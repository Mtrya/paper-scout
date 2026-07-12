"""
LaMem-VLA architectural probe.

A compact, runnable PyTorch sketch of the memory-token injection pattern from
"LaMem-VLA: Dual Latent Memory in Vision-Language-Action Models for Robotic
Manipulation" (arXiv 2607.07608).  The official repository is currently empty,
so this code is my own reconstruction of the method at the level of detail
needed to make the four components (curator / seeker / condenser / weaver)
concrete.

The probe does not train; it simply runs a few timesteps and prints tensor
shapes to show that the sequence assembly in Eq. (9) is well-defined.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBottleneckCompression(nn.Module):
    """
    Learnable compression module C_s that distills visual tokens X_t
    (B, N_i, C) into a compact set of short-term memory tokens v_s
    (B, N_s, C).  The prose calls this an "SE-bottleneck compression module";
    here we implement it as channel-wise SE gating followed by a token-length
    down-projection.  This is an interpretive choice -- the paper gives the
    exact architecture only in the (unavailable) appendix.
    """

    def __init__(self, c: int, ni: int, ns: int, reduction: int = 4):
        super().__init__()
        self.c = c
        self.ni = ni
        self.ns = ns

        # Channel-wise SE gating on every visual token.
        self.se_down = nn.Linear(c, c // reduction)
        self.se_up = nn.Linear(c // reduction, c)

        # Token-length down-projection: N_i -> N_s tokens.
        self.token_mixer = nn.Linear(ni, ns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N_i, C)
        # SE gating.
        gap = x.mean(dim=1)                       # (B, C)
        gate = torch.sigmoid(self.se_up(F.relu(self.se_down(gap))))  # (B, C)
        x = x * gate.unsqueeze(1)                 # broadcast over tokens

        # Reduce token count: treat each channel independently across tokens.
        x = x.transpose(1, 2)                     # (B, C, N_i)
        x = self.token_mixer(x)                   # (B, C, N_s)
        x = x.transpose(1, 2)                     # (B, N_s, C)
        return x


class SmallTransformer(nn.Module):
    """Lightweight two-layer transformer used for the cognition encoder,
    query builder, memory formers, and weaver VLM surrogate."""

    def __init__(self, c: int, heads: int = 4, layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c, nhead=heads, dim_feedforward=4 * c,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(x, mask=mask)


class MemoryVault:
    """
    A simple key-value memory vault with capacity L and the adjacent-pair
    merging rule from Eqs. (4)-(5).  Supports both:
      - short-term vaults: unit = (key: C, value: N_s x C)
      - long-term vaults:  unit = (key: C, value: N_l x C), where N_l == N_a
    """

    def __init__(self, capacity: int, token_dim: int):
        self.capacity = capacity
        self.token_dim = token_dim
        self.keys: List[torch.Tensor] = []    # each (..., C)
        self.values: List[torch.Tensor] = []  # each (..., token_dim, C)

    def write(self, key: torch.Tensor, value: torch.Tensor):
        """Append a new unit and compress if over capacity."""
        self.keys.append(key.detach())
        self.values.append(value.detach())
        if len(self.keys) > self.capacity:
            self._compress_most_redundant_adjacent_pair()

    def _compress_most_redundant_adjacent_pair(self):
        # Eq. (4): find adjacent pair with highest cosine similarity.
        keys = torch.stack(self.keys, dim=0)  # (n, C)
        cos = F.cosine_similarity(keys[:-1], keys[1:], dim=-1)  # (n-1,)
        i = int(cos.argmax().item())

        # Eq. (5): average both key and value tokens.
        new_key = 0.5 * (self.keys[i] + self.keys[i + 1])
        new_value = 0.5 * (self.values[i] + self.values[i + 1])

        self.keys = self.keys[:i] + [new_key] + self.keys[i + 2:]
        self.values = self.values[:i] + [new_value] + self.values[i + 2:]

    def topk(self, query: torch.Tensor, k: int
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-K units by cosine similarity to query.
        Returns padded zero tensors if the vault is empty.
        """
        if len(self.keys) == 0:
            value_shape = (self.token_dim, query.shape[-1])
            zeros = torch.zeros(
                query.shape[0], k * value_shape[0], value_shape[1],
                device=query.device, dtype=query.dtype
            )
            return zeros, torch.zeros(
                query.shape[0], k, device=query.device, dtype=query.dtype
            )

        keys = torch.stack(self.keys, dim=0)  # (n, C)
        # (B, C) vs (n, C) -> (B, n)
        sim = F.cosine_similarity(
            query.unsqueeze(1), keys.unsqueeze(0), dim=-1
        )
        k_eff = min(k, sim.shape[-1])
        top = torch.topk(sim, k_eff, dim=-1)

        # Gather selected values per batch and concatenate along token dim.
        batch_values: List[torch.Tensor] = []
        for b in range(query.shape[0]):
            selected = [self.values[int(idx.item())] for idx in top.indices[b]]
            batch_values.append(torch.cat(selected, dim=0))  # (k*N, C)

        out = torch.stack(batch_values, dim=0)
        if k_eff < k:
            pad = torch.zeros(
                query.shape[0], (k - k_eff) * self.token_dim, query.shape[-1],
                device=query.device, dtype=query.dtype
            )
            out = torch.cat([out, pad], dim=1)
        return out, top.values

    def __len__(self) -> int:
        return len(self.keys)


class LatentMemorySeeker(nn.Module):
    """Builds the context-aware query Q_t (Eq. 6) and retrieves evidence
    from both vaults (Eq. 7)."""

    def __init__(self, c: int, kq: int):
        super().__init__()
        self.kq = kq
        self.query_init = nn.Parameter(torch.randn(kq, c) * 0.02)
        self.query_builder = SmallTransformer(c)

    def forward(self, q_con: torch.Tensor) -> torch.Tensor:
        # q_con: (B, T, C).  Append learnable query slots.
        B = q_con.shape[0]
        q_init = self.query_init.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([q_con, q_init], dim=1)
        # Causal mask keeps the original multimodal states from being
        # perturbed by the query slots: earlier positions never attend later.
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        out = self.query_builder(x, mask=mask)
        return out[:, -self.kq:, :]  # (B, K_q, C)


class MemoryCondenser(nn.Module):
    """Reconstructs retrieved evidence Z into fixed-length memory tokens
    (Eq. 8).  Two independent formers are used for short- and long-term
    evidence because their provenance differs."""

    def __init__(self, c: int, l_slots: int, name: str = "memory"):
        super().__init__()
        self.l_slots = l_slots
        self.name = name
        self.slots = nn.Parameter(torch.randn(l_slots, c) * 0.02)
        self.former = SmallTransformer(c)

    def forward(self, q_t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # q_t: (B, K_q, C); z: (B, K*N, C)
        B = z.shape[0]
        slots = self.slots.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([q_t, z, slots], dim=1)
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        out = self.former(x, mask=mask)
        return out[:, -self.l_slots:, :]  # (B, L, C)


class LaMemVLAProbe(nn.Module):
    """
    Surrogate of the full LaMem-VLA inference loop.

    Inputs per timestep:
      - X_t : visual tokens, shape (B, N_i, C)
      - I   : instruction tokens, shape (B, N_l, C)
      - short_vault, long_vault : MemoryVault objects populated from history

    Outputs per timestep:
      - action_chunk : (B, H, 7)  (dummy; no diffusion training here)
      - z_action     : (B, N_a, C) memory-grounded action tokens
      - the updated vaults (caller should persist them)
    """

    def __init__(self, c: int = 256, ni: int = 16, nl: int = 8,
                 na: int = 4, ns: int = 4, capacity: int = 8,
                 topk: int = 4, kq: int = 2, ls: int = 4, ll: int = 2,
                 action_horizon: int = 16):
        super().__init__()
        self.c = c
        self.ni = ni
        self.nl = nl
        self.na = na
        self.ns = ns
        self.capacity = capacity
        self.topk = topk
        self.ls = ls
        self.ll = ll
        self.action_horizon = action_horizon

        # 1. Curator: compression for short-term visual evidence, and
        #    direct storage of action hidden states in the long-term vault.
        self.short_compressor = SEBottleneckCompression(c, ni, ns)

        # 2. Cognition encoder and query builder (Eq. 6).
        self.cognition_encoder = SmallTransformer(c)
        self.seeker = LatentMemorySeeker(c, kq)

        # 3. Condensers (Eq. 8): one for short-term visual, one for long-term.
        self.short_condenser = MemoryCondenser(c, ls, name="short")
        self.long_condenser = MemoryCondenser(c, ll, name="long")

        # 4. Weaver: source embeddings and the VLM pass that produces action
        #    tokens (Eq. 9).  We use a surrogate transformer in place of the
        #    7B-parameter Prismatic VLM.
        self.source_embed_short = nn.Parameter(torch.randn(c) * 0.02)
        self.source_embed_long = nn.Parameter(torch.randn(c) * 0.02)
        self.action_queries = nn.Parameter(torch.randn(na, c) * 0.02)
        self.weaver = SmallTransformer(c)

        # 5. Surrogate diffusion action expert: maps action tokens + timestep
        #    to a noise prediction / action chunk.  Kept tiny for the probe.
        self.action_mlp = nn.Sequential(
            nn.Linear(na * c + action_horizon * 7 + c, 512),
            nn.ReLU(),
            nn.Linear(512, action_horizon * 7)
        )
        self.time_embed = nn.Linear(1, c)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device),
                          diagonal=1).bool()

    def encode_cognition(self, x: torch.Tensor, inst: torch.Tensor
                         ) -> torch.Tensor:
        """Produce the multimodal cognition representation Q_con from current
        observation and instruction tokens."""
        seq = torch.cat([x, inst], dim=1)  # (B, N_i + N_l, C)
        return self.cognition_encoder(seq)  # all tokens are contextualized

    def retrieve_and_condense(
        self,
        q_con: torch.Tensor,
        short_vault: MemoryVault,
        long_vault: MemoryVault
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Eq. 6
        q_t = self.seeker(q_con)            # (B, K_q, C)
        q_mean = q_t.mean(dim=1)            # (B, C) -- global retrieval vector

        # Eq. 7: top-K retrieval from both vaults (non-differentiable topk).
        z_short, _ = short_vault.topk(q_mean, self.topk)   # (B, K*N_s, C)
        z_long, _ = long_vault.topk(q_mean, self.topk)     # (B, K*N_a, C)

        # Eq. 8: condense into fixed-length latent memory tokens.
        m_short = self.short_condenser(q_t, z_short)       # (B, L_s, C)
        m_long = self.long_condenser(q_t, z_long)          # (B, L_l, C)
        return m_short, m_long

    def weave(self, m_short: torch.Tensor, m_long: torch.Tensor,
              x: torch.Tensor, inst: torch.Tensor) -> torch.Tensor:
        # Eq. 9: prepend source-biased memory tokens to the current inputs.
        B = x.shape[0]
        m_short = m_short + self.source_embed_short.view(1, 1, self.c)
        m_long = m_long + self.source_embed_long.view(1, 1, self.c)
        q_action = self.action_queries.unsqueeze(0).expand(B, -1, -1)
        seq = torch.cat([m_short, m_long, x, inst, q_action], dim=1)
        out = self.weaver(seq)
        return out[:, -self.na:, :]  # memory-grounded action tokens

    def forward(
        self,
        x: torch.Tensor,
        inst: torch.Tensor,
        short_vault: MemoryVault,
        long_vault: MemoryVault,
        diffusion_step: int = 0,
        noisy_action: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Current cognition -> context-aware retrieval -> condensed memory.
        q_con = self.encode_cognition(x, inst)
        m_short, m_long = self.retrieve_and_condense(q_con, short_vault,
                                                     long_vault)

        # Memory is woven into the same continuous sequence as current obs,
        # instruction, and action queries.
        z_action = self.weave(m_short, m_long, x, inst)

        # Surrogate diffusion expert: condition on z_action and a timestep.
        B = z_action.shape[0]
        flat_action_tokens = z_action.reshape(B, -1)
        if noisy_action is None:
            noisy_action = torch.randn(B, self.action_horizon * 7,
                                       device=x.device, dtype=x.dtype)
        t_embed = self.time_embed(
            torch.tensor([[diffusion_step]], dtype=x.dtype, device=x.device)
            .expand(B, -1)
        )
        action_chunk = self.action_mlp(
            torch.cat([flat_action_tokens, noisy_action, t_embed], dim=-1)
        )
        action_chunk = action_chunk.view(B, self.action_horizon, 7)

        # After the forward pass, commit history to the vaults for next step.
        with torch.no_grad():
            for b in range(B):
                # Short-term: compress current visual tokens and store.
                v_s = self.short_compressor(x[b:b + 1]).squeeze(0)
                k_s = v_s.mean(dim=0)
                short_vault.write(k_s, v_s)

                # Long-term: store the action hidden state just produced.
                h_action = z_action[b]                       # (N_a, C)
                k_l = h_action.mean(dim=0)
                long_vault.write(k_l, h_action)

        return action_chunk, z_action


def make_batch(b: int, c: int, ni: int, nl: int,
               device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(b, ni, c, device=device)
    inst = torch.randn(b, nl, c, device=device)
    return x, inst


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running LaMem-VLA probe on {device}")

    # Hyperparameters are intentionally tiny; the goal is shape clarity.
    model = LaMemVLAProbe(
        c=256, ni=16, nl=8, na=4, ns=4,
        capacity=4, topk=4, kq=2, ls=4, ll=2,
        action_horizon=16
    ).to(device)

    B = 1
    x, inst = make_batch(B, model.c, model.ni, model.nl, device)

    # One episode history for this single rollout.
    short_vault = MemoryVault(model.capacity, model.ns)
    long_vault = MemoryVault(model.capacity, model.na)

    print("\n--- Episode roll-out ---")
    for t in range(8):
        # The model writes the current step's evidence back into the vaults.
        action_chunk, z_action = model(
            x, inst, short_vault, long_vault, diffusion_step=t
        )
        print(
            f"t={t}: action_chunk {tuple(action_chunk.shape)}, "
            f"z_action {tuple(z_action.shape)}, "
            f"short_vault_size={len(short_vault)}, "
            f"long_vault_size={len(long_vault)}"
        )

    # Show that a dummy loss can be back-propagated through the whole flow,
    # including through the memory tokens that are part of the input sequence.
    loss = action_chunk.mean()
    loss.backward()
    print("\nDummy loss backward succeeded.")
    print("Gradient on action queries:",
          model.action_queries.grad is not None)
    print("Gradient on source embedding short:",
          model.source_embed_short.grad is not None)


if __name__ == "__main__":
    main()
