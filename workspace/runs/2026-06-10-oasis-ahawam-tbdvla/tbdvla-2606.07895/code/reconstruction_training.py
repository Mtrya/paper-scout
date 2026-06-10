"""
PyTorch reconstruction of TBD-VLA's block diffusion training loop.
Based on the paper (Sec 4.2, Eq 1) and the doubled-layout attention trick
from Arriola et al. "Block diffusion" (ICLR 2025) and Wu et al. "Fast-dLLM v2" (ICLR 2026).

This is NOT official code. It is a compact, runnable sketch meant to make the
mechanism concrete for the investigation thread.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BlockDiffusionTrainingHelper:
    """
    TBD-VLA training specifics:
    - Actions are discretized into N_b bins (paper: 512)
    - Prediction horizon H_p = 16, block size m = 4  => K = 4 blocks
    - Temporal-level token shift: block k predicts block k (but logits are
      generated conditioned on previous blocks, aligning with AR VLM pretraining)
    - Doubled-layout: clean x^0 and noised x^t are concatenated in the input
      sequence with shared RoPE positions, and a custom attention mask ensures
      each clean block only sees prefix + previous clean blocks + its own noised
      block.
    """

    def __init__(
        self,
        num_bins: int = 512,
        horizon: int = 16,
        block_size: int = 4,
        action_dim: int = 7,          # e.g. 6-DOF + gripper
        mask_token_id: int = 510,     # reserved id <MASK>
        pad_token_id: int = 511,
    ):
        self.num_bins = num_bins
        self.horizon = horizon
        self.block_size = block_size
        self.action_dim = action_dim
        self.K = horizon // block_size          # number of temporal blocks
        self.tokens_per_block = block_size * action_dim
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

    def tokenize_action_chunk(self, actions: torch.Tensor) -> torch.Tensor:
        """
        actions: [B, H_p, D_a] continuous actions
        returns: [B, L_t] where L_t = H_p * D_a, each entry is a bin index in [0, N_b)
        """
        B = actions.shape[0]
        # Min-max normalization happens outside; here we assume actions in [0, 1]
        bins = (actions.clamp(0, 1) * (self.num_bins - 1)).long()
        return bins.view(B, -1)  # [B, L_t]

    def forward_process_block(self, clean_block: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Eq 1 (forward process): independently mask each token with probability t ~ U(0,1).
        clean_block: [B, tokens_per_block]
        returns: (noised_block [B, tokens_per_block], mask [B, tokens_per_block])
        """
        B, L = clean_block.shape
        t = torch.rand(B, L, device=clean_block.device)
        mask = torch.rand(B, L, device=clean_block.device) < t
        noised = clean_block.clone()
        noised[mask] = self.mask_token_id
        return noised, mask

    def build_doubled_layout_attention_mask(
        self,
        prefix_len: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build the custom 4D attention mask for the doubled-layout training pass.

        Layout of the input sequence (per sample):
          [prefix (vision+lang+state)]
          [clean_block_0] [noised_block_0]
          [clean_block_1] [noised_block_1]
          ...
          [clean_block_{K-1}] [noised_block_{K-1}]

        Total action tokens in this layout = 2 * H_p * D_a.
        Rules:
          1. Prefix is causal (standard VLM causal mask on prefix).
          2. Noised_block_k can attend to prefix and all clean blocks <= k-1.
          3. Clean_block_k can attend to prefix, all clean blocks <= k-1,
             AND its own noised_block_k.
          4. Nothing attends to padding.
          5. RoPE positions are shared between clean_block_k and noised_block_k
             (they represent the same temporal slot).

        Returns: [B, 1, total_len, total_len] float mask (0 for attend, -inf for block).
        """
        action_tokens = 2 * self.tokens_per_block * self.K
        total_len = prefix_len + action_tokens

        # Build a boolean mask: True = allowed to attend
        mask = torch.zeros(batch_size, total_len, total_len, dtype=torch.bool, device=device)

        # --- Prefix causal mask ---
        prefix_causal = torch.tril(torch.ones(prefix_len, prefix_len, dtype=torch.bool, device=device))
        mask[:, :prefix_len, :prefix_len] = prefix_causal.unsqueeze(0)

        # Helper: start/end indices for each segment
        def seg_start(idx: int, is_clean: bool) -> int:
            # segments ordered: clean_0, noised_0, clean_1, noised_1, ...
            offset = prefix_len + idx * 2 * self.tokens_per_block
            offset += 0 if is_clean else self.tokens_per_block
            return offset

        seg_len = self.tokens_per_block

        for k in range(self.K):
            c_start = seg_start(k, True)
            n_start = seg_start(k, False)

            # Noised block k attends to prefix and previous clean blocks
            allowed_cols = list(range(prefix_len))
            for j in range(k):
                allowed_cols += list(range(seg_start(j, True), seg_start(j, True) + seg_len))
            for row in range(n_start, n_start + seg_len):
                mask[:, row, allowed_cols] = True

            # Clean block k attends to prefix, previous clean blocks, AND noised block k
            allowed_cols = list(range(prefix_len))
            for j in range(k):
                allowed_cols += list(range(seg_start(j, True), seg_start(j, True) + seg_len))
            allowed_cols += list(range(n_start, n_start + seg_len))
            for row in range(c_start, c_start + seg_len):
                mask[:, row, allowed_cols] = True

        # Convert to float additive mask
        float_mask = torch.zeros_like(mask, dtype=torch.float32)
        float_mask = float_mask.masked_fill(~mask, float("-inf"))
        return float_mask.unsqueeze(1)  # [B, 1, total_len, total_len]

    def compute_loss(
        self,
        logits: torch.Tensor,
        clean_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-entropy averaged over masked positions only (paper loss equation).

        logits: [B, total_len, vocab_size]  (we only care about action positions)
        clean_tokens: [B, L_t] ground-truth clean token ids
        mask: [B, L_t] bool, True where token was masked in forward process
        """
        # In the doubled layout, clean token positions are interleaved.
        # For simplicity this reconstruction assumes we extract the clean-action logits.
        B = logits.shape[0]
        # Flatten and gather masked positions
        logits_masked = logits.view(B, -1, logits.shape[-1])
        # We need to index the correct positions; simplified here:
        loss = F.cross_entropy(
            logits_masked[mask].view(-1, self.num_bins),
            clean_tokens[mask].view(-1),
            reduction="mean",
        )
        return loss

    def training_step(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """
        One training step sketch.
        batch contains:
          - prefix_input_ids: [B, prefix_len]  (vision, language, state tokens)
          - actions: [B, H_p, D_a] continuous actions
        """
        prefix = batch["prefix_input_ids"]
        actions = batch["actions"]
        B = prefix.shape[0]
        device = prefix.device

        # 1. Tokenize clean action chunk
        clean_tokens = self.tokenize_action_chunk(actions)  # [B, L_t]

        # 2. Partition into K blocks and apply forward process per block
        noised_blocks = []
        mask_blocks = []
        for k in range(self.K):
            start = k * self.tokens_per_block
            end = start + self.tokens_per_block
            block = clean_tokens[:, start:end]
            n_block, m_block = self.forward_process_block(block)
            noised_blocks.append(n_block)
            mask_blocks.append(m_block)

        noised_tokens = torch.cat(noised_blocks, dim=1)  # [B, L_t]
        mask = torch.cat(mask_blocks, dim=1)              # [B, L_t]

        # 3. Build doubled-layout input
        # Interleave: clean_0, noised_0, clean_1, noised_1, ...
        action_input_ids = []
        for k in range(self.K):
            start = k * self.tokens_per_block
            end = start + self.tokens_per_block
            action_input_ids.append(clean_tokens[:, start:end])
            action_input_ids.append(noised_tokens[:, start:end])
        action_input_ids = torch.cat(action_input_ids, dim=1)  # [B, 2*L_t]

        input_ids = torch.cat([prefix, action_input_ids], dim=1)

        # 4. Build custom attention mask
        attn_mask = self.build_doubled_layout_attention_mask(
            prefix_len=prefix.shape[1],
            batch_size=B,
            device=device,
        )

        # 5. Forward pass through VLM backbone (e.g. Qwen3-VL 2B)
        # The model would use shared RoPE positions for clean/noised pairs.
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs.logits  # [B, total_len, vocab_size]

        # 6. Extract logits at clean-action positions and compute masked CE loss
        # (position extraction omitted for brevity; see full notebook logic)
        loss = self.compute_loss(logits, clean_tokens, mask)
        return loss


if __name__ == "__main__":
    helper = BlockDiffusionTrainingHelper()
    # Sanity-check mask shape
    mask = helper.build_doubled_layout_attention_mask(prefix_len=128, batch_size=2, device=torch.device("cpu"))
    print("Attention mask shape:", mask.shape)  # [2, 1, 128+2*16*7, 128+2*16*7]
    # Verify that noised_block_1 does NOT attend to clean_block_1 (no cheating)
    prefix_len = 128
    seg_len = helper.tokens_per_block  # 28
    k = 1
    c_start = prefix_len + k * 2 * seg_len
    n_start = c_start + seg_len
    # Check noised block 1 row against clean block 1 col
    can_attend = mask[0, 0, n_start, c_start:c_start + seg_len].isfinite()
    print("Noised block 1 attends to clean block 1?", can_attend.any().item())  # should be False
    # Check clean block 1 row against noised block 1 col
    can_attend = mask[0, 0, c_start, n_start:n_start + seg_len].isfinite()
    print("Clean block 1 attends to noised block 1?", can_attend.any().item())  # should be True
