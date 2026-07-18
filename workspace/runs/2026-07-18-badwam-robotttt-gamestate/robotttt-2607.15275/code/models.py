"""Sequence mixers for the RoboTTT mechanism probe.

Three recurrent/attention mixers, all d_model=64, sharing embed/head:

1. TTTLayer  -- the paper's mechanism (Eq. 1-2 of arXiv 2607.15275):
       W_t <- W_{t-1} - eta * grad_W || f_{W_{t-1}}(K_t) - V_t ||^2
       O_t = f_{W_t}(Q_t)
   Fast model f_W: 2-layer MLP with GeLU (as in the paper, App. A.1).
   Fast weights are PER-SEQUENCE (batched), initialised from learned W0.
   eta: learnable inner learning rate, base 0.1 (paper, App. A.1).

2. DeltaRuleLayer -- linear-attention / GDN-lite analogue (paper's GDN
   baseline family): gated delta rule with learned matrix state init S0,
   update-then-apply, per-channel gate beta. Note: the delta rule IS one
   online gradient step on a *linear* fast model, so this doubles as the
   paper's "TTT Linear" ablation with a learned gate.

3. CausalAttention -- standard transformer block (attention + MLP) with a
   KV cache for autoregressive inference.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- TTT ----
def fast_mlp_apply(W, x):
    """x: (B, d_in) -> (B, d_out). W holds batched tensors."""
    h = torch.bmm(x.unsqueeze(1), W["W1"]).squeeze(1) + W["b1"]
    h = F.gelu(h)
    return torch.bmm(h.unsqueeze(1), W["W2"]).squeeze(1) + W["b2"]


class TTTLayer(nn.Module):
    def __init__(self, d=64, hidden=64, base_lr=0.1):
        super().__init__()
        self.d, self.hidden = d, hidden
        self.ln = nn.LayerNorm(d)
        self.theta_q = nn.Linear(d, d, bias=False)
        self.theta_k = nn.Linear(d, d, bias=False)
        self.theta_v = nn.Linear(d, d, bias=False)
        self.theta_o = nn.Linear(d, d, bias=False)
        # learned fast-weight initialisation W0 (unbatched, expanded per fwd)
        self.W0_1 = nn.Parameter(torch.randn(d, hidden) * 0.02)
        self.b0_1 = nn.Parameter(torch.zeros(hidden))
        self.W0_2 = nn.Parameter(torch.randn(hidden, d) * 0.02)
        self.b0_2 = nn.Parameter(torch.zeros(d))
        # learnable inner LR on top of constant base LR 0.1 (paper A.1)
        self.eta = nn.Parameter(torch.full((d,), base_lr))
        self.eta_h = nn.Parameter(torch.tensor(base_lr))

    def init_state(self, B):
        return {
            "W1": self.W0_1.unsqueeze(0).expand(B, -1, -1),
            "b1": self.b0_1.unsqueeze(0).expand(B, -1),
            "W2": self.W0_2.unsqueeze(0).expand(B, -1, -1),
            "b2": self.b0_2.unsqueeze(0).expand(B, -1),
        }

    def _inner_update(self, W, k, v, create_graph):
        pred = fast_mlp_apply(W, k)
        # per-sequence dim-averaged MSE, summed over the batch: each
        # sequence's fast weights take a full-lr step on their own loss
        loss = ((pred - v) ** 2).mean(-1).sum()
        keys = ["W1", "b1", "W2", "b2"]
        grads = torch.autograd.grad(loss, [W[kk] for kk in keys],
                                    create_graph=create_graph)
        e = self.eta
        scales = {"W1": e.view(1, -1, 1), "b1": self.eta_h,
                  "W2": e.view(1, 1, -1), "b2": e.view(1, -1)}
        return {kk: W[kk] - scales[kk] * gg for kk, gg in zip(keys, grads)}

    def forward(self, x, state=None, create_graph=True):
        """x: (B, T, d). Returns (out (B,T,d), fast-weight state)."""
        B, T, _ = x.shape
        xn = self.ln(x)
        Q, K, V = self.theta_q(xn), self.theta_k(xn), self.theta_v(xn)
        W = self.init_state(B) if state is None else state
        if create_graph:
            # TBPTT: state arrives detached at segment boundaries; make it a
            # fresh leaf so the inner gradient step is differentiable within
            # this segment (first segment still traces back to W0 itself).
            W = {kk: (vv if vv.requires_grad else vv.requires_grad_(True))
                 for kk, vv in W.items()}
        outs = []
        for t in range(T):
            if create_graph:
                W = self._inner_update(W, K[:, t], V[:, t], create_graph=True)
            else:  # inference: inner grad without keeping the graph
                with torch.enable_grad():
                    Wd = {kk: vv.detach().requires_grad_(True)
                          for kk, vv in W.items()}
                    W = self._inner_update(Wd, K[:, t], V[:, t],
                                           create_graph=False)
                    W = {kk: vv.detach() for kk, vv in W.items()}
            outs.append(fast_mlp_apply(W, Q[:, t]))
        return self.theta_o(torch.stack(outs, dim=1)), W


def detach_state(state):
    if state is None:
        return None
    if isinstance(state, dict):
        return {k: v.detach() for k, v in state.items()}
    return state.detach()


# -------------------------------------------------------- Delta rule ----
class DeltaRuleLayer(nn.Module):
    """Gated delta rule, update-then-apply. State S: (B, d, d)."""

    def __init__(self, d=64):
        super().__init__()
        self.d = d
        self.ln = nn.LayerNorm(d)
        self.theta_q = nn.Linear(d, d, bias=False)
        self.theta_k = nn.Linear(d, d, bias=False)
        self.theta_v = nn.Linear(d, d, bias=False)
        self.theta_b = nn.Linear(d, d)
        self.theta_o = nn.Linear(d, d, bias=False)
        self.S0 = nn.Parameter(torch.randn(d, d) * 0.02)

    def forward(self, x, state=None, create_graph=True):
        B, T, _ = x.shape
        xn = self.ln(x)
        Q, K, V = self.theta_q(xn), self.theta_k(xn), self.theta_v(xn)
        Bt = torch.sigmoid(self.theta_b(xn))
        S = (self.S0.unsqueeze(0).expand(B, -1, -1)
             if state is None else state)
        outs = []
        for t in range(T):
            v = V[:, t]                                  # (B, d)
            k = F.normalize(K[:, t], dim=-1).unsqueeze(-1)  # (B, d, 1)
            q = F.normalize(Q[:, t], dim=-1)
            err = torch.bmm(S, k).squeeze(-1) - v        # (B, d)
            S = S - (Bt[:, t] * err).unsqueeze(-1) * k.transpose(1, 2)
            outs.append(torch.bmm(S, q.unsqueeze(-1)).squeeze(-1))
        return self.theta_o(torch.stack(outs, dim=1)), S


# ----------------------------------------------------- Transformer ----
class CausalSelfAttention(nn.Module):
    def __init__(self, d=64, heads=2):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, cache=None):
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        if T > 1:  # causal mask (only needed when processing a chunk)
            Tq, Tk = q.shape[2], k.shape[2]
            mask = torch.ones(Tq, Tk, dtype=torch.bool, device=x.device).tril(
                diagonal=Tk - Tq)
            scores = scores.masked_fill(~mask, float("-inf"))
        att = torch.softmax(scores, dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, T, d)
        return self.out(o), (k, v)


class TransformerBlock(nn.Module):
    def __init__(self, d=64, hidden=128, heads=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                 nn.Linear(hidden, d))

    def forward(self, x, cache=None):
        a, cache = self.attn(self.ln1(x), cache)
        x = x + a
        return x + self.mlp(self.ln2(x)), cache


# ------------------------------------------------------------ Policy ----
class Policy(nn.Module):
    """embed -> L mixer layers -> head, per-timestep action regression."""

    def __init__(self, mixer="ttt", d=64, layers=2):
        super().__init__()
        self.mixer_type = mixer
        self.embed = nn.Linear(4, d)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            if mixer == "ttt":
                self.layers.append(TTTLayer(d))
            elif mixer == "delta":
                self.layers.append(DeltaRuleLayer(d))
            elif mixer == "attn":
                self.layers.append(TransformerBlock(d))
            else:
                raise ValueError(mixer)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)

    def forward(self, tokens, states=None, create_graph=True):
        """tokens: (B,T,4). Returns (pred (B,T,2), states)."""
        x = self.embed(tokens)
        new_states = []
        for i, layer in enumerate(self.layers):
            st = None if states is None else states[i]
            if self.mixer_type == "attn":
                y, st = layer(x, st)
            else:
                y, st = layer(x, st, create_graph)
            x = x + y  # residual
            new_states.append(st)
        return self.head(self.ln_f(x)), new_states

    def init_states(self):
        return None


class NoContextPolicy(nn.Module):
    """Per-timestep MLP: cannot use history. Reference for context value."""

    def __init__(self, d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, d), nn.GELU(),
                                 nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, tokens, states=None, create_graph=True):
        return self.net(tokens), None
