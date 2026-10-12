"""CPU harness for CyFA: official 400M checkpoint + official layer code, triton-free naive patches.

Design: load the official CyclicFlowAttentionForCausalLM (float32, CPU), monkeypatch the four
triton-dependent fla modules to their exact PyTorch definitions, and route the CyFA core operator
to the repository's own pure-PyTorch reference `naive_recurrent_cyfa` (ops/naive.py).

Patched fla modules (mathematically the same ops as the triton kernels):
- fla RMSNorm            -> mirrors fla.modules.layernorm.rms_norm_ref (is_rms_norm=True)
- fla FusedRMSNormGated  -> rmsnorm(x)*w then * activation(g), whole feature dim
- fla ShortConvolution   -> causal depthwise conv1d, y[t] = sum_j w[:,j] * x[t-(K-1)+j]
- fla SwiGLULinear       -> down(silu(gate) * up)

The CyFA recurrence itself is NOT reimplemented here: we call the official naive_recurrent_cyfa.
A Recorder optionally stashes per-token (delta, beta, g) and final states, and can apply
clock interventions (delta_fn) for the Table-8 replication.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent / "CyclicFlowAttention"
sys.path.insert(0, str(REPO))

torch.set_grad_enabled(False)

# ---------------------------------------------------------------------------
# naive fla replacements
# ---------------------------------------------------------------------------


def naive_rmsnorm_forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
    dtype = x.dtype
    if residual is not None:
        x = (x + residual).to(x.dtype)
    rstd = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
    out = (x.float() * rstd) * self.weight.float()
    if self.bias is not None:
        out = out + self.bias.float()
    out = out.to(dtype)
    return out if not prenorm else (out, x)


def naive_rmsnorm_gated_forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
    dtype = x.dtype
    if residual is not None:
        x = (x + residual).to(x.dtype)
    rstd = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
    out = x.float() * rstd
    if self.weight is not None:
        out = out * self.weight.float()
    if self.bias is not None:
        out = out + self.bias.float()
    if self.activation in ("swish", "silu"):
        act = F.silu(g.float())
    elif self.activation == "sigmoid":
        act = torch.sigmoid(g.float())
    else:
        raise ValueError(self.activation)
    out = (out * act).to(dtype)
    return out if not prenorm else (out, x)


def naive_shortconv_forward(self, x, residual=None, mask=None, cache=None, output_final_state=False,
                            cu_seqlens=None, chunk_indices=None, **kw):
    # x [B, T, C]; causal depthwise conv: y[t] = sum_j w[:,j] * x[t-(K-1)+j]
    assert cache is None and cu_seqlens is None, "probe harness runs stateless full sequences"
    if mask is not None:
        x = x * mask.unsqueeze(-1)
    B, T, C = x.shape
    K = self.kernel_size[0] if hasattr(self.kernel_size, "__len__") else self.kernel_size
    w = self.weight.squeeze(1).float()  # [C, K]
    xp = F.pad(x.float(), (0, 0, K - 1, 0))
    y = torch.zeros_like(xp[:, :T])
    for i in range(K):
        y = y + xp[:, i:i + T] * w[:, i].unsqueeze(0)
    if self.bias is not None:
        y = y + self.bias.float()
    if self.activation == "silu":
        y = F.silu(y)
    elif self.activation is not None:
        raise ValueError(self.activation)
    y = y.to(x.dtype)
    if residual is not None:
        y = y + residual
    return y, None


def naive_swiglu_linear_forward(self, x, y, weight, bias):
    return F.linear(F.silu(x) * y, weight, bias)


def apply_fla_patches():
    import fla.modules.layernorm as fla_ln
    import fla.modules.fused_norm_gate as fla_fng
    import fla.modules.mlp as fla_mlp
    from fla.modules import ShortConvolution

    fla_ln.RMSNorm.forward = naive_rmsnorm_forward
    fla_fng.FusedRMSNormGated.forward = naive_rmsnorm_gated_forward
    fla_mlp.SwiGLULinear.forward = naive_swiglu_linear_forward
    ShortConvolution.forward = naive_shortconv_forward


# ---------------------------------------------------------------------------
# recorder + op wrapper
# ---------------------------------------------------------------------------


class Recorder:
    """Collects per-token (delta, beta, g) and optional final states from every CyFA op call."""

    def __init__(self, n_layers: int, want_state: bool = False, delta_fn=None):
        self.n_layers = n_layers
        self.want_state = want_state
        self.delta_fn = delta_fn  # fn(layer_idx, delta[B,T,H]) -> delta
        self.traces = []  # list per call: dict(delta, beta, g, layer)
        self.states = []  # final (hk, hv, lambda) per call
        self._call = 0

    def reset(self):
        self.traces.clear()
        self.states.clear()
        self._call = 0

    def observe(self, delta, beta, g):
        layer = self._call % self.n_layers
        if self.delta_fn is not None:
            delta = self.delta_fn(layer, delta)
        self.traces.append({
            "layer": layer,
            "delta": delta.detach().float().cpu().numpy().copy(),
            "beta": beta.detach().float().cpu().numpy().copy(),
            "g": g.detach().float().cpu().numpy().copy(),
        })
        self._call += 1
        return delta


RECORDER: Recorder | None = None


def make_cyfa_op(impl: str = "naive"):
    if impl == "naive":
        from cyclic_flow_attention.ops.naive import naive_recurrent_cyfa as core_op
    elif impl == "chunk":
        from chunk_torch import chunk_cyfa_torch as core_op
    else:
        raise ValueError(impl)

    def cyfa_op(q, k, v, g, delta, beta, readout, q_norm_weight, k_norm_weight,
                q_norm_eps, k_norm_eps, scale=None, initial_state=None,
                output_final_state=False, cu_seqlens=None, checkpoint_level=0, **kw):
        rec = RECORDER
        if rec is not None:
            delta = rec.observe(delta, beta, g)
            output_final_state = output_final_state or rec.want_state
        out, state = core_op(
            q=q, k=k, v=v, g=g, delta=delta, beta=beta, readout=readout,
            q_norm_weight=q_norm_weight, k_norm_weight=k_norm_weight,
            q_norm_eps=q_norm_eps, k_norm_eps=k_norm_eps, scale=scale,
            initial_state=initial_state, output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        if rec is not None and rec.want_state and state is not None:
            rec.states.append(tuple(s.detach().float().cpu().numpy().copy() for s in state))
        return out, state

    return cyfa_op


def apply_cyfa_op_patch(impl: str = "naive"):
    import cyclic_flow_attention.layers.cyfa as cyfa_layer_mod

    op = make_cyfa_op(impl)
    cyfa_layer_mod.chunk_cyfa = op
    cyfa_layer_mod.fused_recurrent_cyfa = op
    return op


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------


def load_model(ckpt_path: str | None = None, op_impl: str = "naive"):
    apply_fla_patches()
    apply_cyfa_op_patch(op_impl)

    from cyclic_flow_attention.models.cyfa import (
        CyclicFlowAttentionConfig,
        CyclicFlowAttentionForCausalLM,
    )

    if ckpt_path is None:
        import glob
        import os
        ckpt_path = sorted(glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--cyxxxxxxxxxx--cyfa-400M-15B/snapshots/*")))[0]
    config = CyclicFlowAttentionConfig.from_pretrained(ckpt_path)
    model = CyclicFlowAttentionForCausalLM.from_pretrained(
        ckpt_path, config=config, dtype=torch.float32,
    )
    model.eval()
    return model, ckpt_path


def load_tokenizer(ckpt_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(ckpt_path)


@torch.no_grad()
def forward_nll(model, ids: torch.Tensor, recorder: Recorder | None = None) -> float:
    """ids [T] int64 -> mean next-token NLL over the sequence (token-level)."""
    return forward_batch_nll(model, ids.unsqueeze(0), recorder=recorder)[0]


@torch.no_grad()
def forward_batch_nll(model, ids: torch.Tensor, recorder: Recorder | None = None,
                      want_logits: bool = False):
    """ids [B, T] int64 -> per-sequence mean next-token NLL (token-level).

    All sequences in a batch share the same length (2048). The Recorder receives
    per-call tensors of shape [B, T, H]; call order is layer 0..L-1 for each forward.
    """
    global RECORDER
    RECORDER = recorder
    try:
        logits = model(ids).logits.float()
    finally:
        RECORDER = None
    lp = F.log_softmax(logits, dim=-1)
    nll = -lp[:, :-1].gather(2, ids[:, 1:].unsqueeze(2)).squeeze(2).mean(dim=1)
    nlls = nll.tolist()
    if want_logits:
        return nlls, logits
    return nlls
