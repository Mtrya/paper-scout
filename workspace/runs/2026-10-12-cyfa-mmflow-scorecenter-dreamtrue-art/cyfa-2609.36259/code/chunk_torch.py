"""Chunkwise pure-PyTorch CyFA op (paper Eq. 85-88 / Appendix B), for fast CPU inference.

Mathematically identical to the official token-by-token `naive_recurrent_cyfa`; only the
summation order changes (chunk-parallel form of the two ScalarGatedLA passes). All decay
factors are products of alpha_t <= 1, so every exp() argument is <= 0 and fp32 is safe
(no overflow; underflow to 0 is the correct semantics of forgetting).

Rotation/RMSNorm helpers are imported from the official ops.naive module, so the temporal
write vector, readout table and modal rotations are literally the same code as the reference.
"""

from __future__ import annotations

import torch

from cyclic_flow_attention.ops.naive import (
    _rmsnorm_forward,
    _rotate_atom,
    _rotate_modal_neg,
    _rotate_modal_pos,
)
from cyclic_flow_attention.ops.utils import build_readout_table, validate_cyfa_inputs


def scalar_gated_la_chunk(q, k, v, g, chunk_size: int = 64, initial_state=None):
    """Chunkwise ScalarGatedLA: S_t = a_t S_{t-1} + k_t v_t^T, O_t = q_t S_t, a_t = exp(g_t).

    q,k: [B,T,H,D] float32; v: [B,T,H,E]; g: [B,T,H]; initial_state: [B,H,D,E] or None.
    Returns (O [B,T,H,E], S_final [B,H,D,E]).
    """
    B, T, H, D = q.shape
    E = v.shape[-1]
    C = chunk_size
    assert T % C == 0
    NC = T // C
    qc = q.transpose(1, 2).reshape(B, H, NC, C, D)
    kc = k.transpose(1, 2).reshape(B, H, NC, C, D)
    vc = v.transpose(1, 2).reshape(B, H, NC, C, E)
    gc = g.transpose(1, 2).reshape(B, H, NC, C)

    log_gam = gc.cumsum(dim=-1)                      # [B,H,NC,C] log gamma_r within chunk
    gam = log_gam.exp()                              # <= 1
    gam_c = gam[:, :, :, -1]                         # [B,H,NC] chunk decay
    # Gamma[r,j] = gam_r / gam_j for j <= r else 0, computed stably as exp(log diff)
    log_ratio = log_gam.unsqueeze(-1) - log_gam.unsqueeze(-2)   # [B,H,NC,C(r),C(j)]
    tril = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device))
    Gamma = torch.where(tril, log_ratio, torch.full_like(log_ratio, -torch.inf)).exp()

    gram = qc @ kc.transpose(-1, -2)                  # [B,H,NC,C,C]
    o_intra = (gram * Gamma) @ vc                     # [B,H,NC,C,E]

    vtilde = vc * (log_gam[:, :, :, -1:] - log_gam).exp().unsqueeze(-1)
    # vtilde[j] = (gam_C / gam_j) * v_j ; k^T @ vtilde accumulates the chunk into the state
    state = initial_state
    if state is None:
        state = torch.zeros(B, H, D, E, dtype=torch.float32, device=q.device)
    outs = []
    for c in range(NC):
        o_inter = (qc[:, :, c] * gam[:, :, c].unsqueeze(-1)) @ state   # [B,H,C,E]
        outs.append(o_inter + o_intra[:, :, c])
        state = state * gam_c[:, :, c, None, None] + kc[:, :, c].transpose(-1, -2) @ vtilde[:, :, c]
    out = torch.stack(outs, dim=2).reshape(B, H, T, E).transpose(1, 2)
    return out, state


def chunk_cyfa_torch(q, k, v, g, delta, beta, readout, q_norm_weight, k_norm_weight,
                     q_norm_eps, k_norm_eps, scale=None, initial_state=None,
                     output_final_state=False, cu_seqlens=None, checkpoint_level=0,
                     chunk_size: int = 64, **_):
    """Drop-in replacement for the official ops with identical semantics, pure torch chunkwise."""
    num_slots = validate_cyfa_inputs(q, k, v, g, delta, beta, readout)
    assert cu_seqlens is None and initial_state is None, "probe harness runs stateless full sequences"
    dtype = q.dtype
    q = _rmsnorm_forward(q, q_norm_weight, q_norm_eps)
    k = _rmsnorm_forward(k, k_norm_weight, k_norm_eps)
    q, k, v = (x.float() for x in (q, k, v))
    g, delta, beta = g.float(), delta.float(), beta.float()
    B, T, H, D = q.shape
    if scale is None:
        scale = D ** -0.5

    lambdas = delta.cumsum(dim=1)                    # [B,T,H], lambda_0 = 0
    lam_flat = lambdas.permute(0, 2, 1).reshape(B * H, T)
    write = beta.permute(0, 2, 1).reshape(B * H, T, 1) * _rotate_atom(lam_flat, num_slots)
    write = write.reshape(B, H, T, num_slots).permute(0, 2, 1, 3)   # [B,T,H,M]

    o1, hk = scalar_gated_la_chunk(q * float(scale), k, write, g, chunk_size)   # o1: [B,T,H,M]

    readout_table = build_readout_table(readout)     # [H,127,M]
    o1f = o1.permute(0, 2, 1, 3).reshape(B * H, T, num_slots)
    rotated = _rotate_modal_pos(lam_flat, o1f, num_slots)
    logits = torch.einsum("hcm,bhtm->bhtc", readout_table, rotated.reshape(B, H, T, num_slots)).reshape(B * H, T, -1)
    probs = torch.softmax(logits, dim=-1)
    coeff = torch.einsum("hcm,bhtc->bhtm", readout_table, probs.reshape(B, H, T, -1)).reshape(B * H, T, num_slots)
    o2 = _rotate_modal_neg(lam_flat, coeff, num_slots).reshape(B, H, T, num_slots).permute(0, 2, 1, 3)

    out, hv = scalar_gated_la_chunk(o2, write, v, g, chunk_size)    # [B,T,H,Dv]

    out = out.to(dtype)
    if output_final_state:
        final_lambda = lambdas[:, -1].transpose(0, 1)  # wait: naive returns [B,H]; lambdas[:, -1] is [B,H]
        final_lambda = lambdas[:, -1, :]
        return out, (hk, hv, final_lambda)
    return out, None
