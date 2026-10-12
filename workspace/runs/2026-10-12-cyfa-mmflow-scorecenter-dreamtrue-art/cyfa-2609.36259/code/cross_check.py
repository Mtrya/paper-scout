"""Fidelity cross-checks for the CPU harness.

1. op-level: random-input equality of the harness path (official naive_recurrent_cyfa invoked
   through the patched layer namespace) vs a direct call to naive_recurrent_cyfa.
2. end-to-end: a 128-token WikiText-2 forward through the patched official model must produce
   a sane NLL; the full calibration number is produced by probe_calib.py.
3. confirms the recorder preserves outputs bit-exactly (delta_fn=None).
"""

from __future__ import annotations

import numpy as np
import torch

import cyfa_cpu
from cyfa_cpu import Recorder


def op_level_check():
    from cyclic_flow_attention.ops.naive import naive_recurrent_cyfa
    from cyclic_flow_attention.ops.utils import build_readout_table

    torch.manual_seed(0)
    B, T, H, DK, DV, C = 1, 37, 4, 256, 256, 127
    q = torch.randn(B, T, H, DK)
    k = torch.randn(B, T, H, DK)
    v = torch.randn(B, T, H, DV)
    a = torch.randn(B, T, H)
    dt_bias = torch.randn(H)
    A = torch.rand(H) * 16
    decay = torch.nn.functional.softplus(a + dt_bias)
    g = -A * decay
    delta = torch.rand(B, T, H) * 0.9 + 0.05
    beta = torch.rand(B, T, H)
    readout = torch.eye(C).expand(H, C, C).clone() + 0.05 * torch.randn(H, C, C)
    qw = torch.rand(DK) + 0.5
    kw = torch.rand(DK) + 0.5

    ref, ref_state = naive_recurrent_cyfa(
        q=q, k=k, v=v, g=g, delta=delta, beta=beta, readout=readout,
        q_norm_weight=qw, k_norm_weight=kw, q_norm_eps=1e-6, k_norm_eps=1e-6,
        output_final_state=True,
    )
    op = cyfa_cpu.make_cyfa_op()
    cyfa_cpu.RECORDER = Recorder(n_layers=24, want_state=True)
    out, state = op(q=q, k=k, v=v, g=g, delta=delta, beta=beta, readout=readout,
                    q_norm_weight=qw, k_norm_weight=kw, q_norm_eps=1e-6, k_norm_eps=1e-6,
                    output_final_state=False, checkpoint_level=0)
    cyfa_cpu.RECORDER = None
    d_out = (out - ref).abs().max().item()
    d_state = max((a - b).abs().max().item() for a, b in zip(state, ref_state))
    print(f"[op-level] max|out diff|={d_out:.3e}  max|state diff|={d_state:.3e}")
    assert d_out == 0.0 and d_state == 0.0, "recorder must be bit-exact"
    print("[op-level] recorder path is bit-exact vs official naive_recurrent_cyfa")


def e2e_smoke():
    model, ckpt = cyfa_cpu.load_model()
    tok = cyfa_cpu.load_tokenizer(ckpt)
    print("checkpoint:", ckpt)
    text = ("The history of the steam engine stretches back into the first century AD; "
            "the aeolipile described by Hero of Alexandria is the earliest known example. "
            "In the following centuries the fundamentals were gradually developed, and "
            "the first commercially successful engine arrived at the beginning of the "
            "eighteenth century. Thomas Newcomen built the atmospheric engine in 1712, "
            "and James Watt later improved it decisively with the separate condenser.")
    ids = torch.tensor(tok(text, add_special_tokens=True).input_ids, dtype=torch.long)
    rec = Recorder(n_layers=24)
    nll = cyfa_cpu.forward_nll(model, ids, recorder=rec)
    print(f"[e2e smoke] tokens={len(ids)} NLL={nll:.4f} PPL={np.exp(nll):.2f}")
    deltas = np.stack([t["delta"] for t in rec.traces])  # [24, B, T, H]
    print("[e2e smoke] delta mean per layer (first 6):", deltas[:, 0, :, :].mean(axis=(1, 2))[:6].round(3))
    assert 1.5 < nll < 6.0, "smoke NLL out of sane range"
    print("[e2e smoke] OK")


def chunk_vs_naive_check():
    """Official naive recurrence vs my chunkwise torch op on random inputs."""
    from cyclic_flow_attention.ops.naive import naive_recurrent_cyfa
    from chunk_torch import chunk_cyfa_torch

    torch.manual_seed(1)
    B, T, H, DK, DV, C = 2, 256, 4, 256, 256, 127
    q = torch.randn(B, T, H, DK)
    k = torch.randn(B, T, H, DK)
    v = torch.randn(B, T, H, DV)
    a = torch.randn(B, T, H)
    dt_bias = torch.randn(H)
    A = torch.rand(H) * 16
    g = -A * torch.nn.functional.softplus(a + dt_bias)
    delta = torch.rand(B, T, H) * 0.9 + 0.05
    beta = torch.rand(B, T, H)
    readout = torch.eye(C).expand(H, C, C).clone() + 0.05 * torch.randn(H, C, C)
    qw = torch.rand(DK) + 0.5
    kw = torch.rand(DK) + 0.5

    ref, ref_state = naive_recurrent_cyfa(
        q=q, k=k, v=v, g=g, delta=delta, beta=beta, readout=readout,
        q_norm_weight=qw, k_norm_weight=kw, q_norm_eps=1e-6, k_norm_eps=1e-6,
        output_final_state=True,
    )
    out, state = chunk_cyfa_torch(
        q=q, k=k, v=v, g=g, delta=delta, beta=beta, readout=readout,
        q_norm_weight=qw, k_norm_weight=kw, q_norm_eps=1e-6, k_norm_eps=1e-6,
        output_final_state=True,
    )
    d_out = (out - ref).abs().max().item()
    rel_out = (out - ref).abs().max().item() / ref.abs().max().item()
    ds = [(a - b).abs().max().item() for a, b in zip(state, ref_state)]
    print(f"[chunk-vs-naive] max|out diff|={d_out:.3e} (rel {rel_out:.3e}); "
          f"state diffs hk={ds[0]:.3e} hv={ds[1]:.3e} lam={ds[2]:.3e}")


if __name__ == "__main__":
    op_level_check()
    chunk_vs_naive_check()
    e2e_smoke()
