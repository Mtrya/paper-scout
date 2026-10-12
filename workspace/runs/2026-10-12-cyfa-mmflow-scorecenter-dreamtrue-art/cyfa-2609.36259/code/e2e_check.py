"""End-to-end fidelity: full 400M model on a real 2048-token WikiText-2 chunk,
official naive recurrence vs my chunkwise torch op. Also times both paths.
Saves results/e2e_check.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import cyfa_cpu
from cyfa_cpu import Recorder

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RES.mkdir(exist_ok=True)


def run(op_impl: str, ids: torch.Tensor):
    model, ckpt = cyfa_cpu.load_model(op_impl=op_impl)
    torch.set_num_threads(16)
    t0 = time.time()
    nlls, logits = cyfa_cpu.forward_batch_nll(model, ids, want_logits=True)
    dt = time.time() - t0
    return nlls[0], logits[0], dt


def main():
    import sys
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    natural = np.load(HERE / "data" / "natural_ids.npy")
    ids = torch.tensor(natural[:1, :T], dtype=torch.long)

    t0 = time.time()
    nll_c, logits_c, t_chunk = run("chunk", ids)
    print(f"[e2e chunk] NLL={nll_c:.4f} time={t_chunk:.1f}s")

    nll_n, logits_n, t_naive = run("naive", ids)
    print(f"[e2e naive] NLL={nll_n:.4f} time={t_naive:.1f}s")

    d = (logits_c - logits_n).abs()
    out = {
        "seq_len": T,
        "nll_chunk": nll_c, "nll_naive": nll_n,
        "logit_max_diff": d.max().item(), "logit_mean_diff": d.mean().item(),
        "time_chunk_s": t_chunk, "time_naive_s": t_naive,
    }
    print(json.dumps(out, indent=2))
    (RES / "e2e_check.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
