"""Per-step inference cost vs context length for the paired TTT arms.

Autoregressive decoding, batch=1, one token at a time, single CPU thread.
Both arms run the same fast MLP and one inner gradient step per step, so the
per-step cost is O(1) in context length for both; the difference is that ttt
differentiates all four fast tensors while ttt_last differentiates only W2.

Implementation note: this measures the sequential PyTorch per-step loop used
throughout this probe. It is NOT the official chunk-parallel TTT kernel, and
these numbers must not be read as that kernel's cost.
"""
import argparse
import json
import time

import torch

from models import Policy

WINDOWS = [(0, 64), (64, 128), (128, 256), (256, 512), (512, 1024),
           (1024, 1536), (1536, 2048)]


@torch.no_grad()
def run_once(mixer, d, T, layers=2):
    torch.manual_seed(0)
    model = Policy(mixer, d=d, layers=layers).eval()
    tokens = torch.randn(1, T, 4) * 0.5
    states = None
    step_ms = []
    for t in range(T):
        t0 = time.perf_counter()
        _, states = model(tokens[:, t:t + 1], states, create_graph=False)
        step_ms.append((time.perf_counter() - t0) * 1e3)
    return step_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--mixers", nargs="+", default=["ttt", "ttt_last"],
                    choices=["ttt", "ttt_last"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    res = {"d": args.d, "T": args.T, "layers": args.layers,
           "threads": args.threads, "reps": args.reps,
           "windows": [f"{a}-{b}" for a, b in WINDOWS],
           "impl": ("sequential PyTorch per-step loop; not the official "
                    "chunk-parallel TTT kernel; no weight norm"),
           "mixers": {}}
    for mixer in args.mixers:
        best = None
        for r in range(args.reps):
            step_ms = run_once(mixer, args.d, args.T, args.layers)
            if best is None:
                best = step_ms
            else:  # element-wise min across reps
                best = [min(a, b) for a, b in zip(best, step_ms)]
            print(f"[{mixer}] rep {r}: total {sum(step_ms):.0f} ms", flush=True)
        win = {}
        for a, b in WINDOWS:
            if b <= args.T:
                seg = best[a:b]
                win[f"{a}-{b}"] = sum(seg) / len(seg)
        res["mixers"][mixer] = {"step_ms": best, "window_ms": win,
                                "total_ms": sum(best)}
    with open(args.out, "w") as f:
        json.dump(res, f)
    for m, d_ in res["mixers"].items():
        print(m, {k: round(v, 3) for k, v in d_["window_ms"].items()},
              flush=True)


if __name__ == "__main__":
    main()
