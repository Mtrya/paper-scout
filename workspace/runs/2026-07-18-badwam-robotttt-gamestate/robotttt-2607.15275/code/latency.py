"""Per-step inference cost vs context length for the three sequence mixers.

Autoregressive decoding, batch=1, one token at a time, single CPU thread:
  - attn: KV-cached causal attention  -> per-step cost grows O(t)
  - ttt:  inner gradient step + apply -> per-step cost O(1)
  - delta: matrix-state recurrence    -> per-step cost O(1)

Weights are irrelevant for wall-clock, so models are freshly instantiated.
"""
import argparse, json, time
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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(1)
    res = {"d": args.d, "T": args.T, "layers": args.layers,
           "windows": [f"{a}-{b}" for a, b in WINDOWS], "mixers": {}}
    for mixer in ["ttt", "delta", "attn"]:
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
        print(m, {k: round(v, 3) for k, v in d_["window_ms"].items()})


if __name__ == "__main__":
    main()
