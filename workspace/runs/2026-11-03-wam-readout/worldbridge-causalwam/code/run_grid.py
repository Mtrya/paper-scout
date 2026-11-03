"""Run the synthetic WAM grid: data condition x action-head sparsity x seed."""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "grid")
os.makedirs(OUT, exist_ok=True)

CONDS = {
    "base": ["--ood_video", "0", "--ood_demos", "0"],
    "vid": ["--ood_video", "48", "--ood_demos", "0"],
    "vid1": ["--ood_video", "48", "--ood_demos", "1", "--demo_steps", "80"],
    "vid5": ["--ood_video", "48", "--ood_demos", "5", "--demo_steps", "80"],
}
TOPK = [36, 1, 2, 4, 8]


def run(cond, topk, seed, extra=None):
    tag = f"{cond}_k{topk}_s{seed}"
    out = os.path.join(OUT, tag + ".json")
    if os.path.exists(out):
        print("skip", tag)
        return
    cmd = [sys.executable, os.path.join(HERE, "synth_wam.py"),
           "--seed", str(seed), "--topk", str(topk), "--train_steps", "1500",
           "--out", out] + CONDS[cond] + (extra or [])
    print("run", tag, flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    if mode == "bench":
        run("base", 36, 0)
        run("vid", 2, 0)
        return
    if mode == "main":
        for cond in ["base", "vid", "vid1"]:
            for topk in [36, 1, 2, 4]:
                for seed in [0, 1]:
                    run(cond, topk, seed)
        for topk in [36, 1, 2, 4]:
            run("vid5", topk, 0)
        return
    if mode == "k1seeds":
        for cond in ["base", "vid", "vid1"]:
            for seed in range(5):
                run(cond, 1, seed)
        return
    if mode == "extra":
        # optional ablations
        for topk in [36, 2, 4]:
            run("base", topk, 0, ["--no_video"])
        return
    raise SystemExit("unknown mode")


if __name__ == "__main__":
    main()
