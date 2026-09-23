"""全量实验编排:进程池并行跑完所有 (方法, 种子, ε) 组合。

网格:
  A × seeds {0,1,2}                    (纯评估,500 eps/种子)
  B,C,D × seeds {0,1,2} × ε=0          (60k 环境步)
  C × ε {0.05,0.15,0.3,0.5} × seeds {0,1,2}(verifier 噪声扫描,60k;0.5 为规格外扩展点)
用法: python run_all.py [--out results] [--workers 6]
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor


def job(method, seed, eps, out, eval_n=None):
    cmd = [sys.executable, "run.py", "--method", method, "--seed", str(seed),
           "--eps", str(eps), "--budget", "60000", "--out", out]
    if eval_n:
        cmd += ["--eval-n", str(eval_n)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout.strip().splitlines()[-1] if r.returncode == 0 else \
        f"FAIL {method}/s{seed}/e{eps}: {r.stderr[-300:]}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jobs = []
    for s in (0, 1, 2):
        jobs.append(("A", s, 0.0, 500))
    for m in ("B", "C", "D"):
        for s in (0, 1, 2):
            jobs.append((m, s, 0.0, None))
    for eps in (0.05, 0.15, 0.3, 0.5):
        for s in (0, 1, 2):
            jobs.append(("C", s, eps, None))

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(job, m, s, e, args.out, en) for m, s, e, en in jobs]
        for f in futs:
            print(f.result(), flush=True)


if __name__ == "__main__":
    main()
