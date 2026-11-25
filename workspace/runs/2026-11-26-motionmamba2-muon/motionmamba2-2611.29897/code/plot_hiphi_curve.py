"""HiPHI 到目标距离曲线图:frames_988 群体均值 + 参考线(随机游走/线性漂移)。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

curves_json = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "window_curve_988.json"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    Path(__file__).parent.parent.parent / "assets" / "mm2-hiphi-curve.png"

d = json.loads(curves_json.read_text())
g = d["distance_to_goal"]
mean = g["mean_curve"]
fps = 25
t = [i / fps for i in range(len(mean))]
nwin = len(g["per_win_mean_cm"])
win_t = [(i + 1) * 52 / fps for i in range(nwin)]
win_d = g["per_win_mean_cm"]

# 参考线:从首窗均值出发的 sqrt(t) 与 t 增长(归一到曲线中点,仅示形状)
import numpy as np
t_mid = t[len(t) // 2]
mid = mean[len(mean) // 2]
sqrt_ref = [mid * (x / t_mid) ** 0.5 for x in t]
lin_ref = [mid * (x / t_mid) for x in t]

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
ax.plot(t, mean, "-", color="#1a6faf", lw=1.8,
        label=f"mean distance to goal ({g['n_rollouts']} rollouts, 48 families)")
ax.plot(t, sqrt_ref, ":", color="#7f8c8d", lw=1.3, label=r"random-walk reference $\propto\sqrt{t}$")
ax.plot(t, lin_ref, "--", color="#c0392b", lw=1.3, label=r"linear-drift reference $\propto t$")
ax.scatter(win_t, win_d, s=18, color="#1a6faf", zorder=3, label="per-window mean")
ax.set_xlabel("physical time (s)")
ax.set_ylabel("root distance to goal (cm)")
ax.set_title(f"HiPHI 39.52 s: distance-to-goal curve (final {g['final_cm']:.1f} cm, "
             f"tail slope {g['tail_loglog_slope']:.2f} in log-log)")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout()
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png)
print("written", out_png, out_png.stat().st_size)
