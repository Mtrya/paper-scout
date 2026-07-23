#!/usr/bin/env python3
"""Fermi estimates for Xiaomi-Robotics-1's 100k-hour UMI data bet.

All numbers are derived from the paper text + public references; assumptions are
marked and tunable at the bottom. Run: python3 umi_logistics.py
"""

HOURS_TOTAL = 100_000  # paper: "over 100,000 hours of real-world manipulation trajectories"

# --- 1. Collection logistics -------------------------------------------------
# UMI (Chi et al. 2024, arXiv 2402.10329) is a hand-held gripper + GoPro; an
# operator collects demonstrations in the wild. Sustained *productive* collection
# per operator-day is far below 8h (travel, setup, battery, fatigue).
for productive_h_per_day, label in [(3, "pessimistic"), (5, "mid"), (7, "optimistic")]:
    operator_days = HOURS_TOTAL / productive_h_per_day
    ops_1yr = operator_days / 250          # 250 working days/yr
    ops_6mo = operator_days / 125
    print(f"[collect:{label:11s}] {productive_h_per_day} productive h/op-day -> "
          f"{operator_days:,.0f} operator-days = {ops_1yr:,.0f} ops x 1yr "
          f"or {ops_6mo:,.0f} ops x 6mo")

# --- 2. Auto-labeling throughput --------------------------------------------
# Paper: Qwen3.5-27B captions fixed-length clips; producer-consumer pipeline,
# "hundreds of captioning requests in flight"; whole corpus labeled in ~2 weeks.
LABEL_DAYS = 14
for clip_s in (10, 20, 30):  # clip length NOT stated in the paper
    clips = HOURS_TOTAL * 3600 / clip_s
    rps = clips / (LABEL_DAYS * 86400)
    print(f"[label:clip={clip_s:2d}s] {clips/1e6:5.1f}M clips -> {rps:6.1f} captions/s "
          f"sustained for 14 days")
# At ~1-3 s per 27B VLM caption on modern GPUs with continuous batching,
# hundreds of in-flight requests -> O(100) captions/s is plausible but implies
# a serious inference fleet (order 10^2 H100-class GPUs).

# --- 3. Storage --------------------------------------------------------------
# Egocentric video, assume single RGB stream 1080p30 H.264 ~8 Mbps + pose track.
for mbps in (4, 8, 16):
    tb = HOURS_TOTAL * 3600 * mbps / 8 / 1e6
    print(f"[storage:{mbps:2d}Mbps] ~{tb:,.0f} TB video (excl. pose/actions, raw frames)")

# --- 4. Scale context --------------------------------------------------------
print()
print("Context: RDT2 (arXiv 2602.03310, Feb 2026), the largest *open* UMI corpus,")
print("is 10,000+ hours -> XR-1 pre-train corpus is ~10x that, and ~50x OXE-scale")
print("teleop corpora. But the paper's scaling EVIDENCE (Fig 5) only spans")
print("2.5k-20k hours (12.5%-100% of ~20k h); 20k->100k is a 5x unvalidated leap.")
