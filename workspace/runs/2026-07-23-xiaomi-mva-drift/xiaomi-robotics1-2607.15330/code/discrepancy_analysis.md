# Discrepancy resolution: abstract vs project page vs paper tables

## RoboDojo — NOT a real discrepancy: two different metrics from the same table

RoboDojo (arXiv 2607.04434, deep-dived in `runs/2026-07-14-genception-lhtb-robodojo/`)
reports every result as **score / success rate**:

- *score* = partial task progress, 0–100 (milestones),
- *success rate* = binary task completion,
- *average* = mean over the 5 capability dimensions (Generalization, Precision,
  Long-Horizon, Memory, Open), not over tasks.

XR-1 Table 5 (paper p.15) reports XR-1 = **20.07 / 13.93%** and the prior SOTA
Hy-Embodied-0.5-VLA = **13.07 / 8.80%**. Therefore:

| Source | Quoted numbers | Metric |
|---|---|---|
| arXiv abstract / intro | 20.07 vs 13.07 | average **score** |
| Project page | 13.93 vs 8.80 | average **success rate** |

The paper body confirms both: "absolute improvements of 7% and 5.13% in terms of
average score and average success rate" (20.07−13.07 = 7.00; 13.93−8.80 = 5.13).
All baseline numbers in Table 5 match the RoboDojo leaderboard exactly, so XR-1
was evaluated on the same protocol. Editorial note: the abstract cherry-picks the
more flattering metric (score gap +53% relative vs SR gap +58% relative — actually
similar, but the absolute numbers 20.07 vs 13.07 look more dramatic than 13.93 vs 8.80).

Two caveats the abstract hides:
1. XR-1 **loses the Memory dimension** (7.81 vs 13.37 score) — it ranks first in
   only 4/5 dimensions, because it feeds no history observation. The paper admits
   this in §3.4.
2. Human teleop reference on the same benchmark is 80.42 score / 76.03% SR
   (RoboDojo paper, Finding 1). XR-1's 20.07/13.93 is SOTA but still far from human.

## RoboCasa365 — a v1→v2 correction; 57.6 is stale, 57.4 is correct

- arXiv **v1** abstract (16 Jul 2026, still shown on the /abs page fetched
  2026-07-23): "57.6% success rate on RoboCasa365".
- Local PDF is **v2** (22 Jul 2026): abstract says **57.4%**, Table 3 says **57.4**,
  but the **intro (§1) still says 57.6%** — a leftover the v2 authors missed.
- Official leaderboard https://robocasa.ai/leaderboard.html (updated 07/21/2026,
  fetched 2026-07-23) lists **Xiaomi-Robotics-1 = 57.4** (80.2 / 57.1 / 32.1 per
  split), rank #1, matching Table 3 exactly. Prior best ABot-M0.6 = 46.6 confirmed.

So the table/leaderboard value 57.4 is the trustworthy one; 57.6 survives only in
the v1 abstract and the v2 intro. Resolution: benchmark-maintainer-verified 57.4;
the 0.2pp ghost is an editing artifact, not an eval difference.

(One more micro-discrepancy: leaderboard lists ABot-M0.5 overall as 40.3 vs 40.4
in Table 3 — rounding only.)
