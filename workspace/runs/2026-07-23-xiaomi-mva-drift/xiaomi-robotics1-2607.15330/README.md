# Deep thread: Xiaomi-Robotics-1 (arXiv 2607.15330v2, 22 Jul 2026)

Interrogation of "Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with
over 100K Hours of Real-World Trajectories". Paper text:
`papers/robotics/xiaomi-robotics1-2607.15330.md`; PDF:
`drafts/xiaomi-robotics1-2607.15330.pdf`.

## What I did

1. Read the full paper; digitized Tables 1–3, 5 and the chart-read Fig. 8
   values into `code/extracted_tables.json`.
2. Resolved the abstract-vs-project-page benchmark discrepancy
   (`code/discrepancy_analysis.md`) using the paper's Table 5, the local RoboDojo
   paper, the arXiv v1/v2 abstract diff, and the official RoboCasa365 leaderboard.
3. Rendered and archived the scaling figures:
   `../assets/xiaomi_fig5_pretrain_scaling.png` (val action MSE vs data/params),
   `../assets/xiaomi_fig8_posttrain_scaling.png` (post-training success vs
   pre-train data/model size). Fig. 11 (auto-label examples) inspected at p.25.
4. Fermi-estimated the 100k-hour UMI bet: `code/umi_logistics.py`
   (+ `umi_logistics_output.txt`).
5. Verified the XR-0→XR-1 lineage against the open XR-0 repo source:
   `code/xr0_vs_xr1.md`, grep evidence in `code/xr0_code_evidence.txt`.
6. External checks: official RoboCasa365 leaderboard (robocasa.ai, fetched
   2026-07-23), RDT2 abstract (arXiv 2602.03310), HF org `XiaomiRobotics` via
   hf-mirror API, GitHub API for both Xiaomi repos.

## Findings (evidence-backed)

### 1. Discrepancy resolved — no eval mystery, one metric swap + one stale typo
- **RoboDojo 20.07 vs 13.93**: same Table 5, two metrics. Abstract quotes
  average **score** (partial progress 0–100): 20.07 vs 13.07. Project page quotes
  average **success rate**: 13.93% vs 8.80%. Paper body states both deltas
  (+7.00 score, +5.13pp SR). Baselines match the RoboDojo leaderboard exactly.
- **RoboCasa365 57.6 vs 57.4**: v1 abstract said 57.6; v2 abstract and Table 3
  say 57.4, and the **official leaderboard (updated 07/21/2026) lists XR-1 at
  57.4**, rank #1 — so 57.4 is benchmark-verified. The v2 intro (§1) still says
  57.6: an editing leftover, not a re-eval.

### 2. Scaling evidence is real but spans only 20k of the 100k hours
- Fig. 5 data scaling: 12.5/25/50/100% of **~20k hours** (2.5k–20k h), val action
  MSE. 2.5k/5k overfit (MSE rises mid-training); 10k→20k still descending but the
  absolute gain is small (~0.48→~0.45 MSE). Model scaling 2B/5B/10B curves nearly
  overlap — the paper itself concludes **data, not parameters, is the bottleneck**
  at this scale. Largest model: 10.5B.
- Post-training transfer (Fig. 8, bar values digitized in
  `code/extracted_tables.json`): out-of-the-box success 26% (no action
  pre-train) → 53% (2.5k h) → 56% (5k h) → 69% (10k h) → 75% (20k h); +6pp for
  the last doubling. Model size: 61% (2B) → 75% (5B) → 79% (10B).
- **Per-task wobble the paper doesn't discuss**: the monotonicity is clean only
  on the 4-task average. Per task it breaks — Shoe Storage 83% at 50% data vs
  75% at 100%; Table Organization 63% at 12.5% vs 56% at 25%; Sofa 80% (5B) vs
  77% (10B). Trial counts per bar are never stated, so these ±10pp swings are
  unquantified noise; the "monotonic scaling" claim is an average-of-4-tasks
  statement.
- "No saturation" is supported over 3 doublings (2.5k→20k h) with visibly
  moderating marginal gains (+27, +16, +6pp). The jump to the headline 100k h is
  a 5× extrapolation the paper never evaluates — the 100k-hour model appears only
  in the benchmark tables, not in any scaling curve. Aspirational, not dishonest,
  but the abstract's "strong scaling behavior" overstates the demonstrated range.

### 3. Auto-labeling: clever, fast, and completely unablated
- Pipeline: fixed-length equal-size clips (length **never stated**), Qwen3.5-27B
  captions state transitions of gripper AND interacted objects; Fig. 11 shows a
  structured format — `Gripper: <action>` + `Object: {name: state-change}`.
  Producer–consumer infra, "hundreds of requests in flight", whole 100k h labeled
  in ~2 weeks (my estimate: 10–30M clips → 10–30 captions/s sustained — needs
  O(100) high-end inference GPUs).
- **No verification step, no quality filtering, no failure-mode analysis, and NO
  ablation of label granularity vs data scale.** The paper never isolates how much
  of the gain comes from state-transition conditioning vs raw hours. The closest
  evidence: the 0%-pretrain baseline in Fig. 8 (26%) shows UMI pre-training
  matters a lot, but every pre-train point uses the same auto-labels, so
  conditioning quality and data scale are fully confounded. This is the paper's
  biggest evidentiary hole: its "crucial" contribution (abstract's word) is
  untested.

### 4. The 100k-hour UMI bet: plausible only for a company with a data army
- Fermi (`code/umi_logistics.py`): at 3–7 productive hours/operator-day,
  100k hours ≈ 14k–33k operator-days ≈ **60–270 full-time collectors for a
  year** (or 2× that for 6 months). Storage ~180–720 TB compressed video.
  Labeling fleet: O(100) GPUs for 2 weeks. This is an industrial data operation
  ~10× RDT2's 10k-hour UMI corpus (the largest *open* one) — no academic lab can
  replicate it, which is precisely the moat.
- Embodiment gap handling is thin: unify EE-frame orientation across UMI and
  robots, relative-delta EE actions, mask missing action dims, then let the 10k h
  cross-embodiment post-training (7.2k h in-house robots + 1k h instruction-UMI +
  open datasets at 0.5:0.5:0.5:8.5) do the transfer. No analysis of UMI-vs-robot
  dynamics gap (gripper compliance, wrist-vs-arm kinematics); the 26%→75% Fig. 8
  curve is the only evidence the transfer works, and it confounds data scale with
  embodiment alignment.

### 5. XR-1 is XR-0's architecture with a data recipe on top
Verified against open XR-0 code (`code/xr0_vs_xr1.md`): same Qwen3-VL + DiT MoT,
same AdaLN flow matching, **identical** Beta(1.5,1)→0.999 noise schedule and
5-step Euler integrator, XR-0's VL dataset and async recipe reused. New in XR-1:
Choice-Policies auxiliary VLM action loss (with the nice finding that letting the
DiT attend to VLM action tokens creates a copy shortcut), 2.6/5.1/10.5B variants,
and the entire UMI data engine. This is a data-scaling paper on a frozen
architecture lineage.

### 6. Reproducibility verdict
- **Verifiable today**: RoboCasa365 rank — already on the official leaderboard
  (57.4, maintainer-verified). RoboDojo baseline numbers — match the public
  leaderboard. Baseline-relative claims on public sim suites could in principle be
  re-run since RoboCasa/RoboCasa365/VLABench/RoboDojo all ship official training
  sets — but XR-1's own numbers can't be re-run: no weights (GitHub repo is
  README+PDF only, 246★; HF org has only XR-0 and the new XR-U0, checked
  2026-07-23). Notably the RoboCasa365 leaderboard marks XR-1 as **not open
  source** while 8/11 other entries are.
- **Not verifiable**: everything real-robot (75% vs π0.5's 40% at <10 h/task,
  with only 10 trials/task → ±13pp CIs), the 100k h corpus, the auto-label
  quality, the 10-min suitcase-packing demo. "Code and model checkpoints will be
  released" — as of today, nothing.

## Weak points (honest)
- Fig. 8 bar values were read off a 220 dpi render (±1–2pp); per-task means don't
  exactly reproduce the printed Average bars, so the average is likely computed
  over rollouts, not task means.
- π0.5's 40% baseline is XR-1's own OpenPi fine-tune on their private tasks — I
  did not cross-check against PI's published numbers (no public equivalent
  exists).
- Clip-length and per-label token cost are inferred, not stated; the labeling-
  fleet estimate is order-of-magnitude.
- Project-page fetch was inherited from the run brief; I verified the arXiv side
  and leaderboards myself.

## Files
- `code/extracted_tables.json` — digitized Tables 1/2/3/5 + Fig. 8 anchors +
  leaderboard cross-check
- `code/discrepancy_analysis.md` — RoboDojo metric swap + RoboCasa365 v1/v2 typo
- `code/umi_logistics.py`, `code/umi_logistics_output.txt` — Fermi estimates
- `code/xr0_vs_xr1.md`, `code/xr0_code_evidence.txt` — lineage verification
- `../assets/xiaomi_fig5_pretrain_scaling.png` — scaling curves (report-worthy)
- `../assets/xiaomi_fig8_posttrain_scaling.png` — post-training transfer bars
