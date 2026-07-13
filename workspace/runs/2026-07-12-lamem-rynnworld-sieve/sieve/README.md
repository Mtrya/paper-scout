# SIEVE deep-dive — structure-aware selection for VLA imitation learning

**Paper:** SIEVE: Structure-Aware Data Selection for Imitation Learning with VLA Models (arXiv 2607.06442)  
**Code:** https://github.com/ChangtiWu/SIEVE, cloned to `code/sieve`  
**Run packet:** `runs/2026-07-12-lamem-rynnworld-sieve/sieve/`  
**Author of this note:** Paper Scout subagent

---

## 1. What question are we trying to answer?

SIEVE claims that VLA imitation-learning datasets should be selected not by trajectory-level similarity or state-action scores, but by exposing the model to reusable **visuo-motor primitives** and the **transition interfaces** between them. The reported result is striking: on Bridge-V2, a 50%-data / 50%-steps SIEVE subset outperforms training on the full dataset.

Our question is: **does the released implementation actually do what the paper says, and can we make the mechanism concrete enough to argue about it?** We read the paper end-to-end, traced the official scripts, and built a standalone probe that reimplements the core two-stage selection algorithm on synthetic data.

---

## 2. Core method (from the paper)

SIEVE has three stages:

1. **Primitive Discovery**
   - Segment each trajectory at physically grounded interaction boundaries (gripper / hand state flips, with a 5-frame confirmation rule).
   - Extract V-JEPA2 features for each segment (8 uniformly sampled frames; concatenate start/mid/end frame embeddings).
   - Reduce to 256-D with PCA, then run MiniBatchKMeans.
   - Choose K automatically on a 10% episode subset by maximizing `(1 − J) · log R`, where `J` is the median pairwise Jaccard similarity of per-trajectory cluster sets and `R` is the median cross-trajectory cluster reuse.

2. **Structural Exposure Allocation (Stage A)**
   - Each trajectory becomes an ordered primitive sequence = a **composition pattern**.
   - Adjacent primitives form a **transition**; single-primitive trajectories get a terminal transition `(c → ∅)`.
   - Allocate a fixed episode budget greedily to maximize
     ```
     F(B) = Σ_c w_c log(1 + n_c(B)) + Σ_e w_e log(1 + n_e(B))
     ```
     where `w_c = q_c / |P|`, `w_e = q_e / |P|`, and `q_c/q_e` count how many unique patterns contain primitive/transition `c/e`. The log gives diminishing returns, encouraging broader coverage over repeated reinforcement.

3. **Learning-Friendly Trajectory Selection (Stage B)**
   - Within each composition-pattern bucket, take the trajectory whose concatenated segment representation has the highest average cosine similarity to the others (the **medoid**), then keep the `B[ℓ]` trajectories closest to it.

The result is a subset that is structurally diverse *and* consists of central, stable realizations of each pattern.

---

## 3. Code inspection: what the implementation does, and where it diverges

We traced `scripts/gripper_pose_seg.py`, `feature.py`, `dim_reduction.py`, `cluster.py`, `select.py`, and `save_as_lerobot.py`.

### 3.1 Pipeline map

| Paper stage | Script | Key data object | Notes |
|---|---|---|---|
| Trajectory segmentation | `gripper_pose_seg.py` | `segments.jsonl` | One line per segment start; records `dataset_name`, `chunk`, `episode_index`, `frame_index`, `index`. |
| Segment representation | `feature.py` | `worker*.zarr/emb` | Per-worker Zarr shards; V-JEPA features are `[start; mid; end]` concatenated. |
| PCA | `dim_reduction.py` | `worker*.zarr/emb` | IncrementalPCA → 256-D, `float16`. |
| Clustering / K search | `cluster.py` | `worker*.zarr/cluster_id` | K selected on 10% subset, then full MiniBatchKMeans; `cluster_id` stored 1-based. |
| Selection | `select.py` | `selected_episodes.jsonl` | Stage A + Stage B (`random`/`centre`/`unsim`). |
| Export | `save_as_lerobot.py` | LeRobot dataset | `direct` copy mode recommended; `lerobot` mode uses `lerobot-edit-dataset` when available. |

### 3.2 Findings that are not obvious from the paper text

**A. Segmentation is implemented as gripper-/hand-command flips; the pose-reversal helper is dead code.**

The paper says trajectories are segmented using "end effector (gripper/dexterous-hand) state flips (i.e., grasp/release flips)". This wording is consistent with treating open/close commands as the boundary signal. The code's default `gripper` mode does exactly that: it thresholds the gripper command column (`action[:, 6]` or `action[:, 7]`) and confirms the new state for `confirm_frames` frames.

What is not obvious from the paper is that the code also computes a smoothed 6-D end-effector pose and defines a `has_confirmed_sign_flip` helper for xyz motion reversals, but **that helper is never called in either `gripper` or `dexterous_hand` mode**. The operational boundary is therefore a *control-signal* flip, not a physical motion reversal. We preserved a patch that would optionally activate the xyz sign-flip boundary.

**B. Stage-A weights are normalized by the mean q, not by |P| as the paper states.**

Equation 12 in the paper says

```
w_c = q_c / |P| ,   w_e = q_e / |P|
```

But `scripts/select.py::compute_feature_weights` computes

```python
mean_q_c = sum(q_c.values()) / len(q_c)
w_c = {k: v / mean_q_c for k, v in q_c.items()}
```

(and analogously for transitions). This forces the *average* primitive weight and the *average* transition weight each to be 1.0, rather than scaling both by `1/|P|`. Because the number of primitives and transitions differs, the relative importance of the primitive term vs. the transition term is different from the paper's formula. The ablation in Table 4 says removing transitions hurts more than removing primitives; the code's mean-normalization may be amplifying whichever term has fewer unique items.

We preserved a patch (`select_paper_weights.patch`) that restores the paper's `|P|` normalization.

**C. The terminal transition is encoded as `(c, -1)`, not `(c, ∅)`.**

For single-primitive trajectories the paper introduces a null state `∅`. The code uses the integer pair `(c, -1)` as a sentinel. Functionally equivalent, but it means the transition vocabulary includes negative-index sentinels that are unique per primitive.

**D. Medoid selection uses cosine similarity including self-similarity.**

The paper defines the medoid as `argmax_i Σ_{j≠i} cos(x_i, x_j)`. The code computes `unit @ unit.T` and sums over all columns, which adds 1.0 for the self term. Since adding a constant does not change the argmax, the result is identical; this is a harmless implementation shortcut.

**E. The default Stage-B mode in Python is `random`, but the shell wrapper defaults to `centre`.**

`scripts/select.py` defaults `--stageb-mode` to `random`; `select.sh` overrides it to `centre`. If a user calls the Python script directly without reading the shell wrapper, they will not get the SIEVE medoid selection. This is a minor footgun.

**F. The K-search heuristic uses `min_avg_cluster_size` to bound the maximum K.**

The paper mentions "uniformly evaluate 20 candidate values of K within a reasonable search range". The code sets `max_k = total_n // min_avg_cluster_size` (default 2000 for Bridge/Fractal, 1000 for RoboCasa). This is a sensible engineering choice that prevents tiny clusters, but it is not described in the paper.

---

## 4. Standalone probe: `code/sieve_probe.py`

Because running V-JEPA on real video is heavy and depends on a LeRobot dataset, we built a minimal, dependency-light probe that reproduces the entire selection logic on synthetic data. It requires only `numpy` and the standard library.

### What the probe does

1. Generates 120 synthetic episodes with binary gripper commands and confirmed-flip segmentation.
2. Assigns each segment a primitive ID by adding noise to a small set of centroids.
3. Derives composition patterns and transitions.
4. Runs Stage-A greedy allocation under **both** the official code's mean-normalized weights and the paper's `|P|` weights.
5. Runs Stage-B medoid selection for both allocations.
6. Reports pattern-length distribution, pattern-frequency Gini coefficient, and top selected patterns.

### Running it

```bash
cd runs/2026-07-12-lamem-rynnworld-sieve/sieve/code
python3 sieve_probe.py --output ../probe_result.json
python3 visualize_probe.py
```

### Probe output (default seed)

```
Synthetic episodes : 120
Selection budget   : 60
Unique patterns    : 24

Weight normalization comparison
  code  mean primitive weight : 1.000
  paper mean primitive weight : 0.219
  code  mean transition weight: 1.000
  paper mean transition weight: 0.083

[code] selection
  active patterns          : 20
  unique patterns retained : 20/24
  single-primitive episodes: 3/7
  pattern Gini before/after: 0.463 -> 0.412
  top-3 before             : [((1, 5), 16), ((3, 7), 14), ((4, 8), 13)]
  top-3 after              : [((2, 6, 2, 6), 8), ((3, 7, 3), 7), ((4, 8, 4, 8), 7)]
```

The paper norm gives the same allocation on this synthetic draw because the primitive/transition weight ratios happen to align, but the absolute weights are very different. The important algorithmic behavior is visible in both cases:

- **The selected subset is more structurally diverse.** Pattern-frequency Gini drops from 0.463 to 0.412.
- **Short, dominant patterns are deprioritized.** The top-3 patterns before selection are 2-primitive; after selection the top patterns are 3- and 4-primitive, exposing more transitions.
- **Medoid selection picks central realizations.** Within each chosen pattern bucket, the probe keeps trajectories closest to the pattern medoid in cosine space.

The generated figure `figures/probe_redistribution.png` shows this redistribution across pattern lengths, sorted frequencies, and the top selected patterns for both normalizations.

---

## 5. What we learned

1. **The SIEVE mechanism is concrete and mostly matches the paper**, but the official implementation contains two implementation details worth noting:
   - segmentation is driven by gripper/hand command flips, while an xyz pose-reversal helper exists but is unused;
   - Stage-A weights are mean-normalized within each term, not divided by `|P|` as Eq. 12 states.

2. **The structure-aware objective does what it claims on toy data.** Even without real video features, the greedy log-utility allocation redistributes the selection budget away from a few dominant short patterns toward longer, more diverse composition patterns. This is the same qualitative effect shown in the paper's Figure 3.

3. **Medoid selection is a clean, cheap BC-friendly proxy.** Replacing it with "most dissimilar" selection (the paper's Table 4 ablation) would likely hurt because outliers within a pattern introduce inconsistent action supervision. The code implements this correctly.

4. **The code is production-oriented.** Multi-worker Zarr sharding, resume logic, per-dataset routing, and LeRobot export fallbacks are all well-engineered but not discussed in the paper.

---

## 6. Suggested claims for the main report

- SIEVE's selection behavior can be reproduced from the paper's equations: greedy log-utility allocation over composition patterns naturally flattens the pattern distribution and favors multi-primitive sequences.
- The released implementation is faithful in spirit but contains two implementation details worth flagging: (a) segmentation is driven by gripper/hand command flips, while an xyz pose-reversal helper exists but is not used; (b) Stage-A weights are mean-normalized per term rather than scaled by `1/|P|` as Eq. 12 indicates. These should be noted when comparing to the paper's formulas.
- The ablation result that transitions matter more than primitives (Table 4) is worth revisiting with the paper's exact weight normalization, because the code's mean-normalization can change the relative influence of the two terms.
- The SIEVE pipeline is modular enough that the heavy V-JEPA/PCA stages can be bypassed for diagnostics: a synthetic probe is sufficient to demonstrate the structure-aware redistribution and medoid-selection behavior.

---

## 7. Artifacts in this packet

```
runs/2026-07-12-lamem-rynnworld-sieve/sieve/
├── README.md                       # this file
├── code/
│   ├── sieve_probe.py              # standalone selection probe
│   └── visualize_probe.py          # matplotlib visualization
├── patches/
│   ├── select_paper_weights.patch  # align Stage-A weights with paper Eq. 12
│   └── segmentation_pose_sign.patch # optionally add xyz sign-flip boundaries
├── figures/
│   └── probe_redistribution.png    # synthetic before/after pattern distributions
└── probe_result.json               # full probe output
```
