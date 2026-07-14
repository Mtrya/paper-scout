# GenCeption (2607.09024) — Investigation

**Question:** Is large-scale text-to-video generation actually a general-purpose
vision pretraining paradigm — and if so, what is the concrete mechanism that
turns a diffusion generator into a single-step perception model?

**Paper:** Wang et al., *Video Generation Models are General-Purpose Vision
Learners*, ECCV 2026. arXiv:2607.09024.  
**Project page:** https://genception.github.io/  
**Base model:** WAN 2.1 (open weights, open code).

---

## Core mechanism

GenCeption's bet is that a text-to-video rectified-flow DiT already contains
the spatial, temporal, and vision-language priors needed for perception, and
that the remaining work is almost entirely a *data-representation* problem
rather than an architectural one.

The adaptation is surprisingly minimal:

1. **Feed-forward reformulation.** The WAN 2.1 DiT normally takes a noisy
   latent `x_t`, a timestep `t`, and text `c`, and predicts the flow velocity
   `v`. GenCeption instead feeds the **clean latent of the input video** as
   `x_0`, fixes `t = 0`, runs **one forward pass**, then **negates the raw
   velocity output** before the VAE decoder. The paper's intuition is that
   `-v = x_0 - eps` is closer to the target-modality latent than the raw
   velocity, and empirically converges faster.

2. **Unified RGB ambient space for dense tasks.** Depth, normals,
   segmentation, dense pose, and even camera raymaps are all rendered as
   ordinary 3-channel RGB target videos. Single-channel targets are
   channel-replicated; 3D targets fill RGB naturally. Camera poses use a
   pixel-space "Rothko" raymap: ray origins live in a central crop, ray
   directions in the surrounding ring, so a 6-channel ray representation fits
   in a standard 3-channel frame.

3. **Learnable tokens for sparse tasks.** For 2D/3D keypoints, `T` extra
   tokens (one per frame) are appended to the video latent sequence. They use
   the base model's native 3D RoPE, with learnable spatial positions and
   temporally interpolated frame indices to stay inside the pretrained
   temporal range. An MLP decodes each token to a `K`-dimensional coordinate
   vector.

4. **One loss everywhere.** Everything is trained with plain L2 — in latent
   space for dense tasks, output space for sparse tasks. Task-specific
   handling is pushed to the data: depth is median-normalized per scene and
   compressed with `d' = clip(α log(d + 1), 0, 1)`; normals and segmentation
   are already in [0,1].

5. **Synthetic training recipe.** Most tasks use 7,500 synthetic human videos
   (800 RenderPeople assets, 200 CMU mocap clips, Blender render passes for
   depth/normal/mask, rigged joints for keypoints). Depth data is augmented
   with TartanAir, Virtual KITTI, and MVS Synth; referring-expression
   segmentation uses real Ref-COCO/MeViS/YouTube-VOS.

This is a real shift: the architecture (backbone + decoder + loss) is frozen
across tasks; the only thing that changes is the target "video" the model is
asked to generate and the text prompt that steers it.

---

## What the evidence shows

I parsed the paper's tables and reproduced the reported comparisons in
`code/extract_tables.py`; the tidy data and plots live in
`../assets/genception/`.

**Table 1 — Specialist vs. generalist.** The 14B generalist is competitive
with or beats task-specific models on several benchmarks:

- **Surface normals** on Sintel: 29.3 mAE vs. Lotus-2 30.3, NormalCrafter 30.7.
- **Depth on KITTI:** 0.048 AbsRel, between DepthAnything3 (0.059) and D4RT
  (0.055) / VGGT-Ω (0.041).
- **Camera pose on Sintel:** generalist RPE-T 0.018 / RPE-R 0.485, competitive
  with VGGT-Ω (0.059 / 0.407).
- **Expression-referring segmentation:** 75.8 J&F on Ref-DAVIS, 69.0 on MeViS,
  ahead of ReferEverything (75.0 / 60.3) and SAM3 variants.
- **3D human keypoints:** 71.8 MPJPE in the specialist 14B, beating Genmo
  (73.0) and TRAM (74.4).

But joint training is not free. The generalist **loses 3D keypoints entirely**
(no value reported), and the paper explicitly notes that surface normal
results move in opposite directions across benchmarks. The token-based sparse
head also disrupts pretrained DiT attention, which the authors flag as the
main architectural pain point.

**Table 2 — Pretraining comparison and scaling.** Under the same 7.5K-video
finetuning set, WAN 2.1 significantly outperforms V-JEPA and VideoMAE V2 on
depth (average AbsRel 0.122 for 1.3B, 0.093 for 14B vs. 0.281 and 0.154–0.175
for the SSL baselines). Scaling from 1.3B to 14B and from 7.5K to 8.08K
videos improves numbers. The model also reaches respectable depth performance
with far less data than D4RT (~1M frames) or VGGT-Ω (~600M frames), though
"7× to 500×" is a wide range and depends heavily on which metric/benchmark
you pick.

**Emergent behaviors.** The paper shows zero-shot sim-to-real transfer from
synthetic humans to real videos, multiple instances, and out-of-distribution
classes (animals, robots). These are qualitative/zero-shot results; the
quantitative tables above are on standard benchmarks.

---

## External signals

- **No GenCeption code, weights, or dataset released.** The project page and
  arXiv entry contain only the paper, videos, and citation. A web search for
  `GenCeption github` returns an unrelated 2024 MLLM-evaluation paper; there
  is no `Wan-Video/GenCeption` repository.

- **Base model is fully open.** I cloned `Wan-Video/Wan2.1` and inspected the
  architecture (`wan/modules/model.py`, `wan/utils/fm_solvers.py`). It is
  indeed a rectified-flow DiT with 3D RoPE, T5 text conditioning, and a
  flow-matching objective. GenCeption's `t=0` trick maps cleanly onto the
  existing `time_embedding`/`time_projection` path. See
  `code/wan_inspection.md`.

- **Related-work landscape.**
  - *Image diffusion adapters:* Marigold, GenPercept, Diception showed that
    Stable-Diffusion-like generators can be finetuned for dense perception.
    GenCeption's novelty is doing this natively in video space, with a single
    feed-forward pass, across dense + sparse tasks.
  - *Video diffusion adapters:* DepthCrafter, NormalCrafter, Geo4D,
    DiffusionRenderer, ReferEverything each tackle one video task (depth,
    normals, 4D reconstruction, inverse/forward rendering, referring
    segmentation). GenCeption unifies them under one architecture.
  - *Concurrent work:* Wiedemer et al. ("Video models are zero-shot learners
    and reasoners") probes video generators without training and gets
    qualitative evidence of reusable priors. Vision Banana ("Image generators
    are generalist vision learners") does the same in the image domain.
    GenCeption distinguishes itself with post-training, a feed-forward
    single-pass design, and quantitative benchmarks.
  - *SSL baselines:* V-JEPA and VideoMAE V2 lack native text steering and, in
    this paper's setup, underperform the generative backbone on depth.

---

## Limitations and blockers

1. **Reproducibility gap.** Without released weights, inference code, or the
   synthetic data pipeline, the numbers are not independently verifiable yet.
   The 14B model also requires ~43 GB VRAM for an 81-frame clip, so even if
   weights appeared, reproduction would be expensive.

2. **Training is costly.** 15,000 steps at batch size 64 on 256 v6e TPUs is
   not a recipe most labs can run.

3. **Joint-training regressions.** The generalist loses the 3D-keypoint task
   and shows mixed results on normals. The paper is honest about this, but it
   undercuts the "one model does everything" framing: sparse coordinate
   regression still needs an architectural add-on that interferes with the
   pretrained backbone.

4. **Data questions.** Most tasks are trained purely on synthetic humans. The
   strong real-world numbers are impressive, but we do not know how much of
   the performance comes from the generative prior versus the synthetic-data
   rendering quality, camera distributions, or benchmark overlap. The
   referring-segmentation task does use real data, so it is not a fully
   uniform recipe.

5. **Metric apples-to-oranges.** The "7×–500× less data" claim compares a
   model trained only on synthetic data against specialists trained on much
   larger real corpora. That is a legitimate point about efficiency, but it
   is not the same as saying the same pipeline beats specialists when both
   see the same real data.

---

## Takeaway

GenCeption is a crisp existence proof that a large video-generative backbone
*can* be repurposed into a generalist perception model with very little
architectural surgery. The mechanism — clean-latent input, `t=0`
conditioning, negated velocity, RGB target space, learnable tokens, one L2
loss — is concrete and elegant. The empirical results are strong enough to
take seriously, especially the comparison with V-JEPA/VideoMAE and the
single-model depth/pose numbers.

At the same time, the paper is best read as a *paradigm demonstration*, not
as a finished universal vision model. The sparse-task head still breaks the
"no architectural change" story, joint training hurts some tasks, and the
absence of released artifacts means the community cannot yet stress-test the
claims. The most important open question is whether these priors are
*uniquely* available in video generators, or whether they simply reflect the
scale and multimodality of the underlying data and compute — a question that
will only be settled once independent reproductions and ablations appear.

---

## Preserved evidence

- `code/extract_tables.py` — parses `paper.html`, extracts Tables 1 and 2,
  writes tidy CSV/JSON, and generates scaling and SOTA plots.
- `code/raymap_and_depth_layout.py` — derives the Rothko raymap packing and
  the median-normalized log-depth mapping in code and figures.
- `code/wan_inspection.md` — notes from inspecting the open WAN 2.1 base
  repository.
- `paper.html` — local copy of the arXiv HTML version used for table
  extraction.
- `wan2.1-base/` — shallow clone of the WAN 2.1 repository (read-only).
- `../assets/genception/` — generated CSVs, JSONs, PNG plots, and the paper
  figures already present there.
