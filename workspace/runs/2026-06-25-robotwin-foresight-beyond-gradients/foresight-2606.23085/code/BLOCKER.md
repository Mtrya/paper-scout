# Code Availability Blocker — Foresight (arXiv 2606.23085)

## What was requested
The deep-dive task asked to locate, clone, and inspect the official Foresight code
repository, preserving key implementation snippets.

## What was found
**No public official Foresight code repository exists as of 2026-06-25.**

Search steps taken:

1. **Paper and project page.** The paper lists "Project Page: Foresight.github.io".
   The live project page is `https://haoranzhangumich.github.io/Forsight_web/`.
   It contains only a static HTML summary of the paper, qualitative videos, and
   result tables. There is no "Code" button, no GitHub link to source, and no
   downloadable artifact.

2. **GitHub search.**
   - `https://api.github.com/search/repositories?q=Foresight+failure+detection+robotic+manipulation` → 0 results.
   - `https://api.github.com/search/repositories?q=foresight+action-conditioned+world+model` → 0 results.
   - `https://api.github.com/search/repositories?q=HaoranZhang+Foresight` → 0 results.
   - Author profile `HaoranZhangumich` exists, but the only Foresight-related
     repository is `Forsight_web` (the static project page). Cloning it failed
     repeatedly with TLS/timeout errors, and FetchURL confirmed it contains only
     rendered HTML/markdown, no implementation.

3. **Web search.** Queries on the title plus "github" or "code" return only
   arXiv mirrors, daily-paper aggregators (DailyArxiv, robot-paper-daily), and
   third-party review sites (Moonlight, ChatPaper). None host source code.

4. **Hugging Face / arXiv code metadata.** The arXiv abstract page has no
   associated code entry. Hugging Face Papers has the arXiv entry but no
   linked repository.

## Why this is a meaningful blocker
Foresight's value is in the integration: how exactly the frozen V-JEPA 2-AC
encoder and trained predictor are loaded per benchmark, how action chunks are
aligned across policies with different action dimensions and replan intervals,
how the causal Transformer detector is trained with per-timestep BCE and
early-detection weighting, and how the FCP threshold is padded across variable-
length rollouts. Without the official repository, these integration details
cannot be verified against the authors' implementation.

## What external signal was preserved instead
The backbone on which Foresight is built — **V-JEPA 2-AC** — has an official,
public implementation at `https://github.com/facebookresearch/vjepa2`. That
repository was cloned and inspected. Key files preserved under `code/`:

- `vjepa2_ac_predictor_arch.py` — action-conditioned predictor architecture.
- `vjepa2_ac_causal_mask.py` — frame-causal attention mask.
- `vjepa2_ac_training_loop.py` — teacher-forcing + autoregressive training objective.
- `vjepa2_inference_wrapper.py` — inference-time encode/predict wrapper.
- `foresight_detector_pseudocode.py` — faithful reconstruction of the detector
  and FCP thresholding from the paper.
- `diagnostic_shape_trace.py` and `diagnostic_shape_trace_pure.py` — shape/sanity
checks for the full pipeline.

These snippets provide the strongest feasible external signal: the actual code
that produces the `z_t^p` latent representation Foresight consumes.

## Baseline code that does exist (noted for reference)
- FAIL-Detect: `https://github.com/CXU-TRI/FAIL-Detect`
- SAFE: `https://github.com/vla-safe/SAFE`
- Gauge: `https://github.com/autoinspection-classification/GaugeFailClassification`

These were not cloned because the task scope is Foresight; their existence
confirms the field is starting to release failure-detection code, but Foresight
itself has not been released as of this run.
