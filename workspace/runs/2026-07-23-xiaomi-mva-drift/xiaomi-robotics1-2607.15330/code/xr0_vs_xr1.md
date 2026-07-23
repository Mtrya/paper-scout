# XR-0 → XR-1 architectural lineage (verified against open XR-0 code)

XR-1 paper never says "we build on XR-0" in one sentence, but the evidence is
unambiguous: XR-1 is a scaled evolution of the open-sourced XR-0 stack.

Evidence from `github.com/XiaomiRobotics/Xiaomi-Robotics-0` (604★, created
2026-02-11, Apache-2.0), file `xr0/mibot/models/VLA/XR0.py` (903 lines, fetched
2026-07-23; grep dump in `xr0_code_evidence.txt`):

| Component | XR-0 (open code) | XR-1 (paper) | Same? |
|---|---|---|---|
| VLM backbone | `Qwen3VLForConditionalGeneration` (Qwen3-VL) | Qwen3-VL | ✓ |
| Action head | DiT cross-attending to VLM KV cache, AdaLN timestep conditioning | DiT on VLM KV cache, AdaLN (adaLN-Zero) | ✓ |
| Objective | flow matching, velocity prediction | flow matching | ✓ |
| Timestep sampling | `Beta(1.5, 1.0)` rescaled to (0, 0.999) | u ~ Beta(1.5, 1), τ=(1−u)·0.999 | ✓ verbatim |
| Inference | Euler integration, `num_steps = 5` | 5-step Euler, Δτ=0.2 | ✓ verbatim |
| Choice Policies auxiliary loss on VLM | **absent** (no "choice" in code) | present, K candidates, winner-takes-all L1 + score regression [Qi et al. 2512.25072] | ✗ NEW |
| Model sizes | 4.7B single size | 2.6B / 5.1B / 10.5B variants (Table 1) | scaled |
| Real-time recipe | asynchronous execution ("real-time inference") | "asynchronous training recipe proposed in [8]" cited for fine-tuning | ✓ inherited |
| VL co-training | pre-trained on cross-embodiment + VL data | co-trained on "high-quality vision-language dataset curated in our previous work [8]" (= XR-0), λ=0.1, VL:UMI = 1:9 | ✓ inherited |

XR-0 itself appears in XR-1's RoboDojo Table 5 as a baseline (6.93 / 4.18%), and
XR-1 cites XR-0 [8] for: the VL dataset, the asynchronous fine-tuning recipe, and
as a downstream fine-tuning baseline.

Conclusion: XR-1 = XR-0 architecture family (Qwen3-VL MoT + flow-matching DiT,
identical noise schedule and integrator) + Choice-Policies auxiliary supervision
+ deliberate size variants, differentiated almost entirely by the DATA recipe
(100k h UMI pre-train with auto-labeled state transitions → 10k h cross-embodiment
post-train). The paper's contribution is a data-scaling story on a frozen
architecture lineage, not an architecture story.

Practical consequence for reproducibility: the open XR-0 repo + checkpoints
(XiaomiRobotics/Xiaomi-Robotics-0-Pretrain on HF) implement the same
architecture, so an outsider can reproduce the *architecture and training
objective* (minus Choice Policies) but NOT the data mixture that the paper's own
Fig. 8 identifies as the dominant factor (26% → 75% from pre-training data alone).
