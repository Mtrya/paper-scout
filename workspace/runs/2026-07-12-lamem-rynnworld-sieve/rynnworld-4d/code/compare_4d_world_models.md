# Comparison: RynnWorld-4D and nearby 4D world-model approaches

This table situates RynnWorld-4D against the methods the paper cites as closest
competitors / predecessors, plus the broader world-action-model landscape.

| Aspect | RynnWorld-4D (this paper) | TesserAct (Zhen et al., 2025) | 4DNeX (Chen et al., 2025) | Cosmos 3 (NVIDIA, 2026) | LingBot-VA (Robbyant, 2026) |
|--------|---------------------------|------------------------------|---------------------------|-------------------------|----------------------------|
| **Core representation** | RGB-D-Flow (2D-aligned latents) | RGB-Depth-Normal (2D-aligned latents) | 6D video = RGB+XYZ pointmap | Omnimodal tokens (lang / image / video / audio / action) | Causal video-action tokens |
| **Backbone** | Wan2.2-TI2V-5B tri-branch DiT | CogVideoX-5b-I2V fine-tuned | Wan2.1-I2V-14B + LoRA | Mixture-of-Transformers (MoT) | Wan-Video family |
| **How geometry/motion is made explicit** | Depth per frame + optical flow → back-project to 3D scene flow | Depth + normals → optimize a 4D mesh/GS scene | XYZ pointmap per pixel | Learned action pseudo-actions + world tokens | Action-conditioned future video |
| **Cross-modal fusion** | Sparse Joint Cross-Modal Attention every 3 layers; per-branch FFNs; shared text K/V | Joint RGB-D-N generation in one model; post-hoc 4D reconstruction | Width-wise fusion of RGB and XYZ tokens | Dual-stream joint attention across all modalities | Action scale/shift per block |
| **Training data scale** | 254.4 M pseudo-labeled frames (human egocentric + robot data) | Synthetic + real RGB-DN videos (smaller, exact size not public) | 4DNeX-10M pseudo-4D samples | Huge synthetic corpus (SDG-*) + real video/action | 60 k hr real robot data (LingBot-VLA claim); LingBot-VA scale not clear |
| **Policy / control** | **Inverse-dynamics head on internal 4D latents** (single forward pass, 4-step flow matching, 9 Hz closed-loop) | Learns inverse dynamics over generated RGB-DN sequences; requires decoding futures | No policy; novel-view rendering only | World-action post-training on DROID etc.; next-token / denoising | World-action model for generalist robot control |
| **Closed-loop real-robot freq** | ~9 Hz effective (1.1 s world-model pass, 10-action chunking) | Not reported as real-time closed-loop | N/A | Not reported for bimanual dexterous tasks | 150 Hz claimed (LingBot-VLA), 40+ FPS for LingBot-World-Fast |
| **Key empirical claim** | SOTA on 6 real-world bimanual dexterous tasks; strong depth/flow metrics (δ₁=0.610, AEPE=0.170) | Strong 4D reconstruction and policy on synthetic + real tasks | Strong single-image-to-4D generation; no robot eval | SOTA on DROID / RoboArena etc. | Strong on GM-100 dual-arm benchmark |
| **What's different** | Keeps 2D-aligned video latent space; explicit **flow** for kinetic cue; policy reads **mid-diffusion internal features** without waiting for decoded video | Uses **normals** instead of flow; reconstructs explicit 4D surface after generation | Generates explicit dynamic point clouds; no action head | Generalist omnimodal pre-training; action as tokens | Action-conditioned video generation as policy training data / control |

## Take-away for the report

RynnWorld-4D is not the first paper to augment video generation with depth, but it
is the most explicit about **optical flow as a kinetic signal** and about turning
the world model into a **real-time inverse-dynamics encoder**.  The engineering
choice that matters most is not the tri-branch architecture per se (TesserAct and
4DNeX also add geometry branches), but the combination of:

1. co-generating **flow** alongside RGB-D,
2. fusing branches with **sparse, position-aware joint attention**, and
3. reading the resulting 4D latents with a **lightweight flow-matching policy head**
   in a single forward pass.

That third point is what lets the system close the loop on a real robot at a
plausible control frequency, whereas TesserAct and 4DNeX are mainly evaluated as
future-predictors / scene reconstructors.
