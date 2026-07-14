# WAN 2.1 base-model inspection note

Cloned from https://github.com/Wan-Video/Wan2.1.git (shallow clone in
`wan2.1-base/`) to understand the backbone GenCeption adapts.

Key observations relevant to GenCeption's claims:

- **Architecture**: WanModel is a DiT with 3D patch embedding
  (`patch_size=(1,2,2)`), 3D RoPE (`rope_params` + `rope_apply` in
  `wan/modules/model.py`), causal-style self-attention windows, and a T5-XXL
  text encoder. Input video latents are 16-channel, spatially downsampled 8x,
  temporally 4x — matching GenCeption's "480x832, 81 frames, temporal factor 4,
  spatial factor 8" description.

- **Rectified-flow conditioning**: The model takes a timestep `t` and produces
  sinusoidal time embeddings (`sinusoidal_embedding_1d`) projected into
  AdaLN-like scale/shift gates (`time_projection` produces a `[6, dim]` tensor
  fed to each block). GenCeption fixes `t=0`, which means the model sees the
  "noise-free" time embedding and the input latent is the clean source video
  rather than Gaussian noise.

- **Velocity output**: WAN is trained with a rectified-flow / flow-matching
  objective (solvers in `wan/utils/fm_solvers.py`). The DiT predicts the
  velocity field `v`. GenCeption negates this velocity output and treats
  `-v ≈ x_0 - eps` as the latent of the target modality video.

- **No released GenCeption checkpoint**: The repo contains only the base
  text-to-video weights and generation pipelines. There is no post-training
  perception head, no task prompts, and no synthetic-data rendering code.
  GenCeption-specific weights/data have not been released as of this
  inspection.

- **Patch implications**: Because the decoder head is a simple
  `Head(dim, out_dim, patch_size)` that unpatchifies back to latents,
  GenCeption can keep the exact same head/decoder for all dense tasks as long
  as targets are formatted into the same latent shape. This aligns with the
  paper's "single decoder, task specification via data representation"
  design.

No code changes were made to the cloned repo; it was read only.
