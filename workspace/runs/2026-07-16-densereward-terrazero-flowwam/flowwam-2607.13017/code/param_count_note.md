# Static parameter-count estimates

These were computed by reading the architecture definitions in `action_dit.py`
and `wan_video_dit_dual_stream.py`. The harness lacks PyTorch, so the counts
were not obtained by instantiating the modules.

## Action Expert (`ActionExpertIDM`)

Defaults from the code and `train.sh`:
- dim = 1024
- num_layers = 30
- num_heads = 16
- ffn_dim = 4096
- action_dim = 14
- text_context_dim = 4096
- joint_state_dim = 14

Approximate breakdown:
- action embedding: 14 * 1024 ≈ 14 k
- proprio projection to text: 14 * 4096 ≈ 57 k
- time MLPs: Linear(256,1024) + Linear(1024,1024) ≈ 1.1 M + 1.0 M; plus
  Linear(1024, 6*1024) projection ≈ 6.3 M
- cond noise-level MLP (video_dim=3072): Linear(256,3072)+Linear(3072,3072)
  ≈ 0.8 M + 9.4 M
- text projection: LayerNorm(4096) + Linear(4096,1024) ≈ 4.2 M
- per-layer block (x30):
  - self-attn: 4 * Linear(1024,1024) ≈ 4.2 M
  - video cross-attn: Linear(1024,1024) + 2*Linear(3072,1024) + Linear(1024,1024)
    ≈ 1.0 M + 6.3 M + 1.0 M = 8.3 M
  - text cross-attn: 4 * Linear(1024,1024) ≈ 4.2 M
  - FFN: Linear(1024,4096) + Linear(4096,1024) ≈ 4.2 M + 4.2 M = 8.4 M
  - norms + modulation ≈ small (~20 k)
  - total per block ≈ 25.1 M
- 30 blocks ≈ 753 M
- head: Linear(1024,14) + modulation ≈ 14 k

Total ≈ **780 M parameters**, matching the paper's Appendix A.1 claim.

## Flow stream (`FlowStreamModule`)

The module is a deep copy of the pretrained DiT's patch embedding and output
head, plus a learnable stream embedding:
- `flow_patch_embedding`: same shape as `dit.patch_embedding`
- `flow_head`: same shape as `dit.head`
- `stream_embed`: (1, 1, dit.dim) where dit.dim=3072 for Wan2.2-5B

The patch embedding and head are each roughly the size of a small conv / linear
patchifier (~200 k params each for a 16-channel VAE latent with patch size
(1,2,2) and dim 3072). The whole FlowStream therefore adds only on the order of
**~400 k trainable parameters** on top of the frozen/shared 5B DiT.

## Video backbone

Wan2.2-TI2V-5B: ~5 B parameters, frozen VAE and T5, DiT fine-tuned.
