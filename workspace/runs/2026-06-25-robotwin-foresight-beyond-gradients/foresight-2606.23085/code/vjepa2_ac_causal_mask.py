"""
Curated excerpt from facebookresearch/vjepa2 src/models/utils/modules.py
Frame-causal attention mask for the action-conditioned predictor.

This is what "action-conditioned" means mechanically in V-JEPA 2-AC:
- Each timestep's patch tokens can attend to patch tokens from the SAME frame
  and ALL PREVIOUS frames, plus the action/state tokens at every frame.
- Future frames are masked out, so the predictor cannot cheat by looking ahead.
- Actions are encoded as per-frame prefix tokens (a_t, s_t, optionally e_t).
"""

import torch


def build_action_block_causal_attention_mask(T, H, W, add_tokens=2):
    """
    Build a block-causal mask for V-JEPA 2-AC.

    Args:
        T: number of frames (num_frames // tubelet_size)
        H, W: spatial patch grid size
        add_tokens: number of conditioning tokens per frame
                    (2 = action + state; 3 = action + state + extrinsics)
    Returns:
        [T*(add_tokens + H*W), T*(add_tokens + H*W)] boolean attention mask.
        True  -> allowed to attend
        False -> masked out
    """
    N_T = add_tokens + (H * W)          # tokens per timestep
    N = T * N_T                         # total tokens
    mask = torch.zeros(N, N).bool()
    mask_block = torch.ones(N_T, N_T).bool()

    for t1 in range(T):
        # Current frame t1 can see frames [0, t1]
        for t2 in range(max(0, t1 - T + 1), t1 + 1):
            mask[t1 * N_T : (t1 + 1) * N_T,
                 t2 * N_T : (t2 + 1) * N_T] = mask_block

    return mask


# Example for Foresight's LIBERO setup:
#   8-frame context, patch_size=16, 256x256 image -> H=W=16 -> H*W=256 tokens/frame
#   add_tokens=2 (action + state)
#   N_T = 2 + 256 = 258 tokens per timestep
#   Total tokens = 8 * 258 = 2064
#
# The mask ensures frame t's 258 tokens only attend to tokens from frames <= t.
# This causality is critical: the failure detector later sees a sequence of
# predicted latents z_t^p that were generated without lookahead.
