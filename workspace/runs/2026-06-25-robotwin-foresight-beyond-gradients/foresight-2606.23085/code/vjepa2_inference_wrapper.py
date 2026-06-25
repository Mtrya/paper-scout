"""
Curated excerpt from facebookresearch/vjepa2 notebooks/utils/world_model_wrapper.py
How V-JEPA 2-AC is used at inference time.

Foresight uses the same encode + predict machinery, but instead of passing the
predicted latents to an MPC planner, it mean-pools them per timestep and feeds
them to a downstream causal failure detector.
"""

import numpy as np
import torch
import torch.nn.functional as F


class WorldModel(object):
    def __init__(self, encoder, predictor, tokens_per_frame, transform,
                 normalize_reps=True, device="cuda:0"):
        self.encoder = encoder
        self.predictor = predictor
        self.normalize_reps = normalize_reps
        self.transform = transform
        self.tokens_per_frame = tokens_per_frame
        self.device = device

    def encode(self, image):
        """Encode a single image into hidden latent tokens z^h."""
        clip = np.expand_dims(image, axis=0)          # [1, H, W, C]
        clip = self.transform(clip)[None, :]          # [1, C, T, H, W]
        B, C, T, H, W = clip.size()
        # Flatten B*T and repeat for tubelet_size=2
        clip = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        clip = clip.to(self.device)
        h = self.encoder(clip)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        if self.normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
        return h

    def predict_next_latent(self, rep, action, state):
        """
        Given current hidden latent rep and an action, return predicted latent.
        Foresight calls this at every replan step and pools the output.
        """
        B, T, N_T, D = rep.size()
        rep = rep.flatten(1, 2)                       # [B, T*N_T, D]
        next_rep = self.predictor(rep, action, state)[:, -self.tokens_per_frame:]
        if self.normalize_reps:
            next_rep = F.layer_norm(next_rep, (next_rep.size(-1),))
        return next_rep


# Foresight's adaptation:
#   z_t^h = Pool(encoder(c_t))      # mean over 256 spatial patches -> 1408-d
#   z_t^p = Pool(predictor(z_t^h, A_t))
#   u_t   = W * z_t^p + p_t         # learned projection + sinusoidal position
#   s_t   = D_theta({u_i}_{i<=t})   # causal Transformer failure score
