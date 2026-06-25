"""
Curated excerpt from facebookresearch/vjepa2 app/vjepa_droid/train.py
Training objective for the action-conditioned predictor.

Foresight's paper (Appendix 7) says:
"The predictor is trained with a combined teacher-forcing and autoregressive-
rollout objective following the V-JEPA 2-AC training procedure ... losses on
LayerNorm-normalized representations are summed: L = L_TF + L_AR"

This file shows exactly that in Meta's official implementation.
"""

import torch
import torch.nn.functional as F


def make_training_step(encoder, predictor, target_encoder, optimizer, scaler,
                       clips, actions, states, extrinsics,
                       tokens_per_frame, max_num_frames, auto_steps=2,
                       loss_exp=1.0, normalize_reps=True, mixed_precision=True,
                       dtype=torch.bfloat16):
    """
    One training step for V-JEPA 2-AC.

    Args:
        clips: [B, C, T, H, W] video frames
        actions: [B, T-1, action_dim] deltas between consecutive states
        states:  [B, T, action_dim] current state sequence
        extrinsics: [B, T, action_dim-1] camera extrinsics (optional)
    """
    def forward_target(c):
        with torch.no_grad():
            # Permute to [B, T, C, H, W], flatten batch*time, insert tubelet dim
            c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
            h = target_encoder(c)
            h = h.view(c.size(0) // max_num_frames, max_num_frames, -1, h.size(-1)).flatten(1, 2)
            if normalize_reps:
                h = F.layer_norm(h, (h.size(-1),))
            return h

    def forward_predictions(z):
        def _step_predictor(_z, _a, _s, _e):
            _z = predictor(_z, _a, _s, _e)
            if normalize_reps:
                _z = F.layer_norm(_z, (_z.size(-1),))
            return _z

        # Teacher forcing: context = all but last frame, actions/states up to T-1
        _z, _a, _s, _e = z[:, :-tokens_per_frame], actions, states[:, :-1], extrinsics[:, :-1]
        z_tf = _step_predictor(_z, _a, _s, _e)

        # Autoregressive rollout: auto_steps future frames (default n=2)
        _z = torch.cat([z[:, :tokens_per_frame], z_tf[:, :tokens_per_frame]], dim=1)
        for n in range(1, auto_steps):
            _a, _s, _e = actions[:, :n + 1], states[:, :n + 1], extrinsics[:, :n + 1]
            _z_nxt = _step_predictor(_z, _a, _s, _e)[:, -tokens_per_frame:]
            _z = torch.cat([_z, _z_nxt], dim=1)
        z_ar = _z[:, tokens_per_frame:]

        return z_tf, z_ar

    def loss_fn(z, h):
        _h = h[:, tokens_per_frame:z.size(1) + tokens_per_frame]
        return torch.mean(torch.abs(z - _h) ** loss_exp) / loss_exp

    with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
        h = forward_target(clips)
        z_tf, z_ar = forward_predictions(h)
        jloss = loss_fn(z_tf, h)
        sloss = loss_fn(z_ar, h)
        loss = jloss + sloss

    if mixed_precision:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()

    return loss.item(), jloss.item(), sloss.item()


# Paper training hyperparameters (Appendix 7):
#   AdamW (beta1=0.9, beta2=0.999), weight decay 0.04
#   Linear warmup 10 epochs, cosine anneal to 0 over 200 epochs total
#   LIBERO: 1x H200, batch 256, peak LR 2e-4
#   BEHAVIOR-1K / ManiSkill-Long: 2x H200, effective batch 512, same LR
#   Real-world: 2x H200, effective batch 32, peak LR 5e-5
