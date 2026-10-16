"""Toy attribution study: is the OneWM benefit a property of the objective, or of the prior?

OneWM holds the architecture fixed and varies only the training objective, which
is a clean design for asking "do the two tasks help each other". It does not
separate the *objective* from the *pretrained Wan2.2 backbone*. Standing
criterion for interface designs: split the interface's inductive bias from the
contribution of the pretraining prior. This script runs that split at a size
that fits on a CPU.

World (fully synthetic, deterministic, no dependencies):
  * a marker and a target live in [0,1]^2 and are rendered as Gaussian bumps on
    an 8x8 grid -- the "latent observation" the video expert consumes;
  * an action is a commanded displacement, a_t in [-0.1, 0.1]^2, and the marker
    moves by exactly that: M_{t+1} = clip(M_t + a_t). The visual consequence of
    an action is therefore "the marker moves by a";
  * demonstrations come from a proportional controller toward the target
    (k=3, clipped), which is a deterministic function of (M_0, T): the policy
    has to read both bumps out of the image, and the world model has to turn
    actions into marker motion.

Geometry copies LIBERO's: 9 latent frames (1 clean + 8 future) and a 32-action
chunk (4 actions per future frame), so the released causal mask aligns actions
to frames exactly as in the paper's main recipe.

Arms (all on the same architecture, differing only in objective/mask):
  onewm      sampling 60% r=0 / 25% partial / 15% r=K  (repo sampler) + action loss
  wam_noisy  every example all-noisy (action_clean_prefix_enabled=false)
  wam_mask   onewm mixture, but video queries cannot read action keys
  wm         every example all-clean (r=K), loss_lambda_action=0

Conditions:
  scratch    all arms start from the same random initialization
  prior      all arms start from one shared *video-only* prior: the video expert
             trained to predict future frames with actions hidden from it (the
             analogue of Wan2.2, which has never seen an action), action expert
             untouched

Reported per arm/seed:
  reach      closed-loop style success: the predicted 32-action chunk is applied
             to the true dynamics; a trial succeeds if the marker ends within
             0.05 of the target (uniform in [0,1]^2)
  action_mse normalized action-chunk error
  video_mse  future-latent prediction error given clean ground-truth actions

    ONEWM_REPO=/path/to/onewm-repo python study_benefit_attribution.py [--seeds 3]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
WORKSPACE = HERE.parents[4]
REPO = Path(os.environ.get("ONEWM_REPO", WORKSPACE / "code" / "onewm-repo"))
sys.path.insert(0, str(REPO))

from onewm.training.flow import add_partial_clean_action_noise  # noqa: E402
from onewm.training.loss import flow_matching_loss  # noqa: E402
from onewm.modules.scheduler import FlowMatchScheduler  # noqa: E402


def _load_probe():
    path = HERE.parent / "probe_mot_coupling.py"
    spec = importlib.util.spec_from_file_location("probe_mot_coupling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROBE = _load_probe()
MoTWAM_mask = PROBE.MoTWAM._build_mot_attention_mask

# ---------------------------------------------------------------- geometry
GRID, SIGMA = 8, 0.12
OBS_DIM = GRID * GRID
TPF = 4  # tokens per latent frame
TOK_DIM = OBS_DIM // TPF
FRAMES, CLEAN, CHUNK = 9, 1, 32
APF = CHUNK // (FRAMES - CLEAN)  # 4 actions per future frame
STEP, K_CTRL, DELTA = 4, 3.0, 0.1
VID_HID, ACT_HID, HEADS, HEAD_DIM, LAYERS = 64, 48, 4, 8, 2
DEV = torch.device("cpu")


def grid_points() -> torch.Tensor:
    lin = torch.linspace(0.05, 0.95, GRID)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


GRID_PTS = grid_points()


def render(marker: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    def bump(c: torch.Tensor) -> torch.Tensor:
        d2 = (GRID_PTS[None, :, :] - c[:, None, :]).pow(2).sum(-1)
        return torch.exp(-d2 / (2 * SIGMA**2))

    return bump(marker) + 0.8 * bump(target)


def make_episodes(n: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    marker = torch.rand(n, 2, generator=g)
    target = torch.rand(n, 2, generator=g)
    obs, actions = [], []
    m = marker.clone()
    obs.append(render(m, target))
    for t in range(CHUNK):
        a = (K_CTRL * (target - m)).clamp(-DELTA, DELTA)
        actions.append(a)
        m = (m + a).clamp(0.0, 1.0)
        if (t + 1) % APF == 0 and (t + 1) // APF <= FRAMES - CLEAN:
            obs.append(render(m, target))
    obs = torch.stack(obs, dim=1)  # [n, 9, 64]
    return {
        "obs": obs.reshape(n, FRAMES, TPF, TOK_DIM),
        "actions": torch.stack(actions, dim=1),  # [n, 32, 2]
        "marker": marker,
        "target": target,
    }


# ---------------------------------------------------------------- model
def build_arms(seed: int):
    """Two experts sharing one attention geometry, as in the released model."""
    from onewm.modules.action_dit import ActionDiT
    from onewm.modules.mot import MoT

    torch.manual_seed(seed)
    video = ActionDiT(
        hidden_dim=VID_HID, action_dim=TOK_DIM, ffn_dim=VID_HID * 4, text_dim=8,
        freq_dim=8, eps=1e-6, num_heads=HEADS, attn_head_dim=HEAD_DIM,
        num_layers=LAYERS, max_seq_len=128,
    )
    action = ActionDiT(
        hidden_dim=ACT_HID, action_dim=2, ffn_dim=ACT_HID * 4, text_dim=8,
        freq_dim=8, eps=1e-6, num_heads=HEADS, attn_head_dim=HEAD_DIM,
        num_layers=LAYERS, max_seq_len=128,
    )
    mot = MoT(experts={"video": video, "action": action}, checkpoint_mixed_attn=False)
    return mot, video, action


def mask_for(block_actions_from_video: bool) -> torch.Tensor:
    m = MoTWAM_mask(
        video_seq_len=FRAMES * TPF,
        action_seq_len=CHUNK,
        video_tokens_per_frame=TPF,
        device=DEV,
        action_video_conditioning="first_frame",
        clean_prefix_latent_steps=CLEAN,
    )
    if block_actions_from_video:
        m = m.clone()
        m[: FRAMES * TPF, FRAMES * TPF :] = False
    return m


CTX = torch.randn(1, 2, 8)
T_SCALAR = torch.tensor([300.0])


def flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], FRAMES * TPF, TOK_DIM)


def unflat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], FRAMES, TPF, TOK_DIM)


def tokens_of(t_frame: torch.Tensor) -> torch.Tensor:
    return t_frame.repeat_interleave(TPF, dim=1)


def run_mot(video, action, mot, obs, act_in, t_video, t_action, mask):
    v_state = video.pre_dit(obs, t_video, CTX, None)
    a_state = action.pre_dit(act_in, t_action, CTX, None)
    out = mot({"video": v_state, "action": a_state}, mask)
    return video.post_dit(out["video"]), action.post_dit(out["action"])


# ---------------------------------------------------------------- training
def train_arm(arm: str, data: dict, seed: int, steps: int, init=None, batch=32):
    torch.manual_seed(seed + 1000)
    mot, video, action = build_arms(seed)
    if init is not None:
        video.load_state_dict(init["video"])
        action.load_state_dict(init["action"])
    params = list(mot.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    sched = FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    block_actions = arm == "wam_mask"
    mask = mask_for(block_actions)
    n = data["obs"].shape[0]
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        obs = data["obs"][idx]
        act = data["actions"][idx]
        if arm == "wm":
            probs = (0.0, 0.0, 1.0)
        elif arm == "wam_noisy":
            probs = (1.0, 0.0, 0.0)
        else:
            probs = (0.60, 0.25, 0.15)
        noise = add_partial_clean_action_noise(
            sched, act, action_is_pad=None, num_future_frames=FRAMES - CLEAN,
            prob_none=probs[0], prob_partial=probs[1], prob_full=probs[2],
            boundary_ratio=0.6, jitter=2, jitter_weights=[0.1, 0.15, 0.5, 0.15, 0.1],
        )
        clean_obs = obs[:, :CLEAN]
        future_obs = obs[:, CLEAN:]
        v_noise = torch.randn_like(future_obs)
        v_sigma = sched.timestep_to_sigma(
            sched.sample_timesteps(batch, DEV, torch.float32)
        )
        v_noisy = (1 - v_sigma)[:, None, None, None] * future_obs + v_sigma[
            :, None, None, None
        ] * v_noise
        v_target = v_noise - future_obs
        v_in = flat(torch.cat([clean_obs, v_noisy], dim=1))
        t_frame = torch.zeros(batch, FRAMES, dtype=torch.float32)
        t_frame[:, CLEAN:] = (v_sigma * 1000)[:, None]
        v_pred, a_pred = run_mot(
            video, action, mot, v_in, noise.noisy, tokens_of(t_frame),
            noise.token_timesteps[:, :CHUNK], mask,
        )
        v_pred = unflat(v_pred)[:, CLEAN:]
        loss_v = flow_matching_loss(
            v_pred.reshape(batch, -1), v_target.reshape(batch, -1),
            v_sigma * 1000, sched,
        )
        loss_a = flow_matching_loss(
            a_pred, noise.target, noise.timesteps, sched,
            is_pad_mask=noise.known_mask, average_only_valid_samples=True,
        )
        lam_a = 0.0 if arm == "wm" else 1.0
        loss = loss_v + lam_a * loss_a
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    return mot, video, action


@torch.no_grad()
def evaluate(video, action, mot, data: dict, arm: str):
    """Denoise the action chunk from noise, execute it, then score world prediction."""
    block_actions = arm == "wam_mask"
    mask = mask_for(block_actions)
    n = data["obs"].shape[0]
    batch = 64
    sched = FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    reach, a_err, v_err, m_err, f_err = [], [], [], [], []
    for lo in range(0, n, batch):
        obs = data["obs"][lo : lo + batch]
        b = obs.shape[0]
        act = data["actions"][lo : lo + batch]
        # --- action generation: integrate the action expert from pure noise
        a_lat = torch.randn(b, CHUNK, 2)
        t_steps, deltas = sched.build_inference_schedule(10, DEV, torch.float32)
        v_in = flat(torch.cat([obs[:, :CLEAN], torch.zeros_like(obs[:, CLEAN:])], dim=1))
        for i in range(10):
            t_frame = torch.full((b, FRAMES), float(t_steps[i]))
            t_frame[:, :CLEAN] = 0.0
            _, a_pred = run_mot(
                video, action, mot, v_in, a_lat, tokens_of(t_frame),
                t_steps[i].reshape(1).expand(b), mask,
            )
            a_lat = a_lat + a_pred * deltas[i]
        # --- execute the predicted chunk in the true dynamics
        m = data["marker"][lo : lo + batch].clone()
        for t in range(CHUNK):
            m = (m + a_lat[:, t].clamp(-DELTA, DELTA)).clamp(0.0, 1.0)
        reach.append(((m - data["target"][lo : lo + batch]).norm(dim=-1) < 0.05).float())
        a_err.append((a_lat - act).pow(2).mean(dim=(1, 2)))
        # --- world prediction given clean ground-truth actions
        noise0 = torch.randn_like(obs[:, CLEAN:])
        v_lat = noise0.clone()
        t_steps, deltas = sched.build_inference_schedule(10, DEV, torch.float32)
        for i in range(10):
            v_full = flat(torch.cat([obs[:, :CLEAN], v_lat], dim=1))
            t_frame = torch.full((b, FRAMES), float(t_steps[i]))
            t_frame[:, :CLEAN] = 0.0
            v_pred, _ = run_mot(
                video, action, mot, v_full, act, tokens_of(t_frame),
                t_steps[i].reshape(1).expand(b), mask,
            )
            v_lat = v_lat + unflat(v_pred)[:, CLEAN:] * deltas[i]
        v_err.append((v_lat - obs[:, CLEAN:]).pow(2).mean(dim=(1, 2, 3)))
        change_true = obs[:, CLEAN:] - obs[:, :1]
        change_pred = v_lat - obs[:, :1]
        m_err.append((change_pred - change_true).pow(2).mean(dim=(1, 2, 3)))
        # does the world prediction follow the actions it is conditioned on?
        # Same initial noise as the reference rollout, so the difference isolates
        # the conditioning change rather than sampler noise.
        v_alt = noise0.clone()
        act_alt = (act + 0.08).clamp(-DELTA, DELTA)
        for i in range(10):
            v_full = flat(torch.cat([obs[:, :CLEAN], v_alt], dim=1))
            t_frame = torch.full((b, FRAMES), float(t_steps[i]))
            t_frame[:, :CLEAN] = 0.0
            v_pred, _ = run_mot(
                video, action, mot, v_full, act_alt, tokens_of(t_frame),
                t_steps[i].reshape(1).expand(b), mask,
            )
            v_alt = v_alt + unflat(v_pred)[:, CLEAN:] * deltas[i]
        f_err.append((v_alt - v_lat).abs().mean(dim=(1, 2, 3)))
    return (
        float(torch.cat(reach).mean()),
        float(torch.cat(a_err).mean()),
        float(torch.cat(v_err).mean()),
        float(torch.cat(m_err).mean()),
        float(torch.cat(f_err).mean()),
    )


def train_prior(data: dict, seed: int, steps: int):
    """Video-only prior: future frames predicted with actions hidden from video."""
    torch.manual_seed(seed + 500)
    mot, video, action = build_arms(seed)
    opt = torch.optim.AdamW(video.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    mask = mask_for(True)
    n = data["obs"].shape[0]
    g = torch.Generator().manual_seed(seed + 7)
    for _ in range(steps):
        idx = torch.randint(0, n, (32,), generator=g)
        obs = data["obs"][idx]
        clean_obs, future_obs = obs[:, :CLEAN], obs[:, CLEAN:]
        v_noise = torch.randn_like(future_obs)
        v_sigma = sched.timestep_to_sigma(
            sched.sample_timesteps(32, DEV, torch.float32)
        )
        v_noisy = (1 - v_sigma)[:, None, None, None] * future_obs + v_sigma[
            :, None, None, None
        ] * v_noise
        v_in = flat(torch.cat([clean_obs, v_noisy], dim=1))
        t_frame = torch.zeros(32, FRAMES)
        t_frame[:, CLEAN:] = (v_sigma * 1000)[:, None]
        v_pred, _ = run_mot(
            video, action, mot, v_in, torch.zeros(32, CHUNK, 2),
            tokens_of(t_frame), torch.tensor([300.0]).expand(32), mask,
        )
        loss = flow_matching_loss(
            unflat(v_pred)[:, CLEAN:].reshape(32, -1),
            (v_noise - future_obs).reshape(32, -1),
            v_sigma * 1000, sched,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {"video": video.state_dict(), "action": action.state_dict()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--prior-steps", type=int, default=700)
    parser.add_argument("--out", default=None)
    parser.add_argument("--conditions", default="scratch,prior")
    parser.add_argument("--arms", default="onewm,wam_noisy,wam_mask,wm")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    train = make_episodes(512, seed=0)
    test = make_episodes(128, seed=1)
    print(
        f"toy geometry: obs {FRAMES}x{TPF}x{TOK_DIM}, chunk {CHUNK}, "
        f"actions/frame {APF}; train={train['obs'].shape[0]} test={test['obs'].shape[0]}"
    )
    rows = []
    conditions = args.conditions.split(",")
    arms = args.arms.split(",")
    for condition in conditions:
        for seed in range(args.seeds):
            init = train_prior(train, seed, args.prior_steps) if condition == "prior" else None
            if args.tag:
                torch.save({"video": init["video"], "action": init["action"]},
                           f"/tmp/onewm_prior_{args.tag}_{seed}.pt")
            for arm in arms:
                mot, video, action = train_arm(
                    arm, train, seed, args.steps, init=init
                )
                reach, a_err, v_err, m_err, f_err = evaluate(video, action, mot, test, arm)
                rows.append(
                    dict(condition=condition, seed=seed, arm=arm, reach=reach,
                         action_mse=a_err, video_mse=v_err, motion_mse=m_err,
                         action_follow=f_err)
                )
                print(
                    f"  {condition:6s} seed={seed} {arm:10s} reach={reach:.3f} "
                    f"action_mse={a_err:.4f} video_mse={v_err:.5f} motion_mse={m_err:.5f} "
                    f"action_follow={f_err:.2e}",
                    flush=True,
                )
    print("\nmeans over seeds")
    print(
        f"  {'condition':9s} {'arm':10s} {'reach':>7s} {'action_mse':>11s} "
        f"{'video_mse':>10s} {'motion_mse':>11s} {'action_follow':>13s}"
    )
    for condition in conditions:
        for arm in arms:
            sel = [r for r in rows if r["condition"] == condition and r["arm"] == arm]
            print(
                f"  {condition:9s} {arm:10s} "
                f"{sum(r['reach'] for r in sel) / len(sel):7.3f} "
                f"{sum(r['action_mse'] for r in sel) / len(sel):11.4f} "
                f"{sum(r['video_mse'] for r in sel) / len(sel):10.5f} "
                f"{sum(r['motion_mse'] for r in sel) / len(sel):11.5f} "
                f"{sum(r['action_follow'] for r in sel) / len(sel):13.3e}"
            )
    if args.out:
        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
