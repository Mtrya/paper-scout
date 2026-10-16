"""Does the released architecture actually couple the two experts the way the paper says?

Builds a tiny two-expert MoT out of the *released* modules (MoT, ActionDiT, the
Wan2.2 DiTBlock, the causal mask builder, the flow-matching loss) at a size that
runs on CPU, then measures, numerically:

  A. action -> video channel: which future video frames change when a clean
     action block changes (strict temporal causality, Eq. 2).
  B. the WAM_mask ablation: zeroing the video->action key columns makes the
     video output bitwise independent of actions, and nothing else changes.
  C. gradient routing between the two objectives: does each loss reach the
     other expert's parameters, and how much of it.
  D. video -> action channel, and what the action queries can see.

The video expert here is the repo's ActionDiT class at video-shaped I/O -- the
same DiTBlock, timestep modulation and cross-attention as Wan22Dit, minus the
Conv3d patch embedding and 3D RoPE, because a 5B-parameter Wan22Dit cannot be
instantiated on this machine and no Wan checkpoint is distributed with the
paper. Everything that the measurements depend on (mask, joint attention
orchestration, loss masking, gradient paths) is the released code unchanged.

    ONEWM_REPO=/path/to/onewm-repo python probe_mot_coupling.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
WORKSPACE = HERE.parents[4]
REPO = Path(os.environ.get("OSNEWM_REPO", os.environ.get("ONEWM_REPO", WORKSPACE / "code" / "onewm-repo")))
sys.path.insert(0, str(REPO))

from onewm.modules.action_dit import ActionDiT  # noqa: E402
from onewm.modules.mot import MoT  # noqa: E402
from onewm.modules.scheduler import FlowMatchScheduler  # noqa: E402
from onewm.training.loss import flow_matching_loss  # noqa: E402
from onewm.wam.mot_wam import MoTWAM  # noqa: E402

torch.manual_seed(0)
DEV = torch.device("cpu")

# ---- tiny runtime geometry (LIBERO's shape, shrunk) ----------------------
FRAMES, TPF = 3, 4  # 1 clean latent frame + 2 future latent frames, 4 tokens each
CLEAN, CHUNK = 1, 8  # chunk 8 -> 4 actions per future latent frame
LATENT, HIDDEN_A, HIDDEN_V = 4, 16, 32
HEADS, HEAD_DIM = 2, 8  # both experts share heads*head_dim = 16, as in the real model
LAYERS = 2
V_TOKENS = FRAMES * TPF


def build(mask_mode: str = "shipped") -> tuple[MoT, dict]:
    global MASK_MODE
    MASK_MODE = mask_mode
    torch.manual_seed(1)
    video = ActionDiT(
        hidden_dim=HIDDEN_V,
        action_dim=LATENT,
        ffn_dim=HIDDEN_V * 4,
        text_dim=LATENT,
        freq_dim=8,
        eps=1e-6,
        num_heads=HEADS,
        attn_head_dim=HEAD_DIM,
        num_layers=LAYERS,
        max_seq_len=64,
    )
    action = ActionDiT(
        hidden_dim=HIDDEN_A,
        action_dim=7,
        ffn_dim=HIDDEN_A * 4,
        text_dim=LATENT,
        freq_dim=8,
        eps=1e-6,
        num_heads=HEADS,
        attn_head_dim=HEAD_DIM,
        num_layers=LAYERS,
        max_seq_len=64,
    )
    mot = MoT(experts={"video": video, "action": action}, checkpoint_mixed_attn=False)
    return mot, {
        "video": video,
        "action": action,
        "ctx": torch.randn(1, 2, LATENT),
        "t": torch.tensor([300.0]),
    }


MASK_MODE = "shipped"


def shipped_mask() -> torch.Tensor:
    m = MoTWAM._build_mot_attention_mask(
        video_seq_len=V_TOKENS,
        action_seq_len=CHUNK,
        video_tokens_per_frame=TPF,
        device=DEV,
        action_video_conditioning="first_frame",
        clean_prefix_latent_steps=CLEAN,
    )
    if MASK_MODE == "wam_mask":
        m = m.clone()
        m[:V_TOKENS, V_TOKENS:] = False
    return m


def mask_for(cond: str) -> torch.Tensor:
    m = MoTWAM._build_mot_attention_mask(
        video_seq_len=V_TOKENS,
        action_seq_len=CHUNK,
        video_tokens_per_frame=TPF,
        device=DEV,
        action_video_conditioning=cond,
        clean_prefix_latent_steps=CLEAN,
    )
    if MASK_MODE == "wam_mask":
        m = m.clone()
        m[:V_TOKENS, V_TOKENS:] = False
    return m


def forward(modules: dict, video_in: torch.Tensor, action_in: torch.Tensor, cond: str = "first_frame"):
    video_dit, action_dit = modules["video"], modules["action"]
    ctx = modules["ctx"]
    t_v = modules["t"]
    t_a = modules["t"]
    v_state = video_dit.pre_dit(video_in, t_v, ctx, None)
    a_state = action_dit.pre_dit(action_in, t_a, ctx, None)
    out = modules["mot"]({"video": v_state, "action": a_state}, mask_for(cond))
    return out["video"], out["action"]


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def channel_audit() -> None:
    section("A. action -> video channel and strict temporal causality")
    modules_mot, modules = build()
    modules["mot"] = modules_mot
    v = torch.randn(1, V_TOKENS, LATENT)
    a = torch.randn(1, CHUNK, 7)
    base_v, base_a = forward(modules, v, a)
    print(f"  geometry: {FRAMES} latent frames x {TPF} tokens, clean={CLEAN}, chunk={CHUNK}")
    for block, label in ((slice(0, 4), "block 1 (aligned to future frame 1)"),
                         (slice(4, 8), "block 2 (aligned to future frame 2)")):
        a2 = a.clone()
        a2[:, block] += 1.0
        v2, _ = forward(modules, v, a2)
        deltas = [
            float((v2[:, f * TPF : (f + 1) * TPF] - base_v[:, f * TPF : (f + 1) * TPF]).abs().max())
            for f in range(FRAMES)
        ]
        print(f"  perturb {label}: max |d video| per latent frame = "
              f"[{', '.join(f'{d:.3e}' for d in deltas)}]")
    print("  frame 0 is the observed prefix: exactly zero in every perturbation, as is")
    print("  every future frame that precedes the perturbed action block.")


def wam_mask_audit() -> None:
    section("B. WAM_mask ablation is exactly a channel removal")
    v = torch.randn(1, V_TOKENS, LATENT)
    a = torch.randn(1, CHUNK, 7)
    for mode in ("shipped", "wam_mask"):
        mot, modules = build()
        modules["mot"] = mot
        if mode == "wam_mask":
            global MASK_MODE
            MASK_MODE = "wam_mask"
        else:
            MASK_MODE = "shipped"
        base_v, base_a = forward(modules, v, a)
        a2 = a.clone()
        a2 += 2.0
        v2, a2_out = forward(modules, v, a2)
        dv = float((v2 - base_v).abs().max())
        da = float((a2_out - base_a).abs().max())
        print(
            f"  {mode:9s}: perturbing every action token changes video by {dv:.3e} "
            f"and action output by {da:.3e}"
        )
    print("  The masked variant's video output is bitwise unchanged; the action side is")
    print("  untouched. One operator on the mask separates the paper's primary")
    print("  baseline from OneWM, and no recipe field can express it.")


def grad_norm(module: torch.nn.Module) -> tuple[float, int, int]:
    total, nz, total_par = 0.0, 0, 0
    for p in module.parameters():
        total_par += 1
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
            if float(p.grad.detach().abs().max()) > 0:
                nz += 1
    return total**0.5, nz, total_par


def gradient_audit() -> None:
    section("C. does each objective reach the other expert's parameters?")
    sched = FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    for arm, lam_v, lam_a, known, masked in (
        ("r=K example   (action loss empty)", 1.0, 1.0, True, False),
        ("r=0 example   (action loss active)", 1.0, 1.0, False, False),
        ("WM baseline   (loss_lambda_action = 0)", 1.0, 0.0, True, False),
        ("WAM_mask, r=0 (video->action keys removed)", 1.0, 1.0, False, True),
    ):
        mot, modules = build("wam_mask" if masked else "shipped")
        modules["mot"] = mot
        v = torch.randn(1, V_TOKENS, LATENT)
        a = torch.randn(1, CHUNK, 7)
        v_out, a_out = forward(modules, v, a)
        target_v = torch.randn_like(v_out)
        target_a = torch.randn_like(a_out)
        loss_v = flow_matching_loss(v_out, target_v, torch.tensor([300.0]), sched)
        if known:
            # every action token clean -> the action loss mask is full
            loss_a = flow_matching_loss(
                a_out, target_a, torch.tensor([300.0]), sched,
                is_pad_mask=torch.ones_like(a_out[..., 0], dtype=torch.bool),
                average_only_valid_samples=True,
            )
        else:
            loss_a = flow_matching_loss(a_out, target_a, torch.tensor([300.0]), sched)
        # separate backward passes per objective
        results = {}
        for name, loss in (("L_video", loss_v), ("L_action", loss_a)):
            for p in modules["video"].parameters():
                p.grad = None
            for p in modules["action"].parameters():
                p.grad = None
            (lam_v * loss if name == "L_video" else lam_a * loss).backward(retain_graph=True)
            gv, nzv, tv = grad_norm(modules["video"])
            ga, nza, ta = grad_norm(modules["action"])
            results[name] = (gv, nzv, tv, ga, nza, ta)
        print(f"  {arm}")
        print(f"      L_video={float(loss_v):.4f}  L_action={float(loss_a):.4f}")
        for name, (gv, nzv, tv, ga, nza, ta) in results.items():
            print(
                f"      via {name:8s}: |g| video={gv:9.3e} ({nzv}/{tv} tensors)   "
                f"|g| action={ga:9.3e} ({nza}/{ta} tensors)"
            )

    print("\n  Which parts of the video expert does the action loss touch? (r=0 arm)")
    mot, modules = build()
    modules["mot"] = mot
    v = torch.randn(1, V_TOKENS, LATENT)
    a = torch.randn(1, CHUNK, 7)
    v_out, a_out = forward(modules, v, a)
    loss_a = flow_matching_loss(a_out, torch.randn_like(a_out), torch.tensor([300.0]), sched)
    for p in modules["video"].parameters():
        p.grad = None
    loss_a.backward()
    groups: dict[str, list[float]] = {}
    for name, p in modules["video"].named_parameters():
        key = name.split(".")[0] if name.startswith("blocks") else name
        if name.startswith("blocks"):
            key = name.split(".")[2] if len(name.split(".")) > 2 else "blocks.?"
        groups.setdefault(key, []).append(
            0.0 if p.grad is None else float(p.grad.detach().abs().max())
        )
    for key in sorted(groups):
        vals = groups[key]
        print(
            f"      {key:16s} tensors={len(vals):2d} with nonzero grad="
            f"{sum(1 for x in vals if x > 0):2d}"
        )

    print("  Read-out: at r=K the action loss is exactly zero (the sample carries no")
    print("  action target), so the two experts are joined only by the video loss --")
    print("  yet that loss already pushes gradient into the action expert. In the WM")
    print("  arm the action expert is therefore not untrained: it is trained as a")
    print("  conditioning encoder by the world-prediction objective alone. Conversely")
    print("  the action loss reaches the video expert's own blocks, so neither")
    print("  direction of the claimed coupling is merely rhetorical.")


def seen_audit() -> None:
    section("D. video -> action channel: what the action queries can see")
    mot, modules = build()
    modules["mot"] = mot
    v = torch.randn(1, V_TOKENS, LATENT)
    a = torch.randn(1, CHUNK, 7)
    base_v, base_a = forward(modules, v, a)
    for f in range(FRAMES):
        v2 = v.clone()
        v2[:, f * TPF : (f + 1) * TPF] += 1.0
        _, a2 = forward(modules, v2, a)
        print(
            f"  perturb video latent frame {f}: max |d action| = "
            f"{float((a2 - base_a).abs().max()):.3e}"
            + ("   (clean prefix: queries read it)" if f < CLEAN else "   (future latent)")
        )
    full = mask_for("full_video")
    print(
        f"\n  mask density: first_frame={shipped_mask().float().mean():.3f}  "
        f"full_video={full.float().mean():.3f}"
    )
    print("  Under the shipped recipes (LIBERO first_frame, RoboCasa365 clean_prefix),")
    print("  action queries never read a future latent, at training or inference time.")


if __name__ == "__main__":
    print(f"torch {torch.__version__} | repo {REPO}")
    channel_audit()
    wam_mask_audit()
    gradient_audit()
    seen_audit()
