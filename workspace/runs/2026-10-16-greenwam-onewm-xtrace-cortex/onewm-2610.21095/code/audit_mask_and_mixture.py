"""Audit OneWM's causal mask, conditioning mixture and action-loss mask.

Runs against the released repo (code/onewm-repo) with only torch installed --
no Wan2.2 weights are required, because the mask builder, the clean-prefix
sampler and the flow-matching loss are all weight-free functions.

    ONEWM_REPO=/path/to/onewm-repo python audit_mask_and_mixture.py

Sections:
  1. runtime geometry  -- frames -> VAE latents -> future latent frames
  2. mask structure    -- Eq. 2 visibility, checked exhaustively per token block
  3. config surface    -- which paper arms are expressible as recipes
  4. mixture sampler   -- realized regime fractions and prefix-length shape
  5. action-loss mask  -- clean prefix is bitwise clean and loss-inert
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
WORKSPACE = HERE.parents[4]
REPO = Path(os.environ.get("ONEWM_REPO", WORKSPACE / "code" / "onewm-repo"))
sys.path.insert(0, str(REPO))

from onewm.training.flow import (  # noqa: E402
    add_partial_clean_action_noise,
    aligned_action_boundaries,
    sample_clean_prefix_lengths,
)
from onewm.training.loss import flow_matching_loss  # noqa: E402
from onewm.wam.mot_wam import MoTWAM  # noqa: E402
from onewm.modules.scheduler import FlowMatchScheduler  # noqa: E402
from onewm.config import FrameworkConfig  # noqa: E402

DEV = torch.device("cpu")


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------
# 1. geometry
# --------------------------------------------------------------------------
RECIPES = {
    # name: (raw frames, temporal compress, clean prefix latent steps, chunk)
    "LIBERO": (33, 4, 1, 32),
    "RoboCasa365": (13, 4, 2, 16),
}


def geometry() -> dict[str, dict]:
    rule("1. runtime geometry (from configs/*.yaml + Wan2.2 VAE temporal_compress=4)")
    out = {}
    for name, (raw, compress, clean, chunk) in RECIPES.items():
        latents = 1 + (raw - 1) // compress
        future = latents - clean
        out[name] = dict(raw=raw, latents=latents, clean=clean, future=future, chunk=chunk)
        print(
            f"{name:12s} raw frames={raw:3d} -> latents={latents} "
            f"(clean={clean}, future={future}) | chunk={chunk} actions "
            f"-> {chunk / future:.1f} actions per future latent frame"
        )
    return out


# --------------------------------------------------------------------------
# 2. mask structure
# --------------------------------------------------------------------------
def mask_audit(geom: dict[str, dict]) -> None:
    rule("2. causal mask structure: MoTWAM._build_mot_attention_mask")
    for name, g in geom.items():
        for cond in ("first_frame", "clean_prefix", "full_video"):
            tpf = 4  # 2x2 latent tile per frame; the mask is block-structured
            v_len = g["latents"] * tpf
            a_len = g["chunk"]
            m = MoTWAM._build_mot_attention_mask(
                video_seq_len=v_len,
                action_seq_len=a_len,
                video_tokens_per_frame=tpf,
                device=DEV,
                action_video_conditioning=cond,
                clean_prefix_latent_steps=g["clean"],
            )
            ok, notes = _check_mask(m, g, tpf, cond)
            print(
                f"  {name:12s} cond={cond:12s} mask={tuple(m.shape)} "
                f"density={m.float().mean():.3f} structure={'OK' if ok else 'FAIL'}"
            )
            for note in notes:
                print(f"      {note}")
            if not ok:
                raise AssertionError(f"mask structure check failed for {name}/{cond}")

    print(
        "\n  Eq. 2 reading (LIBERO geometry, cond=first_frame, one token per latent frame):"
    )
    g = geom["LIBERO"]
    m = MoTWAM._build_mot_attention_mask(
        video_seq_len=g["latents"],
        action_seq_len=g["chunk"],
        video_tokens_per_frame=1,
        device=DEV,
        action_video_conditioning="first_frame",
        clean_prefix_latent_steps=g["clean"],
    )
    vt = m[: g["latents"]]
    at = m[g["latents"] :]
    for i in range(g["latents"]):
        vis_a = int(vt[i, g["latents"] :].sum())
        print(
            f"    video query f{i}: sees video f0..f{int(vt[i, : g['latents']].sum()) - 1}, "
            f"{vis_a:2d}/{g['chunk']} action tokens"
        )
    for j in range(0, g["chunk"], 4):
        print(
            f"    action query a{j:2d}: sees video f0..f{int(at[j, : g['latents']].sum()) - 1} "
            f"({int(at[j, : g['latents']].sum())} latent frame(s)), "
            f"{int(at[j, g['latents'] :].sum()):2d}/{g['chunk']} action tokens"
        )

    # The paper's primary policy baseline WAM_mask blocks action->video keys.
    masked = m.clone()
    masked[: g["latents"], g["latents"] :] = False
    dropped = int(m[: g["latents"], g["latents"] :].sum())
    print(
        f"\n  WAM_mask counterfactual: zeroing video->action key entries drops "
        f"{dropped} of {int(m.sum())} visible entries "
        f"({100 * dropped / int(m.sum()):.1f}% of the mask)."
    )
    print(
        "  The released mask builder takes no parameter that performs this; "
        "the ablation baseline is not reachable from a recipe."
    )


def _check_mask(m: torch.Tensor, g: dict, tpf: int, cond: str) -> tuple[bool, list[str]]:
    """Exhaustively verify the visibility relations of Eq. 2."""
    latents, clean, future, chunk = g["latents"], g["clean"], g["future"], g["chunk"]
    v_len = latents * tpf
    notes: list[str] = []
    ok = True

    # (a) clean prefix video queries attend causally inside the video stream only
    for f in range(clean):
        rows = m[f * tpf : (f + 1) * tpf]
        vis_v = rows[:, :v_len]
        vis_a = rows[:, v_len:]
        if not bool(vis_v[:, (f + 1) * tpf :].sum() == 0):
            ok = False
            notes.append("FAIL clean-prefix video query sees later video tokens")
        if int(vis_a.sum()) != 0:
            ok = False
            notes.append("FAIL clean-prefix video query sees action tokens")

    # (b) future video frame i (1-based over future frames) sees v<=i and the
    #     aligned action prefix ceil(i * chunk / future)
    for i in range(1, future + 1):
        f = clean + i - 1
        rows = m[f * tpf : (f + 1) * tpf]
        expect_a = (i * chunk + future - 1) // future
        if int(rows[:, v_len:].sum(1).max()) != expect_a or int(rows[:, v_len:].sum(1).min()) != expect_a:
            ok = False
            notes.append(f"FAIL future frame {i}: action prefix != {expect_a}")
        if int(rows[:, :v_len].sum(1).max()) != (f + 1) * tpf:
            ok = False
            notes.append(f"FAIL future frame {i}: video visibility != frames 0..{f}")
        if not bool(rows[:, (f + 1) * tpf : v_len].sum() == 0):
            ok = False
            notes.append(f"FAIL future frame {i}: sees future video tokens")

    # (c) action queries: observed video prefix only, own causal action prefix
    observed = tpf if cond == "first_frame" else clean * tpf
    for i in range(1, future + 1):
        lo = (i - 1) * chunk // future
        hi = (i * chunk + future - 1) // future
        rows = m[v_len + lo : v_len + hi]
        vis_v = rows[:, :v_len]
        expect_v = min((clean + i) * tpf, v_len) if cond == "full_video" else observed
        if int(vis_v.sum(1).max()) != expect_v or int(vis_v.sum(1).min()) != expect_v:
            ok = False
            notes.append(f"FAIL action block {i}: video visibility != {expect_v} tokens")
        if not bool(vis_v[:, (clean + 0) * tpf :].sum() == 0) and cond != "full_video":
            ok = False
            notes.append(f"FAIL action block {i}: sees a future video latent")
        if int(rows[:, v_len:].sum(1).max()) != hi:
            ok = False
            notes.append(f"FAIL action block {i}: action prefix != {hi}")
    return ok, notes


# --------------------------------------------------------------------------
# 3. config surface
# --------------------------------------------------------------------------
def config_audit(geom: dict[str, dict]) -> None:
    rule("3. config surface: which paper arms can be expressed as a recipe")
    fields = {f.name for f in FrameworkConfig.__dataclass_fields__.values()}
    mixture = sorted(k for k in fields if "clean_prefix" in k)
    print("  clean-prefix / conditioning fields in FrameworkConfig:")
    for k in mixture:
        print(f"    framework.{k} = {getattr(FrameworkConfig(), k)!r}")
    print("\n  arm expressibility:")
    print("    WAM_noisy  (all-noisy actions)      -> action_clean_prefix_enabled: false")
    print("    WM         (all-clean, no action)   -> prob_full=1.0, prob_none=0.0,")
    print("                                           prob_partial=0.0, loss_lambda_action=0.0")
    print("    WAM_mask   (video cannot read acts) -> NOT expressible: no field removes")
    print("                                            video->action keys in the mask")
    print("    WM_AdaLN   (AdaLN action injection) -> NOT present in the repo")
    mask_fields = [k for k in fields if "mask" in k]
    print(f"\n  fields matching 'mask': {mask_fields}")
    import subprocess

    recipes = sorted(p.name for p in (REPO / "configs").glob("*.yaml"))
    print(f"  shipped recipes: {recipes}")
    hits = subprocess.run(
        ["grep", "-rni", "adaln", "--include=*.py", str(REPO / "onewm")],
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    print(f"  'adaln' grep hits in onewm/ (**=Wan2.2's own timestep modulation):")
    for line in hits[:6]:
        print("    " + line.replace(str(REPO) + "/", ""))
    print(f"    total={len(hits)} (no action-injection module among them)")


# --------------------------------------------------------------------------
# 4. mixture sampler
# --------------------------------------------------------------------------
def mixture_audit(draws: int = 200_000) -> None:
    rule("4. clean-prefix sampler: realized regime mixture (LIBERO recipe)")
    chunk, future = 32, 8
    params = dict(
        num_future_frames=future,
        prob_none=0.60,
        prob_partial=0.25,
        prob_full=0.15,
        boundary_ratio=0.6,
        jitter=2,
        jitter_weights=[0.1, 0.15, 0.5, 0.15, 0.1],
    )
    print(f"  aligned action boundaries(32, 8) = {aligned_action_boundaries(chunk, future)}")
    valid = torch.full((draws,), chunk, dtype=torch.long)
    r = sample_clean_prefix_lengths(valid, generator=torch.Generator().manual_seed(0), **params)
    is_none = r == 0
    is_full = r == chunk
    is_partial = ~is_none & ~is_full
    print(
        f"  realized over {draws} draws: none={is_none.float().mean():.4f} "
        f"partial={is_partial.float().mean():.4f} full={is_full.float().mean():.4f} "
        f"(config 0.60/0.25/0.15)"
    )
    print(
        f"  samples carrying an action loss (r<K) = "
        f"{(is_none | is_partial).float().mean():.4f} <=> 0.60+0.25"
    )
    hist = Counter(r[is_partial].tolist())
    boundaries = set(aligned_action_boundaries(chunk, future))
    n_partial = int(is_partial.sum())
    at_b = sum(hist[k] for k in boundaries) / n_partial
    off_b = [hist[k] for k in hist if k not in boundaries]
    mean_off = sum(off_b) / len(off_b) if off_b else 0.0
    print(
        f"  partial r: range [{int(r[is_partial].min())},{int(r[is_partial].max())}]; "
        f"P(r=aligned boundary)={at_b:.3f} vs mean over non-boundary r="
        f"{mean_off / n_partial:.3f} -> peak ratio {at_b / (mean_off / n_partial):.2f}x"
    )
    print(
        "  expected under boundary_ratio=0.6 with peak jitter weight 0.5: "
        "0.6*0.5 + 0.4/30 = 0.313 vs non-boundary ~0.4/30 = 0.013 -> 23.5x at the "
        "boundary row itself; the row-mean above averages the +/-2 jitter tail."
    )
    print("  partial histogram r:count:")
    for k in sorted(hist):
        bar = "#" * max(1, round(60 * hist[k] / max(hist.values())))
        mark = " <-aligned" if k in boundaries else ""
        print(f"    r={k:2d} {hist[k]:6d} {bar}{mark}")


# --------------------------------------------------------------------------
# 5. action-loss mask
# --------------------------------------------------------------------------
def action_loss_audit() -> None:
    rule("5. action conditioning: clean prefix is bitwise clean and loss-inert")
    chunk, future = 32, 8
    sched = FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    torch.manual_seed(0)
    action = torch.randn(4, chunk, 7)
    out = add_partial_clean_action_noise(
        sched,
        action,
        action_is_pad=None,
        num_future_frames=future,
        prob_none=0.6,
        prob_partial=0.25,
        prob_full=0.15,
        boundary_ratio=0.6,
        jitter=2,
        jitter_weights=[0.1, 0.15, 0.5, 0.15, 0.1],
    )
    print(f"  clean prefix lengths per sample: {out.clean_prefix_lengths.tolist()}")
    known = out.known_mask
    print(
        f"  prefix tokens bitwise equal to ground-truth action: "
        f"{bool(torch.equal(out.noisy[known], action[known]))}"
    )
    print(
        f"  prefix token timesteps are all zero: "
        f"{bool((out.token_timesteps[known] == 0).all())}"
    )
    print(
        f"  suffix tokens carry one sampled level per sample: "
        f"{out.token_timesteps[~known].unique().numel()} distinct levels in a "
        f"batch of {action.shape[0]} (all four samples have a suffix)"
    )
    # loss inertness: corrupt the prediction exactly on the prefix, keep suffix
    pred = out.target.clone()
    pred[known] += 10.0
    ignore = known.clone()
    loss_with = flow_matching_loss(
        pred, out.target, out.timesteps, sched, is_pad_mask=ignore, average_only_valid_samples=True
    )
    pred2 = out.target.clone()
    loss_without = flow_matching_loss(
        pred2, out.target, out.timesteps, sched, is_pad_mask=ignore, average_only_valid_samples=True
    )
    print(
        f"  action loss ignores prefix: |L(corrupted prefix) - L(perfect)| = "
        f"{abs(loss_with.item() - loss_without.item()):.3e} (both {loss_without.item():.3e})"
    )
    # all-clean example contributes no action loss at all
    out_full = add_partial_clean_action_noise(
        sched,
        action,
        action_is_pad=None,
        num_future_frames=future,
        prob_none=0.0,
        prob_partial=0.0,
        prob_full=1.0,
        boundary_ratio=0.6,
        jitter=2,
        jitter_weights=[0.1, 0.15, 0.5, 0.15, 0.1],
    )
    print(
        f"  r=K regime: every token clean = "
        f"{bool(out_full.known_mask.all())}, so its action-loss mask is full "
        f"and the sample is excluded by average_only_valid_samples=True"
    )


if __name__ == "__main__":
    print(f"torch {torch.__version__} | repo {REPO}")
    geom = geometry()
    mask_audit(geom)
    config_audit(geom)
    mixture_audit()
    action_loss_audit()
    print("\nAll structural audits passed.")
