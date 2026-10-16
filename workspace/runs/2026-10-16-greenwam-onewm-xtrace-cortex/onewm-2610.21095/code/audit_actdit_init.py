"""What the ActionDiT "interpolated Wan initialization" actually is.

The paper says the action expert is initialized by "interpolating" Wan2.2-TI2V-5B
transformer weights. This script runs the released converter
(`onewm.tools.preprocess_action_dit_init._convert_tensor`) on every key pair the
real conversion touches, with tensors of the exact shapes, and measures what the
operation preserves.

Wan2.2-TI2V-5B shapes are taken from the published config (dim=3072, ffn=14336,
24 heads x 128, text_dim=4096, freq_dim=256, 30 layers); the numbers below are
random draws, which is sufficient because the resample is a fixed linear map --
its scale behaviour is a property of the map, not of the weights. No Wan
checkpoint is required and none is distributed by the authors.

    ONEWM_REPO=/path/to/onewm-repo python audit_actdit_init.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
WORKSPACE = HERE.parents[4]
REPO = Path(os.environ.get("ONEWM_REPO", WORKSPACE / "code" / "onewm-repo"))
sys.path.insert(0, str(REPO))

from onewm.modules.action_dit import ActionDiT  # noqa: E402
from onewm.tools.preprocess_action_dit_init import _convert_tensor  # noqa: E402

torch.manual_seed(0)

VIDEO = dict(hidden=3072, ffn=14336, heads=24, head_dim=128, text=4096, freq=256, layers=30)
ACTION = dict(hidden=1024, ffn=4096, layers=30, action_dim=7)  # configs/libero.yaml


def rule(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def key_pairs() -> list[tuple[str, tuple[int, ...], tuple[int, ...], str]]:
    """(key, source shape, target shape, role) for one DiT block + stems."""
    h, f, a = VIDEO["hidden"], VIDEO["ffn"], ACTION["hidden"]
    af = ACTION["ffn"]
    att = VIDEO["heads"] * VIDEO["head_dim"]
    txt = VIDEO["text"]
    pairs = [
        ("blocks.0.self_attn.q.weight", (att, h), (att, a), "resample"),
        ("blocks.0.self_attn.k.weight", (att, h), (att, a), "resample"),
        ("blocks.0.self_attn.v.weight", (att, h), (att, a), "resample"),
        ("blocks.0.self_attn.o.weight", (h, att), (a, att), "resample"),
        ("blocks.0.cross_attn.q.weight", (att, h), (att, a), "resample"),
        ("blocks.0.cross_attn.k.weight", (att, txt), (att, txt), "verbatim"),
        ("blocks.0.cross_attn.v.weight", (att, txt), (att, txt), "verbatim"),
        ("blocks.0.cross_attn.o.weight", (h, att), (a, att), "resample"),
        ("blocks.0.ffn.0.weight", (f, h), (af, a), "resample"),
        ("blocks.0.ffn.2.weight", (h, f), (a, af), "resample"),
        ("blocks.0.self_attn.norm_q.weight", (att,), (att,), "verbatim"),
        ("blocks.0.norm3.weight", (h,), (h,), "verbatim"),
        ("blocks.0.modulation", (1, 6, h), (1, 6, a), "resample"),
        ("text_embedding.0.weight", (h, txt), (a, txt), "resample"),
        ("text_embedding.2.weight", (h, h), (a, a), "resample"),
        ("time_embedding.0.weight", (h, VIDEO["freq"]), (a, VIDEO["freq"]), "resample"),
        ("time_embedding.2.weight", (h, h), (a, a), "resample"),
        ("time_projection.1.weight", (6 * h, h), (6 * a, a), "resample"),
    ]
    return pairs


def param_counts() -> None:
    rule("0. module sizes at the two recipe geometries (configs/libero.yaml)")
    h, f, a, af = VIDEO["hidden"], VIDEO["ffn"], ACTION["hidden"], ACTION["ffn"]
    att, txt = VIDEO["heads"] * VIDEO["head_dim"], VIDEO["text"]
    per_video = 4 * att * h + (2 * att * txt + 2 * att * h) + 2 * h * f
    per_action = (3 * att * a + a * att) + (2 * att * txt + att * a + a * att) + 2 * a * af
    v = per_video * VIDEO["layers"]
    act = per_action * ACTION["layers"]
    print(f"  video expert   : {VIDEO['layers']} layers x {per_video / 1e6:7.1f}M = {v / 1e9:.2f}B params")
    print(f"  action expert  : {ACTION['layers']} layers x {per_action / 1e6:7.1f}M = {act / 1e9:.2f}B params")
    print(f"  attention dim (heads*head_dim) = {att} for both experts -> K/V spaces are")
    print("  shape-compatible, which is what lets the two experts share one attention.")


def resample_audit() -> None:
    rule("1. per-key conversion: verbatim copy, decimation, or genuine resample")
    pairs = key_pairs()
    verbatim = [p for p in pairs if p[1] == p[2]]
    print(f"  {'key':38s} {'source shape':16s} {'target shape':16s} outcome")
    for key, sshape, tshape, role in pairs:
        src = torch.randn(*sshape) * 0.02
        tgt = torch.zeros(*tshape)
        out, resized = _convert_tensor(src, tgt, apply_alpha_scaling=True)
        if not resized:
            outcome = "copy (identical shape)"
        else:
            # closest resampled counterpart: strided subsample with the axis ratio
            idx = [
                (torch.arange(n, dtype=torch.float64) * (s - 1) / (n - 1)).round().long()
                if s > 1 and n > 1
                else torch.zeros(n, dtype=torch.long)
                for s, n in zip(sshape, tshape)
            ]
            dec = src.to(torch.float64)[torch.meshgrid(*idx, indexing="ij")]
            alpha = (sshape[-1] / tshape[-1]) ** 0.5
            dev = out.to(torch.float64) / alpha - dec
            rel = float(dev.norm() / dec.norm())
            outcome = f"resample (alpha={alpha:.3f}, ||dev from decimation||/||dec||={rel:.3f})"
        print(f"  {key:38s} {str(sshape):16s} {str(tshape):16s} {outcome}")
    print(
        f"\n  {len(verbatim)}/{len(pairs)} sampled keys are copied verbatim "
        "(norms, text cross-attn k/v); the rest are resampled."
    )


def scale_audit() -> None:
    rule("2. what the resample preserves: iid output scale, not function")
    h, a = VIDEO["hidden"], ACTION["hidden"]
    att, f, af = VIDEO["heads"] * VIDEO["head_dim"], VIDEO["ffn"], ACTION["ffn"]
    cases = [
        ("one axis resized  q/k/v   (3072,3072)->(3072,1024)", (att, h), (att, a)),
        ("both axes resized ffn.0   (14336,3072)->(4096,1024)", (f, h), (af, a)),
    ]
    for label, sshape, tshape in cases:
        row = []
        for alpha_on in (True, False):
            src = torch.randn(*sshape) * 0.02
            out, _ = _convert_tensor(src, torch.zeros(*tshape), apply_alpha_scaling=alpha_on)
            y_src = torch.randn(64, sshape[1]) @ src.T
            y_tgt = torch.randn(64, tshape[1]) @ out.T
            row.append(
                (float(out.std() / src.std()), float(y_tgt.pow(2).mean() / y_src.pow(2).mean()))
            )
        (sd_on, var_on), (sd_off, var_off) = row
        print(
            f"  {label}\n"
            f"      entry std ratio {sd_off:.3f} -> {sd_on:.3f} (alpha {'on':>2}); "
            f"E||y||^2 target/source {var_off:.3f} -> {var_on:.3f}"
        )
    print(
        "  Reading: linear interpolation with align_corners=True spreads each output over\n"
        "  two taps with weights (f, 1-f), f ~ U(0,1), so one resized axis multiplies\n"
        "  variance by E[f^2+(1-f)^2] = 2/3. The last-dim alpha = sqrt(fan_src/fan_tgt)\n"
        "  restores the fan-in but not that 2/3 damping (and a both-axis resample pays it\n"
        "  twice, 4/9). Net: the action expert starts a little quieter than the layers it\n"
        "  was copied from -- the same order of magnitude, not an exact scale match."
    )
    print(
        "\n  What cannot survive is the computation: video latents enter the video expert\n"
        "  through a Conv3d patch embedding and actions enter the action expert through a\n"
        "  freshly initialized Linear(7 -> 1024). The two experts therefore share a token\n"
        "  layout, a K/V geometry (heads*head_dim = 3072) and a scale convention -- not a\n"
        "  function. ActionDiT keys never taken from Wan: action_encoder.weight/bias and\n"
        "  head.weight/bias (configs ship action_expert_head_init: random)."
    )


def resample_is_averaging() -> None:
    rule("3. is the resample a smoothing operator or a projection?")
    h, a = VIDEO["hidden"], ACTION["hidden"]
    src = torch.eye(h)
    tgt = torch.zeros(a, h)
    out, _ = _convert_tensor(src, tgt, apply_alpha_scaling=False)
    rows = out.sum(dim=1)
    support = (out > 1e-6).sum(dim=1).float()
    wmax = out.max(dim=1).values
    print(
        f"  3072 -> 1024 rows of the resampled identity:\n"
        f"    row sums   in [{float(rows.min()):.4f}, {float(rows.max()):.4f}] "
        f"(mean {float(rows.mean()):.4f}) -> convex combinations, no rescaling\n"
        f"    support    mean {float(support.mean()):.2f} source rows "
        f"(max {int(support.max())})\n"
        f"    top weight mean {float(wmax.mean()):.4f} (0.75 = the value expected from "
        "uniformly\n    distributed taps) -> a two-tap linear resample, not a block average"
    )


if __name__ == "__main__":
    print(f"torch {torch.__version__} | repo {REPO}")
    param_counts()
    resample_audit()
    scale_audit()
    resample_is_averaging()
