"""Draw the joint attention mask OneWM trains with, and its WAM_mask ablation.

Renders the released mask for the LIBERO geometry (9 latent frames x 4 tokens,
1 clean frame, 32-action chunk) side by side with the masked variant the paper
uses as its primary policy baseline, plus the abstract block view used in the
paper's Figure 1. Pure PIL, deterministic.

    ONEWM_REPO=/path/to/onewm-repo python make_mask_figure.py --out onewm-mask-visibility.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve()
WORKSPACE = HERE.parents[4]
REPO = Path(os.environ.get("ONEWM_REPO", WORKSPACE / "code" / "onewm-repo"))
sys.path.insert(0, str(REPO))

from onewm.wam.mot_wam import MoTWAM  # noqa: E402

FRAMES, TPF, CLEAN, CHUNK = 9, 4, 1, 32
V_TOKENS = FRAMES * TPF
CELL = 9
PAD_L, PAD_T, PAD_R, PAD_B = 78, 46, 18, 46


def shipped() -> torch.Tensor:
    return MoTWAM._build_mot_attention_mask(
        video_seq_len=V_TOKENS,
        action_seq_len=CHUNK,
        video_tokens_per_frame=TPF,
        device=torch.device("cpu"),
        action_video_conditioning="first_frame",
        clean_prefix_latent_steps=CLEAN,
    )


def masked(m: torch.Tensor) -> torch.Tensor:
    out = m.clone()
    out[:V_TOKENS, V_TOKENS:] = False
    return out


COL_KEEP = (59, 110, 150)
COL_DROP = (205, 150, 90)
CLEAN_TINT = (232, 238, 242)


def draw(m: torch.Tensor, title: str, note: str) -> Image.Image:
    n = m.shape[0]
    w = PAD_L + n * CELL + PAD_R
    h = PAD_T + n * CELL + PAD_B
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    for i in range(n):
        for j in range(n):
            x0, y0 = PAD_L + j * CELL, PAD_T + i * CELL
            if i < V_TOKENS and j < V_TOKENS:
                fill = CLEAN_TINT if i < CLEAN * TPF else (243, 245, 247)
            else:
                fill = (250, 250, 250)
            d.rectangle([x0, y0, x0 + CELL - 1, y0 + CELL - 1], fill=fill)
    base = shipped()
    for i in range(n):
        for j in range(n):
            if not base[i, j]:
                continue
            x0, y0 = PAD_L + j * CELL, PAD_T + i * CELL
            colour = COL_KEEP if m[i, j] else COL_DROP
            d.rectangle([x0, y0, x0 + CELL - 1, y0 + CELL - 1], fill=colour)
    d.rectangle(
        [PAD_L, PAD_T, PAD_L + n * CELL - 1, PAD_T + n * CELL - 1], outline=(120, 120, 120)
    )
    for f in range(1, FRAMES):
        x = PAD_L + f * TPF * CELL
        d.line([x, PAD_T, x, PAD_T + n * CELL], fill=(255, 255, 255))
    d.line(
        [PAD_L, PAD_T + V_TOKENS * CELL, PAD_L + n * CELL, PAD_T + V_TOKENS * CELL],
        fill=(150, 150, 150),
    )
    d.text((PAD_L, PAD_T - 34), title, fill=(20, 20, 20))
    d.text((PAD_L, PAD_T - 20), note, fill=(80, 80, 80))
    d.text((10, PAD_T), "video q", fill=(60, 60, 60))
    d.text((10, PAD_T + V_TOKENS * CELL - 6), "action q", fill=(60, 60, 60))
    d.text((PAD_L + 8, PAD_T + n * CELL + 8), "video keys  (9 latent frames)", fill=(60, 60, 60))
    d.text((PAD_L + V_TOKENS * CELL + 8, PAD_T + n * CELL + 8), "action keys (32)", fill=(60, 60, 60))
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "onewm-mask-visibility.png"))
    args = parser.parse_args()
    m = shipped()
    mm = masked(m)
    left = draw(
        m,
        "OneWM: released causal mask (LIBERO recipe)",
        f"density {m.float().mean():.3f}; future video frame i reads exactly 4i actions",
    )
    right = draw(
        mm,
        "WAM_mask: video queries cannot read action keys",
        f"density {mm.float().mean():.3f} "
        f"({int(m.sum() - mm.sum())} of {int(m.sum())} visible entries dropped); "
        "video output becomes bitwise independent of actions",
    )
    gap = 26
    sheet = Image.new("RGB", (left.width + gap + right.width, left.height), "white")
    sheet.paste(left, (0, 0))
    sheet.paste(right, (left.width + gap, 0))
    d = ImageDraw.Draw(sheet)
    d.text((left.width + gap // 2 - 3, 6), "|", fill=(0, 0, 0))
    sheet.save(args.out)
    print(f"wrote {args.out} ({sheet.width}x{sheet.height})")
    print(
        "legend: dark blue = visible in both variants, orange = visible in OneWM only"
        " (dropped by the ablation), pale = never visible"
    )


if __name__ == "__main__":
    main()
