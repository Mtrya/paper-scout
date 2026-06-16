#!/usr/bin/env python3
"""
Generate report-facing figures/tables for the DreamX-World thread.

Outputs:
  ../../assets/dreamx_code_mapping_table.png
  ../../assets/dreamx_inference_pipeline.png
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ASSET_DIR = Path(__file__).resolve().parents[2] / "assets"


def get_font(size):
    # Try a few common monospace fonts; fall back to default if none exist.
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_table(filename, title, headers, rows, col_widths):
    cell_h = 32
    header_h = 40
    pad = 8
    w = sum(col_widths) + pad * 2
    h = header_h + cell_h * len(rows) + pad * 2
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    font = get_font(14)
    title_font = get_font(16)

    # Title
    d.text((pad, pad), title, fill="black", font=title_font)

    y = pad + header_h - 5
    # Header bg
    d.rectangle([pad, y, w - pad, y + cell_h], fill="#eeeeee", outline="black")
    x = pad
    for i, head in enumerate(headers):
        d.text((x + 4, y + 6), head, fill="black", font=font)
        x += col_widths[i]

    y += cell_h
    for row in rows:
        x = pad
        for i, cell in enumerate(row):
            d.rectangle([x, y, x + col_widths[i], y + cell_h], outline="#cccccc")
            d.text((x + 4, y + 6), str(cell), fill="black", font=font)
            x += col_widths[i]
        y += cell_h

    img.save(filename)
    print(f"Saved {filename}")


def draw_pipeline(filename):
    """Simple block diagram of the AR inference pipeline."""
    w, h = 900, 520
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    font = get_font(14)
    title_font = get_font(18)
    d.text((20, 15), "DreamX-World autoregressive inference pipeline (released code)", fill="black", font=title_font)

    boxes = [
        (80, 70, 220, 110, "Input JSON\n(prompt + actions)"),
        (320, 70, 480, 110, "Trajectory processor\n(action_seq -> poses)"),
        (560, 70, 760, 110, "PRoPE camera dict\n{viewmats, K}"),
        (80, 150, 260, 190, "VAE encode\nstart image -> latent"),
        (320, 150, 520, 190, "Prepare noise\nframe 0 = latent"),
        (560, 150, 760, 190, "Causal DiT\nKV-cache denoising"),
        (80, 230, 300, 270, "Few-step scheduler\n[1000,750,500,250]"),
        (360, 230, 560, 270, "Rolling KV cache\nlocal window + sinks"),
        (600, 230, 780, 270, "Block-Relativistic\nRoPE"),
        (200, 310, 500, 350, "VAE decode_to_pixel\n(cached_decode)"),
        (560, 310, 760, 350, "Post-process\nLab color correction"),
        (320, 390, 520, 430, "Output MP4 video"),
    ]

    for x1, y1, x2, y2, text in boxes:
        d.rounded_rectangle([x1, y1, x2, y2], radius=8, outline="#2c3e50", width=2, fill="#ecf0f1")
        # Compute multiline text size manually
        lines = text.split("\n")
        tw = max((font.getbbox(line)[2] - font.getbbox(line)[0]) for line in lines)
        line_h = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
        th = line_h * len(lines)
        d.multiline_text(((x1 + x2 - tw) // 2, (y1 + y2 - th) // 2 - 6), text, fill="black", font=font, align="center")

    arrows = [
        (220, 90, 320, 90),
        (480, 90, 560, 90),
        (660, 110, 660, 150),
        (200, 170, 320, 170),
        (520, 170, 560, 170),
        (760, 170, 800, 170),
        (800, 170, 800, 250),
        (800, 250, 780, 250),
        (660, 190, 660, 230),
        (300, 230, 360, 250),
        (560, 250, 600, 250),
        (180, 270, 180, 310),
        (440, 270, 440, 310),
        (680, 270, 680, 310),
        (500, 350, 500, 390),
        (500, 430, 500, 460),
    ]
    for x1, y1, x2, y2 in arrows:
        d.line([(x1, y1), (x2, y2)], fill="#2c3e50", width=2)
        # arrow head
        d.polygon([(x2, y2), (x2 - 6, y2 - 4), (x2 - 6, y2 + 4)], fill="#2c3e50")

    # Annotation
    d.text((20, 470), "Observation: the released repo implements the causal inference path but omits training/DMD/RL code.", fill="#7f8c8d", font=font)

    img.save(filename)
    print(f"Saved {filename}")


def main():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    headers = ["Paper contribution", "Code file", "Key symbol / lines", "Status"]
    rows = [
        ["E-PRoPE camera control", "models/prope_utils.py", "prope_qkv", "implemented"],
        ["E-PRoPE camera control", "models/wan_transformer3d.py", "PropeSelfAttention 190-270", "implemented"],
        ["Causal AR generation", "pipeline/pipeline_causal_camera.py", "inference 47-254", "implemented"],
        ["Block-Relativistic RoPE", "wan/modules/causal_camera_model_2_2_prope_infinity.py", "block_relativistic_rope 17-55", "implemented"],
        ["Memory (geometry retrieval)", "—", "not in released code", "missing"],
        ["Event instruction tuning", "—", "only prompt text", "missing"],
        ["DMD distillation training", "—", "not in released code", "missing"],
        ["RL post-training", "—", "not in released code", "missing"],
        ["Inference acceleration", "models/attention_utils.py", "Sage/Flash attention", "implemented"],
        ["Inference acceleration", "utils/fp8_optimization.py", "FP8 conversion", "implemented"],
    ]
    col_widths = [210, 230, 220, 150]
    draw_table(ASSET_DIR / "dreamx_code_mapping_table.png", "DreamX-World: paper contribution -> code mapping", headers, rows, col_widths)

    draw_pipeline(ASSET_DIR / "dreamx_inference_pipeline.png")


if __name__ == "__main__":
    main()
