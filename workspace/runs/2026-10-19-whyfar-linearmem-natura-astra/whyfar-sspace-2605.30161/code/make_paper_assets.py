"""Assemble paper-side figures for the report (crop white borders, cap 1600 px)."""
import os
from PIL import Image, ImageDraw, ImageFont

WS = os.environ.get(
    "PAPER_SCOUT_WS",
    os.path.dirname(os.path.abspath(__file__)) + "/../../../..",
)
OUT = os.path.join(WS, "runs", "2026-10-19-whyfar-linearmem-natura-astra", "assets")
WFLU = os.path.join(WS, "drafts", "images", "whyfarlooksup-2605.30161")
SS = os.path.join(WS, "drafts", "images", "sspace-2610.13527")


def trim(im, pad=8):
    im = im.convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    from PIL import ImageChops
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return im
    l, t, r, b = bbox
    l = max(0, l - pad); t = max(0, t - pad)
    r = min(im.width, r + pad); b = min(im.height, b + pad)
    return im.crop((l, t, r, b))


def cap(im, max_edge=1600):
    w, h = im.size
    s = min(1.0, max_edge / max(w, h))
    if s < 1:
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return im


def save(im, name):
    im = cap(im)
    path = os.path.join(OUT, name)
    im.save(path)
    print(name, im.size)


def main():
    os.makedirs(OUT, exist_ok=True)

    # Single paper figures used by the report
    save(trim(Image.open(os.path.join(WFLU, "img-07.png"))), "whyfar_wflu_teaser.png")
    save(trim(Image.open(os.path.join(WFLU, "img-01.png"))), "whyfar_wflu_probing.png")
    save(trim(Image.open(os.path.join(WFLU, "img-04.png"))), "whyfar_wflu_pca.png")
    save(trim(Image.open(os.path.join(SS, "img-26.png"))), "whyfar_sspace_formulation.png")

    # S-Space per-axis correlations: 2 rows (absolute, pairwise) x 3 axes
    names = [
        ["img-33.png", "img-34.png", "img-35.png"],
        ["img-36.png", "img-37.png", "img-38.png"],
    ]
    tiles = [[trim(Image.open(os.path.join(SS, n))) for n in row] for row in names]
    h = max(im.height for row in tiles for im in row)
    tiles = [[im.resize((int(im.width * h / im.height), h), Image.LANCZOS)
              for im in row] for row in tiles]
    gap = 16
    label_w = 78
    W = label_w + sum(im.width for im in tiles[0]) + gap * 2
    H = sum(h for _ in tiles) + gap
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    y = 0
    for ri, row in enumerate(tiles):
        x = label_w
        for im in row:
            canvas.paste(im, (x, y))
            x += im.width + gap
        label = "absolute\ncoordinates" if ri == 0 else "pairwise\ndifferences"
        draw.multiline_text((4, y + h // 2 - 14), label, fill=(60, 60, 60),
                            font=font, spacing=2)
        y += h + gap
    save(canvas, "whyfar_sspace_axis_correlations.png")


if __name__ == "__main__":
    main()
