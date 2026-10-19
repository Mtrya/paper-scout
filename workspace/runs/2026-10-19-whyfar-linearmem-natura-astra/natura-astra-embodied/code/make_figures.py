#!/usr/bin/env python3
"""Build the thread's derived figures: 12-dimension chart + video contact sheets.

Requires: python3 stdlib, google-chrome (headless), ffmpeg-generated frames in
code/videos/. Run from the thread directory.
"""
from __future__ import annotations

import base64
import html
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VID = Path(os.environ.get("NATURA_VIDEO_DIR", ROOT / "code" / "videos"))
OUT_REPORT = ROOT.parent / "assets"
OUT_THREAD = OUT_REPORT
OUT_REPORT.mkdir(exist_ok=True)
# all figures live at the run-level assets/ directory

DIMS = [
    ("D01", "Pose control & contact", 56.9, 63.2, 14.4, 8.6),
    ("D02", "Bimanual coordination", 50.0, 61.5, 0.0, 0.0),
    ("D03", "In-hand manipulation", 16.7, 16.7, None, None),
    ("D04", "Tool use", 57.3, 66.7, 10.8, 0.0),
    ("D05", "Goal-directed reasoning", 32.8, 35.4, 15.5, 10.4),
    ("D06", "Spatial & scene generalization", 90.9, 100.0, 39.3, 20.6),
    ("D07", "Tool & object generalization", 100.0, 100.0, 40.8, 10.6),
    ("D08", "New-action generalization", 29.4, 64.3, 13.8, 7.8),
    ("D09", "Compositional generalization", 73.7, 77.8, 65.1, 36.2),
    ("D10", "Planning", 38.5, 55.6, 20.6, 25.0),
    ("D11", "Memory", 57.7, 70.0, 37.5, 10.9),
    ("D12", "Recovery", 68.6, 83.3, 75.0, 44.6),
]
SERIES = [
    ("Natura-1 Base (20-DoF hands)", "#2f5d8a", 2),
    ("Natura-1 Agentic (20-DoF hands)", "#2a9d8f", 3),
    ("GPT-6-Astra (gripper)", "#e76f51", 4),
    ("pi-0.5 (gripper)", "#8d99ae", 5),
]


def svg_chart() -> str:
    W, H = 1600, 1100
    left, right, top, bottom = 390, 150, 120, 80
    plot_w = W - left - right
    row_h = (H - top - bottom) / len(DIMS)
    bar_h = 12
    gap = 3
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="40" y="44" font-family="Noto Sans, DejaVu Sans, sans-serif" font-size="26" font-weight="700" fill="#1d2433">Natura-1\'s 12 capability dimensions vs GPT-6-Astra / π0.5</text>',
        '<text x="40" y="72" font-family="Noto Sans, DejaVu Sans, sans-serif" font-size="15" fill="#5b6475">Published scores (%). Natura-1 runs on 20-DoF dexterous hands; both baselines run on the ARX AC One gripper platform.</text>',
    ]
    for xv in range(0, 101, 20):
        x = left + plot_w * xv / 100
        parts.append(f'<line x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{H-bottom+4}" stroke="#e4e8ef" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{top-18}" text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="13" fill="#7a8496">{xv}</text>')
    group_bounds = [(0, 4), (4, 9), (9, 12)]
    for i, (a, b) in enumerate(group_bounds):
        if i % 2 == 0:
            y0 = top + a * row_h
            y1 = top + b * row_h
            parts.append(f'<rect x="{left-8}" y="{y0:.1f}" width="{plot_w+16}" height="{y1-y0:.1f}" fill="#f7f9fc"/>')
    for i, (dim, label, base, agentic, astra, pi05) in enumerate(DIMS):
        y0 = top + i * row_h
        parts.append(f'<text x="{left-14}" y="{y0+row_h/2+5:.1f}" text-anchor="end" font-family="Noto Sans, DejaVu Sans, sans-serif" font-size="14" fill="#1d2433">{html.escape(label)}</text>')
        parts.append(f'<text x="16" y="{y0+row_h/2+5:.1f}" font-family="DejaVu Sans, sans-serif" font-size="11" fill="#98a2b3">{dim}</text>')
        for si, color in enumerate(["#2f5d8a", "#2a9d8f", "#e76f51", "#8d99ae"]):
            val = [base, agentic, astra, pi05][si]
            y = y0 + (row_h - (4 * bar_h + 3 * gap)) / 2 + si * (bar_h + gap)
            if val is None:
                parts.append(f'<line x1="{left}" y1="{y+bar_h/2:.1f}" x2="{left+18}" y2="{y+bar_h/2:.1f}" stroke="#c7ced9" stroke-width="2"/>')
                parts.append(f'<text x="{left+24}" y="{y+bar_h-1:.1f}" font-family="DejaVu Sans, sans-serif" font-size="10" fill="#98a2b3">n/s</text>')
                continue
            w = max(1.5, plot_w * val / 100)
            parts.append(f'<rect x="{left}" y="{y:.1f}" width="{w:.1f}" height="{bar_h}" fill="{color}" rx="2"/>')
            if val >= 92:
                parts.append(f'<text x="{left+w-5:.1f}" y="{y+bar_h-2:.1f}" text-anchor="end" font-family="DejaVu Sans, sans-serif" font-size="10" fill="white">{val:.1f}</text>')
            else:
                parts.append(f'<text x="{left+w+5:.1f}" y="{y+bar_h-2:.1f}" font-family="DejaVu Sans, sans-serif" font-size="10" fill="#4b5563">{val:.1f}</text>')
    legend = [
        ("Natura-1 Base", "#2f5d8a", 390),
        ("Natura-1 Agentic", "#2a9d8f", 580),
        ("GPT-6-Astra (gripper)", "#e76f51", 800),
        ("π0.5 (gripper)", "#8d99ae", 1080),
    ]
    for name, color, lx in legend:
        parts.append(f'<rect x="{lx}" y="{H-42}" width="13" height="13" fill="{color}" rx="2"/>')
        parts.append(f'<text x="{lx+18}" y="{H-32}" font-family="Noto Sans, DejaVu Sans, sans-serif" font-size="13" fill="#4b5563">{name}</text>')
    parts.append(f'<text x="{W-right}" y="{H-10}" text-anchor="end" font-family="DejaVu Sans, sans-serif" font-size="10" fill="#98a2b3">Source: naturewill.com/blog/natura-1 benchmark embed. n/s = not scored.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def chrome_shot(src: Path, out: Path, width: int, height: int) -> None:
    subprocess.run(
        [
            "google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
            "--hide-scrollbars", f"--window-size={width},{height}",
            f"--screenshot={out}", src.as_uri(),
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def sheet_html(frames: list[tuple[str, str]], cols: int, width: int) -> tuple[str, int]:
    tile_w = width // cols
    rows = (len(frames) + cols - 1) // cols
    # all source frames are 16:9 except wontons (1920x1620); use auto height
    tiles = []
    for path, cap in frames:
        tiles.append(
            f'<figure><img src="data:image/png;base64,{b64(VID / path)}"><figcaption>{html.escape(cap)}</figcaption></figure>'
        )
    css = f"""
    <style>
      * {{ box-sizing: border-box; }}
      body {{ margin:0; background:#fff; width:{width}px; font-family: 'Noto Sans','DejaVu Sans',sans-serif; }}
      .grid {{ display:grid; grid-template-columns: repeat({cols}, 1fr); gap:8px; padding:8px; }}
      figure {{ margin:0; background:#0f172a; border-radius:4px; overflow:hidden; }}
      img {{ display:block; width:100%; height:auto; }}
      figcaption {{ color:#e5e7eb; font-size:13px; padding:6px 8px; background:#111827; }}
      h1 {{ font-size:20px; margin:10px 12px 4px; color:#111827; }}
      p {{ font-size:13px; margin:0 12px 8px; color:#4b5563; }}
    </style>
    """
    body = '<div class="grid">' + "".join(tiles) + "</div>"
    return f"<!doctype html><html><head><meta charset='utf-8'>{css}</head><body>{body}</body></html>", rows


def make_sheets() -> None:
    required = ["ww_5.png", "r_0.png", "c_0.png", "l_0.png"]
    if not all((VID / name).exists() for name in required):
        print(f"frames missing under {VID}; skipping video sheets")
        return
    specs = [
        (
            "sheet_wontons.html", "natura_wontons_sheet.png", 3, 1600,
            "Natura-1 · making wontons (72.6 s, 1x): one continuous take, three synchronized views",
            "No scene cut: max ffmpeg scene score 0.0075 across 3,628 frames. Top: third-person; bottom: two close views.",
            [
                ("ww_5.png", "t=5 s: both hands enter the workspace"),
                ("ww_20.png", "t=20 s: wrapper held, right hand works the filling"),
                ("ww_35.png", "t=35 s: bimanual fold begins"),
                ("ww_45.png", "t=45 s: both hands close the wrapper"),
                ("ww_55.png", "t=55 s: shaping the fold"),
                ("ww_65.png", "t=65 s: final placement at the plate"),
            ],
        ),
        (
            "sheet_rod.html", "natura_rod_basket_sheet.png", 5, 1600,
            "Natura-1 · rod pulls an out-of-reach basket (9.2 s, 2x)",
            "No scene cut: max scene score 0.036. The clip starts with the rod already grasped; it does not show the approach.",
            [
                ("r_0.png", "0 s: rod grasped, basket on the high ledge"),
                ("r_2.png", "2 s: pull the near end"),
                ("r_4.png", "4 s: basket slides toward the edge"),
                ("r_6.png", "6 s: basket on the table, left hand reaches in"),
                ("r_8.png", "8 s: yarn lifted out"),
            ],
        ),
        (
            "sheet_card.html", "natura_card_edge_sheet.png", 5, 1600,
            "Natura-1 · new grasp via the box edge (9.1 s, 2x)",
            "No scene cut: max scene score 0.0088. The card is flat at the start and overhangs the box edge before the lift.",
            [
                ("c_0.png", "0 s: card flat on the box, hands open"),
                ("c_2.png", "2 s: left hand slides the card"),
                ("c_4.png", "4 s: card overhangs, fingers find the gap"),
                ("c_6.png", "6 s: lift from below"),
                ("c_8.png", "8 s: two-hand hold"),
            ],
        ),
        (
            "sheet_lemon.html", "natura_lemon_sheet.png", 3, 1600,
            "Natura-1 · lemon tracking under human interference (48.6 s, 2x)",
            "No scene cut: max scene score 0.018. In the sampled frames the human moves the lemon with a bare hand, not a stick; at t=32 s the lemon is placed into the robot's palm.",
            [
                ("l_0.png", "0 s: human hand moves the lemon"),
                ("l_8.png", "8 s: robot follows"),
                ("l_16.png", "16 s: human repositions the lemon"),
                ("l_24.png", "24 s: robot reaches"),
                ("l_32.png", "32 s: human places lemon into the robot hand"),
                ("l_40.png", "40 s: robot holds the lemon"),
            ],
        ),
    ]
    for html_name, png_name, cols, width, title, note, frames in specs:
        # insert a title/note header into the grid HTML
        body, rows = sheet_html(frames, cols, width)
        body = body.replace("<body>", f"<body><h1>{html.escape(title)}</h1><p>{html.escape(note)}</p>")
        src = OUT_THREAD / html_name
        src.write_text(body, encoding="utf-8")
        # approximate height: header 70 + rows * tile height + captions
        # tile height depends on source aspect; use 16:9 except wontons 1.185
        aspect = 0.84375 if "wontons" in png_name else 0.5625
        tile_h = int((width - 2 * 8 - (cols - 1) * 8) / cols * aspect)
        height = 80 + rows * (tile_h + 34) + 12
        chrome_shot(src, OUT_THREAD / png_name, width, height)
        src.unlink()
        print("wrote", OUT_THREAD / png_name, f"{width}x{height}")


if __name__ == "__main__":
    svg = svg_chart()
    svg_path = OUT_REPORT / "natura_12dims.svg"
    svg_path.write_text(svg, encoding="utf-8")
    chrome_shot(svg_path, OUT_REPORT / "natura_12dims.png", 1600, 1100)
    print("wrote", OUT_REPORT / "natura_12dims.png")
    make_sheets()
