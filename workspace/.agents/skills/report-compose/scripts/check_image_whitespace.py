#!/usr/bin/env python3
"""报告图片白边机械检查与裁剪。

扫描报告图片四周的大面积空白(近白行/列),这是反复出现过的交付事故:
带白边的截图进文档后有效内容被压小。两种用法:

  # 检查(任何一边空白占比超过 --warn 阈值则退出码 1)
  python check_image_whitespace.py img1.png img2.jpg ...

  # 自动裁掉白边(原处改写,保留 --pad 像素余量)
  python check_image_whitespace.py --crop img1.png ...

判定:逐行/列计算近白像素占比(采样加速),连续 ≥98% 近白的外围行/列计为白边。
只依赖 Pillow。被 verify_run / push_report 以子进程方式调用,也可独立使用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

WHITE = 245        # 近白阈值(0-255)
ROW_WHITE = 0.98   # 一行/列近白像素占比达到此值才算空白行/列


def measure(path: Path) -> tuple[tuple[float, float, float, float], tuple[int, int]]:
    """返回 ((top, bottom, left, right) 空白占比, (w, h))。"""
    im = Image.open(path).convert("L")
    w, h = im.size
    px = im.load()
    xs = list(range(0, w, 4))
    ys = list(range(0, h, 4))

    def row_white(y: int) -> float:
        return sum(1 for x in xs if px[x, y] > WHITE) / len(xs)

    def col_white(x: int) -> float:
        return sum(1 for y in ys if px[x, y] > WHITE) / len(ys)

    def trim(measure_fn, n: int) -> tuple[int, int]:
        lead = 0
        while lead < n and measure_fn(lead) >= ROW_WHITE:
            lead += 1
        trail = 0
        while trail < n - lead and measure_fn(n - 1 - trail) >= ROW_WHITE:
            trail += 1
        return lead, trail

    t, b = trim(row_white, h)
    l, r = trim(col_white, w)
    return (t / h, b / h, l / w, r / w), (w, h)


def crop(path: Path, pad: int) -> tuple[float, float, float, float]:
    """裁掉白边,原处改写。返回裁剪前各边空白占比。"""
    fracs, (w, h) = measure(path)
    t, b, l, r = fracs
    im = Image.open(path)
    box = (max(0, int(l * w) - pad), max(0, int(t * h) - pad),
           min(w, w - int(r * w) + pad), min(h, h - int(b * h) + pad))
    if box[2] > box[0] and box[3] > box[1] and box != (0, 0, w, h):
        im.crop(box).save(path)
    return fracs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--warn", type=float, default=0.10, help="单边空白占比警告阈值(默认 0.10)")
    ap.add_argument("--crop", action="store_true", help="自动裁掉白边(原处改写)")
    ap.add_argument("--pad", type=int, default=8, help="裁剪时保留的余量像素(默认 8)")
    args = ap.parse_args()

    bad = 0
    for f in args.images:
        if args.crop:
            fracs = crop(f, args.pad)
            action = "cropped"
        else:
            fracs, _ = measure(f)
            action = "checked"
        t, b, l, r = fracs
        worst = max(fracs)
        flag = ""
        if worst > args.warn:
            bad += 1
            flag = f"  <-- 白边超过 {args.warn:.0%}"
        print(f"{f.name}: top {t:.1%} bottom {b:.1%} left {l:.1%} right {r:.1%} [{action}]{flag}")
    sys.exit(1 if bad and not args.crop else 0)


if __name__ == "__main__":
    main()
