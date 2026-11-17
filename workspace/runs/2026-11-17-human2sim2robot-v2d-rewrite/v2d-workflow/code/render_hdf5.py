#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-sheet", type=Path, required=True)
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    with h5py.File(args.input, "r") as h5:
        demo = h5["data"][args.demo]
        names = ["image", "image_left_wrist", "image_right_wrist"]
        arrays = [demo["obs"][name][...] for name in names]
        labels = ["ego", "left wrist", "right wrist"]
    if len({array.shape[0] for array in arrays}) != 1:
        raise ValueError("camera lengths differ")
    if any(array.ndim != 4 or array.shape[-1] != 3 for array in arrays):
        raise ValueError("expected RGB images with shape (time, height, width, 3)")

    def compose(index):
        panels = []
        for array in arrays:
            image = Image.fromarray(array[index].astype(np.uint8))
            panels.append(image.resize((image.width * 2, image.height * 2), Image.Resampling.BICUBIC))
        frame = Image.new("RGB", (sum(image.width for image in panels), panels[0].height + 28), "white")
        draw = ImageDraw.Draw(frame)
        x = 0
        for label, panel in zip(labels, panels):
            draw.text((x + 8, 8), f"{label} | t={index / args.fps:.2f}s", fill="black")
            frame.paste(panel, (x, 28))
            x += panel.width
        return frame

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.output_sheet.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        args.output_video,
        fps=args.fps,
        codec="libx264",
        output_params=["-crf", "28", "-preset", "veryfast"],
    ) as writer:
        for index in range(arrays[0].shape[0]):
            writer.append_data(np.asarray(compose(index)))
    indices = np.linspace(0, arrays[0].shape[0] - 1, 4).round().astype(int)
    rows = [compose(int(index)) for index in indices]
    sheet = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(args.output_sheet)
    print({
        "demo": args.demo,
        "frames": int(arrays[0].shape[0]),
        "source_shapes": [list(array.shape) for array in arrays],
        "video": str(args.output_video),
        "sheet": str(args.output_sheet),
    })


if __name__ == "__main__":
    main()
