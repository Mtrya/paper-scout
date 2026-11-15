#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"objects": {}}
    for directory in sorted((args.reconstruction / "masks").iterdir()):
        if not directory.is_dir() or not directory.name.isdigit():
            continue
        coverages = []
        areas = []
        centroids = []
        image_shape = None
        for path in sorted(directory.glob("*.png")):
            mask = np.asarray(Image.open(path).convert("L")) > 0
            image_shape = list(mask.shape)
            count = int(mask.sum())
            areas.append(count)
            coverages.append(float(mask.mean()))
            if count:
                yy, xx = np.nonzero(mask)
                centroids.append([float(xx.mean()), float(yy.mean())])
            else:
                centroids.append([np.nan, np.nan])
        centroids = np.asarray(centroids)
        valid = np.isfinite(centroids).all(axis=1)
        steps = np.linalg.norm(np.diff(centroids[valid], axis=0), axis=1)
        diagonal = float(np.hypot(*image_shape)) if image_shape else None
        result["objects"][directory.name] = {
            "image_shape_hw": image_shape,
            "frames": len(areas),
            "empty_frames": int(np.sum(np.asarray(areas) == 0)),
            "nonzero_fraction": float(np.mean(np.asarray(areas) > 0)) if areas else 0.0,
            "coverage_fraction": stats(coverages),
            "area_pixels": stats(areas),
            "centroid_step_pixels": stats(steps),
            "centroid_step_image_diagonal": stats(steps / diagonal) if diagonal else {"count": 0},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
