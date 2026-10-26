#!/usr/bin/env python
"""Compare generated videos without treating counterfactual inputs as ground truth."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_metrics import motion_stats, read_video


PAIRS = [
    ("rice_synthcam", "rice_full"),
    ("rice_skelshift", "rice_full"),
    ("rice_skelshuffle", "rice_full"),
    ("rice_noskel", "rice_full"),
    ("rice_none", "rice_full"),
    ("cut_nocam", "cut_full"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow_partial", action="store_true")
    args = parser.parse_args()
    results = {}
    missing = []
    for variant, baseline in PAIRS:
        variant_path = args.dir / f"{variant}_pred.mp4"
        baseline_path = args.dir / f"{baseline}_pred.mp4"
        if not variant_path.exists() or not baseline_path.exists():
            missing.append(f"{variant}_vs_{baseline}")
            continue
        a = read_video(variant_path)
        b = read_video(baseline_path)
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch for {variant} and {baseline}: {a.shape} vs {b.shape}")
        delta = a[1:] - b[1:]
        mse = float(np.mean(delta ** 2))
        mean_abs = np.abs(delta).mean(axis=(1, 2, 3))
        results[f"{variant}_vs_{baseline}"] = {
            "variant": variant,
            "baseline": baseline,
            "frames_compared": len(delta),
            "pair_psnr_db": 99.0 if mse <= 1e-12 else float(10 * np.log10(1 / mse)),
            "mean_abs_rgb": float(mean_abs.mean()),
            "pixel_fraction_mean_rgb_gt_0p1": float((np.abs(delta).mean(axis=3) > 0.1).mean()),
            "motion_mean_variant": motion_stats(a)["motion_mean"],
            "motion_mean_baseline": motion_stats(b)["motion_mean"],
            "mean_abs_rgb_by_frame": mean_abs.tolist(),
        }
    if missing and not args.allow_partial:
        raise RuntimeError(f"missing comparisons: {missing}")
    payload = {
        "metric_scope": "Differences between generated videos with the same first frame, text and seed; not adherence to a ground-truth future.",
        "rgb_scale": "float RGB in [0,1]; pair PSNR uses MAX=1 and excludes the clamped first frame",
        "pixel_fraction_definition": "fraction of pixels whose mean absolute RGB difference exceeds 0.1",
        "missing_comparisons": missing,
        "comparisons": results,
    }
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"written": str(args.out), "n_comparisons": len(results), "missing": missing}, indent=2))


if __name__ == "__main__":
    main()
