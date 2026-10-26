#!/usr/bin/env python
"""Measure local MediaPipe hand keypoints against supplied projected annotations."""
import argparse
import itertools
import json
from pathlib import Path

import imageio
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def match_hands(predicted, target):
    if len(predicted) == 0 or len(target) == 0:
        return [], []
    valid = [np.isfinite(h).all(axis=1) for h in target]
    costs = np.array([
        [np.linalg.norm(p[mask] - h[mask], axis=1).mean() if mask.any() else 1e6
         for h, mask in zip(target, valid)]
        for p in predicted
    ])
    candidates = []
    if len(predicted) <= len(target):
        for assignment in itertools.permutations(range(len(target)), len(predicted)):
            candidates.append((sum(costs[i, j] for i, j in enumerate(assignment)),
                               [(i, j) for i, j in enumerate(assignment)]))
    else:
        for assignment in itertools.permutations(range(len(predicted)), len(target)):
            candidates.append((sum(costs[i, j] for i, j in enumerate(assignment)),
                               [(i, j) for j, i in enumerate(assignment)]))
    pairs = min(candidates, key=lambda item: item[0])[1] if candidates else []
    matched = {(i, j) for i, j in pairs}
    return pairs, [j for j in range(len(target)) if j not in {j for _, j in matched}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--overlay", type=Path)
    args = parser.parse_args()

    labels = np.load(args.labels)["hand_kp"].astype(np.float64)
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    reader = imageio.get_reader(args.video)
    frames = [np.ascontiguousarray(frame[..., :3]) for frame in reader]
    fps = float(reader.get_meta_data().get("fps", 16))
    reader.close()
    if len(frames) != len(labels):
        raise ValueError(f"video has {len(frames)} frames, labels have {len(labels)}")

    records = []
    all_errors = []
    overlays = []
    with vision.HandLandmarker.create_from_options(options) as detector:
        for index, (frame, target) in enumerate(zip(frames, labels)):
            height, width = frame.shape[:2]
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=frame),
                int(round(index * 1000 / fps)),
            )
            predicted = [
                np.array([(point.x * width, point.y * height) for point in hand])
                for hand in result.hand_landmarks
            ]
            pairs, missed = match_hands(predicted, target)
            errors = []
            for pred_index, target_index in pairs:
                valid = np.isfinite(target[target_index]).all(axis=1)
                errors.extend(np.linalg.norm(
                    predicted[pred_index][valid] - target[target_index][valid], axis=1
                ).tolist())
            all_errors.extend(errors)
            records.append({
                "frame": index,
                "detected_hands": len(predicted),
                "matched_hands": len(pairs),
                "missed_reference_hands": missed,
                "mean_l2_px": float(np.mean(errors)) if errors else None,
            })
            if args.overlay and index in {0, 20, 40, 60, 80}:
                image = Image.fromarray(frame)
                draw = ImageDraw.Draw(image)
                for hand in target:
                    for x, y in hand:
                        if np.isfinite(x) and np.isfinite(y):
                            draw.line((x - 4, y, x + 4, y), fill=(0, 255, 0), width=1)
                            draw.line((x, y - 4, x, y + 4), fill=(0, 255, 0), width=1)
                for hand in predicted:
                    for x, y in hand:
                        draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=(255, 50, 50), width=1)
                draw.rectangle((0, 0, 310, 20), fill=(0, 0, 0))
                draw.text((4, 4), f"frame {index}: matched {len(pairs)}/{len(target)}", fill=(255, 255, 255))
                overlays.append(image)

    matched = sum(row["matched_hands"] for row in records)
    expected = sum(len(hand) for hand in labels)
    errors = np.array(all_errors)
    summary = {
        "video": str(args.video),
        "labels": str(args.labels),
        "detector": f"MediaPipe {mp.__version__} HandLandmarker, video mode, all thresholds 0.3",
        "metric": "mean Euclidean distance over visible joints of spatially matched hands; not the paper's HandL2",
        "frames": len(records),
        "detected_hands": sum(row["detected_hands"] for row in records),
        "reference_hands": int(expected),
        "matched_reference_hand_fraction": matched / expected,
        "matched_keypoint_l2_px": {
            "mean": float(errors.mean()) if len(errors) else None,
            "median": float(np.median(errors)) if len(errors) else None,
            "p90": float(np.quantile(errors, 0.9)) if len(errors) else None,
            "n": len(errors),
        },
        "interpretation": "Distance to the supplied projected annotations. These labels are not a valid future for counterfactual camera or skeleton inputs.",
        "per_frame": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"per_frame", "video", "labels", "interpretation"}}, indent=2))

    if args.overlay:
        columns = len(overlays)
        sheet = Image.new("RGB", (640 * columns, 480))
        for index, image in enumerate(overlays):
            sheet.paste(image, (640 * index, 0))
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.overlay)


if __name__ == "__main__":
    main()
