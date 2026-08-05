#!/usr/bin/env python3
"""Causal intervention on DEFT's free-text HIGH_LEVEL_DECISION.

For six public AD-MCQ cases (one per oracle letter), generate Turn 1 once and
then replay Turn 2 under the original, withheld, and counterfactual decision.
Only the HIGH_LEVEL_DECISION line changes; the scene, reasoning, options,
weights, and deterministic decoding remain fixed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import requests
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


SELECTED = (0, 13, 45, 51, 75, 87)
CHOICE_RE = re.compile(r"FINAL_CHOICE:\s*([A-Fa-f])")
HLD_RE = re.compile(r"(?im)^\s*HIGH_LEVEL_DECISION\s*:.*$")
OPTION_RE = re.compile(r"Option ([A-F])\n\s+waypoints xy: \[(.*?)\]", re.S)
POINT_RE = re.compile(r"\((-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)")


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(destination)


def parse_options(prompt: str) -> dict[str, list[tuple[float, float]]]:
    parsed: dict[str, list[tuple[float, float]]] = {}
    for letter, body in OPTION_RE.findall(prompt):
        parsed[letter] = [(float(x), float(y)) for x, y in POINT_RE.findall(body)]
    if len(parsed) != 6:
        raise ValueError(f"expected six options, got {sorted(parsed)}")
    return parsed


def farthest_option(options: dict[str, list[tuple[float, float]]], oracle: str) -> str:
    reference = options[oracle]
    def ade(points: list[tuple[float, float]]) -> float:
        return sum(math.hypot(x - rx, y - ry) for (x, y), (rx, ry) in zip(points, reference)) / len(reference)
    return max((letter for letter in options if letter != oracle), key=lambda letter: ade(options[letter]))


def trajectory_hld(points: list[tuple[float, float]], current_speed: float) -> str:
    final_x, final_y = points[-1]
    direction = "turn left" if final_y > 4.0 else "turn right" if final_y < -4.0 else "go straight"
    last_step = math.hypot(points[-1][0] - points[-2][0], points[-1][1] - points[-2][1]) / 0.5
    mean_speed = math.hypot(final_x, final_y) / 5.0
    if last_step < 0.8:
        speed = "decelerate to a stop"
    elif mean_speed > max(current_speed * 1.35, current_speed + 2.0):
        speed = "accelerate hard"
    elif mean_speed < current_speed * 0.65:
        speed = "decelerate sharply"
    else:
        speed = f"hold approximately {mean_speed:.1f} m/s"
    return f"HIGH_LEVEL_DECISION: {speed} and {direction}; treat this as the binding plan."


def multimodal_user(text: str, video_paths: list[Path]) -> dict[str, Any]:
    pieces = text.split("<video>")
    if len(pieces) != len(video_paths) + 1:
        raise ValueError(f"video placeholder mismatch: {len(pieces)-1} vs {len(video_paths)}")
    content: list[dict[str, Any]] = []
    for index, video in enumerate(video_paths):
        if pieces[index]:
            content.append({"type": "text", "text": pieces[index]})
        content.append({
            "type": "video",
            "video": f"file://{video}",
            "fps": 2.0,
            "min_pixels": 50_176,
            "max_pixels": 50_176,
        })
    if pieces[-1]:
        content.append({"type": "text", "text": pieces[-1]})
    return {"role": "user", "content": content}


def generate(model: Any, processor: Any, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    if isinstance(video_kwargs.get("fps"), list):
        video_kwargs["fps"] = video_kwargs["fps"][0] if video_kwargs["fps"] else None
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_vision_id=True,
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    trimmed = [sequence[len(source):] for source, sequence in zip(inputs.input_ids, output)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def replace_hld(text: str, replacement: str) -> str:
    if HLD_RE.search(text):
        return HLD_RE.sub(replacement, text, count=1)
    return text.rstrip() + "\n" + replacement


def remove_hld(text: str) -> str:
    return HLD_RE.sub("", text, count=1).strip()


def extract_choice(text: str) -> str | None:
    matches = CHOICE_RE.findall(text)
    return matches[-1].upper() if matches else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-json", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.rows_json.read_text())
    rows = {item["row_idx"]: item["row"] for item in payload["rows"]}
    chosen = [rows[index] for index in SELECTED]

    media_root = args.work_dir / "media"
    for row in chosen:
        for item in row["videos"]:
            relative = item["video"]
            if not isinstance(relative, str):
                raise ValueError(f"selected case lacks public media: {row['extra_info']['id']}")
            download(
                "https://hf-mirror.com/datasets/hzxllll/AD-MCQ/resolve/main/" + relative,
                media_root / relative,
            )

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    ).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[int] = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                completed.add(json.loads(line)["row_idx"])

    for row_idx in SELECTED:
        if row_idx in completed:
            continue
        row = rows[row_idx]
        videos = [media_root / item["video"] for item in row["videos"]]
        source_messages = [
            {"role": "system", "content": row["prompt"][0]["content"]},
            multimodal_user(row["prompt"][1]["content"], videos),
        ]
        turn1 = generate(model, processor, source_messages, max_new_tokens=1200)
        options_prompt = row["extra_info"]["options_prompt"]
        options = parse_options(options_prompt)
        oracle = row["extra_info"]["oracle_letter"]
        counterfactual_target = farthest_option(options, oracle)
        counterfactual_hld = trajectory_hld(
            options[counterfactual_target],
            float(row["extra_info"]["velocity_norm"]),
        )
        variants = {
            "original": turn1,
            "withheld": remove_hld(turn1),
            "counterfactual": replace_hld(turn1, counterfactual_hld),
        }
        turn2: dict[str, Any] = {}
        for condition, assistant_text in variants.items():
            messages = source_messages + [
                {"role": "assistant", "content": assistant_text},
                {"role": "user", "content": options_prompt},
            ]
            response = generate(model, processor, messages, max_new_tokens=1200)
            turn2[condition] = {"choice": extract_choice(response), "response": response}
        record = {
            "row_idx": row_idx,
            "id": row["extra_info"]["id"],
            "oracle": oracle,
            "velocity_norm": row["extra_info"]["velocity_norm"],
            "cv_gap": row["extra_info"]["cv_gap"],
            "turn1": turn1,
            "original_hld": HLD_RE.search(turn1).group(0).strip() if HLD_RE.search(turn1) else None,
            "counterfactual_target": counterfactual_target,
            "counterfactual_hld": counterfactual_hld,
            "turn2": turn2,
        }
        with args.output.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({
            "row_idx": row_idx,
            "oracle": oracle,
            "counterfactual_target": counterfactual_target,
            "choices": {name: value["choice"] for name, value in turn2.items()},
        }), flush=True)


if __name__ == "__main__":
    main()
