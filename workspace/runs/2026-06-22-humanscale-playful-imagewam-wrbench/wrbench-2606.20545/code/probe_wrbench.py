#!/usr/bin/env python3
"""Minimal diagnostic probe for the WRBench Natural-25 suite.

This script does not need GPUs or model weights. It:
  1. Loads the Natural-25 scene/event grid and prompt variants.
  2. Verifies the 25-family x 4-event-tier factorial design.
  3. Prints sample event-view records for one family across all four tiers.
  4. Demonstrates the WRBench denominator logic using a row from Table 2.
  5. Emits the actual VLM probe prompts used for D3 (visible spatial) and
     D5 (re-observation gate) so the scoring mechanics are concrete.

Run from anywhere:
    python runs/2026-06-22-humanscale-playful-imagewam-wrbench/wrbench-2606.20545/code/probe_wrbench.py
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# The cloned WRBench source lives next to this script.
SCRIPT_DIR = Path(__file__).resolve().parent
WRBENCH_REPO = SCRIPT_DIR / "WRBench-repo"
WRBENCH_SRC = WRBENCH_REPO / "src"
if str(WRBENCH_SRC) not in sys.path:
    sys.path.insert(0, str(WRBENCH_SRC))

from wrbench.eval.scoring.prompts_v2_probe import (
    DEFAULT_PROMPT_MODE,
    active_probe_catalog,
    build_runtime_v2_probe_prompt,
    probe_by_id,
)

DATA_DIR = WRBENCH_SRC / "wrbench" / "data" / "natural25"
SCENE_CSV = DATA_DIR / "scene_events_25x4.csv"
VARIANTS_JSONL = DATA_DIR / "variants.jsonl"
FAMILIES_JSONL = DATA_DIR / "families.jsonl"
CAMERA_SCOPE_JSON = DATA_DIR / "camera_scopes" / "t2v_rotation_stress_30_60.json"

SAMPLE_FAMILY = "bedroom_cat_bed_jump"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def event_delta_from_reasoning(tier: str, div: str | None) -> str:
    if tier == "T0":
        return "none"
    if tier == "T1":
        return "spatial"
    if tier == "T2" and div == "div_a":
        return "state_only"
    if tier == "T2" and div == "div_b":
        return "full"
    return "unknown"


def verify_factorial_design(variants: list[dict[str, Any]]) -> None:
    families = sorted({v["family_id"] for v in variants})
    assert len(families) == 25, f"expected 25 families, got {len(families)}"

    event_counts = Counter(v["event_delta"] for v in variants)
    assert set(event_counts) == {"none", "spatial", "state_only", "full"}
    assert all(c == 100 for c in event_counts.values()), event_counts

    oov_counts = Counter(v["oov_gap"] for v in variants)
    assert set(oov_counts) == {"none", "static", "yaw_LR", "yaw_RL"}
    assert all(c == 100 for c in oov_counts.values()), oov_counts

    cross = Counter((v["event_delta"], v["oov_gap"]) for v in variants)
    expected = {(e, g): 25 for e in ("none", "spatial", "state_only", "full") for g in ("none", "static", "yaw_LR", "yaw_RL")}
    assert cross == expected, f"cross tab mismatch: {cross}"

    print(f"Natural-25 factorial design checks passed:")
    print(f"  families={len(families)}  variants={len(variants)}")
    print(f"  event tiers={dict(event_counts)}")
    print(f"  camera gaps={dict(oov_counts)}")
    print(f"  cross count per (event, camera) cell=25")


def print_family_samples(variants: list[dict[str, Any]], family_id: str) -> None:
    sample = [v for v in variants if v["family_id"] == family_id and v["oov_gap"] == "yaw_LR"]
    sample = sorted(sample, key=lambda v: (v["reasoning_tier"], v.get("divergence_id") or ""))
    print(f"\n--- Sample event-view records for '{family_id}' (yaw_LR camera) ---")
    for v in sample:
        tier = v["reasoning_tier"] + (f"_{v['divergence_id']}" if v.get("divergence_id") else "")
        print(
            f"variant_id={v['variant_id']}\n"
            f"  tier={tier:12} event_delta={v['event_delta']:10} oov_gap={v['oov_gap']}\n"
            f"  world_state_prompt={v['world_state_prompt']}\n"
            f"  expected_state    ={v['expected_state']}"
        )


def demonstrate_denominator(model: str, support_rate: float, cond_score: float, total: int = 400) -> None:
    """Reconstruct the conditional D6 denominator from reported Table 2 numbers."""
    judgeable = round(support_rate * total)
    summed_scores = cond_score * judgeable
    print(f"\n--- Denominator mechanics for {model} ---")
    print(f"  reported re-observation support = {support_rate:.1%}")
    print(f"  reported conditional re-observed-state score = {cond_score:.3f}")
    print(f"  assumed total generated outputs = {total}")
    print(f"  judgeable rows (D5/D6 denominator) = {support_rate:.1%} x {total} ≈ {judgeable}")
    print(f"  sum of judgeable D6 scores = {cond_score:.3f} x {judgeable} ≈ {summed_scores:.1f}")
    print(f"  conditional mean = {summed_scores:.1f} / {judgeable} = {cond_score:.3f}")

    # Simulate per-row support flags and D6 scores that reproduce the same aggregate.
    rng = random.Random(26062045)
    supports = [1 if i < judgeable else 0 for i in range(total)]
    rng.shuffle(supports)
    # Beta(6, 3) has mean ~0.667; shift/scale to target mean and clip to [0,1].
    scores = []
    for _ in range(judgeable):
        x = rng.betavariate(6.0, 3.0)
        s = 0.55 + 0.30 * x
        scores.append(min(1.0, max(0.0, s)))
    # Adjust to hit the exact target mean.
    if scores:
        offset = cond_score - (sum(scores) / len(scores))
        scores = [min(1.0, max(0.0, s + offset)) for s in scores]
    rows = []
    score_iter = iter(scores)
    for sup in supports:
        rows.append({"support": sup, "d6_score": next(score_iter) if sup else None})
    observed_support = sum(r["support"] for r in rows) / total
    observed_cond = sum(r["d6_score"] for r in rows if r["d6_score"] is not None) / judgeable
    print(f"  simulated support rate = {observed_support:.1%}")
    print(f"  simulated conditional D6 mean = {observed_cond:.3f}")
    print("  => D6 is averaged ONLY over rows that pass the re-observation gate.")


def print_probe_prompts(world_state_prompt: str) -> None:
    print("\n--- Example VLM probe prompts (no model called) ---")
    probes = active_probe_catalog(DEFAULT_PROMPT_MODE)
    probe_map = {p.probe_id: p for p in probes}

    d3_probe = probe_map["D3_POSITION_RELATION"]
    prompt = build_runtime_v2_probe_prompt(
        world_state_prompt=world_state_prompt,
        video_id="demo_bedroom_cat_bed_jump_yawLR",
        probe=d3_probe,
        task_context={"family_id": SAMPLE_FAMILY, "camera_type": "yaw_LR", "event_delta": "full"},
        fps="2",
        frames_used=16,
    )
    print(f"\n[D3 visible-spatial probe]\n{prompt}\n")

    d5_gate = probe_map["D5_GATE_RETURN"]
    prompt = build_runtime_v2_probe_prompt(
        world_state_prompt=world_state_prompt,
        video_id="demo_bedroom_cat_bed_jump_yawLR",
        probe=d5_gate,
        task_context={"family_id": SAMPLE_FAMILY, "camera_type": "yaw_LR", "event_delta": "full"},
        fps="2",
        frames_used=16,
    )
    print(f"\n[D5 re-observation gate probe]\n{prompt}\n")


def main() -> int:
    if not WRBENCH_REPO.is_dir():
        print(f"ERROR: WRBench repo not found at {WRBENCH_REPO}", file=sys.stderr)
        print("Clone or unzip the WRBench repository next to this script.", file=sys.stderr)
        return 1

    variants = load_jsonl(VARIANTS_JSONL)
    families = load_jsonl(FAMILIES_JSONL)
    scenes = load_csv(SCENE_CSV)

    verify_factorial_design(variants)
    print_family_samples(variants, SAMPLE_FAMILY)

    # Sanity-check scene CSV against variant records.
    csv_by_family = {row["family_id"]: row for row in scenes}
    variant_by_family_tier = {(v["family_id"], v["variant_id"]): v for v in variants}
    mismatches = 0
    for row in scenes:
        family_id = row["family_id"]
        for tier, delta in [
            ("T0", "none"),
            ("T1", "spatial"),
            ("T2_div_a", "state_only"),
            ("T2_div_b", "full"),
        ]:
            vid = f"{family_id}__{tier}__none"  # baseline camera for design check
            v = variant_by_family_tier.get((family_id, vid))
            if v and v["event_delta"] != delta:
                mismatches += 1
    print(f"\nScene-CSV / variant event_delta consistency mismatches: {mismatches}")

    # Table 2 row: Gen3C (geometry-cache condition).
    demonstrate_denominator(
        model="Gen3C",
        support_rate=0.730,
        cond_score=0.640,
        total=400,
    )

    # Camera scope used for prompt-only T2V runs.
    if CAMERA_SCOPE_JSON.exists():
        scope = json.loads(CAMERA_SCOPE_JSON.read_text())
        print(f"\n--- T2V rotation-stress camera scope ---")
        print(f"  scope_id={scope['scope_id']}  expected_tasks={scope['expected_task_count']}")
        for cam in scope["cameras"]:
            print(f"    {cam['label']:10} camera_type={cam['camera_type']} preset={cam['preset']} stress_yaw_deg={cam['stress_yaw_deg']}")

    # A concrete prompt from the full-event variant.
    full_variants = [v for v in variants if v["family_id"] == SAMPLE_FAMILY and v["event_delta"] == "full"]
    if full_variants:
        print_probe_prompts(full_variants[0]["world_state_prompt"])

    print("\nProbe complete. No GPUs or VLM weights were used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
