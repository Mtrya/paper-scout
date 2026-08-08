"""Build the full experiment input bundle (deterministic, pure local).

Outputs under data/:
  tasks.jsonl   one line per model query: {task_id, probe, seed, regime, mode,
                condition, batch_idx, memory_ids, system, prompt, image}
  gold.json     task_id -> gold stale labels ({memory_id: bool} for audits,
                bool for probe3 single-entry tasks)
  images/       rendered PNGs referenced by vision tasks
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import testbed
from prompts import (SYSTEM_DETECTION, detection_user_prompt, probe3_prompt,
                     F1_WORDING, F2_AGE, F3_PRIOR, F4_OBS)

N_SEEDS = 30
MASTER_SEED = 2024
BATCH = 10
PROBE3_REPLICAS = 20

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
IMGDIR = os.path.join(DATA, "images")


def batches(entries: list[dict]) -> list[list[dict]]:
    return [entries[i:i + BATCH] for i in range(0, len(entries), BATCH)]


def main() -> None:
    os.makedirs(IMGDIR, exist_ok=True)
    seeds = testbed.instance_seeds(N_SEEDS, MASTER_SEED)
    tasks: list[dict] = []
    gold: dict[str, object] = {}

    # ---- shared blank image
    testbed.blank_image().save(os.path.join(IMGDIR, "blank.png"))

    # ---- per-seed instances, images, probe1/probe2 tasks
    instances: dict[tuple[int, str], testbed.Instance] = {}
    for seed in seeds:
        for regime in ("L1", "L2"):
            inst = testbed.make_instance(seed, regime)
            instances[(seed, regime)] = inst
            img = testbed.render_grid(inst.grid)
            img.save(os.path.join(IMGDIR, f"grid_{seed}_{regime}.png"))
            entries = testbed.memory_entries(inst.grid0)
            labels = testbed.gold_labels(inst)
            obs = testbed.text_observation(inst.grid)

            for b_idx, b_entries in enumerate(batches(entries)):
                base = dict(seed=seed, regime=regime, batch_idx=b_idx,
                            memory_ids=[e["memory_id"] for e in b_entries],
                            system=SYSTEM_DETECTION)
                tid = f"p1_text_{regime}_{seed}_b{b_idx}"
                tasks.append(dict(task_id=tid, probe="p1", mode="text", condition="correct",
                                  prompt=detection_user_prompt(b_entries, "text", obs_text=obs),
                                  image=None, **base))
                gold[tid] = {m: labels[m] for m in base["memory_ids"]}

                tid = f"p1_vision_{regime}_{seed}_b{b_idx}"
                tasks.append(dict(task_id=tid, probe="p1", mode="vision", condition="correct",
                                  prompt=detection_user_prompt(b_entries, "vision"),
                                  image=f"images/grid_{seed}_{regime}.png", **base))
                gold[tid] = {m: labels[m] for m in base["memory_ids"]}

                # probe 2: same prompts, ablated images
                tid = f"p2_blank_{regime}_{seed}_b{b_idx}"
                tasks.append(dict(task_id=tid, probe="p2", mode="vision", condition="blank",
                                  prompt=detection_user_prompt(b_entries, "vision"),
                                  image="images/blank.png", **base))
                gold[tid] = {m: labels[m] for m in base["memory_ids"]}

                mism_seed = seeds[(seeds.index(seed) + 13) % len(seeds)]
                tid = f"p2_mismatch_{regime}_{seed}_b{b_idx}"
                tasks.append(dict(task_id=tid, probe="p2", mode="vision", condition="mismatch",
                                  prompt=detection_user_prompt(b_entries, "vision"),
                                  image=f"images/grid_{mism_seed}_{regime}.png",
                                  mismatch_seed=mism_seed, **base))
                gold[tid] = {m: labels[m] for m in base["memory_ids"]}

    # ---- probe 3: arbitration factors (text mode, single-entry audits)
    rng = np.random.default_rng(777)
    p3_specs = [("F1", lv) for lv in F1_WORDING] + \
               [("F2", lv) for lv in F2_AGE] + \
               [("F3", lv) for lv in F3_PRIOR] + \
               [("F4", lv) for lv in F4_OBS]
    for factor, level in p3_specs:
        for i in range(PROBE3_REPLICAS):
            for kind in ("conflict", "control"):
                g_seed = int(rng.integers(10_000, 9_000_000))
                grid = testbed.generate_grid(g_seed)
                # pick a mid-grid nonterminal conflict cell; keep world solvable
                while True:
                    r = int(rng.integers(2, 6))
                    c = int(rng.integers(2, 6))
                    if (r, c) in (testbed.START, testbed.GOAL):
                        continue
                    old = grid[r][c]
                    grid[r][c] = "H" if kind == "conflict" else "F"
                    if testbed._solvable(grid):
                        break
                    grid[r][c] = old

                mid = testbed.mem_id(r, c)
                wording = F1_WORDING[level] if factor == "F1" else F1_WORDING["neutral"]
                age = F2_AGE[level] if factor == "F2" else ""
                prior = F3_PRIOR[level] if factor == "F3" else ""
                header = F4_OBS[level] if factor == "F4" else F4_OBS["plain"]

                entry_text = f"[{mid}] SAFE at ({r},{c}): {wording}{age}"
                obs = testbed.text_observation(grid)
                tid = f"p3_{factor}_{level}_{'c' if kind == 'conflict' else 'v'}{i:02d}"
                tasks.append(dict(
                    task_id=tid, probe="p3", mode="text", condition=kind,
                    factor=factor, level=level, seed=g_seed, regime=None, batch_idx=None,
                    memory_ids=[mid], system=SYSTEM_DETECTION,
                    prompt=probe3_prompt(entry_text, obs, obs_header=header, prior_note=prior),
                    image=None,
                ))
                gold[tid] = (kind == "conflict")  # conflict => truly stale

    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "tasks.jsonl"), "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(os.path.join(DATA, "gold.json"), "w") as f:
        json.dump(gold, f)

    # ---- sanity report
    stale_ratios = {"L1": [], "L2": []}
    for (seed, regime), inst in instances.items():
        labels = testbed.gold_labels(inst)
        stale_ratios[regime].append(sum(labels.values()) / 64)
    for regime, xs in stale_ratios.items():
        print(f"{regime}: stale ratio mean {np.mean(xs):.3f} +- {np.std(xs):.3f} "
              f"(paper: L1 0.094+-0.011, L2 0.141+-0.014)")
    n_p1 = sum(1 for t in tasks if t["probe"] == "p1")
    n_p2 = sum(1 for t in tasks if t["probe"] == "p2")
    n_p3 = sum(1 for t in tasks if t["probe"] == "p3")
    print(f"tasks: p1={n_p1} p2={n_p2} p3={n_p3} total={len(tasks)} (per model)")
    print(f"seeds: {seeds[:5]} ...")


if __name__ == "__main__":
    main()
