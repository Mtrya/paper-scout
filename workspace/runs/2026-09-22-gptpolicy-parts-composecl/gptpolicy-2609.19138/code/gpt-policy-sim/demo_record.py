"""Record scripted-expert episodes in the sim and package them as GPT-Policy demos.

Produces a reviewed-bundle ``demo.json`` (schema_version 1) with keyframe images,
then runs the official ``gpt_policy.input.demonstration.prepare_demonstration``
to compile it into a portable prepared-input directory (input.json + keyframes)
that ``run_episode.py`` loads straight into ``RunInput.content``.

Modes match the paper's conditions:
- ``video``:        images + text annotations only (state/action stripped by the
                    official preparer) — the "Human Video" analogue.
- ``video+action``: keyframes additionally carry measured EE/joint/gripper
                    state and inter-keyframe action segments — the
                    "Robot Video+Action" analogue.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from gpt_policy.input.demonstration import prepare_demonstration
from gpt_policy.input.manifest import load_manifest
from gpt_policy.input.request import RunInput

from sim_render import SimRenderer
from sim_world import SimRobot, World
from tasks import TASKS

HERE = Path(__file__).resolve().parent
DEMOS_ROOT = HERE / "demos"
SAMPLE_PERIOD_S = 0.1

COORDINATE_FRAME = (
    "Simulated single-arm base frame: +x forward, +y left, +z up; metres; quaternion xyzw. "
    "EE is a point TCP; measured orientation tracks the commanded one. "
    "Gripper normalized: 0 closed, 1 open. Samples are exact simulated measurements."
)


def _state_sample(world: World, t: float) -> dict:
    joints = world.mock_joints(world.ee)
    return {
        "t_s": round(t, 6),
        "left": {
            "joint_measured_rad": [round(float(v), 6) for v in joints],
            "eef_measured_xyz_xyzw": [round(float(v), 6) for v in (*world.ee, *world.ee_quat_xyzw)],
            "gripper_measured": round(float(world.gripper_measured), 6),
            "gripper_command": round(float(world.gripper_command), 6),
        },
    }


def record_episode(task_key: str, seed: int) -> tuple[dict, Path]:
    """Run the scripted expert; capture keyframes at stage transitions."""
    task = TASKS[task_key]
    world = task.build_world(seed)
    robot = SimRobot(world)
    renderer = SimRenderer(world)
    expert = task.expert(world)

    keyframes: list[dict] = []
    samples: list[dict] = []
    requests: list[dict] = []

    def record_sample(t: float) -> None:
        # compress_samples downstream requires strictly increasing sample times.
        if not samples or t > samples[-1]["t_s"]:
            samples.append(_state_sample(world, t))

    def sampler(w: World, prev: np.ndarray, new: np.ndarray) -> None:
        if not samples or w.time_s - samples[-1]["t_s"] >= SAMPLE_PERIOD_S - 1e-9:
            record_sample(w.time_s)

    world.contact_listeners.append(sampler)

    annotations = {}
    step = 0
    while True:
        decision = expert.act(world)
        stage = expert.stage
        if not keyframes or keyframes[-1]["stage"] != stage:
            t_kf = world.time_s
            if keyframes and t_kf <= keyframes[-1]["t_s"]:
                # A stage that completes without executing advances no sim time.
                t_kf = keyframes[-1]["t_s"] + 1e-4
            keyframes.append({
                "t_s": round(t_kf, 6),
                "stage": stage,
                "observation": decision["arguments"].get("note", stage),
                "_images": {},  # filled below
            })
            record_sample(world.time_s)
            for view in ("top", "front"):
                keyframes[-1]["_images"][view] = renderer.capture(view)
        if decision["name"] in ("done", "give_up"):
            break
        requests.append({
            "at_s": round(world.time_s, 6),
            "step": step,
            "name": decision["name"],
            "arguments": decision["arguments"],
            "reported_result": "execution_result",
        })
        robot.execute(decision["name"], decision["arguments"])
        step += 1
        if step > 80:
            raise RuntimeError("expert did not finish within 80 decisions")
    renderer.close()

    success, detail = task.success(world)
    if not success:
        raise RuntimeError(f"expert episode failed: {detail}")

    # Assemble keyframe records: state at keyframe, action segment to the next.
    frames = []
    for i, keyframe in enumerate(keyframes):
        t = keyframe["t_s"]
        state = _state_sample(world, t)  # pose fields replaced below by recorded sample
        nearest = min(samples, key=lambda s: abs(s["t_s"] - t))
        frame = {
            "t_s": t,
            "stage": keyframe["stage"],
            "observation": keyframe["observation"],
            "images": keyframe["_images"],
            "image_times": {view: t for view in keyframe["_images"]},
            "state": {"left": nearest["left"]},
            "alignment": {
                "method": "sim_synchronous_capture",
                "captured_at_s": t,
                "state_observed_at_s": nearest["t_s"],
                "delta_s": round(nearest["t_s"] - t, 6),
                "max_delta_s": 0.1,
                "exposure_synchronized": True,
            },
        }
        if i + 1 < len(keyframes):
            end = keyframes[i + 1]["t_s"]
            segment = [s for s in samples if t <= s["t_s"] < end]
            gaps = [b["t_s"] - a["t_s"] for a, b in zip(segment, segment[1:])]
            frame["action"] = {
                "kind": "measured_single_arm_segment",
                "time_basis": "sim_elapsed_s",
                "samples": segment,
                "requested_tools": [r for r in requests if t <= r["at_s"] < end],
                "max_sample_gap_s": round(max(gaps, default=0.0), 6),
            }
        frames.append(frame)

    demo = {
        "schema_version": 1,
        "title": task.instruction,
        "source": f"gpt-policy-sim scripted expert, task {task.key} ({task.name}), seed {seed}",
        "demonstrator": "scripted_expert_oracle_sim",
        "coverage": "complete_episode",
        "summary": "Simulated expert rollout; stages: " + " -> ".join(f["stage"] for f in frames),
        "outcome": "success",
        "finalization_state": "completed",
        "coordinate_frame": COORDINATE_FRAME,
        "source_robot": {"model": "SIM-X5", "machine": "gpt-policy-sim",
                         "interfaces": ["sim0", None]},
        "keyframes": frames,
    }
    return demo, HERE


def write_demo_source(demo: dict, directory: Path) -> Path:
    """Write demo.json plus keyframe JPEGs (paths relative to demo.json)."""
    if directory.exists():
        shutil.rmtree(directory)
    (directory / "keyframes").mkdir(parents=True)
    frames = []
    for i, frame in enumerate(demo["keyframes"]):
        images = {}
        for j, (view, image) in enumerate(frame["images"].items()):
            name = f"keyframes/{i:04d}-{j:02d}.jpg"
            (directory / name).write_bytes(image.data)
            images[view] = name
        frames.append({**frame, "images": images})
    payload = {**demo, "keyframes": frames}
    (directory / "demo.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return directory / "demo.json"


def build_and_prepare(task_key: str, seed: int, mode: str, root: Path = DEMOS_ROOT) -> Path:
    """Record (if needed) and officially prepare a demo bundle; return its dir."""
    task = TASKS[task_key]
    prepared_dir = root / "prepared" / f"{task.key.lower()}-{mode}-s{seed}"
    if prepared_dir.is_dir():
        return prepared_dir
    demo, _ = record_episode(task_key, seed)
    source_dir = root / "src" / f"{task.key.lower()}-s{seed}"
    source = write_demo_source(demo, source_dir)
    run_input = RunInput(task.instruction, "gpt-6-astra")
    prepared, metadata = prepare_demonstration(run_input, source, mode, prepared_dir)
    manifest = load_manifest(prepared_dir / "input.json")
    print(f"[demo] {task.key} mode={mode} seed={seed}: {metadata['keyframes']} keyframes, "
          f"{metadata['unique_images']} images, {len(manifest.content)} content parts -> {prepared_dir}")
    return prepared_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Record scripted-expert demos from the sim backend")
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--mode", choices=("video", "video+action"), default="video+action")
    parser.add_argument("--out", type=Path, default=DEMOS_ROOT)
    args = parser.parse_args()
    prepared_dir = build_and_prepare(args.task, args.seed, args.mode, args.out)
    manifest = load_manifest(prepared_dir / "input.json")
    print(f"OK: {prepared_dir} ({len(manifest.content)} content parts)")


if __name__ == "__main__":
    main()
