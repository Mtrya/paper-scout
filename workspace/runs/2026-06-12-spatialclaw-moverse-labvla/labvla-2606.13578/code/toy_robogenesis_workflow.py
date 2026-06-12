"""Toy reconstruction of RoboGenesis-style workflow composition.

The real RoboGenesis engine (not released with the LabVLA code) builds laboratory
scenes from generated USD assets, composes long-horizon protocols from atomic
skills, randomizes along six axes, and exports only successful rollouts with
structured annotations. This script captures the *workflow composition* core:
- a protocol is an ordered list of atomic skills,
- each skill carries a success checker,
- a scene provides objects with poses and affordances,
- domain randomization perturbs the scene while preserving protocol semantics.

Run: python toy_robogenesis_workflow.py
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Object:
    name: str
    category: str
    position: tuple[float, float, float]
    affordances: list[str] = field(default_factory=list)
    state: dict = field(default_factory=dict)


@dataclass
class Scene:
    objects: dict[str, Object]
    robot: str = "franka"
    camera_pose: tuple[float, float, float] = (1.0, 0.0, 1.5)
    lighting: str = "neutral"


@dataclass
class SkillStep:
    skill: str
    target: str
    args: dict = field(default_factory=dict)


@dataclass
class Workflow:
    instruction: str
    steps: list[SkillStep]


# -----------------------------------------------------------------------------
# Atomic skill success checkers (simplified physics-aware checks)
# -----------------------------------------------------------------------------
def check_pick(scene: Scene, target: str, state: dict) -> tuple[bool, str]:
    obj = scene.objects.get(target)
    if obj is None:
        return False, f"{target} missing"
    if "graspable" not in obj.affordances:
        return False, f"{target} not graspable"
    return state.get("gripper_has", None) == target, f"gripper_has={state.get('gripper_has')}"


def check_place(scene: Scene, target: str, state: dict) -> tuple[bool, str]:
    # target is the destination object (e.g. hot_plate); placed object is the
    # one released by the most recent place step.
    placed_obj = state.get("last_placed_object", None)
    if placed_obj is None:
        return False, "no placement recorded"
    dst = scene.objects.get(target)
    if dst is None:
        return False, f"{target} missing"
    placed_pos = state.get(f"{placed_obj}_placed_pos", None)
    if placed_pos is None:
        return False, "no placement position"
    tx, ty, tz = dst.position
    px, py, pz = placed_pos
    ok = (abs(px - tx) < 0.05 and abs(py - ty) < 0.05 and abs(pz - tz) < 0.05)
    return ok, f"offset=({px-tx:.3f},{py-ty:.3f},{pz-tz:.3f})"


def check_pour(scene: Scene, target: str, state: dict) -> tuple[bool, str]:
    src, dst = target.split("->")
    transferred = state.get("liquid_transferred", 0.0)
    return transferred > 0.5, f"transferred={transferred:.2f}"


def check_press(scene: Scene, target: str, state: dict) -> tuple[bool, str]:
    return state.get(f"{target}_pressed", False), f"pressed={state.get(f'{target}_pressed', False)}"


SKILL_REGISTRY: dict[str, Callable[[Scene, str, dict], tuple[bool, str]]] = {
    "pick": check_pick,
    "place": check_place,
    "pour": check_pour,
    "press": check_press,
}


# -----------------------------------------------------------------------------
# Domain randomization axes (six axes from the paper, plus instruction paraphrase)
# -----------------------------------------------------------------------------
def randomize_scene_layout(scene: Scene, rng: random.Random) -> Scene:
    """Perturb object poses within a small validated range."""
    s = copy.deepcopy(scene)
    for obj in s.objects.values():
        x, y, z = obj.position
        obj.position = (
            x + rng.uniform(-0.04, 0.04),
            y + rng.uniform(-0.04, 0.04),
            z,
        )
    return s


def randomize_visual_clutter(scene: Scene, rng: random.Random) -> Scene:
    s = copy.deepcopy(scene)
    clutter = ["notepad", "pen", "gloves", "funnel"]
    for name in rng.sample(clutter, k=rng.randint(0, 2)):
        s.objects[name] = Object(
            name=name,
            category="clutter",
            position=(rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 0.0),
            affordances=[],
        )
    return s


def randomize_camera(scene: Scene, rng: random.Random) -> Scene:
    s = copy.deepcopy(scene)
    cx, cy, cz = s.camera_pose
    s.camera_pose = (
        cx + rng.uniform(-0.05, 0.05),
        cy + rng.uniform(-0.05, 0.05),
        cz + rng.uniform(-0.02, 0.02),
    )
    return s


def randomize_object_swap(scene: Scene, rng: random.Random) -> Scene:
    """Swap compatible source/target assets while keeping semantic roles."""
    s = copy.deepcopy(scene)
    beakers = [n for n, o in s.objects.items() if o.category == "beaker"]
    if len(beakers) >= 2:
        a, b = rng.sample(beakers, k=2)
        s.objects[a].position, s.objects[b].position = s.objects[b].position, s.objects[a].position
    return s


def randomize_lighting(scene: Scene, rng: random.Random) -> Scene:
    s = copy.deepcopy(scene)
    s.lighting = rng.choice(["neutral", "warm", "cool", "dim"])
    return s


RANDOMIZATION_AXES = {
    "scene": randomize_scene_layout,
    "clutter": randomize_visual_clutter,
    "camera": randomize_camera,
    "object": randomize_object_swap,
    "lighting": randomize_lighting,
}


# -----------------------------------------------------------------------------
# Workflow executor + success-filtered export
# -----------------------------------------------------------------------------
def execute_workflow(scene: Scene, workflow: Workflow, rng: random.Random) -> dict:
    """Execute an ordered workflow and annotate each step's success.

    In the real engine this is Isaac Sim + motion planning; here we use a
    deterministic cheat state to demonstrate the success-checker/annotation
    pipeline.
    """
    state: dict = {"gripper_has": None}
    step_results = []
    overall_success = True

    for step in workflow.steps:
        checker = SKILL_REGISTRY.get(step.skill)
        if checker is None:
            step_results.append({"skill": step.skill, "success": False, "reason": "unknown skill"})
            overall_success = False
            continue

        # Simulate physical outcome deterministically for the demo.
        if step.skill == "pick":
            state["gripper_has"] = step.target
        elif step.skill == "place":
            held = state.pop("gripper_has", None)
            if held:
                # step.target is the destination object (e.g. hot_plate)
                state[f"{held}_placed_pos"] = scene.objects[step.target].position
                state["last_placed_object"] = held
        elif step.skill == "pour":
            state["liquid_transferred"] = 0.8
        elif step.skill == "press":
            state[f"{step.target}_pressed"] = True

        ok, metric = checker(scene, step.target, state)
        step_results.append({
            "skill": step.skill,
            "target": step.target,
            "success": ok,
            "metric": metric,
        })
        overall_success = overall_success and ok

    return {
        "instruction": workflow.instruction,
        "robot": scene.robot,
        "camera_pose": scene.camera_pose,
        "lighting": scene.lighting,
        "steps": step_results,
        "overall_success": overall_success,
    }


def export_successful_rollouts(
    workflow: Workflow,
    base_scene: Scene,
    axes: list[str] | None = None,
    n: int = 8,
    seed: int = 42,
) -> list[dict]:
    """Randomize, execute, and keep only successful episodes."""
    rng = random.Random(seed)
    axes = axes or list(RANDOMIZATION_AXES.keys())
    exported = []
    for episode_id in range(n):
        scene = copy.deepcopy(base_scene)
        # Apply a random subset of axes.
        active = rng.sample(axes, k=rng.randint(1, len(axes)))
        for ax in active:
            scene = RANDOMIZATION_AXES[ax](scene, rng)
        result = execute_workflow(scene, workflow, rng)
        if result["overall_success"]:
            result["episode_id"] = episode_id
            result["active_axes"] = active
            exported.append(result)
    return exported


# -----------------------------------------------------------------------------
# Demo protocol: "transfer liquid between beakers and heat it"
# -----------------------------------------------------------------------------
def make_heat_protocol() -> tuple[Scene, Workflow]:
    scene = Scene(
        objects={
            "beakerA": Object("beakerA", "beaker", (0.15, 0.0, 0.0), ["graspable", "pourable"], {"filled": 0.5}),
            "beakerB": Object("beakerB", "beaker", (0.0, 0.15, 0.0), ["graspable", "pourable"]),
            "hot_plate": Object("hot_plate", "instrument", (0.25, 0.0, 0.0), ["heatable"]),
            "power_button": Object("power_button", "button", (0.27, 0.0, 0.05), ["pressable"]),
        }
    )
    workflow = Workflow(
        instruction="Transfer liquid from beaker A to beaker B, place beaker B on the hot plate, then press the power button.",
        steps=[
            SkillStep("pick", "beakerA"),
            SkillStep("pour", "beakerA->beakerB"),
            SkillStep("place", "hot_plate"),
            SkillStep("press", "power_button"),
        ],
    )
    return scene, workflow


if __name__ == "__main__":
    scene, workflow = make_heat_protocol()
    print("Base workflow:")
    for s in workflow.steps:
        print(f"  {s.skill}({s.target})")
    print()

    episodes = export_successful_rollouts(workflow, scene, n=12, seed=7)
    print(f"Kept {len(episodes)}/12 successful episodes.")
    for ep in episodes[:3]:
        print(f"\nEpisode {ep['episode_id']}: axes={ep['active_axes']}")
        print(f"  camera={ep['camera_pose']}, lighting={ep['lighting']}")
        for st in ep["steps"]:
            print(f"  {st['skill']:5} -> success={st['success']} ({st.get('metric', '')})")
