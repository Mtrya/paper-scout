#!/usr/bin/env python3
"""
Guava-style harness probe — a minimal, runnable toy manipulation scenario.

The goal is to make Guava's three design ingredients concrete:
  1. Iterative perception-reasoning-action (ReAct) loops.
  2. Semantic action abstractions (grasp, align, move, release, ...).
  3. Multimodal observations (structured text + a visual ASCII render).

No neural model is required. A small hand-written policy plays the role of the
VLM. It executes the task "place the red cube in the basket" and can recover
from a simulated grasp failure. Ablations show what happens when each ingredient
is removed.

Run:
    python guava_probe.py
"""

from __future__ import annotations

import json
import random
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Toy world state
# -----------------------------------------------------------------------------

@dataclass
class Object:
    name: str
    x: float
    y: float
    z: float
    size: float
    color: str = "gray"
    in_basket: bool = False


@dataclass
class Gripper:
    x: float = 0.5
    y: float = 0.5
    z: float = 0.5
    open: bool = True
    holding: Optional[str] = None

    def pos(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class World:
    objects: Dict[str, Object]
    gripper: Gripper = field(default_factory=Gripper)
    home: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    table_z: float = 0.0
    last_result: str = "initialized"
    step: int = 0

    def __post_init__(self):
        # Basket is a container; its interior bottom is slightly above the table.
        self.basket = self.objects.get("basket")

    def is_above(self, obj_name: str, tol: float = 0.08) -> bool:
        obj = self.objects[obj_name]
        return (abs(self.gripper.x - obj.x) < tol and
                abs(self.gripper.y - obj.y) < tol and
                self.gripper.z > obj.z)

    def inside_basket(self, obj_name: str) -> bool:
        obj = self.objects[obj_name]
        if self.basket is None:
            return False
        b = self.basket
        return (abs(obj.x - b.x) < b.size / 2 and
                abs(obj.y - b.y) < b.size / 2 and
                obj.z <= b.z + 0.02)


# -----------------------------------------------------------------------------
# Semantic tool implementations (the "low-level controllers")
# -----------------------------------------------------------------------------

def tool_get_position(world: World, obj: str) -> str:
    if obj not in world.objects:
        return f"error: unknown object '{obj}'"
    o = world.objects[obj]
    return f"{obj} is at ({o.x:.2f}, {o.y:.2f}, {o.z:.2f})"


def tool_get_position_size(world: World, obj: str) -> str:
    if obj not in world.objects:
        return f"error: unknown object '{obj}'"
    o = world.objects[obj]
    return f"{obj} at ({o.x:.2f}, {o.y:.2f}, {o.z:.2f}) size {o.size:.2f}"


def tool_move(world: World, x: float, y: float, z: float) -> str:
    world.gripper.x = float(x)
    world.gripper.y = float(y)
    world.gripper.z = float(z)
    # If holding an object, drag it along.
    if world.gripper.holding:
        held = world.objects[world.gripper.holding]
        held.x = world.gripper.x
        held.y = world.gripper.y
        held.z = world.gripper.z
    return f"moved gripper to ({x:.2f}, {y:.2f}, {z:.2f})"


CLEARANCE = {"small": 0.03, "medium": 0.08, "large": 0.15}


def tool_align(world: World, obj: str, direction: str, clearance: str) -> str:
    if obj not in world.objects:
        return f"error: unknown object '{obj}'"
    o = world.objects[obj]
    dz = CLEARANCE.get(clearance, 0.08)
    dx = dy = 0.0
    if direction == "top":
        pass
    elif direction == "left":
        dx = -o.size / 2 - dz
    elif direction == "right":
        dx = +o.size / 2 + dz
    elif direction == "front":
        dy = -o.size / 2 - dz
    elif direction == "back":
        dy = +o.size / 2 + dz
    else:
        return f"error: unknown direction '{direction}'"
    target_z = o.z + dz
    return tool_move(world, o.x + dx, o.y + dy, target_z)


def tool_grasp(world: World, obj: str, fail_prob: float = 0.3) -> str:
    if obj not in world.objects:
        return f"error: unknown object '{obj}'"
    if not world.gripper.open:
        return "error: gripper is closed; release() first"
    o = world.objects[obj]
    if not world.is_above(obj):
        world.gripper.open = False
        world.gripper.holding = None
        return f"grasp failed: gripper was not aligned above {obj}"
    # Even if roughly above, the gripper must be close enough in z to make contact.
    vertical = world.gripper.z - o.z
    if vertical > 0.15:
        world.gripper.open = False
        world.gripper.holding = None
        return f"grasp failed: gripper too high above {obj} (z={vertical:.2f})"
    # Simulated stochastic grasp failure (first attempts can slip).
    if random.random() < fail_prob:
        world.gripper.open = False
        world.gripper.holding = None
        return f"grasp failed: gripper closed on air near {obj}"
    world.gripper.open = False
    world.gripper.holding = obj
    return f"grasped {obj}"


def tool_release(world: World) -> str:
    if world.gripper.holding:
        held = world.objects[world.gripper.holding]
        # Drop onto table or into basket interior.
        if world.basket and world.is_above("basket", tol=world.basket.size / 2):
            held.z = world.basket.z + 0.01
            held.in_basket = True
        else:
            held.z = world.table_z
            held.in_basket = False
        world.gripper.holding = None
    world.gripper.open = True
    return "released gripper"


def tool_close_gripper(world: World) -> str:
    world.gripper.open = False
    return "gripper closed"


def tool_home_pose(world: World) -> str:
    return tool_move(world, *world.home)


TOOL_SPECS: Dict[str, Callable] = {
    "get_position": tool_get_position,
    "get_position_size": tool_get_position_size,
    "align": tool_align,
    "move": tool_move,
    "grasp": tool_grasp,
    "release": tool_release,
    "close_gripper": tool_close_gripper,
    "home_pose": tool_home_pose,
}


def execute_tool(world: World, name: str, kwargs: Dict) -> str:
    world.step += 1
    if name not in TOOL_SPECS:
        world.last_result = f"error: unknown tool '{name}'"
        return world.last_result
    try:
        world.last_result = TOOL_SPECS[name](world, **kwargs)
    except Exception as e:
        world.last_result = f"error executing {name}: {e}"
    return world.last_result


# -----------------------------------------------------------------------------
# Multimodal observations
# -----------------------------------------------------------------------------

def text_observation(world: World) -> str:
    """Compact symbolic description of the world."""
    parts = [f"Step {world.step}. Last action result: {world.last_result}",
             f"Gripper: pos=({world.gripper.x:.2f},{world.gripper.y:.2f},{world.gripper.z:.2f}) "
             f"open={world.gripper.open} holding={world.gripper.holding}"]
    for name, obj in world.objects.items():
        parts.append(f"{name}: pos=({obj.x:.2f},{obj.y:.2f},{obj.z:.2f}) size={obj.size:.2f} "
                     f"in_basket={obj.in_basket}")
    return "\n".join(parts)


def render_ascii(world: World, grid: int = 11) -> str:
    """Top-down visual grid (z ignored for the 2D view)."""
    # Each cell is about 1/grid world units.
    scale = grid - 1
    canvas = [["." for _ in range(grid)] for _ in range(grid)]

    def world_to_grid(x, y):
        return int(round(x * scale)), int(round(y * scale))

    # Draw objects first.
    for name, obj in world.objects.items():
        gx, gy = world_to_grid(obj.x, obj.y)
        if 0 <= gx < grid and 0 <= gy < grid:
            # Use the first letter of the color + name to show identity.
            label = (obj.color[0].upper() if obj.color else "?") + name[0].upper()
            canvas[grid - 1 - gy][gx] = label  # flip y so +y is up

    # Draw gripper on top.
    gx, gy = world_to_grid(world.gripper.x, world.gripper.y)
    if 0 <= gx < grid and 0 <= gy < grid:
        canvas[grid - 1 - gy][gx] = "+" if world.gripper.open else "x"

    header = "  " + " ".join(str(i % 10) for i in range(grid))
    lines = [header]
    for i, row in enumerate(canvas):
        lines.append(f"{i % 10} " + " ".join(row))
    return "\n".join(lines)


def multimodal_observation(world: World) -> Dict[str, str]:
    return {
        "text": text_observation(world),
        "visual": render_ascii(world),
    }


def text_only_observation(world: World) -> Dict[str, str]:
    """Text-only observation that strips the color attribute.

    This makes the instruction "red cube" impossible to resolve from text alone,
    mimicking the ambiguity Guava's visual channel resolves.
    """
    parts = [f"Step {world.step}. Last action result: {world.last_result}",
             f"Gripper: pos=({world.gripper.x:.2f},{world.gripper.y:.2f},{world.gripper.z:.2f}) "
             f"open={world.gripper.open} holding={world.gripper.holding}"]
    for name, obj in world.objects.items():
        # Color is intentionally omitted.
        parts.append(f"{name}: pos=({obj.x:.2f},{obj.y:.2f},{obj.z:.2f}) size={obj.size:.2f}")
    return {"text": "\n".join(parts), "visual": "[visual observation disabled]"}


# -----------------------------------------------------------------------------
# Agents (stand-ins for a VLM calling tools)
# -----------------------------------------------------------------------------

def run_task_done(world: World, task_target: str) -> bool:
    return world.objects[task_target].in_basket


class ReActAgent:
    """Guava-style agent: think, call one tool, observe, repeat."""

    def __init__(self, low_level: bool = False):
        self.low_level = low_level

    def think_and_act(self, world: World, obs: Dict[str, str], target: str = "red_cube") -> Tuple[str, Dict]:
        """Return (reasoning, tool_call)."""
        g = world.gripper
        held = g.holding
        basket = world.objects.get("basket")
        target_obj = world.objects.get(target)

        # 1. Task already finished?
        if target_obj.in_basket:
            return "Task complete: target already in basket.", {"name": "done", "arguments": {}}

        # 2. Recovery from a failed grasp: if gripper is closed and holding nothing,
        #    release and try again.
        if not g.open and held is None:
            return "Recovery: last grasp failed, opening gripper before retry.", {"name": "release", "arguments": {}}

        # 3. If we are already holding the target, transport and release it.
        if held == target:
            if basket and not world.is_above("basket", tol=basket.size / 2):
                if self.low_level:
                    return "Move directly above basket (low-level numeric).", {
                        "name": "move", "arguments": {"x": basket.x, "y": basket.y, "z": g.z}}
                return "Align above basket for safe placement.", {
                    "name": "align", "arguments": {"obj": "basket", "direction": "top", "clearance": "medium"}}
            if g.z > basket.z + 0.05:
                return "Lower into basket.", {"name": "move", "arguments": {"x": basket.x, "y": basket.y, "z": basket.z + 0.05}}
            return "Release target into basket.", {"name": "release", "arguments": {}}

        # 4. Acquire the target.
        if not world.is_above(target):
            if self.low_level:
                # Low-level agent must compute numeric coordinates itself.
                return "Move directly over target (low-level numeric).", {
                    "name": "move", "arguments": {"x": target_obj.x, "y": target_obj.y, "z": g.z}}
            return "Align above target for grasping.", {
                "name": "align", "arguments": {"obj": target, "direction": "top", "clearance": "medium"}}
        if g.open:
            return "Gripper is aligned and open; grasp target.", {"name": "grasp", "arguments": {"obj": target}}
        return "Unexpected closed gripper before grasp; release and retry.", {"name": "release", "arguments": {}}


class OneShotAgent:
    """Open-loop baseline: builds a fixed plan from the first observation and runs it.

    This mirrors the one-shot code-generation harnesses that Guava argues are brittle.
    """

    def __init__(self, target: str = "red_cube", low_level: bool = False):
        self.target = target
        self.low_level = low_level
        self.plan = self._make_plan()
        self.idx = 0

    def _make_plan(self) -> List[Dict]:
        if self.low_level:
            return [
                {"name": "move", "arguments": {"x": 0.3, "y": 0.3, "z": 0.3}},  # guessed over red_cube
                {"name": "grasp", "arguments": {"obj": self.target}},
                {"name": "move", "arguments": {"x": 0.7, "y": 0.7, "z": 0.3}},  # guessed over basket
                {"name": "release", "arguments": {}},
            ]
        return [
            {"name": "align", "arguments": {"obj": self.target, "direction": "top", "clearance": "medium"}},
            {"name": "grasp", "arguments": {"obj": self.target}},
            {"name": "align", "arguments": {"obj": "basket", "direction": "top", "clearance": "medium"}},
            {"name": "release", "arguments": {}},
        ]

    def think_and_act(self, world: World, obs: Dict[str, str], target: str = "red_cube") -> Tuple[str, Dict]:
        if self.idx < len(self.plan):
            action = self.plan[self.idx]
            self.idx += 1
            return f"Open-loop step {self.idx}/{len(self.plan)}: {action['name']}", action
        return "Plan exhausted.", {"name": "done", "arguments": {}}


# -----------------------------------------------------------------------------
# Harness runner
# -----------------------------------------------------------------------------

def run_harness(world: World,
                agent,
                target: str = "red_cube",
                task_target: Optional[str] = None,
                max_steps: int = 20,
                obs_mode: str = "multimodal") -> Dict:
    """Run a closed-loop interaction and return the trial result.

    target: the object the agent is told to manipulate.
    task_target: the object that actually satisfies the task (defaults to target).
                 Used to measure text-only ablations where the agent picks the wrong object.
    """
    task_target = task_target or target
    history: List[Dict] = []
    for _ in range(max_steps):
        obs = multimodal_observation(world) if obs_mode == "multimodal" else text_only_observation(world)
        reasoning, action = agent.think_and_act(world, obs, target=target)
        if action["name"] == "done":
            history.append({"reasoning": reasoning, "action": "done"})
            break
        result = execute_tool(world, action["name"], action["arguments"])
        history.append({
            "step": world.step,
            "reasoning": reasoning,
            "action": action,
            "result": result,
            "observation": obs,
        })
        if run_task_done(world, task_target):
            break
    return {
        "success": run_task_done(world, task_target),
        "steps": world.step,
        "history": history,
    }


def make_world(seed: Optional[int] = None) -> World:
    if seed is not None:
        random.seed(seed)
    # Red and blue cubes differ only by color; text-only observations omit color.
    objects = {
        "red_cube": Object("red_cube", x=0.25, y=0.40, z=0.0, size=0.08, color="red"),
        "blue_cube": Object("blue_cube", x=0.25, y=0.60, z=0.0, size=0.08, color="blue"),
        "basket":   Object("basket",   x=0.75, y=0.50, z=0.0, size=0.18, color="brown"),
    }
    return World(objects=objects, gripper=Gripper(x=0.5, y=0.5, z=0.5, open=True))


# -----------------------------------------------------------------------------
# Main demonstration
# -----------------------------------------------------------------------------

def print_trial(label: str, result: Dict, verbose: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"Condition: {label}")
    print(f"Success: {result['success']}  |  Steps used: {result['steps']}")
    if verbose:
        for entry in result["history"]:
            print("-" * 40)
            print(f"Step {entry.get('step', '-')}")
            if "observation" in entry:
                print("Observation (text):")
                print(textwrap.indent(entry["observation"]["text"], "  "))
                print("Observation (visual):")
                print(textwrap.indent(entry["observation"]["visual"], "  "))
            print(f"Think: {entry.get('reasoning', '')}")
            print(f"Tool:  {entry.get('action', '')}")
            print(f"Result: {entry.get('result', '')}")


def main():
    random.seed(42)
    print("Guava Harness Probe")
    print("Task: place the red cube in the basket")
    print("Note: the first grasp attempt is randomly allowed to fail to test recovery.\n")

    # 1. Full Guava-style agent (seed=1 causes the first grasp to fail).
    world = make_world(seed=1)
    res = run_harness(world, ReActAgent(low_level=False), target="red_cube", obs_mode="multimodal")
    print_trial("Guava-style: ReAct + semantic tools + multimodal obs", res, verbose=True)

    # 2. Ablation: one-shot open-loop execution.
    world = make_world(seed=1)
    res = run_harness(world, OneShotAgent(target="red_cube", low_level=False), target="red_cube", obs_mode="multimodal")
    print_trial("Ablation: one-shot plan (no ReAct loop)", res, verbose=False)

    # 3. Ablation: low-level action space with ReAct.
    world = make_world(seed=1)
    res = run_harness(world, ReActAgent(low_level=True), target="red_cube", obs_mode="multimodal")
    print_trial("Ablation: low-level numeric moves (no semantic align/grasp)", res, verbose=False)

    # 4. Ablation: text-only observation (color stripped -> wrong cube selection).
    # The agent is forced to pick the first cube it can name; it chooses blue_cube instead of red_cube.
    world = make_world(seed=1)
    res = run_harness(world, ReActAgent(low_level=False), target="blue_cube",
                      task_target="red_cube", obs_mode="text_only")
    print_trial("Ablation: text-only observations (color omitted -> picks blue cube)", res, verbose=False)

    # Monte-Carlo summary over stochastic grasp failures.
    print("\n" + "="*60)
    print("Monte-Carlo summary (20 trials, seed 100-119)")
    conditions = [
        ("ReAct + semantic + multimodal", lambda: run_harness(make_world(), ReActAgent(low_level=False), target="red_cube", task_target="red_cube", obs_mode="multimodal")),
        ("One-shot", lambda: run_harness(make_world(), OneShotAgent(target="red_cube", low_level=False), target="red_cube", task_target="red_cube", obs_mode="multimodal")),
        ("Low-level moves", lambda: run_harness(make_world(), ReActAgent(low_level=True), target="red_cube", task_target="red_cube", obs_mode="multimodal")),
        ("Text-only", lambda: run_harness(make_world(), ReActAgent(low_level=False), target="blue_cube", task_target="red_cube", obs_mode="text_only")),
    ]
    random.seed(100)
    for label, runner in conditions:
        successes = 0
        total_steps = 0
        for i in range(20):
            random.seed(100 + i)
            r = runner()
            successes += int(r["success"])
            total_steps += r["steps"]
        print(f"  {label:35s} success {successes}/20  avg_steps {total_steps/20:.1f}")


if __name__ == "__main__":
    main()
