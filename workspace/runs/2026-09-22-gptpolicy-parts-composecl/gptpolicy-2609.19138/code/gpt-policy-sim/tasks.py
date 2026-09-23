"""Task definitions for the GPT-Policy simulated backend.

Each task provides: a seeded initial-layout builder, a ground-truth success
criterion for evaluation, and a scripted oracle expert (used both by MockAgent
smoke tests and by demo recording). Experts read ground-truth world state —
they exist to exercise the protocol loop and to produce demonstrations, not to
be vision policies.

Tasks:
- T1 gate-button: red cube behind a transparent gate; pressing the blue button
  slides the gate open. Instruction mentions only cube and bowl.
- T2 align-insert: plug lying in a shallow slot, off the alignment mark; grasp
  only succeeds within 1.5 cm of the mark, so the plug must be pushed first.
- T3 hook-retrieve: cube inside a single-open-ended tube the EE cannot enter;
  a hook tool can drag it out.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from sim_world import Container, SimObject, World

DOWN = [1.0, 0.0, 0.0, 0.0]


def move(x, y, z, note):
    return {"name": "move_to", "arguments": {"target": {"pose_xyzquat": [x, y, z, *DOWN]}, "note": note}}


def chunk(points, note):
    return {"name": "move_eef_chunk",
            "arguments": {"poses": [{"pose_xyzquat": [x, y, z, *DOWN]} for x, y, z in points], "note": note}}


def grip(value, note):
    return {"name": "set_gripper", "arguments": {"gripper": value, "note": note}}


def done(summary):
    return {"name": "done", "arguments": {"summary": summary, "hindsight": "scripted expert rollout"}}


def give_up(reason):
    return {"name": "give_up", "arguments": {"reason": reason, "hindsight": "scripted expert failure"}}


class Step:
    def __init__(self, stage: str, decide: Callable[[World], dict], complete: Callable[[World], bool]):
        self.stage = stage
        self.decide = decide
        self.complete = complete


class ScriptedExpert:
    """Stage-machine oracle expert; emits one protocol decision per act() call."""

    def __init__(self, world: World, steps: list[Step]) -> None:
        self.steps = steps
        self.i = 0
        self.attempts = 0
        self.stage = "start"

    def act(self, world: World) -> dict[str, Any]:
        while self.i < len(self.steps):
            step = self.steps[self.i]
            if step.complete(world):
                self.i += 1
                self.attempts = 0
                continue
            self.stage = step.stage
            self.attempts += 1
            if self.attempts > 4:
                return give_up(f"expert stuck at stage {step.stage}")
            return step.decide(world)
        self.stage = "done"
        return done("scripted expert completed all stages")


def _move_step(stage, target: Callable[[World], tuple], note, tol=0.004):
    def decide(world):
        return move(*target(world), note)

    def complete(world):
        return bool(np.linalg.norm(world.ee - np.asarray(target(world))) <= tol)

    return Step(stage, decide, complete)


def _chunk_step(stage, points: Callable[[World], list], note, tol=0.004):
    def decide(world):
        return chunk(points(world), note)

    def complete(world):
        return bool(np.linalg.norm(world.ee - np.asarray(points(world)[-1])) <= tol)

    return Step(stage, decide, complete)


def _grip_step(stage, value, want_grasped, note):
    closing = value <= 0.1

    def decide(world):
        return grip(value, note)

    def complete(world):
        if want_grasped:
            return world.grasped is not None
        reached = world.gripper_measured <= 0.1 if closing else world.gripper_measured >= 0.9
        return world.grasped is None and reached

    return Step(stage, decide, complete)


@dataclass
class Task:
    key: str
    name: str
    instruction: str
    build_world: Callable[[int], World]
    success: Callable[[World], tuple[bool, str]]
    expert: Callable[[World], ScriptedExpert]
    goal_overlay: Callable[[World], None]  # mutate a fresh world into its goal state


# --------------------------------------------------------------------- T1

def _t1_build(seed: int) -> World:
    rng = np.random.default_rng(seed)
    world = World(seed=seed + 50000)

    def jit():
        return rng.uniform(-0.05, 0.05, 2)

    cube_xy = np.array([0.65, 0.50]) + jit()
    bowl_xy = np.array([0.30, 0.30]) + jit()
    button_xy = np.array([0.30, 0.70]) + jit()
    cube = world.add(SimObject("cube", "cube", "red", np.array([*cube_xy, 0.02]),
                               np.full(3, 0.02), graspable=True))
    bowl = world.add(SimObject("bowl", "bowl", "green", np.array([*bowl_xy, 0.0]),
                               np.array([0.06, 0.06, 0.04])))
    button = world.add(SimObject("button", "button", "blue", np.array([*button_xy, 0.008]),
                                 np.full(3, 0.015)))
    gate = {"x": 0.55, "y_span": (0.15, 0.85), "height": 0.45, "open": False, "slide_offset": 0.30}
    world.add(SimObject("gate", "gate", "gray", np.zeros(3), np.zeros(3), meta=gate))
    world.containers.append(Container("bowl", bowl.position[:2].copy(), 0.06, 0.005))

    def gate_blocker(point):
        if gate["open"]:
            return None
        if (abs(point[0] - gate["x"]) <= 0.011
                and gate["y_span"][0] - 0.005 <= point[1] <= gate["y_span"][1] + 0.005
                and point[2] <= gate["height"] + 0.005):
            return "gate"
        return None

    world.blockers.append(gate_blocker)

    def button_listener(w: World, prev: np.ndarray, new: np.ndarray) -> None:
        if gate["open"]:
            return
        if (np.linalg.norm(new[:2] - button.position[:2]) <= 0.015
                and new[2] <= button.top_z + 0.004):
            gate["open"] = True
            w.log("button_pressed", object="button")
            w.log("gate_opened", object="gate")

    world.contact_listeners.append(button_listener)
    return world


def _t1_success(world: World) -> tuple[bool, str]:
    cube, bowl = world.objects["cube"], world.objects["bowl"]
    d = float(np.linalg.norm(cube.position[:2] - bowl.position[:2]))
    ok = world.grasped is None and d <= 0.06 and cube.position[2] <= 0.045
    return ok, f"cube_in_bowl={ok} dist_xy={d:.3f} cube_z={cube.position[2]:.3f} grasped={world.grasped}"


def _t1_expert(world: World) -> ScriptedExpert:
    cube, bowl, button = world.objects["cube"], world.objects["bowl"], world.objects["button"]
    gate = world.objects["gate"].meta
    steps = [
        _move_step("approach-button", lambda w: (button.position[0], button.position[1], 0.08), "到按钮上方"),
        _move_step("press-button", lambda w: (button.position[0], button.position[1], button.top_z + 0.003), "下压按钮"),
        Step("verify-gate-open", lambda w: move(w.ee[0], w.ee[1], 0.12, "抬升确认门开"),
             lambda w: gate["open"] and w.ee[2] >= 0.10),
        _move_step("approach-cube", lambda w: (cube.position[0], cube.position[1], 0.18), "移到方块上方"),
        _move_step("descend-cube", lambda w: (cube.position[0], cube.position[1], 0.026), "下降对方块"),
        _grip_step("grasp-cube", 0.0, True, "闭合夹爪抓方块"),
        _chunk_step("carry-to-bowl", lambda w: [
            (cube.position[0], cube.position[1], 0.22),
            (bowl.position[0], bowl.position[1], 0.22),
            (bowl.position[0], bowl.position[1], 0.10),
        ], "抓起并移到碗上方"),
        _grip_step("release-cube", 1.0, False, "松开方块入碗"),
        _move_step("retract", lambda w: (bowl.position[0], bowl.position[1], 0.20), "抬升离开"),
    ]
    return ScriptedExpert(world, steps)


def _t1_goal(world: World) -> None:
    cube, bowl = world.objects["cube"], world.objects["bowl"]
    cube.position = np.array([bowl.position[0], bowl.position[1], 0.025])
    world.objects["gate"].meta["open"] = True


# --------------------------------------------------------------------- T2

T2_ALIGN_TOLERANCE_M = 0.015


def _t2_build(seed: int) -> World:
    rng = np.random.default_rng(seed)
    world = World(seed=seed + 60000)
    slot_y = 0.50
    marker_x = 0.50
    plug_x = float(np.clip(0.57 + rng.uniform(-0.05, 0.05), 0.36, 0.64))
    box_xy = np.array([0.28, 0.30]) + rng.uniform(-0.05, 0.05, 2)
    plug = world.add(SimObject("plug", "plug", "orange", np.array([plug_x, slot_y, 0.008]),
                               np.full(3, 0.008), graspable=True))
    world.add(SimObject("slot", "slot", "gray", np.zeros(3), np.zeros(3),
                        meta={"y": slot_y, "x_span": (0.35, 0.65), "marker_x": marker_x}))
    world.add(SimObject("box", "box", "brown", np.array([*box_xy, 0.02]),
                        np.array([0.05, 0.05, 0.02]), meta={"socket_xy": box_xy.copy()}))
    world.containers.append(Container("box", box_xy.copy(), 0.045, 0.005))

    def alignment_rule(w: World, obj: SimObject):
        if obj.name == "plug" and abs(obj.position[0] - marker_x) > T2_ALIGN_TOLERANCE_M:
            return "plug_off_alignment_mark"
        return None

    world.grasp_rules.append(alignment_rule)

    def push_listener(w: World, prev: np.ndarray, new: np.ndarray) -> None:
        if w.gripper_measured >= 0.2 or w.grasped is not None:
            return  # pushing happens with closed empty gripper
        if new[2] > 0.030 or abs(new[1] - slot_y) > 0.020:
            return
        if abs(new[0] - plug.position[0]) <= 0.023:
            delta = float(new[0] - prev[0])
            if delta:
                plug.position[0] = float(np.clip(plug.position[0] + delta, 0.36, 0.64))
                w.log("plug_pushed", x=round(plug.position[0], 4))

    world.contact_listeners.append(push_listener)
    return world


def _t2_success(world: World) -> tuple[bool, str]:
    plug = world.objects["plug"]
    socket = world.objects["box"].meta["socket_xy"]
    d = float(np.linalg.norm(plug.position[:2] - socket))
    ok = world.grasped is None and d <= 0.02 and plug.position[2] <= 0.06
    return ok, f"plug_in_box={ok} dist_xy={d:.3f} plug_z={plug.position[2]:.3f} grasped={world.grasped}"


def _t2_expert(world: World) -> ScriptedExpert:
    plug = world.objects["plug"]
    slot = world.objects["slot"].meta
    socket = world.objects["box"].meta["socket_xy"]
    slot_y, marker = slot["y"], slot["marker_x"]
    steps = [
        _grip_step("close-gripper", 0.0, False, "闭合夹爪准备推拨"),  # completes immediately (grasped None)
        _move_step("approach-plug", lambda w: (plug.position[0] + 0.02, slot_y, 0.10), "移到柱子上方"),
        _move_step("lower-behind-plug", lambda w: (plug.position[0] + 0.02, slot_y, 0.020), "降到槽内柱子后方"),
        Step("push-to-mark", lambda w: move(marker + 0.02, slot_y, 0.020, "沿槽推柱子到对齐标记"),
             lambda w: abs(plug.position[0] - marker) <= 0.006),
        _grip_step("open-gripper", 1.0, False, "张开夹爪"),
        _move_step("lift", lambda w: (w.ee[0], slot_y, 0.10), "抬起"),
        _move_step("above-plug", lambda w: (plug.position[0], slot_y, 0.06), "对准柱子"),
        _move_step("descend-plug", lambda w: (plug.position[0], slot_y, 0.012), "下降对柱子"),
        _grip_step("grasp-plug", 0.0, True, "闭合夹爪抓柱子"),
        _chunk_step("carry-to-box", lambda w: [
            (plug.position[0], slot_y, 0.15),
            (socket[0], socket[1], 0.15),
            (socket[0], socket[1], 0.055),
        ], "抓起柱子移到盒子"),
        _grip_step("release-plug", 1.0, False, "松开柱子入盒"),
        _move_step("retract", lambda w: (socket[0], socket[1], 0.18), "抬升离开"),
    ]
    return ScriptedExpert(world, steps)


def _t2_goal(world: World) -> None:
    plug = world.objects["plug"]
    socket = world.objects["box"].meta["socket_xy"]
    plug.position = np.array([socket[0], socket[1], 0.013])


# --------------------------------------------------------------------- T3

def _t3_build(seed: int) -> World:
    rng = np.random.default_rng(seed)
    world = World(seed=seed + 70000)
    tube_y = 0.55
    tube = {"y": tube_y, "z": 0.03, "r_inner": 0.025, "r_outer": 0.03, "x_span": (0.45, 0.75)}
    cube_x = float(0.68 + rng.uniform(-0.03, 0.03))
    cube = world.add(SimObject("cube", "cube", "red", np.array([cube_x, tube_y, 0.025]),
                               np.full(3, 0.02), graspable=True, meta={"in_tube": True}))
    hook_xy = np.array([0.30, 0.30]) + rng.uniform(-0.05, 0.05, 2)
    world.add(SimObject("hook", "hook", "magenta", np.array([*hook_xy, 0.01]),
                        np.array([0.01, 0.07, 0.01]), graspable=True, meta={}))
    bowl_xy = np.array([0.32, 0.78]) + rng.uniform(-0.04, 0.04, 2)
    world.add(SimObject("bowl", "bowl", "green", np.array([*bowl_xy, 0.0]),
                        np.array([0.06, 0.06, 0.04])))
    world.add(SimObject("tube", "tube", "gray", np.zeros(3), np.zeros(3), meta=tube))
    world.containers.append(Container("bowl", bowl_xy.copy(), 0.06, 0.005))

    def tube_blocker(point):
        x0, x1 = tube["x_span"]
        if x0 - 0.004 <= point[0] <= x1 + 0.004:
            if math.hypot(point[1] - tube["y"], point[2] - tube["z"]) <= tube["r_outer"] + 0.01:
                return "tube"
        return None

    world.blockers.append(tube_blocker)

    def tube_rule(w: World, obj: SimObject):
        if obj.name == "cube" and obj.meta.get("in_tube"):
            return "cube_inside_tube_unreachable"
        return None

    world.grasp_rules.append(tube_rule)

    def drag_listener(w: World, prev: np.ndarray, new: np.ndarray) -> None:
        hook = w.objects["hook"]
        tip = hook.meta.get("tip_xyz")
        if w.grasped != "hook" or tip is None:
            return
        prev_tip = hook.meta.get("prev_tip_xyz", tip)
        delta = np.asarray(tip) - np.asarray(prev_tip)
        hook.meta["prev_tip_xyz"] = list(tip)
        if not cube.meta.get("in_tube") and cube.position[0] < tube["x_span"][0] - 0.06:
            return
        if np.linalg.norm(np.asarray(tip[:2]) - cube.position[:2]) <= 0.045 and abs(tip[2] - cube.position[2]) <= 0.02:
            cube.position[0] += delta[0]
            if cube.meta.get("in_tube"):
                # The bore constrains y/z and the closed end caps +x travel.
                cube.position[1] = tube["y"]
                cube.position[0] = float(min(cube.position[0], tube["x_span"][1] - 0.02))
                if cube.position[0] < tube["x_span"][0]:
                    cube.meta["in_tube"] = False
                    cube.position[2] = 0.02
                    w.log("cube_left_tube", x=round(float(cube.position[0]), 4))
            else:
                cube.position[1] += delta[1]
            w.log("cube_dragged", x=round(float(cube.position[0]), 4))

    world.contact_listeners.append(drag_listener)

    original_sync = World._sync_grasped

    def sync_with_hook(self):
        original_sync(self)
        if self.grasped == "hook":
            hook_obj = self.objects["hook"]
            # The held hook points straight down-base +x (toward the tube); its
            # tip rides 14 cm ahead of and 7 cm below the TCP.
            hook_obj.meta.setdefault("held_dir", [1.0, 0.0])
            direction = np.asarray(hook_obj.meta["held_dir"])
            tip_xy = self.ee[:2] + 0.14 * direction
            tip_z = float(self.ee[2] - 0.07)
            hook_obj.meta["tip_xy"] = [float(tip_xy[0]), float(tip_xy[1])]
            hook_obj.meta["tip_xyz"] = [float(tip_xy[0]), float(tip_xy[1]), tip_z]

    world._sync_grasped = types.MethodType(sync_with_hook, world)
    return world


def _t3_success(world: World) -> tuple[bool, str]:
    cube, bowl = world.objects["cube"], world.objects["bowl"]
    d = float(np.linalg.norm(cube.position[:2] - bowl.position[:2]))
    ok = (world.grasped is None and not cube.meta.get("in_tube") and d <= 0.06
          and cube.position[2] <= 0.045)
    return ok, f"cube_in_bowl={ok} dist_xy={d:.3f} in_tube={cube.meta.get('in_tube')} grasped={world.grasped}"


def _t3_expert(world: World) -> ScriptedExpert:
    cube, bowl, hook = world.objects["cube"], world.objects["bowl"], world.objects["hook"]
    tube = world.objects["tube"].meta
    tube_y = tube["y"]
    mouth_x = tube["x_span"][0]

    def insertion_ee(w):
        # High pass: the hook tip stays above the cube (tip z = EE z - 0.07),
        # so it can reach the far side without pushing the cube deeper.
        return (float(cube.position[0] + 0.04 - 0.14), tube_y, 0.13)

    def hook_behind_cube(w):
        return (float(cube.position[0] + 0.04 - 0.14), tube_y, 0.10)

    steps = [
        _move_step("approach-hook", lambda w: (hook.position[0], hook.position[1], 0.10), "移到钩子上方"),
        _move_step("descend-hook", lambda w: (hook.position[0], hook.position[1], 0.015), "下降对钩子"),
        _grip_step("grasp-hook", 0.0, True, "闭合夹爪抓钩子"),
        _move_step("lift-hook", lambda w: (hook.position[0], hook.position[1], 0.14), "抬起钩子"),
        _move_step("insert-hook-high", insertion_ee, "钩尖从上方伸入管内越过方块", tol=0.008),
        _move_step("lower-hook-behind", hook_behind_cube, "钩尖落到方块后方", tol=0.008),
        Step("drag-cube-out", lambda w: _t3_drag_decision(w, hook),
             lambda w: not cube.meta.get("in_tube") and cube.position[0] <= mouth_x - 0.02),
        _move_step("lift-hook-away", lambda w: (w.ee[0], w.ee[1], 0.16), "抬起钩子"),
        _move_step("drop-zone", lambda w: (0.30, 0.42, 0.16), "移开放下钩子"),
        _grip_step("release-hook", 1.0, False, "放下钩子"),
        _move_step("approach-cube", lambda w: (cube.position[0], cube.position[1], 0.12), "移到方块上方"),
        _move_step("descend-cube", lambda w: (cube.position[0], cube.position[1], 0.026), "下降对方块"),
        _grip_step("grasp-cube", 0.0, True, "闭合夹爪抓方块"),
        _chunk_step("carry-to-bowl", lambda w: [
            (cube.position[0], cube.position[1], 0.22),
            (bowl.position[0], bowl.position[1], 0.22),
            (bowl.position[0], bowl.position[1], 0.10),
        ], "抓起并移到碗上方"),
        _grip_step("release-cube", 1.0, False, "松开方块入碗"),
        _move_step("retract", lambda w: (bowl.position[0], bowl.position[1], 0.20), "抬升离开"),
    ]
    return ScriptedExpert(world, steps)


def _t3_drag_decision(world: World, hook: SimObject) -> dict:
    """Keep the EE at drag height and walk it back along the tube axis."""
    return move(float(world.ee[0] - 0.38), float(world.ee[1]), 0.10, "回拖钩子带出方块")


def _t3_goal(world: World) -> None:
    cube, bowl = world.objects["cube"], world.objects["bowl"]
    cube.position = np.array([bowl.position[0], bowl.position[1], 0.025])
    cube.meta["in_tube"] = False


TASKS = {
    "T1": Task("T1", "gate-button", "把红方块放进绿碗",
               _t1_build, _t1_success, _t1_expert, _t1_goal),
    "T2": Task("T2", "align-insert", "把柱子插到盒子里",
               _t2_build, _t2_success, _t2_expert, _t2_goal),
    "T3": Task("T3", "hook-retrieve", "把红方块放进绿碗",
               _t3_build, _t3_success, _t3_expert, _t3_goal),
}
