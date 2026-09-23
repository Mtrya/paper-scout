"""Point-end-effector tabletop world speaking the GPT-Policy robot protocol.

The simulated arm is abstracted as a point TCP with a freely trackable
orientation (measured orientation follows the commanded one). Joint telemetry
is a smooth mock 6-DoF kinematic mapping; consumers should read tcp poses and
images, exactly like the real protocol intends. Task-specific mechanics (gate,
button, tube, plug, hook) are registered by tasks.py through blockers, contact
listeners, containers and grasp rules; this module stays task-agnostic.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from gpt_policy.geometry.poses import quaternion_to_rpy, rpy_to_quaternion
from gpt_policy.motion.coordination import TrajectoryIKError

# Workspace bounds in the (simulated) arm base frame: +x forward, +y left, +z up.
WORKSPACE = {"x": (0.15, 0.85), "y": (0.15, 0.85), "z": (0.0, 0.45)}
TABLE_Z = 0.0
HOME_XYZ = np.array([0.5, 0.0, 0.3])
# Unit quaternion (xyzw) pointing tool +z along base -z: gripper straight down.
DOWN_QUAT_XYZW = np.array([1.0, 0.0, 0.0, 0.0])

EE_SPEED_M_S = 0.1
MOVE_STEP_M = 0.002
GRIPPER_TRAVEL_S = 1.0
GRIPPER_WIDTH_M = 0.088
GRASP_XY_TOLERANCE_M = 0.025
GRASP_Z_TOLERANCE_M = 0.03

Blocker = Callable[[np.ndarray], str | None]
ContactListener = Callable[["World", np.ndarray, np.ndarray], None]
GraspRule = Callable[["World", "SimObject"], str | None]


@dataclass
class SimObject:
    name: str
    kind: str
    color: str
    position: np.ndarray
    half_extents: np.ndarray  # xyz half sizes for support/render
    graspable: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def top_z(self) -> float:
        return float(self.position[2] + self.half_extents[2])


@dataclass
class Container:
    name: str
    xy: np.ndarray
    radius: float
    floor_z: float


class World:
    """Mutable simulated scene; all positions are metres in the base frame."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.objects: dict[str, SimObject] = {}
        self.blockers: list[Blocker] = []
        self.contact_listeners: list[ContactListener] = []
        self.grasp_rules: list[GraspRule] = []
        self.containers: list[Container] = []
        self.ee = HOME_XYZ.copy()
        self.ee_quat_xyzw = DOWN_QUAT_XYZW.copy()
        self.gripper_command = 1.0
        self.gripper_measured = 1.0
        self.grasped: str | None = None
        self._grasp_dz = 0.0
        self.time_s = 0.0
        self.events: list[dict[str, Any]] = []

    # ------------------------------------------------------------- scene API
    def add(self, obj: SimObject) -> SimObject:
        self.objects[obj.name] = obj
        return obj

    def log(self, kind: str, **payload: Any) -> None:
        self.events.append({"t_s": round(self.time_s, 6), "kind": kind, **payload})

    def obstacle_at(self, point: np.ndarray) -> str | None:
        if self.grasped is not None:
            # A carried object follows the TCP; only the bare point is checked.
            pass
        for blocker in self.blockers:
            name = blocker(point)
            if name is not None:
                return name
        return None

    def support_height(self, obj: SimObject, xy: np.ndarray) -> float:
        best = TABLE_Z
        for container in self.containers:
            if np.linalg.norm(xy - container.xy) <= container.radius:
                best = max(best, container.floor_z)
        return best + float(obj.half_extents[2])

    # --------------------------------------------------------------- motion
    def move_to(self, target_xyz: np.ndarray, quat_xyzw: np.ndarray) -> dict[str, Any]:
        """Advance the EE in straight-line micro-steps; stop at first contact."""
        start = self.ee.copy()
        distance = float(np.linalg.norm(target_xyz - start))
        steps = max(1, int(math.ceil(distance / MOVE_STEP_M)))
        blocked = None
        for i in range(1, steps + 1):
            nxt = start + (target_xyz - start) * (i / steps)
            obstacle = self.obstacle_at(nxt)
            if obstacle is not None:
                blocked = {
                    "obstacle": obstacle,
                    "contact_xyz": [float(v) for v in self.ee],
                    "planned_target_xyz": [float(v) for v in target_xyz],
                    "reached_target": False,
                }
                self.log("blocked", **blocked)
                break
            prev = self.ee.copy()
            self.ee = nxt
            self.time_s += MOVE_STEP_M / EE_SPEED_M_S
            self._sync_grasped()
            for listener in self.contact_listeners:
                listener(self, prev, nxt)
        if blocked is None:
            self.ee_quat_xyzw = quat_xyzw
        duration = distance / EE_SPEED_M_S
        return {
            "start_xyz": start,
            "target_xyz": np.asarray(target_xyz, dtype=np.float64),
            "distance_m": distance,
            "duration_s": duration,
            "blocked": blocked,
        }

    def _sync_grasped(self) -> None:
        if self.grasped is None:
            return
        obj = self.objects[self.grasped]
        obj.position = np.array([self.ee[0], self.ee[1], self.ee[2] + self._grasp_dz])

    # -------------------------------------------------------------- gripper
    def set_gripper(self, target: float) -> dict[str, Any]:
        target = float(np.clip(target, 0.0, 1.0))
        closing = target < self.gripper_command
        self.gripper_command = target
        self.time_s += GRIPPER_TRAVEL_S
        attached = None
        released = None
        if closing and target <= 0.1 and self.grasped is None:
            obj, reason = self._find_grasp()
            if obj is not None:
                self.grasped = obj.name
                self._grasp_dz = float(obj.position[2] - self.ee[2])
                self._sync_grasped()
                attached = obj.name
                self.log("grasp_attached", object=obj.name)
            elif reason:
                self.log("grasp_failed", reason=reason)
        elif not closing and target >= 0.9 and self.grasped is not None:
            obj = self.objects[self.grasped]
            self.grasped = None
            obj.position[2] = self.support_height(obj, obj.position[:2])
            released = obj.name
            self.log("released", object=obj.name, at_xyz=[float(v) for v in obj.position])
        # Finger closure is clamped by a held object's width, like the real streamer.
        held = self.objects[self.grasped] if self.grasped else None
        if held is not None and target <= 0.1:
            object_fraction = min(1.0, 2.0 * float(held.half_extents[1]) / GRIPPER_WIDTH_M)
            self.gripper_measured = max(target, object_fraction)
            torque = 1.1
        else:
            self.gripper_measured = target
            torque = 0.25
        return {"attached": attached, "released": released, "torque_nm": torque}

    def _find_grasp(self) -> tuple[SimObject | None, str | None]:
        best: SimObject | None = None
        best_d = GRASP_XY_TOLERANCE_M
        rejection = None
        for obj in self.objects.values():
            if not obj.graspable:
                continue
            d_xy = float(np.linalg.norm(self.ee[:2] - obj.position[:2]))
            d_z = abs(float(self.ee[2] - obj.position[2]))
            if d_xy > GRASP_XY_TOLERANCE_M or d_z > GRASP_Z_TOLERANCE_M:
                continue
            for rule in self.grasp_rules:
                reason = rule(self, obj)
                if reason is not None:
                    rejection = reason
                    best = None
                    break
            else:
                if d_xy <= best_d:
                    best, best_d = obj, d_xy
                continue
            break
        if best is None and rejection is None:
            rejection = "no_graspable_object_in_gripper_window"
        return best, rejection

    # ---------------------------------------------------------------- state
    def mock_joints(self, xyz: np.ndarray) -> np.ndarray:
        """Smooth, plausible 6-DoF mock inverse kinematics for telemetry only."""
        x, y, z = (float(v) for v in xyz)
        j0 = math.atan2(y, x)
        r = max(math.hypot(x, y) - 0.05, 0.06)
        h = 0.42 - z
        l1, l2 = 0.30, 0.30
        d = min(math.hypot(r, h), l1 + l2 - 1e-6)
        cos_el = float(np.clip((r * r + h * h - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1.0, 1.0))
        j2 = -math.acos(cos_el)
        j1 = math.atan2(h, r) - math.atan2(l2 * math.sin(-j2), l1 + l2 * math.cos(-j2))
        j3 = -(j1 + j2) - math.pi / 2
        j4 = 0.5 * math.sin(j0)
        j5 = 0.0
        return np.array([j0, j1, j2, j3, j4, j5])

    def tcp_xyzrpy(self) -> list[float]:
        rpy = quaternion_to_rpy(self.ee_quat_xyzw)
        return [float(v) for v in (*self.ee, *rpy)]

    def state_dict(self) -> dict[str, Any]:
        joints = self.mock_joints(self.ee) + self.rng.normal(0.0, 0.0015, 6)
        tcp = np.asarray(self.tcp_xyzrpy()) + self.rng.normal(0.0, 3e-4, 6)
        held = self.grasped is not None
        return {
            "joint_positions_rad": [float(v) for v in joints],
            "joint_velocities_rad_s": [float(v) for v in self.rng.normal(0.0, 0.002, 6)],
            "joint_torques_nm": [float(v) for v in (0.4, 0.9, 0.6, 0.2, 0.1, 0.05)],
            "joint_command_positions_rad": [float(v) for v in self.mock_joints(self.ee)],
            "tcp_xyzrpy": [float(v) for v in tcp],
            "tcp_xyzquat": [*[float(v) for v in tcp[:3]], *[float(v) for v in self.ee_quat_xyzw]],
            "gripper_position_m": float(self.gripper_measured * GRIPPER_WIDTH_M),
            "gripper_normalized": float(self.gripper_measured),
            "gripper_command_normalized": float(self.gripper_command),
            "gripper_velocity_m_s": 0.0,
            "gripper_torque_nm": 1.1 if held else 0.25,
            "gravity_compensation": True,
            "timestamp_s": time.time(),
            "sim_time_s": float(self.time_s),
        }


def _parse_pose(value: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object with pose_xyzquat")
    pose = value.get("pose_xyzquat")
    if not isinstance(pose, list) or len(pose) != 7:
        raise ValueError(f"{name}.pose_xyzquat must be [x,y,z,qx,qy,qz,qw]")
    pose = np.asarray(pose, dtype=np.float64)
    if not np.isfinite(pose).all():
        raise ValueError(f"{name}.pose_xyzquat must be finite")
    quat = pose[3:]
    norm = float(np.linalg.norm(quat))
    if not 0.99 <= norm <= 1.01:
        raise ValueError(f"{name}.pose_xyzquat quaternion must be a unit quaternion")
    return pose[:3], quat / norm


def _workspace_violation(xyz: np.ndarray) -> str | None:
    for axis, (lo, hi) in zip("xyz", (WORKSPACE["x"], WORKSPACE["y"], WORKSPACE["z"])):
        value = float(xyz["xyz".index(axis)])
        if not lo - 1e-9 <= value <= hi + 1e-9:
            return f"{axis}={value:.4f} outside [{lo}, {hi}]"
    return None


class SimRobot:
    """GPT-Policy robot facade over a World (single arm, 6 mock joints)."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.interface = "sim0"

    @property
    def dof(self) -> int:
        return 6

    def state(self) -> dict[str, Any]:
        return self.world.state_dict()

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "state":
            return self.state()
        if name == "move_to":
            target, quat = _parse_pose(arguments.get("target"), "move_to.target")
            return self._move([target], [quat], arguments)
        if name == "move_eef_chunk":
            requested = arguments.get("poses")
            if not isinstance(requested, list) or not requested:
                raise ValueError("move_eef_chunk.poses must be a non-empty array")
            targets, quats = [], []
            for i, pose in enumerate(requested):
                xyz, quat = _parse_pose(pose, f"move_eef_chunk.poses[{i}]")
                targets.append(xyz)
                quats.append(quat)
            return self._move(targets, quats, arguments)
        if name == "check_path":
            return self._check_path(arguments)
        if name == "set_gripper":
            value = arguments.get("gripper", arguments.get("position"))
            if value is None:
                raise ValueError("set_gripper requires gripper in [0, 1]")
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("gripper must be a finite number in [0, 1]")
            return self._gripper(value)
        if name == "home":
            return self.return_home()
        raise ValueError(f"未知工具: {name}")

    # ------------------------------------------------------------------ moves
    def _move(self, targets, quats, arguments) -> dict[str, Any]:
        world = self.world
        planned_points = [[*world.tcp_xyzrpy()]]
        segments = []
        feedbacks = []
        blocked = None
        for index, (target, quat) in enumerate(zip(targets, quats)):
            violation = _workspace_violation(target)
            if violation is not None:
                raise TrajectoryIKError(
                    segment=index, sample=0, status=-9, status_name="SIM_WORKSPACE_LIMIT",
                    translation_error_m=0.0, rotation_error_rad=0.0,
                    reason=f"sim_workspace_limit: {violation}",
                )
            planned_points.append([*target.tolist(), *quaternion_to_rpy(quat).tolist()])
            move = world.move_to(target, quat)
            segments.append({
                "segment": index,
                "duration_s": move["duration_s"],
                "endpoint_fk_translation_error_m": 0.0 if move["blocked"] is None else float(
                    np.linalg.norm(move["target_xyz"] - world.ee)),
                "endpoint_fk_rotation_error_rad": 0.0,
            })
            feedbacks.append(move)
            if move["blocked"] is not None:
                blocked = {**move["blocked"], "segment": index}
                break
        result = self.state()
        total_duration = float(sum(item["duration_s"] for item in feedbacks))
        samples = [planned_points[0]]
        # Densified trace: linear samples every ~1 cm along requested segments.
        for start, end in zip(planned_points, planned_points[1:]):
            a, b = np.asarray(start[:3]), np.asarray(end[:3])
            n = max(1, int(np.linalg.norm(b - a) / 0.01))
            for i in range(1, n + 1):
                p = a + (b - a) * (i / n)
                samples.append([float(p[0]), float(p[1]), float(p[2]), *end[3:]])
        result["trajectory"] = {
            "planned_duration_s": total_duration,
            "_trace": {
                "model_tcp_points_xyzrpy": planned_points,
                "tcp_samples_xyzrpy": samples[:64],
                "segments": segments,
                "execution_start": {"sim_time_s": world.time_s},
            },
        }
        target_pose = planned_points[-1]
        measured = world.tcp_xyzrpy()
        error = np.asarray(target_pose[:3]) - np.asarray(measured[:3])
        feedback = {
            "joint_residual_rad": [float(v) for v in np.abs(world.rng.normal(0.0, 0.001, 6))],
            "max_joint_residual_rad": 0.001,
            "target_tcp_xyzrpy": target_pose,
            "measured_tcp_xyzrpy": measured,
            "tcp_error_xyz_m": [float(v) for v in error],
            "tcp_translation_error_m": float(np.linalg.norm(error)),
            "tcp_rotation_error_rad": 0.0,
            "gripper_measured_normalized": float(world.gripper_measured),
            "gripper_torque_nm": 1.1 if world.grasped else 0.25,
            "settle": {"settled": True, "method": "simulated_synchronous", "window_s": 0.2},
        }
        if blocked is not None:
            feedback["blocked"] = blocked
            result["status"] = "blocked_by_obstacle"
        result["execution_feedback"] = feedback
        return result

    def _check_path(self, arguments) -> dict[str, Any]:
        requested = arguments.get("poses") or ([arguments["target"]] if "target" in arguments else None)
        if not isinstance(requested, list) or not requested:
            raise ValueError("check_path.poses must be a non-empty array")
        world = self.world
        points = [world.ee.copy()]
        quats = []
        for i, pose in enumerate(requested):
            xyz, quat = _parse_pose(pose, f"check_path.poses[{i}]")
            violation = _workspace_violation(xyz)
            if violation is not None:
                raise TrajectoryIKError(
                    segment=i, sample=0, status=-9, status_name="SIM_WORKSPACE_LIMIT",
                    translation_error_m=0.0, rotation_error_rad=0.0,
                    reason=f"sim_workspace_limit: {violation}",
                )
            points.append(xyz)
            quats.append(quat)
        contacts = []
        for segment, (a, b) in enumerate(zip(points, points[1:])):
            n = max(1, int(np.linalg.norm(b - a) / MOVE_STEP_M))
            for i in range(1, n + 1):
                sample = a + (b - a) * (i / n)
                obstacle = world.obstacle_at(sample)
                if obstacle is not None:
                    contacts.append({
                        "segment": segment, "obstacle": obstacle,
                        "at_xyz": [float(v) for v in sample],
                    })
                    break
        planned = [[*p.tolist(), *quaternion_to_rpy(q).tolist()]
                   for p, q in zip([points[0], *points[1:]], [world.ee_quat_xyzw, *quats])]
        duration = float(sum(np.linalg.norm(b - a) for a, b in zip(points, points[1:])) / EE_SPEED_M_S)
        return {
            "path_check": {
                "accepted": True,
                "executed": False,
                "checks": ["workspace_bounds", "obstacle_intersection"],
                "collision_checked": True,
                "collision_note": "Simulated static-obstacle intersection only; not a full-body collision model.",
                "obstacle_contacts": contacts,
                "replanned_before_execution": True,
                "arms": {"left": {"planned_duration_s": duration,
                                  "planned_tcp_points_xyzrpy": planned}},
            }
        }

    def _gripper(self, target: float) -> dict[str, Any]:
        outcome = self.world.set_gripper(target)
        result = self.state()
        measured = float(self.world.gripper_measured)
        result["execution_feedback"] = {
            "gripper_target_normalized": float(target),
            "gripper_active_command_normalized": float(target),
            "gripper_measured_normalized": measured,
            "gripper_residual_normalized": abs(float(target) - measured),
            "gripper_torque_nm": outcome["torque_nm"],
            "motion": {"duration_s": GRIPPER_TRAVEL_S, "simulated": True},
        }
        return result

    def return_home(self) -> dict[str, Any]:
        world = self.world
        durations = []
        lift = np.array([world.ee[0], world.ee[1], max(world.ee[2], 0.35)])
        for target in (lift, HOME_XYZ):
            move = world.move_to(np.asarray(target, dtype=np.float64), DOWN_QUAT_XYZW)
            durations.append(move["duration_s"])
        world.gripper_command = 1.0
        if world.grasped is None:
            world.gripper_measured = 1.0
        result = self.state()
        result["home"] = {
            "source": "gpt_policy_sim.return_home",
            "time_scale": 1.0,
            "segment_durations_s": durations,
            "target_joint_positions_rad": [float(v) for v in world.mock_joints(HOME_XYZ)],
            "max_joint_residual_rad": 0.001,
            "gripper_target_normalized": 1.0,
            "settle": {"settled": True, "method": "simulated_synchronous", "window_s": 0.2},
        }
        return result

    # -------------------------------------------------------------- lifecycle
    def cancel(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def close(self) -> None:
        pass
