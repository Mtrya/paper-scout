"""Run one GPT-Policy episode against the simulated backend.

Wires the official runtime pieces end to end: RunRecorder, ToolExecutor,
protocol.instructions/observation, run_loop — with either the real Codex
harness (gpt-6-astra via `codex app-server`) or a MockAgent that proxies the
task's scripted oracle expert through the same decide() interface.

Usage:
  python run_episode.py --task T1 --context none [--seed 0] [--mock]
  python run_episode.py --task T2 --context demo-action --demo-seed 1042
  python run_episode.py --task T1 --arm hybrid-biased --seed 0
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from gpt_policy.harness.models import AgentContext, AgentTurn
from gpt_policy.harness.protocol import instructions, observation, output_schema, tool_schemas
from gpt_policy.harness.usage import collect_usage
from gpt_policy.input.manifest import ImagePart, load_manifest
from gpt_policy.input.request import RunInput
from gpt_policy.input.demonstration import save_input
from gpt_policy.recording.trace import RunRecorder
from gpt_policy.runtime.console import RunConsole
from gpt_policy.runtime.runner import run_loop
from gpt_policy.settings import RuntimeConfig
from gpt_policy.tools.catalog import load_tool_catalog
from gpt_policy.tools.runtime import ToolExecutor

from demo_record import build_and_prepare
from hybrid_agent import HYBRID_ARMS, HYBRID_PROTOCOL_NOTES, HybridAgent, augment_catalog
from sim_render import SimRenderer
from sim_world import SimRobot
from tasks import TASKS

HERE = Path(__file__).resolve().parent
MODEL = "gpt-6-astra"
CODEX_BIN = shutil.which("codex") or "/home/deneb/.local/bin/codex"

SIM_NOTES = """
SIMULATION BACKEND NOTES (this run uses a simulated robot; where these lines conflict with hardware-specific lines above, these win):
- The arm is simulated as a point TCP with idealized orientation tracking; joint telemetry is a smooth mock kinematic mapping. Trust tcp_pose feedback and the camera images, not joint values.
- Cameras: 'top' (overhead; image up = world +x forward, image left = world +y) and 'front' (side view from -y; image right = world +x, image up = world +z). There are NO wrist cameras in this simulation; locate_point with camera 'left' is rejected.
- Workspace limits are hard: x,y within [0.15, 0.85] m, z within [0, 0.45] m; targets outside are rejected exactly like IK failures.
- Motions stop at first contact with static obstacles; execution_feedback.blocked then names the obstacle and the contact point. check_path additionally reports static-obstacle intersections under path_check.obstacle_contacts (a real IK-only check would not).
- Gripper physics: when the gripper closes on an object, measured opening stays above the commanded closure and torque rises, as on real hardware. A fully-closed reading with low torque means nothing was grasped.
"""


class SimCameras:
    """CameraSet analogue: two fixed matplotlib views of the world."""

    def __init__(self, renderer: SimRenderer) -> None:
        self.renderer = renderer

    def capture(self) -> dict:
        return {view: self.renderer.capture(view) for view in ("top", "front")}

    def describe(self, images=None) -> list[dict]:
        context = []
        for view in ("top", "front"):
            item = {
                "name": view,
                "device": f"sim://{view}",
                "format": "MJPG (simulated matplotlib render)",
                "width": self.renderer.width,
                "height": self.renderer.height,
            }
            if images is not None and view in images:
                item["captured_at"] = str(images[view].captured_at)
            context.append(item)
        return context

    def close(self) -> None:
        self.renderer.close()


class SimVideo:
    """RunVideo analogue without background threads: render fresh on snapshot()."""

    def __init__(self, cameras: SimCameras) -> None:
        self.cameras = cameras
        self._snapshots = 0
        self._started_at = time.time()
        self._stopped = False

    def snapshot(self, after: float | None = None, timeout: float | None = None) -> dict:
        if self._stopped:
            raise RuntimeError("相机采集已停止")
        images = self.cameras.capture()
        self._snapshots += 1
        if after is not None and min(image.captured_at for image in images.values()) <= after:
            # The two matplotlib renders practically guarantee freshness; this
            # guard only matters on pathologically fast machines.
            time.sleep(0.002)
            images = self.cameras.capture()
        return images

    @property
    def details(self) -> dict:
        return {
            "state": "stopped" if self._stopped else "recording",
            "codec": "per-step JPEG frames in frames/ (sim backend writes no mp4)",
            "views": ["top", "front"],
            "snapshots": self._snapshots,
            "started_at_s": self._started_at,
        }

    def stop(self) -> dict:
        self._stopped = True
        return self.details


class SimLocalizer:
    """PixelLocalizer analogue for the orthographic-ish top view.

    Mirrors the real tool's single-observation behaviour: a camera ray, plus
    base-frame rays for fixed cameras; never a metric position from one image.
    Unknown cameras raise the same ValueError the real localizer raises.
    """

    def __init__(self, renderer: SimRenderer) -> None:
        self.renderer = renderer

    def locate(self, arguments: dict, state: dict, history=None) -> dict:
        name = str(arguments.get("camera", ""))
        pixel = arguments.get("pixel_xy")
        if not isinstance(pixel, list) or len(pixel) != 2:
            raise ValueError("pixel_xy 必须是有限的 [x, y]")
        if name != "top":
            raise ValueError(f"没有 {name} 相机的有效内参")
        world_x, world_y = self.renderer.top_pixel_to_world(pixel)
        return {
            "camera": name,
            "pixel_xy": [float(pixel[0]), float(pixel[1])],
            "ray_camera_xyz": [0.0, 0.0, -1.0],
            "metric_position_available": False,
            "explanation": "单张 RGB 图像只确定相机射线。（模拟顶视为近似正交投影：射线竖直向下，桌面交点可直接读出。）",
            "base_frame_rays": {
                "left": {
                    "frame": "left_base_link",
                    "ray_origin_base_xyz": [world_x, world_y, 1.20],
                    "ray_direction_base_xyz": [0.0, 0.0, -1.0],
                }
            },
            "sim_table_intersection_xyz": [world_x, world_y, 0.0],
        }


class MockAgent:
    """AgentSession that proxies the task's scripted oracle expert."""

    def __init__(self, expert, world) -> None:
        self.expert = expert
        self.world = world
        self.context: AgentContext | None = None
        self.last_context_refresh = None
        self.last_decision_timing = None

    def start(self, context: AgentContext) -> None:
        self.context = context

    def decide(self, turn: AgentTurn) -> dict:
        if self.context is None:
            raise RuntimeError("Agent session has not been started")
        decision = self.expert.act(self.world)
        return {"name": decision["name"], "arguments": decision["arguments"], "_wire": decision}

    def close(self) -> None:
        pass


def build_settings(arm: str = "none") -> dict:
    catalog = json.loads((HERE / "tools.sim.json").read_text(encoding="utf-8"))
    if arm in HYBRID_ARMS:
        catalog = augment_catalog(catalog)
    return {
        "machine": "gpt-policy-sim",
        "agent": "codex",
        "runtime": {"robot_model": "SIM-X5", "interface": "sim0", "right_interface": None},
        "scene": {"safety_notes": [
            "Simulated tabletop world: workspace x,y in [0.15, 0.85] m and z in [0, 0.45] m; table surface at z=0.",
            "Available cameras are 'top' (overhead) and 'front' (side); no wrist cameras exist in this simulation.",
        ]},
        "tool_catalog": catalog,
    }


def build_run_input(task, args, record_dir: Path) -> RunInput:
    context = args.context
    if context == "none":
        return RunInput(task.instruction, MODEL)
    if context == "target":
        goal_world = task.build_world(args.seed)
        task.goal_overlay(goal_world)
        renderer = SimRenderer(goal_world)
        image = renderer.capture("top")
        renderer.close()
        path = record_dir / "target-top.jpg"
        path.write_bytes(image.data)
        return RunInput(task.instruction, MODEL, (ImagePart(
            path, label="目标状态参考图：当前场景下任务完成时的俯视布局（模拟渲染）。其余条件不变。"),))
    mode = {"demo-video": "video", "demo-action": "video+action"}[context]
    prepared = build_and_prepare(task.key, args.demo_seed, mode)
    manifest = load_manifest(prepared / "input.json")
    return RunInput(task.instruction, MODEL, manifest.content, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--context", choices=("none", "target", "demo-video", "demo-action"), default="none")
    parser.add_argument("--arm", choices=("none", "direct") + HYBRID_ARMS, default="none",
                        help="control channel: none/direct = model only; hybrid-* = cerebellum proposal reviewed by the model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo-seed", type=int, default=1042)
    parser.add_argument("--max-decisions", type=int, default=60)
    parser.add_argument("--mock", action="store_true", help="use the scripted oracle expert instead of Codex")
    parser.add_argument("--out", type=Path, default=HERE / "runs")
    args = parser.parse_args()

    task = TASKS[args.task]
    world = task.build_world(args.seed)
    robot = SimRobot(world)
    renderer = SimRenderer(world)
    cameras = SimCameras(renderer)
    video = SimVideo(cameras)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = args.arm if args.arm != "none" else args.context
    record_dir = args.out / f"{task.key}_{label}_s{args.seed}_{timestamp}"
    recorder = RunRecorder(record_dir, {
        "instruction": task.instruction,
        "task_name": f"sim-{task.key.lower()}-{task.name}",
        "max_decisions": args.max_decisions,
        "model": MODEL,
        "agent": {"profile": "mock-expert" if args.mock else "codex", "type": "mock" if args.mock else "codex",
                  "model": MODEL},
        "input": {"context": args.context, "seed": args.seed, "demo_seed": args.demo_seed, "arm": args.arm},
        "robot_model": "SIM-X5",
        "interface": "sim0",
        "right_interface": None,
        "sim_backend": {"world_seed": args.seed, "task": task.key, "context": args.context, "arm": args.arm},
    })

    display = RunConsole()
    settings = build_settings()
    catalog = load_tool_catalog(settings)
    # Host-side tool set stays unmodified: a leaked accept_proposal is a recorded
    # tool rejection, never a silent extra handler. Only the agent sees the
    # augmented catalog (review tool + protocol notes).
    agent_settings = build_settings(args.arm)
    agent_catalog = load_tool_catalog(agent_settings) if args.arm in HYBRID_ARMS else catalog
    localizer = SimLocalizer(renderer)
    tool_executor = ToolExecutor(catalog, ("left",), robot, localizer)
    runtime = RuntimeConfig(
        robot_model="SIM-X5", interface="sim0", right_interface="",
        gripper_open_readout=-3.4, camera_width=renderer.width, camera_height=renderer.height,
        convert_camera_images_to_jpeg=False, camera_jpeg_quality=80, trajectory_hz=100.0,
        task_name=f"sim-{task.key.lower()}", record_dir=record_dir, max_decisions=args.max_decisions,
    )

    agent = None
    status = "failed"
    failure = None
    # The ledger must stay active until recorder.close() has written status.json
    # and attached usage.json, so it is entered/exited explicitly instead of
    # wrapping only run_loop.
    usage_ledger = collect_usage()
    usage_ledger.__enter__()
    try:
        run_input = build_run_input(task, args, record_dir)
        save_input(run_input, record_dir)
        if run_input.manifest is not None:
            recorder.write("input_manifest", run_input.manifest.record())
        base_instructions = instructions(
            "SIM-X5", "sim0", robot.dof, ("left",), settings, agent_catalog,
            task_instruction=task.instruction,
        ) + SIM_NOTES
        if args.arm in HYBRID_ARMS:
            base_instructions += HYBRID_PROTOCOL_NOTES
        base_instructions += (
            f"\nThis task allows at most {runtime.max_decisions} decisions; observation.extra.env_step starts at 0. "
            "Each tool selection counts, including rejected arguments or IK failures. "
            "When the budget is exhausted, the host stops model calls, returns home and saves the recording."
        )
        schema = output_schema(robot.dof, ("left",), agent_catalog)
        function_catalog = tool_schemas(robot.dof, ("left",), agent_catalog)
        recorder.write("protocol", {
            "output_schema": schema,
            "tool_catalog": function_catalog,
            "base_instructions": base_instructions,
            "control_prompt_profile": "default",
            "arm": args.arm,
            "cameras": cameras.describe(),
        })
        expert = task.expert(world)
        if args.mock:
            agent = MockAgent(expert, world)
        else:
            from gpt_policy.harness.config import AgentConfig
            from gpt_policy.harness.providers.codex import CodexSession
            config = AgentConfig(type="codex", model=MODEL, executable=CODEX_BIN,
                                 effort="medium", live_image_window=8)
            agent = CodexSession(config, MODEL, False, 80)
        if args.arm in HYBRID_ARMS:
            agent = HybridAgent(agent, expert, world, args.arm, seed=args.seed,
                                record=recorder.write, mock=args.mock)
        agent.start(AgentContext(base_instructions, function_catalog, schema))
        status = run_loop(runtime, run_input, robot, cameras, video, agent,
                          tool_executor, recorder, observation, display=display)
    except KeyboardInterrupt:
        status = "interrupted"
        recorder.write("interrupted", {"trigger": "ctrl_c"})
    except Exception as exc:
        failure = repr(exc)
        recorder.write("run_error", {"error": failure})
        raise
    finally:
        success, detail = task.success(world)
        recorder.write("sim_evaluation", {
            "success": success, "detail": detail,
            "sim_time_s": world.time_s, "world_events": world.events[-40:],
        })
        # Ground-truth label: the sim, not the model's conclusion, decides success.
        recorder.task_status = "completed" if success else "failed"
        try:
            recorder.set_recording(video.stop())
        except Exception as exc:
            recorder.write("recording_error", {"error": repr(exc)})
        if agent is not None:
            try:
                agent.close()
            except Exception as exc:
                recorder.write("cleanup_error", {"resource": "agent", "error": repr(exc)})
        cameras.close()
        recorder.write("execution_finished", {"status": status, "error": failure})
        final_directory = recorder.close(status, failure)
        usage_ledger.__exit__(None, None, None)
        display.finished(status, final_directory,
                         task_status="completed" if success else "failed", error=failure)
        print(f"[sim] ground-truth success={success} ({detail})")
        print(f"[sim] recording: {final_directory}")
    if status == "failed":
        raise RuntimeError(failure)


if __name__ == "__main__":
    main()
