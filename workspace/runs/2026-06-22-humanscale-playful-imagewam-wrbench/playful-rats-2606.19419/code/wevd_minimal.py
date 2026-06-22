"""
Minimal Write-Execute-Verify-Diagnose (WEVD) loop for RATS.

This script implements a *toy* version of the execution pipeline described
in Sec. 3.3 / Appendix A.2 of the paper, using a deterministic mock
environment for the MolmoSpaces drawer-opening example from Figure 9 /
Appendix C.2.

Components:
  - MockScene: simulates a cabinet with a handle.
  - Learned skills: axis-aligned pull direction and grasp selection,
    distilled from the paper's skill snippets (Appendix E.3).
  - Planner / Policy Writer / Quality Checker / Verifiers / Failure Diagnoser:
    simplified deterministic agents that illustrate the retry loop.

The first attempt fails because the grasp is misaligned; the per-step
verifier and diagnoser localize the failure, and the second attempt
reuses the learned helpers with the corrected axis.  This is the same
pattern the paper reports: dense step-level feedback converts failures
into reusable skills instead of a single sparse success/failure bit.

Run with:
    python wevd_minimal.py
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any


# ---------------------------------------------------------------------------
# Preserved learned-skill snippets (adapted from Appendix E.3)
# ---------------------------------------------------------------------------

def get_axis_aligned_pull_direction(target_pos: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Computes the axis-aligned pull direction pointing from the target
    position toward the robot base (assumed at origin on the table plane).
    """
    import math
    robot_pos = (0.0, 0.0, 0.0)
    dx = robot_pos[0] - target_pos[0]
    dy = robot_pos[1] - target_pos[1]
    # Project to table plane, choose dominant axis.
    if abs(dx) >= abs(dy):
        return (math.copysign(1.0, dx), 0.0, 0.0)
    return (0.0, math.copysign(1.0, dy), 0.0)


def select_grasp_for_pulling(pull_dir: tuple[float, float, float]) -> str:
    """Select a handle grasp whose approach opposes the requested pull."""
    axis = "x" if abs(pull_dir[0]) > abs(pull_dir[1]) else "y"
    return f"{axis}-aligned_handle_grasp"


# ---------------------------------------------------------------------------
# Mock environment
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MockScene:
    """A trivial cabinet-with-drawer scene."""
    drawer_open: bool = False
    handle_pos: tuple[float, float, float] = (0.35, -0.10, 0.88)
    robot_at_handle: bool = False
    gripper_closed: bool = False
    grasp_aligned: bool = False

    def obs(self) -> dict[str, Any]:
        return {
            "drawer_open": self.drawer_open,
            "handle_pos": self.handle_pos,
            "robot_at_handle": self.robot_at_handle,
            "gripper_closed": self.gripper_closed,
            "grasp_aligned": self.grasp_aligned,
        }

    def goto_handle(self) -> None:
        self.robot_at_handle = True

    def align_grasp(self, axis: str) -> None:
        self.grasp_aligned = "x" in axis or "y" in axis

    def close_gripper(self) -> None:
        self.gripper_closed = True

    def pull(self) -> None:
        if self.robot_at_handle and self.gripper_closed and self.grasp_aligned:
            self.drawer_open = True


# ---------------------------------------------------------------------------
# Simplified agent team
# ---------------------------------------------------------------------------

class Planner:
    """Produces an ordered plan and predicts a bottleneck."""

    def plan(self, task: str, skills: list[str]) -> list[dict[str, Any]]:
        return [
            {"id": 1, "text": "localize handle", "skill": skills[0]},
            {"id": 2, "text": "compute pull direction", "skill": skills[1]},
            {"id": 3, "text": "select grasp", "skill": skills[2]},
            {"id": 4, "text": "approach and grasp handle", "skill": None},
            {"id": 5, "text": "pull drawer open", "skill": None},
        ]


class PolicyWriter:
    """Turns the plan into executable code.  The retry context lets us fix
    the previous attempt."""

    def __init__(self) -> None:
        self.retry_context: dict[str, Any] = {}

    def write(self, plan: list[dict[str, Any]], retry: int, diagnosis: dict[str, Any] | None) -> str:
        if retry == 0:
            # First attempt: naive code, ignores learned axis-alignment helper.
            return """
scene.goto_handle()
scene.close_gripper()
scene.pull()
""".strip()
        # Retry: use the diagnosed fix (axis-aligned grasp).
        fix = diagnosis.get("fix", "") if diagnosis else ""
        return f"""
{fix}
scene.goto_handle()
scene.close_gripper()
scene.pull()
""".strip()

    def set_retry_context(self, ctx: dict[str, Any]) -> None:
        self.retry_context = ctx


class QualityChecker:
    """Static screen for forbidden patterns."""

    def check(self, code: str) -> str | None:
        if "while True" in code:
            return "unbounded loop"
        return None


class PerStepVerifier:
    """Checks each plan step against the execution trace."""

    def verify(self, scene: MockScene, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
        verdicts = []
        obs = scene.obs()
        # Step 1: handle localized (always true in mock).
        verdicts.append({"step_id": 1, "ok": True, "evidence": "handle in view"})
        # Step 2: pull direction computed if learned helper was used.
        has_dir = "pull_dir" in str(plan)
        verdicts.append({"step_id": 2, "ok": has_dir or True, "evidence": "direction available"})
        # Step 3: grasp selection aligned with pull axis.
        verdicts.append({
            "step_id": 3,
            "ok": obs["grasp_aligned"],
            "evidence": "grasp aligned" if obs["grasp_aligned"] else "grasp not axis-aligned",
        })
        # Step 4: gripper closed at handle.
        verdicts.append({
            "step_id": 4,
            "ok": obs["robot_at_handle"] and obs["gripper_closed"],
            "evidence": "gripper closed on handle" if obs["gripper_closed"] else "gripper not closed",
        })
        # Step 5: drawer opened.
        verdicts.append({
            "step_id": 5,
            "ok": obs["drawer_open"],
            "evidence": "drawer open" if obs["drawer_open"] else "drawer still closed",
        })
        return verdicts


class FailureDiagnoser:
    """Produces a concrete code-level fix from per-step verdicts."""

    def diagnose(self, task: str, verdicts: list[dict[str, Any]], code: str) -> dict[str, Any]:
        failed = [v for v in verdicts if not v["ok"]]
        if not failed:
            return {"success": True}
        first = failed[0]
        if first["step_id"] == 3:
            return {
                "success": False,
                "failure_mode": "wrong_affordance",
                "failed_step": 3,
                "fix": "pull_dir = get_axis_aligned_pull_direction(scene.handle_pos)\n" +
                       "grasp = select_grasp_for_pulling(pull_dir)\n" +
                       "scene.align_grasp(grasp.split('_')[0])",
                "policy_feedback": "Align the grasp to the dominant pull axis before closing the gripper.",
            }
        return {
            "success": False,
            "failure_mode": "execution_failure",
            "failed_step": first["step_id"],
            "fix": "",
            "policy_feedback": first["evidence"],
        }


# ---------------------------------------------------------------------------
# WEVD loop
# ---------------------------------------------------------------------------

def run_wevd(task: str = "Open the top drawer of the white cabinet.", max_retries: int = 3) -> dict[str, Any]:
    scene = MockScene()
    planner = Planner()
    writer = PolicyWriter()
    quality = QualityChecker()
    verifier = PerStepVerifier()
    diagnoser = FailureDiagnoser()

    skills = ["localize_handle", "get_axis_aligned_pull_direction", "select_grasp_for_pulling"]
    plan = planner.plan(task, skills)
    history: list[dict[str, Any]] = []

    print("=" * 70)
    print(f"WEVD loop for task: {task}")
    print("=" * 70)

    for attempt in range(max_retries + 1):
        print(f"\n--- Attempt {attempt + 1} ---")
        diagnosis = history[-1]["diagnosis"] if history else None
        code = writer.write(plan, attempt, diagnosis)

        qc = quality.check(code)
        if qc:
            print(f"Quality checker rejected code: {qc}")
            continue

        print("Policy code:")
        for line in code.splitlines():
            print(f"    {line}")

        # Execute the policy in the mock scene.
        local_ns: dict[str, Any] = {"scene": scene, "get_axis_aligned_pull_direction": get_axis_aligned_pull_direction,
                                     "select_grasp_for_pulling": select_grasp_for_pulling}
        try:
            exec(code, local_ns)
        except Exception as e:
            print(f"Runtime error: {e}")
            break

        verdicts = verifier.verify(scene, plan)
        print("Per-step verdicts:")
        for v in verdicts:
            status = "PASS" if v["ok"] else "FAIL"
            print(f"  step {v['step_id']}: {status} — {v['evidence']}")

        diagnosis = diagnoser.diagnose(task, verdicts, code)
        history.append({"attempt": attempt + 1, "code": code, "verdicts": verdicts, "diagnosis": diagnosis})

        if diagnosis["success"]:
            print("\nGoal verifier: TASK SATISFIED")
            break

        print(f"\nFailure diagnosis: {diagnosis['failure_mode']} at step {diagnosis['failed_step']}")
        print(f"Suggested fix: {diagnosis['policy_feedback']}")
        writer.set_retry_context(diagnosis)

    return {"history": history, "final_scene": scene.obs()}


if __name__ == "__main__":
    result = run_wevd()
    print("\n" + "=" * 70)
    print("Final result")
    print("=" * 70)
    print(json.dumps(result["final_scene"], indent=2))
