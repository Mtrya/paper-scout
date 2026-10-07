"""Hybrid control channel: a scripted cerebellum proposes, the model reviews.

Mirrors the RoboDojo/RoboLab hybrid setting of arXiv:2609.38537 in the sim
backend: before every decision the low-level "cerebellum" (here the task's
scripted oracle expert, standing in for a learned policy such as pi-0.5)
proposes one ready-to-execute tool call; the model answers with either
``accept_proposal`` (the host executes the proposal verbatim) or its own tool
call (the proposal is discarded). The channel is injected as an append to the
observation text plus a tool entry appended to the catalog, so the official
gpt-policy files stay untouched.

Arms (proposal quality):
  hybrid-oracle     proposal = expert's own next action
  hybrid-noisy      proposal = expert action + per-waypoint xy Gaussian noise (sigma 3 cm)
  hybrid-biased     proposal = expert action transformed by a constant SE(2) error
                    (rotate 15 deg about the current TCP, translate +5 cm x)
  hybrid-biased-soft proposal = same but a pure +2 cm x translation (exploratory arm:
                    the smallest constant error still fatal on T2's 1.5 cm tolerance,
                    used to separate error magnitude from error plausibility)
  hybrid-degenerate proposal = a constant junk chunk aimed at a workspace corner
"""

from __future__ import annotations

import copy
import json
import math
import re
import time
from typing import Any, Callable, Protocol

import numpy as np

from gpt_policy.harness.models import AgentContext, AgentTurn

HYBRID_ARMS = ("hybrid-oracle", "hybrid-noisy", "hybrid-biased", "hybrid-biased-soft",
               "hybrid-degenerate")
ACCEPT_TOOL = "accept_proposal"

NOISE_SIGMA_M = 0.03
BIAS_TRANSLATION_M = 0.05
BIAS_ROTATION_DEG = 15.0
BIAS_SOFT_TRANSLATION_M = 0.02
BIAS_SOFT_ROTATION_DEG = 0.0
DEGENERATE_WAYPOINTS = ((0.18, 0.18, 0.22), (0.18, 0.18, 0.02))
DEGENERATE_NOTE = "策略 rollout:移到工作区角落"

_ENV_STEP = re.compile(r'"env_step"\s*:\s*(\d+)')


def accept_tool_entry() -> dict[str, Any]:
    """Catalog entry for the review tool (handler names the pseudo-host handler)."""
    return {
        "name": ACCEPT_TOOL,
        "handler": "policy.accept_proposal",
        "enabled": True,
        "modes": ["single", "bimanual"],
        "description": "Execute the cerebellum proposal shown in this observation exactly as written; use it instead of re-issuing the proposed tool call yourself.",
        "prompt": "No arguments are required. Select it when the cerebellum proposal shown in the current observation should run unchanged. The host executes the proposed call with the same validation, IK and feedback as any other tool call, and it counts as one decision.",
        "parameters": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "Optional 1-2 sentence Chinese justification for accepting (recorded with the decision).",
                }
            },
            "required": [],
            "additionalProperties": False,
        },
    }


def augment_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the tool catalog dict with the review tool appended."""
    result = copy.deepcopy(catalog)
    if any(tool.get("name") == ACCEPT_TOOL for tool in result["tools"]):
        return result
    result["tools"].append(accept_tool_entry())
    return result


def augment_output_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Add accept_proposal to the outer selection schema's name enum/arguments."""
    result = copy.deepcopy(schema)
    properties = result["properties"]
    names = properties["name"]["enum"]
    if ACCEPT_TOOL not in names:
        names.append(ACCEPT_TOOL)
    arguments = properties["arguments"]["anyOf"]
    accept_schema = {
        "type": "object",
        "properties": {"note": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        "required": ["note"],
        "additionalProperties": False,
    }
    if accept_schema not in arguments:
        arguments.append(accept_schema)
    return result


HYBRID_PROTOCOL_NOTES = """

HYBRID REVIEW PROTOCOL (this run; where these lines conflict with lines above, these win):
- Before each of your decisions, a learned low-level policy (the "cerebellum", in the spirit of pi-0.5) proposes one candidate tool call, computed from its own state estimate and its own rollout. It does not see your camera images. The proposal is appended to the observation under HYBRID CONTROL CHANNEL, as a ready-to-execute {"name": ..., "arguments": ...} object.
- Every turn you must still return exactly one tool selection, and you have exactly two options:
  * "accept_proposal": the host executes the proposed call exactly as written above, with the same validation, IK, execution feedback and budget accounting as any tool call you issue yourself.
  * Any other tool in this protocol: your own action replaces the proposal; the proposal is discarded and never executed.
- The proposal is a candidate, not a command, and not a claim about the scene. Judge it against the current observation and your own geometry, and replace it whenever you disagree; accepting a proposal executes it exactly, including any error it contains, and the host does not silently correct it.
- accept_proposal is a normal decision: it counts against the decision budget and its result arrives as previous_result like any other tool.
"""


def _env_step(observation: str, fallback: int) -> int:
    match = _ENV_STEP.search(observation)
    return int(match.group(1)) if match else fallback


def _map_pose(pose: dict[str, Any], fn: Callable[[np.ndarray], np.ndarray]) -> dict[str, Any]:
    values = pose.get("pose_xyzquat")
    if not isinstance(values, list) or len(values) != 7:
        return pose
    mapped = list(values)
    mapped[:3] = [float(v) for v in fn(np.asarray(values[:3], dtype=float))]
    return {**pose, "pose_xyzquat": mapped}


def map_waypoints(decision: dict[str, Any], fn: Callable[[np.ndarray], np.ndarray]) -> dict[str, Any]:
    """Copy a tool call with every commanded xyz waypoint mapped through fn (z preserved by fn)."""
    result = copy.deepcopy(decision)
    arguments = result.get("arguments")
    if not isinstance(arguments, dict):
        return result
    if isinstance(arguments.get("target"), dict):
        arguments["target"] = _map_pose(arguments["target"], fn)
    for key in ("poses",):
        if isinstance(arguments.get(key), list):
            arguments[key] = [
                _map_pose(pose, fn) if isinstance(pose, dict) else pose for pose in arguments[key]
            ]
    return result


class ProposalSource(Protocol):
    kind: str

    def next(self, world: Any) -> dict[str, Any]: ...


class OracleProposals:
    """Proposal = the expert's own next action."""

    kind = "oracle"

    def __init__(self, expert: Any) -> None:
        self.expert = expert

    @property
    def stage(self) -> str | None:
        return self.expert.stage

    def raw(self, world: Any) -> dict[str, Any]:
        # The stuck-guard is meant for the expert's own closed loop; in the
        # hybrid loop a stuck cerebellum must keep re-proposing its plan rather
        # than degrade into a give_up proposal the reviewer could accept.
        self.expert.attempts = 0
        return self.expert.act(world)

    def next(self, world: Any) -> dict[str, Any]:
        return self.raw(world)


class NoisyProposals(OracleProposals):
    """Proposal = expert waypoints plus per-waypoint xy Gaussian noise."""

    kind = "noisy"

    def __init__(self, expert: Any, seed: int) -> None:
        super().__init__(expert)
        self.seed = seed
        self.calls = 0

    def next(self, world: Any) -> dict[str, Any]:
        proposal = self.raw(world)
        rng = np.random.default_rng([self.seed, 7919, self.calls])
        self.calls += 1

        def fn(point: np.ndarray) -> np.ndarray:
            point = point.copy()
            point[:2] += rng.normal(0.0, NOISE_SIGMA_M, 2)
            return point

        return map_waypoints(proposal, fn)


class BiasedProposals(OracleProposals):
    """Proposal = expert action under a constant SE(2) error (rotation + translation).

    Every commanded xy waypoint p becomes  ee + R(theta) (p - ee) + (dx, 0),
    where ee is the current TCP xy: each proposed motion keeps its length but is
    rotated and shifted forward, i.e. a smooth, confident, systematically wrong
    rollout that converges to a fixed offset instead of the true target.
    """

    kind = "biased"

    def __init__(self, expert: Any, translation: float = BIAS_TRANSLATION_M,
                 rotation_deg: float = BIAS_ROTATION_DEG, kind: str = "biased") -> None:
        super().__init__(expert)
        self.kind = kind
        self.translation = translation
        angle = math.radians(rotation_deg)
        self.rotation = np.array([[math.cos(angle), -math.sin(angle)],
                                  [math.sin(angle), math.cos(angle)]])

    def next(self, world: Any) -> dict[str, Any]:
        proposal = self.raw(world)
        anchor = np.asarray(world.ee[:2], dtype=float)

        def fn(point: np.ndarray) -> np.ndarray:
            mapped = point.copy()
            mapped[:2] = (anchor + self.rotation @ (point[:2] - anchor)
                          + np.array([self.translation, 0.0]))
            return mapped

        return map_waypoints(proposal, fn)


class DegenerateProposals:
    """Proposal = one constant junk chunk aimed at a workspace corner."""

    kind = "degenerate"

    @property
    def stage(self) -> str | None:
        return None

    def __init__(self) -> None:
        self.decision = {
            "name": "move_eef_chunk",
            "arguments": {
                "poses": [{"pose_xyzquat": [x, y, z, 1.0, 0.0, 0.0, 0.0]} for x, y, z in DEGENERATE_WAYPOINTS],
                "note": DEGENERATE_NOTE,
            },
        }

    def next(self, world: Any) -> dict[str, Any]:
        return copy.deepcopy(self.decision)


def make_proposal_source(arm: str, expert: Any, seed: int) -> ProposalSource:
    if arm == "hybrid-oracle":
        return OracleProposals(expert)
    if arm == "hybrid-noisy":
        return NoisyProposals(expert, seed)
    if arm == "hybrid-biased":
        return BiasedProposals(expert)
    if arm == "hybrid-biased-soft":
        return BiasedProposals(expert, translation=BIAS_SOFT_TRANSLATION_M,
                               rotation_deg=BIAS_SOFT_ROTATION_DEG, kind="biased-soft")
    if arm == "hybrid-degenerate":
        return DegenerateProposals()
    raise ValueError(f"not a hybrid arm: {arm}")


def render_proposal(proposal: dict[str, Any], step: int) -> str:
    body = json.dumps({"name": proposal.get("name"), "arguments": proposal.get("arguments")},
                      ensure_ascii=False, separators=(",", ":"))
    return f"""HYBRID CONTROL CHANNEL (decision step {step})
Cerebellum proposal (from the low-level policy's own state estimate, not from your images):
{body}
Answer with exactly one tool call this turn:
- "accept_proposal": the host executes the proposal above exactly as written.
- any other tool: you replace it; the proposal is discarded and never executed.
The proposal is a candidate, not a command, and it can be wrong."""


class MockHybridReviewer:
    """Deterministic stand-in for the model in --mock runs.

    Alternates ACCEPT and REPLACE so both branches of the channel are exercised;
    a REPLACE emits the expert's own oracle action (kept separately from the
    arm's possibly corrupted proposal) so mock episodes still make progress.
    """

    def __init__(self, expert: Any, world: Any) -> None:
        self.expert = expert
        self.world = world
        self.calls = 0
        self.last_context_refresh = None
        self.last_decision_timing = None
        self.context: AgentContext | None = None

    def start(self, context: AgentContext) -> None:
        self.context = context

    def decide(self, turn: AgentTurn) -> dict[str, Any]:
        self.calls += 1
        if self.calls % 2 == 1:
            return {"name": ACCEPT_TOOL, "arguments": {},
                    "_wire": {"name": ACCEPT_TOOL, "arguments": {}}}
        self.expert.attempts = 0
        decision = self.expert.act(self.world)
        return {"name": decision["name"], "arguments": decision["arguments"], "_wire": decision}

    def close(self) -> None:
        pass


class HybridAgent:
    """AgentSession wrapper adding the cerebellum proposal/review channel."""

    def __init__(
        self,
        base: Any,
        expert: Any,
        world: Any,
        arm: str,
        *,
        seed: int = 0,
        record: Callable[[str, dict[str, Any]], None] | None = None,
        mock: bool = False,
    ) -> None:
        self.base = base if not mock else MockHybridReviewer(expert, world)
        self.proposals = make_proposal_source(arm, expert, seed)
        self.world = world
        self.arm = arm
        self.record = record
        self._fallback_step = 0

    def start(self, context: AgentContext) -> None:
        # The run host may already have appended the same notes for recording;
        # do not duplicate them in the live instructions.
        notes = "" if HYBRID_PROTOCOL_NOTES.strip() in context.instructions else HYBRID_PROTOCOL_NOTES
        augmented = AgentContext(
            instructions=context.instructions + notes,
            tools=context.tools + [
                {"type": "function", "function": {
                    "name": ACCEPT_TOOL,
                    "description": accept_tool_entry()["description"],
                    "parameters": {
                        "type": "object",
                        "properties": {"note": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
                        "required": ["note"],
                        "additionalProperties": False,
                    },
                }},
            ],
            output_schema=augment_output_schema(context.output_schema),
        )
        self.base.start(augmented)

    def decide(self, turn: AgentTurn) -> dict[str, Any]:
        step = _env_step(turn.observation, self._fallback_step)
        self._fallback_step = step + 1
        proposal = self.proposals.next(self.world)
        stage = getattr(self.proposals, "stage", None)
        text = render_proposal(proposal, step)
        started = time.perf_counter()
        decision = self.base.decide(AgentTurn(turn.observation + "\n\n" + text, turn.images, turn.content))
        decide_s = time.perf_counter() - started
        choice = "accept" if str(decision.get("name")) == ACCEPT_TOOL else "replace"
        final = dict(proposal) if choice == "accept" else dict(decision)
        final["_wire"] = {"name": final.get("name"), "arguments": final.get("arguments")}
        final["_hybrid"] = {"arm": self.arm, "choice": choice,
                            "proposal_kind": getattr(self.proposals, "kind", self.arm)}
        if self.record is not None:
            self.record("hybrid_step", {
                "step": step,
                "arm": self.arm,
                "proposal_kind": final["_hybrid"]["proposal_kind"],
                "proposal": {"name": proposal.get("name"), "arguments": proposal.get("arguments")},
                "stage": stage,
                "choice": choice,
                "model_action": {"name": decision.get("name"), "arguments": decision.get("arguments")},
                "model_wire": decision.get("_wire"),
                "executed_action": final.get("name"),
                "decide_s": decide_s,
                "proposal_text": text,
            })
        return final

    def close(self) -> None:
        self.base.close()

    # The run loop reads these through getattr() after every decision.
    @property
    def last_context_refresh(self):
        return getattr(self.base, "last_context_refresh", None)

    @property
    def last_decision_timing(self):
        return getattr(self.base, "last_decision_timing", None)
