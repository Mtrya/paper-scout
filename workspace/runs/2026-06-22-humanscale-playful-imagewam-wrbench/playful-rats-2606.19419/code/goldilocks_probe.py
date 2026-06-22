"""
Goldilocks task-selection probe for RATS (Playful Agentic Robot Learning).

This script reimplements the analytical curiosity score from the paper
(Sec. 3.2 / Appendix A.1) and the public code in
`code/rats/rats/agents/curiosity_scoring.py`.

It does two things:

1. Reproduces the candidate-selection trace reported in Table 7 of the
   appendix (MolmoSpaces iteration 15).  The script computes
   N(tau) * F(tau) for the five candidates and confirms that the
   tissue-box lift wins because it sits on the competence frontier.

2. Simulates a 50-iteration play run on a small abstract task family.
   Starting from the same primitives and a few experimental learned
   skills reported in Table 5, the simulation shows how novelty decay
   and competence-frontier dynamics shape the distribution of proposed
   tasks over time.

Run with:
    python goldilocks_probe.py

Outputs are printed; a small JSON log (`goldilocks_sim_log.json`) is
written for further inspection.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Core formulas (mirrored from rats/agents/curiosity_scoring.py)
# ---------------------------------------------------------------------------

def wilson_lower_bound(success: int, total: int, z: float = 1.96) -> float:
    """Conservative reliability estimate used by RATS."""
    if total <= 0:
        return 0.0
    p = success / total
    denom = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return max(0.0, (center - margin) / denom)


def object_skill_novelty(candidate: dict[str, Any], counts: dict[tuple[str, str], int]) -> float:
    """Mean of 1/sqrt(N(o,s)+1) over object-skill pairs."""
    objects = [str(o).strip().lower() for o in candidate.get("objects", []) if str(o).strip()]
    skills = [str(s).strip().lower() for s in candidate.get("required_skills", []) if str(s).strip()]
    if not objects or not skills:
        return 0.5
    pairs = [(o, s) for o in objects for s in skills]
    vals = [1.0 / math.sqrt(counts.get(pair, 0) + 1.0) for pair in pairs]
    return sum(vals) / len(vals)


def competence(candidate: dict[str, Any], skill_lookup: dict[str, float]) -> float:
    """Mean Wilson-lower-bound reliability of required skills."""
    required = [str(s).strip() for s in candidate.get("required_skills", []) if str(s).strip()]
    if not required:
        return 0.5
    rels = []
    for name in required:
        r = skill_lookup.get(name)
        if r is None:
            r = 0.05  # missing skill default from the paper
        rels.append(max(0.0, min(1.0, r)))
    return sum(rels) / len(rels)


def frontier(competence: float) -> float:
    """Goldilocks parabola: peaks at 0.5, value 1.0."""
    c = max(0.0, min(1.0, competence))
    return 4.0 * c * (1.0 - c)


def score_candidate(
    candidate: dict[str, Any],
    counts: dict[tuple[str, str], int],
    skill_lookup: dict[str, float],
) -> dict[str, float]:
    """Return N, r_bar, F, and the product score N*F."""
    nov = object_skill_novelty(candidate, counts)
    comp = competence(candidate, skill_lookup)
    front = frontier(comp)
    return {
        "novelty": nov,
        "competence": comp,
        "frontier": front,
        "score": nov * front,
    }


# ---------------------------------------------------------------------------
# 1. Reproduce Table 7 candidate-selection trace
# ---------------------------------------------------------------------------

def table7_probe() -> None:
    """Check that the analytical ranker selects the tissue-box lift."""
    # Table 7 reports the following candidates at MolmoSpaces iteration 15.
    # The prompt metadata included the skills from Table 5; we encode the
    # Wilson-lower-bound reliability values explicitly derived in the text.
    skill_lookup = {
        "push_surface_inward": 0.9,          # primitive / learned close helpers
        "pull_surface_outward": 0.9,         # primitive / learned open helpers
        "execute_top_down_grasp_and_lift": wilson_lower_bound(5, 12),
        "place_in": 0.05,                    # missing skill default
        "pick": 0.05,                        # missing skill default
    }

    candidates = [
        {
            "language": "Push the top drawer of the white cabinet all the way closed.",
            "family": "Close",
            "objects": ["white_cabinet"],
            "required_skills": ["push_surface_inward"],
        },
        {
            "language": "Pull the partially open drawer out even more.",
            "family": "Open",
            "objects": ["white_cabinet"],
            "required_skills": ["pull_surface_outward"],
        },
        {
            "language": "Pick up the black cloth from the cabinet.",
            "family": "Pick",
            "objects": ["black_cloth"],
            "required_skills": ["pick"],
        },
        {
            "language": "Lift the brown tissue box straight up into the air.",
            "family": "Lift",
            "objects": ["brown_tissue_box"],
            "required_skills": ["execute_top_down_grasp_and_lift"],
        },
        {
            "language": "Put the black cloth inside the open drawer.",
            "family": "Place in",
            "objects": ["black_cloth", "white_cabinet"],
            "required_skills": ["pick", "place_in"],
        },
    ]

    # All object-skill pairs are new at iteration 15, so novelty is 1.0.
    counts: dict[tuple[str, str], int] = {}

    print("=" * 70)
    print("Table 7 candidate-selection reproduction")
    print("=" * 70)
    print(f"{'Family':<12} {'N':>6} {'r_bar':>8} {'F':>8} {'N*F':>8}")
    print("-" * 70)

    best = None
    best_score = -1.0
    for cand in candidates:
        if cand["family"] == "Pick":
            # The cloth pick was vetoed by bridge compatibility in the paper.
            print(f"{cand['family']:<12} {'—':>6} {'—':>8} {'—':>8} {'vetoed':>8}")
            continue
        s = score_candidate(cand, counts, skill_lookup)
        print(f"{cand['family']:<12} {s['novelty']:>6.4f} {s['competence']:>8.4f} "
              f"{s['frontier']:>8.4f} {s['score']:>8.4f}")
        if s["score"] > best_score:
            best_score = s["score"]
            best = cand["family"]

    print("-" * 70)
    print(f"Selected family: {best} (expected: Lift)")
    print()
    assert best == "Lift", "Tissue-box lift should win on the competence frontier"


# ---------------------------------------------------------------------------
# 2. Simulate 50 iterations of play
# ---------------------------------------------------------------------------

@dataclass
class SimSkill:
    name: str
    successes: int = 0
    uses: int = 0
    true_sr: float = 0.5

    def wilson(self) -> float:
        return wilson_lower_bound(self.successes, self.uses)


@dataclass
class SimTask:
    family: str
    objects: list[str]
    required_skills: list[str]


class PlaySimulator:
    """Tiny abstract simulator of RATS play-time task selection."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        # Initial learned skills roughly matching Table 5 / iteration 15.
        self.skills: dict[str, SimSkill] = {
            "execute_top_down_grasp_and_lift": SimSkill(
                "execute_top_down_grasp_and_lift", successes=5, uses=12, true_sr=0.55
            ),
            "push_surface_inward": SimSkill(
                "push_surface_inward", successes=2, uses=2, true_sr=0.9
            ),
            "pull_surface_outward": SimSkill(
                "pull_surface_outward", successes=1, uses=3, true_sr=0.7
            ),
            "place_in": SimSkill("place_in", successes=0, uses=0, true_sr=0.25),
            "pick": SimSkill("pick", successes=0, uses=0, true_sr=0.35),
        }
        self.object_skill_counts: dict[tuple[str, str], int] = {}
        self.log: list[dict[str, Any]] = []
        self.available_objects = [
            "tissue_box", "black_cloth", "white_cabinet", "drawer",
            "sandal", "toothpaste", "walkie_talkie", "metal_ring",
        ]
        self.families = {
            "Lift": ["execute_top_down_grasp_and_lift"],
            "Close": ["push_surface_inward"],
            "Open": ["pull_surface_outward"],
            "Place in": ["pick", "place_in"],
        }

    def skill_lookup(self) -> dict[str, float]:
        return {name: sk.wilson() for name, sk in self.skills.items()}

    def generate_candidates(self, k: int = 4) -> list[dict[str, Any]]:
        """Sample k random (family, object) candidates."""
        cands = []
        for _ in range(k):
            family = self.rng.choice(list(self.families.keys()))
            obj = self.rng.choice(self.available_objects)
            # Place-in tasks use two objects; others use one.
            if family == "Place in":
                other = self.rng.choice([o for o in self.available_objects if o != obj])
                objects = [obj, other]
            else:
                objects = [obj]
            cands.append({
                "family": family,
                "objects": objects,
                "required_skills": self.families[family],
            })
        return cands

    def select_task(self, candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, float]]:
        lookup = self.skill_lookup()
        scored = []
        for cand in candidates:
            s = score_candidate(cand, self.object_skill_counts, lookup)
            scored.append((cand, s))
        scored.sort(key=lambda x: x[1]["score"], reverse=True)
        return scored[0]

    def execute(self, task: dict[str, Any]) -> bool:
        """Simulate one execution attempt.  Success rate = mean true_sr."""
        srs = [self.skills[s].true_sr for s in task["required_skills"]]
        p = sum(srs) / len(srs)
        # Add a small competence-frontier bias: tasks near 0.5 are easier to learn.
        p += 0.1 * frontier(p)
        return self.rng.random() < max(0.05, min(0.95, p))

    def update(self, task: dict[str, Any], success: bool, iteration: int) -> None:
        for skill_name in task["required_skills"]:
            self.skills[skill_name].uses += 1
            if success:
                self.skills[skill_name].successes += 1
        for obj in task["objects"]:
            for skill_name in task["required_skills"]:
                self.object_skill_counts[(obj, skill_name)] = (
                    self.object_skill_counts.get((obj, skill_name), 0) + 1
                )
        self.log.append({
            "iteration": iteration,
            "family": task["family"],
            "objects": task["objects"],
            "skills": task["required_skills"],
            "success": success,
        })

    def run(self, n_iterations: int = 50) -> None:
        for it in range(1, n_iterations + 1):
            candidates = self.generate_candidates()
            task, score = self.select_task(candidates)
            success = self.execute(task)
            self.update(task, success, it)

    def report(self) -> None:
        print("=" * 70)
        print("50-iteration play simulation: task-family distribution")
        print("=" * 70)
        from collections import Counter
        bins = [self.log[i:i + 10] for i in range(0, len(self.log), 10)]
        for idx, bin_log in enumerate(bins, start=1):
            c = Counter(entry["family"] for entry in bin_log)
            print(f"  iterations {(idx - 1) * 10 + 1:>2}-{idx * 10:>2}: "
                  f"Lift={c['Lift']} Close={c['Close']} Open={c['Open']} PlaceIn={c['Place in']}")

        print()
        print("Final skill reliabilities (Wilson lower bound)")
        print("-" * 70)
        for name, sk in sorted(self.skills.items()):
            print(f"  {name:<40} {sk.successes}/{sk.uses}  "
                  f"sr={sk.successes / max(sk.uses, 1):.2f}  wilson={sk.wilson():.3f}")
        print()


def simulation_probe() -> None:
    rng = random.Random(260619419)
    sim = PlaySimulator(rng)
    sim.run(50)
    sim.report()
    out_path = os.path.join(os.path.dirname(__file__), "goldilocks_sim_log.json")
    with open(out_path, "w") as f:
        json.dump(sim.log, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    table7_probe()
    simulation_probe()
