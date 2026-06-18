#!/usr/bin/env python3
"""
OmniAgent probe: tiny active-perception POMDP + TAURA advantage rescaling.

This script demonstrates two core ideas from "Native Active Perception as
Reasoning for Omni-Modal Understanding" (OmniAgent, arXiv 2606.19341):

1. Observation-Thought-Action (OTA) loop as strict information distillation.
   A simple LLM-style agent is dropped into a 1-D hidden-target POMDP. It can
   *look* at a window, *think*, and *answer*. Raw observations are discarded
   after each turn; only textual observations persist in memory.

2. TAURA (Turn-aware Adaptive Uncertainty Rescaled Advantage).
   We synthesise turn-level entropy and rewards for a small group of
   trajectories and compare vanilla GRPO (uniform advantage) with TAURA
   (entropy-rescaled advantage). TAURA up-weights high-entropy discovery turns.

Run with:
    python omniagent_probe.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Part 1: Tiny POMDP active-perception environment
# ---------------------------------------------------------------------------

@dataclass
class TargetPOMDP:
    """A 1-D search POMDP: hidden target, noisy observations, query-driven."""

    target: float = field(default_factory=lambda: random.uniform(0.0, 100.0))
    noise_std: float = 12.0
    max_steps: int = 6

    def observe(self, window: Tuple[float, float], num_samples: int) -> List[str]:
        """Return noisy sample readings inside [start, end]."""
        start, end = window
        if end <= start:
            raise ValueError("window end must be > start")
        pts = np.linspace(start, end, num=num_samples)
        readings = []
        for p in pts:
            signal = np.exp(-0.5 * ((p - self.target) / self.noise_std) ** 2)
            signal += random.gauss(0.0, 0.05)
            readings.append(f"{p:.1f}s: signal={signal:.2f}")
        return readings

    def reward(self, guess: float) -> float:
        """Outcome reward: 1 within 5 units, linear decay otherwise."""
        err = abs(guess - self.target)
        return max(0.0, 1.0 - err / 20.0)


@dataclass
class ActivePerceptionAgent:
    """Hand-written OTA agent with textual memory and action grammar."""

    max_steps: int = 6
    memory: List[str] = field(default_factory=list)
    step: int = 0

    # State of the belief (internal, not exposed to env)
    belief_lo: float = 0.0
    belief_hi: float = 100.0

    def reset(self, query: str) -> None:
        self.memory = [f"Query: {query}", "Belief: target is somewhere in [0.0, 100.0]."]
        self.step = 0
        self.belief_lo, self.belief_hi = 0.0, 100.0

    def act(self, observation: List[str] | None) -> dict:
        """Generate an Observation-Thought-Action triplet."""
        self.step += 1

        if observation is not None:
            obs_text = "; ".join(observation)
            self.memory.append(f"Observation {self.step}: {obs_text}")

        # Simple heuristic: narrow the belief around the highest signal.
        if observation is not None:
            best_p = None
            best_s = -1.0
            for line in observation:
                # parse "12.3s: signal=0.45"
                parts = line.split(":")
                p = float(parts[0].replace("s", ""))
                s = float(parts[1].split("=")[1])
                if s > best_s:
                    best_s, best_p = s, p
            if best_p is not None:
                width = (self.belief_hi - self.belief_lo) / 2.5
                self.belief_lo = max(0.0, best_p - width / 2)
                self.belief_hi = min(100.0, best_p + width / 2)

        thought = (
            f"Step {self.step}/{self.max_steps}. Current belief: "
            f"[{self.belief_lo:.1f}, {self.belief_hi:.1f}]. "
        )

        if self.step < self.max_steps:
            n_samples = 4
            action = {
                "type": "look",
                "start": round(self.belief_lo, 1),
                "end": round(self.belief_hi, 1),
                "num": n_samples,
            }
            thought += f"I will look more closely in this window."
        else:
            guess = (self.belief_lo + self.belief_hi) / 2
            action = {"type": "answer", "content": round(guess, 1)}
            thought += f"Enough evidence; I will answer {guess:.1f}."

        self.memory.append(f"Thought {self.step}: {thought}")
        self.memory.append(f"Action {self.step}: {json.dumps(action)}")
        return {"observation": "; ".join(observation) if observation else "",
                "thought": thought,
                "action": action}


def run_active_perception_episode(seed: int | None = None, verbose: bool = True) -> float:
    """Run one episode of the tiny OTA loop and return the outcome reward."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = TargetPOMDP()
    agent = ActivePerceptionAgent(max_steps=env.max_steps)
    agent.reset("Where is the hidden target? Answer with a single number.")

    observation = None
    reward = 0.0
    if verbose:
        print("=" * 60)
        print("OmniAgent-style tiny active-perception episode")
        print("=" * 60)
    for _ in range(env.max_steps):
        ota = agent.act(observation)
        if verbose:
            print(f"\nTurn {agent.step}")
            print(f"  Observation : {ota['observation'][:120]}...")
            print(f"  Thought     : {ota['thought']}")
            print(f"  Action      : {ota['action']}")

        action = ota["action"]
        if action["type"] == "look":
            observation = env.observe((action["start"], action["end"]), action["num"])
        elif action["type"] == "answer":
            reward = env.reward(action["content"])
            if verbose:
                print(f"\n  Final guess: {action['content']} | target: {env.target:.1f} | reward: {reward:.3f}")
            break
    return reward


# ---------------------------------------------------------------------------
# Part 2: TAURA advantage rescaling on synthetic turn-level data
# ---------------------------------------------------------------------------

def synthesise_trajectories(
    n_queries: int = 2,
    n_rollouts_per_query: int = 4,
    max_turns: int = 5,
    seed: int = 42,
) -> Tuple[List[dict], List[float], List[float]]:
    """
    Build synthetic turn-level records.

    Each trajectory has a final reward and each turn has a mean token entropy.
    We intentionally give the *discovery* turn (the turn that first narrows the
    search) higher entropy, mimicking the paper's fork-step observation.

    Returns:
        records: list of turn records with keys uid, traj_uid, turn, reward, entropy.
        rewards: per-trajectory final rewards.
        entropies: per-trajectory mean entropies.
    """
    random.seed(seed)
    np.random.seed(seed)
    records = []
    traj_rewards = []
    traj_entropies = []

    for q in range(n_queries):
        for r in range(n_rollouts_per_query):
            traj_uid = f"q{q}_r{r}"
            # Random final reward, with one outlier per query.
            final_reward = random.uniform(0.2, 0.9)
            if r == 0:
                final_reward = 0.95  # best rollout
            if r == n_rollouts_per_query - 1:
                final_reward = 0.15  # worst rollout

            # Build turn entropies: one high-entropy discovery turn, others lower.
            n_turns = random.randint(3, max_turns)
            base = random.uniform(0.25, 0.45)
            turn_entropies = [base + random.gauss(0, 0.03) for _ in range(n_turns)]
            discovery_turn = random.randint(1, n_turns - 1)
            turn_entropies[discovery_turn] += random.uniform(0.35, 0.55)

            for t, ent in enumerate(turn_entropies):
                records.append({
                    "uid": f"q{q}",
                    "traj_uid": traj_uid,
                    "turn": t,
                    "final_reward": final_reward,
                    "entropy": ent,
                })

            traj_rewards.append(final_reward)
            traj_entropies.append(np.mean(turn_entropies))

    return records, traj_rewards, traj_entropies


def vanilla_grpo_advantages(records: List[dict]) -> List[float]:
    """Broadcast trajectory-level normalized advantage to every turn uniformly."""
    from collections import defaultdict
    groups = defaultdict(list)
    for rec in records:
        groups[rec["uid"]].append(rec)

    advs = []
    for rec in records:
        g = groups[rec["uid"]]
        rewards = [r["final_reward"] for r in g]
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-6
        adv = (rec["final_reward"] - mean_r) / std_r
        advs.append(adv)
    return advs


def taura_advantages(records: List[dict]) -> List[float]:
    """
    TAURA: rescale trajectory-level advantage by turn entropy.

    w_{i,k} = H_{i,k} / mean(H over all turns in the query group).
    A_hat_{i,k} = A_i * w_{i,k}.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for rec in records:
        groups[rec["uid"]].append(rec)

    # Global group mean entropy over all turns.
    group_mean_entropy = {}
    for uid, g in groups.items():
        group_mean_entropy[uid] = np.mean([r["entropy"] for r in g])

    advs = []
    for rec in records:
        g = groups[rec["uid"]]
        rewards = [r["final_reward"] for r in g]
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-6
        a_i = (rec["final_reward"] - mean_r) / std_r
        w = rec["entropy"] / (group_mean_entropy[rec["uid"]] + 1e-8)
        advs.append(a_i * w)
    return advs


def demo_taura(verbose: bool = True) -> dict:
    """Compare vanilla GRPO and TAURA on synthetic turn-level data."""
    records, rewards, _ = synthesise_trajectories()
    adv_vanilla = vanilla_grpo_advantages(records)
    adv_taura = taura_advantages(records)

    if verbose:
        print("\n" + "=" * 60)
        print("TAURA vs. vanilla GRPO on synthetic turn-level rewards")
        print("=" * 60)
        print(f"{'uid':6} {'turn':5} {'reward':7} {'entropy':8} "
              f"{'GRPO_A':9} {'TAURA_A':9} {'weight':7}")
        print("-" * 60)
        for rec, av, at in zip(records, adv_vanilla, adv_taura):
            w = rec["entropy"] / (np.mean([r["entropy"] for r in records if r["uid"] == rec["uid"]]) + 1e-8)
            print(f"{rec['uid']:6} {rec['turn']:5} {rec['final_reward']:7.2f} "
                  f"{rec['entropy']:8.3f} {av:9.3f} {at:9.3f} {w:7.3f}")

    # Quantify how much more weight high-entropy discovery turns receive.
    median_entropy = np.median([r["entropy"] for r in records])
    high_ent_idx = [i for i, r in enumerate(records) if r["entropy"] > median_entropy]
    low_ent_idx = [i for i, r in enumerate(records) if r["entropy"] <= median_entropy]

    def mean_abs(adv, idx):
        return np.mean([abs(adv[i]) for i in idx])

    result = {
        "mean_abs_advantage_grpo": np.mean(np.abs(adv_vanilla)),
        "mean_abs_advantage_taura": np.mean(np.abs(adv_taura)),
        "high_entropy_turns": len(high_ent_idx),
        "low_entropy_turns": len(low_ent_idx),
        "mean_abs_adv_high_entropy_grpo": mean_abs(adv_vanilla, high_ent_idx),
        "mean_abs_adv_high_entropy_taura": mean_abs(adv_taura, high_ent_idx),
        "mean_abs_adv_low_entropy_grpo": mean_abs(adv_vanilla, low_ent_idx),
        "mean_abs_adv_low_entropy_taura": mean_abs(adv_taura, low_ent_idx),
    }

    if verbose:
        print("\nSummary")
        print(f"  High-entropy turns: {result['high_entropy_turns']}")
        print(f"  Low-entropy turns : {result['low_entropy_turns']}")
        print(f"  Mean |adv| GRPO  : {result['mean_abs_advantage_grpo']:.3f}")
        print(f"  Mean |adv| TAURA : {result['mean_abs_advantage_taura']:.3f}")
        print(f"  High-entropy |adv| GRPO  : {result['mean_abs_adv_high_entropy_grpo']:.3f}")
        print(f"  High-entropy |adv| TAURA : {result['mean_abs_adv_high_entropy_taura']:.3f}")
        print(f"  Low-entropy |adv| GRPO   : {result['mean_abs_adv_low_entropy_grpo']:.3f}")
        print(f"  Low-entropy |adv| TAURA  : {result['mean_abs_adv_low_entropy_taura']:.3f}")
        boost = (result["mean_abs_adv_high_entropy_taura"] /
                 (result["mean_abs_adv_high_entropy_grpo"] + 1e-8))
        print(f"\n  TAURA boosts high-entropy turn magnitude by ~{boost:.2f}x")
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OmniAgent tiny probe")
    parser.add_argument("--part", choices=["pomdp", "taura", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.part in ("pomdp", "all"):
        rewards = []
        for s in range(5):
            r = run_active_perception_episode(seed=args.seed + s, verbose=(s == 0))
            rewards.append(r)
        print(f"\nMean reward over 5 episodes: {np.mean(rewards):.3f} (std={np.std(rewards):.3f})")

    if args.part in ("taura", "all"):
        demo_taura()


if __name__ == "__main__":
    main()
