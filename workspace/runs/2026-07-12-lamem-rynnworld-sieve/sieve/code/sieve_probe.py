#!/usr/bin/env python3
"""
SIEVE core-selection probe.

This standalone script reconstructs the two-stage selection logic from
SIEVE (Wu et al., arXiv 2607.06442) on purely synthetic data so that the
behavior of the algorithm can be inspected without V-JEPA, LeRobot, or
large robot datasets.

It demonstrates:
  1. Synthetic trajectory segmentation driven by gripper-command flips.
  2. Primitive composition patterns and transition interfaces.
  3. Stage-A structural exposure allocation with diminishing returns.
  4. Stage-B medoid selection within each composition-pattern bucket.
  5. The quantitative difference between the official code's weight
     normalization (by mean q_c / mean q_e) and the paper's stated
     normalization (by |P|).

Run with:
    python3 sieve_probe.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

EpisodeKey = str
Pattern = Tuple[int, ...]
Transition = Tuple[int, int]


def confirmed_gripper_flips(cmd: np.ndarray, confirm: int) -> List[int]:
    """Return frame indices where the gripper command changes and persists.

    Mirrors gripper_pose_seg.py::has_confirmed_gripper_flip.
    """
    splits = [0]
    n = len(cmd)
    for i in range(1, n):
        if i + confirm > n:
            break
        if bool(np.all(cmd[i : i + confirm] != cmd[i - 1])):
            if i != splits[-1]:
                splits.append(i)
    return splits


def synthesize_episodes(
    n_episodes: int = 120,
    n_clusters: int = 8,
    feat_dim: int = 16,
    min_len: int = 30,
    max_len: int = 90,
    seed: int = 42,
) -> Tuple[Dict[EpisodeKey, Dict], np.ndarray]:
    """Generate synthetic episodes with gripper-driven segments.

    Each episode is a sequence of frames.  A binary gripper command toggles
    between open/close; confirmed flips define segment boundaries.  Each
    segment is assigned to one of ``n_clusters`` primitive centroids with
    Gaussian noise.

    Returns:
        episodes: mapping episode_key -> {"pattern": [...], "features": [...]}
        centroids: [n_clusters, feat_dim] array of primitive centers.
    """
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((n_clusters, feat_dim))
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12

    episodes: Dict[EpisodeKey, Dict] = {}
    for ep in range(n_episodes):
        length = rng.integers(min_len, max_len + 1)
        # Each episode has 2-4 gripper state intervals.
        n_intervals = rng.integers(2, 5)
        interval_ends = np.sort(rng.integers(1, length, size=n_intervals - 1))
        cmd = np.zeros(length, dtype=np.int32)
        state = 0
        prev = 0
        for end in interval_ends:
            cmd[prev:end] = state
            state = 1 - state
            prev = end
        cmd[prev:] = state

        splits = confirmed_gripper_flips(cmd, confirm=5)
        # Make sure the last frame is a boundary for the final segment.
        if splits[-1] != length:
            splits.append(length)

        features = []
        pattern = []
        for s, e in zip(splits[:-1], splits[1:]):
            # Primitive chosen by the first frame's gripper state and a bit
            # of episode-level structure so patterns are not uniform random.
            base = (int(cmd[s]) * 4 + ep % 4) % n_clusters
            noise = rng.standard_normal(feat_dim) * 0.15
            feat = centroids[base] + noise
            feat /= np.linalg.norm(feat) + 1e-12
            features.append(feat)
            pattern.append(base + 1)  # 1-based like the official code.

        key = f"ep_{ep:04d}"
        episodes[key] = {"pattern": tuple(pattern), "features": features}

    return episodes, centroids


def transitions_of_pattern(pattern: Pattern) -> List[Transition]:
    """Transition list with a sentinel for single-primitive episodes."""
    if len(pattern) == 1:
        return [(pattern[0], -1)]
    return list(zip(pattern[:-1], pattern[1:]))


def compute_feature_weights_code(
    patterns: List[Pattern],
) -> Tuple[Dict[int, float], Dict[Transition, float]]:
    """Weights as implemented in the official code (select.py).

    Code normalizes by the *mean* q_c / q_e across primitives / transitions,
    not by |P| as the paper states.
    """
    q_c = Counter()
    q_e = Counter()
    for p in patterns:
        for c in set(p):
            q_c[c] += 1
        for e in set(transitions_of_pattern(p)):
            q_e[e] += 1

    def norm(q: Counter) -> Dict:
        if not q:
            return {}
        mean_q = sum(q.values()) / len(q)
        return {k: float(v / mean_q) for k, v in q.items()} if mean_q > 0 else {}

    return norm(q_c), norm(q_e)


def compute_feature_weights_paper(
    patterns: List[Pattern],
) -> Tuple[Dict[int, float], Dict[Transition, float]]:
    """Weights as written in the paper: w_c = q_c / |P|, w_e = q_e / |P|."""
    n_patterns = len(patterns)
    if n_patterns == 0:
        return {}, {}
    q_c = Counter()
    q_e = Counter()
    for p in patterns:
        for c in set(p):
            q_c[c] += 1
        for e in set(transitions_of_pattern(p)):
            q_e[e] += 1
    return {k: v / n_patterns for k, v in q_c.items()}, {
        k: v / n_patterns for k, v in q_e.items()
    }


def pattern_occurrence_features(pattern: Pattern) -> Tuple[Counter, Counter]:
    occ_c = Counter(pattern)
    occ_e = Counter(transitions_of_pattern(pattern))
    return occ_c, occ_e


def marginal_gain(
    occ_c: Counter,
    occ_e: Counter,
    n_c: Dict[int, int],
    n_e: Dict[Transition, int],
    w_c: Dict[int, float],
    w_e: Dict[Transition, float],
) -> float:
    gain = 0.0
    for c, occ in occ_c.items():
        w = w_c.get(c, 0.0)
        if w <= 0:
            continue
        now = n_c.get(c, 0)
        gain += w * (math.log1p(now + occ) - math.log1p(now))
    for e, occ in occ_e.items():
        w = w_e.get(e, 0.0)
        if w <= 0:
            continue
        now = n_e.get(e, 0)
        gain += w * (math.log1p(now + occ) - math.log1p(now))
    return gain


def stage_a_allocate(
    pattern_to_episodes: Dict[Pattern, List[EpisodeKey]],
    budget: int,
    w_c: Dict[int, float],
    w_e: Dict[Transition, float],
) -> Dict[Pattern, int]:
    """Greedy budget allocation maximizing structural exposure."""
    patterns = sorted(pattern_to_episodes.keys(), key=lambda p: (len(p), p))
    total = sum(len(v) for v in pattern_to_episodes.values())
    budget = min(budget, total)

    occ_by_pattern = {p: pattern_occurrence_features(p) for p in patterns}
    n_c: Dict[int, int] = defaultdict(int)
    n_e: Dict[Transition, int] = defaultdict(int)
    m_p = {p: 0 for p in patterns}
    cap_p = {p: len(pattern_to_episodes[p]) for p in patterns}

    for _ in range(budget):
        best_pattern = None
        best_gain = float("-inf")
        best_remaining = -1
        for p in patterns:
            remaining = cap_p[p] - m_p[p]
            if remaining <= 0:
                continue
            gain = marginal_gain(
                occ_c=occ_by_pattern[p][0],
                occ_e=occ_by_pattern[p][1],
                n_c=n_c,
                n_e=n_e,
                w_c=w_c,
                w_e=w_e,
            )
            if gain > best_gain or (gain == best_gain and remaining > best_remaining):
                best_gain = gain
                best_pattern = p
                best_remaining = remaining
        if best_pattern is None:
            break
        m_p[best_pattern] += 1
        occ_c, occ_e = occ_by_pattern[best_pattern]
        for c, occ in occ_c.items():
            n_c[c] = n_c.get(c, 0) + int(occ)
        for e, occ in occ_e.items():
            n_e[e] = n_e.get(e, 0) + int(occ)

    return {p: m for p, m in m_p.items() if m > 0}


def stage_b_medoid(
    pattern_to_episodes: Dict[Pattern, List[EpisodeKey]],
    allocation: Dict[Pattern, int],
    episode_vectors: Dict[EpisodeKey, np.ndarray],
) -> List[EpisodeKey]:
    """Select episodes closest to the pattern medoid (cosine space)."""
    selected: List[EpisodeKey] = []
    for p, m in sorted(allocation.items(), key=lambda kv: (len(kv[0]), kv[0])):
        cands = pattern_to_episodes[p]
        if m <= 0:
            continue
        mat = np.stack([episode_vectors[k] for k in cands], axis=0).astype(np.float32)
        norm = np.linalg.norm(mat, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-12)
        unit = mat / norm
        sim_sums = (unit @ unit.T).sum(axis=1)
        medoid = unit[int(np.argmax(sim_sums))]
        dists = 1.0 - (unit @ medoid)
        order = np.argsort(dists, kind="stable")
        selected.extend([cands[int(i)] for i in order[:m].tolist()])
    return selected


def summarize(
    episodes: Dict[EpisodeKey, Dict],
    allocation: Dict[Pattern, int],
    selected: List[EpisodeKey],
    label: str,
) -> Dict:
    """Produce a compact summary of a selection run."""
    pattern_counts_before = Counter(episodes[k]["pattern"] for k in episodes)
    pattern_counts_after = Counter(episodes[k]["pattern"] for k in selected)

    primitive_counts_before = Counter()
    primitive_counts_after = Counter()
    transition_counts_before = Counter()
    transition_counts_after = Counter()

    for k, v in episodes.items():
        p = v["pattern"]
        primitive_counts_before.update(p)
        transition_counts_before.update(transitions_of_pattern(p))
    for k in selected:
        p = episodes[k]["pattern"]
        primitive_counts_after.update(p)
        transition_counts_after.update(transitions_of_pattern(p))

    n_single_before = sum(1 for p in episodes.values() if len(p["pattern"]) == 1)
    n_single_after = sum(1 for k in selected if len(episodes[k]["pattern"]) == 1)

    return {
        "label": label,
        "budget": len(selected),
        "active_patterns": len(allocation),
        "unique_patterns_before": len(pattern_counts_before),
        "unique_patterns_after": len(pattern_counts_after),
        "single_primitive_episodes_before": n_single_before,
        "single_primitive_episodes_after": n_single_after,
        "gini_pattern_before": gini(list(pattern_counts_before.values())),
        "gini_pattern_after": gini(list(pattern_counts_after.values())),
        "top3_before": pattern_counts_before.most_common(3),
        "top3_after": pattern_counts_after.most_common(3),
        "pattern_counts_before": sorted(pattern_counts_before.items()),
        "pattern_counts_after": sorted(pattern_counts_after.items()),
    }


def gini(values: List[int]) -> float:
    """Simple Gini coefficient; 0 = perfectly equal, 1 = maximally skewed."""
    if not values or sum(values) == 0:
        return 0.0
    vals = np.asarray(sorted(values), dtype=np.float64)
    n = len(vals)
    cum = np.cumsum(vals)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def run_probe(n_episodes: int = 120, budget_ratio: float = 0.5, seed: int = 42) -> Dict:
    budget = max(1, int(round(n_episodes * budget_ratio)))
    episodes, _ = synthesize_episodes(n_episodes=n_episodes, seed=seed)

    pattern_to_episodes: Dict[Pattern, List[EpisodeKey]] = defaultdict(list)
    episode_vectors: Dict[EpisodeKey, np.ndarray] = {}
    for k, v in episodes.items():
        pattern_to_episodes[v["pattern"]].append(k)
        episode_vectors[k] = np.concatenate(v["features"], axis=0)

    patterns = list(pattern_to_episodes.keys())
    w_c_code, w_e_code = compute_feature_weights_code(patterns)
    w_c_paper, w_e_paper = compute_feature_weights_paper(patterns)

    alloc_code = stage_a_allocate(pattern_to_episodes, budget, w_c_code, w_e_code)
    alloc_paper = stage_a_allocate(pattern_to_episodes, budget, w_c_paper, w_e_paper)

    sel_code = stage_b_medoid(pattern_to_episodes, alloc_code, episode_vectors)
    sel_paper = stage_b_medoid(pattern_to_episodes, alloc_paper, episode_vectors)

    summary_code = summarize(episodes, alloc_code, sel_code, "code_norm")
    summary_paper = summarize(episodes, alloc_paper, sel_paper, "paper_norm")

    # Compare primitive/transition weight scale ratios.
    mean_w_code = (
        float(np.mean(list(w_c_code.values()))) if w_c_code else 0.0,
        float(np.mean(list(w_e_code.values()))) if w_e_code else 0.0,
    )
    mean_w_paper = (
        float(np.mean(list(w_c_paper.values()))) if w_c_paper else 0.0,
        float(np.mean(list(w_e_paper.values()))) if w_e_paper else 0.0,
    )

    return {
        "n_episodes": n_episodes,
        "budget": budget,
        "n_patterns": len(patterns),
        "mean_primitive_weight": {"code": mean_w_code[0], "paper": mean_w_paper[0]},
        "mean_transition_weight": {"code": mean_w_code[1], "paper": mean_w_paper[1]},
        "code": summary_code,
        "paper": summary_paper,
        "allocation_code": {str(k): v for k, v in alloc_code.items()},
        "allocation_paper": {str(k): v for k, v in alloc_paper.items()},
        "selected_code": sorted(sel_code),
        "selected_paper": sorted(sel_paper),
    }


def main():
    p = argparse.ArgumentParser(description="SIEVE core-selection probe")
    p.add_argument("--n-episodes", type=int, default=120)
    p.add_argument("--budget-ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    result = run_probe(args.n_episodes, args.budget_ratio, args.seed)

    # Pretty print key findings.
    print("=" * 60)
    print("SIEVE core-selection probe")
    print("=" * 60)
    print(f"Synthetic episodes : {result['n_episodes']}")
    print(f"Selection budget   : {result['budget']}")
    print(f"Unique patterns    : {result['n_patterns']}")
    print()
    print("Weight normalization comparison")
    print(f"  code  mean primitive weight : {result['mean_primitive_weight']['code']:.3f}")
    print(f"  paper mean primitive weight : {result['mean_primitive_weight']['paper']:.3f}")
    print(f"  code  mean transition weight: {result['mean_transition_weight']['code']:.3f}")
    print(f"  paper mean transition weight: {result['mean_transition_weight']['paper']:.3f}")
    print()
    for key in ("code", "paper"):
        s = result[key]
        print(f"[{key}] selection")
        print(f"  active patterns          : {s['active_patterns']}")
        print(f"  unique patterns retained : {s['unique_patterns_after']}/{s['unique_patterns_before']}")
        print(f"  single-primitive episodes: {s['single_primitive_episodes_after']}/{s['single_primitive_episodes_before']}")
        print(f"  pattern Gini before/after: {s['gini_pattern_before']:.3f} -> {s['gini_pattern_after']:.3f}")
        print(f"  top-3 before             : {s['top3_before']}")
        print(f"  top-3 after              : {s['top3_after']}")
        print()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            # Convert tuples to lists for JSON serialization.
            json.dump(result, f, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o)
        print(f"Wrote detailed JSON: {args.output}")


if __name__ == "__main__":
    main()
