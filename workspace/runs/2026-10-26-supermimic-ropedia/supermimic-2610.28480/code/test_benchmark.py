"""Whole-clip evaluation stops on the final reference state, not one action later."""

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/npz/lafan1/walk1_subject1.npz"
CHECKPOINT = ROOT / "logs/train30_1024/ckpt_0012000.pt"


def test_successful_clip_stops_at_last_reference_frame(tmp_path):
    if not torch.cuda.is_available() or not SOURCE.exists() or not CHECKPOINT.exists():
        pytest.skip("needs CUDA + LAFAN1 npz + local trained checkpoint")
    from supermimic.eval.benchmark import load_policy_from_checkpoint, run_benchmark

    short = tmp_path / "short_walk.npz"
    long = tmp_path / "long_walk.npz"
    with np.load(SOURCE) as src:
        for path, n in ((short, 48), (long, 512)):
            np.savez(path, **{key: (src[key][:n] if src[key].ndim > 0 and src[key].shape[0] > n else src[key]) for key in src.files})
    policy, cfg = load_policy_from_checkpoint(CHECKPOINT, "cuda:0")
    results = run_benchmark(policy, [str(short), str(long), str(short)], device="cuda:0", num_envs=2,
                            anchor_obs=cfg["env"].get("anchor_obs", False), verbose=False)

    assert [r.name for r in results] == ["short_walk", "long_walk", "short_walk"]
    for result in (results[0], results[2]):
        assert not result.failed, result
        assert result.frames == 48
        assert result.steps == 47
        assert result.norm_return_full_clip == pytest.approx(result.norm_return)
    assert results[1].frames == 512
    assert results[1].steps <= 511
    if not results[1].failed:
        assert results[1].steps == 511


def test_summary_keeps_unearned_rewards_after_early_termination():
    from supermimic.eval.benchmark import ClipResult, summarize

    results = [
        ClipResult("done", 101, 100, 0.8, 0.8, False, -1),
        ClipResult("early", 101, 25, 1.0, 0.25, True, 25),
    ]
    summary = summarize(results)
    assert summary["mean_norm_return"] == pytest.approx(0.9)
    assert summary["mean_norm_return_full_clip"] == pytest.approx(0.525)
    assert summary["failed"] == 1
