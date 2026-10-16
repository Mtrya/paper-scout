#!/usr/bin/env python3
"""Step 1: launch the cached vLLM rotary kernel with cuda-python and time it.

No gwatch involvement yet -- this only proves that the hand-built launch is a real kernel
execution (its output matches a numpy rotary reference) and establishes the baseline timing
the traced run will be compared against.

Usage: launch_smoke.py [--reps 50] [--seqlen 512]
"""

import argparse
import statistics
import sys

import os

import numpy as np
import cuda.bindings.driver as cu

import cuda_launch

CUBIN = os.path.expanduser(
    "~/.triton/cache/YGPFJAPL5ZDQ2SQAGOCZ3FCEYS3AKJOG6TIVYO2YXWIUU7GFXQHA"
    "/rotary_kernel.cubin")


def bf16_round(x: np.ndarray) -> np.ndarray:
    """Round a float32 array to bfloat16 precision (numpy has no bf16)."""
    u = x.astype(np.float32).view(np.uint32)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return rounded.view(np.float32)


def reference_rotary(x: np.ndarray, cos: np.ndarray, sin: np.ndarray,
                     rotary_dim: int) -> np.ndarray:
    """Rotate-half (non-interleaved) rotary embedding on the first rotary_dim head dims.

    vLLM's kernel for the default (non-interleaved, non-conjugate) case writes
    ``out[..., k] = x[..., k] * cos[..., k] - x[..., k + half] * sin[..., k]`` and the
    mirrored expression for the upper half.
    """
    half = rotary_dim // 2
    batch, seqlen, heads, headdim = x.shape
    out = x.astype(np.float32).copy()
    c = cos.astype(np.float32)
    s = sin.astype(np.float32)
    lower = x[..., :half].astype(np.float32)
    upper = x[..., half:rotary_dim].astype(np.float32)
    out[..., :half] = lower * c[:, :half] - upper * s[:, :half]
    out[..., half:rotary_dim] = upper * c[:, half:rotary_dim] + lower * s[:, half:rotary_dim]
    return out


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--cubin", default=CUBIN)
    args = ap.parse_args()

    launch = cuda_launch.Launch(cubin_path=args.cubin, kernel_name="rotary_kernel",
                                seqlen=args.seqlen)
    print(f"grid {launch.grid} block {launch.block}")

    launch.launch()
    launch.synchronize()
    samples = []
    for _ in range(args.reps):
        start, end = launch.new_event(), launch.new_event()
        cu.cuEventRecord(start, launch.stream)
        launch.launch()
        cu.cuEventRecord(end, launch.stream)
        launch.synchronize()
        samples.append(launch.elapsed(start, end))
    print(f"rotary kernel: {len(samples)} launches, median {statistics.median(samples)*1000:.1f} us, "
          f"min {min(samples)*1000:.1f}, max {max(samples)*1000:.1f}")

    out = launch.buffers["out"].download()
    x = launch.buffers["x"].download()
    cos = launch.buffers["cos"].download()
    sin = launch.buffers["sin"].download()
    got = bf16_to_f32(out).astype(np.float32)
    want = reference_rotary(bf16_to_f32(x), bf16_to_f32(cos), bf16_to_f32(sin),
                            launch.rotary_dim)
    # compare only the rows the launch actually covers
    covered = min(launch.grid[0] * launch.block_m, launch.seqlen)
    diff = np.abs(got[:, :covered] - want[:, :covered])
    scale = np.abs(want[:, :, :covered]).max()
    print(f"rows covered by the grid: {covered}/{launch.seqlen}")
    print(f"max |diff| vs numpy reference: {diff.max():.4f} (reference max |value| {scale:.3f})")
    print(f"output nonzero: {int((out != 0).sum())} of {out.size}")


if __name__ == "__main__":
    main()
