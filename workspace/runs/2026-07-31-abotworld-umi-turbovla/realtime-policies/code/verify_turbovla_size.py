#!/usr/bin/env python3
"""Probe: verify TurboVLA's deployed parameter counts from HF checkpoint file sizes.

TurboVLA (arXiv 2607.27205) advertises a "0.2B" model at 32 Hz / <1 GB VRAM.
The code trace (code/turbovla, turbovla/models/turbovla.py) shows the LIBERO
inference graph is DINOv3-B/16 (86.6M) + ~21M of fusion/ACT heads ≈ 107M,
because the BERT text encoder is cached offline (scripts/libero/build_text_cache.py)
and never runs at eval. Checkpoint file sizes independently confirm this:

  - LIBERO .pth are fp32 (4 bytes/param):  ~426.5 MB  -> ~106.6M params
  - RoboTwin .safetensors is bf16 (2 bytes/param): ~868.2 MB -> ~434.1M params
    (= 21M heads + DINOv3-L 304M + live frozen BERT-base 110M)

This script fetches the real byte sizes from the HF API and prints the implied
parameter counts. CPU-only, no downloads of the weights themselves.
"""
import json
import subprocess
import sys

API = "https://huggingface.co/api/models/H-EmbodVis/TurboVLA?blobs=true"

out = subprocess.run(["curl", "-sL", API], capture_output=True, check=True).stdout
data = json.loads(out)

rows = []
for f in data.get("siblings", []):
    name = f["rfilename"]
    size = f.get("size")
    if size and (name.endswith(".pth") or name.endswith(".safetensors")):
        bpp = 4 if name.endswith(".pth") else 2  # fp32 vs bf16, per code trace
        rows.append((name, size, size / bpp))

print(f"{'checkpoint':50s} {'bytes':>14s} {'assumed':>8s} {'implied params':>15s}")
for name, size, params in rows:
    kind = "fp32" if name.endswith(".pth") else "bf16"
    print(f"{name:50s} {size:14,d} {kind:>8s} {params:15,.0f}")

libero = [p for n, _, p in rows if "libero" in n]
robotwin = [p for n, _, p in rows if "robotwin" in n]
if libero:
    print(f"\nLIBERO deployed graph ~= {sum(libero) / len(libero) / 1e6:.1f}M params "
          f"(paper claims 0.2B; BERT ~110M is cached away, not executed)")
if robotwin:
    print(f"RoboTwin deployed graph ~= {robotwin[0] / 1e6:.1f}M params "
          f"(21M heads + DINOv3-L 304M + frozen BERT-base 110M)")
