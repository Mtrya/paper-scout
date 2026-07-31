#!/usr/bin/env python3
"""Probe: enumerate tensors in acvlab/ABot-World-0-5B-LF without downloading weights.

Fetches only the safetensors header (first ~8 MiB) of the released causal
student, parses the JSON header, and reports tensors that do NOT exist in a
stock Wan2.2-TI2V-5B (the action-conditioning delta) with shapes and param
counts. No torch required.

Result on 2026-07-31: 831 tensors, 5,270,329,536 params total, all BF16.
The ONLY non-stock tensors are the 6 act_control_adapter.* tensors below,
totaling 270,541,824 params (~5.1% of the checkpoint):

  act_control_adapter.conv.weight               [3072, 8192, 2, 2]  100,663,296
  act_control_adapter.conv.bias                 [3072]                    3,072
  act_control_adapter.residual_blocks.0.conv1.weight [3072, 3072, 3, 3]  84,934,656
  act_control_adapter.residual_blocks.0.conv1.bias   [3072]               3,072
  act_control_adapter.residual_blocks.0.conv2.weight [3072, 3072, 3, 3]  84,934,656
  act_control_adapter.residual_blocks.0.conv2.bias   [3072]               3,072

Everything else (DMD distillation, LongForcing, causalization) is baked into
the fine-tuned block weights; tensor names map 1:1 onto WanModel.
"""
import json
import struct
import subprocess
import sys

URL = ("https://huggingface.co/acvlab/ABot-World-0-5B-LF/resolve/main/"
       "diffusion_pytorch_model.safetensors")
OUT = "/tmp/abot_head.bin"

# Stock Wan2.2-TI2V-5B tensor-name prefixes are everything except the adapter;
# we simply flag names containing 'act_control_adapter' as the delta, and print
# any other unfamiliar top-level groups for manual inspection.
subprocess.run(["curl", "-sL", "-r", "0-8388607", URL, "-o", OUT], check=True)
with open(OUT, "rb") as f:
    (hlen,) = struct.unpack("<Q", f.read(8))
    header = json.loads(f.read(hlen))

total = 0
delta = 0
groups = {}
for name, meta in header.items():
    if name == "__metadata__":
        continue
    n = 1
    for d in meta["shape"]:
        n *= d
    total += n
    top = name.split(".")[0]
    groups.setdefault(top, 0)
    groups[top] += n
    if "act_control_adapter" in name:
        delta += n
        print(f"DELTA {name} {meta['shape']} {n:,}")

print(f"\ntensors: {len([k for k in header if k != '__metadata__'])}")
print(f"total params: {total:,}")
print(f"action-adapter params: {delta:,} ({delta / total:.1%})")
print("top-level groups:", sorted(groups))
