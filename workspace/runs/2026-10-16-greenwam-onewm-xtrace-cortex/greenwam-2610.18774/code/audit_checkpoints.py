#!/usr/bin/env python3
"""Audit the four released GreenWAM checkpoints on Hugging Face against the paper's claims.

Reads only small files (config.yaml, index json) and the safetensors headers via HTTP range requests.
Never downloads weights. Usage:

    python3 audit_checkpoints.py [--repo SberRoboticsCenter/GreenWAM]

Checks:
  1. the four checkpoint folders of the README table exist with weights + config + stats + SHA256SUMS;
  2. the policy checkpoints carry the action expert (`action_expert: true`) and the world model does not;
  3. parameter counts from the safetensors headers: how many parameters the action expert adds and
     its share of the policy checkpoint;
  4. the action interface the checkpoints were evaluated with (max_action_dim, native_action_dim,
     num_action_contracts, semantic_projectors, loss scale, shift).
"""

from __future__ import annotations

import argparse
import json
import struct
import urllib.parse
import urllib.request

API = "https://huggingface.co/api/models/{repo}"
RESOLVE = "https://huggingface.co/{repo}/resolve/main/{path}"
FOLDERS = ("robotwin", "libero", "calvin", "worldarena")


def get(url, headers=None, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=timeout).read()


def tree(repo):
    doc = json.loads(get(API.format(repo=repo) + "/tree/main"))
    files = []
    for row in doc:
        if row["type"] == "directory":
            files += [t["path"] for t in json.loads(get(API.format(repo=repo) + f"/tree/main/{row['path']}"))]
        else:
            files.append(row["path"])
    return files


def safetensors_header(repo, path):
    raw = get(RESOLVE.format(repo=repo, path=path), headers={"Range": "bytes=0-8000000"})
    n = struct.unpack("<Q", raw[:8])[0]
    doc = json.loads(raw[8 : 8 + n])
    doc.pop("__metadata__", None)
    return doc


def params(header):
    total = expert = 0
    for key, value in header.items():
        n = 1
        for d in value["shape"]:
            n *= d
        total += n
        if "_action_expert" in key:
            expert += n
    return total, expert


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="SberRoboticsCenter/GreenWAM")
    a = ap.parse_args()
    repo = a.repo
    files = tree(repo)
    fails = []

    def check(label, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
        if not ok:
            fails.append(label)

    print(f"repo {repo}: {len(files)} files")
    for folder in FOLDERS:
        have = {f.split("/", 1)[1] for f in files if f.startswith(folder + "/")}
        need = {"config.yaml", "config.json", "model.safetensors.index.json", "SHA256SUMS"}
        check(
            f"{folder}: config + index + SHA256SUMS present",
            need <= have,
            f"{len([f for f in files if f.startswith(folder + '/')])} files, missing {sorted(need - have)}",
        )
    check("action statistics released with the policy checkpoints",
          all(any(f.startswith(f"{x}/") and "stats" in f or f.endswith("_stats.json") for f in files) for x in ("robotwin", "libero", "calvin", "worldarena")))

    # config flags
    flags = {}
    for folder in FOLDERS:
        text = get(RESOLVE.format(repo=repo, path=f"{folder}/config.yaml")).decode()
        flags[folder] = {
            key: next((line.split(":", 1)[1].strip() for line in text.splitlines() if line.strip().startswith(key + ":")), None)
            for key in ("action_expert", "action_gen", "max_action_dim", "native_action_dim", "num_action_contracts", "semantic_projectors", "action_loss_weight", "loss_scale")
        }
    for folder in ("robotwin", "libero", "calvin"):
        check(f"{folder} policy checkpoint has action_expert: true", flags[folder]["action_expert"] == "true", str(flags[folder]))
    check("worldarena checkpoint has no action expert", flags["worldarena"]["action_expert"] is None)
    check("all checkpoints share the 128-wide semantic transport space",
          all(f["max_action_dim"] == "128" for f in flags.values()))
    check("all checkpoints use the shared semantic projectors",
          all(f["semantic_projectors"] == "true" for f in flags.values()))

    # parameter counts
    print()
    counts = {}
    for folder in ("robotwin", "worldarena"):
        total = expert = 0
        for shard in [f for f in files if f.startswith(folder + "/model-") and f.endswith(".safetensors")]:
            t, e = params(safetensors_header(repo, shard))
            total += t
            expert += e
        counts[folder] = (total, expert)
        print(f"      {folder}: {total/1e9:.3f} B parameters, {expert/1e9:.3f} B in *_action_expert* tensors")
    delta = counts["robotwin"][0] - counts["worldarena"][0]
    check("policy checkpoint = world-model checkpoint + the action expert",
          abs(delta - counts["robotwin"][1]) < 1e6 and counts["worldarena"][1] == 0,
          f"delta {delta/1e9:.3f} B")
    check("the action expert is about as large as the generation tower it twins",
          abs(counts["robotwin"][1] / counts["worldarena"][0] - 1410_000_000 / 3_370_000_000) < 0.05,
          f"expert {counts['robotwin'][1]/1e9:.3f} B vs generator {counts['worldarena'][0]/1e9:.3f} B")
    print()
    print("FAILURES:", fails if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
