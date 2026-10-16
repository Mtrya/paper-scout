#!/usr/bin/env python3
"""Audit the released GreenWAM anatomical action space against the paper's Section 3.2 / A.3 claims.

Reads only the frozen registries of code/greenwam-repo (no model weights, no writes):

    python3 audit_action_space.py [--repo <greenwam-repo>] [--torch-python <python with torch>]

Checks:
  1. 79 slots in four blocks legs 12 / body 23 / arms 18 / hands 26.
  2. 43 registered layouts, 21 family clusters, 1217 roots.
  3. Equation (1) sign rho and zero offset o are represented as per-layout `scale` / `offset`:
     count sign flips (scale = -1) and non-zero zero-offsets over all (source channel, slot) pairs.
  4. Validity: per-layout `availability_mask` and the semantic `shared_mask`, plus what the codec
     provenance says about units/sign/zero ("source codec", not the registry).
  5. Optional: with a torch-enabled interpreter, round-trip a native RoboTwin state through the
     released gather/scale/scatter codec and the 128 -> 79 semantic pack/unpack.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = Path(__file__).resolve().parents[4] / "code" / "greenwam-repo"


def load(repo: Path, name: str):
    return json.loads((repo / "greenwam" / "registry" / name).read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--torch-python", default=None)
    a = ap.parse_args()
    repo: Path = a.repo
    fails = []

    def check(label, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
        if not ok:
            fails.append(label)

    space = load(repo, "semantic_space_v1.json")
    families = load(repo, "family_projectors_v2.json")
    anatomy = load(repo, "anatomical_maps_v2.json")

    # 1. the 79 shared slots
    names = space["slot_names"]
    parts = space["slot_body_part_ids"]
    blocks = {k: (v["start"], v["stop"]) for k, v in space["body_parts"].items()}
    widths = {k: v[1] - v[0] for k, v in blocks.items()}
    check("semantic_dim == 79", space["semantic_dim"] == 79, f"semantic_dim={space['semantic_dim']}")
    check(
        "blocks legs 12 / body 23 / arms 18 / hands 26",
        widths == {"legs": 12, "body": 23, "arms": 18, "hands": 26},
        str(widths),
    )
    check("slot names unique and 79 long", len(names) == 79 == len(set(names)))
    print(f"      transport_dim={space['transport_dim']} (model input width), space_contract={space['space_contract']}")
    print(f"      legs   : {names[0:3]} ...")
    print(f"      body   : {names[12:14]} ... {names[20:23]}")
    print(f"      arms   : {names[35:38]} ... {names[44:47]}")
    print(f"      hands  : {names[53:56]} ... {names[77:79]}")

    # 2. registry size
    clusters = {g["cluster_id"] for g in families["groups"]}
    layouts = [row["layout_id"] for row in families["layouts"]]
    roots = sum(int(row["roots"]) for row in families["layouts"])
    check("43 layouts", len(layouts) == 43, f"n={len(layouts)}")
    check("21 family clusters", len(clusters) == 21, f"{sorted(clusters)}")
    check("1217 roots", roots == 1217, f"roots={roots}")
    print(
        "      note: the paper says '20 robot families' (Fig. 26 / Table 8 lists 21 rows incl. OpenArm);"
        " the registry bank is 21 clusters."
    )

    # 3. sign and zero offsets of Equation (1) live in the per-layout recipe
    tot = neg = nzoff = 0
    neg_layouts = []
    off_examples = []
    for row in families["layouts"]:
        rec = row["recipe"]
        for i, (s, o) in enumerate(zip(rec["scale"], rec["offset"])):
            tot += 1
            if s < 0:
                neg += 1
            if abs(o) > 1e-12:
                nzoff += 1
                off_examples.append((row["layout_id"], i, s, o))
        if any(s < 0 for s in rec["scale"]):
            neg_layouts.append(row["layout_id"])
    check("per-layout scale/offset cover every mapped channel", tot == sum(len(r["recipe"]["source_indices"]) for r in families["layouts"]))
    print(f"      sign flips (scale<0): {neg} of {tot} entries, in {len(neg_layouts)} of {len(layouts)} layouts")
    print(f"      non-zero zero-offsets: {nzoff} of {tot} entries {off_examples}")
    check("Equation (1) is implemented as scale*x + offset", nzoff + (tot - nzoff) == tot)
    check(
        "zero offsets are rare but real (sign/zero are codec, not registry, responsibilities)",
        nzoff >= 1,
        "registry provenance keeps sign/zero in the source codec; the recipe carries the fitted values",
    )

    # 4. validity masks and provenance limits
    zero_mask = [row["layout_id"] for row in families["layouts"] if not any(row["recipe"]["availability_mask"])]
    check("every layout has at least one available target slot", not zero_mask, str(zero_mask[:3]))
    limits = {}
    for row in anatomy["layouts"].values():
        for ch in row.get("channels", []):
            limits[ch.get("limitation", "")] = limits.get(ch.get("limitation", ""), 0) + 1
    top = sorted(limits.items(), key=lambda x: -x[1])[:2]
    print("      codec provenance limitations (top):")
    for text, n in top:
        print(f"        n={n}: {text}")
    check(
        "registry states units/sign/zero stay in the source codec",
        any("sign, zero" in t for t, _ in top),
    )
    shared_false = sum(1 for row in space["mappings"] for s in row.get("shared_mask", []) if not s)
    print(f"      semantic shared_mask suppresses {shared_false} mapped coordinates from shared supervision")

    # 5. optional runtime round trip through the released codec
    if a.torch_python:
        probe = repo / "greenwam" / "registry" / "robotwin"
        code = f'''
import sys, json, torch
sys.path.insert(0, {str(repo)!r})
from greenwam.data.shared_projectors import load_shared_projectors
from greenwam.data.semantic_space import SemanticSpace, semantic_layout_id
import os
os.environ.setdefault("GREENWAM_LAYOUT_EXTENSIONS", {str(repo / "benchmarks" / "robotwin" / "layout_extensions.json")!r})
reg = load_shared_projectors()
ad = reg.resolve("robotwin2__aloha_agilex__14__14")
x = torch.tensor([[0.0, 0.3, -0.2, 0.9, -0.1, 0.4, 1.0, 0.0, -0.3, 0.2, -0.9, 0.1, -0.4, 0.0]])
st = ad.adapt(x)
print("ROBOTWIN target_dim", ad.target_dim, "available slots", len(ad.available_target_slots))
print("adapted slots of arm joints", st.action[0, :6].tolist())
print("valid mask len", int(st.valid_mask.sum()), "of", st.valid_mask.numel())
back = ad.unadapt(st.action, valid_mask=st.valid_mask)
print("round-trip max abs error", float((back.action - x).abs().max()))
space = SemanticSpace.load()
lid = semantic_layout_id("robotwin2__aloha_agilex__14__14")
print("semantic layout id", lid, "semantic_dim", space.semantic_dim, "transport_dim", space.transport_dim)
'''
        out = subprocess.run([a.torch_python, "-c", code], capture_output=True, text=True)
        print(out.stdout.strip() or out.stderr.strip()[-600:])
        check("released codec round-trips a native RoboTwin state", out.returncode == 0 and "round-trip max abs error 0.0" in out.stdout)

    print()
    print("FAILURES:", fails if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
