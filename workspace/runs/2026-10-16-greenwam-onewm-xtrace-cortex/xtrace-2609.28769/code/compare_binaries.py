#!/usr/bin/env python3
"""Compare an original cubin with the spliced one the tool produced.

Reports three things the paper's section 5.3 metric implies, at increasing strictness:

  * text-level similarity -- renders each instruction (opcode, operands and the control-bit
    fields the renderer prints) and aligns the two sequences; an instruction counts as
    preserved when its rendering is unchanged.
  * byte-level similarity -- the same alignment over the raw 16-byte encodings. This covers
    every field the renderer does not print, so a changed stall count or scoreboard shows up.
  * control-flow equivalence -- decodes the branch/barrier displacements of both images and
    checks that each transfer still lands on the instruction it used to land on (see
    control_flow_check.py for the same check printed in full).

Usage: compare_binaries.py <origin.cubin> <spliced.cubin> [--json out.json]
"""

import argparse
import difflib
import json
from collections import Counter

import gwatch.cuda.binary as gw_binary

import sasslib


def instruction_bytes(cubin_path: str):
    """{kernel name: (texts, bytes)} for a cubin."""
    cubin = gw_binary.Cubin()
    cubin.fill(cubin_path)
    cubin.parse()
    out = {}
    for name, kd in cubin.get_map_kernel_def().items():
        instrs = kd.get_list_instruction()
        out[name] = ([i.str() for i in instrs], [i.bytes for i in instrs])
    return out


def aligned_preservation(origin_items, traced_items):
    """How many origin items survive the alignment, and which do not."""
    matcher = difflib.SequenceMatcher(a=origin_items, b=traced_items, autojunk=False)
    covered = [False] * len(origin_items)
    for block in matcher.get_matching_blocks():
        for i in range(block.a, block.a + block.size):
            covered[i] = True
    preserved = sum(covered)
    return preserved, [i for i, ok in enumerate(covered) if not ok], matcher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("origin")
    ap.add_argument("spliced")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    origin = instruction_bytes(args.origin)
    traced = instruction_bytes(args.spliced)
    name = next(iter(origin))
    if name not in traced:
        raise SystemExit(f"{name} not in the spliced image ({list(traced)})")
    origin_view = sasslib.load_kernels(args.origin)[name]
    traced_view = sasslib.load_kernels(args.spliced)[name]

    o_text, o_bytes = origin[name]
    t_text, t_bytes = traced[name]

    text_preserved, text_changed_idx, _ = aligned_preservation(o_text, t_text)
    byte_preserved, byte_changed_idx, byte_match = aligned_preservation(o_bytes, t_bytes)

    # instructions identical as bytes but not as text would mean the renderer is unstable;
    # instructions identical as text but not as bytes are control-bit-only changes
    byte_changed_but_text_same = [
        i for i in byte_changed_idx if i not in set(text_changed_idx)
    ]
    text_changed_but_byte_same = [
        i for i in text_changed_idx if i not in set(byte_changed_idx)
    ]

    result = {
        "kernel": name,
        "origin_instructions": len(o_text),
        "traced_instructions": len(t_text),
        "text_preserved": text_preserved,
        "text_similarity": text_preserved / len(o_text),
        "byte_preserved": byte_preserved,
        "byte_similarity": byte_preserved / len(o_text),
        "inserted_instructions": len(t_text) - text_preserved,
        "text_changed_by_opcode": dict(Counter(sasslib.opcode_of(o_text[i]) for i in text_changed_idx)),
        "byte_changed_by_opcode": dict(Counter(sasslib.opcode_of(o_text[i]) for i in byte_changed_idx)),
        "control_bit_only_changes": len(byte_changed_but_text_same),
        "byte_changed_but_text_same_opcodes": dict(
            Counter(sasslib.opcode_of(o_text[i]) for i in byte_changed_but_text_same)),
        "renderer_instability": len(text_changed_but_byte_same),
        "origin_registers": origin_view.num_registers,
        "traced_registers": traced_view.num_registers,
        "origin_basic_blocks": origin_view.nb_basic_blocks,
        "traced_basic_blocks": traced_view.nb_basic_blocks,
    }
    result.update(sasslib.block_changes(origin_view, traced_view))

    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    main()
