#!/usr/bin/env python3
"""Does the spliced binary still branch to the same places?

Binary similarity (the paper's metric) counts an instruction as preserved when its text is
unchanged. That is necessary but not sufficient for a *correct* rewrite: a branch whose
displacement no longer lands on the same instruction is textually identical and semantically
broken. SASS encodes a branch target as a displacement (``sImm``) and a convergence barrier as
an offset (``Sa``), so this script decodes both, maps every target instruction of the origin
through the same alignment the similarity metric uses, and checks that the instrumented kernel's
same branch reaches the same instruction.

Usage: control_flow_check.py <origin.cubin> <spliced.cubin>
"""

import difflib
import re
import sys

from collections import Counter

import sasslib

_BRANCH = re.compile(r"^BRA\b")
_BSSY = re.compile(r"^BSSY\b")
_FIELD = re.compile(r"(\w+)=0x([0-9a-f]+)")


def fields(text):
    return {k: int(v, 16) for k, v in _FIELD.findall(text)}


def branch_targets(kernel):
    """(index, kind, rendered text, target index) for every control transfer."""
    out = []
    for i, text in enumerate(kernel.texts):
        opcode = sasslib.opcode_of(text)
        if opcode == "BRA":
            offset = fields(text).get("sImm")
            if offset is None:
                continue
            out.append((i, "BRA", text, i + offset))
        elif opcode == "BSSY":
            offset = fields(text).get("Sa")
            if offset is None:
                continue
            out.append((i, "BSSY", text, i + offset))
    return out


def main():
    origin = next(iter(sasslib.load_kernels(sys.argv[1]).values()))
    traced = next(iter(sasslib.load_kernels(sys.argv[2]).values()))

    matcher = difflib.SequenceMatcher(a=origin.texts, b=traced.texts, autojunk=False)
    map_a_to_b = {}
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            map_a_to_b[block.a + k] = block.b + k

    origin_targets = branch_targets(origin)
    traced_targets = branch_targets(traced)
    print(f"origin: {origin.n} instructions, {len(origin_targets)} control transfers")
    print(f"traced: {traced.n} instructions, {len(traced_targets)} control transfers")
    print(f"preserved instructions: {sum(b.size for b in matcher.get_matching_blocks())}/{origin.n}")

    # pair each origin transfer with the traced transfer of the same kind, in order
    by_kind_o = [t for t in origin_targets]
    by_kind_t = [t for t in traced_targets]
    if len(by_kind_o) != len(by_kind_t):
        print(f"!! different transfer counts: {len(by_kind_o)} origin vs {len(by_kind_t)} traced")

    moved = 0
    same_disp = 0
    mismatched = []
    for (oi, okind, otext, otarget), (ti, tkind, ttext, ttarget) in zip(by_kind_o, by_kind_t):
        o_target_traced = map_a_to_b.get(otarget)
        disp_same = otext.strip() == ttext.strip()
        if disp_same:
            same_disp += 1
        if o_target_traced is None:
            mismatched.append((oi, okind, otarget, otext, ttarget, ttext,
                               "target instruction was rewritten or dropped"))
            continue
        if o_target_traced != ttarget:
            mismatched.append((oi, okind, otarget, otext, ttarget, ttext,
                               f"origin target maps to T[{o_target_traced}]"))
        else:
            moved += 1

    print(f"same displacement text: {same_disp}/{len(by_kind_o)}")
    print(f"lands on the same instruction: {moved}/{len(by_kind_o)}")
    if mismatched:
        print("\nmismatches:")
        for oi, okind, otarget, otext, ttarget, ttext, why in mismatched[:20]:
            print(f"  O[{oi}] {okind}: origin target O[{otarget}] {why}; "
                  f"traced reaches T[{ttarget}]")
            print(f"      origin: {otext.strip()}")
            print(f"      traced: {ttext.strip()}")

    # how many instructions does each branch skip, before and after?
    print("\nspan (instructions skipped) per transfer, origin -> traced:")
    spans = Counter()
    for (oi, okind, otext, otarget), (ti, tkind, ttext, ttarget) in zip(by_kind_o, by_kind_t):
        spans[(otarget - oi, ttarget - ti)] += 1
    for (o_span, t_span), count in sorted(spans.items()):
        print(f"  origin span {o_span:6d} -> traced span {t_span:6d}   x{count}")


if __name__ == "__main__":
    main()
