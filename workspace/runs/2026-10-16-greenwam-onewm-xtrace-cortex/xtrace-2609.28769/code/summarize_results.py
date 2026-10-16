#!/usr/bin/env python3
"""Collect every static-splice result in this packet's results/ into one table.

Reads the `static_*.json` files written by `run_static_sweep.sh` (each records one kernel's
splice, its fidelity metrics and its search reports), re-runs the origin-vs-spliced binary
comparison for each pair so that byte-level and control-flow numbers land beside the
text-level ones, and writes `results/summary.json` plus a markdown table on stdout.

Usage: summarize_results.py [--json results/summary.json]
"""

import argparse
import glob
import io
import json
import os
import sys
from collections import Counter
from contextlib import redirect_stdout

import compare_binaries

CUBIN_ROOTS = [os.path.expanduser("~/.triton/cache")]


def origin_for(tag_kernel, tag_cid):
    """The cached cubin whose directory hash starts with `tag_cid`."""
    for root in CUBIN_ROOTS:
        for path in glob.glob(os.path.join(root, "*", "*.cubin")):
            if os.path.basename(path)[:-len(".cubin")] != tag_kernel:
                continue
            if os.path.basename(os.path.dirname(path)).startswith(tag_cid):
                return path
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/summary.json")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob("results/static_*_plain.json")):
        stem = os.path.basename(path)[len("static_"):-len("_plain.json")]
        # the tag is <kernel>_<cache-dir-prefix>_<spec>_plain, and every:8 became every_8
        for spec, suffix in (("every:8", "_every_8"), ("mem", "_mem")):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        else:
            raise SystemExit(f"unrecognised tag: {stem}")
        kernel, tag_cid = stem.rsplit("_", 1)
        data = json.load(open(path))
        origin = origin_for(kernel, tag_cid)
        row = {
            "kernel": data["kernel"],
            "cache_dir": tag_cid,
            "sites": data["site_spec"],
            "splice_mode": data.get("splice_mode"),
            "origin_instructions": data["origin_instructions"],
            "origin_registers": data["origin_registers"],
            "sites_requested": data["sites_requested"],
            "sites_spliced": data["sites_spliced"],
            "sites_failed": data["sites_failed"],
            "failed_site_indices": data["failed_site_indices"],
            "probe_instructions_total": data["probe_instructions_total"],
            "probe_instruction_counts": data["probe_instruction_counts"],
            "analysis_ns_median": (data.get("analysis_ns") or {}).get("median"),
            "search_ns_median": (data.get("search_ns") or {}).get("median"),
            "makespan_median": (data.get("effective_makespan_cycles") or {}).get("median"),
            "makespan_max": (data.get("effective_makespan_cycles") or {}).get("max"),
            "register_class": data.get("register_class"),
            "text": data["similarity"],
            "spliced_cubin": data.get("spliced_cubin"),
            "origin_cubin": origin,
        }
        if origin and data.get("spliced_cubin") and os.path.exists(data["spliced_cubin"]):
            buffer = io.StringIO()
            saved = sys.argv
            sys.argv = ["compare_binaries.py", origin, data["spliced_cubin"]]
            try:
                with redirect_stdout(buffer):
                    comparison = compare_binaries.main()
            finally:
                sys.argv = saved
            row["bytes"] = {
                "text_similarity": comparison["text_similarity"],
                "byte_similarity": comparison["byte_similarity"],
                "control_bit_only_changes": comparison["control_bit_only_changes"],
                "byte_changed_by_opcode": comparison["byte_changed_by_opcode"],
                "byte_changed_but_text_same_opcodes":
                    comparison["byte_changed_but_text_same_opcodes"],
                "text_changed_by_opcode": comparison["text_changed_by_opcode"],
                "intact_blocks": comparison["intact_blocks"],
                "origin_blocks": comparison["origin_blocks"],
                "changed_blocks": comparison["changed_blocks"],
                "blocks_with_probe_inside": comparison["blocks_with_probe_inside"],
                "traced_registers": comparison["traced_registers"],
            }
        rows.append(row)

    with open(args.json, "w") as f:
        json.dump(rows, f, indent=2)

    print("| kernel | instrs | regs | sites spliced | probe instrs | text sim | byte sim | "
          "control-bit-only | changed blocks |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        byte = row.get("bytes") or {}
        print(f"| {row['kernel'][:24]} | {row['origin_instructions']} | {row['origin_registers']} | "
              f"{row['sites_spliced']}/{row['sites_requested']} | {row['probe_instructions_total']} | "
              f"{row['text']['binary_similarity']:.4f} | "
              f"{byte.get('byte_similarity', float('nan')):.4f} | "
              f"{byte.get('control_bit_only_changes', '')} | "
              f"{byte.get('changed_blocks', '')}/{byte.get('origin_blocks', '')} |")

    totals = Counter()
    control_bits = Counter()
    for row in rows:
        byte = row.get("bytes") or {}
        for opcode, count in (byte.get("byte_changed_by_opcode") or {}).items():
            totals[opcode] += count
        for opcode, count in (byte.get("byte_changed_but_text_same_opcodes") or {}).items():
            control_bits[opcode] += count
    print(f"\nbyte-level changes by opcode across {len(rows)} splices: {dict(totals.most_common())}")
    print(f"of which control-bit-only changes: {dict(control_bits.most_common())}")
    print(f"totals: {sum(totals.values())} byte-level, {sum(control_bits.values())} control-bit-only")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
