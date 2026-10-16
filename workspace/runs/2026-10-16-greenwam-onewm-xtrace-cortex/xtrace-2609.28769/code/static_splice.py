#!/usr/bin/env python3
"""Splice a full probe set into a compiled cubin and measure what the binary became.

This is the paper's section 5.3 measurement (binary similarity) performed on sm_86 with the
shipped tool, and it needs no GPU: the splicer is a static binary rewriter, so the artifact it
produces -- the instrumented cubin -- can be compared with the origin instruction by
instruction.

The orchestration mirrors `gwatch.cuda.xtrace._sass_instrumenter.TraceInstrumenter_SASS`:

  * the trace-buffer pointer pair is allocated over the pointer's whole live range
    (kernel entry -> every site), preferring per-warp uniform registers
  * the pointer-init prologue is spliced at instruction index 1, each site's probe at its own
    index, deepest pc first, so that inserting at a pc only shifts deeper pcs
  * a site's probe is either the monolithic store (`plain`) or the tool's split splice
    (`auto`/`split`: capture at the site, record store sunk to a low-contention point in the
    same basic block), which is the shipped default

Usage:
    static_splice.py <origin.cubin> --sites mem|sche|every:N|list:1,2,3 [--out spliced.cubin]
"""

import argparse
import json
import os
import statistics

import gwatch.cuda.binary as gw_binary
from gwatch.cuda.assemble import InstrumentOperator, TimestampKind
from gwatch.cuda.xtrace._store_sink import instrument_site_split, store_operators

import sasslib

_RECORD_BYTES = 16
_INIT_INDEX = 1


def pick_sites(view: sasslib.KernelView, spec: str):
    if spec == "mem":
        return sasslib.find_memory_sites(view)
    if spec == "sche":
        exits = [i for i, t in enumerate(view.texts) if sasslib.opcode_of(t) == "EXIT"]
        return [1] + exits
    if spec.startswith("every:"):
        step = max(1, int(spec.split(":", 1)[1]))
        return list(range(1, view.n, step))
    if spec.startswith("list:"):
        return [int(x) for x in spec.split(":", 1)[1].split(",")]
    raise SystemExit(f"unknown site spec {spec!r}")


def splice(cubin_path: str, site_indices, splice_mode="auto", schedule_mode=0,
           per_thread_records=64, exclude_registers=None, out_path=None):
    cubin = gw_binary.Cubin()
    cubin.fill(cubin_path)
    cubin.parse()
    name = next(iter(cubin.get_map_kernel_def()))
    kd = cubin.get_kerneldef_by_name(name)
    kd.parse_cfg()
    kd.parse_register_liveness()
    isize = kd.cfg.instruction_size
    for reg_class, regs in (exclude_registers or {}).items():
        kd.exclude_instrument_registers(reg_class, list(regs))

    origin_after_parse = [i.str() for i in kd.get_list_instruction()]
    origin_regs = int(kd.params.num_registers)

    stride = _RECORD_BYTES * per_thread_records
    init_pc = _INIT_INDEX * isize
    site_pcs = [i * isize for i in site_indices]
    live_pcs = sorted(kd.collect_live_range_pcs(init_pc, site_pcs))

    reg_class_used = None
    last_error = ""
    for reg_class in ("uniform_general", "general"):
        try:
            pair = kd.collect_allocatable_register_pairs(reg_class, live_pcs, 2)
            reg_class_used = reg_class
            break
        except RuntimeError as exc:
            last_error = str(exc)
    if reg_class_used is None:
        return {"spliced": False, "stage": "pointer-allocation", "error": last_error,
                "live_pcs": len(live_pcs)}
    is_uniform = 1 if reg_class_used == "uniform_general" else 0
    record_ptr_reg, thread_limit_reg = pair[0], pair[2]

    actions = [("site", pc, site_indices[k]) for k, pc in enumerate(site_pcs)]
    actions.append(("init", init_pc, _INIT_INDEX))
    actions.sort(key=lambda a: (a[1], a[0] == "site"), reverse=True)

    site_reports = {}
    probe_pcs = {}
    for kind, pc, insert_index in actions:
        if kind == "init":
            with kd.create_instrument_context(pc, schedule_mode=schedule_mode) as ctx:
                ctx.operator(InstrumentOperator.INIT_TRACE_POINTER, {
                    "param_index": 0,
                    "stride": stride,
                    "trace_ptr_reg": record_ptr_reg,
                    "thread_limit_reg": thread_limit_reg,
                    "trace_ptr_is_uniform": is_uniform,
                })
                probe_pcs["init"] = list(ctx.finalize())
                reports = [dict(ctx.report)]
            site_reports["init"] = {"spliced": "init", "insert_index": _INIT_INDEX,
                                    "splices": reports}
            continue

        site_id = insert_index
        if splice_mode in ("auto", "split"):
            _sink, blocks, reports = instrument_site_split(
                kd, isize, insert_index, record_ptr_reg, thread_limit_reg, site_id, stride,
                int(TimestampKind.GLOBALTIMER), _RECORD_BYTES,
                trace_ptr_is_uniform=is_uniform,
                schedule_mode=schedule_mode,
            )
            if blocks:
                probe_pcs[f"site{insert_index}"] = [p for block in blocks for p in block]
                site_reports[insert_index] = {"spliced": "split", "sink_index": _sink,
                                             "insert_index": insert_index, "splices": reports}
                continue
            if splice_mode == "split":
                site_reports[insert_index] = {"spliced": False, "insert_index": insert_index,
                                              "error": "no carrier free over window"}
                continue

        staged = False
        for store_op in store_operators(is_uniform):
            try:
                with kd.create_instrument_context(pc, schedule_mode=schedule_mode) as ctx:
                    ctx.operator(store_op, {
                        "trace_ptr_reg": record_ptr_reg,
                        "thread_limit_reg": thread_limit_reg,
                        "trace_ptr_is_uniform": is_uniform,
                        "value_reg": 255,
                        "value_width_bits": 32,
                        "value_is_uniform": 0,
                        "value_is_smid": 0,
                        "site_id": site_id,
                        "ring_wrap": 1,
                        "stride": stride,
                        "timestamp_kind": int(TimestampKind.GLOBALTIMER),
                        "advance": _RECORD_BYTES,
                    })
                    probe_pcs[f"site{insert_index}"] = list(ctx.finalize())
                    site_reports[insert_index] = {
                        "spliced": "plain", "insert_index": insert_index,
                        "splices": [dict(ctx.report, splice="store")]}
                staged = True
                break
            except RuntimeError as exc:
                last_error = str(exc)
        if not staged:
            site_reports[insert_index] = {"spliced": False, "insert_index": insert_index,
                                          "error": last_error}

    after = [i.str() for i in kd.get_list_instruction()]
    traced_regs = int(kd.params.num_registers)

    # serialize the spliced kernel and re-disassemble it: the artifact compared against the
    # origin is the image a driver would actually load, not an in-memory kernel object
    import tempfile
    out_path = out_path or tempfile.mktemp(suffix=".cubin")
    serialized_bytes = None
    serialization_error = None
    traced_view = None
    try:
        image = gw_binary.Cubin()
        image.add_kerneldef(kd)
        blob = image.serialize()
        with open(out_path, "wb") as f:
            f.write(blob)
        serialized_bytes = len(blob)
        traced_view = sasslib.load_kernels(out_path).get(name)
    except Exception as exc:  # serialize can reject an architecture-family image
        serialization_error = str(exc)

    origin_view = sasslib.load_kernels(cubin_path)[name]
    if traced_view is not None:
        sim = sasslib.similarity(origin_view.texts, traced_view.texts)
        sim.update(sasslib.block_changes(origin_view, traced_view))
        sim["traced_registers"] = traced_view.num_registers
        sim["traced_basic_blocks"] = traced_view.nb_basic_blocks
    else:
        sim = sasslib.similarity(origin_view.texts, after)
        sim["serialization_error"] = serialization_error

    spliced_sites = [k for k, v in site_reports.items() if v.get("spliced") in ("plain", "split")]
    failed_sites = [k for k, v in site_reports.items() if v.get("spliced") is False]
    search_ns = [r["search_ns"] for v in site_reports.values() for r in v.get("splices", [])
                 if isinstance(r, dict) and "search_ns" in r]
    analysis_ns = [r["analysis_ns"] for v in site_reports.values() for r in v.get("splices", [])
                   if isinstance(r, dict) and "analysis_ns" in r]
    makespans = [r["effective_makespan"] for v in site_reports.values()
                 for r in v.get("splices", []) if isinstance(r, dict) and "effective_makespan" in r]

    return {
        "kernel": name,
        "spliced": True,
        "origin_instructions": len(origin_view.texts),
        "origin_registers": origin_regs,
        "traced_instructions_after_splice": len(after),
        "traced_registers": traced_regs,
        "register_class": reg_class_used,
        "pointer_registers": [record_ptr_reg, thread_limit_reg],
        "sites_requested": len(site_indices),
        "site_instruction_indices": list(site_indices),
        "sites_spliced": len(spliced_sites),
        "sites_failed": len(failed_sites),
        "failed_site_indices": failed_sites[:20],
        "probe_instructions_total": len(after) - len(origin_view.texts),
        "probe_instruction_counts": sorted({len(p) for p in probe_pcs.values()}),
        "similarity": sim,
        "serialized_bytes": serialized_bytes,
        "serialization_error": serialization_error,
        "spliced_cubin": out_path,
        "search_ns": {"median": statistics.median(search_ns), "n": len(search_ns)} if search_ns else None,
        "analysis_ns": {"median": statistics.median(analysis_ns), "n": len(analysis_ns)} if analysis_ns else None,
        "effective_makespan_cycles": {"median": statistics.median(makespans),
                                      "max": max(makespans)} if makespans else None,
        "probe_pcs": probe_pcs,
        "site_reports": site_reports,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cubin")
    ap.add_argument("--sites", default="mem")
    ap.add_argument("--splice-mode", default="auto", choices=["auto", "split", "plain"])
    ap.add_argument("--schedule-mode", default="first_feasible", choices=["first_feasible", "least_makespan"])
    ap.add_argument("--records", type=int, default=64)
    ap.add_argument("--out", default=None, help="where to write the spliced cubin")
    ap.add_argument("--json", default=None, help="where to write the result summary")
    ap.add_argument("--exclude-register", action="append", default=[],
                    help="general register id to withhold from instrumentation (repeatable)")
    args = ap.parse_args()

    view = next(iter(sasslib.load_kernels(args.cubin).values()))
    indices = pick_sites(view, args.sites)
    result = splice(
        args.cubin, indices,
        splice_mode=args.splice_mode,
        schedule_mode=1 if args.schedule_mode == "least_makespan" else 0,
        per_thread_records=args.records,
        exclude_registers={"general": [int(x) for x in args.exclude_register]} if args.exclude_register else None,
        out_path=args.out,
    )
    result["site_spec"] = args.sites
    result["splice_mode"] = args.splice_mode
    result["schedule_mode"] = args.schedule_mode
    result["arch"] = view.arch_version
    result["nb_instructions"] = view.n
    result["nb_basic_blocks_before"] = view.nb_basic_blocks

    summary = {k: v for k, v in result.items() if k not in ("probe_pcs", "site_reports")}
    print(json.dumps(summary, indent=2, default=str))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    main()
