#!/usr/bin/env python3
"""Splice, launch and measure without gwatch's launch hijacker in the loop.

`probe_rotary.py` drives the whole toolkit, including the piece that swaps the spliced image
in at launch time. That swap faults on this machine (see the packet README), so this script
takes the same splicing and record-decoding code and does the launch itself:

  1. splice the probes into the cubin with the tool's own `TraceInstrumenter_SASS`, which also
     appends the trace-buffer pointer as the kernel's last argument
  2. serialize that image, load it as a module, and launch it with the argument struct the
     tool's own spliced cubin declares -- the buffer pointer included
  3. read the trace buffer back through the tool's own `collect()` and the timing through CUDA
     events around each launch

Nothing here is a reimplementation of the paper's mechanism: the probe instructions, their
registers, their schedule and the record decoding all come from gwatch. What is ours is the
launch plumbing.

Usage:
    source gwenv.sh
    python selflaunch_trace.py --sites mem --max-sites 16 --reps 40 --out results/self_mem.json
"""

import argparse
import json
import os
import statistics
import time

import numpy as np

import cuda_launch
import sasslib
from probe_rotary import _compare, jsonable, stats, time_block

import gwatch.cuda.binary as gw_binary
from gwatch.cuda.xtrace._buffer import TraceBuffer
from gwatch.cuda.xtrace._schema import TraceSchema_SASS, SassTraceSite
from gwatch.cuda.xtrace._sass_instrumenter import TraceInstrumenter_SASS

KERNEL = "rotary_kernel"
CUBIN = os.path.expanduser(
    "~/.triton/cache/YGPFJAPL5ZDQ2SQAGOCZ3FCEYS3AKJOG6TIVYO2YXWIUU7GFXQHA"
    "/rotary_kernel.cubin")


def choose_sites(view, mode, max_sites):
    if mode == "sche":
        exits = [i for i, t in enumerate(view.texts) if sasslib.opcode_of(t) == "EXIT"]
        return [SassTraceSite(insert_index=1, value_is_smid=True)] + \
               [SassTraceSite(insert_index=i) for i in exits]
    if mode == "mem":
        return [SassTraceSite(insert_index=i)
                for i in sasslib.find_memory_sites(view)][:max_sites]
    if mode.startswith("every:"):
        step = int(mode.split(":")[1])
        return [SassTraceSite(insert_index=i) for i in range(1, view.n, step)][:max_sites]
    raise SystemExit(f"unknown site mode {mode!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="sche")
    ap.add_argument("--max-sites", type=int, default=16)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--records", type=int, default=16)
    ap.add_argument("--clock", default="gpu", choices=["gpu", "sm", "anchor"])
    ap.add_argument("--splice-mode", default="auto", choices=["auto", "split", "plain"])
    ap.add_argument("--schedule-mode", default="first_feasible",
                    choices=["first_feasible", "least_makespan"])
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--cubin", default=CUBIN)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workdir", default="/tmp/xtrace-rotary")
    ap.add_argument("--warps", type=int, default=4, help="warps per block")
    ap.add_argument("--grid", default=None, help="override the launch grid, e.g. 1,1,1")
    ap.add_argument("--buffer-scale", type=float, default=1.0,
                    help="multiply the trace buffer size (oversizing hides an id that overshoots)")
    ap.add_argument("--force-general", action="store_true",
                    help="withhold every uniform register, forcing the general-register probe")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)

    launch = cuda_launch.Launch(cubin_path=args.cubin, kernel_name=KERNEL, seqlen=args.seqlen,
                                num_warps=args.warps)
    if args.grid:
        launch.grid = tuple(int(x) for x in args.grid.split(","))
    origin_view = sasslib.load_kernels(args.cubin)[KERNEL]

    origin_cubin = gw_binary.Cubin()
    origin_cubin.fill(args.cubin)
    origin_cubin.parse()
    origin_kd = origin_cubin.get_kerneldef_by_name(KERNEL)
    launch.param_offsets = sorted(origin_kd.list_param_offsets_reversed)

    # baseline: the uninstrumented kernel, its output, and its timing
    launch.launch_packed()
    launch.synchronize()
    baseline_out = launch.buffers["out"].download()
    baseline = time_block(launch, args.reps)

    sites = choose_sites(origin_view, args.sites, args.max_sites)
    print(f"{KERNEL}: {origin_view.n} instructions, {origin_view.num_registers} registers, "
          f"{len(sites)} sites; baseline median {statistics.median(baseline)*1000:.1f} us")

    # splice with the tool's instrumenter, which also appends the trace-pointer argument
    cubin = gw_binary.Cubin()
    cubin.fill(args.cubin)
    cubin.parse()
    kd = cubin.get_kerneldef_by_name(KERNEL)
    kd.parse_cfg()
    kd.parse_register_liveness()
    if args.force_general:
        kd._gw_instance.exclude_instrument_registers("uniform_general", list(range(64)))
    schema = TraceSchema_SASS(
        sites=sites,
        grid_dim=launch.grid,
        block_dim=launch.block,
        clock_type=args.clock,
        per_thread_records=args.records,
    )
    buffer_bytes = int(schema.total_bytes * args.buffer_scale)
    print(f"grid {launch.grid} block {launch.block}: schema stride {schema.stride_bytes} B, "
          f"records {schema.per_thread_records}, buffer {buffer_bytes} B")
    buffer = TraceBuffer(size_bytes=buffer_bytes, device=0)
    instrumenter = TraceInstrumenter_SASS(
        kernel_prototype=KERNEL,
        kernel_def_sass=kd,
        cubin=cubin,
        trace_buffer=buffer,
        trace_schema=schema,
        clock_type=args.clock,
        splice_mode=args.splice_mode,
        schedule_mode=1 if args.schedule_mode == "least_makespan" else 0,
    )
    t0 = time.time()
    instrumenter.instrument()
    splice_seconds = time.time() - t0

    spliced_path = os.path.join(args.workdir, f"self_{args.sites.replace(':', '_')}.cubin")
    image = gw_binary.Cubin()
    image.add_kerneldef(instrumenter.kernel_def_sass)
    with open(spliced_path, "wb") as f:
        f.write(image.serialize())

    spliced_kd = sasslib.load_kernels(spliced_path)[KERNEL]
    spliced_cubin = gw_binary.Cubin()
    spliced_cubin.fill(spliced_path)
    spliced_cubin.parse()
    spliced_offsets = sorted(spliced_cubin.get_kerneldef_by_name(KERNEL).list_param_offsets_reversed)
    print(f"spliced image: {spliced_kd.n} instructions, registers {spliced_kd.num_registers}, "
          f"params {len(spliced_offsets)} (last at {spliced_offsets[-1]})")

    # the spliced image is the module; the trace pointer rides in the argument struct
    trace_launch = cuda_launch.Launch(cubin_path=spliced_path, kernel_name=KERNEL,
                                      seqlen=args.seqlen, seed=0)
    trace_launch.buffers = launch.buffers          # same inputs and outputs as the baseline
    trace_launch.param_offsets = spliced_offsets
    trace_launch.extra_param_value = int(buffer.data_ptr)

    traced = time_block(trace_launch, args.reps)
    trace_launch.synchronize()
    traced_out = trace_launch.buffers["out"].download()
    records = instrumenter.collect()

    out = {
        "device": "NVIDIA GeForce RTX 3060",
        "arch": "sm_86",
        "kernel": KERNEL,
        "cubin": args.cubin,
        "site_mode": args.sites,
        "num_sites": len(sites),
        "site_instruction_indices": [s.insert_index for s in sites],
        "clock_type": args.clock,
        "splice_mode": args.splice_mode,
        "schedule_mode": args.schedule_mode,
        "per_thread_records": args.records,
        "grid": list(launch.grid),
        "block": list(launch.block),
        "origin": {"instructions": origin_view.n, "registers": origin_view.num_registers,
                   "basic_blocks": origin_view.nb_basic_blocks},
        "spliced": {"instructions": spliced_kd.n, "registers": spliced_kd.num_registers,
                    "basic_blocks": spliced_kd.nb_basic_blocks,
                    "params": len(spliced_offsets)},
        "trace_buffer_bytes": buffer_bytes,
        "schema_total_bytes": schema.total_bytes,
        "splice_seconds": splice_seconds,
        "splice_reports": {str(k): v for k, v in instrumenter.map_site_reports.items()},
        "probe_pcs": {str(k): v for k, v in instrumenter.map_site_probe_pcs.items()},
        "trace_records": len(records),
        "timing": {"baseline": stats(baseline), "traced": stats(traced)},
        "correctness": {
            "traced_equals_baseline_bitwise": bool(np.array_equal(traced_out, baseline_out)),
            "baseline_nonzero_outputs": int((baseline_out != 0).sum()),
            "traced_nonzero_outputs": int((traced_out != 0).sum()),
        },
        "binary_similarity": _compare(args.cubin, spliced_path),
        "spliced_cubin": spliced_path,
    }
    out["timing"]["overhead_percent"] = (
        out["timing"]["traced"]["median_ms"] / out["timing"]["baseline"]["median_ms"] - 1.0
    ) * 100.0
    if records:
        out["record_stats"] = summarize(records)

    buffer.destroy()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(jsonable(out), f, indent=2)
    print(json.dumps(jsonable({k: v for k, v in out.items()
                               if k not in ("probe_pcs", "splice_reports")}), indent=2))


def summarize(records):
    """Per site: how often it fired, over how many threads, and the span it brackets."""
    by_site = {}
    for record in records:
        by_site.setdefault(record["site_id"], []).append(record["timestamp"])
    out = {}
    for site, stamps in sorted(by_site.items()):
        stamps.sort()
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        out[str(site)] = {
            "records": len(stamps),
            "threads": len(stamps),
            "first_ns": stamps[0],
            "last_ns": stamps[-1],
            "kernel_span_ns": stamps[-1] - stamps[0],
            "median_gap_ns": statistics.median(gaps) if gaps else None,
            "min_gap_ns": min(gaps) if gaps else None,
        }
    return out


if __name__ == "__main__":
    main()
