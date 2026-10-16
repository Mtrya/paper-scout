#!/usr/bin/env python3
"""End-to-end Xtrace run on sm_86: splice probes into a real kernel's cubin, run it, measure.

The target is vLLM's rotary embedding kernel, taken as the cubin Triton already compiled into
the local cache, and launched through the CUDA driver API (`cuda_launch.Launch`). That launch
path matters: gwatch's capsule hooks the classic `cuModuleLoad` / `cuLaunchKernel` API, which is
what its scout and its launch hijacker need, and which Triton 3.6 no longer uses.

What one run reports:

  * `timing.baseline`  -- uninstrumented launches, before anything is spliced
  * `timing.scout`     -- the same kernel under the scout pass that captures the launch
  * `timing.traced`    -- launches of the spliced cubin, registered in place of the origin
  * `correctness`      -- the spliced kernel's output vs the uninstrumented kernel's, bitwise
  * `binary_similarity`-- the kept spliced cubin vs the origin cubin (text, bytes, control flow)
  * `trace_results`    -- the records the probes wrote, with span statistics per site

Usage:
    source gwenv.sh
    python probe_rotary.py --sites mem --reps 40 --out results/rotary_mem.json
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

import cuda_launch
import sasslib
from gwatch.cuda.xtrace import do_trace, SassTraceSite
from gwatch.cuda.xtrace.format import Section_IntraKernelTrace
from gwatch.common.format import File
import gwatch.cuda.profile as gw_profile

KERNEL = "rotary_kernel"
CUBIN = os.path.expanduser(
    "~/.triton/cache/YGPFJAPL5ZDQ2SQAGOCZ3FCEYS3AKJOG6TIVYO2YXWIUU7GFXQHA"
    "/rotary_kernel.cubin")


def time_block(launch, reps, warmup=5):
    """Per-launch wall times in milliseconds, each launch bounded by its own event pair."""
    import cuda.bindings.driver as cu
    for _ in range(warmup):
        launch.launch_packed()
    launch.synchronize()
    samples = []
    for _ in range(reps):
        start, end = launch.new_event(), launch.new_event()
        cu.cuEventRecord(start, launch.stream)
        launch.launch_packed()
        cu.cuEventRecord(end, launch.stream)
        launch.synchronize()
        samples.append(launch.elapsed(start, end))
    return samples


def stats(samples):
    xs = sorted(samples)
    return {"n": len(xs), "median_ms": statistics.median(xs), "min_ms": xs[0],
            "max_ms": xs[-1], "mean_ms": statistics.fmean(xs)}


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


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="sche")
    ap.add_argument("--max-sites", type=int, default=24)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--records", type=int, default=32)
    ap.add_argument("--clock", default="gpu", choices=["gpu", "sm", "anchor"])
    ap.add_argument("--splice-mode", default="auto", choices=["auto", "split", "plain"])
    ap.add_argument("--schedule-mode", default="first_feasible",
                    choices=["first_feasible", "least_makespan"])
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--cubin", default=CUBIN)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workdir", default="/tmp/xtrace-rotary")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)

    # the trace context has to exist before the module is loaded, or the capsule never sees it
    ctx = gw_profile.ProfileContext()
    launch = cuda_launch.Launch(cubin_path=args.cubin, kernel_name=KERNEL, seqlen=args.seqlen)
    origin = sasslib.load_kernels(args.cubin)[KERNEL]
    # the launch must carry the argument struct in CU_LAUNCH_PARAM_BUFFER_* extras: that is the
    # form gwatch's launch hijacker can append the trace-buffer pointer to
    cubin_kd = _kernel_def(args.cubin)
    launch.param_offsets = sorted(cubin_kd.list_param_offsets_reversed)
    print(f"kernel params at offsets {launch.param_offsets}")

    launch.launch_packed()
    launch.synchronize()
    baseline_out = launch.buffers["out"].download()

    sites = choose_sites(origin, args.sites, args.max_sites)
    print(f"kernel {KERNEL}: {origin.n} instructions, {origin.num_registers} registers, "
          f"{len(sites)} trace sites")

    baseline = time_block(launch, args.reps)

    kept = os.path.join(args.workdir, f"spliced_{args.sites.replace(':', '_')}.cubin")
    if os.path.exists(kept):
        os.remove(kept)
    os.environ["GWATCH_TRACE_KEEP_CUBIN"] = kept

    calls = {"n": 0}
    timings = {}

    def timed_fn():
        index = calls["n"]
        calls["n"] += 1
        timings[index] = time_block(launch, args.reps, warmup=0)

    section = Section_IntraKernelTrace()
    t0 = time.time()
    failure = None
    try:
        result = do_trace(
            fn=timed_fn,
            kernel_name_pattern=KERNEL,
            instrumentation_tier="sass",
            clock_type=args.clock,
            sass_trace_sites=sites,
            sass_per_thread_records=args.records,
            sass_splice_mode=args.splice_mode,
            sass_schedule_mode=args.schedule_mode,
            profile_context=ctx,
        )
    except Exception as exc:  # keep the numbers we already have; report what broke
        result = None
        failure = f"{type(exc).__name__}: {exc}"
    wall = time.time() - t0

    try:
        launch.synchronize()
    except Exception as exc:
        failure = failure or f"sync after traced run: {exc}"
    traced_out = launch.buffers["out"].download()
    records = (result or {}).get("trace_results") or []

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
        "origin": {"instructions": origin.n, "registers": origin.num_registers,
                   "basic_blocks": origin.nb_basic_blocks},
        "trace_ran": bool(result),
        "failure": failure,
        "trace_records": len(records),
        "compile_results": (result or {}).get("compile_results"),
        "splice_results": (result or {}).get("splice_results"),
        "timing": {
            "baseline": stats(baseline),
            "scout": stats(timings.get(0, [0.0])),
            "traced": stats(timings.get(1, [0.0])),
            "trace_wall_seconds": wall,
        },
        "correctness": {
            "traced_equals_baseline_bitwise": bool(np.array_equal(traced_out, baseline_out)),
            "baseline_nonzero_outputs": int((baseline_out != 0).sum()),
            "traced_nonzero_outputs": int((traced_out != 0).sum()),
        },
    }
    if out["trace_ran"]:
        out["timing"]["overhead_percent"] = (
            out["timing"]["traced"]["median_ms"] / out["timing"]["baseline"]["median_ms"] - 1.0
        ) * 100.0
        out["timing"]["scout_overhead_percent"] = (
            out["timing"]["scout"]["median_ms"] / out["timing"]["baseline"]["median_ms"] - 1.0
        ) * 100.0

    if os.path.exists(kept):
        out["binary_similarity"] = _compare(args.cubin, kept)
        out["spliced_cubin"] = kept

    if records:
        out["record_stats"] = summarize_records(records, sites)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(jsonable(out), f, indent=2)
        section.add_run(result)
        report = File(title=f"Xtrace rotary trace — {args.sites}")
        report.add_section(section)
        report.render(args.out.replace(".json", ".html"))
        out["report"] = args.out.replace(".json", ".html")

    print(json.dumps(jsonable(out), indent=2))


def _kernel_def(cubin_path):
    import gwatch.cuda.binary as gw_binary
    cubin = gw_binary.Cubin()
    cubin.fill(cubin_path)
    cubin.parse()
    return cubin.get_kerneldef_by_name(KERNEL)


def _compare(origin_path, spliced_path):
    """The origin-vs-spliced comparison of compare_binaries, captured instead of printed."""
    import io
    from contextlib import redirect_stdout
    import compare_binaries
    saved_argv = sys.argv
    sys.argv = ["compare_binaries.py", origin_path, spliced_path]
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            return compare_binaries.main()
    finally:
        sys.argv = saved_argv


def summarize_records(records, sites):
    """Per site: how often it fired and the span between its first and last record."""
    by_site = {}
    for record in records:
        site = record["site_id"]
        by_site.setdefault(site, []).append(record["timestamp"])
    out = {}
    for site, stamps in sorted(by_site.items()):
        stamps.sort()
        out[str(site)] = {
            "records": len(stamps),
            "first_ns": stamps[0],
            "last_ns": stamps[-1],
            "span_ns": stamps[-1] - stamps[0],
            "median_gap_ns": statistics.median(
                [b - a for a, b in zip(stamps, stamps[1:])]) if len(stamps) > 1 else None,
        }
    return out


if __name__ == "__main__":
    main()
