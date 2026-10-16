#!/usr/bin/env python3
"""Trace a real kernel with Xtrace's SASS tier on sm_86 and measure its fidelity and cost.

What one run of this script does, in order:

  1. compile and run the target kernel, keep its launch in a gwatch profile context and
     dump the exact cubin the launch used (the *origin* binary)
  2. disassemble the origin with gwatch's own decoder and pick the trace sites from it
  3. time the uninstrumented kernel (baseline)
  4. run `do_trace(instrumentation_tier="sass", ...)`, which splices the probes into the
     cubin and runs the kernel once more with the spliced image registered under the
     origin's name; the spliced image is kept on disk through GWATCH_TRACE_KEEP_CUBIN
  5. disassemble the spliced cubin and compute the paper's binary-similarity metric
     against the origin, per-instruction, with gwatch's own disassembler
  6. compare the spliced kernel's numeric output against the baseline output
  7. time the instrumented kernel over repeated launches and report the median ratio

`--tier sass` is the paper's mechanism (probe spliced into the compiled cubin).
`--tier ptx` is the control: the same toolkit's PTX tier, which splices into the PTX and
JIT-compiles, i.e. the class of pre-compilation instrumentation the paper argues against;
its binary is compared the same way.
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch

import gwatch.cuda.profile as gw_profile
import gwatch.cuda.binary as gw_binary
import sasslib
import targets

from gwatch.cuda.xtrace import do_trace, SassTraceSite
from gwatch.cuda.xtrace.format import Section_IntraKernelTrace


def time_block(launch, reps: int, warmup: int = 5):
    """Per-launch wall times of ``reps`` back-to-back launches, in milliseconds."""
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts[i].record()
        launch()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def stats(samples):
    xs = sorted(samples)
    return {
        "n": len(xs),
        "median_ms": statistics.median(xs),
        "min_ms": xs[0],
        "max_ms": xs[-1],
        "mean_ms": statistics.fmean(xs),
    }


def discover_origin_cubin(launch, kernel_regex, dump_path):
    """Run the kernel once under a scout context and dump the cubin that launch used."""
    ctx = gw_profile.ProfileContext()
    ctx.start_tracing_kernel_launch()
    launch()
    torch.cuda.synchronize()
    kernels = ctx.stop_tracing_kernel_launch()
    names = [k.definition.mangled_prototype for k in kernels]
    matched = [k for k in kernels if kernel_regex in k.definition.mangled_prototype]
    if not matched:
        raise SystemExit(f"no launch matched {kernel_regex!r}; saw {names}")
    target = matched[0]
    name = target.definition.mangled_prototype
    cubin = None
    for candidate in ctx.get_list_binary_cubin():
        if name in candidate.get_map_kernel_def():
            cubin = candidate
            break
    if cubin is None:
        raise SystemExit(f"no loaded cubin carries {name!r}")
    cubin.dump(dump_path)
    return (
        ctx,
        name,
        (target.grid_dim_x, target.grid_dim_y, target.grid_dim_z),
        (target.block_dim_x, target.block_dim_y, target.block_dim_z),
        target.definition.params.arch_version,
    )


def choose_sites(view: sasslib.KernelView, mode: str, max_sites: int):
    """Trace sites as instruction indices into the origin kernel's own instruction list."""
    if mode == "sche":
        exits = [i for i, t in enumerate(view.texts) if sasslib.opcode_of(t) == "EXIT"]
        sites = [SassTraceSite(insert_index=1, value_is_smid=True)]
        sites += [SassTraceSite(insert_index=i) for i in exits]
        return sites
    if mode == "mem":
        indices = sasslib.find_memory_sites(view)
        return [SassTraceSite(insert_index=i) for i in indices[:max_sites]]
    if mode == "loop":
        # a spread of sites through the body of the kernel: every k-th instruction
        step = max(1, view.n // max_sites)
        return [SassTraceSite(insert_index=i) for i in range(1, view.n, step)][:max_sites]
    raise SystemExit(f"unknown site mode {mode!r}")


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="matmul", choices=["matmul", "attn"])
    ap.add_argument("--tier", default="sass", choices=["sass", "ptx"])
    ap.add_argument("--sites", default="sche", choices=["sche", "mem", "loop"])
    ap.add_argument("--max-sites", type=int, default=24)
    ap.add_argument("--warp-level", action="store_true", help="one record per warp, elected lane stores")
    ap.add_argument("--splice-mode", default="auto", choices=["auto", "split", "plain"])
    ap.add_argument("--schedule-mode", default="least_makespan", choices=["first_feasible", "least_makespan"])
    ap.add_argument("--records", type=int, default=64, help="per-thread record budget")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--clock", default="gpu", choices=["gpu", "sm", "anchor"])
    ap.add_argument("--out", default=None, help="where the JSON result lands")
    ap.add_argument("--workdir", default="/tmp/xtrace-probe")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)

    if args.kernel == "matmul":
        launch, tensors, grid, meta = targets.make_matmul()
        kernel_regex, out_index, ref_fn = "matmul", 2, None
    else:
        launch, tensors, grid, meta = targets.make_attn()
        kernel_regex, out_index, ref_fn = "attn", 3, targets.reference_attn

    # compile + first run
    launch()
    torch.cuda.synchronize()
    baseline_out = tensors[out_index].clone()

    # the exact cubin the launch used
    origin_cubin_path = os.path.join(args.workdir, "origin.cubin")
    ctx, kernel_name, grid_dim, block_dim, arch = discover_origin_cubin(
        launch, kernel_regex, origin_cubin_path
    )
    origin = sasslib.load_kernels(origin_cubin_path)[kernel_name]

    sites = choose_sites(origin, args.sites, args.max_sites)
    if args.warp_level:
        for site in sites:
            site.warp_level = True

    # baseline timing, before any instrumentation machinery is engaged
    base_samples = time_block(launch, args.reps)

    kept_cubin = os.path.join(args.workdir, f"{args.tier}_{args.sites}.cubin")
    if os.path.exists(kept_cubin):
        os.remove(kept_cubin)
    os.environ["GWATCH_TRACE_KEEP_CUBIN"] = kept_cubin

    # fn is called twice by do_trace: once to scout the launch, once with the kernel spliced
    calls = {"n": 0}
    timings = {}

    def timed_fn():
        index = calls["n"]
        calls["n"] += 1
        timings[index] = time_block(launch, args.reps, warmup=0)

    section = Section_IntraKernelTrace()
    t0 = time.time()
    result = do_trace(
        fn=timed_fn,
        kernel_name_pattern=kernel_regex,
        dsl="triton",
        instrumentation_tier=args.tier,
        clock_type=args.clock,
        sass_trace_sites=sites if args.tier == "sass" else None,
        sass_per_thread_records=args.records if args.tier == "sass" else None,
        sass_arch=None,
        sass_splice_mode=args.splice_mode,
        sass_schedule_mode=args.schedule_mode,
        scope_name_map=targets.SCOPE_NAMES if args.tier == "ptx" else None,
        ptx_warp_level=args.warp_level,
        profile_context=ctx,
    )
    trace_wall = time.time() - t0

    traced_out = tensors[out_index].clone()
    bit_equal = bool(torch.equal(traced_out, baseline_out))
    if ref_fn is not None:
        ref = ref_fn(*tensors[:3])
        max_ref_diff = float((traced_out.float() - ref.float()).abs().max())
    else:
        max_ref_diff = None

    out = {
        "device": device_name,
        "capability": f"sm_{capability[0]}{capability[1]}",
        "kernel": args.kernel,
        "kernel_name": kernel_name,
        "arch_version": str(arch),
        "tier": args.tier,
        "site_mode": args.sites,
        "num_sites": len(sites),
        "warp_level": args.warp_level,
        "splice_mode": args.splice_mode,
        "schedule_mode": args.schedule_mode,
        "clock_type": args.clock,
        "grid": list(grid_dim),
        "block": list(block_dim),
        "hyperparams": meta,
        "origin": {
            "instructions": origin.n,
            "basic_blocks": origin.nb_basic_blocks,
            "registers": origin.num_registers,
        },
        "correctness": {
            "traced_equals_baseline_bitwise": bit_equal,
            "max_abs_diff_vs_torch_reference": max_ref_diff,
        },
        "timing": {
            "baseline": stats(base_samples),
            "scout": stats(timings.get(0, [0.0])),
            "traced": stats(timings.get(1, [0.0])),
        },
        "trace_records": len(result.get("trace_results", [])) if result else 0,
        "compile_results": result.get("compile_results") if result else None,
        "splice_results": result.get("splice_results") if result else None,
        "trace_wall_seconds": trace_wall,
        "trace_ran": bool(result),
    }
    if out["timing"]["baseline"]["median_ms"] and out["trace_ran"]:
        out["timing"]["overhead_percent"] = (
            out["timing"]["traced"]["median_ms"] / out["timing"]["baseline"]["median_ms"] - 1.0
        ) * 100.0

    # fidelity: disassemble what the trace actually observed and compare with the origin
    if out["trace_ran"] and os.path.exists(kept_cubin):
        traced_view = sasslib.load_kernels(kept_cubin).get(kernel_name)
        if traced_view is not None:
            sim = sasslib.similarity(origin.texts, traced_view.texts)
            sim.update(sasslib.block_changes(origin, traced_view))
            sim["traced_registers"] = traced_view.num_registers
            sim["traced_basic_blocks"] = traced_view.nb_basic_blocks
            sim["cubin_bytes"] = os.path.getsize(kept_cubin)
            out["binary_similarity"] = sim
        else:
            out["binary_similarity"] = {"error": f"{kernel_name} not found in {kept_cubin}"}

    if out["trace_ran"]:
        section.add_run(result)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(jsonable(out), f, indent=2)
        report_path = args.out.replace(".json", ".html")
        from gwatch.common.format import File
        report = File(title=f"Xtrace probe — {args.kernel} / {args.tier} / {args.sites}")
        report.add_section(section)
        report.render(report_path)
        out["report"] = report_path

    print(json.dumps(jsonable(out), indent=2))
    return out


if __name__ == "__main__":
    main()
