#!/usr/bin/env python3
"""How often can Xtrace splice a probe into an sm_86 kernel, and at what cost?

For every instruction index of a cubin (or a sampled subset), this reloads the kernel fresh,
allocates a register pair from the registers that are dead at that address, stages the real
trace-store operator, and asks the splicer to schedule it. Success means the tool found a
splice (instruction order x register allocation x control bits) that its own hazard rules
accept; failure means the same code the paper describes raises "no feasible schedule?".

Each attempt reuses a fresh copy of the cubin, so one attempt cannot perturb the next.

Usage:
    splice_feasibility.py <origin.cubin> [--step N] [--limit N] [--op store|init] [--report out.json]
"""

import argparse
import json
import os
import tempfile
from collections import Counter

import gwatch.cuda.binary as gw_binary
from gwatch.cuda.assemble import InstrumentOperator, TimestampKind

import sasslib

_RECORD_BYTES = 16
_PER_THREAD_RECORDS = 64
_STRIDE = _RECORD_BYTES * _PER_THREAD_RECORDS


def attempt(path, name, index, op_name, schedule_mode=0, reg_class="auto"):
    """One splice attempt, on a private copy of the image."""
    cubin = gw_binary.Cubin()
    cubin.fill(path)
    cubin.parse()
    kd = cubin.get_kerneldef_by_name(name)
    kd.parse_cfg()
    kd.parse_register_liveness()
    isize = kd.cfg.instruction_size
    pc = index * isize

    # the tool's own preference order: a per-warp uniform pair first, general registers second
    classes = ("uniform_general", "general") if reg_class == "auto" else (reg_class,)
    pair = None
    for candidate_class in classes:
        try:
            pair = kd.collect_allocatable_register_pairs(candidate_class, [pc], 2)
            reg_class_used = candidate_class
            break
        except RuntimeError as exc:
            last_error = str(exc)
    if pair is None:
        return {"index": index, "ok": False, "stage": "register-allocation",
                "error": last_error, "reg_classes_tried": list(classes)}
    is_uniform = 1 if reg_class_used == "uniform_general" else 0

    if op_name == "init":
        params = {
            "param_index": 0,
            "stride": _STRIDE,
            "trace_ptr_reg": pair[0],
            "thread_limit_reg": pair[2],
            "trace_ptr_is_uniform": is_uniform,
        }
        op = InstrumentOperator.INIT_TRACE_POINTER
    else:
        params = {
            "trace_ptr_reg": pair[0],
            "thread_limit_reg": pair[2],
            "trace_ptr_is_uniform": is_uniform,
            "value_reg": 255,
            "value_width_bits": 32,
            "value_is_uniform": 0,
            "value_is_smid": 0,
            "site_id": 0,
            "ring_wrap": 1,
            "stride": _STRIDE,
            "timestamp_kind": int(TimestampKind.GLOBALTIMER),
            "advance": _RECORD_BYTES,
        }
        op = InstrumentOperator.STORE_TRACE_RECORD

    try:
        with kd.create_instrument_context(pc, schedule_mode=schedule_mode) as ctx:
            ctx.operator(op, params)
            pcs = list(ctx.finalize())
            report = dict(ctx.report)
    except RuntimeError as exc:
        return {"index": index, "ok": False, "stage": "schedule", "error": str(exc),
                "registers": pair}

    after = [instr.str() for instr in kd.get_list_instruction()]
    inserted = [after[p // isize] for p in pcs]
    return {
        "index": index,
        "ok": True,
        "stage": "spliced",
        "registers": pair,
        "probe_pcs": [hex(p) for p in pcs],
        "probe_instructions": inserted,
        "report": report,
        "nb_instructions_after": len(after),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cubin")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="cap the number of attempts")
    ap.add_argument("--op", default="store", choices=["store", "init"])
    ap.add_argument("--schedule-mode", default="first_feasible", choices=["first_feasible", "least_makespan"])
    ap.add_argument("--reg-class", default="auto", choices=["auto", "general", "uniform_general"])
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    views = sasslib.load_kernels(args.cubin)
    name = next(iter(views))
    view = views[name]

    indices = list(range(1, view.n, args.step))
    if args.limit:
        indices = indices[: args.limit]

    results = []
    for index in indices:
        results.append(attempt(args.cubin, name, index, args.op, reg_class=args.reg_class,
                               schedule_mode=1 if args.schedule_mode == "least_makespan" else 0))

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    print(f"kernel {name}: sm_{view.arch_version}, {view.n} instructions, {view.num_registers} registers")
    print(f"op={args.op} schedule={args.schedule_mode} reg_class={args.reg_class}: "
          f"{len(ok)}/{len(results)} splices succeeded")
    print(f"failure stages: {dict(Counter(r['stage'] for r in failed))}")
    if failed:
        print("first failures: " + ", ".join(
            f"idx {r['index']} ({r['stage']}: {r['error'][:60]})" for r in failed[:8]))
    if ok:
        sizes = Counter(len(r["probe_instructions"]) for r in ok)
        print(f"probe instruction counts: {dict(sizes)}")
        sample = ok[len(ok) // 2]
        print(f"\nexample splice at index {sample['index']} (registers {sample['registers']}):")
        for text in sample["probe_instructions"]:
            print(f"    {text}")
        print(f"report: {json.dumps(sample['report'], indent=2)[:1200]}")

    if args.report:
        with open(args.report, "w") as f:
            json.dump({"kernel": name, "arch": view.arch_version, "n": view.n,
                       "op": args.op, "schedule_mode": args.schedule_mode,
                       "attempts": results}, f, indent=2)
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
