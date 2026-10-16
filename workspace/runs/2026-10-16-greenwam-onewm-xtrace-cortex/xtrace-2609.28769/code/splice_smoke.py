#!/usr/bin/env python3
"""Static smoke test: can Xtrace's splicer actually splice into an sm_86 cubin?

This touches no GPU at all. It loads a cubin already on disk, runs the same two analyses the
splicer runs (CFG + register liveness), asks the splicer for a register allocation at an
insertion point, stages the real trace-store operator, finalizes the splice, and serializes
the result -- so a failure here is about the tool's architecture support, not about our setup.

Usage: splice_smoke.py <origin.cubin> [instruction index ...]
"""

import sys
import tempfile

import gwatch.cuda.binary as gw_binary
from gwatch.cuda.assemble import InstrumentOperator, TimestampKind
import sasslib


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    views = sasslib.load_kernels(path)
    name = next(iter(views))
    view = views[name]
    print(f"kernel {name}: arch=sm_{view.arch_version} regs={view.num_registers} "
          f"instructions={view.n} basic_blocks={view.nb_basic_blocks}")

    cubin = gw_binary.Cubin()
    cubin.fill(path)
    cubin.parse()
    kd = cubin.get_kerneldef_by_name(name)
    kd.parse_cfg()
    kd.parse_register_liveness()
    isize = kd.cfg.instruction_size
    print(f"instruction size = {isize} bytes")

    indices = [int(x) for x in sys.argv[2:]] or [1, view.n // 2, view.n - 2]
    for index in indices:
        pc = index * isize
        # the C++ binding exposes per-class liveness directly; the Python wrapper's
        # get_register_state_at_pc() is absent from this build of the shipped .so
        live = {reg_class: kd._gw_instance.get_live_registers_at_pc(pc, reg_class)
                for reg_class in ("general", "uniform_general", "predicate", "uniform_predicate")}
        print(f"\n[index {index} pc {hex(pc)}] {view.texts[index]}")
        print(f"  live at pc: {live}")
        print(f"  active scoreboards: {kd.get_list_scoreboards_active_over(pc, pc)}")
        try:
            pair = kd.collect_allocatable_register_pairs("general", [pc], 2)
            print(f"  free general register pair: {pair}")
        except RuntimeError as exc:
            print(f"  free general register pair: RuntimeError({exc})")
        try:
            upair = kd.collect_allocatable_register_pairs("uniform_general", [pc], 2)
            print(f"  free uniform register pair: {upair}")
        except RuntimeError as exc:
            print(f"  free uniform register pair: RuntimeError({exc})")

    # splice a real store operator at the middle index and diff the instruction stream
    index = indices[len(indices) // 2]
    with kd.create_instrument_context(index * isize) as ctx:
        ctx.operator(InstrumentOperator.STORE_TRACE_RECORD, {
            "trace_ptr_reg": 4,
            "thread_limit_reg": 6,
            "trace_ptr_is_uniform": 0,
            "value_reg": 255,          # RZ: timestamp-only record
            "value_width_bits": 32,
            "value_is_uniform": 0,
            "value_is_smid": 0,
            "site_id": 0,
            "ring_wrap": 1,
            "stride": 512,
            "timestamp_kind": int(TimestampKind.GLOBALTIMER),
            "advance": 16,
        })
        pcs = ctx.finalize()
        report = ctx.report
    print(f"\nsplice at index {index} -> pcs {[hex(p) for p in pcs]}")
    print(f"splice report: {report}")

    after = [instr.str() for instr in kd.get_list_instruction()]
    sim = sasslib.similarity(view.texts, after)
    print(f"after the splice: origin={sim['origin_instructions']} traced={sim['traced_instructions']} "
          f"preserved={sim['preserved_instructions']} changed={sim['changed_instructions']} "
          f"inserted={sim['inserted_instructions']} similarity={sim['binary_similarity']:.4f}")
    print(f"changed by opcode: {sim['changed_by_opcode']}")
    print(f"inserted by opcode: {sim['inserted_by_opcode']}")
    print("inserted instructions:")
    for pc in pcs:
        print(f"  {hex(pc)}: {after[pc // isize]}")

    serialized = gw_binary.Cubin()
    serialized.add_kerneldef(kd)
    blob = serialized.serialize()
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(blob)
        print(f"\nserialized spliced cubin: {len(blob)} bytes -> {f.name}")

    # the serialized image must disassemble back to the same instruction stream
    reloaded = sasslib.load_kernels(f.name)
    print(f"reloaded kernels: {list(reloaded)}")
    if name in reloaded:
        sim2 = sasslib.similarity(after, reloaded[name].texts)
        print(f"round-trip similarity vs spliced: {sim2['binary_similarity']:.4f} "
              f"(regs {reloaded[name].num_registers})")


if __name__ == "__main__":
    main()
