"""SASS inspection helpers for the Xtrace fidelity measurements.

Three things live here, all built on gwatch's own disassembler (`gwatch.cuda.binary.Cubin`
+ `gwatch.cuda.assemble.KernelDefSASS`) so that the comparison uses the same instruction
decoder the splicer uses:

  * `load_kernels`  -- disassemble a cubin into {kernel name: kernel view}
  * `KernelView`    -- instruction list, opcodes, basic blocks, register count
  * `similarity`    -- the binary-similarity metric of the paper's section 5.3

The metric, verbatim from the paper: align the instruction sequences of the probed and the
unprobed kernel; count an original instruction as *changed* when its operands, its control
bits, or its position within its basic block differ, and as *preserved* otherwise. Probe
instructions have no counterpart in the original and are counted separately as *inserted*.
Binary similarity is preserved / original.

Alignment here is `difflib.SequenceMatcher` over instruction text. A shifted branch target is
part of the instruction text, so a branch whose target moved counts as changed, which is the
paper's own reading ("the changed 2-6% are branch instructions whose target addresses must
shift after inserting instructions"). Probes are inserted *between* original instructions and
never split one, so the longest matching blocks recover the original order.
"""

import difflib
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gwatch.cuda.binary as gw_binary


def opcode_of(text: str) -> str:
    """The bare opcode of a disassembled instruction (predicate and modifiers stripped)."""
    tokens = text.split()
    if tokens and tokens[0].startswith("@"):
        tokens = tokens[1:]
    return tokens[0].split(".")[0] if tokens else ""


@dataclass
class KernelView:
    name: str
    arch_version: str
    num_registers: int
    texts: List[str]
    nb_basic_blocks: int
    blocks: List[Tuple[int, int, int]] = field(default_factory=list)  # (base_pc, end_pc, n)

    @property
    def n(self) -> int:
        return len(self.texts)

    def opcode_counts(self) -> Counter:
        return Counter(opcode_of(t) for t in self.texts)


def load_kernels(cubin_path: str) -> Dict[str, KernelView]:
    """Disassemble every kernel in a cubin."""
    cubin = gw_binary.Cubin()
    cubin.fill(cubin_path)
    cubin.parse()
    out: Dict[str, KernelView] = {}
    kernel_defs = cubin.get_map_kernel_def()
    for name in list(kernel_defs.keys()):
        kd = kernel_defs[name]
        texts = [instr.str() for instr in kd.get_list_instruction()]
        kd.parse_cfg()
        cfg = kd.cfg
        blocks = []
        nb_bb = 0
        if cfg is not None:
            for bb in cfg.basic_blocks:
                blocks.append((bb.base_pc, bb.end_pc, bb.nb_instructions))
            nb_bb = cfg.nb_basic_blocks
        out[name] = KernelView(
            name=name,
            arch_version=str(kd.params.arch_version),
            num_registers=int(kd.params.num_registers),
            texts=texts,
            nb_basic_blocks=nb_bb,
            blocks=blocks,
        )
    return out


def similarity(origin: List[str], traced: List[str]) -> Dict:
    """The paper's binary-similarity metric between an original and a probed instruction list."""
    matcher = difflib.SequenceMatcher(a=origin, b=traced, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())

    # every original instruction not covered by a matching block was altered, dropped or moved
    covered = [False] * len(origin)
    for block in matcher.get_matching_blocks():
        for i in range(block.a, block.a + block.size):
            covered[i] = True
    changed = [origin[i] for i, ok in enumerate(covered) if not ok]
    inserted = len(traced) - matched

    return {
        "origin_instructions": len(origin),
        "traced_instructions": len(traced),
        "preserved_instructions": matched,
        "changed_instructions": len(changed),
        "inserted_instructions": inserted,
        "binary_similarity": matched / len(origin) if origin else 0.0,
        "changed_by_opcode": dict(Counter(opcode_of(t) for t in changed)),
        "inserted_by_opcode": dict(
            Counter(opcode_of(t) for t in _unmatched(traced, matcher))
        ),
    }


def _unmatched(traced: List[str], matcher: difflib.SequenceMatcher) -> List[str]:
    covered = [False] * len(traced)
    for block in matcher.get_matching_blocks():
        for i in range(block.b, block.b + block.size):
            covered[i] = True
    return [traced[i] for i, ok in enumerate(covered) if not ok]


def block_changes(origin: KernelView, traced: KernelView, isize: int = 16) -> Dict:
    """How many of the original's basic blocks survive intact.

    The paper's rule: a basic block counts as changed when any of its original instructions is
    altered, dropped, or moved into another block. Alignment maps every surviving original
    instruction onto its position in the instrumented kernel, and the block's columns are read
    off the instrumented kernel's own CFG: a block survives when every one of its original
    instructions still sits inside a single instrumented block.

    ``blocks_with_probe_inside`` is reported separately: it counts blocks that an inserted probe
    now sits in the middle of. That splits a block's *layout* without changing what the block
    is, so it is not a change by the paper's rule, but it is what makes the linear address
    range of a block no longer contiguous.
    """
    matcher = difflib.SequenceMatcher(a=origin.texts, b=traced.texts, autojunk=False)
    map_a_to_b = {}
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            map_a_to_b[block.a + k] = block.b + k

    # which instrumented block holds each instrumented instruction index
    block_of_index = {}
    for block_id, (base_pc, end_pc, nb) in enumerate(traced.blocks):
        for i in range(base_pc // isize, base_pc // isize + nb):
            block_of_index[i] = block_id

    changed = 0
    intact = 0
    probe_inside = 0
    for base_pc, _end_pc, nb in origin.blocks:
        lo = base_pc // isize
        mapped = [map_a_to_b.get(i) for i in range(lo, lo + nb)]
        if not mapped or any(m is None for m in mapped):
            changed += 1
            continue
        holders = {block_of_index.get(m) for m in mapped}
        if len(holders) == 1:
            intact += 1
            if mapped != list(range(mapped[0], mapped[0] + len(mapped))):
                probe_inside += 1
        else:
            changed += 1
    total = changed + intact
    return {
        "origin_blocks": total,
        "intact_blocks": intact,
        "changed_blocks": changed,
        "blocks_with_probe_inside": probe_inside,
        "unchanged_fraction": intact / total if total else 0.0,
    }


def live_registers_at(kd, pc: int, isize: int = 16) -> Dict[str, List[int]]:
    """Registers live at ``pc``, by class, from gwatch's own liveness analysis.

    The splicer's own rule: a probe may only reuse registers that are dead here.
    """
    kd.parse_register_liveness()
    intervals = kd.get_register_live_intervals()
    live = {}
    for reg_class, map_regs in intervals.items():
        names = []
        for reg, ranges in map_regs.items():
            for lo, hi in ranges:
                if lo <= pc <= hi:
                    names.append(reg)
                    break
        live[reg_class] = sorted(names)
    return live


def register_state_at(kd, pc: int) -> Dict:
    return kd.get_register_state_at_pc(int(pc))


def find_memory_sites(kernel: KernelView, opcodes: Optional[Tuple[str, ...]] = None) -> List[int]:
    """Instruction indices of memory accesses, the paper's trace-mem site set."""
    if opcodes is None:
        opcodes = ("LDG", "STG", "LDS", "STS", "LD", "ST", "LDL", "STL", "ATOM", "RED", "LDGSTS")
    return [i for i, t in enumerate(kernel.texts) if opcode_of(t).startswith(opcodes)]
