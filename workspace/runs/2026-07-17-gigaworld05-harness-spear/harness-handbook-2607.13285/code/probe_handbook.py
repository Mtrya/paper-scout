#!/usr/bin/env python3
"""
probe_handbook.py — minimal Harness Handbook probe on the real Terminus-2 source.

Mirrors the paper's three-phase construction in miniature, replacing the LLM
proposal/review loop with deterministic heuristics, to understand the method's
mechanics and failure modes first-hand:

  Phase I  : stdlib `ast` static extraction — functions, signatures, call edges,
             self._* state reads/writes, unresolved-call audit log, boundary nodes.
  Phase II : hand-authored seed skeleton S0 (execution stages) + rule-based
             function->stage assignment with call-graph propagation fallback.
             (The paper uses an LLM proposer + reviewer here; we don't. The
             quality gap between our crude rules and their loop is itself a
             data point about how much work the LLM structuring does.)
  Phase III: render an L1/L2/L3-style markdown handbook with source locators
             (overview.md, index.md, registers.md, stages/stage-*.md).

Then run one behavior-localization query — the paper's own Terminus-2 Q1
("require marking task_complete three times before grading") — comparing
handbook lookup vs naive grep, scored against the paper's Appendix E answer key.

Usage: python3 probe_handbook.py <terminus_2_dir> <out_dir>
"""

import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

# --------------------------------------------------------------------------
# Seed skeleton S0 (hand-authored, as the paper's function-as-leaf mode assumes)
# --------------------------------------------------------------------------
STAGES = [
    ("stage-1",  "Initialization & Config",
     "Agent construction, LLM client setup, model-info resolution, API surface (name/version)."),
    ("stage-2",  "Prompt & Template Assembly",
     "Loading prompt/timeout templates, skill frontmatter, instruction construction."),
    ("stage-3",  "LLM Query & API Handling",
     "Sending chat to the model, retries, error classification, usage/latency metrics."),
    ("stage-4",  "Response Parsing",
     "Turning raw model text into commands + completion flags (JSON/XML parsers)."),
    ("stage-5",  "Main Agent Loop",
     "The observe-decide-act iteration: episode bookkeeping, observation handling, "
     "command dispatch, and the completion gate that decides loop termination."),
    ("stage-6",  "Command Execution (Terminal I/O)",
     "Sending keystrokes to tmux, capturing panes, key batching, shell-session lifecycle."),
    ("stage-7",  "Context & Token Management",
     "Token counting, output capping, proactive summarization, message unwinding."),
    ("stage-8",  "Trajectory & Recording",
     "Step/trajectory persistence, asciinema recording, ATIF conversion."),
    ("stage-9",  "Subagent Handoff",
     "Spawning subagents with summarized context, aggregating their metrics."),
    ("stage-10", "Per-Run Lifecycle & Reset",
     "run()/perform_task entry points and per-run state reset."),
]

# keyword rules: stage_id -> list of regex tested against qualname+docstring+source
RULES = {
    "stage-1":  [r"^Terminus2\.__init__$", r"_init_llm", r"_resolve_model_info",
                 r"^Terminus2\.name$", r"^Terminus2\.version$"],
    "stage-2":  [r"template", r"frontmatter", r"instruction"],
    "stage-3":  [r"_query_llm", r"error_response", r"usage_metric", r"api_request_time",
                 r"completion_confirmation_message"],
    "stage-4":  [r"parser", r"parse_"],
    "stage-5":  [r"_run_agent_loop", r"_handle_llm_interaction", r"_execute_commands"],
    "stage-6":  [r"^TmuxSession\.", r"tmux", r"send_keys", r"capture_pane", r"_key",
                 r"enter_key", r"newline", r"executing_command"],
    "stage-7":  [r"token", r"summariz", r"unwind", r"limit_output"],
    "stage-8":  [r"trajectory", r"asciinema", r"_dump", r"recording", r"steps?"],
    "stage-9":  [r"subagent", r"handoff", r"continuation"],
    "stage-10": [r"^Terminus2\.perform_task$", r"^Terminus2\.run", r"reset_per_run"],
}

BUILTIN_BOUNDARIES = {"print", "len", "str", "int", "float", "bool", "list", "dict",
                      "set", "tuple", "range", "enumerate", "zip", "isinstance",
                      "super", "open", "min", "max", "sum", "sorted", "any", "all",
                      "repr", "type", "hasattr", "getattr", "setattr", "ValueError",
                      "Exception", "RuntimeError", "Path"}


# --------------------------------------------------------------------------
# Phase I: static fact extraction
# --------------------------------------------------------------------------
class FunctionFacts:
    def __init__(self, qualname, file, lineno, end_lineno, sig, decorators, doc,
                 is_async, class_name):
        self.qualname = qualname
        self.file = file
        self.lineno = lineno
        self.end_lineno = end_lineno
        self.sig = sig
        self.decorators = decorators
        self.doc = doc
        self.is_async = is_async
        self.class_name = class_name
        self.calls = []            # raw callee names seen in body
        self.state_reads = set()   # self._* attributes loaded
        self.state_writes = set()  # self._* attributes stored
        self.self_attr_types = {}  # self._x -> ClassName (from self._x = ClassName(...))
        self.param_types = {}      # param name -> annotation (e.g. session -> TmuxSession)

    @property
    def locator(self):
        return f"{self.file}:{self.lineno}-{self.end_lineno}"


def sig_of(node, src_lines):
    """Best-effort signature: the def line(s) up to the colon."""
    parts = []
    for i in range(node.lineno - 1, min(node.end_lineno, node.lineno + 12)):
        parts.append(src_lines[i].strip())
        if src_lines[i].rstrip().endswith(":"):
            break
    s = " ".join(parts)
    return re.sub(r"\s+", " ", s)[:160]


class Extractor(ast.NodeVisitor):
    """One pass per module. Tracks enclosing class, resolves calls, records state."""

    def __init__(self, module_file, src, funcs, module_defs, imports):
        self.file = module_file
        self.src_lines = src.splitlines()
        self.funcs = funcs                 # qualname -> FunctionFacts (global)
        self.module_defs = module_defs     # module-level function names
        self.imports = imports             # alias -> dotted external name
        self.class_stack = []
        self.fn_stack = []

    # -- scope tracking -----------------------------------------------------
    def visit_ClassDef(self, node):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def _visit_fn(self, node):
        cls = self.class_stack[-1] if self.class_stack else None
        if self.fn_stack:  # nested function: fold into outer, skip separate card
            self.generic_visit(node)
            return
        qual = f"{cls}.{node.name}" if cls else node.name
        decorators = [ast.unparse(d) for d in node.decorator_list]
        ff = FunctionFacts(qual, self.file, node.lineno, node.end_lineno,
                           sig_of(node, self.src_lines), decorators,
                           (ast.get_docstring(node) or "").split("\n")[0][:120],
                           isinstance(node, ast.AsyncFunctionDef), cls)
        for a in list(node.args.args) + list(node.args.kwonlyargs):
            if a.annotation is not None:
                ff.param_types[a.arg] = ast.unparse(a.annotation)
        self.funcs[qual] = ff
        if cls is None:
            self.module_defs.add(node.name)
        self.fn_stack.append(ff)
        self.generic_visit(node)
        self.fn_stack.pop()

    visit_FunctionDef = _visit_fn
    visit_AsyncFunctionDef = _visit_fn

    # -- fact collection ----------------------------------------------------
    def visit_Call(self, node):
        if self.fn_stack:
            ff = self.fn_stack[-1]
            f = node.func
            if isinstance(f, ast.Name):
                ff.calls.append(("name", f.id))
            elif isinstance(f, ast.Attribute):
                base = f.value
                if isinstance(base, ast.Name) and base.id == "self":
                    ff.calls.append(("self", f.attr))
                elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) \
                        and base.value.id == "self":
                    ff.calls.append(("self_attr", f"{base.attr}.{f.attr}"))
                elif isinstance(base, ast.Name) and base.id in self.imports:
                    ff.calls.append(("external", f"{self.imports[base.id]}.{f.attr}"))
                elif isinstance(base, ast.Name):
                    ff.calls.append(("var", f"{base.id}.{f.attr}"))
                else:
                    ff.calls.append(("attr", f.attr))
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if self.fn_stack and isinstance(node.value, ast.Name) and node.value.id == "self":
            ff = self.fn_stack[-1]
            if isinstance(node.ctx, ast.Load):
                ff.state_reads.add(node.attr)
            elif isinstance(node.ctx, (ast.Store, ast.Del)):
                ff.state_writes.add(node.attr)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        if self.fn_stack and isinstance(node.target, ast.Attribute) \
                and isinstance(node.target.value, ast.Name) and node.target.value.id == "self":
            ff = self.fn_stack[-1]
            ff.state_reads.add(node.target.attr)
            ff.state_writes.add(node.target.attr)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        # self._x: ClassName | None = ...  -> declared type of the attribute
        if self.fn_stack and isinstance(node.target, ast.Attribute) \
                and isinstance(node.target.value, ast.Name) and node.target.value.id == "self" \
                and node.annotation is not None:
            cname = ast.unparse(node.annotation).split("|")[0].strip()
            self.fn_stack[-1].self_attr_types[node.target.attr] = cname
        self.generic_visit(node)

    def visit_Assign(self, node):
        # self._x = ClassName(...)  ->  type inference for self._x.method() calls
        if self.fn_stack and isinstance(node.value, ast.Call):
            callee = node.value.func
            cname = None
            if isinstance(callee, ast.Name):
                cname = callee.id
            elif isinstance(callee, ast.Attribute):
                cname = callee.attr
            if cname:
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) \
                            and t.value.id == "self":
                        self.fn_stack[-1].self_attr_types[t.attr] = cname
        self.generic_visit(node)


def extract(repo: Path):
    funcs, edges, boundaries, unresolved = {}, [], defaultdict(set), []
    for path in sorted(repo.glob("*.py")):
        src = path.read_text()
        tree = ast.parse(src)
        rel = path.name
        # import aliases
        imports = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    imports[a.asname or a.name.split(".")[0]] = a.name
            elif isinstance(n, ast.ImportFrom) and n.module:
                for a in n.names:
                    imports[a.asname or a.name] = f"{n.module}.{a.name}"
        module_defs = set()
        Extractor(rel, src, funcs, module_defs, imports).visit(tree)
    # second pass: resolve call edges now that all funcs are known
    class_index = defaultdict(dict)  # class name -> {method: qualname}
    for q, ff in funcs.items():
        if ff.class_name:
            class_index[ff.class_name][q.split(".")[-1]] = q
    module_funcs = {q.split(".")[-1]: q for q, ff in funcs.items() if not ff.class_name}

    self_attr_type_global = {}  # (class, attr) -> ClassName
    for q, ff in funcs.items():
        for attr, cname in ff.self_attr_types.items():
            if cname in class_index:
                self_attr_type_global[(ff.class_name, attr)] = cname

    for q, ff in funcs.items():
        for kind, name in ff.calls:
            target = None
            if kind == "self" and ff.class_name:
                target = class_index.get(ff.class_name, {}).get(name)
                if target is None:  # maybe inherited / defined elsewhere
                    hits = [qq for c, m in class_index.items() if name in m for qq in [m[name]]]
                    if hits:
                        target = hits[0]
            elif kind == "self_attr" and ff.class_name:
                attr, meth = name.rsplit(".", 1)
                cname = self_attr_type_global.get((ff.class_name, attr))
                if cname:
                    target = class_index.get(cname, {}).get(meth)
            elif kind == "name":
                target = module_funcs.get(name)
                if target is None and name in BUILTIN_BOUNDARIES:
                    target = f"<builtin>{name}"
                elif target is None and name in class_index:
                    target = class_index[name].get("__init__")
            elif kind == "var":
                var, meth = name.rsplit(".", 1)
                ann = ff.param_types.get(var, "")
                ann = ann.strip("'\"").split("[")[0]  # strip quotes / generics
                if ann in class_index:
                    target = class_index[ann].get(meth)
            if target:
                edges.append((q, target))
            elif kind in ("self", "attr", "self_attr", "var"):
                unresolved.append((q, name))
            elif kind == "external":
                boundaries[q].add(name)
    # method references are not state: drop self.<attr> that names a method of the class
    for q, ff in funcs.items():
        if ff.class_name and ff.class_name in class_index:
            methods = set(class_index[ff.class_name])
            ff.state_reads -= methods
            ff.state_writes -= methods
    return funcs, edges, boundaries, unresolved, class_index


# --------------------------------------------------------------------------
# Phase II: behavioral organization (heuristic stand-in for the LLM loop)
# --------------------------------------------------------------------------
def assign_stages(funcs, edges):
    assignments, provenance = {}, {}
    for q, ff in funcs.items():
        hay = f"{q}\n{ff.doc}"  # qualname + docstring only: param names are not behavior
        scored = []
        for sid, pats in RULES.items():
            n = sum(1 for p in pats if re.search(p, hay, re.I | re.M))
            if n:
                scored.append((sid, n))
        if scored:
            best = max(n for _, n in scored)
            hits = [sid for sid, n in scored if n == best]
            if len(hits) == 1:
                assignments[q], provenance[q] = hits[0], "rule"
            else:
                order = [s[0] for s in STAGES]
                hits.sort(key=order.index)
                assignments[q], provenance[q] = hits[0], f"rule-ambiguous({','.join(sid for sid, _ in scored)})"
    # call-graph propagation for unassigned: plurality of callers' stages
    callers = defaultdict(set)
    for a, b in edges:
        callers[b].add(a)
    for _ in range(3):
        changed = False
        for q in funcs:
            if q in assignments:
                continue
            votes = defaultdict(int)
            for c in callers.get(q, ()):
                if c in assignments:
                    votes[assignments[c]] += 1
            if votes:
                assignments[q] = max(votes, key=votes.get)
                provenance[q] = "callgraph-propagation"
                changed = True
        if not changed:
            break
    unmapped = [q for q in funcs if q not in assignments]
    return assignments, provenance, unmapped


# --------------------------------------------------------------------------
# Phase III: render the mini handbook
# --------------------------------------------------------------------------
def render(out: Path, funcs, edges, boundaries, assignments, provenance, unmapped):
    out.mkdir(parents=True, exist_ok=True)
    (out / "stages").mkdir(exist_ok=True)
    callers = defaultdict(set)
    callees = defaultdict(set)
    for a, b in edges:
        callers[b].add(a)
        callees[a].add(b)

    # state registers: attr -> readers/writers
    reg_r, reg_w = defaultdict(set), defaultdict(set)
    for q, ff in funcs.items():
        for a in ff.state_reads:
            reg_r[a].add(q)
        for a in ff.state_writes:
            reg_w[a].add(q)

    stage_of = {sid: (sid, title, desc) for sid, title, desc in STAGES}

    (out / "overview.md").write_text(
        "# L1 System Overview — Terminus-2 (probe-generated)\n\n"
        "Terminus-2 is a terminal agent in an observe-decide-act loop: it captures the "
        "tmux pane, prompts an LLM, parses the response into commands, sends keystrokes "
        "to the terminal, and repeats until the model marks the task complete (twice, "
        "via a pending-completion handshake) or limits are hit. Context pressure is "
        "handled by proactive summarization; everything is recorded as trajectories.\n\n"
        "Stages: " + ", ".join(f"`{t}`" for _, t, _ in STAGES) + "\n")

    idx = ["# L2 Index — stages and leaves\n"]
    for sid, title, desc in STAGES:
        members = [q for q, s in assignments.items() if s == sid]
        idx.append(f"\n## {sid} {title}\n{desc}\n")
        for q in sorted(members, key=lambda x: (funcs[x].file, funcs[x].lineno)):
            idx.append(f"- `{q}` — {funcs[q].locator} — {funcs[q].doc or '(no docstring)'}")
    if unmapped:
        idx.append("\n## UNMAPPED (explicit coverage record)\n")
        for q in sorted(unmapped):
            idx.append(f"- `{q}` — {funcs[q].locator}")
    (out / "index.md").write_text("\n".join(idx) + "\n")

    regs = ["# State registers — every self._* attribute with its read/write sites\n"]
    for attr in sorted(set(reg_r) | set(reg_w)):
        regs.append(f"\n## `{attr}`")
        regs.append(f"- writes: {', '.join(f'`{q}` ({funcs[q].locator})' for q in sorted(reg_w[attr])) or '—'}")
        regs.append(f"- reads: {', '.join(f'`{q}` ({funcs[q].locator})' for q in sorted(reg_r[attr])) or '—'}")
    (out / "registers.md").write_text("\n".join(regs) + "\n")

    for sid, title, desc in STAGES:
        members = [q for q, s in assignments.items() if s == sid]
        if not members:
            continue
        lines = [f"# {sid} {title}\n\n{desc}\n"]
        for q in sorted(members, key=lambda x: (funcs[x].file, funcs[x].lineno)):
            ff = funcs[q]
            lines.append(f"\n## L3 `{q}` — `{ff.locator}`")
            lines.append(f"- signature: `{ff.sig}`")
            if ff.decorators:
                lines.append(f"- decorators: {', '.join('`'+d+'`' for d in ff.decorators)}")
            lines.append(f"- assignment provenance: {provenance[q]}")
            if ff.state_reads:
                lines.append(f"- reads state: {', '.join('`'+a+'`' for a in sorted(ff.state_reads))}")
            if ff.state_writes:
                lines.append(f"- writes state: {', '.join('`'+a+'`' for a in sorted(ff.state_writes))}")
            ci = [c for c in sorted(callers[q]) if not c.startswith('<builtin>')]
            ce = [c for c in sorted(callees[q]) if not c.startswith('<builtin>')]
            if ci:
                lines.append(f"- callers: {', '.join('`'+c+'`' for c in ci)}")
            if ce:
                lines.append(f"- calls: {', '.join('`'+c+'`' for c in ce)}")
        (out / "stages" / f"{sid}.md").write_text("\n".join(lines) + "\n")
    return reg_r, reg_w


# --------------------------------------------------------------------------
# Behavior-localization demo: paper's Terminus-2 Q1
# --------------------------------------------------------------------------
def demo_query(repo: Path, out: Path, funcs, reg_r, reg_w, assignments):
    print("\n" + "=" * 74)
    print("BEHAVIOR LOCALIZATION DEMO — paper's Terminus-2 Q1:")
    print('  "require marking task_complete THREE times, with are-you-sure re-prompts"')
    print("=" * 74)
    # Paper Appendix E answer key (their snapshot; ~line drift vs ours):
    key = ["Terminus2.__init__ (flag init)",
           "Terminus2._reset_per_run_state (per-run clear)",
           "Terminus2._run_agent_loop observation branch (set/clear flag)",
           "Terminus2._run_agent_loop completion gate (test flag, return)"]

    # --- handbook path: request -> stage (completion gate lives in Main Loop)
    #     -> register `_pending_completion` -> all R/W sites, verified vs source
    stage_hits = [sid for sid, title, desc in STAGES
                  if re.search(r"terminat|completion", f"{title} {desc}", re.I)]
    reg = "_pending_completion"
    sites = []
    src = (repo / "terminus_2.py").read_text().splitlines()
    for q in sorted(reg_r[reg] | reg_w[reg]):
        ff = funcs[q]
        hits = [i + 1 for i, ln in enumerate(src[ff.lineno - 1:ff.end_lineno], ff.lineno - 1)
                if reg in ln]
        sites.append((q, ff.locator, hits))
    print("\n[handbook path] stages matched:", stage_hits)
    print(f"[handbook path] registers.md entry `{reg}` -> {len(sites)} functions:")
    for q, loc, hits in sites:
        print(f"    {q:<38} {loc:<22} lines with `{reg}`: {hits}")

    # verification step (BGPD source verification): every locator must resolve
    # to source that still contains the register token
    ok = 0
    for q, loc, hits in sites:
        ff = funcs[q]
        body = "\n".join((repo / ff.file).read_text().splitlines()[ff.lineno - 1:ff.end_lineno])
        if reg in body:
            ok += 1
    print(f"[handbook path] locators verified against current source: {ok}/{len(sites)}")

    # --- naive grep path: request keywords -> hits, incl. noise
    import subprocess
    for kw in ["task_complete", "completion", "pending_completion"]:
        r = subprocess.run(["grep", "-rn", kw, str(repo)], capture_output=True, text=True)
        n = len([l for l in r.stdout.splitlines() if l.strip()])
        print(f"[grep] '{kw}': {n} hits")
    print("\n[paper's answer key]")
    for k in key:
        print("   -", k)


def main():
    repo = Path(sys.argv[1])
    out = Path(sys.argv[2])
    funcs, edges, boundaries, unresolved, class_index = extract(repo)
    print(f"Phase I: {len(funcs)} internal functions, "
          f"{len(set(edges))} resolved internal call edges, "
          f"{len(unresolved)} unresolved call sites")
    uc = defaultdict(int)
    for q, name in unresolved:
        uc[name.split('.')[-1]] += 1
    print("  top unresolved callees (dynamic/external-attr):",
          sorted(uc.items(), key=lambda x: -x[1])[:8])

    assignments, provenance, unmapped = assign_stages(funcs, edges)
    prov_count = defaultdict(int)
    for p in provenance.values():
        prov_count[p.split("(")[0]] += 1
    print(f"Phase II: assigned {len(assignments)}/{len(funcs)} functions; "
          f"provenance: {dict(prov_count)}; unmapped: {len(unmapped)}")
    if unmapped:
        for q in unmapped:
            print(f"    UNMAPPED: {q} ({funcs[q].locator})")

    reg_r, reg_w = render(out, funcs, edges, boundaries, assignments, provenance, unmapped)
    print(f"Phase III: handbook rendered to {out}/ "
          f"({len(reg_r | reg_w)} state registers)")

    # failure-mode probes
    print("\nFailure-mode probes:")
    dec = {q: ff.decorators for q, ff in funcs.items() if ff.decorators}
    print(f"  decorated functions (wrappers break naive static edges): {len(dec)}")
    for q, d in list(dec.items())[:6]:
        print(f"    {q}: @{', @'.join(d)}")
    dyn = [q for q, ff in funcs.items() if any('getattr' in c[1] for c in ff.calls)]
    print(f"  functions using getattr (dynamic dispatch): {dyn}")
    cross = [(a, b) for a, b in set(edges)
             if not b.startswith('<builtin>') and funcs[a].file != funcs[b].file]
    print(f"  cross-file call edges resolved via self._attr type inference: {len(cross)}")

    demo_query(repo, out, funcs, reg_r, reg_w, assignments)


if __name__ == "__main__":
    main()
