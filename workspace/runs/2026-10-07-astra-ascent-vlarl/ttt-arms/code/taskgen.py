"""Procedurally generated, procedurally verified text tasks for the online-TTT arm study.

Four families, unified protocol:

  * the agent replies either with ``<cmd>...</cmd>`` (an environment command) or with
    ``<answer>...</answer>`` (its final answer);
  * the environment executes well-formed commands and reports the result;
  * an episode ends on ``<answer>``, on a turn-budget overrun, or on a hard parse error.

Each family ships a one-episode demonstration (see :func:`demo_messages`) that is rendered
into the conversation as few-shot turns, and every task carries its own verifier state, so
success is decided programmatically.  Pure stdlib -- importable and testable without torch.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

MAX_TURNS_BY_FAMILY = {"strxform": 6, "calc": 5, "world": 7, "listops": 6}
FAMILIES = ["strxform", "calc", "world", "listops"]

SYSTEM = (
    "You solve short tasks by interacting with a small environment.\n"
    "Every reply must end with exactly one tagged line, and nothing after it:\n"
    "  <cmd>...</cmd> runs one command from the task's allowed list and prints its result;\n"
    "  <answer>...</answer> submits your final answer and ends the task.\n"
    "The tags are literal: write your real command or answer between them, never the word\n"
    "'cmd' or 'answer' on its own, and never a placeholder such as 'FINAL ANSWER'.\n"
    "Reason briefly first if you like; the tagged line goes last.\n"
)

_CMD_RE = re.compile(r"<cmd>(.*?)</cmd>", re.S)
_ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.S)


def parse_turn(text: str) -> tuple[str, str | None]:
    """Return ("answer", payload) / ("cmd", payload) / ("none", None) for one reply."""
    if "</answer>" in text:
        m = _ANS_RE.search(text)
        return ("answer", (m.group(1) if m else "").strip())
    if "</cmd>" in text:
        m = _CMD_RE.search(text)
        return ("cmd", (m.group(1) if m else "").strip())
    if "<answer>" in text:
        return ("answer", text.split("<answer>", 1)[1].strip())
    if "<cmd>" in text:
        return ("cmd", text.split("<cmd>", 1)[1].strip())
    return ("none", None)


def norm_text(s: str) -> str:
    return s.strip().strip("`'\"").strip()


# --------------------------------------------------------------------------------------
# task container
# --------------------------------------------------------------------------------------

@dataclass
class Task:
    tid: str
    family: str
    level: int
    body: str                       # task statement shown to the model
    tool_doc: str                   # allowed commands
    spec: dict = field(default_factory=dict)

    @property
    def max_turns(self) -> int:
        return MAX_TURNS_BY_FAMILY[self.family]

    def prompt(self) -> str:
        parts = [self.body.strip()]
        if self.tool_doc.strip():
            parts.append(self.tool_doc.strip())
        parts.append(f"You may use at most {self.max_turns} turns.")
        return "\n".join(parts)

    def new_state(self):
        return _new_state(self)


# --------------------------------------------------------------------------------------
# family 1 -- string transformation chains, applied through an `apply` command
# --------------------------------------------------------------------------------------

_WORDS = [
    "river", "garden", "signal", "marble", "harvest", "whisper", "lantern", "compass",
    "glacier", "orchard", "thunder", "puzzle", "meadow", "tunnel", "anchor", "velvet",
    "cactus", "ferry", "hollow", "jungle", "kettle", "legend", "magnet", "nectar",
    "opal", "pebble", "quartz", "ribbon", "saddle", "timber", "urchin", "violet",
]
_ROT13 = str.maketrans(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "nopqrstuvwxyzabcdefghijklmNOPQRSTUVWXYZABCDEFGHIJKLM",
)
_STROPS = ["reverse", "upper", "lower", "rot13"]


def _apply_op(s: str, op: str) -> str:
    if op == "reverse":
        return s[::-1]
    if op == "upper":
        return s.upper()
    if op == "lower":
        return s.lower()
    if op == "rot13":
        return s.translate(_ROT13)
    raise ValueError(op)


def gen_strxform(rng: random.Random, level: int) -> Task:
    k = min(1 + level, 4)
    chain: list[str] = []
    while len(chain) < k:
        cand = [o for o in _STROPS if o not in chain[-2:]]
        chain.append(rng.choice(cand))
    word = rng.choice(_WORDS) if rng.random() < 0.5 else \
        "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(5, 8)))
    cur = word
    for op in chain:
        cur = _apply_op(cur, op)
    steps = "\n".join(f"  {i+1}. {op}" for i, op in enumerate(chain))
    body = (
        "Task: transform the input string by applying the operations below, in the order given.\n"
        f"Operations:\n{steps}\n"
        f"Input string: {word}\n"
        "Report the final string as your answer."
    )
    tool_doc = (
        "Allowed command:\n"
        "  apply <op>  -- applies one operation to the current string and prints the result.\n"
        "     ops: reverse | upper | lower | rot13\n"
        "  show        -- prints the current string.\n"
        "Do the operations one at a time with `apply`, then submit the last printed string."
    )
    return Task(tid="", family="strxform", level=level, body=body, tool_doc=tool_doc,
                spec={"answer": cur, "word": word, "chain": chain})


def _new_state(t: Task):
    if t.family == "world":
        return {"loc": {k: (list(v) if v else None) for k, v in t.spec["init_loc"].items()},
                "room": t.spec["rooms"][0], "inv": []}
    if t.family == "calc":
        return {"value": None}
    if t.family == "listops":
        return {"cur": list(t.spec["start"])}
    if t.family == "strxform":
        return {"cur": t.spec["word"]}
    raise ValueError(t.family)


def step_strxform(task, st, kind, payload):
    if kind == "answer":
        ok = norm_text(payload) == task.spec["answer"]
        return True, "", True, ok
    if kind != "cmd":
        return False, "No command or answer found. Reply with <cmd>...</cmd> or <answer>...</answer>.", False, None
    toks = payload.split()
    if not toks:
        return False, "Empty command.", False, None
    verb = toks[0].lower()
    if verb == "show":
        return True, f"Current string: {st['cur']}", False, None
    if verb == "apply":
        if len(toks) != 2:
            return False, "Usage: apply <reverse|upper|lower|rot13>", False, None
        op = toks[1].lower()
        if op not in _STROPS:
            return False, f"Unknown operation: {op}.", False, None
        st["cur"] = _apply_op(st["cur"], op)
        return True, f"Result: {st['cur']}", False, None
    return False, f"No commands are available for this task.", False, None


# --------------------------------------------------------------------------------------
# family 2 -- arithmetic with a calculator command
# --------------------------------------------------------------------------------------

def _rand_expr(rng: random.Random, n_terms: int, parens: bool) -> str:
    parts = [str(rng.randint(2, 20))]
    for _ in range(n_terms - 1):
        parts += [rng.choice(["+", "-", "*"]), str(rng.randint(2, 12))]
    expr = " ".join(parts)
    if parens:
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        expr = f"({a} + {b}) * {expr}"
    return expr


def _eval_expr(expr: str) -> float:
    expr = expr.strip()
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expr):
        raise ValueError("unsafe")
    val = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - charset restricted above
    if not isinstance(val, (int, float)):
        raise ValueError("non-numeric")
    return float(val)


def gen_calc(rng: random.Random, level: int) -> Task:
    n_terms = min(2 + level, 6)
    expr = _rand_expr(rng, n_terms, parens=(level >= 3))
    val = _eval_expr(expr)
    scale = 1
    if level >= 2:
        scale = rng.choice([3, 4, 5, 6])
    want = val * scale
    body = (
        "Task: compute the exact value of the arithmetic expression below using the `calc` "
        "command.\n"
        f"Expression: {expr}\n"
        + (f"Then multiply that value by {scale}.\n" if scale != 1 else "")
        + "Report the final number as your answer (integers may be written without decimals)."
    )
    tool_doc = (
        "Allowed command:\n"
        "  calc <expression>  -- evaluates an arithmetic expression over integers using "
        "+ - * / and parentheses, and prints the exact value.\n"
        "Use `calc` for every step of the arithmetic; do not compute in your head."
    )
    return Task(tid="", family="calc", level=level, body=body, tool_doc=tool_doc,
                spec={"expr": expr, "value": val, "scale": scale, "answer": want})


def step_calc(task, st, kind, payload):
    if kind == "answer":
        got = norm_text(payload).replace(",", "")
        m = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", got)
        if not m:
            return True, "", True, False
        return True, "", True, abs(float(got) - task.spec["answer"]) <= 0.01
    if kind != "cmd":
        return False, "No command or answer found. Reply with <cmd>...</cmd> or <answer>...</answer>.", False, None
    toks = payload.split(maxsplit=1)
    if not toks:
        return False, "Empty command.", False, None
    if toks[0].lower() != "calc":
        return False, f"Unknown command: {toks[0]}.", False, None
    if len(toks) < 2:
        return False, "Usage: calc <expression>", False, None
    try:
        v = _eval_expr(toks[1])
    except Exception as exc:                                  # noqa: BLE001
        return False, f"The expression could not be evaluated ({type(exc).__name__}).", False, None
    st["value"] = v
    return True, f"Result: {v:g}", False, None


# --------------------------------------------------------------------------------------
# family 3 -- mini text world
# --------------------------------------------------------------------------------------

_ROOMS = ["kitchen", "bedroom", "garden", "study", "cellar"]
_ITEMS = ["apple", "coin", "mug", "sock", "pencil", "stone", "book", "key"]
_CONTAINERS = ["fridge", "box", "shelf", "basket"]


def gen_world(rng: random.Random, level: int) -> Task:
    n_rooms = 2 if level <= 1 else min(2 + level, 4)
    rooms = _ROOMS[:n_rooms]
    items = rng.sample(_ITEMS, 3)
    containers = rng.sample(_CONTAINERS, 1 if level <= 1 else min(level, 3))
    croom = {c: rooms[i % n_rooms] for i, c in enumerate(containers)}
    loc = {}
    for it in items:
        r = rng.choice(rooms)
        c = None
        if rng.random() < 0.3:
            cands = [c for c in containers if croom[c] == r]
            if cands:
                c = rng.choice(cands)
        loc[it] = [r, c]
    goal_container = containers[0]
    cand = [it for it in items
            if not (loc[it][0] == croom[goal_container] and loc[it][1] == goal_container)]
    goal_item = cand[0] if cand else items[0]
    if not cand:
        loc[goal_item] = [rng.choice(rooms), None]
    clauses = [{"item": goal_item, "container": goal_container}]
    if level >= 2 and len(containers) >= 2:
        last = containers[-1]
        rest = [it for it in items if it != goal_item
                and not (loc[it][0] == croom[last] and loc[it][1] == last)]
        if rest:
            clauses.append({"item": rest[0], "container": last})
            if croom[last] == croom[goal_container] and len(rooms) > 1:
                croom[last] = [r for r in rooms if r != croom[goal_container]][0]
    exits = {}
    for i, r in enumerate(rooms):
        nb = []
        if i > 0:
            nb.append(rooms[i - 1])
        if i < n_rooms - 1:
            nb.append(rooms[i + 1])
        exits[r] = nb
    goaltxt = "; ".join(f"put the {c['item']} in the {c['container']}" for c in clauses)
    body = (
        f"Task: you are in a small house and you start in the {rooms[0]}.\n"
        f"Your goal: {goaltxt}.\n"
        f"Rooms (a corridor connects them in this order): {' - '.join(rooms)}.\n"
        "Containers: " + ", ".join(f"the {c} is in the {croom[c]}" for c in containers) + ".\n"
        f"Items: {', '.join(items)}.\n"
        "Go to the item, take it, carry it to the container and put it in."
    )
    tool_doc = (
        "Allowed commands:\n"
        "  look                     -- describe the current room, its exits, its items\n"
        "  go <room>                -- move to an adjacent room\n"
        "  take <item>              -- pick up an item that is in the current room\n"
        "  put <item> <container>   -- put a carried item into a container in this room\n"
        "  inventory                -- list what you are carrying\n"
        "When every goal is met, submit <answer>done</answer>."
    )
    spec = {"rooms": rooms, "exits": exits, "containers": containers, "croom": croom,
            "init_loc": loc, "clauses": clauses, "items": items}
    return Task(tid="", family="world", level=level, body=body, tool_doc=tool_doc, spec=spec)


def step_world(task, st, kind, payload):
    if kind == "answer":
        ok, _ = ver_world(task.spec, st)
        return True, "", True, ok
    if kind != "cmd":
        return False, "No command or answer found. Reply with <cmd>...</cmd> or <answer>...</answer>.", False, None
    toks = payload.split()
    if not toks:
        return False, "Empty command.", False, None
    verb, args = toks[0].lower(), toks[1:]
    spec = task.spec
    if verb == "look":
        r = st["room"]
        here = [i for i in spec["items"]
                if st["loc"][i] is not None and st["loc"][i][0] == r and st["loc"][i][1] is None]
        in_c = [f"{i} (in the {st['loc'][i][1]})" for i in spec["items"]
                if st["loc"][i] is not None and st["loc"][i][0] == r and st["loc"][i][1] is not None]
        cont = [c for c in spec["containers"] if spec["croom"][c] == r]
        return True, (f"You are in the {r}. Exits: {', '.join(spec['exits'][r]) or 'none'}. "
                      f"Containers here: {', '.join(cont) or 'none'}. "
                      f"Items here: {', '.join(here + in_c) or 'none'}."), False, None
    if verb == "go":
        if len(args) != 1:
            return False, "Usage: go <room>", False, None
        dest = args[0].lower()
        if dest not in spec["exits"].get(st["room"], []):
            return False, f"You cannot go to the {dest} from the {st['room']}.", False, None
        st["room"] = dest
        return True, f"You are now in the {dest}.", False, None
    if verb == "take":
        if len(args) != 1:
            return False, "Usage: take <item>", False, None
        it = args[0].lower()
        if it not in spec["items"] or st["loc"][it] is None or st["loc"][it][0] != st["room"]:
            return False, f"The {it} is not in the {st['room']}.", False, None
        st["loc"][it] = None
        st["inv"].append(it)
        return True, f"You take the {it}.", False, None
    if verb == "put":
        if len(args) != 2:
            return False, "Usage: put <item> <container>", False, None
        it, cont = args[0].lower(), args[1].lower()
        if it not in st["inv"]:
            return False, f"You are not carrying the {it}.", False, None
        if cont not in spec["containers"] or spec["croom"][cont] != st["room"]:
            return False, f"There is no {cont} in the {st['room']}.", False, None
        st["inv"].remove(it)
        st["loc"][it] = [st["room"], cont]
        return True, f"You put the {it} in the {cont}.", False, None
    if verb == "inventory":
        return True, f"You are carrying: {', '.join(st['inv']) or 'nothing'}.", False, None
    return False, f"Unknown command: {verb}.", False, None


def ver_world(spec, st):
    for cl in spec["clauses"]:
        where = st["loc"][cl["item"]]
        if where is None:
            return False, f"{cl['item']} is still carried"
        r, c = where
        if c != cl["container"] or r != spec["croom"][cl["container"]]:
            return False, f"{cl['item']} not in {cl['container']}"
    return True, "goal reached"


# --------------------------------------------------------------------------------------
# family 4 -- list operations
# --------------------------------------------------------------------------------------

_LIST_OPS = ["sort", "rsort", "dedup", "reverse", "evens", "odds", "add", "mul"]


def _apply_listop(cur: list, op: str, arg=None):
    if op == "sort":
        return sorted(cur)
    if op == "rsort":
        return sorted(cur, reverse=True)
    if op == "dedup":
        out = []
        for x in cur:
            if x not in out:
                out.append(x)
        return out
    if op == "reverse":
        return list(reversed(cur))
    if op == "evens":
        return [x for x in cur if x % 2 == 0]
    if op == "odds":
        return [x for x in cur if x % 2 != 0]
    if op == "add":
        return [x + arg for x in cur]
    if op == "mul":
        return [x * arg for x in cur]
    raise ValueError(op)


def gen_listops(rng: random.Random, level: int) -> Task:
    n = rng.randint(6, 8)
    start = [rng.randint(1, 20) for _ in range(n)]
    if level >= 1 and rng.random() < 0.7:
        for _ in range(rng.randint(1, 3)):
            i, j = rng.randrange(n), rng.randrange(n)
            start[i] = start[j]
    k = min(1 + level, 4)
    chain = []
    for _ in range(k):
        op = rng.choice(_LIST_OPS)
        arg = rng.randint(2, 5) if op in ("add", "mul") else None
        chain.append((op, arg))
    cur = list(start)
    for op, arg in chain:
        cur = _apply_listop(cur, op, arg)
    steps = "\n".join(f"  {i+1}. {op}" + (f" {arg}" if arg is not None else "")
                      for i, (op, arg) in enumerate(chain))
    body = (
        "Task: transform the starting list by applying the operations below, in the order "
        "given. Duplicates stay in the list unless an operation removes them.\n"
        f"Starting list: {start}\n"
        f"Operations:\n{steps}\n"
        "Report the final list as comma-separated numbers."
    )
    tool_doc = (
        "Allowed command:\n"
        "  listop <op> [arg]  -- applies ONE operation to the CURRENT list and prints the "
        "result. The list itself is kept by the environment; do not retype it.\n"
        "     ops: sort | rsort | dedup | reverse | evens | odds | add <k> | mul <k>\n"
        "Apply the operations one at a time with `listop`, then submit the last printed list."
    )
    return Task(tid="", family="listops", level=level, body=body, tool_doc=tool_doc,
                spec={"start": start, "chain": chain, "answer": cur})


def step_listops(task, st, kind, payload):
    if kind == "answer":
        got = norm_text(payload).strip("[]").replace(" ", "")
        try:
            nums = [int(x) for x in got.split(",") if x != ""]
        except ValueError:
            return True, "", True, False
        return True, "", True, nums == task.spec["answer"]
    if kind != "cmd":
        return False, "No command or answer found. Reply with <cmd>...</cmd> or <answer>...</answer>.", False, None
    toks = payload.split()
    if not toks:
        return False, "Empty command.", False, None
    if toks[0].lower() != "listop":
        return False, f"Unknown command: {toks[0]}.", False, None
    if len(toks) < 2:
        return False, "Usage: listop <op> [arg]", False, None
    op = toks[1].lower()
    arg = None
    if op in ("add", "mul"):
        if len(toks) != 3 or not re.fullmatch(r"\d+", toks[2]):
            return False, f"Usage: listop {op} <k>", False, None
        arg = int(toks[2])
    elif op not in _LIST_OPS:
        return False, f"Unknown list operation: {op}.", False, None
    st["cur"] = _apply_listop(st["cur"], op, arg)
    return True, f"Result: {st['cur']}", False, None


_HANDLERS = {"strxform": step_strxform, "calc": step_calc, "world": step_world,
             "listops": step_listops}
_GENS = {"strxform": gen_strxform, "calc": gen_calc, "world": gen_world,
         "listops": gen_listops}


def step_env(task: Task, st, kind, payload):
    """Dispatch one turn to the family handler. Returns (valid, env_text, finished, ok)."""
    valid, text, finished, res = _HANDLERS[task.family](task, st, kind, payload)
    if isinstance(res, tuple):
        res = bool(res[0])
    return valid, text, finished, (None if res is None else bool(res))


# --------------------------------------------------------------------------------------
# oracle: a reference solution for a task, used to build the demonstrations
# --------------------------------------------------------------------------------------

def _path_between(rooms, a, b):
    ia, ib = rooms.index(a), rooms.index(b)
    step = 1 if ib > ia else -1
    return [rooms[i] for i in range(ia + step, ib + step, step)]


def oracle_turns(task: Task):
    """Return [(assistant_text, env_text), ...] ending with the accepted answer turn."""
    st = task.new_state()
    out = []
    if task.family == "strxform":
        for op in task.spec["chain"]:
            out.append((f"<cmd>apply {op}</cmd>", f"Result: {_apply_op(st['cur'], op)}"))
            st["cur"] = _apply_op(st["cur"], op)
        out.append((f"<answer>{task.spec['answer']}</answer>", ""))
    elif task.family == "calc":
        out.append((f"<cmd>calc {task.spec['expr']}</cmd>", f"Result: {task.spec['value']:g}"))
        ans = task.spec["answer"]
        out.append((f"<answer>{ans:g}</answer>", ""))
    elif task.family == "listops":
        for op, arg in task.spec["chain"]:
            cmd = f"<cmd>listop {op}" + (f" {arg}" if arg is not None else "") + "</cmd>"
            st["cur"] = _apply_listop(st["cur"], op, arg)
            out.append((cmd, f"Result: {st['cur']}"))
        out.append(("<answer>" + ", ".join(map(str, task.spec["answer"])) + "</answer>", ""))
    elif task.family == "world":
        for cl in task.spec["clauses"]:
            it, c = cl["item"], cl["container"]
            if st["loc"][it] is not None and st["loc"][it][1] == c:
                continue
            home = st["loc"][it][0]
            for r in _path_between(task.spec["rooms"], st["room"], home):
                out.append((f"<cmd>go {r}</cmd>", f"You are now in the {r}."))
                st["room"] = r
            out.append((f"<cmd>take {it}</cmd>", f"You take the {it}."))
            st["loc"][it] = None
            st["inv"].append(it)
            for r in _path_between(task.spec["rooms"], st["room"], task.spec["croom"][c]):
                out.append((f"<cmd>go {r}</cmd>", f"You are now in the {r}."))
                st["room"] = r
            out.append((f"<cmd>put {it} {c}</cmd>", f"You put the {it} in the {c}."))
            st["inv"].remove(it)
            st["loc"][it] = [st["room"], c]
        out.append(("<answer>done</answer>", ""))
    return out


_DEMO_SPEC = {"strxform": ("marble", 1), "calc": (None, 1), "world": (None, 0),
              "listops": (None, 1)}


def demo_messages(family: str):
    """One solved episode of ``family`` rendered as chat turns (protocol demonstration)."""
    task = _GENS[family](random.Random(90210), _DEMO_SPEC[family][1])
    msgs = [("user", task.prompt())]
    for a, e in oracle_turns(task):
        msgs.append(("assistant", a))
        if e:
            msgs.append(("user", f"Environment: {e}"))
    return msgs


# --------------------------------------------------------------------------------------
# stream construction
# --------------------------------------------------------------------------------------

def _level_for(phase: int, n_phases: int, family: str) -> int:
    import os
    shift = int(os.environ.get("TTT_LEVEL_SHIFT", "0"))
    frac = (phase + 1) / max(1, n_phases)
    if frac <= 0.34:
        base = 0
    elif frac <= 0.67:
        base = 1
    else:
        base = 2
    if family in ("world", "strxform") and frac > 0.8:
        base = min(3, base + 1)
    return max(0, min(3, base + shift))


def build_stream(seed: int, n_tasks: int, n_phases: int = 4) -> list[Task]:
    rng = random.Random(seed)
    tasks: list[Task] = []
    per_phase = max(1, n_tasks // n_phases)
    for t in range(n_tasks):
        phase = min(n_phases - 1, t // per_phase)
        fam = FAMILIES[t % len(FAMILIES)]
        task = _GENS[fam](rng, _level_for(phase, n_phases, fam))
        task.tid = f"s{seed}-{t:04d}-{fam}"
        tasks.append(task)
    return tasks


def build_val_set(seed: int, n_per_family: int = 5) -> list[Task]:
    """Fixed held-out validation tasks (never trained on)."""
    rng = random.Random(seed)
    tasks: list[Task] = []
    for fam in FAMILIES:
        for j in range(n_per_family):
            task = _GENS[fam](rng, 1 + (j % 2))
            task.tid = f"val{seed}-{fam}-{j}"
            tasks.append(task)
    return tasks
