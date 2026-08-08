"""Multi-player Snake engine: 48x48 arena, N players, 64 food slots.

Reproduction target: MASS (2608.06257) matched Snake benchmark (App. B.1/B.2).
Rules: walls kill; head-into-body kills; eating food grows by 1 and respawns a
new food (the net-added food coordinate is the exogenous input). Dead snakes
leave the board but keep a record slot with alive=0 and a death anchor cell.
"""
import numpy as np

ARENA = 48
N_FOOD = 64
MAX_BODY = 40
MAX_SLOTS = 8
MAX_SPAWNS = 8

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIRS = (( -1, 0), (0, 1), (1, 0), (0, -1))
A_NOOP = 0  # keep current heading; 1..4 = absolute up/right/down/left


def flat(y, x):
    return y * ARENA + x


def unflat(i):
    return divmod(int(i), ARENA)


def in_bounds(y, x):
    return 0 <= y < ARENA and 0 <= x < ARENA


class State:
    __slots__ = ("tick", "food", "snakes")

    def __init__(self, tick, food, snakes):
        self.tick = tick
        self.food = sorted(food)  # flat indices, ascending
        # snakes: list of dicts {body: [flat,..] head-first, heading, alive, dead_at}
        self.snakes = snakes

    def copy(self):
        return State(self.tick, list(self.food),
                     [dict(body=list(s["body"]), heading=s["heading"],
                           alive=s["alive"], dead_at=s["dead_at"]) for s in self.snakes])


def init_state(rng, n_players):
    """Random non-overlapping snakes of length 3, 64 random food."""
    occupied = set()
    snakes = []
    for i in range(n_players):
        while True:
            heading = int(rng.integers(4))
            dy, dx = DIRS[heading]
            hy = int(rng.integers(2, ARENA - 2))
            hx = int(rng.integers(2, ARENA - 2))
            body = [flat(hy - k * dy, hx - k * dx) for k in range(3)]
            cells = set(body)
            # keep snakes well separated at t=0
            ok = cells.isdisjoint(occupied)
            if ok:
                for other in snakes:
                    oy, ox = unflat(other["body"][0])
                    if abs(oy - hy) + abs(ox - hx) < 8:
                        ok = False
                        break
            if ok:
                break
        occupied |= cells
        snakes.append(dict(body=body, heading=heading, alive=True, dead_at=-1))
    food = []
    while len(food) < N_FOOD:
        c = int(rng.integers(ARENA * ARENA))
        if c not in occupied and c not in food:
            food.append(c)
    return State(0, food, snakes)


def step(state, actions, rng=None, spawn_cells=None, deterministic_spawn=False):
    """Advance one tick. Returns (new_state, events).

    actions: list of MAX_SLOTS ints (only alive players' entries used).
    spawn_cells: replay mode — exact coordinates for food respawns (exogenous).
    deterministic_spawn: respawn at first free cell in scan order (probe 2).
    events: dict(eat, death, spawn, spawn_cells).
    """
    n = len(state.snakes)
    food_set = set(state.food)
    events = dict(eat=False, death=False, spawn=False, spawn_cells=[])

    new_heading, new_head, growing, eaten = {}, {}, {}, {}
    for i, s in enumerate(state.snakes):
        if not s["alive"]:
            continue
        a = actions[i] if i < len(actions) else A_NOOP
        h = s["heading"] if a == A_NOOP else a - 1
        dy, dx = DIRS[h]
        hy, hx = unflat(s["body"][0])
        ny, nx = hy + dy, hx + dx
        new_heading[i] = h
        new_head[i] = flat(ny, nx) if in_bounds(ny, nx) else -1
        eaten[i] = new_head[i] in food_set
        growing[i] = eaten[i] and len(s["body"]) < MAX_BODY

    # cells occupied after the move (tails of non-growing snakes vacate)
    occupied = set()
    for j, s in enumerate(state.snakes):
        if not s["alive"]:
            continue
        cells = s["body"] if growing.get(j, False) else s["body"][:-1]
        occupied |= set(cells)

    # collision detection
    dies = {}
    head_claims = {}
    for i in new_head:
        nh = new_head[i]
        if nh == -1 or nh in occupied:
            dies[i] = True
        head_claims.setdefault(nh, []).append(i)
    for nh, claimants in head_claims.items():
        if len(claimants) > 1:  # head-on tie: all die
            for i in claimants:
                dies[i] = True

    snakes_out = []
    for i, s in enumerate(state.snakes):
        s2 = dict(s)
        s2["body"] = list(s["body"])
        if s["alive"]:
            if dies.get(i, False):
                s2["alive"] = False
                nh = new_head[i]
                s2["dead_at"] = nh if nh != -1 else s["body"][0]
                s2["body"] = []
                events["death"] = True
            else:
                nh = new_head[i]
                s2["heading"] = new_heading[i]
                s2["body"] = [nh] + s2["body"]
                if eaten[i]:
                    food_set.discard(nh)
                    events["eat"] = True
                if not growing[i]:
                    s2["body"].pop()
        snakes_out.append(s2)

    # respawn food back to 64 (net-added food = exogenous input)
    body_cells = set()
    for s in snakes_out:
        body_cells |= set(s["body"])
    while len(food_set) < N_FOOD:
        k = len(events["spawn_cells"])
        if spawn_cells is not None and k < len(spawn_cells):
            c = spawn_cells[k]
        elif deterministic_spawn:
            # deterministic hash of (tick, k): spawn is a pure function of state
            free = sorted(c for c in range(ARENA * ARENA)
                          if c not in body_cells and c not in food_set)
            c = free[(1103515245 * (state.tick * 8 + k) + 12345) % len(free)]
        else:
            free = np.array([c for c in range(ARENA * ARENA)
                             if c not in body_cells and c not in food_set])
            c = int(free[rng.integers(len(free))])
        food_set.add(c)
        events["spawn_cells"].append(c)
        events["spawn"] = True

    return State(state.tick + 1, sorted(food_set), snakes_out), events


def _occupied_after(state):
    occ = set()
    for s in state.snakes:
        if s["alive"]:
            occ |= set(s["body"][:-1])  # conservative: ignore tail vacating
    return occ


def heuristic_action(state, i, rng, eps=0.1, deterministic=False):
    """Greedy nearest-food policy with collision avoidance."""
    s = state.snakes[i]
    if not s["alive"]:
        return A_NOOP
    hy, hx = unflat(s["body"][0])
    occ = _occupied_after(state)
    occ.discard(s["body"][0])

    def safe(d):
        dy, dx = DIRS[d]
        ny, nx = hy + dy, hx + dx
        return in_bounds(ny, nx) and flat(ny, nx) not in occ

    cand = [d for d in range(4) if safe(d)]
    if not cand:
        return A_NOOP  # doomed; keep heading and die
    if state.food:
        fy, fx = unflat(min(state.food,
                            key=lambda c: abs(unflat(c)[0] - hy) + abs(unflat(c)[1] - hx)))
        cand.sort(key=lambda d: abs(hy + DIRS[d][0] - fy) + abs(hx + DIRS[d][1] - fx))
    if not deterministic and rng is not None and rng.random() < eps:
        d = int(cand[rng.integers(len(cand))])
    else:
        d = int(cand[0])
    return A_NOOP if d == s["heading"] else d + 1


def gen_episode(seed, n_players, n_steps, eps=0.1, deterministic=False):
    """Returns dict of per-transition arrays (see fields.py for layout)."""
    rng = np.random.default_rng(seed)
    state = init_state(rng, n_players)
    rec = []
    for t in range(n_steps):
        actions = [heuristic_action(state, i, rng, eps, deterministic)
                   for i in range(n_players)] + [A_NOOP] * (MAX_SLOTS - n_players)
        new_state, events = step(state, actions, rng=rng,
                                 deterministic_spawn=deterministic)
        rec.append((state, actions, events, new_state))
        state = new_state
    return rec
