"""Canonical field layout + typed token codec (MASS App. A.2 / B.2).

Field vector = the canonical non-redundant state: tick, 64 sorted food coords,
player_count, and per player slot an ordered body list, length, heading,
survival bit and optional death anchor. Token vocabulary uses disjoint typed
ranges; every field sits at a fixed sequence position so a schema-derived mask
can constrain each output position.

Token layout (matched Snake, 8 player slots):
  BOS, MARK_STATE, MARK_ACTION, MARK_EXOG, MARK_OUTPUT, PAD   (0..5)
  NIBBLE  6..21     (16 hex digits, tick as 4 nibbles MSD-first)
  HEADING 22..25    (up/right/down/left)
  ALIVE   26..27
  BODYLEN 28..68    (0..40)
  PCOUNT  69..76    (player_count 1..8)
  SCOUNT  77..85    (spawn_count 0..8)
  AVAL    86..90    (noop/up/right/down/left)
  CELL    91..2394  (flattened y*48+x)
State segment: 4 nibbles + 64 food + 1 pcount + 8 x 44 slots = 421 tokens.
Prefix: [BOS, MARK_STATE] + 421 + [MARK_ACTION] + 8 + [MARK_EXOG] + 4 + 1 + 8
        + [MARK_OUTPUT] = 447 tokens. Target = next state's 421 tokens.
"""
import numpy as np

from snake_engine import ARENA, N_FOOD, MAX_BODY, MAX_SLOTS, MAX_SPAWNS

BOS, MARK_STATE, MARK_ACTION, MARK_EXOG, MARK_OUTPUT, PAD = range(6)
NIBBLE_BASE = 6        # 16
HEADING_BASE = 22      # 4
ALIVE_BASE = 26        # 2
BODYLEN_BASE = 28      # 41 (0..40)
PCOUNT_BASE = 69       # 8  (1..8)
SCOUNT_BASE = 77       # 9  (0..8)
AVAL_BASE = 86         # 5  (noop + 4 dirs)
CELL_BASE = 91         # 48*48 = 2304
VOCAB_SIZE = CELL_BASE + ARENA * ARENA

N_STATE_TOKENS = 4 + N_FOOD + 1 + MAX_SLOTS * (1 + MAX_BODY + 3)  # 421
SLOT_W = 1 + MAX_BODY + 3  # 44
PREFIX_LEN = 2 + N_STATE_TOKENS + 1 + MAX_SLOTS + 1 + 4 + 1 + MAX_SPAWNS + 1  # 447
SEQ_LEN = PREFIX_LEN + N_STATE_TOKENS  # 868

# field-vector layout (per slot): len, body[40], heading, alive, dead_at
SLOT_FIELDS = 1 + MAX_BODY + 3


def state_to_fields(state, n_players):
    """Engine State -> canonical field dict of int arrays."""
    food = np.full(N_FOOD, -1, dtype=np.int64)
    food[:len(state.food)] = state.food[:N_FOOD]
    slots = []
    for i in range(MAX_SLOTS):
        if i >= n_players:
            slots.append(dict(length=0, body=np.full(MAX_BODY, -1, dtype=np.int64),
                              heading=-1, alive=0, dead_at=-1, inactive=True))
        else:
            s = state.snakes[i]
            body = np.full(MAX_BODY, -1, dtype=np.int64)
            body[:len(s["body"])] = s["body"][:MAX_BODY]
            slots.append(dict(length=len(s["body"]), body=body,
                              heading=s["heading"] if s["alive"] else -1,
                              alive=1 if s["alive"] else 0,
                              dead_at=s["dead_at"], inactive=False))
    return dict(tick=state.tick, food=food, player_count=n_players, slots=slots)


def fields_to_state(fields):
    """Canonical fields -> engine State (for legality checks / rollout compare)."""
    from snake_engine import State
    n = int(fields["player_count"])
    snakes = []
    for i in range(n):
        sl = fields["slots"][i]
        L = int(sl["length"])
        snakes.append(dict(body=[int(c) for c in sl["body"][:L] if c >= 0],
                           heading=int(sl["heading"]) if sl["heading"] >= 0 else 0,
                           alive=bool(sl["alive"]),
                           dead_at=int(sl["dead_at"])))
    return State(int(fields["tick"]), [int(c) for c in fields["food"] if c >= 0], snakes)


def encode_state_tokens(fields):
    """Canonical fields -> 421 state tokens."""
    tok = np.empty(N_STATE_TOKENS, dtype=np.int64)
    tick = int(fields["tick"]) & 0xFFFF
    for k in range(4):
        tok[k] = NIBBLE_BASE + ((tick >> (12 - 4 * k)) & 0xF)
    tok[4:4 + N_FOOD] = [CELL_BASE + c if c >= 0 else PAD for c in fields["food"]]
    tok[4 + N_FOOD] = PCOUNT_BASE + int(fields["player_count"]) - 1
    off = 4 + N_FOOD + 1
    for i, sl in enumerate(fields["slots"]):
        b = off + i * SLOT_W
        if sl["inactive"]:
            tok[b:b + SLOT_W] = PAD
            continue
        L = int(sl["length"])
        tok[b] = BODYLEN_BASE + L
        tok[b + 1:b + 1 + MAX_BODY] = PAD
        for j in range(L):
            tok[b + 1 + j] = CELL_BASE + int(sl["body"][j])
        tok[b + 1 + MAX_BODY] = HEADING_BASE + sl["heading"] if sl["heading"] >= 0 else PAD
        tok[b + 2 + MAX_BODY] = ALIVE_BASE + int(sl["alive"])
        tok[b + 3 + MAX_BODY] = CELL_BASE + sl["dead_at"] if sl["dead_at"] >= 0 else PAD
    return tok


def decode_state_tokens(tok):
    """421 state tokens -> canonical fields (assumes mask-valid decoding)."""
    tok = np.asarray(tok)
    tick = 0
    for k in range(4):
        tick = (tick << 4) | int(tok[k] - NIBBLE_BASE)
    food = np.array([t - CELL_BASE for t in tok[4:4 + N_FOOD]], dtype=np.int64)
    player_count = int(tok[4 + N_FOOD] - PCOUNT_BASE) + 1
    off = 4 + N_FOOD + 1
    slots = []
    for i in range(MAX_SLOTS):
        b = off + i * SLOT_W
        seg = tok[b:b + SLOT_W]
        if np.all(seg == PAD):
            slots.append(dict(length=0, body=np.full(MAX_BODY, -1, dtype=np.int64),
                              heading=-1, alive=0, dead_at=-1, inactive=True))
            continue
        L = int(seg[0] - BODYLEN_BASE)
        body = np.full(MAX_BODY, -1, dtype=np.int64)
        for j in range(L):
            body[j] = seg[1 + j] - CELL_BASE
        heading = seg[1 + MAX_BODY] - HEADING_BASE if seg[1 + MAX_BODY] != PAD else -1
        alive = int(seg[2 + MAX_BODY] - ALIVE_BASE)
        dead_at = seg[3 + MAX_BODY] - CELL_BASE if seg[3 + MAX_BODY] != PAD else -1
        slots.append(dict(length=L, body=body, heading=heading,
                          alive=alive, dead_at=dead_at, inactive=False))
    return dict(tick=tick, food=food, player_count=player_count, slots=slots)


def build_prefix(fields, actions, spawn_cells):
    """Current state + joint actions + exogenous (net-added food) + OUTPUT mark.

    actions: int list len MAX_SLOTS. spawn_cells: list of flat idx (<=8).
    Returns int64[PREFIX_LEN].
    """
    seq = [BOS, MARK_STATE]
    seq += encode_state_tokens(fields).tolist()
    seq.append(MARK_ACTION)
    seq += [AVAL_BASE + int(a) for a in actions]
    seq.append(MARK_EXOG)
    tick = int(fields["tick"]) & 0xFFFF
    seq += [NIBBLE_BASE + ((tick >> (12 - 4 * k)) & 0xF) for k in range(4)]
    seq.append(SCOUNT_BASE + len(spawn_cells))
    seq += [CELL_BASE + int(c) for c in spawn_cells]
    seq += [PAD] * (MAX_SPAWNS - len(spawn_cells))
    seq.append(MARK_OUTPUT)
    assert len(seq) == PREFIX_LEN
    return np.array(seq, dtype=np.int64)


# ---------------------------------------------------------------- masks ----

def static_output_mask():
    """Per-position allowed vocabulary for the 421 output positions (bool[V])."""
    m = np.zeros((N_STATE_TOKENS, VOCAB_SIZE), dtype=bool)
    for k in range(4):
        m[k, NIBBLE_BASE:NIBBLE_BASE + 16] = True
    m[4:4 + N_FOOD, CELL_BASE:CELL_BASE + ARENA * ARENA] = True
    m[4 + N_FOOD, PCOUNT_BASE:PCOUNT_BASE + 8] = True
    off = 4 + N_FOOD + 1
    for i in range(MAX_SLOTS):
        b = off + i * SLOT_W
        m[b, BODYLEN_BASE:BODYLEN_BASE + MAX_BODY + 1] = True
        m[b, PAD] = True                       # inactive slot
        m[b + 1:b + 1 + MAX_BODY, CELL_BASE:CELL_BASE + ARENA * ARENA] = True
        m[b + 1:b + 1 + MAX_BODY, PAD] = True
        m[b + 1 + MAX_BODY, HEADING_BASE:HEADING_BASE + 4] = True
        m[b + 1 + MAX_BODY, PAD] = True
        m[b + 2 + MAX_BODY, ALIVE_BASE:ALIVE_BASE + 2] = True
        m[b + 2 + MAX_BODY, PAD] = True
        m[b + 3 + MAX_BODY, CELL_BASE:CELL_BASE + ARENA * ARENA] = True
        m[b + 3 + MAX_BODY, PAD] = True
    return m


def allowed_tokens(position, decoded, player_count):
    """Dynamic selector: static range + canonical constraints.

    Enforces: food strictly ascending (sorted unique); body cells unique within
    a body; canonical padding after body length; dead vs alive slot consistency;
    fully-padded inactive slots. Movement/collision/growth/death rules are NOT
    encoded here — the model learns them.
    """
    if position < 4:
        return np.arange(NIBBLE_BASE, NIBBLE_BASE + 16)
    if position < 4 + N_FOOD:
        k = position - 4  # food index 0..63
        lo = 0 if k == 0 else int(decoded[position - 1] - CELL_BASE) + 1
        # leave room for the remaining 63-k larger cells
        hi = ARENA * ARENA - (N_FOOD - 1 - k)
        return np.arange(CELL_BASE + lo, CELL_BASE + hi)
    if position == 4 + N_FOOD:
        return np.arange(PCOUNT_BASE, PCOUNT_BASE + 8)
    rel = position - (4 + N_FOOD + 1)
    slot, off = divmod(rel, SLOT_W)
    base = 4 + N_FOOD + 1 + slot * SLOT_W
    if slot >= player_count:
        return np.array([PAD])
    L = int(decoded[base] - BODYLEN_BASE) if off > 0 else None
    if off == 0:
        return np.arange(BODYLEN_BASE, BODYLEN_BASE + MAX_BODY + 1)
    if off <= MAX_BODY:
        j = off - 1
        if j >= L:
            return np.array([PAD])
        used = {int(decoded[base + 1 + k] - CELL_BASE) for k in range(j)}
        return np.array([CELL_BASE + c for c in range(ARENA * ARENA) if c not in used])
    if off == MAX_BODY + 1:  # heading
        return np.array([PAD]) if L == 0 else np.arange(HEADING_BASE, HEADING_BASE + 4)
    if off == MAX_BODY + 2:  # alive
        return np.array([ALIVE_BASE + (0 if L == 0 else 1)])
    # dead_at
    return np.arange(CELL_BASE, CELL_BASE + ARENA * ARENA) if L == 0 else np.array([PAD])


def check_contradiction(fields):
    """Structural violations in decoded fields. Returns list of violation strs."""
    v = []
    food = [int(c) for c in fields["food"]]
    if len(set(food)) != N_FOOD or any(food[i] > food[i + 1] for i in range(len(food) - 1)):
        v.append("food_not_sorted_unique")
    for i, sl in enumerate(fields["slots"]):
        if sl["inactive"]:
            continue
        L = sl["length"]
        body = [int(c) for c in sl["body"][:L]]
        if sl["alive"]:
            if L < 1 or (body and min(body) < 0):
                v.append(f"slot{i}_alive_bad_body")
            if len(set(body)) != L:
                v.append(f"slot{i}_body_dup")
            if sl["heading"] < 0:
                v.append(f"slot{i}_missing_heading")
        else:
            if sl["dead_at"] < 0:
                v.append(f"slot{i}_dead_no_anchor")
    return v
