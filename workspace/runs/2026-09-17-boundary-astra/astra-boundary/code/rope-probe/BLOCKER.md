# BLOCKER — L3 (winding the rope around the peg)

State at hand-off: L0 5/5, L1 5/5, L2 5/5, **L3 0/3** (seeds 1-3, and 0/5).
The obstacle is not the oracle and not the rope physics. It is a coordinate
bug that puts the physical peg **0.7071 m away from the peg the level is graded
against**, so the "winding" trajectories are drags in free space.

## The bug

`core.py`, `build_level` for `"L3"`:

```python
'<body name="peg" pos="%s %s 0">' ... % (-WORLD_OFF, -WORLD_OFF, ...)
```

The task frame is `[0,1]x[0,1]` while MuJoCo's world centres the 1x1 m table on
the origin (`WORLD_OFF = 0.5`), so `(-WORLD_OFF, -WORLD_OFF)` is the table's
**corner**. Everything else about L3 uses the other convention:

| quantity                                   | frame value            |
| ------------------------------------------ | ---------------------- |
| physical peg (world)                       | `[-0.5, -0.5]`         |
| physical peg (task, = world + WORLD_OFF)   | `[0.0, 0.0]` — corner  |
| `task_mod.PEG_XY` = graded axis, reported in `state`, and the centre the init layout is built around | `[0.5, 0.5]` — centre |
| distance between the two                   | **0.7071 m**           |

So the rope never has a post to press against where the winding is measured,
and the peg it *could* touch sits at a corner the rope never visits.

Evidence (current model, 100 moves/seed, free end walked around the graded
axis at r = 0.035): winding delta per lap **+0.00 / +0.01 / +0.00** — the
winding oscillates with the drag and nets zero. Best deltas seen anywhere with
this model: seed1 **+0.755**, seed2 **+0.881**, seed3 **+0.254** (see below).

## The fix (one line, deliberately NOT applied — the environment is frozen)

```diff
--- a/code/rope-probe/core.py
+++ b/code/rope-probe/core.py
@@
-            % (-WORLD_OFF, -WORLD_OFF, PEG_RAD, PEG_H / 2, PEG_H / 2, cfg["fric"])
+            % (0.0, 0.0, PEG_RAD, PEG_H / 2, PEG_H / 2, cfg["fric"])
```

i.e. the peg's body position goes from `-0.5 -0.5 0` to `0 0 0`.
Apply with:

```sh
cd code/rope-probe && sed -i 's/% (-WORLD_OFF, -WORLD_OFF, PEG_RAD, PEG_H/% (0.0, 0.0, PEG_RAD, PEG_H/' core.py
```

L0/L1/L2 models contain no peg, so their physics, CLI and graders cannot change.

## Verification with the fix (in-memory patch, no file modified)

`/tmp/l3fix_test.py` builds each L3 episode, sets `m.body_pos[peg] = [0,0,0]`
on the live model, then runs the oracle's `policy_L3` unchanged:

```
L3/seed1 PASS  winding_delta 1.0585  score 1.0  moves 58
L3/seed2 PASS  winding_delta 1.0686  score 1.0  moves 57
L3/seed3 PASS  winding_delta 1.0761  score 1.0  moves 57
```

The policy that does it: grab the free end, walk it once around the **graded**
peg at r = 0.035 in the direction *opposite* the level's `direction` label
(the label names the sense in which the graded winding must grow, and the
graded winding is the raw winding times the label's sign), stop at
delta >= 1.05. One lap is worth ~1.0 of winding once the rope can push against
the post; without a post the same lap is worth ~0.

## What was tried before the diagnosis (all of it inside `oracle.py`)

- sweep the free end around the graded axis at r ∈ {0.03, 0.045, 0.06, 0.10},
  1-5 laps, both senses, winding tracked every step;
- "lasso" the free end around the *anchored* end at R ∈ {0.03, 0.06, 0.12,
  0.16, 0.18, 0.20, 0.30}, both senses, 1-4 laps (this is what produced the
  +0.755 / +0.881 / +0.254 above — the loop closes around the peg for the
  first two seeds, not for the third);
- swapping which end is grabbed (node 0 vs node n).

Diagnosis: the level as specified is solvable (proved above) and the grader,
the layout generator and the state output are all self-consistent; only the
physical peg's placement is wrong. This is a model bug, not an oracle
algorithm limit — no amount of trajectory search inside `oracle.py` can move
the post.

## Visible in every render

The render added for the framing fix shows it directly: the peg draws as a
small dark dot at the **bottom-left outside the tabletop**, while the rope is
laid around the table's centre. Same bug, same line.

## Files in this run

- `oracle.py` — oracle only (all L3 attempts live here; L0/L1 policies rewritten
  earlier this session)
- `core.py` — joint damping `3.0 -> 0.02` (ball joints were overdamped shut:
  the chain acted as one rigid piece), joint range written in degrees, peg
  contype/conaffinity so the rope can actually touch the post
- `robot_rope.py` — render framing only (fovy-derived distance, explicit
  azimuth/elevation, `MUJOCO_GL` set before `import mujoco`)
- `geom.py` — `crossings_2d` was skipping the first x last segment pair, so
  every self-crossing scene read as 0 crossings (no grader calls it)
