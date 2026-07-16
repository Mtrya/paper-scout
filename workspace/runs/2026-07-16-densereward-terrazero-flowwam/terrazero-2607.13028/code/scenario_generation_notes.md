# TerraZero Procedural Scenario Generation

## Design principle

Treat logged data as **map geometry only**, not as training distribution.  
One real-world map → combinatorially large scenario space via composable axes.

## Composable axes (Figure 3 of paper)

1. **Agent classes** – any subset of {vehicles, pedestrians, cyclists}
2. **Per-class dynamics** – unicycle (pedestrians), bicycle (cars/cyclists), bicycle + tire forces (trucks)
3. **Initialization** – log / random / hybrid
4. **Signal control** – NEMA dual-ring / Christmas (uncoordinated LogNormal) / round-robin
5. **Rule-based road users (NPCs)** – any subset of 8 generators
6. **Per-episode randomization** – reward weights, kinematic coefficients, bounding-box dims, density

## Map sources

| Source | Use | Scale |
|--------|-----|-------|
| Waymo Open Motion Dataset | Sim-agent training / WOSAC | ~576 K scenarios, 487 K train |
| nuPlan (4 cities, RHD + LHD Singapore) | Planner training / val14 / InterPlan | ~262 K scenarios |
| CARLA towns | Transfer study, synthetic geometry | 5 towns |

## Scenario initialization (Algorithm 1)

1. Load `terrabin` (map geometry, logged trajectories, signal phases).
2. Center coordinates.
3. Build grid map + lane-connectivity graph.
4. **Random mode** (default): sample agent count, rejection-sample collision-free on-lane placements with valid goals.
5. **Log mode**: load timestep-0 positions; agents transition to policy control immediately.
6. **Hybrid mode**: per-environment coin flip between random and log initialization.
7. Assign roles: policy-controlled / log-replay / IDM.
8. Per-agent: sample kinematic coeffs, reward params, dimensions, velocity/heading.
9. Goal assignment: forward walk on lane graph, filtered to be ahead of agent.

## Rule-based NPCs (Figure 4)

### Reactive vehicles
- **IDM + pure pursuit** longitudinal/steering; default/assertive/cautious behavior modes.
- **PDM-Closed-style planner**: forward-sim grid of lateral-offset × IDM proposals; score progress/TTC/comfort/lane-keeping.

### Static actors
- Parked vehicles (curb-anchored, car/bus/truck sizes)
- Crashed-vehicle clusters: disc, rear-end chain, T-bone, outward fan
- Construction zones: cone grid, taper, lane block + optional stationary worker
- Isolated static obstacles offset from lane center

### Pedestrians
- Crosswalk pedestrians under **ORCA** reciprocal avoidance; Poisson schedule.
- Patterns: single, packs, bidirectional flows, staggered sequences.
- Jaywalkers: mid-block, perpendicular until clear of far edge.

### Cyclists
- No scripted controller as NPC; only policy-controlled or log-replay.

## Signal controllers

| Controller | Description |
|------------|-------------|
| NEMA | Dual-ring, 8-phase concurrency plan; green/yellow/all-red; barriers keep conflicting movements apart |
| Christmas (default) | Each stop line cycles independently R/G/Y with LogNormal dwells; uncoordinated |
| Round-robin | One approach leg green at a time, rotating in NEMA order |

Extra variety:
- Initial phase randomized at reset.
- Protected left turns flipped to permissive per intersection.

## Domain randomization

- **Kinematic**: `(c_throttle, c_steer, c_acc, c_vel)` each in `[0.5, 1.5]`, sampled per agent per episode.
- **Reward**: 23 reward parameters, each fixed / sampled per agent / disabled (null).
- **Bounding box**: per-episode type-specific ranges; vehicles split into car/truck/bus classes.
- **Goal dropout**: 30% of vehicles get goal observation zeroed (still visible flag tells network it is masked).

All sampled values are included in the ego observation (reward-conditioned + kinematic-conditioned), so the policy adapts online and can be steered at inference by changing weights.
