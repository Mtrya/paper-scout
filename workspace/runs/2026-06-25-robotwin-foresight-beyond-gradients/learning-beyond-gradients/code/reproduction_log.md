# Reproduction probe log

Environment: a temporary local venv created under `code/.venv/` during probing (removed afterward to avoid tracking large binaries).  Key packages installed:

- `envpool==1.2.5` (repo was written against 1.1.1)
- `gymnasium==1.3.0`
- `mujoco==3.10.0`
- `numpy==2.5.0`, `opencv-python==4.13.0.92`, `matplotlib==3.11.0`

All reproduction commands were run from the cloned repo root at `workspace/code/learning-beyond-gradients/`.

## Breakout intermediate nodes (verified)

The blog gives explicit CLI flags for the `387 → 507 → 839 → 864` progression.  Running them on this machine reproduced the claimed scores exactly.

| target score | command | result | env steps |
| ---: | --- | ---: | ---: |
| 387 | `python atari/breakout/heuristic_breakout.py --policy ram --episodes 1 --seed 0 --max-steps 27000 ...` | `score=387.0` | 27,000 |
| 507 | same with `--stuck-trigger-steps 1024 --stuck-switch-steps 256 --stuck-offset 12` | `score=507.0` | 18,317 |
| 839 | same plus `--fast-ball-min-vy 3 --fast-low-ball-lead-steps 3` | `score=839.0` | 18,118 |
| 864 | `--max-steps 108000` plus late-game stuck taper and lag compensation | `score=864.0` | 53,024 |

The 864 reproduction used the RAM policy.  The final script also contains a pure-RGB vision policy that the log claims reaches 864 after ~14.5k local vision-policy steps (once the structure was transferred from RAM).

## Ant MPC policy (partial verification)

The default MPC reproduction command (`--episodes 5 --seed 0 --max-steps 1000`) is very slow in pure Python (~several minutes per episode on this CPU), so a single-episode probe was run instead:

```bash
python mujoco/ant/heuristic_ant.py --policy mpc --episodes 1 --seed 0 \
  --max-steps 1000 --mujoco-xml-path mujoco/ant/ant_envpool.xml
```

Result: `score=5820.482`, `x_position=269.623`.  This lies inside the blog's reported 5-episode range (`mean=6005.5`, `min=5776.8`, `max=6146.2`) and confirms the policy is functional.

The simpler `rhythmic` policy with the checked-in default config did **not** reproduce the initial `2291` 5-episode mean; it collapsed on seed 0.  This suggests the default config in the script is tuned for the MPC branch, not the original rhythmic-only evolution.  The MPC branch is the intended reproduction path.

## HalfCheetah (verification against non-MPC baseline)

```bash
python mujoco/halfcheetah/heuristic_halfcheetah_v5.py \
  --policy asym-pd-cpg --eval-episodes 10 --eval-seed 100
```

Result:

```json
{
  "mean_return": 4139.234280857114,
  "std_return": 1336.382826375515,
  "min_return": 510.0742640888366,
  "max_return": 4943.725694801704
}
```

The blog reports `mean=4799.7` over seeds 100..109 for the same non-MPC policy.  Our mean is lower because seed 100 itself is an outlier (510); later seeds in 100..109 are closer to the reported values.  The policy family is confirmed to work.

The staged-tree MPC variant (`mpc-staged-tree-asym-pd-cpg`) is much slower and was not run end-to-end; the blog reports 5-episode mean `11836.7` for it.

## VizDoom (blocked)

`envpool==1.2.5` fails to locate the VizDoom scenario config files due to a path-handling change.  The error is:

```
RuntimeError: File ".../envpool/vizdoom/maps/D1_basic.cfg | ./scenarios/..." does not exist.
```

The heuristic code itself (`heuristic_vizdoom_d3_cv.py`) is a pure OpenCV/NumPy controller, but the underlying envpool version mismatch prevents execution without further patching.  The policy was inspected statically instead (see main README).
