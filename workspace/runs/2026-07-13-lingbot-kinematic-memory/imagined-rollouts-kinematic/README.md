# Imagined Rollouts Are Kinematic, Not Dynamic — verification thread

This thread reproduces the iKCE diagnostic from:

> Finn Rasmus Schäfer, Korbinian Moller, Yuan Gao, Christian Oefinger, Sebastian Schmidt and Johannes Betz.  
> "Imagined Rollouts Are Kinematic, Not Dynamic: A Diagnosis of Long-Horizon World-Model Failure."  
> arXiv:2607.05966, RSS 2026 World Model Workshop.

The official diagnostic code is at `https://github.com/TUM-AVS/iKCE`; the DreamerV3 PyTorch port is `NM512/dreamerv3-torch@6ef8646`. The released checkpoint lives on Hugging Face as `TUM-AVS/ikce-walker-walk-artifacts`.

## What was done

1. **Cloned the upstream repositories** into `code/ikce` and `code/dreamerv3-torch`.
2. **Created a Python 3.11 virtualenv** (`code/ikce/.venv`) and installed `ikce-diagnostic[dmc]` plus the port's runtime dependencies.
3. **Downloaded the released checkpoint** `checkpoints/walker_walk/latest.pt` and `config.yaml` from Hugging Face with `curl` (the `huggingface_hub` Python client could not reach the Hub from this environment).
4. **Applied the local patch** documented in the paper (set `MUJOCO_GL` only if unset) so the GLFW backend could be used headless.
5. **Ran smoke tests** and the DMC sanity check.
6. **Ran reduced and full friction sweeps** on both the DreamerV3 world model and the real DMC physics, driven by the trained policy, at horizon T=64 with the identity kinematic view (root height z and vertical velocity vz).
7. **Ran a joint-noise positive control** to confirm the WM does respond to kinematic-axis perturbations.
8. **Ran the horizon-emergence reanalysis** on the full sweep, recomputing the log-log friction slope at T ∈ {8,16,32,64} from the saved per-step iKCE traces.

## How to rerun

```bash
cd code/ikce
source .venv/bin/activate
export PYTHONPATH=/path/to/dreamerv3-torch:$PYTHONPATH
export MUJOCO_GL=glfw

# smoke test
python scripts/test_dreamer_rollout.py --checkpoint checkpoints/walker_walk --horizon 16

# reduced sweep
python scripts/run_perturbation_sweep.py configs/probe_wm.yaml
python scripts/run_perturbation_sweep.py configs/probe_physics.yaml

# full headline sweep
python scripts/run_perturbation_sweep.py configs/probe_wm_full.yaml
python scripts/run_perturbation_sweep.py configs/probe_physics_full.yaml

# joint-noise positive control
python scripts/run_perturbation_sweep.py configs/probe_wm_jointnoise.yaml
```

The probe configs and result CSVs are preserved in this thread under `code/` and `assets/`.

## Key reproduced numbers (T=64, identity view, K=20)

| side | μ=1.0 iKCE | log-log slope vs μ | note |
|---|---|---|---|
| WM (imagined) | 2.27e-3 | −0.018 (p=0.36) | statistically flat |
| physics + policy | 9.18e-5 | −0.154 (p=5e-5) | downward with friction |
| WM/physics ratio at μ=1.0 | 24.7× | — | matches paper's ~30× |

The WM is therefore roughly friction-invariant across a 17× friction range that includes the gait-collapse boundary, while real physics shows a clear negative slope. A kinematic perturbation (joint noise) does make the WM's iKCE rise, confirming the flatness is specific to dynamic/regime perturbations.

## Horizon emergence

| source | T=8 | T=16 | T=32 | T=64 |
|---|---|---|---|---|
| physics slope β | +0.005 (CI contains 0) | +0.004 (CI contains 0) | −0.016 (CI contains 0) | **−0.206** (CI excludes 0) |
| WM slope β | −0.065 (CI contains 0) | −0.039 (CI contains 0) | −0.024 (CI contains 0) | −0.020 (CI contains 0) |

The dynamic signature only emerges at T=64, matching the paper's claim that the diagnostic must be run longer than the gait period.

## Files

- `code/probe_*.yaml` — reduced/full sweep configs used for verification.
- `code/analyze_probe.py` — reproduces the plot and log-log slope numbers from the CSVs.
- `assets/ikce_*.csv` — per-sweep summary CSVs.
- `assets/ikce_horizon_sweep.csv` — horizon-emergence slope table.
- `assets/ikce_friction_probe.png` — reproduced headline curve.
- `patches/dreamerv3-torch_mujoco_gl.patch` — local patch needed to run headless with GLFW.

## Blockers / caveats

- The `huggingface_hub` Python client could not download from the Hub in this environment; artifacts were fetched with `curl` instead.
- The run used CPU inference; absolute iKCE magnitudes match the paper's CPU/deterministic expectations but the physics-side slope is somewhat weaker than the paper's CUDA run (−0.154 vs −0.220).
- Only the default `walker_walk` checkpoint was verified; the h=64 actor-horizon ablation and domain-randomization checkpoints were not downloaded.
