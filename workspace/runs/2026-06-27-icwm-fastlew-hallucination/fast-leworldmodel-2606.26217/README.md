# Fast LeWorldModel — Deep Thread

**Paper:** Fast LeWorldModel (arXiv:2606.26217)  
**Thread goal:** Make the action-prefix mechanism concrete, inspect the official code (or the lack of it), and preserve a runnable probe that contrasts Fast-LeWM's prefix prediction with LeWM's autoregressive rollout.

## What I investigated

1. **Read the full paper.** Identified:
   - **Core mechanism:** replace LeWM's repeated one-step latent transition `ẑ_{t+1}=F(z_t,a_t)` with an action-prefix encoder `E_ψ` and a parallel predictor `G_ϕ` that map the anchor latent `z_t` and each action prefix `a_{t:t+k-1}` to the corresponding future latent `ẑ_{t+k}` in one forward pass.
   - **Training objective:** dense prefix MSE loss over `k=1..H` plus SIGReg anti-collapse regularizer.
   - **Planning protocol:** same CEM-based goal-conditioned latent planning as LeWM, but each candidate action sequence is scored by predicting the terminal latent directly from its length-`H` prefix token; an optional self-consistency term can penalize disagreement between direct and two-stage prefix estimates.
   - **Key results:** average success 85.8% → 90.5% (92.0% with self-consistency); dynamics-module time 31.4s → 8.0s on Two-Room; full CEM solve time 54.4s → 28.3s; open-loop latent error grows more slowly with horizon.

2. **Cloned the official Fast-LeWM repository:**
   - `code/fast-leworldmodel-2606.26217` in the workspace is the cloned repo.
   - **It is currently an empty placeholder.** `README.md` says: "We plan to release the full codebase upon paper acceptance." (commit from the live GitHub repo).
   - This means none of the model definition, action-prefix encoder, parallel predictor, training loop, or planning/CEM code can be traced directly.

3. **Inspected the closest available baseline code — LeWM (Maes et al., 2026):**
   - Source: `code/leworldmodel-baseline/` (cloned from https://github.com/lucas-maes/le-wm).
   - LeWM's rollout is exactly the autoregressive chain Fast-LeWM targets. See `code/leworldmodel-baseline/jepa.py`, lines 87–104:
     ```python
     for t in range(n_steps):
         act_emb = self.action_encoder(act)
         emb_trunc = emb[:, -HS:]            # history of latents
         act_trunc = act_emb[:, -HS:]        # history of action embeddings
         pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
         emb = torch.cat([emb, pred_emb], dim=1)
         next_act = act_future[:, t : t + 1, :]
         act = torch.cat([act, next_act], dim=1)
     ```
     The predictor is invoked once per horizon step, each time re-encoding actions and conditioning on previously predicted latents — the compounding error source the paper identifies.
   - LeWM config (`code/leworldmodel-baseline/config/train/model/lewm.yaml`) uses a 6-layer Transformer predictor (`module.ARPredictor`), `history_size=3`, `embed_dim=192`, action encoder via `module.Embedder`. Fast-LeWM's paper swaps this for a 3-layer causal action-prefix Transformer and a 6-layer action-modulated residual MLP predictor; it claims 17.9M params vs LeWM's 18.0M.

4. **Built a toy probe** because the official code is unreleased.
   - File: `code/prefix_vs_ar_probe.py`
   - It creates a nonlinear latent dynamics system, fits (a) a one-step linear model and (b) a set of direct per-horizon linear prefix models, then compares open-loop error and model-call cost.
   - The prefix predictor is evaluated as a single batched forward pass, mirroring Fast-LeWM's "one action-prefix encoding + one parallel prediction" interface.

## How to rerun

```bash
cd runs/2026-06-27-icwm-fastlew-hallucination/fast-leworldmodel-2606.26217
python code/prefix_vs_ar_probe.py
```

Requirements: only NumPy (tested with NumPy 2.5.0). No GPU or PyTorch needed.

The script writes `code/probe_results.json` and prints per-horizon MSE and timing.

## Evidence preserved

- `code/prefix_vs_ar_probe.py` — runnable toy dynamics probe.
- `code/probe_results.json` — output of the probe on the default seed.
- `code/leworldmodel-baseline/` is a clone of the upstream LeWM repo (kept in the workspace `code/` bench; not duplicated in the thread, but the README references the exact file/line).

## What the probe shows

On the default run (horizon 20, 6-D latent state, 3-D action, 8k train trajectories):

- **Error accumulation:** The autoregressive chain's terminal MSE is ~2.2× higher than the prefix predictor's terminal MSE. The gap is small at `k=2` but grows with horizon, matching the paper's Figure 3 pattern.
- **Model-call efficiency:** The autoregressive rollout makes `H=20` model calls; the prefix interface makes ~2 calls (one prefix encoding + one parallel prediction), a 10× algorithmic speedup. Raw NumPy timing is dominated by the single large stacked matrix in this toy implementation and is not representative of GPU-based action encoding; the probe therefore reports both raw CPU time and a model-call-equivalent cost.
- **Limitation:** The probe uses linear models on a synthetic nonlinear system, so the absolute numbers are not comparable to the paper's visual-planning results. The qualitative behavior — slower error growth and fewer model calls for prefix prediction — is the preserved signal.

## Code/artifact status

- **Fast-LeWM official code:** cloned but empty / placeholder. No model, training, or planning code to inspect.
- **LeWM baseline code:** cloned and inspected; the autoregressive rollout loop confirms the target Fast-LeWM is replacing.
- **Probe:** runs end-to-end with only NumPy.

## Suggested report angle / takeaway

Fast-LeWM's contribution is a clean interface change: make action prefixes the prediction unit instead of single-step transitions. This turns rollout into a parallel feed-forward pass, cuts the number of dynamics-model invocations during CEM, and removes the sequential error-accumulation chain. The empirical gains are large on LeWM's own benchmark, but the official implementation is not yet public. The toy probe at least demonstrates that the interface change genuinely reduces error growth and model-call count in a controlled setting. A report should lead with the interface insight, report the speed/accuracy numbers, and note the code-release blocker clearly.
