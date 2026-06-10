# QGF: Test-Time Gradient Guidance of Flow Policies in RL (arXiv 2606.11087)

**Investigation thread:** `runs/2026-06-10-oasis-ahawam-tbdvla/qgf-2606.11087/`  
**Paper:** Zhou et al., *Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning*  
**Repo:** https://github.com/zhouzypaul/qgf (cloned to `code/qgf`)

---

## What QGF Does

QGF (Q-Guided Flow) is a **test-time RL algorithm** for flow-matching policies. The key idea is to keep the actor training purely supervised (standard behavioral-cloning flow matching) and perform all policy optimization at inference time using critic gradients.

### The method in four lines

1. **Train a reference flow policy** $\hat\pi$ via BC with the flow-matching loss (Eq. 2).
2. **Train a critic** $Q(s,a)$ with IQL — purely in-sample, no policy interaction needed.
3. **At test time,** for each denoising step $t$, approximate the clean action with one large Euler step:
   $$\hat a_1 = a_t + (1-t)\,v_\theta(s, a_t, t)$$
4. **Guide the velocity** with the critic gradient evaluated at that approximated clean action, dropping the Jacobian:
   $$a_{t+\delta} = a_t + \delta\Big(v_\theta(s, a_t, t) + \tfrac{1}{\beta}\nabla_{\hat a_1} Q(s, \hat a_1)\Big)$$

This is the **QGF estimator** (Eq. 9 in the paper). It avoids two well-known failure modes:

- **OOD gradient** ($\nabla_{a_t} Q(s, a_t)$): the critic has never seen noisy actions $a_t$ during training, so gradients there are unreliable and biased (Fig. 2).
- **BPTT gradient** ($\nabla_{a_t} Q(s, \text{ODE}(a_t))$): backpropagating through the full denoising chain is expensive and high-variance (Fig. 3).

### Algorithm 1 (runnable pseudocode)

```python
# Inference time
a_0 ~ N(0, I)
for t in [0, dt, 2*dt, ..., 1-dt]:
    v_bc = policy(s, a_t, t)
    # 1) approximate clean action via single Euler step
    a_hat = clip(a_t + (1-t) * v_bc, -1, 1)
    # 2) critic gradient at the clean approximation (J ≈ I)
    g = grad(Q(s, ·))(a_hat)
    # 3) guided denoising step
    a_{t+dt} = a_t + dt * (v_bc + guidance_weight * g)
return a_1
```

The default config in the repo uses `denoise_steps=10`, `apply_jacobian=False`, and `denoised_action_approx="one_euler_step_approx"`.

---

## What Was Investigated

### 1. Code inspection (official repo)

Cloned the repo and inspected the core files:

| File | What it contains |
|------|-----------------|
| `agents/qgf.py` | Main agent. `policy_loss` = flow-matching BC. `critic_loss` = IQL Q-loss. `sample_actions` implements Algorithm 1 with the single-Euler-step approx and optional Jacobian. |
| `agents/edp.py` | EDP baseline (Kang et al. 2023). Uses the same Euler-step approx but bakes Q-maximization into the actor training loss (`-Q(s, a_eval) + bc_weight * BC_loss`). |
| `agents/bptt.py` | BPTT baseline. Backpropagates through the full denoising chain. |
| `utils/networks.py` | `ActorFlowField` = MLP that takes `[obs, noised_action, time_embedding]` and predicts velocity. `Value` = MLP or BroNet critic ensemble. |

**Key preserved snippets:**
- `code/qgf_agent_snippet.py` — the exact inference loop from `agents/qgf.py`
- `code/edp_agent_snippet.py` — how EDP differs at training time

### 2. 1D diagnostic probe

Built `code/qgf_1d_probe.py`, a self-contained simulation that reproduces the qualitative behavior of Figure 2:
- Base policy maps Gaussian noise to a tri-modal distribution at {-2, 0, 2}.
- $Q(s,a) = -\|a - a^*\|^2$ with $a^* = 2$.
- Compares BC (no guidance), OOD gradient, BPTT gradient, and QGF.

**Results:**
- **BC** stays near whichever mode the noise initialized closest to.
- **OOD** gets stuck at the middle mode (0) — the gradient at noisy actions is biased because the critic was never trained on $a_t$.
- **BPTT** mostly works but some trajectories settle at suboptimal modes; the finite-difference proxy for BPTT shows high variance.
- **QGF** cleanly shifts mass toward the optimal mode $a^* = 2$.

Artifacts: `qgf_1d_probe_histogram.png`, `qgf_1d_probe_trajectories.png`.

**To rerun:**
```bash
cd code
python -m venv ../.venv && source ../.venv/bin/activate
pip install matplotlib numpy
python qgf_1d_probe.py
```

### 3. Comparison with related methods

| Method | When does optimization happen? | Gradient estimator | Key limitation |
|--------|-------------------------------|-------------------|----------------|
| **QGF** (ours) | Test time | $\nabla_{\hat a_1} Q$ with single Euler approx, $J\approx I$ | Needs a well-trained base policy |
| **EDP** (Kang et al., NeurIPS 2023) | Train time | Same Euler approx, but gradient flows into actor weights | Must tune `bc_weight`; actor-critic instability |
| **QFQL** (Jang et al.) | Test time | OOD gradient $\nabla_{a_t} Q(s, a_t)$ | Biased; Fig. 2 & 5 show poor performance |
| **QAM** (Li & Levine, 2026) | Train time | Adjoint matching — avoids BPTT via SOC | Complex training; scales worse than QGF (Fig. 9) |
| **FQL** (Park et al., 2025) | Train time | Distills multi-step flow to one-step policy | Distillation cost; expressivity loss |
| **BPTT** (DQL-style) | Test time | Full backprop through denoising chain | High variance, unstable (Fig. 3, 14) |
| **BFN** | Test time | No gradients; sample $N$ actions and pick best | Needs orders of magnitude more FLOPs (Fig. 6) |

The paper’s empirical finding is that **test-time guidance can be as effective as training-time RL** while avoiding the hyperparameter tuning and instability of actor-critic training. QGF is the first gradient-based test-time method to reliably achieve this.

---

## Critical Evaluation

### Claims that hold up

1. **Test-time RL matches training-time RL (Fig. 5).**  
   QGF outperforms or ties EDP (the best training-time baseline) across 20 OGBench tasks with action chunking ($h=5$). The gap is especially clear against QFQL and BPTT. The sample size is large: 10 seeds per task, 100M-transition datasets (and 1B for the hardest tasks). Domains span scene, puzzle-4×4/4×5/4×6, cube-triple/quadruple/octuple.

2. **Compute efficiency vs. BFN (Fig. 6, 7).**  
   BFN with $N=4$ uses ~4× the FLOPs of QGF yet performs worse. QGF+BFN($N=4$) matches BFN($N=16$). This is a strong and practically important result: gradient guidance is a far cheaper form of test-time compute than rejection sampling in high-dimensional action spaces.

3. **QGF is a lower-variance gradient estimator (Fig. 3, 4).**  
   Cosine-similarity probe under Gaussian perturbations shows QGF is the most stable. Surprisingly, including the exact Jacobian (QGF-Jacobian) *increases* variance and hurts performance.

### Claims that need nuance

4. **Scaling with model size (Fig. 9).**  
   QGF scales from 800k → 3.2M params while QAM flatlines. But both methods **overfit at 12.7M params**. The paper frames this as "QGF suffers less," which is true, but the headline "scales with model size" only holds up to a point. The overfitting ceiling is a real limitation for very large policies.

5. **"Any well-trained base policy" assumption.**  
   Section K (Limitations) admits this explicitly, but it is easy to overlook. If the base BC policy is under-trained or fails to cover the data distribution, QGF has little to guide. This is a significant practical caveat: QGF is not a magic wand that fixes a bad policy.

6. **Comparison breadth.**  
   The main results use IQL critics. Fig. 10 shows QGF also works with QAM-style critics (with $Q$-bootstrapping), which is reassuring, but the bulk of the evidence is still under the IQL regime. It is not fully established how QGF behaves with other critic families (e.g., CQL, EDAC).

### What surprised us

- **The first-order Euler approximation outperforms full ODE integration.** The paper hypothesizes that full denoising constrains the output to the *exact* dataset distribution, while the single-step approximation allows the flow to "choose only certain modes." This is subtle and important: a cruder approx can be a better optimizer because it is less constrained.

- **Dropping the Jacobian is better than computing it.** QGF-Jacobian Regularized and QGF-Jacobian Ortho recover some performance, but the simplest estimator ($J=I$) wins. This echoes a broader ML theme (random feedback alignment, synthetic gradients) but is still counter-intuitive.

---

## Connection to the Broader Run Theme

This run covers four papers that all use **flow or diffusion policies** for robotics/RL:

- **OASIS** — online RL with diffusion policies
- **AHA-WAM** — action chunking + diffusion
- **TBD-VLA** — diffusion-style action generation in VLA models
- **QGF** — test-time guidance of flow policies

QGF sits at a conceptual intersection with all three: it leverages the same expressive generative policy class (flow matching), shares the action-chunking setup with AHA-WAM, and raises the question of whether test-time compute — rather than training-time RL — is the right way to optimize these policies. For VLA models (TBD-VLA), where retraining the policy is extremely expensive, QGF-style test-time guidance could be especially relevant.

---

## Evidence Preserved in This Thread

```
qgf-2606.11087/
├── README.md                          # this file
├── qgf_1d_probe_histogram.png         # 1D diagnostic: final action distributions
├── qgf_1d_probe_trajectories.png      # 1D diagnostic: denoising trajectories
├── code/
│   ├── qgf_1d_probe.py                # self-contained 1D simulation
│   ├── qgf_agent_snippet.py           # preserved inference loop from agents/qgf.py
│   └── edp_agent_snippet.py           # preserved training loop from agents/edp.py
```

The full repo is cloned at `code/qgf/` (workspace root).

---

## Key Findings for the Report

1. **QGF is a genuinely new and useful estimator.** The combination of (single Euler-step clean-action approx) + (critic grad at clean action) + (dropping Jacobian) is novel and empirically well-motivated. It outperforms prior test-time methods and competes with the best training-time methods.

2. **Test-time RL is competitive.** The paper makes a strong case that for flow/diffusion policies, expensive actor-critic training may not be necessary. Stable BC + IQL critic + test-time guidance is enough.

3. **But scaling is not unlimited.** Both QGF and QAM overfit at 12.7M parameters. The scaling story is positive up to a point, then hits the same offline-RL overfitting wall.

4. **Base-policy quality matters.** QGF is not robust to a weak base policy. This is an important practical caveat for deployment.

5. **Uncertainty:** We have not independently reproduced the OGBench results (no GPU environment configured). The findings above are based on careful code inspection and the 1D probe. A full reproduction would require the OGBench 100M/1B datasets and substantial compute.

---

*Investigated: 2026-06-10*  
*Agent: Paper Scout research subagent*
