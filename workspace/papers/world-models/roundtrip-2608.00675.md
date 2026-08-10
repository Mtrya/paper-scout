# Round-Trip Consistency: Bidirectional Difusion Models Can Predict Their Own Rollout Errors

Alexander Scheinker

Los Alamos National Laboratory Los Alamos, NM, 87544, USA ascheink@elanlg.ov

## Abstract

Autoregressive models accumulate error over long rollouts, yet at deployment there is no ground truth to measure it against. We train a single conditional latent difusion model that steps a dynamical system forward or backward in time via a direction flag, and show that this bidirectionality supplies a measurement-free test-time error signal: rolling forward i steps and then backward i steps must return the model to its start, so the round-trip discrepancy C<sub>i</sub> is a self-supervised proxy for the unobservable rollout error: no ensembles, no held-out data, no governing equations, for one extra rollout. We validate on compressible magnetohydrodynamics (MHD), an astrophysical turbulent radiative mixing layer, and natural face videos (CelebV-HQ). On held-out MHD trajectories, $\mathcal { C } _ { i }$ ranks rollout error (Spearman 0.91-0.98 at fixed depth; 0.69 ± 0.16 within trajectories), and a simple calibrator fit on training rollouts predicts its magnitude to within 1.14× (68%) and 1.29× (95%) with near-nominal coverage - one nat beyond a depth-only predictor, transferring to all six decoded physical fields. The same signal flags the out-of-distribution Orszag-Tang vortex (AUROC 0.98; 1.0 by depth 10) exactly where sampling-dispersion baselines invert, and it cuts incurred error by 15% at 80% coverage - three times the depthonly baseline. Bidirectional training comes at negative cost, beating direction specialists in both directions, and the backward direction doubles as a fast inverse solver. On LE-PDE-UQ’s turbulent Navier-Stokes benchmark, a single bidirectional model reaches accuracy within 1.3× of their ten-model ensemble at a tenth of the training cost, with the best trainingfree pixel-level calibration. Round-trip consistency turns reversibility into a practical trust signal for generative models.

## 1 Introduction

Learned surrogates increasingly stand in for numerical solvers across scientific computing (Sanchez-Gonzalez et al. 2020; Brandstetter, Worrall, and Welling 2022; Li et al. 2021; Lu et al. 2021; Um et al. 2020; Kochkov et al. 2021; Raissi, Perdikaris, and Karniadakis 2019; Karniadakis et al. 2021), and difusion models now act as probabilistic simulators, from turbulence to operational weather prediction (Kohl, Um, and Thuerey 2023; Cachay et al. 2023; Price et al. 2025; Lam et al. 2023). Deployed autoregressively, these models feed their own predictions back as inputs: small onestep errors compound, the input distribution drifts from the training data (Brandstetter, Worrall, and Welling 2022; Lippe et al. 2023), and accuracy degrades at a rate that varies unpredictably across initial conditions. In deployment: plasma control (Degrave et al. 2022), accelerator diagnostics (Scheinker 2024; Scheinker and Williams 2025; Scheinker, Plastun, and Ostroumov 2026), and weather prediction (Price et al. 2025), the true state is unknown at test time, so a model cannot say howfar into thefuture it can still be trusted.

![](images/ac03047babb5532cbef3fd2b5f8ef33b4e9dccda00090d7ecc1b05cf0ba3e660.jpg)  
Figure 1: Consistency-based, test-time self-supervised error estimation with a bidirectional difusion model: a directional flag $c _ { d } = \pm 1$ selects forward or backward rollout; reversing it rolls back to an estimate $\hat { \bf z } _ { t }$ of the starting point.

Standard remedies quantify predictive uncertainty rather than realized error: deep ensembles (Lakshminarayanan, Pritzel, and Blundell 2017; Price et al. 2025), MC dropout (Gal and Ghahramani 2016), learned variance (AI4Science-WestlakeU 2024), spread of repeated samples (Lippe et al. 2023; Shu and Barati Farimani 2024), and conformal intervals (Angelopoulos and Bates 2023) all measure how much the model disagrees with itself, not whether the learned dynamics are being applied accurately to the trajectory at hand. Dispersion-based signals degrade under distribution shift that autoregressive rollouts induce (Ovadia et al. 2019).

We take a diferent route, built on an old and simple idea: check the consistency of a round trip (Fig. 1). Forward– backward agreement flags unreliable optical-flow correspondences (Sundaram, Brox, and Keutzer 2010; Meister, Hur, and Roth 2018), cycle losses supervise unpaired translation and temporal correspondence (Zhu et al. 2017; Wang, Jabri, and Efros 2019), and cycle defects quantify uncertainty in inverse imaging when an analytic forward model is available (Huang et al. 2023). We port this principle to learned dynamics by training a single conditional latent difusion model with a direction flag $c _ { d } \in \{ + 1 , - 1 \}$ , so that one network represents both a system’s forward evolution and its temporal inverse; the denoiser conditions on the two most recent frames — the least context that determines a velocity (the check applies verbatim for any $n \geq 1 )$ . Reversibility then becomes a checkable invariant: an accurate model composed with its own inverse is the identity, so the discrepancy left after rolling forward i steps and backward i steps, the round-trip consistency error $\mathcal { C } _ { i } ,$ is measurable at test time with no ground truth, no ensemble, and no access to the governing equations, for the price of one extra rollout. The check is necessary rather than suficient because forward and backward errors could in principle cancel, which is precisely why quantifying how faithfully $\mathcal { C } _ { i }$ tracks the true error is the central contribution of this paper.

We validate on three dissipative, nonlinear systems: highresolution compressible magnetohydrodynamics (MHD), with the canonical Orszag–Tang vortex (Orszag and Tang 1979) as a deliberate out-of-distribution stress test; an astrophysical turbulent radiative mixing layer from the Well (Ohana, McCabe et al. 2024); and natural face videos. Our aim is explicitly not state-of-the-art rollout accuracy. Our context length, architecture scale, and sampling schedules are held fixed and modest throughout, but the question such tuning never answers: whether a model can know, at test time and without ground truth, how wrong its rollout is. A trust signal matters most precisely when the surrogate is imperfect, which every deployed surrogate is.

We find that $\mathcal { C } _ { i }$ tracks the true rollout error across depths and trajectories: a simple calibrator fit on training rollouts predicts its magnitude on held-out MHD trajectories to within a factor of 1.14 (68%): one nat better than a depthonly predictor, with the gain transferring to all six decoded physical fields — supports selective prediction (15% lower incurred error at 80% coverage, three times the depth-onlydeferral baseline (Geifman and El-Yaniv 2017)), and flags the out-of-distribution Orszag-Tang vortex immediately (AU-ROC 1.0 by depth 10; 0.98 for the trajectory-mean score). Our contributions are:

• Bidirectional difusion dynamics model. A single conditional latent difusion with direction flag $c _ { d } \in$ $\{ + 1 , - 1 \}$ rolls a system forward (surrogate solver) or backward (inverse solver) in time with one set of weights.

• Round-trip consistency as self-supervised UQ. A testtime error proxy $\mathcal { C } _ { i }$ , computed by cycling a rollout forward and back, requiring no ground truth, no ensembles, and no architectural changes: one extra rollout yields a trust / early-stop criterion for autoregressive generation.

• Quantitative validation and theory. Correlation with true error across depths and systems, out-of-distribution detection, selective prediction, comparisons against sampling-based UQ baselines, and a Lipschitz-style certificate relating $\mathcal { C } _ { i }$ to $\mathcal { E } _ { i }$ (Sec. 5.1).

## 2 Related Work

Surrogates and difusion models for dynamics. Learned simulators range from graph networks and message-passing solvers to neural operators and solver-in-the-loop hybrids (Sanchez-Gonzalez et al. 2020; Brandstetter, Worrall, and Welling 2022; Li et al. 2021; Lu et al. 2021; Um et al. 2020; Kochkov et al. 2021); difusion models entered as probabilistic forecasters, from turbulence benchmarking and forecasting frameworks (Cachay et al. 2023; Kohl, Um, and Thuerey 2023) to operational ensemble weather prediction (Price et al. 2025), with difusion-style refinement stabilizing long rollouts (Lippe et al. 2023). Latent generation with transformer denoisers (Rombach et al. 2022; Peebles and Xie 2023a) makes high-resolution, multi-field states tractable; we operate in this latent autoregressive regime, by 2026 an established class (Shysheya et al. 2024; Li, Zhou, and Barati Farimani 2025; Chen et al. 2024), and add bidirectionality as the enabling structure. Although our multi-field MHD surrogate is, to our knowledge, among the first for that system, we deliberately do not rest the paper’s claims on it: the bidirectional model is infrastructure; the contribution is what its structure makes measurable.

Uncertainty quantification for rollouts. Ensembles, MC dropout, and conformal calibration are the workhorses of deep UQ (Lakshminarayanan, Pritzel, and Blundell 2017; Gal and Ghahramani 2016; Angelopoulos and Bates 2023), and their reliability erodes under exactly the distribution shift that autoregressive rollouts induce (Ovadia et al. 2019). For PDE surrogates, LE-PDE-UQ (AI4Science-WestlakeU 2024) evolves latent uncertainty alongside the state and reports gains over ensembles, Bayesian layers, and dropout, making it the strongest published baseline for our setting (evaluated head-to-head on its own benchmark in Sec. 6.6); sample spread supplies zero-shot uncertainty (Shu and Barati Farimani 2024; Lippe et al. 2023). All of these measure the model’s dispersion, that is how much it disagrees with itself, rather than whether the learned dynamics are applied accurately to the trajectory at hand; we compare against dispersion signals for error detection and selective prediction (Geifman and El-Yaniv 2017; Hendrycks and Gimpel 2017). Relatedly, difusion posterior sampling, score-based data assimilation, and DifusionPDE (Chung et al. 2023; Rozet and Louppe 2023; Huang et al. 2024) solve inverse and stateestimation problems by guiding a prior with test-time observations; our backward direction is instead learned, and the error signal we study requires no observations at all.

Cycle consistency and nearest neighbors. Forward– backward agreement has long flagged unreliable optical-flow correspondences (Sundaram, Brox, and Keutzer 2010; Meister, Hur, and Roth 2018), and cycle losses supervise unpaired translation and temporal correspondence (Zhu et al. 2017; Wang, Jabri, and Efros 2019). Huh et al. (2020) impose time-reversal symmetry as a training regularizer, Arık, Yoder, and Pfister (2022) attach an auxiliary backcasting head, and Chakraborty et al. (2022) reverse predicted trajectories as a training-time loss; our signal needs no auxiliary head, applies at test time, and directly checks the generative dynamics. (Consistency models (Song et al. 2023) are a difusion distillation technique, unrelated to the cycle consistency studied here.) The direct precedent is Huang et al. (2023), who quantify uncertainty in inverse imaging by cycling a known forward operator with a learned inverse over a single, static problem, with supporting theory and an OOD application. Two diferences define our setting: no analytic model exists in either direction. Both legs of the cycle are the same learned bidirectional generator, and the cycle closes over a 2i-step autoregressive rollout in which both legs accumulate error with depth, which is why our central claim is quantitative: that $\mathcal { C } _ { i }$ tracks the depth-dependent $\mathcal { E } _ { i }$ (Sec. 6.1). Among rollout-time signals, DifusionRollout (Yoo et al. 2026) uses the dispersion of repeated samples to schedule step sizes, whereas $\mathcal { C } _ { i }$ is a functional check computable from a single deterministic cycle (compared head-to-head in Table 2). Table 1 situates these signals; no existing method converts its trust signal into a correction of the prediction itself, and such adaptation is deliberately out of scope here (Sec. 7), keeping the detection claims cleanly falsifiable.

Table 1: Test-time trust signals for learned predictors. “Both dir. learned” asks whether forward and backward maps are both neural (no analytic operator); “comp. rollout” whether the signal is defined over a multi-step autoregressive horizon; “test-time data” whether external observations are required at deployment; the last column states what the method does with its signal.
<table><tr><td>Method</td><td>Signal</td><td>Both dir. learned</td><td>Comp. rollout</td><td>Test-time data</td><td>Uses signal for</td></tr><tr><td>Cycle-consistency UQ (Huang et al. 2023)</td><td>cycle defect</td><td>a</td><td></td><td></td><td>error/OOD flags</td></tr><tr><td>PDE-Refiner (Lippe et al. 2023)</td><td>sample spread</td><td></td><td></td><td></td><td>implicit confidence</td></tr><tr><td>LE-PDE-UQ (AI4Science-WestlakeU 2024)</td><td>learned variance</td><td></td><td></td><td></td><td>error bars</td></tr><tr><td>DiffusionRollout (Yoo et al. 2026)</td><td>sample spread</td><td></td><td></td><td></td><td>step-size scheduling</td></tr><tr><td>DiffusionPDE (Huang et al. 2024)</td><td>obs. guidance</td><td>b</td><td></td><td> $\checkmark$ </td><td>constraining sampling</td></tr><tr><td>Cycle loss (Chakraborty et al. 2022)</td><td>training loss</td><td>√</td><td></td><td>n/a</td><td>training regularizer</td></tr><tr><td>Round-trip Ci (ours)</td><td>cycle defect</td><td> $\checkmark$ </td><td> $\checkmark$ </td><td></td><td>flags, selective prediction</td></tr></table>

<sup>a</sup> Forward leg is a known physical operator; only the inverse is learned. <sup>b</sup> One difusion prior guided in both directions by observations.

Generative models as scientific diagnostics. Conditional generative models increasingly serve as virtual diagnostics for large experimental facilities, e.g., reconstructing charged-particle beam phase space from sparse measurements (Scheinker 2024; Scheinker, Plastun, and Ostroumov 2026). The bidirectional MHD application is developed in depth in a preliminary, application-focused preprint by the authors Scheinker (2026); the present paper isolates the round-trip consistency metric, quantifies its fidelity as an error estimator, and evaluates it beyond a single system.

## 3 Background

Problem setup. We observe trajectories $\mathbf { x } _ { 0 } , \ldots , \mathbf { x } _ { T }$ of a dynamical system, with each state $\mathbf { x } _ { t } \in \mathbb { R } ^ { d \times d \times c }$ a multichannel field snapshot (for MHD: density, pressure, velocity, and magnetic-field components). A surrogate learns the transition map and is deployed autoregressively: seeded with observed states, it feeds its own predictions back as inputs to reach depth i. The true rollout error at depth i, $\mathbf { \bar { \mathcal { E } } } _ { i } = \mathrm { M S E } ( \mathbf { z } _ { t + i } , \mathbf { \bar { z } } _ { t + i } )$ (in latent space; decoded-field metrics in the supplementary), is what is needed and cannot be computed at deployment, because $\mathbf { z } _ { t + i }$ is unknown. Our goal is a test-time estimate of $\mathcal { E } _ { i }$ using only the model itself.

Latent encoding and difusion. Following latent generative modeling (Rombach et al. 2022), each physical field is encoded independently by a β-VAE (Kingma and Welling 2014; Higgins et al. 2017) with a weak KL penalty $( \beta = 1 0 ^ { - 3 } )$ , mapping a $5 1 2 \times 5 1 2$ field to a $1 6 \times 1 6 \times 4$ latent (256× compression) that stays compact without collapsing detail; dynamics are learned in this latent space. On the latents we use standard denoising difusion (Ho, Jain, and Abbeel 2020; Song et al. 2021): a network $\epsilon _ { \theta }$ is trained to predict the noise injected by a fixed variance schedule, which defines the reverse-time sampler (Anderson 1982), with conditioning information entering $\epsilon _ { \theta }$ alongside the noised latent. Our contribution lies in what is conditioned on and how the resulting bidirectionality is exploited.

## 4 Bidirectional Latent Difusion Dynamics

A single denoiser $\epsilon _ { \theta }$ learns the bidirectional transition density

$$
\hat { \mathbf { z } } _ { t + c _ { d } } \sim p _ { \theta } \big ( \mathbf { z } _ { t + c _ { d } } \mid \mathbf { z } _ { t } , \mathbf { z } _ { t - c _ { d } } , c _ { d } \big ) , \quad c _ { d } \in \{ + 1 , - 1 \} ,\tag{1}
$$

trained with the conditional noise-prediction loss

$$
\mathcal { L } ( \boldsymbol { \theta } ) = \mathbb { E } _ { t , c _ { d } , k , \epsilon } \big | \big | \epsilon - \epsilon _ { \boldsymbol { \theta } } \big ( \mathbf { z } _ { t + c _ { d } } ^ { ( k ) } , k , \mathbf { z } _ { t } , \mathbf { z } _ { t - c _ { d } } , c _ { d } \big ) \big | \big | _ { 2 } ^ { 2 } ,\tag{2}
$$

with physical time t and direction $c _ { d }$ drawn uniformly over training trajectories and k the denoising step, $\mathbf { z } ^ { ( k ) }$ denoting the correspondingly noised target. Drawing both signs of $c _ { d }$ forces the shared weights to denoise consistently toward the future $( c _ { d } = + 1$ , surrogate solver) and the past $( c _ { d } = - 1$ inverse solver). Rollouts compose (1) autoregressively; decoded fields are $\hat { \mathbf { x } } _ { t } = \mathcal { D } _ { \theta } ( \hat { \mathbf { z } } _ { t } )$ The denoiser is a difusion transformer (Peebles and Xie 2023a) throughout $\scriptstyle ( n = 2$ context frames unless noted): target, context, and anchor frames are patchified into one token sequence, so conditioning on past states acts through attention, while the scalar conditions — difusion step, simulation-time index, and the direction flag c<sub>d</sub> — modulate every block via adaLN-Zero (full architecture in Supp. Sec. A).

## 5 Round-Trip Consistency

Let $\Phi _ { + } ^ { i }$ denote an i-step forward rollout seeded by the true pair $\left( \mathbf { z } _ { t - 1 } , \mathbf { z } _ { t } \right)$ and $\Phi _ { - } ^ { i }$ the i-step backward rollout seeded

by the terminal predicted pair. An error-free model satisfies $\dot { \Phi _ { - } } ^ { i } \circ \Phi _ { + } ^ { i } = \mathrm { I d }$ . Round-trip consistency error at depth i is

$$
\begin{array} { r } { \mathcal { C } _ { i } = \frac { 1 } { 2 } \Big [ \mathrm { M S E } \big ( \mathbf { z } _ { t - 1 } , \tilde { \mathbf { z } } _ { t - 1 } ^ { ( i ) } \big ) + \mathrm { M S E } \big ( \mathbf { z } _ { t } , \tilde { \mathbf { z } } _ { t } ^ { ( i ) } \big ) \Big ] , } \end{array}\tag{3}
$$

where $\tilde { \mathbf { z } } ^ { ( i ) }$ are the returned seeds. Every quantity in (3) is available at test time — the anchor pair is encoded measured data, the returned pair is produced entirely by the model — in contrast to $\mathcal { E } _ { i } .$ . Given a tolerance $\tau ,$ a rollout is trusted to the largest depth with $\mathcal { C } _ { i } \ \leq \ \tau - \ \mathrm { a n }$ early-stop criterion for autoregressive generation.

Properties and cost. $\mathcal { C } _ { i }$ costs one backward rollout per checked depth — a 2× inference overhead, with no change to training. We use deterministic DDIM sampling throughout, which makes the cycle well defined; averaging S stochastic cycles is a variant whose spread we compare as a dispersion baseline (Table 2). The check is necessary, not suficient: Sec. 5.1 bounds when small $\mathcal { C } _ { i }$ certifies small $\mathcal { E } _ { i }$ , and the experiments measure how often cancellation bites in practice.

## 5.1 When Does the Proxy Certify the Error?

Because the model is second-order Markov, the natural state is the pair $\mathbf { s } _ { k } : = ( \mathbf { z } _ { t + k - 1 } , \mathbf { z } _ { t + k } ) \in \mathbb { R } ^ { 2 n }$ , with the normalized norm $\begin{array} { r } { \| \mathbf { s } \| : = \left( \frac { 1 } { 2 n } \| \mathbf { s } \| _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } } \end{array}$ chosen so that $\mathcal { C } _ { i } = \| \mathbf { s } _ { 0 } - \tilde { \mathbf { s } } _ { 0 } \| ^ { 2 }$ matches Eq. (3) exactly. With a deterministic sampler one model step is a map on pairs: we write $\Phi _ { + } , \Phi _ { - } : \mathbb { R } ^ { 2 \bar { n } } \to \mathbb { R } ^ { 2 n }$ for the forward and backward steps, so the forward rollout is $\hat { \mathbf { s } } _ { i } = \Phi _ { + } ^ { i } ( \mathbf { s } _ { 0 } )$ , the returned seed is $\tilde { \mathbf { s } } _ { 0 } = \Phi _ { - } ^ { i } ( \hat { \mathbf { s } } _ { i } )$ , and the pair-level rollout error is $\mathcal { E } _ { i } ^ { \mathrm { p } } : = \left\| \mathbf { s } _ { i } - \hat { \mathbf { s } } _ { i } \right\| ^ { 2 }$ , which dominates the terminal-state error via $\mathcal { E } _ { i } \leq 2 \mathcal { E } _ { i } ^ { \mathrm { p } }$

Assumption 1. Let $\mathcal { D } \subset \mathbb { R } ^ { 2 n }$ contain the two backward trajectories $\{ \Phi _ { - } ^ { k } ( \mathbf { s } _ { i } ) \} _ { k = 0 } ^ { i }$ and $\{ \Phi _ { - } ^ { k } ( \hat { \mathbf { s } } _ { i } ) \} _ { k = 0 } ^ { i }$ . On D, ∀a, b, and some $0 < \mu \le L < \infty$ , the backward step satisfies

$$
\mu \| a - b \| \ \leq \ \| \Phi _ { - } ( a ) - \Phi _ { - } ( b ) \| \ \leq \ L \ \| a - b \| | .\tag{4}
$$

Define the backward residual on true data, $\begin{array} { r l } { \delta _ { i } } & { { } : = } \end{array}$ $\left\| \dot { \Phi _ { - } } ( \mathbf { s } _ { i } ) - \mathbf { s } _ { 0 } \right\|$ , i.e., the error of the backward model when seeded with the true terminal pair.

The upper inequality in (4) is ordinary Lipschitz continuity; the lower one, co-Lipschitz continuity, is precisely the anti-cancellation condition: it forbids the backward map from collapsing distinct terminal states onto the same returned seed (unpacked in Supp. Sec. C).

Proposition 1. Under Assumption 1,

$$
\Big ( \operatorname* { m a x } \big \{ \mu ^ { i } \sqrt { \mathcal { E } _ { i } ^ { \mathrm { p } } } - \delta _ { i } , 0 \big \} \Big ) ^ { 2 } \leq \mathcal { C } _ { i } \leq \Big ( L ^ { i } \sqrt { \mathcal { E } _ { i } ^ { \mathrm { p } } } + \delta _ { i } \Big ) ^ { 2 } .\tag{5}
$$

Corollary 1. Under Assumption 1,

$$
\sqrt { \mathscr { E } _ { i } } \ \leq \ \sqrt { 2 } \ \mu ^ { - i } \Big ( \sqrt { \mathscr { C } _ { i } } + \delta _ { i } \Big ) .\tag{6}
$$

Smal $\textstyle { \mathcal { C } } _ { i }$ certifies small true error when the learned backward map is well conditioned (µ not too small) and accurate on clean data $( \delta _ { i }$ small) – both properties of the model alone, estimable ofline without test-time ground truth.

The proof combines (4) with the triangle inequality around $\Phi _ { - } ^ { i } ( \mathbf { s } _ { i } ) ^ { \phantom { i } }$ ; the full derivation is in Supp. Sec. C. Both constants are measurable ofline: the residual $\delta _ { i }$ is the detector’s noise floor, estimated on validation data (informative when $\mathcal { C } _ { i } \gg \delta _ { i } ^ { 2 } )$ , and the factor $\mu ^ { - i }$ grows geometrically, so the certificate is meaningful at moderate turnaround depths and for well-conditioned backward maps: as $\mu \to 0$ the bound loosens as a collapsing inverse can hide errors. Condition (4) constrains the learned backward map on rollout-visited pairs only; no invertibility of the underlying physical dynamics is assumed. Because (5) pins $\sqrt { \mathscr { C } _ { i } }$ between two afine functions of $\sqrt { \mathcal { E } _ { i } ^ { \mathrm { p } } }$ , it justifies fitting monotone or heteroscedastic Gaussian calibrators of the observed error on $\mathcal { C } _ { i }$ (Nix and Weigend 1994; Kuleshov, Fenner, and Ermon 2018), in the spirit of Huang et al. (2023); Sec. 6.2 fits exactly such a calibrator. Estimation of $( \mu , L )$ along rollouts, certified bi-Lipschitz architectures, and stochastic sampling are discussed in Supp. Sec. C.

## 6 Experiments

Data. Synthetic 2D compressible MHD trajectories with randomized initial density, pressure, and velocity, and magnetic fields derived from a random vector potential, simulated with a constrained-transport scheme (Mocz, Vogelsberger, and Hernquist 2014) at $5 1 2 \times 5 1 2 ;$ 100 steps over 1 s; 500 training and 50 held-out test trajectories (generation details in Supp. Sec. A). Out-of-distribution evaluation uses the Orszag–Tang vortex (Orszag and Tang 1979) with the model trained only on the synthetic distribution. Baselines. Dispersion-based UQ signals computed from the same trained model: S-sample rollout spread (Lippe et al. 2023; Shu and Barati Farimani 2024). MC-dropout (Gal and Ghahramani 2016) and deep ensembles (Lakshminarayanan, Pritzel, and Blundell 2017) are inapplicable to our single trained, dropout-free model without retraining (Table 2). LE-PDE-UQ (AI4Science-WestlakeU 2024) is evaluated headto-head on its own Navier–Stokes benchmark in Sec. 6.6. Metrics. Per-trajectory and per-depth Spearman/Pearson correlation between $\mathcal { C } _ { i }$ and $\mathcal { E } _ { i } ; \mathrm { A U R O C }$ for flagging high-error and OOD rollouts; risk–coverage curves for selective prediction and calibration of the τ rule; rollout MSE per field vs. depth for fidelity.

## 6.1 Does the Proxy Track the Truth?

The leftmost panel of Fig. 2 shows the raw relationship on the held-out MHD trajectories; Sec. 6.2 quantifies it as a calibrated predictor. Table 2 compares C against the natural sampling-dispersion alternative, and the two turn out to be complementary. In-distribution, the S=5 spread is a slightly better magnitude predictor (−0.99 vs. −0.62 nats): dispersion measures the aleatoric width of the realized noise draw, at 5× inference cost and only under stochastic sampling. Out of distribution the two signals invert: on the Orszag–Tang vortex the model is confidently wrong, at depth 5 its dispersion is lower than every in-distribution trajectory’s (AUROC 0.00), ranking the OOD case as the safest in the batch, but C flags it above all fifty (AUROC 1.00). Conditional width has no reason to grow under distribution shift; realized roundtrip drift must. C thus ofers near-parity in-distribution at 2× cost, compatibility with deterministic deployment, and the failure sensitivity a trust signal exists to provide; Sec. 6.6 composes the two.

Table 2: Test-time signals vs. true rollout error on held-out MHD trajectories. ρ<sub>20</sub>: Spearman across trajectories at fixed depth i=20; NLL $\ I \times _ { 6 8 } \colon$ : identically-fit calibrator on the heldout set; OOD: AUROC separating the Orszag–Tang vortex (single canonical trajectory; percentile rank) from the 50 held-out trajectories; cost: inference overhead.
<table><tr><td>Signal</td><td> $\rho _ { 2 0 } \uparrow$ </td><td>NLL↓</td><td> $\times _ { 6 8 }$ </td><td>OOD AUROC ↑</td><td>Cost</td></tr><tr><td>Round-trip C (ours)</td><td>0.97</td><td>-0.62</td><td>1.14</td><td> $\mathbf { 1 . 0 0 } \ ( i \leq 1 0 )$ </td><td>2×</td></tr><tr><td>S=5 rollout spread</td><td>0.98</td><td>-0.991.09</td><td></td><td> $0 . 0 0 \left( i { = } 5 \right) { - } 0 . 5 0$ </td><td>5×</td></tr><tr><td>MC-dropout variance</td><td></td><td>n/a — no dropout layers</td><td></td><td></td><td></td></tr><tr><td>Deep ensemble</td><td></td><td>n/a — single trained model</td><td></td><td></td><td>3× train</td></tr></table>

## 6.2 Calibrated Error Prediction

Rank correlation shows that $\mathcal { C } _ { i }$ orders errors correctly; a deployed trust signal should ${ \bf g 0 }$ further and predict the error’s magnitude. We therefore fit a heteroscedastic Gaussian calibrator (Nix and Weigend 1994), modeling log $\mathcal { E } _ { i } \sim$ $\mathcal { N } ( \mu ( \log \mathcal { C } _ { i } ) , \sigma ^ { 2 } ( \log \mathcal { C } _ { i } ) )$  with polynomial $\mu$ and log σ on the training-trajectory rollouts; degrees (4, 2) are selected by trajectory-level 5-fold cross-validation with the onestandard-error rule (further safeguards in Supp. Sec. E), and the 50 held-out trajectories are touched exactly once, for the evaluation reported here (4,900 depth–trajectory pairs). Two baselines are fit with the identical recipe: a depth-only calibrator that predicts $\mathcal { E } _ { i }$ from the depth i alone, the critical control, since any signal that merely grows with depth matches it and a global constant.

Table 3 summarizes held-out performance. The $\mathcal { C } _ { i }$ -based calibrator predicts the unobservable error to within a factor of 1.14 (68%) and 1.29 (95%) of held-out points with near-nominal coverage and a miscalibration area of 0.013 (Kuleshov, Fenner, and Ermon 2018), at roughly half the interval width of the depth-only baseline (mean log-space σ of 0.13 vs. 0.26). The headline is the proper-score margin: it improves held-out log-likelihood by 1.0 nat over the depth-only calibrator, and the advantage persists at every probed depth (0.6–1.3 nats at $i \in \{ 5 , \bar { 1 0 } , \bar { 2 0 } , 4 0 , 8 0 \} \rangle$ ), establishing that the self-supervised signal carries per-trajectory information beyond the shared growth of error with depth. Consistently, $\mathcal { C } _ { i }$ and $\mathcal { E } _ { i }$ are rank-correlated within trajectories (Spearman $0 . 6 9 \pm 0$ .16 across held-out trajectories; $0 . 6 4 \pm 0 . 1 4$ on training rollouts) and across trajectories at fixed depth (0.91–0.98 $\mathrm { a t } i \in \{ 5 , 1 0 , 2 0 , 4 0 , 8 0 \} ; 0 . 9 7 \mathrm { a t } i = 2 0$ , Table 2, at or slightly above the reliability ceiling derived in Supp. Sec. B; scatter shown in Supp. Fig. 1).

Transfer to decoded field-space errors. The quantity an operator ultimately cares about is the decoded physicalfield error, which interposes decoder reconstruction noise between $\mathcal { C } _ { i }$ and the target. We therefore fit one calibrator per field with the same recipe, taking log decoded per-field error as the target: the gain over the identically-fit depthonly baseline is positive for all six fields (+0.32 to +0.73 nats, mean +0.59), with the true field error predicted to within a factor of 1.17–1.30 (68%; per-field breakdown and discussion in Supp. Sec. E, Table 1 there). The latent, selfsupervised signal thus remains a quantitative predictor all the way down to the decoded physical quantities.

Table 3: Calibrated error prediction on held-out MHD trajectories (50 trajectories $\times \ 9 8$ depths); all calibrators fit on training rollouts only, with the identical heteroscedastic recipe. NLL: held-out Gaussian negative log-likelihood (nats); $\mathrm { \bar { R M S E } _ { l o g } \mathrm { : } }$ RMSE of $\mu$ in log space; $\times _ { 6 8 } / { \times } _ { 9 5 } { \vdots }$ multiplicative factor containing the true error for 68%/95% of points; Cov: empirical $\pm 1 \sigma / \pm 2 \sigma$ coverage (nominal 68.3/95.4); MCA: miscalibration area.
<table><tr><td>Calibrator</td><td>NLL↓</td><td> $\mathrm { R M S E } _ { \log } \downarrow$ </td><td>×68</td><td>×95</td><td> $\mathrm { C o v } _ { 6 8 / 9 5 } ( \% )$ </td><td>MCA↓</td></tr><tr><td>Global constant</td><td>1.44</td><td> $1 . 0 2$ </td><td>1.94</td><td>9.83</td><td> $8 7 . 5 / 9 3 . 7 $ </td><td>0.095</td></tr><tr><td>Depth-only</td><td>0.38</td><td>0.34</td><td>1.41</td><td>1.92</td><td> $5 1 . 8 \dot { / } 8 4 . 3$ </td><td>0.059</td></tr><tr><td>µ(Ci) (ours)</td><td>-0.62</td><td>0.13</td><td>1.14</td><td>1.29</td><td>66.3/95.1</td><td>0.013</td></tr></table>

One limitation is visible and expected: while pooled calibration is near-nominal, depth-conditional calibration drifts at the deepest rollouts (per-depth miscalibration area up to $0 . 2 7 \mathrm { a t } i \geq 4 0 ,$ vs. 0.013 pooled), this is the regime where Corollary $1 \mathrm { { } ^ { \circ } s } \ \mu ^ { - i }$ factor predicts the signal loosens. A trajectory-mean augmentation of the calibrator recovers +0.26 nats and largely repairs the mid-depth drift (full analysis in Supp. Sec. E).

## 6.3 Selective Prediction and OOD Detection

Because $\mu ( \mathcal { C } _ { i } )$ is a calibrated prediction of the error (Sec. 6.2), it serves directly as a deferral score (Geifman and El-Yaniv 2017). Risk–coverage on the held-out trajectories (Supp. Fig. 6): deferring the 20% of predictions with the highest predicted error reduces the mean incurred latent error by 15% (at 90%/70% coverage: 7%/23%), where the critical control, deferring by depth alone (early stopping), achieves only 5% (2%/9%). The per-trajectory information certified in nats by the calibrator is thus directly actionable. For outof-distribution detection we roll the model on the Orszag– Tang vortex, dynamics it has never seen, and compare C against the 50 held-out in-distribution trajectories; with a single canonical OOD trajectory, AUROC equals its percentile rank among them. The alarm rings immediately: AUROC is 1.0 at depths 5 and 10 (the vortex’s cycle error exceeds every in-distribution value), 0.98 for the trajectory-mean score, and weakens at mid depths (0.64–0.88) where in-distribution errors approach the same saturation scale (Supp. Sec. B), the ideal profile: an OOD flag is most valuable before compute is spent on a long rollout (Fig. 3).

## 6.4 Inverse Rollouts

The bidirectionality is also a capability in its own right: setting $c _ { d } = - 1$ turns the trained surrogate into a fast inverse solver at no additional training cost. Seeded with only a true terminal pair $\left( { \bf z } _ { T - 1 } , { \bf z } _ { T } \right)$ of a held-out trajectory, backward rollouts reconstruct the preceding plasma history (decodedfield reconstructions in Supp. Fig. 2) - precisely the rollouts whose residual $\delta _ { i }$ is the noise floor of Supp. Sec. C with backward error growing in depth comparably to the forward direction. Cycling a backward rollout forward defines its own consistency error $\mathcal { C } _ { i } ^ { - }$ , which tracks the backward rollout error just as $\mathcal { C } _ { i }$ tracks the forward one (Supp. Fig. 3); the same round trips viewed in the latent space where all metrics are computed are shown in the forward roll out in Supp. Fig. 4 and backward roll out in Supp. Fig. 5.

![](images/1ac95fb4605666c7a7bbc8f8916e36a8114532330281ebf1cf6bb079b610c5a9.jpg)  
Figure 2: Predicted vs. true RMS errors on test data for the rolled-out latents $\mathbf { z } _ { i }$ and all six MHD fields: the true error (black dots) is predicted as a function of $\mathcal { C } _ { i }$ and falls within ±2σ bands fit from training data alone.

![](images/53309b6a32643215e7e7eaf579c53a7b609fb52801352f7b9c96c9e6a3b62dab.jpg)  
Figure 3: Out-of-distribution detection: distribution of log C over the 50 held-out trajectories at three depths, with the Orszag–Tang vortex marked: total shallow depth separation.

## 6.5 Beyond Physics: Natural Face Videos

As a probe, not a benchmark battle, we train the identical architecture on natural video: face clips from CelebV-HQ (Zhu et al. 2022) (35,666 clips, 15,653 identities). Frames are resized to 256 × 256 and encoded per-frame by a frozen, pretrained Stable Difusion image VAE (Rombach et al. 2022) into $3 2 \times 3 2 \times 4$ latents (no video VAE is trained; cf. the per-field encoding of Sec. 3). The same bidirectional denoiser (1)-(2) then learns facial dynamics. Identity-disjoint held-out clips supply true frames, so $\mathcal { E } _ { i }$ is measurable in the frozen latent space exactly as in the physics settings, and $\mathcal { C } _ { i }$ is computed by the same cycle. Talking-head dynamics are only mildly time-asymmetric, which is what makes the backward direction learnable here; strongly irreversible content (e.g., pouring, smoke) is out of scope. Absolute rollout fidelity is deliberately modest (a competitive video predictor requires scale beyond this probe), and talking-head futures are intrinsically multimodal, so any single rollout’s pointwise latent error grows quickly against the one realized future, this is precisely the regime in which a trust signal matters and, as Sec. 6.2 anticipated, in which the per-trajectory signal is strongest: clip-to-clip dificulty variation dominates the metric noise. On 150 identity-disjoint held-out clips, C ranks clips by error at fixed depth with Spearman 0.79-0.81 (latent) and 0.73-0.80 (decoded pixels) at rollout depths 3-21 (≈ 0.4-2.5 s), without decay at depth. Unlike the radiative layer, the clip factor never saturates. The same calibration recipe transfers: a two-parameter calibrator (a single globa line in log C with constant scatter; selection and configuration in Supp. Sec. A.6) fit on half the held-out clips predicts the disjoint half’s latent error within a factor of 1.28 (68%) with 95.3% coverage at ±2σ, a +0.43-nat gain over the depth-only baseline (+0.34 nats for decoded pixel error). The probe’s question is answered afirmatively: the model predicts the growth of its own error, with no test-time ground truth, even when its predictions have visibly drifted (Fig. 4).

![](images/6f97bbd1ef491636f660b18dac145e528cd6c045180197e54296ad0ab4cd7637.jpg)  
Figure 4: C tends to detect when faces morph OOD.

## 6.6 Learned-Variance UQ on Navier–Stokes

Setup. We compare directly against LE-PDE-UQ (AI4Science-WestlakeU 2024) on its own benchmark: 2D incompressible Navier–Stokes in vorticity form on a unit torus (64<sup>2</sup>, $R e { = } 1 0 ^ { 4 }$ ; the standard neural-operator dataset of Li et al. 2021), 1200 trajectories of 20 steps, using their exact data files, splits (train = first 1000, test = last 200), and protocol: the past 10 steps predict the next, rolled out autoregressively for 10 steps. Our pipeline is unchanged: a per-snapshot cVAE $( 6 4 ^ { 2 } \to 8 \times 8 \times 8 )$ and the identical bidirectional denoiser with n=10 context frames and deterministic 25-step DDIM sampling. Two training additions (context/anchor noise; pushforward-style context augmentation (Brandstetter, Worrall, and Welling 2022)) and checkpoint selection by validation rollout error are used; both are motivated and disclosed in full in Supp. Sec. E. All uncertainty objects are fit on held-out calibration trajectories from the validation split, not training trajectories whose rollout error under-represents deployment error here by 4.8× on average. Final numbers produced on untouched test split (full protocol, and the predicted-vs-measured test bands, in Supp. Sec. E and Supp. Fig. 8). Rollout error on this benchmark grows geometrically, $\bar { e } _ { d } \approx e _ { 1 } r ^ { d - 1 }$ with r ≈ 1.21 per step (fit within ${ \sim } 5 \%$ at every depth; Supp. Fig. 7), the depth-uniform amplification the geometric factors of Sec. 5.1 anticipate. A depth-only calibrator captures the shared rate $r ;$ whatever distinguishes reliable from unreliable rollouts lives in the per-trajectory deviations about this trend (cf. the variance decomposition of Supp. Sec. B) which is what $\mathcal { C } _ { i }$ measures.

Results. Table 4 reports exact metrics and pooling conventions: the uncertainty\_toolbox calibration metrics (Chung et al. 2021) (miscalibration area (MA), MACE, RMSCE) on per-pixel predictions, per-trajectory block relative $L _ { 2 } ,$ and global MAE. A single bidirectional model attains a block relative $L _ { 2 }$ of 0.244 on the untouched test split: within 1.16× of their best single model with uncertainty (0.211) and 1.29× of their most accurate configurations (the ten-mode ensemble, 0.190, and the single model without an uncertainty head, 0.189) at a tenth of the training cost, with a larger MAE (0.204 vs. their 0.161-0.181), reflecting that metric’s weighting of the late, high-amplitude frames of this spin-up flow. Their single-model uncertainty is not free: the σ head costs 11.5% accuracy (0.189 → 0.211), whereas the roundtrip signal attaches to the unmodified model at zero accuracy cost. Averaging S=5 independently seeded rollouts improves accuracy by only 0.6%, the seed-to-seed dispersion is ∼14× smaller than the realized error: the mistakes are systematic given context, the same blindness of self-disagreement that motivates round-trip checking (Sec. 6.1).

For calibration, converting the calibrated $\mu ( \mathcal { C } )$ into a constant-per-frame pixel σ lands at $\mathbf { M A } = 0 . 1 \dot { 0 } 0$ , on par with an identically fitted depth-only σ (0.090): at this benchmark’s moderate heterogeneity (fixed-depth Spearman 0.32– 0.59), a frame-level scalar from C adds little beyond depth. The signal’s value here arrives through shape and composition: the forward–backward–forward cycle disagreement (one deterministic model, 3× rollouts) reaches 0.088, and composing $\mu ( \mathcal { C } ) \mathrm { { ^ { \circ } s } }$ scale with the seed spread’s spatial shape reaches $\mathbf { M A } = 0 . 0 8 2$ , the best training-free result, against 0.058 for their learned single-model σ and 0.014 for their ten-model ensemble. Dispersion alone is not calibration: their ten-model spread-only reaches 0.182 and our $S { = } 5$ seed spread 0.118, both behind the 2× depth-only ablation. The residual gap to learned per-pixel uncertainty is within-frame heteroscedasticity that no frame-level scalar represents (zscore diagnostics in Supp. Sec. E). In latent space the same run completes the paper’s cross-system heterogeneity spectrum (between the radiative layer’s 0.11 and the faces’ 0.79); latent-space correlations and calibration are in Supp. Sec. E.

Table 4: Head-to-head on LE-PDE-UQ’s Navier–Stokes benchmark, their metrics and pooling. LE-PDE-UQ rows: test split, as published. Our rows: one-shot test evaluation $( n { = } 2 0 0$ trajectories; all uncertainty objects fit on the validation split). Cost: training / inference relative to one model, one rollout.
<table><tr><td>Signal</td><td>MA↓</td><td>MACE</td><td>RMSCE</td><td>L2↓</td><td>MAE</td><td>Cost</td></tr><tr><td colspan="7">LE-PDE-UQ (AI4Science-WestlakeU 2024)</td></tr><tr><td>Latent single (no σ)</td><td></td><td></td><td></td><td>0.189</td><td>0.161</td><td>1× tr., no UQ</td></tr><tr><td>Latent single (+σ)</td><td>0.058</td><td>0.057</td><td>0.065</td><td>0.211</td><td>0.181</td><td>1× tr., σ head</td></tr><tr><td>Latent ensemble (10, spread only)</td><td>0.182</td><td>0.181</td><td>0.202</td><td>0.190</td><td>0.161</td><td>10× tr., 10× inf.</td></tr><tr><td>Latent ensemble (10, +σ)</td><td>0.014</td><td>0.014</td><td>0.016</td><td>0.190</td><td>0.161</td><td>10× tr., 10× inf.</td></tr><tr><td colspan="7">Ours (single bidirectional model)</td></tr><tr><td> $\mu ( { \mathcal { C } } ) ,$  constant σ</td><td>0.100</td><td>0.099</td><td>0.112</td><td></td><td>0.244 0.204</td><td>2× rollouts</td></tr><tr><td>+ cycle-disagreement shape</td><td>0.088</td><td>0.087</td><td>0.099</td><td></td><td>0.244 0.204</td><td>3× rollouts</td></tr><tr><td> $S { = } 5$  seed spread alone</td><td>0.118</td><td>0.117</td><td>0.133</td><td></td><td>0.242 0.203</td><td>5× rollouts</td></tr><tr><td>µ(C)× seed-spread shape</td><td>0.082</td><td>0.081</td><td>0.092</td><td></td><td>0.242 0.203</td><td>5× rollouts</td></tr><tr><td>depth-only σ (ablation)</td><td>0.090</td><td>0.089</td><td>0.102</td><td></td><td>0.244 0.204</td><td>2× rollouts</td></tr></table>

## 6.7 Ablations

Does bidirectionality cost accuracy? The opposite, in the low-data regime. On the turbulent radiative layer (80 training trajectories) we train the identical architecture forward-only, backward-only, and bidirectionally $( p _ { \mathrm { b w d } } { = } 0 . 5 )$ at matched compute, evaluated direction-matched with fixed noise (10 seeds; mean $\pm \ \mathrm { S E ; }$ Supp. Sec. D). The shared model outperforms each specialist on the specialist’s own direction: $0 . 0 6 3 0 \pm 0 . 0 0 0 6$ vs. $0 . 0 6 7 5 \pm 0 . 0 0 0 7$ forward and $0 . 0 6 2 1 \pm 0 . 0 0 0 6$ vs. $0 . 0 6 8 9 \pm 0 . 0 0 0 7$ backward, a 7-10% improvement despite seeing only half as many examples per direction, and it overfits markedly later (Supp. Sec. D).

Each trajectory supplies transitions in both temporal directions, so the direction flag acts as free augmentation. The specialists fail severely of-direction (5-7× higher loss), confirming the round-trip check requires both directions: the structure that enables $\mathcal { C } _ { i }$ thus comes at negative cost: one network, half the parameters of two specialists, better accuracy in both directions. A linear analysis explains the mechanism: for stationary dynamics the time-reversed map is a fixed reparameterization of the forward one $( \Sigma A ^ { \top } \Sigma ^ { - 1 } ;$ exactly $A ^ { \dagger }$ for whitened latents), so the direction flag implements a correctly-specified weight tying whose estimation variance is up to half a specialist’s, a gain scaling as $1 / n ,$ largest in the low-data regime measured here (Supp. Sec. D). On the turbulent radiative layer the situation inverts instructively: the pertrajectory factor is smaller (±11% vs. MHD’s ±28%) and is masked by $\mathcal { C } _ { i } { ' } s$ own per-depth noise, the variance decomposition of Supp. Sec. B bounds the observable fixed-depth correlation at $\approx 0 . 1 2$ , matching the measured 0.11 $\mathrm { ( } n \mathrm { = } 7 2 )$ so a depth-only calibrator is already near-optimal here. The per-trajectory signal is thus valuable in proportion to how sharply trajectories difer relative to the metric’s noise: large on MHD, marginal here, and maximal by construction for natural video, where clip-to-clip heterogeneity dominates.

## 7 Discussion and Conclusion

Round-trip consistency turns a structural property of one network representing both temporal directions of a system into a practical trust signal: an error meter that travels with the model, needs no ground truth, ensembles, or governing equations, and costs one extra rollout. The evidence is quantitative: the meter predicts held-out MHD rollout error to within a factor of 1.14 (68%) with near-nominal coverage, and a single bidirectional model approaches a ten-model ensemble’s accuracy at a tenth of its training cost (Secs. 6.2 and 6.6). Bidirectional difusion models can, in a precise and calibrated sense, predict their own rollout errors. The approach has clear limitations. The check is necessary rather than suficient: cancellation is empirically rare (95.1% of held-out points within the calibrated ±2σ band), but a small $\mathcal { C } _ { i }$ is evidence, not certification, and Proposition 1 makes the dependence on backward-map conditioning explicit. Backward dynamics of strongly dissipative systems are ill-posed in the continuum; the learned inverse remains informative at the horizons studied, but very long horizons and stif systems deserve dedicated study. And $\mathcal { C } _ { i }$ lives in latent space; relating it to physically calibrated field-space metrics is a natural refinement. Because $\mathcal { C } _ { i }$ is a measurable scalar cost, it can in principle be minimized online by gradient-free adaptive feedback (Scheinker and Scheinker 2016), turning error detection into error correction at test time; ongoing work beyond this paper’s scope. The principle may also extend to discrete sequence models, where time-reversed language models already provide unsupervised feedback (Yerram et al. 2024); the severe non-injectivity of language $( \mu \to 0$ in Sec. 5.1) makes that a distinct and interesting regime.

## Acknowledgments

This work was funded by Los Alamos National Laboratory and TPUs were provided by Google’s TPU developer program.

## References

AI4Science-WestlakeU. 2024. Uncertainty Quantification for Forward and Inverse Problems of PDEs via Latent Global Evolution. arXiv preprint arXiv:2402.08383.

Anderson, B. D. O. 1982. Reverse-time difusion equation models. Stochastic Processes and their Applications, 12(3): 313–326.

Angelopoulos, A. N.; and Bates, S. 2023. Conformal Prediction: A Gentle Introduction. Foundations and Trends in Machine Learning, 16(4): 494–591.

Arık, S. Ö.; Yoder, N. C.; and Pfister, T. 2022. Self-Adaptive Forecasting for Improved Deep Learning on Non-Stationary Time-Series. arXiv preprint arXiv:2202.02403.

Baxter, J. 2000. A Model of Inductive Bias Learning. JAIR, 12: 149–198.

Behrmann, J.; Grathwohl, W.; Chen, R. T. Q.; Duvenaud, D.; and Jacobsen, J.-H. 2019. Invertible Residual Networks. In International Conference on Machine Learning.

Brandstetter, J.; Worrall, D.; and Welling, M. 2022. Message Passing Neural PDE Solvers. In International Conference on Learning Representations.

Cachay, S. R.; Zhao, B.; Joren, H.; and Yu, R. 2023. DYfusion: A Dynamics-informed Difusion Model for Spatiotemporal Forecasting. In Advances in Neural Information Processing Systems, volume 36.

Chakraborty; et al. 2022. Improving Motion Forecasting for Autonomous Driving with the Cycle Consistency Loss. In Machine Learning for Autonomous Driving Workshop, NeurIPS. ArXiv:2211.00149.

Chen, B.; Martí Monsó, D.; Du, Y.; Simchowitz, M.; Tedrake, R.; and Sitzmann, V. 2024. Difusion Forcing: Next-Token Prediction Meets Full-Sequence Difusion. In Advances in Neural Information Processing Systems, volume 37.

Chung, H.; Kim, J.; McCann, M. T.; Klasky, M. L.; and Ye, J. C. 2023. Difusion Posterior Sampling for General Noisy Inverse Problems. In International Conference on Learning Representations.

Chung, Y.; Char, I.; Guo, H.; Schneider, J.; and Neiswanger, W. 2021. Uncertainty Toolbox: an Open-Source Library for Assessing, Visualizing, and Improving Uncertainty Quantification. arXiv preprint arXiv:2109.10254.

Degrave, J.; Felici, F.; Buchli, J.; Neunert, M.; Tracey, B.; Carpanese, F.; et al. 2022. Magnetic control of tokamak plasmas through deep reinforcement learning. Nature, 602: 414–419.

Dhariwal, P.; and Nichol, A. 2021. Difusion models beat gans on image synthesis. Advances in neural information processing systems, 34: 8780–8794.

Esser, P.; Rombach, R.; and Ommer, B. 2021. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 12873–12883.

Gal, Y.; and Ghahramani, Z. 2016. Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. In International Conference on Machine Learning.

Geifman, Y.; and El-Yaniv, R. 2017. Selective Classification for Deep Neural Networks. In Advances in Neural Information Processing Systems, volume 30.

Hang, T.; Gu, S.; Li, C.; Bao, J.; Chen, D.; Hu, H.; Geng, X.; and Guo, B. 2023. Eficient Difusion Training via Min-SNR Weighting Strategy. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

Hendrycks, D.; and Gimpel, K. 2017. A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. In International Conference on Learning Representations.

Higgins, I.; Matthey, L.; Pal, A.; Burgess, C.; Glorot, X.; Botvinick, M.; Mohamed, S.; and Lerchner, A. 2017. beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. In International Conference on Learning Representations.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising Difusion Probabilistic Models. In Advances in Neural Information Processing Systems, volume 33.

Huang, J.; Yang, G.; Wang, Z.; and Park, J. J. 2024. DiffusionPDE: Generative PDE-Solving under Partial Observation. In Advances in Neural Information Processing Systems, volume 37.

Huang, L.; Li, J.; Ding, X.; Zhang, Y.; Chen, H.; and Ozcan, A. 2023. Cycle Consistency-Based Uncertainty Quantification of Neural Networks in Inverse Imaging Problems. Intelligent Computing, 2: 0071. ArXiv:2305.12852.

Huh, I.; et al. 2020. Time-Reversal Symmetric ODE Network. In Advances in Neural Information Processing Systems, volume 33.

Karniadakis, G. E.; Kevrekidis, I. G.; Lu, L.; Perdikaris, P.; Wang, S.; and Yang, L. 2021. Physics-informed machine learning. Nature Reviews Physics, 3: 422–440.

Kingma, D. P.; and Welling, M. 2014. Auto-Encoding Variational Bayes. In 2nd International Conference on Learning Representations (ICLR).

Kochkov, D.; Smith, J. A.; Alieva, A.; Wang, Q.; Brenner, M. P.; and Hoyer, S. 2021. Machine learning–accelerated computational fluid dynamics. Proceedings of the National Academy ofSciences, 118(21): e2101784118.

Kohl, G.; Chen, L.-W.; and Thuerey, N. 2026. Benchmarking Autoregressive Conditional Difusion Models for Turbulent Flow Simulation. Neural Networks, 108641.

Kohl, G.; Um, K.; and Thuerey, N. 2023. Benchmarking Autoregressive Conditional Difusion Models for Turbulent Flow Simulation. arXiv preprint arXiv:2309.01745.

Kuleshov, V.; Fenner, N.; and Ermon, S. 2018. Accurate Uncertainties for Deep Learning Using Calibrated Regression. In International Conference on Machine Learning.

Lakshminarayanan, B.; Pritzel, A.; and Blundell, C. 2017. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. In Advances in Neural Information Processing Systems, volume 30.

Lam, R.; Sanchez-Gonzalez, A.; Willson, M.; Wirnsberger, P.; Fortunato, M.; Alet, F.; et al. 2023. Learning skillful medium-range global weather forecasting. Science, 382(6677): 1416–1421.

Li, Z.; Kovachki, N.; Azizzadenesheli, K.; Liu, B.; Bhattacharya, K.; Stuart, A.; and Anandkumar, A. 2021. Fourier Neural Operator for Parametric Partial Diferential Equations. In International Conference on Learning Representations.

Li, Z.; Zhou, A.; and Barati Farimani, A. 2025. Generative Latent Neural PDE Solver using Flow Matching. arXiv preprint arXiv:2503.22600.

Lippe, P.; Veeling, B. S.; Perdikaris, P.; Turner, R. E.; and Brandstetter, J. 2023. PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers. In Advances in Neural Information Processing Systems, volume 36.

Lu, L.; Jin, P.; Pang, G.; Zhang, Z.; and Karniadakis, G. E. 2021. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3: 218–229.

Maurer, A.; Pontil, M.; and Romera-Paredes, B. 2016. The Benefit of Multitask Representation Learning. JMLR, 17(81): 1–32.

Meister, S.; Hur, J.; and Roth, S. 2018. UnFlow: Unsupervised Learning of Optical Flow with a Bidirectional Census Loss. In Proceedings of the AAAI Conference on Artificial Intelligence.

Mocz, P.; Vogelsberger, M.; and Hernquist, L. 2014. A constrained transport scheme for MHD on unstructured static

and moving meshes. Monthly Notices of the Royal Astronomical Society, 442(1): 43–55.

Nichol, A. Q.; and Dhariwal, P. 2021. Improved Denoising Difusion Probabilistic Models. In Proceedings of the 38th International Conference on Machine Learning (ICML), volume 139, 8162–8171.

Nix, D. A.; and Weigend, A. S. 1994. Estimating the Mean and Variance of the Target Probability Distribution. In IEEE International Conference on Neural Networks.

Ohana, R.; McCabe, M.; et al. 2024. The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, volume 37.

Orszag, S. A.; and Tang, C.-M. 1979. Small-scale structure of two-dimensional magnetohydrodynamic turbulence. Journal ofFluid Mechanics, 90(1): 129–143.

Ovadia, Y.; Fertig, E.; Ren, J.; Nado, Z.; Sculley, D.; Nowozin, S.; Dillon, J.; Lakshminarayanan, B.; and Snoek, J. 2019. Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty under Dataset Shift. In Advances in Neural Information Processing Systems, volume 32.

Peebles, W.; and Xie, S. 2023a. Scalable Difusion Models with Transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 4195–4205.

Peebles, W.; and Xie, S. 2023b. Scalable Difusion Models with Transformers. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), 4195–4205.

Perez, E.; Strub, F.; De Vries, H.; Dumoulin, V.; and Courville, A. 2018. Film: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI conference on artificial intelligence, volume 32.

Price, I.; Sanchez-Gonzalez, A.; Alet, F.; Andersson, T. R.; El-Kadi, A.; Masters, D.; Ewalds, T.; Stott, J.; Mohamed, S.; Battaglia, P.; Lam, R.; and Willson, M. 2025. Probabilistic weather forecasting with machine learning. Nature, 637: 84–90.

Raissi, M.; Perdikaris, P.; and Karniadakis, G. E. 2019. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational Physics, 378: 686–707.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 10684– 10695.

Rozet, F.; and Louppe, G. 2023. Score-based Data Assimilation. In Advances in Neural Information Processing Systems, volume 36.

Sanchez-Gonzalez, A.; Godwin, J.; Pfaf, T.; Ying, R.; Leskovec, J.; and Battaglia, P. 2020. Learning to Simulate Complex Physics with Graph Networks. In International Conference on Machine Learning.

Scheinker, A. 2024. cDVAE: VAE-guided difusion for particle accelerator beam 6D phase space projection diagnostics. Scientific Reports, 14: 29303.

Scheinker, A. 2026. Bidirectional Autoregressive Latent Difusion for Forward and Inverse Magnetohydrodynamics. arXiv preprint arXiv:2606.29620.

Scheinker, A.; Plastun, A.; and Ostroumov, P. 2026. Phase-Flow4D: Physically constrained 4D beam reconstruction via feedback-guided latent difusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4729–4737.

Scheinker, A.; and Scheinker, D. 2016. Bounded extremum seeking with discontinuous dithers. Automatica, 69: 250– 257.

Scheinker, A.; and Williams, A. 2025. Latent difusion can map beam loss to two-dimensional phase-space projections. Physical Review Accelerators and Beams, 28(9): 094602.

Shu, D.; and Barati Farimani, A. 2024. Zero-Shot Uncertainty Quantification using Difusion Probabilistic Models. arXiv preprint arXiv:2408.04718.

Shysheya, A.; Diaconu, C.; Bergamin, F.; Perdikaris, P.; Hernández-Lobato, J. M.; Turner, R. E.; and Mathieu, E. 2024. On Conditional Difusion Models for PDE Simulations. In Advances in Neural Information Processing Systems, volume 37.

Sohn, K.; Lee, H.; and Yan, X. 2015. Learning Structured Output Representation using Deep Conditional Generative Models. In Advances in Neural Information Processing Systems (NeurIPS), volume 28.

Song, J.; Meng, C.; and Ermon, S. 2021. Denoising Diffusion Implicit Models. In 9th International Conference on Learning Representations (ICLR).

Song, Y.; Dhariwal, P.; Chen, M.; and Sutskever, I. 2023. Consistency Models. In International Conference on Machine Learning.

Song, Y.; Sohl-Dickstein, J.; Kingma, D. P.; Kumar, A.; Ermon, S.; and Poole, B. 2021. Score-Based Generative Modeling through Stochastic Diferential Equations. In International Conference on Learning Representations.

Sundaram, N.; Brox, T.; and Keutzer, K. 2010. Dense Point Trajectories by GPU-Accelerated Large Displacement Optical Flow. In European Conference on Computer Vision.

Um, K.; Brand, R.; Fei, Y.; Holl, P.; and Thuerey, N. 2020. Solver-in-the-Loop: Learning from Diferentiable Physics to Interact with Iterative PDE-Solvers. In Advances in Neural Information Processing Systems, volume 33.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention Is All You Need. In Advances in Neural Information Processing Systems, volume 30.

Virmaux, A.; and Scaman, K. 2018. Lipschitz regularity of deep neural networks: analysis and eficient estimation. In Advances in Neural Information Processing Systems, volume 31.

von Platen, P.; Patil, S.; Lozhkov, A.; Cuenca, P.; Lambert, N.; Rasul, K.; Davaadorj, M.; Nair, D.; Paul, S.; Berman, W.; Xu, Y.; Liu, S.; and Wolf, T. 2022. Difusers: State-of-the-art difusion models. https://github.com/huggingface/difusers.

Wang, X.; Girshick, R.; Gupta, A.; and He, K. 2018. Nonlocal Neural Networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Wang, X.; Jabri, A.; and Efros, A. A. 2019. Learning Correspondence from the Cycle-Consistency of Time. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2566–2576.

Yerram, V.; Madhavan, R.; Addepalli, S.; Suggala, A.; Shanmugam, K.; and Jain, P. 2024. Time-Reversal Provides Unsupervised Feedback to LLMs. In Advances in Neural Information Processing Systems, volume 37.

Yoo, S.; Koo, J.; Choi, D.; and Sung, M. 2026. DifusionRollout: Uncertainty-Aware Rollout Planning in Long-Horizon PDE Solving. Transactions on Machine Learning Research. ArXiv:2602.13616.

Zhu, H.; Wu, W.; Zhu, W.; Jiang, L.; Tang, S.; Zhang, L.; Liu, Z.; and Loy, C. C. 2022. CelebV-HQ: A Large-Scale Video Facial Attributes Dataset. In European Conference on Computer Vision.

Zhu, J.-Y.; Park, T.; Isola, P.; and Efros, A. A. 2017. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. In Proceedings of the IEEE International Conference on Computer Vision, 2223–2232.

This document contains the experimental configurations, additional analyses, and full derivations supporting the main paper. Works cited only here appear in this document’s own reference list.

# Supplementary Materials: Round-Trip Consistency: Bidirectional Difusion Models Can Predict Their Own Rollout Errors

Appendix

## A Experimental Configurations

## A.1 Embedding into Latent Space

The first step for each of the data sets that we work with is to compress to a lower dimensional latent space by using a variational autoencoder (VAE) or a conditional variational autoencoder (cVAE) (Kingma and Welling 2014; Sohn, Lee, and Yan 2015). For the CelebV-HQ dataset, we utilize a pre-trained Stable Difusion VAE (Rombach et al. 2022), whose architecture is inherited from a VQGAN approach (Esser, Rombach, and Ommer 2021). The model was aquired using the difusers library from Huggingface (von Platen et al. 2022), class name AutoencoderKL, difusers version 0.4.2 with silu activation functions and block out channels [128, 256, 512, 512], 4 latent channels, 2 layers per residual block, and norm groups of size 32.

For all other data, each of the independent multi-channel inputs, such as the six fields of MHD, or the four fields of the turbulent radiative layer, a single cVAE is used for each field, guided by a one-hot encoded conditional vector via Feature-wise Linear Modulation (FiLM) (Perez et al. 2018). Each cVAE uses multiple residual blocks at each stage of compression and our scale/shift

$$
\mathrm { F i L M } ( \mathbf { x } ) = \gamma ( \mathbf { c } ) \odot \mathbf { x } + \beta ( \mathbf { c } )\tag{7}
$$

is applied to the GroupNorm output inside residual blocks, as done in adaptive group normalization (AdaGN) used in modern difusion U-Nets (Dhariwal and Nichol 2021). Singlehead spatial self-attention (Vaswani et al. 2017), equivalent to a non-local block (Wang et al. 2018), is applied at the lowest resolution on both sides of the latent bottleneck.

For The Well, the cVAE trained for 500 epochs, with a batch size of 64, with Adam, with a learning rate of $1 e - 4 .$ KL divergence weight of $\beta = 1 e - 3 .$ , 16 latent channels, 32 input activations which were then multiplied by factors of [2, 4, 8, 16, 32] through the encoder and reverse of that through the decoder, resulting in a latent space representation ofsize 4×12×16 ofeach input field image ofsize 128×384× 1, with 2 residual blocks at each resolution and attention at the pinch point with 4 heads, with a total parameter count of 240 M.

For the Navier Stokes / LE-PDE-UQ cVAE, it was trained for 500 epochs with a batch size of 32 and then an additional 500 epochs with a batch size of 8, each with Adam with a learning weight of 1e − 4, KL weight of $1 e \mathrm { ~ - ~ } 4 .$ , two residual blocks per resolution in the encoder and decoder, and 4 attention heads in the attention layers on either side of the pinch. The model has 16 input activations followed by 3 down and up stages whose activation numbers are 32,

64, and 128, with a latent channel size of 8, so that an input image of size $6 4 \times 6 4 \times 1$ is encoded into a latent of size $8 \times 8 \times 8$ . The cVAE had a total parameter count of 3.9 M.

The MHD work was all done on a Google V6e TPU, all other work was done on NVIDIA H200 GPUs.

## A.2 Denoiser Architecture and Conditioning

All experiments use a denoising difusion model (Ho, Jain, and Abbeel 2020) with the same transformer denoiser setup; only the latent geometry $( H _ { \ell } , W _ { \ell } , F )$ changes between systems. The denoiser is a difusion transformer (DiT) (Peebles and Xie 2023b): the noisy target frame, the N nearest context frames, and an anchor frame are decomposed into $2 \times 2$ patches, linearly embedded, and concatenated into a single token sequence with learned positional embeddings, so that self-attention acts jointly over space and a short temporal window. The scalar conditions (difusion timestep, the simulation-time index of the target, and a temporal direction flag) are fused into one conditioning vector (sinusoidal embeddings for the two timesteps, a learned two-entry embedding for the direction) that modulates every transformer block via adaLN-Zero (Peebles and Xie 2023b). The direction flag makes a single network bidirectional: it is trained on a mixture of forward and reverse-time examples in which the anchor is the initial condition $z _ { \mathrm { 0 } }$ when predicting $z _ { t + 1 }$ from the N frames ending $\mathrm { a t } ~ z _ { t }$ (direction +1), and the final state $z _ { T - 1 }$ when predicting $z _ { t - 1 }$ from the N frames ending at $z _ { t + 1 }$ (direction −1); context slots are ordered nearest-last, so each slot carries the same distance-to-target meaning in both directions. Training minimizes the standard ϵ-prediction objective (Ho, Jain, and Abbeel 2020) under a cosine noise schedule (Nichol and Dhariwal 2021) with min-SNR-γ loss weighting $( \gamma = 5 )$ (Hang et al. 2023), and an exponential moving average of the parameters is retained for sampling. At inference, each frame is drawn with a deterministic DDIM sampler (Song, Meng, and Ermon 2021) and the model is rolled out autoregressively in either time direction, appending each generated frame to the sliding context window, in the spirit of autoregressive conditional difusion emulators for turbulent flow (Kohl, Chen, and Thuerey 2026).

Tokenization. Each latent frame is a channels-last array of shape $H _ { \ell } \times W _ { \ell } \times F$ . The input to the denoiser is the ordered frame list [ noised target, $\mathrm { c o n t e x t 1 } , \ldots , \mathrm { c o n t e x t } _ { n }$ anchor ], with contexts ordered farthest-to-nearest (the frame adjacent to the target is last) and the anchor fixed to $z _ { \mathrm { 0 } }$ for forward and $z _ { T - 1 }$ for backward prediction; near a trajectory boundary, missing context slots are padded with the anchor frame. Every frame is split into non-overlapping $p \times p$ patches $\scriptstyle ( p = 2 )$ giving $( H _ { \ell } / \bar { p ) } ( W _ { \ell } / p )$ tokens of dimension $\bar { p ^ { 2 } } F$ per frame, each mapped by a shared linear layer to the transformer width. The $( n { + } 2 )$ frames’ tokens are concatenated along the sequence axis, and a single learned positional embedding (initialized $0 . 0 2 \mathcal { N } ( 0 , \mathbf { I } ) )$ over the full sequence encodes both spatial location and frame role, since each frame occupies a fixed block of positions. Conditioning on past states therefore requires no extra machinery: the target’s tokens attend to the context and anchor tokens through ordinary self-attention.

Scalar conditioning (adaLN-Zero). The three scalars: denoising step k, the target frame’s simulation-time index t, and the direction flag $c _ { d } \in \{ + 1 , - 1 \}$ form a single conditioning vector

$$
\begin{array} { r c l } { c } & { = } & { \mathrm { M L P } ( \operatorname { S i L U } \left[ W _ { k } \sin ( k ) \right] + \operatorname { S i L U } \left[ W _ { t } \sin ( t ) \right] } \\ & & { + E _ { d } [ \mathbb { 1 } _ { c _ { d } < 0 } ] ) \in \mathbb { R } ^ { 2 5 6 } , } \end{array}\tag{8}
$$

where sin(·) is a sinusoidal embedding and $E _ { d }$ a learned twoentry table. Following adaLN-Zero (Peebles and Xie 2023a), each transformer block applies pre-norm LayerNorm without afine parameters and derives, from c through a zeroinitialized linear map, six per-block vectors: shift, scale, and residual gate for the attention sublayer and again for the MLP sublayer. Zero initialization makes every block the identity at the start of training. The final layer applies a two-parameter adaLN modulation followed by a zero-initialized linear projection back to patch dimension; only the target frame’s tokens pass through it and are un-patchified into the noise prediction $\hat { \epsilon } \in \mathbb { R } ^ { \breve { H } _ { \ell } \times W _ { \ell } \times F }$

Block and size details. For the difusion transformer, as describe in A.2, we used width 384, depth 12, 6 attention heads, MLP ratio 4 with GELU, conditioning width 256, patch size 2, fixed across all systems. For the face-video runs the attention uses qk-normalization (per-head LayerNorm applied to queries and keys before the dot product), which we found necessary to prevent attention-logit growth and the associated late-training divergence; the physics runs did not require it. Per system, with $n { = } 2$ context frames the token sequence lengths are: MHD, $\left( H _ { \ell } , W _ { \ell } , F \right) = \left( 1 6 , 1 6 , 2 4 \right)$ (six fields, four latent channels each), 4 × 64 = 256 tokens; turbulent radiative layer, $( 4 , 1 2 , F )$ with $F = 6 4$ (4 fields × 16 latent channels), $4 \times 1 2 = 4 8$ tokens; face videos, (32, 32, 4) (frozen Stable Difusion VAE latents), $4 \times 2 5 6 = 1 0 2 4$ tokens. Training uses the ϵ-prediction objective of the main paper with min-SNR-γ weighting $( \gamma { = } 5 )$ and an exponential moving average of the weights (decay 0.999); sampling uses deterministic DDIM. Optimization schedules and per-system data processing are given in the configuration appendices.

Parameter counts. For the CelebV-HQ difusion model, it is working on latent images of size $3 2 \times 3 2 \times 4$ and this was the most challenging data set by far because of the multimodal nature, with each videos actual dynamics independent of each other. For DDIM training, 1000 difusion steps were used, and for sampling at inference time we used 30 steps. With a patch size of 2, hidden token dimension of 2048, 16 attention heads were used per transformer and the model was 16 transformers deep, the conditional embedding dimension used was 512, and the total parameter count was 912 million parameters.

For the MHD data, for DDIM training, 1000 difusion steps were used, and for sampling at inference time we used 50 steps. The difusion model had a depth of 12 with 6 attention heads each, a conditional dimension of 256, hidden token dimension of 384, conditional embedding dimension of 256, resulting in a total parameter count of 29 million parameters.

For the Well data, the same setup was used as for MHD with 200 difusion steps for DDIM training and 25 steps for sampling. The difusion model had a depth of 12 with 6 attention heads each, a conditional dimension of 256, hidden token dimension of 384, conditional embedding dimension of 256, resulting in a total parameter count of 29 million parameters.

For the Navier-Stokes data, the same setup was used as for MHD and Well with 200 difusion steps for DDIM training and 25 steps for sampling. The difusion model had a depth of 12 with 6 attention heads each, a conditional dimension of 256, hidden token dimension of 384, conditional embedding dimension of 256, resulting in a total parameter count of 29 million parameters.

## A.3 2D Compressible MHD

For modeling 2D MHD, we utilize the tool and approach of P. Mocz (Mocz, Vogelsberger, and Hernquist 2014), which considers the conservation law dynamics

$$
\frac { \partial \mathbf { U } } { \partial t } + \nabla \cdot \mathbf { F ( \mathbf { U } ) } = 0 ,\tag{9}
$$

$$
\mathbf { U } = \left( \rho \atop \rho \mathbf { v } \right) , \quad \mathbf { F } ( \mathbf { U } ) = \left( \begin{array} { c } { \rho \mathbf { v } } \\ { \rho \mathbf { v v } ^ { T } + p - \mathbf { B } \mathbf { B } ^ { T } } \\ { \rho e \mathbf { v } + p \mathbf { v } - \mathbf { B } ( \mathbf { v } \cdot \mathbf { B } ) } \\ { \mathbf { B } \mathbf { v } ^ { T } - \mathbf { v } \mathbf { B } ^ { T } , } \end{array} \right)\tag{10}
$$

where $\begin{array} { r } { p = p _ { \mathrm { g a s } } + \frac { 1 } { 2 } \mathbf { B } ^ { 2 } } \end{array}$ is the total gass pressure, $e = u +$ $\begin{array} { r } { { \frac { 1 } { 2 } } { \bf v } ^ { 2 } + { \frac { 1 } { 2 \rho } } { \bf B } ^ { 2 } } \end{array}$ is the total energy per unit mass, and u is the thermal energy per unit mass.

For MHD, at each time step one MHD state is represented by six fields $( \rho , P , v _ { x } , v _ { y } , B _ { x } , B _ { y } ) .$ , having size $5 1 2 \times 5 1 2 \times 6 .$ We split that into 6 single-channel inputs of size $5 1 2 \times 5 1 2 \times 1$ and then encode the first field with a conditional vector $[ 1 , 0 , 0 , 0 , 0 , 0 ]$ , the second with $[ 0 , 1 , 0 , 0 , 0 , 0 ] , \ldots$ , and the sixth with $[ 0 , \mathrm { { \dot { 0 } } } , 0 , 0 , 0 , 0 , 1 ]$ , with FiLM conditioning on each residual block of both the encoder and decoder, as described in A.1. Each channel is encoded into a 16×16×4 latent $( \beta = 1 0 ^ { - 3 } )$ . The number of weights in the encoder’s downsampling layers of the cVAE are [32, 64, 128, 256, 512], which is mirrored for the decoder, with 1 residual block per resolution and an 8 headed attention block on either side of the latent bottleneck. With this configuration our cVAE model has 192 million parameters. We use 500 train and 50 test trajectories where each trajectory consists of 100 steps over 1 second, as generated by using a spatial resolution of $5 1 2 \times 5 1 2$ , with 100 steps over 1 s via the constrained transport code (Mocz, Vogelsberger, and Hernquist 2014). The random vector potential generating the initial magnetic fields is a mixture of 100 Gaussians. The model is trained for 1M steps with a batch size of 16. Latent roll out error vs round-trip consistency for the MHD data is shown, colored by turnaround depth, in Fig. 5. Examples of forward and backward autoregressive MHD field rollouts are shown in Fig. 6.

## A.4 External 2D Benchmark: Turbulent Radiative Layer (the Well)

turbulent\_radiative\_layer\_2D from the Well (Ohana, McCabe et al. 2024): an astrophysical radiative turbulent mixing layer (Kelvin–Helmholtz instability between hot dilute and cold dense gas with radiative cooling); 90 trajectories (10 seeds × 9 cooling times) of 101 steps at $1 2 \bar { 8 } \times 3 8 4$ , four fields (density, pressure, $v _ { x } , v _ { y } ) .$ periodic in x and zero-gradient in $y .$ . We use the oficial train/valid/test split and report the benchmark’s standard VRMSE alongside our metrics for anchoring. Each field is encoded by its own VAE with the recipe of Appendix A.3, yielding 4×12×4 latents per field (rectangular latent grids; no square-shape assumptions); the bidirectional denoiser configuration is unchanged unless noted.

![](images/b8ca0365eed35bf4e88732c49d9aa0cc5331ed47a5e739c74122e1d8c841ac5c.jpg)  
Figure 5: Round-trip consistency $\mathcal { C } _ { i }$ vs. true latent rollout error $\mathcal { E } _ { i }$ on the 50 held-out MHD trajectories, colored by depth i (turnaround-anchored cycles; deterministic sampler). Fixed-depth Spearman is 0.91–0.98.

Comparison to The Well baselines. Following the evaluation protocol of Ohana, McCabe et al. (2024), we report the variance-scaled root mean squared error,

$$
\mathrm { V R M S E } \left( \hat { u } , u \right) = \sqrt { \frac { \left. ( \hat { u } - u ) ^ { 2 } \right. _ { \Omega } } { \left. ( u - \left. u \right. \Omega ) ^ { 2 } \right. _ { \Omega } + \varepsilon } } , \qquad \varepsilon = 1 0 ^ { - 7 } ,\tag{11}
$$

where $\langle \cdot \rangle _ { \Omega }$ is the spatial mean over the $1 2 8 \times 3 8 4$ grid and the denominator uses the unbiased spatial variance, matching the reference implementation. VRMSE is computed per snapshot and per field in physical units—latent predictions are first decoded with the cVAE—and then averaged, so a score of 1 corresponds to predicting the spatial mean of the target field. On single-step prediction for the turbulent\_radiative\_layer\_2D test split, with each frame predicted from the two preceding ground-truth frames, our model attains VRMSE = 0.59 (density 0.21, pressure 1.32, $v _ { x } \ 0 . 3 0 , v _ { y } \ 0 . 5 4 )$ , versus 0.50 (FNO), 0.50 (TFNO), 0.24 (U-net), and 0.20 (CNextU-net) for the baselines of Ohana, McCabe et al. (2024). Three protocol diferences favor the baselines: they condition on four past states rather than our two; they operate at full spatial resolution, whereas our predictions pass through a 64×-compressed latent bottleneck, cVAE reconstruction of the ground truth alone yields $\mathrm { V R M S E } = 0 . 3 0$ on this split (0.67 on pressure), an irreducible floor in this latent space that already exceeds the total one-step error of the two strongest baselines—and they are deterministic regressors, whereas we report a single stochastic sample. Consistent with the per-field analysis of Ohana, McCabe et al. (2024), error is concentrated in the pressure field; notably, this holds for the reconstruction floor as well, and the one-step error is roughly twice the floor in every field, indicating that the pressure field on this dataset is intrinsically dificult both to compress and to predict. We emphasize that the model was not tuned for singlestep accuracy: our contribution concerns forward–backward rollout consistency, for which competitive (rather than stateof-the-art) single-step fidelity is enough. Over autoregressive rollouts, seeded and conditioned throughout on a two-frame context, our time-windowed VRMSE is 1.12 over steps 6–12 and 2.86 over steps 13–30, versus 0.54/1.01 for CNextU-net, 0.66/1.04 for U-net, 1.79/3.54 for FNO, and 6.01/>10 for TFNO: despite the latent bottleneck and shorter context, our model outperforms both spectral baselines at every horizon while trailing the convolutional ones. We note that pointwise VRMSE structurally favors deterministic regressors at long horizons on chaotic dynamics: as predictability is lost, a regressor reverting to the conditional mean approaches a score of 1, whereas an emulator drawing a sharp sample that is statistically consistent with, but decorrelated from, the reference incurs $\mathrm { V R M S E } \approx \sqrt { 2 }$ even in the ideal case. Part of the long-horizon gap therefore reflects the metric’s preference for mean-reverting predictions rather than sample fidelity per se.

## A.5 Navier Stokes

For the 2D turbulent radiative layer data, the governing equations are

$$
\begin{array} { r c l } { \displaystyle \frac { \partial \rho } { \partial t } } & { = } & { - \nabla \cdot ( \rho v ) , } \end{array}\tag{12}
$$

$$
\begin{array} { r c l } { \displaystyle \frac { \partial \rho v } { \partial t } } & { = } & { \displaystyle - \nabla \cdot ( \rho v v + P ) , } \end{array}\tag{13}
$$

$$
\frac { \partial E } { \partial t } = - \nabla \cdot ( ( E + P ) \boldsymbol { v } ) - \frac { E } { t _ { \mathrm { c o o l } } } ,\tag{14}
$$

$$
E = \frac { P } { \gamma - 1 } , \gamma = \frac { 5 } { 3 } ,\tag{15}
$$

where $\rho$ is density, v is 2D velocity, $P$ is pressure, $E$ is total energy, and $t _ { \mathrm { c o o l } }$ the cooling time.

## A.6 CelebV-HQ Face Videos

Evaluation. $\mathcal { E } _ { i }$ as latent-space MSE on identity-disjoint test clips (decoded LPIPS as a secondary check); $\mathcal { C } _ { i }$ as defined in the main paper, with the deterministic sampler (S-averaged variant ablated). Some examples of autoregressive CELEBV-HQ rollouts and their $\mathcal { C } _ { i }$ values are shown in Fig. 12 and Fig. 13.

## B Why the Per-Trajectory Signal Is Strong on MHD and Marginal on the Radiative Layer: A Variance Decomposition

For each system we remove the depth trend from the log quantities and split the residual variance into a between-trajectory component (a per-trajectory dificulty factor) and withintrajectory fluctuation $( \sigma _ { b } , \sigma _ { w }$ below; $\begin{array} { r } { \dot { \mathrm { I C C } } = \dot { \sigma _ { b } ^ { 2 } } / ( \sigma _ { b } ^ { 2 } + \sigma _ { w } ^ { 2 } ) ) } \end{array}$ on the training ensembles (n=100 MHD, $n { = } 7 2$ radiative layer):

<table><tr><td>System</td><td></td><td> $\sigma _ { b }$ </td><td> $\sigma _ { w }$ </td><td>ICC</td><td>rel.</td></tr><tr><td>MHD</td><td>E</td><td>0.250</td><td>0.074</td><td>0.92</td><td>0.96</td></tr><tr><td rowspan="3">Rad. layer</td><td>C</td><td>0.349</td><td>0.118</td><td>0.90</td><td>0.95</td></tr><tr><td>ε</td><td>0.104</td><td>0.191</td><td>0.23</td><td>0.48</td></tr><tr><td>C</td><td>0.057</td><td>0.220</td><td>0.06</td><td>0.25</td></tr></table>

Here reliability $\mathrm { ( r e l . ) } = \sigma _ { b } / \sqrt { \sigma _ { b } ^ { 2 } + \sigma _ { w } ^ { 2 } }$ is the correlation between a single-depth reading and the trajectory factor it estimates. Both systems possess a per-trajectory factor; its magnitude difers (±28% vs. ±11% multiplicative), and so does the metric noise it must be read through.

Attenuation ceiling. If the underlying factors of C and $\mathcal { E }$ are correlated at $\rho _ { f }$ , the observable fixed-depth correlation across trajectories is bounded by classical measurement-error attenuation,

$$
\begin{array} { r c l } { \rho _ { \mathrm { o b s } } } & { = } & { \rho _ { f } \cdot \frac { \sigma _ { b } ^ { \mathscr { C } } } { \sqrt { ( \sigma _ { b } ^ { \mathscr { C } } ) ^ { 2 } + ( \sigma _ { w } ^ { \mathscr { C } } ) ^ { 2 } } } \cdot \frac { \sigma _ { b } ^ { \mathscr { E } } } { \sqrt { ( \sigma _ { b } ^ { \mathscr { E } } ) ^ { 2 } + ( \sigma _ { w } ^ { \mathscr { E } } ) ^ { 2 } } } } \\ & { \leq } & { 0 . 9 1 ~ \mathrm { ( M H D ) } , \quad 0 . 1 2 ~ ( \mathrm { r a d } . \mathrm { l a y e r } ) . } \end{array}
$$

The radiative-layer measurement matches its ceiling (0.11 at $n { = } 8 0$ , averaged over probed depths), implying $\rho _ { f } \approx$ 1; independently, the MHD calibrator’s residual reduction $\mathrm { ( 0 . 3 4  }$ 0.13 log-units, ≈ 85% of residual variance) back-solves to $\rho _ { f } \approx 0 . 9 6 \substack { - 1 . 0 . \mathrm { A } }$ single model therefore explains both systems: the trajectory factors of C and E are essentially perfectly coupled; what varies is whether the factor is large enough, relative to the per-depth noise ofa single cycle, to be resolved, trivially on MHD, marginally on the radiative layer. The measured MHD fixed-depth Spearman (0.91-0.98 across probed depths, $n { = } 5 0 )$ sits at and slightly above the 0.91 ceiling: the static-factor model treats the within-trajectory fluctuations of C and E as independent, but they share the forward rollout leg and co-move, so the ceiling is conservative where the model is accurate; at $\scriptstyle i = 8 0$ , where error saturation decouples the legs, the measurement (0.85 train / 0.91 val) matches the bound. Consistently, the pointwise $\mu ( \mathcal { C } )$ calibrator trails the identically fit depth-only calibrator by 0.78 nats on the radiative layer’s held-out test trajectories (with the trajectory factor unresolvable, C acts only as a noisy depth proxy) and $\mathcal { C } _ { i }$ retains only a weak within-rollout association there (Spearman $0 . 3 6 \pm \mathrm { { \bar { 0 . 1 3 } } }$ , train). Averaging log $\mathcal { C }$ over depths raises its reliability $( \mathrm { t o } \approx 0 . 9 3$ even on the radiative layer), capping the recoverable trajectory-mean gain there at ≈ +0.11 nats; the measured gain on the nine-trajectory test split is $- 0 . 0 0 8 \pm 0 . 0 0 5 .$ , the nine calibration trajectories are too few to estimate $\mathbf { a } \pm 1 1 \%$ factor, while the same construction earns +0.26 nats on MHD. Incidentally, $\mathit { c } _ { \mathrm { { s } } }$ between-std exceeds $\mathcal { E } \mathrm { { : } }$ on MHD: the round trip compounds forward and backward factors, making it a slightly amplified reading of trajectory dificulty.

## C Detailed Derivations for the Sandwich Bound

Notation, written out. Each latent frame is $\mathbf { z } \in \mathbb { R } ^ { n }$ (the flattened latent). The pair state stacks two consecutive frames, $\mathbf { s } _ { k } = ( \mathbf { z } _ { t + k - 1 } , \mathbf { z } _ { t + k } ) \in \mathbb { R } ^ { 2 n }$ , and $\begin{array} { r } { \| \mathbf { s } \| = \left( \frac { 1 } { 2 n } \| \mathbf { s } \| _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } } \end{array}$ is the RMS over all 2n entries, so that for any pair diference

$$
\begin{array} { r l r } { \left\| \left( a _ { 1 } , a _ { 2 } \right) - \left( b _ { 1 } , b _ { 2 } \right) \right\| ^ { 2 } } & { = } & { \frac { 1 } { 2 n } \left( \| a _ { 1 } - b _ { 1 } \| _ { 2 } ^ { 2 } + \| a _ { 2 } - b _ { 2 } \| _ { 2 } ^ { 2 } \right) } \\ & { = } & { \frac { 1 } { 2 } \big ( \mathrm { M S E } ( a _ { 1 } , b _ { 1 } ) } \\ & { } & { + \mathrm { M S E } ( a _ { 2 } , b _ { 2 } ) \big ) . \qquad ( 1 6 ) } \end{array}
$$

Applied to $( \mathbf { s } _ { 0 } , \tilde { \mathbf { s } } _ { 0 } )$ this is exactly the definition of $\mathcal { C } _ { i }$ in the main paper: the $\frac { 1 } { 2 n }$ normalization is chosen so pair-level and frame-level (MSE) quantities interconvert without stray constants. With a deterministic sampler, one model step is the shift-and-append map on pairs,

$$
\Phi _ { + } ( a , b ) = \bigl ( b , \ : f _ { \theta } ^ { + } ( a , b ) \bigr ) , \qquad \Phi _ { - } ( a , b ) = \bigl ( f _ { \theta } ^ { - } ( a , b ) , \ : a \bigr ) ,\tag{17}
$$

where $f _ { \theta } ^ { \pm }$ is one full DDIM generation conditioned on the pair (and the anchor and scalars, suppressed). The forward rollout is $\hat { \mathbf { s } } _ { i } = \Phi _ { + } ^ { i } ( \mathbf { s } _ { 0 } )$ , whose components we write $\hat { \mathbf { s } } _ { i } =$ $\left( \hat { \mathbf { z } } _ { t + i - 1 } , \hat { \mathbf { z } } _ { t + i } \right)$ ; by (17), the older component of $\hat { \mathbf { s } } _ { i }$ equals the newer component of $\hat { \bf s } _ { i - 1 }$ , both are the same chain’s prediction of frame $t + i - 1$

Lemma 1 (Pair error is the average of two frame errors). With $\mathcal { E } _ { k } = \mathrm { M S E } ( \mathbf { z } _ { t + k } , \hat { \mathbf { z } } _ { t + k } )$ and the convention $\mathcal { E } _ { 0 } : = 0$ (the seed is exact),

$\begin{array} { r } { \mathcal { E } _ { i } ^ { \mathrm { p } } ~ = ~ \frac { 1 } { 2 } \big ( \mathcal { E } _ { i - 1 } + \mathcal { E } _ { i } \big ) } \end{array}$ , hence $\textstyle { \frac { 1 } { 2 } } { \mathcal { E } } _ { i } \ \leq \ { \mathcal { E } } _ { i } ^ { \mathrm { p } }$ and $\mathcal { E } _ { i } ~ \leq ~ 2 \mathcal { E } _ { i } ^ { \mathrm { p } }$ 2 (18) with $\mathcal { E } _ { i } = 2 \mathcal { E } _ { i } ^ { \mathrm { p } }$ if $\mathcal { E } _ { i - 1 } = 0 \mathrm { ~ } ( \mathrm { e . g . , a t ~ } i = 1$ , where the older slot of $ { \hat { \mathbf { s } } } _ { 1 }$ is still the true seed ${ \bf z } _ { t } )$

Proof. $\begin{array} { r l r } { \mathscr { E } _ { i } ^ { \mathrm { p } } ~ = ~ \left\| { \bf s } _ { i } - \hat { \bf s } _ { i } \right\| ^ { 2 } = } & { { } \frac { 1 } { 2 n } \big ( \| { \bf z } _ { t + i - 1 } - \hat { \bf z } _ { t + i - 1 } \| _ { 2 } ^ { 2 } + } \end{array}$ $\begin{array} { r } { \| \mathbf { z } _ { t + i } - \hat { \mathbf { z } } _ { t + i } \| _ { 2 } ^ { 2 } \big ) = \frac { 1 } { 2 } ( \mathcal { E } _ { i - 1 } + \mathcal { E } _ { i } ) } \end{array}$ , using that the older component of $\hat { \mathbf { s } } _ { i }$ is the chain’s f $\mathrm { r a m e } - ( t + i - 1 )$ ) prediction. Dropping the nonnegative $\mathcal { E } _ { i - 1 }$ gives the bounds; the equality condition is immediate. □

Statements recalled from the main paper. For selfcontainedness we restate the objects whose proofs are completed here (numbering refers to the main paper). Assumption 1 posits constants $0 < \mu \le L < \infty$ such that, on the set D containing the two backward trajectories, all pairs $a , b$ satisfy

$$
\mu \| a - b \| \leq \| \Phi _ { - } ( a ) - \Phi _ { - } ( b ) \| \leq L \| a - b \| ,\tag{19}
$$

with the backward residual $\delta _ { i } : = \left\| \Phi _ { - } ^ { i } ( \mathbf { s } _ { i } ) - \mathbf { s } _ { 0 } \right\|$ . Proposition 1 (the sandwich bound) states

$$
\begin{array} { r } { \Big ( \operatorname* { m a x } \big \{ \mu ^ { i } \sqrt { \mathcal E _ { i } ^ { \mathrm { p } } } - \delta _ { i } , 0 \big \} \Big ) ^ { 2 } \ \leq \ \mathcal C _ { i } \ \leq \ \Big ( L ^ { i } \sqrt { \mathcal E _ { i } ^ { \mathrm { p } } } + \delta _ { i } \Big ) ^ { 2 } , } \end{array}\tag{20}
$$

and Corollary 1 (the certificate) states

$$
\sqrt { \mathscr { E } _ { i } } \ \leq \ \sqrt { 2 } \mu ^ { - i } \Big ( \sqrt { \mathscr { C } _ { i } } + \delta _ { i } \Big ) .\tag{21}
$$

The assumption, unpacked. Assumption 1 of the main paper, recalled in (19), constrains the backward step Φ<sub>−</sub> only on the set D of pairs actually visited by the two backward trajectories in play from the true terminal pair and from the predicted one not globally. The upper inequality is ordinary Lipschitz continuity: one backward step cannot amplify a pair discrepancy by more than L. The lower inequality (co-Lipschitz, or expansiveness) says one backward step cannot shrink a discrepancy below a factor $\mu > 0$ . For diferentiable $\Phi _ { - }$ this is equivalent to the singular values of its Jacobian lying in $[ \mu , L ]$ along the visited segment, and it is precisely the anti-cancellation condition: cancellation means two diferent terminal pairs (true and predicted) get mapped backward onto nearly the same returned seed, which is exactly a collapse of distance that $\mu > 0$ forbids. If the learned backward map does collapse $( \mu \to 0 )$ , the certificate weakens honestly: the bound (21) blows up rather than silently failing. The residual $\delta _ { i } ~ = ~ \| \Phi _ { - } ^ { i } ( \mathbf { s } _ { i } ) - \bar { \bf s _ { 0 } } \|$ isolates the backward model’s own inaccuracy on clean inputs: it is what the round-trip would report even for a perfect forward rollout, i.e., the detector’s noise floor.

Lemma 2 (i-step separation propagation). Let $u _ { k } : = \Phi _ { - } ^ { k } ( \mathbf { s } _ { i } )$ and $v _ { k } : = \Phi _ { - } ^ { k } ( \hat { \mathbf { s } } _ { i } )$ for $k = 0 , \ldots , i ,$ , so $u _ { 0 } = \mathbf { s } _ { i } , v _ { 0 } = \hat { \mathbf { s } } _ { i } .$ and $v _ { i } = \tilde { \bf s } _ { 0 }$ . Under the bi-Lipschitz condition (19), for $k =$ $0 , \ldots , i ,$

$$
\mu ^ { k } \left\| \mathbf { s } _ { i } - \hat { \mathbf { s } } _ { i } \right\| \leq \left\| u _ { k } - v _ { k } \right\| \leq L ^ { k } \left\| \mathbf { s } _ { i } - \hat { \mathbf { s } } _ { i } \right\| .\tag{22}
$$

Proof. Induction on k. Base $k \ = \ 0 :$ both sides equal $\| \mathbf { s } _ { i } - \hat { \mathbf { s } } _ { i } \|$ . Step: $u _ { k } , v _ { k } \in \mathcal { D }$ by the definition of $\bar { \mathcal { D } }$ (it contains both backward trajectories), so (19) applies with $a = u _ { k } , b = v _ { k } \colon | | u _ { k + 1 } - \bar { v _ { k + 1 } } | | = | | \Phi _ { - } ( u _ { k } ) - \bar { \Phi _ { - } } ( v _ { k } ) | | \leqslant$ $\left\lceil \mu \ \left\| u _ { k } - v _ { k } \right\| , L \ \left\| u _ { k } - v _ { k } \right\| \right\rceil ;$ multiplying the inductive bounds by $\mu$ and L respectively gives (22) at $k { \mathrm { + } } 1$ □

Proof of Proposition 1 of the main paper, in full. Set $k = i$ in Lemma 2 and abbreviate $P : = \sqrt { \mathcal { E } _ { i } ^ { \mathrm { p } } } = \| \mathbf { s } _ { i } - \hat { \mathbf { s } } _ { i } \| \colon$

$$
\mu ^ { i } P \le \| u _ { i } - v _ { i } \| \le L ^ { i } P .\tag{23}
$$

The quantity of interest is $\sqrt { \mathscr { C } _ { i } } = \| \mathbf { s } _ { 0 } - v _ { i } \|$ (the returned seed’s distance to the true seed). Insert u and use the triangle inequality both ways, together with $\| \mathbf { s } _ { 0 } - u _ { i } \| = \delta _ { i }$ by definition:

$$
\left\| \mathbf { s } _ { 0 } - v _ { i } \right\| \leq \left\| \mathbf { s } _ { 0 } - u _ { i } \right\| + \left\| u _ { i } - v _ { i } \right\| \leq \delta _ { i } + L ^ { i } P ,\tag{24}
$$

$$
\begin{array} { r } { \| \mathbf { s } _ { 0 } - v _ { i } \| \geq \| u _ { i } - v _ { i } \| - \| u _ { i } - \mathbf { s } _ { 0 } \| \geq \mu ^ { i } P - \delta _ { i } . } \end{array}\tag{25}
$$

Since $\| \mathbf { s } _ { 0 } - v _ { i } \| \geq 0$ always, (25) strengthens to $\left\| \mathbf { s } _ { 0 } - v _ { i } \right\| \geq$ max $\{ \stackrel { . . . } { \mu } ^ { i } P - \delta _ { i } , 0 \}$ . Both outer bounds are now nonnegative, so squaring preserves the ordering and yields (20). □

ProofofCorollary 1 ofthe main paper, in full. Inequality (25) before clipping rearranges, for every case (whether or not the max clipped to zero), to

$$
\mu ^ { i } \sqrt { \mathcal { E } _ { i } ^ { \mathrm { p } } } \leq \sqrt { \mathcal { C } _ { i } } + \delta _ { i } \Longrightarrow \sqrt { \mathcal { E } _ { i } ^ { \mathrm { p } } } \leq \mu ^ { - i } \big ( \sqrt { \mathcal { C } _ { i } } + \delta _ { i } \big ) .\tag{26}
$$

Combining with Lemma 1, $\begin{array} { r } { \sqrt { \mathscr { E } _ { i } } \leq \sqrt { 2 \mathscr { E } _ { i } ^ { \mathrm { p } } } = \sqrt { 2 } \sqrt { \mathscr { E } _ { i } ^ { \mathrm { p } } } \leq } \end{array}$ $\sqrt { 2 } \mu ^ { - i } \left( \sqrt { \mathscr { C } _ { i } } + \delta _ { i } \right)$ , which is (21). □

Remarks on the constants. (i) Everything is estimable offline. $\delta _ { i }$ is measured directly on held-out data (backward rollouts from true terminal pairs; reported in the experiments). $\mu$ and L along visited pairs are the extreme singular values of the backward step’s Jacobian, estimable by Jacobian–vector products or finite-diference probes $\| \mathbf { \bar { \Phi } } _ { - } ( a + \epsilon r ) - \Phi _ { - } ( \bar { a ) } \| / ( \epsilon \| r \| )$ over random directions r at pairs a sampled from backward rollouts, these are modelonly quantities requiring no test-time ground truth. See Virmaux and Scaman (2018) for network-wide Lipschitz estimation. (ii) Tightness. When $\delta _ { i } \to 0$ the sandwich pins the ratio $\mathcal { C } _ { i } / \dot { \mathcal { E } } _ { i } ^ { \mathrm { p } }$ into $[ \mu ^ { 2 i } , L ^ { 2 i } ] ;$ ; if additionally $\mu \approx L \approx 1$ (the backward step nearly preserves pair distances along the attractor), $\mathcal { C } _ { i }$ tracks $\mathcal { E } _ { i } ^ { \mathrm { p } }$ almost proportionally, this is the regime the strong empirical correlations suggest the trained models occupy. (iii) Honest looseness. The window $[ \mu ^ { 2 i } , L ^ { 2 i } ]$ widens exponentially in i whenever $\mu < 1 < L ,$ , so the worst-case guarantee degrades with depth even though the empirical relationship stays tight; the calibrator of the main paper is the practical instrument, with (21) explaining when it can be trusted and (20)’s lower branch explaining when it cannot (cancellation, $\mu \to 0 )$ . (iv) Certified constants. Architectures with certified bi-Lipschitz bounds, such as invertible residual networks (Behrmann et al. 2019), ofer a route to guaranteed $( \mu , L )$ rather than estimated ones. (v) Stochastic sampling. With a stochastic sampler, Proposition 1 of the main paper applies verbatim to the deterministic probabilityflow sampler, or to S-averaged cycles $\begin{array} { r } { \bar { \mathcal { C } } _ { i } = \frac { 1 } { S } \sum _ { s } \mathcal { C } _ { i } ^ { ( s ) } } \end{array}$ with the additional sampling-variance term absorbed into $\delta _ { i }$

## D Why Bidirectional Training Can Beat Specialist Training: A Linear Analysis

The ablation in the main paper finds that one bidirectional model outperforms direction specialists on each specialist’s own task at matched compute. (Protocol of the measured ablation, for reference: the identical architecture is trained forward-only, backward-only, and bidirectionally with $p _ { \mathrm { b w d } } { = } 0 . 5$ on the turbulent radiative layer, 80 training trajectories, at matched compute; all three checkpoints are evaluated direction-matched with fixed noise, 10 seeds, mean $\pm ~ \mathrm { S E . } )$ Here we isolate the mechanism in the simplest setting that exhibits it: stationary linear dynamics with a shared linear hypothesis.

Setting. Let $\mathbf { z } _ { t } \in \mathbb { R } ^ { d }$ be a zero-mean stationary Gaussian process with one-step dynamics $\mathbf { z } _ { t + 1 } = A \mathbf { z } _ { t } + \varepsilon _ { t } , \varepsilon _ { t } \sim$ $\bar { \mathcal { N } } ( 0 , Q )$ ) independent of $\mathbf { z } _ { t } .$ , and stationary covariance $\Sigma =$ $\mathbb { E } [ \mathbf { z } _ { t } \mathbf { z } _ { t } ^ { \top } ]$ satisfying $\Sigma = A \bar { \Sigma } A ^ { \top } + Q$

Lemma 3 (Time reversal is a reparameterization, not a new task). The reverse-time conditional is linear with

$$
\begin{array} { r } { \mathbb { E } [ \mathbf { z } _ { t } \mid \mathbf { z } _ { t + 1 } ] = A _ { - } \mathbf { z } _ { t + 1 } , \qquad A _ { - } = \Sigma A ^ { \top } \Sigma ^ { - 1 } , } \end{array}\tag{27}
$$

and residual covariance $Q _ { - } = \Sigma - \Sigma A ^ { \top } \Sigma ^ { - 1 } A \Sigma \succeq 0$ . In particular, for whitened latents $( \Sigma = I$ , which per-channel standardization approximately enforces), $A _ { - } = \overline { { A } } ^ { \top }$ exactly: the forward and backward prediction tasks share one parameter object, and a single network queried with a direction flag is a correctly specified weight tying.

Proof. $\left( \mathbf { z } _ { t } , \mathbf { z } _ { t + 1 } \right)$ is jointly Gaussian with $\mathrm { C o v } ( { \bf z } _ { t + 1 } , { \bf z } _ { t } ) =$ AΣ, hence $\begin{array} { r l r } { \mathrm { C o v } ( { \mathbf z } _ { t } , { \mathbf z } _ { t + 1 } ) } & { { } = } & { \Sigma A ^ { \top } } \end{array}$ The Gaussian conditional-mean formula gives $\begin{array} { r l r l } { \mathbb { E } [ \mathbf { z } _ { t } } & { { } | } & { \mathbf { z } _ { t + 1 } ] } & { { } = } \end{array}$ $\mathrm { C o v } ( \mathbf { z } _ { t } , \mathbf { z } _ { t + 1 } ) \mathrm { C o v } ( \mathbf { z } _ { t + 1 } ) ^ { - 1 } \mathbf { z } _ { t + 1 } = \Sigma \tilde { A } ^ { \top } \Sigma ^ { - 1 } \mathbf { z } _ { t + 1 }$ , and the conditional covariance is $\begin{array} { r } { \Sigma { - } \Sigma \dot { A } ^ { \top } \Sigma ^ { - 1 } A \Sigma } \end{array}$ , positive semidefinite as a Gaussian conditional covariance. With $\Sigma = I$ the stationarity identity reads $A A ^ { \top } + Q = I _ { \mathrm { { i } } }$ , so all singular values of A are at most one and $Q _ { - } ~ = ~ I - A ^ { \top } A ~ \succeq ~ 0$ directly. □

Estimation with and without tying. Lemma 3 converts the training question into: the same matrix A is observable through two regression views: forward pairs, in which it acts as A, and backward pairs, in which it acts as $A ^ { \top }$ (taking $\Sigma =$ I). A direction specialist uses one view; the bidirectional model, both. We compare them in the standard fixed-design idealization.

Proposition 2 (Tying halves the estimation variance). Suppose A is estimated by least squares from m forward observations $y _ { k } = A x _ { k } + \varepsilon _ { k }$ and m backward observations $v _ { k } = A ^ { \top } u _ { k } + \varepsilon _ { k } ^ { \prime } ,$ , with $( \mathbb { E } [ x x ^ { \top } ] = \mathbb { E } [ u u ^ { \top } ] = I )$ and independent noise of equal scale $\boldsymbol { \sigma } ^ { 2 }$ per coordinate. Then the expected squared estimation error obeys

$$
\mathbb { E } \Vert \hat { A } - A \Vert _ { F } ^ { 2 } = \frac { \sigma ^ { 2 } d ^ { 2 } } { m _ { + } + m _ { - } } \big ( 1 + o ( 1 ) \big ) .\tag{28}
$$

At matched total sample budget m, the specialist $( m _ { + } =$ m, $m _ { - } = 0 )$ attains $\stackrel { \cdot } { \sigma } ^ { 2 } d ^ { 2 } / \stackrel { \cdot } { m } ,$ while the tied bidirectional estimator $( m _ { + } = m _ { - } = m / 2 )$ attains the same $\sigma ^ { 2 } d ^ { 2 } / m$ on the union of views, but per parameter of the deployed system it has solved both directions with one $d \times d$ matrix, where the specialists require two. Equivalently, at matched per-direction data n (each trajectory supplies both views), the tied estimator’s error is $\sigma ^ { 2 } d ^ { 2 } / ( 2 \dot { n } )$ : half the specialist’s $\sigma ^ { 2 } d ^ { 2 } / n$

Proof. Vectorize: each forward observation contributes d scalar equations in the entries of A with isotropic regressors; each backward observation contributes d scalar equations in the entries of $A ^ { \top } , \mathrm { i . e . }$ e., in the same unknowns under a fixed permutation. Stacking gives a least-squares problem for $\mathrm { v e c } ( A ) \in \mathbb { R } ^ { d ^ { 2 } }$ with $( m _ { + } + m _ { - } ) d$ equations and asymptotically isotropic design; the standard OLS risk $\sigma ^ { 2 } \times ( \mathbf { \dot { \# } p a r a m s } ) /$ (#obs rows) $\times d = \sigma ^ { 2 } d ^ { 2 } / ( m _ { + } + m _ { - } )$ follows. The two readings of the budget give the two displayed comparisons. □

Remarks. (i) Where the measured efect lives. The gain is a variance reduction, so it scales as $1 / n$ and vanishes with abundant data. The mechanism predicts the benefit is largest in the low-data regime, which is exactly where the ablation measures it (80 trajectories), and variance reduction is regularization, matching the observed delayed overfitting (epoch ∼250 vs. ∼180). (ii) Why the idealization overstates the constant. The two views of a trajectory share its underlying randomness, so the independent-noise factor of two is an upper bound on the mechanism; the measured 7-10% perdirection improvement is consistent with partially redundant views plus optimization efects. (iii) Of-direction failure of specialists. The converse is immediate: a forward specialist’s objective never constrains the transpose view, so nothing anchors its behavior under the backward query, which is consistent with the $5 { - } 7 \times$ of-direction degradation measured. (iv) Beyond linear. For the nonlinear denoiser the tying acts through a shared representation rather than a shared matrix; multitask representation bounds (Baxter 2000; Maurer, Pontil, and Romera-Paredes 2016) give the analogous complexity-per-task reduction when tasks share structure, which Lemma 3 identifies here as near-exact rather than merely assumed. What is specific to our setting is that the same tying that improves the forward model is the structure that makes the round-trip meter of the main paper available at all.

Table 5: Per-field calibration of decoded reconstruction errors on held-out trajectories. ∆NLL: log-likelihood gain over the identically-fit depth-only calibrator (nats); other columns as in Table 3 of the main paper.
<table><tr><td>Field</td><td>∆NLL↑</td><td> $\times _ { 6 8 }$ </td><td> $\times { } _ { 9 5 }$ </td><td> $\mathrm { C o v _ { 6 8 / 9 5 } }$  (%) MCA↓</td></tr><tr><td> $\rho$ </td><td>+0.73</td><td>1.18</td><td>1.39 63.6</td><td>92.5 0.022</td></tr><tr><td> $v _ { x }$ </td><td>+0.73</td><td>1.17</td><td>1.37 65.1</td><td>93.1 0.058</td></tr><tr><td> $v _ { y }$ </td><td>+0.72</td><td>1.19</td><td>1.38 63.0</td><td>93.7 0.040</td></tr><tr><td> $P$ </td><td>+0.67</td><td>1.30</td><td>1.64 67.2</td><td>/95.1 0.042</td></tr><tr><td> $B _ { x }$ </td><td>+0.32</td><td>1.27</td><td>1.66 74.3</td><td>/96.1 0.024</td></tr><tr><td> $B _ { y }$ </td><td>+0.36</td><td>1.26</td><td>1.57 63.9</td><td>/92.0 0.059</td></tr></table>

## E Additional Results and Protocol Details

Per-field calibration ofdecoded MHD errors. Table 5 accompanies the “Transfer to decoded field-space errors” paragraph of the main paper. Per-field polynomial degrees are selected by the same trajectory-level cross-validation protocol as the latent calibrator, additionally restricted to monotone candidates. The gains are largest for the hydrodynamic quantities $( \rho , v _ { x } , v _ { y } , P ; + 0 . 6 7 – + 0 . 7 3$ nats) and smaller for the magnetic components $( B _ { x } , B _ { y } ; + 0 . 3 2 – + 0 . 3 6 )$ , whose advantage over the depth-only baseline concentrates at deeper rollouts $( i ~ \geq ~ 4 0 )$ ; pressure is the hardest field in absolute terms.

MHD calibration: safeguards, depth-conditional behavior, and a trajectory-mean augmentation. Two safeguards apply to every calibrator fit reported in the main paper: the mean function $\mu$ is verified monotone on the data range, and inputs are clamped to the training support. While pooled calibration of the $\mu ( \mathcal { C } _ { i } )$ calibrator is near-nominal, depthconditional calibration drifts at the deepest rollouts: the perdepth miscalibration area reaches 0.27 at $i \geq 4 0$ , against 0.013 pooled, with the same drift recurring per decoded field, this is the regime where the certificate’s $\mu ^ { - i }$ factor predicts the signal loosens. Augmenting the calibrator with the trajectory mean of log C and the depth (five additional coeficients, same fitting recipe) yields a further +0.26 nats (paired per-trajectory, ±0.02) and largely repairs the middepth calibration drift (miscalibration area $0 . 2 3  0 . 0 9$ at i=40) while leaving shallow-depth calibration intact; adding temporal-shape features of the C-curve (early-window level, slope) brings no further gain $( + 0 . 0 1 \pm 0 . 0 2 ) $ , indicating that per-depth $\mathcal { C } _ { i }$ values act as exchangeable measurements of a trajectory-level quality factor rather than carrying timing information. Because the trajectory mean reads the full curve, this variant suits post-hoc assessment of a completed rollout (a prefix mean provides a causal analogue); residual drift at the deepest depths $( 0 . 2 7 \to 0 . 2 2 \mathrm { a t } i = 8 0 )$ is where per-depth recalibration or a split-conformal wrapper (Angelopoulos and Bates 2023) would restore finite-sample coverage, which we leave to future work.

LE-PDE-UQ: training additions and model selection. Because their baseline trains with a weighted four-step unrolled objective while ours is single-step, we disclose two training additions: Gaussian noise on context and anchor frames, and pushforward-style context augmentation (Brandstetter, Worrall, and Welling 2022) (1–3 unrolled no-gradient steps supply model-generated contexts; targets remain the true next frames), applied as a short low-learning-rate finetune. Checkpoints are selected by validation rollout error rather than the validation denoising loss: the ϵ-prediction objective is a teacher-forced, single-step surrogate that reflects neither the iterated sampler nor robustness to self-generated contexts, and in our runs the two anti-correlate: the ϵ-selected checkpoint is the worst rollout model in the sweep.

LE-PDE-UQ: calibration protocol. All fitted uncertainty objects including the heteroscedastic calibrators, the depthonly baseline tables, pixel-scale factors, and shape-mixing weights λ, are fit on rollouts from held-out calibration trajectories (drawn from the validation split), never on the surrogate’s training trajectories. This departs from the MHD calibration of the main paper, where fitting on training rollouts suficed, for a measured reason: here the surrogate’s held-out rollout error exceeds its training-trajectory rollout error by a factor of 4.8 on average (up to 6.3 at depth 10), so training-set rollouts under-represent deployment error and calibrators fit on them transfer systematically undersized uncertainty; calibration data must resemble the data being calibrated (Kuleshov, Fenner, and Ermon 2018; Angelopoulos and Bates 2023). During development we fit on one interleaved half of the validation trajectories and report on the other; final numbers are produced once, fitting on the full validation split and evaluating on the untouched test split. Model selection uses validation rollout error and is disjoint from calibration fitting.

LE-PDE-UQ: pixel z-score diagnostics and latent-space results. The residual gap between the best training-free pixel calibration of the main paper and learned per-pixel uncertainty is within-frame heteroscedasticity that no framelevel scalar represents: pixel z-scores under the composed calibrator are over-concentrated $\mathrm { a t } \pm 1 \sigma ( P ( | z | < 1 ) \approx 0$ .80 vs. nominal 0.68) with heavy tails (kurtosis ${ \approx } 1 0 )$ . The perframe scale is right on average while extreme pixels exceed Gaussian tails. In the latent space where $\mathcal { C } _ { i }$ is computed, the same run completes the cross-system spectrum of the main paper: $\mathcal { C } _ { i }$ and $\mathcal { E } _ { i }$ are rank-correlated within trajectories (Spearman $0 . 8 8 \pm 0 . 1 7$ , test) and across trajectories at fixed depth (0.32-0.59), and the identically fit latent calibrator improves pooled test NLL by +0.04 nats over depth-only, the gain concentrated at shallow depths (NLL 0.06 vs. 0.16 at $i { = } 1 ; 0 . 3 4$ vs. 0.48 at $i { = } 4 )$ and gone by i=10 as errors approach the attractor scale.

## F Inverse Rollouts

This section collects the evidence behind the “Inverse Rollouts” subsection of the main paper. All results use the single trained bidirectional model with the direction flag set to $c _ { d } = - 1$ and the deterministic sampler; no additional training or fine-tuning is involved, and one backward step costs the same as one forward step. Backward rollouts are seeded with the true terminal pair $\left( \mathbf { z } _ { T - 1 } , \mathbf { z } _ { T } \right)$ of a held-out test trajectory and iterated toward $t { = } 0 ;$ these are exactly the rollouts whose residual $\delta _ { i }$ defines the noise floor of the sandwich bound (Sec. C), so the reconstructions of Fig. 6 visualize what $\delta _ { i }$ measures. For the direction-symmetric consistency check of Fig. 19, a backward rollout of depth i is cycledforward for i steps and compared with its own terminal seed pair, yielding the mirror-image round-trip error $\mathcal { C } _ { i } ^ { - }$ : the defining formula of the main paper with the roles of $\Phi _ { + }$ and Φ exchanged. Fig. 8 shows the same forward rollout in the latent space where all metrics are computed and Fig. 9 shows the backward rollout.

LE-PDE-UQ: additional details. The LE-PDE-UQ system evolves according to a set of 2D Navier-Stokes equations governing a viscous, incompressible fluid in vorticity form in a unit torus:

$$
\begin{array} { r l r } { \displaystyle \frac { \partial w ( t , x ) } { \partial t } } & { = } & { - u ( t , x ) \cdot \nabla w ( t , x ) + \nu \Delta w ( t , x ) + f ( x ) , } \\ { \nabla \cdot u ( t , x ) } & { = } & { 0 , } \\ { w ( 0 , x ) } & { = } & { w _ { 0 } ( x ) , \ x \in ( 0 , 1 ) ^ { 2 } , \ t \in [ 0 , T ] \qquad ( 2 9 ) } \end{array}
$$

where $w ( t , x ) = \nabla \times u ( t , x )$ is the vorticity, $\nu \in \mathbb { R } _ { + }$ is the viscosity coeficient, the domain is discretized into a 64 × 64 grid and $R e = 1 0 ^ { 4 }$ (turbulent).

Additional CELEBV-HQ Rollouts Additional forward autoregressive rollouts of CELEBV-HQ data, using just the first 2 images in a video, for held out test data, are shown in Figures 12, 13.

Additional Well Turbulent Flow Rollouts Figures 15-18 show forward and backward rollouts of all 9 test trajectories of The Well turbulent flow data. An asymmetry is visible here in that after approximately 50 steps, it is easier for the model to go backwards in time rather than forwards because the initial conditions are very similar for all examples, while the final conditions vary widely.

![](images/601203c7b0dc91cf7a529e061678e8c1854a2ead9d29c70b55850442047de284.jpg)  
Figure 6: Physical-field inverse rollouts. Backward rollout $( c _ { d } \mathrm { = } - 1 )$ of a held-out MHD test trajectory seeded with only the true terminal pair $\left( { \bf z } _ { T - 1 } , { \bf z } _ { T } \right)$ : VAE-decoded backward reconstructions vs. ground truth at increasing backward depth i (top to bottom, left 6 columns), with pixelwise absolute error beneath each pair. Reconstructions stay sharp and physically coherent at shallow-to-moderate backward depths and degrade gracefully as backward error accumulates, mirroring the forward direction. The forward rollout (bottom to top) is shown in the right 6 columns for all 6 MHD fields.

![](images/a8fd6907e02ca6e2d52b9c794f4b9411aced46fa760092e07c2c35b56cd634b4.jpg)

![](images/09c53abe8ced7609fb54d8d1c004ffd9b627840d8cc5a6581c0843367e84c971.jpg)  
Figure 7: Predicting unknown latent rollout error: Round-trip consistency forward or backward in time. Forward rollouts cycled backward (the default $\mathcal { C } _ { i }$ , left) and backward rollouts cycled forward $( \mathcal { C } _ { i } ^ { - }$ , right), each plotted against the corresponding true rollout error on the 50 held-out MHD validation trajectories, colored by rollout depth. The backward cycle tracks backward error when the model is deployed as an inverse solver.

![](images/7afcfe3f88d6bc86639cbbb9bd0c4fd3fc197d96bc321c07e477e32947897f98.jpg)  
Figure 8: Latent rollouts, forward. Latent trajectories of a held-out MHD test trajectory in the 16×16×4 per-field latent space where $\mathcal { C } _ { i } , \mathcal { E } _ { i } ,$ and $\delta _ { i }$ are computed: encoded ground truth (top of each row), forward rollout from the initial pair (from left to right), shown at matched times. Drift away from the encoded truth grows with depth — the latent-level picture behind the decoded fields of Fig. 6.

![](images/5c3198f703b6ea0b30f81ada764faf2240c89f5cfa09cd3c49f64ae246bfbcf9.jpg)  
Figure 9: Latent rollouts, backward. Latent trajectories of a held-out MHD test trajectory in the 16×16×4 per-field latent space where $\mathcal { C } _ { i } , \mathcal { E } _ { i }$ , and $\delta _ { i }$ are computed: encoded ground truth (top of each row), backward rollout from the terminal pair (from right to left), shown at matched times. Drift away from the encoded truth grows with depth — the latent-level picture behind the decoded fields of Fig. 6.

![](images/e6011b5af14e0cfc9f4610e756f921bf665951c20345f15afc72b6954174ab67.jpg)  
Figure 10: Forward-rollout error on the Navier–Stokes benchmark: mean relative $L _ { 2 }$ vs. rollout depth with the interquartile range across trajectories. Growth is geometric, $e _ { d } \approx \dot { e } _ { 1 } r ^ { d - 1 }$ with $r \approx 1 . 2 1$ per step (dashed).

![](images/8a8ad0695df692df7a0489ac7b0b90d515f8677c9967cb764f809b4fb85329db.jpg)  
Figure 11: The color bands show predicted $\mu \pm 2 \sigma$ of $\mathcal { E } _ { i }$ as a function of $\mathcal { C } _ { i } .$ , which was fit on validation data, relative to the actual measurements of $( \mathcal { C } _ { i } , \mathcal { E } _ { i } )$ of the test data for the LE-PDE-UQ test dataset colored by rollout distance.

![](images/ae6bab494bc7f8a06ecf662e8a3f7cabba108469d9881a15371960cc6001cf72.jpg)

Figure 12: CELEBV-HQ rollouts show with error and C values. While for such a multi-modal dataset it is impossible for the model to predict rollouts for test data, it is able to flag, in a self-supervised way, OOD data with increased C levels.

![](images/f17af474ae680096514a7957e2494014ccb079c8d809cb9725666153cbbbd577.jpg)

![](images/7f4c3121337c2a247dee34bc34b52a278d6ce492cc031af641c4f68b67f9d6d2.jpg)  
Figure 14: Forward rollout 99 steps for one test trajectory.

![](images/2abfa38f054bcbe46d0e1730f0be6f34332841195bd0e27422fd1536ff37c972.jpg)  
Figure 15: Backward rollout 99 steps for one test trajectory.

![](images/fb3200b961f3e274ac49ecac1bd9f756c9e327022ff7e6f0515cebac7af6840e.jpg)  
Figure 16: Comparing forward and backward rollout of the denisity fields for 99 steps for test trajectories 1, 2, and 3. For each of the 3 images, the 1st row shows forward autoregressive generation using $c _ { d } = + 1$ , the 2nd row shows the absolute diference between forward generated density and the true density, the 3rd row shows the true density at each time step, the 4th row shows the absolute diference between backward generated density and the true density, and finally the 5th row shows the backward autoregressive generation using $c _ { d } = - 1$

![](images/2f309aac5e64fc8b5d1c09d87d0c6f174ad14b54aaf922c3c790b9a113bc480f.jpg)  
Figure 17: Comparing forward and backward rollout of the denisity fields for 99 steps for test trajectories 4, 5, and 6. For each of the 3 images, the 1st row shows forward autoregressive generation using $c _ { d } = + 1$ , the 2nd row shows the absolute diference between forward generated density and the true density, the 3rd row shows the true density at each time step, the 4th row shows the absolute diference between backward generated density and the true density, and finally the 5th row shows the backward autoregressive generation using $c _ { d } = - 1$

![](images/a598eb0b09359a6ee8b2db4a0484b39dfc33c40f381966daaedf189115122ba4.jpg)  
Figure 18: Comparing forward and backward rollout of the denisity fields for 99 steps for test trajectories 7, 8, and 9. For each of the 3 images, the 1st row shows forward autoregressive generation using $c _ { d } = + 1$ , the 2nd row shows the absolute diference between forward generated density and the true density, the 3rd row shows the true density at each time step, the 4th row shows the absolute diference between backward generated density and the true density, and finally the 5th row shows the backward autoregressive generation using $c _ { d } = - 1$

![](images/65d5491aef8ec4b8d5674a77812c2491b787c677f822f2295c04a8f9aa174550.jpg)  
Figure 19: MHD forward (top) and backward (bottom) autoregressive rollout error accumulation.

![](images/1a15d1ccb830af56ad1b94e9522c661738b0bba678878832ac92986f8e8ece35.jpg)

![](images/30818ed3b8a3fb77d2926ba8b65c793efc7716675d613dbd0a1feaba3876f58f.jpg)  
Figure 20: Round-trip consistency forward or backward in time. Forward rollouts cycled backward (the default $\mathcal { C } _ { i } ,$ left) and backward rollouts cycled forward (the mirror image $\mathcal { C } _ { i } ^ { - } , \mathrm { r i g h t } )$ , each plotted against the corresponding true rollout error on the 50 held-out MHD validation trajectories, colored by steo number. The backward cycle tracks backward error when the model is deployed as an inverse solver. The x-axis of the backward rollouts is reversed.