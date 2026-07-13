                                         Imagined Rollouts Are Kinematic, Not Dynamic:
                                        A Diagnosis of Long-Horizon World-Model Failure
                                                               Finn Rasmus Schäfer1 , Korbinian Moller1 , Yuan Gao1 , Christian Oefinger1 ,
                                                                               Sebastian Schmidt2 and Johannes Betz1
                                                                                          1
                                                                                          Autonomous Vehicle Systems Lab
                                                                         Technical University of Munich, Garching b. München, Germany
                                                                                            Email: finn.schaefer@tum.de
                                                                                   2
                                                                                     Data Analytics and Machine Learning Group
                                                                         Technical University of Munich, Garching b. München, Germany


                                           Abstract—Long-horizon failure in world models is convention-           Central Claim
                                        ally attributed to compounding error, a generic framing that
                                        does not distinguish what kind of error compounds. We propose             Current world models imagine kinematically rather
                                        a kinematic-vs-dynamic reframing: world models tend to imagine            than dynamically: they extrapolate position-velocity-
                                        kinematically rather than dynamically. We operationalize this as          acceleration trajectories that are internally consistent with
                                        the imagined Kinematic-Consistency Error, a per-step diagnostic           linear kinematic update rules, but inconsistent with the




arXiv:2607.05966v1 [cs.RO] 7 Jul 2026
                                        that measures how far a rollout departs from a closed-form                physical constraints that produce real motion.
                                        kinematic null, paired with a perturbation protocol that tests
                                        whether iKCE responds when physical conditions cross a regime
                                        boundary. We instantiate the diagnostic on a released DreamerV3            Kinematic fallback is a third, structurally distinct account of
                                        checkpoint trained on DMC walker-walk, where imagined iKCE
                                        runs roughly two orders of magnitude above that of matched              long-horizon world-model failure, alongside the two positions
                                        real-physics rollouts. Across a friction sweep that crosses the gait-   that dominate the literature: predictable-representation engi-
                                        collapse boundary, the model’s iKCE stays statistically flat even       neering (the Dreamer line [5]) and error-compounding bounds
                                        as the trained policy’s reward collapses through the same range,        (MBPO [8]). Predictable-representation engineering attributes
                                        providing the kinematic-not-dynamic signature. The diagnostic           long-horizon reliability to stable, predictable latent represen-
                                        distinguishes kinematic from dynamic imagination at horizons
                                        longer than the embodiment’s gait period.                               tations and pursues it through normalization and balancing
                                                                                                                techniques; a single algorithm with fixed hyperparameters
                                                                                                                across 150+ tasks empirically validates the position, yet it
                                                                                                                is silent on what those representations should contain. Error-
                                                                                                                compounding bounds derive an explicit quadratic-in-horizon
                                                                I. Introduction                                 bound on the gap between model-based and true returns under
                                                                                                                policy distribution shift, and respond pragmatically by limiting
                                                                                                                model trust to short branched rollouts from real states. A world
                                           World models have become a load-bearing component of                 model whose latent contains rich kinematic features but no
                                        recent embodied AI, serving as latent simulators for plan-              dynamic features satisfies both: stable enough for Dreamer-
                                        ning [3, 4, 5] and as generative environments for self-                 line predictability, accurate enough inside the training regime
                                        supervised learning [6, 7]. A widely-noted failure mode is the          for Janner-line short-rollout bounds, yet still biased toward
                                        deterioration of imagined rollouts over long horizons, conven-          kinematic continuation once conditioning pushes its rollouts
                                        tionally attributed to compounding error [8]. This framing is           across a physical-regime boundary. The three accounts are
                                        accurate but underspecified: it does not distinguish what kind          therefore not mutually exclusive but different layers of the
                                        of error compounds or which feature dimensions deteriorate.             same failure surface; because they predict distinct empirical
                                        Across four recent Observations in driving VLM/VLA and                  signatures, the protocol of Section III can target the kinematic-
                                        trajectory-prediction benchmarks indicate that the deterioration        fallback layer specifically.
                                        carries a specific structural signature that the compounding-              We make three contributions. We (i) recast long-horizon
                                        error framing obscures.                                                 world-model failure in kinematic-vs-dynamic terms, dis-
                                           We adopt the classical mechanics distinction between kine-           tinguishing a structural-content layer from the variance-
                                        matic and dynamic motion. We define kinematic as motion                 engineering and error-compounding layers studied in prior
                                        described purely through position, velocity, and acceleration           work; (ii) introduce imagined kinematic-consistency error
                                        time series, without invoking the forces or physical constraints        (iKCE) together with a conditioning-perturbation protocol
                                        that produced it. As dynamic, we define motion that requires            that operationalize this account as a falsifiable evaluation
                                        those constraints (e.g., mass, friction, contact) to be repro-          diagnostic; and (iii) instantiate the diagnostic on an open-
                                        duced correctly.                                                        weight checkpoint (DreamerV3 on DMC walker-walk), where
it exhibits both predicted signatures of kinematic imagination,   planners on this benchmark are not doing substantially more
with controls ruling out the principal confounds. The two         than kinematic extrapolation.
signatures are a kinematic-null residual ∼180× above matched         (iv) Physics-consistency scoring on fine-tuned VLAs. Gao
physics at T =16, and statistical invariance of the imagined      et al. [2] introduce a Kinematic Consistency Error (KCE) that
rollouts to a friction sweep that crosses the empirical gait-     scores a trajectory by checking each predicted next position
collapse boundary. The diagnostic signature is this regime-       against a closed-form kinematic extrapolation of the current
invariance, not the absolute iKCE magnitude: a trivially kine-    state, and reuse it as a training loss for a fine-tuned 4B VLA.
matic predictor would produce zero iKCE.                          On their style-conditioned benchmark, KCE shows no mono-
                                                                  tonic relationship to model scale or modality: the strongest
                        II. Evidence                              generalist (Gemini-3-Pro, 0.06–0.11 m) and the smaller fine-
   Our diagnosis is motivated by four existing observations,      tuned models (0.08–0.12 m) overlap, with no ordering by
each inconclusive alone and explained only in isolation by        parameter count or sensor richness. A data/training deficit,
its original authors, but jointly forming a coherent structural   not a capacity one. A deficit that is invariant to model
signature.                                                        scale and modality is unlikely to reflect a capacity limitation.
   (i) Representational diagnostic on driving VLMs and            It points instead to the training signal and data distribution
VLAs. Schäfer et al. [11] present EgoDyn-Bench, a video-         rather than to model size.
QA diagnostic that decouples physical reasoning from visual          Conclusion. The four observations are concentrated in the
perception. (i), the weighted physics consistency rate (WPCR)     driving and VLA setting but employ heterogeneous method-
saturates with a single static frame: rising from ∼20 with no     ological registers, from representational probing to behavioral
visual input to ∼97 with one frame and remaining essentially      perturbation to physics-consistency scoring, and converge on
flat as additional frames are added or temporally shuffled.       a single structural deficit: the learned representation is domi-
(ii), reintroducing video to a text-only baseline recovers only   nated by kinematic features, and the dynamic features required
∼2.6pp on balanced accuracy under the best encoding, while        for physical-regime-conditional behavior are systematically
text-only input already achieves 59.6% BAcc. The authors          under-represented.
characterize this as a “functional decoupling between vision
                                                                                    III. Diagnostic Protocol
and language”: ego-motion understanding is derived almost ex-
clusively from the language modality, with visual observations    A. iKCE: Imagined Kinematic Consistency Error
Implication for world models is structural. Contributing          Definition (Imagined Kinematic-Consistency Error). For an
static context rather than temporal evidence. Imagined rollouts                          T
                                                                                      t }t=0 produced by a world model, with
                                                                  imagined rollout {x̂WM
that depend on such encoders for motion-conditional features      x̂t a chosen kinematic state vector (e.g. [x, y, v, a, θ]⊤ ), the
extrapolate from representations that under-encode the tempo-     imagined kinematic-consistency error is
ral dynamics they would need to imagine correctly.                                        T −1
   (ii) Sensor-degraded behavioral diagnostic. Priyadershi                       . 1 X                              2
                                                                                           t+1 − kin x̂t
                                                                                         x̂WM          WM
                                                                            iKCE =                                       ,     (1)
and Frtunikj [10] stress-test Alpamayo R1, a 10B-parameter                         T t=0
driving VLA, across 1,996 scenarios under eight sensor
                                                                  where kin(·) is any closed-form kinematic predictor (e.g.,
perturbations. Under heavy Gaussian noise (σ = 70), the
                                                                  constant-velocity or constant-acceleration) chosen to match the
authors characterize the failure mode as one where the tra-
                                                                  WM’s underlying embodiment and output space.
jectory decoder “fails via collapsing kinematic priors while
the language branch continues producing coherent but safety-         Equation 1 follows the mathematical form of the kinematic-
irrelevant explanations.” Independent observation. This is        consistency loss in Gao et al. [2], which supervises a fine-tuned
an independent observation of the same kinematic-fallback         VLA at training time. We repurpose it as a test-time diagnostic
failure mode, in a third-party VLA under naturalistic sensor      on imagined rollouts (the prefixed “i” denotes imagined),
degradation rather than controlled diagnostic stimuli.            measuring kinematic inconsistency rather than reducing it. We
   (iii) Open-loop trajectory-prediction baselines. Zhai et al.   repurpose it as an evaluation diagnostic: applied at test time
[13] train a 3-layer MLP that consumes only the ego vehi-         to imagined rollouts, iKCE measures how far each predicted
cle’s kinematic state and matches perception-based end-to-        next state departs from the kinematic extrapolation of its
end planners on nuScenes open-loop L2 (0.29 m vs. 0.37 m          predecessor.
for VAD-Base), with no camera, LiDAR, or HD-map input.               Counter-intuitively, low iKCE does not indicate dynamic
The authors read this as a benchmark artifact, attributing it     imagination: a world model with near-zero iKCE predicts,
to the trajectory distribution of nuScenes and to a coarse        by construction, next states that coincide with the kine-
collision-evaluation grid, and call for rethinking the open-      matic continuation of their predecessors: it imagines kine-
loop evaluation scheme. We accept the empirical finding but       matically. The signature of dynamic imagination is the op-
read its significance differently. Third independent signature.   posite: iKCE positive, growing with horizon, and responsive
We read the same result as a third independent signature          to physical-regime conditioning (friction transients, contact
of kinematic imagination: when an ego-state-only predictor        events, regime-boundary crossings). iKCE is therefore neces-
saturates the dominant open-loop metric, perception-based         sary but not sufficient to certify dynamic imagination.
                                                                                                       TABLE I
B. Conditioning perturbations                                            iKCE on the (z, vz ) view at µ=1.0 baseline, K=20 rollouts, 95%
   Static iKCE measures internal kinematic self-consistency               bootstrap CIs. WM iKCE exceeds physics by at least an order of
                                                                       magnitude at both measurement horizons. The ratio narrows at T =64
along a single rollout. To turn this into a diagnostic that            due to per-step dilution from the WM’s smooth long-horizon tail (see
separates kinematic from dynamic imagination, we drive iKCE                                      §V, limitation (iii)).
through a dose-response curve over the conditioning state. For
each base rollout, the world model generates K imagined roll-           Source                           Mean iKCE           95% CI
outs under controlled perturbations of physically meaningful            Horizon T =16
conditioning parameters, initial velocity v0 , friction coefficient     Real physics (matched policy)     4.2×10−5       [3.2, 5.3]×10−5
µ, or lateral-acceleration limit alat,max for driving, terrain          DreamerV3 WM (imagined)           7.7×10−3      [5.2, 10.3]×10−3
compliance, or payload for legged locomotion, analogous to              Ratio (WM / physics)               ∼ 180×               —
physically-grounded parameters for other embodiments. The               Horizon T =64
perturbation set is embodiment-specific. The protocol is not.           Real physics (matched policy)     8.6×10−5      [6.4, 11.0]×10−5
   Two diagnostic signatures emerge from the resulting                  DreamerV3 WM (imagined)           2.6×10−3       [2.0, 3.2]×10−3
{iKCEk }K                                                                                                   ∼ 30×
           k=1 ensemble. First, the shape of iKCE(∥∆∥) as
                                                                        Ratio (WM / physics)                                    —
a function of perturbation magnitude: a kinematic imaginer
produces a curve that scales smoothly with ∥∆∥ regardless
of physical regime, because it is extrapolating the same lin-          anchor in Appendix A4 for the analytic lower bound). The
ear update structure in every case. iKCE measures per-step             WM’s imagination carries a substantial residual against any
kinematic-null deviation. Physical-regime perturbations whose          constant-velocity null, with the H1 being a lower bound on
effects are slow relative to the per-step timescale (e.g., friction-   the magnitude separation.
driven slipping in legged locomotion, which accumulates over
                                                                          Hypothesis 2 (H2): WM imagination is friction-invariant
multiple footfalls) are not visible at horizons shorter than
                                                                       across a regime boundary. We sweep surface friction across
their characteristic accumulation time. The diagnostic protocol
                                                                       13 magnitudes in [0.1, 1.7], spanning the regime boundary at
should be applied at horizons longer than the embodiment’s
                                                                       µ=0.20 where the trained gait first drops below 50% of base-
gait period: ∼25 ms × 64 steps ≈ 1.6 s is sufficient
                                                                       line episodic reward (empirically determined, see Fig. 1). For
for walker-class locomotion. Driving trajectories are usually
                                                                       each µ, we compute iKCE on (a) real-physics rollouts under
planned over longer horizons. Second, rollout-pair monotone
                                                                       the trained actor and (b) WM-imagined rollouts conditioned
consistency: physics predicts monotone responses to specific
                                                                       on the first 5 perturbed observations.
perturbation pairs (higher initial velocity implies longer stop-
ping distance under braking; heavier payload implies slower               At T =64, WM iKCE across the sweep is statistically flat
acceleration), and for each such pair (∆a , ∆b ) we check              (max/min spread 1.32×, 95% CIs overlap at every µ, see
whether the imagined rollouts respect the monotonicity. A              Fig. 2). A log-log regression on the same sweep makes this
kinematic world model trivially respects linear monotonicities         falsifiable (Appendix A): the WM slope’s 95% bootstrap CI
but fails physical-regime-conditional ones, where the mono-            (βWM = −0.009, [−0.096, +0.082]) contains zero, while the
tonicity only holds above or below a physical threshold.               physics slope’s (βphys = −0.220, [−0.301, −0.142]) does not.
   This is the closest defensible analog to EgoDyn-Bench’s             The trained policy’s reward, by contrast, collapses through
weighted pairwise consistency rate [11]: rather than checking          the same range (from ∼650 at µ=0.5 to ∼200 at µ=0.10),
answer consistency across tagged question pairs on a single            a real behavioral regime change to which the WM’s imagined
observation, we check rollout consistency across controlled            rollouts are blind. Real-physics iKCE under the trained actor
conditioning perturbations on a single base scenario. The              shows more cell-to-cell variability than the WM, with elevated
diagnostic is reframed rather than contrived, and produces             values in the low-µ region (means 1.04-1.30 × 10−4 across
falsifiable curves on any embodiment whose state admits a              µ ∈ [0.1, 0.3] vs. 0.70–0.92 × 10−4 across µ ∈ [0.5, 1.7]).
kinematic predictor.                                                   This confirms that the iKCE metric is not degenerate, while
                                                                       the WM’s invariance to the same perturbation provides the
                IV. Experiments and Results                            kinematic, not dynamic, signature. The structural difference
   The experimental setup is documented in Appendix A.                 is reinforced by appendix controls: a per-step decomposition
Hypothesis 1 (H1): iKCE is non-degenerate on a pub-                    (Fig. 6) shows that the physics and WM residuals differ
lished checkpoint. At both measurement horizons, the trained           qualitatively in temporal structure, not just in magnitude, and
DreamerV3 walker-walk world model produces an imagined                 the same H2 signature reappears under a richer kinematic slice
iKCE at least an order of magnitude above matched policy-              (Fig. 8). At the shorter T =16 horizon, neither side shows
driven real-physics rollouts on the same conditioning (see             friction sensitivity. The dynamic signature emerges only as
Table I: ∼180× at T =16, ∼30× at T =64). The narrowing                 slip accumulates into per-step deviation at the longer horizon
at the longer horizon is consistent with the per-step dilution         (quantified in Appendix Fig. 3). Controls supporting H2 are
noted in §V, limitation (iii): the WM’s smooth post-transient          reported in detail in the appendix: per-step structure under the
tail averages down the integrated metric. iKCE is therefore            actor ablation (Fig. 7), robustness to the kinematic-state choice
non-degenerate at both horizons (see the trivial-WM scale              (Fig. 8), and a joint-noise positive control (Fig. 2, right panel)
                              800                                                             pre-empts the natural counterfactual that scaling the actor’s



   episodic reward (0–1000)
                                                                                              imagination horizon would have closed the gap.
                              600                                                                Limitations of the present measurement. Several limi-
                                                                                              tations bound the result. (i) The empirical result rests on a
                              400                                                             single embodiment (DMC walker-walk, a 2D 9-DOF system)
                                                                      trained policy reward
                                                                                              restricted to the (z, vz ) sub-slice, and on a single open-weight
                              200                                                             WM family. Extending the flatness test for H2 to quadruped,
                                                                      50% baseline
                                                                      regime µ = 0.20         humanoid, and driving embodiments, as well as to other WM
                               0
                                    0.2   0.4   0.6     0.8       1        1.2      1.4       families such as GAIA-1 [7] or R2Dreamer [9], would further
                                                friction multiplier µ                         broaden the evidence base. (ii) The policy-OOD confound of
                                                                                              the physics-side signature is bounded, not eliminated, by the
Fig. 1. Empirical regime boundary. Mean episodic reward of the trained                        domain-randomization control (Appendix B2): the H2 contrast
DreamerV3 walker policy across friction (K=10, 95% bootstrap CI). The                         survives under a policy in-distribution at every swept µ, while
regime boundary at µ=0.20 (red dashed) is the friction at which mean reward
first drops below 50% of the µ=1.0 baseline (dotted). This boundary anchors                   the partition of the original low-µ elevation holds at the point-
the dashed reference line in Fig. 2.                                                          estimate level only. Two residuals remain: the DR policy adapts
                                                                                              its gait to friction in closed loop, so the physics response
                                                                                              is not policy-free; and imagined rollouts condition on only
confirming the WM responds to kinematic-axis perturbations.                                   five observations (under one gait period), so the WM-side
   Actor training horizon ablation. An actor-training-horizon                                 flatness is informative only up to the regime evidence the
ablation (see Appendix Fig. 4) rules out the most concerning                                  prefix can carry. (iii) Per-step displacement decays over the
confound: retraining at imag horizon = 64 produces identical                                  rollout horizon, so the WM’s low long-horizon iKCE reflects
WM iKCE friction spread (1.32×), confirming H2 is not an                                      in part reduced motion magnitude rather than purely cleaner
artifact of the default actor’s 15-step training horizon.                                     kinematic imagination.
   Domain-randomization control. A domain-randomization                                          Open directions. Embodiment extension to quadruped-walk
control (Appendix B2, Fig. 5) bounds the policy-OOD con-                                      would test the diagnostic on richer contact dynamics than
found: under a policy trained with µ ∼ U(0.1, 1.7), the WM                                    the planar walker provides. Fixed open-loop action sequences,
                              DR
slope still contains zero (βWM     = −0.026) and the physics                                  applied identically across physics and WM rollouts, would
                           DR
slope still excludes it (βphys   = −0.114), so the kinematic-                                 remove the closed-loop policy adaptation left in place by the
not-dynamic contrast survives with the policy in-distribution                                 domain-randomization control. An explicit-conditioning exper-
at every swept friction.                                                                      iment, in which friction or contact indicators are appended
                                                                                              to the WM’s observation, together with a conditioning-prefix-
                                    V. Discussion & open directions                           length sweep (5–64 observed steps), would distinguish repre-
                                                                                              sentational absence from architectural insensitivity – and from
   iKCE is a per-step kinematic null fit integrated over a                                    regime evidence the prefix simply cannot carry. A driving-WM
horizon. It diagnoses kinematic imagination by the absence of                                 cross-anchor on a model exposing an ego-pose head (Vista [1],
regime sensitivity, not by the presence of dynamic prediction                                 DriveDreamer [12]) would connect the measurement to the
quality. A WM that achieves low iKCE everywhere has been                                      autonomous-driving evidence of §II.
correctly identified as kinematic by our protocol, but has not
been certified as a useful predictor. A high-iKCE WM with
                                                                                                                        References
friction sensitivity has been certified as dynamic but not as
accurate. The diagnostic is structural, not predictive, by design.                             [1] Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta,
   A downstream behavioral prediction. If imagined rollouts                                        Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang
are kinematically structured but not dynamically faithful, then                                    Li. Vista: A generalizable driving world model with high
policy gradients propagated through long imagined rollouts                                         fidelity and versatile controllability, 2024. URL https:
optimize the actor against a trajectory distribution that diverges                                 //arxiv.org/abs/2405.17398.
from real dynamics in directions iKCE itself does not capture                                  [2] Yuan Gao, Dengyuan Hua, Mattia Piccinini, Finn Ras-
(rotational drift, contact-event timing, accumulated absolute-                                     mus Schäfer, Korbinian Moller, Lin Li, and Johannes
state error). Long-horizon actor training should therefore be                                      Betz. StyleVLA: Driving style-aware vision language
unstable and yield a weaker deployed policy. The h=64                                              action model for autonomous driving. arXiv preprint
ablation is consistent with this prediction: under matched                                         arXiv:2603.09482, 2026.
hyperparameters, the long-horizon actor converged to ∼400                                      [3] David Ha and Jürgen Schmidhuber. Recurrent world
episodic reward versus ∼955 for the default h=15 checkpoint,                                       models facilitate policy evolution. In Advances in Neural
and exhibited training instability throughout. We do not claim                                     Information Processing Systems (NeurIPS), 2018.
a causal link. Long-horizon Dreamer training is known to be                                    [4] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben
sensitive to multiple factors, but the observation is what one                                     Villegas, David Ha, Honglak Lee, and James Davidson.
would predict from the kinematic-not-dynamic hypothesis, and                                       Learning latent dynamics for planning from pixels. In
                                  friction (physical-regime)                                             joint noise (kinematic control)
              10−2                                                               10−2



              10−3                                           physics + policy    10−3
       iKCE                                                  WM (imagined)
                                                                                                                                       physics + policy
                                                                                                                                       WM (imagined)
                                                             regime µ = 0.20
              10−4                                                               10−4


                     0.2   0.4    0.6    0.8      1      1.2     1.4    1.6             0     5 · 10−2       0.1      0.15       0.2       0.25       0.3
                                     friction multiplier µ                                                     joint-noise σ (rad)

Fig. 2. iKCE diverges in physics, stays flat in imagination. Identity kinematic view (z, vz ) at horizon T = 64. Left: friction sweep µ ∈ [0.1, 1.7] (physical
regime axis). Real-physics rollouts under the trained policy (blue) show modestly elevated iKCE in the low-µ region near the empirical regime boundary
(µ=0.20, dashed line (see Fig. 1)). WM-imagined rollouts (red) are statistically flat across the entire 17× sweep. Right: joint-noise sweep σ ∈ [0, 0.3] rad
(kinematic control axis). Both channels respond similarly, confirming that the WM is not insensitive to all perturbations, only to dynamic ones. Shaded bands:
95% bootstrap CI from K = 20 rollouts per cell.



     International Conference on Machine Learning (ICML),                               and Jingdong Wang. Rethinking the open-loop evaluation
     2019.                                                                              of end-to-end autonomous driving in nuscenes, 2023.
 [5] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timo-                              URL https://arxiv.org/abs/2305.10430.
     thy Lillicrap. Mastering diverse domains through world
     models. Nature, 640:647–653, 2025.
 [6] Anthony Hu, Gianluca Corrado, Nicolas Griffiths, Zak
     Murez, Corina Gurau, Hudson Yeo, Alex Kendall,
     Roberto Cipolla, and Jamie Shotton. Model-based im-
     itation learning for urban driving. In Advances in Neural
     Information Processing Systems (NeurIPS), 2022.
 [7] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez,
     George Fedoseev, Alex Kendall, Jamie Shotton, and
     Gianluca Corrado. Gaia-1: A generative world model
     for autonomous driving, 2023. URL https://arxiv.org/abs/
     2309.17080.
 [8] Michael Janner, Justin Fu, Marvin Zhang, and Sergey
     Levine. When to trust your model: Model-based policy
     optimization. In Advances in Neural Information Pro-
     cessing Systems (NeurIPS), 2019.
 [9] Naoki Morihira, Amal Nahar, Kartik Bharadwaj, Ya-
     suhiro Kato, Akinobu Hayashi, and Tatsuya Harada.
     R2-dreamer: Redundancy-reduced world models without
     decoders or augmentation, 2026. URL https://arxiv.org/
     abs/2603.18202.
[10] Abhinaw Priyadershi and Jelena Frtunikj. Lost in fog:
     Sensor perturbations expose reasoning fragility in driving
     VLAs. arXiv preprint arXiv:2605.21446, 2026.
[11] Finn Rasmus Schäfer, Yuan Gao, Dingrui Wang, Thomas
     Stauner, Stephan Günnemann, Mattia Piccinini, Sebastian
     Schmidt, and Johannes Betz. Egodyn-bench: Evaluating
     ego-motion understanding in vision-centric foundation
     models for autonomous driving, 2026. URL https://arxiv.
     org/abs/2604.22851.
[12] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen,
     Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards
     real-world-driven world models for autonomous driving,
     2023. URL https://arxiv.org/abs/2309.09777.
[13] Jiang-Tian Zhai, Ze Feng, Jinhao Du, Yongqiang Mao,
     Jiang-Jiang Liu, Zichang Tan, Yifu Zhang, Xiaoqing Ye,
                            Appendix                                    original training run. The same procedure on the physics-side
   This appendix documents (i) the experimental configura-              sweep (seed 0 only, as physics is not a learned model) gives
tion (Table II), (ii) the methodological details behind the             βphys = −0.220, CI [−0.301, −0.142], comfortably excluding
headline numbers (§A1-A4: regime-boundary determination,                zero. The low-µ elevation reported in §IV is statistically real.
the flatness and horizon-emergence tests, and the trivial-WM            The quantitative form of the kinematic-not-dynamic signature
scale anchor), (iii) the controls supporting H2 (§B1-B6: actor-         is therefore: across three seeds, |βWM | is bounded above
training-horizon ablation, domain-randomization control, per-           by ∼0.13 (one decade of µ changes WM iKCE by at most
step structure decomposition, per-step structure under the actor        ∼13%), while |βphys | ≈ 0.22 (∼25% per decade), with non-
ablation, robustness to the kinematic-state choice, and the             overlapping confidence intervals.
joint-noise positive control), and (iv) implementation specifics           3) Horizon-emergence test.: Section III-B argues that the
needed to reproduce the measurement (§C1-C2). Code and                  diagnostic should be applied at horizons longer than the
data will be released upon acceptance.                                  embodiment’s gait period. We sharpen this claim quantitatively
                                                                        by repeating the flatness regression of the preceding paragraph
                             TABLE II                                   at four sub-horizons T ∈ {8, 16, 32, 64}, re-integrating each
   Experimental setup. Horizons h=16 and h=64 refer to the iKCE
       rollout length, evaluated on the same trained policy.            rollout’s saved per-step iKCE trace from the existing T =64
                                                                        sweep (no new rollouts, see Fig. 3). The dynamic signature
  Field                Value                                            in physics emerges with horizon: the slope βphys grows in
                                                                        magnitude from +0.012 (CI [−0.121, +0.146]) at T =8 to
  World model          DreamerV3 (NM512 PyTorch port, com-
                       mit 6ef8646)
                                                                        −0.221 (CI [−0.301, −0.142]) at T =64, crossing out of the
  Task                 DMC walker-walk, dmc_proprio config              CI-contains-zero region between T =32 and T =64 – consistent
  Training             1M env steps, seed 0, RTX 5090                   with friction effects accumulating over multiple footfalls before
  Final reward         955 ± 30 (mean over last 100k steps)             becoming detectable in the per-step kinematic-null residual.
  Evaluation policy    Trained actor (same checkpoint on both           The WM-side slope βWM is statistically indistinguishable
                       physics and WM sides)
  Backend              dm_control 1.0.20, mujoco 3.1.6
                                                                        from zero at every horizon tested (−0.028, −0.012, −0.013,
  K                    20 rollouts per perturbation cell                −0.009 at T =8, 16, 32, 64, with CIs of width ≤ 0.32 all
  Kinematic spec       (z, vz ) root-vertical-motion (1D)               straddling zero). The contrast is the H2-emergence claim stated
  Extrapolation        constant velocity                                quantitatively: the dynamic signature emerges with horizon in
                                                                        physics but not in the WM. Note that long-horizon iKCE in
                                                                        both channels is in part dilated by reduced per-step motion
A. Methodological Details                                               magnitude (§V, limitation (iii)). The present result is robust
   1) Regime-boundary determination.: The boundary                      to that effect because the WM-physics contrast widens with
µ=0.20 referenced in Fig. 2 (dashed line) and Fig. 1 is not             horizon rather than shrinks, which is the opposite of what a
chosen a priori. It is determined empirically from the policy’s         horizon-degenerate metric would produce.
reward collapse. We roll out the trained actor for K=10
episodes at each of 12 friction multipliers µ ∈ [0.1, 1.5],                                           0.2




                                                                          β = ∂ log(iKCE)/∂ log(µ)
compute the mean episodic reward and a 95% bootstrap CI
per cell, and define the regime boundary as the largest µ at
which mean reward has dropped below 50% of its µ=1.0                                                    0
baseline. The µ=1.0 mean over this sweep is ∼650 (K=10
episodes), giving a threshold of ∼325. On our checkpoint,
the boundary lies at µ=0.20. (The 955 ± 30 final reward in                                           −0.2
Table II averages over the last 100k training steps and is not                                                  physics + policy
                                                                                                                WM (imagined)
directly comparable.)
   2) Flatness test for H2.: We make the “statistically flat”                                               8                 16           32         64
claim falsifiable by regressing log(iKCE) on log(µ) across                                                            measurement horizon T (steps)
the T =64 friction sweep, with each of the K=20 rollouts at
each of the 13 µ values contributing one observation (n=260             Fig. 3. Horizon-emergence test. Slope β = ∂ log(iKCE)/∂ log(µ) of
per seed). To rule out a seed-dependent artifact, we repeat the         the friction sweep at four measurement horizons T ∈ {8, 16, 32, 64},
                                                                        computed by re-integrating the saved T =64 per-step iKCE traces (no new
regression for three independently-trained DreamerV3 walker-            rollouts). Physics slope (blue) grows in magnitude with T and crosses out
walk checkpoints (seeds {0, 1, 2}, matched hyperparameters              of the CI-contains-zero region by T =64. WM slope (red) is statistically
and step budget; see Appendix A). A 95% percentile boot-                indistinguishable from zero at every horizon. Dashed line marks the H2
                                                                        flatness target (β = 0). Shaded bands: 95% percentile bootstrap CI over
strap (1000 resamples over rollouts) on each slope gives:               1000 resamples at the rollout level.
  (0)                                           (1)
βWM = −0.009, CI [−0.096, +0.082]; βWM = +0.031, CI
                       (2)
[−0.072, +0.129]; βWM = +0.038, CI [−0.039, +0.123], all                   4) Trivial-WM scale anchor.: The iKCE scale has an an-
three WM slope CIs contain zero, so H2’s flatness claim                 alytic lower bound that anchors the magnitudes in Table I.
survives a falsifiable statistical test across seeds, not just on the   A trivial “WM” that imagines by applying the kinematic
predictor to its own current state, x̂WM
                                      t+1 = kin(x̂t ), produces
                                                  WM
                                                                            imagine regime-conditional rollouts – and it remains friction-
iKCE = 0 by construction: each predicted next state is                      invariant (Fig. 5, left). The flatness signature is therefore not
identically the kinematic continuation of its predecessor, so               an artifact of the training-time friction distribution.
every residual in Eq. 1 is zero. The measured ordering on                      Physics side: under the DR policy, the physics slope remains
                                                                                                   DR
walker-walk is therefore                                                    strictly negative (βphys    = −0.114, CI [−0.201, −0.024]), at
                                                                            roughly half the default-policy magnitude (−0.220). Read
       0
      |{z}           <     4.2 × 10−5           ≪ 7.7 × 10−3     (T =16).   at the point-estimate level, this partitions the original low-
                           | {z }                 | {z }
 trivial kinematic       matched real physics     DreamerV3 WM              µ elevation into comparable parts: about half attributable to
The WM lies further from the trivial-kinematic baseline                     an out-of-distribution policy slipping, about half a genuine
than real physics does, ruling out a naive reading in which                 friction response of the contact dynamics that persists when
“imagining kinematically” would imply small absolute iKCE.                  the policy is in-distribution everywhere. We state this partition
The diagnostic signature, per Section III, is friction-invariance           at the point-estimate level only: a percentile bootstrap on the
under perturbation, not low absolute magnitude. The order                   difference of the two physics slopes does not resolve it at
                                                                                        default     DR
holds for T =64 as well.                                                    K=20 (βphys         − βphys = −0.107, CI [−0.221, +0.004]).
                                                                               Under matched DR conditions on both sides, the H2 contrast
B. Controls                                                                 retains its falsifiable form: the physics slope’s CI excludes zero
   1) Actor-training-horizon control.: A natural concern is                 while the WM slope’s contains it. The slope difference itself
that the WM’s friction-invariance at T =64 reflects the default             is directionally consistent but not resolved at this sample size
                                                                               DR        DR
actor operating out-of-distribution from its training horizon               (βphys  − βWM     = −0.087, CI [−0.211, +0.049]).
(imag horizon = 15) rather than a structural property of                       This control addresses the training-distribution side of lim-
WM imagination. To rule this out, we retrain an identical                   itation (ii). Two residual caveats remain, both deferred to the
checkpoint at imag horizon = 64, matching the measure-                      open directions: the DR policy still adapts its gait to friction
ment horizon, with the same seed, hyperparameters, and total                in closed loop, so the fully policy-free variant (fixed open-
training budget. Fig. 4 shows the result: WM iKCE friction                  loop action sequences applied identically to both channels)
spread under the h=64-trained actor is identical to the default-            remains the cleaner disambiguation; and the imagined rollouts
actor headline (1.32× in both cases, CIs overlapping at every               condition on only five observed steps (125 ms, under one gait
µ), confirming that friction-invariance is not an artifact of actor         period), which bounds how much regime evidence even a
training horizon.                                                           dynamically capable imaginer could extract from the prefix.
   2) Domain-randomization control.: Limitation (ii) in §V
                                                                                                          TABLE III
names the most consequential confound of the physics-side                           Domain-randomization control: flatness regression
H2 signature: the evaluation policy acted only at µ=1.0 during                β = ∂ log(iKCE)/∂ log(µ) at T =64, 95% percentile bootstrap CIs
training, so the elevated low-µ physics iKCE may reflect an in-             (1000 resamples at the rollout level, K=20 per cell). All four WM
                                                                                checkpoints are statistically flat regardless of the friction
distribution policy slipping under out-of-distribution friction              distribution seen at training time; both physics slopes are strictly
rather than a genuine friction response of the contact dynam-                                             negative.
ics. To quantify this confound, we train a fourth, otherwise
identical DreamerV3 checkpoint with per-episode domain ran-                  Side      Checkpoint / policy              β    95% CI
domization of friction, µ ∼ U (0.1, 1.7) drawn at every episode              WM        seed 0 (fixed µ=1.0)       −0.009     [−0.096, +0.082]
reset (matched hyperparameters and step budget; evaluation                   WM        seed 1 (fixed µ=1.0)       +0.031     [−0.072, +0.129]
reward 930 ± 36 across the full sweep range). Both the DR                    WM        seed 2 (fixed µ=1.0)       +0.038     [−0.039, +0.123]
policy and the DR world model are therefore in-distribution at               WM        DR (µ ∼ U(0.1, 1.7))       −0.026     [−0.123, +0.076]
every friction value of the H2 sweep. The DR policy exhibits                 Physics   default policy (seed 0)    −0.220     [−0.301, −0.142]
no reward collapse anywhere in the tested range, so the regime               Physics   DR policy                  −0.114     [−0.201, −0.024]
boundary of Fig. 1 is a property of the fixed-µ policy and does
not transfer to this control.                                                  3) Per-step structure decomposition.: The integrated iKCE
   Table III reports the flatness regression of Appendix A2                 of Table I and Fig. 2 averages over the rollout horizon and
under the DR checkpoint, alongside the fixed-µ results;                     so does not reveal whether the WM’s imagined residual has
Fig. 5 shows the underlying sweeps. WM side: the DR-                        the same temporal structure as physics. Fig. 6 decomposes per-
                                                DR
trained world model is statistically flat (βWM        = −0.026,             step iKCE at three friction values µ ∈ {0.15, 1.0, 1.5}: physics
CI [−0.123, +0.076]), indistinguishable from the three fixed-               exhibits sparse contact-event spikes whose positions shift with
µ seeds. This closes a data-coverage loophole in the seed-                  µ (consistent with footfall dynamics driving the kinematic-
level flatness test: a world model trained only at µ=1.0 has                null residual), while the WM shows a one-to-two-step encoder-
never observed friction variation and cannot have learned                   decoder transient followed by a smooth, friction-invariant tail.
friction-conditional latent dynamics, so its flatness is partially          The two residuals are not merely different in magnitude (the
guaranteed by construction. The DR world model was trained                  H1 ratio) but qualitatively different in temporal structure, the
on transitions spanning the full sweep range, could in principle            WM’s imagined rollouts do not reproduce the contact-event
infer the friction regime from its conditioning prefix and                  signature that defines the physics-side residual.
                                               friction (physical-regime)                                                 joint noise (kinematic control)
                       10−2                                                                        10−2



                       10−3                                                 physics + policy       10−3
       iKCE                                                                 WM (imagined)
                                                                                                                                                        physics + policy
                                                                                                                                                        WM (imagined)
                                                                            regime µ = 0.20
                       10−4                                                                        10−4


                                  0.2    0.4   0.6      0.8      1        1.2      1.4     1.6             0   5 · 10−2       0.1       0.15      0.2       0.25        0.3
                                                    friction multiplier µ                                                       joint-noise σ (rad)

Fig. 4. Actor-training-horizon ablation. Identity view (z, vz ) at T =64, with the actor retrained at imag horizon = 64 (matching the measurement
horizon). WM iKCE friction spread is identical to the default-actor headline (1.32× in both cases), confirming the friction-insensitivity of imagined rollouts
is not an artifact of the default actor’s h=15 training horizon. Shaded bands: 95% bootstrap CI from K = 20 rollouts per cell.

                                               friction (physical-regime)                                                 joint noise (kinematic control)
                       10−2                                                                        10−2



                       10−3                                                                        10−3
       iKCE                                                                     physics + policy                                                        physics + policy
                                                                                WM (imagined)                                                           WM (imagined)

                       10−4                                                                        10−4


                                  0.2    0.4   0.6      0.8      1        1.2      1.4     1.6             0   5 · 10−2       0.1       0.15      0.2       0.25        0.3
                                                    friction multiplier µ                                                       joint-noise σ (rad)

Fig. 5. Domain-randomization control. Identity view (z, vz ) at T =64 with friction domain-randomized at training time (µ ∼ U (0.1, 1.7) per episode).
Left: friction sweep µ ∈ [0.1, 1.7] (physical-regime axis). WM-imagined rollouts remain statistically flat despite the world model having been trained on
the full friction range; physics-side iKCE under the DR policy retains a strictly negative slope, with the point estimate at roughly half the default-policy
magnitude. Right: joint-noise sweep σ ∈ [0, 0.3] rad (kinematic control axis). The DR-trained WM responds to kinematic-axis perturbations, replicating the
positive control of Fig. 2 under the DR checkpoint. Shaded bands: 95% bootstrap CI from K=20 rollouts per cell.

                                                      physics + policy                                                              WM (imagined)
                       10−3
                                                                                      µ = 0.15                                                                   µ = 0.15
                                                                                      µ = 1.0       10−2                                                         µ = 1.0


       per-step iKCE
                                                                                      µ = 1.5                                                                    µ = 1.5
                       10−5
                                                                                                    10−4


                       10−7

                              0         10     20          30        40          50         60             0     10          20         30       40         50        60
                                                      imagined step t                                                               imagined step t

Fig. 6. Per-step iKCE structure. Log-y. Physics (left) has discrete contact-event spikes whose positions shift with µ. WM (right) shows a one- to two-step
encoder-decoder transient followed by a smooth tail. The per-step structure is essentially constant across friction.



   4) Per-step structure under the actor-ablation checkpoint.:                                     main result uses the root-vertical-motion slice (z, vz ). Figure 8
Figure 6 reports the per-step decomposition under the default                                      repeats the protocol with a richer representation, the walker’s
actor. Figure 4 reports the integrated iKCE under the retrained                                    gait degrees of freedom. The qualitative pattern of Fig. 2
h=64 actor. Figure 7 combines the two controls: per-step WM                                        reappears: WM iKCE is flat across friction, while physics
iKCE under both the default (h=15) and retrained (h=64)                                            shows low-µ elevation, confirming that H2 reflects a property
actor, at the same three friction values µ ∈ {0.15, 1.0, 1.5}.                                     of the WM’s imagination rather than the particular state slice
The transient-plus-smooth-tail structure is unchanged across                                       used to evaluate it. This generalizes the diagnostic claim: any
actor training horizons, ruling out the joint concern that the                                     kinematic state extraction whose dynamics are sensitive to the
per-step signature reflects an actor-out-of-distribution artifact                                  perturbation axis can serve as a probe, with no special status
rather than a property of the WM’s imagination.                                                    accorded to the identity slice.
  5) Robustness to the kinematic-state choice.: iKCE depends                                         6) Joint-noise as a kinematic positive control.: The friction
on a chosen kinematic state vector x̂t (Definition 1). The                                         sweep is a dynamic perturbation (physically-grounded, regime-
                                                    WM, actor h=15                                                                WM, actor h=64
                                                                                   µ = 0.15                                                                    µ = 0.15
                       10−2                                                        µ = 1.0     10−2                                                            µ = 1.0


       per-step iKCE
                                                                                   µ = 1.5                                                                     µ = 1.5


                       10−4
                                                                                               10−4



                              0         10     20          30       40       50         60               0      10           20       30        40        50        60
                                                    imagined step t                                                               imagined step t

Fig. 7. Per-step actor ablation. WM per-step iKCE at three friction values for both actor checkpoints. The transient-plus-smooth-tail structure is independent
of the actor training horizon.

                                              friction (physical-regime)                                                joint noise (kinematic control)
                        2                                                physics + policy        2                                                    physics + policy
                                                                         WM (imagined)                                                                WM (imagined)
                                                                         regime µ = 0.20

              iKCE     1.5                                                                     1.5



                        1                                                                        1

                                  0.2   0.4   0.6    0.8        1     1.2    1.4      1.6            0       5 · 10−2       0.1      0.15       0.2       0.25       0.3
                                                friction multiplier µ                                                         joint-noise σ (rad)

Fig. 8. Gait DOFs view. Same protocol as Fig. 2 but using the walker gait kinematic slice. Physics still shows low-µ elevation, WM is still flat, the finding
is not an artifact of the (z, vz ) identity view.



crossing). The joint-noise sweep is its kinematic counterpart                                 walker joint angle, and kin(·) applies one-step extrapola-
(zero-mean Gaussian noise added to the joint-position channel                                 tion per joint. Friction perturbations are applied by scal-
of every observation before the WM encoder, with the physics                                  ing the MuJoCo friction tuple of every geom by µ at
state left unperturbed). A kinematic imaginer should respond                                  episode reset (model.geom_friction[:, 0] *= mu).
to the joint-noise sweep. The perturbation directly modifies                                  Joint-noise perturbations add zero-mean Gaussian noise with
the kinematic state x̂t that drives the extrapolation, while a                                standard deviation σ (rad) to the joint-position channel of every
dynamic imaginer should respond to the friction sweep. The                                    observation before the WM encoder, with the physics state it-
right panel of Fig. 2 confirms that both channels respond                                     self left unperturbed. For WM-imagined rollouts, we condition
to joint noise, ruling out the alternative explanation that the                               on the first 5 perturbed observations (encoder unroll), then
WM’s iKCE is simply insensitive to all perturbations. The                                     roll out free imagination for the remaining T − 5 steps. This
contrast (response to joint noise, non-response to friction) is                               matches the standard Dreamer evaluation protocol. All 95%
the diagnostic signature named in the paper’s title.                                          confidence intervals are computed using a percentile bootstrap
                                                                                              with 1000 resamples across the K=20 rollouts per cell.
C. Reproducibility
   1) Code and data availability.: The diagnostic pipeline,
trained checkpoints, perturbation-sweep CSVs, and PGFPlots
figure sources for this paper are released at https://github.
com/TUM-AVS/iKCE. The DreamerV3 implementation is the
NM512 PyTorch port at commit 6ef8646. The upstream al-
gorithm is [5]. Checkpoints are released at https://huggingface.
co/fnc1901/ikce-walker-walk-artifacts. All experiments ran on
a single RTX 5090 (training: ∼24 h per checkpoint. The
full perturbation sweep, including the actor-horizon ablation:
∼2 h).
   2) Implementation specifics.: For the identity view, the
kinematic predictor is constant-velocity on the root ver-
tical state: kin([zt , żt ]) = [zt + ∆t żt , żt ] with ∆t =
25 ms (the DMC physics timestep). For the gait view, x̂t
stacks the unit-circle embedding (cos θj , sin θj ) of each
