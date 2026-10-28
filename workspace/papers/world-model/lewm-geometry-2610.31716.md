<!-- page 1 of 13 -->

arXiv:2610.31716v1 [cs.LG] 22 Oct 2026

NeurReps – Extended Abstract Track 1–13, 2026

Symmetry and Geometry in Neural Representations

# Does LeWorldModel Preserve Local Action Geometry?

Vasil Vasilev

vasil.g.vasilev.2022@elsys-bg.org

Technological School Electronic Systems (TUES)

Siddharth Nayak

sidnayak@mit.edu

Massachusetts Institute of Technology

## Abstract

We examine how closely pretrained LeWorldModel (LeWM) reproduces the effects of small action changes. Using matched simulator and model responses, we measure action sensitivity with Gram matrices at three perturbation scales. Across nearly a thousand states from Reacher and OGBench Cube, the most sensitive action directions differ by median angles of about 36◦ and 42◦, respectively, at the smallest scale. Increasing the perturbation tenfold reduces median normalized Gram discrepancy by roughly 70% on Reacher and 47% on Cube. Geometry measured in pixels and embeddings varies more across scales than the chosen physical-state geometry. On held-out directions, environment Grams achieve median embedding response errors of 17% and 28%, compared with 26% and 38% for an isotropic reference with the same trace. LeWM’s errors are higher at this scale. These findings support scale-specific evaluation of action sensitivity; its effect on control remains an open question.

Keywords: world models, JEPA, action sensitivity, representation geometry, Gram matrix

## 1. Introduction

World models support planning through learned predictive dynamics (Ha and Schmidhuber, 2018; Hafner et al., 2019, 2024). Joint-Embedding Predictive Architectures learn in representation space (Assran et al., 2023; Bardes et al., 2024), including action-conditioned systems such as V-JEPA 2 and LeWorldModel (LeWM) (Assran et al., 2025; Maes et al., 2026). Accurate prediction at a sampled action does not establish that nearby action directions have the correct relative effects. This motivates evaluating action sensitivity alongside aggregate prediction and control scores.

We study pretrained LeWM on Reacher and OGBench Cube using simulator counterfactuals and a shared frozen encoder. Related symmetry-aware and geometric perspectives motivate this diagnostic (Lillemark et al., 2026; Cayco-Gajic and Pellegrino, 2026). Our contributions are a matched finite-scale comparison, calibration against simple references, and validation on held-out action directions in three distinct output metrics. We distinguish sensitivity in the encoder’s output space from physical response and goal-directed action choice.

## 2. Measuring Finite-Scale Action Geometry

Let $H _ { t }$ be a fixed three-observation context, $E _ { \theta }$ the frozen encoder, and $A _ { t } = ( a _ { t } , \ldots , a _ { t + 4 } )$ the recorded five-action block. We perturb all five actions by a shared normalized offset

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">© 2026 V. Vasilev & S. Nayak.</span></small>

<!-- page 2 of 13 -->

Vasilev Nayak

$$
\begin{aligned} { u \in \mathbb { R } ^ { d } } & { { } \colon } \\ { } & { { } A _ { t } ( u ) = ( a _ { t } + D u , \ldots , a _ { t + 4 } + D u ) , \qquad D = \operatorname { d i a g } ( \operatorname { a c t i o n ~ s t a n d a r d ~ d e v i a t i o n s } ) . } \\ \end{aligned}
$$

$F _ { t } ^ { \mathrm { m o d e l } } ( u )$ predicts the next embedding from $H _ { t }$ and $A _ { t } ( u ) ;   F _ { t } ^ { \mathrm { e n v } } ( u )$ encodes the rendered endpoint after executing the same block from the restored simulator state. Both maps use the same output space. With $e _ { i }$ the ith coordinate vector, define

$$
\begin{aligned} { J _ { t , \epsilon } ^ { b } e _ { i } = } & { { } \frac { F _ { t } ^ { b } ( h _ { i } e _ { i } ) - F _ { t } ^ { b } ( - h _ { i } e _ { i } ) } { 2 h _ { i } } , \qquad h _ { i } = \frac { \epsilon ( a _ { i } ^ { \operatorname* { m a x } } - a _ { i } ^ { \operatorname* { m i n } } ) } { 2 D _ { i i } } , } \\ { G _ { t , \epsilon } ^ { b } = } & { { } \; ( J _ { t , \epsilon } ^ { b } ) ^ { \top } J _ { t , \epsilon } ^ { b } . } \\ \end{aligned}
$$

where $b \in \{ \mathrm { e n v } , \mathrm { m o d e l } \}$ The shared offset gives d action coordinates. The identity $\| J _ { t , \epsilon } ^ { b } u \| ^ { 2 }   =   u ^ { \top } G _ { t , \epsilon } ^ { b } u$ motivates predicting response magnitudes. We test this approximation at each scale because contacts and rasterization can limit local linearity.

We measure relative sensitivity using $\widetilde { G } = G / \operatorname { t r } ( G )$ and $D _ { G } = \| \widetilde { G } ^ { \mathrm { m o d e l } } - \widetilde { G } ^ { \mathrm { e n v } } \| _ { F }$ . Zero or nonfinite traces are invalid; the primary additive constant is zero. We separately report absolute gain, condition numbers and eigenspace angles. Calibration compares $D _ { G }$ with isotropy $( I _ { d } / d )$ , random orthogonal conjugation of the environment metric, and shuffledstate matching. Each reference removes a specific source of information. Definitions and episode-bootstrap uncertainty are in Appendix A.

One released checkpoint per environment is evaluated without training. The common noclipping bank retains 458 Reacher anchors $( d = 2 ;$ 42 action-bound exclusions) and 500 Cube anchors $( d = 5 )$ . Five wrapper steps span 0.20 and 0.25 s, respectively. We report $\epsilon = 1 0 ^ { - 3 }$ (primary), $3 \times 1 0 ^ { - 3 }$ and $1 0 ^ { - 2 }$ . Following an audit of action alignment, output coordinates and state restoration, we regenerate trajectories in the current simulator from the original dataset sampling anchors. The revised estimates supersede the earlier results (Appendix C). Appendix D.4 documents replay exceptions and the resulting protocol change.

## 3. Results

The model captures some action structure but remains misaligned. The comparison separates sensitivity strength from action-direction alignment. Eigenvalues describe the squared response strengths along the corresponding eigenvectors; similar condition numbers can therefore coexist with different sensitive directions. Trace-normalized $D _ { G }$ captures both relative spectral shape and orientation, while deliberately discarding a common multiplicative gain.

At the primary scale, the most sensitive directions differ by a median $3 6 ^ { \circ }$ on Reacher and $4 2 ^ { \circ }$ on Cube. The latter also shows about $4 0 ^ { \circ }$ of misalignment between its leading two-dimensional subspaces (Figure 1). Median $D _ { G }$ is 0.465 on Reacher and 0.508 on Cube; their 95% episode-bootstrap intervals are [0.423, 0.500] and [0.487, 0.529].

Reacher’s environment and model condition numbers are both near 4.5, despite their directional mismatch. Cube’s median model condition number is roughly four times the environment value. The numerical floor affects about a third of Cube environment matrices, so the upper tail requires care. Substantial misalignment remains when we restrict the comparison to well-separated leading eigenvalues (Appendix C).

2

<!-- page 3 of 13 -->

Local Action Geometry in LeWM

Table 1: Median primary-scale normalized Gram discrepancy (lower is closer). Randomized references use 1,000 draws per state.

| Environment | Model | Isotropic | Rotation | Shuffle |
| --- | --- | --- | --- | --- |
| Reacher | 0.465 | 0.449 | 0.632 | 0.530 |
| Cube | 0.508 | 0.609 | 0.885 | 0.560 |

![Chart block](doc:8b3b37b/tier:basic/page:3/block:4)

![Chart block](doc:8b3b37b/tier:basic/page:3/block:5)

Figure 1: Action-sensitivity ellipses at $\epsilon = 1 0 ^ { - 3 }$ We select the sampled state nearest the median $D _ { G }$ in each environment. Each curve maps the unit circle through $B ^ { 1 / 2 }$ where B is the trace-normalized Gram restricted to the displayed plane. Reacher shows both action dimensions. Cube shows the environment’s leading twodimensional plane, which cannot reveal all misalignment in its five-dimensional action space.

Cube has lower discrepancy than all three references. Reacher improves on rotation and shuffle, while isotropy is closer to its environment metric (Table 1). Paired model-minusshuffle median differences are −0.059 for Cube (95% interval $[ - 0 . 0 7 4 , - 0 . 0 2 9 ] )$ and −0.047 for Reacher $( [ - 0 . 0 7 1 , - 0 . 0 1 4 ] )$ . Thus matching the state hel<u>ps within this bank.</u> Absolute sensitivity is also lower in the model. The median ratio $\sqrt { \mathrm { t r } ( G ^ { \mathrm { m o d e l } } ) / \mathrm { t r } ( G ^ { \mathrm { e n v } } ) }$ indicates about half the environment’s sensitivity on Cube and four-fifths on Reacher.

Visual geometry changes more with perturbation scale. A tenfold increase in perturbation size reduces median $D _ { G }$ by about 70% on Reacher and 47% on Cube. To investigate this scale dependence, we measure the same simulator responses in scaled physical pose, RGB pixels and frozen embeddings. Reacher’s median physical-pose Gram change is below one millionth, while its pixel and embedding changes are about 0.11 and 0.43. Cube shows the same ordering: approximately 0.03, 0.08 and 0.49, respectively.

These comparisons show how strongly the output metric affects the measurement. Rendering, encoding and nonlinear dynamics may each contribute; their individual effects remain unresolved. Physical-response outliers are retained (Appendix D). We report all three scales because changing the perturbation changes the response under study.

3

<!-- page 4 of 13 -->

Vasilev Nayak

Table 2: Held-out embedding response WAPE (%, lower is better), $\epsilon = 1 0 ^ { - 3 }$ . Isotropy uses the environment trace; oracle gain rescales the model Gram to that trace.

| Environment | Env. Gram | Isotropic | Model | Model + oracle gain |
| --- | --- | --- | --- | --- |
| Reacher | 17.0 | 26.2 | 33.2 | 26.6 |
| Cube | 28.0 | 38.2 | 56.4 | 32.3 |

The Gram predicts response sizes along held-out directions. Fo<u>r eigh</u>t held-out sphere directions per state, reused across scales, we compare $\hat { r } ( u )   =   \sqrt { u ^ { \top } G _ { \epsilon } u }$ with the centered response $r ( u ) = \lVert F ( u ) { - } F ( - u ) \rVert / 2$ . Coordinate Grams use none of these outcomes. We measure weighted absolute percentage error (WAPE) as $\textstyle \sum _ { j } | \hat { r } _ { j } - r _ { j } | / \sum _ { j } r _ { j }$ for each state, then report the median across states. At the primary scale, environment embedding Grams have lower error than isotropy, while model Grams have higher error (Table 2). The 95% intervals for the paired median advantage of environment Grams span roughly six to eight percentage points on Reacher and eight to twelve on Cube.

Gain error helps explain Cube’s different outcomes on the two tests. $D _ { G }$ compares relative geometry, while WAPE also measures response size. Giving the model the environment’s trace reduces median WAPE by about 24 percentage points, bringing it below isotropy and within five points of the environment Gram. This oracle correction uses simulator information, so its role is to separate gain error from the remaining mismatch.

Physical-pose prediction has median WAPE below 0.04% in both environments at the primary scale. Pixel errors are about 5% on Reacher and 15% on Cube. Predicting onesided embedding displacements is harder than predicting centered responses: median errors rise from 17% to 32% on Reacher and from 28% to 43% on Cube. Appendix D gives the full results and retained failures.

## 4. Discussion and Conclusion

LeWM captures part of the environment’s action structure, with agreement depending on perturbation scale and output metric. Held-out tests give the Gram a concrete interpretation: it predicts response magnitudes within the declared measurement space and scale. The original 50-state subsets gave identical repeated results. Later image differences required a documented change to the Cube replay protocol, which limits claims about reproducibility. Additional checkpoints are needed to estimate variation across training seeds.

The implications for training and control remain open. For example, $J _ { \mathrm { m o d e l } }   =   - J _ { \mathrm { e n v } }$ gives equal Grams and opposite outcome directions. Evaluating signed, goal-directed responses and closed-loop performance would establish whether better sensitivity estimates help action selection (Appendix B).

4

<!-- page 5 of 13 -->

Local Action Geometry in LeWM

## References

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture, 2023. URL [https://arxiv.org/abs/2301.08243](https://arxiv.org/abs/2301.08243).

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-jepa 2: Self-supervised video models enable understanding, prediction and planning, 2025. URL [https://arxiv.org/abs/2506.09985](https://arxiv.org/abs/2506.09985).

Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video, 2024. URL [https://arxiv.org/abs/2404.08471](https://arxiv.org/abs/2404.08471).

N Alex Cayco-Gajic and Arthur Pellegrino. Geometry-aware similarity metrics for neural representations on riemannian and statistical manifolds, 2026. URL [https://arxiv.org/abs/2603.28764](https://arxiv.org/abs/2603.28764).

David Ha and Jürgen Schmidhuber. World models. 2018. doi: 10.5281/ZENODO.1207631. URL [https://zenodo.org/record/1207631](https://zenodo.org/record/1207631).

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels, 2019. URL [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551).

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models, 2024. URL [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104).

Hansen Jin Lillemark, Benhao Huang, Fangneng Zhan, Yilun Du, and Thomas Anderson Keller. Flow equivariant world models: Memory for partially observed dynamic environments, 2026. URL [https://arxiv.org/abs/2601.01075](https://arxiv.org/abs/2601.01075).

Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. Leworldmodel: Stable end-to-end joint-embedding predictive architecture from pixels, 2026. URL [https://arxiv.org/abs/2603.19312](https://arxiv.org/abs/2603.19312).

5

<!-- page 6 of 13 -->

Vasilev Nayak

## Appendix A. Geometry Metrics and Additional Derivations

## A.1. Trace-normalized metric discrepancy

Relative directional geometry is invariant to positive rescaling of $G _ { t }$ . We therefore use $\widetilde { G } _ { t } = G _ { t } / \operatorname { t r } ( G _ { t } )$ and $D _ { G } ( t ) = \| \widetilde { G } _ { t } ^ { \mathrm { m o d e l } }   -   \widetilde { G } _ { t } ^ { \mathrm { e n v } } \| _ { F }$ . The primary trace-normalization constant is $\delta = 0$ . We mark zero or non-finite traces as invalid. We separately test additive constants $1 0 ^ { - 3 0 } , 1 0 ^ { - 1 2 } , 1 0 ^ { - 8 }$ Small $D _ { G }$ indicates similar relati<u>ve sen</u>sitivity and orientation. We assess absolute gain separately. We also report $S _ { b } = \sqrt { \mathrm { t r } ( G _ { t } ^ { b } ) }$ and $S _ { \mathrm { m o d e l } } / S _ { \mathrm { e n v } }$ . The shared frozen encoder gives this ratio a common latent scale. Its interpretation depends on that representation.

## A.2. Eccentricity

We define $\kappa ( G _ { t } ) \: = \: \lambda _ { \operatorname* { m a x } } ( G _ { t } ) / \lambda _ { \operatorname* { m i n } } ^ { \operatorname { e f f } } ( G _ { t } )$ , where $\lambda _ { \mathrm { m i n } } ^ { \mathrm { e f f } }   =   \operatorname* { m a x } \{ \lambda _ { \mathrm { m i n } } , 1 0 ^ { - 6 } \lambda _ { \mathrm { m a x } } , 1 0 ^ { - 1 2 } \}$ We report the fraction of floor-limited matrices; condition numbers in that subset should be read as capped estimates. For additional analysis we use $D _ { \kappa } ( t ) = \log \kappa ( G _ { t } ^ { \mathrm { m o d e l } } )   -   \log \kappa ( G _ { t } ^ { \mathrm { e n v } } )$ and $E _ { \kappa } ( t ) = | D _ { \kappa } ( t ) |$ . Positive $D _ { \kappa }$ indicates eccentricity amplification relative to the environment and negative values indicate eccentricity suppression.

## A.3. Directional and subspace alignment

For dominant eigenvectors we use $\theta _ { 1 } ( t ) = \cos ^ { - 1 } ( | ( v _ { t , 1 } ^ { \mathrm { e n v } } ) ^ { \top } v _ { t , 1 } ^ { \mathrm { m o d e l } } | )$ ; the absolute value removes the arbitrary sign of an eigenvector.

For a leading k-dimensional eigenspace, let $V _ { t , k } ^ { \mathrm { e n v } }   =   [ v _ { t , 1 } ^ { \mathrm { e n v } } ~ \cdots ~ v _ { t , k } ^ { \mathrm { e n v } } ]$ and define $V _ { t , k } ^ { \mathrm { m o d e l } }$ analogously. If $\sigma _ { i }$ are the singular values of $( V _ { t , k } ^ { \mathrm { e n v } } ) ^ { \top } V _ { t , k } ^ { \mathrm { m o d e l } }$ , then the principal angles are $\theta _ { i } = \cos ^ { - 1 } ( \sigma _ { i } )$ . We use the maximum principal angle for the leading two-dimensional Cube subspaces. Near eigenvalue ties, these directions are not reliably identified. We report the trace-normalized gaps $\widetilde { \lambda } _ { 1 } - \widetilde { \lambda } _ { 2 }$ and $\widetilde { \lambda } _ { 2 } - \widetilde { \lambda } _ { 3 }$ (descending eigenvalues), and condition angle summaries on both branches having gaps at least 0.05 or 0.1. These checks describe angle sensitivity; all states remain in the $D _ { G }$ analysis.

## A.4. Calibration and uncertainty

We compare the model with three reference discrepancies:

$$
\begin{aligned} { D _ { \operatorname { i s o } } ( t ) } & { { } = \| \widetilde { G } _ { t } ^ { \operatorname { e n v } } - I _ { d } / d \| _ { F } , } \\ { D _ { \operatorname { r o t } } ( t , Q ) } & { { } = \| \widetilde { G } _ { t } ^ { \operatorname { e n v } } - Q ^ { \top } \widetilde { G } _ { t } ^ { \operatorname { e n v } } Q \| _ { F } , } \\ { D _ { \operatorname { s h u f f l e } } ( t , \pi ) } & { { } = \| \widetilde { G } _ { t } ^ { \operatorname { m o d e l } } - \widetilde { G } _ { \pi ( t ) } ^ { \operatorname { e n v } } \| _ { F } . } \\ \end{aligned}
$$

Here Q is Haar-distributed orthogonal and π is a uniformly sampled derangement of the retained states. Rotation preserves the environment’s spectrum while randomizing its actioncoordinate orientation; shuffling tests whether matching the current state helps relative to another sampled state. Each reference removes a different source of information. Isotropy provides a direction-agnostic comparison. An isotropic matrix has no unique dominant direction, so its angle score is undefined.

6

<!-- page 7 of 13 -->

Local Action Geometry in LeWM

We use 1,000 draws, take each state’s median null discrepancy, then summarize across states. Paired differences subtract that per-state reference from the model discrepancy. Confidence intervals use 2,000 episode-cluster bootstrap resamples (seed 42). These intervals quantify uncertainty across sampled anchors. Estimating training-seed uncertainty requires more checkpoints. The first 50 eligible anchors receive three complete computations per scale. We report two-way single-measure absolute-agreement ICC for repeats and separately across scales, together with cross-scale Spearman rank correlations. Agreement across repeats does not imply agreement across perturbation scales.

## A.5. Anchor drift in the new bank

The coherent live state need not equal the archived step-10 state. For Cube, the median per-state maximum coordinate difference is $5 . 8 8 \times 1 0 ^ { - 4 }$ in qpos and $4 . 1 0 \times 1 0 ^ { - 4 }$ in qvel; the respective maxima are 0.0262 and 2.2505. These audit values retain the coordinates’ original units. We keep all high-drift anchors. Reacher’s maximum differences are below $3 \times 1 0 ^ { - 1 5 }$ , indicating near-identical replay of its archived states. Cube’s drift means that the new bank samples somewhat different trajectories.

## A.6. Environment-relative interpretation

A complementary theoretical comparison is

$$
R _ { t } = ( G _ { t } ^ { \mathrm { e n v } } + \eta I ) ^ { - 1 / 2 }   G _ { t } ^ { \mathrm { m o d e l } }   ( G _ { t } ^ { \mathrm { e n v } } + \eta I ) ^ { - 1 / 2 }   , \qquad \eta > 0 .
$$

For positive-definite $G _ { t } ^ { \mathrm { e n v } }$ , taking $\eta = 0$ and defining $x = ( G _ { t } ^ { \mathrm { e n v } } ) ^ { 1 / 2 } \Delta a$ gives $\Delta \boldsymbol { a } ^ { \top } G _ { t } ^ { \mathrm { e n v } } \Delta \boldsymbol { a } =$ $x ^ { \top } x$ and $\Delta a ^ { \top } G _ { t } ^ { \mathrm { m o d e l } } \Delta a = x ^ { \top } R _ { t } x$ . Thus $R _ { t }$ describes model geometry in coordinates where the environment metric is locally Euclidean. In this unregularized positive-definite case, if $G _ { t } ^ { \mathrm { m o d e l } } = c _ { t } G _ { t } ^ { \mathrm { e n v } }$ , then $R _ { t } = c _ { t } I$ . For Cube, $G _ { t } ^ { \mathrm { e n v } }$ is often close to singular, so inverse-metric quantities can amplify numerical noise; this is why $D _ { G }$ and principal subspace angles are our primary empirical comparisons.

## Appendix B. What Gram Agreement Does Not Establish

Gram agreement characterizes relative response magnitudes. Outcome direction remains ambiguous: for example, $J _ { \mathrm { m o d e l } } = - J _ { \mathrm { e n v } }$ gives identical Gram matrices but opposite latent responses. More generally, left multiplication by an orthogonal output-space map leaves $J ^ { \top } J$ unchanged. Trace normalization also removes overall sensitivity scale. Goal-directed action preferences, planning efficiency and closed-loop success therefore require separate measurements. Isotropy remains useful as a comparison baseline. In an anisotropic environment, using it as a training target would suppress genuine differences in action sensitivity.

## Appendix C. Bank Construction and Robustness

Matched transition construction. The current-simulator bank uses MuJoCo 3.11.0, OGBench 1.2.1 and $2 2 4   \times   2 2 4$ images, with an opaque Cube arm and hidden target markers. Each eligible anchor must reproduce its live context image exactly after snapshot restoration, and two unperturbed branch rollouts must agree in pixels exactly and in qpos/qvel

7

<!-- page 8 of 13 -->

Vasilev Nayak

Table 3: New-bank finite-scale geometry at $\epsilon \; = \; 1 0 ^ { - 3 } .$ Values are median [IQR] where reported; N = 458 Reacher, N = 500 Cube. Lower $D _ { G }$ and angles indicate closer agreement; κ describes anisotropy.

| Env. | $d _ { a }$ | $D _ { G }$ | $\kappa _ { \mathrm { e n v } }$ | $\kappa _ { \mathrm { m o d e l } }$ |  | Top-1 | Top-2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | Reacher 2 0.465 [0.270,0.708] | $4 . 4 9$ | $4 . 5 0$ |  | $3 6 . 0 3 [ 1 8 . 0 4 , 5 3 . 8 5 ] ^ { \circ }$ |  |
| Cube | 5 | 0.508 [0.397, 0.633] |  |  |  |  | 2.84×103 1.07×10441.52 [25.03, 66.09]°40.27 [27.13, 58.12]° |

within $1 { \overline { { 0 ^ { - 1 2 } } } } ,$ . The bank retains context images, counterfactual images and embeddings, response matrices, anchor identifiers, and checkpoint/source hashes. Historical results remain archived. Full native simulator data and mutable wrapper state are restored before each branch without an intervening forward call. These are the primary-bank checks; the later response-validation replay protocol is separately documented in Appendix D.4.

Cube finite-difference scale. Because rendering and contacts can introduce finite-scale effects, we repeat both model and environment measurements over a ten-fold perturbation range. Table 4 reports Cube results; the cross-environment trends appear in Figure 2. Each scale defines its own finite-response measurement; convergence to a smooth derivative remains untested.

Table 4: Cube robustness across finite-difference scales.

| ϵ | Median D<sub>G</sub> | Top-1 angle | Top-2 max angle |
| --- | --- | --- | --- |
| 10<sup>-3</sup> | 0.508 | 41.52◦ | 40.27◦ |
| 3×10<sup>-3</sup> | 0.408 | 34.82◦ | 30.38◦ |
| 10<sup>-2</sup> | 0.271 | 20.96◦ | 18.48◦ |

Scope of the revision. The corrected Reacher measurements do not support the historical direction-preserving/eccentricity-amplifying interpretation: both median condition numbers are near 4.5, but the median dominant-direction angle is 36.03◦. Primary-scale paired model-minus-isotropic $D _ { G }$ differences are +0.083 (95% interval [0.064, 0.108]) for Reacher and −0.108 ([−0.128, −0.094]) for Cube. Paired median differences need not equal differences of separately computed medians.

We do not transfer historical gripper-contact labels to the new trajectories. The earlier contact/non-contact figure and claim are therefore not retained as evidence for this bank. Resolving contact-specific effects requires measuring contact events along these live counterfactuals.

Repeated runs agree, while changing scale changes the result. On Cube’s first 50 eligible anchors, all three repeats produced identical response matrices and $D _ { G }$ at every scale (absolute-agreement ICC 1). Across scales, however, $D _ { G }$ has ICC 0.220 and pairwise Spearman correlations 0.220–0.432. Between $1 0 ^ { - 3 }$ and $1 0 ^ { - 2 }$ , the environment’s dominant direction changes by a median 38.99◦, versus $0 . 0 1 ^ { \circ }$ for the model. The model’s leading direction stays nearly fixed while the environment embedding’s direction rotates by almost 39◦. This supports reporting each scale separately. Rendering, nonlinear dynamics and contact effects remain possible contributors. Reacher likewise repeats exactly on its 50- anchor subset, while its median $D _ { G }$ falls from 0.465 to 0.308 to 0.141 across the three

8

<!-- page 9 of 13 -->

Local Action Geometry in LeWM

![Chart block](doc:8b3b37b/tier:basic/page:9/block:2)

![Chart block](doc:8b3b37b/tier:basic/page:9/block:3)

![Chart block](doc:8b3b37b/tier:basic/page:9/block:4)

Figure 2: Geometry measurements change with perturbation scale. Lines show medians; shading covers the middle half of states. Each environment uses the same eligible states at every scale.

scales; dominant angles fall from $3 6 . 0 3 ^ { \circ }$ to $2 0 . 7 0 ^ { \circ }$ to $8 . 8 9 ^ { \circ }$ . Its cross-scale ICC is 0.200 and rank correlations range from 0.205 to 0.375.

Spectral gaps and numerical floors. At the primary scale, requiring both Cube top-1 gaps to be at least 0.1 retains 449 states with median angle 41. $. 0 9 ^ { \circ } ;$ the corresponding top-2 gap check retains 302 with median maximum angle $3 4 . 6 6 ^ { \circ }$ . The mismatch therefore is not explained solely by nearly tied eigenspaces. The condition-number floor is active for 32.4% of Cube environment metrics and none of its model metrics, limiting interpretation of the upper tail. Adding a trace constant up to $1 0 ^ { - 8 }$ changes Cube $D _ { G }$ by less than $9 \times 1 0 ^ { \circ }$ −1<sup>0</sup> at all three scales. Large condition numbers do not, by themselves, imply a small trace or a dominant trace regularizer. Neither Reacher branch is floor-limited at the primary scale; its maximum trace-constant effect is below $3 \times 1 0 ^ { - 1 1 }$ across scales. Requiring both Reacher top-1 gaps to exceed 0.1 retains 454 states with median angle $3 5 . 8 0 ^ { \circ }$

Camera-ready protocol revision. These measurements supersede the historical esti mates under a revised protocol. An audit found action-block/output-coordinate inconsistencies in the available evaluation code and incomplete historical simulator-state restoration. We therefore use the original datasets as sampling anchors for a new current-simulator bank: initialize at each anchor’s context start, execute ten recorded actions, and render fresh context frames at steps 0, 5 and 10. Counterfactuals branch from that live step-10 state, which can differ from the archived state. This keeps context and outcomes internally consistent within the current runtime. The original collection runtime remains unavailable. Both branches now use matched finite differences; the earlier description paired model autodiff with environment finite differences.

## Appendix D. Output-Space and Held-Out Response Validation

## D.1. Separate output metrics

We repeat the coordinate and held-out perturbations on the same eligible anchors and at all three declared scales. The physical-pose metric uses MuJoCo’s generalized-position tangent from the unperturbed endpoint to each perturbed endpoint, computed with mj differentiatePos at unit time. Translational components are divided by 0.1 m and rotational components by

9

<!-- page 10 of 13 -->

Vasilev Nayak

1 rad, then the vector is divided by $\sqrt { n _ { v } }$ . Here $n _ { v } = 2$ for Reacher and $n _ { v } = 2 0$ for Cube, including robot, gripper and object coordinates. This avoids treating quaternion signs as physical displacements, but still makes an explicit, non-unique choice of physical units and coordinate weights.

The supplemental pose-plus-velocity metric concatenates this vector with $T ( v   -   v _ { 0 } )$ using the same component scaling and $1 / \sqrt { n _ { v } }$ factor, then divides the concatenation by ${ \sqrt { 2 } } ;   T$ is the five-step horizon. Pixels use flattened RGB values divided by $2 5 5 \sqrt { N _ { \mathrm { R G B } } }$ . Embeddings use the frozen encoder’s Euclidean metric without trace normalization of output vectors. Grams are constructed independently in each space. Gain depends on the chosen units and coordinate weights, so comparisons across spaces require care.

Within each output metric, scale dependence is measured by $\| \widetilde { G } _ { t , 0 . 0 0 1 } ^ { \mathrm { e n v } } - \widetilde { G } _ { t , 0 . 0 1 } ^ { \mathrm { e n v } } \| _ { F } ,$ , summarized across anchors. This measures how the Gram changes between two perturbation scales.

## D.2. Held-out directions and error definitions

For each anchor, draw eight independent normalized Gaussian vectors $v _ { j }$ on the unit sphere in raw half-range coordinates, using seed 20261021 + sample ID (plus 100000 for Reacher). Define

$$
u _ { j } = \operatorname { d i a g } ( h _ { 1 } , \ldots , h _ { d } ) v _ { j } , \qquad \hat { r } _ { j } = \sqrt { u _ { j } ^ { \top } G _ { \epsilon } u _ { j } } , \qquad r _ { j } = \left\| \frac { F ( u _ { j } ) - F ( - u _ { j } ) } { 2 } \right\| .
$$

The same $v _ { j }$ are reused across scales. No held-out endpoint is used to construct the coordinate Gram or choose its gain. The primary per-state error is

$$
\mathrm { W A P E } = \frac { \sum _ { j = 1 } ^ { 8 } | \hat { r } _ { j } - r _ { j } | } { \sum _ { j = 1 } ^ { 8 } r _ { j } } .
$$

We mark a zero denominator as invalid. Each state contributes one WAPE value, which combines its eight directions. Paired median-difference intervals use 2,000 episode-cluster bootstrap draws with directions held fixed. Their scope is uncertainty across sampled states and episodes; training-seed and direction-resampling uncertainty require additional runs.

For the one-sided check, define $r _ { j } ^ { \pm } = \| F ( \pm u _ { j } ) - F ( 0 ) \|$ and compare the same $\hat { r } _ { j }$ with both values, summing errors and actual magnitudes over all sixteen endpoints. A centered secant can cancel effects that remain in a one-sided displacement; the two tests are not equivalent.

## D.3. Magnitude calibration and all-scale results

The magnitude reference is $G _ { \mathrm { i s o } }   =   \mathrm { t r } ( G _ { \mathrm { e n v } } ) I _ { d } / d .$ . For embedding targets we also test the unnormalized model Gram and

$$
G _ { \mathrm { m o d e l } } ^ { \mathrm { o r a c l e } } = G _ { \mathrm { m o d e l } } \frac { \mathrm { t r } \big ( G _ { \mathrm { e n v } } \big ) } { \mathrm { t r } \big ( G _ { \mathrm { m o d e l } } \big ) } .
$$

Both references use environment trace information. The oracle correction helps quantify gain error. Deploying this correction would require access to the environment trace. Isotropy

10

<!-- page 11 of 13 -->

Local Action Geometry in LeWM

is defined in normalized action coordinates, whereas directions are sampled in raw half-range coordinates, so its predicted magnitude need not be constant across the sampled directions. Unlike $D _ { G }$ , WAPE includes gain and the finite-scale approximation error on new directions.

Table 5: Median embedding response WAPE (%) at all scales. The one-sided column uses the environment Gram. Lower is better.

| Environment | ϵ | Env. Gram | Isotropic | Model | Oracle gain | One-sided |
| --- | --- | --- | --- | --- | --- | --- |
| Reacher | 0.001 | 17.0 | 26.2 | 33.2 | 26.6 | 32.2 |
| Reacher | 0.003 | 11.3 | 22.2 | 20.6 | 18.0 | 23.9 |
| Reacher | 0.01 | 6.8 | 20.2 | 10.3 | 9.2 | 13.8 |
| Cube | 0.001 | 28.0 | 38.2 | 56.4 | 32.3 | 42.6 |
| Cube | 0.003 | 21.6 | 34.5 | 39.1 | 25.1 | 33.9 |
| Cube | 0.01 | 13.5 | 32.9 | 23.9 | 16.9 | 22.7 |

At the primary scale, the model’s paired median WAPE exceeds isotropy by roughly five to seven percentage points on Reacher and fourteen to nineteen on Cube (95% intervals). After oracle gain correction, Reacher’s excess error narrows to about one to three points, while Cube gains an advantage of about four to six points. The exact intervals, in WAPE fractions, are [0.046, 0.069] and [0.138, 0.189] before correction, and [0.007, 0.031] and [−0.065, −0.036] afterward, respectively. These comparisons show why response magnitude and normalized geometry need separate evaluation.

Table 6: Distinct environment metrics: median normalized Gram change from $\epsilon = 0 . 0 0 1$ to 0.01, and primary-scale held-out centered WAPE. WAPE is reported as a fraction (multiply by 100 for percent).

| Environment | Output metric | Gram change | WAPE |
| --- | --- | --- | --- |
| Reacher | Physical pose | 9.32 × 10<sup>-7</sup> | 2.97 × 10<sup>-9</sup> |
| Reacher | Pose + velocity | 2.11 × 10<sup>-6</sup> | 6.37 × 10<sup>-9</sup> |
| Reacher | Pixels | 0.1125 | 0.0498 |
| Reacher | Embedding | 0.4331 | 0.1702 |
| Cube | Physical pose | 0.0270 | 0.0004 |
| Cube | Pose + velocity | 0.0266 | 0.0004 |
| Cube | Pixels | 0.0808 | 0.1483 |
| Cube | Embedding | 0.4853 | 0.2801 |

Retained failures and interpretation. One Reacher state has a large physical-prediction error: sample 389 reaches about 117% WAPE at the primary scale. The other 457 states are below 1%. Several endpoints show a much larger angular response; the physical cause remains unresolved. Cube’s physical-pose cross-scale Gram change has upper quartile 0.111 and maximum 1.172 despite a median of 0.027. All such states remain in the analysis. Cube’s median physical-pose WAPE rises to about 1.47% at $\epsilon = 0 . 0 1$ . These results qualify median summaries and preclude attributing every visual scale effect solely to rendering.

11

<!-- page 12 of 13 -->

Vasilev Nayak

![Chart block](doc:8b3b37b/tier:basic/page:12/block:2)

Figure 3: Left: how much the environment Gram changes with perturbation scale in each output space. Right: error in predicting held-out embedding response magnitudes. Lines show medians. Reacher v2 and Cube v3 identify replay protocols; each uses its environment’s original LeWM checkpoint. Oracle references use the environment Gram’s trace.

## D.4. Replay protocol amendment and verification

The primary discrepancy bank and its 50-anchor repeatability checks are distinct from this subsequent response-validation run. The latter retains all 458 Reacher and 500 Cube anchors, with 10,992 and 12,000 direction–scale pairs. Fresh environment coordinate endpoints are evaluated alongside held-out endpoints. Reacher uses the v2 protocol with exact parent context and baseline images and had no replay exceptions.

Cube’s initial strict replay attempts encountered intermittent one-level RGB differences, first at a coordinate endpoint and later in context images. Partial v1/v2 runs are archived and are not pooled with production. The separately versioned v3 protocol permits at most one RGB intensity level of parent context/baseline difference, records the changed values, and recomputes model responses from the fresh context. Relative model response change versus the parent must not exceed 1%; baseline embedding replay change divided by the coordinate centered-response norm must not exceed 1%. Coordinate image mismatches trigger repeated rollouts with the same one-level image and 1% embedding-response gates.

12

<!-- page 13 of 13 -->

Local Action Geometry in LeWM

Physical qpos/qvel tolerances remain 10−12. We introduced these gates after observing replay differences.

Production v3 stopped at anchor 292 with a baseline embedding replay ratio of 1.193%, above the fixed 1% threshold. An isolated unchanged rerun matched exactly; one unchanged production resume then completed the bank. The thresholds and full set of an chors remained unchanged. This retry qualifies cross-run reproducibility; the cause of the intermittent discrepancy remains unknown.

In accepted v3 outputs, two contexts changed by at most three scalar RGB values, each by one level; baseline images matched exactly. Maximum relative model-response change was 0.02525%, and maximum normalized model-Gram change was $4 . 3 4 \times 1 0 ^ { - 5 }$ . Ten coordinate audit records across six states had maximum relative embedding replay difference 0.32462%; repeated qpos/qvel errors were zero. These accepted-output bounds describe the completed outputs. The earlier failure remains part of the reproducibility record. Five dedicated response-validation tests pass. Independent analysis verified artifact hashes and reconstructed Gram predictions and pixel/embedding response magnitudes, with maximum reconstruction error zero.

13