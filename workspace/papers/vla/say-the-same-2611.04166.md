<!-- page 1 of 23 -->

arXiv:2611.04166v1 [cs.LG] 3 Nov 2026

# SAY THE SAME, ACT DIFFERENTLY: Diagnosing and Steering Text–Action Divergence in Reasoning VLAs

**Zihao Feng**<sup>1</sup> **Qingzhao Zhang**<sup>2</sup> **Chunyu Xia**<sup>1</sup> **Bo Yu**<sup>3</sup> **Z. Morley Mao**<sup>1</sup> **Ramesh Govindan**<sup>1</sup>

<sup>1</sup>University of Southern California <sup>2</sup>University of Arizona <sup>3</sup>General Motors {zihaofen, chunyuxi, morley.mao, ramesh}@usc.edu qzzhang@arizona.edu bo.3.yu@gm.com

## Abstract

Reasoning vision-language-action (VLA) models pair physical control with naturallanguage rationales that supervisors can read to monitor embodied safety. Prior attacks on VLA reasoning corrupt the rationale itself and leave visible evidence in the text. We show that visual perturbations can instead alter the physical action while keeping the generated rationale word-for-word identical. At the shared text/action interface of Alpamayo for driving and InstructVLA for manipulation, our text-neutral action-subspace (TNAS) diagnostic shows that the locally most action-sensitive direction is nearly orthogonal to the active text-loss gradients, with 1.87% and 5.28% norm overlap. At equal hidden-state perturbation norms across both models, interventions along this text-neutral residual shift actions 4.64× to 47.8× more than text-gradient perturbations, whereas text-gradient perturbations induce 21.2× to 277× more text drift. Steering visual perturbations along this direction reaches the same decoupling from the image. Across 154 obstacle-annotated driving scenes in Alpamayo, TNAS-guided attacks induce a new collision under a word-for-word identical rationale in 52.6% of scenes under full-image perturbations and 35.1% under a localized forward-camera patch, compared with 4.55% for unguided action attacks. Relative to a text-preserving baseline, TNAS guidance raises exact rationale preservation from 83.0% to 89.5% and cuts teacher-forced text drift by 68.4% at a comparable collision rate. In InstructVLA, TNAS guidance cuts text drift by 40.5% and preserves the exact response on 85.0% of examples alongside a 1.73-unit action-chunk shift. An unchanged rationale therefore cannot certify a safe action, so monitors for reasoning VLAs must verify the predicted action channel directly.

## 1 Introduction

Vision-language-action (VLA) models translate visual observations and language instructions into embodied actions. Reasoning VLAs additionally produce natural-language scene descriptions, rationales, or plans alongside continuous action predictions [1–5]. Unlike latent representations or raw motor commands, this language channel gives human supervisors an explicit, readable account of what the model perceives and intends to do. A supervisor can inspect that account for a missed obstacle, an incorrect target object, or an intention that conflicts with the instruction, potentially before the action is executed. The same idea drives chain-of-thought monitoring for language agents [6, 7], and for embodied models the stakes are even higher because an erroneous physical motion can cause collisions or physical damage. Whether this interface can be trusted for safety hinges on one question: can the action become unsafe without any change in what the model says?

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">40th Conference on Neural Information Processing Systems (NeurIPS 2026).</span></small>

<!-- page 2 of 23 -->

![Image block](doc:e4f9533/tier:basic/page:2/block:1)

Figure 1: Rationale-preserving action divergence in reasoning VLAs. On a clean driving input, the VLM and action decoder read the same unobservable hidden state to produce a matching rationale and stopping action. Adding an imperceptible perturbation to the input shifts that shared hidden state along a text-neutral, action-sensitive direction. The model still outputs “Stop to keep distance to the lead vehicle” word for word, but its predicted action flips from stopping to continuing into the lead vehicle.

Recent attacks on reasoning VLAs redirect behavior by changing the reasoning itself. TRAP steers chain-of-thought and downstream actions toward an adversary’s target with adversarial patches [8], while Trinh et al. [9] corrupt the intermediate reasoning trace directly, for example by substituting object names. Both leave evidence in the channel a monitor reads. Because the reasoning text has changed, an incorrect object reference or an instruction-inconsistent plan is visible to a supervisor. In contrast, as illustrated in Figure 1, we study rationale-preserving action divergence, where a perturbed input changes the predicted action while leaving the generated rationale word-for-word identical to the clean run. When a perturbed driving input produces a colliding plan under the exact same rationale as the clean input, a monitor restricted to that rationale receives identical evidence in both cases.

This failure mode is rooted in how reasoning and action share a representation. In the reasoning VLAs studied here, action prediction reads continuous hidden states that also produce the text [4, 5]. Because the rationale is a discrete, many-to-one readout of those continuous states, many distinct hidden states can map to the same generated text. Under ordinary inputs the text and action co-vary, which makes rationales informative in typical operation. For a rationale monitor to certify safety, however, any unsafe change in the action must also change the rationale. Prior reasoning-hijacking attacks show that altering the text alters the action, leaving open the converse question: does the shared hidden state contain action-sensitive variation that leaves the text readout unchanged, and can visual inputs reach it?

We answer this question with the TEXT-NEUTRAL ACTION-SUBSPACE (TNAS) diagnostic and a corresponding input-level attack. At the shared text–action interface, TNAS identifies the locally most action-sensitive direction in the hidden state, measures its overlap with the active text-loss gradients, and projects out that text-gradient component to isolate a text-neutral action direction $( d _ { \mathrm { T N A S } } )$ . We first perturb the clean hidden state directly along $d _ { \mathrm { T N A S } }$ and control directions, under both fixed text-drift budgets and equal perturbation norms, to verify that this local direction shifts the predicted action while leaving the token distribution nearly unchanged. We then align image perturbations with $d _ { \mathrm { T N A S } }$ during input-level optimization to steer the model toward silent action divergence under paired free generation. Across both Alpamayo for autonomous driving and InstructVLA for robotic manipulation, we find the same geometric decoupling and summarize our main contributions as follows:

## Contributions.

• **Rationale-preserving action divergence through visual inputs.** We demonstrate that visual perturbations can substantially alter a reasoning VLA’s action while preserving the exact generated rationale. In Alpamayo, TNAS-guided perturbations introduce a geometric collision absent from the clean plan while leaving the rationale word-for-word unchanged in 52.6% of annotated scenes under full-image perturbations and 35.1% under a localized forward-camera patch, and in InstructVLA, they preserve the exact response in 85.0% of examples with a mean action-chunk shift of 1.73 normalized units.

2

<!-- page 3 of 23 -->

• **A local geometric mechanism for text–action decoupling.** We introduce TNAS and show that the action-sensitive direction at the shared interface overlaps the active text-loss span by only 1.87% of its norm in Alpamayo and 5.28% in InstructVLA. Under equal perturbation norms across both models, perturbing the hidden state along the projected text-neutral direction shifts actions 4.64× to 47.8× more than text-gradient perturbations, whereas text-gradient perturbations induce 21.2× to 277× more teacher-forced KL.

• **Mechanism-guided attacks with stronger text preservation at the same level of action deviation.** Directly combining an action-deviation loss and a text-preservation penalty creates competing gradients that interfere with each other during optimization. Adding TNAS alignment to the attack objective resolves this conflict by steering hidden-state updates along the text-neutral direction: at the strongest guidance weights, exact rationale preservation rises from 83.0% to 89.5% in Alpamayo and from 77.5% to 85.0% in InstructVLA, while teacher-forced text drift falls by 68.4% and 40.5% at an unchanged driving collision rate.

## 2 TEXT-NEUTRAL ACTION-SUBSPACE Diagnostic

## 2.1 Setup and probe construction

**Models and cohorts.** We evaluate Alpamayo, an autonomous driving VLA, on 200 clips and InstructVLA, a robotic manipulation VLA, on 200 Bridge-distribution examples, both fixed as described in Appendix A.1. Alpamayo is our primary safety case study because its predicted trajectory can be scored against logged traffic actors on 154 obstacle-annotated clips. With InstructVLA, we test whether the same geometry holds in a second embodiment that uses a flow-matching action expert and concise task-intention responses.

**Shared-state hook and intervention coordinates.** For a block of K transformer layers of hidden width d and an active reasoning- or response-token span R, an additive intervention $\delta \; \in \; \mathbb { R } ^ { K d }$ broadcasts one vector per layer over R. A usable block must expose both a nonzero text basis and a strong action interface in the same state. As detailed in Appendix A.4.1, selecting the qualifying block with the strongest action response gives reasoning-token layers 32–35 in Alpamayo and response-token layers 14–17 in InstructVLA. The pooled state $h \in \mathbb { R } ^ { \check { K } d }$ concatenates per-layer token means on R and supplies the input-level alignment readout.

**Action and text readouts.** As detailed in Appendix A.2.1, under the hooked forward pass, both action heads decode with a fixed 10-step Euler flow-matching solver and paired initial noise: Alpamayo integrates acceleration–curvature controls into 2D planar waypoints $\dot { \tau } _ { t } ( \delta ) \in \mathbb { R } ^ { 2 }$ , and InstructVLA predicts normalized 7-DoF end-effector actions $a _ { t } ( \dot { \delta } ) \in \mathbb { R } ^ { 7 }$ (3D position, 3D rotation, and gripper). We measure action change as the mean displacement from the clean output $( \delta = 0 )$ , over $T = 6 4$ waypoints at 10 Hz (6.4 s, in meters) for Alpamayo and over $H = 1 6$ chunk steps for InstructVLA:

$$
\Delta _ { \operatorname { A D E } } ( \delta ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left\| \tau _ { t } ( \delta ) - \tau _ { t } ( 0 ) \right\| _ { 2 } , \qquad \Delta _ { \operatorname { I V L A } } ( \delta ) = \frac { 1 } { H } \sum _ { t = 1 } ^ { H } \left\| a _ { t } ( \delta ) - a _ { t } ( 0 ) \right\| _ { 2 } .\tag{1}
$$

These fixed-prefix readouts hold the clean rationale or response fixed as the text prefix, so the paired clean and perturbed passes differ only through the intervention. As defined in Appendix A.2.2, text drift is the forward teacher-forced KL divergence, from the clean to the perturbed next-token distribution, averaged over R.

**Action-Jacobian direction.** The action-Jacobian direction $d _ { \mathrm { a c t } }$ is a hidden-state direction to which the action output is locally sensitive. We do not take the gradient of $\Delta _ { \mathrm { A D E } }$ or $\Delta _ { \mathrm { I V L A } }$ directly. Both are norms of a displacement that is zero at $\delta = 0$ , so neither has a well-defined gradient at the clean point. We instead work with the Jacobian of the action output itself. Let $A _ { \mathrm { h o o k } } ( \delta )$ be the flattened action output of the hooked pass, namely the 64 XY waypoints for Alpamayo and the $1 6 \times 7$ action chunk for InstructVLA, with local Jacobian

$$
J _ { A } = \left. \frac { \partial A _ { \mathrm { h o o k } } ( \delta ) } { \partial \delta } \right| _ { \delta = 0 } .\tag{2}
$$

As detailed in Appendix A.4, we obtain $d _ { \mathrm { a c t } }$ by backpropagating through the action readout to the hooked state and normalizing the resulting row-space direction of $J _ { A }$ . The same direction then goes through text-span residualization and the hidden-state intervention tests.

3

<!-- page 4 of 23 -->

**Active text-loss span.** The text side needs losses whose gradients with respect to $\delta$ are nonzero at the clean point $\delta = 0$ . Teacher-forced KL and clipped hinge losses do not qualify, because the KL gradient vanishes when the clean and perturbed distributions coincide and hinge losses saturate with zero gradient whenever the clean token already clears the margin threshold. We therefore capture local text sensitivity through two complementary losses that remain active at $\delta = 0 .$ , namely target negative log-likelihood (NLL), which tracks the probability mass assigned to the clean token across the vocabulary, and signed target-margin drop, which tracks the logit gap to the strongest competing token that governs top-1 token flips. For token $t \in R$ with clean target $y _ { t }$ and logits $z _ { t } ,$ let $p _ { t } = \mathrm { s o f t m a x } ( z _ { t } )$ and $m _ { t } = z _ { t , y _ { t } } - \operatorname { m a x } _ { v \neq y _ { t } } z _ { t , v }$ . Writing superscripts cln and adv for the clean and perturbed passes, the two losses are

$$
\mathcal { L } _ { \operatorname { N L L } } ( \delta ) = \frac { 1 } { | R | } \sum _ { t \in R } - \operatorname { l o g } p _ { t , y _ { t } } ^ { \operatorname { a d v } } , \qquad \mathcal { L } _ { \operatorname { m a r g i n } } ( \delta ) = \frac { 1 } { | R | } \sum _ { t \in R } \bigl ( m _ { t } ^ { \operatorname { c l n } } - m _ { t } ^ { \operatorname { a d v } } \bigr ) .\tag{3}
$$

Gram–Schmidt on their clean-state gradients g<sub>NLL</sub> and $g _ { \mathrm { m a r g i n } }$ gives an orthonormal basis of the text-loss span $\mathcal { T } = \operatorname { s p a n } \{ g _ { \operatorname { N L L } } , g _ { \operatorname { m a r g i n } } \}$ , which has rank 2 on every evaluated example.

**TNAS residual.** With $P _ { T }$ the orthogonal projector onto $\mathcal { T } ,$ the text-neutral action-subspace (TNAS) direction is

$$
d _ { \operatorname { T N A S } } = \operatorname { n o r m a l i z e } \bigl ( ( I - P _ { \mathcal { T } } ) d _ { \operatorname { a c t } } \bigr ) .\tag{4}
$$

By construction, $d _ { \mathrm { T N A S } }$ is orthogonal to both text-loss gradients at $\delta = 0$ , so a small step along it leaves the NLL and margin of the clean rationale unchanged to first order while still moving the action.

## 2.2 Clean-state geometry

## Probe 1. How much of the action does the text carry?

**Rationale-only action prediction**. To evaluate the consistency between the rationale and the action under ordinary inputs, we use a frozen Qwen3-14B model [10], configured as described in Appendix A.5, to predict from the clean rationale alone whether Alpamayo’s planned trajectory comes to rest within the 6.4 s horizon. Across the 1,181 clips, where 24.5% (289/1,181) of planned trajectories actually come to rest, the reader’s prediction matches the trajectory outcome on 84.3% of clips. Clips predicted to stop come to rest 63.6% of the time, compared with 5.76% for clips predicted to continue. Even so, the text does not pin down the action. The 1,181 clips share only 258 distinct rationales, 36.9% carry a rationale that also appears with the opposite stopping outcome, and the most frequent rationale (“Keep distance to the lead vehicle since it is directly ahead in our lane”) alone covers 245 clips with 10.1 m to 208 m of forward displacement.

**Hidden-state gradient projection**. The hidden-state geometry locates this gap at the shared interface. To quantify how much of the action gradient lies in the text span, we project the action direction $d _ { \mathrm { a c t } }$ onto $\dot { \mathcal { T } }$ and divide by the norm of $d _ { \mathrm { a c t } }$ , giving the projection-norm ratio $\rho = \| P _ { \mathcal { T } } d _ { \mathrm { a c t } } \| / \| d _ { \mathrm { a c t } } \|$ Across the 200 Alpamayo clips, $\rho$ averages 1.87% (versus $\rho = 1$ for text-gradient controls), roughly twice the 0.978% mean of isotropic random unit vectors reported in Appendix $\mathrm { A . 6 , }$ and across the 200 InstructVLA examples, $\rho$ averages 5.28%. Because the text-span project<u>ion</u> $P _ { \mathcal { T } } d _ { \mathrm { a c t } }$ and its orthogonal complement $( I - P _ { \mathcal { T } } ) d _ { \mathrm { a c t } }$ are perpendicular, removing $\mathcal { T }$ leaves $\sqrt{1 - \rho^{2}} > 99.8\%$ of $d _ { \mathrm { a c t } } { } ^ { \prime } \mathrm { s }$ norm intact on average in both models, meaning that almost the entire action-sensitive direction is orthogonal to the text span.

**Takeaway 1. The rationale carries action information, but the text registers only a small part of what moves the action.** Read alone, Alpamayo’s rationale predicts whether the plan stops on 84.3% of clips, yet 36.9% of clips share their rationale with the opposite outcome. At the shared interface, the projection-norm ratio $\rho$ of $d _ { \mathrm { a c t } }$ onto the active text span averages 1.87% in Alpamayo and 5.28% in InstructVLA, so projecting out the text span retains on average more than 99.8% of the action direction’s norm in both models.

## 2.3 Hidden-state intervention results

Probe 1 measures first-order gradients at $\delta = 0$ . To test whether this local separation holds when finite hidden-state perturbations pass through the nonlinear language and action heads, Probe 2 and Probe 3 compare $d _ { \mathrm { T N A S } }$ and the unprojected action direction $d _ { \mathrm { a c t } }$ against three non-action control directions, namely the text-loss gradients $( g _ { \mathrm { N L L } } ,   g _ { \mathrm { m a r g i n } } )$ and random unit vectors $( u _ { \mathrm { r a n d } } )$ , on the

4

<!-- page 5 of 23 -->

a

Probe 2: fixed text KL

![Chart block](doc:e4f9533/tier:basic/page:5/block:3)

b

Probe 3: direction-specific causal control

![Chart block](doc:e4f9533/tier:basic/page:5/block:6)

![Chart block](doc:e4f9533/tier:basic/page:5/block:7)

![Chart block](doc:e4f9533/tier:basic/page:5/block:8)

![Chart block](doc:e4f9533/tier:basic/page:5/block:9)

![Chart block](doc:e4f9533/tier:basic/page:5/block:10)

![Chart block](doc:e4f9533/tier:basic/page:5/block:11)

![Chart block](doc:e4f9533/tier:basic/page:5/block:12)

![Chart block](doc:e4f9533/tier:basic/page:5/block:13)

Figure 2: Hidden-state intervention probes. Top: schematic of the two protocols $( g _ { \mathrm { t e x t } }$ denotes g<sub>NLL</sub> and $g _ { \mathrm { m a r g i n } } )$ . (a) Probe 2 adjusts each direction’s step size so that all directions incur the same text-KL budget B (shaded) and compares the resulting action shift. (b) Probe 3 adds every direction at the same per-example norm $| q | S _ { \mathrm { c l i p } }$ for $q \in \{ 0 . 5 , 1 , 2 \}$ (shaded, light to dark) and compares the resulting text KL and action shift. Middle and bottom: cohort means on 200 Alpamayo clips and 200 InstructVLA examples, with full values tabulated in Appendix A.7. (c,f) Action shift at fixed teacher-forced KL budgets B. (d,g) Action shift and (e,h) teacher-forced text KL across equal-norm strengths q, where $d _ { \mathrm { a c t } }$ (hollow diamonds) nearly overlaps d<sub>TNAS</sub>.

same 200 Alpamayo clips and 200 InstructVLA examples. Figure 2 summarizes both probes, and Appendices A.4 and $\mathrm { A } . \bar { \mathcal { I } }$ provide the implementation details and full tables. As shown in Figure 2, Probe 2 compares these directions under a fixed text-drift budget in panel a, and Probe 3 compares them under a fixed hidden-state perturbation norm in panel b.

**Probe 2. Fixed text KL.** To test how much action change a hidden-state perturbation can induce before noticeably altering the rationale, we hold teacher-forced text KL fixed across directions and compare the resulting action shifts. As illustrated in Figure $\mathrm { 2 a , } \mathrm { c , } \mathrm { f , }$ for each direction, we select the step size on the perturbation grid whose teacher-forced KL matches a target budget B.

**Alpamayo**. Perturbing along d<sub>TNAS</sub> shifts the planned trajectory by 0.227 m at $\widetilde { B } = 1 0 ^ { - 4 }$ and 0.504 m at $\dot { B = 5 \times 1 0 ^ { - 4 } }$ , matching $d _ { \mathrm { a c t } }$ and exceeding the controls $\widehat { ( \leq 5 . 1 0 }$ mm) by 63.3× to 212×.

**InstructVLA**. Perturbing along d<sub>TNAS</sub> and $d _ { \mathrm { a c t } }$ likewise produces the largest action shifts at both budgets, reaching $1 . 3 7   \times   1 0 ^ { \underline { { \sim } } _ { 2 } }$ and $1 . 6 6 \times 1 0 ^ { - 2 }$ at $\tilde { B } = 1 0 ^ { - \tilde { 3 } }$ versus at most ${ \mathrm { 9 . 1 8 } } \times 1 0 ^ { - 3 }$ for the controls.

**Probe 3. Direction-specific causal control.** In Probe 2, matching a fixed text-KL budget allows directions that barely perturb the text to take larger hidden-state steps than text-sensitive directions, as Figure 2a illustrates. To verify that $d _ { \mathrm { T N A S } }$ is intrinsically more action-selective per unit norm rather than only benefiting from a larger step size, Probe 3 fixes the hidden-state perturbation magnitude across all directions, as depicted in Figure 2b. Starting from the clean state, Probe 3 adds each direction with the same per-example norm $| q | S _ { \mathrm { c l i p } } ,$ where $\widetilde { S _ { \mathrm { c l i p } } } = | \langle h _ { \mathrm { c l e a n } } , d _ { \mathrm { T N A S } } \rangle |$ and $q \in \{ 0 . 5 , 1 , 2 \}$ , as

5

<!-- page 6 of 23 -->

detailed in Appendix $\mathrm { A . 8 . }$ Figure 2d,e,g,h then shows how the action and text channels respond. Across both models, $d _ { \mathrm { T N A S } }$ and the text-gradient controls show opposite selectivity, while $d _ { \mathrm { a c t } }$ lies on top of $d _ { \mathrm { T N A S } }$ across all $q .$

**Alpamayo**. The trajectory shift along $d _ { \mathrm { T N A S } }$ grows with $q$ to 0.142 m at $q   =   2$ , exceeding the text-gradient controls by $\dot { 4 } 4 . 0 \times$ to $4 7 . 8 \times$ and the random control by $5 3 . 9 \times$ . Conversely, at $q = 2$ dTNAS induces only $9 . 8 \mathring { 7 } \times 1 0 ^ { - 5 }$ teacher-forced KL, whereas the text-gradient controls induce $2 6 9 \times$ to $2 7 7 \times$ higher KL.

**InstructVLA**. $\mathrm { A t }   q = 2 ,   d _ { \mathrm { T N A S } }$ shifts the action by 0.156, exceeding the text-gradient controls by 4.64× to 5.51× and the random control by 12.6×, while keeping teacher-forced KL at $2 . 2 0 \times 1 0 ^ { - \tilde { 2 } }$ compared with $2 1 . 2 \times \mathrm { t o } \; 2 6 . 3 \times$ higher KL for the text-gradient controls.

**Takeaway 2. Hidden-state interventions along** $d _ { \mathrm { T N A S } }$ **shift the action while barely moving the text.** Under a fixed text-KL budget, $d _ { \mathrm { T N A S } }$ shifts Alpamayo’s trajectory $6 3 . 3 \times$ to 212× more than the controls. Under an equal perturbation norm across both models, $d _ { \mathrm { T N A S } }$ shifts the action $4 . 6 4   \times   \mathrm { t o }$ 47.8× more than the text-gradient controls and 12.6× to 53.9× more than the random control, while the text-gradient controls induce 21.2× to 277× more teacher-forced KL.

## 3 TNAS-Guided Visual Perturbations

Section 2 showed that direct hidden-state interventions along $d _ { \mathrm { T N A S } }$ shift the predicted action while leaving the next-token distribution nearly unchanged. We now test whether an adversary modifying only the visual input can excite this same direction end-to-end in Alpamayo and InstructVLA, resolv ing the optimization conflict between altering the physical action and preserving the generated text.

## 3.1 Attack setup

**Threat model.** We consider white-box digital perturbations applied to the normalized, processed visual input tensor immediately before the visual encoder, with gradients backpropagated end-to-end through the vision encoder, language backbone, and action head. In Alpamayo, full-image attacks perturb all 16 camera-frame inputs across the four cameras and four temporal frames. In InstructVLA, both full-image and localized attacks perturb the VLM/SigLIP visual stream. Section 3.3 further restricts the attacker to a spatially localized patch in each domain to mimic real-world scenarios where perturbations are attached to local objects or surfaces in the scene.

**Optimization.** Every attack runs 200 steps of projected gradient descent (PGD) [11–14] on its maximization objective L,

$$
\delta ^ { ( k + 1 ) } = \Pi _ { \mathcal { S } } \Big [ \delta ^ { ( k ) } + \alpha \operatorname { s i g n } \Bigl ( \nabla _ { \delta } \mathcal { L } ( \delta ^ { ( k ) } ) \Bigr ) \Big ] \: ,\tag{5}
$$

where $\Pi _ { S }$ projects onto the $\ell _ { \infty }$ ball of radius ε and the allowed spatial mask. Full-image attacks use $\varepsilon = 0 . 0 5$ and step size $\alpha = 5 \times 1 0 ^ { - 4 }$ in normalized pixel coordinates, whereas localized patches use a larger local budget $( \varepsilon = 0 . 5 , \alpha = 5 \times 1 0 ^ { - 3 } )$ to compensate for their restricted spatial support. Further implementation details on differentiable action rollouts are provided in Appendix A.2.

**Evaluation metrics.** Our evaluation covers both controlled fixed-prefix readouts and end-to-end free generation, along with open-loop safety checks in Alpamayo:

• **Fixed-prefix readouts.** Under the fixed clean text prefix, we report the Section 2.1 readouts: action shift $( \Delta _ { \mathrm { A D E } } \; \mathrm { o r } \; \Delta _ { \mathrm { I V L A } } )$ , teacher-forced KL $\mathrm { ( K L _ { t f } ) }$ , and hidden-state alignment cos $( \Delta h , d _ { \mathrm { T N A S } } )$ .

• **Free-generation exact match.** As detailed in Appendices A.1–A.2.3, under free generation, the model autoregressively generates its own text and decodes the corresponding action, and Exact reports the fraction of examples whose generated rationale or response matches the clean one word for word.

• **Silent collision rate.** To evaluate whether trajectory divergence creates concrete driving hazards, we score Alpamayo’s free-generated 6.4 s (64-waypoint) plans against time-aligned logged traffic actors on the 154 clips with obstacle annotations [15, 16], following the protocol detailed in Appendix $\mathrm { A . 3 }$ . At each 0.1 s planning step, we project the ego vehicle and logged actors as oriented bounding boxes in bird’s-eye view, inferring ego heading from consecutive waypoints, and test for overlap using the separating-axis theorem. A clip has an open-loop collision if the

6

<!-- page 7 of 23 -->

Table 1: Alpamayo input attacks. Fixed-prefix and free-generation columns use the canonical 200 clips. The open-loop column uses the 154 of them with obstacle annotations, as described in Section 3.1.

<table><tr><td rowspan="3">Attack</td><td colspan="3">Fixed-prefix</td><td colspan="2">Free gen.</td></tr><tr><td>ADE</td><td> $\mathrm { K L } _ { \mathrm { t f } } \downarrow$ </td><td>TNAS</td><td>Exact</td><td>Open loop Silent Coll.</td></tr><tr><td>(m)↑</td><td></td><td>cos</td><td>(%)↑</td><td>(%)↑</td></tr><tr><td>Action-only PGD</td><td>70.1</td><td> $2 . 7 8 \times 1 0 ^ { - 1 }$ </td><td> $4 . 4 0 \times 1 0 ^ { - 4 }$ </td><td>9.5</td><td>4.55</td></tr><tr><td>Text-preserving PGD</td><td>69.2</td><td> $3 . 2 6 \times 1 0 ^ { - 1 }$ </td><td> $8 . 0 0 \times 1 0 ^ { - 4 }$ </td><td>83.0</td><td>48.7</td></tr><tr><td>TNAS Full image</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\lambda _ { \mathrm { c o s } } = 5$ </td><td>67.4</td><td> $3 . 3 2 \times 1 0 ^ { - 1 }$ </td><td> $1 . 6 4 \times 1 0 ^ { - 2 }$ </td><td>80.5</td><td>46.8</td></tr><tr><td> $\lambda _ { \mathrm { c o s } } = 1 0$ </td><td>67.8</td><td> $2 . 9 9 \times 1 0 ^ { - 1 }$   $2 . 4 6 \times 1 0 ^ { - 1 }$ </td><td> $3 . 9 3 \times 1 0 ^ { - 2 }$   $7 . 5 7 \times 1 0 ^ { - 2 }$ </td><td>81.0</td><td>47.4</td></tr><tr><td> $\lambda _ { \mathrm { c o s } } = 1 5$ </td><td>65.5</td><td> $1 . 8 9 \times 1 0 ^ { - 1 }$ </td><td> $1 . 3 3 \times 1 0 ^ { - 1 }$ </td><td>84.0 88.0</td><td>45.5</td></tr><tr><td> $\lambda _ { \mathrm { c o s } } = 2 0$   $\lambda _ { \mathrm { c o s } } = 3 0$ </td><td>63.2 53.0</td><td> $1 . 0 3 \times 1 0 ^ { - 1 }$ </td><td> $2 . 4 6 \times 1 0 ^ { - 1 }$ </td><td>89.5</td><td>45.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>52.6</td></tr><tr><td> $\mathrm { T N A S ~ P a t c h }$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\lambda _ { \mathrm { c o s } } = 5$ </td><td>38.7</td><td> $1 . 3 9 \times 1 0 ^ { - 1 }$ </td><td> $1 . 5 5 \times 1 0 ^ { - 2 }$ </td><td>70.5</td><td>33.8</td></tr><tr><td> $\lambda _ { \mathrm { c o s } } = 1 0$ </td><td>39.8</td><td> $1 . 2 4 \times 1 0 ^ { - 1 }$ </td><td> $4 . 2 4 \times 1 0 ^ { - 2 }$ </td><td>70.5</td><td>35.1</td></tr><tr><td> $\lambda _ { \mathrm { c o s } } = 1 5$ </td><td>37.2</td><td> $1 . 0 7 \times 1 0 ^ { - 1 }$ </td><td> $6 . 7 4 \times 1 0 ^ { - 2 }$ </td><td>71.0</td><td>31.8</td></tr><tr><td> $\lambda _ { \mathrm { c o s } } = 2 0$ </td><td>34.8</td><td> $9 . 0 0 \times 1 0 ^ { - 2 }$ </td><td> $1 . 0 4 \times 1 0 ^ { - 1 }$ </td><td>76.0</td><td>34.4</td></tr><tr><td> $\lambda _ { \mathrm { c o s } } = 3 0$ </td><td>30.0</td><td> $5 . 7 0 \times 1 0 ^ { - 2 }$ </td><td> $1 . 6 2 \times 1 0 ^ { - 1 }$ </td><td>80.5</td><td>34.4</td></tr></table>

ego box intersects any actor box at least once over the 6.4 s horizon, and the silent collision rate reports the fraction of the 154 clips where a collision-free clean plan is perturbed into a collision while the free-generated rationale remains word-for-word identical to the clean rationale.

**Baselines.** We compare against two baseline objectives that represent the intuitive output-level approaches to inducing action deviation alone and to inducing action deviation while preserving the generated text.

**Action-only PGD** only maximizes an action-displacement loss $\mathcal { L } _ { \mathrm { a c t } } ( \delta _ { \mathrm { i m g } } )$ between the perturbed and clean action predictions over the T-step horizon:

$$
\mathcal { L } _ { \operatorname { a c t } } ( \delta _ { \operatorname { i m g } } ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } D _ { \operatorname { a c t } } \bigl ( a _ { t } ( \delta _ { \operatorname { i m g } } ) , \thinspace a _ { t } ( 0 ) \bigr ) .\tag{6}
$$

Following Equation 1, the per-step discrepancy $D _ { \mathrm { a c t } }$ is the planar waypoint displacement $\| \tau _ { t } ( \delta _ { \mathrm { i m g } } ) -$ $\tau _ { t } ( 0 ) \| _ { 2 }$ for Alpamayo and the mean squared coordinate difference across the 7-DoF action vector for InstructVLA.

**Text-preserving PGD**, detailed in Appendix A.2, subtracts an explicit output-level penalty on the two active text losses defined in Section 2.1, namely the clean-target NLL increase $\Delta \mathcal { L } _ { \mathrm { N L L } }$ and the targetversus-runner-up margin drop ∆m, following joint action–text formulations for reasoning VLAs [8]:

$$
\mathcal { L } _ { \operatorname { a c t + t x t } } ( \delta _ { \operatorname { i m g } } ) = \mathcal { L } _ { \operatorname { a c t } } ( \delta _ { \operatorname { i m g } } ) - \lambda _ { \operatorname { t x t } } \left[ \Delta \mathcal { L } _ { \operatorname { N L L } } + \Delta m \right]\tag{7}
$$

## 3.2 TNAS-guided full-image attack

**Output-level action and text objectives compete.** In Table 1, Action-only PGD in Alpamayo shifts the planned trajectory by 70.1 m ADE, but it also rewrites the driving rationale on 90.5% of clips, leaving only 4.55% silent collisions. Adding the output-level text penalty in Text-preserving PGD keeps the clean rationale intact on 83.0% of clips and raises silent collisions to 48.7%, yet the two objectives work against each other during optimization. In Equation 7, both terms update the same input pixels: the action loss maximizes trajectory displacement, while the text penalty constrains only the clean target tokens’ NLL and top-1 margin. Holding those target tokens in place against unguided action updates pushes vocabulary-wide teacher-forced KL from $\widetilde { 2 . 7 8 } \times 1 0 ^ { - 1 }$ under Action-only PGD to $3 . 2 6 \stackrel { \circ } { \times } 1 0 ^ { - 1 }$ , which lowers the chance that stochastic decoding reproduces the clean rationale.

**Resolving the conflict with TNAS guidance.** To resolve this conflict at its geometric source, **TNAS-guided PGD** augments the text-preserving objective with a term that rewards cosine alignment

7

<!-- page 8 of 23 -->

Table 2: InstructVLA input attacks. Fixed-prefix and free-generation columns use the canonical 200 examples.

<table><tr><td rowspan="2">Attack</td><td colspan="3">Fixed-prefix</td><td>Free gen.</td></tr><tr><td>Action shift (norm.) ↑</td><td> $\mathrm { K L } _ { \mathrm { t f } } \downarrow$ </td><td>TNAS cos</td><td>Exact (%)↑</td></tr><tr><td>Action-only PGD</td><td>2.04</td><td> $4 . 6 6 \times 1 0 ^ { - 2 }$ </td><td> $1 . 2 0 \times 1 0 ^ { - 3 }$ </td><td>72.5</td></tr><tr><td>Text-preserving PGD</td><td>1.97</td><td> $3 . 4 6 \times 1 0 ^ { - 2 }$ </td><td> $2 . 9 0 \times 1 0 ^ { - 3 }$ </td><td>77.5</td></tr><tr><td>TNAS Full image</td><td></td><td></td><td></td><td></td></tr><tr><td> $\lambda _ { \mathrm { c o s } } = 0 . 1 0$ </td><td>1.85</td><td> $2 . 6 7 \times 1 0 ^ { - 2 }$ </td><td> $1 . 2 2 \times 1 0 ^ { - 1 }$ </td><td>81.0</td></tr><tr><td> $\lambda _ { \mathrm { c o s } } = 0 . 1 5$ </td><td>1.80</td><td> $2 . 3 5 \times 1 0 ^ { - 2 }$ </td><td> $1 . 6 6 \times 1 0 ^ { - 1 }$ </td><td>81.5</td></tr><tr><td> $\lambda _ { \mathrm { c o s } } = 0 . 2 0$ </td><td>1.73</td><td> $\underline { { 2 . 0 6 \times 1 0 ^ { - 2 } } }$ </td><td> $\underline { { 2 . 1 3 \times 1 0 ^ { - 1 } } }$ </td><td>85.0</td></tr><tr><td> $\mathrm { T N A S ~ P a t c h }$ </td><td></td><td></td><td></td><td></td></tr><tr><td> $\lambda _ { \mathrm { c o s } } = 0 . 2 0$ </td><td>1.32</td><td> $2 . 8 0 \times 1 0 ^ { - 2 }$ </td><td> $1 . 1 3 \times 1 0 ^ { - 1 }$ </td><td>77.5</td></tr></table>

between the input-induced hidden-state shift $\Delta h ( \delta _ { \mathrm { i m g } } )$ and each example’s TNAS direction $d _ { \mathrm { T N A S } } ;$

$$
\mathcal { L } _ { \operatorname { T N A S } } ( \delta _ { \operatorname { i m g } } ) = \mathcal { L } _ { \operatorname { a c t + t x t } } ( \delta _ { \operatorname { i m g } } ) + \lambda _ { \operatorname { c o s } } \operatorname { c o s } \bigl ( \Delta h ( \delta _ { \operatorname { i m g } } ) , d _ { \operatorname { T N A S } } \bigr ) .\tag{8}
$$

As shown in Section 2, shifting the hidden state along d<sub>TNAS</sub> drives action displacement while leaving the two text losses in Equation 7 unchanged to first order. Maximizing cosine similarity in Equation 8 directly steers the direction of $\Delta h ( \delta _ { \mathrm { i m g } } )$ toward $d _ { \mathrm { T N A S } }$ . During optimization, this allows PGD updates to increase action displacement with minimal disturbance to the text logits and reduces the conflict between the action and text terms in $\mathcal { L } _ { \mathrm { a c t + t x t } }$

**TNAS guidance quiets the text channel.** In Table 1, increasing $\lambda _ { \mathrm { c o s } }$ from 5 to 30 raises cosine alignment between ∆h and $d _ { \mathrm { T N A S } }$ at every step from $1 . 6 4 \times 1 0 ^ { - \widetilde { 2 } } \; \mathrm { t o } \; \widetilde { 2 } . 4 6 \times 1 0 ^ { - 1 }$ , compared with $8 . \widetilde { 0 0 } \times 1 0 ^ { - 4 }$ under Text-preserving PGD. Even though the cosine term never accesses token logits or text losses, steering the hidden state toward $d _ { \mathrm { T N A S } }$ pulls teacher-forced KL down monotonically across the sweep to $\bar { 1 . 0 3 } \times 1 0 ^ { - 1 }$ at $\lambda _ { \mathrm { c o s } } = 3 0$ , a 68.4% reduction relative to Text-preserving PGD. Because lower distributional drift keeps the perturbed next-token distribution closer to the clean one during stochastic decoding, free-generation exact match also rises monotonically across the sweep and reaches 89.5% at $\lambda _ { \mathrm { c o s } } = 3 0$ , up from 83.0% without TNAS guidance.

**Steering toward** $d _ { \mathrm { T N A S } }$ **concentrates displacement into silent collisions.** Under a fixed pixel budget, weighting the objective toward $d _ { \mathrm { T N A S } }$ trades redundant fixed-prefix displacement, from 69.2 m under Text-preserving PGD to 53.0 m at $\lambda _ { \mathrm { c o s } } = 3 0$ , for stronger TNAS alignment without reducing collisions. On the 154 obstacle-annotated scenes, the free-generated plans continue to collide on 61.7% of clips at $\lambda _ { \mathrm { c o s } } = 3 0$ versus 61.0% under Text-preserving PGD, so the higher exact rationale preservation translates directly into silent collisions. The final step of the sweep brings the largest gains in text stability. Between $\lambda _ { \mathrm { c o s } }   =   2 0$ and $\lambda _ { \mathrm { c o s } }   =   3 0$ , TNAS alignment nearly doubles and teacher-forced KL falls the most, while silent collisions reach 52.6%, the highest in the sweep. TNAS guidance therefore buys text stability without giving up silent collisions.

**Replication in InstructVLA.** In Table 2, the same geometric guidance transfers to robotic manipulation in InstructVLA, where perturbations modify only the VLM visual stream while a single taskintention sentence averaging 9.60 words, such as “I will move the cup to the sink,” covers an entire 16- step 7-DoF action chunk. Across the $\lambda _ { \mathrm { c o s } }$ sweep from 0.10 to 0.20, cosine alignment between $\Delta h$ and $d _ { \mathrm { T N A S } }$ rises at every step from $1 . 2 2 \times \overset { \circ \infty } { 1 0 ^ { - 1 } } \mathbin { \mathfrak { t o } } \overset { \bullet } { 2 } . 1 3 \times 1 0 ^ { - 1 }$ , whereas Action-only PGD at $1 . 2 0   \times   1 0 ^ { - 3 }$ and Text-preserving PGD at $1 2 . 9 0 \times 1 0 ^ { - 3 }$ leave the hidden-state shift unaligned with $d _ { \mathrm { T N A S } }$ . Even the smallest guided weight $\lambda _ { \mathrm { c o s } } = 0 . 1 0$ already improves over Text-preserving PGD on every text metric, and steering further toward $d _ { \mathrm { T N A S } }$ pulls teacher-forced KL down monotonically from $\tilde { 2 } . 6 7 \times 1 0 ^ { - 2 }$ to $2 . 0 6 \times \tilde { 1 0 } ^ { - 2 }$ at $\lambda _ { \mathrm { c o s } } = 0 . 2 0$ , a 40.5% reduction from $3 . 4 6 \times 1 0 ^ { - 2 }$ under Text-preserving PGD and 55.8% below $4 . 6 6 \times 1 0 ^ { - 2 }$ under Action-only PGD. This lower distributional drift raises freegeneration exact match monotonically across the sweep from 81.0% to 85.0% at $\lambda _ { \mathrm { c o s } } = 0 . 2 0$ , up from 77.5% under Text-preserving PGD. Even though this coarser many-to-one intention bottleneck already leaves 72.5% of responses unchanged under Action-only PGD without any text penalty, steering along $d _ { \mathrm { T N A S } }$ still eliminates one-third of the response changes that remain under Text-preserving PGD. Under the fixed pixel budget, weighting the objective toward $d _ { \mathrm { T N A S } }$ trades only a small part of

8

<!-- page 9 of 23 -->

the fixed-prefix action shift, from 2.04 under Action-only PGD and 1.97 under Text-preserving PGD to 1.73 at $\lambda _ { \mathrm { c o s } } = 0 . 2 0$ , so it retains 87.5% of Text-preserving PGD’s action displacement alongside this stronger text stability. As in Alpamayo, the final step from $\lambda_{ cos } = 0.15 \;  to  \; \bar{\lambda_{ cos }} = 0.20$ brings the largest exact-match gain within the guided sweep, jumping from 81.5% to $8 5 . 0 \%$ as TNAS alignment climbs from $1 . 6 6 { \times } \tilde { 1 0 } ^ { - 1 }   \mathrm { t o }   2 . 1 3 { \times } \tilde { 1 0 } ^ { - 1 }$ and teacher-forced KL drops from $2 . 3 5   \times   1 0 ^ { - 2 }$ to $2 . 0 6   \bar { \times }   1 0 ^ { - 2 }$

**Takeaway 3. TNAS guidance resolves the action–text optimization conflict, suppressing text drift while increasing silent collisions.** Output-level text penalties compete with the action gradient and leave hidden-state updates unaligned with d<sub>TNAS</sub>. Steering along d<sub>TNAS</sub> lowers teacher-forced KL by 68.4% in Alpamayo and 40.5% in InstructVLA, raises exact rationale preservation from 83.0% to 89.5% and from 77.5% to 85.0%, and produces silent collisions on 52.6% of annotated driving scenes alongside a 1.73-unit manipulation action shift.

## 3.3 Localized patch attack

Our full-image attacks assume a digital attacker who can add a perturbation to the preprocessed image tensor fed to the vision encoder, which in practice requires access to the model’s software pipeline. Moving toward localized attacks on objects in the scene, such as a patch on the back of a front vehicle, requires the perturbation to operate within a restricted spatial region while remaining consistent across views and time. To test whether an attacker can still steer the shared representation along $d _ { \mathrm { T N A S } }$ under these constraints, we confine the perturbation to a localized patch and retain the TNAS-guided objective.

**Shared forward-camera patch in Alpamayo.** In Alpamayo, we optimize a single clip-specific patch on the road surface ahead of the vehicle and tie it across the two forward-facing cameras and all four timestamps, covering 13.1% of visual tokens. The front-telephoto crop receives the base pattern, the front-wide crop receives a bilinearly resized version on the corresponding road region, and the two cross-view cameras remain clean. In Table 1, increasing $\lambda _ { \mathrm { c o s } }$ from 5 to 30 raises cosine alignment between ∆h and $d _ { \mathrm { T N A S } }$ at every step from $1 . 5 5 \times 1 0 ^ { - 2 }$ to $1 . 6 2 \times 1 0 ^ { - 1 }$ and pulls teacher-forced KL down monotonically from $1 . 3 9 \dot { } \times 1 \dot { 0 } ^ { - 1 }$ to $5 . 7 0 \times 1 0 ^ { - 2 }$ , a 59.0% reduction that lifts free-generation exact match from 70.5% to 80.5%. With the surrounding road and cross-view cameras left untouched, fixed-prefix trajectory displacement reaches 38.7 m ADE at $\lambda _ { \mathrm { c o s } } = 5$ and 39.8 m at $\lambda _ { \mathrm { c o s } } = 1 0$ , and weighting the objective toward $\lambda _ { \mathrm { c o s } } = 3 0$ trades part of this shift to 30.0 m for stronger text stability while keeping silent collisions between 31.8% and 35.1% of obstacle-annotated scenes.

**VLM-branch patch in InstructVLA.** In InstructVLA, we confine SigLIP updates to a $0 . 4 \times 0 . 4$ lower-center rectangle covering 16% of the image area. In Table 2, TNAS-guided PGD at $\lambda _ { \mathrm { c o s } } = 0 . 2 0$ steers the hidden-state shift from this patch to $\tilde { 1 . 1 3 } \times 1 0 ^ { - 1 }$ cosine alignment with dTNAS and lowers teacher-forced KL $\mathrm { t o } \; 2 . 8 0 \times 1 0 ^ { - 2 }$ , 19.1% below the $3 . 4 6 \times 1 0 ^ { - 2 }$ of full-image Text-preserving PGD. Free-generation exact match reaches 77.5%, matching full-image Text-preserving PGD and exceeding the 72.5% of full-image Action-only PGD. With the remaining 84% of the image left untouched, the patch shifts the predicted action chunk by 1.32 and retains 76.4% of the 1.73 displacement from full-image TNAS-guided PGD at $\lambda _ { \mathrm { c o s } } = 0 . \dot { 2 } 0$

**Takeaway 4. TNAS-guided action divergence persists under localized patch constraints.** In Alpamayo, a forward-camera patch on 13.1% of visual tokens cuts teacher-forced KL by 59.0% across the $\lambda _ { \mathrm { c o s } }$ sweep, preserves up to 80.5% of rationales word for word, and induces silent collisions on up to 35.1% of annotated scenes. In InstructVLA, a patch on 16% of the image shifts the action chunk by 1.32 units while matching the 77.5% exact preservation of full-image Text-preserving PGD with 19.1% lower teacher-forced KL.

## 4 Related Work

**Reasoning vision-language-action models.** VLA policies connect visual and language representations to embodied control [17–20], with autonomous driving models and benchmarks extending this paradigm to multi-camera road scenes [21–27]. Reasoning-conditioned VLAs expose naturallanguage scene descriptions, plans, or chain-of-thought traces alongside continuous action prediction [1, 2, 28, 3]. We study Alpamayo-1.5 [4] and InstructVLA [5], two representative embodiments

9

<!-- page 10 of 23 -->

that pair autoregressive VLM text generation with continuous diffusion or flow-matching action heads reading from a shared hidden state.

**Monitoring model reasoning.** Chain-of-thought monitoring treats verbalized reasoning as a safety signal and asks when that signal stops reflecting behavior [6, 7]. In language models, faithfulness studies distinguish plausible post-hoc explanations from the causal drivers of an answer [29–32]. In embodied VLAs, prior work addresses unperturbed mismatches between textual plans and actions at the output level through RL consistency rewards [4] or runtime action verification [33]. Concurrent with our work, Foutter et al. [34] observe that Alpamayo-1.5 can state that it will yield while steering without decelerating, and post-train driving planners with a learned critic that rewards semantic consistency among observations, reasoning steps, and trajectories. We instead diagnose why text and action decouple at the shared hidden state, where TNAS shows that the locally most action-sensitive direction is nearly orthogonal to the active text-loss gradients, and steer visual inputs along this subspace to alter the physical action while keeping the generated rationale word-for-word identical. The TNAS geometry justifies pairing rationale monitors with execution-channel monitors [35].

**Adversarial attacks on driving and embodied policies.** Adversarial examples were first studied as imperceptible $\ell _ { p } .$ bounded perturbations of the full digital image [36, 11, 37, 12]. Later work showed that such perturbations survive printing, camera capture, and viewpoint changes [38, 39], and that a localized patch such as a printed sticker or accessory can fool image classifiers and face recognition systems [40–43]. In autonomous driving, physical patches appear as dirt-like road patterns that steer automated lane centering out of its lane [44], roadside billboards that mislead steering models [45], and patches on the rear of a lead vehicle that corrupt monocular depth estimation [46]. Attacks on driving stacks also target trajectory prediction [13] and collaborative perception [14]. On VLAs, a small patch in the camera view can derail robotic manipulation [47] or hijack chain-of-thought reasoning [8], and other attacks corrupt the reasoning trace directly [9] or implant backdoors [48–50]. Reasoning-hijacking attacks visibly alter the text, and unconstrained action attacks corrupt it as a byproduct. Our TNAS-guided attack instead holds the generated rationale fixed while steering the action, under both full-image perturbations and a localized patch.

**Representation steering and subspace projection.** Representation engineering and activation steering change model behavior by adding a direction to intermediate hidden states at inference time [51, 52]. Linear concept erasure projects hidden states so that a linear readout can no longer recover a protected attribute [53, 54]. TNAS applies this projection to the steering direction itself at the shared text–action interface. Removing the active text-loss span from the locally most action-sensitive direction leaves a residual that moves the action readout while keeping the text readout unchanged to first order. Activation steering writes such directions into the hidden state, whereas our attack reaches the TNAS residual by perturbing only the visual input.

## 5 Conclusion

We studied rationale-preserving action divergence in reasoning VLAs, where a perturbed visual input alters the physical action while leaving the generated rationale word for word unchanged. To diagnose this vulnerability at the shared text–action interface, we introduced the text-neutral action-subspace (TNAS) probe, which projects active text-loss gradients out of the locally most action-sensitive hidden-state direction to isolate a text-neutral action residual. Across Alpamayo for autonomous driving and InstructVLA for robotic manipulation, the action-sensitive direction overlaps the active text span by only 1.87% and 5.28% of its norm, and equal-norm interventions along the TNAS residual shift actions 4.64× to 47.8× more than text-gradient perturbations while text-gradient perturbations induce 21.2× to 277× more text drift. Steering visual perturbations along this residual resolves the optimization conflict between action displacement and text preservation and cuts teacher-forced text drift by 68.4% in Alpamayo and 40.5% in InstructVLA relative to text-preserving baselines. Under free generation, TNAS-guided attacks preserve the rationale word for word while inducing new collisions on 52.6% of annotated driving scenes under full-image perturbations and 35.1% under a localized forward-camera patch, and preserve 85.0% of manipulation responses alongside a 1.73-unit action-chunk shift. Because an unchanged rationale cannot certify a safe action, embodied safety monitors must verify the predicted action channel directly alongside the text.

10

<!-- page 11 of 23 -->

## Acknowledgments and Disclosure of Funding

We thank the area chair and the anonymous reviewers for their valuable feedback and constructive suggestions. We also specifically thank Chufan Shi for his valuable help and insightful discussions throughout this work. This material is based upon work supported by the National Science Foundation under Grant No. 2333489. The authors declare no competing interests.

## References

[1] Xiaoyu Tian, Junru Gu, Bailin Li, Yicheng Liu, Yang Wang, Zhiyong Zhao, Kun Zhan, Peng Jia, Xianpeng Lang, and Hang Zhao. Drivevlm: The convergence of autonomous driving and large vision-language models. arXiv preprint arXiv:2402.12289, 2024.

[2] Bo Jiang, Shaoyu Chen, Bencheng Liao, Xingyu Zhang, Wei Yin, Qian Zhang, Chang Huang, Wenyu Liu, and Xinggang Wang. Senna: Bridging large vision-language models and end-to-end autonomous driving. arXiv preprint arXiv:2410.22313, 2024.

[3] Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. Robotic control via embodied chain-of-thought reasoning. arXiv preprint arXiv:2407.08693, 2024.

[4] Yan Wang, Wenjie Luo, Junjie Bai, Yulong Cao, Tong Che, Ke Chen, Yuxiao Chen, Jenna Diamond, Yifan Ding, Wenhao Ding, et al. Alpamayo-r1: Bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088, 2025.

[5] Shuai Yang, Hao Li, Bin Wang, Yilun Chen, Yang Tian, Tai Wang, Hanqing Wang, Feng Zhao, Yiyi Liao, and Jiangmiao Pang. Instructvla: Vision-language-action instruction tuning from understanding to manipulation. arXiv preprint arXiv:2507.17520, 2025.

[6] Bowen Baker, Joost Huizinga, Leo Gao, Zehao Dou, Melody Y. Guan, Aleksander Madry, Wojciech Zaremba, Jakub Pachocki, and David Farhi. Monitoring reasoning models for misbehavior and the risks of promoting obfuscation. arXiv preprint arXiv:2503.11926, 2025.

[7] Tomek Korbak, Mikita Balesni, Elizabeth Barnes, Yoshua Bengio, Joe Benton, Joseph Bloom, Mark Chen, Alan Cooney, Allan Dafoe, Anca Dragan, et al. Chain of thought monitorability: A new and fragile opportunity for AI safety. arXiv preprint arXiv:2507.11473, 2025.

[8] Zhengxian Huang, Wenjun Zhu, Haoxuan Qiu, Xiaoyu Ji, and Wenyuan Xu. Trap: Hijacking vla cot-reasoning via adversarial patches. arXiv preprint arXiv:2603.23117, 2026.

[9] Tuan Duong Trinh, Naveed Akhtar, and Basim Azam. Altered thoughts, altered actions: Probing chain-of-thought vulnerabilities in vla robotic manipulation. arXiv preprint arXiv:2603.12717, 2026.

[10] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[11] Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. In International Conference on Learning Representations, 2015.

[12] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations, 2018.

[13] Qingzhao Zhang, Shengtuo Hu, Jiachen Sun, Qi Alfred Chen, and Z Morley Mao. On adversarial robustness of trajectory prediction for autonomous vehicles. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15159–15168, 2022.

[14] Qingzhao Zhang, Shuowei Jin, Ruiyang Zhu, Jiachen Sun, Xumiao Zhang, Qi Alfred Chen, and Z Morley Mao. On data fabrication in collaborative vehicular perception: Attacks and countermeasures. In 33rd USENIX Security Symposium (USENIX Security 24), pages 6309–6326, 2024.

[15] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, Lewei Lu, Xiaosong Jia, Qiang Liu, Jifeng Dai, Yu Qiao, and Hongyang Li. Planningoriented autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17853–17862, 2023.

11

<!-- page 12 of 23 -->

[16] Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, and Xinggang Wang. VAD: Vectorized scene representation for efficient autonomous driving. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 8340–8350, 2023.

[17] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

[18] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.

[19] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.

[20] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[21] Zewei Zhou, Tianhui Cai, Seth Z Zhao, Yun Zhang, Zhiyu Huang, Bolei Zhou, and Jiaqi Ma. Autovla: A vision-language-action model for end-to-end autonomous driving with adaptive reasoning and reinforcement fine-tuning. arXiv preprint arXiv:2506.13757, 2025.

[22] Hidehisa Arai, Keita Miwa, Kento Sasaki, Kohei Watanabe, Yu Yamaguchi, Shunsuke Aoki, and Issei Yamamoto. Covla: Comprehensive vision-language-action dataset for autonomous driving. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 1933–1943. IEEE, 2025.

[23] Xingcheng Zhou, Xuyuan Han, Feng Yang, Yunpu Ma, Volker Tresp, and Alois Knoll. Opendrivevla: Towards end-to-end autonomous driving with large vision language action model. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 13782–13790, 2026.

[24] Haoyu Fu, Diankun Zhang, Zongchuang Zhao, Jianfeng Cui, Dingkang Liang, Chong Zhang, Dingyuan Zhang, Hongwei Xie, Bing Wang, and Xiang Bai. Orion: A holistic end-to-end autonomous driving framework by vision-language instructed action generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 24823–24834, 2025.

[25] Katrin Renz, Long Chen, Elahe Arani, and Oleg Sinavski. Simlingo: Vision-only closed-loop autonomous driving with language-action alignment. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 11993–12003, 2025.

[26] Hao Shao, Yuxuan Hu, Letian Wang, Guanglu Song, Steven L. Waslander, Yu Liu, and Hongsheng Li. LMDrive: Closed-loop end-to-end driving with large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15120–15130, 2024.

[27] Yongkang Li, Kaixin Xiong, Xiangyu Guo, Fang Li, Sixu Yan, Gangwei Xu, Lijun Zhou, Long Chen, Haiyang Sun, Bing Wang, et al. Recogdrive: A reinforced cognitive framework for end-to-end autonomous driving. arXiv preprint arXiv:2506.08052, 2025.

[28] Cheng Yin, Yankai Lin, Wang Xu, Sikyuen Tam, Xiangrui Zeng, Zhiyuan Liu, and Zhouping Yin. Deepthinkvla: Enhancing reasoning capability of vision-language-action models. arXiv preprint arXiv:2511.15669, 2025.

[29] Miles Turpin, Julian Michael, Ethan Perez, and Samuel Bowman. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. Advances in Neural Information Processing Systems, 36:74952–74965, 2023.

[30] Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, et al. Measuring faithfulness in chain-of-thought reasoning. arXiv preprint arXiv:2307.13702, 2023.

[31] Iván Arcuschin, Jett Janiak, Robert Krzyzanowski, Senthooran Rajamanoharan, Neel Nanda, and Arthur Conmy. Chain-of-thought reasoning in the wild is not always faithful. In Workshop on Reasoning and Planning for Large Language Models, 2025. URL [https://openreview.net/forum?id=L8094Whth0](https://openreview.net/forum?id=L8094Whth0).

12

<!-- page 13 of 23 -->

[32] Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong, Marianna Apidianaki, and Chris Callison-Burch. Faithful chain-of-thought reasoning. In Jong C. Park, Yuki Arase, Baotian Hu, Wei Lu, Derry Wijaya, Ayu Purwarianti, and Adila Alfa Krisnadhi, editors, Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 305–329, Nusa Dua, Bali, November 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.ijcnlp-main.20. URL [https://aclanthology.org/2023.ijcnlp-main.20/.](https://aclanthology.org/2023.ijcnlp-main.20/)

[33] Yilin Wu, Anqi Li, Tucker Hermans, Fabio Ramos, Andrea Bajcsy, and Claudia Pérez-D’Arpino. Do what you say: Steering vision-language-action models via runtime reasoning-action alignment verification. In IEEE International Conference on Robotics and Automation (ICRA), 2026.

[34] Matthew Foutter, Matteo Cercola, Lena Wild, Yunshan Wang, Michelle Li, Daniele Gammelli, and Marco Pavone. Do vision-language-action models mean what they say? On the role of faithfulness in embodied reasoning. arXiv preprint arXiv:2607.04681, 2026.

[35] Florian Seligmann, Emiliyan Gospodinov, Enes Ulas Dincer, and Gerhard Neumann. VLA-FAIL: Efficient task failure detection for finetuned vision-language-action models. arXiv preprint arXiv:2606.21386, 2026.

[36] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. In International Conference on Learning Representations, 2014.

[37] Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In 2017 IEEE Symposium on Security and Privacy, pages 39–57, 2017.

[38] Alexey Kurakin, Ian Goodfellow, and Samy Bengio. Adversarial examples in the physical world. In International Conference on Learning Representations Workshop, 2017.

[39] Anish Athalye, Logan Engstrom, Andrew Ilyas, and Kevin Kwok. Synthesizing robust adversarial examples. In International Conference on Machine Learning, pages 284–293. PMLR, 2018.

[40] Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, and Justin Gilmer. Adversarial patch. arXiv preprint arXiv:1712.09665, 2017.

[41] Danny Karmon, Daniel Zoran, and Yoav Goldberg. LaVAN: Localized and visible adversarial noise. In International Conference on Machine Learning, pages 2507–2515. PMLR, 2018.

[42] Mahmood Sharif, Sruti Bhagavatula, Lujo Bauer, and Michael K. Reiter. Accessorize to a crime: Real and stealthy attacks on state-of-the-art face recognition. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security, pages 1528–1540, 2016.

[43] Kevin Eykholt, Ivan Evtimov, Earlence Fernandes, Bo Li, Amir Rahmati, Chaowei Xiao, Atul Prakash, Tadayoshi Kohno, and Dawn Song. Robust physical-world attacks on deep learning visual classification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1625–1634, 2018.

[44] Takami Sato, Junjie Shen, Ningfei Wang, Yunhan Jia, Xue Lin, and Qi Alfred Chen. Dirty road can attack: Security of deep learning based automated lane centering under physical-world attack. In 30th USENIX Security Symposium (USENIX Security 21), pages 3309–3326, 2021.

[45] Zelun Kong, Junfeng Guo, Ang Li, and Cong Liu. PhysGAN: Generating physical-world-resilient adversarial examples for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14254–14263, 2020.

[46] Zhiyuan Cheng, James Liang, Hongjun Choi, Guanhong Tao, Zhiwen Cao, Dongfang Liu, and Xiangyu Zhang. Physical attack on monocular depth estimation with optimal adversarial patches. In European Conference on Computer Vision, pages 514–532, 2022.

[47] Taowen Wang, Cheng Han, James Liang, Wenhao Yang, Dongfang Liu, Luna Xinyu Zhang, Qifan Wang, Jiebo Luo, and Ruixiang Tang. Exploring the adversarial vulnerabilities of vision-language-action models in robotics. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6948–6958, 2025.

[48] Jiayu Li, Yunhan Zhao, Xiang Zheng, Zonghuan Xu, Yige Li, Xingjun Ma, and Yu-Gang Jiang. Attack-VLA: Benchmarking adversarial and backdoor attacks on vision-language-action models. arXiv preprint arXiv:2511.12149, 2025.

13

<!-- page 14 of 23 -->

[49] Xueyang Zhou, Guiyao Tie, Guowen Zhang, Hechang Wang, Pan Zhou, and Lichao Sun. BadVLA: Towards backdoor attacks on vision-language-action models via objective-decoupled optimization. arXiv preprint arXiv:2505.16640, 2025.

[50] Qi Li, Bo Yin, Weiqi Huang, Ruhao Liu, Bojun Zou, Runpeng Yu, Jingwen Ye, Weihao Yu, and Xinchao Wang. Vision-language-action safety: Threats, challenges, evaluations, and mechanisms. arXiv preprint arXiv:2604.23775, 2026.

[51] Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A top-down approach to ai transparency. arXiv preprint arXiv:2310.01405, 2023.

[52] Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J Vazquez, Ulisse Mini, and Monte MacDiarmid. Steering language models with activation engineering, 2025. URL [https://openreview.net/forum?id=2XBPdPIcFK.](https://openreview.net/forum?id=2XBPdPIcFK)

[53] Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, and Yoav Goldberg. Null it out: Guarding protected attributes by iterative nullspace projection. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020.

[54] Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, and Stella Biderman. LEACE: Perfect linear concept erasure in closed form. In Advances in Neural Information Processing Systems, 2023.

[55] Zhiqi Li, Guo Chen, Shilong Liu, Shihao Wang, Vibashan VS, et al. Eagle 2: Building post-training data strategies from scratch for frontier vision-language models. arXiv preprint arXiv:2501.14818, 2025.

14

<!-- page 15 of 23 -->

## A Appendix

The appendix gives the protocols and full values behind the main-text claims. The cohorts that every experiment shares are fixed in Appendix A.1. Appendix A.2 specifies the attacks, metrics, and paired tests behind Tables 1 and 2, and Appendix A.3 defines the silent-collision readout. The TNAS construction and the hook choice of Section 2.1 are detailed in Appendix A.4. For Probe 1, Appendix A.5 evaluates the rationale-only reader and Appendix A.6 audits the projections. Appendix A.7 lists the values plotted in Figure 2, and Appendix A.8 details the sign-aligned amplification protocol of Probe 3. Asset terms and compute resources close the appendix in Appendices A.9 and A.10.

## A.1 Datasets and paired cohorts

**Alpamayo evaluation subset.** The Alpamayo experiments use the gated NVIDIA PhysicalAI-Autonomous-Vehicles HuggingFace dataset together with the nvidia/Alpamayo-1.5-10B checkpoint. We start from the clip manifest distributed with the model release, which lists 1,181 unique clip IDs, and download each clip’s egomotion stream and four camera streams through the physical\_ai\_av dataset interface: the $1 2 0 ^ { \circ }$ cross-left, front-wide, and cross-right cameras and the $\dot { 3 } 0 ^ { \dot { \circ } }$ front-telephoto camera. Probes read each clip at the canonical reference time $t _ { 0 } = 5 { , } 1 0 0 { , } 0 0 0 \; \mu \mathrm { s }$ with all four cameras and four temporal frames, for 16 images per clip.

The 200-clip subset is constructed deterministically. We list every clip ID that has a non-empty clean rationale and a resolved local cache, sort the list lexicographically, shuffle it with Python’s random.Random(0), and take the first 200 entries. All Alpamayo probe and input experiments draw from this one ordered list.

**InstructVLA evaluation subset.** The InstructVLA experiments use 200 Bridge-distribution examples from the VLA-IT language evaluation set. The released VLA-IT split fixes the example list, so no sampling seed is involved. The hidden-state probes and the full-image and patch input evaluations all use this ordered list, and every comparison is paired by example ID. Appendices A.4.1 and A.4.2 describe hook selection and the differentiable action readout.

**Evaluation cohorts.** Each model’s probe and input results use its 200-example subset. The Alpamayo safety analysis uses the 154 of those 200 clips that have obstacle annotations in the dataset release, as described in Appendix A.3. Appendix A.6 also reports the projection ratio on the 1,180 manifest clips with a nonempty rationale, and the benign rationale-only reader of Appendix A.5 uses all 1,181 clips of the manifest.

## A.2 Input protocols and text/action metrics

**Alpamayo attack settings.** All Alpamayo attacks use the objectives of Section 3.1 with text weight $\lambda _ { \mathrm { t x t } } = 1$ . The text penalty is the raw change in clean-target NLL plus the target-margin drop, with no hinge. During optimization, both trajectories come from a differentiable 4-step flow rollout with seed 42, and every reported readout uses the 10-step solver of Appendix A.2.1. PGD starts from uniform noise in $\left[ - \alpha , \alpha \right]$ , where α is the step size, drawn with seed 0 for each clip and method, and every row reports the final step 200.

The patch attack optimizes one pattern per clip. The processor resizes each camera frame to $3 2 0 \times 5 7 6$ pixels, a grid of $\bar { 2 0 } \times 3 6$ input patches of $1 6 \times 1 6$ pixels. In the front-telephoto frame, the pattern covers $1 2 \times 2 2$ patches, or $1 9 2 \times 3 5 2$ pixels, centered horizontally and flush with the bottom edge. A bilinear resize of the whole pattern covers 8 × 14 patches, or $1 2 8 \times 2 2 4$ pixels, of the front-wide frame, centered horizontally with its top edge at the vertical center. Both copies repeat in all four frames of their camera, and the cross-left and cross-right frames stay clean. The patched area is 1,504 of the 11,520 input patches, or 13.1%. Figure 3 shows this layout on two clips.

**InstructVLA text-loss adaptation.** The InstructVLA text-preserving baseline uses a one-sided hinge on clean-response positions $R .$ Let $m _ { t }$ be the target-token logit minus the largest non-target logit, let $\mathcal { L } _ { \mathrm { N L L } }$ be mean target-token NLL, and write $\bar { [ x ] _ { + } } = \operatorname* { m a x } ( \bar { x , 0 } )$ . Its text penalty is

$$
\mathcal { L } _ { \operatorname { t x t } } ^ { \operatorname { I V L A } } = \frac { 1 } { | R | } \sum _ { t \in R } [ m _ { t } ^ { \operatorname { c l n } } - m _ { t } ^ { \operatorname { a d v } } - \tau _ { m } ] _ { + } + [ \mathcal { L } _ { \operatorname { N L L } } ^ { \operatorname { a d v } } - \mathcal { L } _ { \operatorname { N L L } } ^ { \operatorname { c l n } } - \tau _ { n } ] _ { + } .
$$

15

<!-- page 16 of 23 -->

![Image block](doc:e4f9533/tier:basic/page:16/block:1)

Figure 3: Shared forward-camera patch in Alpamayo. Illustrative patches on two clips of the evaluation cohort, optimized for 200 PGD steps at $\lambda _ { \mathrm { c o s } } = 2 0$ in the patch layout of Table 1. Rows are the four cameras and columns the four input frames. Red boxes mark the patch. In every frame, the optimized pattern fills the front-telephoto box and a resized copy fills the front-wide box, while the cross-view cameras stay clean.

The reported hinge family uses $\tau _ { m } = \tau _ { n } = 0 ,$ unit action and text weights, and no additional KL penalty. The maximized objective is the mean squared action-coordinate displacement minus this penalty, plus $\lambda _ { \operatorname { c o s } } \operatorname { c o s } ( \Delta h , \tilde { d } _ { \operatorname { T N A S } } )$ for guided rows. The text-preserving baseline is the $\lambda _ { \mathrm { c o s } } = 0$ member, and the action-only arm drops both the text and the guidance terms. Within the full-image hinge family, the runs differ only in the guidance weight.

**InstructVLA attack settings.** The localized run uses a rectangular mask of 40% of the input height and 40% of its width, centered below the image center, and keeps the perturbed input inside [−1, 1]. As in Alpamayo, PGD starts from seeded uniform noise in $\left[ - \alpha , \alpha \right]$ , restricted to the mask for the patch, and every row reports the final step 200. The perturbation enters the VLM/SigLIP input, and the action expert keeps its clean DINO input in both full-image and patch runs.

16

<!-- page 17 of 23 -->

## A.2.1 Action decoding and metrics

Both models decode actions with an explicit 10-step Euler flow-matching solver from $t = 0 \; \mathrm { t o } \; t = 1$ without classifier-free guidance, and share the initial Gaussian noise between paired runs through seed 42. Alpamayo’s solver state is a $6 4 \times 2$ sequence of normalized acceleration and curvature controls, which the released unicycle action space integrates into the waypoints of Equation 1. InstructVLA’s solver state is the $1 6 \times \dot { 7 }$ normalized action chunk itself, and $\Delta _ { \mathrm { I V L A } }$ reads the sampler’s normalized outputs before deployment clipping, gripper thresholding, and dataset-specific unnormalization, so its coordinates can leave [−1, 1].

## A.2.2 Definition and implementation of teacher-forced KL

The teacher-forced KL metric of Section 2.1 is the forward KL from the clean to the perturbed next-token distribution, summed over the vocabulary and averaged over the active rationale/response span. With clean and perturbed per-token logits $z _ { t } ^ { \vec { \mathrm { c l n } } } , z _ { t } ^ { \mathrm { a d v } } \in \mathbb { R } ^ { \overleftarrow { V } }$ and $p _ { t } ^ { \bullet } = \mathrm { s o f t m a x } ( z _ { t } ^ { \bullet } )$ ),

$$
\operatorname { K L _ { t f } } ( \delta ) \; = \; \frac { 1 } { | R | } \sum _ { t \in R } \sum _ { v = 1 } ^ { V } p _ { t , v } ^ { \operatorname { c l n } } \bigl [ \operatorname { l o g } p _ { t , v } ^ { \operatorname { c l n } } - \operatorname { l o g } p _ { t , v } ^ { \operatorname { a d v } } \bigr ] ,\tag{9}
$$

which is computed in log-space for numerical stability by applying log\_softmax separately to $z _ { t } ^ { \mathrm { c l n } }$ and $z _ { t } ^ { \mathrm { a d v } }$ before forming the inner sum. The metric is reported in nats. The clean rationale or response is fixed as the teacher-forced target sequence at every $t \in R$ . At each position, the paired runs share the textual prefix and target token, and they differ only through the specified hidden-state intervention for local probes or visual-input intervention for input attacks.

**Cross-table reading.** Hidden-state probes select steps from finite alpha grids under the modelspecific KL-budget rules of Appendix A.4, while input attacks optimize under visual-input budgets with no fixed KL target. Both report Equation 9 on the corresponding model’s active text span.

## A.2.3 Free-generation exact match

**Alpamayo sampled decoding.** For each clean or perturbed visual input, we run the VLM autoregressively without teacher forcing, with one sampled sequence, top-p = 0.98, temperature 0.6, at most 256 new tokens, and generation seed 42 shared between paired runs. Generation stops after the trajectory-start token, after which we extract the rendered chain-of-thought span. Sharing the random seed between paired clean and perturbed runs isolates the effect of the visual perturbation from sampling noise.

**InstructVLA greedy decoding.** Clean and perturbed responses are generated separately with sampling disabled and a single beam. The decoder extracts the generated suffix, truncates it at the designated meta and end markers, &lt;new_token_0&gt; and <|im\_end|>, and strips surrounding whitespace. Exact match is equality of these cleaned strings. The 200 clean responses are short, templated intention statements of 9.60 words on average, with a median of 9, a range of 5 to 17, and 178 distinct strings. Each one summarizes a full 16-step action chunk, the many-to-one intention bottleneck of Section 3.2.

For either model, if $c _ { \mathrm { c l n } }$ and $c _ { \mathrm { a d v } }$ are the strings extracted under its protocol, the metric is

$$
\operatorname { E x a c t } _ { \operatorname { f r e e } } = \mathbf { 1 } \{ c _ { \operatorname { c l n } } = c _ { \operatorname { a d v } } \} .\tag{10}
$$

## A.2.4 Paired tests for the input tables

Every row of Tables 1 and 2 is paired by example through the shared seeds above, so Table 3 compares arms with exact paired tests on the comparisons interpreted in Section 3. At the strongest guidance weights, TNAS guidance significantly lowers teacher-forced KL and raises exact match over Text-preserving PGD in both models. In Alpamayo, its attacked-plan and silent-collision counts stay level with Text-preserving PGD and far above Action-only PGD. The InstructVLA patch matches the exact match of full-image Text-preserving PGD with significantly lower KL.

## A.3 Logged-scene open-loop safety protocol

**Cohort, time alignment, and geometry.** Displacement and geometric collision readouts are standard in open-loop planning evaluations [15, 16]. We score Alpamayo’s free-generation plans

17

<!-- page 18 of 23 -->

Table 3: Paired tests for the input tables. Binary outcomes list the counts of the two arms and the discordant examples favoring each arm, tested with exact McNemar tests. KL rows count the examples on which the first arm has lower teacher-forced KL, tested with exact sign tests. Exact match and KL use 200 examples, and collisions use the 154 annotated clips.

<table><tr><td>Outcome</td><td>Counts</td><td>Discordant</td><td>p</td></tr><tr><td>Alpamayo: full image</td><td> $\lambda _ { \mathrm { c o s } } = 3 0 \; \mathrm { v s } .$ </td><td>Text-preserving PGD</td><td></td></tr><tr><td>Exact match</td><td> $1 7 9 \; \mathrm { v s . } \; 1 6 6$ </td><td>20/7</td><td>0.0192</td></tr><tr><td>Lower  $\mathrm { K L } _ { \mathrm { t f } }$ </td><td> $1 7 9 \; \mathrm { o f } \; 2 0 0$ </td><td></td><td> $1 . 9 5 \times 1 0 ^ { - 3 2 }$ </td></tr><tr><td>Attacked collision</td><td> $9 5 \; \mathrm { v s . } \; 9 4$ </td><td>22/21</td><td>1.00</td></tr><tr><td>Silent collision</td><td> $8 1 \; \mathrm { v s . } \; 7 5$ </td><td>25/19</td><td>0.451</td></tr><tr><td colspan="4">Alpamayo: full image  $\lambda _ { \mathrm { c o s } } = 3 0 \; \mathrm { v s }$  . Action-only PGD</td></tr><tr><td>Silent collision</td><td> $8 1 \; \mathrm { v s . } \; 7$ </td><td>75/1</td><td> $2 . 0 4 \times 1 0 ^ { - 2 1 }$ </td></tr><tr><td>InstructVLA: full image</td><td> $\lambda _ { \mathrm { c o s } } = 0 . 2 0 \; \mathrm { v s } .$ </td><td>. Text-preserving PGD</td><td></td></tr><tr><td>Exact match</td><td> $\widetilde { 1 } 7 0 \: \mathrm { v s . } \: 1 5 5$ </td><td>24/9</td><td>0.0135</td></tr><tr><td>Lower  $\mathrm { K L } _ { \mathrm { t f } }$ </td><td> $1 6 3 \; \mathrm { o f } \; 2 0 0$ </td><td></td><td> $4 . 5 8 \times 1 0 ^ { - 2 0 }$ </td></tr><tr><td colspan="4">InstructVLA: patch  $\lambda _ { \mathrm { c o s } } = 0 . 2 0 \; \mathrm { v s } .$ </td></tr><tr><td> $\mathrm { E x a c t   m a t c h }$ </td><td> $1 5 5 \; \mathrm { v s . } \; 1 5 5$ </td><td>20/20</td><td>. full-image Text-preserving PGD 1.00</td></tr><tr><td> $\mathbf { L o w e r \: K L _ { t f } }$ </td><td> $1 3 0 \; \mathrm { o f } \; 2 0 0$ </td><td></td><td> $2 . 6 5 \times 1 0 ^ { - 5 }$ </td></tr></table>

against time-aligned logged actors on the same 154 annotated clips for every arm, in the ego frame fixed at $t _ { 0 } .$ . Actor boxes are transformed from their observation-time rig frames into that fixed frame using logged ego poses, then interpolated in position and unwrapped heading to the planning timestamps within their observed intervals. A single-observation track is used only at its nearest planning timestamp if within 0.05 s. Following standard open-loop protocol, actors follow their recorded trajectories over the 6.4 s horizon.

Ego dimensions and the rear-axle-to-box-center offset come from the clip’s vehicle metadata, and the ego heading inferred from consecutive waypoints keeps its previous value for steps shorter than 1 mm. Actor dimensions are the per-track median annotated length and width. Apparent ego self-tracks, whose median center lies within 1 m of the expert ego footprint, are removed.

**Silent collision rate.** For clip $i ,$ let $M _ { i }$ denote exact free-generation rationale equality, $C _ { i }$ the clean-plan collision flag, and $A _ { i }$ the attacked-plan collision flag. We report

$$
\frac { 1 } { 1 5 4 } \sum _ { i = 1 } ^ { 1 5 4 } \mathbf { 1 } \{ M _ { i } \wedge A _ { i } \wedge \neg C _ { i } \} .
$$

The three flags come from paired records for the same clip and decoding branch. On these 154 clips, the exact-match counts $\sum _ { i } M _ { i }$ are 16, 131, $1 2 6 / 1 2 8 / \dot { 1 3 } 3 / 1 3 9 / 1 4 1$ , and $1 1 1 / 1 1 1 / 1 1 0 / 1 1 7 / 1 ^ { 2 }$ 23 for the action-only, text-preserving, full-image $\dot{\lambda_{ cos }}^{\prime} = \dot{ } 5/\dot{10}/15/20/30,$ and patch $\lambda _ { \mathrm { c o s } } ~ =$ $5 / 1 0 / 1 5 / 2 0 / 3 0$ arms, and the attacked-plan collision counts $\sum _ { i } A _ { i }$ are 91, 94, $9 4 / 9 2 / 8 6 / 8 7 / 9 5$ and $\dot { 7 } 8 / \dot { 8 } 2 / \dot { 7 } 3 / 7 4 / 7 2 .$

## A.4 TNAS implementation and hook selection

**Pooled summaries and additive intervention coordinates.** For a layer block $L = \{ \ell _ { 1 } , \ldots , \ell _ { K } \}$ and active token span R, the pooled summary concatenates per-layer token means,

$$
\bar { H } _ { \ell } = \frac { 1 } { | R | } \sum _ { t \in R } H _ { \ell } ^ { ( t ) } , \qquad h = \left( \bar { H } _ { \ell _ { 1 } } , \ldots , \bar { H } _ { \ell _ { K } } \right) \in \mathbb { R } ^ { K d } .\tag{11}
$$

The intervention handle broadcasts one vector per layer over the same span,

$$
H _ { \ell } ^ { ( t ) } \leftarrow H _ { \ell } ^ { ( t ) } + \delta _ { \ell } , \qquad t \in R , \; \ell \in L ,\tag{12}
$$

with one unscaled additive coordinate per layer and hidden dimension. The summary h is not a standalone sufficient input to the model: a readout written at $h + \delta$ denotes the full hooked forward pass, with the remaining inputs and decoding configuration fixed.

18

<!-- page 19 of 23 -->

Table 4: Alpamayo layer-window sweep. Matched teacher-forced text-KL budget $1 0 ^ { - 4 }$ on 50 clips per block. At layers 24 to 27, 47 clips have a TNAS row and 49 a random row within the budget.

| Layers | Mean ρ (%) | TNAS ADE (m) | Random ADE (m) |
| --- | --- | --- | --- |
| 24 to 27 | 2.89 | 0.195 | 0.00284 |
| 28 to 31 | 2.34 | 0.161 | 0.00276 |
| 32 to 35 | 1.99 | 0.226 | 0.00356 |

**Hooked action Jacobian and residual.** With $A _ { \mathrm { h o o k } }$ and $J _ { A }$ the flattened action output of that pass and its Jacobian at $\delta = 0 ,$ , as defined in Section 2.1, Alpamayo targets

$$
d _ { \operatorname { a c t } } ^ { \star } \in \operatorname { a r g } \operatorname* { m a x } _ { \| v \| _ { 2 } = 1 } \lVert J _ { A } v \rVert _ { 2 } ,
$$

materializing the Jacobian by differentiating each output coordinate and using the first right-singular vector from a thin torch.linalg.svd. A Jacobian norm at most $1 0 ^ { - 1 2 }$ returns a zero direction. InstructVLA uses the normalized Gaussian-output VJP construction in Appendix A.4.2. Both models form the text gradients and residual $r   =   ( \bar { I - P _ { \mathcal { T } } } ) d _ { \mathrm { a c t } }$ in these unscaled δ coordinates, before intervention scaling.

**Matched-budget perturbations.** For unit directions $u \: \in \: \{ d _ { \operatorname { T N A S } } , d _ { \operatorname { a c t } } , g _ { \operatorname { N L L } } , g _ { \operatorname { m a r g i n } } , u _ { \operatorname { r a n d } } \}$ matched-budget probes use the additive hook and model-specific clean-state scales. For Alpamayo, the step is per-layer RMS-scaled,

$$
\delta _ { \ell } = s \cdot \alpha \cdot \sigma _ { h , \ell } \cdot u _ { \ell } , \qquad s \in \{ - 1 , + 1 \} ,\tag{13}
$$

where $\sigma _ { h , \ell } = \operatorname { r m s } ( \bar { H } _ { \ell } )$ . Equivalently, $\delta = s \alpha S u$ with $S = \operatorname { d i a g } ( \sigma _ { h , \ell _ { 1 } } I _ { d } , \ldots , \sigma _ { h , \ell _ { K } } I _ { d } )$ .InstructVLA applies a single captured clean-state scale $\sigma _ { \mathrm { I V L A } }   =   \operatorname* { m a x } ( 0 . 0 1   \mathrm { r m s } ( h _ { \mathrm { m e t a } } ) , 1 0 ^ { - 3 } )$ across hooked layers, so $\delta ~ = ~ s \alpha \sigma _ { \mathrm { I V L A } } u .$ Here $h _ { \mathrm { m e t a } }$ concatenates the per-layer mean clean meta-token states, captured separately from the response-token intervention. Alpamayo’s action-facing and random directions use $\vec { \alpha \in } \{ 0 . 0 3 , 0 . 1 , \bar { 0 } . 3 , 0 . 5 , 1 . 0 , 3 . 0 , 6 . 0 \}$ , and its text-gradient directions use $\{ 0 . 0 3 , 0 . 1 , 0 . 1 5 , 0 . 2 5 , 0 . 3 , \dot { 0 . 4 , 0 . 5 , 1 . 0 } \}$ . InstructVLA uses α $\in \{ 0 . 2 5 , \bar { 0 . 5 , 1 , 2 , 4 , 8 , 1 6 , 3 2 } \}$ as detailed in Appendix A.4.2. For each direction and text-KL budget, Alpamayo selects the available row with the largest achieved teacher-forced KL no greater than the target budget, and InstructVLA selects the row closest to the target budget within the 1.4B ceiling described in Appendix A.4.2. Alpamayo random controls in Probes 2 and 3 use 16 isotropic unit-vector seeds per clip, and the InstructVLA matched-budget grids use random seeds 0 and 1.

## A.4.1 Layer-window sensitivity and hook viability

The Alpamayo layer block comes from a sweep over three late VLM blocks, layers 24 to 27, 28 to 31, and 32 to 35. Each block runs the five directions of Probe 2 with alphas {0.03, 0.1, 0.3, 0.5, 1.0, 3.0, 6.0} at a teacher-forced text-KL budget of $1 0 ^ { - 4 }$ on the same 50 clips drawn from the 200-clip cohort. All three blocks recover rank-2 active text spans on every clip and show the same action/random asymmetry. As Table 4 shows, layers 32 to 35 give the strongest matched-budget action response, the selection rule of Section 2.1, and also the smallest projection ρ. Adjacent blocks reproduce the same pattern at smaller magnitude.

**InstructVLA hook/gradient viability audit.** For InstructVLA, a viability audit checks that a candidate hook exposes both a nonzero text basis on its token span and a nonzero action sensitivity through the flow action expert. Table 5 compares meta- and response-token spans in three four-layer windows. Response-token spans are feasible in all three windows, and layers 14 to 17 give the largest action response among them. Meta-token spans move the action but have zero text gradients, so they are pure action-interface states rather than shared text/action states. We therefore select the response-token 14 to 17 window.

## A.4.2 InstructVLA action-direction estimator and budget selection

Gradients pass from the response-token hook through latent action features to the flow expert by bypassing inference-only detaches and using out-of-place Euler updates. To estimate the action-sensitive direction without materializing the full Jacobian through the flow rollout, we draw an output-space

19

<!-- page 20 of 23 -->

Table 5: InstructVLA hook viability audit. Audit on three Bridge-distribution examples. Feasible cells have nonzero text gradients, a rank-2 text basis, and a nonzero action proxy. The selected hook is layers 14 to 17 on response tokens.

| Layers | Span | NLL grad | Margin grad | Action proxy | Feasible (%) |
| --- | --- | --- | --- | --- | --- |
| 4 to 7 | meta | 0 | 0 | 1.39 × 10<sup>-2</sup> | 0 |
| 4 to 7 | response | 0.109 | 0.849 | 4.82 × 10<sup>-3</sup> | 100 |
| 14 to 17 | meta | 0 | 0 | 2.07 × 10<sup>-2</sup> | 0 |
| 14 to 17 | response | 0.199 | 1.27 | 7.54 × 10<sup>-3</sup> | 100 |
| 24 to 27 | meta | 0 | 0 | 1.94 × 10<sup>-2</sup> | 0 |
| 24 to 27 | response | 0.0779 | 0.388 | 1.51 × 10<sup>-3</sup> | 100 |

Gaussian vector $z _ { 0 }$ and normalize $w _ { 0 } = z _ { 0 } / ( \| z _ { 0 } \| _ { 2 } + 1 0 ^ { - 8 } )$ , then compute a single vector-Jacobian product at $\delta = 0$ followed by a central-difference directional gain check $( \epsilon = 0 . 0 1 )$

$$
v = J _ { A } ^ { \top } w _ { 0 } , \qquad d _ { \operatorname { a c t } } = \frac { v } { \| v \| _ { 2 } } , \qquad j = \frac { A _ { \operatorname { h o o k } } ( \epsilon d _ { \operatorname { a c t } } ) - A _ { \operatorname { h o o k } } ( - \epsilon d _ { \operatorname { a c t } } ) } { 2 \epsilon } .
$$

A VJP norm at most $1 0 ^ { - 1 2 }$ returns a zero direction, and $\| j \| _ { 2 }$ is recorded as the directional gain.

**Finite-grid budget selection.** Each evaluation estimates the per-example directions once on the 200 example IDs and holds them fixed throughout its perturbation sweep. The matched-budget sweep uses $\alpha \in \{ 0 . 2 5 , 0 . 5 , 1 , 2 , 4 , 8 , 1 6 , 3 2 \}$ so that action-aligned directions reach the $B = 1 0 ^ { - 3 }$ KL budget on all 200 examples. For nominal budget $B \in \{ 5 \times \overset { \mathcal { C } } { 1 0 ^ { - 4 } } , 1 0 ^ { - 3 } \}$ , the analysis selects the row with achieved teacher-forced KL closest to B within a 1.4B ceiling. On this grid, action-aligned directions reach a maximum mean text KL of about $1 . 5 0 \times 1 0 ^ { - 3 } ,$ while text-aligned directions reach about $1 0 ^ { - 2 }$ . Across the five directions in Table 7, mean achieved KL is 97.8% to 101% of B at $B = 5 \times 1 0 ^ { - 4 }$ and 94.4% to 98.9% of B at $B = \dot { 1 0 } ^ { - 3 }$ , keeping at least 186 examples per direction at $B = 5 \times 1 0 ^ { - 4 }$ and all 200 examples at $B = 1 0 ^ { - 3 }$

## A.5 Benign rationale-only reader

**Cohort and reader.** As described in Appendix A.1, the reader uses all 1,181 clips of the release manifest. Each clip’s clean rationale and planned trajectory come from one greedy Alpamayo run with generation and flow seed 42, and the clips contain 258 distinct rationale strings. Qwen/Qwen3-14B in bf16, with thinking disabled and greedy decoding, receives one rationale string per call together with the horizon and the label definition, and returns a binary stop prediction and a stop probability. It never receives the image, the trajectory, the ego speed, or the map. Each distinct string is read once, so identical rationales receive identical predictions, and all 258 outputs parse. These benign statistics use greedy rationales and are not paired with the sampled-decoding attack results of Section 3.

**Prompt and readout.** Each distinct rationale is inserted into the user prompt below, which goes through the Qwen3 chat template, and the reply is capped at 200 new tokens. The binary prediction is the will\_stop field of the returned JSON object, and the stop probability is the stop\_prob number that the model writes into the same object. No token probabilities are used. The three remaining fields do not enter this paper’s analysis. In the prompt, [RATIONALE] stands for the rationale string.

```txt
You are given one sentence: the rationale that an autonomous-driving model produced to explain its own
driving plan for the next 6.4 seconds.
Predict what the vehicle will actually do. You see only this sentence. You do not see the camera images,
the planned trajectory, the vehicle's current speed, or the map.
Sentence: "[RATIONALE]"
Answer with a single JSON object and nothing else:
{"will_stop": <true|false>,
"stop_prob": <number 0..1>,
"longitudinal": "<accelerate|maintain|decelerate|stop>",
"lateral_dir": "<left|right|none>",
"lateral_mag": "<none|small|large>"}
Definitions:
- will_stop: the vehicle comes to rest (below 0.5 m/s) before the 6.4 s horizon ends.
```

20

<!-- page 21 of 23 -->

Table 6: Benign rationale-only reader. Planned outcomes against reader predictions on 1,181 clean Alpamayo clips. Shares are within each prediction column.

<table><tr><td></td><td colspan="2">Reader: will stop</td><td colspan="2">Reader: keep going</td><td></td></tr><tr><td>Plan outcome</td><td>Clips</td><td>Share (%)</td><td>Clips</td><td>Share (%)</td><td>Total</td></tr><tr><td>Comes to rest</td><td>243</td><td>63.6</td><td>46</td><td>5.76</td><td>289</td></tr><tr><td>Does not come to rest</td><td>139</td><td>36.4</td><td>753</td><td>94.2</td><td>892</td></tr><tr><td>Total</td><td>382</td><td>100</td><td>799</td><td>100</td><td>1,181</td></tr></table>

- stop\_prob: your probability that will\_stop is true. - longitudinal: speed in the final 0.8 s vs the first 0.8 s. "stop" if it comes to rest, "decelerate" if below 70% of the initial speed, "accelerate" if above 130%, else "maintain". - lateral\_dir / lateral\_mag: net sideways departure from the starting heading, in metres, measured at its largest point. Count sideways motion caused by the road curving as well as by a deliberate lane change or nudge. "left" and "right" are from the driver's seat. none = under 1 m, small = 1 to 3 m, large = 3 m or more.

**Stop label.** Speeds are taken from consecutive planned waypoints at 10 Hz. A plan comes to rest within the horizon if its mean speed over the final 0.8 s of the 6.4 s horizon is below 0.5 m/s. Table 6 reports the reader’s predictions against this label. The reader recovers 84.1% of stopping and 84.4% of non-stopping plans, 84.3% overall. Its stop probability ranks stopping plans with AUROC 0.882, above 0.820 for a no-text baseline that ranks clips by entry speed from the ego history, with tied scores assigned their average rank.

**Mixed-outcome rationales.** A rationale has mixed outcomes when its clips include both stopping and non-stopping plans. Of the 258 rationales, 23 are mixed, and together they cover 436 clips. Because the label is read at the end of the horizon, mixed outcomes include vehicles waiting at a red light that depart within the horizon. All 19 non-stopping clips with the rationale “Stop for the red traffic light” start at rest. The statistic therefore measures how much of the 6.4 s plan a sentence leaves unspecified, not how often the sentence is wrong.

## A.6 Projection audits

**Projection statistics.** Across the 200 Alpamayo clips, the projection ratio ρ of Probe 1 has mean 1.87% and median 1.79%. Extending the cohort to all 1,180 manifest clips with a nonempty rationale leaves the mean at 1.87%. The random unit-vector reference draws 1,000 isotropic unit vectors per clip. Over these 200,000 samples it has mean 0.978%, median 0.916%, 95th percentile 1.91%, and 99th percentile 2.36%. Of the 200 clips, 43.5% exceed the random 95th percentile and 25.5% exceed the 99th, against 5% and 1% expected by chance. The largest per-clip value is $\rho = 5 . 2 8 \%$ , so the TNAS residual keeps more than 99.8% of the action direction’s norm on every clip. In InstructVLA, ρ has mean 5.28% and median 4.90% over the 200 examples, and its mean square of 0.00382 leaves 99.8% of the norm on average. Rescaling either text-gradient basis vector by 0.01, 0.1, 1, 10, or 100 before re-orthonormalization changes the action projection ratio by at most $4 . 8 4 \times 1 0 ^ { - 8 }$ , so the ratio depends only on the text span.

## A.7 Hidden-state intervention probe values

Tables 7 and 8 list the cohort means plotted in Figure 2c–h.

## A.8 Sign-aligned clean-state amplification

For the clean-state amplification of Probe 3, we sweep a raw scalar s with

$$
h ( s , u ) = h _ { \operatorname { c l e a n } } + s \thinspace S _ { \operatorname { c l i p } } u , \qquad S _ { \operatorname { c l i p } } = \big | \langle h _ { \operatorname { c l e a n } } , d _ { \operatorname { T N A S } } \rangle \big | .\tag{14}
$$

For both models, write $c   =   \langle h _ { \mathrm { c l e a n } } , d _ { \mathrm { T N A S } } \rangle$ and aggregate by the sign-aligned parameter $q =$ $s \cdot \operatorname { s g n } ( c )$ . Thus $q = 0$ is the clean state, and all directions have matched per-clip norm $\| q S _ { \mathrm { c l i p } } \bar { u } \|$ at fixed q. Because the SVD sign of a direction is arbitrary, q aligns the amplification curves by the sign of the natural projection. The InstructVLA clean-state amplification sweep averages four random seeds per example for $u _ { \mathrm { r a n d } } ,$ , while deterministic named directions are evaluated once.

21

<!-- page 22 of 23 -->

Table 7: Probe 2: Matched-budget action shifts in Alpamayo and InstructVLA. Alpamayo entries are mean ADE shifts in meters on 200 clips, or 199 for $g _ { \mathrm { N L L } }$ at $B = 1 0 ^ { - 4 }$ , at nominal teacher-forced KL budgets B. InstructVLA entries are mean per-step normalized 7-DoF action shifts with the admissible count n, which covers all 200 examples at $\tilde { B } = 1 0 ^ { - 3 }$ as detailed in Appendix A.4.2.

<table><tbody><tr><td rowspan="2"></td><td>Alpamayo</td><td>ADE shift (m)</td><td colspan="4">InstructVLA norm. ℓ<sub>2</sub> shift</td></tr><tr><td></td><td></td><td>B = 5 ×</td><td>10<sup>-4</sup></td><td>B = 10</td><td>-3</td></tr><tr><td>Direction</td><td>B = 10<sup>-4</sup></td><td>B = 5 × 10<sup>-4</sup></td><td>Shift</td><td>n</td><td>Shift</td><td>n</td></tr><tr><td>d<sub>TNAS</sub></td><td>0.227</td><td>0.504</td><td>0.0131</td><td>189</td><td>0.0137</td><td>200</td></tr><tr><td>d<sub>act</sub></td><td>0.181</td><td>0.527</td><td>0.0144</td><td>186</td><td>0.0166</td><td>200</td></tr><tr><td>urand</td><td>0.00359</td><td>0.00510</td><td>0.00848</td><td>194</td><td>0.00752</td><td>200</td></tr><tr><td>g<sub>NLL</sub></td><td>0.00240</td><td>0.00238</td><td>0.00760</td><td>186</td><td>0.00918</td><td>200</td></tr><tr><td>g<sub>margin</sub></td><td>0.00229</td><td>0.00252</td><td>0.00916</td><td>186</td><td>0.00817</td><td>200</td></tr></tbody></table>

Table 8: Probe 3: Norm-matched clean-state amplification on 200 Alpamayo clips and 200 InstructVLA examples. At each strength $q ,$ all directions receive the same per-example intervention norm $| q | S _ { \mathrm { c l i p } } ,$ and $q = 0$ is the clean state. The KL columns report teacher-forced KL at $q = 2 .$

<table><tr><td rowspan="3"></td><td colspan="5">Alpamayo</td><td colspan="5">InstructVLA</td></tr><tr><td colspan="3">ADE shift (m)</td><td colspan="2"></td><td colspan="3">Norm. shift  $\ell _ { 2 }$ </td><td colspan="2">KL</td></tr><tr><td>Direction  $q = 0 . 5$ </td><td> $q = 1$ </td><td></td><td> $q = 2$ </td><td> $q = 2$ </td><td> $q = 0 . 5$ </td><td> $q = 1$ </td><td> $q = 2$ </td><td></td><td> $q = 2$ </td></tr><tr><td> $d _ { \mathrm { T N A S } }$ </td><td>0.0317</td><td>0.0673</td><td>0.142</td><td></td><td> $9 . 8 7 \times 1 0 ^ { - 5 }$ </td><td>0.0734</td><td>0.110</td><td>0.156</td><td></td><td> $2 . 2 0 \times 1 0 ^ { - 2 }$ </td></tr><tr><td> $d _ { \mathrm { a c t } }$ </td><td>0.0316</td><td>0.0675</td><td>0.142</td><td></td><td> $1 . 0 3 \times 1 0 ^ { - 4 }$ </td><td>0.0734</td><td>0.110</td><td>0.155</td><td></td><td> $2 . 7 1 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Urand</td><td>0.00223</td><td>0.00261</td><td>0.00264</td><td></td><td> $6 . 4 5 \times 1 0 ^ { - 5 }$ </td><td>0.00861</td><td>0.00999</td><td>0.0124</td><td></td><td> $3 . 6 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>gNLL</td><td>0.00234</td><td>0.00262</td><td>0.00297</td><td></td><td> $2 . 6 6 \times 1 0 ^ { - 2 }$ </td><td>0.0134</td><td>0.0197</td><td>0.0283</td><td></td><td> $5 . 8 0 \times 1 0 ^ { - 1 }$ </td></tr><tr><td> $g _ { \mathrm { m a r g i n } }$ </td><td>0.00217</td><td>0.00263</td><td>0.00323</td><td></td><td> $2 . 7 3 \times 1 0 ^ { - 2 }$ </td><td>0.0158</td><td>0.0198</td><td>0.0337</td><td></td><td> $4 . 6 6 \times 1 0 ^ { - 1 }$ </td></tr></table>

## A.9 Existing asset licenses and terms

The experiments use existing datasets, checkpoints, and annotator models only for analysis. We do not redistribute raw clips, VLA-IT examples, checkpoints, adapter weights, or a repackaged derived dataset. Alpamayo uses the gated nvidia/PhysicalAI-Autonomous-Vehicles dataset under the NVIDIA Autonomous Vehicle Dataset License Agreement (nvidia-av-dataset), accessed through HuggingFace, and the nvidia/Alpamayo-1.5-10B checkpoint under the NVIDIA License for non-commercial research or evaluation use. The Alpamayo inference code is Apache-2.0. The benign rationale-only reader uses Qwen/Qwen3-14B [10] under Apache-2.0. InstructVLA uses the public ShuaiYang03/instructvla\_finetune\_v2\_xlora\_freeze\_head\_instruction checkpoint, ShuaiYang03/InstructVLA\_Assets, and the ShuaiYang03/InstructVLA\_embodied\_ mm\_evaluation\_assets language-evaluation files. We credit these releases and the InstructVLA paper, and we use only the fixed evaluation examples. As of November 2026, the public InstructVLA release pages and the InternRobotics/InstructVLA code repository list no explicit license for these assets, so we treat their license status as unspecified. The InstructVLA asset page identifies nvidia/Eagle2-2B [55] as the VLM base, whose HuggingFace card lists CC-BY-NC-4.0. For all assets with gated or restrictive terms, the experiments use local access only and do not distribute the protected asset contents. Rationale-preserving visual manipulations also pose a misuse risk: they can alter predicted actions without changing the visible explanation. Runnable attack pipelines and optimized perturbation tensors are not distributed with this manuscript.

## A.10 Compute resources

The experiments ran on a single HPC node with up to four NVIDIA A100-80GB GPUs and one Python worker per GPU. Each Alpamayo worker loads one model copy and uses about 25 GB of GPU memory. The InstructVLA probes use the same GPU type. We estimate that reproducing the Alpamayo hidden-state probes and clean-state amplification curves from cached model and data artifacts takes 75 to 85 A100 GPU-hours, including about 25 GPU-hours for the local-probe alpha grids and 30 GPU-hours for the intervention sweeps. The InstructVLA local probes and clean-state amplification take an estimated 20 to 24 GPU-hours. The input-attack costs below are measured wall-clock times summed over workers. Each Alpamayo guidance weight takes about 47 GPU-hours

22

<!-- page 23 of 23 -->

on the 200 clips, or about 14 GPU-minutes per clip, for both the full-image and the patch attack. The Action-only and Text-preserving rows come from a shared 200-clip run of about 85 GPU-hours. Each InstructVLA input-attack arm takes about 13 GPU-hours on the 200 examples. Merging, row selection, summaries, and the open-loop collision scoring run on CPU and take minutes per stage.

23