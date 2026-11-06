<!-- page 1 of 25 -->

arXiv:2611.05377v1 [cs.CV] 3 Nov 2026

# DREAMX: VIDEO DIFFUSION MODEL IS A UNIVERSAL LATENT ACTION LEARNER

**Jisoo Kim**<sup>1∗</sup> **Chaehyun Kim**<sup>1∗</sup> **Junsu Kim**<sup>1</sup> **Jiho Park**<sup>1</sup> **Seongchan Kim**<sup>1</sup> **Eunbeen Hong**<sup>1</sup> **Soowon Son**<sup>1</sup> **Fu-En Yang**<sup>2</sup> **Seungryong Kim**<sup>1 1</sup>KAIST <sup>2</sup>NVIDIA

**Project page:** [https://cvlab-kaist.github.io/DreamX](https://cvlab-kaist.github.io/DreamX)

![Image block](doc:f9b8e34/tier:basic/page:1/block:5)

Figure 1: DreamX learns a shared action interface from egocentric human videos and robot videos. Compact, embodiment-agnostic latent actions extracted from a pretrained video diffusion model capture motion consistently across changes in embodiment and scene appearance. The same representation enables transferring motion from a source video to a target video, decoding human motion into executable robot actions without paired human–robot training data, and training a lightweight latent policy that jointly predicts latent actions and robot actions.

## ABSTRACT

Learning a universal action representation from video requires modeling dynamics that are consistent across different executions of the same action yet invariant to changes in embodiment and appearance. Existing latent action models infer actions from visual transitions, but their representations often entangle the underlying action with irrelevant visual variations, limiting transfer across embodiments. We introduce **DreamX**, which repurposes a pretrained video diffusion model as a universal latent action learner. Our approach extracts compact latent actions from the video model’s rich spatiotemporal features, compressing each latent video frame by 300× (1,200 → 4 tokens). In addition, by leveraging the video model’s generative prior, these latent actions capture action-relevant dynamics disentangled from visual appearance. The resulting latent space serves as a shared action interface across embodiments (human and robot), environments, and action parameterizations. On LARY-Bench, DreamX achieves state-of-the-art performance among latent action models. Moreover, actions encoded in one setting remain effective in another: the latent space supports any-to-any video transfer and zero-shot action transfer across diverse embodiments and environments. Furthermore, this shared interface also enables policy learning in a compact latent action space, reducing the computational cost of video-diffusion-based policies.

1

<!-- page 2 of 25 -->

## 1 INTRODUCTION

Generalist robot policies, including vision-language-action (VLA) models (Kim et al., 2024; Bjorck et al., 2025; Intelligence et al., 2025) and world action models (WAMs) (Kim et al., 2026b; Ye et al., 2026), benefit from diverse robot demonstrations, but collecting such data at scale remains costly. Human egocentric videos offer a promising alternative as a scalable source of diverse manipulation behaviors, and recent works leverage these videos for policy learning (Li et al., 2025; Kareer et al., 2025). However, action supervision remains tied to the embodiment that produces it. Human videos lack robot action labels, and even robot demonstrations use different control spaces across embodiments. The same behavior can therefore correspond to different low-level actions, making action labels difficult to share directly across embodiments. Overcoming this mismatch calls for a common action representation that preserves shared motion and effects across embodiments.

Latent action models (LAMs) offer a promising route toward learning such a representation from video. Although humans and robots have different bodies and control spaces, their actions can all be observed as changes in a visual scene. LAMs typically use dedicated encoders to compress these changes into compact representations, called latent actions (Ye et al., 2025; Chen et al., 2025c; Bu et al., 2025b), where these encoders are trained using a decoder that reconstructs future frames solely from the current frame and latent actions. However, reconstruction accuracy alone does not ensure that latent actions isolate the underlying action. The learned representations can also encode domain-specific visual information, such as viewpoint and scene appearance, whenever it helps reconstruction (Nikulin et al., 2025). Capturing action-relevant structure shared across embodiments while separating it from domain-specific visual factors therefore remains a challenge.

Pretrained video diffusion models provide a promising foundation for addressing this challenge. First, through large-scale video generation, they learn rich spatiotemporal representations that capture motion, correspondence, geometry, and semantic structure (Nam et al., 2026; Wiedemer et al., 2025; Son et al., 2025). Such representations encode how scenes, objects, and agents evolve over time, making them a natural source of action-relevant dynamics. Second, these models possess a generative prior that enables future-frame synthesis conditioned on an initial frame and an additional signal. With scene appearance supplied by the initial frame, additional conditioning can guide how the scene evolves. Together, these priors motivate extracting latent actions directly from pretrained spatiotemporal representations and supervising their learning with a pretrained generative decoder.

We introduce **DreamX**, a framework for learning universal latent actions across embodiments using a pretrained video diffusion model. We pass learnable query tokens alongside the video tokens through the model’s spatiotemporal attention, allowing the queries to gather action-relevant information across frames. The queries condense frame-to-frame changes represented by **1,200** frame tokens into just 4 latent-action tokens, yielding a 300× reduction in token count. These latent actions replace text conditioning in the same model to reconstruct future frames from the initial frame. With scene appearance supplied by the initial frame and visual detail by the pretrained generative prior, the reconstruction objective encourages latent actions to capture the motion needed to explain the observed future.

Trained on heterogeneous human and robot videos, DreamX achieves state-of-the-art performance among latent action models on LARYBench (Nie et al., 2026) and enables any-to-any video transfer and human-to-robot action transfer without paired training data. As a downstream application, we also show that the learned latent actions can serve as prediction targets for an observationconditioned robot policy, without requiring future-video generation during execution. Together, these results suggest that DreamX learns a shared, transferable latent action space that supports motion transfer across embodiments and can also serve as a basis for downstream policy learning.

In summary, our contributions are as follows:

• We introduce DreamX, a framework that uses the spatiotemporal and generative priors of a pretrained video diffusion model to learn universal latent actions. The resulting representations achieve state-of-the-art performance among latent action models on LARYBench.

• We demonstrate that the learned latent space supports any-to-any video transfer and humanto-robot action transfer without paired source–target training data. We further show the use of compact latent-action codes as prediction targets for downstream policy learning.

2

<!-- page 3 of 25 -->

## 2 RELATED WORKS

**Latent action models.** Latent action models (LAMs) learn action representations from visual transitions, enabling action-unlabeled videos to supervise downstream control. LAPO (Schmidt & Jiang, 2024) and Genie (Bruce et al., 2024) established the paradigm, recovering discrete latent actions through an inverse–forward dynamics bottleneck over adjacent frames. LAPA (Ye et al., 2025) further scales this recipe for vision-language-action pretraining, and Moto (Chen et al., 2025c) tokenizes visual motion for manipulation. Later work incorporates additional cues: UniVLA (Bu et al., 2025b) grounds latent actions with semantic visual features, villa-X (Chen et al., 2025b) integrates language and robot-state dynamics, and CoMo (Yang et al., 2026) and DiLA (Zhang et al., 2026) separate motion from visual content. Across these variants, the action space is learned from scratch on a local visual transition, often a single frame pair. Such supervision offers no view of how motion persists beyond one step, so the encoder entangles action with appearance and yields representations tied to their training embodiments.

**Emergent capabilities of video generative models.** Video generative models acquire spatiotemporal structure by modeling how scenes evolve over time. Their internal representations support correspondence, point tracking, optical flow, and geometric understanding without task-specific supervision (Nam et al., 2026; Velez et al., 2025; Vedder et al., 2024; Huang et al., 2025), and ´ text-conditioned models further exhibit semantic grounding over objects, behaviors, and interactions (Wiedemer et al., 2025). Rather than encoding physical laws explicitly, these representations capture motion, geometry, and semantics jointly: which entities are present, where they are, and how they move over time. This is precisely the composition a latent action must express, which is a structured change of a specific entity in a specific place, and it is acquired from web-scale video rather than from a single frame pair.

**Video generative priors for latent action learning.** Recent work connects these two lines. VPP (Hu et al., 2024) conditions a manipulation policy on the predictive visual representations inside a fine-tuned video diffusion model, but still learns the policy from action labels. AdaWorld (Gao et al., 2025) learns latent actions for adaptable video prediction, and VideoWorld2 (Ren et al., 2026b) pairs a pretrained video diffusion model for generation with a separate latent dynamics model. In both, the video model acts as a generative prior supplying pixels or rollouts, while the action representation is produced by an external module trained from scratch on frame-level transitions. Thus, the latent action encoder does not directly read the video diffusion backbone’s internal representations. We instead read latent actions directly from the model’s rich spatiotemporal representation, making maximal use of video pretraining and requiring no per-embodiment action encoder.

## 3 METHOD

Given a video and instruction, DreamX uses learnable queries to extract compact latent actions from a pretrained video diffusion backbone’s spatiotemporal features. Future-frame reconstruction conditioned on these latent actions and the initial frame provides generative supervision (Sec. 3.2). The learned representation enables video and action transfer with a target initial frame or robot state (Sec. 3.3). For control, the DreamX policy predicts compact codes and low-level actions without generating future video (Sec. 3.4).

## 3.1 PRELIMINARIES: VIDEO DIFFUSION MODELS

We build on Cosmos-Predict2.5 (Ali et al., 2025), a pretrained video diffusion transformer. A causal video autoencoder compresses a clip into N latent frames, $\mathbf { V }   =   ( V _ { 0 } , \ldots , V _ { N - 1 } )$ , where $V _ { 0 }$ encodes only the initial frame. These frames are patchified into spatial tokens and jointly modeled through spatiotemporal attention. Through video pretraining, the model learns spatiotemporal priors that capture spatial context and temporal relationships across the clip. Conditioned on an initial frame and an additional signal such as text, collectively denoted by $c _ { \flat }$ the model learns future-video generation through flow matching. Given clean video latents x and noise $\boldsymbol { \epsilon }   \sim   \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , we form $x _ { \tau } = ( 1 - \tau ) x + \tau \epsilon$ , with $\tau \in [ 0 , 1 ]$ , and minimize

$$
\mathcal { L } _ { \operatorname { v i d e o } } = \mathbb { E } _ { x , c , \epsilon , \tau } \left[ \left\| v _ { \theta } ( x _ { \tau } , \tau ; c ) - ( \epsilon - x ) \right\| _ { 2 } ^ { 2 } \right] ,\tag{1}
$$

where $v _ { \theta }$ predicts the flow velocity.

3

<!-- page 4 of 25 -->

![Image block](doc:f9b8e34/tier:basic/page:4/block:1)

Figure 2: DreamX architecture and attention masks. (a) Learnable queries read out the pretrained video backbone’s spatiotemporal representations to encode latent actions. A decoder initialized from the same pretrained backbone reconstructs future video from the latent actions and the initial frame, with auxiliary action supervision when labels are available. (b) Encoder and decoder attention masks control information flow among video, query, and action tokens. In the decoder, video and action tokens read the latent-action queries but do not attend to one another.

By optimizing this objective over large-scale video data, the model acquires a generative prior over scene appearance and plausible dynamics. When the initial frame supplies the scene’s appearance, additional conditioning steers how the scene evolves. We exploit this property by using latent actions to steer future-frame reconstruction, whose objective supervises what the latent actions encode.

## 3.2 LEARNING UNIVERSAL LATENT ACTIONS FROM VIDEO DIFFUSION PRIORS

DreamX adapts a pretrained video diffusion backbone for two complementary roles: encoding observed dynamics into latent actions and supervising them through future-frame reconstruction to encourage separation of motion from visual appearance. Building on Sec. 3.1, we introduce learnable queries that read out the video representations to summarize observed dynamics as latent actions, while the decoder reconstructs the future from these latent actions and the initial frame. Leveraging a video diffusion backbone for both the encoder and decoder allows its generative prior to account for visual appearance, encouraging the latent action tokens to focus on motion.

**Latent Action Encoding from Spatiotemporal Dynamics.** DreamX uses pretrained video model itself to encode latent actions. As shown in Figure $2 ( \mathrm { a } )$ , we append learnable query tokens q to the video token sequence and process them jointly through the backbone’s spatiotemporal attention. We assign a group of K query tokens $\mathbf { q } _ { g }$ to each latent frame $V _ { g } .$ . Once contextualized by the video, their output features form the latent-action group $Q _ { g }$ . All N groups are extracted in a single forward,

$$
Q = \mathcal { E } _ { \phi } ( \mathbf { q } ; \mathbf { V } , \ell ) ,\tag{2}
$$

where $\mathcal { E } _ { \phi } ( \cdot )$ is the encoder and ℓ is the language instruction supplied through the backbone’s original cross-attention. As shown in Figure 2(b), query-to-video attention is frame-causal, so $Q _ { g }$ directly attends to $V _ { 0 : g } ,$ while query-to-query attention is unrestricted. This lets the queries encode temporally grounded motion from the pretrained dynamics, with query-layout settings in Appendix A.2.

**Generative Decoding for Action-Centric Latent Learning.** Encoding from video alone may retain scene- and embodiment-specific visual cues. We therefore use the same pretrained video backbone as the encoder to supervise latent actions through future-frame reconstruction, conditioned on $V _ { 0 }$ and $Q _ { 0 : N - 1 }$ without text. As $V _ { 0 }$ provides scene appearance and the pretrained generative prior captures plausible evolution, Q is encouraged to encode the motion needed to explain the future, rather than visual cues already available to the decoder.

As shown in Figure 2(a), the decoder $\mathcal { D } _ { \theta } ( \cdot )$ generates future frames conditioned on $Q _ { \mathrm { 0 : } N - 1 }$ , concatenated with the video tokens along the temporal sequence. When action labels are available, it also takes the proprioceptive state $p _ { 0 }$ and decodes action chunks as an auxiliary task:

$$
\hat { V } , \hat { A } = \mathcal { D } _ { \theta } ( Q _ { 0 : N - 1 } , V _ { 0 } , p _ { 0 } ) .\tag{3}
$$

Under the decoder attention mask in Figure 2(b), latent-action queries serve solely as conditioning signals and do not attend to any tokens. The video and action streams attend to these queries without attending to each other. Specifically, video frame $V _ { t }$ attends to the query prefix $Q _ { 0 : t }$ , while action chunk $A _ { m }$ attends to $Q _ { 0 : m }$

4

<!-- page 5 of 25 -->

![Image block](doc:f9b8e34/tier:basic/page:5/block:1)

Figure 3: Downstream applications of DreamX. (a) Motion transfer. Source latent actions condition the DreamX decoder to generate video from a target initial frame or low-level actions from a target robot state, without paired source–target training data. (b) DreamX policy. A variational bottleneck compresses latent actions into compact codes while preserving the token layout. Conditioned on the current observation and instruction, the policy generates these codes alongside low-level actions through a separate DiT action head.

**Training Objective.** We train both the encoder and the decoder end-to-end with the standard flowmatching objective applied to video and action generation as Eq. 4, with $\lambda   =   0$ for video clips without action labels:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { v i d e o } } + \lambda   \mathcal { L } _ { \mathrm { a c t i o n } } .\tag{4}
$$

## 3.3 MOTION TRANSFER VIA LATENT ACTIONS

Building upon the rich generative prior from the video diffusion backbone, DreamX learns latent actions that capture motion transferable across embodiments, even though cross-embodiment transfer is not an explicit training objective. This capability emerges from the pretraining described in the previous section and requires no additional training. The auxiliary robot action branch further encourages the latent actions to encode action-relevant cues and further enables motion extracted from a source video to be decoded into continuous actions executable by a target embodiment.

Given a source video, we extract its latent actions using the frozen DreamX encoder, $Q ^ { \mathrm { s r c } } ~ =$ $\mathcal { E } _ { \phi } ( \mathbf { q } ; \mathbf { V } ^ { \mathrm { s r c } } , \ell ^ { \mathrm { s r c } } )$ , which captures the motion to be transferred. As shown in Figure 3(a), we reuse the same $Q ^ { \mathrm { s r c } }$ in two complementary ways. For video transfer, we combine it with a target initial frame $V _ { 0 } ^ { \mathrm { t g i } }$ and use the frozen DreamX decoder $\mathcal { D } _ { \theta }$ to generate a target video that reproduces the source motion under the target scene and embodiment. For action transfer, we instead combine $Q ^ { \mathrm { s r c } }$ with the target robot state $\widetilde { p _ { 0 } ^ { \mathrm { t g t } } }$ and decode the same motion into executable low-level actions,

$$
\hat { V } ^ { \mathrm { t g t } } = \mathcal { D } _ { \theta } \big ( Q ^ { \mathrm { s r c } } , V _ { 0 } ^ { \mathrm { t g t } } \big ) , \qquad \hat { A } ^ { \mathrm { t g t } } = \mathcal { D } _ { \theta } \big ( Q ^ { \mathrm { s r c } } , p _ { 0 } ^ { \mathrm { t g t } } \big ) .\tag{5}
$$

Thus, the shared latent action realizes the same underlying motion either as a generated video in a new domain or as executable actions on a target robot, without paired source–target datasets.

## 3.4 POLICY LEARNING WITH COMPACT LATENT ACTIONS

As a downstream application, we examine whether the latent actions learned by DreamX can serve as prediction targets for an observation-conditioned robot policy. Whereas motion transfer uses latent actions extracted from a source demonstration, the policy predicts them from the current observation and instruction. To this end, we repurpose the same pretrained video backbone used by DreamX to predict latent actions in place of future video (Kim et al., 2026a). Although Q is compact in token count, each token retains the full backbone width, making direct diffusion high-dimensional (Chen et al., 2025a). We therefore compress Q into a low-dimensional code Z first, which serves as the policy’s prediction target.

**Latent Action Compression.** Following (Gao et al., 2026), we learn a variational bottleneck that compresses each latent-action token from backbone width D to $d \ll D$ while preserving the token layout, as shown in Figure 3(b). With the DreamX encoder frozen, we train the bottleneck using feature reconstruction and KL regularization. After training, we use its posterior mean to obtain deterministic compact codes Z for policy learning. The formulation and implementation details are provided in Appendix A.3.

5

<!-- page 6 of 25 -->

Table 1: LARYBench action regression and classification. Regression reports normalized action MSE and classification reports accuracy. Bold and underline denote the best and second-best results.

|  |  | A | ction regressi | on ↓ |  |  | Action classific | ation ↑ |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Method | CALVIN | VLABench | RoboCOIN | AgiBotWorld-Beta | Avg. | Atomic robot | Composite human | Composite robot | Avg. |
| Latent actionsfrom 2frames |  |  |  |  |  |  |  |  |  |
| LAPA | 0.96 | 0.95 | 0.96 | 1.00 | 0.97 | 22.27 | 14.61 | 23.64 | 20.17 |
| UniVLA | 0.82 | 0.74 | 0.94 | 0.97 | 0.87 | 18.62 | 19.08 | 18.56 | 18.75 |
| villa-X | 0.86 | 0.72 | 0.94 | 0.97 | 0.87 | 15.00 | 17.80 | 29.90 | 20.90 |
| LAPA-MAGVIT2 | 0.59 | 0.36 | 0.80 | 0.83 | 0.65 | 33.12 | 59.70 | 29.53 | 40.78 |
| LAPA-SigLIP2 | 0.57 | 0.30 | 0.86 | 0.88 | 0.65 | 46.83 | 54.74 | 29.44 | 43.67 |
| LAPA-DINOv2 | 0.55 | 0.26 | 0.85 | 0.86 | 0.63 | 58.04 | 55.86 | 34.19 | 49.36 |
| LAPA-DINOv3 | 0.50 | 0.25 | 0.82 | 0.84 | 0.60 | 56.27 | 64.19 | 27.04 | 49.17 |
| Latent actionsfrom video |  |  |  |  |  |  |  |  |  |
| MoTo (Chen et al., 2025c) | 0.43 | 0.13 | 0.54 | 0.68 | 0.44 | 41.57 | 55.11 | 56.34 | 51.01 |
| Adaworld (Gao et al., 2025) | 0.64 | 0.26 | 0.94 | 0.94 | 0.69 | 31.58 | 49.8 | 51.48 | 44.29 |
| VideoWorld2 (Ren et al., 2026a) | 0.72 | 0.41 | 0.82 | 0.85 | 0.70 | 14.50 | 30.24 | 37.36 | 27.37 |
| DiLA (Zhang et al., 2026) | 0.43 | 0.09 | 0.51 | 0.52 | 0.38 | 58.51 | 65.47 | 62.76 | 62.25 |
| DreamX (Ours) | 0.15 | 0.05 | 0.26 | 0.22 | 0.17 | 68.03 | 72.72 | 68.26 | 69.67 |
| DreamX (Ours) + FAE-32 | 0.23 | 0.07 | 0.31 | 0.26 | 0.22 | 65.90 | 69.49 | 66.06 | 67.15 |

**Policy Learning on Compact Latent Action Space.** With the learned bottleneck frozen, the compact codes serve as prediction targets for the DreamX policy π(·). As shown in Figure 3(b), the policy generates these codes from the current frame $V _ { 0 }$ and instruction ℓ, while a separate DiT action head predicts low-level actions conditioned on the generator’s latent-action features, the same observation and instruction, and the state $p _ { 0 }$

$$
\pi ( Z , A \mid V _ { 0 } , \ell , p _ { 0 } ) .\tag{6}
$$

Action tokens attend to latent-action features, but not vice versa. This separation allows human videos to supervise latent-code prediction without robot action labels, while robot demonstrations additionally supervise low-level action prediction. We train the policy with

$$
\mathcal { L } _ { \mathrm { p o l i c y } } = \mathcal { L } _ { \mathrm { l a t e n t } } + \lambda   \mathcal { L } _ { \mathrm { a c t i o n } } ,\tag{7}
$$

where both terms follow Eq. 1, applied to compact codes Z and low-level actions A respectively, with λ = 0 for human videos without action labels. Further details are provided in Appendix A.4.

## 4 EXPERIMENTAL RESULTS

We evaluate whether DreamX learns universal latent actions by examining how well they capture motion (Sec. 4.1) and whether they can transfer across scenes and embodiments (Secs. 4.2–4.3). We further assess their utility for learning effective and efficient policies from human and robot demonstrations (Sec. 4.4), followed by ablations and analysis of the learned representation (Secs. 4.5–4.6).

**Implementation details.** We instantiate DreamX with Cosmos-Predict2.5 2B and pretrain on robot demonstrations from MolmoAct2-BimanualYAM (Fang et al., 2026), Bridge (Walke et al., 2023), DROID (Khazatsky et al., 2024), and AgiBot (Bu et al., 2025a), together with human videos from EgoDex (Hoque et al., 2026) and EgoVerse (Punamiya et al., 2026). Videos are processed at 480 × 640 resolution with variable-length clips of up to 29 frames. We assign four query tokens to each VAE latent frame. We adapt the encoder and decoder using separate rank-16 LoRA modules for 78,000 training steps. Further training details are provided in Appendix A.2.

## 4.1 REPRESENTATION QUALITY

We examine how well DreamX’s latent actions capture motion on LARYBench (Nie et al., 2026). Regression asks whether the representation retains enough fine-grained motion to recover endeffector trajectories, and classification whether it carries enough semantic information to distinguish different motions. Table 1 compares DreamX with latent action models that encode either frame pairs or video clips. DreamX achieves the best performance across all benchmarks. Compared with the strongest baseline, DiLA, DreamX reduces regression MSE from 0.38 to 0.17 and increases classification accuracy from 62.25% to 69.67%. These results support leveraging video models as a source of action-relevant representations: reading latent actions directly from their pretrained priors yields representations that capture spatial, temporal, and semantic structure across embodiments. FAE-32, our 32-channel compressed representation (Sec. 3.4), also outperforms all baselines, indicating that even the compact codes retain substantial motion and semantic information. Further evaluation details are provided in Appendix B.4.

6

<!-- page 7 of 25 -->

![Image block](doc:f9b8e34/tier:basic/page:7/block:1)

Figure 4: Qualitative Any-to-Any video transfer. Each example shows a source motion and the corresponding transferred video generated from a target-domain initial observation. These results demonstrate that the learned latent action transfers motion across substantial changes in embodiment and scene appearance. Blue dots indicate the end-effector trajectory throughout the video frames.

## 4.2 VIDEO TRANSFER

We evaluate whether DreamX learns universal latent actions by transferring source motion across scenes and embodiments while preserving the appearance specified by a target initial frame. We examine any-to-any video transfer and quantitatively evaluate zero-shot human-to-robot transfer. Zeroshot evaluation details and additional transfer examples are provided in Appendices B.5 and D.1.

**Any-to-any video transfer.** Figure 4 shows motion transfer across environments for the same robot, between different robot embodiments, and from humans to robots, with a DAVIS example extending beyond manipulation. DreamX reproduces the source motion while preserving the target scene and embodiment. These results support the universality of the learned latent actions across embodiments and environments, extending to general-domain videos.

**Zero-shot human-to-robot transfer.** We evaluate zero-shot human-to-robot transfer on HumanAndRobot (Xie et al., 2026), comparing with AdaWorld (Gao et al., 2025) and VideoWorld 2 (Ren et al., 2026b), both of which use diffusionbased video decoders. Paired robot recordings serve only as evaluation references; **no paired human–robot data is used for training**. DreamX outperforms both baselines on every metric in Table 2 and follows the source human motion while preserving the target robot and scene (Figure 5). In contrast,

Table 2: Zero-shot human-to-robot video transfer on H&R. DreamX outperforms baselines across all metrics without paired human–robot training data.

<table><tr><td>Method</td><td colspan="4">FVD↓ LPIPS↓ PSNR↑ SSIM↑</td></tr><tr><td>AdaWorld</td><td>170.19</td><td>0.344</td><td>22.95</td><td>0.846</td></tr><tr><td>VideoWorld2 241.92</td><td></td><td>0.299</td><td>22.53</td><td>0.881</td></tr><tr><td>Ours</td><td>57.44</td><td>0.179</td><td>26.52</td><td>0.917</td></tr></table>

7

<!-- page 8 of 25 -->

![Image block](doc:f9b8e34/tier:basic/page:8/block:1)

Grab red cube and pull to plate

Figure 5: Zero-shot human-to-robot video transfer on the H&R benchmark. DreamX preserves the source motion and target robot appearance, while the baselines exhibit visual degradation or human-hand leakage.

AdaWorld exhibits visual degradation in this out-of-distribution setting, while VideoWorld 2 occasionally introduces a human hand despite its appearance-disentangling design. These results suggest that DreamX better separates transferable motion from appearance, enabling more consistent human-to-robot transfer.

## 4.3 ACTION TRANSFER

We evaluate action transfer to examine whether our latent actions form a universal representation space that connects similar motions across embodiments. Following Sec. 3.3, we extract latent actions from a human video and decode them into low-level actions for the target robot. Execution is open-loop without visual observations: only the latent actions specify the motion, while proprioception provides the robot’s initial state. Notably, this transfer is learned **without paired human–robot training data**, so the two embodiments are connected through the shared latent action space. Figure 6 shows that the robot reproduces the stroke structure of the letter “N,” preserving the demonstrated motion beyond a coarse task outcome. In single-arm pushing, it follows the forward-thenleft sequence, retaining the direction change and temporal order. Bimanual pushing further shows corresponding movements of both arms, suggesting that the shared latent representation retains armspecific motion and transfers it to the corresponding robot arms. Evaluation protocols and additional examples are provided in Appendices B.2 and D.2, respectively.

## 4.4 POLICY EVALUATION

We conduct a real-world drawer-closing study to examine whether DreamX’s learned latent actions can support downstream policy learning. Given the instruction close the drawer, the robot must close the upper drawer, the lower drawer, or both sequentially. We compare DreamX with Mimic-Video (Pai et al., 2025), both trained using human videos and robot demonstrations. During rollout, Mimic-Video generates future video, while DreamX generates compact latent actions alongside low-level actions.

Table 3 reports an average subgoal success rate of 96.7% for DreamX, averaging reaching and pushing completion

Table 3: Real-world drawer-closing SR (%) and inference latency. DreamX achieves higher SR than Mimic-Video with 19.1× faster inference, predicting compact latent actions instead of future video.

<table><tr><td colspan="5"></td></tr><tr><td>Method</td><td>Upper Lower</td><td></td><td>Both</td><td>Latency↓ Avg.↑ (ms)</td></tr><tr><td>Mimic-Video</td><td>50.0</td><td>50.0 50.0</td><td>50.0</td><td>28,226</td></tr><tr><td>DreamX</td><td>90.0</td><td>100.0 100.0</td><td>96.7</td><td>1,474</td></tr></table>

8

<!-- page 9 of 25 -->

![Image block](doc:f9b8e34/tier:basic/page:9/block:1)

Figure 6: Qualitative human-to-robot action transfer. Given a human demonstration, DreamX extracts a latent action and decodes it into low-level robot actions without paired human–robot training data. We show diverse transferred behaviors spanning writing, single-arm pushing, and bimanual manipulation. Across substantial embodiment differences, the robot reproduces the motion structure of the human demonstration.

within each configuration and weighting the three configurations equally. Pushing completion rates are 80%, 100%, and 100% for the upper, lower, and both-open configurations, respectively. We also report an inference latency of 1.47 s per policy call for DreamX. These results demonstrate the feasibility of using the learned latent actions as prediction targets for a robot policy in the evaluated drawer-closing setting. See Appendix B.3, C.1, and D.3 for additional evaluation details and results.

## 4.5 ABLATION STUDY

We examine how auxiliary action supervision affects DreamX’s latent actions, using the same subsets of MolmoAct2, EgoVerse, and EgoDex to limit computational cost. Table 4 shows that action supervision slightly improves regression MSE from 0.24 to 0.23, while classification accuracy slightly decreases from 65.3% to 64.8%. The comparable performance suggests that DreamX captures motion and action semantics from video alone, without relying on action labels.

## 4.6 ANALYSIS

**Sensitivity to latent-action perturbations.** We examine whether latent actions guide the motion reconstructed by the decoder. Keeping the initial frame and sampling noise fixed, we perturb the latent actions or replace them with zeros. In Figure 7, the original latent actions reconstruct the grasping and lifting motion, whereas perturbation disrupts object manipulation and zeroing largely suppresses motion. The surrounding scene remains largely unchanged, supporting the role of latent actions in specifying motion beyond appearance. Quantitative results are provided in Appendix C.3.

Table 4: Encoder supervision ablation. LARYBench regression MSE and classification accuracy.

|  | Reg. avg. | Cls. avg. |
| --- | --- | --- |
| Supervision | ↓ | ↑ |
| RGB only | 0.24 | 65.3 |
| RGB + action | 0.23 | 64.8 |

![Image block](doc:f9b8e34/tier:basic/page:9/block:10)

Figure 7: Sensitivity analysis. Perturbing latent actions alters motion while preserving scene appearance.

9

<!-- page 10 of 25 -->

![Image block](doc:f9b8e34/tier:basic/page:10/block:1)

Figure 8: Query-to-video attention. We visualize attention maps from latent queries to video tokens during encoding. Across human and robot demonstrations, latent queries attend to dynamic regions, such as moving hands and grippers, capturing the underlying actions in each video.

**What latent actions attend to.** To examine where DreamX reads motion information, Figure 8 visualizes query-to-video attention during latent-action encoding of human and robot demonstrations. Attention highlights the hands and grippers involved in object interactions, suggesting that the queries read action-relevant regions across embodiments. Additional visualizations are provided in Appendix D.4.

## 5 CONCLUSION

We present DreamX, a framework that leverages pretrained video diffusion priors to learn universal latent actions. Its spatiotemporal representations provide action-relevant dynamics, while its generative prior supervises latent-action learning through future-frame reconstruction. DreamX achieves state-of-the-art performance among latent action models on LARYBench and enables any-to-any video transfer and human-to-robot action transfer without paired training data. Predicting these compact latent actions also supports effective and efficient policy learning from human and robot demonstrations. Together, these results highlight pretrained video models as a foundation for learning reusable action representations across embodiments.

10

<!-- page 11 of 25 -->

## AI USE STATEMENT

We used generative AI tools to assist with manuscript drafting and editing for clarity and readability, code editing during implementation, and discussions during the research process. Research ideas, methodological choices, experimental designs, and interpretation of results were formulated, evaluated, and ultimately determined by the authors. All AI-assisted text and code were reviewed by the authors, and code changes were tested for correctness. We take responsibility for the final content of this work, including text, claims, code, and artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

All datasets used in this work are publicly available, including the human egocentric and robot demonstration datasets listed in Appendix A.1. Human videos may contain identifiable people or surroundings, so any redistributed examples should follow the source datasets’ usage terms and privacy requirements. Video-transfer outputs are synthetic and should be identified as such. Cross-embodiment transfer and robot policies may fail outside the evaluated settings and require safety validation before deployment.

## REPRODUCIBILITY STATEMENT

Section 3 describes the model, training objectives, transfer procedure, and policy. Appendix A documents the data sources and implementation details, while Appendix B specifies the benchmark, transfer, and real-world policy evaluation protocols. Supplementary videos illustrate video-transfer and action-transfer results; policy rollouts are provided in Appendix D.3.

## REFERENCES

Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, et al. World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062, 2025.

Johan Bjorck, Fernando Castaneda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, ˜ Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first international conference on machine learning, 2024.

Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025a.

Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. Univla: Learning to act anywhere with task-centric latent actions. arXiv preprint arXiv:2505.06111, 2025b.

Junyu Chen, Dongyun Zou, Wenkun He, Junsong Chen, Enze Xie, Song Han, and Han Cai. Dc-ae 1.5: Accelerating diffusion model convergence with structured latent space. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1–10. IEEE, 2025a.

Xiaoyu Chen, Hangxing Wei, Pushi Zhang, Chuheng Zhang, Kaixin Wang, Yanjiang Guo, Rushuai Yang, Yucen Wang, Xinquan Xiao, Li Zhao, Jianyu Chen, and Jiang Bian. villa-x: Enhancing latent action modeling in vision-language-action models, 2025b. URL [https://arxiv.org/abs/2507.23682](https://arxiv.org/abs/2507.23682).

Yi Chen, Yuying Ge, Weiliang Tang, Yizhuo Li, Yixiao Ge, Mingyu Ding, Ying Shan, and Xihui Liu. Moto: Latent motion token as the bridging language for learning robot manipulation from videos. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 19752–19763. IEEE, 2025c.

11

<!-- page 12 of 25 -->

Haoquan Fang, Jiafei Duan, Donovan Clay, Sam Wang, Shuo Liu, Weikai Huang, Xiang Fan, Wei-Chuan Tsai, Shirui Chen, Yi Ru Wang, et al. Molmoact2: Action reasoning models for real-world deployment. arXiv preprint arXiv:2605.02881, 2026.

Shenyuan Gao, Siyuan Zhou, Yilun Du, Jun Zhang, and Chuang Gan. Adaworld: Learning adaptable world models with latent actions. arXiv preprint arXiv:2503.18938, 2025.

Yuan Gao, Chen Chen, and Jiatao Gu. One layer is enough: Adapting pretrained visual encoders for image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4688–4697, 2026.

Ryan Hoque, Peide Huang, David Yoon, Jian Zhang, et al. Egodex: Learning dexterous manipulation from large-scale egocentric video. In International Conference on Learning Representations, volume 2026, pp. 4218–4237, 2026.

Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video prediction policy: A generalist robot policy with predictive visual representations. arXiv preprint arXiv:2412.14803, 2024.

Zixuan Huang, Xiang Li, Zhaoyang Lv, and James M. Rehg. How much 3d do video foundation models encode?, 2025. URL [https://arxiv.org/abs/2512.19949](https://arxiv.org/abs/2512.19949).

Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π0. 5: a vision-languageaction model with open-world generalization, 2025. URL https://arxiv. org/abs/2504.16054, 1(2): 3, 2025.

Simar Kareer, Dhruv Patel, Ryan Punamiya, Pranay Mathur, Shuo Cheng, Chen Wang, Judy Hoffman, and Danfei Xu. Egomimic: Scaling imitation learning via egocentric video. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 13226–13233. IEEE, 2025.

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

Dongwon Kim, Gawon Seo, Jinsung Lee, Minsu Cho, and Suha Kwak. Planning in 8 tokens: A compact discrete tokenizer for latent world model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026a.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, et al. Cosmos policy: Fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163, 2026b.

Guangrun Li, Yaoxu Lyu, Zhuoyang Liu, Chengkai Hou, Jieyu Zhang, and Shanghang Zhang. H2r: A human-to-robot data augmentation for robot pre-training from videos. arXiv preprint arXiv:2505.11920, 2025.

Jisu Nam, Soowon Son, Dahyun Chung, Jiyoung Kim, Siyoon Jin, Junhwa Hur, and Seungryong Kim. Emergent temporal correspondences from video diffusion transformers. Advances in Neural Information Processing Systems, 38:123683–123727, 2026.

Dujun Nie, Fengjiao Chen, Qi Lv, Jun Kuang, Xiaoyu Li, Xuezhi Cao, and Xunliang Cai. Lary: A latent action representation yielding benchmark for generalizable vision-to-action alignment. arXiv preprint arXiv:2604.11689, 2026.

Alexander Nikulin, Ilya Zisman, Denis Tarasov, Nikita Lyubaykin, Andrei Polubarov, Igor Kiselev, and Vladislav Kurenkov. Latent action learning requires supervision in the presence of distractors. arXiv preprint arXiv:2502.00379, 2025.

12

<!-- page 13 of 25 -->

Jonas Pai, Liam Achenbach, Victoriano Montesinos, Benedek Forrai, Oier Mees, and Elvis Nava. mimic-video: Video-action models for generalizable robot control beyond vlas. arXiv preprint arXiv:2512.15692, 2025.

Ryan Punamiya, Simar Kareer, Zeyi Liu, Josh Citron, Ri-Zhao Qiu, Xiongyi Cai, Alexey Gavryushin, Jiaqi Chen, Davide Liconti, Lawrence Y Zhu, et al. Egoverse: An egocentric human dataset for robot learning from around the world. arXiv preprint arXiv:2604.07607, 2026.

Zhongwei Ren, Yunchao Wei, Xiao Yu, Guixun Luo, Yao Zhao, Bingyi Kang, Jiashi Feng, and Xiaojie Jin. Videoworld 2: Learning transferable knowledge from real-world videos, 2026a. URL [https://arxiv.org/abs/2602.10102](https://arxiv.org/abs/2602.10102).

Zhongwei Ren, Yunchao Wei, Xiao Yu, Guixun Luo, Yao Zhao, Bingyi Kang, Jiashi Feng, and Xiaojie Jin. Videoworld 2: Learning transferable knowledge from real-world videos. arXiv preprint arXiv:2602.10102, 2026b.

Dominik Schmidt and Minqi Jiang. Learning to act without actions. In International Conference on Learning Representations, volume 2024, pp. 9379–9395, 2024.

Soowon Son, Honggyu An, Jisu Nam, Hyunah Ko, Chaehyun Kim, Dahyun Chung, Siyoon Jin, Jung Yi, Junhwa Hur, and Seungryong Kim. Probing and leveraging video diffusion transformer features for robust point tracking. arXiv preprint arXiv:2512.20606, 2025.

Kyle Vedder, Neehar Peri, Nathaniel Chodosh, Ishan Khatri, Eric Eaton, Dinesh Jayaraman, Yang Liu, Deva Ramanan, and James Hays. Zeroflow: Scalable scene flow via distillation, 2024. URL [https://arxiv.org/abs/2305.10424](https://arxiv.org/abs/2305.10424).

Pedro Velez, Luisa F. Polan ´ ´ıa, Yi Yang, Chuhan Zhang, Rishabh Kabra, Anurag Arnab, and Mehdi S. M. Sajjadi. From image to video: An empirical study of diffusion representations, 2025. URL [https://arxiv.org/abs/2502.07001](https://arxiv.org/abs/2502.07001).

Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. Bridgedata v2: A dataset for robot learning at scale. In Conference on robot learning, pp. 1723–1736. PMLR, 2023.

Thaddaus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, ¨ Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners, 2025. URL [https://arxiv.org/abs/2509.20328](https://arxiv.org/abs/2509.20328).

Sicheng Xie, Haidong Cao, Zejia Weng, Zhen Xing, Haoran Chen, Shiwei Shen, Jiaqi Leng, Zuxuan Wu, and Yu-Gang Jiang. Human2robot: Learning robot actions from paired human-robot videos. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 11078–11086, 2026.

Jiange Yang, Yansong Shi, Haoyi Zhu, Mingyu Liu, Kaijing Ma, Yating Wang, Gangshan Wu, Tong He, and Limin Wang. Como: Learning continuous latent motion from internet videos for scalable robot learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 42352–42363, 2026.

Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, and Minjoon Seo. Latent action pretraining from videos, 2025. URL [https://arxiv.org/abs/2410.11758](https://arxiv.org/abs/2410.11758).

Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026.

Tianqiu Zhang, Muyang Lyu, Yufan Zhang, Fang Fang, and Si Wu. Dila: Disentangled latent action world models. arXiv preprint arXiv:2605.15725, 2026.

13

<!-- page 14 of 25 -->

This appendix provides implementation details, evaluation protocols, additional results and analysis.

**A. IMPLEMENTATION DETAILS** A.1 Training Data A.2 DreamX Pretraining A.3 Latent-Action Compression A.4 Policy Training and Inference

B. EXPERIMENTAL SETUP AND EVALUATION P B.1 Real-World Robot Setup B.2 Action Transfer Details B.3 Policy Evaluation Details B.4 LARYBench Evaluation Details B.5 Zero-Shot Human-to-Robot Transfer Details

**C. ADDITIONAL QUANTITATIVE RESULTS** C.1 Policy Evaluation C.2 Ablation Study C.3 Representation Analysis

**D. ADDITIONAL QUALITATIVE RESULTS AND ANALYSIS** D.1 Any-to-Any Video Transfer D.2 Human-to-Robot Action Transfer D.3 Policy Rollouts D.4 Query-to-Video Attention

Video transfer, action transfer videos are provided in the supplementary materials.

## A IMPLEMENTATION DETAILS

## A.1 TRAINING DATA

We pretrain DreamX on 2.19k hours of human and robot video spanning six datasets (Figure 9). Human demonstrations contribute 469 hours from EgoDex (Hoque et al., 2026) and EgoVerse (Punamiya et al., 2026), complementing robot demonstrations from Bridge (Walke et al., 2023), DROID (Khazatsky et al., 2024), AgiBot (Bu et al., 2025a), and MolmoAct-BimanualYAM (Fang et al., 2026).

## A.2 DREAMX PRETRAINING

We initialize DreamX from Cosmos-Predict2.5 2B and train at 480 × 640 resolution for 78,000 steps. Training clips contain 5–29 frames in increments of four, corresponding to 2–8 VAE latent frames. We assign four query tokens to each latent frame, yielding up to 32 query tokens per clip. The encoder and decoder use separate rank-16 LoRA modules, while the pretrained backbone weights remain frozen. Learning rates are $1 0 ^ { - 4 }$ for the LoRA parameters and query embeddings and ${ \mathrm { ~ 5 ~ } } \times 1 0 ^ { - 5 }$ for the remaining trainable parameters, with 500 warm-up steps. Video and action losses receive equal weights for action-labeled clips; clips without action labels use only the video loss.

## A.3 LATENT-ACTION COMPRESSION

We train a variational bottleneck on latent actions $Q$ extracted by the frozen DreamX encoder. Following (Gao et al., 2026), the bottleneck compresses each token from backbone width D to d ≪ D while preserving the token layout.

14

<!-- page 15 of 25 -->

![Chart block](doc:f9b8e34/tier:basic/page:15/block:1)

Figure 9: Pretraining data composition. DreamX learns from six datasets spanning human and robot demonstrations. Sector areas represent video duration, with human datasets highlighted in warm colors.

**Formulation.** The encoder $f _ { \psi }$ parameterizes a factorized Gaussian posterior $q _ { \psi } ( Z \mid Q )$ through

$$
\left( \mu _ { \psi } ( Q ) , \operatorname { l o g } \sigma _ { \psi } ^ { 2 } ( Q ) \right) = f _ { \psi } ( Q ) .
$$

During bottleneck training, we sample compact codes using

$$
Z = \mu _ { \psi } ( Q ) + \sigma _ { \psi } ( Q ) \odot \epsilon , \qquad \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) ,\tag{8}
$$

and reconstruct the original latent actions as $\hat { Q } = g _ { \psi } ( Z )$

Only the bottleneck encoder $f _ { \psi }$ and decoder $g _ { \psi }$ are optimized, using

$$
\mathcal { L } _ { \operatorname { F A E } } = \left\| \hat { Q } - Q \right\| _ { 2 } ^ { 2 } + \beta \operatorname { K L } \bigl ( q _ { \psi } ( Z \mid Q ) \left\| \operatorname { \mathcal { N } } ( \mathbf { 0 } , \mathbf { I } ) \right) .\tag{9}
$$

The reconstruction term encourages the compact codes to retain the information in $Q ,$ while the KL term regularizes their posterior toward a standard Gaussian. We compute the reconstruction loss in a fixed normalized feature space.

**Architecture and Training.** The bottleneck encoder uses a single self-attention block to produce a 32-dimensional code for each 2,048-dimensional latent-action token. A six-layer transformer decoder reconstructs the original features. The token layout is preserved: for a 32-token input, the bottleneck maps a $3 2 \times 2 \bar { , } 0 4 8$ representation to $3 2 \times 3 \dot { 2 }$ compact codes.

We train the bottleneck for 50,000 steps with a learning rate of $1 0 ^ { - 4 }$ , 500 warm-up steps, and a KL weight of $\beta = 0 . 0 1$ . After training, we freeze the bottleneck and use the posterior mean,

$$
Z = \mu _ { \psi } ( Q ) ,
$$

to obtain deterministic compact codes for downstream policy learning and evaluation.

## A.4 POLICY TRAINING AND INFERENCE

To convert the pretrained video model into a latent-action generator, we train it to predict compact latent-action codes conditioned on the current observation and language instruction. We use approximately one-third of the DreamX pretraining data and optimize rank-128 LoRA modules in self-attention, cross-attention, and feed-forward layers, together with trainable input and output projections. The generator is trained for 100,000 steps on two NVIDIA H200 GPUs with a global batch size of 60, a learning rate of $1 0 ^ { - 4 }$ , and 1,000 warm-up steps. The Action DiT is not separately pre-trained; it is introduced during task-specific policy learning and trained jointly with the generator. Task-specific training settings are provided in Appendix B.3.

15

<!-- page 16 of 25 -->

## B EXPERIMENTAL SETUP AND EVALUATION PROTOCOLS

## B.1 REAL-WORLD ROBOT SETUP

We conduct both action-transfer and policy experiments on a bimanual YAM robot. The action space consists of 14- dimensional absolute commands, comprising six joint positions and one gripper command per arm. We leverage one Intel RealSense D405 camera for observation, which is mounted on top view.

![Image block](doc:f9b8e34/tier:basic/page:16/block:4)

Figure 10: BimanualYAM setup.

## B.2 ACTION TRANSFER DETAILS

We extract latent actions from human demonstrations using the frozen DreamX encoder and decode them into robot commands. The resulting trajectory is executed open-loop, without visual feedback, conditioned on the human latent actions and the robot’s initial proprioceptive state.

We evaluate three types of motion transfer. **Writing** tests whether latent actions preserve finegrained spatial trajectories and stroke order; a marker attached to the right gripper reproduces the demonstrated shape. **Single-arm pushing** tests whether the representation captures changes in motion direction and their temporal order, requiring the robot to follow a forward-then-left sequence. **Bimanual manipulation** tests whether latent actions retain arm-specific motion and transfer it to the corresponding robot arms. Additional examples are provided in Section D.2.

## B.3 POLICY EVALUATION DETAILS

**Task and observations.** We evaluate three drawer-closing configurations: only the upper drawer open, only the lower drawer open, and both drawers open, as illustrated in Figure 11. All configurations use the instruction close the drawer. When both drawers are open, the robot must close the upper drawer before the lower drawer. A fixed top-view RGB camera observes the workspace.

![Image block](doc:f9b8e34/tier:basic/page:16/block:11)

(a) Upper drawer open

![Image block](doc:f9b8e34/tier:basic/page:16/block:13)

(b) Lower drawer open

![Image block](doc:f9b8e34/tier:basic/page:16/block:15)

(c) Both drawers open

Figure 11: Real-world drawer-closing setup. Initial observations for the three task configurations, all conditioned on close the drawer. When both drawers are open, the robot must close the upper drawer before the lower drawer.

**Training.** We compare DreamX with Mimic-Video (Pai et al., 2025), using human videos and robot demonstrations for task-specific learning. DreamX starts from the pretrained latent-action generator, while Mimic-Video starts from a video backbone pretrained on MolmoAct. Each method then incorporates an action head for task-specific learning, and both are trained for 30,000 steps. For DreamX, human videos supervise latent-action prediction, while robot demonstrations additionally supervise low-level actions. During rollout, DreamX generates compact latent actions alongside low-level actions, whereas Mimic-Video generates future video.

**Evaluation.** We evaluate reaching and pushing as two subgoals: reaching measures whether the robot approaches the drawer for interaction, and pushing measures whether it completes the required closure. The reported subgoal success rate averages the two completion rates within each configuration, and the overall average weights the three configurations equally. We additionally report inference latency per policy call. Separate reaching and pushing results are provided in Section C.1. We collected 45 trajectories to train the policy and evaluated its success rate over 10 trials in each of three scenarios.

16

<!-- page 17 of 25 -->

## B.4 LARYBENCH EVALUATION DETAILS

**Evaluation protocol.** We freeze all representation models and train a common attentive probe for action regression and classification. Each input token is projected to 768 dimensions through a linear layer, layer normalization, and GELU. For representations spanning multiple transitions, learned temporal embeddings identify the transition positions before the temporal and token dimensions are flattened. A learned-query attentive pooler aggregates the tokens into a single feature for the regression or classification head.

**Feature extraction.** The regression benchmark originally provides only two frames per sample. Using the accompanying metadata, we recover the corresponding video sequences from the source datasets, including the intervening frames, for video-based models.

We extract each model’s native representation without fine-tuning and preserve its temporal and token structure. Pair-based models encode consecutive frame pairs, while video-based models encode clips using their supported temporal windows. Features are precomputed for probe training.

## B.5 ZERO-SHOT HUMAN-TO-ROBOT TRANSFER DETAILS

We evaluate on 1,008 clips spanning 21 tasks from HumanAndRobot (Xie et al., 2026), selecting 48 clips per task. Each method receives a human source video and the initial frame of its paired robot recording. The remaining robot frames serve only as evaluation references; no paired human–robot data is used for training. We compare DreamX with AdaWorld (Gao et al., 2025) and VideoWorld 2 (Ren et al., 2026b) using their native generation horizons.

For a common evaluation horizon, we retain the first 20 frames of each output and exclude the conditioning frame, yielding 19 generated frames per clip. We report FVD using a Kinetics-400- pretrained I3D network, VGG-based LPIPS, PSNR, and SSIM against the corresponding robot reference videos.

## C ADDITIONAL QUANTITATIVE RESULTS

## C.1 POLICY EVALUATION

**Subgoal completion** Table 5 reports reaching and pushing completion separately. Averaging each pair gives the per-configuration SR in Table 3: 50.0% for Mimic-Video in all configurations, and 90.0%, 100.0%, and 100.0% for DreamX.

Table 5: Reaching and pushing completion rates (%) for real-world drawer closing. Each subgoal contributes equally to the SR reported in the main table.

<table><tr><td></td><td colspan="2">Upper</td><td colspan="2">Lower</td><td colspan="2">Both</td></tr><tr><td>Method</td><td>Reach</td><td>Push</td><td>Reach</td><td>Push</td><td>Reach</td><td>Push</td></tr><tr><td>Mimic-Video</td><td>80</td><td>20</td><td>100</td><td>0</td><td>100</td><td>0</td></tr><tr><td>DreamX</td><td>100</td><td>80</td><td>100</td><td>100</td><td>100</td><td>100</td></tr></table>

**Policy learning from human demonstrations** We additionally train a policy using only human videos for the upper-only drawer-closing configuration, while retaining robot demonstrations for the lower-only and both-open configurations. Human videos supervise latent-action prediction without low-level action labels. As shown in Table 6, the policy achieves a 100% subgoal success rate in upper drawer task even if trained only with human dataset.

Table 6: Policy learning with human-only demonstrations of the upper-only configuration. Task columns report approach/push completion rates (%); SR denotes subgoal success rate (%), averaged equally over reaching and pushing and across the three configurations.

| Method | Upper | Lower | Both | SR↑ |
| --- | --- | --- | --- | --- |
| DreamX | 100/100 | 100/100 | 100/100 | 100 |

17

<!-- page 18 of 25 -->

## C.2 ABLATION STUDY

**Latent-action conditioning on CALVIN.** We compare the DreamX policy with and without allowing the action head to attend to latent-action features. In the variant without latent actions, we block the action head’s attention to these features. Table 7 reports the percentage of evaluation sequences completing at least k consecutive tasks, for $k   =   1 , \ldots , 5 ,$ and the average completed sequence length. Conditioning on latent actions improves success rates at all five sequence lengths and increases the average length from 2.378 to 2.643. These results suggest that latent-action information provides useful guidance for robot action prediction in the DreamX policy.

Table 7: CALVIN policy ablation with and without latent-action conditioning. Columns ≥ k report success rates (%) for completing at least k consecutive tasks; Avg. Len. is the average completed sequence length.

| Policy | ≥1 ≥2≥3 |  |  | ≥4 | ≥5 | Avg. Len.↑ |
| --- | --- | --- | --- | --- | --- | --- |
| DreamX policy w/o latent actions |  | 83.4 61.7 | 42.5 | 29.8 20.4 |  | 2.378 |
| DreamX policy |  | 85.7 67.0 |  |  | 49.5 36.4 25.7 | 2.643 |

## C.3 REPRESENTATION ANALYSIS

**Motion structure across embodiments** We examine whether latent actions exhibit common directional structure across robot embodiments using 2,362 RoboCOIN clips spanning ten robot types and six end-effector translation directions.

Figure 12 visualizes 60 directionconditioned latent-action centroids using t-SNE. Centroids for the same direction exhibit similar organization across robot types. This aggregate analysis supports shared directional structure across embodiments.

![Chart block](doc:f9b8e34/tier:basic/page:18/block:8)

**Effect of latent-action conditioning** We quantify how changes to latent actions affect decoded motion and appearance. For latent actions $Q   \in   \mathbb { R } ^ { \overleftarrow { N _ { q } } \times D }$ we perturb the decoder’s conditioning as

Figure 12: Directional structure across robot embodiments. Colors denote motion directions and marker shapes denote robot types.

$$
\widetilde { Q } _ { i , d } = Q _ { i , d } + \alpha s _ { d } \epsilon _ { i , d } , \qquad \epsilon _ { i , d } \sim \mathcal { N } ( 0 , 1 ) ,\tag{10}
$$

where $s _ { d }$ is the channel-wise standard deviation estimated from training features. We evaluate $\alpha   \in   \{ 0 , 0 . 1 , 0 . 3 , 1 . 0 \}$ and a zero-query control on eight selected high-motion clips, with two decoder seeds per clip. Within each comparison, we fix the initial frame, decoder noise, denoising schedule, and perturbation draw, rescaling only the perturbation magnitude. The stronger qualitative perturbation in Figure 7 uses $\alpha = 2$

Table 8 reports mean changes relative to the unperturbed reconstruction, and Figure 13 shows their distributions across clips and decoder seeds. We compute Farneback optical flow at ¨ $1 6 0 \times 1 2 0$ and measure mean flow-vector deviation and motion magnitude relative to the unperturbed output. RGB differences use intensities in [0, 1] and exclude the conditioning frame. A fixed static-region mask selects pixels below the 40th percentile of temporal RGB variation in the reference video, followed by $5 \times 5$ erosion.

Increasing perturbation strength produces larger changes in decoded motion, while static-region appearance changes remain smaller than whole-image changes. Zeroing the latent actions reduces

18

<!-- page 19 of 25 -->

the mean per-sample motion ratio to 0.456. Together with the qualitative examples, these measurements support the role of latent actions in steering motion while the decoder retains much of the scene appearance.

| Conditioning | Motion ratio | Flow RGB deviation L1 | Static RGB L1 |
| --- | --- | --- | --- |
| Original | 1.000 | 0.000 0.0000 | 0.0000 |
| Noise (α = 0.1) | 1.000 | 0.141 0.0054 | 0.0030 |
| Noise (α = 0.3) | 0.995 | 0.211 0.0077 | 0.0034 |
| Noise (α = 1.0) | 0.946 | 0.481 0.0215 | 0.0077 |
| Zero queries | 0.456 | 0.866 0.0758 | 0.0227 |

![Chart block](doc:f9b8e34/tier:basic/page:19/block:3)

![Chart block](doc:f9b8e34/tier:basic/page:19/block:4)

![Chart block](doc:f9b8e34/tier:basic/page:19/block:5)

Table 8: Sensitivity to latent-action perturbations. Means over eight clips and two decoder seeds. Flow deviation (pixels/frame) and RGB L1 are measured against unperturbed reconstructions. Motion ratio is averaged per clip and seed; no perturbation has ratio 1.

Figure 13: Distribution of perturbation effects. Motion ratio, flow deviation, and RGB change for the same eight clips and two decoder seeds as Table 8. Boxes show quartiles and medians; whiskers extend to observations within 1.5 times the interquartile range, with outliers shown individually.

## D ADDITIONAL QUALITATIVE RESULTS AND ANALYSIS

## D.1 ANY-TO-ANY VIDEO TRANSFER

## D.1.1 IN-DOMAIN TRANSFER

Figure 14 provides additional any-to-any video transfer examples across embodiments and environments, complementing Figure 4 in the main paper.

## D.1.2 ADDITIONAL ZERO-SHOT HUMAN-TO-ROBOT TRANSFER

Figure 15 provides additional zero-shot human-to-robot video transfer comparisons on HumanAndRobot, complementing Figure 5. All methods use the evaluation protocol in Section B.5.

## D.2 HUMAN-TO-ROBOT ACTION TRANSFER

Figure 16 provides additional human-to-robot action transfer examples for writing and single-arm manipulation, complementing Figure 6. The experimental setup is described in Section B.2.

## D.3 POLICY ROLLOUTS

Figure 17 provides illustrative real-world drawer-closing rollouts for the upper-only, lower-only, and both-open configurations.

## D.4 QUERY-TO-VIDEO ATTENTION

Figure 18 provides additional query-to-video attention visualizations across human and robot demonstration sequences, complementing Figure 8. We average query-to-video attention over all heads, the four query slots within each temporal group, and the final seven transformer blocks.

19

<!-- page 20 of 25 -->

![Image block](doc:f9b8e34/tier:basic/page:20/block:1)

Figure 14: Additional qualitative Any-to-Any video transfer results. Each example shows a source motion and the corresponding transferred sequence generated from a target-domain initial frame.

20

<!-- page 21 of 25 -->

![Image block](doc:f9b8e34/tier:basic/page:21/block:1)

Grab pencil

![Image block](doc:f9b8e34/tier:basic/page:21/block:3)

Grab cup

Figure 15: Additional zero-shot human-to-robot video transfer results on the HumanAndRobot benchmark.

21

<!-- page 22 of 25 -->

![Image block](doc:f9b8e34/tier:basic/page:22/block:1)

Fold cloth

![Image block](doc:f9b8e34/tier:basic/page:22/block:3)

Figure 15: Additional zero-shot human-to-robot video transfer results on the HumanAndRobot benchmark (continued).

22

<!-- page 23 of 25 -->

![Image block](doc:f9b8e34/tier:basic/page:23/block:1)

![Image block](doc:f9b8e34/tier:basic/page:23/block:2)

Figure 15: Additional zero-shot human-to-robot video transfer results on the HumanAndRobot benchmark (continued).

23

<!-- page 24 of 25 -->

![Image block](doc:f9b8e34/tier:basic/page:24/block:1)

Figure 16: Additional qualitative human-to-robot action transfer results. DreamX transfers human demonstrations into executable robot actions while preserving task-specific trajectory structure and temporal ordering across writing and single-arm manipulation examples.

24

<!-- page 25 of 25 -->

![Image block](doc:f9b8e34/tier:basic/page:25/block:1)

![Image block](doc:f9b8e34/tier:basic/page:25/block:2)

Close the top drawer

![Image block](doc:f9b8e34/tier:basic/page:25/block:4)

Close the bottom drawer

![Image block](doc:f9b8e34/tier:basic/page:25/block:6)

Close both drawers

Figure 17: Real-world drawer-closing rollouts. Six frames are sampled at equal fractions of each rollout’s duration.

![Image block](doc:f9b8e34/tier:basic/page:25/block:9)

![Image block](doc:f9b8e34/tier:basic/page:25/block:10)

Figure 18: Query-to-video attention across time. Human demonstrations (left) and robot demonstrations (right), each shown with the input frame beside its attention overlay. Rows follow temporal order and highlight attention around the acting embodiment and objects.

25