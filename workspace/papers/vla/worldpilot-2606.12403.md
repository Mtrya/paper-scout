# World Pilot: Steering Vision-Language-Action Models with World-Action Priors

Zefu Lin1,2\* Rongxu Cui3\* Junjia Xu3\* Xiaojuan Jin1 Wenling Li3 Lue Fan1# Zhaoxiang Zhang1,2#

1 Institute of Automation, Chinese Academy of Sciences (CASIA) 2 Nanjing University 3 Beihang University {linzefu2022, lue.fan}@ia.ac.cn Equal contribution. # Corresponding author.

Project website: https://world-pilot.github.io/

![](images/bbd0566441912877e5cc98e56e61549593a48a8fb2ed8754872a679831faccf1.jpg)  
Figure 1: World Pilot steers a VLA with priors from a World-Action Model. VLA methods generate actions from a VLM’s encoding of the scene. World Pilot adds two priors from a WAM into the decision chain, with Latent Steering routing a scene-evolution latent into VLM hidden states and Action Steering feeding a trajectory-level motion prior to the action generator. This gives the VLA an anticipated view of the scene and a motion hint alongside its semantic conditioning. World Pilot reaches state-of-the-art performance on LIBERO-Plus and real-robot tasks.

Abstract: Vision-Language-Action (VLA) models inherit semantic grounding from large-scale pretraining and perform competently across in-distribution manipulation tasks. This grounding, however, is built on static image-text pairs, whereas manipulation is a continuous, contact-rich process whose dynamics this pretraining cannot capture. We present World Pilot, a VLA framework that augments the policy with priors from a World-Action Model (WAM), routed into the decision chain through two complementary pathways. Latent Steering conditions the perception layer on a scene-evolution latent, and Action Steering supplies an anticipated trajectory as a motion prior to the action generator. Together the two priors equip the VLA with an anticipated view of the scene and a trajectory-level motion hint alongside its semantic conditioning, and the scene-evolution prior remains effective even when supplied by a video-pretrained world model that has not been action-post-trained. World Pilot attains a state-of-the-art Total success rate of 84.7% on the LIBERO-Plus zero-shot OOD benchmark and the highest success rate on every real-robot setting across four manipulation tasks, with the largest margins under shifts in viewpoint, geometry, deformable state, and pose.

Keywords: Vision-Language-Action Models, World Action Models

## 1 Introduction

Vision-Language-Action (VLA) policies [1, 2, 3, 4] inherit semantic grounding from the image-text pretraining of their VLM backbones and perform competently within the manipulation distribution on which they are fine-tuned [5, 6, 7]. What such pretraining cannot supply is a model of how a scene evolves under action. Image-text pairs are static [8, 9, 10], and the action generator downstream of the VLM consumes purely semantic hidden states with no internal account of the dynamics it must produce [11, 12, 13, 14, 15]. Consistent with this gap, VLAs become fragile once viewpoint, geometry, or contact tolerance drifts away from the training distribution [16, 17, 18, 19].

Video pretraining is the natural complement. Action-conditioned scene evolution is present in video by construction [20, 21, 22], and video-pretrained World-Action Models (WAMs) such as Cosmos Policy [23], mimic-video [24], and DreamZero [25] acquire representations of scene dynamics that transfer broadly across embodiments and visual conditions [26, 27, 28, 29, 30, 31, 32, 33, 34]. Their outputs map onto exactly what VLAs lack: a scene-evolution latent describing how the visible state will change, and a coarse action-trajectory hypothesis sketching the actions whose effects the latent forecasts [35, 36, 37]. Because both predictions come from a shared encoder under joint training, they remain structurally aligned. The two are therefore naturally complementary, with semantic grounding supplied by the VLA and scene dynamics supplied by the WAM.

Realizing this complementarity in practice, however, requires more than placing the two models side by side. Whether the WAM’s signals translate into a more capable policy depends on which signals to extract, in what form to carry them, and at which layers of the VLA to inject them, so that dynamics knowledge reaches the parts of the policy that need it without being diluted in transit.

We answer this question with World Pilot, a VLA framework that routes WAM outputs into the policy through two complementary pathways. Latent Steering injects the scene-evolution latent into VLM hidden states through a residual cross-attention update at the perception layer, supplying spatiotemporal dynamics anticipation. We route the latent rather than a decoded future image because pixel content carries action-irrelevant detail such as texture, lighting, background, and generation artifacts that dilute the dynamics structure the latent encodes directly [38, 39, 40, 41]. Action Steering compresses the anticipated trajectory into a single prefix token at the flow-matching action generator, supplying intent-to-motion grounding through a trajectory-level signal that biases generation toward the WAM’s overall motion shape. The single-token form leaves the generator free to commit to a specific continuous chunk informed by both the prior and the dynamics-enhanced hidden states. The two priors enter at different layers because they carry different kinds of information, and both are additive. Throughout fine-tuning the WAM is kept frozen, with gradient updates restricted to the VLA parameters and the lightweight fusion modules. Both pathways therefore steer the VLA with an existing world model rather than co-train a new one, and VLA fine-tuning never propagates back into the WAM to disturb its pretrained world prior.

The form and entry point of each prior are not freely interchangeable. Several otherwise plausible alternatives, including a decoded future image in place of the latent, per-step trajectory tokens at the action generator, and flow-matching initialization from the WAM’s trajectory, each tie the policy too tightly to a noisy intermediate output and forfeit part of the WAM’s complementary dynamics signal. Our ablations (Section 4.3) benchmark World Pilot against these alternatives under matched training conditions, and only World Pilot’s specific configuration consistently converts the WAM’s complementarity into measurable gain on the LIBERO-Plus OOD benchmark.

We evaluate World Pilot on LIBERO-Plus [42] and RoboCasa [43], and on four real-robot manipulation tasks. World Pilot reaches a state-of-the-art Total success rate of 84.7% on LIBERO-Plus and the highest success rate on every real-robot setting, while remaining competitive on RoboCasa. Margins are largest under shifts in viewpoint, geometry, deformable state, and pose; ablations show that each pathway contributes independently and that the scene-evolution prior remains effective even when supplied by a video-pretrained world model that has not been action-post-trained.

![](images/2314c7e62ae70f094ca06594b4ebcb6c711f1fbf7e00c55713e89151448fe8fb.jpg)  
Figure 2: World Pilot architecture. A semantic pathway encodes images and language with a VLM into hidden states. Two prior pathways from a World-Action Model enter the same decision chain, with Latent Steering routing a scene-evolution latent into the VLM hidden states and Action Steering compressing the anticipated trajectory into a prior token for the flow-matching action generator.

## 2 Related Work

Vision-Language-Action Models. Vision-Language-Action (VLA) policies attach an action generator to a Vision-Language Model (VLM) backbone [44, 45, 46, 47], producing continuous robot actions from visual observations and language instructions [1, 2, 3, 4]. Recent systems such as $\pi _ { 0 . 5 }$ [5], ABot-M0 [6], and CogACT [7] achieve competent in-distribution performance on standard manipulation benchmarks. Their conditioning, however, is built from image-text pretraining alone, with no representation of how the scene will evolve under actions, and they remain fragile under shifts in appearance, viewpoint, and physical interaction [48, 49, 50, 51, 52].

World-Action Models. World-Action Models (WAMs) such as Cosmos Policy [23], mimicvideo [24], and DreamZero [25] are pretrained on large-scale video sequences [26, 27, 28, 29, 30], learning the action-conditioned scene evolution and contact dynamics that image-text pretraining cannot capture. Their video-pretrained representations transfer broadly across embodiments and visual conditions [31, 32, 33, 34, 20, 21, 22]. A natural design question is how to combine WAMderived priors with a VLA’s instruction-following pipeline [53, 54, 55, 56, 57], and prior work has explored several routes. Motus [58] and DreamVLA [59] jointly generate future images and actions in a unified framework, but the visual reconstruction loss pushes the action representation to absorb appearance details unrelated to control. π0.7 [60] and VISTA [61] use predicted future images or subgoal images to guide policy learning; pixel-space outputs encode appearance details such as texture, lighting, background, and generation artifacts that are largely irrelevant to action selection and dilute the control-relevant structure of the underlying world-model latent [62, 63, 64, 65, 38, 39, 40]. Being-H0.7 [66] and WoG [41] pass world-model knowledge through latents or implicit features, reducing pixel-level information loss, but still rely on static future snapshots rather than continuous spatiotemporal evolution. World Pilot instead routes two signals from a WAM into a VLA pipeline: a scene-evolution latent that conditions VLM hidden states through Latent Steering, and an anticipated action trajectory that conditions the action generator through Action Steering.

## 3 Method

## 3.1 Problem Formulation

We study robot manipulation policies conditioned on visual observations and natural-language instructions. At each time step, the policy receives these inputs together with an optional proprioceptive state and predicts an action chunk $\mathbf { A } _ { t } = \left( a _ { t } , \ldots , a _ { t + K - 1 } \right)$ that controls the robot over a future horizon. A standard Vision-Language-Action (VLA) policy encodes images and language with a Vision-Language Model (VLM) into multimodal hidden states, from which an action generator produces ${ \bf A } _ { t }$ . This pipeline inherits semantic grounding from image-text pretraining, but image-text pairs do not capture how a scene evolves under actions, and the action generator that follows operates on these purely semantic VLM hidden states.

World Pilot extends this pipeline with a video-pretrained World-Action Model (WAM) that, from the same inputs, jointly predicts a scene-evolution latent and a coarse action-trajectory hypothesis from a shared encoder, so the two outputs are structurally aligned, with the trajectory describing the actions whose effects the latent forecasts. Let $\mathbf { O } _ { t }$ denote the visual observation, ℓ the language instruction, and $\mathbf { q } _ { t }$ the proprioceptive state when available. The WAM branch returns $\mathbf { Z } _ { t } ^ { w }$ and $\mathbf { \Delta A } _ { t } ^ { w }$ and World Pilot predicts the executable action chunk as

$$
\begin{array} { r } { ( \mathbf { Z } _ { t } ^ { w } , \widetilde { \mathbf { A } } _ { t } ^ { w } ) = W _ { \phi } ( \mathbf { O } _ { t } , \ell , \mathbf { q } _ { t } ) , \qquad \hat { \mathbf { A } } _ { \theta , t } = \pi _ { \theta } ( \mathbf { O } _ { t } , \ell , \mathbf { q } _ { t } ; \mathbf { Z } _ { t } ^ { w } , \widetilde { \mathbf { A } } _ { t } ^ { w } ) , } \end{array}\tag{1}
$$

where $\mathbf { Z } _ { t } ^ { w }$ is the scene-evolution latent, $\widetilde { \mathbf { A } } _ { t } ^ { w }$ is the anticipated action trajectory used as a motion prior, and $\hat { \mathbf { A } } _ { \theta , t }$ is the action chunk produced by World Pilot.

Two pathways route the WAM outputs into the policy (Fig. 2). A semantic backbone first encodes the current images and instruction into VLM hidden states $\mathbf { H } _ { t }$ . Latent Steering (Section 3.2) conditions $\mathbf { H } _ { t }$ on $\mathbf { Z } _ { t } ^ { w }$ to produce dynamics-enhanced hidden states $\begin{array} { r } { \bar { \bf H } _ { t } , } \end{array}$ , supplying spatiotemporal dynamics anticipation at the perception layer. Action Steering (Section 3.3) encodes $\widetilde { \mathbf { A } } _ { t } ^ { w }$ into a single trajectory-level prior token $\mathbf { s } _ { t } ^ { w }$ , supplying intent-to-motion grounding at the action-generation layer. Both pathways are additive: Latent Steering adds a residual to $\mathbf { H } _ { t }$ that preserves the token sequence, and Action Steering inserts a single prefix token into the generator without altering its denoising recurrence, so each pathway can be ablated independently as Section 4 confirms.

## 3.2 Latent Steering

Latent Steering enriches VLM hidden states with cues about anticipated scene evolution. The sceneevolution latent $\mathbf { Z } _ { t } ^ { w }$ carries compact information about predicted object motion, contact outcomes, and local state changes. We use this latent rather than a decoded future image because pixel content carries action-irrelevant detail such as texture, lighting, background, and generation artifacts that dilute the dynamics structure the latent encodes directly [38, 39, 41].

Given the current observation and instruction, the WAM predicts $\mathbf { Z } _ { t } ^ { w }$ as a per-view latent representation of the future visual state. Concretely, the WAM encodes $\mathbf { O } _ { t }$ with a VAE and denoises it via a Diffusion Transformer (DiT), yielding $\mathbf { Z } _ { t } ^ { w }$ . World Pilot projects this latent through a dynamics encoder $f _ { \mathrm { d y n } }$ and adds a temporal embedding $\rho _ { \mathrm { f u t } }$ that marks the tokens as future-scene tokens, giving $\mathbf D _ { t } ^ { w } = f _ { \mathrm { d y n } } ( \mathbf Z _ { t } ^ { w } ) + \rho _ { \mathrm { f u t } } ;$ without this tag, the prior’s contribution diminishes empirically. Let $\mathbf { H } _ { t } \in \mathbb { R } ^ { L \times d }$ denote the VLM hidden states. The Latent Steering block applies cross-attention from $\mathbf { H } _ { t }$ to $\mathbf { D } _ { t } ^ { w }$ and adds the result back as a residual,

$$
\bar { \mathbf { H } } _ { t } = \mathbf { H } _ { t } + \mathrm { C r o s s A t t n } ( \mathbf { H } _ { t } , \mathbf { D } _ { t } ^ { w } ) .\tag{2}
$$

Cross-attention lets each VLM token attend selectively to the parts of $\mathbf { D } _ { t } ^ { w }$ most relevant to its spatial region, rather than receiving a single global modulation. The residual form preserves the original VLM token order and hidden-state structure, so $\bar { \mathbf { H } } _ { t }$ feeds directly into the standard VLA actiongeneration path with no further adaptation or downstream interface change.

## 3.3 Action Steering

Action Steering supplies the action generator with a soft trajectory-level context derived from $\widetilde { \mathbf { A } } _ { t } ^ { w }$ This context guides generation rather than replacing it, and the executed trajectory remains the output of the VLA action generator under standard action supervision.

The WAM produces $\widetilde { \mathbf { A } } _ { t } ^ { w }$ with a horizon and action dimension that depend on the task. World Pilot aligns this trajectory to the VLA action horizon K by resampling and encodes the result with an action encoder $f _ { \mathrm { a c t } }$ into a single prior token $\mathbf { s } _ { t } ^ { w } = f _ { \mathrm { a c t } } ( \mathrm { A l i g n } _ { K } ( \widetilde { \mathbf { A } } _ { t } ^ { w } ) )$ ). A single token summarizes the trajectory’s overall shape rather than pinning generation to per-step targets, leaving the generator free to commit to a specific continuous chunk that reflects both the prior and the dynamics-enhanced hidden states. Per-step conditioning, in contrast, ties each output step to the corresponding WAM step, which we find empirically to be less robust when the WAM trajectory is approximate.

The flow-matching action generator denoises a noisy trajectory $\mathbf { X } _ { \tau , t }$ at flow time τ toward the clean action chunk. World Pilot extends its input to $\left[ { \bf { u } } _ { t } ; { \bf { s } } _ { t } ^ { w } ; { \bf { Q } } _ { t } ; { \bf { X } } _ { \tau , t } \right]$ , where $\mathbf { u } _ { t }$ is the optional state token and $\mathbf { Q } _ { t }$ are learned future-query tokens. The dynamics-enhanced VLM hidden states $\bar { \mathbf { H } } _ { t }$ provide the cross-attention condition. $\mathbf { s } _ { t } ^ { w }$ enters as a prefix rather than as part of the noisy trajectory, so it conditions the denoising recurrence through self-attention without itself being denoised. Table 6 compares this single-token form against three alternative ways of feeding $\widetilde { \mathbf { A } } _ { t } ^ { w }$ to the generator and shows that the encoded single token attains the highest success rate.

## 3.4 Policy Training

Each training sample provides the observation, language instruction, optional proprioceptive state, and expert action chunk $\mathbf { A } _ { t } ^ { \star }$ . Throughout fine-tuning the WAM $W _ { \phi }$ is kept frozen, with gradient updates restricted to the VLA-side parameters $\theta$ (the VLM backbone, the dynamics encoder $f _ { \mathrm { d y n } }$ and Latent Steering cross-attention, the action encoder $f _ { \mathrm { { a c t } } }$ , and the flow-matching action generator). We therefore treat the WAM as an external prior model rather than a component to be co-trained, which is the sense in which World Pilot steers the VLA with an existing world model: VLA finetuning does not propagate gradients back into the WAM, and its forward pass can be precomputed and cached so that it is excluded from the inner training loop. At inference, both the VLA and the WAM run online and produce the priors from the live observation at every decision step, and the fusion paths in Sections 3.2–3.3 consume identically shaped priors at training and inference, so the learned fusion behavior transfers directly to online execution.

Following ABot-M0 [6], we adopt the clean-action parameterization of the flow-matching action generator, which is equivalent to a reweighted velocity-space objective induced by the action-tovelocity transformation. The parameterization keeps the supervision target equal to the expert chunk $\mathbf { A } _ { t } ^ { \star }$ , so the WAM priors enter only through the conditioning path and need no separate prior loss. Given Gaussian noise ϵ and a sampled flow time τ , the noisy trajectory is $\mathbf { X } _ { \tau , t } = \tau \mathbf { A } _ { t } ^ { \star } + ( 1 - \tau ) \epsilon .$ and the action generator predicts a clean action chunk

$$
\hat { \mathbf { A } } _ { \theta , t } = g _ { \theta } \big ( \mathbf { X } _ { \tau , t } , \tau , \mathbf { u } _ { t } , \mathbf { s } _ { t } ^ { w } , \mathbf { Q } _ { t } \mid \bar { \mathbf { H } } _ { t } \big ) .\tag{3}
$$

The training objective is

$$
\mathcal { L } _ { \mathrm { W o r l d } \mathrm { P i l o t } } = \mathbb { E } _ { \tau , \epsilon } \left[ w ( \tau ) \left| \left| \hat { \mathbf { A } } _ { \boldsymbol { \theta } , t } - \mathbf { A } _ { t } ^ { \star } \right| \right| _ { 2 } ^ { 2 } \right] , \qquad w ( \tau ) = \frac { 1 } { ( 1 - \tau ) ^ { 2 } } ,\tag{4}
$$

where $w ( \tau )$ implements the equivalent velocity-space loss under this parameterization. Optimizing this objective end-to-end teaches World Pilot how to use the world priors provided by Latent Steering and Action Steering to guide the action generator toward lower-error actions.

## 4 Experimental Results

## 4.1 Main Experiments

We build World Pilot on the ABot-M0 [6], with Qwen3-VL [67] as the VLM backbone and a DiTbased flow-matching action head, and use Cosmos Policy [23] as the WAM with 5-step denoising. WAM outputs are precomputed during training and run online at evaluation. We apply dropout with rate 0.3 to the WAM conditions $\mathbf { D } _ { t } ^ { w }$ and $\mathbf { s } _ { t } ^ { w }$ to prevent the policy from over-relying on the priors. We fine-tune World Pilot on 8 RTX PRO 6000 GPUs and report success rate.

We evaluate on two simulation benchmarks. LIBERO-Plus [42] is an OOD suite of 10,030 perturbed tasks built on LIBERO [68] that covers seven axes of perturbation (background, camera, language, light, layout, robot, noise). Models are trained only on LIBERO and evaluated zero-shot on the perturbations, with Total reporting the success rate averaged over all perturbed tasks. RoboCasa [43] emphasizes long-horizon manipulation in everyday kitchen scenes.

World Pilot reaches the highest Total success rate on LIBERO-Plus [42] at 84.7% averaged over three random seeds, a 2.6-point margin over the strongest reported baseline (Table 1), and leads on Camera, Light, Background, and Noise while placing close behind the strongest baselines on Language, Robot, and Layout. On the appearance axes (Light, Background, Noise), World Pilot leads on all three, consistent with image-text pretraining at the VLM and video pretraining at the WAM both contributing appearance robustness. On Camera, World Pilot reaches 82.8 (+13.2 over the next baseline), the largest per-axis gain in the table; the WAM’s video pretraining covers diverse camera poses, and the scene-evolution latent carries this coverage into the policy, narrowing the gap that pretraining leaves open. LIBERO-Plus reports that a high Language score reflects insensitivity to instruction perturbations rather than robustness [42], so we read this column as a sanity check. We treat Total as the primary indicator of broad OOD robustness, since the perturbation a deployed scene presents is unknown and Total aggregates over all 10,030 perturbed tasks. On LIBERO [68] itself, recent strong baselines already sit above 98% with little headroom, so World Pilot’s gains concentrate on the OOD axes. On RoboCasa, World Pilot is competitive with the strongest reported baseline, so the same conditioning design carries over to long-horizon kitchen tasks.

Table 1: Simulation results on LIBERO, LIBERO-Plus, and RoboCasa. All LIBERO-Plus numbers come from training on LIBERO only and evaluating zero-shot on its OOD perturbations. The LIBERO-Plus numbers for Cosmos Policy [23] and DreamVLA [59] are our own runs, as is the RoboCasa number for ABot-M0 [6], and the remaining LIBERO-Plus baselines are taken from ABot-M0 and Being-H0.7 [66]. We rerun ABot-M0 on RoboCasa because the ABot-M0 paper reports RoboCasa on the GR1 split rather than the original benchmark used here.
<table><tr><td rowspan="2">Method</td><td rowspan="2">LIBERO</td><td colspan="8">LIBERO-Plus</td><td rowspan="2">RoboCasa</td></tr><tr><td>Camera</td><td></td><td>Robot Language</td><td>Light</td><td>Background Noise</td><td></td><td>Layout</td><td>Total</td></tr><tr><td>OpenVLA [69]</td><td>84.7</td><td>0.8</td><td>3.5</td><td>23.0</td><td>8.1</td><td>34.8</td><td>15.2</td><td>28.5</td><td>15.6</td><td></td></tr><tr><td>WorldVLA [70]</td><td>81.8</td><td>0.1</td><td>27.9</td><td>41.6</td><td>43.7</td><td>17.1</td><td>10.9</td><td>38.0</td><td>25.0</td><td>1</td></tr><tr><td>UniVLA [38]</td><td>95.2</td><td>1.8</td><td>46.2</td><td>69.6</td><td>69.0</td><td>81.0</td><td>21.2</td><td>31.9</td><td>42.9</td><td>1</td></tr><tr><td>To [12]</td><td>94.4</td><td>13.8</td><td>6.0</td><td>58.8</td><td>85.0</td><td>81.4</td><td>79.0</td><td>68.9</td><td>53.6</td><td>42.4</td></tr><tr><td>TT0.5[5]</td><td>96.9</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>77.4</td><td>41.4</td></tr><tr><td>RIPT-VLA</td><td>93.6</td><td>55.2</td><td>31.2</td><td>77.6</td><td>88.4</td><td>91.6</td><td>73.5</td><td>74.2</td><td>68.4</td><td>1</td></tr><tr><td>DreamVLA [59]</td><td>97.5</td><td>26.2</td><td>17.6</td><td>67.0</td><td>77.5</td><td>71.5</td><td>53.6</td><td>43.5</td><td>48.9</td><td>1</td></tr><tr><td>Being-H0.7 [66]</td><td>99.2</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>82.1</td><td>62.1</td></tr><tr><td>Cosmos Policy [23]</td><td>98.5</td><td>69.6</td><td>51.0</td><td>89.6</td><td>97.7</td><td>85.7</td><td>87.3</td><td>83.7</td><td>79.7</td><td>67.1</td></tr><tr><td>ABot-M0 [6]</td><td>98.6</td><td>60.4</td><td>67.9</td><td>86.4</td><td>96.2</td><td>91.6</td><td>86.4</td><td>82.6</td><td>80.5</td><td>54.0</td></tr><tr><td>World Pilot (Ours)</td><td>98.5</td><td>82.8</td><td>60.6</td><td>87.2</td><td>98.6</td><td>96.4</td><td>93.6</td><td>80.5</td><td>84.7</td><td>65.5</td></tr></table>

![](images/f3429d2798f0728a6adb64ec79864822e6f961112506201de790d4bbe8841b12.jpg)  
Figure 3: Real-robot evaluation setup and task scenes. The robot platform (left), in-distribution scenes matching the training conditions (middle), and out-of-distribution scenes (right) under changes in appearance, geometry, deformable state, or pose.

Table 2: Real-robot success rates on four manipulation tasks. Each task has one in-distribution (ID) setting that matches training and two out-of-distribution (OOD) variants that perturb appearance, geometry, deformable state, or pose, and success is measured over 20 trials per setting. Parenthesized red values give the absolute drop from the corresponding ID setting.
<table><tr><td>Method</td><td colspan="3">Stack Blocks</td><td colspan="3">Fold Towel</td><td colspan="3">Fruit-to-Plate</td><td colspan="3">Container-Lid Alignment</td></tr><tr><td></td><td>ID</td><td>Color</td><td></td><td></td><td></td><td>Height ID Direction Novel Towel ID Novel Fruit Layout</td><td></td><td></td><td></td><td></td><td>ID Novel Object Lid Pose</td><td></td></tr><tr><td>π0.5[5]</td><td></td><td>40 15 (-25)</td><td>0(-40)</td><td></td><td>5525 (-30)</td><td></td><td>10 (-45)</td><td>35</td><td>10 (-25)</td><td>5 (-30) 40</td><td>15 (-25)</td><td>5 (-35)</td></tr><tr><td>ABot-M0 [6]</td><td></td><td></td><td></td><td></td><td>60 25 (-35) 10(-50) 50 20 (-30)</td><td>5 (-45)</td><td>65</td><td>30 (-35)</td><td></td><td>15(-50） 65</td><td>30 (-35)</td><td>15 (-50)</td></tr><tr><td>Cosmos Policy [23] 6530(-35) 15(-50) 4515(-30)</td><td></td><td></td><td></td><td></td><td></td><td>10 (-35)</td><td></td><td>70</td><td>35 (-35)</td><td>20(-50) 60</td><td>25 (-35)</td><td>10 (-50)</td></tr><tr><td>World Pilot (Ours) 70 55 (-15) 50 (-20) 8575 (-10)</td><td></td><td></td><td></td><td></td><td></td><td></td><td>70 (-15)</td><td>90</td><td>75 (-15)</td><td>70 (-20) 80</td><td>70 (-10)</td><td>65 (-15)</td></tr></table>

OOD settings. Stack Blocks, block color and stacking height. Fold Towel, towel direction and towel instance. Fruit-to-Plate, fruit category and fruit/plate layout. Container-Lid Alignment, object category and lid pose, where success requires the lid to be aligned with the container rim and fully closed.

## 4.2 Real-World Experiments

The platform and the ID/OOD scenes are illustrated in Fig. 3. We compare World Pilot with ABot-M0 [6], Cosmos Policy [23], and $\pi _ { 0 . 5 }$ [5] on the same robot-arm platform and RGB inputs, across four manipulation tasks (stacking blocks, folding towels, placing fruit on a plate, and container-lid alignment), each with one ID and two OOD variants that change geometry, deformable state, or pose. For each task we collect 100 ID teleoperated demonstrations, fine-tune all methods for 10,000 steps under a matched optimizer, batch size, and learning-rate schedule, and run 20 trials per task setting per method, scoring a trial as successful if the robot reaches the final state specified by the language instruction within the allowed time. We read individual cell differences below 10 percentage points as within trial-level variance and rely on consistent direction across the 12 task-setting cells.

World Pilot attains the highest success rate on every setting in Table 2, with the largest margins under OOD perturbations. World Pilot’s ID-to-OOD drop stays within 20 absolute points, while other baselines drop by 25 to 50. Container-lid alignment is the most stringent setting, requiring tight geometric tolerance for closure; under OOD pose and object changes World Pilot succeeds in 13 to 14 of 20 trials, while no baseline exceeds 6. Both priors thus remain effective when the object’s geometry, pose, or appearance moves outside the training distribution, providing trajectorylevel conditioning and anticipated scene-state cues that VLM hidden states do not carry.

## 4.3 Ablations

We organize ablations by pathway, with Latent Steering on the perception side and Action Steering on the action-generator side. We first show that each pathway contributes, then probe the source and form of Latent Steering’s prior, and finally vary how the action-trajectory prior enters the generator.

Each pathway contributes. We first isolate the two pathways on LIBERO-Plus [42] (Table 3). Latent Steering alone reaches 83.7% (+3.2 over the 80.5% ABot-M0 [6] baseline), Action Steering alone reaches 83.1% (+2.6), and combining the two pathways gives the strongest result at 84.7%, indicating that anticipated scene dynamics and trajectory-level priors contribute complementary signals beyond the VLM’s semantic representation.

Latent Steering: the world prior is already present before action fine-tuning. We test whether the world prior Latent Steering consumes is already supplied by a world model that produces only future-scene predictions, namely Cosmos-Predict [71], or whether it requires the further action posttraining that adapts Cosmos-Predict into Cosmos Policy [23]. Cosmos-Predict is pretrained on large-scale, filtered, VLM-captioned video and image data covering Physical AI scenes such as driving, robot manipulation, human activity, navigation, natural physical dynamics, first-person views, and synthetic rendering, so its scene-evolution latent already encodes broadly transferable dynamics structure before any action-side adaptation. We take Cosmos-Predict in scene-prediction-only mode, route its scene-evolution latent through Latent Steering with Action Steering disabled, and additionally evaluate on RoboTwin2.0 (clean) [72] to test how broadly the world prior transfers. This world-model-only signal still improves over ABot-M0 on every benchmark (Table 4), reaching 82.6 (+2.1) on LIBERO-Plus [42], 62.7 (+8.7) on RoboCasa, and 85.3 (+4.1) on RoboTwin2.0 (clean). The action post-training that adapts Cosmos-Predict into Cosmos Policy [23] further sharpens the signal (the Cosmos-Policy-based counterpart in Table 3 reaches 83.7% on LIBERO-Plus with Latent Steering only, +1.1 over the Cosmos-Predict-only setting under matched projection head, dropout, and training schedule), but the prior takes effect even without it.

Table 3: Contribution of each prior pathway on LIBERO-Plus. Each pathway is enabled individually in isolation and then evaluated in combination, and green values mark absolute gains over the ABot-M0 baseline.
<table><tr><td>Variant</td><td>Success (%)</td></tr><tr><td>ABot-M0 [6] (baseline)</td><td>80.5</td></tr><tr><td>Latent Steering only</td><td>83.7 (+3.2)</td></tr><tr><td>Action Steering only</td><td>83.1 (+2.6)</td></tr><tr><td>Full World Pilot</td><td>84.7 (+4.2)</td></tr></table>

Table 4: World-model-only prior transfer. The WAM is replaced by Cosmos-Predict [71], which has not been action-post-trained and produces only future latents, with only Latent Steering (LS) active. RoboTwin2.0 results are reported on the clean split.
<table><tr><td>Benchmark</td><td></td><td>ABot-Mo +LS (world model)</td></tr><tr><td>LIBERO-Plus</td><td>80.5</td><td>82.6 (+2.1)</td></tr><tr><td>RoboCasa</td><td>54.0</td><td>62.7 (+8.7)</td></tr><tr><td>RoboTwin2.0</td><td>81.2</td><td>85.3 (+4.1)</td></tr></table>

Table 6: Action-prior form on LIBERO-Plus. Four ways of feeding the WAM trajectory $\widetilde { \mathbf { A } } _ { t } ^ { w }$ to the flow-matching generator are compared, varying granularity and entry point, where Ours marks World Pilot’s default setting.

Table 5: Future-scene representation on LIBERO-Plus. Latent rows take the WAM cue at a Cosmos denoising step, and the decoded variant replaces the latent with a decoded future image to test whether pixel-space realism helps.
<table><tr><td>Future Information</td><td>Success (%)</td></tr><tr><td>Future latent (1 step)</td><td>84.6</td></tr><tr><td>Future latent (3 steps)</td><td>84.5</td></tr><tr><td>Future latent (5 steps)</td><td>84.7</td></tr><tr><td>Decoded future image</td><td>83.5</td></tr></table>

<table><tr><td>Action Prior Form</td><td>Success (%)</td></tr><tr><td>Single encoded token (Ours)</td><td>84.7</td></tr><tr><td>Per-step encoded tokens</td><td>83.6</td></tr><tr><td>Flow init. from  $\widetilde { \mathbf { A } } _ { t } ^ { w }$ </td><td>84.1</td></tr><tr><td>Raw  $\widetilde { \mathbf { A } } _ { t } ^ { w }$ </td><td>83.0</td></tr></table>

Latent Steering: latent injection over decoded future images. Given that the latent prior is already useful from world-model pretraining alone, we ask in what form it should enter Latent Steering (Table 5), evaluating 1-step, 3-step, and 5-step latents (taken from intermediate Cosmos denoising states) together with a fully decoded future image passed through the VLA’s image encoder. Latent injection is stable across denoising depths, with the three latent variants falling within 0.2 points of each other (84.5 to 84.7%), since World Pilot relies on state-transition cues and local dynamics structure encoded in the latent rather than on pixel-level realism. Replacing the latent with a fully decoded future image instead lowers Total to 83.5% (a 1.2-point drop), as pixel-level decoding adds visual artifacts and dilutes the dynamics structure.

Action Steering: how the trajectory prior conditions the generator. We vary how the trajectory $\widetilde { \mathbf { A } } _ { t } ^ { w }$ enters the action generator (Table 6). We compare the single trajectory-level prior token against three alternatives: per-step encoded tokens, flow-matching initialization from $\widetilde { \mathbf { A } } _ { t } ^ { w }$ , and use of the raw $\widetilde { \mathbf { A } } _ { t } ^ { w }$ as the action prior. The single-token form gives the strongest result at 84.7%. Per-step tokens (83.6%) and the raw trajectory (83.0%) pin generation to noisy step-level signals, propagating WAM trajectory noise and compounding errors across the chunk. Flow-matching initialization recovers part of this gap (84.1%) but ties the final output to the WAM’s action quality, leaving the generator less room to correct the prior with VLA-side cues. Compressing the trajectory into a single conditioning token keeps the prior as guidance while the generator commits to a chunk that reflects both the prior and the dynamics-enhanced hidden states.

## 5 Conclusion and Limitations

We propose a training recipe that augments VLA policy learning with priors from a World-Action Model (WAM), routed through Latent Steering on the perception side and Action Steering on the action-generator side. We instantiate this recipe as World Pilot, which attains state-of-the-art performance on LIBERO-Plus [42] and the highest success rate on every real-robot setting.

Limitations. World Pilot inherits its WAM’s coverage, so when test scenes fall outside the WAM’s video pretraining distribution, both priors degrade and the gains shrink. The improvements are also uneven: World Pilot trails on the Language, Robot, and Layout axes of LIBERO-Plus, and real-robot OOD success still drops by 10 to 20 points relative to ID, so the priors reduce but do not eliminate the effect of OOD shifts. By design, the WAM and VLA are coupled only through the action loss, a modular choice that keeps either component interchangeable with stronger world models or different VLA backbones but does not pursue the tighter prior-policy co-adaptation that joint training could provide. Each decision step also incurs an extra WAM forward pass, which limits applicability to high-frequency reactive control. Three directions follow from these limitations: uncertainty-aware prior gating to handle WAM-coverage drops, joint WAM-VLA co-tuning to close the prior-policy loop, and prior distillation or adaptive querying to reduce the per-step overhead.

## Acknowledgments

If a paper is accepted, the final camera-ready version will (and probably should) include acknowledgments. All acknowledgments go at the end of the paper, including thanks to reviewers who gave useful comments, to colleagues who contributed to the ideas, and to funding agencies and corporate sponsors that provided financial support.

## References

[1] X. Chen, Y. Chen, Y. Fu, N. Gao, J. Jia, W. Jin, H. Li, Y. Mu, J. Pang, Y. Qiao, Y. Tian, B. Wang, B. Wang, F. Wang, H. Wang, T. Wang, Z. Wang, X. Wei, C. Wu, S. Yang, J. Ye, J. Yu, J. Zeng, J. Zhang, J. Zhang, S. Zhang, F. Zheng, B. Zhou, and Y. Zhu. Internvla-m1: A spatially guided vision-language-action framework for generalist robot policy, 2025. URL https://arxiv.org/abs/2510.13778.

[2] H. Zhen, X. Qiu, P. Chen, J. Yang, X. Yan, Y. Du, Y. Hong, and C. Gan. 3d-vla: A 3d vision-language-action generative world model, 2024. URL https://arxiv.org/abs/ 2403.09631.

[3] S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation, 2024. URL https://arxiv.org/abs/2410. 07864.

[4] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, P. Florence, C. Fu, M. G. Arenas, K. Gopalakrishnan, K. Han, K. Hausman, A. Herzog, J. Hsu, B. Ichter, A. Irpan, N. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, L. Lee, T.-W. E. Lee, S. Levine, Y. Lu, H. Michalewski, I. Mordatch, K. Pertsch, K. Rao, K. Reymann, M. Ryoo, G. Salazar, P. Sanketi, P. Sermanet, J. Singh, A. Singh, R. Soricut, H. Tran, V. Vanhoucke, Q. Vuong, A. Wahid, S. Welker, P. Wohlhart, J. Wu, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich. Rt-2: Vision-language-action models transfer web knowledge to robotic control, 2023. URL https://arxiv.org/abs/2307.15818.

[5] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al. π0.5: A vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[6] Y. Yang, S. Zeng, T. Lin, X. Chang, D. Qi, J. Xiao, H. Liu, R. Chen, Y. Chen, D. Huo, et al. Abot-m0: Vla foundation model for robotic manipulation with action manifold learning. arXiv preprint arXiv:2602.11236, 2026.

[7] Q. Li, Y. Liang, Z. Wang, L. Luo, X. Chen, M. Liao, F. Wei, Y. Deng, S. Xu, Y. Zhang, et al. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650, 2024.

[8] S. Gu, Y. Cai, T. Wang, S. Wu, and Y. Fu. Say, dream, and act: Learning video world models for instruction-driven robot manipulation. arXiv preprint arXiv:2602.10717, 2026. URL https: //arxiv.org/abs/2602.10717.

[9] J. Wang et al. MVISTA-4D: View-consistent 4d world model with test-time action inference for robotic manipulation. arXiv preprint arXiv:2602.09878, 2026. URL https://arxiv. org/abs/2602.09878.

[10] L. Fan, Z. Xu, C. Cao, W. Zhang, M. Yuan, and J. Chen. Aim: Intent-aware unified world action modeling with spatial value maps, 2026. URL https://arxiv.org/abs/2604.11135.

[11] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. In Proceedings of Robotics: Science and Systems, Daegu, Republic of Korea, July 2023. doi:10.15607/RSS.2023.XIX.016.

[12] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. pi 0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[13] D. Qu, H. Song, Q. Chen, Y. Yao, X. Ye, Y. Ding, Z. Wang, J. Gu, B. Zhao, D. Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. arXiv preprint arXiv:2501.15830, 2025.

[14] R. Zhang, M. Dong, Y. Zhang, L. Heng, X. Chi, G. Dai, L. Du, D. Wang, Y. Du, and S. Zhang. Mole-vla: Dynamic layer-skipping vision language action model via mixture-of-layers for efficient robot manipulation. arXiv preprint arXiv:2503.20384, 2025.

[15] J. Zhang, Y. Guo, Y. Hu, X. Chen, X. Zhu, and J. Chen. Up-vla: A unified understanding and prediction model for embodied agent. arXiv preprint arXiv:2501.18867, 2025.

[16] L. Fan, K. Chen, Z. Xu, M. Yuan, P. Huang, and W. Huang. Language reasoning in visionlanguage-action model for robotic grasping. In 2024 China Automation Congress (CAC), pages 6656–6661. IEEE, 2024.

[17] J. Wen, Y. Zhu, J. Li, M. Zhu, Z. Tang, K. Wu, Z. Xu, N. Liu, R. Cheng, C. Shen, et al. Tinyvla: Towards fast, data-efficient vision-language-action models for robotic manipulation. IEEE Robotics and Automation Letters, 2025.

[18] C. Li, J. Wen, Y. Peng, Y. Peng, F. Feng, and Y. Zhu. Pointvla: Injecting the 3d world into vision-language-action models. arXiv preprint arXiv:2503.07511, 2025.

[19] J. Liu, H. Chen, P. An, Z. Liu, R. Zhang, C. Gu, X. Li, Z. Guo, S. Chen, M. Liu, et al. Hybridvla: Collaborative diffusion and autoregression in a unified vision-language-action model. arXiv preprint arXiv:2503.10631, 2025.

[20] J. Guo, Q. Li, P. Li, Z. Chen, N. Sun, Y. Su, H. Wang, Y. Zhang, X. Li, and H. Liu. Unified 4d world action modeling from video priors with asynchronous denoising, 2026. URL https: //arxiv.org/abs/2604.26694.

[21] R. Zheng, J. Wang, S. Reed, J. Bjorck, Y. Fang, F. Hu, J. Jang, K. Kundalia, Z. Lin, L. Magne, A. Narayan, Y. L. Tan, G. Wang, Q. Wang, J. Xiang, Y. Xu, S. Ye, J. Kautz, F. Huang, Y. Zhu, and L. Fan. Flare: Robot learning with implicit world modeling, 2025. URL https://arxiv. org/abs/2505.15659.

[22] L. Li, Q. Zhang, Y. Luo, S. Yang, R. Wang, F. Han, M. Yu, Z. Gao, N. Xue, X. Zhu, Y. Shen, and Y. Xu. Causal world modeling for robot control, 2026. URL https://arxiv.org/abs/ 2601.21998.

[23] M. J. Kim, Y. Gao, T.-Y. Lin, Y.-C. Lin, Y. Ge, G. Lam, P. Liang, S. Song, M.-Y. Liu, C. Finn, and J. Gu. Cosmos policy: Fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163, 2026.

[24] J. Pai, L. Achenbach, V. Montesinos, B. Forrai, O. Mees, and E. Nava. mimic-video: Videoaction models for generalizable robot control beyond vlas. arXiv preprint arXiv:2512.15692, 2025.

[25] S. Ye, Y. Ge, K. Zheng, S. Gao, S. Yu, G. Kurian, S. Indupuru, Y. L. Tan, C. Zhu, J. Xiang, A. Malik, K. Lee, W. Liang, N. Ranawaka, J. Gu, Y. Xu, G. Wang, F. Hu, A. Narayan, J. Bjorck, J. Wang, G. Kim, D. Niu, R. Zheng, Y. Xie, J. Wu, Q. Wang, R. Julian, D. Xu, Y. Du, Y. Chebotar, S. Reed, J. Kautz, Y. Zhu, L. J. Fan, and J. Jang. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026.

[26] A. Ye, B. Wang, C. Ni, G. Huang, G. Zhao, H. Li, H. Li, J. Li, J. Lv, J. Liu, M. Cao, P. Li, Q. Deng, W. Mei, X. Wang, X. Chen, X. Zhou, Y. Wang, Y. Chang, Y. Li, Y. Zhou, Y. Ye, Z. Liu, and Z. Zhu. Gigaworld-policy: An efficient action-centered world–action model, 2026. URL https://arxiv.org/abs/2603.17240.

[27] D. Ha and J. Schmidhuber. World models. 2018. doi:10.5281/ZENODO.1207631. URL https://zenodo.org/record/1207631.

[28] M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, Mojtaba, Komeili, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus, S. Arnaud, A. Gejji, A. Martin, F. R. Hogan, D. Dugas, P. Bojanowski, V. Khalidov, P. Labatut, F. Massa, M. Szafraniec, K. Krishnakumar, Y. Li, X. Ma, S. Chandar, F. Meier, Y. LeCun, M. Rabbat, and N. Ballas. V-jepa 2: Self-supervised video models enable understanding, prediction and planning, 2025. URL https://arxiv. org/abs/2506.09985.

[29] C. Deng, D. Zhu, K. Li, C. Gou, F. Li, Z. Wang, S. Zhong, W. Yu, X. Nie, Z. Song, G. Shi, and H. Fan. Emerging properties in unified multimodal pretraining, 2025. URL https: //arxiv.org/abs/2505.14683.

[30] H. Zhen, Q. Sun, H. Zhang, J. Li, S. Zhou, Y. Du, and C. Gan. Tesseract: Learning 4d embodied world models, 2025. URL https://arxiv.org/abs/2504.20995.

[31] J. Bruce, M. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps, Y. Aytar, S. Bechtle, F. Behbahani, S. Chan, N. Heess, L. Gonzalez, S. Osindero, S. Ozair, S. Reed, J. Zhang, K. Zolna, J. Clune, N. de Freitas, S. Singh, and T. Rocktaschel. Genie: Generative interactive environments, 2024. URL ¨ https://arxiv. org/abs/2402.15391.

[32] S. Zhou, Y. Du, J. Chen, Y. Li, D.-Y. Yeung, and C. Gan. Robodreamer: Learning compositional world models for robot imagination, 2024. URL https://arxiv.org/abs/2404. 12377.

[33] J. Yang, K. Lin, J. Li, W. Zhang, T. Lin, L. Wu, Z. Su, H. Zhao, Y.-Q. Zhang, L. Chen, P. Luo, X. Yue, and H. Li. Rise: Self-improving robot policy with compositional world model, 2026. URL https://arxiv.org/abs/2602.11075.

[34] NVIDIA. Cosmos-predict2: World simulation model for physical ai, 2025. URL https: //github.com/nvidia-cosmos/cosmos-predict2.

[35] R. Li, H. Zhang, J. Jin, Q. Zeng, Z. Zhuang, Y. Tang, S. Lyu, and D. Wang. World-value-action model: Implicit planning for vision-language-action systems, 2026. URL https://arxiv. org/abs/2604.14732.

[36] J. Quevedo, A. K. Sharma, Y. Sun, V. Suryavanshi, P. Liang, and S. Yang. Worldgym: World model as an environment for policy evaluation, 2025. URL https://arxiv.org/abs/2506. 00613.

[37] S. Gao, S. Zhou, Y. Du, J. Zhang, and C. Gan. Adaworld: Learning adaptable world models with latent actions, 2025. URL https://arxiv.org/abs/2503.18938.

[38] Q. Bu, Y. Yang, J. Cai, S. Gao, G. Ren, M. Yao, P. Luo, and H. Li. Univla: Learning to act anywhere with task-centric latent actions. arXiv preprint arXiv:2505.06111, 2025.

[39] M. Liu, J. Shu, H. Chen, Z. Li, C. Zhao, J. Yang, S. Gao, H. Chen, and C. Shen. Stamo: Unsupervised learning of generalizable robot motion from compact state representation, 2025.

[40] C. Zhang, T. Pearce, P. Zhang, K. Wang, X. Chen, W. Shen, L. Zhao, and J. Bian. What do latent action models actually learn?, 2025.

[41] Y. Su, S. Chen, H. Shi, M. Liu, Z. Zhang, N. Huang, W. Zhong, Z. Zhu, Y. Liu, and X. Liu. World guidance: World modeling in condition space for action generation. arXiv preprint arXiv:2602.22010, 2026.

[42] S. Fei, S. Wang, J. Shi, Z. Dai, J. Cai, P. Qian, L. Ji, X. He, S. Zhang, Z. Fei, J. Fu, J. Gong, and X. Qiu. Libero-plus: In-depth robustness analysis of vision-language-action models. arXiv preprint arXiv:2510.13626, 2025.

[43] S. Nasiriany, A. Maddukuri, L. Zhang, A. Parikh, A. Lo, A. Joshi, A. Mandlekar, and Y. Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. In Robotics: Science and Systems (RSS), 2024.

[44] M. Zhu, Y. Zhu, J. Li, Z. Zhou, J. Wen, X. Liu, C. Shen, Y. Peng, and F. Feng. Objectvla: End-to-end open-world object manipulation without demonstration. arXiv preprint arXiv:2502.19250, 2025.

[45] S. Dey, J.-N. Zaech, N. Nikolov, L. Van Gool, and D. P. Paudel. Revla: Reverting visual domain limitation of robotic foundation models. arXiv preprint arXiv:2409.15250, 2024.

[46] B. Zhang, Y. Zhang, J. Ji, Y. Lei, J. Dai, Y. Chen, and Y. Yang. Safevla: Towards safety alignment of vision-language-action model via safe reinforcement learning. arXiv preprint arXiv:2503.03480, 2025.

[47] Y. Fan, P. Ding, S. Bai, X. Tong, Y. Zhu, H. Lu, F. Dai, W. Zhao, Y. Liu, S. Huang, et al. Long-vla: Unleashing long-horizon capability of vision language action model for robot manipulation. arXiv preprint arXiv:2508.19958, 2025.

[48] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, J. Ibarz, B. Ichter, A. Irpan, T. Jackson, S. Jesmonth, N. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, K.-H. Lee, S. Levine, Y. Lu, U. Malla, D. Manjunath, I. Mordatch, O. Nachum, C. Parada, J. Peralta, E. Perez, K. Pertsch, J. Quiambao, K. Rao, M. Ryoo, G. Salazar, P. Sanketi, K. Sayed, J. Singh, S. Sontakke, A. Stone, C. Tan, H. Tran, V. Vanhoucke, S. Vega, Q. Vuong, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich. Rt-1: Robotics transformer for real-world control at scale. In arXiv preprint arXiv:2212.06817, 2022.

[49] H. Li, P. Ding, R. Suo, Y. Wang, Z. Ge, D. Zang, K. Yu, M. Sun, H. Zhang, D. Wang, and W. Su. Vla-rft: Vision-language-action reinforcement fine-tuning with verified rewards in world simulators, 2025. URL https://arxiv.org/abs/2510.00406.

[50] Y. Shen, F. Wei, Z. Du, Y. Liang, Y. Lu, J. Yang, N. Zheng, and B. Guo. Videovla: Video generators can be generalizable robot manipulators, 2025. URL https://arxiv.org/abs/ 2512.06963.

[51] C.-L. Cheang, G. Chen, Y. Jing, T. Kong, H. Li, Y. Li, Y. Liu, H. Wu, J. Xu, Y. Yang, H. Zhang, and M. Zhu. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation, 2024. URL https://arxiv.org/abs/2410.06158.

[52] Q. Zhao, Y. Lu, M. J. Kim, Z. Fu, Z. Zhang, Y. Wu, Z. Li, Q. Ma, S. Han, C. Finn, et al. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. arXiv preprint arXiv:2503.22020, 2025.

[53] X. Liu, Z. Bai, H. Ci, K. Y. Ma, and M. Z. Shou. World-vla-loop: Closed-loop learning of video world model and vla policy, 2026. URL https://arxiv.org/abs/2602.06508.

[54] J. Xiao, Y. Yang, X. Chang, R. Chen, F. Xiong, M. Xu, W.-S. Zheng, and Q. Zhang. World-env: Leveraging world model as a virtual environment for vla post-training, 2026. URL https: //arxiv.org/abs/2509.24948.

[55] Y. Li, Y. Zhu, J. Wen, C. Shen, and Y. Xu. Worldeval: World model as real-world robot policies evaluator, 2025. URL https://arxiv.org/abs/2505.19017.

[56] J. Won, K. Lee, H. Jang, D. Kim, and J. Shin. Dual-stream diffusion for world-model augmented vision-language-action model, 2025. URL https://arxiv.org/abs/2510.27607.

[57] T. Ma, J. Zheng, Z. Wang, C. Jiang, A. Cui, J. Liang, and S. Yang. Dit4dit: Jointly modeling video dynamics and actions for generalizable robot control, 2026. URL https://arxiv. org/abs/2603.10448.

[58] H. Bi, H. Tan, S. Xie, Z. Wang, S. Huang, H. Liu, R. Zhao, Y. Feng, C. Xiang, Y. Rong, H. Zhao, H. Liu, Z. Su, L. Ma, H. Su, and J. Zhu. Motus: A unified latent action world model. arXiv preprint arXiv:2512.13030, 2025.

[59] W. Zhang, H. Liu, Z. Qi, Y. Wang, X. Yu, J. Zhang, R. Dong, J. He, F. Lu, H. Wang, Z. Zhang, L. Yi, W. Zeng, and X. Jin. Dreamvla: A vision-language-action model dreamed with comprehensive world knowledge. arXiv preprint arXiv:2507.04447, 2025.

[60] Physical Intelligence, B. Ai, A. Amin, R. Aniceto, A. Balakrishna, G. Balke, K. Black, et al. π0.7: a steerable generalist robotic foundation model with emergent capabilities. arXiv preprint arXiv:2604.15483, 2026.

[61] Q. Long, Y. Wang, J. Song, J. Zhang, P. Li, W. Wang, Y. Wang, H. Li, S. Xie, G. Yao, et al. Scaling world model for hierarchical manipulation policies. arXiv preprint arXiv:2602.10983, 2026.

[62] S. Li, Y. Gao, D. Sadigh, and S. Song. Unified video action model, 2025. URL https: //arxiv.org/abs/2503.00200.

[63] X. Xu, H. Li, J. Ye, Y. Chen, J. Zeng, X. Chen, L. Xu, D. Lin, W. Li, and J. Pang. Futurevla: Joint visuomotor prediction for vision-language-action model, 2026. URL https://arxiv. org/abs/2603.10712.

[64] S. Miao, N. Feng, J. Wu, Y. Lin, X. He, D. Li, and M. Long. Jepa-vla: Video predictive embedding is needed for vla models, 2026. URL https://arxiv.org/abs/2602.11832.

[65] Q. Lv, W. Kong, H. Li, J. Zeng, Z. Qiu, D. Qu, H. Song, Q. Chen, X. Deng, and J. Pang. F1: A vision-language-action model bridging understanding and generation to actions. CoRR, abs/2509.06951, 2025. doi:10.48550/ARXIV.2509.06951. URL https://doi.org/10. 48550/arXiv.2509.06951.

[66] BeingBeyond Team. Being-h0.7: A latent world-action model from egocentric videos. Technical report / project page, 2026. URL https://research.beingbeyond.com/being-h07.

[67] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[68] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. arXiv preprint arXiv:2306.03310, 2023.

[69] M. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[70] J. Cen, S. Huang, Y. Yuan, K. Li, H. Yuan, C. Yu, Y. Jiang, J. Guo, X. Li, H. Luo, F. Wang, F. Wang, and D. Zhao. Rynnvla-002: A unified vision-language-action and world model. arXiv preprint arXiv:2511.17502, 2025.

[71] NVIDIA, A. Ali, J. Bai, M. Bala, Y. Balaji, A. Blakeman, T. Cai, J. Cao, T. Cao, E. Cha, Y.-W. Chao, P. Chattopadhyay, M. Chen, Y. Chen, Y. Chen, S. Cheng, Y. Cui, J. Diamond, Y. Ding, J. Fan, L. Fan, L. Feng, F. Ferroni, S. Fidler, X. Fu, R. Gao, Y. Ge, J. Gu, A. Gupta, S. Gururani, I. El Hanafi, A. Hassani, Z. Hao, J. Huffman, J. Jang, P. Jannaty, J. Kautz, G. Lam, X. Li, Z. Li, M. Liao, C.-H. Lin, T.-Y. Lin, Y.-C. Lin, H. Ling, M.-Y. Liu, X. Liu, Y. Lu, A. Luo, Q. Ma, H. Mao, K. Mo, S. Nah, Y. Narang, A. Panaskar, L. Pavao, T. Pham, M. Ramezanali, F. Reda, S. Reed, X. Ren, H. Shao, Y. Shen, S. Shi, S. Song, B. Stefaniak, S. Sun, S. Tang, S. Tasmeen, L. Tchapmi, W.-C. Tseng, J. Varghese, A. Z. Wang, H. Wang, H. Wang, H. Wang, T.-C. Wang, F. Wei, J. Xu, D. Yang, X. Yang, H. Ye, S. Ye, X. Zeng, J. Zhang, Q. Zhang, K. Zheng, A. Zhu, and Y. Zhu. World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062, 2025.

[72] T. Chen, Z. Chen, B. Chen, Z. Cai, Y. Liu, Z. Li, Q. Liang, X. Lin, Y. Ge, Z. Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025.