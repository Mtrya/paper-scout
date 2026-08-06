# ST-WAM: Semantic-Temporal World Action Model for Robust Manipulation under Visual Distribution Shifts

Mingxin Wang<sup>1,2</sup>, Bin Hu<sup>1</sup>, Bin Qian<sup>1</sup>, Kaitao Jiang<sup>2</sup>, Haoning Wu<sup>3</sup>, Feng Yan<sup>4</sup>, Bowen Jing<sup>5</sup>, Ruiyang Hao<sup>6</sup>, Enyi Wang<sup>1</sup>, Kangning Niu<sup>2</sup>, Yandan Yang<sup>2</sup>, Mu Xu<sup>2</sup>, Yan Wang<sup>1</sup>, Houde Liu<sup>1∗</sup>, Tianlun Li<sup>2∗</sup>

<sup>1</sup>Tsinghua University, <sup>2</sup>AMAP-CV-LAB, Alibaba Group <sup>3</sup>Shanghai Jiao Tong University, <sup>4</sup>Xi’an Jiaotong University <sup>5</sup>The University of Manchester, <sup>6</sup>King’s College London

## Abstract

World Action Models (WAMs) have emerged as a promising paradigm by jointly modeling robot actions and future visual dynamics. However, their reliance on pixel-generative future supervision can entangle action-relevant state transitions with task-irrelevant visual content, limiting robustness under visual distribution shifts. We identify Training-Distribution Hallucination, a recurring phenomenon in which futures conditioned on visually shifted observations hallucinate trainingdomain content rather than remain faithful to the current scene. A controlled frame-triplet diagnosis further shows that DI-NOv3 features remain more stable across visual shifts while better preserving task-state distinctions than Wan-VAE latents. Rather than correcting the predicted futures, we propose Semantic-Temporal WAM (ST-WAM) to improve action robustness by using DINOv3 as a shared semantic representation for future prediction and history retrieval while retaining fine-grained VAE dynamics. Its Dual-Space Future Experts (DSFE) jointly predict future VAE latents and DINO features, while Current-Anchored Intent Retrieval (CAIR) retrieves task-relevant evidence from recent DINO history under the current visual-language context. ST-WAM is trained end-toend without additional embodied pretraining or task-specific annotations, and requires no explicit future generation at inference. It achieves 98.7% on LIBERO and 92.8% on RoboTwin 2.0; more importantly, compared with Fast-WAM, it improves zero-shot LIBERO-Plus performance by 21.3 percentage points and more than doubles real-world success under visual shifts from 25.8% to 61.5%. These results demonstrate that semantic-temporal modeling efectively complements pixelgenerative dynamics for robust manipulation. The project page is available at https://thu-wangmx.github.io/st-wam/.

## Introduction

Leveraging world-dynamics priors acquired through largescale video pretraining, World Action Models (WAMs) (Bi et al. 2026; Kim et al. 2026) jointly model future visual states—typically in VAE latent spaces—and robot actions, providing a promising alternative to VLAs that directly map current observations to actions (Black et al. 2024; Kim et al. 2024; Physical Intelligence et al. 2025). Despite their strong performance on standard manipulation benchmarks, the robustness ofvideo-generative WAMs under visual distribution shifts remains unclear.

![](images/a0c0b93b77f8cda24073c21fc0c15afaefd86991db969069e2ed70a99964c91f.jpg)

![](images/f5ba907563d08e3f4701d45761ac9d2c643a58b5ebbed1cd5ecfbfb8b163002e.jpg)

![](images/c26d4e04ac7f6d367555c0978082f76f032c068eebe45016be874a0a17193e3c.jpg)  
Figure 1: (a) Three representative Training-Distribution Hallucination cases from video-generative WAMs under LIBERO-Plus background-texture, illumination, and camera-viewpoint shifts. (b) A representative diagnostic triplet and a comparison of DINOv3 and Wan-VAE cosine similarities across same-state frames.

To investigate this question, we analyze two representative video-generative WAMs. As illustrated in Fig. 1(a), when LingBot-VA (Li et al. 2026b) and Fast-WAM-Joint (Yuan et al. 2026), both trained only on LIBERO, are evaluated zero-shot on LIBERO-Plus, their predicted videos progressively drift toward LIBERO-style content under perturbations to background textures, illumination, and other scene characteristics. We refer to this phenomenon as Training-Distribution Hallucination: when the current observation deviates from the training distribution, the predicted future hallucinates training-domain content rather than remaining faithful to the current scene. To assess its prevalence, we manually audit the predicted futures of both models on 30 randomly sampled cases under each of three visual shifts—background, illumination, and camera viewpoint— totaling 180 predictions; 70.6% distinctly exhibit Training-Distribution Hallucination. Moreover, Fast-WAM and Fast-WAM-Joint drop from success rates of 97.6% and 98.5% on LIBERO to 51.5% and 59.0% on LIBERO-Plus, respectively (Zhang et al. 2026d). Taken together, the recurring hallucination pattern across two representative models and the substantial zero-shot performance drops reveal a clear robustness limitation of video-generative WAMs under visual distribution shifts.

These observations shift the central question from whether WAMs can predict the future to how the future should be represented for robust control. Many recent video-generative WAMs model future states in VAE latent spaces (Yuan et al. 2026; Li et al. 2026b; Ye et al. 2026b). Optimized primarily for visual reconstruction, these representations can entangle action-relevant transitions with task-irrelevant or hallucinated visual content. Recent studies have explored alternative future representations, including latent action, semantic masks, and spatial value maps (Chen et al. 2026a; Lou et al. 2026; Yu et al. 2026; Fan et al. 2026). However, existing approaches often rely on large-scale embodied pretraining, multi-stage training, or specialized pipelines for auxiliary supervision. We instead seek a semantic representation that can be extracted directly from raw observations while remaining stable under visual shifts and discriminative across task states. Large-scale self-supervised visual encoders, such as DINOv3 (Siméoni et al. 2025), provide a promising basis through their semantically structured features.

As illustrated in Fig. 1(b), we conduct a controlled representation diagnosis using 290 frame triplets from LIBERO and LIBERO-Plus. Each triplet contains two initial frames from the same task with identical robot and object states but diferent visual conditions, together with a final frame from the same LIBERO demonstration as a diferent-state reference. DINOv3 achieves an average cosine similarity of 0.904 between the same-state initial frames, compared with 0.686 for Wan-VAE latents. Moreover, when comparing the shifted initial frame with the other two frames, DINOv3 yields higher similarity to the state-matched clean frame than to the diferent-state final frame in 95.2% of triplets, versus 60.0% for Wan-VAE. DINOv3 therefore exhibits both stronger same-state stability under visual shifts and better diferent-state discriminability than Wan-VAE latents. Additional details are provided in the supplementary material.

These properties make DINOv3 suitable for two complementary temporal roles: as a future prediction target, it supervises task-relevant state evolution; as a history representation, it provides evidence of recent task progress when visual shifts make current-frame cues unreliable. Rather than correcting hallucinated future videos, we use these complementary semantic cues to improve action robustness.

In this paper, we propose Semantic-Temporal WAM (ST-WAM), an end-to-end WAM that uses DINOv3 as a shared semantic representation across two complementary temporal directions. Prospectively, Dual-Space Future Experts (DSFE) jointly model future VAE latents and DINO features in a three-branch Mixture-of-Transformers with the action expert, coupling fine-grained visual dynamics with visually stable semantic transitions. Retrospectively, Current-

Anchored Intent Retrieval (CAIR) uses the current visuallanguage context to retrieve task-relevant evidence from recent DINO history for action generation. ST-WAM requires no additional embodied pretraining, multi-stage training, or task-specific semantic annotations.

Extensive experiments across LIBERO, LIBERO-Plus, RoboTwin 2.0, and five real-world tasks demonstrate the effectiveness of ST-WAM while retaining eficient inference. Without embodied pretraining, ST-WAM achieves 98.7% on LIBERO and outperforms Fast-WAM by 21.3 percentage points under zero-shot transfer to LIBERO-Plus. It further achieves leading performance on RoboTwin 2.0 and more than doubles real-world success under visual distribution shifts from 25.8% with Fast-WAM to 61.5%.

Our main contributions are summarized as follows:

• We identify Training-Distribution Hallucination in video-generative WAMs, where predictions conditioned on visually shifted observations drift toward trainingdomain content, and provide a controlled frame-triplet diagnosis showing that, compared with VAE latents, DI-NOv3 ofers greater same-state stability under visual shifts while better distinguishing diferent task states.

• We propose ST-WAM, which unifies semantic modeling across complementary prospective and retrospective directions: Dual-Space Future Experts (DSFE) jointly model future dynamics in VAE and DINO spaces, while Current-Anchored Intent Retrieval (CAIR) extracts taskrelevant evidence from recent DINO history for action generation.

• Extensive evaluations in simulation and the real world demonstrate that, without additional embodied pretraining, ST-WAM maintains strong in-distribution performance while substantially improving robustness under visual distribution shifts.

## Related Work

## Vision-Language-Action Models

VLA models transfer semantic knowledge from pretrained vision-language models to directly map observations and instructions into robot actions (Kim et al. 2024; Black et al. 2024; Physical Intelligence et al. 2025, 2026; Yan et al. 2025; Du et al. 2026). Recent methods incorporate temporal context or future prediction: IntentVLA models short-horizon intent from recent history, while DreamVLA, VLA-JEPA, and DeFI introduce predictive objectives into VLA learning (Lian et al. 2026; Zhang et al. 2026b; Sun et al. 2026; Zhang et al. 2026c; Song et al. 2026). In contrast, we study how future and historical information should be represented within WAMs.

## Video-Generative World Action Models

World Action Models jointly model future visual states and robot actions. Representative methods include DreamZero (Ye et al. 2026b), LingBot-VA (Li et al. 2026b), and Motus (Bi et al. 2026), which couple video prediction with action generation. Fast-WAM (Yuan et al. 2026) and GigaWorld-Policy (Ye et al. 2026a) omit explicit future video generation during deployment to improve inference eficiency. Despite their diferent inference designs, their future supervision remains rooted in pixel-generative objectives, potentially entangling action-relevant dynamics with task-irrelevant visual factors.

## WAMs with Alternative Future Representations

Beyond pixel-generative objectives, recent works explore semantic or spatially structured future representations. Some methods predict semantic masks (Yu et al. 2026; Lou et al. 2026), while others model geometric-semantic cues, spatial value maps, or compact latent conditions (Fan et al. 2026; Ma et al. 2026; Su et al. 2026; Luo et al. 2026; Li et al. 2026a; Liu et al. 2026; Zhang et al. 2026a). LDA-1B (Lyu et al. 2026) and LaWAM (Chen et al. 2026a) model future states directly in DINO feature space, but rely on large-scale embodied pretraining and multi-stage training. In contrast, ST-WAM retains fine-grained VAE dynamics while incorporating both DINOv3 future supervision and DINO-based history retrieval within an end-to-end WAM, achieving competitive performance without embodied pretraining.

## Methodology

## Problem Formulation

Given a current multi-view observation $\mathbf { o } _ { t } ,$ proprioceptive state $\mathbf { s } _ { t } .$ , and language instruction ℓ, the policy predicts an action chunk $\mathbf { a } _ { t : t + H - 1 }$ . We also use a short history $\mathcal { H } _ { t }$ of M preceding observations and a future observation sequence $\mathbf { O } _ { t + 1 : t + K }$ . Let $E _ { \mathrm { V A E } }$ and $E _ { \mathrm { D I N O } }$ denote the frozen Wan2.2 VAE (Wan et al. 2025) and DINOv3 encoders, respectively. We encode the observations into two complementary spaces:

$$
\begin{array} { r } { \mathbf { z } ^ { v } = E _ { \mathrm { V A E } } ( \mathbf { o } _ { t : t + K } ) , \quad \mathbf { z } ^ { s } = E _ { \mathrm { D I N O } } ( \mathbf { o } _ { t : t + K } ) . } \end{array}\tag{1}
$$

For each $r \in \{ v , s \}$ , we partition $\mathbf { z } ^ { r }$ into current conditioning tokens $\mathbf { z } _ { \mathrm { c u r } } ^ { r }$ and future prediction targets $\mathbf { z } _ { \mathrm { f u t } } ^ { r } , \ \mathrm { i . e . , } \ \mathbf { z } ^ { r } \ =$ $[ \mathbf { z } _ { \mathrm { c u r } } ^ { r } ; \mathbf { z } _ { \mathrm { f u t } } ^ { r } ]$ . During training, ST-WAM models the following conditional joint distribution:

$$
\begin{array} { r } { p _ { \theta } \left( \mathbf { a } _ { t : t + H - 1 } , \mathbf { z } _ { \mathrm { f u t } } ^ { v } , \mathbf { z } _ { \mathrm { f u t } } ^ { s } \mid \mathbf { o } _ { t } , \mathbf { s } _ { t } , \boldsymbol { \ell } , \mathcal { H } _ { t } \right) . } \end{array}\tag{2}
$$

At inference, ST-WAM reduces to an action-only policy for eficient deployment:

$$
\mathbf { a } _ { t : t + H - 1 } \sim \pi _ { \boldsymbol { \theta } } \big ( \cdot | \mathbf { o } _ { t } , \mathbf { s } _ { t } , \boldsymbol { \ell } , \mathcal { H } _ { t } \big ) .\tag{3}
$$

## Dual-Space Future Experts

Unified Visual–Semantic Future Modeling. DSFE models future states in complementary VAE-latent and DINOfeature spaces. The frozen VAE of Wan2.2-TI2V-5B (Wan et al. 2025) encodes the observation sequence into visual latents $\mathbf { z } ^ { v }$ , while a frozen DINOv3 encoder (Siméoni et al. 2025) extracts dense, frame-wise semantic features $\mathbf { z } ^ { s }$ . After modality-specific embedding, the pretrained Wan2.2 Video DiT models $\mathbf { z } _ { \mathrm { f u t } } ^ { v }$ to preserve the fine-grained visual dynamics inherited from video pretraining, whereas a semantic future DiT models $\mathbf { z } _ { \mathrm { f u t } } ^ { s }$ to learn semantic state transitions.

Three-Branch Mixture-of-Transformers. As shown in Fig. 2(a), the visual and semantic future DiTs, together with the action DiT, form a three-branch Mixture-of-Transformers (Liang et al. 2024). Each branch retains its own parameters and prediction head, while layer-wise mixed attention enables mutual refinement between the two future spaces and allows the action branch to integrate current evidence from both. During flow-matching training, the three experts jointly denoise future VAE latents, future DINO features, and action tokens, with branch-specific heads estimating flow velocities in their respective spaces.

Structured Cross-Branch Attention Mask. As illustrated in Fig. 3, we apply an asymmetric mask at each mixedattention layer. Clean current VAE and DINO tokens interact within and across their spaces but cannot read future or action tokens, forming leakage-free anchors. The two noisy future streams attend to both current anchors and each other, enabling mutual refinement across the two future spaces while remaining isolated from action tokens. Action tokens attend to both current streams and themselves but cannot access either future stream. This routing prevents future-target leakage into action generation and action-token interference with future modeling, while allowing the future streams to be omitted at inference for eficient deployment.

## Current-Anchored Intent Retrieval

Current-Anchored Semantic Queries. As shown in Fig. 2(b), given the current observation $\mathbf { o } _ { t }$ and language instruction ℓ, a frozen Qwen3-VL model extracts its final-layer multimodal hidden states:

$$
\mathbf { H } _ { t } ^ { q } = E _ { \mathrm { Q w e n } } ( \mathbf { o } _ { t } , \boldsymbol { \ell } ) \in \mathbb { R } ^ { B \times L _ { q } \times d _ { q } } ,\tag{4}
$$

where $L _ { q }$ is the sequence length and $d _ { q }$ is the hidden dimension of Qwen3-VL. A bank of $N _ { I }$ learnable queries $\mathbf { Q } .$ each with dimension $d _ { r }$ , attends to the Qwen3-VL features to produce a compact set of current semantic tokens:

$$
{ \bf U } _ { t } ^ { 0 } = \mathrm { M H A } \left( { \bf Q } , { \cal P } _ { q } { \bf H } _ { t } ^ { q } , { \cal P } _ { q } { \bf H } _ { t } ^ { q } \right) .\tag{5}
$$

Here, $\mathbf { Q }$ serves as the query, while the projected Qwen3-VL features serve as keys and values. $P _ { q }$ is a learnable linear projection from the Qwen3-VL hidden dimension $d _ { q }$ to the hidden dimension $d _ { r } ,$ and $\mathbf { U } _ { t } ^ { 0 } \ \in \ \mathbb { R } ^ { B \times N _ { I } \times d _ { \tau } }$ <sup>r</sup> denotes the current semantic tokens before history fusion. These tokens jointly encode the current scene and instruction, serving as semantic anchors that guide subsequent intent retrieval.

Semantic History Retrieval. For each observation in the short history $\mathcal { H } _ { t }$ , the frozen DINOv3 encoder extracts dense patch features, providing a visually stable semantic representation from which task-relevant historical evidence can be retrieved. The features from all M observations are linearly projected, augmented with learnable temporal embeddings, and concatenated into a history-token sequence $\mathbf { R } _ { t }$

Starting from $\mathbf { U } _ { t } ^ { 0 }$ , CAIR applies L cross-attention blocks, using the current semantic tokens as queries and $\mathbf { R } _ { t }$ as keys and values. This current-conditioned interaction selectively retrieves historical evidence relevant to the present task state while reducing sensitivity to task-irrelevant visual variations.

![](images/9a03d3e74d7b5aeec400e1a176228b3aa24c92107b9f74912e7aa10f1602bb01.jpg)

![](images/35e9b0c0148a0ead80defb5a9d2d7a7e22f81368ef31129d2fcee136174f424c.jpg)  
Figure 2: Overview of ST-WAM. (a): Dual-Space Future Experts (DSFE) jointly model future dynamics in the VAE visuallatent space and the DINOv3 semantic space. The visual and semantic future experts interact with the action expert through a three-branch Mixture-of-Transformers. (b): Current-Anchored Intent Retrieval (CAIR).

![](images/a1cf13ea8be322a9e9425de0ae7cffb048eb298e7c6b8cf63d68cbaee83a3ea6.jpg)  
Figure 3: Structured cross-branch attention masks during training and inference.

After L blocks, the refined tokens $\mathbf { U } _ { t } ^ { L }$ are projected into the action-context space:

$$
\mathbf { I } _ { t } = P _ { o } \left( \mathbf { U } _ { t } ^ { L } \right) \in \mathbb { R } ^ { B \times N _ { I } \times d _ { c } } ,\tag{6}
$$

where $P _ { o }$ is the output projection and $d _ { c }$ is the context dimension of the action expert. We refer to $\mathbf { I } _ { t }$ as short-horizon intent tokens: a latent, label-free summary of recent task progress relevant to current action decision, rather than an explicitly supervised variable.

For expert conditioning, the frozen T5 encoder maps the language instruction into the shared language context $\mathbf { C } _ { \ell } =$ $E _ { \mathrm { T 5 } } ( \ell )$ . After appending the projected proprioceptive token $P _ { p } ( \mathbf { s } _ { t } )$ , where $P _ { p }$ is a learnable projection, the contexts of the three experts are

$$
\mathbf { C } _ { v } = \mathbf { C } _ { s } = [ \mathbf { C } _ { \ell } ; P _ { p } ( \mathbf { s } _ { t } ) ] , \qquad \mathbf { C } _ { a } = [ \mathbf { C } _ { \ell } ; P _ { p } ( \mathbf { s } _ { t } ) ; \mathbf { I } _ { t } ] .\tag{7}
$$

Each expert receives its corresponding context through crossattention at every DiT block. Thus, the short-horizon intent tokens are injected only into the action expert and optimized end-to-end through action flow matching, allowing semantic evidence from history to guide action generation without altering the conditioning contexts of the DSFE branches.

Together, DSFE and CAIR use DINOv3 in complementary temporal directions: DSFE prospectively supervises future semantic dynamics, whereas CAIR retrospectively retrieves task-relevant evidence from recent semantic history for robust action generation under visual distribution shifts.

## Joint Flow-Matching Objective

We jointly train the visual future, semantic future, and action experts using flow matching (Lipman et al. 2022). Let the clean branch targets be ${ \bf x } ^ { v } = { \bf z } _ { \mathrm { f u t } } ^ { v } , { \bf x } ^ { s } = { \bf z } _ { \mathrm { f u t } } ^ { s } ,$ and $\mathbf { x } ^ { a } = \mathbf { a } _ { t : t + H - 1 } .$ . For each branch $r \in \{ v , s , a \}$ , we sample a timestep $\tau _ { r } \in [ 0 , 1 ]$ and Gaussian noise $\epsilon ^ { r } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ The noisy input is constructed through linear interpolation:

$$
\mathbf { x } _ { \tau _ { r } } ^ { r } = ( 1 - \tau _ { r } ) \mathbf { x } ^ { r } + \tau _ { r } \mathbf { \epsilon } ^ { r } .\tag{8}
$$

Here, $\tau _ { r } = 0$ corresponds to the clean target and $\tau _ { r } = 1$ to pure noise. The corresponding target velocity is

$$
\mathbf { u } ^ { r } = \epsilon ^ { r } - \mathbf { x } ^ { r } .\tag{9}
$$

Because the two future experts exchange information through mixed attention, we set $\tau _ { v } = \tau _ { s } = \tau _ { f }$ to synchronize their denoising stages, while sampling $\tau _ { a }$ independently from the same timestep distribution. Gaussian noises remain independent across all three branches, and action denoising is therefore decoupled from the two training-only future flows.

The clean current tokens, noisy future tokens, and noisy action tokens are processed in a single MoT forward pass under the structured attention mask, producing branch-specific velocity estimates $\widehat { \mathbf { u } } _ { \theta } ^ { r }$ . For each branch, we optimize

$$
\begin{array} { r } { \mathcal { L } _ { r } = \mathbb { E } \left[ w ( \tau _ { r } ) \lVert \widehat { \mathbf { u } } _ { \theta } ^ { r } - \mathbf { u } ^ { r } \rVert _ { 2 } ^ { 2 } \right] , \qquad r \in \{ v , s , a \} , } \end{array}\tag{10}
$$

where $w ( \tau _ { r } )$ denotes the timestep-dependent weighting of the flow scheduler. The overall training objective is

$$
\mathcal { L } = \lambda _ { v } \mathcal { L } _ { v } + \lambda _ { s } \mathcal { L } _ { s } + \lambda _ { a } \mathcal { L } _ { a } ,\tag{11}
$$

where $\mathcal { L } _ { v } , \mathcal { L } _ { s } .$ , and $\mathcal { L } _ { a }$ denote the visual, semantic, and action expert losses, respectively, and $\lambda _ { v } , \lambda _ { s } ,$ and $\lambda _ { a }$ are their corresponding loss weights.

## Experiments

## Experimental Setup

Benchmarks and Protocols. We evaluate in-distribution manipulation performance on the four LIBERO suites (Liu et al. 2023): Spatial, Object, Goal, and Long, covering 40 tasks with 50 evaluation rollouts per task. For out-ofdistribution evaluation, we directly evaluate the LIBEROtrained policy on LIBERO-Plus (Fei et al. 2025) without finetuning, which comprises 10,030 test cases spanning seven perturbation dimensions. We additionally evaluate bimanual manipulation on RoboTwin 2.0 (Chen et al. 2025) and ST-WAM is trained on a mixture of 2,500 clean and 25,000 heavily randomized demonstrations and each task is evaluated over 100 trials in both clean and randomized settings.

Baselines. We compare ST-WAM against a broad range of representative methods. These include VLAs such as $\pi _ { 0 }$ (Black et al. 2024), $\pi _ { 0 . 5 }$ (Physical Intelligence et al. 2025); video-generative WAMs such as Fast-WAM (Yuan et al. 2026), Motus (Bi et al. 2026), and LingBot-VA (Li et al. 2026b); and methods exploring alternative future representations, such as LaWAM (Chen et al. 2026a), Mask World Model (Lou et al. 2026), MaskWAM (Yu et al. 2026).

Real-World Evaluation. We evaluate ST-WAM on an Agilex Piper 6-DoF single-arm robot using 50 demonstrations per task under fixed nominal visual conditions. We consider five tasks with diverse temporal and geometric requirements: (1) Arrange Flowers, inserting three bouquets into a vase; (2) Drawer Organization, opening a drawer, placing a pen inside, and closing it; (3) Bean Scooping, transferring beans from a plate to a bowl and returning the spoon; (4) Arrange Fruits, placing five fruits into a basket; and (5) Hang Mug, hanging a mug on a rack. We compare ST-WAM with $\pi _ { 0 }$ (Black et al. 2024) and Fast-WAM (Yuan et al. 2026), with all methods post-trained separately for each task using the same demonstrations. We test them under nominal and four visual-shift conditions without further fine-tuning: Background, which replaces the tabletop texture with unseen patterns; Lighting, which changes the illumination intensity; Object Appearance, which changes object colors or instances while preserving their geometry and task function; and Compound, which applies all three shifts simultaneously. Each method is evaluated over 30 trials per task and condition using predefined object initializations to ensure fair comparison.

Implementation Details. The visual future expert uses the 5B Video DiT of Wan2.2-TI2V-5B (Wan et al. 2025), while the semantic and action experts use 1B DiTs initialized from the pretrained Wan2.2 weights. The Wan2.2 VAE and T5 encoder, DINOv3 ViT-S/16 visual encoder (Siméoni et al. 2025), and Qwen3-VL-4B-Instruct (Bai et al. 2025) remain frozen throughout training.

We set the action horizon to H = 32 and the future horizon to $K = 8$ , with future observations sampled every four control steps. For CAIR, we set M = 4 and construct $\mathcal { H } _ { t }$ from observations at frame indices $\{ t - 2 4 , t - 1 6 , t - 8 , t - 1 \}$ The retrieval uses L = 2 cross-attention blocks and $N _ { I } = 8$ learnable queries, producing eight intent tokens.

<table><tr><td>Method</td><td>Emb. PT. Spat. Obj. Goal Long Avg.</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="5">Vision-Language-Action Models</td></tr><tr><td>π0 (Black et al. 2024)</td><td>Yes</td><td>98.0</td><td>96.8</td><td>94.4</td><td>88.494.4</td></tr><tr><td>π0.5 (Physical Intelligence et al. 2025)</td><td>Yes</td><td>98.8 98.2 99.7</td><td>98.0 98.1</td><td></td><td> $9 2 . 4 9 6 . 9 $  97.4 98.6</td></tr><tr><td colspan="6">IntentVLA (Lian et al. 2026) No 99.3</td></tr><tr><td colspan="6">Video-Generative World Action Models</td></tr><tr><td>Fast-WAM (Yuan et al. 2026)</td><td>No</td><td>98.2</td><td>100.0 97.0</td><td>95.2</td><td>97.6</td></tr><tr><td>Motus (Bi et al. 2026) LingBot-VA (Li et al. 2026b)</td><td>Yes</td><td>96.8</td><td>99.8</td><td>96.6</td><td>97.6 97.7</td></tr><tr><td></td><td>Yes</td><td>98.5</td><td>99.6</td><td>97.2</td><td>98.5 98.5</td></tr><tr><td colspan="6">WAMs with Alternative Future Representations</td></tr><tr><td>Mask World Model (Lou et al. 2026)</td><td>No</td><td>98.8 100.0</td><td>98.2</td><td>96.0</td><td>98.3</td></tr><tr><td>MaskWAM (Yu et al. 2026)</td><td>No</td><td>98.8 100.0</td><td>98.2</td><td>96.4</td><td>98.4</td></tr><tr><td>GeoSem-WAM (Ma et al. 2026)</td><td>No</td><td>99.0</td><td>100.0 98.2</td><td>97.0</td><td>98.6</td></tr><tr><td>LaWAM (Chen et al. 2026a)</td><td>Yes</td><td>99.4</td><td>99.6 98.4</td><td>97.0</td><td>98.6</td></tr><tr><td>ST-WAM (Ours)</td><td>No</td><td>99.0</td><td>100.0 99.0</td><td>96.8</td><td>98.7</td></tr></table>

Table 1: Success rates (%) on the four LIBERO suites. Emb. PT. indicates large-scale pretraining on robot trajectories or embodied videos before LIBERO adaptation.

<table><tr><td>Method</td><td>Emb. PT.</td><td>Clean</td><td>Random</td><td> $\operatorname { A v g } .$ </td></tr><tr><td> $\pi _ { 0 }$  (Black et al. 2024)</td><td>Yes</td><td>65.92</td><td>58.40</td><td>62.16</td></tr><tr><td>π0.5 (Physical Intelligence et al. 2025)</td><td>Yes</td><td>82.74</td><td>76.76</td><td>79.75</td></tr><tr><td>GigaWorld-Policy (Ye et al. 2026a)</td><td>Yes</td><td>87.00</td><td>85.00</td><td>86.00</td></tr><tr><td>Motus (Bi et al. 2026)</td><td>Yes</td><td>88.66</td><td>87.02</td><td>87.84</td></tr><tr><td>LaWAM (Chen et al. 2026a)</td><td>Yes</td><td>92.64</td><td>89.80</td><td>91.22</td></tr><tr><td>Fast-WAM (Yuan et al. 2026)</td><td>No</td><td>91.88</td><td>91.78</td><td>91.83</td></tr><tr><td>LingBot-VA (Li et al. 2026b)</td><td>Yes</td><td>92.90</td><td>91.50</td><td>92.20</td></tr><tr><td>GeoSem-WAM (Ma et al. 2026)</td><td>No</td><td>92.94</td><td>92.14</td><td>92.54</td></tr><tr><td>ST-WAM (Ours)</td><td>No</td><td>93.06</td><td>92.48</td><td>92.77</td></tr></table>

Table 2: Success rates (%) on RoboTwin 2.0 under the standard mixed clean-and-randomized training setting.

We train on LIBERO and RoboTwin 2.0 for 10 and 5 epochs with global batch sizes of 128 and 1,024, respectively. We use AdamW $( \mathrm { l r } = 1 \times 1 0 ^ { - 4 }$ , weight decay 0.01), cosine decay, BF16 precision, gradient clipping at 1.0, and a shifted flow-matching schedule with shift 5.0; the loss weights are $( \lambda _ { v } , \lambda _ { s } , \lambda _ { a } ) \overset { \smile } { = } ( 1 . 0 , 0 . 0 2 , 1 . 0 )$ . At inference, we use 10 flowintegration steps and execute 10 actions before replanning; additional details are provided in the supplementary material.

## Main Results

Performance on LIBERO. As shown in Table 1, ST-WAM achieves an average success rate of 98.7%, the highest among the compared methods. Without embodied pretraining, it outperforms recent strong WAMs such as Motus (97.7%) (Bi et al. 2026) and LingBot-VA (98.5%) (Li et al. 2026b). Thus, ST-WAM further improves in-distribution performance.

Performance on RoboTwin 2.0. As shown in Table 2, ST-WAM achieves success rates of 93.06% and 92.48% in the clean and randomized settings, respectively, yielding the highest average success rate of 92.77% among the compared methods. Without embodied pretraining, ST-WAM outperforms both the embodied-pretrained LingBot-VA (Li et al.

<table><tr><td>Method</td><td>Emb. PT.</td><td>Camera</td><td>Robot</td><td>Lang.</td><td>Light</td><td>BG</td><td>Noise</td><td>Layout</td><td>Overall</td></tr><tr><td>UniVLA (Bu et al. 2025)</td><td>Yes</td><td>1.8</td><td>46.2</td><td>69.6</td><td>69.0</td><td>81.0</td><td>21.2</td><td>31.9</td><td>42.9</td></tr><tr><td>π0 (Black et al. 2024)</td><td>Yes</td><td>13.8</td><td>6.0</td><td>58.8</td><td>85.0</td><td>81.4</td><td>79.0</td><td>68.9</td><td>53.6</td></tr><tr><td>π0-FAST (Pertsch et al. 2025)</td><td>Yes</td><td>65.1</td><td>21.6</td><td>61.0</td><td>73.2</td><td>73.2</td><td>74.4</td><td>68.8</td><td>61.6</td></tr><tr><td>RIPT-VLA (OFT) (Tan et al. 2025)</td><td>Yes</td><td>55.2</td><td>31.2</td><td>77.6</td><td>88.4</td><td>91.6</td><td>73.5</td><td>74.2</td><td>68.4</td></tr><tr><td>OpenVLA-OFT (Kim, Finn, and Liang 2025)</td><td>Yes</td><td>56.4</td><td>31.9</td><td>79.5</td><td>88.7</td><td>93.3</td><td>75.8</td><td>74.2</td><td>69.6</td></tr><tr><td>X-VLA (Zheng et al. 2025)</td><td>Yes</td><td>23.4</td><td>89.7</td><td>75.7</td><td>88.2</td><td>96.0</td><td>62.7</td><td>71.8</td><td>71.4</td></tr><tr><td>Fast-WAM (Yuan et al. 2026)</td><td>No</td><td>16.4</td><td>44.5</td><td>68.9</td><td>78.2</td><td>53.7</td><td>37.7</td><td>60.7</td><td>51.5</td></tr><tr><td>Fast-WAM-Joint (Yuan et al. 2026)</td><td>No</td><td>34.0</td><td>55.1</td><td>88.9</td><td>90.0</td><td>44.9</td><td>33.3</td><td>73.4</td><td>59.0</td></tr><tr><td>ST-WAM (Ours)</td><td>No</td><td>55.4</td><td>60.1</td><td>79.3</td><td>93.0</td><td>74.2</td><td>79.5</td><td>74.3</td><td>72.8</td></tr></table>

Table 3: Zero-shot success rates (%) on LIBERO-Plus. Baseline results from (Zhang et al. 2026d; Chen et al. 2026b).

2026b) (92.20%) and the closely related Fast-WAM (Yuan et al. 2026) (91.83%). These consistent results demonstrate the efectiveness of ST-WAM for bimanual manipulation across clean and randomized environments.

Zero-Shot Generalization on LIBERO-Plus. As shown in Table 3, ST-WAM achieves an overall success rate of 72.8%. Without embodied pretraining, it surpasses several embodied-pretrained VLAs, including OpenVLA-OFT (69.6%), RIPT-VLA (68.4%), and X-VLA (71.4%). More importantly, ST-WAM improves the closely matched Fast-WAM baseline (Yuan et al. 2026) from 51.5% to 72.8%, a gain of 21.3 percentage points. This improvement is consistent across all seven perturbation categories, with particularly large gains of 39.0 and 41.8 percentage points under camera and sensor-noise perturbations, respectively. It also outperforms Fast-WAM-Joint on all six non-language perturbations. Together, these results suggest that complementing fine-grained VAE dynamics with DINO-based semantic supervision reduces reliance on low-level visual cues and substantially improves out-of-distribution robustness.

Inference Eficiency. On RoboTwin 2.0, we benchmark the complete action-chunk inference call, including all model components active at deployment, on a single NVIDIA A100-80GB GPU using BF16 precision and 10 flowintegration steps. Averaged over 20 synchronized runs, ST-WAM generates a 32-step action chunk in 756.17 ms, compared with 609.30 ms for Fast-WAM (Yuan et al. 2026). This 1.24× latency represents a moderate overhead for improved robustness while retaining sub-second inference.

Real-World Generalization. As shown in Fig. 4 and Table 4, ST-WAM achieves 79.3% average success under the nominal condition, outperforming Fast-WAM and π<sub>0</sub> by 14.6 and 32.0 percentage points, respectively. Under visual distribution shifts, ST-WAM achieves 61.5%, surpassing Fast-WAM and π<sub>0</sub> by 35.7 and 28.7 points, respectively, and retaining a substantial advantage under the compound shift (48.0% vs. 15.3%). Notably, Fast-WAM drops by 38.9 points from the nominal to shifted conditions, compared with only 17.8 points for ST-WAM, suggesting that pixel-generative future representations are particularly sensitive to visual distribution shifts. Removing the semantic future expert or CAIR reduces shifted-condition performance to 41.0% and 43.7%, respectively, confirming their complementary contributions to real-world robustness. Task-wise results under visual shifts are provided in the supplementary material.

![](images/e3f72f975f2565a539dc2267e5d4c3ad383eb7a521616a4d00cbd3f78bfeabbb.jpg)  
Figure 4: Real-world evaluation of ST-WAM on five tasks with diverse temporal and geometric requirements.

## Ablation and Qualitative Analysis

Beyond the component-level real-world ablations, we conduct finer-grained controlled studies on LIBERO and LIBERO-Plus, as shown in Table 5, to disentangle the design choices underlying DSFE and CAIR. Specifically, we address three critical questions:

Q1: Are VAE and DINO future representations complementary or interchangeable? We first compare diferent future representation spaces without intent conditioning. The DINO Future Only variant achieves 39.7% on LIBERO-Plus, below the 51.5% of the VAE-based Fast-WAM. In contrast, jointly modeling future states in the VAE and DINO spaces improves the success rate to 66.4% even without CAIR. These results indicate that DINO semantics cannot directly replace VAE dynamics: VAE latents preserve fine-grained visual and motion information, whereas DINO features provide complementary object- and state-level semantics.

Q2: Is explicit semantic future prediction necessary? Keeping CAIR fixed, w/o Semantic Future Expert removes the semantic stream and achieves 63.5% on LIBERO-Plus. The parameter-matched Semantic Expert w/o Future Obj. retains the semantic DiT, current-DINO conditioning, and mixed-attention interactions, but removes its future target and loss, yielding 62.9%. In contrast, the full model reaches

<table><tr><td></td><td colspan="6">Nominal Environment</td><td colspan="5">Visual Distribution Shifts</td></tr><tr><td>Method</td><td>Flower</td><td>Drawer</td><td>Scoop</td><td>Fruit</td><td>Hang</td><td>Avg.</td><td>BG</td><td>Light</td><td>Obj. App.</td><td>Comp.</td><td>Avg.</td></tr><tr><td>π0 (Black et al. 2024)</td><td>56.7</td><td>46.7</td><td>30.0</td><td>46.7</td><td>56.7</td><td>47.3</td><td>33.3</td><td>40.7</td><td>35.3</td><td>22.0</td><td>32.8</td></tr><tr><td>Fast-WAM (Yuan et al. 2026)</td><td>70.0</td><td>66.7</td><td>46.7</td><td>66.7</td><td>73.3</td><td>64.7</td><td>27.3</td><td>35.3</td><td>25.3</td><td>15.3</td><td>25.8</td></tr><tr><td>w/o Semantic Future Expert</td><td>76.7</td><td>73.3</td><td>56.7</td><td>73.3</td><td>80.0</td><td>72.0</td><td>43.3</td><td>50.0</td><td>41.3</td><td>29.3</td><td>41.0</td></tr><tr><td>w/o CAIR</td><td>80.0</td><td>76.7</td><td>60.0</td><td>76.7</td><td>83.3</td><td>75.3</td><td>46.0</td><td>52.7</td><td>44.0</td><td>32.0</td><td>43.7</td></tr><tr><td>ST-WAM (Ours)</td><td>86.7</td><td>80.0</td><td>66.7</td><td>76.7</td><td>86.7</td><td>79.3</td><td>66.0</td><td>70.0</td><td>62.0</td><td>48.0</td><td>61.5</td></tr></table>

Table 4: Real-world success rates (%) under the nominal condition and visual distribution shifts. Results for each visual shift are averaged across all five tasks. “Comp.” combines background, lighting, and object-appearance shifts.
<table><tr><td>Variant</td><td>Future Prediction</td><td>Intent Conditioning</td><td>LIBERO</td><td>LIBERO-Plus</td></tr><tr><td>Fast-WAM (Yuan et al. 2026)</td><td>VAE</td><td>None</td><td>97.6</td><td>51.5</td></tr><tr><td>DINO Future Only</td><td>DINO</td><td>None</td><td>96.3</td><td>39.7</td></tr><tr><td>Dual-Space w/o CAIR</td><td>VAE + DINO</td><td>None</td><td>97.8</td><td>66.4</td></tr><tr><td>w/o Semantic Future Expert</td><td>VAE</td><td>CAIR (DINO history)</td><td>97.3</td><td>63.5</td></tr><tr><td>Semantic Expert w/o Future Obj.</td><td>VAE</td><td>CAIR (DINO history)</td><td>95.8</td><td>62.9</td></tr><tr><td>Naive History Retrieval</td><td>VAE + DINO</td><td>Unanchored DINO history</td><td>96.3</td><td>56.5</td></tr><tr><td>Qwen Current Only</td><td>VAE + DINO</td><td>Qwen current</td><td>96.5</td><td>62.3</td></tr><tr><td>CAIR with VAE History</td><td>VAE + DINO</td><td>CAIR (VAE history)</td><td>96.3</td><td>64.7</td></tr><tr><td>ST-WAM (Ours)</td><td>VAE + DINO</td><td>CAIR (DINO history)</td><td>98.7</td><td>72.8</td></tr></table>

Table 5: Ablation results on LIBERO and LIBERO-Plus. “Intent Conditioning” denotes the alternative action-expert contexts used to ablate the design choices of CAIR.

72.8%, showing that the gain cannot be explained by current DINO features or additional model capacity alone, but requires explicit future-semantic prediction.

![](images/9878e07c003035b6675f16a9c606a33f0a4824c89b5334c1fb2be3c421c0f2be.jpg)  
Figure 5: Attention heatmaps on two representative tasks. Warmer colors indicate stronger relative attention within each map.

Q3: How should short-horizon history be represented and retrieved? The Naive History Retrieval variant compresses DINO history with unanchored learnable queries and achieves only 56.5% on LIBERO-Plus, versus 72.8% for the full model, demonstrating the necessity of a current semantic anchor. Removing history entirely in Qwen Current Only yields 62.3%, ruling out VLM-derived current-frame semantics alone as the source of improvement. Replacing DINO history with Wan-VAE latents in CAIR with VAE History, while retaining the same Qwen anchor and retrieval architecture, obtains 64.7%. All three alternative conditioning schemes underperform the Dual-Space w/o CAIR baseline (66.4%), indicating that improperly represented or retrieved context can be detrimental; only current-anchored retrieval from DINO history surpasses this baseline, reaching 72.8%. Together, these results show that efective intent modeling requires both a current visual-language anchor and a visually stable DINO representation of recent history. Detailed results for each subset are provided in the supplementary material.

Cross-Branch Attention Visualization. Fig. 5 visualizes the attention from action queries to the current VAE and DINO tokens in the MoT mixed self-attention. Across both tasks, action-to-DINO attention aligns more closely with the manipulated objects and interaction regions, whereas actionto-VAE attention is distributed over broader scene areas. This qualitative pattern suggests that DINO provides task-focused semantic cues complementary to the fine-grained visual dynamics represented by VAE latents.

## Conclusion

Motivated by the robustness limitations exposed by Training-Distribution Hallucination, we introduced ST-WAM to improve video-generative World Action Models under visual distribution shifts. ST-WAM uses DINOv3 in complementary temporal directions: DSFE complements finegrained VAE dynamics with visually stable future semantics, while CAIR retrieves task-relevant intent from recent semantic history under the current visual-language context. Extensive simulation and real-world experiments demonstrate substantially improved robustness while preserving strong in-distribution performance, eficient action-only inference, and freedom from large-scale embodied pretraining. Overall, our results highlight semantic-temporal modeling as an efective direction for robust world-action learning beyond pixel-centric futures. Future work will extend ST-WAM beyond visual distribution shifts to changes in physical dynamics and embodiments.

## References

Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; Cheng, Z.; Deng, L.; Ding, W.; Gao, C.; Ge, C.; et al. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Bi, H.; Tan, H.; Xie, S.; Wang, Z.; Huang, S.; Liu, H.; Zhao, R.; Feng, Y.; Xiang, C.; Rong, Y.; et al. 2026. Motus: A unified latent action world model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 35101–35113.

Black, K.; Brown, N.; Driess, D.; Esmail, A.; Equi, M.; Finn, C.; Fusai, N.; Groom, L.; Hausman, K.; Ichter, B.; et al. 2024. π : A Vision-Language-Action Flow Model for General Robot Control. arXiv preprint arXiv:2410.24164.

Bu, Q.; Yang, Y.; Cai, J.; Gao, S.; Ren, G.; Yao, M.; Luo, P.; and Li, H. 2025. Univla: Learning to act anywhere with task-centric latent actions. arXiv preprint arXiv:2505.06111.

Chen, J.; Wang, K.; Chen, K.; Chen, S.; Gao, F.; Tang, W.; Li, Z.; Liu, W.; Yao, Z.; Li, B.; et al. 2026a. Lawam: Latent world action models for eficient dynamics-aware robot policies. arXiv preprint arXiv:2606.15768.

Chen, R.; Yang, Y.; Tang, Z.; Huo, D.; Lin, T.; Wu, H.; Liu, H.; Chen, Y.; Zheng, L.; Yuan, B.; et al. 2026b. Abot-m0. 5: Unified mobility-and-manipulation world action model. arXiv preprint arXiv:2607.00678.

Chen, T.; Chen, Z.; Chen, B.; Cai, Z.; Liu, Y.; Li, Z.; Liang, Q.; Lin, X.; Ge, Y.; Gu, Z.; et al. 2025. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088.

Du, F.; Yan, F.; Wu, J.; Xu, X.; Zhang, W.; Wang, W.; Guo, Y.; Qian, B.; He, Z.; Wang, F.; and Yang, H. 2026. CF-VLA: Eficient Coarse-to-Fine Action Generation for Vision-Language-Action Policies. arXiv:2604.24622.

Fan, L.; Xu, Z.; Cao, C.; Zhang, W.; Yuan, M.; and Chen, J. 2026. Aim: Intent-aware unified world action modeling with spatial value maps. arXiv preprint arXiv:2604.11135.

Fei, S.; Wang, S.; Shi, J.; Dai, Z.; Cai, J.; Qian, P.; Ji, L.; He, X.; Zhang, S.; Fei, Z.; et al. 2025. Libero-plus: In-depth robustness analysis of vision-language-action models. arXiv preprint arXiv:2510.13626.

Kim, M. J.; Finn, C.; and Liang, P. 2025. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645.

Kim, M. J.; Gao, Y.; Lin, T.-Y.; Lin, Y.-C.; Ge, Y.; Lam, G.; Liang, P.; Song, S.; Liu, M.-Y.; Finn, C.; et al. 2026. Cosmos policy: Fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163.

Kim, M. J.; Pertsch, K.; Karamcheti, S.; Xiao, T.; Balakrishna, A.; Nair, S.; Rafailov, R.; Foster, E.; Lam, G.; Sanketi, P.; et al. 2024. Openvla: An open-source vision-languageaction model. arXiv preprint arXiv:2406.09246.

Li, B.; Yin, X.; Lin, M.; Zhang, Y.; and Xu, D. 2026a. EgoWAM: World Action Models Beyond Pixels with Inthe-Wild Egocentric Human Data. In Robot World Models.

Li, L.; Zhang, Q.; Luo, Y.; Yang, S.; Wang, R.; Han, F.; Yu, M.; Gao, Z.; Xue, N.; Zhu, X.; et al. 2026b. Causal World Modeling for Robot Control. arXiv preprint arXiv:2601.21998.

Lian, S.; Yu, B.; Lin, X.; Shen, Z.; Yang, L. T.; Jin, Y.; Liu, H.; Wu, C.; Yuan, H.; Huang, C.; et al. 2026. IntentVLA: Short-Horizon Intent Modeling for Aliased Robot Manipulation. arXiv preprint arXiv:2605.14712.

Liang, W.; Yu, L.; Luo, L.; Iyer, S.; Dong, N.; Zhou, C.; Ghosh, G.; Lewis, M.; Yih, W.-t.; Zettlemoyer, L.; et al. 2024. Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models. arXiv preprint arXiv:2411.04996.

Lipman, Y.; Chen, R. T.; Ben-Hamu, H.; Nickel, M.; and Le, M. 2022. Flow matching for generative modeling. In The eleventh international conference on learning representations.

Liu, B.; Zhu, Y.; Gao, C.; Feng, Y.; Liu, Q.; Zhu, Y.; and Stone, P. 2023. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36: 44776–44791.

Liu, Y.; Sun, P.; Li, S.; Xie, Y.; Zhang, L.; Chao, X.; Dong, S.; Chen, F.; Zhang, X.-P.; and Ding, W. 2026. Oa-wam: Object-addressable world action model for robust robot manipulation. arXiv preprint arXiv:2605.06481.

Lou, Y.; Chi, X.; Zhang, X.; Qian, Z.; Li, C.; Zhang, R.; Lyu, Y.; Song, G.; Fu, C.; Xu, H.; et al. 2026. Mask World Model: Predicting What Matters for Robust Robot Policy Learning. arXiv preprint arXiv:2604.19683.

Luo, H.; Zhang, W.; Feng, Y.; Zheng, S.; Xu, H.; Xu, C.; Xi, Z.; Fu, Y.; and Lu, Z. 2026. Being-h0. 7: A latent world-action model from egocentric videos. arXiv preprint arXiv:2605.00078.

Lyu, J.; Liu, K.; Zhang, X.; Liao, H.; Feng, Y.; Zhu, W.; Shen, T.; Chen, J.; Zhang, J.; Dong, Y.; et al. 2026. Lda-1b: Scaling latent dynamics action model via universal embodied data ingestion. arXiv preprint arXiv:2602.12215.

Ma, F.; Peng, D.; Yue, W.; Cao, J.; Wang, B.; Zhang, Q.; and Ma, J. 2026. GeoSem-WAM: Geometry-and Semantic-Aware World Action Models. arXiv preprint arXiv:2606.03188.

Pertsch, K.; Stachowicz, K.; Ichter, B.; Driess, D.; Nair, S.; Vuong, Q.; Mees, O.; Finn, C.; and Levine, S. 2025. Fast: Eficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747.

Physical Intelligence; Ai, B.; Amin, A.; Aniceto, R.; Balakrishna, A.; Balke, G.; Black, K.; Bokinsky, G.; Cao, S.; Charbonnier, T.; et al. 2026. π : A Steerable Generalist Robotic Foundation Model with Emergent Capabilities. arXiv preprint arXiv:2604.15483.

Physical Intelligence; Black, K.; Brown, N.; Darpinian, J.; Dhabalia, K.; Driess, D.; Esmail, A.; Equi, M.; Finn, C.; Fusai, N.; et al. 2025. π : A Vision-Language-Action Model with Open-World Generalization. arXiv preprint arXiv:2504.16054.

Siméoni, O.; Vo, H. V.; Seitzer, M.; Baldassarre, F.; Oquab, M.; Jose, C.; Khalidov, V.; Szafraniec, M.; Yi, S.; Ramamonjisoa, M.; et al. 2025. Dinov3. arXiv preprint arXiv:2508.10104.

Song, W.; Zhou, Z.; Zhao, H.; Chen, J.; Ding, P.; Yan, H.; Huang, Y.; Tang, F.; Wang, D.; and Li, H. 2026. Reconvla: Reconstructive vision-language-action model as efective robot perceiver. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 18549–18557.

Su, Y.; Chen, S.; Shi, H.; Liu, M.; Zhang, Z.; Huang, N.; Zhong, W.; Zhu, Z.; Liu, Y.; and Liu, X. 2026. World guidance: World modeling in condition space for action generation. arXiv preprint arXiv:2602.22010.

Sun, J.; Zhang, W.; Qi, Z.; Ren, S.; Liu, Z.; Zhu, H.; Sun, G.; Jin, X.; and Chen, Z. 2026. Vla-jepa: Enhancing vision-language-action model with latent world model. arXiv preprint arXiv:2602.10098.

Tan, S.; Dou, K.; Zhao, Y.; and Krähenbühl, P. 2025. Interactive post-training for vision-language-action models. arXiv preprint arXiv:2505.17016.

Wan, T.; Wang, A.; Ai, B.; Wen, B.; Mao, C.; Xie, C.-W.; Chen, D.; Yu, F.; Zhao, H.; Yang, J.; et al. 2025. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314.

Yan, F.; Liu, F.; Huang, Y.; Guan, Z.; Zheng, L.; Zhong, Y.; Feng, C.; and Ma, L. 2025. RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 13707–13718.

Ye, A.; Wang, B.; Ni, C.; Huang, G.; Zhao, G.; Li, H.; Li, H.; Li, J.; Lv, J.; Liu, J.; et al. 2026a. GigaWorld-Policy: An Eficient Action-Centered World–Action Model. arXiv preprint arXiv:2603.17240.

Ye, S.; Ge, Y.; Zheng, K.; Gao, S.; Yu, S.; Kurian, G.; Indupuru, S.; Tan, Y. L.; Zhu, C.; Xiang, J.; et al. 2026b. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922.

Yu, H.; Lin, H.; Zhang, J.; Zhang, W.; Gu, C.; Li, H.; and Tan, P. 2026. Maskwam: Unifying mask prompting and prediction for world-action models. arXiv preprint arXiv:2606.13515.

Yuan, T.; Dong, Z.; Liu, Y.; and Zhao, H. 2026. Fast-wam: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666.

Zhang, J.; Zhu, J.; Su, T.; Ma, C.; Huang, Z.; Xu, Y.; and Wang, H. 2026a. Learning 4D Geometric Priors for Inference-Eficient World Action Models. arXiv preprint arXiv:2607.05468.

Zhang, W.; Liu, H.; Qi, Z.; Wang, Y.; Yu, X.; Zhang, J.; Dong, R.; He, J.; Wang, H.; Zhang, Z.; et al. 2026b. Dreamvla: a vision-language-action model dreamed with comprehensive world knowledge. Advances in Neural Information Processing Systems, 38: 24195–24228.

Zhang, W.; Zhang, B.; Qi, Z.; Zeng, W.; Jin, X.; and Zhang, L. 2026c. Disentangled robot learning via separate forward and inverse dynamics pretraining. arXiv preprint arXiv:2604.16391.

Zhang, Z.; Li, Z.; Rahmati, B.; Yang, R. H.; Ma, Y.; Rasouli, A.; Pakdamansavoji, S.; Wu, Y.; Zhang, L.; Cao, T.; et al. 2026d. Do world action models generalize better than vlas? a robustness study. arXiv preprint arXiv:2603.22078.

Zheng, J.; Li, J.; Wang, Z.; Liu, D.; Kang, X.; Feng, Y.; Zheng, Y.; Zou, J.; Chen, Y.; Zeng, J.; et al. 2025. X-vla: Softprompted transformer as scalable cross-embodiment visionlanguage-action model. arXiv preprint arXiv:2510.10274.