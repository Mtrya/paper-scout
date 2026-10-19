<!-- page 1 of 43 -->

arXiv:2605.30161v1 [cs.CV] 28 May 2026

# Why Far Looks Up: Probing Spatial Representation in Vision-Language Models

Cheolhong Min<sup>1</sup>, Jaeyun Jung<sup>1</sup>, Daeun Lee<sup>1</sup>, Hyeonseong Jeon<sup>1</sup>, Yu Su<sup>2</sup>, Jonathan Tremblay<sup>3</sup>, Chan Hee Song<sup>3,†,‡</sup>, and Jaesik Park<sup>1,†</sup>

1 Seoul National University

2 The Ohio State University

3 NVIDIA

lusong@nvidia.com, {cheolhong.min, jaesik.park}@snu.ac.kr

Abstract. Vision-language models (VLMs) achieve strong performance on spatial reasoning benchmarks, yet it remains unclear whether this reflects structured 3D understanding or reliance on statistical shortcuts in natural images. We introduce a representation-level analysis framework that constructs minimal contrastive pairs to measure how spatial axes are organized and disentangled within VLM embeddings. Our analysis across multiple model families reveals a consistent vertical-distance entanglement: models conflate vertical image position with distance, mirroring the perspective bias of natural photographs. This bias produces a significant accuracy gap between perspective-consistent and counter-heuristic examples, and intensifies under data scaling even as overall benchmark accuracy improves. We further show that models with similar benchmark scores can exhibit different internal representations, and that these differences predict accuracy and robustness across diverse spatial reasoning benchmarks. To isolate this bias from evaluation-set skew, we introduce SpatialTunnel, a synthetic benchmark designed to expose spatial shortcut biases by removing common correlations present in natural images. Experiments suggest that the entanglement is model-intrinsic, and that models with well-separated spatial axes exhibit greater robustness, indicating that well-structured spatial representations lead to more reliable spatial reasoning across diverse benchmarks. Code and benchmark are available on the [project website](https://cheolhong0916.github.io/whyfarlooksup.github.io/).

Keywords: Vision-Language Models· Spatial Understanding· Representation Analysis

## 1 Introduction

Spatial reasoning is a core capability for Vision-Language Models (VLMs), particularly as these systems are increasingly deployed in robotics [29, 32, 41, 54], embodied agents [1, 48, 51], and multimodal assistants [2, 24, 42] that observe

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">Co-corresponding author ‡ Project Lead</span></small>

<!-- page 2 of 43 -->

2

C. Min et al.

![Image block](doc:bf97463/tier:basic/page:2/block:3)

Fig. 1: Many VLMs answer spatial questions via a perspective-driven shortcut, e.g., objects located higher in the image are further away in 3D. By confusing 2D vertical position with 3D distance, models fail systematically on counter examples. Our SpatialTunnel benchmark and contrastive probing expose this vertical-distance entanglement. In contrast, strong spatial VLMs show disentangled axes and consistent correctness across both real and synthetic settings.

and interact with physical environments. Although modern VLMs are primarily trained on 2D image–text pairs [3, 4,16,39], they achieve strong performance on spatial reasoning benchmarks [20, 23, 55], and recent work continues to improve these results through scaling and spatial training data [12, 13, 50, 53, 65]. These advances suggest that current models possess meaningful spatial understanding. However, it remains unclear whether strong benchmark accuracy reflects robust spatial reasoning or the exploitation of statistical regularities in natural images.

Many spatial relations can be partially inferred from correlations that arise naturally in photographic data rather than from explicit reasoning about 3D spatial structure. For example, perspective in everyday photographs introduces a consistent relationship between vertical image position and depth: objects appearing higher in the image are often farther from the camera, as in Figure 1. Such correlations allow models to rely on shortcuts that substitute vertical cues for depth reasoning, achieving high benchmark accuracy while internally conflating distinct spatial dimensions.

This limitation highlights a broader challenge in evaluating spatial understanding in VLMs. Behavioral benchmarks measure whether a model produces correct answers, but they provide limited insight into how those answers are obtained. Two models may achieve similar performance while relying on different internal mechanisms: one encoding spatial relations in a structured, separable manner, and another depending on correlated cues present in natural imagery which become brittle under distribution shift. Distinguishing these pos-

<!-- page 3 of 43 -->

Why Far Looks Up

3

sibilities requires examining how spatial information is represented inside the model, rather than relying on output-level performance.

Recent work has revealed persistent spatial reasoning failures through controlled benchmarks [30, 61, 62] and has begun probing internal model behavior such as attention dynamics [11]. However, these efforts primarily assess individual task performance or local mechanisms, leaving the global geometric organization of spatial relations in representation space largely unexplored.

We address this gap from two complementary angles. First, we analyze how spatial relations along three core 3D axes – horizontal (left / right), vertical (above / below), and depth (close / far) – are organized within VLM internal embeddings, using controlled contrastive examples that vary only the spatial relation between objects while holding confounds such as object identity fixed. Second, we introduce SpatialTunnel, a synthetic benchmark designed to remove perspective-driven biases in spatial evaluation. Its tunnel geometry decouples vertical image position from depth, enabling balanced assessment beyond the correlations present in natural image benchmarks.

Across multiple VLM families, our experiments reveal that horizontal relations form stable, opposing directions in representation space, whereas vertical and depth relations are frequently entangled, suggesting reliance on perspectivedriven cues. Moreover, models with more structured spatial representations perform better across diverse spatial reasoning benchmarks, including EmbSpatial-Bench [20], CV-Bench [55], and BLINK [23]. Evaluations on SpatialTunnel further expose biases hidden under standard benchmark settings, and models with more structured representations exhibit greater robustness when these correlations are removed. Together, these results suggest that benchmark accuracy alone may overestimate the spatial reasoning capabilities of current VLMs. Our contributions are threefold:

Representation-level analysis of spatial reasoning. We introduce a framework for analyzing how spatial relations are organized within VLM embeddings, diagnosing whether models encode structured spatial reasoning or rely on shortcut cues.

– Spatial representations predict robustness. We show that models with similar benchmark performance can exhibit markedly different internal spatial representations, and that models with more structured spatial representations exhibit greater robustness and generalization.

– A bias-controlled synthetic benchmark for spatial reasoning. We construct a synthetic dataset that decouples vertical image position from depth, revealing shortcut biases hidden under standard benchmark settings.

## 2 Related Work

Spatial Understanding Datasets and Benchmarks. Recent benchmarks have revealed persistent weaknesses in VLM spatial reasoning despite strong semantic performance. Controlled evaluations such as What’s Up [30] and COM-FORT [62] show that models frequently fail on basic positional distinctions

<!-- page 4 of 43 -->

4

C. Min et al.

and frame-of-reference consistency. To probe deeper spatial competence, subsequent work has expanded along several axes: egocentric and cross-video reasoning [20, 55], 6DoF diagnostic tasks [57], and multi-step spatial referring [65]. In parallel, simulation-based datasets [45, 50, 54] provide large-scale supervision for physical dynamics, yet spatial performance often plateaus with data scaling [61]. While these efforts effectively measure whether models succeed or fail, they do not examine what cues models rely on internally—in particular, none isolate the entanglement between vertical image position and perceived depth that arises from perspective projection. Our work targets this gap by constructing controlled synthetic environments and contrastive splits that systematically expose this bias.

Probing Internal Representations of Vision-Language Models. Recent work has moved beyond behavioral evaluation to examine the internal states of VLMs. Linear probing studies show that vision encoders inherently represent monocular depth cues [15] and bind geometric coordinates to object activations in early layers [31], while unified extraction frameworks [47] facilitate systematic comparison across model families. On the mechanistic side, ADAPTVIS [11] analyzes attention dynamics during spatial reasoning, and Spatial Forcing [37] explicitly aligns intermediate layers with 3D structure. However, these approaches primarily detect the presence of individual spatial primitives or adjust local attention behavior; they do not examine how different spatial dimensions are jointly organized—in particular, whether depth and vertical cues occupy separable or entangled directions in representation space. We address this gap through controlled contrastive analysis of internal embeddings, directly measuring the geometric relationship between spatial axes to reveal entanglement that isolated probing cannot detect.

## 3 Perspective Projection Bias in Spatial Understanding

Vision-language models are increasingly expected to reason about 3D spatial relationships from a single RGB image, e.g., answering questions such as “Is the chair closer to the camera than the table?” However, monocular images provide only a 2D projection of the 3D scene, requiring models to infer spatial structure from indirect visual cues. A central question is whether current VLMs genuinely learn such 3D reasoning, or instead rely on the visual cues that happen to correlate with depth in image space.

In this section, we analyze how VLMs perform spatial reasoning across multiple model families and benchmarks. Our analysis reveals a systematic bias arising from perspective projection: models frequently use an object’s vertical position in the image as a proxy for its distance from the camera. We term this phenomenon vertical-distance entanglement, where image-plane vertical position becomes conflated with depth. Across multiple models and benchmarks, we show that this bias consistently emerges and leads to systematic errors in spatial reasoning.

<!-- page 5 of 43 -->

Why Far Looks Up

5

## 3.1 What is Vertical-Distance Entanglement?

Perspective projection and vertical position. From the observer’s viewpoint, objects farther away on a common ground surface appear higher in the image. This phenomenon gives rise to the classical elevation cue: for objects lying on the ground plane, those nearer to the horizon line are perceived as being farther from the observer [15] (see Appendix A for details).

![Image block](doc:bf97463/tier:basic/page:5/block:5)

Fig. 2: Consistent vs. counter examples. Consistent: Farther object appears higher in the image; Counter: Farther object appears lower.

Entanglement as a shortcut. We hypothesize that VLMs exploit this correla-

tion as a shortcut: when asked about relative depth, they partially rely on the vertical positions of objects rather than reasoning about 3D structure. We refer to this phenomenon as vertical-distance entanglement, indicating the tendency of a model to treat above ≈ far and below ≈ close when answering depth-related questions.

Consistent and counter-heuristic examples. To systematically analyze this entanglement, we categorize depth-related samples into two groups, consistent and counter (Figure 2). The classification is based on whether the groundtruth spatial relationship aligns with the vertical-position heuristic.

We implement this by comparing the vertical center coordinates of the two queried objects in pixel space: if the farther object has a smaller y-coordinate (i.e., higher in the image), the example is consistent; otherwise, it is counter. If a model exhibits no entanglement, its accuracy should be comparable on both groups. Conversely, a systematic accuracy gap between the two groups would constitute evidence that the model relies on the vertical-position shortcut.

## 3.2 Experimental Setup

Models. We evaluate three VLM families spanning different architectures: Molmo 7B-O-0924 [16], NVILA-Lite-2B [40], and Qwen2.5-VL-3B-Instruct [4]. To analyze how spatial fine-tuning affects entanglement, we train variants of each model at multiple data scales (80k, 400k, 800k, and 2M samples); base models refer to the original pretrained weights without additional fine-tuning. We also include RoboRefer-2B-SFT [65], which shares the NVILA-Lite-2B base but is trained on more than 20M samples including RGB and RGB-D images, and Qwen3-VL-235B-A22B-Instruct [3] as a large-scale reference.

Training data. Recent work has attributed VLMs’ limited spatial understanding to a lack of spatial reasoning data during training, motivating several spatialfocused datasets [9, 19, 45, 50, 60, 65]. To study the effect of data scaling within

<!-- page 6 of 43 -->

6

C. Min et al.

Table 1: Distribution of consistent, counter, and ambiguous examples. Existing spatial benchmarks are skewed toward consistent examples, mirroring the natural statistics of perspective projection in real-world images.

| Type E | mbSpatial-Benc | h CV-Bench-3D | Definition |
| --- | --- | --- | --- |
| Consistent | 976 (80.9%) | 363 (60.5%) | GT aligns with heuristic |
| Counter | 129 (10.7%) | 65 (10.8%) | GT contradicts heuristic |
| Ambiguous | 101 (8.4%) | 172 (28.7%) | ∆y &lt; 5% of image height |

and across model families, we uniformly mix five existing spatial understanding datasets (i.e., SAT [45], RoboSpatial [50], SPAR-7M [60], RefSpatial [65], PRISM [19]) and subsample at four target scales (80k, 400k, 800k, and 2M) for supervised fine-tuning (see Appendix B.2 and B.3 for details).

## 3.3 Evidence from Existing Benchmarks

We first examine whether vertical-distance entanglement is observable on established spatial reasoning benchmarks that use real-world images: EmbSpatial-Bench [20] and the 3D-spatial split of CV-Bench [55].

Data distribution is skewed toward consistent examples. We classify all depth-related questions in both benchmarks into consistent, counter, and ambiguous categories following the criteria defined in Section 3.1. As shown in Table 1, consistent examples account for 80.9% of EmbSpatial-Bench and 60.5% of CV-Bench-3D, while counter examples constitute only about 10% in each. This heavy skew reflects the natural statistics of real-world photographs: in most everyday scenes, farther objects do appear higher in the image.

Models systematically fail on counter examples. We evaluate a range of VLMs spanning different architectures and scales on the two benchmarks, reporting accuracy separately for consistent and counter subsets (Table 2). Across all models and all training scales, accuracy on consistent examples significantly exceeds that on counter examples. For instance, Qwen2.5-VL fine-tuned on 2M samples achieves 60.9% on the consistent split of EmbSpatial-Bench but only 24% on counter examples, yielding a 36.9 percentage-point gap. This pattern holds regardless of model family (Molmo, NVILA, or Qwen2.5-VL), model size, or the amount of spatial fine-tuning data, suggesting that vertical-distance entanglement is a widespread phenomenon rather than an artifact of any single architecture, training recipe, or data scale.

## 4 Behavioral Analysis with a Synthetic Dataset

The accuracy gap in Section 3 indicates that VLMs systematically fail on counter examples in real-world datasets. However, real photographs conflate multiple

<!-- page 7 of 43 -->

Why Far Looks Up

7

Table 2: Accuracy on consistent vs. counter examples across models and benchmarks. All models exhibit a substantial accuracy gap, with consistent examples outperforming counter examples. Results are reported on depth-related questions from EmbSpatial-Bench and CV-Bench-3D. Indented rows denote fine-tuned variants at the given spatial data scale.

|  | EmbSpat | ial-Bench | CV-Be | nch-3D |
| --- | --- | --- | --- | --- |
| Model | Consistent | Counter | Consistent | Counter |
| Molmo-7B-O-0924 [16] | 63.5 | 34.9 (-28.6) | 93.1 | 75.4 (-17.7) |
| + 80k | 60.6 | 29.5 (-31.1) | 80.2 | 56.9 (-23.3) |
| + 400k | 62.7 | 27.1 (-35.6) | 89.5 | 56.9 (-32.6) |
| + 800k | 65.2 | 34.1 (-31.1) | 88.7 | 70.8 (-17.9) |
| + 2M | 65.3 | 39.5 (-25.8) | 90.6 | 72.3 (-18.3) |
| NVILA-Lite-2B [40] | 49.0 | 27.1 (-21.9) | 74.4 | 40.0 (-34.4) |
| + 80k | 57.7 | 15.5 (-42.2) | 71.6 | 50.8 (-20.8) |
| + 400k | 61.1 | 34.1 (-27.0) | 81.3 | 58.5 (-22.8) |
| + 800k | 63.2 | 38.8 (-24.4) | 84.6 | 67.7 (-16.9) |
| + 2M | 60.7 | 41.1 (-19.6) | 97.2 | 93.8 (-3.4) |
| RoboRefer-2B-SFT [65] | 87.0 | 59.7 (-27.3) | 98.9 | 95.4 (-3.5) |
| Qwen2.5-VL-3B-Instruct [4] | 54.7 | 32.6 (-22.1) | 75.5 | 55.4 (-20.1) |
| + 80k | 50.6 | 30.2 (-20.4) | 69.7 | 60.0 (-9.7) |
| + 400k | 52.6 | 27.1 (-25.5) | 65.8 | 58.5 (-7.3) |
| + 800k | 55.8 | 26.4 (-29.4) | 61.2 | 58.5 (-2.7) |
| + 2M | 60.9 | 24.0 (-36.9) | 62.0 | 53.8 (-8.2) |
| Qwen3-VL-235B-A22B-Instruct [3] | 73.3 | 41.7 (-31.6) | 98.1 | 90.8 (-7.3) |

depth cues (e.g., vertical position, apparent size, and occlusion), making it difficult to isolate the contribution of any single cue. To enable controlled interventions, we introduce SpatialTunnel, a synthetic dataset that decouples an object’s vertical image-plane position from its 3D depth by design, allowing the two factors to be manipulated independently.

## 4.1 SpatialTunnel Benchmark

To evaluate spatial relations in a controlled manner, we require an environment with two key properties: (i) objects can be positioned arbitrarily, enabling queries over any spatial relation (e.g., left/right, above/below, near/far); and (ii) an object’s vertical position can be adjusted independently of its depth, allowing us to construct image groups that differ only in vertical placement while preserving depth ordering.

To satisfy these requirements, we build a tunnel-shaped synthetic scene in Blender [5] (Figure 3). Each scene consists of a single-point-perspective corridor whose walls, ceiling, and floor are symmetric about the camera’s optical axis, where objects are placed anywhere on the interior tunnel surfaces. Because ob-

<!-- page 8 of 43 -->

8

C. Min et al.

![Image block](doc:bf97463/tier:basic/page:8/block:3)

Fig. 3: SpatialTunnel holds the two objects at fixed depths while sweeping their angular positions around the tunnel cross-section, so that 2D image-plane layout varies independently of depth ordering.

jects near the top and bottom of the image can be equidistant from the camera, the common heuristic “higher in the image ⇒ farther” no longer holds. We parameterize each object by its depth z and an angular position θ on the tunnel cross-section. Holding z fixed while varying θ moves the object up/down and left/right in the image without changing its depth ordering, enabling matched counterfactual pairs that flip vertical arrangement while preserving depth.

We construct a synthetic benchmark suite, SpatialTunnel, that enables controlled spatial interventions in a single-point-perspective corridor. Specifically, we place two objects at predetermined depths and sweep each object along the tunnel cross-section, discretizing the interior into 16 angular positions (see Figure 3). This yields a $1 6 \times 1 6$ Cartesian grid over $( \theta _ { 1 } , \theta _ { 2 } )$ , enabling heatmap-style diagnostics of model behavior across configurations (see Figure 4). To increase visual diversity and improve robustness, we randomize object appearance (color, size, and shape) and scene lighting across renders. Additional synthetic variants for other spatial cues $( e . g .$ , object size) and auxiliary analyses are provided in the Appendix C.4.

## 4.2 Experimental Setup on SpatialTunnel

Given a rendered RGB image containing two objects, the model is asked a binary depth-comparison question. In our setup, an object is always placed farther from the camera than the other, and the VLM is asked to answer the questions like $^ { \mathrm { ~ \textell ~ } } I s \quad \{ o b j _ { 1 } \}$ closer to / farther from the camera than $\{ o b j _ { 2 } \} ?$ Following prior work [28, 58, 62], we define a local probability by extracting the logits for Yes and No at the first generated token. We then compute the predicted probability as

$$
p = \sigma ( \ell _ { \tt Y e s } - \ell _ { \tt N o } ) .
$$

The correctness score for a single query is defined as $v = p$ if the ground-truth answer is Yes, and $v   =   1 - p$ if it is No. We report the following metrics for all VLMs described in Section 3.2. Following the definition in Section 3.1, samples are partitioned into consistent and counter subsets. We report four metrics: (1) Mean accuracy (v), the mean correctness score across all images and questions; (2) Consistent accuracy $( v _ { \mathrm { c o n s } } )$ , the mean correctness score on consistent examples; (3) Counter accuracy $( v _ { \mathrm { c t r } } )$ , the mean correctness score on counter

<!-- page 9 of 43 -->

Why Far Looks Up

9

![Chart block](doc:bf97463/tier:basic/page:9/block:3)

(a) Results on consistent samples

![Chart block](doc:bf97463/tier:basic/page:9/block:5)

(b) Results on counter samples.

Fig. 4: Mean accuracy heatmaps on SpatialTunnel for Molmo-7B. Each cell indexes a joint angular configuration $( \theta _ { 1 } , \theta _ { 2 } )$ of the two objects (red = higher accuracy; blue = lower). Gray indicates configurations outside the subset. From base → 400k → 2M training samples, accuracy on (a) perspective-consistent cells improves steadily. In contrast, (b) counter cells remain substantially harder, with the largest drop at 400k and a partial recovery at 2M.

examples; and (4) Accuracy gap $( \varDelta = v _ { \mathrm { c o n s } } - v _ { \mathrm { c t r } } )$ , the accuracy difference between the two subsets, quantifying the vertical-distance entanglement. A model with no directional bias would yield $\varDelta \approx 0$

## 4.3 Results on SpatialTunnel: Vertical-Distance Entanglement

Consistent with Section 3.3, we observe that the vertical-distance entanglement is universal. Across all base and fine-tuned models, accuracy is consistently higher on the consistent subset than on the counter subset, yielding a positive accuracy gap ∆. Table 3 and Figure 4 summarize model behavior on SpatialTunnel.

For example, base Qwen2.5-VL-3B achieves $v _ { \mathrm { c o n s } } \; = \; 0 . 7 7 6$ but only $v _ { \mathrm { c t r } } ~ = ~ 0 . 3 6 0$ , indicating strong reliance on the vertical-position shortcut. While base NVILA-Lite-2B produces a narrower gap, its sub-0.5 overall accuracy suggests near-random performance rather than meaningful depth understanding. Figure 4 visualizes positional bias at the cell level for

Table 3: Consistent vs. Counter accuracy on SpatialTunnel. v: mean correctness score; $v _ { \mathrm { c o n s } }$ and $v _ { \mathrm { c t r } } \colon$ scores on consistent and counter subsets; $\varDelta = v _ { \mathrm { c o n s } } - v _ { \mathrm { c t r } } .$

| Model | v vcons vctr ∆ |
| --- | --- |
| Molmo-7B-O-0924 | 0.528 0.565 0.487 +0.078 |
| + 80k | 0.496 0.507 0.486 +0.021 |
| + 400k | 0.501 0.593 0.409 +0.184 |
| + 800k | 0.531 0.628 0.430 +0.198 |
| + 2M | 0.666 0.703 0.630 +0.073 |
| NVILA-Lite-2B | 0.488 0.504 0.471 +0.033 |
| + 80k | 0.499 0.562 0.438 +0.124 |
| + 400k | 0.669 0.804 0.538 +0.267 |
| + 800k | 0.646 0.728 0.571 +0.157 |
| + 2M | 0.812 0.875 0.749 +0.127 |
| RoboRefer-2B-SFT | 0.793 0.816 0.770 +0.046 |
| Qwen2.5-VL-3B | 0.570 0.776 0.360 +0.416 |
| + 80k | 0.512 0.585 0.440 +0.145 |
| + 400k | 0.503 0.588 0.418 +0.171 |
| + 800k | 0.499 0.600 0.398 +0.202 |
| + 2M | 0.500 0.648 0.353 +0.295 |
| Qwen3-VL-235B | 0.908 0.948 0.880 +0.068 |

Molmo-7B variants. If predictions were insensitive to 2D placement, accuracy would be approximately uniform across the grid. Instead, most models show pronounced contrast between consistent and counter regions. The results suggest that large-scale spatial training reduces this reliance. RoboRefer [65], trained on more than 20M QA pairs, achieves the smallest gap $( \varDelta   =   + 0 . 0 4 6 )$ among models performing above chance. Qwen3-VL-235B attains the highest mean accuracy $( v = 0 . 9 0 8 )$ with a similarly small gap $( \varDelta = + 0 . 0 6 8 )$ , indicating that very large-scale pretraining can substantially alleviate this bias even without targeted spatial fine-tuning.

<!-- page 10 of 43 -->

10

C. Min et al.

![Image block](doc:bf97463/tier:basic/page:10/block:3)

Fig. 5: Contrastive probing for representation-level spatial analysis. Given a spatial-relation VQA, we construct a minimal question pair by swapping the object order, which flips the ground-truth relation. We extract the final-token hidden state at an intermediate layer for each question and compute a delta vector as their difference, isolating the relational displacement in embedding space. Aggregated across samples, these vectors summarize the model’s internal spatial representations and enable diagnosing systematic confounds among spatial cues.

## 5 Representation Analysis via Contrastive Probing

Sections 3–4 established vertical-distance entanglement as a model-intrinsic phenomenon through behavioral evaluation. We now turn to internal representations to examine how spatial axes are encoded and what distinguishes models that exhibit robust spatial reasoning.

## 5.1 Beyond Benchmark Accuracy

Behavioral accuracy alone can be a misleading indicator of spatial understanding. Table 4 reports performance across five spatial reasoning tasks spanning different formats, dimensionalities, and difficulty levels. Beyond EmbSpatial-Bench and the 3D-spatial split of CV-Bench (CV-3D) used in Table 2, we additionally include the 2D-spatial split of CV-Bench (CV-2D) and the spatial relationship and relative depth splits of BLINK [23]. Detailed task descriptions are provided in the Appendix B.4.

Fine-tuned variants of Molmo, NVILA, and Qwen show inconsistent patterns across benchmarks. For example, NVILA (2M) achieves 93.8% on CV-3D Depth but only 62.9% on BLINK Spatial Relation, while Qwen (2M) scores 78.3% on BLINK Spatial Relation but drops to 52.2% on CV-3D Distance. No single accuracy figure reliably indicates how well these models have internalized 3D spatial concepts. In contrast, RoboRefer-SFT-2B and Qwen3-VL-235B achieve consistently high performance across all benchmarks. As we show in Section 5.4, these models also exhibit the most structured internal representations under our probing framework, including high axis coherence, well-separated PCA clusters, and low entanglement. This suggests that representation quality underlies robust spatial reasoning across benchmarks, motivating us to look beyond accuracy and analyze model-internal representations directly.

<!-- page 11 of 43 -->

Why Far Looks Up

11

Table 4: Performance comparison across spatial understanding benchmarks. Fine-tuned variants of Molmo, NVILA, and Qwen exhibit inconsistent performance fluctuations depending on the benchmark, while RoboRefer and Qwen3-VL-235B, which show strong representation structure under our probing framework (Section 5.4), achieve consistently high performance across all evaluated tasks. Bold denotes best. BLINK confidence intervals are reported in Appendix B.5.

<table><tr><td rowspan="2">Model</td><td>EmbSpatial CV-2D</td><td></td><td colspan="2">CV-3D</td><td colspan="2">BLINK</td></tr><tr><td>Overall</td><td></td><td></td><td></td><td></td><td>Relation Depth Distance Rel. Depth Spat. Rel.</td></tr><tr><td>Molmo-7B-O-0924 [16]</td><td>60.7</td><td>76.3</td><td>84.5</td><td>68.5</td><td>78.2</td><td>70.6</td></tr><tr><td>+ 80k</td><td>52.9</td><td>62.3</td><td>71.0</td><td>67.5</td><td>72.6</td><td>60.8</td></tr><tr><td>+ 400k</td><td>64.9</td><td>84.3</td><td>80.0</td><td>70.8</td><td>72.6</td><td>68.5</td></tr><tr><td>+ 800k</td><td>69.1</td><td>90.0</td><td>82.0</td><td>70.8</td><td>75.0</td><td>61.5</td></tr><tr><td>+ 2M</td><td>74.3</td><td>93.7</td><td>87.3</td><td>81.3</td><td>71.0</td><td>69.2</td></tr><tr><td>NVILA-Lite-2B [40]</td><td>54.0</td><td>58.6</td><td>69.2</td><td>52.3</td><td>64.5</td><td>67.1</td></tr><tr><td>+ 80k</td><td>65.1</td><td>78.9</td><td>66.2</td><td>60.8</td><td>53.2</td><td>74.1</td></tr><tr><td>400k 十</td><td>62.1</td><td>83.2</td><td>74.3</td><td>67.0</td><td>71.8</td><td>63.6</td></tr><tr><td>+ 800k</td><td>69.7</td><td>85.5</td><td>78.2</td><td>71.3</td><td>57.3</td><td>65.0</td></tr><tr><td>+ 2M</td><td>69.4</td><td>91.4</td><td>93.8</td><td>87.2</td><td>70.2</td><td>62.9</td></tr><tr><td>[65] RoboRefer-SFT-2B</td><td>92.0</td><td>96.5</td><td>95.7</td><td>90.5</td><td>84.7</td><td>79.7</td></tr><tr><td>Qwen2.5-VL-3B [4]</td><td>62.3</td><td>67.4</td><td>70.3</td><td>60.2</td><td>68.6</td><td>83.9</td></tr><tr><td>+ 80k</td><td>57.3</td><td>59.7</td><td>64.7</td><td>61.5</td><td>58.1</td><td>79.7</td></tr><tr><td>400k 十</td><td>58.6</td><td>58.2</td><td>62.0</td><td>54.5</td><td>58.9</td><td>78.3</td></tr><tr><td>+ 800k</td><td>60.9</td><td>59.4</td><td>58.7</td><td>51.2</td><td>58.1</td><td>79.0</td></tr><tr><td>+ 2M</td><td>65.7</td><td>68.8</td><td>58.5</td><td>52.2</td><td>53.2</td><td>78.3</td></tr><tr><td>Qwen3-VL-235B [3]</td><td>82.0</td><td>96.5</td><td>93.3</td><td>91.0</td><td>84.7</td><td>90.2</td></tr></table>

## 5.2 Contrastive Probing

Given an image, we construct a pair of questions that differ only in the ordering of the queried objects, such as swapping “Is A to the left or right of $B _ { \cdot } ^ { \mathcal { O } ^ { \prime \prime } }$ with “Is B to the left or right of A?”. Consequently, the ground-truth answer for the swapped query becomes the spatial inverse of the original; for instance, a left relationship is inverted to right. For probing, we extract hidden states at a fixed intermediate layer L∗ per model family. Let $h _ { q } \in \mathbb { R } ^ { d }$ denote the final-token hidden state at layer L∗for question q. For a question pair $( q _ { 1 } , q _ { 2 } )$ , we define delta vector $\delta = h _ { q _ { 2 } } - h _ { q _ { 1 } }$ as the representation-space displacement induced by the swap. By repeating this across many images, we obtain a set of delta vectors per spatial category (e.g., above, below, far, close, left, right). This procedure aims to isolate the latent encoding of spatial directions by neutralizing common visual components. We define two metrics over these delta vectors:

Axis coherence. For each spatial axis (horizontal, vertical, distance), we pool the delta vectors from both opposing categories (e.g., far and close for the distance axis). To align directions, we negate the deltas from the opposing category

<!-- page 12 of 43 -->

12

C. Min et al.

so that all vectors point toward the canonical direction:

$$
\tilde { \delta } ^ { ( i ) } =  \begin{cases} { \delta ^ { ( i ) } } & { \mathrm { i f ~ c a t e g o r y ~ i s ~ c a n o n i c a l ~ ( \mathit { e . g . , ~ f a r } ) } } \\ { - \delta ^ { ( i ) } } & { \mathrm { i f ~ c a t e g o r y ~ i s ~ o p p o s i t e ~ ( \mathit { e . g . , ~ c l o s e } ) } } \end{cases}\tag{1}
$$

Axis coherence is the mean pairwise cosine similarity over the sign-corrected set:

$$
\operatorname { C o h _ { a x i s } } = \frac { 2 } { N ( N - 1 ) } \sum _ { i < j } \operatorname { c o s } ( \tilde { \delta } ^ { ( i ) } , \; \tilde { \delta } ^ { ( j ) } ) .\tag{2}
$$

High coherence indicates that the model encodes the axis as a stable, consistent direction in representation space.

VD-Entanglement Index. To quantify the degree of vertical-distance entanglement at the representation level, we compute the mean delta vector $\mu _ { c }$ for each category $c \in \{$ {above, below, far, close} and define the VD-Entanglement Index (VD-EI):

$$
\begin{array} { r l } & { \mathrm { V D - E I } = \frac { 1 } { 4 } \big [ \cos ( \mu _ { \mathrm { a b o v e } } , \mu _ { \mathrm { f a r } } ) + \cos ( \mu _ { \mathrm { b e l o w } } , \mu _ { \mathrm { c l o s e } } ) \quad } \\ & { \quad - \cos ( \mu _ { \mathrm { a b o v e } } , \mu _ { \mathrm { c l o s e } } ) - \cos ( \mu _ { \mathrm { b e l o w } } , \mu _ { \mathrm { f a r } } ) \big ] . } \end{array}\tag{3}
$$

The first two terms measure the similarity between perspective-aligned pairs (above↔far, below↔close); the last two measure perspective-opposing pairs. A positive value indicates that vertical and distance representations are directionally coupled in the manner predicted by perspective projection; zero indicates independence. We extract hidden states from EmbSpatial-Bench [20] images at a fixed intermediate layer per model family, following previous approaches [10, 25, 49] (see Appendix D.3 and D.4 for details of layer selection).

## 5.3 Distance Coherence and Counter Accuracy

Distance coherence is the weakest axis. Across all models and training scales in Table 5, Coh<sub>D</sub> is the lowest among the three axes. Fine-tuning substantially increases vertical coherence (e.g., Molmo: $0 . 2 3 \; \rightarrow \; 0 . 5 7 ;$ Qwen: $0 . 2 9 \rightarrow 0 . 5 9 )$ , but Coh<sub>D</sub> grows by a comparatively smaller margin.

Distance coherence growth accompanies counter accuracy improvement. Figure 6a plots Coh<sub>D</sub> against counter accuracy on EmbSpatial-Bench, the same dataset from which the coherence values are derived. At early scaling steps $( e . g . ,   8 0 \mathrm { k } )$ , counter accuracy sometimes drops before CohD has meaningfully increased. However, once spatial fine-tuning data reaches sufficient scale, a consistent pattern emerges across NVILA (from 80k onward) and Molmo (from 400k onward): as $\mathrm { C o h _ { D } }$ increases, counter accuracy rises in tandem, tracing an upward trajectory in Figure 6a. In contrast, Qwen’s $\mathrm { C o h _ { D } }$ remains nearly flat throughout scaling $( 0 . 0 4 3 \rightarrow 0 . 0 5 2 )$ , and its counter accuracy declines, widening the consistent-counter gap. This suggests that as models form a more coherent distance representation through spatial data scaling, they become more robust to the vertical-position shortcut. Conversely, when distance coherence stagnates, continued scaling does not resolve the entanglement.

<!-- page 13 of 43 -->

Why Far Looks Up

13

![Chart block](doc:bf97463/tier:basic/page:13/block:3)

(a) Counter Accuracy vs. Distance Coherence

![Chart block](doc:bf97463/tier:basic/page:13/block:5)

(b) Distance Coherence vs. VD-EI

Fig. 6: Internal probing analysis of spatial representations. (a) Positive correlation between behavioral accuracy on counter examples and internal distance coherence (Coh<sub>D</sub>). (b) Comparing distance coherence (Coh<sub>D</sub>) against geometric entanglement (VD-EI) within the NVILA family; RoboRefer occupies a unique region of high coherence and low entanglement. Unlabeled points denote base models, and numeric labels (e.g., 80k) indicate data-mix fine-tuned variants.

Cross-domain validity of distance coherence. To examine whether Coh<sub>D</sub> reflects a reusable representation rather than a benchmark-specific artifact, we measure Coh<sub>D</sub> on SpatialTunnel and compare it against counter accuracy on two other benchmarks. Coh<sub>D</sub> computed on SpatialTunnel correlates with counter accuracy on both EmbSpatial-Bench and CV-Bench-3D $( \rho = 0 . 7 5 9$ and 0.804, respectively; both $p < 1 0 ^ { - 3 } )$ . This cross-domain alignment supports the view that Coh<sub>D</sub> captures predictive signal beyond in-domain computation artifacts (see Appendix D.5 for additional details).

## 5.4 What Characterizes Strong Spatial Representations

We compare the NVILA-Lite-2B scaling series and RoboRefer-2B-SFT [65], which share the same base architecture, to identify what representation profile accompanies robust spatial reasoning. Figure 7 visualizes the delta vectors via PCA. In the base model (i.e., NVILA-Lite-2B), distance delta vectors are collapsed near the origin without forming a distinguishable axis. Fine-tuning with 2M samples initiates directional spreading, but vertical and distance clusters remain overlapped. RoboRefer exhibits three cleanly separated clusters, each aligned with a distinct principal component. Figure 6b quantifies this contrast. The NVILA scaling trajectory yields only marginal gains in Coh<sub>D</sub> while VD-EI remains high (0.54–0.64). RoboRefer occupies a distinct region: the highest Coh<sub>D</sub> (0.182) and the lowest VD-EI (0.362) in the family, corresponding to 59.7% counter accuracy on EmbSpatial-Bench versus 41.1% for NVILA (2M).

<!-- page 14 of 43 -->

14

C. Min et al.

![Image block](doc:bf97463/tier:basic/page:14/block:3)

Fig. 7: PCA of delta vectors across models. Each point is a delta vector colored by axis (orange: horizontal, green: vertical, purple: distance), with darker/lighter shades distinguishing opposing categories within each axis (e.g., left vs. right). Molmo (2M), NVILA (2M), and Qwen (2M) show separation along the horizontal and vertical axes, but distance delta vectors remain poorly distinguished. RoboRefer and Qwen3 exhibit three clearly separated clusters, each aligned with a distinct principal component.

These results suggest that high Coh<sub>D</sub>, together with low VD-EI as a complementary signal, accompanies robust spatial reasoning across benchmarks. Within the NVILA scaling series, incremental increases in fine-tuning scale yield only modest gains in Coh<sub>D</sub>. In contrast, RoboRefer and Qwen3 exhibit well-separated axes and strong overall performance, showing such structure can emerge under substantially richer training regimes. RoboRefer reflects a different training regime (e.g., additional supervision and much larger training), so we treat it as an illustrative reference rather than attributing its gains to a single factor. Overall, Coh<sub>D</sub> can serve as a practical diagnostic for whether training improves spatial representations.

Table 5: Axis coherence and VD-Entanglement Index. Coh measures directional consistency within axes. Coh<sub>D</sub> is consistently the lowest across all models. Bold denotes best.

| Model | CohH | Cohv | CohD | VD-EI |
| --- | --- | --- | --- | --- |
| Molmo-7B | 0.143 | 0.228 | 0.075 | 0.279 |
| + 80k | 0.122 | 0.332 | 0.072 | 0.388 |
| + 400k | 0.236 | 0.597 | 0.096 | 0.459 |
| + 800k | 0.247 | 0.559 | 0.107 | 0.514 |
| + 2M | 0.239 | 0.574 | 0.112 | 0.474 |
| NVILA-2B | 0.323 | 0.289 | 0.052 | 0.539 |
| + 80k | 0.295 | 0.497 | 0.070 | 0.606 |
| + 400k | 0.242 | 0.574 | 0.095 | 0.589 |
| + 800k | 0.278 | 0.498 | 0.089 | 0.591 |
| + 2M | 0.241 | 0.553 | 0.104 | 0.550 |
| RoboRefer-2B | 0.649 | 0.830 | 0.182 | 0.362 |
| Qwen2.5-3B | 0.367 | 0.293 | 0.043 | 0.457 |
| + 80k | 0.386 | 0.315 | 0.040 | 0.456 |
| + 400k | 0.450 | 0.452 | 0.042 | 0.451 |
| + 800k | 0.473 | 0.538 | 0.045 | 0.429 |
|  |  |  |  |  |
| + 2M | 0.485 | 0.586 | 0.052 | 0.472 |

## 6 Conclusion

We introduced a representation-level diagnostic framework that reveals verticaldistance entanglement as a pervasive, model-intrinsic bias across VLM families and scales. Our analysis shows that models with more structured spatial representations – characterized by high distance coherence and low VD-Entanglement Index – not only exhibit stronger counter-heuristic robustness but also achieve higher accuracy across diverse spatial reasoning benchmarks. To isolate this bias from evaluation-set confounds, we introduced the synthetic benchmark SpatialTunnel, which removes perspective-driven correlations present in natural images. Together, these results demonstrate that representational structure, rather than benchmark accuracy alone, provides a reliable indicator of robust spatial reasoning in vision-language models.

<!-- page 15 of 43 -->

Why Far Looks Up

15

## References

1. Ahn, M., Brohan, A., Brown, N., Chebotar, Y., Cortes, O., David, B., Finn, C., Fu, C., Gopalakrishnan, K., Hausman, K., et al.: Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691 (2022) 1

2. Anthropic: Claude opus 4 & claude sonnet 4 system card. Tech. rep., Anthropic (May 2025), [https : / / www - cdn . anthropic . com / 4263b940cabb546aa0e3283f35b686f4f3b2ff47.pdf](https://www-cdn.anthropic.com/4263b940cabb546aa0e3283f35b686f4f3b2ff47.pdf) 1

3. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025) 2, 5, 7, 11, 22

4. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923 (2025) 2, 5, 7, 11, 21

5. Blender Online Community: Blender - a 3D modelling and rendering package. Blender Foundation (2025), [https://www.blender.org/](https://www.blender.org/) 7

6. Brazil, G., Kumar, A., Straub, J., Ravi, N., Johnson, J., Gkioxari, G.: Omni3d: A large benchmark and model for 3d object detection in the wild. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 13154–13164 (2023) 24

7. Chang, A., Dai, A., Funkhouser, T., Halber, M., Niessner, M., Savva, M., Song, S., Zeng, A., Zhang, Y.: Matterport3d: Learning from rgb-d data in indoor environments. arXiv preprint arXiv:1709.06158 (2017) 24

8. Chang, A.X., Funkhouser, T., Guibas, L., Hanrahan, P., Huang, Q., Li, Z., Savarese, S., Savva, M., Song, S., Su, H., et al.: Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012 (2015) 23

9. Chen, B., Xu, Z., Kirmani, S., Ichter, B., Sadigh, D., Guibas, L., Xia, F.: Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14455–14465 (2024) 5

10. Chen, H., Lin, J., Chen, X., Fan, Y., Jin, X., Su, H., Dong, J., Fu, J., Shen, X.: Rethinking visual layer selection in multimodal llms. arXiv e-prints pp. arXiv–2504 (2025) 12, 32

11. Chen, S., Zhu, T., Zhou, R., Zhang, J., Gao, S., Niebles, J.C., Geva, M., He, J., Wu, J., Li, M.: Why is spatial reasoning hard for VLMs? an attention mechanism perspective on focus areas. In: Forty-second International Conference on Machine Learning (2025), [https://openreview.net/forum?id=k7vcuqLK4X](https://openreview.net/forum?id=k7vcuqLK4X) 3, 4

12. Chen, S., Uy, M.A., Song, C.H., Ladhak, F., Murali, A., Qu, Q., Birchfield, S., Blukis, V., Tremblay, J.: SpaceTools: Tool-augmented spatial reasoning via double interactive RL. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2026), [https://arxiv.org/abs/2512.04069,](https://arxiv.org/abs/2512.04069)to appear 2

13. Cheng, A.C., Yin, H., Fu, Y., Guo, Q., Yang, R., Kautz, J., Wang, X., Liu, S.: Spatialrgpt: Grounded spatial reasoning in vision-language models. In: Advances in Neural Information Processing Systems (NeurIPS) (2024) 2

14. Dai, A., Chang, A.X., Savva, M., Halber, M., Funkhouser, T., Nießner, M.: Scannet: Richly-annotated 3d reconstructions of indoor scenes. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5828–5839 (2017) 22, 24

<!-- page 16 of 43 -->

16

C. Min et al.

15. Danier, D., Aygün, M., Li, C., Bilen, H., Mac Aodha, O.: Depthcues: Evaluating monocular depth perception in large vision models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 20049–20059 (2025) 4, 5, 21

16. Deitke, M., Clark, C., Lee, S., Tripathi, R., Yang, Y., Park, J.S., Salehi, M., Muennighoff, N., Lo, K., Soldaini, L., et al.: Molmo and pixmo: Open weights and open data for state-of-the-art vision-language models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 91–104 (2025) 2, 5, 7, 11, 21

17. Deitke, M., Schwenk, D., Salvador, J., Weihs, L., Michel, O., VanderBilt, E., Schmidt, L., Ehsani, K., Kembhavi, A., Farhadi, A.: Objaverse: A universe of annotated 3d objects. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 13142–13153 (2023) 23

18. Deitke, M., VanderBilt, E., Herrasti, A., Weihs, L., Ehsani, K., Salvador, J., Han, W., Kolve, E., Kembhavi, A., Mottaghi, R.: Procthor: Large-scale embodied ai using procedural generation. Advances in Neural Information Processing Systems 35, 5982–5994 (2022) 22

19. Deshpande, A., Deng, Y., Ray, A., Salvador, J., Han, W., Duan, J., Zeng, K.H., Zhu, Y., Krishna, R., Hendrix, R.: Graspmolmo: Generalizable task-oriented grasping via large-scale synthetic data generation. arXiv preprint arXiv:2505.13441 (2025) 5, 6, 23, 24

20. Du, M., Wu, B., Li, Z., Huang, X.J., Wei, Z.: Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). pp. 346–355 (2024) 2, 3, 4, 6, 12, 24

21. Eppner, C., Mousavian, A., Fox, D.: Acronym: A large-scale grasp dataset based on simulation. In: 2021 IEEE International Conference on Robotics and Automation (ICRA). pp. 6222–6227. IEEE (2021) 23

22. Eppner, C., Murali, A., Garrett, C., O’Flaherty, R., Hermans, T., Yang, W., Fox, D.: scene\_synthesizer: A python library for procedural scene generation in robot manipulation. Journal of Open Source Software 10(105), 7561 (2025) 23

23. Fu, X., Hu, Y., Li, B., Feng, Y., Wang, H., Lin, X., Roth, D., Smith, N.A., Ma, W.C., Krishna, R.: Blink: Multimodal large language models can see but not perceive. In: European Conference on Computer Vision. pp. 148–166. Springer (2024) 2, 3, 10, 25

24. Google: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities (2025), [https://arxiv.org/abs/2507.06261](https://arxiv.org/abs/2507.06261) 1, 27

25. Gurnee, W., Tegmark, M.: Language models represent space and time. In: Kim, B., Yue, Y., Chaudhuri, S., Fragkiadaki, K., Khan, M., Sun, Y. (eds.) International Conference on Learning Representations. vol. 2024, pp. 2483–2503 (2024), [https://proceedings.iclr.cc/paper\_files/paper/2024/file/0a6059857ae5c82ea9726ee9282a7145-Paper-Conference.pdf](https://proceedings.iclr.cc/paper_files/paper/2024/file/0a6059857ae5c82ea9726ee9282a7145-Paper-Conference.pdf) 12, 32

26. Hata, K., Savarese, S.: Cs231a course notes 1: Camera models. url: https://web.stanford. edu/class/cs231a/course\_ notes/01-camera-models. pdf (2015) 20

27. Hoiem, D., Savarese, S.: Representations and techniques for 3D object recognition and scene interpretation. Springer Nature (2022) 21

28. Hu, J., Levy, R.: Prompting is not a substitute for probability measurements in large language models. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. pp. 5040–5060 (2023) 8

<!-- page 17 of 43 -->

Why Far Looks Up

17

29. Intelligence, P., Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Galliker, M.Y., Ghosh, D., Groom, L., Hausman, K., Ichter, B., Jakubczak, S., Jones, T., Ke, L., LeBlanc, D., Levine, S., Li-Bell, A., Mothukuri, M., Nair, S., Pertsch, K., Ren, A.Z., Shi, L.X., Smith, L., Springenberg, J.T., Stachowicz, K., Tanner, J., Vuong, Q., Walke, H., Walling, A., Wang, H., Yu, L., Zhilinsky, U.: π0.5: a vision-language-action model with open-world generalization (2025), [https://arxiv.org/abs/2504.16054](https://arxiv.org/abs/2504.16054) 1

30. Kamath, A., Hessel, J., Chang, K.W.: What’s “up” with vision-language models? investigating their struggle with spatial reasoning. In: Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP) (2023) 3

31. Kang, R., Chen, H., Gkioxari, G., Perona, P.: Linear mechanisms for spatiotemporal reasoning in vision language models. arXiv preprint arXiv:2601.12626 (2026) 4

32. Kim, M.J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E.P., Sanketi, P.R., Vuong, Q., Kollar, T., Burchfiel, B., Tedrake, R., Sadigh, D., Levine, S., Liang, P., Finn, C.: Openvla: An open-source vision-language-action model. In: Agrawal, P., Kroemer, O., Burgard, W. (eds.) Proceedings of the 8th Conference on Robot Learning (CoRL). Proceedings of Machine Learning Research, vol. 270, pp. 2679–2713. PMLR (06–09 Nov 2025), [https://proceedings.mlr.press/v270/kim25c.html](https://proceedings.mlr.press/v270/kim25c.html) 1

33. Kolve, E., Mottaghi, R., Han, W., VanderBilt, E., Weihs, L., Herrasti, A., Deitke, M., Ehsani, K., Gordon, D., Zhu, Y., et al.: Ai2-thor: An interactive 3d environment for visual ai. arXiv preprint arXiv:1712.05474 (2017) 24

34. Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.J., Shamma, D.A., et al.: Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision 123(1), 32–73 (2017) 24

35. Kuznetsova, A., Rom, H., Alldrin, N., Uijlings, J., Krasin, I., Pont-Tuset, J., Kamali, S., Popov, S., Malloci, M., Kolesnikov, A., et al.: The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International journal of computer vision 128(7), 1956–1981 (2020) 23

36. Lazarow, J., Griffiths, D., Kohavi, G., Crespo, F., Dehghan, A.: Cubify anything: Scaling indoor 3d object detection. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 22225–22233 (2025) 23

37. Li, F., Song, W., Zhao, H., Wang, J., Ding, P., Wang, D., Zeng, L., Li, H.: Spatial forcing: Implicit spatial representation alignment for vision-language-action model. arXiv preprint arXiv:2510.12276 (2025) 4

38. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: European conference on computer vision. pp. 740–755. Springer (2014) 24

39. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744 (2023) 2

40. Liu, Z., Zhu, L., Shi, B., Zhang, Z., Lou, Y., Yang, S., Xi, H., Cao, S., Gu, Y., Li, D., et al.: Nvila: Efficient frontier visual language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4122–4134 (2025) 5, 7, 11, 21

41. NVIDIA, :, Bjorck, J., Castañeda, F., Cherniadev, N., Da, X., Ding, R., Fan, L.J., Fang, Y., Fox, D., Hu, F., Huang, S., Jang, J., Jiang, Z., Kautz, J., Kundalia, K., Lao, L., Li, Z., Lin, Z., Lin, K., Liu, G., Llontop, E., Magne, L., Mandlekar, A.,

<!-- page 18 of 43 -->

18

C. Min et al.

Narayan, A., Nasiriany, S., Reed, S., Tan, Y.L., Wang, G., Wang, Z., Wang, J., Wang, Q., Xiang, J., Xie, Y., Xu, Y., Xu, Z., Ye, S., Yu, Z., Zhang, A., Zhang, H., Zhao, Y., Zheng, R., Zhu, Y.: Gr00t n1: An open foundation model for generalist humanoid robots (2025), [https://arxiv.org/abs/2503.14734](https://arxiv.org/abs/2503.14734) 1

42. OpenAI: Openai gpt-5 system card (2025), [https://arxiv.org/abs/2601.03267](https://arxiv.org/abs/2601.03267)1, 27

43. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021) 32

44. Raistrick, A., Mei, L., Kayan, K., Yan, D., Zuo, Y., Han, B., Wen, H., Parakh, M., Alexandropoulos, S., Lipson, L., et al.: Infinigen indoors: Photorealistic indoor scenes using procedural generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21783–21794 (2024) 23

45. Ray, A., Duan, J., II, E.L.B., Tan, R., Bashkirova, D., Hendrix, R., Ehsani, K., Kembhavi, A., Plummer, B.A., Krishna, R., Zeng, K.H., Saenko, K.: SAT: Dynamic spatial aptitude training for multimodal language models. In: Second Conference on Language Modeling (2025), [https://openreview.net/forum?id=DW8U8ZWa1U](https://openreview.net/forum?id=DW8U8ZWa1U)4, 5, 6, 22, 24

46. Savva, M., Chang, A.X., Hanrahan, P.: Semantically-enriched 3d models for common-sense knowledge. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. pp. 24–31 (2015) 23

47. Sheta, H., Huang, E.H., Wu, S., Alenabi, I., Hong, J., Lin, R., Ning, R., Wei, D., Yang, J., Zhou, J., et al.: From behavioral performance to internal competence: Interpreting vision-language models with vlm-lens. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. pp. 886–895 (2025) 4

48. Singh, I., Blukis, V., Mousavian, A., Goyal, A., Xu, D., Tremblay, J., Fox, D., Thomason, J., Garg, A.: Progprompt: Generating situated robot task plans using large language models. In: 2023 IEEE International Conference on Robotics and Automation (ICRA). pp. 11523–11530 (2023). [https://doi.org/10.1109/ICRA48891.2023.10161317](https://doi.org/10.1109/ICRA48891.2023.10161317) 1

49. Skean, O., Arefin, M.R., Zhao, D., Patel, N.N., Naghiyev, J., LeCun, Y., Shwartz-Ziv, R.: Layer by layer: Uncovering hidden representations in language models. In: Forty-second International Conference on Machine Learning (2025), [https://openreview.net/forum?id=WGXb7UdvTX](https://openreview.net/forum?id=WGXb7UdvTX) 12, 32

50. Song, C.H., Blukis, V., Tremblay, J., Tyree, S., Su, Y., Birchfield, S.: Robospatial: Teaching spatial understanding to 2d and 3d vision-language models for robotics. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 15768–15780 (2025) 2, 4, 5, 6, 22, 24

51. Song, C.H., Wu, J., Washington, C., Sadler, B.M., Chao, W.L., Su, Y.: Llmplanner: Few-shot grounded planning for embodied agents with large language models. In: ICCV (2023) 1

52. Szeliski, R.: Computer vision: algorithms and applications. Springer Nature (2022) 20

53. Tan, H., Zhou, E., Li, Z., Xu, Y., Ji, Y., Chen, X., Chi, C., Wang, P., Jia, H., Ao, Y., Cao, M., Chen, S., Li, Z., Liu, M., Wang, Z., Rong, S., Lyu, Y., Zhao, Z., Co, P., Li, Y., Han, Y., Xie, S., Yao, G., Wang, S., Zhang, L., Yang, X., Jiao, Y., Shi, D., Xie, K., Nie, S., Men, C., Lin, Y., Wang, Z., Huang, T., Zhang, S.: Robobrain 2.5: Depth in sight, time in mind. arXiv preprint arXiv:2601.14352 (2026) 2

<!-- page 19 of 43 -->

Why Far Looks Up

19

54. Team, G.R., Abeyruwan, S., Ainslie, J., Alayrac, J.B., Arenas, M.G., Armstrong, T., Balakrishna, A., Baruch, R., Bauza, M., Blokzijl, M., et al.: Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020 (2025) 1, 4

55. Tong, S., II, E.L.B., Wu, P., Woo, S., IYER, A.J., Akula, S.C., Yang, S., Yang, J., Middepogu, M., Wang, Z., Pan, X., Fergus, R., LeCun, Y., Xie, S.: Cambrian-1: A fully open, vision-centric exploration of multimodal LLMs. In: The Thirty-eighth Annual Conference on Neural Information Processing Systems (2024), [https://openreview.net/forum?id=Vi8AepAXGy](https://openreview.net/forum?id=Vi8AepAXGy) 2, 3, 4, 6, 24, 32

56. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al.: Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023) 32

57. Wang, X., Ma, W., Zhang, T., de Melo, C.M., Chen, J., Yuille, A.: Spatial457: A diagnostic benchmark for 6d spatial reasoning of large mutimodal models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 24669–24679 (2025) 4

58. Wang, Y., Shi, F.: Logical forms complement probability in understanding language model (and human) performance. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 16862–16877 (2025) 8

59. Yeshwanth, C., Liu, Y.C., Nießner, M., Dai, A.: Scannet++: A high-fidelity dataset of 3d indoor scenes. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 12–22 (2023) 22

60. Zhang, J., Chen, Y., Zhou, Y., Xu, Y., Huang, Z., Mei, J., Chen, J., Yuan, Y.J., Cai, X., Huang, G., et al.: From flatland to space: Teaching vision-language models to perceive and reason in 3d. arXiv preprint arXiv:2503.22976 (2025) 5, 6, 22, 24

61. Zhang, W., Huang, Y., Xu, Y., Huang, J., Zhi, H., Ren, S., Xu, W., Zhang, J.: Why do mllms struggle with spatial understanding? a systematic analysis from data to architecture (2025), [https://arxiv.org/abs/2509.02359](https://arxiv.org/abs/2509.02359) 3, 4

62. Zhang, Z., Hu, F., Lee, J., Shi, F., Kordjamshidi, P., Chai, J., Ma, Z.: Do visionlanguage models represent space and how? evaluating spatial frame of reference under ambiguities. In: The Thirteenth International Conference on Learning Representations (2025), [https://openreview.net/forum?id=84pDoCD4lH](https://openreview.net/forum?id=84pDoCD4lH) 3, 8

63. Zheng, J., Zhang, J., Li, J., Tang, R., Gao, S., Zhou, Z.: Structured3d: A large photo-realistic dataset for structured 3d modeling. In: European Conference on Computer Vision. pp. 519–535. Springer (2020) 22

64. Zhou, B., Zhao, H., Puig, X., Xiao, T., Fidler, S., Barriuso, A., Torralba, A.: Semantic understanding of scenes through the ade20k dataset. International journal of computer vision 127(3), 302–321 (2019) 24

65. Zhou, E., An, J., Chi, C., Han, Y., Rong, S., Zhang, C., Wang, P., Wang, Z., Huang, T., Sheng, L., Zhang, S.: Roborefer: Towards spatial referring with reasoning in vision-language models for robotics. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025), [https://openreview.net/forum?id=OGxalNUHbJ](https://openreview.net/forum?id=OGxalNUHbJ) 2, 4, 5, 6, 7, 9, 11, 13, 22, 23, 24

<!-- page 20 of 43 -->

20

C. Min et al.

## Appendices

The following appendices provide supplementary material that complements the main text.

Appendix A Ground-Plane Geometry and Vertical Image Position.

– Appendix B Additional Details on Experiment Setup.

– Appendix C Additional Details on SpatialTunnel.

– Appendix D Additional Details on Contrastive Probing.

## A Ground-Plane Geometry and Vertical Image Position

We derive how perspective projection on a flat ground plane induces a monotonic relationship between an object’s depth and its vertical image coordinate under a standard pinhole camera model [26, 52].

Setup. Consider a pinhole camera with focal length $f ,$ whose optical axis is aligned with the world Z-axis, and whose center is at height $H _ { c } > 0$ above a horizontal ground plane [26].

World coordinates are $( X , Y , Z )$ , with the ground plane defined as $Y = 0$ and the camera is located at $( 0 , H _ { c } , 0 )$ , looking along the positive Z-axis. In the camera coordinate system, a point on the ground plane has coordinates

$$
( X _ { c } , Y _ { c } , Z _ { c } ) = ( X , - H _ { c } , Z ) ,
$$

where $Z > 0$ denotes the depth of the point.

Perspective projection. Under the pinhole camera model, the projection of $( X _ { c } , Y _ { c } , Z _ { c } )$ onto the image plane at distance $f$ along the $Z _ { c } .$ -axis is given by [26]

$$
u = f \frac { X _ { c } } { Z _ { c } } , \quad v = f \frac { Y _ { c } } { Z _ { c } } .
$$

Substituting the ground-plane coordinates $( X _ { c } , Y _ { c } , Z _ { c } ) = ( X , - H _ { c } , Z )$ yields

$$
u = f \frac { X } { Z } , \quad v = - f \frac { H _ { c } } { Z } .
$$

Thus, for points on the ground plane with fixed camera height $H _ { c } ,$ , the vertical image coordinate satisfies

$$
v ( Z ) = - f \frac { H _ { c } } { Z } .
$$

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">‡ We assume zero camera tilt, square pixels, zero skew, and no lens distortion for simplicity.</span></small>

<!-- page 21 of 43 -->

Why Far Looks Up

21

Depth–height relationship. From the expression above, the magnitude of the vertical coordinate obeys

$$
| v ( Z ) | \propto \frac { 1 } { Z } ,
$$

so that increasing depth Z decreases the magnitude |v|. Adopting the standard image convention that the v-axis increases downward, we define the image-frame coordinate as

$$
v _ { \mathrm { i m g } } ( Z ) = - v ( Z ) = \frac { f H _ { c } } { Z } > 0 ,
$$

which confirms that ground-plane points appear below the principal point. Under the zero-tilt assumption, the horizon coincides with the principal point at $v _ { \mathrm { i m g } } =$ 0 [27], and as $Z \to \infty .$ , we have $v _ { \mathrm { i m g } } ( Z ) \to 0 ^ { + }$ , meaning that points farther along the ground plane project closer to the horizon line and therefore appear higher in the image. Consequently, for objects resting on a common ground plane, greater depth corresponds to a higher vertical position in the image, which is precisely the classical elevation monocular depth cue exploited in both human perception and recent depth-cue benchmarks [15].

## B Additional Details on Experiment Setup

In this section, we detail the models, training data sources, data mix composition, and benchmarks used in the experiments described in Section 3.

## B.1 Models

We conduct experiments on the following vision-language models, each capable of spatial reasoning.

– Molmo-7B-O-0924 [16]: Molmo-7B-O-0924 is a 7B-parameter open visionlanguage model from the Molmo family, trained on the PixMo dataset of around one million carefully curated image–text pairs, using an OLMo-7B backbone with OpenAI CLIP as the vision encoder.

NVILA-Lite-2B [40]: NVILA-Lite-2B is a compact 2B-parameter visual language model built on the NVILA architecture, using a scale-then-compress design that processes high-resolution images and long videos efficiently by compressing visual tokens for faster inference and lower compute cost while maintaining competitive accuracy on standard benchmarks.

Qwen2.5-VL-3B-Instruct [4]: Qwen2.5-VL-3B-Instruct is a 3B-parameter multi-modal vision-language model from the Qwen2.5-VL family, designed to process images, documents, and videos together with text. It combines a Vision Transformer encoder with a Qwen2.5-series language decoder to support instruction-following tasks such as OCR, document understanding, and general visual reasoning.

<!-- page 22 of 43 -->

22

C. Min et al.

RoboRefer-2B-SFT [65]: RoboRefer-2B-SFT is a 2B-parameter visionlanguage model for robotics that is supervised-finetuned on RefSpatial and related instruction-following and referring datasets to handle spatial referring instructions in complex 3D scenes. In the second SFT step, RefSpatial is reused with both RGB and RGB-D inputs so that the image encoder learns robust spatial understanding from RGB alone while using depth as an auxiliary training signal, enabling both RGB-only and RGB-D inference at test time.

Qwen3-VL-235B-A22B-Instruct [3]: Qwen3-VL-235B-A22B-Instruct is a large open-weight Mixture-of-Experts vision–language model (235B parameters, 22B active) that combines text generation with visual understanding over images and video. It is an instruction-tuned Qwen3-VL variant designed for general-purpose multimodal tasks such as visual question answering, document parsing, and multilingual OCR in chat-style interactions.

## B.2 Training Data Sources

A number of benchmarks and datasets for spatial understanding in VLMs have been proposed by the community. Rather than generating training data from scratch, we leverage existing datasets and compose data mixes at varying scales to train the models. Below we describe each dataset used in our experiments.

SAT [45]: SAT is a synthetic spatial reasoning dataset built in the ProcTHOR-10K [18] interactive 3D indoor simulation environment, using about 22K procedurally generated apartment scenes composed of 1K object assets and rendered into 2D views. It contains 175K automatically generated question–answer pairs over 20K scenes, constructed from perfect 3D geometry and simulator metadata without human annotation, and is split into 127K static spatial QAs (relative position, depth, counting) and 48K dynamic spatial QAs grouped into Egocentric Movement, Object Movement, Allocentric Perspective, Goal Aiming, and Action Consequence, where actions in the simulator change spatial relationships across frames.

RoboSpatial [50]: RoboSpatial is a large-scale 2D/3D spatial reasoning dataset built from real indoor and tabletop environments, where egocentric RGB images are paired with 3D scans instead of a synthetic simulator. The data are collected as 3D scene scans and corresponding first-person images, and then automatically annotated with around 3M spatial relations over 1M images and 5K scans, capturing rich object–object and object–space relationships relevant for robotics. The benchmark defines tasks such as spatial affordance prediction (where an object can be placed or an action can be executed), spatial relationship prediction (e.g., left/right, in front/behind, on/under), and robot manipulation tasks that test whether models can use these spatial cues to guide real-world actions.

– SPAR-7M [60]: SPAR-7M is a large-scale spatial reasoning dataset built from indoor 3D scenes (e.g., ScanNet [14], ScanNet++ [59], Structured3D [63])

<!-- page 23 of 43 -->

Why Far Looks Up

23

with around 7M QA pairs and 33 tasks covering a wide range of spatial perception and reasoning skills. For our experiments, we sample from the following tasks: (1) Multi-view object spatial relation, which requires describing object–camera spatial relations in a multi-view setting (2) Single-view object spatial relation, a single-view multiple-choice task that asks models to select the correct relative position between two objects (3) Single-view spatial imagination, which evaluates single-view spatial imagination by asking models to verbally infer observer-centric relations beyond the directly visible configuration (4) Object count, which focuses on numerical reasoning by predicting object counts in the scene and (5) Multi-view spatial imagination, a multi-view spatial imagination task that requires describing how object–object relations change as the camera moves.

RefSpatial [65]: RefSpatial is a large-scale spatial referring dataset built from 2D web images (OpenImages [35]) and 3D embodied videos (CA-1M [36]) plus simulated scenes from Infinigen [44] with Objaverse [17] assets. It contains 2.5M RGB-D samples and 20M QA pairs over 31 spatial relations and up to 5 reasoning steps, covering qualitative/quantitative spatial QA and point-based location/placement supervision. The tasks include object location (pointing to a described object), free-space placement (pointing to a feasible placement location), and multi-step spatial reasoning with explicit intermediate steps on simulated scenes. In our work, we sample from all Ref-Spatial sources but use only the RGB images and associated annotations, discarding depth maps.

PRISM [19]: PRISM is a large-scale synthetic task-oriented grasping dataset built in a procedurally generated tabletop simulation using ShapeNet-Sem [8, 46] objects, ACRONYM [21] grasp annotations, and SceneSynthesizer-based scene composition [22], where 2,300+ object instances are rendered in heavily randomized scenes with calibrated RGB-D views, natural language task instructions, and associated 6-DoF grasp poses. GPT-based pipelines generate and match grasp-centric descriptions and manipulation tasks to appropriate grasps, yielding hundreds of thousands of samples for training and evaluating language-conditioned grasp prediction on both seen and novel objects.

## B.3 Data Mix Composition

We construct four training data mixes of increasing scale using the five spatial datasets described in Section B.2. For the 80k through 800k mixes, each dataset contributes an equal number of samples. For the 2M mix, the per-dataset allocation is adjusted to accommodate differences in total dataset size, with smaller datasets (e.g., SAT) included in full while larger ones (e.g., RefSpatial) are subsampled. Table 6 summarizes the per-dataset sample counts at each scale.

Within each dataset, samples are drawn proportionally across all constituent sub-files (e.g., the six task categories of SAT, the seven QA types of RefSpatial). Detailed per-file sampling ratios and the ready-to-use data mix configurations will be publicly released.

<!-- page 24 of 43 -->

24

C. Min et al.

Table 6: Per-dataset sample counts for each data mix scale. The 80k–800k mixes use uniform allocation across datasets. The 2M mix uses all available samples from smaller datasets and subsamples larger ones (RefSpatial at ∼3.3%, SAT and PRISM in full).

<table><tr><td>Dataset</td><td>Available</td><td>80k</td><td>400k</td><td>800k</td><td>2M</td></tr><tr><td>SAT [45]</td><td>172,384</td><td>15,997</td><td>79,997</td><td>159,998</td><td>172,384</td></tr><tr><td>RoboSpatial [50]</td><td>300,000</td><td>15,999</td><td>79,998</td><td>159,999</td><td>300,000</td></tr><tr><td>SPAR-7M [60]</td><td>608,538</td><td>15,997</td><td>79,997</td><td>159,996</td><td>608,538</td></tr><tr><td>RefSpatial [65]</td><td>16,363,656</td><td>15,994</td><td>79,996</td><td>159,997</td><td>540,397</td></tr><tr><td>PRISM [19]</td><td>378,844</td><td>16,000</td><td>80,000</td><td>160,000</td><td>378,844</td></tr><tr><td colspan="6">Total 79,987 399,988 799,990 2,000,163</td></tr></table>

## B.4 Benchmarks

We evaluate on the following benchmarks which are designed to test VLM’s spatial understanding ability.

EmbSpatial-Bench [20]: EmbSpatial-Bench is introduced to systematically evaluate and improve large vision-language models’ spatial understanding for embodied tasks, addressing the gap that most existing Visual Spatial Reasoning benchmarks are 2D, dataset-centric (e.g., COCO [38]/VG [34]), and object-centric rather than agent-centric, thus misaligned with real navigation and manipulation settings. To better reflect embodied scenarios, the authors construct EmbSpatial-Bench from 3D indoor environments in MP3D [7], AI2-THOR [33], and ScanNet [14], rendering egocentric RGB-D views and using camera parameters and 3D coordinates to automatically derive 2D bounding boxes and spatial relation triplets between objects. They define six fundamental relations (above, below, left, right, close, far) in the agent’s egocentric coordinate system to cover all three axes, convert these relations into multiple-choice QA pairs, and apply automatic filtering based on bounding box statistics followed by human verification to ensure object recognizability, relation correctness, and plausibility of distractor options.

CV-Bench [55]: CV-Bench (CV-Bench-2D, CV-Bench-3D) is a vision-centric multiple-choice benchmark built by repurposing standard vision datasets ADE20K [64], COCO [38], and Omni3D [6] into VQA-style examples that probe fundamental 2D and 3D understanding. All images come from these real-world datasets rather than a synthetic simulator, and each instance is manually inspected, resulting in 2,638 high-quality examples with naturallanguage questions and four-way answer choices. The 2D split focuses on classic perception skills such as spatial relationship reasoning and object counting, while the 3D split targets depth ordering and relative distance understanding derived from the rich 3D annotations of Omni3D.

<!-- page 25 of 43 -->

Why Far Looks Up

25

Table 7: BLINK performance with Wilson 95% confidence intervals. Point estimates (accuracy, %) are shown with Wilson 95% CIs for BLINK Rel. Depth (n=124) and Spat. Rel. (n=143). Bold marks the best point estimate within each column.

| Model R | el. Depth (n=124) | Spat. Rel. (n=143) |
| --- | --- | --- |
| Molmo-7B-O-0924 | 78.2 [70.2, 84.6] | 70.6 [62.7, 77.5] |
| + 80k | 72.6 [64.1, 79.7] | 60.8 [52.7, 68.5] |
| + 400k | 72.6 [64.1, 79.7] | 68.5 [60.5, 75.6] |
| + 800k | 75.0 [66.7, 81.8] | 61.5 [53.4, 69.1] |
| + 2M | 71.0 [62.4, 78.2] | 69.2 [61.2, 76.2] |
| NVILA-Lite-2B | 64.5 [55.8, 72.4] | 67.1 [59.1, 74.3] |
| + 80k | 53.2 [44.5, 61.8] | 74.1 [66.4, 80.6] |
| + 400k | 71.8 [63.3, 78.9] | 63.6 [55.5, 71.1] |
| + 800k | 57.3 [48.5, 65.6] | 65.0 [56.9, 72.4] |
| + 2M | 70.2 [61.6, 77.5] | 62.9 [54.8, 70.4] |
| RoboRefer-SFT-2B | 84.7 [77.3, 90.0] | 79.7 [72.4, 85.5] |
| Qwen2.5-VL-3B | 68.6 [59.9, 76.1] | 83.9 [77.0, 89.0] |
| + 80k | 58.1 [49.3, 66.4] | 79.7 [72.4, 85.5] |
| + 400k | 58.9 [50.1, 67.1] | 78.3 [70.9, 84.3] |
| + 800k | 58.1 [49.3, 66.4] | 79.0 [71.6, 84.9] |
| + 2M | 53.2 [44.5, 61.8] | 78.3 [70.9, 84.3] |
| Qwen3-VL-235B | 84.7 [77.3, 90.0] | 90.2 [84.2, 94.1] |

BLINK [23]: BLINK is a benchmark that repurposes 14 classic computer vision datasets into 3,807 visually prompted multiple-choice questions to assess fine-grained visual perception abilities such as relative depth estimation, spatial reasoning, visual correspondence, forensics detection, and multi-view understanding in multimodal large language models. The images span abstract diagrams, synthetic scenes, and real-world photographs, covering diverse settings from object-centric views to outdoor landscapes without relying on a single simulation engine, and each task is constructed by overlaying simple visual markers and natural-language questions on existing annotated datasets. Among its tasks, the relative depth (Rel. Depth) and spatial relation (Spat. Rel.) settings require models to compare which of two or more marked points is closer or to reason about geometric and positional relations between regions in the image, providing a focused probe of low-level 3D and spatial understanding beyond object recognition.

## B.5 BLINK Confidence Intervals

To contextualize performance differences on small BLINK subsets, we report Wilson 95% confidence intervals for the two BLINK splits used in Table 4: Rel. Depth (n=124) and Spat. Rel. (n=143).

<!-- page 26 of 43 -->

26

C. Min et al.

## C Additional Details on SpatialTunnel

This section presents additional details for SpatialTunnel, including scene setup, the VQA protocol, proprietary-model results, and the object-size variant.

## C.1 Scene Generation Details

All scenes in SpatialTunnel are rendered using Blender.

Object placement. The tunnel has a square cross-section of $2   \mathrm { m } \times 2   \mathrm { m }$ . Each scene contains two objects placed at different depths, with obj<sub>1</sub> always farther from the camera than obj<sub>2</sub>. To vary image-plane layout while preserving groundtruth depth, each object is swept independently over 16 discrete angular positions on the tunnel cross-section. Holding depth fixed while varying θ changes the object’s image-plane position without altering its distance from the camera.

This construction yields matched image pairs that differ only in 2D layout while preserving the underlying depth ordering.

Randomization. For each scene, we independently randomize the following factors:

– Shape. Each object is instantiated as either a sphere or a cube.

Color. Each object is assigned one of seven colors: red, green, blue, yellow, cyan, magenta, or black. Materials are implemented with a Principled BSDF shader. Surface roughness is sampled uniformly from [0.05, 1.0]. Objects are constrained to have distinct (color, shape) combinations.

– Size. In the phase-variation setting, the base sizes are $s _ { 1 } = 0 . 2$ for $\mathrm { o b j _ { 1 } }$ and $s _ { 2 } = 0 . 1$ for $\mathrm { o b j _ { 2 } } ,$ each multiplied by an independent random scale factor in [1.0, 1.5]. In the size-variation setting, sizes are controlled systematically as described in Section C.4.

– Lighting. We use a Nishita sky texture in Blender. The sun rotation is sampled uniformly from [1.25π, 1.75π] radians, and the background intensity is fixed to 0.15.

## C.2 VQA Protocol

Given a rendered image containing two objects $\mathrm { ( o b j _ { 1 } }$ and $\mathrm { o b j _ { 2 } ) }$ , the model is asked a binary depth-comparison question. To control for wording effects and answer-polarity bias, we instantiate four question templates per image, varying both the queried object order and the direction of comparison:

1. $^ { \cdot } I s$ the $\{ o b j _ { 1 } \}$ closer to the camera than the $\{ o b j _ { 2 } \} ?$ GT: No

2. $\mathit { `` I s ~ t h e ~ } \{ \mathit { o b j _ { 2 } } \}$ closer to the camera than the $\{ o b j _ { 1 } \} ?$ GT: Yes

3. $\mathit { `` I s ~ t h e ~ } \{ o b j _ { 2 } \}$ farther from the camera than the $\{ o b j _ { 1 } \} ? ''$

GT: No

4. “Is the $\{ o b j _ { 1 } \}$ farther from the camera than the $\{ o b j _ { 2 } \} ? ''$

GT: Yes

<!-- page 27 of 43 -->

Why Far Looks Up

27

For each joint angular configuration $( \theta _ { 1 } , \theta _ { 2 } )$ , we render 12 scene instances with independently randomized shape, color, size, and lighting, for a total of $1 6 \times 1 6 \times 1 2 = 3 { , } 0 7 2$ images. With four question templates per image, this yields 12,288 question-image pairs. Unless otherwise noted, responses are evaluated using the probability-based protocol described in Section 4.2. We then average the four template-level correctness scores to obtain a single score for each configuration cell.

## C.3 Proprietary Model Results

We additionally evaluate three proprietary configurations on SpatialTunnel: GPT-5.2 [42] in its default configuration, GPT-5.2 with reasoning enabled, and Gemini-2.5-Pro [24]. Because token-level logits are not exposed by the Azure API endpoints we tested, we evaluate models using final Yes/No outputs and report exact-match accuracy. We instantiate four question templates per image and average the resulting accuracies. Table 8 summarizes these proprietary results alongside the open-source baselines that are recomputed with exact-match accuracy.

Under the default setting, GPT-5.2 attains a mean exact-match accuracy of 0.613 , with $A c c _ { \mathrm { c o n } } = 0 . 6 7 3$ and $A c c _ { \mathrm { c t r } } = 0 . 5 5 2$ , yielding a gap of $\varDelta = 0 . 1 2 0$ . The positive gap indicates better performance on perspective-consistent cells than on counter cells, matching the directional bias observed in the open-source models.

Enabling reasoning improves GPT-5.2 from 0.613 to 0.953 mean accuracy and reduces the gap from $\varDelta   =   0 . 1 2 0$ to $\varDelta   =   0 . 0 5 8$ . Gemini-2.5-Pro also performs strongly, achieving 0.919 mean accuracy with a slightly negative gap, $\varDelta = - 0 . 0 2 8$ . Taken together, the proprietary-model results show that enabling reasoning in GPT-5.2 both improves exact-match accuracy and reduces the consistent-counter gap on SpatialTunnel, while Gemini-2.5-Pro likewise exhibits a near-zero gap.

## C.4 Extending the Analysis to Object Size

Section 4 showed that many VLMs rely on vertical image position as a shortcut for depth. We next examine whether the same failure mode extends to another cue, object size. In natural images, larger objects often appear closer to the camera. If a model relies on this cue, its depth judgments should degrade when relative size conflicts with the true depth ordering.

To test this, we construct a size-controlled variant of SpatialTunnel in which the two object sizes, denoted by $s _ { 1 }$ for $\mathrm { o b j _ { 1 } }$ and $s _ { 2 }$ for $\mathrm { o b j _ { 2 } } ,$ , are anti-correlated under the constraint $s _ { 1 }   +   s _ { 2 } \; = \; 0 . 4$ . Specifically, we sweep $s _ { 1 }$ over 10 equal intervals, yielding 11 values in total, and set $s _ { 2 } = 0 . 4 - s _ { 1 }$ . The object depths are held fixed throughout, with $\mathrm { o b j _ { 1 } }$ always farther from the camera than $\mathrm { o b j _ { 2 } }$ As $s _ { 1 }$ increases, the scene moves from a size-consistent regime, where the farther object is smaller, to a size-conflicting regime, where the farther object is larger than the nearer one.

<!-- page 28 of 43 -->

28

C. Min et al.

Table 8: Exact-match response accuracy on SpatialTunnel under final-output evaluation. All models in this table are scored using discrete Yes/No outputs and averaged over the four question templates. These values are therefore not directly comparable to the logit-based correctness scores v reported in Section 4.

| Model | $\mathbf { A c c } _ { \mathrm { a l l } } \mathbf { A c c } _ { \mathrm { c o n } } \mathbf { A c c } _ { \mathrm { c t r } }$ |  |  | Δ |
| --- | --- | --- | --- | --- |
| Molmo-7B |  |  |  |  |
| base | 0.581 | 0.718 | 0.434 | +0.284 |
| + 80k | 0.500 | 0.500 | 0.500 | +0.000 |
| + 400k | 0.502 | 0.693 | 0.308 | +0.384 |
| + 800k | 0.551 | 0.768 | 0.316 | +0.452 |
| + 2M | 0.731 | 0.795 | 0.673 | +0.121 |
| NVILA-Lite-2B |  |  |  |  |
| base | 0.468 | 0.510 | 0.419 | +0.091 |
| + 80k | 0.454 | 0.545 | 0.364 | +0.181 |
| + 400k | 0.794 | 0.971 | 0.607 | +0.364 |
| + 800k | 0.851 | 0.979 | 0.727 | +0.252 |
| + 2M | 0.977 | 0.998 | 0.955 | +0.043 |
| RoboRefer-2B-SFT | 0.919 | 0.956 | 0.879 | +0.076 |
| Qwen2.5-VL-3B |  |  |  |  |
| base | 0.493 | 0.653 | 0.342 | +0.310 |
| + 80k | 0.495 | 0.610 | 0.388 | +0.222 |
| + 400k | 0.498 | 0.576 | 0.423 | +0.153 |
| + 800k | 0.504 | 0.606 | 0.406 | +0.200 |
| + 2M | 0.506 | 0.681 | 0.335 | +0.346 |
| Qwen3-VL-235B | 0.929 | 0.958 | 0.909 | +0.050 |
| GPT-5.2 (default) | 0.613 | 0.673 | 0.552 | +0.120 |
| GPT-5.2 (reasoning) | 0.953 | 0.980 | 0.922 | +0.058 |
| Gemini-2.5-Pro | 0.919 | 0.905 | 0.933 | -0.028 |

Figure 8 shows a representative scene rendered under six $\left( s _ { 1 } , s _ { 2 } \right)$ configurations. As $s _ { 1 }$ increases from left to right, obj<sub>1</sub> grows while obj<sub>2</sub> shrinks correspondingly. At the left endpoint $(s_{1} = 0.10,\; s_{2} = 0.30)$ , apparent size agrees with the true depth ordering. At the right endpoint $(s_{1} = 0.30,\ s_{2} = 0.10)$ , the farther object appears substantially larger than the nearer one, which creates a strong cue that contradicts ground-truth depth.

We evaluate the same open-source VLMs as in Section 3 and report mean logit-based accuracy v over all configurations. Table 9 also reports accuracy at the two endpoints of the size sweep, $v _ { s _ { 1 } = 0 . 1 }$ and $v _ { s _ { 1 } = 0 . 3 }$ . We define the size-bias gap as

$$
\varDelta _ { s } = v _ { s _ { 1 } = 0 . 1 } - v _ { s _ { 1 } = 0 . 3 } ,
$$

which quantifies the size-distance entanglement. Larger positive values of $\varDelta _ { s }$ indicate stronger reliance on apparent size as a proxy for depth.

<!-- page 29 of 43 -->

Why Far Looks Up

29

![Image block](doc:bf97463/tier:basic/page:29/block:3)

Fig. 8: Object-size variation in SpatialTunnel. A representative scene rendered under six $\left( s _ { 1 } , s _ { 2 } \right)$ configurations with $s _ { 1 } + s _ { 2 } = 0 . 4 ,$ where $s _ { 1 }$ and $s _ { 2 }$ denote the sizes of $\mathrm { o b j _ { 1 } }$ and $\mathrm { o b j _ { 2 } } ,$ respectively. $\mathrm { o b j _ { 1 } }$ is always farther from the camera than $\mathrm { o b j } _ { 2 } .$ As $s _ { 1 }$ increases from left to right, the farther object grows while the nearer object shrinks, moving from a size-consistent to a size-conflicting configuration.

![Chart block](doc:bf97463/tier:basic/page:29/block:5)

Fig. 9: Correctness as a function of object size. Mean logit-based correctness, averaged over all question templates, as a function of obj<sub>1</sub> size (bottom axis) and $\mathrm { o b j _ { 2 } }$ size (top axis), with $s _ { 1 }   +   s _ { 2 } = 0 . 4$ . Each curve corresponds to one training-data variant. Molmo and NVILA show clear degradation as the farther object becomes larger than the nearer one, whereas Qwen remains near chance throughout, indicating weak depth reasoning in this setting.

Figure 9 plots mean correctness as a function of $s _ { 1 }$ (bottom axis) and $s _ { 2 } =$ $0 . 4 - s _ { 1 }$ (top axis) for each model family. Models that rely on the size cue show clear performance degradation as $s _ { 1 }$ increases, that is, as the farther object becomes larger than the nearer one. This confirms that apparent size acts as a confounding cue for depth, just as vertical position does in the main text.

Analysis. The object-size results closely mirror the vertical-distance entanglement results in the main text. In both settings, performance is higher when a simple cue agrees with the true depth ordering, and lower when that cue is put in conflict with depth.

1. Qwen is largely insensitive to size, but remains near chance. All Qwen variants cluster around chance performance $( v   \approx   0 . 5 0 )$ and exhibit negligible gaps $( | \varDelta _ { s } | < 0 . 0 2 )$ . This weak sensitivity should not be interpreted as robustness. Rather, it reflects limited depth discrimination in this setting.

2. Fine-tuning improves mean accuracy but often amplifies size-based shortcut reliance in Molmo and NVILA. For Molmo-7B (2M), mean accuracy rises to $v = 0 . 8 0 1$ , but the size-bias gap also grows to $\varDelta _ { s } = + 0 . 2 4 6$

<!-- page 30 of 43 -->

30

C. Min et al.

Table 9: Object-size variant results. Mean logit-based accuracy v across all $\left( s _ { 1 } , s _ { 2 } \right)$ configurations, accuracy at the two endpoints of the size sweep, and the size-bias gap $\varDelta _ { s }$ . Larger positive $\varDelta _ { s }$ indicates stronger reliance on apparent size as a depth cue.

<table><tr><td>Model</td><td>V</td><td> $\bf { v _ { s _ { 1 } = 0 . 1 } } \bf { v _ { s _ { 1 } = 0 . 3 } }$ </td><td></td><td> $\mathbf { \Delta _ { s } }$ </td></tr><tr><td>Qwen2.5-VL-3B</td><td></td><td></td><td></td><td></td></tr><tr><td>base</td><td>0.510</td><td>0.507</td><td>0.515</td><td>-0.008</td></tr><tr><td>+ 80k</td><td>0.513</td><td>0.511</td><td>0.516</td><td>-0.005</td></tr><tr><td>+ 400k</td><td>0.500</td><td>0.494</td><td>0.510</td><td>-0.016</td></tr><tr><td>+ 800k</td><td>0.498</td><td>0.489</td><td>0.505</td><td>-0.016</td></tr><tr><td>+ 2M</td><td>0.522</td><td>0.528</td><td>0.509</td><td>+0.018</td></tr><tr><td>Molmo-7B</td><td></td><td></td><td></td><td></td></tr><tr><td>base</td><td>0.589</td><td>0.609</td><td>0.553</td><td>+0.057</td></tr><tr><td>+ 80k</td><td>0.516</td><td>0.525</td><td>0.503</td><td>+0.022</td></tr><tr><td>+ 400k</td><td>0.605</td><td>0.656</td><td>0.532</td><td>+0.124</td></tr><tr><td>+ 800k</td><td>0.606</td><td>0.648</td><td>0.551</td><td>+0.097</td></tr><tr><td>+ 2M</td><td>0.801</td><td>0.888</td><td>0.643</td><td>+0.246</td></tr><tr><td>NVILA-Lite-2B</td><td></td><td></td><td></td><td></td></tr><tr><td>base</td><td>0.515</td><td>0.530</td><td>0.487</td><td>+0.043</td></tr><tr><td>+ 80k</td><td>0.526</td><td>0.555</td><td>0.482</td><td>+0.073</td></tr><tr><td>+ 400k</td><td>0.727</td><td>0.815</td><td>0.586</td><td>+0.229</td></tr><tr><td>+ 800k</td><td>0.686</td><td>0.779</td><td>0.557</td><td>+0.222</td></tr><tr><td>+ 2M</td><td>0.828</td><td>0.895</td><td>0.689</td><td>+0.207</td></tr><tr><td></td><td></td><td></td><td>0.743</td><td>+0.061</td></tr><tr><td colspan="5">RoboRefer-2B-SFT 0.804 0.804</td></tr></table>

Similarly, NVILA-Lite-2B (2M) reaches $v = 0 . 8 2 8$ with $\varDelta _ { s } = + 0 . 2 0 7$ . This is the same qualitative pattern observed for vertical position: fine-tuning improves aggregate performance, yet also increases sensitivity to a correlated cue.

3. RoboRefer mitigates size bias while retaining high accuracy. RoboRefer achieves $v   =   0 . 8 0 4 ,$ comparable to the strongest fine-tuned checkpoints, while exhibiting a much smaller gap $( \varDelta _ { s }   =   + 0 . 0 6 1 )$ . This mirrors the trend in the vertical-distance analysis, where RoboRefer also shows a relatively small gap, suggesting greater robustness to shortcut depth cues.

Taken together with the vertical-position intervention, the size-variation results show the same qualitative pattern: accuracy is higher when a simple cue agrees with the true depth ordering, and lower when that cue conflicts with depth. For the models we study, depth judgments therefore remain sensitive to multiple cues that often correlate with depth, including image height and apparent size. These cues can support performance when they are aligned with the underlying 3D layout, but accuracy may decline when that correlation is weakened or reversed. Thus, high average accuracy on depth queries should be interpreted with some caution, as it may not always reflect equally robust 3D spatial reasoning.

<!-- page 31 of 43 -->

Why Far Looks Up

31

## D Additional Details on Contrastive Probing

In this section, we detail the swap pair construction methodology for each spatial category, describe the layer selection methodology, examine the cross-domain consistency of distance coherence, and present full heatmap and PCA visualizations for all model families.

## D.1 Swap Pair Construction

For horizontal and vertical pairs, we construct minimal contrastive pairs by symmetrically swapping the two queried objects: the question “Is A to the left or right of B?” becomes “Is B to the left or right of $\mathrm { A } _ { \cdot \cdot } ^ { \mathrm { { \it 2 9 } } } ,$ , which flips the groundtruth label while keeping all other visual context fixed.

For distance (far/close) pairs, the construction differs due to the format of EmbSpatial-Bench depth questions, which are presented as four-choice questions rather than direct relational queries. We identify the target object from the correct answer option, and sample the reference object uniformly at random from the remaining distractor options. The original question asks whether the target is far or close relative to the reference; the swapped question reverses these roles. This yields the same label-flip structure as the horizontal and vertical cases, while adapting to the available annotation format.

## D.2 A Brief Illustration of VD-EI

VD-EI is positive when perspective-aligned pairs (above↔far, below↔close) are more similar than perspective-opposing pairs, and is near zero when the aligned and opposing terms largely cancel. In the extreme, VD-EI tends to be largest when aligned cosines are high while opposing cosines are negative (anti-aligned).

## D.3 Layer Selection Methodology

This section provides supplementary details on layer selection criteria and permodel justifications. The probing code extracts all layers; the user should select the appropriate L∗from the saved outputs.

Selection criteria. For each model family, we select a single representative layer L∗ at which to compute all probing metrics reported in the main text. The selection is guided by the following criteria, applied in order of priority (i.e., if criteria conflict, earlier criteria take precedence):

1. Axis coherence plateau. Coherence across all three spatial axes (horizontal, vertical, distance) is at or near its peak, indicating that stable axis-level structure has formed in the representation space.

<!-- page 32 of 43 -->

32

C. Min et al.

2. VD-EI stability. The VD-Entanglement Index should be at a meaningful plateau rather than in a transient region, ensuring that the selected layer captures the entanglement phenomenon we aim to analyze. When criteria conflict (e.g., VD-EI oscillates), we prioritize criterion (1) and select a layer from the shared high-coherence region across all three axes.

3. Avoidance of final layers. The selected layer should not fall in the last few layers of the model, as these tend to be optimized for next-token prediction rather than rich representational encoding.

Supporting evidence for intermediate-layer selection. The preference for intermediate layers over final layers is well supported by prior work across both language and vision-language models.

Spatial representations in LLMs have been shown to form in early layers and plateau around the model midpoint; for instance, layer 50 out of 80 (63%) in Llama-2-70b [56] was identified as the primary analysis layer for spatial probing [25]. A systematic study across 32 probing tasks further confirms that intermediate layers encode richer representations than final layers, which become increasingly specialized for output generation [49]. In the multi-modal setting, visual layer selection experiments demonstrate that middle CLIP-ViT [43] layers outperform deep layers for spatial reasoning; on CV-Bench [55], layer 18 out of 24 outperforms the penultimate layer by 3%, suggesting that vision-centric tasks such as spatial and positional reasoning benefit from mid-depth features rather than the deepest ones [10].

Per-model selection. We describe the layer selection rationale for each model family below, with specific coherence and VD-EI values drawn from the full layer-wise trajectories.

Molmo-7B-O-0924 (32 layers): L∗ = 23 (72%). Coherence across all three axes peaks in the L20–25 range, with horizontal ∼0.24, vertical ∼0.55, and distance ∼0.10 at L23. VD-EI reaches a plateau between L15 and L23 (0.5–0.7 for fine-tuned variants, 0.25 for base model), with L23 at the upper end of this range. PCA visualizations confirm clearer cluster separation at L23 compared to neighboring layers.

NVILA-Lite-2B (28 layers): L∗ = 20 (71%). Vertical coherence plateaus between L18 and L25 (0.40–0.60 for fine-tuned variants, 0.80 for RoboRefer). Horizontal coherence is stable across L15–27 (0.20–0.30 for fine-tuned variants, 0.65 for RoboRefer). Distance coherence peaks between L18 and L25 (0.03–0.05 for fine-tuned variants, 0.18 for RoboRefer). VD-EI plateaus at L17–26 (0.5–0.6 for fine-tuned variants), while RoboRefer shows around 0.25. PCA visualizations are shown at L=25, where cluster separation is more visually distinguishable.

Qwen2.5-VL-3B-Instruct (36 layers): $L ^ { * } = 2 8 ~ \left( 7 8 \% \right)$ . Horizontal coherence reaches ∼0.37 at L28. Vertical coherence is ∼0.35 at L28, with a peak of

<!-- page 33 of 43 -->

Why Far Looks Up

33

0.55 at L33. Distance coherence remains low at ${ \sim } 0 . 0 3 5$ . VD-EI peaks at L20–22 (0.50), dips around L25, and rebounds to 0.50 at L28. L28 balances meaningful entanglement with reasonable coherence while remaining below the outputspecialized final layers.

Qwen3-VL-235B-A22B-Instruct (94 layers): $L ^ { * }   =   8 7 \quad ( 9 3 \% )$ . Coherence across all axes forms very late in this model. Horizontal coherence peaks at L83–87 (0.65), vertical peaks at L87–90 (0.63), and distance peaks at L85–90 (0.23). VD-EI oscillates between 0.3 and 0.7 in the upper layers without a clear plateau. The selected depth of 93% notably exceeds the 71–75% range observed in smaller models, likely reflecting architectural differences: Qwen3- VL-235B-A22B-Instruct is a 94-layer Mixture-of-Experts model with 235B total parameters (22B active), which may delay the formation of stable spatial representations to later layers. We select $L ^ { * } = 8 7$ from the shared high-coherence region across all three axes, despite VD-EI oscillation in the upper layers. Importantly, the resulting cross-model $\mathrm { C o h _ { D } }$ ranking is robust to alternative valid layer choices (see below).

## D.4 Robustness to Alternative Layer Choices.

Our layer selection follows the predefined protocol above and is applied independently per model family before cross-model comparison. Candidate ranges are defined as the union of layers where $\mathrm { C o h _ { H } ,   C o h _ { V } }$ , and $\mathrm { C o h _ { D } }$ are near peak; when the criteria differ, we prioritize high axis coherence. To quantify sensitivity, we sample 1K random layers within each candidate range (without refitting) and recompute the cross-model Coh<sub>D</sub> ranking; the resulting rankings show high agreement with the reported ordering (Spearman $\rho = 0 . 9 2 8 )$ .

## D.5 Cross-Domain Consistency of Distance Coherence

Figure 10 compares $\mathrm { C o h } _ { D }$ computed on the synthetic dataset (SpatialTunnel) with that computed on EmbSpatial-Bench. The absolute values differ between the two domains, as Coh<sub>D</sub> in SpatialTunnel is generally higher; however, the relative ordering of models within each family is largely consistent.

For the NVILA family (highlighted in Figure 10), the ranking is preserved across both datasets: RoboRefer $\mathrm { >   N V I L A   2 M   >   N V I L A }$ 400k $\approx \mathrm { N V I L A }$ 800k $\mathrm { >   N V I L A   8 0 k   >   N V I L A }$ base. The Molmo family exhibits minor rank swaps among adjacent checkpoints, but the overall trend of increasing coherence with training scale, from 80k to 2M, is shared across domains. For the Qwen family, the Qwen2.5-VL-3B scale variants cluster tightly in both settings, while Qwen3-235B shows a markedly different profile.

These results suggest that the absolute magnitude of $\mathrm { C o h } _ { D }$ can be influenced by the environment, but it provides a reliable relative comparison when models are evaluated under the same data condition. We publicly release both evalua tion datasets and the probing pipeline so that future users can benchmark new models under identical conditions and compare against the values reported in

<!-- page 34 of 43 -->

34

C. Min et al.

![Chart block](doc:bf97463/tier:basic/page:34/block:3)

Fig. 10: Distance Coherence measured on synthetic (SpatialTunnel) vs. real (EmbSpatial-Bench) datasets. Gray bars denote Coh<sub>D</sub> computed on SpatialTunnel; red dots denote Coh<sub>D</sub> on EmbSpatial-Bench. Although the absolute magnitudes differ across domains, the relative ordering within each model family is largely preserved. The red box highlights the NVILA family, where the ranking is identical across both datasets.

this paper, enabling Coh<sub>D</sub> to serve as a reproducible measure of spatial representation quality.

## D.6 Heatmap and PCA Results

This section presents the cross-category similarity heatmaps and PCA visualizations for all model families. Similarity is computed as the cosine similarity between each category’s mean delta vector.

As shown in the heatmap results (Figure 11, 12, 13), the similarity between opposing categories on the same axis (e.g., left and right on the horizontal axis) is consistently near −1, indicating that the model encodes opposite spatial directions as antiparallel vectors in representation space. Additionally, similarity between horizontal and the other axes (i.e., vertical and distance) is close to zero, suggesting that the horizontal axis is encoded independently from both vertical and distance representations. However, between the vertical and distance axes, the perspective-aligned pairs above↔far and below↔close exhibit meaningful positive similarity in the range of 0.1–0.65 across all models. This confirms that vertical and distance representations are directionally coupled, consistent with the entanglement phenomenon described in the main text.

<!-- page 35 of 43 -->

Why Far Looks Up

35

![Chart block](doc:bf97463/tier:basic/page:35/block:3)

![Chart block](doc:bf97463/tier:basic/page:35/block:4)

![Chart block](doc:bf97463/tier:basic/page:35/block:5)

![Chart block](doc:bf97463/tier:basic/page:35/block:6)

![Chart block](doc:bf97463/tier:basic/page:35/block:7)

Fig. 11: Cross-category similarity heatmaps for the Molmo family. Each cell shows the cosine similarity between mean delta vectors of two categories. Variants range from vanilla (base Molmo-7B) to 2M (SFT with 2M-sample data mix).

<!-- page 36 of 43 -->

36

C. Min et al.

![Chart block](doc:bf97463/tier:basic/page:36/block:3)

![Chart block](doc:bf97463/tier:basic/page:36/block:4)

![Image block](doc:bf97463/tier:basic/page:36/block:5)

![Image block](doc:bf97463/tier:basic/page:36/block:6)

![Chart block](doc:bf97463/tier:basic/page:36/block:7)

![Chart block](doc:bf97463/tier:basic/page:36/block:8)

Fig. 12: Cross-category similarity heatmaps for the NVILA family. Variants include NVILA-Lite-2B from vanilla (base) through 2M (SFT), plus RoboRefer (RoboRefer-2B-SFT).

<!-- page 37 of 43 -->

Why Far Looks Up

37

![Chart block](doc:bf97463/tier:basic/page:37/block:3)

![Chart block](doc:bf97463/tier:basic/page:37/block:4)

![Chart block](doc:bf97463/tier:basic/page:37/block:5)

![Chart block](doc:bf97463/tier:basic/page:37/block:6)

![Chart block](doc:bf97463/tier:basic/page:37/block:7)

![Chart block](doc:bf97463/tier:basic/page:37/block:8)

Fig. 13: Cross-category similarity heatmaps for the Qwen family. Variants include Qwen2.5-VL-3B-Instruct (vanilla through 2M) and Qwen3-VL-235B-A22B-Instruct.

<!-- page 38 of 43 -->

38

C. Min et al.

![Chart block](doc:bf97463/tier:basic/page:38/block:3)

![Chart block](doc:bf97463/tier:basic/page:38/block:4)

![Chart block](doc:bf97463/tier:basic/page:38/block:5)

![Chart block](doc:bf97463/tier:basic/page:38/block:6)

![Chart block](doc:bf97463/tier:basic/page:38/block:7)

Fig. 14: 2D PCA of delta vectors for the Molmo family. Each point represents a per-sample delta vector, colored by spatial category. Opposing categories (e.g., left vs. right) separate along shared principal components, while far/close overlap with above/below, reflecting vertical-distance entanglement.

<!-- page 39 of 43 -->

Why Far Looks Up

39

![Chart block](doc:bf97463/tier:basic/page:39/block:3)

![Chart block](doc:bf97463/tier:basic/page:39/block:4)

![Chart block](doc:bf97463/tier:basic/page:39/block:5)

![Chart block](doc:bf97463/tier:basic/page:39/block:6)

![Chart block](doc:bf97463/tier:basic/page:39/block:7)

![Chart block](doc:bf97463/tier:basic/page:39/block:8)

Fig. 15: 2D PCA of delta vectors for the NVILA family. RoboRefer shows notably tighter distance-axis clusters (far/close) separated from vertical categories, consistent with its higher Coh<sub>D</sub> and lower VD-EI.

<!-- page 40 of 43 -->

40

C. Min et al.

![Chart block](doc:bf97463/tier:basic/page:40/block:3)

![Chart block](doc:bf97463/tier:basic/page:40/block:4)

![Chart block](doc:bf97463/tier:basic/page:40/block:5)

![Chart block](doc:bf97463/tier:basic/page:40/block:6)

![Chart block](doc:bf97463/tier:basic/page:40/block:7)

![Chart block](doc:bf97463/tier:basic/page:40/block:8)

Fig. 16: 2D PCA of delta vectors for the Qwen family. Variants include Qwen2.5-VL-3B-Instruct and Qwen3-VL-235B-A22B-Instruct. Qwen3-VL-235B exhibits markedly cleaner cluster separation across all three axes.

<!-- page 41 of 43 -->

Why Far Looks Up

41

![Chart block](doc:bf97463/tier:basic/page:41/block:3)

![Chart block](doc:bf97463/tier:basic/page:41/block:4)

![Chart block](doc:bf97463/tier:basic/page:41/block:5)

![Chart block](doc:bf97463/tier:basic/page:41/block:6)

![Chart block](doc:bf97463/tier:basic/page:41/block:7)

Fig. 17: 3D PCA of delta vectors for the Molmo family. A distinct distance axis does not clearly emerge, although delta vectors in the horizontal and vertical axes appear more well-clustered with data scaling.

<!-- page 42 of 43 -->

42

C. Min et al.

![Chart block](doc:bf97463/tier:basic/page:42/block:3)

![Chart block](doc:bf97463/tier:basic/page:42/block:4)

![Chart block](doc:bf97463/tier:basic/page:42/block:5)

![Chart block](doc:bf97463/tier:basic/page:42/block:6)

![Chart block](doc:bf97463/tier:basic/page:42/block:7)

![Chart block](doc:bf97463/tier:basic/page:42/block:8)

Fig. 18: 3D PCA of delta vectors for the NVILA family. RoboRefer’s distance clusters (far/close) occupy a distinct subspace from vertical categories, unlike the finetuned variants.

<!-- page 43 of 43 -->

Why Far Looks Up

43

![Image block](doc:bf97463/tier:basic/page:43/block:3)

Fig. 19: 3D PCA of delta vectors for the Qwen family. Variants include Qwen2.5-VL-3B-Instruct and Qwen3-VL-235B-A22B-Instruct. Qwen3-VL-235B shows clear three-way separation among horizontal, vertical, and distance axes in 3D space.