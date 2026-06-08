# Stream3D-VLM: Online 3D Spatial Understanding with Incremental Geometry Priors

Hanxun Yu1,2∗†, Xuan Qu1,2†, Lei Ke2, Boqiang Zhang2, Yuxin Wang2,3, Jianke Zhu1,4, and Dong Yu2

1Zhejiang University, 2Tencent Hunyuan, 3HKUST, 4Shenzhen Loop Area Institute

§ Project Page: https://stream3d-vlm.github.io/

Abstract. Despite advances in 3D scene understanding, existing 3D Large Multimodal Models operate in offline settings, requiring complete scene observations or predefined video clips. In this paper, we present an online 3D vision-language model that enables real-time spatial understanding from streaming video. Our approach adopts an autoregressive streaming control modeling based on the LLM’s next-token prediction objective to learn when to respond, and employs a lightweight Visual–Spatial Feature Integration (VSFI) module to incrementally inject temporally aligned geometry priors into the visual stream. To alleviate long-context decoding overhead, we propose a plug-and-play Geometry-Adaptive Voxel Compression (GAVC) module for efficient visual token compression. To address the scarcity of streaming 3D–language data, we further develop a scalable data generation pipeline that curates over 1M online spatio-temporal 3D QA pairs and establishes a comprehensive benchmark spanning 29 tasks. Extensive experiments show that our approach significantly outperforms both proprietary and open-source models across online and offline 3D spatial understanding, reasoning, and grounding tasks.

Keywords: 3D Vision-Language Models · Spatial Intelligence · Online Scene Understanding

## 1 Introduction

With the rapid progress of Multimodal Large Language Models (MLLMs) [2, 8, 23,36,51,53] in visual–language reasoning and interaction, growing attention has focused on extending them to more complex 3D vision tasks, such as autonomous robotics, AR/VR glasses, and embodied agents. The goal is to equip MLLMs with robust 3D spatial understanding, enabling them to function more effectively in real-world application scenarios. Early pioneering works [10, 17, 21, 32, 58] on 3D Large Multimodal Models rely on explicit 3D sensor inputs (e.g., point clouds, meshes, or depth maps) aligned with LLMs through instruction data. While effective at capturing geometric structure, their dependence on scarce 3D data prevents them from scaling to large-scale training, thereby limiting model capacity. Recent advances in feed-forward 3D reconstruction [22, 24, 30, 33–35, 54] enable methods [11,56] that require only RGB videos, injecting rich spatial information and scaling training on abundant 2D data. However, as shown in Figure 1, both paradigms remain strictly offline, requiring complete 3D scene observations or predefined video clips before interaction and task execution. In contrast, realworld embodied applications demand real-time interaction at arbitrary moments during video streaming.

![](images/18d64491b3639870773af00a43af4a980f36da0896e04449b8c9683b3dd4a573.jpg)  
(a) Previous 3D Large Multimodal Models

![](images/9cb110708e9e1f19a2349f8daab98aa8cc3c72776f911a46a2f69861a42ce9df.jpg)  
(b) Stream3D-VLM (ours)  
Fig. 1: Comparison between previous 3D LMMs and our Stream3D-VLM. (a) Previous methods typically operate in offline settings, requiring complete 3D observations or predefined video clips. (b) In contrast, our method enables real-time 3D spatial understanding and interaction on streaming video by incrementally integrating geometry priors from StreamVGGT [59].

Given prior efforts [6,31] in online 2D video understanding, which mainly target egocentric narration and forecasting, one might expect such models to extend to 3D tasks. However, 3D vision–language problems, including visual grounding and camera-related estimation, require deep reasoning over object-object and object-camera spatial relationships, i.e., the geometric structures inherent to the 3D world. Our experiments show that even with large-scale 3D–language fine-tuning, existing online 2D VLMs perform poorly on 3D tasks, highlighting the urgent need for a general-purpose 3D LMM capable of real-time interaction and powerful 3D spatial reasoning.

To address this challenge, we propose Stream3D-VLM, the first 3D vision–language model enabling online spatial understanding and interaction across multiple 3D–language tasks solely on streaming video. In this setting, the model is required to autonomously determine not only what to answer, but also when to respond, allowing natural and flexible interaction during continuous video input, as shown in Figure 2. Built upon the LLM’s native autoregressive training objective, we reformulate streaming control as a next-token prediction problem, enabling the model to learn response timing without degrading text generation quality. To support continuous 3D scene comprehension, we incrementally extract temporally aligned geometry priors by a streaming feed-forward 3D reconstruction model and inject them into the visual stream via a lightweight Visual–Spatial Feature Integration (VSFI) module. This design eliminates reliance on sparse 3D sensor data and supports scalable training on in-the-wild 2D videos. To handle long-term visual context during online inference, we introduce a plug-and-play Geometry-Adaptive Voxel Compression (GAVC) module that dynamically compresses visual tokens guided by spatial coordinates, preserving structural integrity while reducing redundancy and latency, enabling real-time deployment.

![](images/1284d48ed6e29c0d485518f459b65fb3858f4d5fdf56213aa1752019a4d3fab2.jpg)  
Fig. 2: Qualitative examples of Stream3D-VLM on streaming videos from ScanNet++ [50]. (a) Backward: The model aggregates historical visual-spatial features to precisely calculate the camera trajectory over a recent temporal window. (b) Realtime: The model performs metric reasoning to estimate the physical size of the coffee table in the current frame. (c) Forward: The model continuously monitors the video stream and responds proactively only when the target object becomes visible.

A key challenge in developing an online 3D spatial understanding model lies in the lack of large-scale streaming 3D–language data. Existing datasets for online 2D VLMs mainly target egocentric narration and activity recognition, offering limited supervision for 3D geometry and spatial relationships. To address this gap, we develop a scalable data generation pipeline to construct a large-scale online 3D spatio-temporal QA dataset with explicit timestamps, comprising over 1 million QA pairs across 5.2k videos. Furthermore, we introduce Stream3D-Bench, a novel benchmark consisting of 29 tasks over 518 videos, to systematically evaluate online 3D spatial understanding by measuring both model performance and temporal response accuracy.

Extensive experiments demonstrate that our approach achieves state-of-theart online 3D spatial understanding and reasoning, while maintaining leading performance on offline tasks such as visual grounding and dense captioning. We believe this work lays a fundamental step toward deploying 3D LMMs in real-world embodied applications. To summarize, our main contributions are threefold:

– We propose Stream3D-VLM, the first online 3D spatial understanding model solely on streaming video. By incrementally integrating geometry priors, it achieves effective 3D perception. A plug-and-play token compression module further reduces visual redundancy and inference latency, enabling real-time deployment.

– We develop a scalable data generation pipeline to curate online 3D spatiotemporal QA data with explicit timestamps for instruction tuning, addressing the scarcity of large-scale streaming 3D-language data.

– We establish Stream3D-Bench, a novel benchmark with 29 tasks across 518 videos to rigorously evaluate and advance online 3D spatial understanding.

## 2 Related Work

3D Large Multimodal Models. Build on advances in Large Language Models, recent works [12–15, 28, 37, 39, 44, 55] extend their capabilities to 3D data. Early methods [16, 29, 52] extract features from point clouds, meshes, or depth maps and align them with LLMs via adapters. While effective at capturing geometric structure, they rely on sparse and costly 3D data, limiting the model’s scalability and capacity. More recent methods [42,57] rely solely on RGB videos and integrate spatial cues by geometry-aware encoders from 3D reconstruction models [34, 35], eliminating reliance on explicit 3D sensors and enabling scalable training on 2D videos. However, both paradigms operate in offline settings, requiring complete 3D scenes or manually selected video clips, which diverges from real-world streaming scenarios. In this work, we propose an online 3D LMM framework that enables real-time interaction while maintaining strong 3D spatial understanding and reasoning, achieving leading performance across both online and offline 3D-language tasks.

Online Video Understanding. Driven by the growing real-time demands like AR glasses and autonomous robotics, recent research has advanced online video understanding, including action detection [45] and forecasting [38]. VideoLLMonline [6] pioneers general-purpose LLMs for streaming video dialogue, while VideoLLM-MoD [43] uses a mixture-of-depth strategy to scale resolution efficiently. StreamChat [26] and VideoChat-Online [18] retain key video tokens via dynamic memory banks, and TimeChat-Online [49] and StreamingAssistant [20] reduce temporal redundancy for faster decoding. Despite these advances, online 2D VLMs struggle with 3D tasks due to limited geometric and spatial understanding, emphasizing the need for a general-purpose online 3D LMM. By incrementally integrating geometry priors into the visual stream, Stream3D-VLM enables continuous and real-time 3D scene comprehension.

![](images/da3b26fccf7bc44d2d3dc967e95fe26733f4361886c132adff3ab182d9b99e1b.jpg)  
Fig. 3: Illustration of our data generation pipeline. Guided by a comprehensive task taxonomy spanning five cognitive competencies and three temporal interaction modes, the pipeline leverages detailed metadata from RGB-D video streams and a hybrid generation strategy to construct a large-scale spatio-temporal 3D QA dataset and the Stream3D-Bench for evaluating online 3D spatial understanding.

## 3 Streaming 3D-Language Data Generation

To endow our model to perceive, reason, and interact with 3D environments in an online setting, we introduce a scalable pipeline for generating large-scale spatiotemporal 3D-language data with explicit timestamps for instruction tuning. As shown in Figure 3, we construct a comprehensive dataset of over 1 million QA pairs across 29 tasks spanning 5.2k 3D scans. Additionally, we carefully curate a high-quality benchmark, Stream3D-Bench, to assess the online 3D spatial understanding and reasoning of MLLMs.

## 3.1 Task Taxonomy

We define streaming 3D vision-language tasks along two orthogonal dimensions: Cognitive Competencies and Temporal Interaction Modes, forming a structured taxonomy that captures the core capabilities of an online 3D assistant.

Cognitive Competencies. We organize tasks along a hierarchical spectrum of spatial perception, ranging from ego-centric motion to global scene structure and fine-grained object-level analysis. Ego-Motion Estimation captures the agent’s own motion (e.g., path length and rotation angles). Environment Measurement quantifies scene-level properties (e.g., room area and inter-object distances). Object–Camera Relationship records spatial relations between the agent and objects, including distance and direction. Object Chronology tracks objects over time, including counting, appearance order, and first/last observations. Object Attributes describe object-level features, such as category, color, and location.

Temporal Interaction Modes. We define three interaction modes based on the relative query-answer timing. Backward Tracing (Memory) retrieves information from past frames that is invisible in the current view, probing long-term memory capability. Realtime Perception (Observation) grounds responses in visual evidence from the current frame, emphasizing immediate spatial perception. Forward Response (Monitoring) entails asynchronous interaction, requiring the model to continuously monitor incoming frames and generate responses only when future conditions are satisfied.

## 3.2 Data Collection and Processing

Data Sources. Our pipeline builds on the train splits of three widely used 3D datasets: ScanNet [9], ScanNet++ [50], and ARKitScenes [4]. Specifically, we utilize their RGB video streams, depth maps, camera parameters, and 3D instance segmentation annotations. Captured in real-world environments, these data preserve realistic characteristics such as motion blur, illumination changes, and sensor noise.

Metadata Computing. A key challenge in streaming 3D data generation lies in accurately grounding 2D frames in precise 3D properties. Hence, we develop a Core Annotation Engine that extracts structured metadata per frame. Object Visibility is computed by projecting 3D geometry onto the image plane with depth-aware occlusion and temporally thresholding valid timestamps. In parallel, Camera Kinematics is derived by analyzing the camera trajectory over time, including cumulative path length, net displacement, and horizontal rotation. Geometric Measurement further captures spatial relationships like camera-object distances, relative azimuth, and global scene attributes. Further details on data generation are provided in the Supplementary Material.

## 3.3 Question-Answer Generation

Building on the task taxonomy and computed metadata, we adopt a hybrid strategy combining geometric precision and semantic richness to curate 1M+ QA pairs across 29 tasks.

Rule-based Generation. For Ego-Motion Estimation, Object–Camera Relationship, Environment Measurement, and Object Chronology, we employ a templatebased generator leveraging the computed metadata in Sec. 3.2. It processes timeseries signals and fills natural-language templates to generate QA pairs. For example, in Object Chronology tasks like Appearance Order, the generator extracts object entry timestamps and populates diverse linguistic templates.

QA Transfer with VLM Verification. For Object Attributes, we transfer existing offline QAs [1] into the streaming setting via temporal grounding, linking each object-centric query to the frame $t ^ { * }$ where the object is sufficiently visible. We then apply VLM-based verification to ensure answerability under streaming conditions (e.g., motion blur or partial occlusion). Specifically, we use GPT-5 [27] to validate the visual evidence at $t ^ { * }$ and discard samples lacking adequate visual support.

Table 1: Comparison of our proposed Stream3D-Bench with existing benchmarks focusing on spatial intelligence in 3D worlds.
<table><tr><td>Comparison</td><td>VSI-Bench OST-Bench</td><td></td><td>Stream3D-Bench</td></tr><tr><td># Task Types</td><td>8</td><td>15</td><td>29</td></tr><tr><td> $\# \ : \mathrm { Q A }$  Pairs</td><td>5k</td><td>10k</td><td>10k</td></tr><tr><td>Input Format</td><td>Video Clips</td><td>Video Clips</td><td>Streaming Video</td></tr><tr><td>Evaluation Granularity</td><td>Holistic</td><td>Holistic</td><td>Past /Present /Future</td></tr><tr><td>Response Timing Required</td><td>X</td><td>X</td><td></td></tr></table>

## 3.4 Stream3D-Bench

Overview. To evaluate MLLMs’ online 3D spatial understanding from streaming video, we construct Stream3D-Bench using our QA generation pipeline. It comprises 10,037 manually curated high-quality samples spanning 518 real videos sourced from the validation sets of ScanNet, ScanNet++, and ARKitScenes. Stratified sampling ensures balanced coverage of all five content categories and three temporal interaction modes (Sec. 3.1). As shown in Table 1, Stream3D-Bench significantly extends existing benchmarks such as VSI-Bench [46] and OST-Bench [25] in both task diversity and question complexity. More visualizations are provided in the Supplementary Material.

Metrics. Stream3D-Bench employs task-specific evaluation metrics. For Numerical Answers, we follow VSI-Bench and report Mean Relative Accuracy. For Multiple-Choice Answers, we use Exact Match. For Open-Ended Answers, we adopt an LLM-as-a-judge protocol with GPT-4o [19] to assess correctness. To evaluate the temporal precision inherent in streaming tasks, we introduce a new metric Answer-Timing Accuracy (AT A), measuring how well the predicted response time aligns with the GT timestamp. Let $t _ { \mathrm { p r e d } }$ denote the predicted response time and $t _ { \mathrm { g t } }$ represent the earliest answerable GT timestamp. With a delay penalty factor $\beta = 0 . 5$ , the timing score is defined as

$$
S ( t _ { \mathrm { p r e d } } ) = \mathbb { I } ( t _ { \mathrm { p r e d } } \geq t _ { \mathrm { g t } } ) \cdot \exp \bigl ( - \beta ( t _ { \mathrm { p r e d } } - t _ { \mathrm { g t } } ) \bigr )\tag{1}
$$

AT A is computed by averaging the timing score over all samples. We additionally report Time-To-First-Token (TTFT), end-to-end latency, and memory usage to enable a comprehensive evaluation of inference efficiency.

## 4 Stream3D-VLM Architecture

Our goal is to equip MLLM with online 3D spatial understanding and reasoning from streaming video, enabling diverse real-time visual interaction tasks. Figure 4 illustrates our framework. In this section, we first introduce a streaming decision learning strategy that utilizes the LLM’s native next-token prediction objective to model response timing after a user query. Next, we present the Visual-Spatial Feature Integration (VSFI) module to inject geometry priors into the visual stream for continuous 3D scene comprehension. Finally, we propose a plug-and-play Geometry-Adaptive Voxel Compression (GAVC) module that dynamically compresses visual tokens guided by spatial coordinates, preserving structural integrity while significantly reducing redundancy and latency during long-context online inference.

![](images/a8b0780e7e5a96c03fcdee16a44b090334e12e92b0fddfa3ea3babec2dbe9e5e.jpg)  
Fig. 4: Overview of our proposed Stream3D-VLM. Our pipeline processes ⊕ Addition Pipeline\_v2streaming video as a temporally ordered input sequence. We utilize the LLM’s native next-token prediction to jointly optimize a streaming control loss and the standard language modeling (LM) loss, enabling the model to learn when to respond or keep silent. We then suggest the VSFI module to inject temporally aligned geometric priors from a 3D reconstruction model into the visual stream. To mitigate long-context redundancy in online inference, we further propose a plug-and-play GAVC module that dynamically compresses visual tokens guided by 3D structure, enabling real-time deployment.

## 4.1 Autoregressive Streaming Mechanism

Existing 3D LMMs are primarily designed for offline inference on pre-segmented video clips. When deployed in streaming scenarios, they necessitate repeatedly reprocessing the entire historical context at each incoming frame to decide whether to respond, leading to excessive memory usage and high inference latency. This inefficiency severely limits their applicability in real-time settings. Streaming Control Modeling. Inspired by recent advances in online 2D video LLMs [6, 43], we reformulate streaming control as a next-token prediction problem. Specifically, we exploit the LLM’s native autoregressive objective to enable the model to learn when to answer—skipping redundant frames and triggering responses only when necessary, rather than generating outputs at every time step. To this end, we introduce two special decision tokens, <SEP> and <END>, which indicate whether the model should continue ingesting visual inputs or stop and initiate response generation, as illustrated below:

$$
\begin{array} { r l } & { \underbrace { \mathrm { U S E R : } < \mathrm { i m g } > \mathrm { Q u e r y } } _ { \mathrm { C o n t e x t ~ H i s t o r y } } \underbrace { < \mathrm { i m g } > < \mathrm { S E P } > < \mathrm { i m g } > < \mathrm { S E P } > } _ { \mathrm { S t r e a m i n g ~ C o n t i n u a t i o n } } } \\ & { \underbrace { < \mathrm { i m g } > < \mathrm { E N D } > } _ { \mathrm { R e s p o n s e ~ T r i g g e r } } \underbrace { \texttt { A S S I S T A N T : } < \mathrm { t x t } > . . . < \mathrm { t x t } > } _ { \mathrm { R e s p o n s e ~ G e n e r a t i o n } } } \end{array}\tag{2}
$$

Joint Training Objective. To unify streaming control and language generation within a single autoregressive framework, we modify the training label mask to supervise these decision tokens explicitly. The overall training objective is a weighted sum of the streaming decision loss $\mathcal { L } _ { \mathrm { s t r e a m } }$ and the standard language modeling loss $\mathcal { L } _ { \mathrm { L M } } \mathrm { : }$

$$
\begin{array} { r l } & { \displaystyle \mathcal { L } _ { \mathrm { s t r e a m } } = \frac { 1 } { | \mathcal { D } | } \sum _ { t \in \mathcal { D } } \mathrm { C E } _ { t } , \quad \mathcal { L } _ { \mathrm { L M } } = \frac { 1 } { | \mathcal { T } | } \sum _ { t \in \mathcal { T } } \mathrm { C E } _ { t } , } \\ & { \mathrm { C E } _ { t } \equiv \mathrm { C E } \big ( p _ { \theta } ( y _ { t } \mid y _ { < t } ) , y _ { t } \big ) , ~ \mathcal { L } = \lambda \mathcal { L } _ { \mathrm { s t r e a m } } + \mathcal { L } _ { \mathrm { L M } } , } \end{array}\tag{3}
$$

where D is the set of streaming decision tokens $y _ { t } \in \{ < \mathrm { S E P } > , < \mathrm { E N D } > \}$ , T contains all remaining tokens, CE(·) denotes the cross-entropy loss, $p _ { \theta }$ is the model’s next-token prediction distribution, and λ balances the two objectives. During inference, the model decides whether to respond at each frame via next token prediction (<SEP> or <END>).

## 4.2 Visual-Spatial Feature Integration

To endow MLLMs with spatial understanding without explicit 3D inputs, we use latent geometry from feed-forward 3D reconstruction models, enabling scalable training on large-scale 2D videos. Specifically, we adopt StreamVGGT [59] to incrementally extract geometry priors from streaming video in online settings. Latent Geometric Encoding. Given an incoming RGB frame $I _ { t } \in \mathbb { R } ^ { 3 \times H \times W }$ we first extract 2D visual tokens $\mathbf { H } _ { t } ^ { \mathrm { 2 D } } \in \mathbb { R } ^ { N \times D ^ { v i s } }$ via the MLLM’s native vision encoder. In parallel, StreamVGGT’s spatial encoder produces latent geometry tokens $\mathbf { G } _ { t } \in \mathbf { \bar { \mathbb { R } } } ^ { K \times D ^ { j e o } }$ along with a camera token $\mathbf { c } _ { t } \doteq \mathbb { R } ^ { 1 \times D ^ { g e o } }$ encoding global camera and scene information. We concatenate these tokens and project them into the LLM embedding space via a lightweight two-layer MLP:

$$
\mathbf { H } _ { t } ^ { \mathrm { 3 D } } = \mathrm { M L P } ( [ \mathbf { c } _ { t } \ ; \mathbf { G } _ { t } ] ) \in \mathbb { R } ^ { ( K + 1 ) \times D ^ { v i s } } .\tag{4}
$$

Cross-Attention Fusion. To inject geometric priors into the semantic stream, we treat $\mathbf { H } _ { t } ^ { \mathrm { 2 D } }$ as queries and $\mathbf { H } _ { t } ^ { \mathrm { 3 D } }$ as keys and values. A stack of cross-attention blocks yields geometry-enhanced visual tokens $\mathbf { H } _ { t } ^ { \mathrm { f } } \in \mathbb { R } ^ { N \times D ^ { v i s } }$

$$
\mathbf { H } _ { t } ^ { \mathrm { f } } = \mathrm { s o f t m a x } \left( \frac { ( W _ { Q } \mathbf { H } _ { t } ^ { \mathrm { 2 D } } ) ( W _ { K } \mathbf { H } _ { t } ^ { \mathrm { 3 D } } ) ^ { \top } } { \sqrt { d _ { k } } } \right) ( W _ { V } \mathbf { H } _ { t } ^ { \mathrm { 3 D } } ) ,\tag{5}
$$

where $W _ { Q } , W _ { K }$ , and $W _ { V }$ are learnable projections and $d _ { k }$ denotes the key dimension. A residual connection is applied to preserve the original semantics as $\mathbf { H } _ { t } ^ { \mathrm { f } }  \mathbf { H } _ { t } ^ { \mathrm { f } } + \mathbf { H } _ { t } ^ { \mathrm { 2 D } }$

## 4.3 Geometry-Adaptive Voxel Compression

After visual-spatial feature integration, we introduce a plug-and-play module that dynamically compresses visual tokens guided by spatial coordinates, effectively reducing long-context visual redundancy during online inference.

3D Voxel Construction. For each incoming frame $I _ { t } \in \mathbb { R } ^ { 3 \times H \times W }$ , we first use StreamVGGT’s prediction heads to estimate the depth map $D _ { t } \in \mathbb { R } ^ { H \times W }$ and camera intrinsics and extrinsics $\left( \mathbf { K } _ { t } , \mathbf { E } _ { t } \right)$ . Given the 2D patch coordinates $( u _ { j } , v _ { j } )$ in $I _ { t } .$ , each patch is back-projected to 3D position:

$$
{ \bf p } _ { t , j } = { \bf E } _ { t } ^ { - 1 } \left( D _ { t } ( u _ { j } , v _ { j } ) { \bf K } _ { t } ^ { - 1 } [ u _ { j } , v _ { j } , 1 ] ^ { \top } \right) .\tag{6}
$$

The 2D tokens are then lifted into spatially-aware 3D voxels via sinusoidal positional encoding: $\mathbf { v } _ { t , j } = \mathbf { H } _ { t , j } ^ { \mathrm { f } } + \mathrm { P E } ( \mathbf { p } _ { t , j } )$

Dynamic Clustering. To enable dynamic compression while preserving the inherent 3D structure, the newly constructed voxels at time t are collected as:

$$
\mathcal { V } _ { t } = \{ ( \mathbf { v } _ { t , j } , \mathbf { p } _ { t , j } ) \} _ { j = 1 } ^ { N } .\tag{7}
$$

We then apply spatial K-Means clustering to adaptively partition the combined voxel set into K clusters:

$$
\{ \mathcal { C } _ { k } \} _ { k = 1 } ^ { K } = \mathrm { K M e a n s } ( \{ \mathbf { p } \ | \ ( \mathbf { v } , \mathbf { p } ) \in \mathcal { V } _ { t } \} , K ) ,\tag{8}
$$

where each cluster $\mathcal { C } _ { k }$ groups spatially proximal voxels in 3D space, enabling structure-aware dynamic compression. The clustering process is executed in parallel on the GPU, incurring negligible additional latency.

Dual-Attention Aggregation. Within each cluster $\mathcal { C } _ { k } .$ , voxel features are aggregated via a dual-attention mechanism that models both feature similarity and spatial proximity. The cluster center’s feature and coordinate are computed as:

$$
\bar { \mathbf { v } } _ { k } = \frac { 1 } { | \mathcal { C } _ { k } | } \sum _ { j \in \mathcal { C } _ { k } } \mathbf { v } _ { j } , \ \bar { \mathbf { p } } _ { k } = \frac { 1 } { | \mathcal { C } _ { k } | } \sum _ { j \in \mathcal { C } _ { k } } \mathbf { p } _ { j } .\tag{9}
$$

Each voxel $\mathbf { v } _ { j } \in \mathcal { C } _ { k }$ is assigned feature and spatial weights:

$$
s _ { j } ^ { \mathrm { f } } = \cos ( \mathbf { v } _ { j } , \bar { \mathbf { v } } _ { k } ) , \ s _ { j } ^ { \mathrm { p } } = \exp \left( - \| \mathbf { p } _ { j } - \bar { \mathbf { p } } _ { k } \| ^ { 2 } / 2 \sigma _ { k } ^ { 2 } \right) ,\tag{10}
$$

where $\sigma _ { k }$ is the voxel distance standard deviation to the center. The two are combined as $w _ { j } = \alpha s _ { i } ^ { \mathrm { f } } + ( 1 - \alpha ) s _ { i } ^ { \mathrm { p } }$

The aggregated feature for cluster k is $\begin{array} { r } { \mathbf { v } _ { k } ^ { \prime } = \sum _ { j \in \mathcal { C } _ { k } } w _ { j } \mathbf { v } _ { j } } \end{array}$ . These compressed voxels $\{ \mathbf { v } _ { k } ^ { \prime } \} _ { k = 1 } ^ { K }$ , encoding both rich semantics and spatial cues, are then fed to the LLM with prompt tokens for efficient autoregressive generation.

## 5 Experiments

## 5.1 Experimental Setup

Datasets and Benchmarks. To evaluate our model across online and offline 3D-language tasks, we train it on the curated 1M+ streaming 3D spatio-temporal

Table 2: Evaluation results on Stream3D-Bench. Stream3D-VLM consistently outperforms all competing models, delivering the most accurate response timing and the lowest inference latency. NA/MCA/OEA denote numerical, multiple-choice, and open-ended answers, respectively. FT indicates that the model is fine-tuned on our curated online 3D spatio-temporal QA dataset for fair comparison. Bold and underlined values indicate the best and the second-best results, respectively. The results are reported under a 1fps streaming video setting.
<table><tr><td rowspan="2">Methods Avg.</td><td rowspan="2"></td><td colspan="7">Backward TracingRealtime Perception Forward Response|</td><td rowspan="2">Answer-</td><td rowspan="2">TTFT Timing Acc.Latency</td><td rowspan="2">End2EndMemory Latency</td><td rowspan="2">Usage</td><td rowspan="2">Image ↓Resolution</td></tr><tr><td></td><td>NAMCA</td><td></td><td>NA MCA</td><td>OEA</td><td>NA MCA</td><td>OEA</td></tr><tr><td>Proprietary Models (API)</td><td></td><td></td><td>OEA</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-40</td><td>28.0</td><td>10.536.3</td><td>24.9</td><td>31.1 34.8</td><td></td><td>51.1</td><td>10.2 29.9</td><td>23.0</td><td>55.9%</td><td></td><td>二</td><td>二</td><td>1296×968</td></tr><tr><td>GPT-5</td><td>35.0</td><td>18.744.1</td><td>33.6</td><td>38.046.2</td><td></td><td>52.8</td><td>12.9 30.6</td><td>37.7</td><td>61.7%</td><td></td><td></td><td></td><td>1296×968</td></tr><tr><td>Open-source Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLaVA-Video-7B</td><td>20.4</td><td>12.7 34.6</td><td>11.4</td><td>21.6 28.6</td><td></td><td>38.0</td><td>10.2 20.3</td><td>6.5</td><td>55.5%</td><td>650ms</td><td>2.84s</td><td>25.8G</td><td>1152×768</td></tr><tr><td>InternVL3-8B</td><td>24.1</td><td>22.1 34.4</td><td>6.8</td><td>30.7 31.2</td><td></td><td>40.0</td><td>18.3 27.2</td><td>5.9</td><td>19.8%</td><td>1034ms</td><td>5.65s</td><td>40.5G</td><td>1024×765</td></tr><tr><td>InternVL3.5-8B</td><td>27.0</td><td>21.4 36.2</td><td>17.6</td><td>28.2 38.2</td><td></td><td>42.4</td><td>25.2 25.8</td><td>8.2</td><td>23.0%</td><td>1771ms</td><td>6.72s</td><td>43.1G</td><td>1024×765</td></tr><tr><td>Qwen2.5-VL-7B</td><td>22.5</td><td>11.6 36.3</td><td>13.6</td><td>23.1 33.2</td><td></td><td>39.4</td><td>11.3 28.4</td><td>5.5</td><td>19.2%</td><td>325ms</td><td>3.16s</td><td>40.0G</td><td>1008×784</td></tr><tr><td>Qwen2.5-VL-32B</td><td>29.3</td><td>14.1 40.3</td><td>35.7</td><td>28.7 37.6</td><td></td><td>43.2</td><td>22.4 30.2</td><td>11.5</td><td>35.2%</td><td>560ms</td><td>5.87s</td><td>91.6G</td><td>1008×784</td></tr><tr><td>Qwen3-VL-8B</td><td>19.0</td><td>2.533.2</td><td>20.2</td><td>16.0 33.4</td><td></td><td>28.4</td><td>4.624.7</td><td>8.3</td><td>61.7%</td><td>150ms</td><td>1.56s</td><td>28.8G</td><td>1024×768</td></tr><tr><td>Qwen3-VL-32B</td><td>23.1</td><td>4.934.2</td><td>35.9</td><td>19.4 32.0</td><td></td><td>31.8</td><td>7.329.2</td><td>13.5</td><td>66.4%</td><td>282ms</td><td>3.22s</td><td>65.5G</td><td>1024×768</td></tr><tr><td>Streaming Dialogue VLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VideoLLM-online-8B (FT)</td><td>34.6</td><td>28.8 40.9</td><td>27.8</td><td>31.3 37.9</td><td></td><td>44.6</td><td>38.2 32.7</td><td>29.2</td><td>70.2%</td><td>120ms</td><td>1.29s</td><td>28.4G</td><td>384×384</td></tr><tr><td>Qwen2.5-VL-7B (FT)</td><td>47.8</td><td>45.656.6</td><td>45.4</td><td>45.744.6</td><td></td><td>48.7</td><td>46.6 57.4</td><td>39.8</td><td>73.1%</td><td>106ms</td><td>0.53s</td><td>36.8G</td><td>504×392</td></tr><tr><td>Stream3D-VLM-4B</td><td>54.6</td><td>58.1 61.2</td><td>45.2</td><td>58.857.3</td><td></td><td>52.8</td><td>55.360.6</td><td>41.9</td><td>75.4%</td><td>43ms</td><td>0.24s</td><td>20.7G</td><td>504×392</td></tr><tr><td>Stream3D-VLM-8B</td><td>58.8</td><td>59.8 67.9</td><td>50.5</td><td>61.4 61.5</td><td></td><td>54.3</td><td>58.1 66.7</td><td>49.4</td><td>86.7%</td><td>62ms</td><td>0.39s</td><td>36.6G</td><td>504×392</td></tr></table>

QA pairs, alongside the VSI-590K dataset [48] including real videos from the train splits of S3DIS, ScanNet, ScanNet++, ARKitScenes, and Aria Digital Twin. For evaluation, we employ Stream3D-Bench for online spatial understanding, VSI-Bench [46] for offline spatial reasoning, and downstream 3D scene understanding tasks, including ScanRefer [5] for visual grounding, ScanQA [1] for question answering, and Scan2Cap [7] for dense captioning.

Implementation Details. Our model is based on Qwen2.5-VL-3B/7B [3] and integrates StreamVGGT-1B [59] as the spatial encoder to supply incremental 3D geometry priors temporally aligned with the streaming video input. We train our model in a unified end-to-end multi-task instruction tuning paradigm for a single epoch on the mixed dataset. We adopt the AdamW optimizer with a weight decay of 0.03, using randomly sampled data at each step. During training, the MLLM visual encoder and spatial encoder are frozen, while the proposed VSFI module and LLM backbone are fully trainable.

Inference Pipeline. During online inference, the video is input as a frame-byframe stream with a default 1 FPS. Our model ingests each frame sequentially and generates tokens on-the-fly. In the whole process, KV caching is employed to accelerate decoding, implicitly reusing previously generated tokens without explicit concatenation across frames.

## 5.2 Main Results

Evaluation on Stream3D-Bench. Since existing offline models cannot natively process streaming videos, we provide the full video and specify the query time, prompting joint prediction of response timing and answers. We also finetune an online 2D VLM [6] and Qwen2.5-VL-7B with streaming decision learning on the curated 3D QA pairs as baselines. As shown in Table 2, our method achieves strong spatial understanding and temporal reasoning across three categories, significantly outperforming both proprietary and open-source models. Evaluation on VSI-Bench. To examine whether streaming prediction affects offline performance, we evaluate our method on VSI-Bench. As shown in Table 3, our 8B model achieves the highest average accuracy of 65.9%, outperforming both open-source general models and specialized spatial reasoning models. Notably, the 4B variant reaches 55.2% accuracy, surpassing much larger 72B models and the proprietary model Gemini-2.5 Pro [8]. These results demonstrate our method’s strong 3D spatial understanding and reasoning, even in offline settings. Evaluation on ScanQA, ScanRefer, and Scan2Cap. We also evaluate Stream3D-VLM on other 3D scene understanding tasks—question answering, visual grounding, and dense captioning in Table 4. Despite using online videos without explicit 3D data, our method outperforms others across all tasks and metrics. The 4B variant remains competitive, exceeding many larger methods.

Table 3: Evaluation results on VSI-Bench. Despite being designed for streaming scenarios, Stream3D-VLM also performs well across all subtasks of the offline spatial perception and reasoning benchmark, significantly surpassing both commercial and open-source models.
<table><tr><td rowspan="2"></td><td rowspan="2">Methods OnlinelAvg. ↑</td><td rowspan="2"></td><td colspan="4">Numerical Answer</td><td colspan="4">Multiple-Choice Answer</td></tr><tr><td>Obj. Cnt.Abs.Dist. Obj. Size Room Size Rel. Dist.Rel. Dir.Route Plan Appr. Order</td><td colspan="2"></td><td></td><td colspan="4"></td></tr><tr><td>Proprietary Models (API)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-40</td><td>X</td><td>34.0</td><td>46.2</td><td>5.3</td><td>43.8</td><td>38.2</td><td>37.0</td><td>41.3</td><td>31.5</td><td>28.5</td></tr><tr><td>Gemini-1.5 Pro</td><td>×</td><td>45.4</td><td>56.2</td><td>30.9</td><td>64.1</td><td>43.6</td><td>51.3</td><td>46.3</td><td>36.0</td><td>34.6</td></tr><tr><td>Gemini-2.5Pro</td><td>×</td><td>51.5</td><td>43.8</td><td>34.9</td><td>64.3</td><td>42.8</td><td>61.1</td><td>47.8</td><td>45.9</td><td>71.3</td></tr><tr><td>Open-source Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LongVILA-8B</td><td>X</td><td>21.6</td><td>29.1</td><td>9.1</td><td>16.7</td><td>0.0</td><td>29.6</td><td>30.7</td><td>32.5</td><td>25.5</td></tr><tr><td>VILA-1.5-40B</td><td>X</td><td>31.2</td><td>22.4</td><td>24.8</td><td>48.7</td><td>22.7</td><td>40.5</td><td>25.7</td><td>31.5</td><td>32.9</td></tr><tr><td>Qwen2.5-VL-7B</td><td>X</td><td>33.0</td><td>40.9</td><td>14.8</td><td>43.4</td><td>10.7</td><td>38.6</td><td>38.5</td><td>33.0</td><td>29.8</td></tr><tr><td>Qwen2.5-VL-72B</td><td>X</td><td>37.0</td><td>25.1</td><td>29.3</td><td>54.5</td><td>38.8</td><td>38.2</td><td>37.0</td><td>34.0</td><td>28.9</td></tr><tr><td>LLaVA-OneVision-72B</td><td>X</td><td>40.2</td><td>43.5</td><td>23.9</td><td>57.6</td><td>37.5</td><td>42.5</td><td>39.9</td><td>32.5</td><td>44.6</td></tr><tr><td>LLaVA-NeXT-Video-72B</td><td>X</td><td>40.9</td><td>48.9</td><td>22.8</td><td>57.4</td><td>35.3</td><td>42.4</td><td>36.7</td><td>35.0</td><td>48.6</td></tr><tr><td>Spatial Reasoning Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SpaceR-7B</td><td>X</td><td>45.5</td><td>57.8</td><td>28.2</td><td>59.9</td><td>47.1</td><td>40.1</td><td>45.4</td><td>33.5</td><td>52.1</td></tr><tr><td>Spatial-MLLM-4B</td><td>X ×</td><td>48.4</td><td>65.3</td><td>34.8</td><td>63.1</td><td>45.1</td><td>41.3</td><td>46.2</td><td>33.5</td><td>46.3</td></tr><tr><td>VG LLM-8B</td><td>×</td><td>50.7</td><td>67.9</td><td>37.7</td><td>58.6</td><td>62.0</td><td>46.6</td><td>40.7</td><td>32.4</td><td>59.2</td></tr><tr><td>VLM-3R-8B</td><td></td><td>60.9</td><td>70.2</td><td>49.4</td><td>69.2</td><td>67.1</td><td>65.4</td><td>80.5</td><td>45.4</td><td>40.1</td></tr><tr><td>Stream3D-VLM-4B</td><td></td><td>55.2</td><td>68.6</td><td>38.4</td><td>65.9</td><td>56.4 68.7</td><td>49.9</td><td>54.3</td><td>42.3</td><td>66.0</td></tr><tr><td>Stream3D-VLM-8B</td><td></td><td>65.9</td><td>72.4</td><td>50.1</td><td>71.5</td><td></td><td>70.8</td><td>73.3</td><td>46.2</td><td>73.8</td></tr></table>

Table 4: Evaluation results on ScanQA, ScanRefer, and Scan2Cap. Stream3D-VLM, while operating in the online setting, excels in traditional 3D scene understanding tasks, including Question Answering, Visual Grounding, and Dense Captioning.
<table><tr><td rowspan="2">Methods Onlinel</td><td rowspan="2"></td><td colspan="5">ScanQA</td><td colspan="2">ScanRefer</td><td colspan="4">Scan2Cap@0.50</td></tr><tr><td>BLEU-4METEOR ROUGE CIDEr EM Acc@0.25 Acc@0.50 BLEU-4 METEOR ROUGE CIDEr</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Task-Specific Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ScanQA</td><td>X</td><td>10.1</td><td>13.1</td><td>33.3</td><td>64.921.0</td><td></td><td>一</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ScanRefer</td><td>X</td><td>1</td><td>一</td><td></td><td></td><td></td><td>37.3</td><td>24.3</td><td></td><td></td><td></td><td></td></tr><tr><td>Scan2Cap</td><td>X</td><td></td><td></td><td></td><td>一</td><td></td><td></td><td>二</td><td>22.4</td><td>21.4</td><td>43.5</td><td>35.2</td></tr><tr><td>3D-Vista</td><td>X</td><td>13.1</td><td>15.2</td><td>38.6</td><td>76.6</td><td>27.0</td><td>50.6</td><td>45.8</td><td>34.0</td><td>27.1</td><td>54.3</td><td>66.9</td></tr><tr><td>3D/2.5D-Input Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>3D-LLM</td><td>X</td><td>12.0</td><td>14.5</td><td>35.7</td><td>69.4</td><td>20.5</td><td>30.3</td><td></td><td>8.1</td><td>13.1</td><td>33.2</td><td>二</td></tr><tr><td>LEO</td><td>X</td><td>11.5</td><td>16.2</td><td>39.3</td><td>80.0</td><td>24.5</td><td></td><td></td><td>38.2</td><td>27.9</td><td>58.1</td><td>72.4</td></tr><tr><td>Inst3D-LMM</td><td>X</td><td>14.9</td><td>18.4</td><td>42.6</td><td>88.6</td><td>24.6</td><td>57.8</td><td>51.6</td><td>38.3</td><td>27.5</td><td>57.2</td><td>79.7</td></tr><tr><td>LLaVA-3D</td><td>X</td><td>14.5</td><td>20.7</td><td>49.6</td><td>91.7</td><td>27.0</td><td>54.1</td><td>42.4</td><td>41.1</td><td>30.2</td><td>63.4</td><td>79.2</td></tr><tr><td>Video-3D LLM</td><td>X</td><td>16.2</td><td>19.8</td><td>49.0</td><td>102.1</td><td>30.1</td><td>58.1</td><td>51.7</td><td>40.2</td><td>28.5</td><td>61.7</td><td>80.0</td></tr><tr><td>Only Video-Input Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SPAR-8B</td><td>X</td><td>15.3</td><td></td><td></td><td>90.7</td><td>27.7</td><td>48.8</td><td>43.1</td><td></td><td></td><td></td><td></td></tr><tr><td>Spatial-MLLM-4B</td><td>X</td><td>14.8</td><td>18.4</td><td>45.0</td><td>91.8</td><td>26.3</td><td>49.3</td><td>44.2</td><td>37.8</td><td>27.3</td><td>57.8</td><td>76.5</td></tr><tr><td>VG LLM-8B</td><td>X</td><td>16.0</td><td>18.0</td><td>44.2</td><td>98.6</td><td>27.3</td><td>54.4</td><td>47.9</td><td>40.1</td><td>28.4</td><td>62.0</td><td>76.4</td></tr><tr><td>Stream3D-VLM-4B</td><td></td><td>16.6 17.8</td><td>19.5</td><td>47.9</td><td>102.6</td><td>29.2 104.530.9</td><td>56.5 58.4</td><td>51.9 52.5</td><td>41.6 42.8</td><td>28.8 31.0</td><td>63.5</td><td>78.8</td></tr><tr><td>Stream3D-VLM-8B</td><td></td><td></td><td>21.0</td><td>50.2</td><td></td><td></td><td></td><td></td><td></td><td></td><td>64.2</td><td>81.2</td></tr></table>

Table 5: Ablation of streaming loss design. We explore different loss functions and weighting factors to identify an optimal trade-off across multiple evaluation metrics under the online input setting.
<table><tr><td rowspan="2">Configs</td><td colspan="3">Stream3D-Bench</td><td rowspan="2">Answer- Timing Acc.1 →</td><td rowspan="2">End2End Latency</td></tr><tr><td>NA ↑</td><td>MCA ↑</td><td>OEA ↑</td></tr><tr><td colspan="6">Loss Function (fixed 入= 2.0)</td></tr><tr><td>Focal Loss</td><td>58.1</td><td>63.2</td><td>49.8</td><td>77.4%</td><td>0.52s</td></tr><tr><td> Standard CE</td><td>59.8</td><td>65.4</td><td> 51.4</td><td>86.7%</td><td>0.39s</td></tr><tr><td colspan="6">Weighting Factor (using Standard CE Loss)</td></tr><tr><td>入=1.0</td><td>59.6</td><td>65.0</td><td>53.3</td><td>80.3%</td><td>0.42s</td></tr><tr><td>入 = 2.0</td><td>59.8</td><td>65.4</td><td>51.4</td><td>86.7%</td><td>0.39s</td></tr><tr><td>入=3.0</td><td>57.0</td><td>63.8</td><td>50.2</td><td>89.2%</td><td>0.40s</td></tr></table>

Table 6: Ablation study on the impact of 3D geometry prior modeling in the VSFI module. We compare the full Stream3D-VLM model with a fine-tuned Qwen2.5-VL-7B baseline and with variants that remove individual components.
<table><tr><td rowspan="2">Settings</td><td colspan="3">Stream3D-Bench</td><td colspan="2">VSI-Bench</td></tr><tr><td>NA ↑</td><td>MCA ↑</td><td>OEA ↑ 1</td><td>NA ↑</td><td>MCA ↑</td></tr><tr><td>Baseline (Visual Only)</td><td>46.0</td><td>52.9</td><td>44.6</td><td>42.9</td><td>46.2</td></tr><tr><td>w/o Camera Tokens w/o Geometry Tokens</td><td>55.4</td><td>60.4</td><td>46.3</td><td>60.5</td><td>61.4</td></tr><tr><td rowspan="2">Fusion: Addition</td><td>52.9</td><td>58.6</td><td>50.2</td><td>56.1</td><td>55.7</td></tr><tr><td>57.6</td><td>65.8</td><td>49.2</td><td>63.5</td><td>63.4</td></tr><tr><td>Fusion: Concat.+ MLP</td><td>53.5</td><td>60.3</td><td>50.1</td><td>60.8</td><td>62.1</td></tr><tr><td> Full Model (Cross-Attn.)</td><td>59.8</td><td>65.4</td><td>51.4</td><td>65.7</td><td>66.0</td></tr></table>

## 5.3 Ablation Studies

Ablation Analysis of Streaming Loss Configurations. We explore different loss functions and weighting factors for streaming prediction to balance performance and efficiency. Table 5 shows that standard cross-entropy and focal loss perform similarly. However, over-weighting the streaming loss harms overall performance, while underweighting reduces response time accuracy. Optimal results occur at a streaming-to-generation loss weight ratio of 2.0.

Effects of Visual-Spatial Feature Integration (VSFI). Table 6 evaluates strategies for combining latent geometric features. Using either camera information or geometry tokens alone improves over the baseline, while combining both performs best. Moreover, a fusion design with stacked cross-attention blocks and skip connections outperforms patch-level addition or MLP-based concatenation. Impacts of Geometry-Adaptive Voxel Compression (GAVC). Table 7 compares GAVC with other token compression baselines such as random pruning, average pooling, and VisionZip [47]. Unlike methods based on semantic redundancy or attention scores that ignore 3D structure, GAVC dynamically updates voxels guided by spatial properties, better handling irregular voxel distributions. As shown in Figure 5, GAVC maintains competitive accuracy with substantially reduced latency, even at a 25% token retention ratio.

Table 7: Comparison of token compression strategies at a 50% retention ratio. Unlike baseline methods that rely solely on semantic redundancy or attention scores, GAVC performs spatially guided dynamic updates, thereby preserving geometric consistency under streaming video inputs.
<table><tr><td rowspan="2">Methods</td><td colspan="2">Stream3D-Bench</td><td colspan="4">ScanQA</td></tr><tr><td colspan="4">[NA ↑ MCA ↑ OEA ↑|B-4 ↑ ROUGE ↑ CIDEr ↑ EM ↑</td></tr><tr><td colspan="7">Geometry-Unaware 1 Baselines 8.5</td></tr><tr><td>Random</td><td>35.6</td><td>40.2</td><td>36.8</td><td>32.6 41.6</td><td>72.4 94.5</td><td>18.8</td></tr><tr><td>Avg. Pooling</td><td>47.8</td><td>52.9</td><td>45.2</td><td>11.4 13.0</td><td></td><td>24.0</td></tr><tr><td>VisionZip</td><td>49.2</td><td>53.8</td><td>41.6</td><td>43.8</td><td>87.2</td><td>24.4</td></tr><tr><td colspan="7">Geometry-Adaptive Compression</td></tr><tr><td>GAVC (Ours)</td><td>59.8</td><td>65.4</td><td> 51.4</td><td>17.8 50.2</td><td>104.5</td><td>30.9</td></tr></table>

![](images/60dd2d0e210fd0147c629202e82876bc16d9778ba96d77402ca2546b1b78015c.jpg)

![](images/fabafd81f2fd7a38ac534b141adbf86e3728b4414831a727d51893341ff86291.jpg)  
Fig. 5: Ablation study of the token retention ratio in the GAVC module. We analyze the trade-off between performance and efficiency by reporting response accuracy, answer-timing accuracy, TTFT, and end-to-end latency on Stream3D-Bench.

## 6 Conclusion

In this paper, we propose Stream3D-VLM, the first online 3D spatial understanding model solely on streaming video. We reformulate streaming control as a next-token prediction problem, enabling the model to learn when to respond or keep silent. For continuous 3D scene comprehension, we introduce a VSFI module that incrementally injects temporally aligned geometric priors into the visual stream. We further propose a GAVC module to dynamically compress visual tokens guided by 3D structure, reducing long-context redundancy during online inference. Extensive experiments demonstrate that our model achieves leading performance across diverse online and offline 3D-language tasks.

## References

1. Azuma, D., Miyanishi, T., Kurita, S., Kawanabe, M.: Scanqa: 3d question answering for spatial scene understanding. In: proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 19129–19139 (2022)

2. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025)

3. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al.: Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923 (2025)

4. Baruch, G., Chen, Z., Dehghan, A., Dimry, T., Feigin, Y., Fu, P., Gebauer, T., Joffe, B., Kurz, D., Schwartz, A., et al.: Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. arXiv preprint arXiv:2111.08897 (2021)

5. Chen, D.Z., Chang, A.X., Nießner, M.: Scanrefer: 3d object localization in rgb-d scans using natural language. In: European conference on computer vision. pp. 202–221. Springer (2020)

6. Chen, J., Lv, Z., Wu, S., Lin, K.Q., Song, C., Gao, D., Liu, J.W., Gao, Z., Mao, D., Shou, M.Z.: Videollm-online: Online video large language model for streaming video. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18407–18418 (2024)

7. Chen, Z., Gholami, A., Nießner, M., Chang, A.X.: Scan2cap: Context-aware dense captioning in rgb-d scans. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3193–3203 (2021)

8. Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al.: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261 (2025)

9. Dai, A., Chang, A.X., Savva, M., Halber, M., Funkhouser, T., Nießner, M.: Scannet: Richly-annotated 3d reconstructions of indoor scenes. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5828–5839 (2017)

10. Deng, J., He, T., Jiang, L., Wang, T., Dayoub, F., Reid, I.: 3d-llava: Towards generalist 3d lmms with omni superpoint transformer. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 3772–3782 (2025)

11. Fan, Z., Zhang, J., Li, R., Zhang, J., Chen, R., Hu, H., Wang, K., Qu, H., Wang, D., Yan, Z., et al.: Vlm-3r: Vision-language models augmented with instruction-aligned 3d reconstruction. arXiv preprint arXiv:2505.20279 (2025)

12. Fu, R., Liu, J., Chen, X., Nie, Y., Xiong, W.: Scene-llm: Extending language model for 3d visual reasoning. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 2195–2206. IEEE (2025)

13. Hong, Y., Zhen, H., Chen, P., Zheng, S., Du, Y., Chen, Z., Gan, C.: 3d-llm: Injecting the 3d world into large language models. Advances in Neural Information Processing Systems 36, 20482–20494 (2023)

14. Hu, W., Lin, J., Long, Y., Ran, Y., Jiang, L., Wang, Y., Zhu, C., Xu, R., Wang, T., Pang, J.: G2vlm: Geometry grounded vision language model with unified 3d reconstruction and spatial reasoning. arXiv preprint arXiv:2511.21688 (2025)

15. Huang, J., Ma, X., Linghu, X., Fan, Y., He, J., Tan, W., Li, Q., Zhu, S.C., Chen, Y., Jia, B., et al.: Leo-vl: Towards 3d vision-language generalists via data scaling with efficient representation. arXiv preprint arXiv:2506.09935 (2025)

16. Huang, J., Yong, S., Ma, X., Linghu, X., Li, P., Wang, Y., Li, Q., Zhu, S.C., Jia, B., Huang, S.: An embodied generalist agent in 3d world. In: Proceedings of the 41st International Conference on Machine Learning. pp. 20413–20451 (2024)

17. Huang, X., Wu, J., Xie, Q., Han, K.: 3drs: Mllms need 3d-aware representation supervision for scene understanding. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025)

18. Huang, Z., Li, X., Li, J., Wang, J., Zeng, X., Liang, C., Wu, T., Chen, X., Li, L., Wang, L.: Online video understanding: Ovbench and videochat-online. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 3328–3338 (2025)

19. Hurst, A., Lerer, A., Goucher, A.P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al.: Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024)

20. Jin, X., Yu, H., Yu, B., Liu, K., Liu, J., Tao, K., Pei, Y., Wang, H., Dang, F., Liu, J., et al.: Streamingassistant: Efficient visual token pruning for accelerating online video understanding. arXiv preprint arXiv:2512.12560 (2025)

21. Kang, W., Huang, H., Shang, Y., Shah, M., Yan, Y.: Robin3d: Improving 3d large language model via robust instruction tuning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 3905–3915 (2025)

22. Lan, Y., Luo, Y., Hong, F., Zhou, S., Chen, H., Lyu, Z., Yang, S., Dai, B., Loy, C.C., Pan, X.: Stream3r: Scalable sequential 3d reconstruction with causal transformer. arXiv preprint arXiv:2508.10893 (2025)

23. Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., et al.: Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326 (2024)

24. Lin, H., Chen, S., Liew, J., Chen, D.Y., Li, Z., Shi, G., Feng, J., Kang, B.: Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647 (2025)

25. Lin, J., Zhu, C., Xu, R., Mao, X., Liu, X., Wang, T., Pang, J.: Ost-bench: Evaluating the capabilities of mllms in online spatio-temporal scene understanding. arXiv preprint arXiv:2507.07984 (2025)

26. Liu, J., Yu, Z., Lan, S., Wang, S., Fang, R., Kautz, J., Li, H., Alvare, J.M.: Streamchat: Chatting with streaming video. arXiv preprint arXiv:2412.08646 (2024)

27. OpenAI.: Gpt-5 system card (2025), https://cdn.openai.com/gpt-5-systemcard.pdf

28. Ouyang, K., Liu, Y., Wu, H., Liu, Y., Zhou, H., Zhou, J., Meng, F., Sun, X.: Spacer: Reinforcing mllms in video spatial reasoning. arXiv preprint arXiv:2504.01805 (2025)

29. Qi, Z., Zhang, Z., Fang, Y., Wang, J., Zhao, H.: Gpt4scene: Understand 3d scenes from videos with vision-language models. arXiv preprint arXiv:2501.01428 (2025)

30. Su, Z., Ye, W., Feng, H., Fan, K., Zhang, J., Yu, D., Liu, Z., Wong, N.: Xstreamvggt: Extremely memory-efficient streaming vision geometry grounded transformer with kv cache compression. arXiv preprint arXiv:2601.01204 (2026)

31. Tang, H., Zhang, C., Jin, M., Yu, Q., Wang, Z., Jin, X., Zhang, Y., Du, M.: Time series forecasting with llms: Understanding and enhancing model capabilities. ACM SIGKDD Explorations Newsletter 26(2), 109–118 (2025)

32. Wang, H., Zhao, Y., Wang, T., Fan, H., Zhang, X., Zhang, Z.: Ross3d: Reconstructive visual instruction tuning with 3d-awareness. arXiv preprint arXiv:2504.01901 (2025)

33. Wang, H., Zhou, H., Liu, H., Yan, L.: 4d-vggt: A general foundation model with spatiotemporal awareness for dynamic scene geometry estimation. arXiv preprint arXiv:2511.18416 (2025)

34. Wang, J., Chen, M., Karaev, N., Vedaldi, A., Rupprecht, C., Novotny, D.: Vggt: Visual geometry grounded transformer. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 5294–5306 (2025)

35. Wang, Q., Zhang, Y., Holynski, A., Efros, A.A., Kanazawa, A.: Continuous 3d perception model with persistent state. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 10510–10522 (2025)

36. Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., et al.: Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265 (2025)

37. Wang, X., Li, Z., Xu, Y., Qi, J., Yang, Z., Ma, R., Liu, X., Zhang, C.: Spatial 3d-llm: Exploring spatial awareness in 3d vision-language models. In: 2025 IEEE International Conference on Multimedia and Expo (ICME). pp. 1–6. IEEE (2025)

38. Wang, X., Feng, M., Qiu, J., Gu, J., Zhao, J.: From news to forecast: Integrating event analysis in llm-based time series forecasting with reflection. Advances in Neural Information Processing Systems 37, 58118–58153 (2024)

39. Wang, Y., Ke, L., Zhang, B., Qu, T., Yu, H., Huang, Z., Yu, M., Xu, D., Yu, D.: N3d-vlm: Native 3d grounding enables accurate spatial reasoning in visionlanguage models. arXiv preprint arXiv:2512.16561 (2025)

40. Wei, H., Tang, H., Jia, X., Wang, Z., Yu, H., Li, Z., Satoh, S., Van Gool, L., Wang, Z.: Physical adversarial attack meets computer vision: A decade survey. IEEE Transactions on Pattern Analysis and Machine Intelligence 46(12), 9797– 9817 (2024)

41. Wei, H., Yu, H., Zhang, K., Wang, Z., Zhu, J., Wang, Z.: Moiré backdoor attack (mba): A novel trigger for pedestrian detectors in the physical world. In: Proceedings of the 31st ACM International Conference on Multimedia. pp. 8828–8838 (2023)

42. Wu, D., Liu, F., Hung, Y.H., Duan, Y.: Spatial-mllm: Boosting mllm capabilities in visual-based spatial intelligence. arXiv preprint arXiv:2505.23747 (2025)

43. Wu, S., Chen, J., Lin, K.Q., Wang, Q., Gao, Y., Xu, Q., Xu, T., Hu, Y., Chen, E., Shou, M.Z.: Videollm-mod: Efficient video-language streaming with mixtureof-depths vision computation. Advances in Neural Information Processing Systems 37, 109922–109947 (2024)

44. Xu, R., Wang, X., Wang, T., Chen, Y., Pang, J., Lin, D.: Pointllm: Empowering large language models to understand point clouds. In: European Conference on Computer Vision. pp. 131–147. Springer (2024)

45. Yan, T., Zeng, W., Xiao, Y., Tong, X., Tan, B., Fang, Z., Cao, Z., Zhou, J.T.: Crossglg: Llm guides one-shot skeleton-based 3d action recognition in a crosslevel manner. In: European Conference on Computer Vision. pp. 113–131. Springer (2024)

46. Yang, J., Yang, S., Gupta, A., Han, R., Fei-Fei, L., Xie, S.: Thinking in Space: How Multimodal Large Language Models See, Remember and Recall Spaces. In: CVPR (2025)

47. Yang, S., Chen, Y., Tian, Z., Wang, C., Li, J., Yu, B., Jia, J.: Visionzip: Longer is better but not necessary in vision language models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 19792–19802 (2025)

48. Yang, S., Yang, J., Huang, P., Brown, E., Yang, Z., Yu, Y., Tong, S., Zheng, Z., Xu, Y., Wang, M., Lu, D., Fergus, R., LeCun, Y., Fei-Fei, L., Xie, S.: Cambrian-s: Towards spatial supersensing in video. arXiv preprint arXiv:2511.04670 (2025)

49. Yao, L., Li, Y., Wei, Y., Li, L., Ren, S., Liu, Y., Ouyang, K., Wang, L., Li, S., Li, S., et al.: Timechat-online: 80% visual tokens are naturally redundant in streaming videos. In: Proceedings of the 33rd ACM International Conference on Multimedia. pp. 10807–10816 (2025)

50. Yeshwanth, C., Liu, Y.C., Nießner, M., Dai, A.: Scannet++: A high-fidelity dataset of 3d indoor scenes. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 12–22 (2023)

51. Yu, H., Li, W., Qu, X., Wang, S., Chen, J., Zhu, J.: Visiontrim: Unified vision token compression for training-free mllm acceleration. arXiv preprint arXiv:2601.22674 (2026)

52. Yu, H., Li, W., Wang, S., Chen, J., Zhu, J.: Inst3d-lmm: Instance-aware 3d scene understanding with multi-modal instruction tuning. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 14147–14157 (2025)

53. Yu, H., Qu, X., Wang, Y., Zhu, J., Ke, L.: Unlocking dense metric depth estimation in vlms. arXiv preprint arXiv:2605.15876 (2026)

54. Yuan, S., Yang, Y., Yang, X., Zhang, X., Zhao, Z., Zhang, L., Zhang, Z.: Infinitevggt: Visual geometry grounded transformer for endless streams. arXiv preprint arXiv:2601.02281 (2026)

55. Zhang, J., Chen, Y., Zhou, Y., Xu, Y., Huang, Z., Mei, J., Chen, J., Yuan, Y.J., Cai, X., Huang, G., et al.: From flatland to space: Teaching vision-language models to perceive and reason in 3d. arXiv preprint arXiv:2503.22976 (2025)

56. Zhao, R., Zhang, Z., Xu, J., Chang, J., Chen, D., Li, L., Sun, W., Wei, Z.: Spacemind: Camera-guided modality fusion for spatial reasoning in vision-language models. arXiv preprint arXiv:2511.23075 (2025)

57. Zheng, D., Huang, S., Li, Y., Wang, L.: Learning from videos for 3d world: Enhancing mllms with 3d vision geometry priors. arXiv preprint arXiv:2505.24625 (2025)

58. Zhu, C., Wang, T., Zhang, W., Pang, J., Liu, X.: Llava-3d: A simple yet effective pathway to empowering lmms with 3d capabilities. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4295–4305 (2025)

59. Zhuo, D., Zheng, W., Guo, J., Wu, Y., Zhou, J., Lu, J.: Streaming 4d visual geometry transformer. arXiv preprint arXiv:2507.11539 (2025)

## Supplementary Material

In this part, we provide more details and additional experimental results on our approach. The supplementary material is organized as follows:

• § A: Metadata computing details;

• § B: Detailed data generation pipeline;

• § C: Stream3D-1M dataset statistics;

• § D: Stream3D-Bench details;

• § E: More visualization results;

• § F: Evaluation prompts for offline models.

## A Metadata Computing Details

This section details the computational logic for the metadata introduced in Section 3.2 of the main paper.

## A.1 Visibility and Occlusion Reasoning

To determine the visibility of an object instance $O _ { i }$ at frame t, we project its 3D mesh vertices $V _ { i }$ onto the 2D image plane via the camera intrinsic matrix K and the extrinsic pose $E _ { t }$ . Let $\mathbf { p } _ { p r o j } ~ = ~ ( u , v )$ denote the projected pixel coordinates of a vertex $\mathbf { x } \in V _ { i } .$ , and $d _ { \mathbf { x } }$ be its corresponding depth in the camera coordinate system. We determine the vertex-level visibility $\mathcal { V } _ { t } ( { \bf x } )$ by enforcing geometric consistency between the projected vertex depth and the sensor depth map $D _ { t } ( u , v )$

$$
\mathcal { V } _ { t } ( \mathbf { x } ) = \mathbb { I } \left( \left| d _ { \mathbf { x } } - D _ { t } ( u , v ) \right| < \tau _ { o c c } \right) \cdot \mathbb { I } \left( 0 \leq u < W , 0 \leq v < H \right) ,\tag{11}
$$

where $\mathbb { I } ( \cdot )$ represents the indicator function, and $\tau _ { o c c }$ denotes the occlusion tolerance threshold (empirically set to 0.05m).

We quantify the object visibility score for the current frame as the ratio of the number of currently visible vertices to the maximum number of visible vertices observed across the entire video sequence, inspired by recent work [39, 55]. Finally, to obtain a binary visibility label, we threshold this ratio using values adapted to specific tasks.

## A.2 Camera Kinematics Computation

We analyze the geometric properties of the camera pose trajectory, denoted as $\scriptstyle \{ { \mathbf { P } } _ { t } \} _ { t = 0 } ^ { T } .$ . Let $\mathbf { t } _ { t } \in \mathbb { R } ^ { 3 }$ represent the camera center (translation) and $\mathbf { R } _ { t } \in S O ( 3 )$ represent the orientation at frame t. We compute the following kinematic metrics:

• Path Length: Calculated as the cumulative Euclidean distance traversed by the camera center across consecutive frames:

$$
L = \sum _ { i = 1 } ^ { T } \| \mathbf t _ { i } - \mathbf t _ { i - 1 } \| _ { 2 } .\tag{12}
$$

• Displacement: Defined as the Euclidean distance between the initial and final camera positions:

$$
\varDelta = \| \mathbf { t } _ { T } - \mathbf { t } _ { 0 } \| _ { 2 } .\tag{13}
$$

• Direction: Defined as the azimuthal angle (i.e., clock position) of the final position relative to the initial pose. This is computed by projecting the displacement vector $\left( \mathbf { t } _ { T } - \mathbf { t } _ { 0 } \right)$ onto the local coordinate system of the initial camera $\mathbf { P } _ { 0 }$

• Rotation: Defined as the variation in horizontal orientation between the start and end frames. It is obtained by calculating the signed angle between the optical axes (forward vectors) of $\mathbf { R } _ { T }$ and $\mathbf { R } _ { 0 }$ projected onto the horizontal plane.

## A.3 Geometric Measurement

Leveraging the dense point clouds derived from 3D instance segmentation, we compute the following metrics to characterize the geometric properties of the scene and objects:

• Object-Camera Relationship: We analyze the egocentric spatial relation of an object with respect to the camera. Specifically, the Direction (discretized as clock positions) is computed based on the object centroid in the camera’s local coordinate system. The Distance is defined as the minimum Euclidean distance between the camera center and the set of points belonging to the object instance.

• Inter-Object Distance: To quantify the spatial separation between two objects, we compute the minimum Euclidean distance between their respective 3D point sets. This calculation is efficiently accelerated using a KD-Tree structure to query the nearest neighbor points.

• Object Size: We estimate the spatial extent of an object by constructing an Oriented Bounding Box (OBB) around its point cloud. The dimensions are parameterized by the length, width, and height of this bounding volume.

• Room Area: The approximate area of the room is derived from the global scene point cloud. We project the scene’s bounding box onto the horizontal plane and calculate the product of its length and width.

## B Detailed Data Generation Pipeline

## B.1 Rule-based Generation Templates

We constructed diverse linguistic templates for each rule-based task.

• Ego-Motion Estimation: Questions focus on monitoring future motion (Forward) or recalling past motion (Backward). Example: “How much total distance has the camera traveled in the past {N} seconds?”

• Object–Camera Relationship: Covers Distance, Direction (e.g., “to your left"), and Location (Clock position + Distance). Example: “Where is the {Object} relative to me right now?” (Realtime) vs. “I can’t see the {Object} anymore. Where is it relative to me?” (Backward).

• Environment Measurement: Includes Object Size, Room Area, and Interobject Distance. Example: What is the shortest distance between the {Object\_A} and the {Object\_B} in meters?

• Object Chronology: Involves counting unique instances, determining appearance order, and timestamps. Example: “How many {Object\_Class} have you seen so far?” or “Did you see the {Object\_A} before or after the $\{ O b j e c t \_ B \} ? ^ { \prime \prime }$

Table A1 presents the complete collection of question templates for all 29 tasks, hierarchically structured according to three temporal interaction modes and five cognitive categories.

## B.2 QA Transfer with VLM Verification

For semantic tasks (Attribute, Position, Recognition), we adapt static ScanQA data.

1. Temporal Grounding: We align the static question to the video timeline by identifying the frame t∗ where the target object is most clearly visible (highest visibility ratio).

2. VLM Verification: We employ GPT-5 [27] with the prompt: “You are an expert in visual question answering verification. Please examine whether the provided image frame contains sufficient visual evidence to support the given Question and Answer pair. You must assess if the target object is clearly perceivable. Return ‘1‘ if the answer can be strictly inferred from the visual content, and ‘0‘ otherwise.” to ensure the question is answerable from the visual stream alone, filtering out questions relying on unobservable context.

## B.3 Construction of Target Object Whitelist

When constructing streaming 3D QA pairs for object-centric tasks, we observed that many object labels are noisy, ambiguous, or otherwise unsuitable for meaningful question generation. To ensure high-quality QA, we employed GPT-5 [27] to rigorously curate a Target Object Whitelist from the full label sets of ScanNet and ScanNet++. We adopted a unified high-precision filtering strategy based on three core criteria:

1. Specific Identity: Retain concrete object classes $( e . g .$ , chair, monitor) while discarding broad hypernyms (e.g., furniture, electronics) to ensure semantic precision.

Tab-A1: Taxonomy and question templates of Stream3D-Bench. The benchmark comprises 29 tasks, hierarchically structured according to temporal interaction modes and cognitive categories. Highlighted variables are dynamically instantiated for each scene.
<table><tr><td>Mode|</td><td>Category</td><td>Specific Task</td><td>Question Template /Example</td></tr><tr><td rowspan="8">Paaralarereeg</td><td rowspan="4">Ego-Motion</td><td>Cam.Direction</td><td>Where is the current camera location relative to its position {n} secondsago? What is the total path length covered by the camera during the</td></tr><tr><td>Cam.Path</td><td>last {n} seconds?</td></tr><tr><td>Cam.Rotation</td><td>What was the camera&#x27;s horizontal rotation angle over the past {n} seconds?</td></tr><tr><td>Cam.Displacement</td><td>How far has the camera moved during the last {n} seconds? Provide a detailed summary of the camera&#x27;s motion over the last</td></tr><tr><td rowspan="2">Env.Measure.</td><td>Cam.Comprehensive</td><td>{n} seconds. Based on the observed footage，estimate the room’s length,</td></tr><tr><td>Room Area</td><td>width,and area.</td></tr><tr><td rowspan="2">Obj-Cam Rel.</td><td>Obj-Cam Distance</td><td>The {object} is currently out of view. How far away is it from me？ Where is the {object} relative to me now? Please use directional</td></tr><tr><td>Obj-Cam Direction Obj-Cam Location</td><td>terms. Provide both the clock direction and the distance of the {object}</td></tr><tr><td rowspan="2">Chronology</td><td>Object Counting Appearance Time</td><td>relative to me. How many {category}(s) have you seen so far? At what timestamp (in seconds) did the {object} first appear?</td></tr><tr><td>Appearance Order</td><td>What is the first-time appearance order of {choice a}， {choice b}，{choice c}?</td></tr><tr><td rowspan="4">rrerreeeereeareen</td><td>Env.Measure.</td><td>Object Size Obj-Obj Distance</td><td>Estimate the size of the {object}. What is the distance between {object A} and {object B} cur- rently?</td></tr><tr><td>Obj-Cam Rel.</td><td>Absolute Distance Relative Distance</td><td>How far is this {object} from my current position? Which of the following objects ({a,b,c,d}）is closest to me?</td></tr><tr><td rowspan="2">Attributes</td><td>Object Property</td><td>What is the color/material/shape of the {object} currently in view?</td></tr><tr><td>Object Position</td><td>Is the {object} currently located on the floor or on another ob- ject?</td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td>Object Recognition</td><td>Identify the {object} currently at the center of the view.</td></tr><tr><td>Cam.Direction</td><td>Where will the camera be located compared to here in {n} sec- onds?</td></tr><tr><td rowspan="6">morrsesn pereorg Env.Measure.</td><td rowspan="6">Ego-Motion</td><td>Cam.Path</td><td>In{n} seconds, how much ground will the camera cover in total? How many degrees will the camera turn horizontally in {n} sec-</td></tr><tr><td>Cam.Rotation</td><td>onds?</td></tr><tr><td>Cam.Displacement.</td><td>How far will the camera be from its current position in {n} seconds?</td></tr><tr><td></td><td>Give a comprehensive report on the camera&#x27;s motion over the</td></tr><tr><td>Cam.Comprehensive</td><td>next {n} seconds.</td></tr><tr><td>Room Area</td><td>Once the room is sufficiently covered, tell me its length,width, and area.</td></tr><tr><td rowspan="2">Obj-Cam Rel.</td><td>Obj-Cam Distance</td><td>Wait {n} seconds, then tell me the distance to the {object}. In{n} seconds,report both the clock direction and distance of</td></tr><tr><td>Object Finding</td><td>Obj-Cam Location the{object} relative to me. Obj-Cam Direction Wait {n} seconds, then indicate the direction of the {object). I&#x27;m moving,help locate the {object} and report its clock direc-</td></tr></table>

2. Distinct Entity: Select standalone functional entities rather than dependent parts (e.g., retain door but discard door frame).

3. Spatial Localizability: Prioritize objects with finite, well-defined boundaries, excluding continuous structural elements such as floor, ceiling, and wall.

The prompts used to curate the Target Object Whitelist are shown in Figure A1. Note: While structural elements like walls can theoretically support specific tasks (e.g., Object-to-Object Distance), they are unsuitable for the majority of object-centric tasks (e.g., Object Size or Object Counting). To maintain consistency across the benchmark, we applied this whitelist globally. This approach ensures a clear and consistent definition of “object" across tasks, prevents referential ambiguity (e.g., distinguishing between multiple walls in a single view), and focuses evaluation on the model’s ability to perceive distinct 3D instances.

<System Message>:   
You are a data annotation assistant for a 3D dataset. Your task is to classify whether a   
label represents a \*\*specific, spatially localized 3D instance\*\* suitable for diverse spatial   
reasoning tasks (e.g., estimating size, calculating distance/direction, or object finding).   
<User Instruction>:   
Please evaluate the label based on the following strict criteria:   
\*\*Return "1" (Keep) ONLY if the label meets ALL of these conditions:\*\*   
1. \*\*Specific Identity:\*\* It must refer to a concrete object class (e.g., 'chair', 'monitor'),   
NOT a broad category or hypernym (e.g., 'furniture', 'electronics', 'appliance', 'equipment').   
2. \*\*Distinct Entity:\*\* It must be a whole, standalone functional entity (e.g., 'door',   
'window'), NOT a dependent component or frame (e.g., 'door frame', 'window sill', 'table leg',   
'handle').   
3. \*\*Localizable:\*\* It must have clear boundaries (e.g., 'rug', 'poster', 'sink'), NOT be a   
structural boundary of the room (e.g., 'wall', 'floor', 'ceiling').   
\*\*Return "0" (Discard) if the label falls into ANY of these categories:\*\*   
1. \*\*Broad/Vague Categories:\*\* 'furniture', 'prop', 'item', 'object', 'stuff', 'misc',   
'unknown'.   
2. \*\*Structural Parts/Frames:\*\* 'door frame', 'window frame', 'molding', 'baseboard', 'trim',   
'column', 'beam', 'pipe'.   
3. \*\*Room Boundaries:\*\* 'wall', 'floor', 'ceiling'.   
4. \*\*Materials/Textures:\*\* 'wood', 'metal', 'carpet' (if wall-to-wall), 'tile'.   
5. \*\*Partial/Fragments:\*\* 'fragment', 'part of...', 'corner'.  
Fig-A1: Prompts employed to curate the Target Object Whitelist for object-centric 3D online QA.

## C Stream3D-1M Dataset Statistics

We present a comprehensive statistical analysis of the Stream3D-1M Dataset, a large-scale dataset containing approximately 1 million (1,003,203) questionanswer pairs derived from 5,154 unique scans. This dataset is designed to train Multimodal Large Language Models (MLLMs) with robust streaming 3D capabilities. The detailed composition is summarized in Figure A2 and Table A2.

Scene Diversity and Data Sources. To ensure the model generalizes across diverse environments, our dataset aggregates scenes from three distinct sources. As detailed in Table A2 (Top), ARKitScenes [4] contributes the majority of raw video data, accounting for 60.1% of the total videos. This vast diversity is crucial for preventing the model from overfitting to the specific interior styles of ScanNet-like environments. ScanNet [9], and ScanNet++ [50] contribute 23.3% and 16.6% of the videos, respectively, serving as the foundation for finegrained semantic and geometric reasoning.

![](images/286dbe04227ee5390441ad20a33febc6a12e775da512448b44c6bd7322f5486c.jpg)  
Fig-A2: Statistical distribution of Stream3D-1M dataset. The figure visualizes the composition of the training set (over 1M samples) across three key dimensions: (a) Data Source, highlighting that ScanNet++ contributes the plurality of QA pairs (43.5%) due to its dense annotations; (b) Task Category, dominated by Ego-Motion Estimation tasks (50.1%) which serve as the foundation for spatial tracking; and (c) Interaction Mode, demonstrating a strong emphasis on long-term memory (Backward, 46.3%) and active monitoring (Forward, 33.9%).

QA Distribution and Density. While ARKitScenes offers scene breadth, ScanNet++ provides annotation depth. Despite comprising fewer scenes, Scan-Net++ contributes the largest portion of QA pairs (43.5%), reflecting its highquality, dense annotations that support complex spatial reasoning. Conversely, ARKitScenes contributes 28.6% of the QAs, primarily focusing on camera motion and trajectories, balancing the dataset between semantic richness and kinematic diversity.

Task Category Distribution. The distribution of task types reflects the fundamental requirements of a streaming 3D assistant:

• Ego-Motion Estimation (50.1%): This category dominates the dataset. Since ego-motion estimation is a prerequisite for spatial tracking and is applicable to all data sources, it serves as a large-scale pre-training foundation.

• Object-Camera Relation (25.0%) & Object Chronology (14.3%): These tasks constitute the core of spatial interaction, requiring the model to update object states relative to the viewer dynamically.

• Environment Measurement (8.2%) & Object Attributes (2.4%): These tasks are inherently scarcer as they rely on specific geometric conditions or high-quality VLM-verified semantic attributes, ensuring high precision over quantity.

Temporal Mode Distribution. The dataset emphasizes memory and prediction capabilities. Backward Tracing accounts for 46.3% of the data, training the model’s long-term history retention. Realtime Perception (19.8%) and Forward Response (33.9%) ensure the model remains responsive to current and future events.

![](images/d5ff3a19ce3b4a9a21dee74e49a22d7281c8240580243e9123fc9219282a3098.jpg)  
Fig-A3: Statistical distribution of Stream3D-Bench. The four pie charts illustrate the composition of the benchmark across diverse dimensions: (a) Data Source, showing a robust integration of ScanNet, ScanNet++, and ARKitScenes; (b) Task Category, covering five core cognitive capabilities; (c) Interaction Mode, balancing tasks across Memory (Backward), Observation (Realtime), and Monitoring (Forward) phases; and (d) Answer Type, maintaining a uniform distribution among Openended, Numerical, and Multiple-choice formats.

## D Stream3D-Bench Details

To systematically evaluate the capabilities of MLLMs in online 3D spatial understanding, we construct Stream3D-Bench, a comprehensive benchmark comprising 10,037 high-quality samples derived from 518 unique videos. As shown in Figure A3 and Table A3, the benchmark is rigorously balanced across data sources, task categories, interaction modes, and answer types, ensuring a fair and robust assessment.

Data Sources and Diversity. The benchmark integrates diverse indoor 3D data to prevent overfitting to specific sensor characteristics. Specifically, the data is sourced from ScanNet (312 scans), ScanNet++ (100 scans), and ARKitScenes (106 scans). While ScanNet and ScanNet++ provide dense semantic annotations for object-centric tasks, ARKitScenes contributes significantly to camera motion tasks, enriching the diversity of trajectory dynamics.

Task Categories. Stream3D-Bench covers five core capability dimensions. To ensure balanced evaluation, we apply a stratified sampling strategy where most specific sub-tasks (Camera Path (Backward), Object Finding (Forward)) are explicitly balanced to approximately 356 samples each. Exceptions occur only in tasks with strict geometric constraints, such as Room Area, where valid samples are inherently scarcer. The overall distribution across categories is:

• Ego-Motion Estimation (35.5%): The largest category, assessing egomotion perception across all three datasets.

• Object-Camera Relationship (31.9%): Evaluates spatial reasoning capabilities like distance and direction estimation.

• Environment Measurement (11.3%): Focuses on quantifying scene attributes (e.g., object size, room area).

• Object Attributes (10.6%): Tests attribute recognition and property understanding.

Tab-A2: Detailed statistics of the Stream3D-1M dataset.  
Tab-A3: Detailed statistics of Stream3D-Bench composition.
<table><tr><td>Attribute</td><td>Count</td><td>|Percent</td><td>Attribute</td><td>|Count |Percent</td><td></td></tr><tr><td>Scan Count by Source</td><td></td><td></td><td>Data Source</td><td></td><td></td></tr><tr><td>ARKitScenes</td><td>3,098</td><td>60.1%</td><td>ScanNet</td><td>4,525</td><td>45.1%</td></tr><tr><td>ScanNet</td><td>1,201</td><td>23.3%</td><td>ScanNet++</td><td>4,082</td><td>40.7%</td></tr><tr><td>ScanNet++</td><td>855</td><td>16.6%</td><td>ARKitScenes</td><td>1,430</td><td>14.2%</td></tr><tr><td>Total Scans</td><td>5,154</td><td>100.0%</td><td>Task Category</td><td></td><td></td></tr><tr><td>QA Count by Source</td><td></td><td></td><td>Ego-Motion Estimation</td><td>3,560</td><td>35.5%</td></tr><tr><td>ScanNet++</td><td>436,812</td><td>43.5%</td><td>Object-Camera Relationship</td><td>3,204</td><td>31.9%</td></tr><tr><td>ScanNet</td><td>279,859</td><td>27.9%</td><td>Environment Measurement</td><td>1,137</td><td>11.3%</td></tr><tr><td>ARKitScenes</td><td>286,532</td><td>28.6%</td><td>Object Attributes</td><td>1,068</td><td>10.6%</td></tr><tr><td>Total QAs</td><td>1,003,203</td><td>100.0%</td><td>Object Chronology</td><td>1,068</td><td>10.6%</td></tr><tr><td>Task Category</td><td></td><td></td><td>Interaction Mode</td><td></td><td></td></tr><tr><td>Ego-Motion Estimation</td><td>503,072</td><td>50.1%</td><td>Backward</td><td>4,145</td><td>41.3%</td></tr><tr><td>Object-Camera Relationship</td><td>250,650</td><td>25.0%</td><td>Forward</td><td>3,400</td><td>33.9%</td></tr><tr><td>Object Chronology</td><td>143,475</td><td>14.3%</td><td>Realtime</td><td>2,492</td><td>24.8%</td></tr><tr><td>Environment Measurement</td><td>82,126</td><td>8.2%</td><td>Test Type</td><td></td><td></td></tr><tr><td>Object Attributes</td><td>23,880</td><td>2.4%</td><td>Open-Ended</td><td>3,629</td><td>36.2%</td></tr><tr><td>Interaction Mode</td><td></td><td></td><td>Numerical</td><td>3,204</td><td>31.9%</td></tr><tr><td>Backward</td><td>464,241</td><td>46.3%</td><td>Multiple-Choice</td><td>3,204</td><td>31.9%</td></tr><tr><td>Forward</td><td>340,153</td><td>33.9%</td><td></td><td></td><td></td></tr><tr><td>Realtime</td><td>198,809</td><td>19.8%</td><td></td><td></td><td></td></tr></table>

• Object Chronology (10.6%): Assesses temporal awareness, such as appearance order and counting.

Temporal Interaction Modes. The benchmark evaluates agents across three distinct temporal phases, ensuring the model can handle memory, observation, and prediction:

• Backward Tracing (41.3%): Assessing long-term memory and history retrieval.

• Realtime Perception (24.8%): Testing instant spatial perception and current-frame analysis.

• Forward Response (33.9%): Evaluating future monitoring and asynchronous response capabilities.

Answer Types. To validate different output modalities, the benchmark maintains a near-perfect balance among answer formats: Open-ended (36.2%), Numerical (31.9%), and Multiple-choice (31.9%). This uniform distribution prevents bias towards any specific answering style.

## E More Visualization Results• Object Positi• Object Recog• Counting • Time • Order

In this section, we present comprehensive visualization examples covering all 29 tasks in Stream3D-Bench, hierarchically structured along three temporal interaction patterns and five cognitive categories, as illustrated in Figures A4–A8.

![](images/f485f067ce4a0d7fe1cd9e51533dcbd0e78b6989254c696317db5c20c673086f.jpg)  
Fig-A4: Stream3D-Bench Examples (Part 1).

![](images/fc6be4945bbd100c5495974e364bce24842a7a0dbdb5bb9042cf1d8179a66f9c.jpg)  
Fig-A5: Stream3D-Bench Examples (Part 2).

![](images/ead36ed45dd8824ac27ed06533b895f95bd0aceeb63569f64dde09a65b88503a.jpg)  
Fig-A6: Stream3D-Bench Examples (Part 3).

![](images/d36889cae08c2f33293b2b6d6ec374ca4fc0fbc15098b27fe22bb390baf0a72f.jpg)  
Fig-A7: Stream3D-Bench Examples (Part 4).

![](images/236b315b232d61ea261fcbe08fded8a2f2f99ba636d4f398e9fc16f519bd5d24.jpg)  
Fig-A8: Stream3D-Bench Examples (Part 5).

## F Evaluation Prompts for Offline Models

Since existing offline models [10,14,15,21,28,37] cannot natively process streaming videos or generate online responses, we adapt them to Stream3D-Bench by providing the full video as input and explicitly specifying the query time. The models are prompted to jointly predict when to respond and what answer to produce. As illustrated in Figure A9, we explicitly inform the model of the total video duration and require it to simulate a streaming (online) scenario. Specifically, the query timestamp and the user question are given simultaneously, and the model must determine at what time after the query it has accumulated sufficient visual evidence to respond. The answer is constrained to rely only on visual information available up to the predicted response time.

Although this protocol enables a coarse evaluation of existing offline MLLMs on Stream3D-Bench, it inherently places them in a favorable and unrealistic setting, as the entire video is provided upfront, effectively granting access to more visual information than would be available in a true streaming scenario. Despite this disadvantageous comparison setup, experimental results consistently show that our real-time streaming 3D spatial understanding model significantly outperforms offline baselines across multiple evaluation metrics, including both answer accuracy and response-time precision. These results further demonstrate the superiority and effectiveness of our approach for online 3D spatial understanding. In future work, we plan to explore a broader range of streaming feedforward 3D reconstruction models [22, 24, 30, 33, 54] to further investigate and enhance model generalization and stability [40, 41], thereby facilitating better adaptation to real-world embodied applications.

![](images/f5b2f38410b514c95bd3101debaae49754ac90fa502479106fff8f264fdb05ec.jpg)  
Fig-A9: Evaluation prompts for offline MLLMs on our Stream3D-Bench.