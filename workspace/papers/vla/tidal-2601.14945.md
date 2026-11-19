<!-- page 1 of 8 -->

arXiv:2601.14945v3 [cs.RO] 14 Sep 2026

# TIDAL: Temporally Interleaved Diffusion and Action Loop for Dynamic Manipulation

Yuteng Sun<sup>1,2</sup>, Haoran Wang<sup>1,3</sup>, Ruofei Bai<sup>1,3</sup>, Zhengguo Li<sup>1</sup>, Jun Li<sup>1</sup>, Wenyu Liang<sup>1</sup>, Meng Yee (Michael) Chuah<sup>1</sup>, and Wei-Yun Yau<sup>1,∗</sup>

**Abstract— Large-scale Vision-Language-Action (VLA) models offer semantic generalization but suffer from high inference latency because they adopt a low-frequency batch-and-execute paradigm. This frequency mismatch creates an execution blind spot, causing failures in dynamic environments where targets move during the open-loop execution window. We propose TIDAL (Temporally Interleaved Diffusion and Action Loop), a hierarchical framework that decouples semantic reasoning from high-frequency actuation. TIDAL operates as a backbone-agnostic scheduler for diffusion-based VLAs, using a dual-frequency architecture to redistribute the computational budget. Specifically, a low-frequency macro-intent loop caches semantic embeddings, while a high-frequency micro-control loop interleaves single-step flow integration with execution, conditioning on the latest state fused with motion cues. To handle the resulting latency shift, we introduce a temporally misaligned training strategy where the policy learns stalenessaware compensation, conditioning on stale semantic intent alongside real-time proprioception. TIDAL is architectural, making it orthogonal to system-level optimizations. Experiments show an average 2.5x performance gain over open-loop baselines in dynamic interception tasks. Despite a slight decrease in static success rates, our approach yields a 4x increase in feedback frequency and extends the effective horizon of semantic embeddings beyond the native action chunk size. Under non-paused physics and on a real robot, TIDAL remains robust to unmodeled dynamics, while standard open-loop baselines fail due to latency-induced error accumulation.**

## I. INTRODUCTION

The pursuit of generalist embodied agents has been accelerated by Vision-Language-Action models, exemplified by systems like GR00T [1] and $\pi _ { 0 }$ [2]. These models increasingly adopt hierarchical architectures that integrate a large Vision-Language Model with continuous diffusion or flow-matching heads. While these models offer strong semantic grounding from large-scale pre-training, their scale creates a severe computational bottleneck. Current VLAs typically operate on a batch-and-execute paradigm: the system pauses to process observations and computes an entire long-horizon action chunk, which is then executed open-loop. This introduces a fundamental frequency mismatch. Dynamic manipulation requires high-frequency control, whereas large-scale VLA inference often incurs substantial latency and struggles to satisfy such control demands in practice [3], [4]. In real-world deployment, this bottleneck is worsened by the mandatory execution time of the generated action chunk [5], [6]. The

combination of inference latency and open-loop execution creates an execution blind spot, where the robot remains unresponsive to environmental changes, often leading to failures. Simply increasing the replanning frequency is often compute-prohibitive. Higher replanning rates can increase end-to-end delay, making newly generated trajectories stale by the time they are executed under non-paused physics, where the environment continues to evolve during policy inference rather than waiting for computation to finish. On edge hardware with a fixed compute budget, increasing the control update rate is better achieved by scheduling rather than by more frequent backbone queries.

Existing latency mitigation strategies incur significant trade-offs. System-level optimizations, such as quantization (e.g., BitVLA [7]) or token pruning (e.g., FlashVLA [8]), provide speedups but remain bounded by the sequential attention mechanism. Distilling small-scale policies like TinyVLA [9] trades rich semantic priors for throughput. More recently, asynchronous scheduling [5] and dual-system architectures [10] have attempted to decouple execution from reasoning. However, these methods often require careful stream management or separate policy networks. We argue that high-frequency control should not require losing the semantic intelligence of large foundation models, nor should it require training distinct policies.

We propose TIDAL (Temporally Interleaved Diffusion and Action Loop), a backbone-agnostic framework that bridges this frequency gap via compute-efficient interleaved scheduling. Our core insight is that semantic intent exhibits high temporal persistence, whereas physical state is highly transient. Consequently, the expensive VLM backbone need not be queried at every step. Instead of discarding the semantic context, we extend its lifespan by fusing it with highfrequency, real-time proprioceptive updates. TIDAL employs a dual-frequency scheduling strategy: a low-frequency macroloop caches semantic embeddings, while a high-frequency micro-loop interleaves single-step flow integration with execution. This redistributes the computational budget, refreshing the action chunk every ∼110 ms (≈ 9 Hz) compared to the standard ∼400 ms baseline cycle, yielding a 4× increase in feedback frequency. Although this imposes a higher optimization burden that slightly impacts static task convergence, it provides the critical reactivity required for dynamic interaction and extends the lifespan of semantic embeddings beyond the native action chunk size.

Deploying this decoupled architecture introduces a temporal misalignment between the stale semantic condition

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>1</sup>Authors are with the Institute of Advanced Intelligence and Computing (IAIC), Agency for Science, Technology and Research (A\*STAR), Singapore.</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>2</sup>Tsinghua University, Beijing, China.</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>3</sup>Nanyang Technological University, Singapore.</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>∗</sup>Correspondence: Yau\_Wei\_Yun@a-star.edu.sg.</span></small>

<!-- page 2 of 8 -->

![Image block](doc:2f5209d/tier:basic/page:2/block:1)

Fig. 1. Comparison of inference paradigms for dynamic interception under target morphology variation given the instruction “grasp the can.” (Top) Standard Baseline: The system performs blocking inference at the start and generates a fixed 16-step action chunk conditioned on the initial macro observation I<sub>macro</sub> and proprioception $s _ { \mathrm { p r o p } }$ via intent E. Open-loop execution follows a stale trajectory and can miss the moving target. (Bottom) TIDAL (Ours): The macro-loop queries the VLM once to cache intent E, while the micro-loop performs single-step interleaved flow integration using the latest fused state s˜ ∗ t from proprioception s ∗ prop and motion cues mt. The policy executes only the immediate chunk (N = 4) and discards the tail, refreshing control at ∼9,Hz for robust interception.

and the current physical state. To address this, we introduce a temporally misaligned training strategy where the policy learns to compensate for intent staleness, using current proprioception signals to reconcile stale semantic conditions with the rapidly evolving physical state. Additionally, we resolve the velocity insensitivity of static vision encoders by incorporating a Differential Motion Predictor with contactgated injection.

Our primary contributions are:

1) A serial, time-interleaved scheduler for VLAs that increases control update rate without increasing backbone calls.

2) A training paradigm that enables robust compensation for stale semantic cues under temporal misalignment.

3) A flow-matching inference/training recipe that enables reliable one-step Euler chunk generation for highfrequency interleaving.

4) TIDAL achieves an average 2.5× performance gain in dynamic interception tasks. Robust performance under non-paused inference evaluation and real-world deployment on a physical manipulator validates its efficacy.

## II. RELATED WORK

## A. Latency Mitigation and Real-Time Inference

Current methods bridge the inference-actuation gap via model simplification, adaptive computation, or parallel scheduling.

1) Model Distillation and Architectural Simplification: One strategy improves deployment efficiency through lightweight VLA variants, compressed backbones, or efficient architectures, including compact models [9], [11], [12] and state-space

architectures such as [13]. These methods typically trade model capacity or compatibility for speed, potentially weakening the semantic reasoning of larger backbones. TIDAL instead retains the full backbone and improves responsiveness through scheduling.

2) Static Compression and Adaptive Inference: Static compression techniques like quantization [7] and token pruning [8], [14] directly reduce memory footprint and computational cost. Dynamic inference mechanisms further accelerate forward passes via confidence-based early exits [15], layer routing [16], and parallel decoding [17]. Alongside system-level kernel fusion [18], these methods reduce per-step computation but remain bound by the sequential observe-thenact bottleneck. TIDAL is orthogonal to these optimizations; it acts as a temporal scheduler that can leverage such accelerated backbones to further densify replanning cycles.

3) Asynchronous Pipeline and Future Awareness: Recent works optimize the inference pipeline via asynchronous execution to break sequential dependencies. Real-Time Chunking [5] inpaints future actions during execution, while VLASH [6] uses a future-state-aware paradigm. VLASH trains the policy to compensate for latency by conditioning on predicted future states. Similarly, Spec-VLA [19] accelerates decoding via speculative sampling. These approaches reduce latency exposure by overlapping computation with execution or by forecasting future states. In contrast, under a serial compute constraint and non-paused physics, reactive updates from the latest state become critical. TIDAL shares the insight of training with temporal misalignment but diverges in execution. Pipelined approaches require high VLM throughput to minimize the prediction horizon. In contrast, TIDAL’s lightweight micro-loop reduces effective calculation time to

<!-- page 3 of 8 -->

minimal levels. This allows TIDAL to condition directly on the current fused state, avoiding the complexity and uncertainty of predicting future system states.

## B. Decoupled Architectures and Temporal Redundancy

Physical states change rapidly, yet high-level semantic intent persists over time [20]. While methods like VLA-Cache [21] exploit this by reusing visual features, recent frameworks like DuoCore-FS [22] and Fast-in-Slow [10] take this further by decoupling reasoning from motor control. They implement a parallel architecture: a slow VLM updates a semantic buffer, while a fast policy queries it at high frequency. However, running two models concurrently demands significant memory bandwidth and parallel compute, which are often prohibitive for edge devices.

To avoid parallel execution requirements, TIDAL achieves similar decoupling but shifts from a parallel to an interleaved strategy. By treating the VLM and action head as serial, temporally multiplexed components, we eliminate the need for parallel hardware. We freeze the heavy semantic embedding and use a single-step flow integration mechanism for the high-frequency micro-loop. This approach ensures real-time reactivity within a strict serial backbone compute budget, which we validate under realistic, non-paused physical constraints.

## C. Action Chunking and Execution Strategies

Action chunking [23] distributes the computational cost of VLA backbones [24], [25]. Recent works like Real-Time Chunking [5] improve throughput by overlapping inference with execution via inpainting or chunk splicing. However, these asynchronous methods condition actions on stale observations captured at the start of the inference delay.

TIDAL moves away from this rigid predict-then-execute paradigm. Leveraging Flow Matching [26], we reinterpret chunk generation as repeated conditional vector-field queries rather than a single static open-loop rollout. By injecting the latest proprioceptive state into the action head at each microupdate, our interleaved integration enables high-frequency reactivity without the complexity of asynchronous stream management or full sequence regeneration.

## III. METHODOLOGY

## A. Hierarchical Dual-Frequency Inference Scheduling

TIDAL splits VLA inference into two nested processes: a low-frequency macro-intent loop for semantic reasoning and a high-frequency micro-control loop for closed-loop refinement. This decomposition enables high-rate control updates while keeping the 3B-parameter backbone at a low query rate by caching semantic embeddings and interleaving flow matching integration steps.

1) Macro-Intent Loop (Semantic Caching): The macroloop extracts high-level intent from visual observations. Let $l _ { \mathrm { i n s t r } }$ denote the natural language instruction and $I _ { t _ { m } }$ denote the RGB observation at macro-step m. We use a pre-trained

VLM backbone $\Phi _ { \mathrm { V L M } }$ to map these inputs to a dense intent embedding $\mathcal { E } _ { m } ;$

$$
\mathcal { E } _ { m } = \Phi _ { \mathrm { V L M } } ( I _ { t _ { m } } , l _ { \mathrm { i n s t r } } )\tag{1}
$$

The embedding $\mathcal { E } _ { m }$ is stored in a latent intent buffer and remains frozen for the macro-cycle (horizon $H = 1 6 )$ . This caching strategy isolates the control loop from the high latency of $\Phi _ { \mathrm { V L M } } ,$ exploiting the fact that semantic intent persists over short execution horizons.

2) Micro-Control Loop (Interleaved Flow Execution): The micro-loop functions as a reactive policy that queries the action head at high frequency. To align with the standard execution horizon $( H   =   1 6 )$ while maximizing feedback density, we structure the inference as a cascade of $K = 4$ interactions per macro-cycle. Within each macro-cycle, we re-index time locally so that the macro observation is at $t = 0$ Given a cached intent $\mathcal { E } ,$ the loop iterates through latency stages $k \in \{ 0 , \ldots , K - 1 \}$ . At each stage, the loop observes the current state $\tilde { s } _ { k \cdot N }$ . To reduce inference latency, we bypass multi-step ODE solvers and generate the full-horizon action trajectory $\hat { \mathbf { a } } \in \mathbb { R } ^ { H \times D _ { a } }$ via a single-step Euler approximation $( \Delta \tau = 1 )$ from $\tau = 0$ to 1:

$$
\mathbf { \hat { a } } = \mathbf { x } _ { 0 } + \Delta \tau \thinspace v _ { \theta } ( \mathbf { x } _ { 0 } , 0 , \tilde { s } _ { k \cdot N } , \mathcal { E } ) , \qquad \Delta \tau = 1 .\tag{2}
$$

where $\mathbf { x } _ { 0 } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ is the source noise trajectory at $\tau = 0$ and $v _ { \theta }$ predicts the conditional flow-matching velocity field along the optimal-transport path. We treat this as a one-step flow map optimized to be accurate near the source frontier $( \tau   \approx   0 )$ , rather than a generic ODE solver. After chunk generation, we execute only the first N steps $( \hat { \mathbf { a } } _ { [ 0 , N ) } )$ on the robot hardware, discarding the rest $( \hat { \mathbf { a } } _ { [ N , H ) } )$

TIDAL distributes the typical integration budget across the execution timeline. Inspired by streaming execution paradigms [27], [28], we use a single Euler step as a latency-efficient update for the interleaved control loop at the chosen chunk size. By querying this field at every chunk boundary using the latest state $\tilde { s } _ { k \cdot N }$ , TIDAL actively corrects action chunk drift without increasing the backbone compute budget.

## B. Temporally Misaligned Training

Synchronous training creates a distribution shift in our architecture because semantic conditions $\mathcal { E } \; ( I _ { t - \Delta } )$ lag behind the physical state $\tilde { s } _ { t } .$ . Inspired by recent findings in asynchronous control [6], [22], we address this via a temporally misaligned training strategy that explicitly decouples visual and physical input timestamps, enabling convergence despite semantic lag.

1) Dynamic Latency Injection: We simulate variable inference latencies by creating training samples with randomized time-lags. Instead of fixed windows $H ,$ we sample extended trajectory segments of length $L ;$

$$
L = H + ( K - 1 ) \times N\tag{3}
$$

where $H = 1 6$ is the action horizon predicted by the policy, $N = 4$ is the prefix executed before the next micro-update, and $K = 4$ is the number of micro-updates performed under one cached macro intent. $L$ is the training segment length

<!-- page 4 of 8 -->

![Image block](doc:2f5209d/tier:basic/page:4/block:1)

Fig. 2. Temporally Misaligned Training Strategy. We decouple visual and physical timestamps to handle macro–micro latency. Latency Injection: A frozen semantic intent from I<sub>macro</sub> at t=0 is reused across latency stages $( k = 0 , \dot { \dots } , 3 )$ and paired with later fused states $( \tilde { s } * 0 , \tilde { s } \mathbin { \dot { * } } 4 , \tilde { s } * 8 , \tilde { s } \mathbin { \dot { * } } 1 2 ) .$ Optimization Objectives: The DiT predicts the flow vector field v<sub>θ</sub> over the $H { = } 1 6 { - } \mathsf { s t e p }$ horizon, supervised by $( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } )$ . A horizon-weighted loss emphasizes the immediate execution prefix (N=4), while time-biased sampling concentrates τ near the noise source for single-step integration.

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 TIDAL Hierarchical Inference Logic
Require: VLM Backbone $\Phi _ { \mathrm { V L M } } ,$ Policy Vector Field vθ,
Prediction Horizon $H = 1 6 ,$ Execution Chunk $N = 4 ,$
Latency Stages $K   =   4 ,$ Micro-Cam Buffer B (single-
frame, initialized by warm-up for N steps)
1: loop
2: {Macro-Intent Loop}
3: Capture high-res image $I _ { \mathrm { m a c r o } }$
4: $\mathcal { E } \gets \Phi _ { \mathrm { V L M } } ( I _ { \mathrm { m a c r o } } , l _ { \mathrm { i n s t r } } )$ {Freeze semantic intent}
5: {Micro-Control Loop}
6: for $k = 0$ to $K - 1$ do
7: Capture proprioception $s _ { \mathrm { p r o p } }$ and micro frame $I _ { k \cdot N }$
8: $I _ { \mathcal { B } } \gets \mathcal { B }$ {previous micro frame}
9: $\mathcal { B } \leftarrow I _ { k \cdot N }$ {update buffer}
10: $m _ { k \cdot N } \leftarrow \phi _ { \mathsf { m o t i o n } } ( \mathcal { T } ( I _ { k \cdot N } ) - \mathcal { T } ( I _ { \mathcal { B } } ) )$
11: $\tilde { s } _ { k \cdot N } \gets \mathtt { F u s e } ( s _ { \mathtt { p r o p } } , m _ { k \cdot N } )$ {State Fusion}
12: $\mathbf { x } _ { 0 }   \sim   \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ {Sample fresh noise trajectory of
length H}
13: $\mathbf { \hat { v } } \leftarrow v _ { \theta } ( \mathbf { x } _ { 0 } , \tau = 0 , \tilde { s } _ { k \cdot N } , \mathcal { E } )$ {Query vector field at
source frontier}
14: $\mathbf { \hat { a } } \gets \mathbf { x } _ { 0 }   +   \mathbf { \hat { v } }   \cdot   1$ {Single-step Euler integration $( \Delta \tau =$
1)}
15: Execute $\hat { \mathbf { a } } _ { [ 0 , N ) }$ {Run N steps, discard $H - N \}$
16: Wait for execution
17: end for
18: end loop
</div>

required so that every sampled latency stage still has an H-step supervision target. We set $L = 2 8 .$

During training, we discretize the potential delay into fixed stages, sampling a latency index $k \sim \mathcal { U } \{ 0 , K - 1 \}$ for each iteration. The policy input tuple consists of:

• Frozen Intent (at physical step $t = 0 )$ : Visual backbone input $I _ { 0 } .$

• Current State (at physical step $t = k   \cdot   N )$ : Proprioceptive state $\tilde { s } _ { k \cdot N }$

• Action Target (at physical step $t = k \cdot N )$ : Ground truth sequence $\mathbf { a } _ { [ k N ,   k N + H ) }$ (length H).

Restricting latency to a fixed set of discrete multiples of N, rather than sampling from a continuous interval, reduces the variance of the learning objective. This simplifies the optimization landscape, allowing the model to converge to distinct compensation modes for each stage of intent staleness, rather than fitting an infinite spectrum of delays.

2) Horizon-Weighted & Time-Biased Flow Matching: The action chunk generator is implemented using Conditional Flow Matching (CFM) [29], which regresses a timedependent vector field $v _ { \theta }$ . We define the optimal transport path $\psi _ { \tau } ( \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } ) = ( 1 - \tau ) \mathbf { x } _ { 0 } + \tau \mathbf { x } _ { 1 }$ between the Gaussian noise source $\mathbf { x } _ { 0 } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ and the target action sequence x<sub>1</sub>. The objective minimizes the expected flow matching error:

$$
\begin{aligned} { \mathcal { L } ( \theta ) = \mathbb { E } _ { k , \tau , \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } } \Bigg [ \sum _ { i = 0 } ^ { H - 1 } w _ { i } \cdot \Big \| v _ { \theta } ( \psi _ { \tau } ( \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } ) , \tau , \tilde { s } _ { k \cdot N } , \mathcal { E } _ { 0 } ) ^ { ( i ) } } \\ { - \left. ( \mathbf { x } _ { 1 } ^ { ( i ) } - \mathbf { x } _ { 0 } ^ { ( i ) } ) \right\| ^ { 2 } \Bigg ] } \\ \end{aligned}\tag{4}
$$

where $\tau \in [ 0 , 1 ]$ is the continuous flow time, $k \in \{ 0 , 1 , 2 , 3 \}$ represents the sampled latency stage, and i indexes the action horizon steps. $\tilde { s } _ { k \cdot N }$ and $\mathcal { E } _ { 0 }$ denote the time-misaligned proprioception and frozen semantic intent, respectively. The micro-loop executes only the initial chunk of length $N = 4$ before the next update. Therefore, we apply a horizon-aware mask $w _ { i } ,$ setting $w _ { i } = 2 . 0$ for $i < N$ and 1.0 otherwise. This asymmetry forces the vector field to prioritize precision in the imminent execution window $( i < N )$ over distal trajectory smoothness.

<!-- page 5 of 8 -->

Standard Flow Matching samples the flow time step τ uniformly $( \tau \sim \mathcal { U } [ 0 , 1 ] )$ , allocating equal capacity across the probability path. However, our micro-loop relies exclusively on single-step Euler integration starting from pure noise $( \tau =$ 0). The policy is only queried at the source frontier. Training capacity spent minimizing error at intermediate flow times $( \mathrm { e . g . , } ~ \tau = 0 . 5 )$ is therefore wasted, as these states are never visited during inference. To align the objective with our architecture, we bias the sampling distribution $p ( \tau )$ heavily towards the noise source. We sample an auxiliary variable $u \sim \mathbf { B e t a } ( \alpha , \beta )$ and set $\tau = 1   -   u$ . With $\alpha = 5 . 0$ and $\beta = 1 . 0$ the probability mass concentrates near $\tau \approx 0 .$ . This ensures the vector field is most accurate exactly where the single-step solver queries it, maximizing the efficiency of the limited inference budget.

## C. Dynamic Perception via Motion Injection

We introduce a lightweight Differential Motion Predictor $( \phi _ { \mathrm { m o t i o n } } )$ to inject high-frequency kinematic features directly into the policy.

We extract motion cues using temporal difference tensors. The input $\Delta I _ { t } = \mathcal { T } ( I _ { t } ) - \mathcal { T } ( I _ { t - \delta } )$ is the normalized pixelwise difference between the current frame and a history frame with lag $\delta = N$ (i.e., one micro-loop update interval). A lowlatency CNN processes this input into a low-dimensional embedding $m _ { t }   \in   \mathbb { R } ^ { 6 4 }$ . We pretrain $\phi _ { \mathrm { m o t i o n } }$ using a loss to predict the target’s future position and velocity, forcing $m _ { t }$ to encode predictive physical dynamics and momentum.

To prevent control instability during manipulation where relative motion is zero, we employ a hard contact-gating mechanism. Based on a binary contact state $c _ { t } \; \in \; \{ 0 , 1 \}$ derived from gripper sensors and the robot’s proprioceptive joint state $s _ { \mathrm { p r o p } }$ , we construct the fused policy state $\tilde { s } _ { t }$ via concatenation:

$$
\tilde { s } _ { t } = \mathsf { C o n c a t } \Big ( s _ { \mathsf { p r o p } } , ( 1 - c _ { t } ) \cdot m _ { t } \Big )\tag{5}
$$

This formulation enforces a modal switch: the policy attends to the full motion vector during the approach phase $( c _ { t } = 0 )$ and receives a zero-masked vector during manipulation $( c _ { t } =$ 1).

## IV. EXPERIMENTS

We evaluate whether TIDAL’s decoupling and scheduling enable robust dynamic manipulation without losing generalist capabilities.

## A. Experimental Setup

1) Environments: We evaluate TIDAL on two benchmarks in the RoboCasa suite.

We use 8 long-horizon tasks from the official RoboCasa-GR1 benchmark [1], [30] with stationary targets to verify that our architecture retains generalist capabilities.

We construct a Dynamic Interception benchmark in which success requires the robot to intercept and grasp a moving target, place it into a drawer, and close the drawer. The target starts with a random cardinal-direction velocity and

makes random 90-degree turns at boundaries, yielding non-deterministic trajectories. We stratify the benchmark into Easy and Hard tiers; using the left hand makes right/up motions kinematically favorable.

2) Data Collection: For the dynamic task, we collect 2,000 successful fine-tuning episodes using a paused-simulation oracle that pauses physics at every step to run full VLA inference, producing ideal closed-loop trajectories—a behavior achievable only when computational latency is ignored.

3) Baselines and Computational Protocol: We use GR00T-N1.5-3B [1] as the shared VLA backbone, fixing the prediction horizon at $H = 1 6$ to isolate performance gains derived purely from scheduling efficiency. To quantify architectural latency, we use the latency numbers reported for NVIDIA Jetson AGX Orin in [1]. The Effective Update Frequency is derived from the total cycle time $( T _ { \mathrm { c y c l e } } = T _ { \mathrm { i n f e r e n c e } } + T _ { \mathrm { e x e c u t i o n } } )$ at a 50 Hz physical control rate (20 ms per execution step).

We compare TIDAL against three primary configurations:

• **Open-Loop Baseline [1]:** Queries the VLM and a 4- step DiT of GR00T-N1.5-3B once at $t = 0 ,$ generating a 16-step chunk that is executed completely open-loop (∼2.4 Hz).

**Compute-Heavy Replan:** A short-execution fullreplanning variant that queries the full VLM and 4-step DiT every 4 physical steps. It generates a 16-step chunk but executes only the first 4 steps. It includes our motion predictor for dynamic tracking.

• **TIDAL (Ours):** Queries the VLM once every 16 steps to cache intent. Every 4 steps, it queries the flow head for a single Euler integration, updating the trajectory based on the latest observation.

TABLE I

INFERENCE LATENCY AND REAL-TIME UPDATE PROFILES

| Method | Cost | T<sub>inf</sub> (ms) | T<sub>exec</sub> | T<sub>cycle</sub> | Update |
| --- | --- | --- | --- | --- | --- |
| Baseline | 1.0× | 93 | 320 | 413 | ∼2.4Hz |
| Comp-Heavy | >4.0× | 93 | 80 | 173 | ∼5.8Hz |
| TIDAL (Avg) | ∼1.05× | ∼30 | 80 | ∼110 | ∼9.0Hz |
| TIDAL (Micro) |  | ∼19 |  | ∼99 | ∼10.1Hz |

Table I shows that the Open-Loop Baseline has $T _ { \mathrm { c y c l e } } { = } 4 1 3   \mathrm { m s }$ , while Compute-Heavy reduces it to 173 ms at an unsustainable $> 4 . 0 \times$ cost. TIDAL remains approximately iso-compute by amortizing one macro-step and four microsteps, with $T _ { \mathrm { i n f } } \approx 3 0$ ms on average (down to ∼19 ms for micro-steps), enabling an effective ∼9–10 Hz update rate without parallel hardware.

## B. Main Results

Table II reports success rates on RoboCasa static tasks and our Dynamic Interception benchmark. To evaluate the true impact of inference latency, we report dynamic performance under two physics protocols: the idealized Paused simulation where the environment waits for computation and the realistic Non-Paused evaluation where physics advances during policy inference, approximating real-world deployment.

<!-- page 6 of 8 -->

TABLE II

SUCCESS RATE COMPARISON ON STATIC AND DYNAMIC TASKS UNDER DIFFERENT PHYSICS PROTOCOLS

<table><tbody><tr><td rowspan="2">Method</td><td rowspan="2">Compute Cost</td><td>Static Benchmark</td><td colspan="2">Dynamic (Easy)</td><td>Dynamic (Hard)</td></tr><tr><td>Standard</td><td>Paused (Ideal)</td><td>Non-Paused (Realistic)</td><td>Paused (Ideal)</td></tr><tr><td>Open-Loop Baseline</td><td>1.0×</td><td>0.6088</td><td>0.31</td><td>0.09 (↓71.0%)</td><td>0.16</td></tr><tr><td>Compute-Heavy Replan</td><td>&gt;4.0×</td><td>-</td><td>0.63</td><td>0.17 (↓73.0%)</td><td>-</td></tr><tr><td>TIDAL (Ours)</td><td>∼1.05×</td><td>0.5938 (↓2.5%)</td><td>0.61</td><td>0.30 (↓50.8%)</td><td>0.36</td></tr></tbody></table>

\*Note: Bold/underline indicate best/second-best. The >4.0× compute cost is highlighted as a severe penalty: while brute-force computation performs well in idealized Paused physics, its inference latency leads to a severe ∼73% drop in the realistic Non-Paused environment. TIDAL robustly maintains performance with minimal compute overhead.

1) Static Generalization: On the official RoboCasa static benchmark, the Baseline achieves a 60.88% success rate. TIDAL achieves 59.38%, demonstrating that our hierarchical decoupling incurs only a slight drop. The slight drop is expected, as our temporally misaligned training objective and single-step flow integration create a more complex optimization landscape compared to standard open-loop training. However, this negligible trade-off confirms that TIDAL successfully retains the rich semantic priors of the 3B VLA backbone while freeing up computational budget for high-frequency control.

## 2) Dynamic Interception (Idealized Paused Protocol):

In the standard paused simulation, the Open-Loop Baseline fails in dynamic tasks because it blindly follows an outdated trajectory. To establish an upper performance bound, we introduce Compute-Heavy Replan, which uses aggressive full replanning—querying the full VLM and a 4-step DiT every 4 steps at a > 4.0× compute cost—achieving a 0.63 success rate. TIDAL achieves an almost identical 0.61 at ∼1.05× compute cost. In a paused world, these results suggest that locally updating the flow field can approach the performance of full semantic replanning.

3) Dynamic Interception (Realistic Non-Paused Protocol): The limitation of compute-heavy replanning is exposed under the non-paused protocol, where the environment continues to evolve during inference. The Baseline collapses to a 9% success rate. The compute-heavy replan also experiences a severe drop from 0.63 to 0.17; although it updates frequently and perceives motion, its massive computational latency means targets move before new plans execute. Simply increasing replanning frequency without solving inference lag fails in physical reality. In contrast, TIDAL sustains a highly robust 0.30 success rate. By isolating high-frequency updates to the lightweight micro-loop, TIDAL minimizes reaction delay, actively correcting positional drift while the cached semantic plan remains valid.

4) Failure Modes: While interleaving substantially improves reactivity, the task remains a long-horizon manipulation problem and does not saturate even in a static setting (success ∼0.8) due to semantic and appearance variability. In the paused dynamic setting, remaining failures typically happen after interception during transport and placement into the right-side drawer. The object occasionally slips or is

released, or ends up with a pose error that prevents a clean insert, consistent with suboptimal contact under dynamic grasps and limited grasp stability. Under the non-paused protocol, additional failures occur when the target moves too quickly or changes direction suddenly.

## C. Ablation Study

We isolate the contributions of TIDAL’s two core components: hierarchical inference scheduling and differential motion predictor. Table III shows the results on the Dynamic (Easy) task.

The TIDAL scheduling framework without motion features yields a success rate of 0.33, marginally better than the Baseline. Although operating at ∼9 Hz, it lacks explicit velocity data. Injecting motion features into the Baseline improves performance to 0.43. While motion embeddings provide crucial initial velocity cues at t = 0, open-loop execution prevents any mid-trajectory corrections. TIDAL achieves 0.61, outperforming both variants. This highlights a critical synergy: the motion predictor provides the signal to track dynamics, while the hierarchical architecture provides the computational opportunity to act on that signal reactively.

TABLE III

ABLATION ANALYSIS ON DYNAMIC (EASY) TASKS

| Configuration | Hierarchical Micro-Loop | Motion Predictor | Success Rate | Key Limitation |
| --- | --- | --- | --- | --- |
| Baseline | ✗ | ✗ | 0.31 | High Latency + Static Features |
| w/o Motion | ✓ | ✗ | 0.33 | Motion Blindness |
| Base + Motion | ✗ | ✓ | 0.43 | Execution Blind Spot |
| TIDAL (Full) | ✓ | ✓ | 0.61 | - |

## D. Semantic Intent Lifespan and Efficiency

We analyze cached intent validity on a single static task from the official suite by increasing the number of physical steps executed per macro intent refresh. This forces the microloop to act longer under stale semantics, testing the policy’s tolerance to intent staleness.

Fig. 3 shows a stable profile: extending intent lifespan from the standard 16 steps (0.72) to 44 steps (0.60) causes limited degradation. This suggests that cached embeddings remain useful when paired with real-time proprioceptive updates.

This finding challenges the native action chunk limit. While the GR00T backbone is trained to predict a fixed 16-step

<!-- page 7 of 8 -->

![Chart block](doc:2f5209d/tier:basic/page:7/block:1)

Fig. 3. Analysis of Semantic Intent Lifespan. The shaded regions indicate the Stable Semantics Zone (green) where intent remains valid, and the Degradation Zone (red) where performance collapses. The native action chunk limit is 16 steps.

![Image block](doc:2f5209d/tier:basic/page:7/block:3)

Fig. 4. Real-World Dynamic Interception Workspace. The overhead view illustrates the stochastic nature of the task. Colored overlays represent diverse historical trajectories captured during evaluation, highlighting unmodeled non-linearities, decelerations due to friction, and sharp deflections caused by block obstacles.

chunk, TIDAL’s hierarchical architecture extends the useful horizon of a single intent query to ∼44 steps. The policy uses the old intent to condition new, locally correct flows. A sharp performance cliff appears at 48 steps, indicating where the robot’s state diverges too far from the initial snapshot. Although variance increases at extreme latencies, the trend confirms that control ability degrades as semantic lag exceeds the physical correlation window.

## E. Real-World Deployment

To validate interleaved scheduling under unmodeled realworld stochasticity, we deploy TIDAL on a KUKA iiwa 14 with a Robotiq 2F-140 gripper and an RGB camera.

1) Task Setup and Environmental Stochasticity: As shown in Fig. 4, objects roll down a ramp into a flat workspace with block obstacles, introducing trajectory uncertainty from friction, spin, and obstacle deflections. To isolate scheduling effects, we also equip the open-loop baseline with the motion predictor. It still executes the full 16-step chunk, whereas TIDAL executes only N = 4 steps before each micro-loop update.

2) Reactivity and Error Bounding: In real-world experiments, TIDAL achieves an 86% success rate, substantially

![Chart block](doc:2f5209d/tier:basic/page:7/block:10)

Fig. 5. Temporal Degradation of Action Chunks. Tracking error vs. action index diverges for the open-loop 16-step rollout, while TIDAL executes only the first 4 steps to bound error within gripper tolerance.

outperforming the motion-augmented open-loop baseline (34%) over 100 trials, yielding roughly a 2.5× improvement in dynamic environments.

The motion-augmented baseline is accurate under near-linear rolls but fails once real perturbations (friction/spin/collisions) change the target trajectory. Executing a 16-step chunk open-loop, it accumulates tracking error throughout the rollout (Fig. 5). Once the target deviates from its initial motion at t = 0, the later chunk steps effectively become stale open-loop extrapolation.

TIDAL instead bounds error by truncating execution to N=4 steps and triggering a micro-loop update (Fig. 5). Although tracking error can increase slightly within each 4-step window, the next update reconditions on fresh proprioception and motion, preventing divergence. This frequent refresh keeps error within the gripper’s tolerance under unmodeled disturbances.

## V. CONCLUSION

We presented TIDAL, a framework designed to enable high-frequency control with large VLA models. By interleaving single-step flow generation with action execution, our approach achieves a 4x increase in update frequency on edge hardware while maintaining the same backbone computational budget. This significantly reduces the execution blind spot found in standard batch-and-execute baselines.

We leverage temporally misaligned training to compensate for stale semantics, while single-step flow integration minimizes latency. This results in a 2.5x performance gain on dynamic interception tasks. Our analysis reveals that updating the policy with real-time proprioception effectively extends the lifespan of cached semantic embeddings beyond their native horizon. Since TIDAL is backbone-agnostic, it remains orthogonal to system-level accelerations. TIDAL sustains robust performance under non-paused inference protocols, where compute-heavy baselines degrade sharply due to latency. Furthermore, real-world deployment on a physical manipulator confirms these findings. By caching temporally persistent semantic intent and interleaving low-latency flow updates with execution, TIDAL turns long-horizon VLA plans

<!-- page 8 of 8 -->

into a high-rate closed-loop controller, keeping tracking error bounded under non-paused inference and unmodeled realworld dynamics without increasing backbone queries.

## ACKNOWLEDGMENT

OpenAI’s ChatGPT was used to assist with implementation debugging and English-language editing. All technical content, analyses, and conclusions were reviewed and verified by the authors. This research is partly supported by the National Research Foundation, Singapore, under its Thematic CRP programme award no NRF-T-CRP-2025-0008 and by A\*STAR under the National Robotics Programme grant no H25-MNR3495.

## REFERENCES

[1] J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, S. Huang et al., “Gr00t n1: An open foundation model for generalist humanoid robots,” arXiv preprint arXiv:2503.14734, 2025.

[2] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter et al., “π<sub>0</sub>: A vision-language-action flow model for general robot control,” arXiv preprint arXiv:2410.24164, 2024.

[3] K. Kawaharazuka, J. Oh, J. Yamada, I. Posner, and Y. Zhu, “Visionlanguage-action models for robotics: A review towards real-world applications,” IEEE Access, vol. 13, pp. 162 467–162 504, 2025.

[4] R. Shao, W. Li, L. Zhang, R. Zhang, Z. Liu, R. Chen, and L. Nie, “Large vlm-based vision-language-action models for robotic manipulation: A survey,” arXiv preprint arXiv:2508.13073, 2025.

[5] K. Black, M. Y. Galliker, and S. Levine, “Real-time execution of action chunking flow policies,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id=UkR2zO5uww

[6] J. Tang, Y. Sun, Y. Zhao, S. Yang, Y. Lin, Z. Zhang, J. Hou, Y. Lu, Z. Liu, and S. Han, “Vlash: Real-time vlas via future-state-aware asynchronous inference,” arXiv preprint arXiv:2512.01031, 2025.

[7] H. Wang, C. Xiong, R. Wang, and X. Chen, “Bitvla: 1-bit visionlanguage-action models for robotics manipulation,” arXiv preprint arXiv:2506.07530, 2025.

[8] X. Tan, Y. Yang, P. Ye, J. Zheng, B. Bai, X. Wang, J. Hao, and T. Chen, “Think twice, act once: Token-aware compression and action reuse for efficient inference in vision-language-action models,” arXiv preprint arXiv:2505.21200, 2025.

[9] J. Wen, Y. Zhu, J. Li, M. Zhu, Z. Tang, K. Wu, Z. Xu, N. Liu, R. Cheng, C. Shen, Y. Peng, F. Feng, and J. Tang, “Tinyvla: Toward fast, dataefficient vision-language-action models for robotic manipulation,” IEEE Robotics and Automation Letters, vol. 10, no. 4, pp. 3988–3995, 2025.

[10] H. Chen, J. Liu, C. Gu, Z. Liu, R. Zhang, X. Li, X. He, Y. Guo, C.-W. Fu, S. Zhang, and P.-A. Heng, “Fast-in-slow: A dual-system VLA model unifying fast manipulation within slow reasoning,” in The Thirtyninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id=4asFznbzJg

[11] M. Shukor, D. Aubakirova, F. Capuano, P. Kooijmans, S. Palma, A. Zouitine, M. Aractingi, C. Pascal, M. Russi, A. Marafioti et al., “Smolvla: A vision-language-action model for affordable and efficient robotics,” arXiv preprint arXiv:2506.01844, 2025.

[12] C.-Y. Hung, Q. Sun, P. Hong, A. Zadeh, C. Li, U. Tan, N. Majumder, S. Poria et al., “Nora: A small open-sourced generalist vision language action model for embodied tasks,” arXiv preprint arXiv:2504.19854, 2025.

[13] J. Liu, M. Liu, Z. Wang, P. An, X. Li, K. Zhou, S. Yang, R. Zhang, Y. Guo, and S. Zhang, “Robomamba: Efficient vision-language-action model for robotic reasoning and manipulation,” in Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, Eds., vol. 37. Curran Associates, Inc., 2024, pp. 40 085–40 110. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2024/file/46a126492ea6fb87410e55a58df2e189-Paper-Conference.pdf

[14] Y. Yang, Y. Wang, Z. Wen, L. Zhongwei, C. Zou, Z. Zhang, C. Wen, and L. Zhang, “EfficientVLA: Training-free acceleration and compression for vision-language-action models,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id=SELYlDHZk2

[15] W. Song, J. Chen, P. Ding, Y. Huang, H. Zhao, D. Wang, and H. Li, “Ceed-vla: Consistency vision-language-action model with early-exit decoding,” arXiv preprint arXiv:2506.13725, 2025.

[16] R. Zhang, M. Dong, Y. Zhang, L. Heng, X. Chi, G. Dai, L. Du, Y. Du, and S. Zhang, “Mole-vla: Dynamic layer-skipping vision language action model via mixture-of-layers for efficient robot manipulation,” arXiv preprint arXiv:2503.20384, 2025.

[17] W. Song, J. Chen, P. Ding, H. Zhao, W. Zhao, Z. Zhong, Z. Ge, Z. Li, D. Wang, L. Wang, J. Ma, and H. Li, “Pd-vla: Accelerating visionlanguage-action model integrated with action chunking via parallel decoding,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025, pp. 13 162–13 169.

[18] Y. Ma, Y. Zhou, Y. Yang, T. Wang, and H. Fan, “Running vlas at real-time speed,” arXiv preprint arXiv:2510.26742, 2025.

[19] S. Wang, R. Yu, Z. Yuan, C. Yu, F. Gao, Y. Wang, and D. F. Wong, “Spec-VLA: Speculative decoding for vision-languageaction models with relaxed acceptance,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, C. Christodoulopoulos, T. Chakraborty, C. Rose, and V. Peng, Eds. Suzhou, China: Association for Computational Linguistics, Nov. 2025, pp. 26 928–26 940. [Online]. Available: https://aclanthology.org/2025.emnlp-main.1367/

[20] W. Guan, Q. Hu, A. Li, and J. Cheng, “Efficient vision-languageaction models for embodied manipulation: A systematic survey,” arXiv preprint arXiv:2510.17111, 2025.

[21] S. Xu, Y. Wang, C. Xia, D. Zhu, T. Huang, and C. Xu, “VLA-cache: Efficient vision-language-action manipulation via adaptive token caching,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id=QZYZ0Xm58q

[22] T. Zou, H. Zeng, Y. Nong, Y. Li, K. Liu, H. Yang, X. Ling, X. Li, and L. Ma, “Asynchronous fast-slow vision-language-action policies for whole-body robotic manipulation,” arXiv preprint arXiv:2512.20188, 2025.

[23] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware,” in Proceedings of Robotics: Science and Systems, Daegu, Republic of Korea, July 2023.

[24] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. P. Foster, P. R. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn, “Openvla: An open-source vision-language-action model,” in Proceedings of The 8th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, P. Agrawal, O. Kroemer, and W. Burgard, Eds., vol. 270. PMLR, 06–09 Nov 2025, pp. 2679–2713. [Online]. Available: https://proceedings.mlr.press/v270/kim25c.html

[25] R. Xu, J. Zhang, M. Guo, Y. Wen, H. Yang, M. Lin, J. Huang, Z. Li, K. Zhang, L. Wang, Y. Kuang, M. Cao, F. Zheng, and X. Liang, “A0: An affordance-aware hierarchical model for general robotic manipulation,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2025, pp. 13 491–13 501.

[26] X. Liu, C. Gong, and qiang liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=XVjTT1nw5z

[27] S. Jiang, X. Fang, N. Roy, T. Lozano-Pérez, L. P. Kaelbling, and S. Ancha, “Streaming flow policy: Simplifying diffusion/flow policies by treating action trajectories as flow trajectories,” in ICRA 2025 Workshop: Beyond Pick and Place, 2025.

[28] S. H. Høeg, Y. Du, and O. Egeland, “Fast policy synthesis with variable noise diffusion models,” in 2025 IEEE International Conference on Robotics and Automation (ICRA), 2025, pp. 4821–4828.

[29] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=PqvMRDCJT9t

[30] S. Nasiriany, A. Maddukuri, L. Zhang, A. Parikh, A. Lo, A. Joshi, A. Mandlekar, and Y. Zhu, “RoboCasa: Large-Scale Simulation of Household Tasks for Generalist Robots,” in Proceedings of Robotics: Science and Systems, Delft, Netherlands, July 2024.