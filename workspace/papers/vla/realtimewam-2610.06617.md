Title: RealtimeWAM: One-Step Asynchronous World Action Models

URL Source: https://arxiv.org/html/2610.06617

Published Time: Tue, 06 Oct 2026 02:41:39 GMT

Markdown Content:
Chengtao Lv Jinyang Du 1 1 footnotemark: 1 Shuyi Feng Yang Yong Shiqiao Gu ††thanks: Equal Contribution.Shunzi Yang Ruihao Gong Shen Ren Tianwei Zhang Wenya Wang 2 2 footnotemark: 2††thanks: Corresponding Authors.Affiliation:Nanyang Technological University Beihang University Sensetime Continental Automotive Singapore Email:[jinyangdu@buaa.edu.cn](mailto:jinyangdu@buaa.edu.cn)

###### Abstract

World Action Models (WAMs) incorporate visual representations from video generation backbones to guide action prediction. Recent efficient WAMs adopt Mixture-of-Transformers (MoT) architectures and compute video representations once for reuse by the action expert. However, intra-expert iteration (_i.e._, multi-step action denoising) and inter-expert waiting (_i.e._, sequential execution of the video and action experts) still limit inference efficiency. To this end, we present RealtimeWAM, an extremely efficient WAM variant with one-step action generation and asynchronous inference, addressing these two bottlenecks. To reduce intra-expert iteration, we propose Teacher-Anchored Consistency Distillation (TACD) to address a local-global error gap: low local consistency error alone does not guarantee accurate final actions. TACD supplements local consistency with explicit supervision from the frozen teacher’s multi-step rollout endpoint, enabling accurate one-step action generation. Additionally, we propose Cross-Expert Wavefront Pipelining (CEWP) to eliminate unnecessary expert-level waiting. It overlaps the two experts through block-wise sharing of the video KV cache, synchronizing only immediately before the corresponding action attention consumes it. Extensive experiments across diverse benchmarks (_e.g._, LIBERO, LIBERO-Plus and RoboTwin) and model variants (_e.g._, Fast-WAM and Faster-WAM) demonstrate the superiority of RealtimeWAM. Notably, RealtimeWAM maintains near-lossless performance (_i.e._, <1\% drop) across these benchmarks while delivering significant end-to-end speedup (_e.g._, \sim 25\times on H100). Our code and checkpoints are available via this [link](https://github.com/ModelTC/LightX2V/tree/main/examples/realtimewam).

## 1 Introduction

World Action Models (WAMs)([Yuan et al., 2026](https://arxiv.org/html/2610.06617#bib.bib4); [Ye et al., 2026](https://arxiv.org/html/2610.06617#bib.bib5); [Bi et al., 2026](https://arxiv.org/html/2610.06617#bib.bib12); [Li et al., 2026a](https://arxiv.org/html/2610.06617#bib.bib13); [Kim et al., 2026](https://arxiv.org/html/2610.06617#bib.bib14); [Zhao et al., 2026](https://arxiv.org/html/2610.06617#bib.bib15)) have emerged as a promising paradigm for robot manipulation by jointly modeling future visual observations and robot actions. Unlike conventional Vision-Language-Action (VLA) models([Shukor et al., 2025](https://arxiv.org/html/2610.06617#bib.bib6); [Intelligence et al., 2025](https://arxiv.org/html/2610.06617#bib.bib7); [Kim et al., 2024](https://arxiv.org/html/2610.06617#bib.bib8); [Bjorck et al., 2025](https://arxiv.org/html/2610.06617#bib.bib9); [Black et al., 2024](https://arxiv.org/html/2610.06617#bib.bib10); [Liu et al., 2025](https://arxiv.org/html/2610.06617#bib.bib11)) that map visual observations and language instructions directly to actions, WAMs incorporate future visual prediction as an additional learning objective. This supervision encourages policies to capture the physical dynamics and temporal structure of robot–environment interactions.

Based on whether video and action modeling use the same network, WAM architectures can be broadly divided into two categories. Representative shared-backbone WAMs([Kim et al., 2026](https://arxiv.org/html/2610.06617#bib.bib14); [Ye et al., 2026](https://arxiv.org/html/2610.06617#bib.bib5)) jointly denoise video and action tokens within a single backbone. Consequently, iterative denoising of long token sequences through the heavy video backbone increases inference latency. In contrast, some efficient Mixture-of-Transformers (MoT)-based WAMs([Yuan et al., 2026](https://arxiv.org/html/2610.06617#bib.bib4); [Zhao et al., 2026](https://arxiv.org/html/2610.06617#bib.bib15)) offer a promising alternative: the video expert computes representations once, while a separate action expert reuses them for action generation.

Despite advancements in inference pipeline design, these MoT-based WAMs still incur substantial latency. Specifically, even on advanced GPUs (_e.g._, H100 in Fig.[1](https://arxiv.org/html/2610.06617#S1.F1 "Figure 1 ‣ 1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models")), these models (_e.g._, Fast-WAM([Yuan et al., 2026](https://arxiv.org/html/2610.06617#bib.bib4))) require over 300 ms for a single inference call. We identify two inference bottlenecks that remain insufficiently addressed in existing MoT-based WAMs: ➀ Intra-expert iteration arises from multi-step action denoising. We observe that the action expert accounts for nearly 90% of end-to-end latency for action generation across diverse GPUs (see Fig.[1](https://arxiv.org/html/2610.06617#S1.F1 "Figure 1 ‣ 1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models")). ➁ Inter-expert waiting stems from sequential execution of the video and action experts. As shown in Fig.[4](https://arxiv.org/html/2610.06617#S4.F4 "Figure 4 ‣ 4.2 Cross-Expert Wavefront Pipelining ‣ 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models")(a), the action expert begins execution only after the video expert completes its forward pass, causing unnecessary waiting.

Figure 1: Inference efficiency of Fast-WAM (left) and Faster-WAM (right) across diverse hardware. Vertical bars compare inference latency between the original models and RealtimeWAM, while stacked bars show the runtime proportions of key components.

Motivated by the above analysis, we present RealtimeWAM, an extremely efficient post-trained variant based on any MoT-based WAM. To our knowledge, it is the first one-step WAM to achieve near-lossless performance compared with its multi-step counterparts.

To combat intra-expert iteration, we propose Teacher-Anchored Consistency Distillation (TACD) for one-step action generation. TACD addresses a local-global error gap in conventional consistency distillation, which only constrains local consistency while neglecting to explicitly penalize the global target action error. TACD adopts the frozen teacher’s multi-step rollout endpoint as a numerical proxy for the exact clean endpoint, explicitly aligning the student’s velocity prediction with the corresponding interval-averaged teacher velocity. From a theoretical perspective, we show that this teacher anchor directly constrains the total endpoint error, complementing local consistency supervision with an explicit action endpoint constraint.

To reduce inter-expert waiting, we introduce Cross-Expert Wavefront Pipelining (CEWP), which transforms coarse expert-wise execution into a fine-grained block-wise pipeline. Under conventional expert-wise execution, the Action Expert waits for the Video Expert to construct the complete block-wise KV cache. We first conduct an in-depth dependency analysis, observing that each action block requires the video KV cache only from its corresponding video block, rather than the complete output of the Video Expert. Motivated by this observation, CEWP overlaps the execution of the two experts through block-wise sharing of the video KV cache, synchronizing only immediately before the corresponding action attention consumes it, which significantly reduces unnecessary waiting.

RealtimeWAM achieves impressive results on representative MoT-based WAM architectures, including Fast-WAM([Yuan et al., 2026](https://arxiv.org/html/2610.06617#bib.bib4)) and Faster-WAM([Zhao et al., 2026](https://arxiv.org/html/2610.06617#bib.bib15)), across diverse downstream benchmarks (_e.g._, LIBERO([Liu et al., 2023](https://arxiv.org/html/2610.06617#bib.bib1)), LIBERO-Plus([Fei et al., 2025](https://arxiv.org/html/2610.06617#bib.bib3)), and RoboTwin 2.0([Chen et al., 2025](https://arxiv.org/html/2610.06617#bib.bib2))). On these benchmarks, it delivers approximately 14\sim 25\times inference acceleration with an average <1\% accuracy drop compared with the original multi-step models. Notably, RealtimeWAM’s per-call inference latency is significantly shorter than the corresponding execution time (_e.g._, 30 Hz control frequency), enabling real-time action generation. These results demonstrate that RealtimeWAM remains effective across diverse robot configurations, task horizons, and distribution conditions, and provide a practical path toward efficient WAM deployment.

To summarize, our contributions are as follows:

\blacktriangleright We propose RealtimeWAM, a one-step asynchronous WAM that addresses both intra-expert iteration and inter-expert waiting in any MoT-based WAM for efficient robot manipulation.   
\blacktriangleright We propose Teacher-Anchored Consistency Distillation (TACD), which mitigates the local-global error gap by augmenting local consistency supervision with a teacher endpoint constraint for accurate one-step action generation.   
\blacktriangleright We introduce Cross-Expert Wavefront Pipelining (CEWP), which reduces unnecessary synchronization by overlapping block-wise execution of the Video and Action Experts according to their data dependencies.   
\blacktriangleright Extensive experiments demonstrate that RealtimeWAM achieves real-time action generation with <1\% accuracy degradation (_i.e._, 12.2 ms inference latency with \sim 25\times speedup on an NVIDIA H100 GPU).

## 2 Related Work

### 2.1 World Action Models

While Vision-Language-Action (VLA) models([Shukor et al., 2025](https://arxiv.org/html/2610.06617#bib.bib6); [Intelligence et al., 2025](https://arxiv.org/html/2610.06617#bib.bib7); [Kim et al., 2024](https://arxiv.org/html/2610.06617#bib.bib8); [Bjorck et al., 2025](https://arxiv.org/html/2610.06617#bib.bib9); [Black et al., 2024](https://arxiv.org/html/2610.06617#bib.bib10); [Liu et al., 2025](https://arxiv.org/html/2610.06617#bib.bib11); [Wang et al., 2026b](https://arxiv.org/html/2610.06617#bib.bib17); [Zheng et al., 2026](https://arxiv.org/html/2610.06617#bib.bib18)) rely on pretrained vision-language backbones to map visual observations and language instructions directly to actions, World Action Models (WAMs)([Yuan et al., 2026](https://arxiv.org/html/2610.06617#bib.bib4); [Ye et al., 2026](https://arxiv.org/html/2610.06617#bib.bib5); [Bi et al., 2026](https://arxiv.org/html/2610.06617#bib.bib12); [Li et al., 2026a](https://arxiv.org/html/2610.06617#bib.bib13)) incorporate future visual representations from video generation backbones to guide action prediction. Modern WAMs can be broadly classified into two categories. ➀ One line of work focuses on shared-backbone WAMs, which model future video and robot actions within a shared backbone, including DreamZero([Ye et al., 2026](https://arxiv.org/html/2610.06617#bib.bib5)) and Cosmos([Kim et al., 2026](https://arxiv.org/html/2610.06617#bib.bib14)). ➁ In contrast, Mixture-of-Transformers (MoT)-based WAMs, such as Motus([Bi et al., 2026](https://arxiv.org/html/2610.06617#bib.bib12)), employ separate video and action experts that interact through attention. Among these methods, Fast-WAM([Yuan et al., 2026](https://arxiv.org/html/2610.06617#bib.bib4)) reduces inference overhead by computing video representations once for reuse by the action expert. Subsequent works, including Light-WAM([Li et al., 2026c](https://arxiv.org/html/2610.06617#bib.bib16)) and Faster-WAM([Zhao et al., 2026](https://arxiv.org/html/2610.06617#bib.bib15)), further improve computational efficiency and out-of-distribution (OOD) robustness. Nevertheless, these methods may incur substantial performance degradation on challenging benchmarks or achieve limited speedups.

### 2.2 Few-Step Distillation

Diffusion models([Sohl-Dickstein et al., 2015](https://arxiv.org/html/2610.06617#bib.bib27); [Ho et al., 2020](https://arxiv.org/html/2610.06617#bib.bib28); [Song et al., 2020](https://arxiv.org/html/2610.06617#bib.bib29)) and flow models([Lipman et al., 2022](https://arxiv.org/html/2610.06617#bib.bib30); [Liu et al., 2022](https://arxiv.org/html/2610.06617#bib.bib31); [Albergo et al., 2023](https://arxiv.org/html/2610.06617#bib.bib32)) have become cornerstones of modern high-quality image and video generation. However, their iterative sampling cost has motivated a series of few-step acceleration methods, including progressive distillation([Salimans and Ho, 2022](https://arxiv.org/html/2610.06617#bib.bib33)), consistency models([Song et al., 2023](https://arxiv.org/html/2610.06617#bib.bib21)), distribution matching distillation([Yin et al., 2024](https://arxiv.org/html/2610.06617#bib.bib22)), MeanFlow([Geng et al., 2025](https://arxiv.org/html/2610.06617#bib.bib20)), and adversarial distillation([Sauer et al., 2024](https://arxiv.org/html/2610.06617#bib.bib34)). These distillation techniques have recently been introduced into World Action Models([Akbari et al., 2026](https://arxiv.org/html/2610.06617#bib.bib19); [Yang et al., 2026](https://arxiv.org/html/2610.06617#bib.bib35)). The work most relevant to ours is Flash-WAM([Akbari et al., 2026](https://arxiv.org/html/2610.06617#bib.bib19)), which identifies a modality-dependent noise-regime mismatch and introduces modality-specific consistency parameterizations. However, ➀ it focuses on few-step distillation for WAMs that retain future-video denoising during inference, such as LingBot-VA([Li et al., 2026a](https://arxiv.org/html/2610.06617#bib.bib13)); and ➁ its ultra-few-step (_i.e._, one-step) distillation still exhibits noticeable performance degradation (_e.g._, RoboTwin and real-world tasks). These limitations motivate our investigation of one-step action distillation for MoT-based WAMs.

### 2.3 Asynchronous Inference for Robot Policies

Asynchronous inference has been widely studied for real-time robot control because it mitigates execution stalls, stale observations, and delayed responses caused by policy latency. Existing approaches can be broadly grouped into three categories: execution-level asynchrony, module-level asynchrony, and sampling-level asynchrony. Execution-level methods overlap prediction of the next action chunk with execution of the current one and maintain continuity through prefix conditioning, state correction, or chunk fusion([Black et al., 2025a](https://arxiv.org/html/2610.06617#bib.bib23); [Black et al., 2025b](https://arxiv.org/html/2610.06617#bib.bib36); [Tang et al., 2025](https://arxiv.org/html/2610.06617#bib.bib37); [Wang et al., 2026a](https://arxiv.org/html/2610.06617#bib.bib24); [Motubrain Team, 2026](https://arxiv.org/html/2610.06617#bib.bib26)). Module-level methods decouple a slow semantic or world module from a fast action module, allowing cached high-level representations to guide high-frequency action updates([Hirose et al., 2026](https://arxiv.org/html/2610.06617#bib.bib39); [Cai et al., 2026](https://arxiv.org/html/2610.06617#bib.bib38)). Sampling-level methods assign different action positions, chunks, modalities, or denoising streams distinct sampling schedules, enabling early action output or selective refinement without waiting for full decoding([Jiang et al., 2025](https://arxiv.org/html/2610.06617#bib.bib40); [Lu et al., 2026](https://arxiv.org/html/2610.06617#bib.bib41); [Li et al., 2026b](https://arxiv.org/html/2610.06617#bib.bib25)). In contrast, RealtimeWAM employs finer-grained asynchrony at the block level, distinguishing it from prior approaches.

## 3 Preliminaries

### 3.1 Efficient Mixture-of-Transformers (MoT)-based WAMs

WAMs with a Mixture-of-Transformers (MoT) architecture([Bi et al., 2026](https://arxiv.org/html/2610.06617#bib.bib12); [Li et al., 2026a](https://arxiv.org/html/2610.06617#bib.bib13)) comprise a pretrained Video Expert for world modeling and an Action Expert for action generation. Corresponding blocks in the two experts interact through shared attention. While the two experts are jointly trained on video and action prediction, recent efficient MoT-based WAMs([Yuan et al., 2026](https://arxiv.org/html/2610.06617#bib.bib4); [Zhao et al., 2026](https://arxiv.org/html/2610.06617#bib.bib15)) require only a single forward pass through the Video Expert during inference. Specifically, the Video Expert processes the current observation o and language instruction l to extract block-wise keys and values:

z(o,l)=F_{\psi}^{\mathrm{KV}}(o,l)=\left\{(K_{i}^{v},V_{i}^{v})\right\}_{i=1}^{N}.(1)

Here, F_{\psi}^{\mathrm{KV}} denotes the KV extraction function of the Video Expert with parameters \psi. The latent world representation z(o,l) consists of keys K_{i}^{v} and values V_{i}^{v} from each of the N video transformer blocks. Subsequently, these efficient MoT-based WAMs use this representation to parameterize the action distribution:

p_{\theta}(a\mid o,l)=p_{\theta}\!\left(a\mid z(o,l)\right).(2)

Here, a denotes an action chunk with prediction horizon H, and \theta denotes the parameters of the Action Expert. Action generation requires n denoising steps along 0=t_{0}<\cdots<t_{n}=1. The action distribution p_{\theta}\!\left(a\mid z(o,l)\right) is implicitly defined by the denoising map \Phi_{\theta}^{z}=\Phi_{\theta,t_{1}}^{z}\circ\Phi_{\theta,t_{2}}^{z}\circ\cdots\circ\Phi_{\theta,t_{n}}^{z}, where a_{t_{j-1}}=\Phi_{\theta,t_{j}}^{z}(a_{t_{j}})=a_{t_{j}}-(t_{j}-t_{j-1})v_{\theta}(a_{t_{j}},t_{j}). Here, a_{t_{j}} denotes a noisy action chunk and v_{\theta} the predicted velocity.

### 3.2 Consistency Distillation for Action Generation

Consistency distillation (CD)([Song et al., 2023](https://arxiv.org/html/2610.06617#bib.bib21)) learns to map noisy states (_i.e._, action chunks in our paper) along the same teacher denoising trajectory to a common clean endpoint (_i.e._, t=0), enabling few-step generation. It uses three models: a frozen teacher \theta_{\mathrm{T}}, a trainable student \theta_{\mathrm{S}}, and an exponential moving average (EMA) target \theta_{\mathrm{ema}}. During training, given a clean action chunk a_{0}^{\mathrm{data}} from the training data, we randomly sample t\in[0,1] and Gaussian noise \epsilon\sim\mathcal{N}(0,I) to construct a_{t}=(1-t)a_{0}^{\mathrm{data}}+t\epsilon. The student and EMA target use the endpoint mapping

f_{\theta}(a_{t},t)=a_{t}-t\,v_{\theta}(a_{t},t).(3)

For a neighboring time 0\leq s<t, the teacher produces

\tilde{a}_{s}=\texttt{Solver}(a_{t},t,s;\theta_{\mathrm{T}})\approx a_{t}-(t-s)\,v_{\theta_{\mathrm{T}}}(a_{t},t),(4)

where \texttt{Solver}(a_{t},t,s;\theta_{\mathrm{T}}) denotes one numerical integration step from t to s using the frozen teacher \theta_{\mathrm{T}}. For neighboring time steps with small t-s, treating the velocity as locally constant yields the first-order approximation above. The student is optimized with

\mathcal{L}_{\mathrm{CD}}=\mathbb{E}_{a_{t},t,s}\!\left[\left\|f_{\theta_{\mathrm{S}}}(a_{t},t)-\operatorname{sg}\!\left[f_{\theta_{\mathrm{ema}}}(\tilde{a}_{s},s)\right]\right\|_{2}^{2}\right],(5)

where \operatorname{sg} denotes stop-gradient. After each student update, \theta_{\mathrm{ema}}\leftarrow\mu\theta_{\mathrm{ema}}+(1-\mu)\theta_{\mathrm{S}}, with decay rate \mu\in[0,1). Therefore, consistency distillation learns to map noisy action chunks directly to clean endpoints, enabling few-step action generation without traversing the full teacher trajectory.

## 4 RealtimeWAM

In this work, we propose RealtimeWAM, a one-step asynchronous world action model for post-training acceleration of any MoT-based WAM. RealtimeWAM addresses intra-expert iteration through Teacher-Anchored Consistency Distillation (Sec.[4.1](https://arxiv.org/html/2610.06617#S4.SS1 "4.1 Teacher-Anchored Consistency Distillation ‣ 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models")) and inter-expert waiting through Cross-Expert Wavefront Pipelining (Sec.[4.2](https://arxiv.org/html/2610.06617#S4.SS2 "4.2 Cross-Expert Wavefront Pipelining ‣ 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models")).

### 4.1 Teacher-Anchored Consistency Distillation

![Image 1: Refer to caption](https://arxiv.org/html/2610.06617v1/method1_new.png)

Figure 2: Overview of Teacher-Anchored Consistency Distillation (TACD). The frozen Video Expert provides shared KV conditioning, while the action student is trained with local consistency supervision and a teacher-anchored constraint derived from the frozen teacher’s multi-step rollout.

As discussed above, multi-step action denoising dominates inference latency (Fig.[1](https://arxiv.org/html/2610.06617#S1.F1 "Figure 1 ‣ 1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models")) while video latent representations are computed in a single forward pass in the Video Expert. Therefore, our target is to achieve high-quality few-step action generation (especially one-step) to combat intra-expert iteration. Recent step distillation strategies([Song et al., 2023](https://arxiv.org/html/2610.06617#bib.bib21); [Yin et al., 2024](https://arxiv.org/html/2610.06617#bib.bib22); [Geng et al., 2025](https://arxiv.org/html/2610.06617#bib.bib20); [Huang et al., 2026](https://arxiv.org/html/2610.06617#bib.bib49)) have been extensively studied in image and video generation. However, we identify a distinctive characteristic of action generation in WAMs: denoising is strongly conditioned on visual observations, robot states, and language instructions, which can confine feasible actions (_e.g._, grasp positions and gripper commands) to a narrow range. Within this constrained action space, even subtle deviations can affect task success rates owing to the high sensitivity of robotic control to action errors.

![Image 2: Refer to caption](https://arxiv.org/html/2610.06617v1/loss_curves3.png)

Figure 3: Training curves of two distillation methods: consistency loss (Left) and squared distance between the student’s velocity v_{\theta_{\mathrm{S}}} and the teacher’s average velocity towards the clean endpoint u_{\theta_{\mathrm{T}}} (Right).

However, directly applying consistency distillation to WAMs raises a significant challenge, dubbed the local-global error gap. Specifically, although consistency distillation([Song et al., 2023](https://arxiv.org/html/2610.06617#bib.bib21)) can effectively minimize local consistency errors, it does not considerably reduce deviations from the global target action. As shown in Fig.[3](https://arxiv.org/html/2610.06617#S4.F3 "Figure 3 ‣ 4.1 Teacher-Anchored Consistency Distillation ‣ 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models") (right), the error between the teacher’s clean action targets and student’s predictions does not decrease considerably for consistency distillation. Such large deviations from the clean action targets can induce inaccuracies in robotic control.

Formally, let a_{t} denote a noisy action chunk and \tilde{a}_{s} the teacher-updated action defined analogously to Eq.([4](https://arxiv.org/html/2610.06617#S3.E4 "In 3.2 Consistency Distillation for Action Generation ‣ 3 Preliminaries ‣ RealtimeWAM: One-Step Asynchronous World Action Models")). Let a_{0}^{\star} denote the exact clean endpoint of the frozen teacher ODE initialized at a_{t}, where \star indicates the theoretical exact solution. Omitting the shared conditioning z(o,l) for brevity, the total error decomposes into a local consistency error and a global target error:

\underbrace{f_{\theta_{\mathrm{S}}}(a_{t},t)-a_{0}^{\star}}_{\mathbf{e}_{\mathrm{total}}:\ \text{Total error}}=\underbrace{f_{\theta_{\mathrm{S}}}(a_{t},t)-f_{\theta_{\mathrm{ema}}}(\tilde{a}_{s},s)}_{\mathbf{e}_{\mathrm{local}}:\ \text{Local consistency error}}+\underbrace{f_{\theta_{\mathrm{ema}}}(\tilde{a}_{s},s)-a_{0}^{\star}}_{\mathbf{e}_{\mathrm{global}}:\ \text{Global target error}}.(6)

Consistency Distillation directly minimizes \mathbb{E}\|\mathbf{e}_{\mathrm{local}}\|_{2}^{2} but does not explicitly penalize \mathbf{e}_{\mathrm{global}}. Consequently, a small local consistency error alone does not guarantee a small total error \mathbf{e}_{\mathrm{total}} (see Appendix[C.1](https://arxiv.org/html/2610.06617#A3.SS1 "C.1 Local Consistency and Endpoint Error ‣ Appendix C Complete Error Derivation ‣ RealtimeWAM: One-Step Asynchronous World Action Models")). Therefore, our key insight is to preserve local consistency supervision while introducing an additional constraint based on the teacher’s target action (_i.e._, a_{0}^{\star}), thereby mitigating the global deviation that the local objective may leave unresolved.

Motivated by this insight, we propose Teacher-Anchored Consistency Distillation (TACD), a novel post-training step distillation framework tailored for MoT-based WAMs. Specifically, since the exact endpoint a_{0}^{\star} is unavailable in practice, we adopt the frozen teacher’s multi-step rollout endpoint a_{0}^{\mathrm{T}} as a proxy and compute the corresponding interval-averaged velocity. It is worth noting that the three Action Experts (_i.e._, the frozen teacher, trainable student, and EMA target) share the same video KV cache z(o,l) from a single frozen Video Expert. Starting from the same noisy action a_{t} under this shared conditioning z(o,l), we have

a_{0}^{\mathrm{T}}=\texttt{Solver}(a_{t},t,0;\theta_{\mathrm{T}}),\qquad u_{\theta_{\mathrm{T}}}(a_{t},t)=\frac{a_{t}-a_{0}^{\mathrm{T}}}{t},\quad t>0,(7)

where Solver denotes numerical integration from t to 0 using K teacher denoising steps.

We further directly supervise the student’s velocity prediction v_{\theta_{\mathrm{S}}}(a_{t},t) with the teacher’s interval-averaged velocity u_{\theta_{\mathrm{T}}}(a_{t},t), yielding the teacher-anchored loss and the overall training objective:

\mathcal{L}_{\mathrm{TA}}=\mathbb{E}_{a_{t},t}\!\left[\left\|v_{\theta_{\mathrm{S}}}(a_{t},t)-\operatorname{sg}\!\left[u_{\theta_{\mathrm{T}}}(a_{t},t)\right]\right\|_{2}^{2}\right],\qquad\mathcal{L}=\mathcal{L}_{\mathrm{CD}}+\lambda\mathcal{L}_{\mathrm{TA}},(8)

where \lambda>0 weights the teacher-anchored loss. Equivalently, \mathcal{L}_{\mathrm{TA}} directly supervises the student’s clean action prediction against the teacher’s rollout endpoint, with weight 1/t^{2} (see Appendix[C.2](https://arxiv.org/html/2610.06617#A3.SS2 "C.2 Equivalence to Weighted Endpoint Supervision ‣ Appendix C Complete Error Derivation ‣ RealtimeWAM: One-Step Asynchronous World Action Models")).

To relate this supervision to the exact teacher endpoint, let \bm{\delta}=a_{0}^{\mathrm{T}}-a_{0}^{\star} denote the teacher’s numerical integration error. The total endpoint error in Eq.([6](https://arxiv.org/html/2610.06617#S4.E6 "In 4.1 Teacher-Anchored Consistency Distillation ‣ 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models")) then satisfies

\mathbf{e}_{\mathrm{total}}=\mathbf{e}_{\mathrm{local}}+\mathbf{e}_{\mathrm{global}}=-t\!\left[v_{\theta_{\mathrm{S}}}(a_{t},t)-u_{\theta_{\mathrm{T}}}(a_{t},t)\right]+\bm{\delta}.(9)

Consequently, teacher anchoring directly constrains the combined error \mathbf{e}_{\mathrm{local}}+\mathbf{e}_{\mathrm{global}}, complementing the local consistency objective with a direct endpoint constraint 1 1 1 The total endpoint error \mathbf{e}_{\mathrm{total}} also depends on the teacher’s numerical integration error \bm{\delta}, which can be controlled by increasing the number of teacher denoising steps K (see Appendix[C.3](https://arxiv.org/html/2610.06617#A3.SS3 "C.3 Total Endpoint Error Bound ‣ Appendix C Complete Error Derivation ‣ RealtimeWAM: One-Step Asynchronous World Action Models")).. In addition, for 0<t\leq 1, this relation yields an upper bound on the expected squared endpoint error, \mathbb{E}\|\mathbf{e}_{\mathrm{total}}\|_{2}^{2}\leq 2\mathcal{L}_{\mathrm{TA}}+2\mathbb{E}\|\bm{\delta}\|_{2}^{2} (see Appendix[C.3](https://arxiv.org/html/2610.06617#A3.SS3 "C.3 Total Endpoint Error Bound ‣ Appendix C Complete Error Derivation ‣ RealtimeWAM: One-Step Asynchronous World Action Models")).

### 4.2 Cross-Expert Wavefront Pipelining

Figure 4: Comparison of expert execution schedules. (a) Conventional execution waits for the complete video KV cache. (b) Cross-Expert Wavefront Pipelining releases block-wise KV cache entries after projection and synchronizes only before action attention consumes them, overlapping the Video and Action Experts.

Teacher-Anchored Consistency Distillation reduces intra-expert iteration by enabling one-step action generation. However, inter-expert waiting persists because the Video and Action Experts still execute sequentially (Fig.[4](https://arxiv.org/html/2610.06617#S4.F4 "Figure 4 ‣ 4.2 Cross-Expert Wavefront Pipelining ‣ 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models")(a)). Specifically, the Action Expert waits for the complete video KV cache, creating a global synchronization barrier that prevents action computation from starting even when its required context is already available.

We therefore analyze the actual data dependencies between the Video Expert and Action Expert in a fine-grained manner. We first decompose each DiT block into three sequential components: Q/K/V projection \mathcal{P}_{i}, attention \operatorname{Attn}_{i}, and the remaining computation \mathcal{R}_{i}. Let h_{i-1}^{v} and h_{i-1}^{a} denote the video and action hidden states entering block i\in\{1,\ldots,N\}, respectively. Their projections are

(Q_{i}^{v},K_{i}^{v},V_{i}^{v})=\mathcal{P}_{i}^{v}(h_{i-1}^{v}),\qquad(Q_{i}^{a},K_{i}^{a},V_{i}^{a})=\mathcal{P}_{i}^{a}(h_{i-1}^{a}).(10)

Here, (K_{i}^{v},V_{i}^{v}) constitute the i-th entry of z(o,l) in Eq.([1](https://arxiv.org/html/2610.06617#S3.E1 "In 3.1 Efficient Mixture-of-Transformers (MoT)-based WAMs ‣ 3 Preliminaries ‣ RealtimeWAM: One-Step Asynchronous World Action Models")). The Video Expert attends to its own tokens, while the Action Expert additionally incorporates the corresponding video KV cache:

\displaystyle h_{i}^{v}\displaystyle=\mathcal{R}_{i}^{v}\!\left(h_{i-1}^{v},\operatorname{Attn}_{i}^{v}\!\left(Q_{i}^{v},K_{i}^{v},V_{i}^{v}\right)\right),(11)
\displaystyle h_{i}^{a}\displaystyle=\mathcal{R}_{i}^{a}\!\left(h_{i-1}^{a},\operatorname{Attn}_{i}^{a}\!\left(Q_{i}^{a},K_{i}^{v}\mathbin{\|}K_{i}^{a},V_{i}^{v}\mathbin{\|}V_{i}^{a}\right)\right),

where \mathbin{\|} denotes token-wise concatenation. Action attention \operatorname{Attn}_{i}^{a} is thus the only operation in action block i that directly consumes the video KV cache (K_{i}^{v},V_{i}^{v}), with the dependency

\mathcal{P}_{i}^{v}\longrightarrow\operatorname{Attn}_{i}^{a}\longleftarrow\mathcal{P}_{i}^{a}.(12)

Our key observations are: ➀ The action projection \mathcal{P}_{i}^{a} can execute once h_{i-1}^{a} is available, without waiting for the video KV cache (K_{i}^{v},V_{i}^{v}). ➁ Action attention \operatorname{Attn}_{i}^{a} can begin once both \mathcal{P}_{i}^{a} and \mathcal{P}_{i}^{v} complete, without waiting for video attention \operatorname{Attn}_{i}^{v} or video states h_{j}^{v} with j\geq i. ➂ Existing methods([Yuan et al., 2026](https://arxiv.org/html/2610.06617#bib.bib4); [Zhao et al., 2026](https://arxiv.org/html/2610.06617#bib.bib15)) enforce \mathcal{R}_{N}^{v}\rightarrow\mathcal{P}_{1}^{a}, introducing unnecessary waiting.

Based on this dependency analysis, we propose Cross-Expert Wavefront Pipelining (CEWP), which transforms coarse expert-wise execution into a fine-grained block-wise wavefront across the Video and Action Experts. We assign the Video and Action Experts to two non-blocking CUDA streams, \mathcal{S}_{v} and \mathcal{S}_{a}, respectively. Each stream preserves program order, while a CUDA event e_{i} enforces the dependency at block i:

\displaystyle\mathcal{S}_{v}:\displaystyle\mathcal{P}_{i}^{v}\rightarrow\operatorname{record}(e_{i})\rightarrow\operatorname{Attn}_{i}^{v}\rightarrow\mathcal{R}_{i}^{v},(13)
\displaystyle\mathcal{S}_{a}:\displaystyle\mathcal{P}_{i}^{a}\rightarrow\operatorname{wait}(e_{i})\rightarrow\operatorname{Attn}_{i}^{a}\rightarrow\mathcal{R}_{i}^{a},

where e_{i} signals the availability of (K_{i}^{v},V_{i}^{v}). The video stream records e_{i} immediately after \mathcal{P}_{i}^{v} and continues its remaining computation, while the action stream waits only before \operatorname{Attn}_{i}^{a} (shown in Fig.[4](https://arxiv.org/html/2610.06617#S4.F4 "Figure 4 ‣ 4.2 Cross-Expert Wavefront Pipelining ‣ 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models")(b)). This late synchronization allows the action projection to proceed independently and delays cross-stream waiting until the video KV cache is actually consumed. Action blocks thus consume video KV cache entries incrementally as the video stream advances, forming a staggered wavefront along model depth.

## 5 Experiments

### 5.1 Implementation Details

We implement RealtimeWAM based on Fast-WAM([Yuan et al., 2026](https://arxiv.org/html/2610.06617#bib.bib4)) and Faster-WAM([Zhao et al., 2026](https://arxiv.org/html/2610.06617#bib.bib15)), yielding RealtimeWAM* and RealtimeWAM†, respectively. We freeze the Video Expert and optimize only rank-128 low-rank adaptation (LoRA)([Hu et al., 2021](https://arxiv.org/html/2610.06617#bib.bib42)) parameters in the Action Expert. In Teacher-Anchored Consistency Distillation (TACD), we obtain the teacher endpoint through a 10-step rollout of the frozen teacher and set the teacher-anchored loss weight to \lambda=0.2. Training proceeds for 30,000 iterations with AdamW, a learning rate of 10^{-4}, and a weight decay of 10^{-2}. We set the exponential moving average (EMA) decay to 0.995. For a fair comparison, all evaluated step-distillation methods, including distribution matching distillation (DMD)([Yin et al., 2024](https://arxiv.org/html/2610.06617#bib.bib22)), consistency distillation([Song et al., 2023](https://arxiv.org/html/2610.06617#bib.bib21)), and MeanFlow distillation([Geng et al., 2025](https://arxiv.org/html/2610.06617#bib.bib20); [Gu et al., 2026](https://arxiv.org/html/2610.06617#bib.bib51)), adopt the same training settings. Moreover, we compare against other efficient WAMs, including Flash-WAM([Akbari et al., 2026](https://arxiv.org/html/2610.06617#bib.bib19)) and Light-WAM([Li et al., 2026c](https://arxiv.org/html/2610.06617#bib.bib16)). For latency evaluation, we include VAE encoding time but exclude text encoding time, as instructions are encoded only once per episode.

### 5.2 Evaluation Benchmarks

We evaluate RealtimeWAM on LIBERO([Liu et al., 2023](https://arxiv.org/html/2610.06617#bib.bib1)), RoboTwin 2.0([Chen et al., 2025](https://arxiv.org/html/2610.06617#bib.bib2)), and LIBERO-Plus([Fei et al., 2025](https://arxiv.org/html/2610.06617#bib.bib3)), using task success rate as the evaluation metric. On LIBERO, we evaluate four task suites: Spatial, Object, Goal, and Long. For RoboTwin 2.0, we follow the multi-task training setting in previous work([Yuan et al., 2026](https://arxiv.org/html/2610.06617#bib.bib4)), training on 50 tasks using a total of 2,500 clean demonstrations and 25,000 randomized demonstrations. We evaluate performance under both clean and randomized conditions. To assess out-of-distribution (OOD) robustness, we further evaluate success rates on all seven perturbation subsets of LIBERO-Plus. We additionally evaluate RealtimeWAM on real-world dual-arm manipulation tasks (see Appendix[F.6](https://arxiv.org/html/2610.06617#A6.SS6 "F.6 Real-World Evaluation of Long-Horizon Garment Folding ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models")).

### 5.3 Main Results

RoboTwin 2.0. We report detailed results on RoboTwin 2.0 (Tab.[1](https://arxiv.org/html/2610.06617#S5.T1 "Table 1 ‣ 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models")). Notably, RealtimeWAM* achieves a success rate of 90.84% with one-step action generation using only LoRA fine-tuning, incurring a reduction of just 0.67% compared with 10-step Fast-WAM. Surprisingly, RealtimeWAM† achieves nearly lossless performance with one-step action generation, attaining a success rate of 92.64% with a reduction of only 0.29% compared with 10-step Faster-WAM. It also considerably outperforms other efficient WAMs (_e.g._, 81.41% for one-step Flash-WAM) and step-distillation baselines (_e.g._, 88.22% for DMD*).

LIBERO. A similar trend is observed on LIBERO (Tab.[1](https://arxiv.org/html/2610.06617#S5.T1 "Table 1 ‣ 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models")). RealtimeWAM* achieves an overall success rate of 97.0% with one-step action generation, matching the performance of 10-step Fast-WAM. Compared with directly reducing Fast-WAM to a single denoising step (95.0%), RealtimeWAM* improves the success rate by 2.0%, demonstrating the effectiveness of step distillation in preserving policy performance. It also outperforms one-step Flash-WAM (95.1%) and DMD* (96.1%), while remaining competitive with Light-WAM (97.2%). Detailed results for individual LIBERO task suites are provided in Appendix[F.3](https://arxiv.org/html/2610.06617#A6.SS3 "F.3 Detailed LIBERO Results ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models") (Tab.[F.1](https://arxiv.org/html/2610.06617#A6.T1 "Table F.1 ‣ F.3 Detailed LIBERO Results ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models")).

Table 1: Success rates (%) on RoboTwin 2.0 and LIBERO. NFE denotes the number of denoising steps for video and action generation. Methods marked with * and † are built on Fast-WAM and Faster-WAM, respectively.

Method NFE RoboTwin 2.0 LIBERO
Video Action Clean\uparrow Random\uparrow Overall\uparrow Overall\uparrow
WAM and VLA Baselines
\pi_{0}([Black et al., 2024](https://arxiv.org/html/2610.06617#bib.bib10))––65.92 58.40 62.16 94.1
\pi_{0.5}([Intelligence et al., 2025](https://arxiv.org/html/2610.06617#bib.bib7))––82.74 76.76 79.75 96.9
X-VLA([Zheng et al., 2026](https://arxiv.org/html/2610.06617#bib.bib18))––72.90 72.80 72.85 98.1
Motus([Bi et al., 2026](https://arxiv.org/html/2610.06617#bib.bib12))10 10 88.66 87.02 87.84 97.7
LingBot-VA([Li et al., 2026a](https://arxiv.org/html/2610.06617#bib.bib13))25 50 92.90 91.50 92.20 98.5
Fast-WAM([Yuan et al., 2026](https://arxiv.org/html/2610.06617#bib.bib4))1 10 91.82 91.19 91.51 97.0
Faster-WAM([Zhao et al., 2026](https://arxiv.org/html/2610.06617#bib.bib15))1 10 93.20 92.66 92.93 98.9
Other Efficient Variants
Flash-WAM([Akbari et al., 2026](https://arxiv.org/html/2610.06617#bib.bib19))1 2 88.42 82.66 85.54 96.1
Flash-WAM([Akbari et al., 2026](https://arxiv.org/html/2610.06617#bib.bib19))1 1 82.56 80.26 81.41 95.1
Light-WAM([Li et al., 2026c](https://arxiv.org/html/2610.06617#bib.bib16))1 1 76.40 76.30 76.40 97.2
Few-Step Distillation
Fast-WAM([Yuan et al., 2026](https://arxiv.org/html/2610.06617#bib.bib4))1 1 70.42 68.12 69.27 95.0
Mean Flow*([Geng et al., 2025](https://arxiv.org/html/2610.06617#bib.bib20))1 1 85.74 85.72 85.73 96.9
Consistency Distillation*([Song et al., 2023](https://arxiv.org/html/2610.06617#bib.bib21))1 1 89.88 89.38 89.63 96.9
DMD*([Yin et al., 2024](https://arxiv.org/html/2610.06617#bib.bib22))1 1 88.64 87.80 88.22 96.1
RealtimeWAM*1 1 91.96 89.72 90.84 97.0
RealtimeWAM†1 1 92.98 92.30 92.64 99.0

LIBERO-Plus. We further evaluate robustness to distribution shifts on LIBERO-Plus (Tab.[2](https://arxiv.org/html/2610.06617#S5.T2 "Table 2 ‣ 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models")). With one-step action generation, RealtimeWAM† achieves an overall success rate of 73.0%, only 0.6% below Faster-WAM (73.6%). It slightly improves performance under camera, sensor-noise, and layout perturbations, indicating that step distillation largely preserves out-of-distribution robustness.

Table 2: Success rates (%) on the seven perturbation subsets of LIBERO-Plus.

Method Camera\uparrow Robot\uparrow Lang.\uparrow Light\uparrow Backg.\uparrow Noise\uparrow Layout\uparrow Overall\uparrow
UniVLA([Bu et al., 2025](https://arxiv.org/html/2610.06617#bib.bib45))1.8 46.2 69.6 69.0 81.0 21.2 31.9 42.9
OpenVLA-OFT([Kim et al., 2025](https://arxiv.org/html/2610.06617#bib.bib46))56.4 31.9 79.5 88.7 93.3 75.8 74.2 69.6
\pi_{0}([Black et al., 2024](https://arxiv.org/html/2610.06617#bib.bib10))13.8 6.0 58.8 85.0 81.4 79.0 68.9 53.6
\pi_{0}-Fast([Pertsch et al., 2025](https://arxiv.org/html/2610.06617#bib.bib47))65.1 21.6 61.0 73.2 73.2 74.4 68.8 61.6
WorldVLA([Cen et al., 2025](https://arxiv.org/html/2610.06617#bib.bib48))0.1 27.9 41.6 43.7 17.1 10.9 38.0 25.0
Fast-WAM([Yuan et al., 2026](https://arxiv.org/html/2610.06617#bib.bib4))18.8 45.7 70.1 83.2 45.7 29.8 62.7 49.1
Faster-WAM([Zhao et al., 2026](https://arxiv.org/html/2610.06617#bib.bib15))53.8 71.6 94.7 96.3 61.3 63.6 79.1 73.6
RealtimeWAM†54.2 69.0 92.5 94.7 59.4 64.6 79.3 73.0

### 5.4 Ablation Study

Table 3: Ablation of loss terms and expert fine-tuning on RoboTwin 2.0. \mathcal{L}_{\mathrm{CD}} and \mathcal{L}_{\mathrm{TA}} denote the consistency and teacher-anchored losses, respectively. Video/Action indicate whether each expert is fine-tuned. Results are success rates (%); best results are in bold.

\mathcal{L}_{\mathrm{CD}}\mathcal{L}_{\mathrm{TA}}Video Action Clean\uparrow Random\uparrow Overall\uparrow
✓✗✓✓89.88 89.38 89.63
✓✗✗✓89.76 89.94 89.85
✓✓✗✓91.96 89.72 90.84

Table 4: Effect of teacher rollout steps K in TACD on RoboTwin 2.0. K denotes the number of teacher denoising steps used to obtain a_{0}^{\mathrm{T}}. We use K=10 by default. Results are success rates (%); best results are in bold.

K Clean\uparrow Random\uparrow Overall\uparrow
5 91.02 89.76 90.39
10 91.96 89.72 90.84
20 91.22 90.32 90.77

Effect of different components. We first validate the effectiveness of the key components of our distillation framework (Tab.[4](https://arxiv.org/html/2610.06617#S5.T4 "Table 4 ‣ 5.4 Ablation Study ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models")). Fine-tuning the Video Expert during action step distillation reduces the overall success rate while introducing additional training overhead (_i.e._, longer training time and higher peak memory usage, as reported in Appendix[A](https://arxiv.org/html/2610.06617#A1 "Appendix A Limitations ‣ RealtimeWAM: One-Step Asynchronous World Action Models")). With the Video Expert frozen, augmenting \mathcal{L}_{\mathrm{CD}} with \mathcal{L}_{\mathrm{TA}}, yielding Teacher-Anchored Consistency Distillation (TACD), further improves the reported overall success rate from 89.85% to 90.84%. This improvement is consistent with our analysis: the teacher anchor provides an endpoint reference from the complete teacher trajectory, supplementing local consistency supervision with an explicit endpoint constraint to mitigate global deviations that the local objective may leave unresolved.

Choice of teacher rollout steps K. We further present results for different numbers of teacher rollout steps K (Tab.[4](https://arxiv.org/html/2610.06617#S5.T4 "Table 4 ‣ 5.4 Ablation Study ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models")). With K=5, the rollout may produce a less accurate approximation a_{0}^{\mathrm{T}} to the exact clean endpoint a_{0}^{\star}, resulting in a larger numerical integration error \bm{\delta} and weaker endpoint supervision. This setting achieves a success rate of 90.39%, which improves to 90.84% when K increases to 10. Further increasing K to 20 yields no additional improvement (90.77%) while requiring more teacher computation. We therefore adopt K=10 to balance policy performance and teacher rollout cost.

Figure 5: Inference latency of RealtimeWAM built on Fast-WAM (left) and Faster-WAM (right) on a single NVIDIA H100 GPU.

Efficiency Results. RealtimeWAM reduces inference latency through algorithm and system co-optimization (Fig.[5](https://arxiv.org/html/2610.06617#S5.F5 "Figure 5 ‣ 5.4 Ablation Study ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models")). In addition to TACD and CEWP, we employ CUDA Graph and efficient kernels from LightX2V([LightX2V Contributors, 2025](https://arxiv.org/html/2610.06617#bib.bib43)) (see Appendix[F.2](https://arxiv.org/html/2610.06617#A6.SS2 "F.2 Efficient Kernel Implementation ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models")). On Fast-WAM and Faster-WAM, TACD reduces latency from 299.7 and 218.9 ms to 65.8 and 57.1 ms, yielding 4.56\times and 3.83\times speedups, respectively. After enabling CUDA Graph, CEWP further reduces latency from 23.3 to 17.4 ms on Fast-WAM and from 26.2 to 22.4 ms on Faster-WAM, providing additional speedups of 1.34\times and 1.17\times. With all optimizations enabled, RealtimeWAM achieves inference latencies of 12.2 and 16.1 ms, corresponding to overall speedups of 24.55\times and 13.56\times over the respective baselines. Detailed latency results on other GPUs (_e.g._, RTX 4090D and RTX 5090) are provided in Appendix[F.5](https://arxiv.org/html/2610.06617#A6.SS5 "F.5 Latency on Additional GPUs ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models") (Tab.[F.3](https://arxiv.org/html/2610.06617#A6.T3 "Table F.3 ‣ F.5 Latency on Additional GPUs ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models")).

## 6 Conclusion

We presented RealtimeWAM, a post-training framework that accelerates MoT-based WAMs by addressing intra-expert iteration and inter-expert waiting. Teacher-Anchored Consistency Distillation complements local consistency supervision with a teacher endpoint constraint, enabling accurate one-step action generation. Cross-Expert Wavefront Pipelining overlaps block-wise execution of the Video and Action Experts through fine-grained synchronization, reducing unnecessary waiting. RealtimeWAM enables real-time action generation (_e.g._, 12.2 ms latency) with near-lossless policy performance on MoT-based WAMs, including Fast-WAM and Faster-WAM, while delivering significant end-to-end speedups across diverse hardwares (_e.g._, approximately 14\sim 25\times on H100).

### AI use statement

In this work, we used generative AI tools solely to improve the language and readability of the manuscript. We did not use these tools for any tasks requiring disclosure under the ICLR 2027 AI Policy for Authors. The authors reviewed all AI-assisted edits to ensure that they preserved the original technical meaning. We take responsibility for the final content of this work, including the AI-assisted language edits.

## References

*   Akbari et al. (2026)A. Akbari, C. Zhang, A. Akbari, L. Zhao, Y. Chen, W. Chen, X. Zhang, G. Yuan, and Y. Wang Flash-wam: modality-aware distillation for world action models. arXiv preprint arXiv:2606.05254. Cited by: [§F.1](https://arxiv.org/html/2610.06617#A6.SS1.p1.1 "F.1 Result sources and aggregation ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table F.1](https://arxiv.org/html/2610.06617#A6.T1.2.1.12.1 "In F.3 Detailed LIBERO Results ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table F.1](https://arxiv.org/html/2610.06617#A6.T1.2.1.13.1 "In F.3 Detailed LIBERO Results ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§2.2](https://arxiv.org/html/2610.06617#S2.SS2.p1.1 "2.2 Few-Step Distillation ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§5.1](https://arxiv.org/html/2610.06617#S5.SS1.p1.1 "5.1 Implementation Details ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table 1](https://arxiv.org/html/2610.06617#S5.T1.6.1.12.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table 1](https://arxiv.org/html/2610.06617#S5.T1.6.1.13.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Albergo et al. (2023)M. S. Albergo, N. M. Boffi, and E. Vanden-Eijnden Stochastic interpolants: a unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797. Cited by: [§2.2](https://arxiv.org/html/2610.06617#S2.SS2.p1.1 "2.2 Few-Step Distillation ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Bi et al. (2026)H. Bi, H. Tan, S. Xie, Z. Wang, S. Huang, H. Liu, R. Zhao, Y. Feng, C. Xiang, Y. Rong, et al.Motus: a unified latent action world model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp.35101–35113. Cited by: [Table F.1](https://arxiv.org/html/2610.06617#A6.T1.2.1.7.1 "In F.3 Detailed LIBERO Results ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§1](https://arxiv.org/html/2610.06617#S1.p1.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§2.1](https://arxiv.org/html/2610.06617#S2.SS1.p1.1 "2.1 World Action Models ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§3.1](https://arxiv.org/html/2610.06617#S3.SS1.p1.1 "3.1 Efficient Mixture-of-Transformers (MoT)-based WAMs ‣ 3 Preliminaries ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table 1](https://arxiv.org/html/2610.06617#S5.T1.6.1.7.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Bjorck et al. (2025)J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, et al.Gr00t n1: an open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734. Cited by: [§1](https://arxiv.org/html/2610.06617#S1.p1.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§2.1](https://arxiv.org/html/2610.06617#S2.SS1.p1.1 "2.1 World Action Models ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Black et al. (2024)K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al.\pi_{0}: a vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164. Cited by: [Table F.1](https://arxiv.org/html/2610.06617#A6.T1.2.1.4.1 "In F.3 Detailed LIBERO Results ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§1](https://arxiv.org/html/2610.06617#S1.p1.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§2.1](https://arxiv.org/html/2610.06617#S2.SS1.p1.1 "2.1 World Action Models ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table 1](https://arxiv.org/html/2610.06617#S5.T1.6.1.4.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table 2](https://arxiv.org/html/2610.06617#S5.T2.2.1.4.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Black et al. (2025a)K. Black, M. Y. Galliker, and S. Levine Real-time execution of action chunking flow policies. arXiv preprint arXiv:2506.07339. Cited by: [§F.6](https://arxiv.org/html/2610.06617#A6.SS6.p2.1 "F.6 Real-World Evaluation of Long-Horizon Garment Folding ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§2.3](https://arxiv.org/html/2610.06617#S2.SS3.p1.1 "2.3 Asynchronous Inference for Robot Policies ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Black et al. (2025b)K. Black, A. Z. Ren, M. Equi, and S. Levine Training-time action conditioning for efficient real-time chunking. arXiv preprint arXiv:2512.05964. Cited by: [§2.3](https://arxiv.org/html/2610.06617#S2.SS3.p1.1 "2.3 Asynchronous Inference for Robot Policies ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Bu et al. (2025)Q. Bu, Y. Yang, J. Cai, S. Gao, G. Ren, M. Yao, P. Luo, and H. Li Univla: learning to act anywhere with task-centric latent actions. arXiv preprint arXiv:2505.06111. Cited by: [Table 2](https://arxiv.org/html/2610.06617#S5.T2.2.1.2.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Cai et al. (2026)J. Cai, L. Ling, S. Chu, Z. Liu, J. Kang, Z. Liang, W. Xu, Y. Mao, W. Zhang, X. Yang, et al.AHA-wam: asynchronous horizon-adaptive world-action modeling with observation-guided context routing. arXiv preprint arXiv:2606.09811. Cited by: [§2.3](https://arxiv.org/html/2610.06617#S2.SS3.p1.1 "2.3 Asynchronous Inference for Robot Policies ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Cen et al. (2025)J. Cen, C. Yu, H. Yuan, Y. Jiang, S. Huang, J. Guo, X. Li, Y. Song, H. Luo, F. Wang, et al.Worldvla: towards autoregressive action world model. arXiv preprint arXiv:2506.21539. Cited by: [Table 2](https://arxiv.org/html/2610.06617#S5.T2.2.1.6.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Chen et al. (2025)T. Chen, Z. Chen, B. Chen, Z. Cai, Y. Liu, Z. Li, Q. Liang, X. Lin, Y. Ge, Z. Gu, et al.RoboTwin 2.0: a scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088. Cited by: [§1](https://arxiv.org/html/2610.06617#S1.p7.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§5.2](https://arxiv.org/html/2610.06617#S5.SS2.p1.1 "5.2 Evaluation Benchmarks ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Fei et al. (2025)S. Fei, S. Wang, J. Shi, Z. Dai, J. Cai, P. Qian, L. Ji, X. He, S. Zhang, Z. Fei, J. Fu, J. Gong, and X. Qiu LIBERO-Plus: in-depth robustness analysis of vision-language-action models. arXiv preprint arXiv:2510.13626. Cited by: [§1](https://arxiv.org/html/2610.06617#S1.p7.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§5.2](https://arxiv.org/html/2610.06617#S5.SS2.p1.1 "5.2 Evaluation Benchmarks ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Geng et al. (2025)Z. Geng, M. Deng, X. Bai, J. Z. Kolter, and K. He Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447. Cited by: [Table F.1](https://arxiv.org/html/2610.06617#A6.T1.2.1.17.1 "In F.3 Detailed LIBERO Results ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§2.2](https://arxiv.org/html/2610.06617#S2.SS2.p1.1 "2.2 Few-Step Distillation ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§4.1](https://arxiv.org/html/2610.06617#S4.SS1.p1.1 "4.1 Teacher-Anchored Consistency Distillation ‣ 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§5.1](https://arxiv.org/html/2610.06617#S5.SS1.p1.1 "5.1 Implementation Details ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table 1](https://arxiv.org/html/2610.06617#S5.T1.6.1.17.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Gu et al. (2026)Y. Gu, G. Fang, Y. Jiang, W. Mao, S. Han, H. Cai, and M. Z. Shou Anyflow: any-step video diffusion model with on-policy flow map distillation. In European Conference on Computer Vision, pp.163–181. Cited by: [§5.1](https://arxiv.org/html/2610.06617#S5.SS1.p1.1 "5.1 Implementation Details ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Hairer et al. (1993)E. Hairer, S.P. Nørsett, and G. Wanner Solving ordinary differential equations i: nonstiff problems. Springer. Cited by: [§C.3](https://arxiv.org/html/2610.06617#A3.SS3.SSS0.Px1.p1.1 "Teacher numerical integration error. ‣ C.3 Total Endpoint Error Bound ‣ Appendix C Complete Error Derivation ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Hirose et al. (2026)N. Hirose, C. Glossop, D. Shah, and S. Levine Asyncvla: an asynchronous vla for fast and robust navigation on the edge. arXiv preprint arXiv:2602.13476. Cited by: [§2.3](https://arxiv.org/html/2610.06617#S2.SS3.p1.1 "2.3 Asynchronous Inference for Robot Policies ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Ho et al. (2020)J. Ho, A. Jain, and P. Abbeel Denoising diffusion probabilistic models. NeurIPS 33, pp.6840–6851. Cited by: [§2.2](https://arxiv.org/html/2610.06617#S2.SS2.p1.1 "2.2 Few-Step Distillation ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Hu et al. (2021)E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen Lora: low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685. Cited by: [§5.1](https://arxiv.org/html/2610.06617#S5.SS1.p1.1 "5.1 Implementation Details ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Huang et al. (2026)Y. Huang, X. Zhou, J. Zhang, L. Bo, and T. Pang MeanFlowNFT: bringing forward-process rl to average-velocity generators. arXiv preprint arXiv:2607.15273. Cited by: [§4.1](https://arxiv.org/html/2610.06617#S4.SS1.p1.1 "4.1 Teacher-Anchored Consistency Distillation ‣ 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Intelligence et al. (2025)P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al.\pi_{0.5}: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054. Cited by: [Table F.1](https://arxiv.org/html/2610.06617#A6.T1.2.1.5.1 "In F.3 Detailed LIBERO Results ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§1](https://arxiv.org/html/2610.06617#S1.p1.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§2.1](https://arxiv.org/html/2610.06617#S2.SS1.p1.1 "2.1 World Action Models ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table 1](https://arxiv.org/html/2610.06617#S5.T1.6.1.5.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Jiang et al. (2025)Y. Jiang, S. Cheng, Y. Ding, F. Gao, and B. Qi Asyncvla: asynchronous flow matching for vision-language-action models. arXiv preprint arXiv:2511.14148. Cited by: [§2.3](https://arxiv.org/html/2610.06617#S2.SS3.p1.1 "2.3 Asynchronous Inference for Robot Policies ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Kim et al. (2025)M. J. Kim, C. Finn, and P. Liang Fine-tuning vision-language-action models: optimizing speed and success. arXiv preprint arXiv:2502.19645. Cited by: [Table 2](https://arxiv.org/html/2610.06617#S5.T2.2.1.3.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Kim et al. (2026)M. J. Kim, Y. Gao, T. Lin, Y. Lin, Y. Ge, G. Lam, P. Liang, S. Song, M. Liu, C. Finn, et al.Cosmos policy: fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163. Cited by: [§1](https://arxiv.org/html/2610.06617#S1.p1.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§1](https://arxiv.org/html/2610.06617#S1.p2.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§2.1](https://arxiv.org/html/2610.06617#S2.SS1.p1.1 "2.1 World Action Models ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Kim et al. (2024)M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al.Openvla: an open-source vision-language-action model. arXiv preprint arXiv:2406.09246. Cited by: [§1](https://arxiv.org/html/2610.06617#S1.p1.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§2.1](https://arxiv.org/html/2610.06617#S2.SS1.p1.1 "2.1 World Action Models ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Li et al. (2026a)L. Li, Q. Zhang, Y. Luo, S. Yang, R. Wang, F. Han, M. Yu, Z. Gao, N. Xue, X. Zhu, et al.Causal world modeling for robot control. arXiv preprint arXiv:2601.21998. Cited by: [Table F.1](https://arxiv.org/html/2610.06617#A6.T1.2.1.8.1 "In F.3 Detailed LIBERO Results ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§1](https://arxiv.org/html/2610.06617#S1.p1.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§2.1](https://arxiv.org/html/2610.06617#S2.SS1.p1.1 "2.1 World Action Models ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§2.2](https://arxiv.org/html/2610.06617#S2.SS2.p1.1 "2.2 Few-Step Distillation ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§3.1](https://arxiv.org/html/2610.06617#S3.SS1.p1.1 "3.1 Efficient Mixture-of-Transformers (MoT)-based WAMs ‣ 3 Preliminaries ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table 1](https://arxiv.org/html/2610.06617#S5.T1.6.1.8.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Li et al. (2026b)Z. Li, J. Tang, and Z. Liu FlashVLA: streaming action decoding for fast and asynchronous vla inference. arXiv preprint arXiv:2608.27384. Cited by: [§2.3](https://arxiv.org/html/2610.06617#S2.SS3.p1.1 "2.3 Asynchronous Inference for Robot Policies ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Li et al. (2026c)Z. Li, D. Cheng, Y. Wang, S. Wang, X. Xu, L. Weng, J. Wang, and J. Wang Light-wam: efficient world action models with state-fusion action decoding. arXiv preprint arXiv:2606.08242. Cited by: [§F.1](https://arxiv.org/html/2610.06617#A6.SS1.p1.1 "F.1 Result sources and aggregation ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table F.1](https://arxiv.org/html/2610.06617#A6.T1.2.1.14.1 "In F.3 Detailed LIBERO Results ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§2.1](https://arxiv.org/html/2610.06617#S2.SS1.p1.1 "2.1 World Action Models ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§5.1](https://arxiv.org/html/2610.06617#S5.SS1.p1.1 "5.1 Implementation Details ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table 1](https://arxiv.org/html/2610.06617#S5.T1.6.1.14.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   LightX2V Contributors (2025)LightX2V Contributors LightX2V: light video generation inference framework. GitHub. Note: [https://github.com/ModelTC/lightx2v](https://github.com/ModelTC/lightx2v)Cited by: [§5.4](https://arxiv.org/html/2610.06617#S5.SS4.p3.1 "5.4 Ablation Study ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Lipman et al. (2022)Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le Flow matching for generative modeling. arXiv preprint arXiv:2210.02747. Cited by: [§2.2](https://arxiv.org/html/2610.06617#S2.SS2.p1.1 "2.2 Few-Step Distillation ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Liu et al. (2023)B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone LIBERO: benchmarking knowledge transfer for lifelong robot learning. arXiv preprint arXiv:2306.03310. Cited by: [§1](https://arxiv.org/html/2610.06617#S1.p7.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§5.2](https://arxiv.org/html/2610.06617#S5.SS2.p1.1 "5.2 Evaluation Benchmarks ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Liu et al. (2025)S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu Rdt-1b: a diffusion foundation model for bimanual manipulation. In International Conference on Learning Representations, Vol. 2025, pp.29982–30009. Cited by: [§1](https://arxiv.org/html/2610.06617#S1.p1.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§2.1](https://arxiv.org/html/2610.06617#S2.SS1.p1.1 "2.1 World Action Models ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Liu et al. (2022)X. Liu, C. Gong, and Q. Liu Flow straight and fast: learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003. Cited by: [§2.2](https://arxiv.org/html/2610.06617#S2.SS2.p1.1 "2.2 Few-Step Distillation ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Lu et al. (2026)Y. Lu, Z. Liu, X. Fan, Z. Yang, J. Hou, J. Li, K. Ding, and H. Zhao Faster: rethinking real-time flow vlas. arXiv preprint arXiv:2603.19199. Cited by: [§2.3](https://arxiv.org/html/2610.06617#S2.SS3.p1.1 "2.3 Asynchronous Inference for Robot Policies ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Motubrain Team (2026)Motubrain Team World action models in real time: an empirical study of smooth execution via asynchronous deployment. arXiv preprint arXiv:2608.01880. Cited by: [§2.3](https://arxiv.org/html/2610.06617#S2.SS3.p1.1 "2.3 Asynchronous Inference for Robot Policies ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Pertsch et al. (2025)K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, and S. Levine Fast: efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747. Cited by: [Table 2](https://arxiv.org/html/2610.06617#S5.T2.2.1.5.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Salimans and Ho (2022)T. Salimans and J. Ho Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512. Cited by: [§2.2](https://arxiv.org/html/2610.06617#S2.SS2.p1.1 "2.2 Few-Step Distillation ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Sauer et al. (2024)A. Sauer, D. Lorenz, A. Blattmann, and R. Rombach Adversarial diffusion distillation. In European Conference on Computer Vision, pp.87–103. Cited by: [§2.2](https://arxiv.org/html/2610.06617#S2.SS2.p1.1 "2.2 Few-Step Distillation ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Shukor et al. (2025)M. Shukor, D. Aubakirova, F. Capuano, P. Kooijmans, S. Palma, A. Zouitine, M. Aractingi, C. Pascal, M. Russi, A. Marafioti, et al.Smolvla: a vision-language-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844. Cited by: [§1](https://arxiv.org/html/2610.06617#S1.p1.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§2.1](https://arxiv.org/html/2610.06617#S2.SS1.p1.1 "2.1 World Action Models ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Sohl-Dickstein et al. (2015)J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, pp.2256–2265. Cited by: [§2.2](https://arxiv.org/html/2610.06617#S2.SS2.p1.1 "2.2 Few-Step Distillation ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Song et al. (2023)Y. Song, P. Dhariwal, M. Chen, and I. Sutskever Consistency models. In Proceedings of the 40th International Conference on Machine Learning, pp.32211–32252. Cited by: [Table F.1](https://arxiv.org/html/2610.06617#A6.T1.2.1.18.1 "In F.3 Detailed LIBERO Results ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§2.2](https://arxiv.org/html/2610.06617#S2.SS2.p1.1 "2.2 Few-Step Distillation ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§3.2](https://arxiv.org/html/2610.06617#S3.SS2.p1.1 "3.2 Consistency Distillation for Action Generation ‣ 3 Preliminaries ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§4.1](https://arxiv.org/html/2610.06617#S4.SS1.p1.1 "4.1 Teacher-Anchored Consistency Distillation ‣ 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§4.1](https://arxiv.org/html/2610.06617#S4.SS1.p2.1 "4.1 Teacher-Anchored Consistency Distillation ‣ 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§5.1](https://arxiv.org/html/2610.06617#S5.SS1.p1.1 "5.1 Implementation Details ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table 1](https://arxiv.org/html/2610.06617#S5.T1.6.1.18.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Song et al. (2020)Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456. Cited by: [§2.2](https://arxiv.org/html/2610.06617#S2.SS2.p1.1 "2.2 Few-Step Distillation ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Steele (2004)J. M. Steele The cauchy-schwarz master class: an introduction to the art of mathematical inequalities. Cambridge University Press. Cited by: [§C.1](https://arxiv.org/html/2610.06617#A3.SS1.p2.4 "C.1 Local Consistency and Endpoint Error ‣ Appendix C Complete Error Derivation ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§C.3](https://arxiv.org/html/2610.06617#A3.SS3.p1.3 "C.3 Total Endpoint Error Bound ‣ Appendix C Complete Error Derivation ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Tang et al. (2025)J. Tang, Y. Sun, Y. Zhao, S. Yang, Y. Lin, Z. Zhang, J. Hou, Y. Lu, Z. Liu, and S. Han Vlash: real-time vlas via future-state-aware asynchronous inference. arXiv preprint arXiv:2512.01031. Cited by: [§2.3](https://arxiv.org/html/2610.06617#S2.SS3.p1.1 "2.3 Asynchronous Inference for Robot Policies ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Wang et al. (2026a)H. Wang, G. Zhang, Y. Yan, Y. Shang, R. R. Kompella, and G. Liu Real-time robot execution with masked action chunking. arXiv preprint arXiv:2601.20130. Cited by: [§2.3](https://arxiv.org/html/2610.06617#S2.SS3.p1.1 "2.3 Asynchronous Inference for Robot Policies ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Wang et al. (2026b)Y. Wang, P. Ding, L. Li, C. Cui, Z. Ge, X. Tong, W. Song, H. Zhao, W. Zhao, P. Hou, et al.Vla-adapter: an effective paradigm for tiny-scale vision-language-action model. In Proceedings of the AAAI conference on artificial intelligence, Vol. 40, pp.18638–18646. Cited by: [§2.1](https://arxiv.org/html/2610.06617#S2.SS1.p1.1 "2.1 World Action Models ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Yang et al. (2026)L. Yang, Z. Jiang, C. Sheng, and Z. Tang WAM-opd: on-policy distillation for world action models. arXiv preprint arXiv:2608.22364. Cited by: [§2.2](https://arxiv.org/html/2610.06617#S2.SS2.p1.1 "2.2 Few-Step Distillation ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Ye et al. (2026)S. Ye, Y. Ge, K. Zheng, S. Gao, S. Yu, G. Kurian, S. Indupuru, Y. L. Tan, C. Zhu, J. Xiang, et al.World action models are zero-shot policies. arXiv preprint arXiv:2602.15922. Cited by: [Appendix A](https://arxiv.org/html/2610.06617#A1.p1.1 "Appendix A Limitations ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§1](https://arxiv.org/html/2610.06617#S1.p1.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§1](https://arxiv.org/html/2610.06617#S1.p2.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§2.1](https://arxiv.org/html/2610.06617#S2.SS1.p1.1 "2.1 World Action Models ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Yin et al. (2024)T. Yin, M. Gharbi, T. Park, R. Zhang, E. Shechtman, F. Durand, and W. T. Freeman Improved distribution matching distillation for fast image synthesis. arXiv preprint arXiv:2405.14867. Cited by: [Table F.1](https://arxiv.org/html/2610.06617#A6.T1.2.1.19.1 "In F.3 Detailed LIBERO Results ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§2.2](https://arxiv.org/html/2610.06617#S2.SS2.p1.1 "2.2 Few-Step Distillation ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§4.1](https://arxiv.org/html/2610.06617#S4.SS1.p1.1 "4.1 Teacher-Anchored Consistency Distillation ‣ 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§5.1](https://arxiv.org/html/2610.06617#S5.SS1.p1.1 "5.1 Implementation Details ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table 1](https://arxiv.org/html/2610.06617#S5.T1.6.1.19.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Yuan et al. (2026)T. Yuan, Z. Dong, Y. Liu, and H. Zhao Fast-wam: do world action models need test-time future imagination?. arXiv preprint arXiv:2603.16666. Cited by: [Appendix A](https://arxiv.org/html/2610.06617#A1.p1.1 "Appendix A Limitations ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Appendix B](https://arxiv.org/html/2610.06617#A2.p1.1 "Appendix B Cross-Expert Wavefront Pipelining: Pseudocode ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table F.1](https://arxiv.org/html/2610.06617#A6.T1.2.1.16.1 "In F.3 Detailed LIBERO Results ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table F.1](https://arxiv.org/html/2610.06617#A6.T1.2.1.9.1 "In F.3 Detailed LIBERO Results ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§1](https://arxiv.org/html/2610.06617#S1.p1.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§1](https://arxiv.org/html/2610.06617#S1.p2.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§1](https://arxiv.org/html/2610.06617#S1.p3.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§1](https://arxiv.org/html/2610.06617#S1.p7.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§2.1](https://arxiv.org/html/2610.06617#S2.SS1.p1.1 "2.1 World Action Models ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§3.1](https://arxiv.org/html/2610.06617#S3.SS1.p1.1 "3.1 Efficient Mixture-of-Transformers (MoT)-based WAMs ‣ 3 Preliminaries ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§4.2](https://arxiv.org/html/2610.06617#S4.SS2.p2.5 "4.2 Cross-Expert Wavefront Pipelining ‣ 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§5.1](https://arxiv.org/html/2610.06617#S5.SS1.p1.1 "5.1 Implementation Details ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§5.2](https://arxiv.org/html/2610.06617#S5.SS2.p1.1 "5.2 Evaluation Benchmarks ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table 1](https://arxiv.org/html/2610.06617#S5.T1.6.1.16.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table 1](https://arxiv.org/html/2610.06617#S5.T1.6.1.9.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table 2](https://arxiv.org/html/2610.06617#S5.T2.2.1.7.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Zhao et al. (2026)W. Zhao, H. Jiang, X. Shi, L. Liu, F. Huang, Z. Su, W. Sui, and X. Wang Faster-wam: efficient inference-time future conditioning for robust world action models. arXiv preprint arXiv:2608.04404. Cited by: [Appendix A](https://arxiv.org/html/2610.06617#A1.p1.1 "Appendix A Limitations ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Appendix B](https://arxiv.org/html/2610.06617#A2.p1.1 "Appendix B Cross-Expert Wavefront Pipelining: Pseudocode ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§F.1](https://arxiv.org/html/2610.06617#A6.SS1.p1.1 "F.1 Result sources and aggregation ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table F.1](https://arxiv.org/html/2610.06617#A6.T1.2.1.10.1 "In F.3 Detailed LIBERO Results ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§1](https://arxiv.org/html/2610.06617#S1.p1.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§1](https://arxiv.org/html/2610.06617#S1.p2.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§1](https://arxiv.org/html/2610.06617#S1.p7.1 "1 Introduction ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§2.1](https://arxiv.org/html/2610.06617#S2.SS1.p1.1 "2.1 World Action Models ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§3.1](https://arxiv.org/html/2610.06617#S3.SS1.p1.1 "3.1 Efficient Mixture-of-Transformers (MoT)-based WAMs ‣ 3 Preliminaries ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§4.2](https://arxiv.org/html/2610.06617#S4.SS2.p2.5 "4.2 Cross-Expert Wavefront Pipelining ‣ 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§5.1](https://arxiv.org/html/2610.06617#S5.SS1.p1.1 "5.1 Implementation Details ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table 1](https://arxiv.org/html/2610.06617#S5.T1.6.1.10.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table 2](https://arxiv.org/html/2610.06617#S5.T2.2.1.8.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 
*   Zheng et al. (2026)J. Zheng, J. Li, Z. Wang, D. Liu, X. Kang, Y. Feng, Y. Zheng, J. Zou, Y. Chen, J. Zeng, et al.X-vla: soft-prompted transformer as scalable cross-embodiment vision-language-action model. In International Conference on Learning Representations, Vol. 2026, pp.60580–60606. Cited by: [Table F.1](https://arxiv.org/html/2610.06617#A6.T1.2.1.6.1 "In F.3 Detailed LIBERO Results ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [§2.1](https://arxiv.org/html/2610.06617#S2.SS1.p1.1 "2.1 World Action Models ‣ 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), [Table 1](https://arxiv.org/html/2610.06617#S5.T1.6.1.6.1 "In 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). 

###### Appendix

1.   [1 Introduction](https://arxiv.org/html/2610.06617#S1 "In RealtimeWAM: One-Step Asynchronous World Action Models")
2.   [2 Related Work](https://arxiv.org/html/2610.06617#S2 "In RealtimeWAM: One-Step Asynchronous World Action Models")
    1.   [2.1 World Action Models](https://arxiv.org/html/2610.06617#S2.SS1 "In 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models")
    2.   [2.2 Few-Step Distillation](https://arxiv.org/html/2610.06617#S2.SS2 "In 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models")
    3.   [2.3 Asynchronous Inference for Robot Policies](https://arxiv.org/html/2610.06617#S2.SS3 "In 2 Related Work ‣ RealtimeWAM: One-Step Asynchronous World Action Models")

3.   [3 Preliminaries](https://arxiv.org/html/2610.06617#S3 "In RealtimeWAM: One-Step Asynchronous World Action Models")
    1.   [3.1 Efficient Mixture-of-Transformers (MoT)-based WAMs](https://arxiv.org/html/2610.06617#S3.SS1 "In 3 Preliminaries ‣ RealtimeWAM: One-Step Asynchronous World Action Models")
    2.   [3.2 Consistency Distillation for Action Generation](https://arxiv.org/html/2610.06617#S3.SS2 "In 3 Preliminaries ‣ RealtimeWAM: One-Step Asynchronous World Action Models")

4.   [4 RealtimeWAM](https://arxiv.org/html/2610.06617#S4 "In RealtimeWAM: One-Step Asynchronous World Action Models")
    1.   [4.1 Teacher-Anchored Consistency Distillation](https://arxiv.org/html/2610.06617#S4.SS1 "In 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models")
    2.   [4.2 Cross-Expert Wavefront Pipelining](https://arxiv.org/html/2610.06617#S4.SS2 "In 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models")

5.   [5 Experiments](https://arxiv.org/html/2610.06617#S5 "In RealtimeWAM: One-Step Asynchronous World Action Models")
    1.   [5.1 Implementation Details](https://arxiv.org/html/2610.06617#S5.SS1 "In 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models")
    2.   [5.2 Evaluation Benchmarks](https://arxiv.org/html/2610.06617#S5.SS2 "In 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models")
    3.   [5.3 Main Results](https://arxiv.org/html/2610.06617#S5.SS3 "In 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models")
    4.   [5.4 Ablation Study](https://arxiv.org/html/2610.06617#S5.SS4 "In 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models")

6.   [6 Conclusion](https://arxiv.org/html/2610.06617#S6 "In RealtimeWAM: One-Step Asynchronous World Action Models")
7.   [References](https://arxiv.org/html/2610.06617#bib "In RealtimeWAM: One-Step Asynchronous World Action Models")
8.   [A Limitations](https://arxiv.org/html/2610.06617#A1 "In RealtimeWAM: One-Step Asynchronous World Action Models")
9.   [B Cross-Expert Wavefront Pipelining: Pseudocode](https://arxiv.org/html/2610.06617#A2 "In RealtimeWAM: One-Step Asynchronous World Action Models")
10.   [C Complete Error Derivation](https://arxiv.org/html/2610.06617#A3 "In RealtimeWAM: One-Step Asynchronous World Action Models")
    1.   [C.1 Local Consistency and Endpoint Error](https://arxiv.org/html/2610.06617#A3.SS1 "In Appendix C Complete Error Derivation ‣ RealtimeWAM: One-Step Asynchronous World Action Models")
    2.   [C.2 Equivalence to Weighted Endpoint Supervision](https://arxiv.org/html/2610.06617#A3.SS2 "In Appendix C Complete Error Derivation ‣ RealtimeWAM: One-Step Asynchronous World Action Models")
    3.   [C.3 Total Endpoint Error Bound](https://arxiv.org/html/2610.06617#A3.SS3 "In Appendix C Complete Error Derivation ‣ RealtimeWAM: One-Step Asynchronous World Action Models")

11.   [D Inference Equivalence of Asynchronous Execution](https://arxiv.org/html/2610.06617#A4 "In RealtimeWAM: One-Step Asynchronous World Action Models")
12.   [E Inference Latency Analysis](https://arxiv.org/html/2610.06617#A5 "In RealtimeWAM: One-Step Asynchronous World Action Models")
13.   [F More Experimental Details and Results](https://arxiv.org/html/2610.06617#A6 "In RealtimeWAM: One-Step Asynchronous World Action Models")
    1.   [F.1 Result sources and aggregation](https://arxiv.org/html/2610.06617#A6.SS1 "In Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models")
    2.   [F.2 Efficient Kernel Implementation](https://arxiv.org/html/2610.06617#A6.SS2 "In Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models")
    3.   [F.3 Detailed LIBERO Results](https://arxiv.org/html/2610.06617#A6.SS3 "In Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models")
        1.   [F.4 Additional Ablation Study](https://arxiv.org/html/2610.06617#A6.SS4 "In F.3 Detailed LIBERO Results ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models")

    4.   [F.5 Latency on Additional GPUs](https://arxiv.org/html/2610.06617#A6.SS5 "In Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models")
    5.   [F.6 Real-World Evaluation of Long-Horizon Garment Folding](https://arxiv.org/html/2610.06617#A6.SS6 "In Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models")

## Appendix A Limitations

1) Applicability. RealtimeWAM targets MoT-based WAMs and is not directly applicable to shared-backbone architectures such as DreamZero([Ye et al., 2026](https://arxiv.org/html/2610.06617#bib.bib5)). Step distillation in shared-backbone WAMs more closely resembles step distillation in video generation, making action-specific characteristics harder to isolate and motivating our focus on MoT architectures. Within this scope, our framework supports both models that omit explicit future-video prediction at inference (_e.g._, Fast-WAM([Yuan et al., 2026](https://arxiv.org/html/2610.06617#bib.bib4))) and those that retain it (_e.g._, Faster-WAM([Zhao et al., 2026](https://arxiv.org/html/2610.06617#bib.bib15))).

2) Training overhead. The teacher-anchored loss \mathcal{L}_{\mathrm{TA}} introduces additional training overhead by requiring multi-step rollouts from the frozen teacher. Nevertheless, this overhead remains manageable: as shown in Tab.[A.1](https://arxiv.org/html/2610.06617#A1.T1 "Table A.1 ‣ Appendix A Limitations ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), adding \mathcal{L}_{\mathrm{TA}} increases training time from 5h 21min to 8h 3min (approximately 50%), comparable to jointly fine-tuning the Video and Action Experts (8h 1min). By freezing the Video Expert, TACD reduces peak GPU memory from 65.98 to 24.73 GiB compared with joint fine-tuning, a 62.5% reduction, while retaining the same memory footprint as action-only consistency distillation.

Table A.1: Training time and peak GPU memory for 30,000 training iterations on 16 NVIDIA H100 GPUs. \mathcal{L}_{\mathrm{CD}} and \mathcal{L}_{\mathrm{TA}} denote the consistency and teacher-anchored losses, respectively. Video/Action indicate whether each expert is fine-tuned.

\mathcal{L}_{\mathrm{CD}}\mathcal{L}_{\mathrm{TA}}Video Action Training Time\downarrow Peak Memory (GiB)\downarrow
✓✗✓✓8h 1min 65.98
✓✗✗✓5h 21min 24.73
✓✓✗✓8h 3min 24.73

## Appendix B Cross-Expert Wavefront Pipelining: Pseudocode

Our Cross-Expert Wavefront Pipelining (CEWP) supports MoT-based WAMs, including both models that omit explicit future-video prediction at inference (_e.g._, Fast-WAM([Yuan et al., 2026](https://arxiv.org/html/2610.06617#bib.bib4))) and those that retain it (_e.g._, Faster-WAM([Zhao et al., 2026](https://arxiv.org/html/2610.06617#bib.bib15))). Algorithm[1](https://arxiv.org/html/2610.06617#algorithm1 "Algorithm 1 ‣ Appendix B Cross-Expert Wavefront Pipelining: Pseudocode ‣ RealtimeWAM: One-Step Asynchronous World Action Models") presents a unified schedule for one paired forward pass. Input preparation, attention masks, output heads, and denoising schedules follow the base model.

Let \mathcal{C} denote the video-conditioned action blocks, indexed from 1. Fast-WAM uses \mathcal{C}=\{1,\ldots,N\}, whereas our Faster-WAM implementation uses \mathcal{C}=\{1,5,\ldots,N\}. At each conditioning block, Fuse returns the single-block KV for Fast-WAM or the base model’s weighted fusion of the accumulated interval KV for Faster-WAM. The video stream then records a readiness event, which the action stream waits on only before attention. Intervening action blocks require no cross-expert synchronization.

Algorithm 1 Cross-Expert Wavefront Pipelining for MoT-based WAMs.

def cewp(hv,ha,N,C,retain_video=False):

caller=current_stream()

Sv,Sa=Stream(),Stream()

ready={i:Event()for i in C}

Sv.wait_stream(caller)

Sa.wait_stream(caller)

pending,cache=[],{}

for i in range(1,N+1):

with stream(Sv):

qv,kv,vv=P_v[i](hv)

pending.append((kv,vv))

if i in C:

cache[i]=Fuse[i](pending)

pending=[]

ready[i].record(Sv)

hv=R_v[i](hv,Attn_v[i](qv,kv,vv,Mv[i]))

with stream(Sa):

qa,ka,va=P_a[i](ha)

mask=None

if i in C:

Sa.wait_event(ready[i])

kc,vc=cache[i]

kc.record_stream(Sa)

vc.record_stream(Sa)

ka,va=cat(kc,ka),cat(vc,va)

mask=Ma[i]

ha=R_a[i](ha,Attn_a[i](qa,ka,va,mask))

caller.wait_stream(Sv)

caller.wait_stream(Sa)

ha.record_stream(caller)

hv.record_stream(caller)

action_pred=ActionHead(ha)

video_pred=VideoHead(hv)if retain_video else None

return action_pred,video_pred

## Appendix C Complete Error Derivation

### C.1 Local Consistency and Endpoint Error

We fix the model parameters and omit the shared conditioning z(o,l). For 0<t\leq 1, let a_{\tau}^{\star} be the unique teacher ODE solution with a_{t}^{\star}=a_{t}, and let a_{0}^{\star} denote its exact endpoint. All expectations below use the same training sampling distribution, and we assume finite second moments for the errors and velocity residuals.

Adding and subtracting the EMA prediction makes the two error terms explicit:

\displaystyle\mathbf{e}_{\mathrm{total}}\displaystyle=f_{\theta_{\mathrm{S}}}(a_{t},t)-a_{0}^{\star}
\displaystyle=\underbrace{f_{\theta_{\mathrm{S}}}(a_{t},t)-f_{\theta_{\mathrm{ema}}}(\tilde{a}_{s},s)}_{\mathbf{e}_{\mathrm{local}}}+\underbrace{f_{\theta_{\mathrm{ema}}}(\tilde{a}_{s},s)-a_{0}^{\star}}_{\mathbf{e}_{\mathrm{global}}}.

Expanding the squared norm and taking expectations gives

\mathbb{E}\|\mathbf{e}_{\mathrm{total}}\|_{2}^{2}=\mathbb{E}\|\mathbf{e}_{\mathrm{local}}\|_{2}^{2}+\mathbb{E}\|\mathbf{e}_{\mathrm{global}}\|_{2}^{2}+2\mathbb{E}\langle\mathbf{e}_{\mathrm{local}},\mathbf{e}_{\mathrm{global}}\rangle.

Cauchy–Schwarz([Steele, 2004](https://arxiv.org/html/2610.06617#bib.bib44)) bounds the cross term as

\displaystyle\left|\mathbb{E}\langle\mathbf{e}_{\mathrm{local}},\mathbf{e}_{\mathrm{global}}\rangle\right|\displaystyle\leq\mathbb{E}\!\left[\|\mathbf{e}_{\mathrm{local}}\|_{2}\|\mathbf{e}_{\mathrm{global}}\|_{2}\right]
\displaystyle\leq\sqrt{\mathbb{E}\|\mathbf{e}_{\mathrm{local}}\|_{2}^{2}}\sqrt{\mathbb{E}\|\mathbf{e}_{\mathrm{global}}\|_{2}^{2}}.

Substitution yields

\displaystyle\mathbb{E}\|\mathbf{e}_{\mathrm{total}}\|_{2}^{2}\displaystyle\leq\left(\sqrt{\mathbb{E}\|\mathbf{e}_{\mathrm{local}}\|_{2}^{2}}+\sqrt{\mathbb{E}\|\mathbf{e}_{\mathrm{global}}\|_{2}^{2}}\right)^{2},(C.1)
\displaystyle\mathbb{E}\|\mathbf{e}_{\mathrm{total}}\|_{2}^{2}\displaystyle\geq\left(\sqrt{\mathbb{E}\|\mathbf{e}_{\mathrm{local}}\|_{2}^{2}}-\sqrt{\mathbb{E}\|\mathbf{e}_{\mathrm{global}}\|_{2}^{2}}\right)^{2}.

Thus, a small expected local error alone does not ensure a small expected total error without control of the global target error. In particular, \mathbb{E}\|\mathbf{e}_{\mathrm{local}}\|_{2}^{2}=0 implies \mathbb{E}\|\mathbf{e}_{\mathrm{total}}\|_{2}^{2}=\mathbb{E}\|\mathbf{e}_{\mathrm{global}}\|_{2}^{2}: agreement with the EMA target does not by itself establish that the target is accurate.

### C.2 Equivalence to Weighted Endpoint Supervision

The student and teacher use the same noisy action a_{t} and conditioning z(o,l). By Eqs.([3](https://arxiv.org/html/2610.06617#S3.E3 "In 3.2 Consistency Distillation for Action Generation ‣ 3 Preliminaries ‣ RealtimeWAM: One-Step Asynchronous World Action Models")) and([7](https://arxiv.org/html/2610.06617#S4.E7 "In 4.1 Teacher-Anchored Consistency Distillation ‣ 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models")),

f_{\theta_{\mathrm{S}}}(a_{t},t)=a_{t}-t\,v_{\theta_{\mathrm{S}}}(a_{t},t),\qquad a_{0}^{\mathrm{T}}=a_{t}-t\,u_{\theta_{\mathrm{T}}}(a_{t},t).

Subtracting the two endpoints gives

f_{\theta_{\mathrm{S}}}(a_{t},t)-a_{0}^{\mathrm{T}}=-t\!\left[v_{\theta_{\mathrm{S}}}(a_{t},t)-u_{\theta_{\mathrm{T}}}(a_{t},t)\right].(C.2)

Since the teacher target is frozen, stop-gradient does not change its value. Taking squared norms, dividing by t^{2}, and averaging gives

\displaystyle\mathcal{L}_{\mathrm{TA}}\displaystyle=\mathbb{E}_{a_{t},t}\!\left[\left\|v_{\theta_{\mathrm{S}}}(a_{t},t)-u_{\theta_{\mathrm{T}}}(a_{t},t)\right\|_{2}^{2}\right](C.3)
\displaystyle=\mathbb{E}_{a_{t},t}\!\left[\left\|\frac{a_{t}-f_{\theta_{\mathrm{S}}}(a_{t},t)}{t}-\frac{a_{t}-a_{0}^{\mathrm{T}}}{t}\right\|_{2}^{2}\right]
\displaystyle=\mathbb{E}_{a_{t},t}\!\left[\frac{\left\|f_{\theta_{\mathrm{S}}}(a_{t},t)-a_{0}^{\mathrm{T}}\right\|_{2}^{2}}{t^{2}}\right],\qquad t>0.

Thus, the velocity loss is exactly a weighted endpoint loss, with 1/t^{2} inside the expectation. The equivalence holds for any teacher solver through the definition of u_{\theta_{\mathrm{T}}}.

### C.3 Total Endpoint Error Bound

Let \bm{\delta}=a_{0}^{\mathrm{T}}-a_{0}^{\star}. Adding this numerical integration error to Eq.([C.2](https://arxiv.org/html/2610.06617#A3.E2 "In C.2 Equivalence to Weighted Endpoint Supervision ‣ Appendix C Complete Error Derivation ‣ RealtimeWAM: One-Step Asynchronous World Action Models")) gives

\displaystyle\mathbf{e}_{\mathrm{total}}\displaystyle=\left(f_{\theta_{\mathrm{S}}}(a_{t},t)-a_{0}^{\mathrm{T}}\right)+\bm{\delta}
\displaystyle=-t\!\left[v_{\theta_{\mathrm{S}}}(a_{t},t)-u_{\theta_{\mathrm{T}}}(a_{t},t)\right]+\bm{\delta},

which recovers Eq.([9](https://arxiv.org/html/2610.06617#S4.E9 "In 4.1 Teacher-Anchored Consistency Distillation ‣ 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models")). Applying a consequence of Cauchy–Schwarz([Steele, 2004](https://arxiv.org/html/2610.06617#bib.bib44))2 2 2 For real vectors p,q, Cauchy–Schwarz gives |\langle p,q\rangle|\leq\|p\|_{2}\|q\|_{2}. Together with 2ab\leq a^{2}+b^{2}, this yields \|p+q\|_{2}^{2}=\|p\|_{2}^{2}+\|q\|_{2}^{2}+2\langle p,q\rangle\leq 2\|p\|_{2}^{2}+2\|q\|_{2}^{2}., taking expectations, and using 0<t\leq 1 yields

\displaystyle\mathbb{E}\|\mathbf{e}_{\mathrm{total}}\|_{2}^{2}\displaystyle\leq 2\mathbb{E}\!\left[t^{2}\left\|v_{\theta_{\mathrm{S}}}(a_{t},t)-u_{\theta_{\mathrm{T}}}(a_{t},t)\right\|_{2}^{2}\right]+2\mathbb{E}\|\bm{\delta}\|_{2}^{2}(C.4)
\displaystyle\leq 2\mathcal{L}_{\mathrm{TA}}+2\mathbb{E}\|\bm{\delta}\|_{2}^{2}.

This bounds the combined error \mathbf{e}_{\mathrm{local}}+\mathbf{e}_{\mathrm{global}} and does not require the two components to decrease separately.

#### Teacher numerical integration error.

Following the standard error analysis of the explicit Euler method([Hairer et al., 1993](https://arxiv.org/html/2610.06617#bib.bib50)), we estimate the teacher’s numerical integration error for a uniform K-step rollout. Let h=t/K, \tau_{k}=t-kh, and \hat{a}_{\tau_{k}} denote the Euler iterates initialized at \hat{a}_{\tau_{0}}=a_{t}. Assume the teacher velocity is L-Lipschitz in state and \|d^{2}a_{\tau}^{\star}/d\tau^{2}\|_{2}\leq M. For D_{k}=\|\hat{a}_{\tau_{k}}-a_{\tau_{k}}^{\star}\|_{2}, Taylor expansion gives D_{k+1}\leq(1+Lh)D_{k}+Mh^{2}/2, with D_{0}=0. Summing this recurrence gives, for L>0,

\|\bm{\delta}\|_{2}=D_{K}\leq\frac{Mh}{2L}\bigl[(1+Lh)^{K}-1\bigr]\leq\frac{Me^{Lt}t^{2}}{2K}.

For L=0, the bound is Mt^{2}/(2K). With uniform L,M, the numerical contribution to Eq.([C.4](https://arxiv.org/html/2610.06617#A3.E4 "In C.3 Total Endpoint Error Bound ‣ Appendix C Complete Error Derivation ‣ RealtimeWAM: One-Step Asynchronous World Action Models")) is O(K^{-2}). Increasing the number of teacher rollout steps K therefore tightens the error bound, providing explicit control over the teacher’s numerical integration error.

## Appendix D Inference Equivalence of Asynchronous Execution

Cross-Expert Wavefront Pipelining (CEWP) changes only the execution schedule, using the same parameters, inputs, attention masks, denoising timestep, and operators as sequential inference. We fix the initial action noise and any future-video noise so that both executions start from identical states (h_{0}^{v},h_{0}^{a}). In our implementation, video KV cache entries are published after their required preprocessing, including key normalization and RoPE. The action stream waits on the corresponding event before concatenating video and action KV.

Suppose both executions receive identical states at block i. Equation([10](https://arxiv.org/html/2610.06617#S4.E10 "In 4.2 Cross-Expert Wavefront Pipelining ‣ 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models")) then gives identical Q/K/V tensors. The event dependency in Eq.([13](https://arxiv.org/html/2610.06617#S4.E13 "In 4.2 Cross-Expert Wavefront Pipelining ‣ 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models")) ensures that action attention consumes the same completed video KV cache. Applying the same attention and remaining-block operators in Eq.([11](https://arxiv.org/html/2610.06617#S4.E11 "In 4.2 Cross-Expert Wavefront Pipelining ‣ 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models")) therefore produces identical next-block states for both experts. Induction from i=0 gives, in exact arithmetic,

\widehat{h}_{i}^{v}=h_{i}^{v},\qquad\widehat{h}_{i}^{a}=h_{i}^{a},\qquad i\in\{0,\ldots,N\}.(D.1)

where the hat denotes asynchronous execution. The unchanged action output head preserves this equality. CEWP is thus another valid topological ordering of the original computation graph, requiring no architectural modification or additional training, and is independent of TACD.

For sparse conditioning in Algorithm[1](https://arxiv.org/html/2610.06617#algorithm1 "Algorithm 1 ‣ Appendix B Cross-Expert Wavefront Pipelining: Pseudocode ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), the same argument applies: each event is recorded after the required KV fusion, whose inputs and operator remain unchanged, while non-conditioning action blocks require no video KV. When future-video prediction is retained, the unchanged video output head likewise preserves the video prediction.

## Appendix E Inference Latency Analysis

We model the execution time of the two experts, excluding common costs such as VAE encoding. Let T_{v} and T_{a} denote the standalone latencies of one Video Expert pass and one Action Expert pass, respectively, and S the number of action denoising steps. Since the video KV cache is computed once and reused across steps, sequential execution requires

T_{\mathrm{seq}}(S)=T_{v}+ST_{a}.(E.1)

After TACD, the Action Expert requires only one denoising step (S=1), reducing the sequential latency to

T_{\mathrm{seq}}(1)=T_{v}+T_{a}.(E.2)

CEWP further overlaps the two remaining expert passes through the block-wise schedule in Eq.([13](https://arxiv.org/html/2610.06617#S4.E13 "In 4.2 Cross-Expert Wavefront Pipelining ‣ 4 RealtimeWAM ‣ RealtimeWAM: One-Step Asynchronous World Action Models")). We model its latency as

T_{\mathrm{pipe}}(1)=T_{v}+T_{a}-T_{\mathrm{overlap}}+\Delta,(E.3)

where T_{\mathrm{overlap}} denotes the execution time saved through overlap, and \Delta accounts for additional synchronization, scheduling, and resource-contention costs. Under ideal overlap, T_{\mathrm{overlap}}\approx\min(T_{v},T_{a}), giving

T_{\mathrm{pipe}}(1)\approx\max(T_{v},T_{a})+\Delta.(E.4)

This is an ideal reference: block dependencies and shared GPU resources limit the overlap attainable in practice. TACD and CEWP thus address complementary sources of latency: TACD reduces the number of action passes, while CEWP overlaps the remaining video and action passes.

## Appendix F More Experimental Details and Results

### F.1 Result sources and aggregation

For results quoted from prior work in Tabs.[1](https://arxiv.org/html/2610.06617#S5.T1 "Table 1 ‣ 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models") and[2](https://arxiv.org/html/2610.06617#S5.T2 "Table 2 ‣ 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), we retain the scores and aggregates reported in Light-WAM([Li et al., 2026c](https://arxiv.org/html/2610.06617#bib.bib16)), Flash-WAM([Akbari et al., 2026](https://arxiv.org/html/2610.06617#bib.bib19)), and Faster-WAM([Zhao et al., 2026](https://arxiv.org/html/2610.06617#bib.bib15)). On LIBERO and RoboTwin 2.0, we evaluate Fast-WAM and Faster-WAM using 10 action denoising steps, our implementations of the few-step distillation baselines (DMD, MeanFlow, and consistency distillation), and the RealtimeWAM variants. Unless otherwise specified, RealtimeWAM variants use the action-only LoRA post-training setting described in Sec.[5.1](https://arxiv.org/html/2610.06617#S5.SS1 "5.1 Implementation Details ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), with the Video Expert kept frozen to maintain the teacher conditioning during distillation. For our evaluations, Overall is the arithmetic mean of the four suite-level success rates on LIBERO and of the Clean and Random success rates on RoboTwin 2.0. On LIBERO-Plus, Overall is the average success rate across perturbation categories, weighted by the number of trials in each category.

### F.2 Efficient Kernel Implementation

We apply five operator-level optimizations to reduce kernel launches and intermediate memory traffic. (1) We fuse modulation and residual gating into separate element-wise kernels, preserving BF16 intermediate rounding while keeping LayerNorm independent. (2) We fuse each Q/K RMSNorm into a single Triton kernel, retaining FP32 statistics and BF16 intermediate rounding. (3) We fuse RoPE type conversions and complex rotation into a single Triton kernel while retaining FP64 arithmetic. (4) We replace static all-True attention masks with None, preserving masks that exclude attention positions. (5) We pack self-attention Q/K/V and cross-attention K/V projections into single GEMMs, leaving cross-attention Q separate. These optimizations complement CUDA Graph, which reduces CPU submission overhead but does not automatically fuse operators within the graph.

### F.3 Detailed LIBERO Results

Table F.1: Detailed success rates (%) on the four LIBERO task suites.

Method NFE LIBERO
Video Action Spatial\uparrow Object\uparrow Goal\uparrow LIBERO-10\uparrow Overall\uparrow
WAM and VLA Baselines
\pi_{0}([Black et al., 2024](https://arxiv.org/html/2610.06617#bib.bib10))––96.8 98.8 95.8 85.2 94.1
\pi_{0.5}([Intelligence et al., 2025](https://arxiv.org/html/2610.06617#bib.bib7))––98.8 98.2 98.0 92.4 96.9
X-VLA([Zheng et al., 2026](https://arxiv.org/html/2610.06617#bib.bib18))––98.2 98.6 97.8 97.6 98.1
Motus([Bi et al., 2026](https://arxiv.org/html/2610.06617#bib.bib12))10 10 96.8 99.8 96.6 97.6 97.7
LingBot-VA([Li et al., 2026a](https://arxiv.org/html/2610.06617#bib.bib13))20 50 98.5 99.6 97.2 98.5 98.5
Fast-WAM([Yuan et al., 2026](https://arxiv.org/html/2610.06617#bib.bib4))1 10 97.0 99.4 96.6 94.8 97.0
Faster-WAM([Zhao et al., 2026](https://arxiv.org/html/2610.06617#bib.bib15))1 10 99.2 99.6 98.4 98.4 98.9
Other Efficient Variants
Flash-WAM([Akbari et al., 2026](https://arxiv.org/html/2610.06617#bib.bib19))1 2 97.0 92.8 96.4 98.0 96.1
Flash-WAM([Akbari et al., 2026](https://arxiv.org/html/2610.06617#bib.bib19))1 1 96.0 92.6 96.0 95.8 95.1
Light-WAM([Li et al., 2026c](https://arxiv.org/html/2610.06617#bib.bib16))1 1 98.2 99.6 97.8 93.0 97.2
Few-Step Distillation
Fast-WAM([Yuan et al., 2026](https://arxiv.org/html/2610.06617#bib.bib4))1 1 94.8 96.8 96.6 91.8 95.0
Mean Flow*([Geng et al., 2025](https://arxiv.org/html/2610.06617#bib.bib20))1 1 97.4 99.6 96.8 93.8 96.9
Consistency Distillation*([Song et al., 2023](https://arxiv.org/html/2610.06617#bib.bib21))1 1 96.4 99.4 96.8 95.2 96.9
DMD*([Yin et al., 2024](https://arxiv.org/html/2610.06617#bib.bib22))1 1 97.0 99.4 95.8 92.2 96.1
RealtimeWAM*1 1 97.0 99.4 97.2 94.2 97.0
RealtimeWAM†1 1 98.8 99.8 99.6 97.6 99.0

Tab.[F.1](https://arxiv.org/html/2610.06617#A6.T1 "Table F.1 ‣ F.3 Detailed LIBERO Results ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models") reports success rates on each LIBERO task suite, complementing the overall results in Tab.[1](https://arxiv.org/html/2610.06617#S5.T1 "Table 1 ‣ 5.3 Main Results ‣ 5 Experiments ‣ RealtimeWAM: One-Step Asynchronous World Action Models"). With one-step action generation, RealtimeWAM* matches the 97.0% overall success rate of 10-step Fast-WAM, while RealtimeWAM† achieves 99.0%, comparable to 10-step Faster-WAM (98.9%). Compared with Faster-WAM, RealtimeWAM† improves Goal success by 1.2 percentage points (99.6% vs. 98.4%), with modest reductions on Spatial and LIBERO-10, indicating that overall performance is preserved despite variations across task suites.

### F.4 Additional Ablation Study

We further investigate the start and end points of the teacher rollout used to construct \mathcal{L}_{\mathrm{TA}}. We compare three intervals: 1\!\to\!0, t\!\to\!r, and t\!\to\!0, where 1 denotes pure noise, 0 denotes the clean endpoint, and t and r<t denote intermediate noise levels. As shown in Tab.[F.4](https://arxiv.org/html/2610.06617#A6.SS4 "F.4 Additional Ablation Study ‣ F.3 Detailed LIBERO Results ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), t\!\to\!0 achieves the highest overall success rate of 90.84%, outperforming both the full-interval rollout 1\!\to\!0 and the intermediate-interval rollout t\!\to\!r. We therefore adopt t\!\to\!0 as the default teacher rollout interval.

Table F.2: Effect of teacher rollout intervals on RoboTwin 2.0. Results are success rates (%); best results are in bold.

Rollout Interval Clean\uparrow Random\uparrow Overall\uparrow
1\!\to\!0 91.26 89.90 90.63
t\!\to\!r 90.96 89.46 90.23
t\!\to\!0 91.96 89.72 90.84

### F.5 Latency on Additional GPUs

Tab.[F.3](https://arxiv.org/html/2610.06617#A6.T3 "Table F.3 ‣ F.5 Latency on Additional GPUs ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models") extends the latency evaluation to RTX 5090 and RTX 4090D. TACD alone yields 3.5–5.0\times speedups, while CUDA Graph and CEWP further reduce latency across all four model–GPU configurations. With all optimizations enabled, RealtimeWAM achieves 14.1\times and 17.7\times speedups on Fast-WAM, and 6.9\times and 8.1\times on Faster-WAM, for RTX 5090 and RTX 4090D, respectively. The current kernels are optimized primarily for NVIDIA H-series GPUs (_e.g._, H100), which may explain their smaller gains on RTX 5090 and RTX 4090D. We expect dedicated kernel tuning for these GPUs to yield larger speedups.

Table F.3: Inference latency (ms) on additional GPUs. Each entry reports latency followed by the cumulative speedup in parentheses, relative to the unoptimized baseline for the same model and GPU.

TACD CUDA Graph CEWP Efficient Kernels Fast-WAM Faster-WAM
RTX 5090 RTX 4090D RTX 5090 RTX 4090D
✗✗✗✗251.5 (1.0\times)480.8 (1.0\times)214.1 (1.0\times)362.4 (1.0\times)
✗✓✗✗97.8 (2.6\times)106.1 (4.5\times)81.5 (2.6\times)96.0 (3.8\times)
✓✗✗✗54.3 (4.6\times)96.8 (5.0\times)60.7 (3.5\times)91.7 (4.0\times)
✓✓✗✗30.8 (8.2\times)35.3 (13.6\times)38.5 (5.6\times)54.9 (6.6\times)
✓✓✓✗24.3 (10.3\times)31.2 (15.4\times)34.7 (6.2\times)51.1 (7.1\times)
✓✓✓✓17.8 (14.1\times)27.2 (17.7\times)30.9 (6.9\times)44.7 (8.1\times)

![Image 3: Refer to caption](https://arxiv.org/html/2610.06617v1/garment_folding_sequence_cropped.png)

Figure F.1: Real-world garment folding with AgileX PiPER. Four keyframes from the recorded rollout show the initial unfolded T-shirt, the configuration after the first fold, a narrow configuration after further lengthwise folding, and the final folded configuration. The rollout was executed with the one-step RealtimeWAM* model. Consecutive actions proceed without perceptible pauses, demonstrating the smooth, low-latency inference that sustains long-horizon deformable-object manipulation.

### F.6 Real-World Evaluation of Long-Horizon Garment Folding

We qualitatively evaluate RealtimeWAM* on a real-world T-shirt-folding task using a bimanual setup comprising two PiPER robotic arms from AgileX Robotics 3 3 3[https://global.agilex.ai/products/piper](https://global.agilex.ai/products/piper), with policy inference running on an NVIDIA RTX 4090 GPU. As shown in Figure[F.1](https://arxiv.org/html/2610.06617#A6.F1 "Figure F.1 ‣ F.5 Latency on Additional GPUs ‣ Appendix F More Experimental Details and Results ‣ RealtimeWAM: One-Step Asynchronous World Action Models"), the task involves successive grasping and folding operations that transform an unfolded garment into a compact configuration. It requires temporally coordinated decisions because each fold changes the garment geometry and the grasp locations available for subsequent actions. Early misalignment can therefore affect later folds, making this task a useful setting for examining long-horizon manipulation of deformable objects.

To meet the real-time requirement of this sequential setting, we deploy the RealtimeWAM* model distilled from Fast-WAM, using TACD for one-step action generation and CEWP for block-wise overlap between the Video and Action Experts. In the physical evaluation, consecutive actions proceeded without perceptible pauses, enabling smooth real-time execution even without execution-level asynchrony between policy inference and robot execution, such as Real-Time Chunking (RTC)([Black et al., 2025a](https://arxiv.org/html/2610.06617#bib.bib23)). The resulting low-latency inference allows the policy to maintain continuous control throughout the long-horizon folding sequence. In the context of RealtimeWAM*, this task motivates preserving action quality under one-step action generation while reducing inference overhead during sequential execution.

