

 Report GitHub Issue
 ×

 Title:

 Content selection saved. Describe the issue below:

 Description:

 Submit without GitHub
 Submit in GitHub

 arXiv is now an independent nonprofit!
 Learn more
 ×

 Back to arXiv

 Why HTML?

 Report Issue

 Back to Abstract

 Download PDF

Abstract

1 Introduction

2 Related work

3 Evoke

3.1 Recurrent Session Formulation

3.2 Long-Horizon Supervision

3.3 Long-Horizon Interactive Teacher

3.4 Geometric World State

3.5 Bounded Three-Step Inference

4 Experiments

4.1 Experimental Setup and Benchmark Performance

4.2 Long-Session Stability and Bounded Runtime

4.3 Effect of Long-Horizon Teacher Supervision

4.4 Geometric Recall and Timed Interaction

5 Conclusion

References

6 Training curves

7 Cost of the geometric path

8 Long-Horizon Generation and Teacher Ablations

9 Qualitative rollouts

10 Public leaderboards in full

Protocol deviations.

11 WBench public leaderboard

 License: arXiv.org perpetual non-exclusive license

arXiv:2608.13546v1 [cs.CV] 13 Aug 2026

1]MoE Key Lab of BIPC, USTC
2]Shanghai Innovation Institute
3]Alaya Lab
\contribution[*]Equal contribution, work done during internship at Alaya Lab
\contribution[†]Project lead
\contribution[‡]Corresponding author
\correspondence\projecthttps://evoke-world.github.io/Evoke/
\codehttps://github.com/SII-YuanyangYin/Evoke

 Alaya-EVOKE: From Linear-Scaling Supervision to Endless World

Yuanyang Yin

Gongxuan Wang

Yifan Zhan

Chuanhao Li

Kaipeng Zhang

Feng Zhao

Affiliation: [

Affiliation: [

Affiliation: [

Email: yyyin@mail.ustc.edu.cn

August 13, 2026

Abstract

Interactive world models must simultaneously support persistent memory, responsive user interaction, and long-horizon generation, yet these requirements place conflicting demands on the underlying model. Maintaining history in the denoiser context or key-value cache incurs growing cost over time, forcing a trade-off between session length and retained memory, while low-latency interaction typically relies on few-step generation whose capabilities are ultimately bounded by its teacher. Evoke addresses both limitations by externalizing persistent world state and redesigning the teacher for long-horizon interactive generation. Scene geometry is maintained in an external, camera-indexed world state bank, from which only information relevant to the current view is retrieved, keeping the denoiser context bounded as the session grows. Rather than treating the teacher as a fixed high-quality generator, the teacher is explicitly designed for long-horizon supervision. Its sparse attention scheme combines chunk-wise grouping, retrieval of selected distant frames, and a linear-attention global state, yielding linear growth in activation memory and computation while enabling supervision over long temporal horizons. Such supervision exposes content drift that remains locally plausible within short windows, while per-chunk conditioning enables prompt changes and event control throughout the sequence. A 30-second long-horizon distribution-matching objective, applied under self-forced rollouts, transfers both capabilities to a three-step student that uses no classifier-free guidance (CFG), improving resistance to long-term content drift while preserving responsive conditioning. With bounded context and recurrent external memory, Evoke supports open-ended, continuously evolving generation; on a single H200 at , each  s chunk is generated in  s. As a three-step world model, Evoke achieves state-of-the-art performance on WBench while remaining competitive in visual quality on VBench-Long and VBench-2.0.

1 Introduction

Figure 1: Two hours of uninterrupted generation. Representative two-hour rollouts under continuous camera control, generated in three steps per chunk without classifier-free guidance. Each row shows six frames sampled uniformly over the full session, illustrating sustained coherence over hour-scale generation.

Interactive world models must jointly support persistent memory, responsive interaction, and long-horizon generation (Bruce et al. 2024; Ball et al. 2025; He et al. 2025b; Sun et al. 2025; Mao et al. 2025). A revisited scene should remain consistent with its earlier appearance; camera motions and text instructions should take effect promptly; and a session should extend from seconds to minutes or even hours. While substantial progress has been made on each capability in isolation, combining all three remains challenging. Their conjunction exposes a systems-level question that model scaling alone does not resolve (Yin et al. 2025b): how to maintain persistent world state without an ever-growing context, and how to equip a low-latency generator with the capabilities required for long-horizon interaction.

Current interactive world models typically place two demanding requirements on the denoiser: preserving persistent world state and generating responses at interactive latency. When past observations are retained as additional context frames or an accumulated key-value cache (Chen et al. 2024; Zhang and Agrawala 2025; Li et al. 2025a; Yang et al. 2025), the cost of each denoising step grows with the session history. Windowing and cache eviction can bound this cost only by discarding information (Cui et al. 2025), while retrieval and streaming conditioning still operate under a finite denoiser-side context budget (Hong et al. 2025; Wang et al. 2026; Feng et al. 2025). Interactive generation introduces a different constraint: low latency requires inference in only a few denoising steps, which is typically achieved by distillation from a slower teacher (Yin et al. 2024b; Yin et al. 2024a; Yin et al. 2025a; Huang et al. 2025a). The resulting student is therefore constrained by the temporal horizon and conditioning capabilities represented in its teacher supervision, limiting both long-term consistency and responsiveness to changes introduced during a session. These limitations suggest that persistent state and interactive capabilities need not both be carried by the deployed denoiser. Evoke decouples them: persistent spatial state is maintained in an external world state bank, while the teacher is explicitly designed to provide the long-horizon and dynamically conditioned supervision required by the few-step student. This separation forms the central design principle of Evoke and enables bounded-cost, long-running interaction.

For persistent world state, Evoke uses an explicit, bounded world state bank indexed by camera pose. Rather than retaining the observed history within the denoiser through context frames, key-value caches, or learned retrieval, the store records scene geometry externally and queries it directly from the current viewpoint. Previously observed surfaces that re-enter the camera frustum are rendered into the current view and provided to the generator as pixel-space conditioning, extending warp-based history conditioning (Wang and He 2026) from short-range frame correspondences to persistent geometric memory. As a result, the denoiser operates with a fixed number of history and rendered memory frames, keeping its token budget independent of session duration. The world state bank is itself bounded and supports explicit read, write, and eviction; the current implementation retains a finite temporal window of geometry, providing constant-cost recall within the retained coverage rather than indefinite memory of every previously visited location.

The world state bank follows a geometry-based construction similar to prior work (Wang and He 2026; Ren et al. 2025; Shen et al. 2026; Wu et al. 2026): generated observations are lifted into scene geometry and later rendered from the current camera pose to provide view-aligned conditioning. This explicit geometric representation keeps persistent state outside the denoiser while preserving direct access to previously observed regions.

A distilled student can acquire only capabilities expressed by its teacher supervision. Yet distillation is commonly treated primarily as an acceleration mechanism (Yin et al. 2024b; Yin et al. 2024a; Lin et al. 2025b; Yin et al. 2025a), leaving key teacher properties largely fixed: attention may remain bidirectional and quadratic, training clips short, and text conditioning constant throughout a sequence. Recent work has examined the mismatch between bidirectional teachers and causal students (Zhu et al. 2026).
Evoke focuses on two complementary teacher-side design variables: supervision horizon and conditioning schedule. Short-horizon supervision provides no joint constraint over distant moments, while a single global prompt does not expose how newly introduced instructions should affect an ongoing rollout. The supervision horizon is particularly important because different failure modes emerge at different temporal scales. Degradation drift, such as exposure growth, saturation shifts, and texture degradation, alters local statistics and can therefore be detected within relatively short windows. Existing long-video objectives can substantially improve such local stability (Cui et al. 2025; Liu et al. 2025), but this improvement may coincide with reduced temporal dynamics or diversity, where suppressing variation also makes short-window statistics easier to preserve.
Content drift presents a different challenge: scene identity, object appearance, or spatial layout may evolve gradually while every local window remains individually plausible. Such inconsistencies become apparent only when sufficiently distant moments are compared and therefore cannot be resolved simply by enforcing stronger local stability. This distinction makes supervision horizon an explicit teacher-side design variable: extending the teacher horizon exposes long-range inconsistencies while retaining the dynamics required for an evolving world. Per-chunk conditioning complements long-horizon supervision by exposing the teacher to prompt changes and newly introduced events within the same rollout. Evoke therefore designs the teacher jointly for long-range consistency and dynamic conditioning, providing both capabilities for subsequent transfer to the few-step student. Whether the benefit of increasing supervision horizon eventually saturates is evaluated empirically in Section 4.2.

These requirements lead to a redesigned Evoke Teacher for long-horizon supervision. The teacher adopts chunk-wise sparse attention (Jia et al. 2025; Cai et al. 2025), where each chunk accesses bounded local context, a small set of distant frames, and a linear-attention global state (Yang et al. 2023; Team et al. 2025; Chen et al. 2025b), reducing attention growth from quadratic to linear with sequence length. The same chunk structure assigns an independent text condition to each chunk, allowing instruction changes to be represented within a single rollout. A 30-second full-window distribution-matching objective (Yin et al. 2024b; Yin et al. 2024a), applied to self-forced rollouts (Huang et al. 2025a), jointly transfers long-range supervision and dynamic conditioning to a three-step, CFG-free (Ho and Salimans 2022) student. At inference time, the resulting student supports scheduled prompt changes during an ongoing session, enabling elements to be introduced or withdrawn on demand, a capability we refer to as evocation (Fig. 2).

Figure 2: Timed prompt switching modifies the sky while preserving anchored scene structure. Using the same input image, camera trajectory, and three-step CFG-free student, rows differ only in text. Rows 1 to 3 introduce a fireball, hot-air balloon, and fireworks, respectively, at chunk and remove them at chunk (amber borders denote active chunks); the objects appear and disappear while the castle, jungle, and dragon remain consistent. Row 4 uses a static castle prompt with no evocation.

Open-ended generation additionally requires the recurrent process itself to remain independent of elapsed time. A rollout represented on a continuously growing positional axis (Su et al. 2024) may eventually exceed the positional range encountered during training (Cui et al. 2025).
Evoke instead reuses the same local positional layout for every generated chunk, while persistent world state is accessed through camera pose rather than temporal position. Session duration therefore increases only the number of recurrent generation steps, without expanding the positional range or denoiser context of any individual call.

The three-step student generates a  s chunk in  s on a single H200 at , with the world state bank enabled. Figure 1 shows seven uninterrupted two-hour rollouts under continuous camera control, demonstrating coherent hour-scale generation. Evoke further achieves state-of-the-art performance on WBench (Ying et al. 2026) while remaining competitive with many-step systems on VBench-2.0 (Zheng et al. 2025) and VBench-Long (Huang et al. 2024; Huang et al. 2025b), despite using only three sampling steps and no classifier-free guidance. Quantitative long-session evaluation is reported separately in Sec. 4.2, together with ablations on supervision horizon and conditioning granularity. These experiments isolate the two teacher-side design variables, measuring how long-horizon supervision affects content drift and how per-chunk conditioning governs mid-session responsiveness. Our contributions can be summarized as follows.

(i) We formulate interactive world generation as a bounded recurrent process that decouples persistent world state from the denoiser: scene geometry is maintained in an external, camera-indexed world state bank, keeping the denoiser context and positional range independent of session duration.

(ii) We redesign the teacher for long-horizon interactive supervision through chunk-wise sparse attention and per-chunk conditioning, enabling efficient supervision over long temporal horizons while exposing both long-range content drift and mid-sequence instruction changes.

(iii) We transfer these capabilities to a three-step, CFG-free student using a -second long-horizon distribution-matching objective under self-forced rollouts, yielding bounded-cost, open-ended generation with strong long-horizon consistency and responsive mid-session control.

2 Related work

Interactive video world models (Bruce et al. 2024; Ball et al. 2025; Valevski et al. 2024; Zhang et al. 2025; NVIDIA 2025; Bar et al. 2025; Decart 2026) differ fundamentally in how they represent and retain information across a running session. One line of work maintains history within the denoiser, using additional context frames or an expanding key-value cache (Chen et al. 2024; Zhang and Agrawala 2025; Li et al. 2025a), and controls the resulting growth in per-step cost through windowing, cache eviction, retrieval, or streaming video-to-video conditioning (Cui et al. 2025; Liu et al. 2025; Hong et al. 2025; Wang et al. 2026; Feng et al. 2025); several systems have demonstrated substantially extended generation horizons (Ball et al. 2025; He et al. 2025b; Sun et al. 2025; Yang et al. 2025; Chen et al. 2025a; Dalal et al. 2025; Li et al. 2025b).
A second line externalizes part of this history through geometry (Wang and He 2026; Ren et al. 2025; Shen et al. 2026; Wu et al. 2026): previously generated observations are lifted into a scene representation (Mildenhall et al. 2021; Kerbl et al. 2023), using monocular depth or feed-forward geometry estimation (Lin et al. 2025a; Wang et al. 2025), and rendered into the target view, allowing revisited content to re-enter the generator as view-aligned pixel conditioning rather than as an ever-growing token history.
A third line targets interactive inference through few-step distillation, building on few-step samplers developed for image diffusion (Salimans and Ho 2022; Luo et al. 2023; Song et al. 2023). Distribution-matching objectives (Yin et al. 2024b; Yin et al. 2024a; Yin et al. 2025a) and self-forcing (Huang et al. 2025a) compress slower teachers into low-step students, alongside adversarial post-training (Lin et al. 2025b), while recent work further examines the mismatch between bidirectional teacher supervision and causal student inference (Zhu et al. 2026).

Evoke builds on the latter two directions while shifting the design focus from individual mechanisms to where persistent state and transferable capabilities reside. Following geometry-based conditioning (Wang and He 2026; Ren et al. 2025), Evoke renders previously observed scene content into the current view; however, the underlying world state is maintained in an explicit external world state bank with bounded read, write, and eviction, keeping the denoiser context independent of session duration.
On the distillation side, Evoke adopts distribution matching (Yin et al. 2024b; Huang et al. 2025a) but treats the teacher itself as a design variable rather than merely a source of high-quality targets. Beyond attention causality (Zhu et al. 2026), the teacher is explicitly structured around two properties required for long-horizon interaction: supervision horizon and conditioning schedule. Long-horizon supervision exposes content inconsistencies that remain locally plausible over short windows, while per-chunk conditioning allows instruction changes to be represented within the same rollout. These properties are then transferred to the few-step student through long-horizon distribution matching.

3 Evoke

Evoke realizes long-horizon interactive generation through two complementary components: a bounded recurrent student with externalized world state, and a teacher designed for long-horizon interactive supervision. At each recurrent step, the three-step student generates one video chunk from a short local history, view-aligned geometry retrieved from an external camera-indexed world state bank, and a per-chunk text condition, without increasing the context length or positional span of any individual generation step as the session grows (Fig. 3). The teacher combines chunk-wise sparse attention with per-chunk conditioning to provide supervision across distant moments and changing instructions, and these capabilities are transferred to the student through a -second long-horizon distribution-matching objective applied under self-forced rollouts. The following sections formalize the recurrent formulation, motivate the required supervision horizon, and detail the teacher, world state bank, distillation procedure, and inference process.

Figure 3: With bounded retention, per-step cost does not grow with session
length. The camera pose reads the world state bank into the target view; the student
injects that render at the coarsest of its three coarse-to-fine evaluations, alongside a
short parametric history and the current text condition. The emitted chunk then updates
the store and the history.

3.1 Recurrent Session Formulation

Figure 4: An hour-long session stays bounded rather than degrading. One continuous
 min Evoke rollout, three steps per chunk without classifier-free guidance
( chunks; faint per-chunk, bold two-minute mean). (a) Color is flat over the hour. (b) Scene identity plateaus at
cosine , the level real footage scores against itself  s apart.
(c) Opening-window shift stays far below unrelated-scene drift. A stability
claim, not fidelity ().

Evoke represents long-running generation as a bounded recurrent process over video chunks. At recurrent step , the student generates a chunk containing latent frames, corresponding to pixel frames or  s at  fps. Each step is conditioned on a camera trajectory (He et al. 2024; He et al. 2025a), containing camera-to-world extrinsics and intrinsics resampled to  fps, and a text condition that may vary across steps. A session is therefore formed by repeatedly applying the same fixed-shape generation interface for an arbitrary number of steps.

The recurrent interface separates transient denoiser context from persistent geometric world state, the loop drawn in Fig. 3. Let denote a bounded local history and the world state bank. At each step, geometry relevant to the current camera trajectory is first rendered from , the next video chunk is then generated from the rendered geometry, local history, and text condition, and the resulting observation is written back to the store:

(1)

Both the local history and world state bank operate under fixed budgets. Consequently, extending a session increases only the number of recurrent calls, without increasing the context length, positional span, or computational footprint of an individual call. Figure 4 illustrates an hour-scale session under this formulation, and Sec. 4.2 evaluates the resulting runtime behavior.

This bounded recurrence determines how Evoke can continue generating for long durations, but does not by itself ensure long-range content consistency or responsiveness to newly introduced instructions. These capabilities arise from two complementary sources: the supervision used to train , which determines what long-horizon behavior the student can acquire, and the external state , which preserves previously observed scene information across recurrent steps. The following sections analyze these two roles separately, beginning with the supervision horizon.

3.2 Long-Horizon Supervision

The bounded recurrent formulation keeps each inference step independent of session duration, but it does not by itself prevent errors from accumulating across recurrent steps. The remaining question is how much of a rollout must be visible to the training objective in order to constrain long-range behavior. Let denote the distribution of student trajectories and the corresponding data distribution. Under self-forced distribution matching (Yin et al. 2024b; Yin et al. 2024a; Huang et al. 2025a), supervision is evaluated over windows of consecutive chunks:

(2)

where denotes the divergence whose gradient is estimated from the teacher and critic scores. By construction, this objective depends only on trajectory statistics observable within a -chunk window. Two trajectories may therefore receive identical supervision at this horizon even if their behavior diverges over longer temporal spans. The supervision horizon consequently determines not only how much generated history is exposed during training, but also which forms of rollout drift can produce a learning signal.

This distinction is important because long-rollout errors arise at different temporal scales. Degradation drift, such as exposure shifts, saturation changes, and progressive texture degradation, eventually becomes visible within a local window. Its cause, however, may precede that window. During a self-forced rollout, the student conditions on its own generated history rather than ground-truth observations (Huang et al. 2025a; Cui et al. 2025), so earlier errors perturb the conditioning distribution encountered by subsequent denoising steps. A short window may capture the resulting degradation while exposing only a narrow range of the preceding perturbations that produced it. Increasing therefore broadens the distribution of rollout-induced histories under which the student is supervised and improves its ability to remain stable under accumulated conditioning shift.
Content drift imposes an additional constraint. Scene identity, object appearance, or spatial layout may evolve gradually while every short window remains individually plausible. Such errors become identifiable only when sufficiently distant moments are jointly covered by the supervision window (Cui et al. 2025; Liu et al. 2025).
Long-horizon supervision thus serves two complementary roles: it enlarges the range of rollout perturbations under which the recurrent generator learns to remain stable, and it exposes long-range inconsistencies that cannot be identified from local trajectory statistics alone.

Not all long-range inconsistency should be resolved by extending the supervision horizon. For temporal content drift, the correct evolution is governed by dependencies in the data distribution across distant moments; no particular past observation uniquely specifies the desired continuation. This failure is therefore naturally addressed through longer-horizon supervision. Spatial revisit inconsistency has a different source. When the camera returns to a previously observed surface, the relevant state has already been produced by the session, and the problem is to make that specific observation available again. Evoke accordingly assigns these two cases to different mechanisms: long-horizon teacher supervision constrains temporal evolution, while the world state bank preserves observed scene state and retrieves it through camera pose. A longer supervision window cannot substitute for an unavailable observation, and stored geometry alone cannot determine how unobserved content should evolve.

Importantly, the required supervision horizon need not scale with the final session duration. The goal is not to train the student to recover from arbitrarily corrupted histories, but to make its recurrent dynamics stable over the range of perturbations encountered during normal generation. As the student becomes more robust to accumulated conditioning shift, its own rollouts remain within a progressively better behaved region of trajectory space, reducing the need to cover increasingly extreme failure states. Extending should therefore provide large gains while it continues to expose new rollout perturbations and previously invisible long-range inconsistencies, followed by diminishing returns once the relevant temporal scale is covered. Evoke uses approximately  s of long-horizon supervision and tests this prediction by sweeping : resistance to drift should improve with the supervision horizon and then approach saturation rather than scale proportionally with the desired rollout length.

This analysis determines the requirements placed on the remaining components. The teacher must support sufficiently long windows at a cost compatible with repeated distillation queries, motivating the efficient long-sequence architecture in Sec. 3.3. Its supervision must also expose the time-varying conditioning encountered at deployment, since a teacher evaluated under a single global prompt cannot demonstrate how an instruction introduced mid-session should affect the rollout. Finally, previously observed scene geometry must remain explicitly accessible when revisited, which motivates the external world-state mechanism in Sec. 3.4.

3.3 Long-Horizon Interactive Teacher

Figure 5: Chunk-wise sparse attention makes long, time-varying supervision
affordable. Each query chunk of a  s clip attends to a fixed set of key sources
and reads the text segment covering it, so attention cost grows linearly in length; the
sampled timestep selects the high- or low-noise expert. Below: camera-control training
and few-step distillation precede long-rollout distribution matching against the shared
teacher/critic backbone.

Figure 6: A student inherits photometric stability from a long-horizon teacher. Two
three-step students with identical distillation recipes and teachers differing only in
temporal horizon; clips, trajectory, prompt, seed and inference flags are shared, so only
the checkpoint differs. (a) The short-horizon student settles at of its
opening brightness, the long-horizon one at . (b) Seven of eight clips
improve (Wilcoxon ). (c) Brightness separates the two most; sharpness
not at all. Claim: exposure stability, not image quality.

The previous section shows that the long-horizon behavior acquired by a few-step student is directly limited by the supervision expressed by its teacher. Evoke therefore treats the teacher not as a fixed source of high-quality scores, but as a model whose supervision is explicitly designed for long-horizon interactive generation. Figure 6 provides a controlled comparison: two students are distilled with identical recipes from teachers that differ only in temporal horizon, and the student supervised by the long-horizon teacher exhibits substantially greater stability over extended rollouts; Sec. 4.3 provides the full analysis. The Evoke Teacher is built on the B Wan2.2 A14B diffusion transformer (Peebles and Xie 2023; Wan et al. 2025a; Wan et al. 2025b) and retains its high-noise and low-noise experts selected according to the sampled timestep.
The teacher and critic share the same backbone rather than residing as two separate models. Enabling a LoRA adapter (Hu et al. 2021) yields the critic, while disabling it yields the teacher, so the teacher and critic scores are always computed within the same expert.

To make long-window supervision computationally practical, the Evoke Teacher partitions a sequence into chunks of nine latent frames and applies chunk-wise sparse attention (Jia et al. 2025; Cai et al. 2025). Each query chunk accesses a bounded set of sources: a first-frame global sink, local context with a one-frame overlap, spatially compressed nearby frames, a small set of selected distant frames, and a global state accumulated through linear attention (Yang et al. 2023; Team et al. 2025; Chen et al. 2025b), as illustrated in Fig. 5. Because the amount of information accessed by each chunk remains bounded, the resulting attention computation grows approximately linearly rather than quadratically with sequence length, as in other sparse-attention designs for long video generation (Li et al. 2025c), making repeated long-window scoring feasible during distillation. The same chunk partition provides an independent text-conditioning context for every chunk, allowing prompt changes to be represented within a single long sequence. This factorization also brings the teacher closer to the student’s chunk-wise recurrent inference, since local context and retrieved distant frames are drawn from the past while only limited non-causal paths remain. The mismatch between bidirectional teacher attention and causal student inference has been studied previously (Zhu et al. 2026); Evoke instead focuses on the complementary axes of supervision horizon and conditioning schedule.

Evoke transfers this long-horizon supervision to the few-step student through full-window distribution matching distillation (DMD) (Yin et al. 2024b; Yin et al. 2024a). Training starts from one ground-truth prefix chunk followed by a self-forced rollout of generated chunks , yielding latent frames, or pixel frames corresponding to approximately  s. Since the student performs three function evaluations per chunk, constructing the rollout requires student forward passes. Rather than sampling a short window from this trajectory, the teacher and critic jointly score all latent frames, so the supervision horizon spans the complete self-forced rollout. Let denote the student prediction, the teacher prediction, and the critic prediction. We define the score difference and normalize it by over the region that receives the distribution-matching gradient:

(3)

The mask excludes the ground-truth prefix and the first generated chunk . At this boundary, the teacher treats its first chunk according to the image-model first-frame distribution, whereas the student performs video continuation; directly matching the two introduces boundary flicker. The DMD gradient is therefore applied only to the remaining generated chunks , and is computed over the same mask so that excluded frames do not affect the update scale.

An important distinction is that the supervision horizon need not equal the gradient horizon. History is detached between consecutive rollout chunks, restricting each backward graph to a single chunk and allowing chunk-level gradients to be computed independently. This shortens the gradient horizon without shortening the supervision horizon: the teacher and critic still evaluate the complete  s trajectory jointly, so each local student update is derived from a distribution discrepancy defined over the full rollout. Long-horizon supervision therefore does not require backpropagation through the entire trajectory. Detachment bounds activation memory, while full-window scoring preserves the long-range training signal.

The teacher must also express the time-varying conditioning encountered at deployment. At inference, the student models

(4)

where the text condition may change at every recurrent step. A teacher evaluated under a single global prompt instead supervises a fixed-condition trajectory distribution and never demonstrates how a newly introduced instruction should affect an ongoing rollout. Evoke therefore assigns text conditions at the chunk level and includes prompt transitions within the training sequence, aligning both the supervision horizon and conditioning schedule with deployment. Training captions are segmented at  s intervals and mapped to their corresponding latent chunks, while each student rollout chunk receives the text condition associated with its temporal position. At inference, the same representation is exposed as a timed prompt schedule, allowing text-driven elements to be introduced or withdrawn during an ongoing session, a capability referred to as evocation. Figure 2 shows a controlled example in which only the timed text schedule changes while the camera trajectory, warp history, and random seed remain fixed, and Figure 11 in the appendix shows the same interface at the teacher’s own horizon: four-minute teacher rollouts driven by twelve consecutive instructions of  s each. This formulation also enables a direct conditioning ablation: with the supervision horizon held fixed, per-chunk teacher conditioning can be replaced by a single global prompt to isolate its effect on mid-session responsiveness, as evaluated in Sec. 4.4.

Pure distribution matching does not explicitly enforce adherence to the requested camera trajectory . Evoke therefore retains the supervised warp-conditioning objective from the preceding training stage as an auxiliary regularizer for camera controllability, while long-horizon distribution matching transfers the teacher’s long-range stability and dynamic-conditioning capabilities. Full-window training further uses sequence parallelism (Jacobs et al. 2023), activation recomputation, and independent chunk-level backward graphs to control the cost of the -frame teacher scoring and -chunk student rollout; the corresponding systems implementation and efficiency analysis are reported in the experiments.

3.4 Geometric World State

Evoke externalizes persistent world state that cannot remain within the bounded denoiser context (Wu et al. 2026). The student retains only the most recent latent frames, corresponding to approximately  s, organized into long-, mid-, and short-range history tiers of , , and frames, respectively. Observations outside this local history are no longer represented in the denoiser context, even though they may become relevant again when the camera revisits a previously observed region. Other designs keep such recalled history inside the denoiser, through a compressed cache or retrieved memory tokens (Hong et al. 2025; Wang et al. 2026). Evoke therefore maintains an external world state bank , using camera pose both as a control signal for generation and as an address for retrieving previously observed scene content. This realizes the spatial component identified in Sec. 3.2: long-range temporal evolution is learned through supervision, while previously observed spatial state is recovered explicitly.

At each recurrent step, newly generated observations are written into as world-space geometry. A monocular depth model (Lin et al. 2025a) estimates depth for frames of the generated chunk under the known camera trajectory.
The resulting depth maps are unprojected with the corresponding camera intrinsics and extrinsics and appended to the world state bank. Geometry from different chunks is inserted independently, which avoids the scale drift and rendering artifacts observed when depth estimates are fused across long generated sequences. To read the store, the current camera pose directly determines which stored observations are geometrically relevant. Stored source views are ranked by co-visibility with the target view, and up to eight sufficiently distinct sources are selected and rendered through batched projection with -buffering, following prior warp-conditioned generators (Wang and He 2026; Ren et al. 2025). The read operation returns both a view-aligned warped observation and a per-pixel visibility mask. In contrast to learned retrieval, recall is determined directly by camera geometry, which is appropriate for spatial revisit because the desired content has already been observed rather than inferred anew.

Figure 7: The world state bank puts a place back when the camera returns. Three held-out
street scenes follow one authored  s trajectory, generated by Evoke in three steps per
chunk without classifier-free guidance: a turn that carries
the opening view out of frame, and back. Left to right: opening view; turn
endpoint; the bank’s return render; the frame generated from it—the black band is what
the bank no longer holds and the model must inpaint. Right: retaining only the
latest  s of geometry buys nothing while the view is new and costs of
coverage at the return. Coverage bounds recall from above; it is not a fidelity measure.

The visibility mask controls how the rendered state enters the denoiser. Regions supported by stored geometry retain informative warp conditioning, whereas unsupported regions are suppressed by increasing the warp noise level; pixels with visibility below are assigned and therefore contribute no visual information. Supported regions receive substantially lighter noise, with . The same visibility signal is pooled at the patch resolution of each history tier and used to remove unsupported history tokens from the denoiser sequence, a visible-token selection introduced by prior work (Wang and He 2026). Consequently, geometric memory contributes only where the store provides reliable coverage: observed surfaces can be propagated into the current view, while uncovered regions remain available for synthesis rather than being constrained by unreliable warps.

The external state is operated under a fixed retention budget in long-session inference. The configuration used for hour-scale experiments retains pixel frames, corresponding to  s of geometry; because every third frame is ingested, the active source pool contains at most frames. Together with the fixed upper bound on retrieved source views, this keeps both storage and per-step geometric conditioning bounded with respect to session duration. The resulting guarantee is therefore persistent recall within retained geometric coverage rather than permanent recall of every location ever observed. When a retained surface is revisited, its state can be recovered through camera pose without extending the denoiser context, while observations outside the retention budget no longer constrain subsequent generation. Figure 7 illustrates this behavior by rendering a previously observed region when the camera returns to its stored geometry.

3.5 Bounded Three-Step Inference

Evoke generates each chunk with three CFG-free (Ho and Salimans 2022) denoising evaluations over a coarse-to-fine latent pyramid (Jin et al. 2024) at resolutions , , and .
Geometric conditioning is injected only at the coarsest stage, where it establishes large-scale spatial structure before the higher-resolution stages refine appearance and detail. Visibility-based pruning further removes unsupported geometric tokens, so the additional conditioning cost depends on the coverage of the world state bank rather than on session duration.

Together with bounded local history, bounded geometric retrieval, and local positional indices, each recurrent call operates within a session-independent context and positional range. Extending a session therefore increases only the number of recurrent calls, not the size of an individual call. Runtime and geometric-conditioning overhead are evaluated in Sec. 4.4.

4 Experiments

We evaluate Evoke along four dimensions: overall world-model and video-generation quality, stability and efficiency over hour-scale sessions, the effect of long-horizon teacher supervision, and the interaction between timed text control and persistent geometric memory. Together, these experiments assess both the quality of the three-step student and the design choices that enable it to operate continuously. Qualitative rollouts across egocentric, cinematic, embodied, and event-driven settings are collected in Fig. 12 in the appendix.

4.1 Experimental Setup and Benchmark Performance

Models and protocol.
The Evoke student is built on Helios (Yuan et al. 2026) and progressively trained for camera-controllable generation and few-step inference, followed by long-horizon distillation (long-distill) and a short post-distillation continuation (post-distill). Training uses the Sekai video dataset (Li et al. 2026) together with additional internal video data. For controlled distillation experiments, we compare a few-step student distilled from a short-horizon teacher with its counterpart distilled using the Evoke long-horizon teacher; the released Evoke student is obtained by continuing the latter training pipeline. The Evoke Teacher is adapted from the B Wan2.2 A14B diffusion transformer (Wan et al. 2025a; Wan et al. 2025b) and trained for long-sequence supervision with the architecture described in Sec. 3.3. Unless stated otherwise, the released student uses three denoising evaluations per chunk without classifier-free guidance.

All runtime measurements use a single H200 at resolution with the full VAE decoder and without inference-specific acceleration such as key-value caching, compilation, quantization, or a distilled decoder. Long-session quantitative evaluation uses eight -minute rollouts of chunks each, with the world state bank bounded to  s of retained observations. Additional verification and detailed protocol variations are provided in the appendix.

Table 1: WBench navigation split, cases, identical for all systems. All
scores , higher is better; bold is best,
underline second best. Group rows are the unweighted mean over their
group’s metrics; peer numbers are as reported by their authors.

Metric
Yume 1.5
Matrix-Game 2.0

HY-World 1.5

ar-distill

HY-GameCraft

LingBot-World

fast

LingBot-World v2

fast

Genie 3
Happy Oyster
Evoke (ours)

Video Quality

Aesthetic
58.7
54.0
60.1
52.6
62.6
64.4
51.6
56.6
66.12

Imaging
63.3
60.3
65.4
58.7
63.8
67.5
59.3
63.9
67.86

Flickering
93.0
94.6
93.5
93.7
92.4
91.4
95.0
94.0
94.31

Dynamic
96.8
94.9
91.1
96.8
95.6
96.2
92.4
94.2
96.84

Smoothness
97.0
98.2
98.1
97.6
96.0
96.5
97.8
97.0
97.86

HPSv3-Norm
57.0
41.0
60.5
38.3
65.7
74.6
55.2
58.3
73.75

Quality avg. (6)
77.63
73.83
78.12
72.95
79.35
81.77
75.22
77.33
82.79

Setting

Scene
53.1
49.4
53.5
50.6
63.4
66.7
61.1
57.4
74.68

Subject
91.7
84.9
90.8
82.5
92.4
86.9
83.8
91.1
92.84

Setting avg. (2)
72.40
67.15
72.15
66.55
77.90
76.80
72.45
74.25
83.76

Interaction

Navigation
72.0
80.6
87.5
67.8
79.4
82.8
73.3
85.1
78.63

Consistency

Background
90.3
86.9
92.7
86.5
90.9
92.5
90.7
91.4
92.27

Spatial
71.5
64.5
90.6
60.5
77.2
82.3
79.9
77.7
84.26

Gated Spatial
71.4
64.5
84.9
60.5
76.9
78.7
78.4
75.8
82.45

Segment
99.4
21.0
100.0
99.4
98.1
98.1
93.6
96.2
100.00

Perspective
48.0
29.2
62.5
17.9
82.8
84.5
54.5
75.0
69.74

Subject
88.8
87.2
89.1
82.6
88.6
88.9
90.4
91.5
91.03

Geometric
88.0
86.1
92.0
88.3
85.4
87.1
88.6
87.2
92.68

Photometric
83.3
81.3
83.1
85.0
79.1
79.8
84.5
79.8
82.53

Consistency avg. (8)
80.09
65.09
86.86
72.59
84.88
86.49
82.58
84.33
86.87

Physical

Causal Fidelity
72.7
59.3
74.0
68.3
72.5
76.7
71.7
69.3
82.44

Visual Plausibility
57.7
55.0
58.6
56.5
58.8
61.4
59.7
57.6
61.67

Physical avg. (2)
65.20
57.15
66.30
62.40
65.65
69.05
65.70
63.45
72.06

Interactive world-model performance.
On the -case navigation split of WBench (Ying et al. 2026), Evoke leads the Video Quality, Setting, and Physical group averages among the evaluated few-step systems (Mao et al. 2025; He et al. 2025b; HunyuanWorld 2025; Li et al. 2025a; Robbyant Team et al. 2026; Gao et al. 2026; Ball et al. 2025; Happy Oyster 2026), while remaining on par with the strongest result in Consistency. The largest gains occur in scene-level and causal-fidelity measures, consistent with the emphasis of Evoke on persistent scene state and long-running generation. Navigation and Perspective remain comparatively weaker, reflecting limitations of the current camera-control path that are analyzed separately.

Table 2: Evoke on two public leaderboards, against the top-10 of each. Higher is better; the full tables are Tables 3
and 4 in the appendix.

Benchmark
Evoke
Rank
Leader
Nearest peer

VBench-2.0
66.77
1 of 10
—
Veo 3, 66.72

VBench-Long
85.11
7 of 10
IPOW, 88.26
Veo 3, 85.06

Evoke is sampled in steps with no classifier-free guidance, against peers
running their own many-step default samplers, so neither comparison is
step-matched. “Leader” is the best peer total and is “—” where Evoke
leads. Protocol deviations are stated in Appendix 10.

General video quality.
Despite using only three CFG-free evaluations, Evoke remains competitive with many-step systems on both VBench-2.0 (Zheng et al. 2025) and VBench-Long (Huang et al. 2024; Huang et al. 2025b). It obtains an overall score of on VBench-2.0 and on VBench-Long, showing that the efficiency required for interactive generation does not come at the cost of a large degradation in general video quality. Full leaderboard results and per-dimension breakdowns are provided in Appendix 10.

4.2 Long-Session Stability and Bounded Runtime

Figure 1 qualitatively demonstrates uninterrupted two-hour generation under continuous camera control. For quantitative analysis, the eight -minute sessions show that extending the rollout does not lead to progressively increasing visual degradation. Photometric statistics stabilize after an initial transient and exhibit little subsequent drift over the remaining session. Content descriptors change more rapidly at the beginning of a rollout and then evolve substantially more slowly; importantly, real-video controls exhibit comparable decorrelation over time. We therefore interpret these measurements as evidence against runaway long-session degradation rather than as evidence of permanent scene-identity preservation.

The computational behavior remains bounded over the same sessions. Once the fixed retention budget is filled, the active geometric source pool no longer grows with elapsed time, and the cost of a recurrent step remains stable throughout the rollout. Session duration therefore increases the number of recurrent calls without increasing the state or computation required by an individual call. The student denoises each  s chunk in  s on a single H200; this number measures diffusion wall clock, while the complete instrumented inference path, including geometry rendering and video I/O, is analyzed in Sec. 4.4.

4.3 Effect of Long-Horizon Teacher Supervision

We next isolate the effect of the teacher by distilling students with otherwise matched recipes from short- and long-horizon teachers. Across paired long-rollout evaluations, the student supervised by the Evoke Teacher exhibits substantially stronger photometric stability than its short-teacher counterpart (Fig. 6). This establishes that long-horizon teacher training can transfer improved resistance to progressive photometric drift into the few-step student. The measured content descriptor does not significantly separate the two students, so we restrict this conclusion to photometric stability rather than attributing all forms of long-range consistency to the teacher horizon.

We further examine whether this improvement can be explained simply by making drift detectable over a longer scoring window. Controlled scoring sweeps reveal little additional change in teacher-critic detectability once the window exceeds a short horizon, and no consistent sharp threshold emerges across the tested perturbations. The empirical benefit of the long-horizon teacher therefore does not reduce to a simple window-length detectability effect. Detailed sweeps, score-noise analysis, and additional teacher ablations are reported in the appendix. These results motivate treating long-horizon supervision as a property of the complete trained teacher and rollout distribution rather than attributing its effect to window length alone.

4.4 Geometric Recall and Timed Interaction

Pose-addressed recall.
We first test whether the world state bank actually restores information that has left the denoiser context, measuring revisit PSNR between two  s windows rendered at identical camera poses. Across leave-and-return trajectories with different revisit intervals, recall remains near the far-pose floor measured within the same run when the retained geometry is shorter than the time spent away, and improves consistently once the retention window covers the revisit. Across the controlled conditions, of comparisons follow this predicted transition, and retention windows at least as long as the time away score – dB higher than shorter ones. The resulting – dB plateau indicates recognizable rather than pixel-faithful reconstruction, defining the scope of the current geometric memory. Figure 7 visualizes the same mechanism at short timescale: when the camera returns to a previously observed view, retained geometry restores the corresponding regions instead of requiring them to be regenerated from the short denoiser history.

Figure 8: A timed prompt switch introduces unanchored content; the anchored floor is not a
switching property. One checkpoint over schedules ( per group). Blue requests
a new object in unanchored space, red requests overwriting content pinned by the source
frame and warp history. (a) Unanchored clauses are realized in of cases
against an ceiling, anchored ones in against —the anchored floor is
mostly what this checkpoint renders at all, not the reach of switching. (b)
Pausing the anchor separates neither group, but it also freezes the camera, so it does
not isolate the anchor. (c) Per-chunk hit rate; shading marks assertion.

Timed text control.
The per-chunk conditioning interface allows text instructions to change during an ongoing session. In the controlled evocation evaluation, a clause introduced mid-session is realized in of cases when it targets previously unanchored content, but only when realizing it would require replacing geometry already supported by the world state bank (Fig. 8). The qualitative examples in Fig. 2 show the same behavior: timed instructions can introduce and withdraw new elements while anchored scene structure remains stable. This contrast exposes a useful interaction between the two control paths: text governs content that remains free to evolve, whereas persistent geometry resists overwriting observations that have already been anchored. A matched short- versus long-teacher comparison does not significantly separate mid-session realization rates; accordingly, per-chunk conditioning is interpreted as the interface that expresses timed control rather than as an isolated causal explanation for the capability.

Cost of geometric memory.
The geometric path remains bounded by construction and adapts to the amount of retrieved coverage. Only a fixed number of source observations are rendered, and unsupported warp tokens are removed before denoising. Consequently, additional geometric-conditioning cost depends on how much previously observed content is visible in the current view rather than on how long the session has been running. Detailed wall-clock decomposition, the dependence on geometric coverage, camera-trajectory controls for revisit evaluation, and training-system efficiency are provided in the appendix.

5 Conclusion

We present Evoke, a three-step video world model for long-horizon interactive generation.
Evoke decouples persistent world state from ever-growing model context through external
geometric memory, while a teacher redesigned for long-horizon generation provides extended
supervision for the few-step student. This design enables continuous camera and text control
while keeping the state and context of each recurrent generation step bounded as the session
grows. Experiments show that Evoke achieves competitive performance on interactive
world-model benchmarks while sustaining stable hour-scale generation with bounded
computational cost.

Several directions remain open. First, the current geometric world state primarily preserves
coarse scene structure, while fine-grained consistency of object identity, appearance, and
local details remains limited. Richer object-level or semantic world representations may
provide stronger long-term identity consistency. Second, a persistent world should model not
only static geometry but also dynamic state, including object motion, state transitions, and
their long-term evolution. Developing explicit representations that can continuously update
such dynamic world state is an important next step. Finally, further inference acceleration
remains necessary for truly real-time interaction, including higher-compression video VAEs,
more efficient few-step generators, and lower-cost geometric conditioning.

References

Ball et al. (2025)

Philip J. Ball, Jakob Bauer, Frank Belletti, et al.

Genie 3: A new frontier for world models, 2025.

https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/.

Bar et al. (2025)

Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun.

Navigation world models.

In CVPR, 2025.

Bruce et al. (2024)

Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al.

Genie: Generative interactive environments.

In ICML, 2024.

Cai et al. (2025)

Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, et al.

Mixture of contexts for long video generation.

arXiv preprint arXiv:2508.21058, 2025.

Chen et al. (2024)

Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann.

Diffusion forcing: Next-token prediction meets full-sequence diffusion.

Advances in Neural Information Processing Systems, 37:24081–24125, 2024.

Chen et al. (2025a)

Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, et al.

Skyreels-v2: Infinite-length film generative model.

arXiv preprint arXiv:2504.13074, 2025a.

Chen et al. (2025b)

Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, et al.

Sana-video: Efficient video generation with block linear diffusion transformer.

arXiv preprint arXiv:2509.24695, 2025b.

Cui et al. (2025)

Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh.

Self-forcing++: Towards minute-scale high-quality video generation.

arXiv preprint arXiv:2510.02283, 2025.

Dalal et al. (2025)

Karan Dalal, Daniel Koceja, Jiarui Xu, Yue Zhao, Shihao Han, Ka Chun Cheung, Jan Kautz, Yejin Choi, Yu Sun, and Xiaolong Wang.

One-minute video generation with test-time training.

In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17702–17711, 2025.

Decart (2026)

Decart.

Introducing oasis 3: The first interactive world model for physical ai.

https://decart.ai/publications/introducing-oasis-3-first-interactive-world-model-for-physical-ai, June 2026.

Blog post; no accompanying paper.

Feng et al. (2025)

Tianrui Feng, Zhi Li, Shuo Yang, Haocheng Xi, Muyang Li, Xiuyu Li, Lvmin Zhang, Keting Yang, Kelly Peng, Song Han, et al.

Streamdiffusionv2: A streaming system for dynamic and interactive video generation.

arXiv preprint arXiv:2511.07399, 2025.

Gao et al. (2026)

Zelin Gao, Qiuyu Wang, Jiapeng Zhu, Jingye Chen, Zichen Liu, Qingyan Bai, Jiahao Wang, Yufeng Yuan, Hanlin Wang, Yichong Lu, et al.

Infinite worlds with versatile interactions.

arXiv preprint arXiv:2607.07534, 2026.

Happy Oyster (2026)

Happy Oyster.

Happy oyster: Real-time world model for interactive creation, 2026.

https://www.happyoyster.com.

He et al. (2024)

Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang.

Cameractrl: Enabling camera control for text-to-video generation.

arXiv preprint arXiv:2404.02101, 2024.

He et al. (2025a)

Hao He, Ceyuan Yang, Shanchuan Lin, Yinghao Xu, Meng Wei, Liangke Gui, Qi Zhao, Gordon Wetzstein, Lu Jiang, and Hongsheng Li.

Cameractrl ii: Dynamic scene exploration via camera-controlled video diffusion models.

In CVPR, 2025a.

He et al. (2025b)

Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, et al.

Matrix-game 2.0: An open-source real-time and streaming interactive world model.

arXiv preprint arXiv:2508.13009, 2025b.

Ho and Salimans (2022)

Jonathan Ho and Tim Salimans.

Classifier-free diffusion guidance.

arXiv preprint arXiv:2207.12598, 2022.

Hong et al. (2025)

Yicong Hong, Yiqun Mei, Chongjian Ge, Yiran Xu, Yang Zhou, Sai Bi, Yannick Hold-Geoffroy, Mike Roberts, Matthew Fisher, Eli Shechtman, Kalyan Sunkavalli, Feng Liu, Zhengqi Li, and Hao Tan.

Relic: Interactive video world model with long-horizon memory.

arXiv preprint arXiv:2512.04040, 2025.

Hu et al. (2021)

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.

Lora: Low-rank adaptation of large language models.

arXiv preprint arXiv:2106.09685, 2021.

Huang et al. (2025a)

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman.

Self forcing: Bridging the train-test gap in autoregressive video diffusion.

arXiv preprint arXiv:2506.08009, 2025a.

Huang et al. (2024)

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al.

Vbench: Comprehensive benchmark suite for video generative models.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

Huang et al. (2025b)

Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al.

Vbench++: Comprehensive and versatile benchmark suite for video generative models.

IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025b.

HunyuanWorld (2025)

Team HunyuanWorld.

Hy-world 1.5: A systematic framework for interactive world modeling with real-time latency and geometric consistency.

arXiv preprint, 2025.

Jacobs et al. (2023)

Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He.

Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models.

arXiv preprint arXiv:2309.14509, 2023.

Jia et al. (2025)

Weinan Jia, Yuning Lu, Mengqi Huang, Hualiang Wang, Binyuan Huang, Nan Chen, Mu Liu, Jidong Jiang, and Zhendong Mao.

Moga: Mixture-of-groups attention for end-to-end long video generation.

arXiv preprint arXiv:2510.18692, 2025.

Jin et al. (2024)

Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin.

Pyramidal flow matching for efficient video generative modeling.

arXiv preprint arXiv:2410.05954, 2024.

Kerbl et al. (2023)

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis.

3d gaussian splatting for real-time radiance field rendering.

ACM Transactions on Graphics, 2023.

Li et al. (2025a)

Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu.

Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition.

arXiv preprint arXiv:2506.17201, 2025a.

Li et al. (2025b)

Wuyang Li, Wentao Pan, Po-Chien Luan, Yang Gao, and Alexandre Alahi.

Stable video infinity: Infinite-length video generation with error recycling.

arXiv preprint arXiv:2510.09212, 2025b.

Li et al. (2025c)

Xingyang Li, Muyang Li, Tianle Cai, Haocheng Xi, Shuo Yang, Yujun Lin, Lvmin Zhang, Songlin Yang, Jinbo Hu, Kelly Peng, et al.

Radial attention: Sparse attention with energy decay for long video generation.

arXiv preprint arXiv:2506.19852, 2025c.

Li et al. (2026)

Zhen Li, Chuanhao Li, Xiaofeng Mao, Shaoheng Lin, Ming Li, Shitian Zhao, Zhaopan Xu, Xinyue Li, Yukang Feng, Jianwen Sun, et al.

Sekai: A video dataset towards world exploration.

Advances in Neural Information Processing Systems, 38, 2026.

Lin et al. (2025a)

Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang.

Depth anything 3: Recovering the visual space from any views.

arXiv preprint arXiv:2511.10647, 2025a.

Lin et al. (2025b)

Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, and Lu Jiang.

Diffusion adversarial post-training for one-step video generation.

arXiv preprint arXiv:2501.08316, 2025b.

Liu et al. (2025)

Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu.

Rolling forcing: Autoregressive long video diffusion in real time.

arXiv preprint arXiv:2509.25161, 2025.

Luo et al. (2023)

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao.

Latent consistency models: Synthesizing high-resolution images with few-step inference.

arXiv preprint arXiv:2310.04378, 2023.

Mao et al. (2025)

Xiaofeng Mao, Zhen Li, Chuanhao Li, Xiaojie Xu, Kaining Ying, Tong He, Jiangmiao Pang, Yu Qiao, and Kaipeng Zhang.

Yume-1.5: A text-controlled interactive world generation model.

arXiv preprint arXiv:2512.22096, 2025.

Mildenhall et al. (2021)

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng.

Nerf: Representing scenes as neural radiance fields for view synthesis.

Communications of the ACM, 2021.

NVIDIA (2025)

NVIDIA.

Cosmos world foundation model platform for physical ai.

arXiv preprint arXiv:2501.03575, 2025.

Peebles and Xie (2023)

William Peebles and Saining Xie.

Scalable diffusion models with transformers.

In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

Ren et al. (2025)

Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, and Jun Gao.

Gen3c: 3d-informed world-consistent video generation with precise camera control.

In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6121–6132. IEEE, 2025.

Robbyant Team et al. (2026)

Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, Yihang Chen, Jie Liu, Yansong Cheng, Yao Yao, Jiayi Zhu, Yihao Meng, Kecheng Zheng, Qingyan Bai, Jingye Chen, Zehong Shen, Yue Yu, Xing Zhu, Yujun Shen, and Hao Ouyang.

Advancing open-source world models.

arXiv preprint arXiv:2601.20540, 2026.

Salimans and Ho (2022)

Tim Salimans and Jonathan Ho.

Progressive distillation for fast sampling of diffusion models.

arXiv preprint arXiv:2202.00512, 2022.

Shen et al. (2026)

Tianchang Shen, Sherwin Bahmani, Kai He, Sangeetha Grama Srinivasan, Tianshi Cao, Jiawei Ren, Ruilong Li, Zian Wang, Nicholas Sharp, Zan Gojcic, et al.

Lyra 2.0: Explorable generative 3d worlds.

arXiv preprint arXiv:2604.13036, 2026.

Song et al. (2023)

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever.

Consistency models.

arXiv preprint arXiv:2303.01469, 2023.

Su et al. (2024)

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu.

Roformer: Enhanced transformer with rotary position embedding.

Neurocomputing, 568:127063, 2024.

Sun et al. (2025)

Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo.

Worldplay: Towards long-term geometric consistency for real-time interactive world modeling.

arXiv preprint arXiv:2512.14614, 2025.

Team et al. (2025)

Kimi Team, Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi Hu, Fanqing Meng, Chengyin Liu, Xin Men, Songlin Yang, Zhiyuan Li, et al.

Kimi linear: An expressive, efficient attention architecture.

arXiv preprint arXiv:2510.26692, 2025.

Valevski et al. (2024)

Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter.

Diffusion models are real-time game engines.

arXiv preprint arXiv:2408.14837, 2024.

Wan et al. (2025a)

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al.

Wan: Open and advanced large-scale video generative models.

arXiv preprint arXiv:2503.20314, 2025a.

Wan et al. (2025b)

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al.

Wan: Open and advanced large-scale video generative models.

arXiv preprint arXiv:2503.20314, 2025b.

Wang et al. (2025)

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny.

Vggt: Visual geometry grounded transformer.

In CVPR, 2025.

Wang and He (2026)

Yifan Wang and Tong He.

Warp-as-history: Generalizable camera-controlled video generation from one training video.

arXiv preprint arXiv:2605.15182, 2026.

Wang et al. (2026)

Zile Wang, Zexiang Liu, Jiaxing Li, Kaichen Huang, Baixin Xu, Fei Kang, Mengyin An, Peiyu Wang, Biao Jiang, Yichen Wei, et al.

Matrix-game 3.0: Real-time and streaming interactive world model with long-horizon memory.

arXiv preprint arXiv:2604.08995, 2026.

Wu et al. (2026)

Tong Wu, Shuai Yang, Ryan Po, Yinghao Xu, Ziwei Liu, Dahua Lin, and Gordon Wetzstein.

Video world models with long-term spatial memory.

Advances in Neural Information Processing Systems, 38:49371–49393, 2026.

Yang et al. (2025)

Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al.

Longlive: Real-time interactive long video generation.

arXiv preprint arXiv:2509.22622, 2025.

Yang et al. (2023)

Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim.

Gated linear attention transformers with hardware-efficient training.

arXiv preprint arXiv:2312.06635, 2023.

Yin et al. (2024a)

Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman.

Improved distribution matching distillation for fast image synthesis.

In NeurIPS, 2024a.

Yin et al. (2024b)

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park.

One-step diffusion with distribution matching distillation.

In CVPR, 2024b.

Yin et al. (2025a)

Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang.

From slow bidirectional to fast autoregressive video diffusion models.

In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22963–22974, 2025a.

Yin et al. (2025b)

Yuanyang Yin, Yaqi Zhao, Mingwu Zheng, Ke Lin, Jiarong Ou, Rui Chen, Victor Shea-Jay Huang, Jiahao Wang, Xin Tao, Pengfei Wan, et al.

Towards precise scaling laws for video diffusion transformers.

In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18155–18165. IEEE, 2025b.

Ying et al. (2026)

Kaining Ying, Hengrui Hu, Siyu Ren, Jiamu Li, Fengjiao Chen, Ziwen Wang, Xuezhi Cao, Xunliang Cai, and Henghui Ding.

Wbench: A comprehensive multi-turn benchmark for interactive video world model evaluation.

arXiv preprint arXiv:2605.25874, 2026.

Yuan et al. (2026)

Shenghai Yuan, Yuanyang Yin, Zongjian Li, Xinwei Huang, Xiao Yang, and Li Yuan.

Helios: Real real-time long video generation model.

arXiv preprint arXiv:2603.04379, 2026.

Zhang and Agrawala (2025)

Lvmin Zhang and Maneesh Agrawala.

Packing input frame context in next-frame prediction models for video generation.

arXiv preprint arXiv:2504.12626, 2025.

Zhang et al. (2025)

Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, et al.

Matrix-game: Interactive world foundation model.

arXiv preprint arXiv:2506.18701, 2025.

Zheng et al. (2025)

Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, et al.

Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness.

arXiv preprint arXiv:2503.21755, 2025.

Zhu et al. (2026)

Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu.

Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation.

arXiv preprint arXiv:2602.02214, 2026.

\beginappendix

6 Training curves

Figure 9 is the optimization record of the two distillation stages that
produce the released Evoke student: long-distill against the Evoke Teacher
on GPUs, and the short post-distill continuation that yields the
released student.

Figure 9: Both distillation stages train stably. Long-distill (Evoke Teacher,
 GPUs, steps) and the short post-distill continuation that
yields the released student; faint per-step, bold -step exponential moving average.
(a) DMD stays bounded in both. (b) The critic converges.
(c) The normalized gradient entering the loss plateaus; we release at its onset.
(d) Gradient norms stay bounded across GPUs and eight scheduler restarts.

7 Cost of the geometric path

Figure 10 breaks one recurrent step’s wall clock into the denoiser and the geometric
path, and isolates the coarsest-stage token surcharge that §3.5
attributes to warp conditioning. It is also the source of the coverage fit reported
there and in §4.4.

Figure 10: The geometric path costs of the denoiser and of a recurrent step that skips
it. Per-chunk wall clock on one H200 at , 3 denoising steps, CFG off; arms differ
only by the named flag. (a) Warp rendering and coarsest-stage token admission
are separable costs, and stages 1 and 2 are unchanged throughout—tokens, not geometry,
carry denoiser cost. (b) Geometry adds  s per chunk, scaling with warp
coverage: unobserved space costs less exactly where memory offers less.

8 Long-Horizon Generation and Teacher Ablations

The long-session evaluation in Sec. 4.2 consists of eight continuous
rollouts, each spanning recurrent steps, minutes, and generated
frames. All runs use three CFG-free evaluations per chunk at on a single
H200, with the world state bank restricted to  s of retained observations. The reported
 s latency refers to diffusion wall clock rather than end-to-end execution. A further
hour-scale rollout was run as a consistency check; its photometric and content-similarity
trajectories remain within the range of the eight-session evaluation.

Long-session behavior is evaluated along complementary photometric and content-based
measures. Photometric statistics quantify progressive changes in brightness, saturation,
and related low-level appearance, while the content descriptor measures similarity to each
session’s own opening segment. The latter is a self-consistency measure rather than a
reference-based identity metric. Real-video controls exhibit substantial descriptor
decorrelation as the camera naturally traverses new content, so absolute similarity values
should not be interpreted as permanent preservation of scene identity. The relevant
observation is instead whether generated trajectories continue to diverge progressively with
elapsed time. Full per-session curves and the corresponding real-video controls are provided
here to complement the aggregated results in the main paper.

To isolate the effect of long-horizon teacher supervision, we additionally compare few-step
students distilled with matched recipes from the short-horizon teacher and the Evoke
teacher. The long-horizon teacher yields a clear improvement in photometric stability over
extended rollouts, while the measured content descriptor does not significantly separate the
two students. Sharpness does not improve under the same comparison, and the final continued
checkpoint shows no measured drift advantage over the long-horizon distilled checkpoint.
These results therefore localize the demonstrated transfer benefit to photometric stability
rather than implying a uniform improvement across all measures of long-range consistency.

We further examine whether this benefit can be reduced to a simple dependence on the
scoring-window length . Across controlled perturbation conditions, teacher-critic
detectability changes little once the window exceeds chunks, and no consistent sharp
threshold emerges as increases. Varying the temporal distance between the perturbation
and the scored chunk produces similar response curves, indicating that a simple
window-coverage explanation is insufficient for the observed teacher advantage. Analysis
over logged training steps further shows that stochastic variation from individual
teacher-critic evaluations is large relative to the systematic drift component, although
aggregation across evaluations can recover the latter. Thus, the empirical benefit of the
Evoke Teacher should be attributed to the trained long-horizon supervision process as a
whole rather than to a single detectability threshold determined by .

Finally, the timed-conditioning ablation separates the control interface from its causal
attribution. Mid-session prompt changes are supported by the per-chunk conditioning
representation, but matched short- and long-horizon teacher variants do not significantly
differ in event-realization rate at the current sample size. The main paper therefore treats
per-chunk conditioning as the mechanism by which timed instructions are represented, without
attributing the observed evocation capability solely to the teacher horizon or conditioning
schedule.

Figure 11 shows that interface at the Evoke Teacher’s own horizon rather than the
student’s: four four-minute rollouts generated by the teacher, each driven by twelve
consecutive instructions held for  s. Every instruction takes visible effect inside its own segment and the session
runs across all twelve without a reset. The tiles are selected within their segments rather
than sampled at a fixed offset, as the caption states, so the figure documents the behavior
of the conditioning schedule and carries no measurement.

Figure 11: Four-minute Evoke Teacher rollouts under a twelve-prompt schedule. Four sessions (a–d) generated by the Evoke Teacher rather than the three-step student, each driven by twelve instructions held for  s apiece. The grey strip above each block is that schedule on a common time axis: the eight segments illustrated below are filled dark, the four that are not remain pale. Each tile is one frame selected inside its own segment rather than sampled at a fixed offset.

9 Qualitative rollouts

Figure 12 collects nine sessions across four settings: egocentric action and
manipulation, cinematic scenes and styles, embodied and industrial scenes, and timed events.
Each row samples five frames evenly along its own clip rather than at chosen moments, so a
row shows what the rollout does over its whole length; the event rows are longer because the
timed instruction fires in the last third. The rings in the lower corners of every tile are
the interactive demo’s joystick overlay, which shows the camera command driving the
motion. Like Fig. 11, the sheet is qualitative
and carries no measurement.

Figure 12: Qualitative rollouts across four domains. Nine sessions generated by Evoke in three steps per chunk without classifier-free guidance, one per row, each shown as five frames evenly spaced over the clip:  s in (a–c) and  s in (d), where a timed event fires in the last third. The rings in the lower corners are the engine’s own joystick overlay.

10 Public leaderboards in full

Table 2 in the main text summarizes two public leaderboards (Zheng et al. 2025; Huang et al. 2025b);
Tables 3 and 4 give the full top-ten of each,
and Tables 5 and 6 the per-dimension
scores behind them. Peer rows are the public leaderboard as of 2026-08-09. Both
breakdowns are reported at the scale on which columns are comparable across rows:
VBench-2.0 aggregates raw dimension scores without normalization, whereas
VBench-Long min–max-normalizes before weighting, so the Evoke row of
Table 6 is de-normalized back to raw with the official
constants.

Protocol deviations.

Our VBench-2.0 row departs from the official
protocol in four declared ways: one
sample per prompt rather than five, with Diversity keeping the its
metric requires;  s clips; resolution; and prompt
augmentation, which the official protocol permits and which leaves case selection,
dimension membership and every answer key byte-identical to the official metadata.
Our VBench-Long row departs in three ways: one sample per prompt;
 s clips rather than  s, which the official code supports and which two
controlled probes put at Total in our favor; and the same prompt
augmentation. Single-sampling inflates variance but not the expectation, and the
 margins reported in Table 2 are of that order, so neither
ranking should be over-read.

Table 3: VBench-2.0, top-10 of the public leaderboard, Evoke
included. All scores , higher is better; bold is best,
underline second best.

Model
Total
Creat.
Comm.
Contr.
Human
Phys.

Evoke (3 step)
66.77
55.22
76.62
41.99
94.23
65.80

Veo 3
66.72
60.85
69.48
47.04
86.88
69.35

JT-CV
64.60
51.80
66.94
45.47
83.53
75.23

ABot-World v0.1
64.47
55.29
72.93
44.13
85.36
64.61

Vidu Q1 (2025-04-17)
62.70
56.54
65.98
38.13
81.24
71.63

ToMoviee 2.0
61.78
45.96
67.41
45.37
80.68
69.47

Wan2.1
60.20
55.25
63.98
37.32
81.60
62.84

Seedance 1.0 Pro (2025-05-28)
59.81
53.04
64.31
39.84
77.06
64.81

Kling 1.6
59.00
48.58
65.45
33.05
83.56
64.35

Sora-480p
58.38
60.57
64.32
22.09
87.72
57.18

Table 4: VBench-Long, top-10 of the public leaderboard, Evoke
included. All scores , higher is better; bold is best,
underline second best.

Model
Total
Quality
Semantic

IPOW
88.26
87.83
90.01

Vidu Q1 (2025-04-17)
87.41
87.28
87.94

IPOC (2025-04-14)
86.57
87.00
84.84

Wan2.1(2025-02-24)
86.22
86.67
84.44

IPOC
85.71
86.12
84.09

MiracleVision V5
85.23
86.68
79.43

Evoke (3 step)
85.11
85.55
83.36

Veo 3
85.06
85.70
82.49

LanDiff
84.87
85.41
82.72

Wan2.1
84.70
85.64
80.95

Table 5: Per-dimension VBench-2.0 breakdown for the systems of
Table 3. Raw scores in , higher is better;
bold is best, underline second best.

Creativity
Commonsense
Controllability
Human Fidelity
Physics

Model

Composition

Diversity

Instance Preserv.

Motion Rational.

Dyn. Spatial Rel.

Dyn. Attribute

Motion Order

Human Interaction

Complex Landscape

Complex Plot

Camera Motion

Human Anatomy

Human Identity

Human Clothes

Mechanics

Thermotics

Material

Multi-View Cons.

Evoke (3 step)
70.38
40.07
87.72
65.52
36.23
50.55
35.35
71.00
18.00
12.42
70.37
89.23
93.45
100.00
80.00
75.00
83.33
24.87

Veo 3
68.57
53.13
92.98
45.98
45.89
63.74
40.40
80.67
21.78
21.87
54.94
90.26
70.84
99.53
81.82
75.51
84.11
35.95

JT-CV
58.75
44.86
73.53
60.34
44.05
54.21
35.35
84.67
19.11
14.87
66.05
93.64
64.51
92.44
76.69
66.21
81.25
76.78

ABot-World v0.1
55.06
55.52
92.98
52.87
31.88
61.90
39.39
80.67
21.33
12.01
61.73
91.77
66.27
98.05
68.42
73.97
85.34
30.72

Vidu Q1 (2025-04-17)
60.94
52.14
86.55
45.40
15.46
55.68
34.01
73.67
17.56
10.67
59.88
88.87
73.31
81.54
81.75
64.03
72.82
67.92

ToMoviee 2.0
51.78
40.13
85.96
48.85
40.10
61.17
37.71
75.33
17.78
15.46
70.06
76.78
75.16
90.10
74.24
67.35
82.46
53.84

Wan2.1
54.51
56.00
88.89
39.08
31.40
45.42
35.35
80.00
16.44
13.14
39.51
82.59
65.73
96.48
71.74
67.16
75.68
36.79

Seedance 1.0 Pro (2025-05-28)
51.91
54.18
80.92
47.70
37.20
44.32
29.63
79.67
16.67
13.98
57.41
64.05
67.13
100.00
63.16
64.34
77.38
54.37

Kling 1.6
43.89
53.26
92.40
38.51
20.77
19.41
29.29
72.00
17.33
10.83
61.73
86.99
71.95
91.75
65.55
59.46
68.00
64.38

Sora-480p
53.65
67.48
94.15
34.48
19.81
8.06
15.15
58.00
15.33
11.11
27.16
86.45
78.57
98.15
62.22
43.36
64.94
58.22

Evoke is sampled in steps (three-stage pyramid, one step per stage) with no CFG, i.e. network evaluations per chunk, against peers running their own many-step default samplers.
The four protocol deviations of Table 3 apply unchanged.
Diversity alone is generated from the official raw prompts, since
augmenting it was measured to cost points.

Table 6: Per-dimension VBench-Long breakdown for the systems of
Table 4. Raw scores in , before the
min–max normalization the aggregates apply; higher is better,
bold is best, underline second best.

Quality dimensions
Semantic dimensions

Model

Subject Cons.

Background Cons.

Temporal Flicker

Motion Smooth.

Aesthetic

Imaging

Dynamic Degree

Object Class

Multiple Objects

Human Action

Color

Spatial Rel.

Scene

Appearance Style

Temporal Style

Overall Cons.

IPOW
97.31
97.67
99.76
97.94
67.42
68.86
95.56
97.43
95.96
97.00
97.15
97.80
76.32
23.60
27.59
26.77

Vidu Q1 (2025-04-17)
95.80
97.11
99.34
98.47
67.32
69.07
91.85
97.40
93.89
99.60
99.04
93.74
67.41
22.37
26.46
27.24

IPOC (2025-04-14)
96.60
97.57
99.78
97.60
65.22
69.02
93.06
92.41
87.25
98.00
86.79
95.46
61.45
24.01
25.87
26.85

Wan2.1(2025-02-24)
96.62
97.58
99.38
97.42
63.45
69.63
94.26
96.64
86.59
99.20
94.39
85.70
61.24
21.59
26.18
27.49

IPOC
96.68
97.21
99.11
97.12
64.11
68.43
92.59
94.14
85.73
99.40
88.64
90.55
54.01
24.59
25.79
27.56

MiracleVision V5
97.32
96.35
98.62
99.02
65.28
65.67
93.52
93.04
81.60
98.00
89.92
76.22
46.80
21.72
25.16
26.94

Evoke (3 step)
97.92
98.10
99.52
98.96
64.43
64.94
71.39
91.65
85.67
99.00
94.76
82.82
62.15
22.25
24.70
27.31

Veo 3
97.36
96.89
99.30
99.16
63.81
68.23
72.43
93.89
82.20
99.40
82.48
84.26
57.43
23.55
25.97
27.88

LanDiff
91.86
98.24
99.42
97.11
65.24
65.78
93.33
94.86
87.13
96.60
91.22
73.58
54.84
25.60
25.28
27.43

Wan2.1
95.92
97.39
99.53
96.92
61.53
67.28
94.35
94.24
81.44
98.80
87.79
81.08
53.67
21.13
25.69
27.44

Evoke is sampled in steps (three-stage pyramid, one step per stage) with no CFG, i.e. network evaluations per chunk, against peers running their own many-step default samplers.
The three protocol deviations of Table 4 apply unchanged.
The Evoke row is stored normalized in the source data and is de-normalized
here with the official constants so that it shares the leaderboard’s scale.

11 WBench public leaderboard

Table 1 compares Evoke with eight few-step interactive systems on the
WBench navigation split (Ying et al. 2026). Table 7 places the same run in the public
WBench leaderboard for that split, which ranks thirty systems of any sampling budget by
the unweighted mean of the five group scores; on this split the Interaction group is the
navigation dimension alone. The comparison is not step-matched: Evoke is sampled in three
steps without classifier-free guidance, whereas most rows around it run their own
many-step defaults. Our row leads the board by Average, a margin of the same order
as the ones §4.1 declines to read as wins.

Table 7: WBench public leaderboard, Navi split ( cases), top ten by Average, Evoke included. Average is the unweighted mean of the five group scores; peer rows are the public leaderboard as of 2026-08-12. All scores , higher is better; bold is best, underline second best.

#
Model
Average
Quality
Setting
Inter.
Consist.
Phys.

1
Evoke (ours, 3 step)†
80.8
82.8
83.8
78.6
86.9
72.1

2
HiDream-O1-World
80.7
81.9
81.9
79.5
88.0
72.1

3
LingBot-World v2 fast
79.4
81.8
76.8
82.8
86.5
69.1

4
Kling 3.0
79.0
81.4
91.0
69.4
83.7
69.3

5
LingBot-World base-camera
78.5
78.9
72.6
80.1
89.9
71.2

6
Wan 2.7
78.1
81.5
91.4
64.4
81.6
71.8

7
HY-World 1.5 ar-distill
78.1
78.1
72.2
86.8
86.9
66.3

8
HY-Video 1.5
77.9
77.6
85.6
71.4
87.4
67.4

9
LingBot-World fast
77.4
79.4
77.9
79.2
84.9
65.7

10
Happy Oyster
76.8
77.3
74.2
84.9
84.3
63.5

† Our own evaluation of the released student, not a leaderboard submission. Evoke is sampled in three steps with no classifier-free guidance against peers running their own default samplers, so the comparison is not step-matched.

 Experimental support, please
 view the build logs
 for errors. Generated by

 L
 A
 T
 E

 xml

 .

 Instructions for reporting errors

 We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile
 support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the
 methods listed below:

 Click the "Report Issue" (

 ) button, located in the page header.

 Tip: You can select the relevant text first, to include it in your report.

 Our team has already identified the following issues. We appreciate your time reviewing and reporting rendering errors we
 may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability
 should not be a barrier to accessing research. Thank you for your continued support in championing open access for
 all.

 Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a list of packages that need conversion, and welcome developer contributions.

 We gratefully acknowledge support from
 our major funders,
 member institutions, ,
 and all contributors.

 About
 ·
 Help
 ·
 Contact
 ·
 Subscribe
 ·
 Copyright
 ·
 Privacy
 ·
 Accessibility
 ·
 Operational Status (opens in new tab)

 Major funding support from

