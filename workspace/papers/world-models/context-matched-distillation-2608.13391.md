

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

2 Related Work

3 Preliminaries

4 Methodology

4.1 Motivation

4.2 Context-Matched Distillation

4.3 Long-Video Prefixes

4.4 Camera-Conditioned Causal Distillation

5 Experiments

5.1 Training

5.2 Quantitative Evaluation

5.3 Ablative experiments

5.4 Qualitative Analysis

6 Conclusion

References

 License: arXiv.org perpetual non-exclusive license

arXiv:2608.13391v1 [cs.CV] 13 Aug 2026

Context-Matched Distillation: Teacher Causality for Autoregressive Video Distillation

Hmrishav Bandyopadhyay 1,2  Xuanchi Ren 1  Zijian Huang 1  Jay Zhangjie Wu 1  Tianshi Cao 1  Ruilong Li 1  Bryan Chu 1  Sanja Fidler 1  Yi-Zhe Song 2  Zian Wang 1

1 NVIDIA  2 SketchX, CVSSP, University of Surrey

https://hmrishavbandy.github.io/cmd-site/

Abstract

Interactive autoregressive video generation demands both low-latency rollouts and precise online control. Few-step distillation accelerates generation by reducing denoising steps, while online control imposes a causal constraint: frames and blocks should depend on history and controls available during generation. Existing video distribution matching distillation (DMD) pipelines, however, often supervise causal few-step students using bidirectional teachers that score complete clips. The score for a target can therefore depend on future frames and controls that were unavailable when the student generated it, misaligning teacher supervision with the student’s causal information set. We introduce Context-Matched Distillation (CMD), a causal DMD framework that aligns teacher supervision with the information available when each target is generated. CMD replaces bidirectional full-clip scoring with a causal teacher that evaluates each target without access to future frames or controls. The same causal teacher initializes the few-step student, establishing a consistent causal formulation across teacher training, student distillation, and inference. Beyond aligning the temporal information boundary, Prefix Scoring matches supervision to the student’s realized rollout context by evaluating each target under the cached student-generated prefix that produced it. Prefix Corruption further stabilizes training by perturbing unreliable prefixes produced early in training while preserving this target–context alignment. With a simple causal formulation, CMD naturally extends to frame-wise and chunk-wise generation, long video distillation, and camera-conditioned distillation. Experiments demonstrate state-of-the-art aggregate performance among autoregressive methods on both short- and long-video benchmarks, together with substantially improved adherence to time-varying camera controls.

Figure 1: Context-Matched Distillation supports long-horizon and camera-controllable autoregressive video generation from a single image. Starting from the conditioning image at s (left), our frame-wise (Frame AR) and multi-frame (Chunk AR) models autoregressively extend each scene. The upper rows show long-video rollouts sampled at the indicated times; the lower rows show camera-controlled rollouts guided by sequential translation (W/A/S/D) and look-direction commands. Frame AR incorporates updated controls after each latent frame, whereas Chunk AR updates controls at multi-frame block boundaries. 

\abscontent

1 Introduction

Autoregressive (AR) video models generate visual observations sequentially, making them a natural fit for interactive generative world models [48, 15, 32, 56]. To support practical interaction, these models must sustain long-horizon rollouts, accept frequent control updates, and faithfully execute time-varying control signals such as camera poses. Every generated frame or block is reused as context for the next prediction, so even small errors can compound into long-term drift in appearance, geometry, and motion.

Few-step distillation reduces the cost of each autoregressive transition by compressing multi-step denoising into a small number of student evaluations
[47, 46, 48].
Methods such as Self-Forcing further roll out the student during training, exposing it to its own generated histories and reducing exposure bias
[15, 11].
However, standard distribution matching distillation (DMD) pipelines still use a bidirectional teacher to score each completed generated clip.
A target can therefore be supervised using future frames and controls that were unavailable when the causal student produced it.
We call this the teacher–student context mismatch.

The context mismatch is especially consequential for camera control and long video generation.
In camera-conditioned distillation, a bidirectional teacher sees the complete camera trajectory, while the student must execute each pose update online
[12, 37].
Future pose changes can therefore affect the supervision assigned to the current transition.
For long rollouts, treating each local clip as self-contained both discards the preceding rollout context used to generate its early targets and allows those targets to attend to later frames inside the scoring window
[2, 42].
In both cases, the teacher scores a target under a different context from the one in which the deployed generator acts.

To align these contexts, we introduce Context-Matched Distillation (CMD), which scores each target with a causal teacher that cannot access later frames or controls.
CMD first fine-tunes a pretrained bidirectional model into a multi-step causal teacher with a Diffusion-Forcing objective over independently noised histories [3], then initializes the few-step student from those weights.
During distillation, the student is rolled out autoregressively, while the frozen teacher scores its noised targets causally.
The same causal architecture and conditioning are therefore used throughout teacher training, distillation, and student inference.
As a practical side benefit, adding a new control does not require separately adapting both a bidirectional conditional teacher and a causal student, and no additional ODE-matching or consistency-distillation stage is needed for initialization
[48, 57, 53, 10].

Causal scoring prevents future information from affecting a target score, but it does not by itself ensure that the teacher observes the actual history that produced the target.
A naive implementation conditions on preceding noised DMD targets rather than the realized student rollout.
We therefore introduce Prefix Scoring, which supplies the cached student-generated prefix used to produce each target.
Early student prefixes may contain structured rollout artifacts unlike the independent corruptions seen during teacher training. Prefix Corruption therefore perturbs the cached prefix to reduce the teacher’s sensitivity to these unreliable details while preserving causal ordering and control signals.
A block-causal attention mask computes all target-specific scores in a single teacher pass.

Figure 2: Context-Matched Distillation.
A causal student first generates an autoregressive rollout. Self-Forcing scores its complete noised rollout with a bidirectional teacher, allowing the score of a target to depend on future blocks. CMD instead uses a causal teacher: Base CMD conditions on preceding noised DMD blocks, Prefix Scoring uses the cached clean student prefix that produced the target, and Prefix Corruption perturbs this prefix to stabilize supervision when early rollouts are unreliable.

CMD’s causal target–prefix scoring applies naturally to frame-wise and chunk-wise generation, bounded-context long videos, and time-varying camera control. Across short- and long-video evaluations, CMD consistently improves quality over bidirectional scoring baselines. For camera-conditioned generation, it substantially reduces camera-pose errors, confirming the importance of matching teacher supervision to the temporal information available during online generation. Ablations further isolate the contributions of causal scoring, matched rollout prefixes, and prefix corruption.

We summarize our contributions as follows:

• 

We introduce Context-Matched Distillation, which uses a causal teacher to score every generated target without access to later frames or controls.

• 

We introduce Prefix Scoring to condition each score on the student-generated history that produced its target, together with Prefix Corruption for robust training on unreliable early rollouts.

• 

We apply the same formulation across frame-wise and chunk-wise generation, long-video rollout, and camera control, achieving strong short- and long-video quality and substantially lower camera-pose errors.

2 Related Work

Block-causal autoregressive video generation.
Video diffusion pipelines traditionally use bidirectional attention across all frames [35, 18], which promotes clip-level appearance and object consistency but scales quadratically with the number of tokens [7].
Causal and block-causal models instead generate frames or short frame blocks sequentially, reusing cached history across generation steps [9, 48, 15, 55, 11]. They support streaming and interactive generation [32, 56, 23], while multi-step causal models can also be pretrained directly with flow matching or Diffusion Forcing [3, 33, 11]. Most few-step variants are obtained by distilling a pretrained bidirectional model with a causal attention mask [48, 15, 24, 42, 58] with distribution matching distillation [47], often supplemented with adversarial objectives [8, 21, 15]. To reduce the mismatch between a bidirectional teacher and a causal student, recent works use a causal teacher to initialize the causal student through ODE matching [57] or consistency distillation [53], or to fine-tune a learned student for long-video generation [5]. Unlike these approaches, we train a causal teacher for multi-step generation and use the same teacher for student initialization and distillation with causal prefixes. Our approach reduces the teacher–student context mismatch, enabling high-quality video generation with precise control.

Long video generation.
Long-video diffusion has been extended beyond a model’s training horizon through tuning-free noise rescheduling [28], fixed-memory diagonal denoising [17], autoregressive short- and long-term memory modules [14], and Diffusion-Forcing-based infinite-length generation [4].
Recent few-step autoregressive systems use rolling windows and attention sinks for real-time long rollout [42, 24], while Infinity-RoPE extends the positional horizon and refreshes the cache for responsive action control at inference time [43].
Context-management methods compress history into a bounded token budget with importance-weighted frame packing [50], deep sinks and attention-based cache pruning [44], or sink–compressed–recent cache partitions [25].
Learned alternatives combine a local attention window with recurrent state-space memory [49], or retrieve pose- and time-indexed frames from an explicit memory bank [38].
Training-based methods further use sink tokens [6], longer-video supervision and hierarchical denoising [2], or a long-context teacher with slow–fast memory [5].
These methods improve the student’s memory or extend the teacher’s temporal reach, but distillation with a bidirectional teacher still requires scoring bounded subclips of a much longer causal rollout.
We instead query a causal teacher under exactly the bounded context used by the student, so every frame is supervised under the context that generated it without increasing the teacher’s attention window.

Camera-controlled video generation.
Camera motion is commonly injected into video generators through spatially aligned ray maps or dedicated pose adapters, enabling trajectory-conditioned generation and separating viewpoint changes from object motion [12, 39, 37, 29, 30].
This line of work has been extended to iterative scene exploration over wider viewpoint ranges and more geometrically consistent novel views [13, 39].
Geometry-aware alternatives transform or modulate attention using relative camera poses [26, 19, 20], or encode positions directly in ray space [45].
Recent causal world models extend autoregressive generation to long-horizon rollout with discrete actions and interactive 6-DoF camera control using these representations [31, 56, 32, 52, 10].
Few-step distillation further reduces rollout latency for real-time camera-controlled novel-view generation [40, 32, 52].
However, a bidirectional teacher that receives the complete camera trajectory can use future poses when scoring the current frame, whereas an online autoregressive student only observes past and current controls.
We use a frame-relative ray map and score each camera update with a causal teacher under the generated prefix, eliminating this future-camera lookahead during distillation.

3 Preliminaries

Distribution Matching Distillation. Distribution matching distillation [47, 46] distills a few-step student generator from a multi-step teacher by matching the student distribution to the teacher distribution . DMD uses the difference between their score functions, and , as a proxy for the distribution mismatch and minimizes this. The resulting gradient is written as

(1)

For readability, we suppress the diffusion timestep and write for the score evaluated on a noised version of , i.e. where , and timestep is .

While DMD was introduced for distillation of few-step text-to-image models [47, 46, 1], it has recently been applied to distilling text-to-video and image-to-video models, including causal and frame-wise generators [48, 51, 15, 57, 53, 54, 8]. In these applications, the distillation strategy largely follows the T2I setting: the student generates a full sample, and the teacher provides a score over the corresponding full video clip.

Autoregressive Video Generation. We represent an image-to-video latent sequence as , where is the fixed input-image latent and are the generated video latents. The sequence represents per-frame conditioning signals such as camera motion or user controls for the generated frames. Standard video diffusion models (Diffusion Transformers) generate a full video clip jointly, typically using bidirectional temporal attention over all frames. This allows each latent frame to depend on both past and future frames during denoising, but the cost of full attention grows quadratically with the number of frames. As a result, extending such models to long or interactive generation is expensive. Autoregressive video generation instead decomposes the video distribution over time [9, 48, 15]. In a chunk-wise autoregressive model, we partition the generated latents and their controls into aligned blocks and , each containing frames. The model factorizes as

(2)

Causality is enforced across generated blocks, while frames within the current block may interact bidirectionally. We treat the input image as part of the ordinary causal history rather than something that is permanently retained beyond local attention windows.
Setting recovers frame-wise autoregressive generation [53, 8], whose factorization is

(3)

Diffusion Forcing. Diffusion Forcing [3] trains causal sequence models by independently corrupting frames and denoising each target under a noisy history. By varying the corruption applied to past frames, it exposes the model to imperfect context during training rather than conditioning only on clean data, improving robustness when generated frames are reused as context at inference time. We use this objective to pre-train the causal teacher used for Context-Matched Distillation.

4 Methodology

We introduce Context-Matched Distillation (CMD) to distill a multi-step causal teacher into a few-step causal student for autoregressive video generation. We begin by fine-tuning a pretrained bidirectional teacher into multi-step causal and block-causal variants. The student models are initialized from the teacher weights and trained by scoring their generated samples against the teacher distribution. Specifically, the student is rolled out on-policy to produce generated targets, while the causal teacher scores each target under the same temporal information boundary as the student, without access to future frames or controls. Building on base CMD, Prefix Scoring replaces the noised scoring history for the current frame with the cached student-generated prefix used to generate it.
A block-causal attention mask pairs every target with its corresponding prefix while allowing all targets to be scored in a single teacher pass. We further introduce Prefix Corruption to reduce sensitivity to artifacts in a weak student’s generated prefixes. Section 4.1 motivates CMD by identifying the information-set mismatch between bidirectional teacher scoring and causal student generation. Section 4.2 presents the core distillation method, including causal-teacher pre-training, Prefix Scoring, efficient block-causal scoring, and Prefix Corruption. Finally, section 4.3 and section 4.4 present natural extensions of CMD to long-video generation and camera control, respectively, with minimal changes to its formulation.

4.1 Motivation

Conditioning mismatch in causal training. We discuss the formulation for frame-wise causal video generation pipelines in eq. 3. This factorization makes direct DMD supervision from a bidirectional teacher problematic [57], where . The fake sample in eq. 1 is , but each frame , , is generated using only the causal information set . In contrast, a bidirectional teacher provides by scoring the full video in its native full-clip denoising setting. Its supervision for frame can therefore depend on information that is unavailable when the causal student generates that frame, including future latents and future conditioning . Thus, although the DMD update is applied to a student sample, the teacher score is computed under a different information set from the one used by the student at inference time.

Context-Matched Distillation (CMD) addresses this conditioning mismatch by replacing the bidirectional full-clip teacher score with a score from a causal teacher queried under the same information set available to the student. For , we use the causal information set , where corresponds to past frames, including image conditioning . The teacher score used in DMD for frame can be written as 
rather than as the -th component of a bidirectional full-clip score (see fig. 2). The student score is evaluated under the same causal information boundary, yielding a frame-level score difference
.
This preserves the standard DMD score-difference update form while aligning the teacher supervision with the causal information available during autoregressive inference. The resulting information-set alignment and efficient joint scoring procedure are summarized in fig. 3.

Figure 3: Efficient Prefix Scoring. The causal student first generates an on-policy rollout and caches the clean prefix used to produce each block. Corrupted prefixes and their corresponding noised DMD targets are then packed under a block-causal attention mask, allowing all targets to be scored in parallel. Finally, the real and fake scores are evaluated under the same prefix–target context to form the context-matched DMD update.

4.2 Context-Matched Distillation

Pre-training Causal Teacher.
As discussed in section 4.1, CMD requires teacher scoring to follow the same causal information boundary as student generation. Towards this, we pre-train a causal multi-step teacher with a standard flow-matching objective to perform generation without bidirectional context.
Given a real video latent sequence with , we keep the input image clean and independently corrupt each input frame with noise at random timesteps. A block mask restricts target frame to , where denotes preceding frames corrupted independently. The teacher is trained with the standard flow-matching loss

(4)

where . Following Diffusion Forcing [3], this improves robustness to noisy context from the past. The learned flow field can be used to score the causal student pipeline as , following [47].

Causal Student Distillation. We use the pretrained causal teacher for initializing the few-step student to carry forward the teacher’s causal inductive bias. Notably, our formulation does not require expensive student initialization through ODE [58, 48] or Consistency Distillation [57, 53] based pre-training. During distillation, the student is rolled out exactly as it would be during inference, following backward-simulation [46] in Self-Forcing [15]. Specifically, we set and use the student to generate . For , latent frames are generated with velocities , where corresponds to past-frame latents detached from gradient computation.
For base CMD, each student-generated target is noised to obtain . These noised targets are arranged temporally and scored with the causal teacher and fake-score network using a block-causal attention mask. The scoring context for target is

where contains the preceding noised DMD targets. The causal attention mask prevents the score for from accessing future frames or controls. Base CMD provides causal teacher supervision matched to the temporal information boundary of student generation.

Prefix Scoring.
Although base CMD matches the temporal information boundary, its scoring history consists of preceding noised DMD targets rather than the realized history under which the student generated the current target. In practice, we instantiate CMD with Prefix Scoring. During the on-policy student rollout, we cache the generated history and controls that condition each target. For DMD supervision, frame generated by the student is noised at timestep with to obtain . The frozen causal teacher and the fake-score network evaluate this noised target using the cached causal history , yielding and , respectively, for the update in eq. 1. Rather than using previous noised DMD targets as causal context, Prefix Scoring therefore matches the scoring context to the context under which the target was generated.
Naively, Prefix Scoring would require a separate teacher pass for each frame. Instead, the teacher receives the input image and student-generated frames as conditioning memory, and uses a block-causal mask which lets each scoring query attend only to its causal prefix and controls , allowing all generated frames to be scored jointly in one pass, akin to Teacher Forcing [9].

Prefix Corruption. Autoregressive video methods have addressed rollout error through self-generated histories [15], self-resampling that simulates imperfect histories [11], anti-drifting context construction [50], and gradients through historical context representations [58]. Complementary to these, a naive teacher scoring on the exact generated prefix is not ideal, as the student can drift heavily right after initialization in few-step settings. This can dominate training when the causal teacher score is conditioned on heavily drifted, out-of-distribution artifacts. To prevent student drift from dominating score quality, we introduce prefix corruption, where the conditioning prefix is corrupted with Gaussian noise that reduces teacher focus on common drift artifacts such as color and texture. Specifically, we corrupt the prefix as:

(5)

Here is a corruption hyperparameter and is set to a small value by default. The Prefix Scoring DMD objective can now be written as:

(6)

Separate from noising in naive causal DMD, Prefix Corruption is a controlled relaxation of Prefix Scoring to stabilize distillation, and perturbs the on-policy prefix rather than the base CMD scoring sequence. The prefix is additionally corrupted using a dedicated noise level, independent of randomized DMD timesteps for scoring. This controls reliability of the generated context for the teacher without changing the target-side DMD.

4.3 Long-Video Prefixes

Figure 4: Prefix Scoring for long-video distillation. A fixed-capacity bidirectional teacher treats each local clip as self-contained, omitting earlier rollout context while allowing access to future blocks. Prefix Scoring instead evaluates all targets in one causal teacher pass under their corresponding bounded rollout prefixes.

Causal long-video models manage their history through rolling windows and sink frames [42, 24, 43], context packing or cache compression [50, 44, 25], or learned long-term memory [49, 38]. We use a local attention window for both the student and the causal teacher, as illustrated in fig. 4. Let denote the number of preceding frames retained by the student. For , the student generates frame under the bounded causal context in . Under the same memory budget, a bidirectional teacher instead scores a bounded clip such as as a self-contained sample. This omits the pre-clip context that influenced the clip’s earliest frames during the student rollout and allows those frames to attend to later frames in the clip. The resulting score therefore does not match the student’s causal factorization.

In contrast, CMD replaces the bidirectional local-clip score with a causal teacher that prevents each target from accessing later frames in the scoring window. With Prefix Scoring, the causal teacher additionally evaluates each target under exactly the same bounded context as the student, i.e. . In practice, we train -frame videos using contiguous rollouts of generated frames, with included only in the first rollout. After each rollout, its generated frames are cached as context for the next. Consequently, the context of a frame near a rollout boundary may include frames from the preceding rollout. These cached frames provide the local attention context and, when applicable, the sink tokens for both the student and the causal teacher.

For long-video training, we replace the fixed prefix-corruption level in eq. 5 with a frame-dependent schedule over the full rollout. Early frames, before substantial drift has accumulated, receive weak prefix corruption, whereas later frames receive stronger corruption as rollout error grows. Applying the score-distillation loss to every generated frame across the rollouts therefore supervises the full causal trajectory while requiring attention only over a bounded prefix for each target. Longer target videos add rollout chunks to be scored without expanding the teacher’s attention window, allowing the same local causal formulation to scale to arbitrarily long videos.

4.4 Camera-Conditioned Causal Distillation

Existing video models encode pose sequences through ray maps, camera adapters, or pose-aware attention [12, 37, 20, 13], and recent autoregressive systems support interactive camera updates during streaming generation [40, 32]. Prior work has used cameras relative to the previous frame [32], while relative camera representations have shown improvements in novel-view synthesis quality over absolute alternatives [20]. We adopt this parameterization of frame-relative cameras for training our teacher and student pipelines, embedding the camera as the motion of the current view relative to a generated view. Let denote the camera-to-world pose for frame and the corresponding intrinsics. We convert the trajectory into frame-relative increments

(7)

and use these increments, rather than the absolute pose sequence, as the camera condition. For each latent-frame location , we unproject a ray using and transform it by to obtain a spatial ray embedding

(8)

where is implemented as a ray map [32]. This representation is spatially aligned with the video tokens and preserves the local geometry of the commanded camera motion, while removing dependence on an arbitrary global coordinate frame. The key distinction is how the camera representation is used by the teacher score. In full-clip distillation, even a relative camera trajectory can be provided to a bidirectional teacher as a complete sequence. The score for frame may then depend on future camera updates , allowing the
teacher to explain the current frame using motion that is unavailable to the autoregressive student at inference time. This creates the same information-set mismatch as future-frame lookahead: the student must choose from , while the teacher score is computed with access to the whole trajectory.
CMD prevents this mismatch by restricting the teacher score for frame to visual context and camera embeddings available up to that frame. With Prefix Scoring, this visual context is the cached generated prefix under which the target was produced.

5 Experiments

Table 1: Short-video comparison of CMD and autoregressive baselines on VBench-I2V. Q: Quality Score; IS/IB: I2V Subject/Background Consistency; CM: Camera Motion; SC/BC: Subject/Background Consistency; TF: Temporal Flickering; MS: Motion Smoothness; AQ/IQ: Aesthetic/Imaging Quality; DD: Dynamic Degree.

Method

Chunk
Size

Aggregate
I2V
Consistency
Quality

Total

I2V

Q

IS

IB

CM

SC

BC

TF

MS

AQ

IQ

DD

CausVid [48]

3

80.77

84.06

77.48

89.05

90.99

15.57

93.45

95.35

98.14

98.67

63.13

70.43

20.57

Self-Forcing [15]

3

85.69

92.02

79.35

95.74

96.76

26.13

96.82

96.80

98.04

99.11

62.29

72.31

24.15

LongLive [42]

3

84.93

91.22

78.64

95.03

96.22

24.90

96.72

96.97

98.43

99.24

62.76

72.04

14.80

Rolling Forcing [24]

3

85.54

92.61

78.46

96.24

97.28

25.69

96.37

96.89

98.04

98.90

62.36

72.01

17.07

Context Forcing [5]

3

85.41

90.66

80.16

94.47

95.63

27.76

94.75

95.90

97.80

98.74

62.79

72.11

42.28

LingBot-World [32]

3

86.86

93.76

79.96

96.00

98.04

42.54

91.96

95.16

96.06

97.37

62.77

68.96

64.39

Causal Forcing [57]

1

87.63

93.70

81.56

96.57

98.04

34.47

91.22

93.50

94.74

98.15

59.42

70.13

87.15

Causal Forcing++ [53]

1

87.35

95.36

79.35

98.63

99.35

27.34

96.35

97.41

98.70

99.19

62.82

71.48

23.66

Ours

1

88.46

96.44

80.47

98.12

98.79

63.64

95.77

97.36

97.13

98.14

60.62

71.91

48.21

Ours

4

88.47

96.54

80.40

97.64

98.44

76.12

94.77

96.72

96.71

97.99

60.82

71.51

52.93

5.1 Training

We fine-tune Cosmos-Predict2.5-2B [27] for autoregressive I2V generation with generated and curated videos [35] for non-camera controlled video generation and with DL3DV [22] for camera-controlled video generation. We generate video captions for all generated videos with Qwen3-VL-8B-Instruct [41]. This data combination contains diverse camera motion coupled with dynamic environments involving people and commonplace objects. We first train the teacher model on generated data as a causal multi-step video generator and then distill it into a few-step generator. For non-camera controlled video generation, the chunk-1/chunk-4 teachers are both trained for 8K iterations. Short-video chunk-1/chunk-4 students are trained for 3.2K/5.1K iterations while long-video students are fine-tuned from short-video checkpoints for an additional 0.8K/0.9K iterations respectively. For camera-controlled I2V, we train the chunk-1/chunk-4 teachers and students for 11K/8K and 3K/0.9K iterations, respectively.

Baselines: Autoregressive I2V.
Among native few-step block-causal I2V systems, we compare with LingBot-World [32], which trains a bidirectional teacher model as a Mixture-of-Experts-style DiT and distills it into a causal student with DMD for low-latency interactive generation. In addition to native I2V, we compare with frame-causal and block-causal T2V pipelines adapted to I2V by replacing the first generated frame with the input image: (i) CausVid [48] trains a causal student model from a bidirectional teacher with DMD, where the student is initialized by ODE matching against the teacher’s trajectory. (ii) Self-Forcing [15] builds upon CausVid’s [48] distillation recipe, but trains on samples rolled out by the student itself rather than only clean video data, reducing exposure bias from generated-history conditioning at inference time. (iii) LongLive [42] targets real-time long-video generation using a bounded rolling context. (iv) Rolling Forcing [24] performs autoregressive generation with a rolling denoising window, where multiple future frames are kept at different noise levels and shifted forward as frames are emitted; it also uses attention sinks to preserve early-frame context during long rollouts. (v) Context Forcing [5] uses long-context teacher supervision to improve consistency across autoregressive rollouts. (vi) Causal Forcing [57] modifies the ODE-matching initialization in CausVid [48] by matching the student against a causal teacher instead of the original bidirectional Wan [35] teacher. (vii) Causal Forcing++ [53] keeps the causal teacher from Causal Forcing [57] but replaces ODE-matching initialization with causal consistency distillation, after which the student is further optimized with DMD.

Table 2: Long-video comparison of CMD and autoregressive baselines on SANA-WM [56]. Q/S: Quality/Semantic Scores; SC/BC: Subject/Background Consistency; TF: Temporal Flickering; MS: Motion Smoothness; DD: Dynamic Degree; AQ/IQ: Aesthetic/Imaging Quality.

Method

Chunk
Size

Aggregate
Consistency
Quality

Q

S

Total

SC

BC

TF

MS

DD

AQ

IQ

LingBot-World [32]

3

80.23

24.20

69.03

91.00

93.82

97.35

98.11

51.25

60.20

67.37

LongLive [42]

3

80.04

23.86

68.80

95.43

95.08

98.38

99.16

8.75

61.20

73.07

Rolling Forcing [24]

3

80.04

23.66

68.77

96.09

95.41

98.60

99.13

3.75

61.63

73.48

Context Forcing [5]

3

80.72

23.38

69.25

91.72

93.45

98.06

98.95

32.50

62.03

72.92

Ours

1

81.39

24.51

70.02

88.59

91.93

96.24

98.13

67.50

60.70

74.58

Ours

4

80.84

24.50

69.57

86.97

91.34

95.83

97.84

77.50

60.50

71.02

Table 3: Camera-controlled video generation on the Simple and Hard SANA-WM [56] benchmark splits.

Method

Chunk Size

Split
SANA-WM [56] VBench Scores
Camera Error

Quality 

Semantic 

Total 

Rot. 

Trans. 

CamMC 

HY-WorldPlay [31]

4

Simple

0.8178

0.1786

0.6900

2.5840

0.1639

0.1866

Hard

0.8215

0.1740

0.6920

6.5991

0.2483

0.3225

LingBot-World [32]

3

Simple

0.8173

0.1841

0.6906

4.7122

0.1562

0.2194

Hard

0.8144

0.1792

0.6873

6.2291

0.1774

0.2618

SANA-WM [56]

3

Simple

0.8355

0.1061

0.6896

1.2104

0.1316

0.1416

Hard

0.8263

0.1073

0.6825

2.0453

0.1822

0.1982

minWM [52]

4

Simple

0.8102

0.1826

0.6847

2.8637

0.2174

0.2388

Hard

0.8167

0.1775

0.6889

6.9128

0.2811

0.3538

Ours

1

Simple

0.8260

0.2413

0.7091

1.3675

0.1195

0.1312

Hard

0.8184

0.2423

0.7032

2.5607

0.1447

0.1698

Ours

4

Simple

0.8195

0.1810

0.6918

0.8601

0.0981

0.1027

Hard

0.8139

0.1751

0.6862

1.2718

0.1109

0.1196

Baselines: Action Control I2V.
Relevant action-controllable I2V world models include the following. (i) HY-WorldPlay [31] uses reconstituted context memory for long-horizon streaming generation together with PRoPE camera conditioning and discrete action embeddings. (ii) LingBot-World [32] injects camera ray maps and discrete actions through adaptive normalization, then distills the causal model for real-time rollout. (iii) SANA-WM [56] combines latent-frame UCPE with raw-frame camera ray maps for 6-DoF camera control. (iv) minWM [52] provides a full-stack recipe for adding PRoPE camera conditioning to video backbones and converting them into few-step autoregressive world models.

5.2 Quantitative Evaluation

Video Quality.
For short-horizon I2V generation, we evaluate CMD with chunk sizes one and four on the VBench-I2V benchmark [16], including five fixed seeds per image–prompt pair for every method. We report the official normalized Total, I2V, and Quality scores together with the raw component scores in table 1. Our chunk-4 model achieves the best Total (88.47), I2V (96.54), and Camera Motion (76.12) scores, improving over the strongest baseline by 0.84, 1.18, and 33.58 percentage points, respectively. Our chunk-1 model ranks second on all three metrics, with scores of 88.46, 96.44, and 63.64.

Long-Video Quality.
For long-horizon evaluation, we use the 80 images and original prompts from the Simple split of the SANA-WM benchmark [56]. Each method is evaluated over 501 frames at 16 fps (30 seconds) and resolution without camera input. We report VBench scores [16], as in SANA-WM [56], in table 2. Our chunk-1 model achieves the best score on every aggregate metric in the main comparison. Relative to the strongest external baseline in each column, it improves Q by 0.67 percentage points over Context Forcing [5], S by 0.31 points over LingBot-World [32], and Total by 0.77 points over Context Forcing [5]. Our chunk-4 model ranks second on every aggregate metric, with Q, S, and Total scores of 80.84, 24.50, and 69.57, respectively, and has the highest raw Dynamic Degree (0.7750). LongLive [42] and Rolling Forcing [24] attain higher temporal-consistency scores but substantially lower Dynamic Degree (0.0875 and 0.0375, respectively).

Camera-Controlled Evaluation.
For camera-controlled evaluation, we use the Simple and Hard splits of the SANA-WM benchmark [56]. Each method generates eight 125-frame windows per scene, yielding 640 videos per split. Following SANA-WM [56], we report Quality, Semantic, and Total scores together with rotation, translation, and Camera Matrix Consistency (CamMC) errors in table 3. To compute the camera errors, we stitch the windows, recover camera poses with [36], and align the recovered trajectory to the target using Umeyama [34]. Rotation error is the mean angular difference in degrees, translation error is the mean Euclidean distance, and CamMC is the mean Frobenius distance between the aligned camera matrices; all three are lower-is-better. Our chunk-1 model achieves the best Semantic and Total scores on both splits, while our chunk-4 model achieves the lowest rotation, translation, and CamMC errors on both splits.

Table 4: Ablations over short-video generation with CMD variants on VBench-I2V. PS denotes Prefix Scoring and PCk Prefix Corruption with ; Full CMD uses PS with PC256. Q: Quality Score; IS/IB: I2V Subject/Background Consistency; CM: Camera Motion; SC/BC: Subject/Background Consistency; TF: Temporal Flickering; MS: Motion Smoothness; AQ/IQ: Aesthetic/Imaging Quality; DD: Dynamic Degree.

Method
Aggregate
I2V
Consistency
Quality

Total

I2V

Q

IS

IB

CM

SC

BC

TF

MS

AQ

IQ

DD

Bidir. teacher

82.65

90.81

74.48

93.90

95.74

36.23

85.10

90.28

94.86

96.16

53.80

57.37

82.85

Base CMD

88.33

96.40

80.25

98.25

98.84

60.66

96.08

97.55

97.58

98.36

61.02

71.89

42.36

+ PS

87.25

94.83

79.67

97.20

97.69

55.49

92.47

94.33

94.79

97.07

58.53

70.83

69.02

+ PS + PC128

88.04

96.18

79.90

98.20

98.86

56.15

95.53

97.38

97.24

98.18

60.49

71.91

42.44

+ PS + PC512

88.06

96.23

79.89

98.31

98.88

55.75

96.28

97.67

97.62

98.37

61.06

71.98

37.15

Full CMD (PC256)

88.46

96.44

80.47

98.12

98.79

63.64

95.77

97.36

97.13

98.14

60.62

71.91

48.21

5.3 Ablative experiments

Short video generation. We report ablations over short-video generation in table 4. We compare (i) Bidir. teacher: the bidirectional teacher jointly scores the complete generated clip and can attend to both past and future frames; and (ii) Base CMD: the teacher is causal, but previous noisy DMD frames provide its context instead of a separately cached prefix. We then ablate prefix corruption using the scheduler timestep , which determines : (iii) Base CMD + Prefix Scoring (): the generated prefixes remain clean; (iv) Full CMD w. : weaker noise is added to the prefixes; (v) Full CMD w. : our default prefix-noise level is used; and (vi) Full CMD w. : the prefixes receive stronger corruption. These four configurations otherwise use the same Prefix Scoring setup. Moving from bidirectional scoring to base CMD raises Total by 5.68 points and places base CMD second on all aggregate metrics. Compared with base CMD, our default prefix-conditioned model further improves Total by 0.13 points, Camera Motion by 2.98 points, and Dynamic Degree by 5.85 points. Clean prefixes do not automatically improve aggregate performance over Base CMD; prefix corruption is important, with giving the best aggregate result and Camera Motion score, while weaker or stronger corruption gives lower Camera Motion scores.

Table 5: Ablations over long-video generation with CMD variants on SANA-WM [56] benchmark. Full CMD includes Prefix Scoring and Prefix Corruption. Q/S: Quality/Semantic Scores; SC/BC: Subject/Background Consistency; TF: Temporal Flickering; MS: Motion Smoothness; DD: Dynamic Degree; AQ/IQ: Aesthetic/Imaging Quality.

Method
Aggregate
Consistency
Quality

Q

S

Total

SC

BC

TF

MS

DD

AQ

IQ

Bidir. teacher

76.49

23.93

65.98

83.66

89.77

95.32

96.59

67.50

54.45

65.44

Base CMD

80.75

23.86

69.37

87.44

91.44

96.16

98.06

70.00

59.12

73.20

Full CMD

81.39

24.51

70.02

88.59

91.93

96.24

98.13

67.50

60.70

74.58

Table 6: Camera-control ablations on the SANA-WM benchmark [56] over camera representation and CMD variants. All variants use chunk size 1. Full CMD includes Prefix Scoring and Prefix Corruption.

Method

Camera
Split
SANA-WM [56] VBench Scores 
Camera error 

Quality
Semantic
Total
Rot.
Trans.
CamMC

Bidir. teacher

PRoPE
Simple
0.6794
0.1133
0.5662
11.2383
0.5413
0.6726

Hard
0.6779
0.1142
0.5651
12.4233
0.5481
0.6885

Full CMD

Simple
0.7775
0.1736
0.6567
1.2546
0.4043
0.4083

Hard
0.7750
0.1691
0.6538
3.2393
0.4311
0.4543

Bidir. teacher

Ray map
Simple
0.7998
0.1878
0.6774
5.7306
0.2858
0.3329

Hard
0.7936
0.1818
0.6712
7.1274
0.3027
0.3733

Base CMD

Simple
0.8214
0.1036
0.6778
1.7265
0.1264
0.1416

Hard
0.8128
0.1054
0.6713
2.1606
0.1257
0.1468

Full CMD

Simple
0.8260
0.2413
0.7091
1.3675
0.1195
0.1312

Hard
0.8184
0.2423
0.7032
2.5607
0.1447
0.1698

Long video generation. We report ablations over long video generation in table 5. We compare (i) Bidir. teacher: the score teacher processes each clip with bidirectional attention and no preceding rollout context; (ii) Base CMD: the teacher is causal within each clip, but receives no cross-clip prefix; and (iii) Full CMD: the causal teacher receives the same bounded cross-clip prefix as the student. All three causal students are evaluated using the same seed and 80 SANA-WM [56] prompts. Here, the prefix carries the student’s bounded history across clip boundaries. Base CMD improves Total by 3.39 points over bidirectional scoring. Compared with Bidir. teacher, Full CMD improves Total by 4.04 points and Semantic by 0.58 points.

Figure 5: Qualitative comparisons for long-video (top) and short-video (bottom) generation.

Figure 6: LLM preferences for our chunk-1 and chunk-4 models over 100 comparisons per model-baseline pair.

Camera control. We report chunk-1 camera-control ablations in table 6. We compare (i) Bidir. teacher + PRoPE: the teacher jointly scores the generated frames with bidirectional attention and PRoPE conditioning [20]; (ii) Full CMD + PRoPE: block-causal scoring is used with the same PRoPE conditioning; (iii) Bidir. teacher + Ray map: the bidirectional teacher instead uses ray map conditioning; (iv) Base CMD + Ray map: the teacher is causal but receives previous noisy DMD frames rather than a separately cached prefix; and (v) Full CMD + Ray map: the causal teacher uses ray map conditioning and the cached student prefix. All variants use the same prompts and evaluation protocol. With PRoPE, our causal formulation improves every metric over bidirectional scoring. With ray map, Base CMD improves Quality by 0.0216/0.0192 over bidirectional scoring on Simple/Hard and reduces all camera errors. Adding the cached prefix then improves Total by a further 0.0313/0.0319, driven by Semantic gains of 0.1377/0.1369. It also further reduces all camera errors on the Simple split for the benchmark, while Base CMD retains lower camera errors on Hard. Compared with PRoPE, the full ray map model improves all SANA-WM VBench [56] metrics, translation, and CamMC on both splits.

5.4 Qualitative Analysis

Visual comparisons.
Figure 5 compares representative rollouts across both evaluation settings. In the long-video example, our model preserves the first-person vehicle structure while progressing through distinct indoor spaces over 32 seconds. The baselines show either less scene progression or larger discontinuities in the vehicle and surrounding structure. In the short-video example, our model keeps the horse coherent while moving it forward along the rooftop towards the dome. In contrast, the baselines either lose the horse, change its placement abruptly, or remain close to the initial view. These examples complement the aggregate evaluations by showing that our model maintains visual continuity while producing clear scene progression over both short and extended horizons.

Automated pairwise preference.
We use a blind, forced-choice LLM judge as our primary perceptual evaluation as it explicitly evaluates whether the requested action and camera motion are fulfilled. We use Gemini 3.1 Pro Preview with temperature and high reasoning effort. We sample 100 VBench-I2V examples with a fixed seed and compare our chunk-1 and chunk-4 models with every baseline using the same prompt and reference image. Within each pair, videos are randomly assigned to A/B without revealing method names. With six baselines, this yields 100 comparisons per model–baseline pair, or 1,200 pairwise trials over the same 100 prompts. Gemini reports criterion-level preferences for prompt subject and scene, requested action fulfillment, camera-motion compliance, reference-image fidelity, motion integrity, and visual quality and composition. A single overall winner is then chosen from the batch, based on a hierarchical rule: required-motion fulfillment first, then reference fidelity and motion integrity and finally visual quality/composition. As shown in fig. 6, our chunk-1 model wins 75%, 64%, 62%, 66%, 85%, and 60% of comparisons against Causal Forcing [57], LingBot-World [32], Causal Forcing++ [53], Self-Forcing [15], Context Forcing [5], and SANA-WM [56], respectively. Our chunk-4 model wins 82%, 76%, 64%, 72%, 88%, and 74% against the same baselines.

6 Conclusion

We present Context-Matched Distillation (CMD), a causal DMD framework for few-step autoregressive video generation. CMD addresses the teacher–student information-set mismatch by ensuring that teacher scores cannot depend on future frames or controls. We use the same causal model family for teacher training, student initialization, and inference, avoiding a separate task-specific bidirectional teacher. Within CMD, Prefix Scoring evaluates each target using the student-generated history and controls available when that target was produced. Prefix Corruption makes this supervision more robust when early student rollouts contain severe artifacts. The same causal formulation supports long-video distillation and camera-controlled generation with frame-relative conditioning. Across short and long rollouts, CMD improves generation quality over bidirectional scoring and substantially reduces camera-pose errors. Our ablations isolate the benefits of causal scoring, matched rollout context, and prefix corruption. Overall, CMD shows that matching supervision to the student’s causal information state provides a practical foundation for efficient, long-horizon, and controllable video generation.

References

Bandyopadhyay et al. [2026]

Hmrishav Bandyopadhyay, Rahim Entezari, Jim Scott, Reshinth Adithyan, Yi-Zhe
Song, and Varun Jampani.

Sd3.5-flash: Distribution-guided distillation of generative flows.

In ECCV, 2026.

Cai et al. [2026]

Shengqu Cai, Weili Nie, Chao Liu, Julius Berner, Lvmin Zhang, Nanye Ma,
Hansheng Chen, Maneesh Agrawala, Leonidas Guibas, Gordon Wetzstein, et al.

Mode seeking meets mean seeking for fast long video generation.

arXiv preprint arXiv:2602.24289, 2026.

Chen et al. [2024]

Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ
Tedrake, and Vincent Sitzmann.

Diffusion forcing: Next-token prediction meets full-sequence
diffusion.

In NeurIPS, 2024.

Chen et al. [2025]

Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan,
Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, et al.

Skyreels-v2: Infinite-length film generative model.

arXiv preprint arXiv:2504.13074, 2025.

Chen et al. [2026]

Shuo Chen, Cong Wei, Sun Sun, Tiancheng Shen, Ping Nie, Kai Zou, Ge Zhang,
Ming-Hsuan Yang, and Wenhu Chen.

Context forcing: Consistent autoregressive video generation with long
context.

arXiv preprint arXiv:2602.06028, 2026.

Cui et al. [2026]

Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai,
Yuanhao Ban, and Cho-Jui Hsieh.

Lol: Longer than longer, scaling video generation to hour.

arXiv preprint arXiv:2601.16914, 2026.

Dao [2024]

Tri Dao.

Flashattention-2: Faster attention with better parallelism and work
partitioning.

In ICLR, 2024.

Feng et al. [2026]

Jiaqi Feng, Justin Cui, Yuanhao Ban, and Cho-Jui Hsieh.

One-forcing: Towards stable one-step autoregressive video generation.

arXiv preprint arXiv:2605.23458, 2026.

Gao et al. [2024]

Kaifeng Gao, Jiaxin Shi, Hanwang Zhang, Chunping Wang, Jun Xiao, and Long Chen.

Ca2-vdm: Efficient autoregressive video diffusion model with causal
generation and cache sharing.

arXiv preprint arXiv:2411.16375, 2024.

Gao et al. [2026]

Zelin Gao, Qiuyu Wang, Jiapeng Zhu, Jingye Chen, Zichen Liu, Qingyan Bai,
Jiahao Wang, Yufeng Yuan, Hanlin Wang, Yichong Lu, Ka Leong Cheng, Haojie
Zhang, Jian Gao, Tianrui Feng, Yuzheng Liu, Yao Yao, Yinghao Xu, Xing Zhu,
Yujun Shen, and Hao Ouyang.

Infinite worlds with versatile interactions.

arXiv preprint arXiv:2607.07534, 2026.

Guo et al. [2025]

Yuwei Guo, Ceyuan Yang, Hao He, Yang Zhao, Meng Wei, Zhenheng Yang, Weilin
Huang, and Dahua Lin.

End-to-end training for autoregressive video diffusion via
self-resampling.

arXiv preprint arXiv:2512.15702, 2025.

He et al. [2024]

Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and
Ceyuan Yang.

CameraCtrl: Enabling camera control for text-to-video generation.

arXiv preprint arXiv:2404.02101, 2024.

He et al. [2025]

Hao He, Ceyuan Yang, Shanchuan Lin, Yinghao Xu, Meng Wei, Liangke Gui, Qi Zhao,
Gordon Wetzstein, Lu Jiang, and Hongsheng Li.

Cameractrl ii: Dynamic scene exploration via camera-controlled video
diffusion models.

In ICCV, 2025.

Henschel et al. [2024]

Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram
Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi.

Streamingt2v: Consistent, dynamic, and extendable long video
generation from text.

arXiv preprint arXiv:2403.14773, 2024.

Huang et al. [2025a]

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman.

Self forcing: Bridging the train-test gap in autoregressive video
diffusion.

arXiv preprint arXiv:2506.08009, 2025a.

Huang et al. [2025b]

Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma,
Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, et al.

Vbench++: Comprehensive and versatile benchmark suite for video
generative models.

IEEE TPAMI, 2025b.

Kim et al. [2024]

Jihwan Kim, Junoh Kang, Jinyoung Choi, and Bohyung Han.

Fifo-diffusion: Generating infinite videos from text without
training.

In NeurIPS, 2024.

Kong et al. [2024a]

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng
Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al.

Hunyuanvideo: A systematic framework for large video generative
models.

arXiv preprint arXiv:2412.03603, 2024a.

Kong et al. [2024b]

Xin Kong, Shikun Liu, Xiaoyang Lyu, Marwan Taher, Xiaojuan Qi, and Andrew J
Davison.

Eschernet: A generative model for scalable view synthesis.

In CVPR, 2024b.

Li et al. [2025]

Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, and Angjoo Kanazawa.

Cameras as relative positional encoding.

In NeurIPS, 2025.

Lin et al. [2025]

Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang
Zhao, Xuefeng Xiao, and Lu Jiang.

Autoregressive adversarial post-training for real-time interactive
video generation.

In NeurIPS, 2025.

Ling et al. [2024]

Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu,
Qianyu Guo, Zixun Yu, Yawen Lu, et al.

Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d
vision.

In CVPR, 2024.

Liu et al. [2026]

Fangfu Liu, Kai He, Tianchang Shen, Tianshi Cao, Sanja Fidler, Yueqi Duan, Jun
Gao, Igor Gilitschenski, Zian Wang, and Xuanchi Ren.

Gamma-world: Generative multi-agent world modeling beyond two
players.

arXiv preprint arXiv:2605.28816, 2026.

Liu et al. [2025]

Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu.

Rolling forcing: Autoregressive long video diffusion in real time.

arXiv preprint arXiv:2509.25161, 2025.

Mao et al. [2026]

Xiaofeng Mao, Shaohao Rui, Kaining Ying, Bo Zheng, Chuanhao Li, Mingmin Chi,
and Kaipeng Zhang.

Packforcing: Short video training suffices for long video sampling
and long context inference.

arXiv preprint arXiv:2603.25730, 2026.

Miyato et al. [2024]

Takeru Miyato, Bernhard Jaeger, Max Welling, and Andreas Geiger.

Gta: A geometry-aware attention mechanism for multi-view
transformers.

In ICLR, 2024.

NVIDIA et al. [2025]

NVIDIA, Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman,
Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, et al.

World simulation with video foundation models for physical ai.

arXiv preprint arXiv:2511.00062, 2025.

Qiu et al. [2024]

Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and
Ziwei Liu.

Freenoise: Tuning-free longer video diffusion via noise rescheduling.

In ICLR, 2024.

Ren et al. [2025]

Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin
Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, and Jun Gao.

Gen3c: 3d-informed world-consistent video generation with precise
camera control.

In 2025 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pp. 6121–6132. IEEE, 2025.

Shen et al. [2026]

Tianchang Shen, Sherwin Bahmani, Kai He, Sangeetha Grama Srinivasan, Tianshi
Cao, Jiawei Ren, Ruilong Li, Zian Wang, Nicholas Sharp, Zan Gojcic, Sanja
Fidler, Jiahui Huang, Huan Ling, Jun Gao, and Xuanchi Ren.

Lyra 2.0: Explorable generative 3d worlds.

arXiv preprint arXiv:2604.13036, 2026.

Sun et al. [2025]

Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang,
Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo.

Worldplay: Towards long-term geometric consistency for real-time
interactive world modeling.

arXiv preprint arXiv:2512.14614, 2025.

Team et al. [2026]

Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong
Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, et al.

Advancing open-source world models.

arXiv preprint arXiv:2601.20540, 2026.

Teng et al. [2025]

Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai
Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al.

Magi-1: Autoregressive video generation at scale.

arXiv preprint arXiv:2505.13211, 2025.

Umeyama [1991]

Shinji Umeyama.

Least-squares estimation of transformation parameters between two
point patterns.

IEEE TPAMI, 1991.

Wan et al. [2025]

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen,
Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al.

Wan: Open and advanced large-scale video generative models.

arXiv preprint arXiv:2503.20314, 2025.

Wang et al. [2025]

Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi
Chen, Jiangmiao Pang, Chunhua Shen, and Tong He.

: Permutation-equivariant visual geometry learning.

arXiv preprint arXiv:2507.13347, 2025.

Wang et al. [2024]

Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia,
Ping Luo, and Ying Shan.

Motionctrl: A unified and flexible motion controller for video
generation.

In ACM SIGGRAPH, 2024.

Xiao et al. [2025]

Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and
Xingang Pan.

Worldmem: Long-term consistent world simulation with memory.

In NeurIPS, 2025.

Xu et al. [2024]

Dejia Xu, Weili Nie, Chao Liu, Sifei Liu, Jan Kautz, Zhangyang Wang, and Arash
Vahdat.

Camco: Camera-controllable 3d-consistent image-to-video generation.

arXiv preprint arXiv:2406.02509, 2024.

Xu et al. [2026]

Youcan Xu, Jiaxin Shi, Zhen Wang, Wensong Song, Feifei Shao, Chen Liang, Jun
Xiao, and Long Chen.

Realcam: Real-time novel-view video generation with interactive
camera control.

arXiv preprint arXiv:2605.06051, 2026.

Yang et al. [2025a]

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen
Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al.

Qwen3 technical report.

arXiv preprint arXiv:2505.09388, 2025a.

Yang et al. [2025b]

Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang,
Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al.

Longlive: Real-time interactive long video generation.

arXiv preprint arXiv:2509.22622, 2025b.

Yesiltepe et al. [2026]

Hidir Yesiltepe, Tuna Meral, Adil Kaan Akan, Kaan Oktay, and Pinar Yanardag.

Infinity-rope: Action-controllable infinite video generation emerges
from autoregressive self-rollout.

In CVPR, 2026.

Yi et al. [2025]

Jung Yi, Wooseok Jang, Paul Hyunbin Cho, Jisu Nam, Heeji Yoon, and Seungryong
Kim.

Deep forcing: Training-free long video generation with deep sink and
participative compression.

arXiv preprint arXiv:2512.05081, 2025.

Yin et al. [2026]

Minghao Yin, Jiahao Lu, Wenbo Hu, Wang Zhao, Shan Ying, and Kai Han.

Raype: Ray-space positional encoding for 3d-aware video generation.

arXiv preprint arXiv:2606.27345, 2026.

Yin et al. [2024a]

Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman,
Fredo Durand, and William T Freeman.

Improved distribution matching distillation for fast image synthesis.

In NeurIPS, 2024a.

Yin et al. [2024b]

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand,
William T Freeman, and Taesung Park.

One-step diffusion with distribution matching distillation.

In CVPR, 2024b.

Yin et al. [2025]

Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli
Shechtman, and Xun Huang.

From slow bidirectional to fast autoregressive video diffusion
models.

In CVPR, 2025.

Yu et al. [2025]

Yifei Yu, Xiaoshan Wu, Xinting Hu, Tao Hu, Yangtian Sun, Xiaoyang Lyu, Bo Wang,
Lin Ma, Yuewen Ma, Zhongrui Wang, et al.

Videossm: Autoregressive long video generation with hybrid
state-space memory.

arXiv preprint arXiv:2512.04519, 2025.

Zhang et al. [2025a]

Lvmin Zhang, Shengqu Cai, Muyang Li, Gordon Wetzstein, and Maneesh Agrawala.

Frame context packing and drift prevention in next-frame-prediction
video diffusion models.

In NeurIPS, 2025a.

Zhang et al. [2025b]

Peiyuan Zhang, Yongqi Chen, Runlong Su, Hangliang Ding, Ion Stoica, Zhengzhong
Liu, and Hao Zhang.

Fast video generation with sliding tile attention.

arXiv preprint arXiv:2502.04507, 2025b.

Zhao et al. [2026a]

Min Zhao, Hongzhou Zhu, Bokai Yan, Zihan Zhou, Yimin Chen, Wenqiang Sun, Kaiwen
Zheng, Guande He, Xiao Yang, Chongxuan Li, Fan Bao, and Jun Zhu.

minWM: A full-stack open-source framework for real-time interactive
video world models.

arXiv preprint arXiv:2605.30263, 2026a.

Zhao et al. [2026b]

Min Zhao, Hongzhou Zhu, Kaiwen Zheng, Zihan Zhou, Bokai Yan, Xinyuan Li, Xiao
Yang, Chongxuan Li, and Jun Zhu.

Causal forcing++: Scalable few-step autoregressive diffusion
distillation for real-time interactive video generation.

arXiv preprint arXiv:2605.15141, 2026b.

Zheng et al. [2026]

Kaiwen Zheng, Guande He, Min Zhao, Jintao Zhang, Huayu Chen, Jianfei Chen,
Chen-Hsuan Lin, Ming-Yu Liu, Jun Zhu, and Qianli Ma.

Causal-rcm: A unified teacher-forcing and self-forcing open recipe
for autoregressive diffusion distillation in streaming video generation and
interactive world models.

arXiv preprint arXiv:2606.25473, 2026.

Zhou et al. [2025]

Deyu Zhou, Quan Sun, Yuang Peng, Kun Yan, Runpei Dong, Duomin Wang, Zheng Ge,
Nan Duan, and Xiangyu Zhang.

Taming teacher forcing for masked autoregressive video generation.

In CVPR, 2025.

Zhu et al. [2026a]

Haoyi Zhu, Haozhe Liu, Yuyang Zhao, Tian Ye, Junsong Chen, Jincheng Yu, Tong
He, Song Han, and Enze Xie.

SANA-WM: Efficient minute-scale world modeling with hybrid linear
diffusion transformer.

arXiv preprint arXiv:2605.15178, 2026a.

Zhu et al. [2026b]

Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu.

Causal forcing: Autoregressive diffusion distillation done right for
high-quality real-time interactive video generation.

arXiv preprint arXiv:2602.02214, 2026b.

Zhuang et al. [2026]

Junhao Zhuang, Shiyi Zhang, Yuxuan Bian, Yaowei Li, Yawen Luo, Yijun Liu,
Weiyang Jin, Songchun Zhang, Xianglong He, Xuying Zhang, et al.

Self gradient forcing: Native long video extrapolation.

arXiv preprint arXiv:2607.20368, 2026.

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

