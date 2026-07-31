# HiFi-UMI: Learning Deployable Manipulation Policies from High-Fidelity UMI Data Alone

Simple AI

Learning deployable manipulation policies is bottlenecked by the scarcity of data that is both highfidelity and scalable. Real-robot teleoperation is accurate but costly to scale; robot-free UMI capture scales readily, and current practice uses the resulting data mainly for pre-training, adding a small real-robot “anchor” at post-training time. We ask whether raising the fidelity of robot-free UMI data—rather than shrinking the real-robot fraction—can remove that anchor. We present HiFi-UMI, a portable UMI data-production system that co-designs hardware and software for trajectory accuracy, inter-gripper relative pose, synchronization, and field of view: head-mounted ofline stereo-inertial SLAM, native rather than reconstructed relative pose, a shared microsecond GPIO trigger, and two wide-angle cameras per hand covering ∼200<sup>◦</sup>. It reaches 3 mm workspace-local end-efector accuracy with no external tracking infrastructure. Using this corpus, we demonstrate zero-robot post-training: a policy post-trained solely on HiFi-UMI demonstrations deploys directly on a real robot and matches in-domain teleoperation across three backbones spanning the vision-language-action (VLA) and worldaction-model (WAM) families, with success-rate diferences of −2.5, +3.1, and −0.6 percentage points on StarVLA-QwenPI, OpenPI-π , and LingBot-VA, respectively; the strongest policy reaches 85% on a precision insertion task—even though the teleoperation baseline is collected in the evaluation scene and no HiFi-UMI trajectory is. Pre-training on 4,000 hours from the same corpus lowers action error on ten unseen tasks by 41% and, on StarVLA-QwenPI, raises real-robot success by a further 18.1 percentage points. We open-source HiFi-UMI-2K—2,000 hours of microsecond-synchronized, ultra-wide-FoV demonstrations, each automatically reconstructed and validated through simulation replay—as a large-scale, high-fidelity resource for the robot-learning community.

Website: https://cloud.simpleai.tech/simple-world-lab/hifi-umi/ Dataset: https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K

## 1 Introduction

Learning deployable manipulation policies has become bottlenecked not by model capacity but by data. The dominant paradigm for acquiring deployable skills is teleoperated demonstration on real robots [1–3], which yields perfectly embodied, directly trainable trajectories but is expensive to scale: it requires the target robot, a teleoperation rig, and a skilled operator for every hour of data. Recent eforts show how steeply that cost scales: AgiBot World assembled 2,976 hours of manipulation data from 100 dual-arm humanoid robots in a purpose-built 4,000 m<sup>2</sup> facility [4], and RoboMIND assembled 305 hours across four embodiments, each needing its own teleoperation hardware built to match that arm [5]. As a result, the community has increasingly turned to cheaper, robot-free sources of manipulation data.

The Universal Manipulation Interface (UMI) [6] is the most prominent of these: a handheld, instrumented gripper that lets a human collect in-the-wild demonstrations with no robot in the loop, at a fraction of the cost of teleoperation. UMI and its derivatives have enabled large demonstration corpora [7] and even datascaling studies of imitation learning [8]. Yet current handheld capture inherits a set of fidelity deficiencies that limit how far such data can be trusted. Pose is typically recovered by online visual(-inertial) SLAM, which drifts over long horizons and fails under low texture or motion blur [6, 9]; inter-gripper relative pose

## > 98%

## HiFi-UMI Data Collection

## High-Fidelity Data Engine

![](images/f371abef46deeaee7a5d4cfa69ec6c9b44fdf4f3639e125af24efa107a98760a.jpg)  
3 mm Trajectory accuracy  
40 μs Synchronization  
Ultra-wide 6-view sensing

![](images/955c84e668acf61d70c594199c6fb18db4647a9e7ce20ede61f11a115a32fb63.jpg)

![](images/012761963ba8697107cf3c035710ce13e6cacd5e223b3b03fa49683341a360ea.jpg)  
Trajectory Reconstruction Success Rate  
> 98% Replay Success Rate

## Learning & Deployment

![](images/b3c3c1a095aad2ca6f51bd21f590f3ce9708a24cefed90c86dcca1022368cb07.jpg)  
Trajectory Reconstruction & Replay  
Direct real-robot deployment

![](images/2495e1402e8abc20d362c402ef8252acb995d60c9dbd4ad0b3882f7ec033803d.jpg)

![](images/fad2ec63f04a9f0af80ba7948577b7ae3c2021c270b271fe81e77624f2c96714.jpg)

![](images/b33083872768f9416e082bc915fdbf0744c410365a8197f8b3eb018ab4ae85cd.jpg)  
Real-robot Evaluations  
Zero-Robot Post-training (VLA)

![](images/0a73b46b76df462aaaf71d82d4e0d426a888245064a9c47360d358edc85708b6.jpg)  
Zero-Robot Post-training (WAM)

![](images/4f88ed5b9acad64f5023e884373eb62dd722c125aeda5b9276054b450f30d9d4.jpg)  
Enabling Scalable Pre-training  
Figure 1 Overview of HiFi-UMI, a robot-free framework for scalable manipulation data collection, high-fidelity data processing, and direct real-robot policy deployment. HiFi-UMI Capture combines 3 mm trajectory accuracy, microsecond-level synchronization, and ultra-wide six-view sensing. The resulting data are reconstructed, replayed, quality-checked, annotated, and curated by the HiFi-UMI data engine, which has collected more than 20,000 hours of data across over 480 scenes, released the curated 2,000-hour HiFi-UMI-2K subset, and achieved trajectoryreconstruction and replay validity rates above 98% and 98%, respectively (≈96% cumulative). The curated corpus supports pre-training and post-training of both VLA and WAM policies without any teleoperated robot data for the target task, enabling direct deployment on real robots. HiFi-UMI-only post-training performs on par with in-domain teleoperation on all three backbones, difering by at most 3.1 percentage points on the two VLA policies and by 0.6 percentage points on the WAM policy. Moreover, the held-out action prediction error decreases consistently with increasing pre-training data, following a clear power-law scaling trend (α = 0.268, R<sup>2</sup> = 0.993).

is reconstructed from cross-camera co-visibility rather than measured natively, introducing error precisely on the coordinated tasks where it matters most; sensor streams use software or wireless alignment rather than hardware triggers; and one 155<sup>◦</sup> wrist fisheye per hand leaves blind spots and weak depth cues [6, 9].

These limitations are not incidental—they are the reason that, throughout the field, robot-free data is largely confined to a pre-training role while real-robot teleoperation is assumed necessary for the post-training that grounds a policy for deployment [10, 11]. Even the most aggressive recent attempts to minimize teleoperation preserve this division. The portable VR-based systems ActiveUMI [12] and XRZero-G0 [13] reduce the realrobot fraction to a small share of a much larger robot-free corpus, but neither eliminates it. RDT2 [14] obtains zero-shot cross-embodiment transfer from robot-free data alone but mixes in real-robot data once deployment-grade performance on a specific arm is required. It has never been clear, however, whether posttraining genuinely requires real-robot data, or whether the fidelity of the robot-free data available so far has simply not been high enough.

We ask whether suficiently high-fidelity UMI data can break this division. We present HiFi-UMI, a portable data-production system designed end-to-end for fidelity, replayability, and automated curation (Fig. 1), and we use it to test a deliberately strong hypothesis—which we call zero-robot post-training: that high-fidelity robot-free data alone, used for post-training, can yield manipulation policies that deploy directly on a real robot without any teleoperated real-robot data in post-training. This reframes the question away from mixing ratios. Where prior work approaches the teleoperation baseline by shrinking the real-robot fraction to a small but non-zero anchor, we ask whether raising the fidelity of the robot-free data itself—so that its trajectories, inter-gripper pose, and timing are as trustworthy as teleoperation—can remove the need for that anchor altogether. Fig. 2 makes this fidelity tangible: our ofline reconstruction preserves a handwriting trajectory well enough to render legible millimeter-scale strokes.

![](images/4206d493c22359415ac6c2d3c4757d968463cf72214e4a2c75b2e3d9a31bc8ed.jpg)  
(a) Pen writing via HiFi-UMI

![](images/ad8a2a3c67ab103abad29a724343cac60a576b6a88cf6e9aac08580701399488.jpg)  
(b) Reconstructed 3D end-effector

![](images/a598c84bfde357b1143e222bdb6a75267d073a48789cb9c7f376e06c1534bbd9.jpg)  
(c) Time-aligned reconstruction sequence  
Figure 2 Qualitative visualization of 3D trajectory reconstruction in a handwriting task. (a) First-person observation of a handwriting demonstration captured using HiFi-UMI. (b) The reconstructed 3D end-efector trajectory, shown in a global view and a zoomed-in view of the written characters. The marked 4 mm width of the lowercase letter “e” provides a spatial reference for the scale of the reconstructed writing. (c) Time-aligned visualization at six synchronized instants $t _ { 1 } { - } t _ { 6 }$ , including the corresponding head-camera observations, pen-tip close-ups, and cumulative 3D trajectory reconstructions.

HiFi-UMI attacks each of the deficiencies above. Pose fidelity comes from head-mounted stereo ofline SLAM, which yields low-drift long-horizon trajectories, together with natively accurate inter-gripper relative pose obtained jointly with the world-frame pose of both hands. Sensing fidelity comes from microsecond-level synchronization across all sensors via a single shared GPIO hardware trigger, and from an ultra-wide field of view—two non-parallel stereo cameras covering roughly 200<sup>◦</sup> horizontally and over $2 0 0 ^ { \circ }$ vertically. Two further choices target the interaction itself and the cost of producing usable data: a full-palm glove form factor that better preserves the operator’s natural force and contact than a trigger gripper, and real-time online slicing with in-situ data-quality monitoring that catches capture anomalies during collection rather than after it. Together these turn handheld capture into a production-grade data engine that reconstructs, replays, quality-checks, annotates, and curates every demonstration automatically—trajectory reconstruction and replay validation each pass over 98%, a cumulative $9 8 \% \times 9 8 \% \approx 9 6 \%$ and it is this engine’s output, and nothing else, that we ask to carry a policy all the way to deployment. The same HiFi-UMI corpus supplies both pre-training and post-training, yielding policies that run directly on a real bimanual robot without teleoperated post-training data. These stages correspond to the three panels of Fig. 1.

The hypothesis holds across every comparison we run. On three backbones spanning both the vision-languageaction (VLA) and world-action-model (WAM) families, HiFi-UMI-only post-training matches in-domain teleoperation: the diferences are −2.5, +3.1, and −0.6 percentage points, of both signs and each within the sampling noise of our protocol. Parity holds under an asymmetry that favors the baseline—the teleoperation data is collected in the evaluation scene and no HiFi-UMI trajectory is—and the strongest policy reaches 85% on a precision insertion task. Because a claim of this kind is only as good as the evaluation behind it, every comparison runs under a benchmark frozen before evaluation begins, with test-instance construction separated from policy execution, randomized policy order, and recorded termination reasons; the six conditions receive 960 real-robot rollouts in total. Using 4,000 hours of the same corpus for pre-training then lowers action error on ten unseen tasks by 41% and, on StarVLA-QwenPI, raises real-robot success by a further 18.1 percentage points—and the structure of that transfer is itself informative, tracking coverage of interaction dynamics in the pre-training mixture rather than whether a test object has been seen before. We provide, to our knowledge, the first controlled demonstration that handheld robot-free post-training, with no real-robot data at all, matches in-domain teleoperation on the same robot, replicated across three backbones. We treat fidelity as the design principle behind that result rather than a variable we isolate through controlled degradation, and leave such an ablation—cleanly separating fidelity from sample count and scene coverage—to future work.

In summary, the primary contributions of this work are:

• A data-production system whose hardware–software co-design remedies the trajectory-accuracy, intergripper-pose, synchronization, and field-of-view deficiencies of prior handheld capture: head-mounted offline stereo SLAM and a shared GPIO trigger give 3 mm end-efector accuracy and microsecond cross-sensor alignment with no external tracking infrastructure, and an automated engine reconstructs, replays, and validates every demonstration, retaining 96% of raw captures as robot-executable data.

• Evidence that HiFi-UMI alone sufices for post-training: across three VLA and WAM backbones, UMIonly post-training matches in-domain teleoperation on the same robot (−2.5, +3.1, and −0.6 percentage points), with all gaps within sampling noise despite native sample-count diferences.

• Pre-training on the same robot-free corpus raises both the data eficiency and the ceiling of downstream post-training: on StarVLA-QwenPI, 4,000 hours cut ofline action error on ten unseen tasks by 41% and, at matched post-training data, raise real-robot success by 18.1 percentage points, matching the scratchinitialized baseline with a quarter of the task data. Transfer depends more on whether pre-training covered a task’s kind of physical interaction than on whether its objects have been seen.

• HiFi-UMI-2K, an open 2,000-hour, microsecond-synchronized, replayable, ultra-wide-FoV subset, produced by the same pipeline for deployment-grade post-training without real-robot teleoperation.

## 2 Related Work

## 2.1 Manipulation Datasets

Manipulation datasets now span a spectrum from fully grounded robot demonstrations to scalable but weakly grounded human video. At the high-fidelity end, real-robot teleoperation corpora remain the standard substrate for deployable policy learning: BridgeData V2 [2] and RH20T [15] established large-scale multitask and multimodal collection, DROID [3] emphasized in-the-wild diversity across scenes and operators, and Open X-Embodiment [16] aggregated heterogeneous robot datasets to study cross-embodiment transfer. More recent eforts such as RoboMIND [5], RoboMIND 2.0 [17], and AgiBot World [4] push this regime toward larger trajectory scale with stronger standardization, quality control, bimanual and mobile settings, and richer sensory streams. At the opposite end, egocentric human-video corpora such as Ego4D [18] and Ego-Exo4D [19] ofer scale and naturalness but lack executable robot actions, while EgoDex [20] narrows this gap with manipulation-centric video and paired 3D hand-pose annotations.

Between these extremes, UMI-style data has emerged as a distinct middle tier of robot-free yet action-grounded supervision. UMI [6] introduced portable handheld grippers for collecting low-cost, information-rich demonstrations without a robot body, enabling direct transfer to hardware-agnostic policies. FastUMI-100K [21] and the 10,000-hour corpus of RDT2 [14] show that this recipe scales to corpora rivaling the largest teleoperation eforts; we defer the corresponding devices to Sec. 2.2. This hierarchy is naturally viewed as a data pyramid [10]: web and human video provide scale, teleoperation provides embodiment-specific grounding, and UMI-style demonstrations occupy the middle by preserving actionable wrist-view geometry and relative end-efector motion without robot-specific collection. Our work targets this middle tier directly: by increasing the fidelity, synchronization, and retargetability of UMI data, we test whether robot-free handheld demonstrations can serve not only as scalable pre-training data but also as deployment-relevant supervision.

## 2.2 UMI and Portable Data-Collection Devices

The Universal Manipulation Interface [6] introduced a handheld instrumented gripper that recovers globalscale end-efector trajectories via ORB-SLAM3 and an IMU, using a single wrist-mounted 155<sup>◦</sup> fisheye camera per gripper and side mirrors for implicit depth. It is the foundation for a growing family of portable capture devices. Tab. 1 compares them along the axes that matter here: pose acquisition and accuracy, cross-sensor synchronization, sensing coverage, whether inter-gripper relative pose is measured natively or reconstructed, gripper form factor, and portability. FastUMI [7] swaps bespoke SLAM for an of-the-shelf tracker, improving robustness and scaling collection. Its successor FastUMI Pro, the capture platform used by VISTA [22], fuses an external lighthouse tracker with onboard visual-inertial SLAM, reaching millimeter-level pose at the cost of fixed infrastructure. DexCap [23] and DexUMI [24] extend capture to dexterous hands via mocap gloves and wearable exoskeletons, and DexWild [25] scales in-the-wild human-hand demonstrations for dexterous policies. ARCap [26] adds in-headset augmented-reality feedback so demonstrations remain kinematically valid, and exoskeleton systems such as AirExo [9, 27] pursue whole-arm capture without a robot. A parallel line pairs head cameras with hand capture—EgoMimic [28] and H-RDT [11]—to obtain action signals.

Despite this progress, fidelity limitations remain pervasive across these devices, and the methods that consume robot-free data still lean on paired robot demonstrations. AirExo-2 [9] explicitly attributes the shortcomings of handheld UMI-style devices to two causes: reliance on visual SLAM for pose estimation, which yields action inaccuracies, and a limited camera field of view, which hinders depth perception. The original UMI [6] likewise notes SLAM and scale-ambiguity failures, latency discrepancies between collection and inference, and constrained single-camera coverage. Crucially, methods that do leverage human or egocentric data for learning still rely on paired robot data: EgoMimic [28] co-trains with teleoperated demonstrations, and H-RDT [11] fine-tunes on robot data after human pre-training. In this line, robot-free data is not asked to ground a deployable policy on its own.

A recent line of work replaces on-device handheld SLAM with tracking infrastructure outside the gripper, whether a headset worn by the operator or base stations placed in the room. ActiveUMI [12] rigidly mounts a copy of the robot’s gripper onto a VR controller and records the operator’s head motion and egocentric attention. XRZero-G0 [13] pairs headset tracking with dual purpose-built grippers and a closed-loop qualityinspection pipeline to build large robot-free datasets. These systems improve tracking robustness and, like ours, recover the inter-gripper relative pose natively, since the headset observes both controllers together. Their remaining design choices difer from ours in ways that matter for fidelity. Pose comes from online headset tracking rather than ofline SLAM optimization; sensor streams are aligned by software spatiotempora matching rather than a hardware trigger; and coverage comes from a small number of discrete camera views rather than ultra-wide stereo optics. They reduce the real-robot fraction to a small share of a much larger robot-free corpus, but neither eliminates it. RDT2 [14] instead scales redesigned UMI hardware to over 10,000 hours and attains zero-shot cross-embodiment transfer on simple open-vocabulary tasks with no real-robot data. Only when deployment-grade performance on a specific arm is required does it mix a small amount of real-robot data into an optional post-training variant. That hardware, however, tracks the end-efector with external infrared base stations rather than onboard SLAM, so every collection site must first be instrumented. Closest to our setting, VISTA [22] post-trains a bimanual policy on curated handheld data alone and deploys it on real robots, establishing that robot-free post-training can work. Because all of its baselines are trained on that same handheld corpus, however, the comparison isolates model and curation design rather than the data source, and the question of how robot-free supervision stands against teleoperation is left open. We share this emphasis on post-collection validation, but also build fidelity into the capture device itself, so that most trajectories pass validation rather than being screened out.

Our system addresses these fidelity limitations through ofline stereo SLAM, a shared GPIO hardware trigger, and non-parallel cameras for ultra-wide coverage. No prior handheld system places robot-free post-training against teleoperation on the same robot [12–14, 22]. We instead hold the backbone, the recipe, and the deployment stack fixed and change only whether the task-specific demonstrations come from HiFi-UMI or from teleoperation on the evaluation robot.

## 2.3 Manipulation Foundation Models

Recent manipulation foundation models fall into two families of low-level control backbone, separated by whether action generation is coupled to a prediction of the future. VLA policies are purely reactive, mapping the current observation and a language instruction directly to actions. World-action models (WAMs) add a predictive component, and that coupling takes two forms. Some predict future observations only as an auxiliary training signal and discard the predictor at test time. Others generate a future at every inference step and decode the action from it, so that the quality of the imagined future directly gates control.

Within the VLA family, RT-2 [29] established the tokenized-action recipe, and OpenVLA [30] and Octo [31] scaled open generalist policies over heterogeneous robot data. Recent systems favor continuous action heads and stronger post-training recipes: π<sub>0</sub> [32] and π<sub>0.5</sub> [33] pair a pretrained VLM with a flow-matching action expert, an architecture now widely reused, while GR00T N1 [10], GR-3 [34], Gemini Robotics [35], and SmolVLA [36] trade model scale against inference cost and deployability. A separate line treats the action interface itself as the design variable, with FAST [37] on action tokenization and OpenVLA-OFT [38] on chunked decoding and fine-tuning; our own action representation inherits these choices (Sec. 5). LingBot-VLA [39], Qwen-VLA [40], and Qwen-RobotManip [41] pursue large-scale aligned pre-training and unified embodied interfaces.

Within the WAM family, early systems explored the idea before the terminology settled. GR-1 [42] and GR-2 [43] jointly predict future images and actions after video-generative pre-training, UniPi [44] casts policy learning as text-guided video generation, and VPP [45] conditions control on representations drawn from a video difusion model. Recent WAMs make the coupling explicit in the policy backbone: DreamZero [46] builds a real-time closed-loop policy on a pretrained video-difusion backbone, and WorldVLA [47] models image and action generation jointly in an autoregressive framework. LingBot-VA [48] learns frame prediction and policy execution through causal world modeling, decoding each action chunk from the future latents it has just generated.

These trends reinforce the value of scalable data. Yet robot-free sources such as human video, egocentric video, and UMI-style demonstrations are still rarely used as post-training supervision, a gap commonly attributed to limited trajectory fidelity, synchronization, retargetability, and geometric consistency. StarVLA [49] and its QwenPI-style implementations provide an open, modular ecosystem in which that attribution can be tested directly.

We therefore select three backbones that difer along three axes. StarVLA-QwenPI is a modular open implementation, so we control its initialization and can add a large-scale pre-training arm. OpenPI-π<sub>0.5</sub> is a strong publicly released checkpoint that we did not build. LingBot-VA derives actions from an imagined future rather than from the current observation alone. What makes them comparable is not their architecture but their interface: each consumes the same supervision, however diferently it tensorizes it. Changing the source of that supervision is therefore one intervention applied three times, and agreement among the three is evidence about the data rather than about any single architecture.

## 3 Data Collection and Processing Pipeline

The core hypothesis of this work—that robot-free demonstrations can, on their own, ground a deployable policy—stands or falls on the fidelity of the data. If trajectories drift over long horizons, if sensor streams are misaligned in time, or if the gripper’s view is too narrow to resolve contact, then no amount of scale recovers a deployment-grade action signal [10], and a real-robot anchor becomes unavoidable. We therefore treat data production as a system-design problem and co-design hardware and software so that fidelity is enforced at the source rather than repaired after the fact. This section describes the resulting system: a wearable capture device whose four subsystems each target a specific fidelity axis (Sec. 3.1); an explicit, two-level definition of what constitutes usable data (Sec. 3.2); a six-stage processing flywheel that turns raw captures into training-ready episodes (Sec. 3.3); and the end-to-end fidelity the system delivers (Sec. 3.4).

## 3.1 HiFi-UMI Capture Device

![](images/e02b3e5ce296ed21b135cc978ee00b2eb475808aa9c5561fe68ecf88e5130228.jpg)  
Figure 3 Overview of the HiFi-UMI capture device. A head-mounted stereo camera pair with an integrated IMU enables ofline stereo-inertial SLAM; per-hand marker cubes are localized in the same head-camera frame; each hand carries two non-parallel wide-angle fisheye cameras (top and bottom) for ultra-wide coverage; a full-palm glove gripper preserves natural contact; and a shared GPIO trigger synchronizes all sensors.

The HiFi-UMI capture device (Fig. 3) is organized around four requirements that our design treats as jointly suficient for deployment-grade robot-free data: (i) accurate and scalable pose acquisition, (ii) a manipulationoriented gripper morphology, (iii) wide-coverage multimodal sensing, and (iv) online quality control. We address each in turn, and summarize how our design compares to representative robot-free systems in Tab. 1.

## 3.1.1 Pose Acquisition and Accuracy

The core tension in UMI-style capture [6] is to reconcile trajectory accuracy, tracking robustness, deployment flexibility, and hardware cost. Existing approaches sit at diferent points of this trade-of: wrist-camera visualinertial odometry (VIO) [50], VR headset-and-controller tracking, base-station tracking, and motion capture. Base-station and motion-capture systems deliver high accuracy but require instrumented environments, which precludes scalable in-the-wild collection. VR-headset solutions inherit mature commercial tracking stacks, at the cost of higher hardware expense and greater system complexity. Wrist-camera VIO is cheaper and lighter than either, yet its view is routinely occluded by the hand or the manipulated object, leaving trajectories vulnerable to tracking failure and accumulated drift.

We instead adopt a wearable scheme built on ofline stereo-inertial SLAM [51] together with fiducial-marker localization [52]. Rather than tracking each wrist independently, a head-mounted stereo rig estimates the global camera trajectory, and each hand is localized relative to the head via a rigidly attached marker cube observed by the same head cameras. Composing the global head trajectory with the two relative handto-head poses yields globally consistent trajectories for both hands. This design is motivated by a simple observation: a head-mounted viewpoint is far more stable than a wrist-mounted one, which is routinely corrupted by nearby moving objects, self-occlusion, and rapid hand motion during manipulation [9]. In our experiments the scheme attains 3 mm end-efector accuracy—comparable to VR-controller tracking—while remaining lighter and lower-cost than VR-headset-based rigs.

The head-relative formulation is especially advantageous for bimanual manipulation. Because both marker cubes are observed in a single head-camera frame, the inter-gripper relative pose is measured natively and inherits the same accuracy as the per-gripper pose, rather than being reconstructed post hoc from crosscamera co-visibility as in prior handheld rigs. Moreover, because head motion is typically far smaller than hand motion, head-relative tracking substantially reduces accumulated drift.

## 3.1.2 Gripper Morphology

Existing UMI grippers fall broadly into trigger-based and finger-sleeve-based designs. Trigger devices emulate parallel-jaw commands directly, but give the operator weak tactile correspondence with the manipulated object. Finger-sleeve designs are mechanically simpler and preserve direct contact sensation, enabling more natural and dexterous manipulation. Many, however, adopt a narrow, elongated geometry: well suited to fine manipulation, but limiting on larger or heavier objects.

To broaden task coverage without sacrificing contact fidelity, we design an asymmetric two-finger, glove-like full-palm gripper inspired by the human hand. The two fingers correspond respectively to the thumb and the opposing four fingers, and are deliberately asymmetric in shape and width: the narrower fingertip region supports precise, small-object manipulation, while the wider proximal region provides a larger contact patch and firmer support for heavy objects. This morphology preserves the operator’s natural force distribution and contact geometry more faithfully than a trigger interface.

## 3.1.3 Cameras and Sensors

Each hand carries two non-parallel fisheye cameras, yielding about 200<sup>◦</sup> of horizontal and vertical coverage. This ultra-wide view reduces occlusion and improves observability around the gripper. Together with the stereo head cameras, the complete device integrates six cameras. It further carries IMUs on the head and both hands—for pose estimation and motion-state monitoring—and high-precision encoders on the grippers to measure opening angle. Critically, every sensor is driven by a single, unified GPIO external trigger [53], providing microsecond-level temporal synchronization across all cameras, IMUs, and encoders. This hardwarelevel synchronization replaces the software or wireless alignment used by prior systems and removes a key source of action-label noise.

## 3.1.4 Online Interaction and Quality Control

To suppress low-quality data at its source, the device performs quality monitoring during recording. It detects common failure modes—including underexposure, motion blur, excessively fast operator motion, and hands leaving the head cameras’ field of view—and issues real-time voice feedback so the operator can correct the capture in situ rather than discovering the problem in post-processing. Online temporal slicing lets operators mark task and subtask boundaries during collection, reducing later segmentation efort. Alongside online monitoring, it improves throughput and reliability.

## 3.2 Data-Quality Criteria

Data quality is enforced as a set of hard constraints on sensors, trajectories, and annotations—the technical requirements a capture must meet before it can serve as an action label—mirroring the curation and qualitycontrol protocols of large-scale robot datasets [3]. Beyond these constraints, the content-level properties of the corpus—annotation correctness, task and scene composition, and coverage of rare behaviors—are governed by the annotation, verification, and export stages of the processing pipeline (Sec. 3.3) rather than by a separate admission test.

Sensors. Camera field of view, layout, and viewing direction must satisfy the system specification; multisensor timestamps must be accurately synchronized; images must be properly exposed, free of severe motion blur, and free of unexpected occlusion; and all other sensor readings must be valid and accurate.

Table 1 Comparison of representative robot-free data-collection systems along key design and data-fidelity axes. Pose acquisition: how 6-DoF pose is estimated—on-device visual(-inertial) odometry/SLAM (VIO / VI-SLAM), headset inside-out tracking (VR inside-out), or external base-station tracking. Pos. err.: reported end-efector positional error (mm), taken from each system’s own publication and rounded; because each is measured against a diferent reference and motion profile, these values should be read as order-of-magnitude rather than compared directly. The HiFi-UMI value is measured in this work (Tab. 2). Sync.: cross-sensor time-alignment, reported as its typical latency scale and mechanism; only HiFi-UMI reaches microsecond-level alignment via a hardware GPIO trigger, whereas prior systems, where reported, use millisecond-level software timestamp alignment. Views / FoV: number of camera views and peak field-of-view coverage. Rel. pose: how the inter-gripper relative pose is obtained—measured natively when both ends are observed together in one operator-mounted frame (native), reconstructed post hoc from cross-camera co visibility (reconstructed), or diferenced from two poses tracked against external infrastructure (external). Gripper: end-efector actuation form factor (e.g. handheld trigger, finger-sleeve, or full-palm glove). Port.: portability— untethered, in-the-wild capture with no instrumented infrastructure (High) vs. dependence on fixed base stations or motion-capture rigs (Low). “–” marks a property not reported by the cited source or not applicable to single-arm capture. Together, these choices yield HiFi-UMI’s main advantages over prior systems: millimeter-level end-efector accuracy (∼3 mm) obtained from head-mounted ofline stereo-inertial SLAM without external tracking infrastructure; the tightest, microsecond-level synchronization via a GPIO hardware trigger; the widest sensing coverage at both hands; and greater ease of use—fully portable with no external base stations, and operated through a full-palm glove rather than a trigger for more natural manipulation.

<table><tr><td>System</td><td>Pose acquisition</td><td>Pos. err. (mm)</td><td>Sync.</td><td>Views FoV</td><td>Rel. pose</td><td>Gripper</td><td>Port.</td></tr><tr><td>UMI [6]</td><td>wrist VI-SLAM</td><td>~6</td><td>ms (software)</td><td>2 / 155°</td><td>reconstructed</td><td>trigger</td><td>High</td></tr><tr><td>FastUMI [7]</td><td>dedicated VI module (T265)</td><td>~10</td><td>ms (software)</td><td>1 / 155°</td><td></td><td>trigger</td><td>High</td></tr><tr><td>DAS fingers [54]</td><td>wrist VIO</td><td></td><td></td><td>2  / 150°</td><td>reconstructed</td><td>finger-sleeve</td><td>High</td></tr><tr><td>ActiveUMI [i2]</td><td>VR inside-out</td><td>~4</td><td>ms (software)</td><td>3 / -</td><td>native</td><td>trigger</td><td>High</td></tr><tr><td>TacUMI [55]</td><td>base station</td><td></td><td></td><td>1 / -</td><td>external</td><td>trigger</td><td>Low</td></tr><tr><td>RDT2 [14]</td><td>base station</td><td></td><td></td><td>2 / -</td><td>external</td><td>trigger</td><td>Low</td></tr><tr><td>FastUMI Pro [22]</td><td>base station + wrist VIO</td><td>~3</td><td></td><td>2  / 180°</td><td>external</td><td>trigger</td><td>Low</td></tr><tr><td>XRZero-G0 [13]</td><td>VR inside-out</td><td>~4</td><td></td><td>3 /-</td><td>native</td><td>trigger and finger-sleeve</td><td>High</td></tr><tr><td>HiFi-UMI (Ours)</td><td>head stereo-inertial SLAM</td><td>~3</td><td>µs (GPIO)</td><td>6 / 200°</td><td>native</td><td>full-palm glove</td><td>High</td></tr></table>

Trajectories. The coordinate frames of every 6-DoF trajectory must be consistently and explicitly defined; the reconstructed trajectories must faithfully reflect the human demonstration, meet the required precision, and be executable under robot replay; and the gripper width or opening angle must be measured accurately.

Annotations. Language annotations must be consistent with the visual content, and the temporal boundaries of subtask segments must meet the required timing accuracy.

These constraints are secured primarily by the device design and the processing pipeline. The current usabledata ratio is approximately 96%, which is itself cumulative: it is the product of the two gates applied in series by the pipeline—trajectory reconstruction and whole-body-control replay validation—each of which passes approximately 98% of the captures that reach it (Sec. 3.3).

## 3.3 HiFi-UMI Data Processing Pipeline

The HiFi-UMI pipeline (Fig. 4) comprises six stages: data collection and upload, trajectory reconstruction and automatic cleaning, simulation retargeting, AI-assisted annotation, human verification, and data analysis and export. Each stage both filters invalid data and augments valid data with structured metadata, so that quality control is distributed across the pipeline rather than deferred to a single expensive review step.

## 3.3.1 Data Collection and Upload

Four mechanisms operate at capture time. Multi-sensor synchronization is guaranteed in hardware via the shared GPIO trigger; on-device processing raises online warnings for corrupted or low-quality segments, removing invalid samples at the source; a human-in-the-loop interface lets operators mark subtask boundaries during collection, reducing downstream segmentation cost; and the device streams captures to the cloud over Wi-Fi in real time, so collection and upload proceed concurrently.

![](images/393822ab1b72ed72911b41bf116256cec6bd0edbb91cc896c630f16c106a8fba.jpg)

![](images/fe18d1656e4de81349be2fb38244bccdfe5c8a41e331aeba374db36af2edfa2d.jpg)  
Arrange the pillows on the bed.  
Data Processing Flywheel

![](images/7e4195f472e384f39f8074fce448211e641ab896340ca0cfc8ba55943909e704.jpg)

![](images/1ac8527132279a8bb603fef937f7c66816dd41590b56a194ad7fbdcb5347931a.jpg)

![](images/bfbae6a4738d1c72c0d10c8027b020c5b755ad71d4fa1907614cb6719457b734.jpg)  
Model Training

## 3. Al Annotation & Verification

![](images/15d2089c216af8244a8792bb04e7c5570dc7f35e4aa4489d4837eaff69fcd30b.jpg)

## 4. Analysis, Training & Evaluation

![](images/12f8bf22b24f3cd3430b9d6bb8e1fdd8b04f1db536d25b0a31dd6daa4b34a3fd.jpg)

![](images/d5f078170b536dc383b5595885ad585f11c9b66b8db53fee181341fd251abde6.jpg)  
Figure 4 The HiFi-UMI data flywheel. Raw captures pass through six stages—collection and upload, trajectory reconstruction and automatic cleaning, simulation retargeting, AI-assisted annotation, human verification, and analysis and export—each of which enriches the data with metadata and removes or flags invalid samples.

## 3.3.2 Trajectory Reconstruction and Automatic Cleaning

![](images/9f2c873b04549b91e8114ab9edf450df748e0215795de82439517430f6488478.jpg)  
Open the sliding screen window upward.

![](images/10f4bce344a5b78922639e27d657ad46c2eed8e80d0005a6e2cf366e9f0ce85b.jpg)

![](images/5073803ee17d2797d916f12e321d97a86e2e6c3d77773b6ff9b553c85d068ea2.jpg)

![](images/8e7cf6f0a4a25eede62f2f66759a7e96c5eccd8f8c2f36dd57d73aca7c12dab9.jpg)  
Neaten the toiletries around the sink and clean the sink.

![](images/07f21d23ed510e3a4a92fe971c5fda4b817b41da0cbd46432aed57eb9dc934f1.jpg)  
Organize the items on the bathroom shelf ano clean the shower fixtures

![](images/46574e74f3ad968ee2dc60aed60ed4117faf3912c6cffd4bebb82db0c049ce26.jpg)  
Move the books from the table to the sofa.

Figure 5 Representative HiFi-UMI tasks: head-camera views (left) and 3D end-efector trajectories (right).

We reconstruct the head trajectory with ofline stereo-inertial SLAM and recover both hand trajectories by detecting their fiducial markers in the head cameras (Fig. 5). Operating ofline lets the optimizer exploit future as well as past observations. Because manipulation continually alters the scene—violating the static-world assumption underlying standard loop closure [56]—we deliberately forgo global loop closure and instead impose a local-consistency constraint over a dynamic sliding window, which bounds global drift to the centimeter level over long horizons while preserving the millimeter-level local accuracy reported above. As SLAM optimization exhibits occasional stochastic failures, trajectories flagged as abnormal are automatically recomputed. A subsequent automatic-cleaning pass detects and annotates residual issues such as abnormal SLAM estimates, inconsistent trajectories, or other anomalous data patterns. This stage reconstructs 98% of trajectories; the remaining failures are automatically detected and removed.

## 3.3.3 Simulation Retargeting

Whether a trajectory can be replayed on the target robot is a prerequisite for its use in policy training. We develop a whole-body motion-control algorithm [57] for the target embodiment and validate every reconstructed trajectory by replaying it in simulation, discarding trajectories that are kinematically or dynamically infeasible. In our experiments this replay validation succeeds for 98% of reconstructed trajectories. Because reconstruction and replay validation are applied in series, the cumulative basic-validity yield is $9 8 \% \times 9 8 \% \approx 9 6 \%$ of raw captures, the figure reported in Sec. 3.2.

## 3.3.4 AI-Assisted Annotation

An annotation model supplements subtask segmentation and generates draft labels [58]. Exploiting the multiview captures unique to our device, the model reasons jointly over the head-mounted and hand-centric views to infer task progress, object interactions, and action boundaries; this multi-view design resolves ambiguity in cases where the object of interest is occluded in one view but visible in another. The stage emits structured metadata—task- and subtask-level language descriptions, temporal segment boundaries, manipulated objects, and candidate abnormal events—giving each demonstration an initial structured representation that substantially reduces later manual efort. Each label carries a confidence or uncertainty score, so that low-confidence samples can be routed preferentially to human review.

## 3.3.5 Human Verification

Human annotators perform sampling-based inspection and final verification. Rather than reviewing all raw data from scratch, they concentrate on samples flagged by automatic quality checks, low-confidence AI annotations, or distribution-level analysis, which keeps manual efort low without sacrificing reliability. During verification they confirm that language descriptions match the visual content, that subtask boundaries are temporally accurate, and that each demonstration satisfies its intended task definition; they correct or supplement AI labels—for example, adjusting temporal boundaries, refining coarse descriptions, adding missing object information, or marking samples for removal or down-weighting.

Verification outcomes are stored as structured quality-control metadata, so that every sample can be traced to its annotation source, review status, rejection reason, and correction history. Such traceability is essential at scale: it enables downstream analysis of failure modes, annotator consistency, and data-quality trends across tasks, scenes, devices, and collection batches.

## 3.3.6 Data Analysis and Export

Finally, the curated data is analyzed statistically before export. The analysis summarizes the dataset along multiple dimensions—task category, scene type, object category, action pattern, trajectory quality, annotation quality, and replay success—yielding a quantitative view of the distribution that surfaces over-represented, under-represented, or otherwise imbalanced subsets. Guided by these statistics, training sets are assembled by explicitly balancing task attributes against the requirements of the target model: the export process can control the proportions of scenes, tasks, objects, action types, and recovery behaviors, down-weight redundant subsets, or up-weight rare but important behaviors. We deliberately collect rare failure-and-recovery episodes to provide supervision for robust closed-loop execution.

Export converts the curated data into training-ready formats. Beyond raw observations and action trajectories, each exported episode carries synchronized multi-view video, calibrated trajectories, gripper states, language annotations, subtask boundaries, and quality-control metadata, so that a single data lake can serve heterogeneous training configurations—task-level imitation learning, subtask-conditioned policy learning, VLA training, and ofline evaluation. In efect, this analysis-and-export stage closes the loop between collection and training: instead of treating all demonstrations as equally useful, the system constructs datasets through explicit filtering, balancing, and versioned export, improving data utilization and making downstream model performance traceable to specific data choices.

## 3.4 Processed-Data Quality

Tab. 2 reports the end-to-end fidelity of the processed data. The pipeline delivers 3 mm end-efector accuracy, cross-sensor timing ofsets below 40 µs, fewer than two dropped frames per hour of capture, a 98% trajectoryreconstruction success rate, and gripper-state error below 0.1<sup>◦</sup>. Within a typical manipulation workspace—on the order of 2 m of accumulated head-trajectory length—the recovered end efector attains a mean translational error of 3 mm against base-station tracking ground truth (used only for this accuracy evaluation, not for routine capture). This is the regime our target manipulation scenarios operate in: they depend on locally consistent trajectories and accurate relative hand poses within the workspace. Taken together, these numbers indicate that the trajectory accuracy, timing, and gripper reconstruction of HiFi-UMI captures are on par with what real-robot teleoperation provides—the property our central hypothesis requires.

Table 2 End-to-end fidelity of the processed data.
<table><tr><td>Metric</td><td>Description</td><td>Value</td></tr><tr><td>Pose accuracy</td><td>Local end-effector error (~2 m workspace)</td><td>3mm</td></tr><tr><td>Synchronization</td><td>Cross-sensor timing offset</td><td> $< 4 0 \mu \mathrm { s }$ </td></tr><tr><td>Frame-drop rate</td><td>Dropped frames (6 cameras @ 25 fps)</td><td>&lt; 1 per 270,000 frames</td></tr><tr><td>Reconstruction</td><td>Trajectories passing SLAM reconstruction</td><td>98%</td></tr><tr><td>Gripper-state error</td><td>Opening-angle error</td><td>&lt; 0.1°</td></tr></table>

Fidelity must, however, survive into the policy’s action space. We represent actions in a robot-centric endefector frame, predicting relative pose increments together with an absolute gripper opening. We detail this action representation in Sec. 5.

## 4 Dataset and Release

The dataset is the direct product of the pipeline of Sec. 3. To date it comprises over 20,000 hours of processed manipulation data spanning more than 4.32 million episodes across 480+ scenes. Every episode is captured with the six-camera configuration of Sec. 3.1—stereo head views plus two ultra-wide fisheye views per hand— and is exported with synchronized multi-view video, calibrated bimanual trajectories, gripper states, language annotations, and subtask boundaries. Tab. 3 summarizes the headline statistics.

Two properties beyond raw scale make this corpus usable for our study. First, fidelity is enforced during collection and processing rather than audited afterwards, so every retained episode clears the bar of Tab. 2. Second, the composition of the corpus is measured rather than assumed: Fig. 6 shows its distribution over tasks, scenes, objects, and manipulation attributes, and the analysis-and-export stage of Sec. 3.3 uses these statistics to control the proportions of any exported training set. From the full corpus we curate and publicly release HiFi-UMI-2K, a representative 2,000-hour subset selected by the same stage to balance task, scene, and attribute coverage. HiFi-UMI-2K inherits the export format of the full corpus, so each released episode carries the same synchronized six-view video, calibrated bimanual trajectories, gripper states, language annotations, and subtask boundaries described above. Versioning and per-sample quality metadata (Sec. 3.3) make every experimental subset reproducible and traceable to its batch, device, and review history, linking performance to data choices. HiFi-UMI-2K is distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license, which permits redistribution and derivative use, including for commercial purposes, provided the source is attributed. Because the capture device records head-mounted egocentric video, all human faces appearing in the released recordings are masked before distribution, so HiFi-UMI-2K contains no facial imagery. The dataset is intended for research on manipulation learning and on robot data pipelines; we discourage its use for behavioral surveillance or identity tracking.

## 5 Baselines and Training Setup

Design principle. Our goal is to test whether high-fidelity UMI data is suficiently action-aligned for posttraining across reactive VLAs and WAMs, rather than for one policy implementation. We instantiate three strong foundation-policy baselines: StarVLA-QwenPI, OpenPI-π<sub>0.5</sub>, and LingBot-VA. We standardize the task definition, physical action semantics, initial-state distribution, success criteria, and safety conditions; each backbone retains its native observation, temporal sampling, action tensorization, normalization, and deployment interface. Within each backbone, we vary only the task-specific data source while holding the architecture, initialization, optimization, and interfaces fixed. Agreement across backbones is convergent evidence about the data, not a pooled architecture comparison.

![](images/2a6fb607792c9d4689b9759c1108c2d32580987f7c591da9b18045c634da4fec.jpg)  
Figure 6 Overview of the dataset distribution across tasks, scenes, objects, and manipulation attributes.

Table 3 Dataset statistics for the full processed corpus (“Collected”) and curated HiFi-UMI-2K subset (“Released”).
<table><tr><td>Property</td><td>Value</td></tr><tr><td>Collected</td><td></td></tr><tr><td>Hours</td><td>20,000+</td></tr><tr><td>Episodes</td><td>4,320,000+</td></tr><tr><td>Scenes</td><td>480+</td></tr><tr><td>Released (HiFi-UMI-2K)</td><td></td></tr><tr><td>Hours</td><td>2,000</td></tr><tr><td>Episodes</td><td>482,100+</td></tr><tr><td>Scenes</td><td>110+</td></tr><tr><td>Camera views per episode</td><td>6</td></tr></table>

Policy interfaces and physical action semantics. At a high level, each backbone consumes episodes of the form

$$
\tau = \left\{ \left( o _ { t } ^ { ( m ) } , q _ { t } ^ { ( m ) } , \ell , a _ { t : t + H _ { m } - 1 } ^ { ( m ) } \right) \right\} _ { t = 1 } ^ { T } ,\tag{1}
$$

where m indexes the backbone, $o _ { t } ^ { ( m ) }$ denotes its native visual observation, $q _ { t } ^ { ( m ) }$ denotes its native robot-state input, ℓ is the natural-language task instruction, and $a _ { t : t + H _ { m } - 1 } ^ { ( m ) }$ is its native future action tensor. The visual observation $o _ { t } ^ { ( m ) }$ is drawn from the four wrist views of the capture rig—two per hand—for every backbone and every condition; the head-mounted stereo pair is used only for trajectory reconstruction during capture and is never provided as input to the policy. Within each backbone, the UMI and teleoperation variants use identical prompts, camera selection, temporal ofsets, tensor layout, normalization convention, and evaluation protocol.

At the physical robot interface, all three backbones use the same active bimanual end-efector semantics. For arm $j ,$ every future pose in a chunk beginning at $t _ { 0 }$ is expressed relative to the same current-observation pose:

$$
\begin{array} { r } { \Delta \mathbf { T } _ { t _ { 0 } , h } ^ { j , ( m ) } = \left( \mathbf { T } _ { t _ { 0 } } ^ { j } \right) ^ { - 1 } \mathbf { T } _ { t _ { 0 } + \delta _ { h } ^ { ( m ) } } ^ { j } , } \end{array}\tag{2}
$$

where $\delta _ { h } ^ { ( m ) }$ is the native future ofset at index h for backbone $m .$ . Thus, chunk rows share one measured anchor and are not defined recursively relative to the preceding action target. Translation is expressed in the anchor end-efector frame, orientation uses Rotation6D [59], and the gripper target remains absolute. Each arm therefore contributes $3 + 6 + 1 = 1 0$ active physical channels, giving 20 channels for bimanual tasks. This common physical convention does not impose a common tensorization: StarVLA-QwenPI uses the 20 channels directly, $\mathrm { O p e n P I } { - \pi _ { 0 . 5 } }$ pads them within its native 32-dimensional action tensor, and LingBot-VA maps them into its native 30-dimensional tensor. Each backbone retains its own horizon, padding, masking, temporal ofsets, and normalization statistics.

## 5.1 StarVLA-QwenPI

StarVLA-QwenPI is our Qwen-based VLA baseline and follows the modular backbone–action-head design of StarVLA [49]. We use Qwen3-VL-4B-Instruct [60], with 36 transformer layers and a hidden width of 2,560, together with a π-style conditional flow-matching DiT action head [32, 61, 62]. Qwen encodes the multi-view observation and instruction, and each DiT block cross-attends to the corresponding layer-wise Qwen features.

The oficial QwenPI path lacks action-side self-attention, limiting token mixing within predicted chunks. We retain cross-attention to all 36 Qwen layers and add an attention-only self-attention residual after every odd-numbered DiT block, coupling the $H = 2 0$ action steps without repeating feed-forward computation.

For a ground-truth action chunk a and Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , I )$ , we sample $u \sim$ Beta(1.5, 1.0) and set $\tau = ( s - u ) / s$ , where $s = 0 . 9 9 9$ . The action head is trained to predict the vector field from noise to data:

$$
\begin{array} { r } { a ^ { \tau } = ( 1 - \tau ) \epsilon + \tau a , \ } \\ { \mathcal { L } _ { \mathrm { F M } } = \mathbb { E } \Big [ \| v _ { \theta } ( a ^ { \tau } , \tau , z _ { t } ) - ( a - \epsilon ) \| _ { 2 } ^ { 2 } \Big ] . } \end{array}\tag{3}
$$

At inference, we integrate the learned vector field from Gaussian noise using 8 explicit-Euler steps. The resulting chunk contains $H = 2 0$ actions [38, 63], of which the robot executes the first $H _ { \mathrm { e x e c } } = 1 0$ in a receding-horizon manner [64]. Images are resized to 224×224, and action dimensions are normalized using statistics from the training split.

## 5.2 OpenPI-π<sub>0.5</sub>

Our second VLA baseline, the JAX implementation of $\mathrm { O p e n P I } { - \pi _ { 0 . 5 } }$ [33], combines a PaliGemma [65] visuallanguage stream with a Gemma continuous action expert. It uses a SigLIP So400m/14 visual encoder followed by an 18-layer Gemma-2B language model, plus an 18-layer Gemma-300M action expert. The streams have modality-specific parameters and Mixture-of-Transformers joint attention [32].

Images and the tokenized task instruction form the visual-language prefix. Proprioceptive state is discretized and serialized together with the instruction rather than injected as a continuous action-side token. Action tokens attend to the complete prefix and to one another, allowing the full action chunk to be predicted jointly.

We use the continuous flow-matching path released in OpenPI. Given conditioning context $c = ( o , q , \ell )$ , action chunk $^ { a , }$ Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , I )$ , and flow time $t ,$ we construct $x _ { t } = ( 1 - t ) a + t \epsilon$ and train the action expert to predict the velocity field:

$$
\mathcal { L } _ { \mathrm { F M } } = \mathbb { E } \left[ \left. v _ { \theta } \left( x _ { t } , t \mid c \right) - \left( \epsilon - a \right) \right. _ { F } ^ { 2 } \right] .\tag{4}
$$

This is our sole fine-tuning objective; unlike the full $\pi _ { 0 . 5 }$ recipe, we omit knowledge insulation, autoregressive subtask generation, and auxiliary text loss. At inference, OpenPI generates a continuous action chunk with 10 Euler steps for the VLA receding-horizon executor (Sec. 5.7.1).

## 5.3 LingBot-VA

Our third baseline is LingBot-VA [48], a causal WAM that predicts future visual states before recovering the actions that realize them. Let $z _ { t } = E _ { \mathrm { V A E } } ( o _ { t } )$ denote the latent of the synchronized multi-view observation and $h _ { t } = ( z _ { \leq t } , a _ { < t } )$ the video–action history, with ℓ denoting the task instruction. LingBot-VA factorizes joint prediction as

$$
\begin{array} { r l } & { p _ { \theta } ( a _ { t : t + H - 1 } , z _ { t + 1 : t + K } \mid h _ { t } , \ell ) } \\ & { \quad = p _ { \theta } ( a _ { t : t + H - 1 } \mid z _ { t + 1 : t + K } , h _ { t } , \ell ) p _ { \theta } ( z _ { t + 1 : t + K } \mid h _ { t } , \ell ) . } \end{array}\tag{5}
$$

The second factor predicts future video latents, while the first acts as an inverse-dynamics model that decodes a continuous action chunk from the predicted visual transition. LingBot-VA uses a block-causal mask to order the interleaved video–action chunks, ensuring that sequence modeling conforms to the factorization described above. Future video latents and continuous actions are trained jointly with continuous flow matching, $\mathcal { L } _ { \mathrm { L i n g B o t } } = \mathcal { L } _ { \mathrm { v i d e o } } + \mathcal { L } _ { \mathrm { a c t i o n } }$ , using equal loss weights.

LingBot-VA follows the chunk-anchored physical convention in equation (2). Its 20 active bimanual dimensions are inserted into the native 30-dimensional action tensor through a fixed channel map, and unused channels are masked from the action loss. Rotation6D is formed from the first two rows of the relative rotation matrix. Each profile reuses its condition-specific normalization statistics throughout inference.

At deployment, receding-horizon prediction uses a bounded rolling KV cache, video/action guidance scales of $5 / 1 , 8 / 1 6$ denoising steps, an attention window of 24, and the WAM protocol in Sec. 5.7.2.

## 5.4 Training Variants

## We evaluate the following training conditions:

1. Real-robot teleoperation post-training. The policy is post-trained on task-specific demonstrations collected directly on the target robot through teleoperation. This setting represents the conventional deployment-oriented training recipe, where the post-training data is fully embodied in the target robot’s observation space, action space, dynamics, and gripper embodiment. We use it as the real-robot reference for comparison with UMI post-training.

2. UMI post-training. The policy is post-trained using only our high-fidelity UMI demonstrations for the target tasks. No real-robot teleoperation trajectories are included in this setting. This is the central experimental condition of the paper: it directly tests whether suficiently accurate robot-free demonstrations can serve not merely as pre-training data, but as the sole post-training source for producing deployable manipulation policies on a real robot.

3. UMI pre-training + UMI post-training. The policy first undergoes continued pre-training on the full-scale UMI corpus, and is then post-trained on the task-specific UMI subset. Both stages use robotfree UMI data, with no real-robot teleoperation episodes. This variant tests whether broad robot-free manipulation pre-training provides reusable visual-motor priors [8] that further improve downstream UMI-only specialization. We instantiate this pre-training stage on StarVLA-QwenPI only; OpenPI-π<sub>0.5</sub> and LingBot-VA are post-trained from their publicly released base checkpoints throughout.

Within a given comparison, the variants share the same policy backbone and its native action tensorization, observation interface, normalization convention, temporal sampling, and deployment controller. Therefore, performance diferences mainly reflect the source and stage of the training data rather than changes in model architecture or robot execution protocol. In particular, comparing real-robot teleoperation post-training against UMI post-training measures whether high-fidelity UMI data can replace the conventional real-robot anchor, while comparing UMI post-training against UMI pre-training plus UMI post-training measures the value of scaling robot-free data before task-specific adaptation.

## 5.5 Data Processing and Episode Filtering

All UMI trajectories are first transformed into the deployment robot’s end-efector action convention. The reconstructed hand poses are expressed in a shared world frame and converted into robot end-efector frames. Each backbone-specific loader then selects its native future ofsets $\delta _ { h } ^ { ( m ) }$ and constructs the chunk-anchored relative targets in equation (2). The resulting 20 active physical channels are packed, padded, and masked according to the backbone’s native tensor layout. The teleoperation pipeline uses the same backbone-specific physical target convention and tensor layout as its UMI counterpart.

Before backbone-specific conversion, we apply source-trajectory validity checks. Because the corpus is already high-fidelity (Sec. 3.4), these checks rarely trigger on the UMI data. Episodes are removed if they contain any of the following: SLAM tracking failure, missing camera frames, timestamp discontinuity, severe hand-pose outliers, action spikes above a physical threshold, incomplete task execution, or inconsistent gripper-state reconstruction. Long episodes are sliced into coherent segments with idle prefixes and sufixes removed. Filtering precedes the train/validation split to keep near-duplicate windows from crossing splits.

For normalization, each condition uses the statistics bound to its deployment profile under the corresponding backbone’s unchanged normalization convention. Position, rotation, and gripper channels are normalized separately. For bimanual tasks, left and right arms use separate statistics.

## 5.6 Training and Optimization

VLA pre-training. Pre- and post-training minimize the flow-matching behavior-cloning objective. We optimize StarVLA-QwenPI end to end with AdamW [66] $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5 , \epsilon = 1 0 ^ { - 8 }$ , and weight decay $1 0 ^ { - 8 } )$ in mixed precision, with gradient clipping at 1.0 and no accumulation.

We pre-train on 4,000 hours of multi-task UMI data. One pass through this training mixture corresponds to 180,000 optimization steps. For the scaling and OOD analyses, the final exported checkpoint is obtained after a further 5,000-step linear learning-rate decay. Training uses an efective global batch size of 2,048. The schedule begins with a 3,000-step linear warm-up. Peak learning rates are $1 0 ^ { - 4 }$ for the action DiT and projection layers, $2 . 5 { \times } 1 0 ^ { - 5 }$ for the remaining backbone parameters, and $1 0 ^ { - 5 }$ for the Qwen–action interface. We use one independently sampled flow time and noise realization per training example, randomize the ordering of camera views, and save checkpoints every 5,000 steps.

VLA post-training. For task-specific post-training of VLA models, the UMI-pretrained variant initializes all parameters from the 185k-step pre-training checkpoint. We train for 50,000 steps with an efective global batch size of 512. After a 2,500-step linear warm-up, the learning rate follows cosine decay to $7 { \times } 1 0 ^ { - 7 } ;$ the peak rates are $5 \times 1 0 ^ { - 5 }$ for the action DiT and $1 0 ^ { - 5 }$ for the Qwen backbone and Qwen–action interface. For each task sample, we draw eight independent flow-time/noise realizations and average their flow-matching losses. The Qwen-VL-initialized StarVLA-QwenPI baseline instead begins task-specific training with a randomly initialized action policy, as described in the post-training comparison below. For OpenPI-π<sub>0.5</sub>, we follow the OpenPI fine-tuning pipeline and initialize from pi05\_base using the same converted UMI data.

WAM post-training. For LingBot-VA, we follow the released task-specific post-training implementation and initialize the video–action Transformer from lingbot-va-base [48]. We update all Transformer parameters while keeping the causal VAE and text encoder frozen. For each task and data-source condition, we train for 3,500 steps with a global batch size of 32. We use AdamW [66] with $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 ) , \epsilon = 1 0 ^ { - 8 }$ , and weight decay 0.01, excluding bias, normalization, and other one-dimensional parameters. The learning rate is linearly warmed up to $1 0 ^ { - 5 }$ over the first 25 steps, kept constant through step 3,000, and cosine-annealed to zero over the final 500 steps. The video and action flow-matching losses are weighted equally.

Validation and checkpoint selection. During validation, we report the held-out flow-matching loss, per-dimension action error after de-normalization, gripper error, and rollout-level metrics on the real robot. We do not select checkpoints solely by validation loss. Instead, we choose checkpoints using a small fixed validation protocol and then freeze them before the final blind real-robot evaluation.

## 5.7 Deployment Protocol

Across the two evaluation tracks, we standardize the task definition, initial-state distribution, physical robot action semantics, safety wrapper, workspace and velocity limits, and success and termination criteria. We oth erwise preserve each backbone’s native temporal execution interface. Consequently, UMI- and teleoperation-

post-trained policies are compared under strictly matched deployment settings within each backbone, without forcing the VLA and WAM backbones into a common control cadence or chunk-consumption rule.

## 5.7.1 VLA Deployment

Following the latency-matching principle of UMI [6], we deploy the two VLA backbones with timestamped receding-horizon control. Sensor streams are aligned to a common observation time,

$$
t ^ { \mathrm { o b s } } = t ^ { \mathrm { n o w } } - \operatorname* { m a x } _ { s } \tau _ { s } ^ { \mathrm { o b s } } ,\tag{6}
$$

where $\tau _ { s } ^ { \mathrm { o b s } }$ is the calibrated latency of sensor stream s. Images, proprioception, and gripper state therefore correspond to the same physical instant. For a query anchored at $t _ { 0 } ,$ , every row of the returned pose chunk is restored independently from the synchronized query-time end-efector pose according to equation (2), rather than being recursively integrated from the preceding target.

Predicted targets are assigned timestamps using the backbone-specific action interval and submitted to the latency-compensated robot command bufer. StarVLA-QwenPI uses the $H = 2 0 , H _ { \mathrm { e x e c } } = 1 0$ configuration described above; $\mathrm { O p e n P I } { - \pi _ { 0 . 5 } }$ retains its native fixed horizon and replanning interval. For each backbone, all deployment parameters are held unchanged between its UMI- and teleoperation-post-trained variants.

## 5.7.2 WAM Deployment

LingBot-VA instead follows its native block-causal streaming protocol. After an episode reset, the first model call generates 12 executable actions, while subsequent calls predict a native two-block chunk containing 24 actions. For each matched UMI-versus-teleoperation comparison, we use the same execution schedule—12 actions after reset and 24 actions thereafter—for both post-training variants. At the start of every executed chunk, we measure the current end-efector poses and use them as the shared chunk anchors. Each predicted pose target is independently restored according to equation (2), without accumulating action rows over time. The evaluated profiles retain a source-time stride of three, i.e., one source observation per three native action slots. Together with the VAE temporal downsampling factor of four, this gives 12 native action slots per latent video frame. The low-level robot controller tracks the resulting timestamped targets.

During execution, the client records one synchronized multi-view key observation every three executed actions. At each subsequent request, the action history from the preceding block and the newly collected real observations are first used to update LingBot-VA’s bounded rolling KV cache. The next video–action chunk is then predicted from this updated context, keeping the causal history grounded in the robot’s executed trajectory. The evaluated baseline performs synchronous inference at chunk boundaries, and the cache is reset whenever the episode or task prompt changes.

## 6 Experiments

We design our experiments to answer a central question: can high-fidelity UMI data serve as a post-training source for directly deployable real-robot manipulation policies, without relying on real-robot teleoperation data? To separate the efect of data source from that of model architecture, we conduct the comparison across three policy backbones spanning both foundation-policy families: the VLA policies StarVLA-QwenPI and OpenPI-π<sub>0.5</sub>, and the WAM LingBot-VA.

## 6.1 Experimental Design

## 6.1.1 Evaluation Benchmark

All real-robot experiments use a stationary bimanual platform with two seven-joint Tianji Robotics Marvin M6 force-controlled arms. At deployment, the platform matches the HiFi-UMI end efector: it carries the same gripper and four wrist cameras described in Sec. 3.1. The head stereo pair is used only for ofline capture reconstruction and excluded from deployment, so the policy receives a strict subset of the recorded views. Capture and deployment thus have physically identical contact and observation interfaces; the residual embodiment gap is confined to arm kinematics. Policies emit end-efector pose targets, interpolated to 125 Hz and streamed to a controller that solves inverse kinematics at 125 Hz and sends joint-angle commands over EtherCAT at 1 kHz.

To ensure a fair comparison across policies, we follow established real-robot evaluation protocols [67] and conduct all rollouts through a standardized benchmark system. Each rollout is jointly conducted by two evaluators with separated responsibilities: a policy operator loads and launches the assigned policy, while a scene operator independently constructs the sampled test instance, including object placement and environment configuration, according to the benchmark specification. This separation limits operator bias and keeps initial conditions randomized and reproducible across policies.

Before evaluation, we freeze the task definitions, object sets, language instructions, policy checkpoints, initialcondition bank, task-specific timeouts, and safety rules. For every rollout, task-relevant objects are newly randomized within a predefined collision-free and robot-reachable tabletop region. All policies are evaluated under the same initial-condition distribution, while the policy order is randomized to reduce temporal and operator bias. We conduct 40 rollouts for each task–policy pair.

The primary metric is binary task success. A rollout is successful only if the complete task objective is achieved before the fixed timeout, the final state remains stable for at least two seconds, and no safety intervention occurs. A rollout terminates upon success, timeout, prolonged lack of task progress, repeated inefective motion, unrecovered object dropping, incorrect-object interaction, or a safety stop. All termination reasons are recorded for failure analysis. Hardware or communication faults not caused by the policy are marked as invalid trials and repeated.

## 6.1.2 Task Suite

Our benchmark comprises four tabletop manipulation tasks that span contact-rich interaction, deformableobject manipulation, constrained placement, and semantic sorting (Fig. 7):

1. Stain Wiping. The robot picks up a towel from the table, wipes the designated stain until the target region is clean, and finally places the towel back on the table.

2. Shirt Folding. The robot folds the left and right sleeves of a shirt and then symmetrically folds the lower hem to produce the desired final configuration.

3. Remote Insertion. The robot grasps a remote and inserts it into the target storage box.

4. Produce Sorting. The robot places a designated vegetable onto the pink plate and a designated fruit onto the blue plate according to their semantic categories.

The tasks span complementary capabilities: Stain Wiping tests sustained contact and spatial coverage; Shirt Folding, bimanual deformable-object coordination; Remote Insertion, precise constrained placement; and Produce Sorting, semantic category-conditioned manipulation.

## 6.1.3 Data Collection Regimes

We construct separate UMI and real-robot teleoperation datasets for every task.

UMI demonstrations. The UMI demonstrations are collected by multiple operators across diferent physical sites, backgrounds, lighting conditions, and tabletop appearances. For each task, we use 3,200 trajectories for post-training, corresponding to approximately 10–20 hours of demonstrations depending on the task duration. Because UMI collection does not require the target robot, independent operators can collect data concurrently across multiple environments.

Real-robot teleoperation demonstrations. The teleoperation demonstrations are collected directly on one target robot in the same physical environment used for evaluation. For each task, we collect approximately 300 trajectories, corresponding to approximately 3–7 hours of demonstrations depending on the task duration. Each collection session requires both a teleoperator and an additional assistant for scene reset, object placement, safety supervision, and recovery.

![](images/1d6b9f1329f13512c1e39585afacb26930d28b2147a2d0f174cde145c2142163.jpg)

![](images/3eaba19c2321745eb72d133d1292fd1abb146cdc1e7ba1f7b2a0c168ef6ecaaf.jpg)

![](images/9f755becc79bcf432297eafccf71d720e7ae78ef9e71490f42bd0a52c7d4fb32.jpg)

![](images/7869deaeb4b1d64f7290b43cf936db0f4aa6c3aab1b2687cbe0f694e91b8b4fa.jpg)

![](images/e9ddc04b3663b272b1280ab459d2c537242e90845ff40b3100f0cf65a9842827.jpg)

![](images/d27c50318103d8b07dd807779bd401c75410fc9457163377343195a62e1dd833.jpg)

![](images/8c1df4ccda499566d95a958303e785d1b39522bc82eb95c0a971f3c0007113a3.jpg)

![](images/d7eb4a854ffc1b5337460f5645b4e6e77cdfb4d7992304448488883e0cc98179.jpg)  
(b) Shirt Folding

(a) Stain Wiping  
![](images/4135b41c3a626a7b238396abb1a855399856c118d92151560061c4942c9bdea4.jpg)

![](images/2b364f19229fc0bfae8ce56e9b5c95e72f949f3489a467ace3aef74978f71b4b.jpg)

![](images/13c8ef0f84dca389c95bbe632037cbbb9255945568b05c16e96c555876a2b4ec.jpg)  
(c) Remote Insertion

![](images/63a92741d204f66f6a080fbad99893b3e072f95f9e1ee8e31805d968c35eb9ba.jpg)

![](images/fcab837d20cd96391f40f3073fccf93f21682048a89430dbb1e60dd20e039c53.jpg)

![](images/e913c049741d27e3ad4b935c4c571e4a205d295d78f1c4014e008040a1a1b8fc.jpg)

![](images/0830f8505afaa5bc5915860f91d90a1b39d4fd7cb1c848202dbfe81db11ce8f8.jpg)  
(d) Produce Sorting

![](images/e4f2469fc4b9a69f31497ea6fe67c113b1197d7f834b6d7b55b7930b722e9b01.jpg)  
Figure 7 The four benchmark tasks, shown as first-person HiFi-UMI demonstrations. Each panel shows representative stages of one task, captured from the head-mounted camera during handheld data collection. The suite spans contact-rich wiping, deformable-object manipulation, constrained placement, and semantic object sorting. Real-robot executions of the same tasks are shown in Fig. 8.

Table 4 Seven training conditions. Each is evaluated on four tasks with 40 rollouts per task. C1–C6 compare posttraining sources within each backbone; C1 versus C7 compares initialization under matched HiFi-UMI post-training. C3–C6 use public base checkpoints, while only C7 adds HiFi-UMI pre-training (Sec. 5.6).
<table><tr><td></td><td>Backbone</td><td>Initialization / pre-training</td><td>Post-train data Reported in</td><td></td></tr><tr><td>C1</td><td></td><td>StarVLA-QwenPI Qwen3-VL, scratch action head</td><td>HiFi-UMI</td><td>Secs. 6.2 and 6.3</td></tr><tr><td>C2</td><td>StarVLA-QwenPI</td><td>Qwen3-VL, scratch action head</td><td>Teleoperation</td><td>Sec. 6.2</td></tr><tr><td>C3</td><td>OpenPI-π0.5</td><td>pi05_base</td><td>HiFi-UMI</td><td>Secs. 6.2 and 6.3</td></tr><tr><td>C4</td><td>OpenPI-π0.5</td><td>pi05_base</td><td>Teleoperation</td><td>Sec. 6.2</td></tr><tr><td>C5</td><td>LingBot-VA</td><td>lingbot-va-base</td><td>HiFi-UMI</td><td>Sec. 6.2</td></tr><tr><td>C6</td><td>LingBot-VA</td><td>lingbot-va-base</td><td>Teleoperation</td><td>Sec. 6.2</td></tr><tr><td>C7</td><td>StarVLA-QwenPI</td><td>Qwen3-VL → HiFi-UMI pre-training</td><td>HiFi-UMI</td><td>Sec. 6.3</td></tr></table>

In our collection pipeline, obtaining one usable teleoperation trajectory requires several times more wall-clock time than obtaining one UMI trajectory. In addition to human operation time, real-robot collection incurs overhead from robot execution, environment reset, safety checks, and hardware recovery. The reported sizes therefore reflect practical pipeline throughput, not a trajectory-count-matched design.

No UMI trajectory is collected in the evaluation scene. Consequently, the UMI-post-trained policies are evaluated under a scene-level distribution shift in background, illumination, tabletop appearance, and overall visual context. In contrast, the teleoperation demonstrations are collected using the same robot setup and environment as the evaluation. This asymmetry provides a conservative test of whether diverse, multi-site UMI data can transfer to an unseen deployment scene.

## 6.1.4 Policy Comparisons and Evaluation Protocol

Our study varies three factors: the policy backbone, the source of task-specific post-training data, and—for StarVLA-QwenPI only—the initialization from which post-training starts. The design is deliberately unbalanced: the data-source axis is evaluated on all three backbones, whereas the initialization axis is evaluated on one. Tab. 4 enumerates the resulting seven training conditions and indicates where each is reported.

Two comparisons follow from this design, each holding everything else fixed. The data-source comparison (C1 vs. C2, C3 vs. C4, C5 vs. C6) fixes the backbone, the initialization, the optimization recipe, the action representation, the normalization protocol, and the deployment executor, and changes only where the taskspecific demonstrations come from. The initialization comparison (C1 vs. C7) fixes the post-training data— the same 3,200 HiFi-UMI trajectories per task—together with the recipe and the deployment stack, and changes only the checkpoint from which post-training starts. Both arms of that comparison begin from the same Qwen3-VL weights, and C7 difers only in that those weights are first carried through large-scale HiFi-UMI pre-training. The comparison therefore isolates the visual-motor prior acquired from HiFi-UMI data rather than the vision-language prior itself. $\mathrm { O p e n P I } { - \pi _ { 0 . 5 } }$ and LingBot-VA are post-trained from their publicly released base checkpoints throughout and do not participate in the initialization comparison.

![](images/6a71e0ec02311657770c9f34f95721ccd8bd4cb4d4b9dd5d882f1c1ebd24dab9.jpg)

![](images/e5e3fedf8d4a745930013d41c4803731bea820e7db8500da318fa06c2fb19bd3.jpg)

![](images/25f29eafa26b7f8243c551ffc8ef63ab3017711293e05e99099a01981ab44106.jpg)

![](images/457cb26db319637d67549433c23daea24e491b5e24da63d60dbd5bbfc39367b3.jpg)  
(a) Stain Wiping

![](images/87d0e39beb607b077643bf6e4203e2538feb9f1c98aff5398f6096f41d796624.jpg)

![](images/38ce61fbfb5178e008e718ac4596f0f897c6d47f3e59b4ebc235b60f17b995bb.jpg)

![](images/79a08db9db9bd44ff414751252c1bd8f5d46ea7e7b2ac7feaff84017b60b4195.jpg)

![](images/147f4529514176b62429720efbde2287abc8567a16152a46cc77c09a102eb541.jpg)

![](images/4eecb58bed27a9709e2c1e48bbc2c458edf8235f8905a195cf04d8fc7afae3f7.jpg)  
(b) Shirt Folding

![](images/f34e9c35e96ff5ea312e2faf0cb29e6ea4c84a0021d94fb495e1ad862d397612.jpg)

![](images/d95620143719f487135db73680767631d7c71825ffa75dd0336833f4cc675f10.jpg)

![](images/5caea01c28c0056bf00e10490f2d46e2b3b97bc845df29a39847652735b8e30f.jpg)

![](images/40aff695bef332ea393a3d986d88fa3a2db4c6aa97ee215f22a81ff40174ae13.jpg)  
(c) Remote Insertion

![](images/13b7e6663d2a6c2bbf434f44b8310f725cd7335195c86fa83f1283d3ef37ab94.jpg)

![](images/a89d153404ecf846c2cd4fc020abd35ea387b17af1ec5998350f136cb1609f1f.jpg)

![](images/2f49431023c98a4d54a9565c8b5b2d5cb6624cf9502d871cfff2e41864ddab86.jpg)

![](images/a46fc4f7127a16a597293c7c8103f679be81d13ebd04dc8c1c219f0b52bfe15d.jpg)

![](images/4b032a592d03ea54f9d5eaa7fe6b45287c8880c7ef00c3e5c05e156172a00433.jpg)  
(d) Produce Sorting

![](images/3b05f485b824dc82927e7ba63339db8132c930c3be8e8db1b6dfd3e76bc316b9.jpg)

![](images/1c2238812c8767699d93d55730ee23585c933ee2cd7fc3e7b1fe6196f89adeb8.jpg)  
Figure 8 Representative successful real-robot executions of the four benchmark tasks, with time progressing from left to right in each row.

For every condition–task pair, we conduct 40 real-robot rollouts. Before each rollout, the manipulated objects are randomly repositioned within the permitted workspace. Evaluation is performed on two identically configured robot platforms under the same physical safety and task-evaluation protocol, while retaining the backbone-specific execution clients described above. The primary metric is the task success rate, $N _ { \mathrm { s u c c e s s } } / N _ { \mathrm { r o l l o u t } }$ , under the success and termination criteria fixed in Sec. 6.1.1; partial completion of the task objectives listed in Sec. 6.1.2 is counted as failure.

## 6.2 Can UMI-Only Post-Training Match Teleoperation-Based Post-Training?

Across both evaluation tracks below, using HiFi-UMI rather than teleoperation as the task-specific posttraining source yields approximate aggregate parity under the evaluated setting. Taking UMI minus teleoperation, the aggregate success-rate diferences are −2.5, +3.1, and −0.6 percentage points for StarVLA-QwenPI, $\mathrm { O p e n P I } { - \pi _ { 0 . 5 } }$ , and LingBot-VA, respectively, with no consistent direction across backbones. Because the VLA and WAM tracks use diferent model-specific experimental settings, including temporal, training, and deployment protocols, their absolute success rates are not directly comparable. We therefore interpret their agreement as convergent evidence from three controlled within-backbone comparisons rather than as a pooled comparison across policy paradigms. We report the two VLA backbones and the WAM in turn.

## 6.2.1 Evaluation Track I: VLA Backbones

We compare UMI-only post-training against conventional real-robot teleoperation post-training using two VLA backbones. Within each backbone, the model architecture, initialization, optimization recipe, action representation, and deployment stack are held fixed; the task-specific post-training dataset is supplied by either the HiFi-UMI or teleoperation collection pipeline, while the model and deployment stack are held fixed within each backbone. In total, this evaluation comprises 640 real-robot rollouts across four tasks, four policy variants, and 40 trials per task–policy pair. Figure 8 shows successful real-robot rollouts for all four tasks.

Aggregate comparison. As shown in Fig. 9, UMI-only post-training closely matches teleoperation-based post-training for both VLA backbones. On StarVLA-QwenPI, the UMI-post-trained policy achieves 51.3% success (82/160), compared with 53.8% (86/160) for its teleoperation-trained counterpart, corresponding to a diference of only 2.5 percentage points. On $\mathrm { O p e n P I } { - \pi _ { 0 . 5 } }$ , UMI post-training achieves 77.5% success (124/160), exceeding teleoperation post-training at 74.4% (119/160) by 3.1 percentage points. Pooled, UMI yields

![](images/00f08dc6f192bb01185c510f2f0819747f9f4ccaa1a3974d35861343c7f84268.jpg)  
Figure 9 Real-robot success rates across the two VLA backbones and post-training data sources. Each task-policy pair is evaluated over 40 rollouts, while the aggregate result pools all four tasks (160 rollouts per policy). Solid bars denote UMI post-training and hatched bars denote real-robot teleoperation post-training. Bold labels indicate the highest success rate within each task.

206/320 (64.4%) versus 205/320 (64.1%) for teleoperation; within-backbone comparisons remain primary.

Notably, the aggregate UMI–Teleop gap remains small for both VLA backbones, whereas the diference between the two backbones is substantially larger. This indicates that the conclusion does not depend on a particular VLA architecture: replacing teleoperation demonstrations with high-fidelity UMI demonstrations does not systematically reduce real-robot performance.

Task-level comparison. The direction of the UMI–Teleop diference varies across tasks rather than consistently favoring either data source. On Stain Wiping, the two conditions perform similarly for StarVLA-QwenPI and achieve an exact tie at 65.0% for OpenPI-π<sub>0.5</sub>. Teleoperation provides a modest advantage on Shirt Folding, reaching 60.0% versus 52.5% with StarVLA-QwenPI and 80.0% versus 77.5% with OpenPI-π<sub>0.5</sub>. Conversely, UMI post-training is slightly higher on Remote Insertion: StarVLA-QwenPI reaches 52.5% versus 50.0% with teleoperation, and OpenPI-π reaches 85.0% versus 77.5%. On Produce Sorting, OpenPI also yields a higher rate with UMI (82.5% versus 75.0%). These diferences may reflect UMI’s broader variation in objects, backgrounds, and collection sites, which can aid spatial grounding and object-conditioned control.

Because each task–policy pair contains 40 rollouts, one successful trial corresponds to 2.5 percentage points. Most task-level diferences therefore represent only one to three rollouts. We consequently interpret small diferences as approximate parity rather than decisive superiority of one data source.

Generalization under scene shift. This comparison is conservative for UMI with respect to evaluationscene visual shift, but it is not sample matched. The UMI demonstrations are collected by multiple operators across environments that difer from the evaluation scene in background, illumination, tabletop appearance, and spatial layout. In contrast, the teleoperation demonstrations are collected directly on the target robot in the same environment used for evaluation. Despite this scene-level distribution shift, UMI-post-trained policies maintain aggregate performance comparable to their in-environment teleoperation counterparts. This is consistent with diverse, high-fidelity UMI data providing accurate supervision under scene shift.

(a) Task-specific data scaling  
![](images/f50433b43c4f7f27001a8c01eb2a4170c28a930d64b0f5d898a691b87da9286c.jpg)

(b) Effect of UMI pre-training  
![](images/e2df55ea201a1acb066eea0fb430ccee9a1045a9a5a8d0c19754ed7d52140649.jpg)  
Figure 10 Two complementary Remote Insertion scaling studies. (a) Task-specific data scaling for OpenPI-π post-trained on increasingly large UMI subsets. (b) Initialization ablation for StarVLA-QwenPI post-training; the dashed line marks the Qwen-VL-initialized scratch action policy trained on 3,200 episodes. The UMI-pretrained policy exceeds this baseline with only 800 episodes. Each setting is evaluated over 40 real-robot rollouts.

Across the two VLA backbones, the results provide evidence that high-fidelity UMI data can serve as the sole task-specific post-training source for directly deployable VLA policies, without requiring a real-robot teleoperation anchor. The comparison is not sample matched: each task is post-trained on 3,200 UMI trajectories but 300 teleoperation trajectories. We therefore interpret these experiments as a comparison between practical data-production pipelines, rather than as a claim of equal-sample data eficiency. Under this practical collection regime, however, removing real-robot teleoperation from post-training does not produce a corresponding loss in deployment performance.

How Much UMI Data Is Needed for VLA Deployment? While the preceding VLA comparisons demonstrate that UMI-only post-training can match teleoperation-based post-training, an important practical question remains: how much UMI data is required to obtain a deployable manipulation policy? To answer it, we study how UMI demonstration count afects real-robot success.

We select the Remote Insertion task for this analysis. This task provides a controlled yet challenging evaluation setting: it requires a sequence of manipulation skills, including object grasping, transportation, precise pose alignment, and insertion into a constrained target region. Compared with highly deformable-object tasks such as shirt folding, Remote Insertion has lower intrinsic variance, allowing us to better isolate the efect of demonstration quantity on policy learning.

Following recent studies on data scaling in robotic imitation learning [8], we train the same OpenPI-π<sub>0.5</sub> backbone using diferent subsets of our UMI demonstrations. Specifically, we construct five training sets containing 400, 800, 1,600, 3,200, and 6,400 episodes, respectively. All other training configurations, including model initialization, optimization settings, action representation, and evaluation protocol, are kept identical. Each trained policy is evaluated on the real robot with 40 independent rollouts under randomized initial object configurations.

As shown in Fig. 10a, increasing the amount of UMI post-training data leads to substantial performance improvements in the low-data regime. The success rate improves from 37.5% with 400 demonstrations to 65.0% with 800 demonstrations, indicating that additional UMI trajectories rapidly improve the policy’s ability to acquire the basic manipulation skill. Further scaling to 1,600 and 3,200 demonstrations continues to improve performance, reaching 70.0% and 85.0% success rates, respectively.

However, after approximately 3,200 demonstrations, the performance improvement largely saturates. Increasing the training set size from 3,200 to 6,400 episodes does not provide additional gains, with the success rate slightly decreasing from 85.0% to 82.5%. This suggests that performance plateaus by 3,200 episodes at the resolution of our 40-rollout evaluation.

![](images/c304b5c9ef426c2c525b005d8a41edffc43aad91b428abdaeeda4b28fce24b46.jpg)  
Figure 11 Real-robot success rates for LingBot-VA under UMI versus teleoperation post-training. Each task–datasource pair is evaluated over 40 rollouts, while the aggregate result pools all four tasks (160 rollouts per post-training source). Solid bars denote UMI post-training and hatched bars denote real-robot teleoperation post-training. Bold labels indicate the highest success rate within each task.

Together with the previous UMI-versus-teleoperation comparison, these results support the conclusion that high-fidelity UMI data provides scalable task-specific supervision. Rather than requiring a small amount of expensive robot-collected “anchor” data, increasing robot-free demonstrations substantially improves taskspecific post-training in the low-data regime and reaches competitive deployment performance without a real-robot teleoperation anchor.

## 6.2.2 Evaluation Track II: WAM Policy

We conduct a separate UMI-versus-teleoperation comparison within LingBot-VA, a WAM that generates actions through predicted future video and inverse-dynamics decoding. We construct UMI- and teleoperationpost-trained variants from the same LingBot-VA base checkpoint, while keeping the model architecture, optimization schedule, video–action objective, action representation, inference settings, and deployment stack fixed. The task-specific post-training data are supplied by the corresponding UMI or teleoperation collection pipeline. For each task–policy pair, we conduct 40 real-robot rollouts, yielding 320 rollouts across four tasks and two policy variants. Because this WAM evaluation follows a model-specific training and evaluation protocol, its absolute success rates are not directly compared with those of the VLA track. As an ofline diagnostic, we additionally evaluate action prediction while conditioning on ground-truth future video. This oracle conditioning removes accumulated future-video generation error from the diagnostic and allows us to probe the inverse-action component more directly.

Aggregate comparison. Across all four tasks, UMI-only post-training achieves an aggregate success rate of 56.9% (91/160), closely matching the 57.5% (92/160) obtained with real-robot teleoperation post-training. This comparable performance suggests that the two data sources provide similar efectiveness for post-training under the evaluated setting, rather than indicating a clear advantage of either source. Under the separate WAM evaluation protocol, replacing teleoperation demonstrations with HiFi-UMI does not result in a systematic degradation in closed-loop deployment performance.

Task-level comparison. The direction of the success-rate diference varies across tasks rather than consistently favoring either data source. UMI post-training performs slightly better on the Stain Wiping task, achieving 62.5% success compared with 60.0% for teleoperation, and similarly reaches 57.5% on the Produce Sorting task compared with 55.0%. The two variants achieve the same success rate of 65.0% on the Shirt Folding task. Conversely, teleoperation post-training performs better on the Remote Insertion task, reaching 50.0% success compared with 42.5% for UMI post-training. With 40 rollouts per task–policy pair, these small variations do not consistently favor either source.

Beyond binary success, qualitative inspection of the robot rollouts reveals a systematic diference in execution tempo: UMI-post-trained policies tend to produce larger-amplitude, more continuous, and more naturallooking motions. Successful rollouts often reach task milestones with fewer pauses, whereas the teleoperation post-trained policies more often execute incremental corrections. These observations are qualitative, as completion time is also influenced by recovery behaviors and termination conditions.

More direct nominal execution does not necessarily imply stronger recovery behavior. This distinction is most evident on the Remote Insertion task: the UMI-post-trained policy often performs a decisive initial grasp, but may require multiple attempts under imperfect contact conditions. These retries reduce the overall execution eficiency and are consistent with the lower success rate observed on this task. This result highlights the distinction between nominal execution eficiency and recovery robustness, suggesting that broader coverage of contact correction and regrasp behaviors remains important for reliable closed-loop deployment

Ground-truth-video analysis shows stable cross-domain pose decoding. The closed-loop evaluation above measures the performance of the full WAM pipeline, but it does not isolate whether failures arise from future-video prediction or action decoding. As LingBot-VA’s action prediction is conditioned on predicted future visual latents, evaluation with generated video combines visual-generation and action-decoding errors. To disentangle these factors, we replace the generated future-video latents with cached ground-truth latents and evaluate the action decoder on the Stain Wiping, Shirt Folding, Remote Insertion, and Produce Sorting tasks. This ground-truth-video protocol serves as an oracle diagnostic rather than a deployable inference mode: it removes future-video generation error from the evaluation, but does not by itself establish that video generation is the dominant closed-loop bottleneck. We evaluate three settings: a teleoperationpost-trained model on real-robot held-out observations (Real→Real), a UMI-post-trained model on the same held-out domain (UMI→Real), and the same UMI-post-trained model on episode-disjoint UMI held-out observations (UMI→UMI).

Metrics and aggregation. We evaluate the complete first executed action chunk, with horizon $H = 1 2$ After applying the same de-normalization and chunk-anchor $\operatorname { S E } ( 3 )$ reconstruction to predictions and targets, we compute the bimanual XYZ trajectory error as

$$
E _ { \mathrm { X Y Z } } = 1 0 ^ { 3 } \sqrt { \frac { 1 } { 6 H } \sum _ { t = 1 } ^ { H } \sum _ { b \in \{ L , R \} } \big \lVert \hat { \mathbf { p } } _ { t , b } - \mathbf { p } _ { t , b } \big \rVert _ { 2 } ^ { 2 } } ,\tag{7}
$$

where the factor $1 0 ^ { 3 }$ converts meters to millimeters. Evaluating the complete chunk preserves the native chunk-anchor action contract and measures the whole trajectory that is passed to the controller, rather than a selected shorter prefix. For rotation, we first convert Rotation6D predictions and targets to valid rotation matrices and form strict adjacent-frame increments, $\Delta \mathbf { R } _ { t , b } = \mathbf { R } _ { t - 1 , b } ^ { \top } \mathbf { R } _ { t , b }$ . We then report

$$
E _ { \mathrm { r o t } } = \frac { 1 } { 2 ( H - 1 ) } \sum _ { t = 2 } ^ { H } \sum _ { b \in \{ L , R \} } d _ { \mathrm { S O } ( 3 ) } \Big ( \widehat { \Delta \mathbf { R } } _ { t , b } , \Delta \mathbf { R } _ { t , b } \Big ) ,\tag{8}
$$

in degrees, where $d _ { \mathrm { S O ( 3 ) } }$ is the geodesic angle between two rotations. Step $t = 1$ is omitted because its preceding pose is the external chunk anchor rather than a previous target in the predicted sequence. Adjacent increments therefore measure local rotation direction and magnitude without allowing accumulated anchorrelative drift to dominate the metric.

For the learned policies, we first average evaluation records and the three inference seeds within each source episode. For the random references, we instead average 256 independently sampled action chunks per record before episode aggregation. We then average source episodes within each task and assign equal weight to the four tasks. The reported 95% confidence intervals use 20,000 stratified bootstrap resamples with source episode as the sampling unit, avoiding the treatment of correlated windows or stochastic samples as independent observations. For each evaluation record, the random reference samples every time step and each of the 20 active action channels independently from Uniform[−1, 1] in the corresponding task’s frozen UMI checkpoint normalization space, applies that checkpoint’s inverse quantile normalization, and passes the resulting chunk through the same physical reconstruction and scoring pipeline. The primary analysis excludes the two gripper channels because their acquisition semantics are not calibrated across interfaces. In particular, when no grasp is present, real-robot teleoperation defaults to a fully open gripper, whereas UMI retains the operator’s hand-gesture angle. Including the channels would therefore conflate pose-decoding fidelity with an interface diference in gripper behavior.

![](images/8f63372b58c7b4a8f7e1f4b13e6547d2124ac1bd25fbe21383efd4523ca4bede.jpg)

![](images/44ad518eb1c2d7871f40f192ea93dc99f93eac96d58b912abb5b3c5610f50a36.jpg)  
Figure 12 WAM pose-decoding fidelity under ground-truth-video conditioning. Points denote equal-task means across Stain Wiping, Shirt Folding, Remote Insertion, and Produce Sorting; whiskers denote 95% source-episode bootstrap confidence intervals. Translation is the bimanual XYZ RMSE over the complete first $H = 1 2$ chunk in the native chunk-anchor representation. Rotation is the SO(3) geodesic error between predicted and ground-truth adjacent-frame increments over $t = 2 , \ldots , 1 2 .$ Logarithmic axes preserve resolution among learned policies while retaining true-random actions as a common scale reference. Gripper channels are excluded because their physical semantics are not calibrated across the UMI and real-robot interfaces. Lower is better.

As shown in Figure 12, the UMI-post-trained policy achieves a full-chunk XYZ RMSE of 24.33 mm and a frame-to-frame SO(3) error of $0 . 6 5 ^ { \circ }$ on held-out real-robot observations. On episode-disjoint held-out UMI observations, the corresponding errors are 21.13 mm and $0 . 8 8 ^ { \circ }$ , resulting in cross-domain diferences of only 3.20 mm in translation and $0 . 2 3 ^ { \circ }$ in rotation. On the same held-out real-robot domain, the teleoperation-posttrained reference achieves 21.64 mm XYZ RMSE and $0 . 4 6 ^ { \circ } \mathrm { S O ( 3 ) }$ error. For comparison, true-random actions yield 117.57 mm and $1 2 6 . 4 7 ^ { \circ }$ on real-robot observations, and 123.80 mm and $1 2 6 . 4 9 ^ { \circ }$ on UMI observations Relative to the domain-matched Random→Real reference, UMI→Real reduces translation and rotation error by 79.3% and 99.5%, respectively. These results indicate that, when future-video generation is bypassed, both UMI- and teleoperation-post-trained pose decoders achieve comparable centimeter-level translation accuracy and sub-degree local rotation accuracy. The small gap between UMI→Real and UMI→UMI further suggests that the UMI-trained decoder generalizes across observation domains without substantial degradation.

## 6.3 Does Large-Scale UMI Pre-Training Yield a Better Base Model?

The two post-training tracks show that high-fidelity UMI demonstrations can replace real-robot teleoperation as the task-specific data source. We next isolate a separate question on StarVLA-QwenPI: whether the same robot-free data can support a reusable base model. We pre-train StarVLA-QwenPI on 4,000 hours of multitask UMI data. A useful initialization should improve not only prediction on held-out samples from this corpus, but also transfer to unseen tasks and real-robot post-training. We study these properties in turn.

![](images/6081e602201b73cdc22d7428d6dced62f7cef51389413929d2358101a9fbb127.jpg)

![](images/de30cb30c96dd11865530adbe96da7ee8a16f482f247ed102d425cd3af1ebaeb.jpg)  
Figure 13 Held-out action-prediction error during large-scale UMI pre-training. (a) Error across training, with the learning-rate decay shaded. (b) Pre-decay checkpoints on logarithmic axes with a power-law fit. Because the corpus is fixed, the fit measures exposure scaling rather than dataset-size scaling.

Scaling on held-out data. The 4,000-hour training mixture is drawn from the large-scale corpus described in Sec. 4 and spans diverse scenes, objects, and manipulation skills. We reserve a fixed set of action chunks for evaluation and use the same fixed-step Euler integration procedure for the flow-matching policy at every checkpoint. This isolates the efect of greater training exposure from changes in the evaluation data or sampler. One pass through the mixture ends at 180k steps, followed by a short learning-rate decay.

Figure 13 shows that held-out action error falls by 61% over one pass through the corpus. At a fixed model size and training recipe, we fit

$$
{ \mathcal { L } } _ { \mathrm { h e l d o u t } } ( S ) = { \mathcal { L } } _ { \infty } + A S ^ { - \alpha } ,\tag{9}
$$

where S denotes the cumulative number of UMI action chunks processed globally. The fitted exponent is $\alpha = 0 . 2 6 8$ , with $R ^ { 2 } = 0 . 9 9 3$ before learning-rate decay. This close log–log fit shows that the model continues to convert greater UMI exposure into better action prediction throughout the training pass. The trend is therefore not an artifact of the final decay schedule.

Transfer to unseen tasks. Lower error on the pre-training distribution does not by itself imply broader visual-motor competence. We therefore construct a balanced evaluation set from ten separately collected manipulation tasks that are absent from pre-training. The same action chunks and inference settings are used for every checkpoint, and the scaling fit uses only checkpoints before learning-rate decay.

Figure 14a shows a 41% reduction in mean OOD error, and every unseen task improves. Aggregate OOD error also follows a power-law trend, although with a smaller exponent of $\alpha = 0 . 0 9 5$ . The family curves reveal a clear ordering: utensil and tableware interactions improve fastest, granular transfer lies in the middle, and cloth folding improves more slowly. Scaling is therefore shared across task families, but its rate depends on which interaction patterns are represented during pre-training.

The task-level comparison in Fig. 14b makes this dependence more concrete. Rigid utensil-to-receptacle tasks have the lowest final error, while garment folding remains the most dificult. Granular transfer also changes markedly with target geometry. This structure mirrors the pre-training mixture: object and receptacle placement accounts for more than one third of its frames, while textile folding accounts for less than one percent. The model can therefore reuse abundant rigid-object pick-and-place experience for novel utensils, whereas cloth folding must extrapolate from sparse deformable-object supervision. OOD transfer is governed more by coverage of interaction dynamics than by whether the test object itself has appeared before.

This analysis also gives a concrete data-collection priority. Deformable-object data should cover more garment topologies, initial configurations, bimanual regrasps, and fold transitions. Granular-material data should vary container geometry, fill level, and motion type. The OOD benchmark thus serves as a diagnostic for rebalancing the next pre-training mixture, rather than only as a single aggregate score.

(a) Task-family OOD scaling  
![](images/a1ac98d50d70cf4f81c17cb8023efce52eccd40f67d872c9f3570cd006deccbc.jpg)

Cumulative UMI action chunks (million)  
(b) Improvement by unseen task  
![](images/04962bc48bc339715ae71db9bb79a2074f0c0442d46b6b07717f50a176507d27.jpg)  
Figure 14 Generalization to ten unseen UMI tasks. (a) Mean action MSE by manipulation family with pre-decay power-law fits; diamonds denote the final model export. (b) Task-level error at the first checkpoint and final export. Colors identify broad manipulation families.

Benefits for post-training. Ofline action error is useful only if it predicts a better starting point for deployment. On Remote Insertion, we compare with a Qwen-VL-initialized model whose action policy starts from scratch; the shared visual-language backbone isolates UMI’s visual-motor prior. With only 800 taskspecific episodes, the UMI-pretrained policy already exceeds the Qwen-VL-initialized baseline trained on four times as much data, as shown in Fig. 10b. Scaling to 1,600 episodes raises success to 80%, and the advantage remains at the matched 3,200-episode scale. The important pattern is that strong performance appears earlier and remains above the task-specific baseline: pre-training improves both data eficiency and the performance reached after post-training.

We next test whether this advantage extends beyond a single task. On each of the four benchmark tasks from Sec. 6.1.2, we post-train two StarVLA-QwenPI policies using the same 3,200 task-specific trajectories. One starts from Qwen-VL with a randomly initialized action policy, while the other starts from our UMI-pretrained checkpoint; the data split, action representation, normalization, optimization schedule, and deployment protocol are otherwise identical. Initialization is therefore the only controlled diference. $\mathrm { O p e n P I } { - \pi _ { 0 . 5 } }$ is included as a cross-architecture reference rather than as part of this controlled comparison.

Figure 15 shows a positive transfer efect on every benchmark task. UMI pre-training raises aggregate StarVLA-QwenPI success by 18.1 percentage points, with particularly strong gains on wiping, folding, and insertion. Because the architecture and post-training recipe are fixed, this improvement can be attributed to the visual-motor initialization rather than additional task-specific supervision. $\mathrm { O p e n P I } { - \pi _ { 0 . 5 } }$ provides context for the resulting performance level, but is not a controlled baseline.

The three evaluations connect pre-training exposure to deployment: prediction improves throughout the training pass, the same trend extends to unseen tasks, and the resulting checkpoint learns downstream tasks with less data. More importantly, the task-family analysis identifies why transfer difers across skills. Scaling the corpus is efective when it expands coverage of the interaction dynamics that the policy must later reuse.

![](images/c043b160122d2ecce40397c2bc186a4090442ccfb6075fb4982f67f990770d2b.jpg)  
Figure 15 Real-robot success after UMI-only post-training. StarVLA-QwenPI is initialized either from Qwen-VL or our UMI-pretrained checkpoint; OpenPI-π<sub>0.5</sub> provides a cross-architecture reference. Each task contains 40 rollouts per policy. Bold labels indicate the highest success rate within each group.

## 7 Discussion

## Empirical Findings and Implications

Across the four evaluated tasks and three tested backbones—two VLAs and one WAM—HiFi-UMI serves as the sole task-specific post-training source and achieves approximate aggregate parity with in-domain teleoperation. This finding concerns the source of task-specific post-training data rather than the complete pre-training history of the models: $\mathrm { O p e n P I } { - \pi _ { 0 . 5 } }$ and LingBot-VA retain their publicly released pre-trained initializations, while no target-task teleoperation data are introduced in the HiFi-UMI post-training conditions. Our additional HiFi-UMI pre-training is not required for this aggregate post-training result; separately, on StarVLA-QwenPI, pre-training on 4,000 hours of HiFi-UMI further improves post-training data eficiency and final deployment performance, shifting the eficiency–performance frontier upward. These results also suggest that, once task-specific demonstrations are suficiently action-aligned, data composition and coverage may become as important as volume alone. Characterizing which interaction dynamics a deployable policy most depends on, and when that coverage saturates, remains a central open question.

## Limitations and Future Work

Evaluation scope and generality. Our zero-robot post-training evidence covers four tabletop bimanual tasks and three backbones under scene-level distribution shift; its generality to other tasks, embodiments, and shifts remains untested. The pre-training evidence is narrower: scaling and downstream gains are measured only on StarVLA-QwenPI. Broader tasks and embodiments, together with additional VLA and WAM backbones, are needed to establish the scope of both findings.

Task-level statistical resolution. Each task–policy pair has 40 rollouts, so one additional success changes the estimated rate by 2.5 percentage points. The resulting uncertainty limits fine-grained task-level comparisons, because a few outcomes can reverse the ordering of methods separated by small gaps. We therefore treat per-task diferences as descriptive and base the parity claim on aggregate evidence across tasks. More trials per condition would narrow the uncertainty and allow task-level efects to be characterized more precisely.

Fidelity is validated as a whole, not decomposed. We treat fidelity as a design principle realized jointly by trajectory accuracy, inter-gripper relative pose, synchronization, and field of view, and we do not isolate these factors through controlled degradation. Our results thus show that high fidelity sufices, but not how much of each property a deployable policy requires. A systematic ablation—selectively degrading each factor while holding sample count and scene coverage fixed—would quantify its marginal contribution and turn “high fidelity helps” into an actionable specification of the fidelity required for deployment.

Data eficiency and transfer across post-training sources. Our parity comparison is not samplematched: without pre-training, UMI-only post-training uses roughly ten times as many demonstrations as the teleoperation baseline, so the result compares practical data-production pipelines rather than per-trajectory eficiency. Large-scale UMI pre-training sharply improves both eficiency and attainable performance, but two questions remain: whether the gains continue as the pre-training corpus grows or eventually saturate, and whether the same initialization benefits post-training on real-robot teleoperation data as much as it benefits post-training on UMI data. Testing both at larger scales would distinguish a generally reusable initialization efect from a benefit specific to matched-domain training.

## 8 Conclusion

We revisit the assumption that robot-free demonstrations can seed but not finish a policy without a teleoperated post-training anchor. We argue that the limitation is fidelity, not the robot-free setting. HiFi-UMI tests this claim with portable capture co-designed for trajectory accuracy, inter-gripper relative pose, synchronization, and field of view; its pipeline has generated over 20,000 hours.

Across three VLA and WAM backbones, policies post-trained solely on HiFi-UMI match teleoperation within roughly 3 percentage points, with diferences of both signs within sampling noise, and reach 85% on precision insertion despite scene shift and zero teleoperated data. Separately, 4,000 hours of pre-training cut action error on ten unseen tasks by 41% and raise real-robot success by 18.1 percentage points at matched posttraining data, reaching the scratch-initialized baseline with one quarter of the task data. Thus, robot-free data of this fidelity can support deployable manipulation, not just pre-training. We release HiFi-UMI-2K, a 2,000-hour, microsecond-synchronized, replayable, ultra-wide-FoV subset of that corpus.

## Author Contributions

Core Contributors

Yuteng Wei<sup>\*</sup>, Jinming Ma<sup>\*</sup>, Jiawei Wang<sup>\*†</sup>, Weitao Zhou<sup>\*†</sup>, Yushen Zuo, Ke Rui, Minglei Li<sup>†</sup> <sup>✉</sup>.

Contributors

Jinhao Zhang, Zhikang Pan, Xiang Wang, Haoran Jia, Huan Du, Zicheng Zeng, Jun Ma, Guiyu Qin, Di Zhang, Xiaofei Li.

<sup>\*</sup>Equal contribution. <sup>†</sup>Project leaders. <sup>✉</sup>Corresponding author.

Correspondence: liminglei@simpleai.tech.

## References

[1] Anthony Brohan, Noah Brown, Justice Carbajal, et al. RT-1: Robotics transformer for real-world control at scale. In Robotics: Science and Systems (RSS), 2023.

[2] Homer Walke, Kevin Black, Abraham Lee, et al. BridgeData V2: A dataset for robot learning at scale. In Conference on Robot Learning (CoRL), 2023. arXiv:2308.12952.

[3] Alexander Khazatsky, Karl Pertsch, Suraj Nair, et al. DROID: A large-scale in-the-wild robot manipulation dataset. In Robotics: Science and Systems (RSS), 2024. arXiv:2403.12945.

[4] AgiBot-World-Contributors, Qingwen Bu, Jisong Cai, et al. AgiBot World Colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025.

[5] Kun Wu, Chengkai Hou, Jiaming Liu, et al. RoboMIND: Benchmark on multi-embodiment intelligence normative data for robot manipulation. arXiv preprint arXiv:2412.13877, 2024.

[6] Cheng Chi, Zhenjia Xu, Chuer Pan, et al. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. In Robotics: Science and Systems (RSS), 2024. arXiv:2402.10329.

[7] Zhaxizhuoma, Kehui Liu, Chuyue Guan, et al. FastUMI: A scalable and hardware-independent universal manipulation interface with dataset. arXiv preprint arXiv:2409.19499, 2024.

[8] Fanqi Lin, Yingdong Hu, Pingyue Sheng, et al. Data scaling laws in imitation learning for robotic manipulation. In International Conference on Learning Representations (ICLR), 2025. arXiv:2410.18647.

[9] Hongjie Fang, Chenxi Wang, Yiming Wang, et al. AirExo-2: Scaling up generalizable robotic imitation learning with low-cost exoskeletons. arXiv preprint arXiv:2503.03081, 2025.

[10] Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, et al. GR00T N1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

[11] Hongzhe Bi, Lingxuan Wu, Tianwei Lin, et al. H-RDT: Human manipulation enhanced bimanual robotic manip ulation. arXiv preprint arXiv:2507.23523, 2025.

[12] Qiyuan Zeng, Chengmeng Li, Jude St. John, et al. ActiveUMI: Robotic manipulation with active perception from robot-free human demonstrations. arXiv preprint arXiv:2510.01607, 2025.

[13] James Wang, Primo Pu, Zephyr Fung, et al. XRZero-G0: Pushing the frontier of dexterous robotic manipulation with interfaces, quality and ratios. arXiv preprint arXiv:2604.13001, 2026.

[14] Songming Liu, Bangguo Li, Kai Ma, et al. RDT2: Exploring the scaling limit of UMI data towards zero-shot cross-embodiment generalization. arXiv preprint arXiv:2602.03310, 2026.

[15] Hao-Shu Fang, Hongjie Fang, Zhenyu Tang, et al. RH20T: A comprehensive robotic dataset for learning diverse skills in one-shot. arXiv preprint arXiv:2307.00595, 2023.

[16] Open X-Embodiment Collaboration, Abby O’Neill, Abdul Rehman, et al. Open X-embodiment: Robotic learning datasets and RT-X models. In IEEE International Conference on Robotics and Automation (ICRA), 2024. arXiv:2310.08864.

[17] Chengkai Hou, Kun Wu, Jiaming Liu, et al. RoboMIND 2.0: A multimodal, bimanual mobile manipulation dataset for generalizable embodied intelligence. arXiv preprint arXiv:2512.24653, 2025.

[18] Kristen Grauman, Andrew Westbury, Eugene Byrne, et al. Ego4D: Around the world in 3,000 hours of egocentric video. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. arXiv:2110.07058.

[19] Kristen Grauman, Andrew Westbury, Lorenzo Torresani, et al. Ego-Exo4D: Understanding skilled human activity from first- and third-person perspectives. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. arXiv:2311.18259.

[20] Ryan Hoque, Peide Huang, David J. Yoon, et al. EgoDex: Learning dexterous manipulation from large-scale egocentric video. arXiv preprint arXiv:2505.11709, 2025.

[21] Kehui Liu, Zhongjie Jia, Yang Li, et al. FastUMI-100K: Advancing data-driven robotic manipulation with a large-scale UMI-style dataset. arXiv preprint arXiv:2510.08022, 2025.

[22] Siyuan Yang, Linzheng Guo, Ouyang Lu, et al. VISTA: Vision-grounded and physics-validated adaptation of UMI data for VLA training. arXiv preprint arXiv:2606.04708, 2026.

[23] Chen Wang, Haochen Shi, Weizhuo Wang, et al. DexCap: Scalable and portable mocap data collection system for dexterous manipulation. In Robotics: Science and Systems (RSS), 2024. arXiv:2403.07788.

[24] Mengda Xu, Han Zhang, Yifan Hou, et al. DexUMI: Using human hand as the universal manipulation interface for dexterous manipulation. In Conference on Robot Learning (CoRL), 2025. arXiv:2505.21864.

[25] Tony Tao, Mohan Kumar Srirama, Jason Jingzhou Liu, et al. DexWild: Dexterous human interactions for in-the-wild robot policies. Robotics: Science and Systems (RSS), 2025.

[26] Sirui Chen, Chen Wang, Kaden Nguyen, et al. ARCap: Collecting high-quality human demonstrations for robot learning with augmented reality feedback. In IEEE International Conference on Robotics and Automation (ICRA), 2025. arXiv:2410.08464.

[27] Hongjie Fang, Hao-Shu Fang, Yiming Wang, et al. AirExo: Low-cost exoskeletons for learning whole-arm manipulation in the wild. In IEEE International Conference on Robotics and Automation (ICRA), 2024. arXiv:2309.14975.

[28] Simar Kareer, Dhruv Patel, Ryan Punamiya, et al. EgoMimic: Scaling imitation learning via egocentric video. In IEEE International Conference on Robotics and Automation (ICRA), 2025. arXiv:2410.24221.

[29] Anthony Brohan, Noah Brown, Justice Carbajal, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning (CoRL), 2023. arXiv:2307.15818.

[30] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, et al. OpenVLA: An open-source vision-language-action model. In Conference on Robot Learning (CoRL), 2024. arXiv:2406.09246.

[31] Octo Model Team, Dibya Ghosh, Homer Walke, et al. Octo: An open-source generalist robot policy. In Robotics: Science and Systems (RSS), 2024. arXiv:2405.12213.

[32] Kevin Black, Noah Brown, Danny Driess, et al. π<sub>0</sub>: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[33] Physical Intelligence, Kevin Black, Noah Brown, et al. π<sub>0.5</sub>: A vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[34] ByteDance Seed. GR-3 technical report. arXiv preprint arXiv:2507.15493, 2025.

[35] Google DeepMind. Gemini robotics: Bringing AI into the physical world. arXiv preprint arXiv:2503.20020, 2025.

[36] Mustafa Shukor, Dana Aubakirova, Francesco Capuano, et al. SmolVLA: A vision-language-action model for afordable and eficient robotics. arXiv preprint arXiv:2506.01844, 2025.

[37] Karl Pertsch, Kyle Stachowicz, Brian Ichter, et al. FAST: Eficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.

[38] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. Robotics: Science and Systems (RSS), 2025. arXiv:2502.19645.

[39] Wei Wu, Fan Lu, Yunnan Wang, et al. A pragmatic VLA foundation model. arXiv preprint arXiv:2601.18692, 2026.

[40] Qiuyue Wang, Mingsheng Li, Jian Guan, et al. Qwen-VLA: Unifying vision-language-action modeling across tasks, environments, and robot embodiments. arXiv preprint arXiv:2605.30280, 2026.

[41] Haoqi Yuan, Zhixuan Liang, Anzhe Chen, et al. Qwen-RobotManip technical report: Alignment unlocks scale for robotic manipulation foundation models. arXiv preprint arXiv:2606.17846, 2026.

[42] Hongtao Wu, Ya Jing, Chilam Cheang, et al. Unleashing large-scale video generative pre-training for visual robot manipulation. In International Conference on Learning Representations (ICLR), 2024. arXiv:2312.13139.

[43] Chi-Lam Cheang, Guangzeng Chen, Ya Jing, et al. GR-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024.

[44] Yilun Du, Mengjiao Yang, Bo Dai, et al. Learning universal policies via text-guided video generation. In Advances in Neural Information Processing Systems (NeurIPS), 2023. arXiv:2302.00111.

[45] Yucheng Hu, Yanjiang Guo, Pengchao Wang, et al. Video prediction policy: A generalist robot policy with predictive visual representations. In International Conference on Machine Learning (ICML), 2025. arXiv:2412.14803.

[46] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026.

[47] Jun Cen, Chaohui Yu, Hangjie Yuan, et al. WorldVLA: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539, 2025.

[48] Lin Li, Qihang Zhang, Yiming Luo, et al. Causal world modeling for robot control. arXiv preprint arXiv:2601.21998, 2026.

[49] StarVLA Community. StarVLA: A lego-like codebase for vision-language-action model developing. arXiv preprint arXiv:2604.05014, 2026. Code available at: https://github.com/starVLA/starVLA.

[50] Tong Qin, Peiliang Li, and Shaojie Shen. VINS-Mono: A robust and versatile monocular visual-inertial state estimator. IEEE Transactions on Robotics, 34(4):1004–1020, 2018.

[51] Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, et al. ORB-SLAM3: An accurate open-source library for visual, visual-inertial, and multimap SLAM. IEEE Transactions on Robotics, 37(6):1874–1890, 2021. arXiv:2007.11898.

[52] Edwin Olson. AprilTag: A robust and flexible visual fiducial system. In IEEE International Conference on Robotics and Automation (ICRA), pp. 3400–3407, 2011.

[53] Florian Tschopp, Michael Riner, Marius Fehr, et al. VersaVIS—an open versatile multi-camera visual-inertial sensor suite. Sensors, 20(5):1439, 2020. arXiv:1912.02469.

[54] GenRobot AI. DAS Fingers: A high-precision multimodal data-collection device for embodied AI. https: //www.genrobot.ai/products/finger, 2025. Product page. Accessed: 2026-07-08.

[55] Tailai Cheng, Kejia Chen, Lingyun Chen, et al. TacUMI: A multi-modal universal manipulation interface for contact-rich tasks. arXiv preprint arXiv:2601.14550, 2026.

[56] Berta Bescos, José M. Fácil, Javier Civera, et al. DynaSLAM: Tracking, mapping, and inpainting in dynamic scenes. IEEE Robotics and Automation Letters, 3(4):4076–4083, 2018. arXiv:1806.05620.

[57] Huy Ha, Yihuai Gao, Zipeng Fu, et al. UMI on legs: Making manipulation policies mobile with manipulationcentric whole-body controllers. arXiv preprint arXiv:2407.10353, 2024.

[58] Pierre Sermanet, Tianli Ding, Jefrey Zhao, et al. RoboVQA: Multimodal long-horizon reasoning for robotics. In IEEE International Conference on Robotics and Automation (ICRA), pp. 645–652, 2024. arXiv:2311.00899.

[59] Yi Zhou, Connelly Barnes, Jingwan Lu, et al. On the continuity of rotation representations in neural networks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. arXiv:1812.07035.

[60] Qwen Team. Qwen3 Technical Report. arXiv preprint arXiv:2505.09388, 2025.

[61] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, et al. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023. arXiv:2210.02747.

[62] William Peebles and Saining Xie. Scalable difusion models with transformers. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023. arXiv:2212.09748.

[63] Tony Z. Zhao, Vikash Kumar, Sergey Levine, et al. Learning fine-grained bimanual manipulation with low-cost hardware. In Robotics: Science and Systems (RSS), 2023.

[64] Cheng Chi, Siyuan Feng, Yilun Du, et al. Difusion policy: Visuomotor policy learning via action difusion. In Robotics: Science and Systems (RSS), 2023.

[65] Lucas Beyer, Andreas Steiner, André Susano Pinto, et al. PaliGemma: A versatile 3b VLM for transfer. arXiv preprint arXiv:2407.07726, 2024.

[66] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2019. arXiv:1711.05101.

[67] Jose Barreiros, Andrew Beaulieu, Aditya Bhat, et al. A careful examination of large behavior models for multitask dexterous manipulation. Science Robotics, 11(113):eaea6201, 2026.