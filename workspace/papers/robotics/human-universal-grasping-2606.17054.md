Predict human grasps

# Human Universal Grasping

Kevin Yuanbo Wu1,† Tianxing Zhou1,2 Isaac Tu1 Billy Yan1 Irmak Guzey1 David Fouhey1 Dandan Shan1,3,‡ Lerrel Pinto1,‡

1New York University 2Tsinghua University 3University of Michigan

https://grasping.io

Abstract: Humans can grasp objects effortlessly, whereas multi-fingered robots are far from this level of generality. We argue that the most natural source of robot grasping data is from humans, who pick up thousands of objects every day. We present HUG, a flow-matching model that generates diverse human grasps for any user-specified object in a single RGB-D image captured from a stereo camera. Using smart glasses, we first collect 1M-HUGS, an egocentric dataset of human grasps spanning 1M frames (27.8 hrs) and 6,707 object instances across 41 buildings. Next, to model the distribution of natural human grasps, our novel flow-matching model fuses RGB and depth observations to output a grasp parameterized by wrist translation, wrist rotation, and MANO hand pose. Predicted grasps can be retargeted to various robot hands, enabling zero-shot grasping in everyday scenes. To standardize evaluation, we build a new simulated benchmark, HUG-BENCH, of 90 unseen objects from five geometric categories and various sizes, with metric-scale 3D meshes. We evaluate HUG in the real world on the 30-object test set of HUG-BENCH across multiple stereo cameras, robot embodiments, and household environments. HUG outperforms the state-of-the-art grasping baselines by +23% and +34% on our challenging object set. Code, data, benchmark, checkpoints, and an interactive demo are released on our website.

Keywords: Learning from Humans, Dexterous Grasping

Train on human grasps  
![](images/1c424d2206b73213c965992b94ab21d2918aeafe7c88f459d9587c54c8a3f78f.jpg)  
Figure 1: HUG learns dexterous grasping without any robot data. Trained solely on egocentric human grasp data, HUG generates diverse human grasps for real-world objects in a single RGBD image captured from a stereo camera, which can be retargeted to robot hands for zero-shot, in-thewild dexterous grasping.

## 1 Introduction

Grasping arbitrary objects unlocks downstream manipulation, from sorting groceries to operating tools. Humans do this effortlessly, yet current models for dexterous robot hands remain far from such generality. A major bottleneck is data: robots need the diverse, real-world grasping experience humans accumulate daily.

Prior work attacks this from two angles. Synthetic grasps are produced in simulation by optimizing analytic objectives like force-closure [1, 2], sampling learned generators or RL policies [3, 4], or reconstructing hands from web images [5, 6]; they suffer the sim-to-real gap and usually require retraining for each new hand. Teleoperation [7, 8, 9, 10] yields real grasps on the target embodiment but is tedious and cannot cover the open world.

We instead train on in-the-wild human grasps, modeling how people naturally grasp objects rather than every physically valid grasp, yielding reliably executable grasps. Two recent advances make this practical. First, lightweight egocentric sensors like Aria Gen 2 [11] stream calibrated RGB-D and hand tracking, making grasp collection as easy as wearing glasses. Second, anthropomorphic robot hands [12, 13, 14, 15] and learned retargeting [10, 16, 17, 18] have narrowed the human-robot morphology gap, facilitating direct robot deployment. Together they unlock a previously infeasible pipeline: collect human grasps at scale, learn from them, and retarget for deployment.

We demonstrate this with HUG (Human Universal Grasping), which generates diverse human grasps for objects in a single RGB-D image captured from a stereo camera that can be retargeted to robot hands. HUG has three steps: (1) collect 1M-HUGS, an egocentric dataset of 1M image-grasp pairs from 6,707 recordings captured with Aria Gen 2 across 41 buildings; (2) train a flow-matching model mapping objects in an RGB-D to a MANO [19] grasp; (3) retarget the predicted grasp to robot hands with no per-embodiment training.

We introduce HUG-BENCH, 90 challenging unseen objects spanning five geometric categories and three size bins. Unlike simulation-only benchmarks, HUG-BENCH starts from real objects, reconstructs each into a metric-scale mesh from egocentric recordings, and evaluates in both simulation and the real world. On the 30 object test set in the real world, HUG reaches 66.7% tabletop success, beating baselines by +23% and +34%, and 62.0% in-the-wild, generalizing zero-shot across stereo cameras, robot hands, and unseen homes.

In summary, our contributions are the following, all of which we open-source:

1. To our knowledge, HUG is the first grasping framework trained purely on human data and deployable across multiple robot embodiments.

2. Dataset. 1M-HUGS, 1M egocentric image-grasp pairs of natural human grasps across 6,707 recordings and 41 buildings, with MANO-fit hand poses and metric depth (§ 3).

3. Method. HUG, a point-conditioned flow-matching model that predicts MANO grasps from RGB-D and retargets to multiple embodiments without per-hand training (§ 4).

4. Benchmark. HUG-BENCH, 90 unseen objects with metric-scale 3D meshes for paired simulation and real-world evaluation (§ 5).

## 2 Related Work

Robotic object grasp prediction. Grasp prediction is long-standing in robotics. Early work targeted two-fingered grippers via self-supervised collection [20] or large-scale datasets [21, 22, 23, 24, 25], achieving strong real-world performance [26, 27]. However, multi-fingered hands remain harder. As dexterous teleoperation data is hard to collect, prior work mostly trains in simulation, via reinforcement learning [28, 29, 30, 31] or generative grasp synthesis [3, 32, 33], both of which struggle with sim-to-real gaps and require retraining per robot. Like HUG, these methods [31, 32, 33] use 3D representations, but rely on a complete object point cloud, which hinders generalization during real-world deployment. HUG instead predicts from single-view camera-frame RGB-D and, trained on human grasping data, is scalable, generalizable, and easily retargeted to robot hands.

![](images/456b26a416cc0fbc50e5d106324d0efa0c8efbd69459d98c9bd43252db8b869c.jpg)  
Figure 2: 1M-HUGS dataset. Our training data comprises 1M egocentric frames of human grasps, spanning 6,707 object instances. Each entry provides synchronized RGB and grayscale views, metric depth, an object mask, and a MANO hand pose with wrist transformation in the camera frame.

Robot learning from non-robot datasets. Given the difficulty of collecting robot-specific data, recent work learns robot behaviors from non-teleoperated datasets [27, 34, 35, 36, 37, 38] via advances in motion tracking [39, 40] and hand-object reconstruction [41, 42] and interaction synthesis [43]. Early efforts used in-domain human data with rich 3D annotations [38, 44, 45, 46], but tying collection to deployment limited scalability. In-the-wild human datasets [5, 47, 48, 49, 50, 51] scale broadly but lack the precise 3D signals for reliable policies, requiring downstream engineering [50, 51] or co-training with robot data [52, 53]. Smart glasses [11, 54] bridge this gap by capturing in-the-wild data with stereo depth and accurate 3D hand poses. Most of these efforts focus on two-finger grippers [55], and some additionally rely on robot data for training [56]. Unlike these, HUG learns multi-fingered dexterous grasping from in-the-wild human data alone, with no robot data, and retargets to multiple robot embodiments at deployment.

## 3 1M-HUGS Dataset

1M-HUGS captures a diverse set of natural human grasps with egocentric smart glasses in everyday environments (Figure 2), differing from existing datasets with grasps from simulation or lab settings (Table 1). This section presents the collection protocol, filtering pipeline, and resulting dataset.

Video recording protocol. We record with Aria Gen 2 glasses [11], which provide synchronized RGB and stereo grayscale views together with 6-DoF camera poses and 3D hand landmarks. In one recording, the wearer stands in front of a target object and moves their head for 15-30 seconds, capturing the static scene from diverse viewpoints without hands visible, before reaching in with their right hand and grasping the object. The grasp pose is then propagated back into preceding no-hand frames with Aria Gen 2’s camera poses, so a single physical grasp yields hundreds of (object-only image, grasp) training pairs from diverse viewpoints at no additional annotation cost.

Curation. Before filtering, each recording is localized to the grasped object and verified. A vision-language model identifies the grasped object, SAM3 [58] propagates its mask across all frames, and stability and proximity heuristics select the grasp frame. Every recording is then human-reviewed in a web interface before it enters the dataset. We detail the full pipeline in Appendix B.1 and B.2.

Frame filtering. We evaluate frames against five criteria: (i) the object mask is non-empty; (ii) stereo depth with S2M2 [59] marks ≥ 60% of the depth map as confident; (iii) the 2D pro-

Table 1: Comparison of grasping datasets. 1M-HUGS captures real in-the-wild human grasps and automatically yields many (image, grasp) pairs from dynamic views by back-propagating grasp across the no-hand frames.
<table><tr><td>Dataset</td><td>Real data</td><td>H. Grasp</td><td># Obj.</td><td>#I-G pair</td></tr><tr><td>DexGraspNet [1]</td><td>×</td><td>×</td><td>5.4K</td><td>1.3M</td></tr><tr><td>Dex1B [3]</td><td>×</td><td>×</td><td>4.4K</td><td>1B</td></tr><tr><td>Web2Grasp [5]</td><td>(web)</td><td>×</td><td>1K</td><td>2.1K</td></tr><tr><td>AnyDexGrasp [2]</td><td>&lt;(lab)</td><td>?</td><td>144</td><td>10K</td></tr><tr><td>DexYCB [57]</td><td>(lab)</td><td>r</td><td>20</td><td>1K</td></tr><tr><td>1M-HUGs (Ours)</td><td>√(wild)</td><td>「</td><td>~1.5K</td><td>1.0M</td></tr></table>

jection of the grasp’s hand landmarks intersects the object mask; (iv) at least five grasp landmarks lie within the image bounds; and (v) there is no hand in the current frame. Each surviving frame is cropped and resized to 224×224. We provide more details in Appendix B.3.

![](images/965d287a84111e341a404d4e285eb6d71c5ff226ee262fb84de73d7d95756f24.jpg)  
Figure 3: HUG architecture. Conditioned on an RGB-D image and a query point on the target object, HUG predicts MANO hand grasps via a flow-matching transformer over fused RGB and point cloud features. Predicted human grasps are then retargeted to robot hands.

Dataset statistics and labels. The dataset contains 6,707 recordings across 41 buildings with an estimated ∼1.5K unique objects. Within each building we grasp whatever we can find, spanning hundreds of distinct environments (kitchens, bedrooms, etc.). After filtering, 1M RGB and 1M grayscale frames (from the left stereo camera) remain, for 2M training entries; each grayscale frame shares its timestep with an RGB frame and lets the model generalize to monochrome cameras. Each entry contains a 224×224 image, camera intrinsics, depth map, object mask, and the grasp pose in the camera frame. Since Aria provides only 21 hand landmarks, we fit a full articulated MANO [19] hand to them (details in Appendix A). MANO decouples hand geometry into a shape parameter $\beta$ (overall hand size and proportions) and a pose parameter θ (joint articulation). MANO is convenient for three reasons. First, with $\beta$ fixed to a canonical hand size, the same θ denotes the same grasp across collectors, removing per-person hand size variation. Second, the resulting articulated mesh can be loaded into simulation to execute predicted grasps (§ 5.2). Third, fitting under an anatomicalvalidity prior [60] regularizes grasps into physically valid poses. The full pipeline, from raw Aria recordings through stereo depth, MANO fitting, and dataset construction, is released as aria2mano.

## 4 Method

Figure 3 shows HUG, a flow-matching model that, trained on real-world human grasps, generates diverse natural grasps for any user-specified object in a single RGB-D image from a stereo camera.

## 4.1 Model Architecture

Given an RGB-D observation and a 2D pixel click (u, v) on the target object, HUG predicts a 99- dim grasp state $\mathbf { x } = [ \mathbf { t } , \mathbf { R } _ { 6 \mathrm { d } } , \pmb { \theta } _ { 6 \mathrm { d } } ] \in \mathbb { R } ^ { 9 9 }$ , where $\mathbf { t } \in \mathbb { R } ^ { 3 }$ is the wrist translation in the camera frame in the OpenCV convention, $\mathbf { R } _ { 6 \mathrm { d } } \in \mathbb { R } ^ { 6 }$ is the global wrist rotation in the continuous 6D rotation representation of Zhou et al. [61], and $\theta _ { 6 \mathrm { d } } ~ \in ~ \mathbb { R } ^ { 1 5 \times 6 }$ collects the 6D rotations of the 15 MANO finger joints. The depth image is back-projected to a metric point cloud (PC), and the click is lifted to a 3D query point $\mathbf { p } _ { q } \in \mathbb { R } ^ { 3 }$ using its depth value and the camera intrinsics K. The MANO shape $\beta \in \mathbb { R } ^ { 1 0 }$ is held fixed at a single canonical value, so the network predicts only articulation and placement; we recompute all grasps in 1M-HUGS and HUG-BENCH to this shape for consistency (Appendix B.3).

Encoders. The RGB image is encoded with a frozen DINOv2-Base ViT with register tokens [62, 63], producing N = 256 patch tokens. The metric point cloud is cropped to a 0.3 m radius ball around the 3D query point ${ \bf p } _ { q }$ . From this crop, $N _ { p } = 4 0 9 6$ points are sampled and passed to a trainable PointNeXt [64] U-Net that outputs $N = 2 5 6$ per-region tokens together with their metric XYZ centroids $\{ { \mathbf { c } } _ { i } \} _ { i = 1 } ^ { N }$ . Cropping keeps the tokens dense around the target object; we set the radius to 0.3 m, roughly the largest object graspable with one hand (Appendix C.1).

![](images/878d8ac565959f3ad886bee8b8a92cf6ae19b36d7d076b605aa0ae4f4fbff617.jpg)  
Figure 4: Predicted grasps on HUG-BENCH. HUG’s predicted grasps for 30 unseen objects across six scenes of the HUG-BENCH test split. HUG generalizes across a variety of object shapes and sizes, environments, and camera viewpoints. Top row: small 2, medium 1, large 1. Bottom row: large 2, medium 2, small 1. See Appendix Table 6 for the objects in each scene.

RGB-PC fusion transformer. We fuse the two encoder streams with point painting: each point cloud centroid is projected to the RGB image via camera intrinsics K, its DINOv2 patch feature is bilinearly sampled, concatenated with the corresponding PC token, and projected by a two-layer MLP into a $D _ { f } = 1 0 2 4$ -dim fused token $\mathbf { f } _ { i \cdot }$ (Without point painting, the two streams are concatenated into 2N tokens; we ablate this in § 5.2.) Both the query point ${ \bf p } _ { q }$ and the PC centroids {ci} are encoded with a shared random Fourier feature encoder γ(·) [65] to retain metric information. The fused tokens $\{ \mathbf { f } _ { i } \} _ { i = 1 } ^ { N }$ cross-attend to the query token $\mathbf { q } = \mathrm { M L P } ( \gamma ( \mathbf { p } _ { q } ) )$ and are refined by a 4- layer pre-norm transformer, producing the scene-conditioning tokens $\mathbf { s } \in \mathbb { R } ^ { N \times D _ { f } }$ used by the flow transformer. Since K enters HUG only through back-projection and projection, never as a learned parameter, the model transfers across stereo cameras with different intrinsics (Appendix C.2).

Flow transformer. We split the grasp state into three distinct groups: translation (3-dim), wrist rotation (6-dim), and finger pose (90-dim). Each group is normalized and projected into a $D _ { m } = 5 1 2 \cdot$ dim token; separate tokens keep geometrically distinct components from over-mixing and balance the gradient signal across groups. These tokens are passed through L= 6 DiT [66] blocks that crossattend to the scene tokens s and are conditioned on the timestep via AdaLN-Zero modulation. Three linear heads decode the output tokens back to 3, 6, and 90 dims, and the velocity is integrated by the flow-matching ODE in normalized space, with the resulting clean state de-normalized to produce the predicted grasp state.

## 4.2 Training

Loss. We combine a velocity-prediction MSE ${ \mathcal { L } } _ { \mathrm { v } }$ with geometric supervision: we recover the predicted clean state $\hat { \mathbf { x } } _ { 0 } = \mathbf { x } _ { t } - t f _ { \phi } ( \mathbf { x } _ { t } , t , \mathbf { s } )$ , pass it through MANO, and supervise the resulting 3D hand landmarks in the camera frame with an L1 loss $ { \mathcal { L } } _ { \mathrm { 3 D } }$ , using $\lambda _ { \mathrm { v } } = 1$ and $\lambda _ { \mathrm { 3 D } } = 2 0$ :

$$
\mathcal { L } = \lambda _ { \mathrm { v } } \mathcal { L } _ { \mathrm { v } } + \lambda _ { \mathrm { 3 D } } \left( 1 - t \right) \mathcal { L } _ { \mathrm { 3 D } } ,\tag{1}
$$

where the (1−t) weight concentrates the geometric loss on near-clean steps where $\hat { \mathbf { x } } _ { 0 }$ is meaningful.

Training details. We train for 100K steps with AdamW at learning rate $1 0 ^ { - 4 }$ and batch size 128, using a 5K-step linear learning rate warmup. Only the PointNeXt encoder, RGB-PC fusion module, and flow transformer are optimized. Training timesteps are drawn uniformly from [0, 1], and at inference we generate samples with 50-step Euler integration of the learned ODE. We keep an EMA from step 50K, validate in MuJoCo [67] (§ 5.2) every 5K steps. Training with DDP on two RTX 5090s (batch size 64 per GPU) takes ∼10 h including MuJoCo validation (Appendix C.3).

![](images/dbcb3fde391a4aacee751706a5c503fdf8f9a5d7c95dd78ea5c21d048b04f413.jpg)  
Figure 5: Real world grasping with HUG. Grasp executions on unseen objects from HUG-BENCH test split in an unseen home, performed by a YOR mobile manipulator equipped with WUJI hands.

## 5 Experiments

We introduce HUG-BENCH (§ 5.1), then evaluate HUG in simulation (§ 5.2) and real-world (§ 5.3).

## 5.1 HUG-BENCH

Evaluation dataset. We propose a difficult-to-grasp set of objects for evaluation. HUG-BENCH comprises 90 objects spanning five geometric categories (cylindrical, spheroidal, prismatic, appendaged, amorphous) and three size bins (small, medium, large), with six objects per class combination (four val, two test). These objects are hard to grasp: many objects are articulated, very short (∼1 cm tall), or large and unwieldy, requiring diverse grasp poses and precise spatial accuracy for successful grasping. All 90 objects are unseen during training. The 30 test objects for real-world evaluation (Figure 6) can be purchased on Amazon for ∼250 USD, linked on our website. The val objects are used only in simulation for checkpoint selection, so only the unseen test objects are purchased for unbiased real-world evaluation.

Metric-scale simulation assets. To produce simulationready evaluation assets of the HUG-BENCH objects at metric scale, we extend Multi-view SAM3D (MV-SAM3D) [68], injecting Aria camera intrinsics, extrinsics, and stereo depth into their pipeline. We collect scans of 18 scenes, with five objects per scene using Aria Gen 2

![](images/826e7c05669f7f67f5b8e4aa22886e74ef10f95114dd258c1a3178cc3ae40f21.jpg)

![](images/2c5e493d1b4e7bc0d15d09dba3433b884b6bb9deaf38da14e58827554e32acb8.jpg)  
Figure 6: HUG-BENCH test split. 30 unseen objects (top) from 5 geometric categories × 3 size bins (2 per class), with their simulation assets (bottom).

glasses. We pass in five spread-out views per scene to the modified MV-SAM3D. We inspect the resulting meshes in Viser [69], editing their scale and pose to align them with the Aria Gen 2 SLAM semidense point cloud, stereo depth points, and their 2D projections onto the input views. Lastly, we make the meshes watertight with Alpha Wrap [70] and obtain their convex decomposition with CoACD [71] for simulation. For each object we also record 10 human grasps with Aria Gen 2, which we replay as a ground-truth human grasp oracle (§ 5.2). We release this scan-to-asset pipeline, together with the MuJoCo environment, as aria2mesh (Appendix D).

Simulation environment. Each predicted grasp is evaluated in MuJoCo [67] (Figure 8) on a fresh single-object scene with the target resting on a table under gravity. A position-actuated MANO right hand (Appendix E.1) executes an open-loop pre-grasp → grasp → lift rollout with gravity on. The pre-grasp pose offsets the predicted grasp by 3 cm along the two non-lateral wrist axes with fingers opened; the wrist then linearly interpolates to the predicted pose while the fingers close to their predicted MANO joint angles, with a small extra flexion to apply force. Then, the wrist lifts straight up by 0.5 m. A grasp succeeds if the object is no longer in contact with the surface after the lift.

<table><tr><td></td><td colspan="2">HUG-BENCH val</td><td colspan="2">HUG-BENCH test</td></tr><tr><td>Method</td><td> $S \mathbb { R } \left( \% \right) \uparrow$ </td><td> $\operatorname { F C } \operatorname { e r r o r } \left( \operatorname { m m } \right) \downarrow$ </td><td> $\mathrm { S R } \left( \% \right) \uparrow$ </td><td> $\operatorname { F C } \operatorname { e r r o r } \left( \operatorname { m m } \right) \downarrow$ </td></tr><tr><td>RGB + PC (full HUG) </td><td> ${ \bf 7 1 . 5 \pm 1 . 8 }$ </td><td> ${ \bf 1 9 . 0 \pm 0 . 8 }$ </td><td> ${ \bf 7 3 . 0 \pm 2 . 6 }$ </td><td> ${ \bf 1 4 . 6 \pm 0 . 9 }$  1</td></tr><tr><td>w/o crop</td><td> $6 1 . 2 \pm 2 . 0$ </td><td> $2 1 . 6 \pm 0 . 9$ </td><td> $5 8 . 0 \pm 2 . 8$ </td><td> $2 5 . 7 \pm 1 . 5$ </td></tr><tr><td>w/o point paint</td><td> $6 1 . 8 \pm 2 . 0$ </td><td> $2 1 . 9 \pm 1 . 0$ </td><td> $5 8 . 3 \pm 2 . 8$ </td><td> $2 3 . 3 \pm 1 . 7$ </td></tr><tr><td>w/o 3D loss</td><td> $3 9 . 2 \pm 2 . 0$ </td><td> $3 3 . 0 \pm 1 . 2$ </td><td> $3 2 . 7 \pm 2 . 7$ </td><td> $3 5 . 7 \pm 2 . 2$ </td></tr><tr><td>PC only</td><td> $6 4 . 2 \pm 2 . 0$ </td><td> $2 5 . 6 \pm 1 . 2$ </td><td> $7 0 . 7 \pm 2 . 6$ </td><td> $2 2 . 1 \pm 1 . 5$ </td></tr><tr><td>w/o crop</td><td> $4 7 . 3 \pm 2 . 0$ </td><td> $3 2 . 6 \pm 1 . 5$ </td><td> $5 0 . 0 \pm 2 . 9$ </td><td> $3 2 . 8 \pm 2 . 2$ </td></tr><tr><td>RGB only</td><td> $2 6 . 8 \pm { 1 . 8 }$ </td><td> $9 5 . 4 \pm 3 . 6$ </td><td> $2 9 . 7 \pm 2 . 6$ </td><td> $1 0 8 . 6 \pm 5 . 1$ </td></tr><tr><td>Human grasp (oracle)</td><td> $9 0 . 3 \pm 1 . 2$ </td><td> $9 . 4 \pm 0 . 3$ </td><td> $9 4 . 0 \pm 1 . 4$ </td><td> $7 . 4 \pm 0 . 3$ </td></tr></table>

![](images/278e043cc071cc390070eb4e2b15e62247962edd98627e86a3c3e610eb78f9b0.jpg)  
Table 2: Simulation results and ablations. Simulation success rate (SR) and fingertip contact (FC) error on HUG-BENCH over 10 grasps per object (600 val, 300 test; mean ± SE). Each model is evaluated on the unseen test objects at its best-val-SR checkpoint. Human grasp represents an oracle upper bound.  
Figure 7: Dataset scaling. Impact of dataset size on HUG-BENCH SR and FC error (Eq. 2); training sets are nested proper subsets.

## 5.2 Simulation Grasping Evaluation

Metrics. Success rate (SR, %) is the fraction of trials in which the object remains grasped at the end of the lift. SR alone is coarse, for example, failures can result from open-loop execution, so it does not fully capture grasp quality. We therefore also report fingertip contact error (FC error, mm), measuring how close the thumb and the closest supporting finger come to the object surface:

$$
\mathrm { F C } = \textstyle \frac { 1 } { 2 } \bigg ( | d _ { \mathrm { t h u m b } } | + \operatorname* { m i n } _ { f \in \mathcal { F } } | d _ { f } | \bigg ) ,\tag{2}
$$

where $d _ { i }$ is the signed distance from fingertip i to the object surface and $\mathcal { F }$ is the set of non-thumb fingers (smaller is better). FC error assumes a good grasp brings the thumb and a supporting fingertip near the object surface. Both metrics are averaged over 10 grasps per object on 60 val and 30 test objects; we pick checkpoints by best val SR. Per-object simulation success rates are reported in Appendix Table 6, and additional simulation results in Appendix E.

Ablation study. Table 2 reports the model architecture ablation. Our full model takes a dual-modal RGB+PC input, applies a 0.3 m crop of the point cloud around the user-clicked point, augments each point with DINO point painting features, and uses an auxiliary 3D loss on fingertip placement (Eq. 1). To isolate the contribution of each input stream, we additionally compare against singlemodality PC-only and RGB-only variants.

The human grasp oracle replays the 10 recorded human grasps on each object, estimating an upper bound on what is achievable in our simulator. It falls short of 100% due to hand-tracking error in our Aria Gen 2 data [72], slight inaccuracies in object assets, and open-loop execution failures (Appendix E.2). Our full model reaches 71.5% val SR and 73.0% test SR, within ∼20 points of the oracle (90.3 and 94.0%). Figure 4 shows that it produces stable, human-like grasps across object shapes, sizes, and viewpoints. The 3D loss is the most critical component: removing it cuts test SR by over 40 points to 32.7% and more than doubles FC error from 14.6 to 35.7 mm, confirming that explicit 3D supervision is essential for accurate fingertip placement. The 0.3 m crop and DINO point

![](images/a141452214cab2d43a1cc5413c119df7965a3b2c4c33114cc4f7871fcd68af57.jpg)  
Figure 8: Real-to-sim grasping. Evaluating HUG on HUG-BENCH in simulation using real captured inputs.

painting each cost ∼10 val-SR and ∼15 test-SR points when removed, showing that both a targeted spatial window for dense point cloud context and rich per-point features matter.

![](images/dd5efbf9a817db60fea02586b87d61c843b04d6a700a27fd4cc6d5bf533807d4.jpg)  
RGB + Point Cloud Point Cloud Only RGB Only

Figure 9: Single-modality failures. Cases where RGB-only or PC-only prediction fails but RGB+PC succeeds. Objects, left to right: pineapple, hair brush, anchovies, spoon, softball.

RGB and depth are complementary. On the modality axis, PC-only remains a strong standalone baseline at 64.2% val SR and 70.7% test SR, while RGB-only collapses to 26.8% val SR and 29.7% test SR. The contrast is sharper in FC error: RGB-only rises to 95 mm val and 109 mm test, and PC-only rises to 25.6 mm val and 22.1 mm test, up from 19.0 and 14.6 mm for the full model, showing that RGB grounding sharpens fingertip placement. Figure 9 shows that the two streams are complementary: RGB-only rarely reaches the vicinity of the object, while PConly reaches it but lacks semantic grounding, grasping the leafy top of a pineapple or the bristles of a brush rather than the body or handle, and snapping to a larger nearby object when depth is unreliable on small targets. RGB+PC resolves these cases, motivating the dual-modal design.

Data scaling. We train the full RGB+PC model on different dataset sizes: 25K, 50K, 100K, 250K, 500K, and 1M (all) RGB frames. Figure 7 shows the data scaling study. From 25K to 1M frames, test SR climbs from 33% to 73% and FC error falls from 54.2 mm to 14.6 mm. Neither saturates at 1M, suggesting the model is still data-bound, not capacity-bound at this scale. The val and test curves track tightly across all sizes, so gains transfer to held-out objects without overfitting.

## 5.3 Real-World Grasping Evaluation

We evaluate HUG on the 30 HUG-BENCH test objects in the real world, running 10 trials per object, or 300 per method (Table 3). We retarget predicted MANO grasps to the robot hand (Appendix F.2) and execute it open-loop pre-grasp → grasp → lift. All rollout videos are on our website.

Baselines. We compare HUG against two recent learning-based grasping methods. Dex1B [3] is a generative multi-fingered grasping model trained on 1B simulated demonstrations by combining grasp optimization with generative sampling, representing the sim-toreal paradigm for dexterous grasping. CAP (Contact-Anchored Policies) [27] conditions parallel-jaw grasping on a specified contact anchor and reports strong real-world performance compared to other methods [22, 73].

![](images/472100d16c4de6954a617a901f1334ecda393246bb957b9b2bf7db5b9c135eb3.jpg)  
MANO

![](images/91c3c6a64c2dc37f4ac2442e26752aa017c54a0bf92b89bf0d594fe7e9a9cd53.jpg)  
MANO-sim

![](images/95377ab9b39b20883547fefa76ddbd58d25b2bbac635ce1532d148f43698caeb.jpg)  
Ability  
WUJI  
Figure 10: Hand sizes. The fixed-shape MANO hand alongside its simulation mesh and the Ability and WUJI robot hands. WUJI is a similar size to HUG’s fixed hand size, but Ability is much smaller.

Tabletop experiments. HUG and Dex1B are deployed on a 6-DoF Ability hand [13] mounted on a 7-DoF xArm [74], using a third-person ZED stereo camera for input observations. CAP follows its published configuration of an iPhone wrist camera and a parallel-jaw gripper for execution. Like HUG, Dex1B is executed open-loop and occasionally predicts a grasp whose trajectory intersects the table; we count severe table-intersecting grasps as a failure to avoid damaging our hardware.

HUG reaches 66.7% overall success on the 30 test objects, exceeding Dex1B (43.7%) and CAP (32.7%) by +23% and +34% and grasping 28/30 objects at least once. The evaluation involves no test-set bias (Appendix F.1): the checkpoint with the best val SR (§ 5.2) is deployed directly on the test split with no per-object tuning, and each trial is a single grasp prediction followed by one open-loop execution without motion planning.

<table><tr><td colspan="3">HUG-BENCH test split</td><td colspan="3">Tabletop ZED + XArm + Ability</td><td>Simulation MANO Hand</td><td colspan="2">In-the-wild Aria + YOR+WUJI</td></tr><tr><td>Geometry</td><td colspan="2">Size Object</td><td>Dex1B</td><td>CAP</td><td>HUG</td><td>HUG</td><td>Location</td><td>HUG</td></tr><tr><td rowspan="7">Cylindrical</td><td>S</td><td>Glue stick</td><td>7/10</td><td>10/10</td><td>6/10</td><td>5/10</td><td>Stool</td><td>7/10</td></tr><tr><td></td><td>Pepper shaker</td><td>8/10</td><td>5/10</td><td>10/10</td><td>10/10</td><td>Kitchen island</td><td>6/10</td></tr><tr><td></td><td>Umbrella</td><td>4/10</td><td>7/10</td><td>5/10</td><td>10/10</td><td>Couch 1</td><td>6/10</td></tr><tr><td>M</td><td>Bowl</td><td>3/10</td><td>0/10</td><td>4/10</td><td>8/10</td><td>Kitchen island</td><td>1/10</td></tr><tr><td></td><td>Spray bottle</td><td>4/10</td><td>1/10</td><td>9/10</td><td>10/10</td><td>Side table</td><td>9/10</td></tr><tr><td>L</td><td>Wine bottle</td><td>7/10</td><td>3/10</td><td>3/10</td><td>10/10</td><td>Kitchen counter</td><td>10/10</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="5">Spheroidal</td><td>S</td><td>Strawberry</td><td>4/10</td><td>0/10</td><td>7/10</td><td>4/10</td><td>Bed 1 Bed2</td><td>6/10</td></tr><tr><td></td><td>Hacky sack</td><td>5/10</td><td>8/10</td><td>9/10</td><td>10/10</td><td></td><td>10/10</td></tr><tr><td>M</td><td>Pear</td><td>10/10</td><td>4/10</td><td>10/10</td><td>10/10</td><td>Dining table</td><td>10/10</td></tr><tr><td></td><td>Softball</td><td>8/10</td><td>0/10</td><td>5/10</td><td>10/10 10/10</td><td>Ottoman</td><td>8/10</td></tr><tr><td>L</td><td>Pineapple Football</td><td>8/10 0/10</td><td>5/10 0/10</td><td>10/10 0/10</td><td>8/10</td><td>Dining chair Nightstand</td><td>9/10 0/10</td></tr><tr><td rowspan="5"></td><td>S</td><td></td><td></td><td></td><td></td><td>6/10</td><td>Bed 1</td><td></td></tr><tr><td></td><td>Eraser Match box</td><td>3/10 5/10</td><td>0/10</td><td>6/10 8/10</td><td>9/10</td><td>Couch 2</td><td>6/10</td></tr><tr><td></td><td>Card deck</td><td>3/10</td><td>3/10 0/10</td><td>8/10</td><td>4/10</td><td>Bed2</td><td>8/10 8/10</td></tr><tr><td>M</td><td>Sponge</td><td>3/10</td><td>6/10</td><td>6/10</td><td>8/10</td><td>Kitchen counter</td><td>7/10</td></tr><tr><td></td><td>Wipe dispenser</td><td>0/10</td><td>0/10</td><td>0/10</td><td>3/10</td><td>Coffee table 1</td><td>4/10</td></tr><tr><td rowspan="5"></td><td>L</td><td>Storage bin</td><td>0/10</td><td>0/10</td><td>10/10</td><td>7/10</td><td>Couch 3</td><td>8/10</td></tr><tr><td></td><td>Nail clipper</td><td>3/10</td><td>0/10</td><td>5/10</td><td>3/10</td><td>Couch 2</td><td>4/10</td></tr><tr><td>Lock</td><td></td><td>6/10</td><td>6/10</td><td>5/10</td><td>8/10</td><td>Dresser</td><td>6/10</td></tr><tr><td>AppendagedM</td><td>Dustpan</td><td>5/10</td><td>2/10</td><td>6/10</td><td>6/10</td><td>Bed 1</td><td>6/10</td></tr><tr><td></td><td>Handbell</td><td>8/10</td><td>5/10</td><td>10/10</td><td>7/10</td><td>Couch 1</td><td>4/10</td></tr><tr><td rowspan="5"></td><td>L</td><td>Saucepan</td><td>5/10</td><td>3/10</td><td>4/10</td><td>10/10</td><td>Stove</td><td>7/10</td></tr><tr><td></td><td>Picnic basket</td><td>1/10</td><td>0/10</td><td>9/10</td><td>5/10</td><td>Desk</td><td>6/10</td></tr><tr><td>S</td><td>Rubber duck</td><td>3/10</td><td>4/10</td><td>10/10</td><td>5/10</td><td>Couch 3</td><td>9/10</td></tr><tr><td></td><td>Tape measure</td><td>3/10</td><td>8/10</td><td>8/10</td><td>9/10</td><td>Coffee table 1</td><td>5/10</td></tr><tr><td>M Grapes</td><td>Tape dispenser</td><td>4/10 2/10</td><td>2/10</td><td>3/10</td><td>10/10</td><td>Desk</td><td>4/10</td></tr><tr><td rowspan="5"></td><td></td><td></td><td></td><td>6/10</td><td>10/10</td><td>5/10 4/10</td><td>Dining table</td><td>3/10</td></tr><tr><td>L</td><td>Headphones</td><td>7/10</td><td>6/10</td><td>6/10</td><td>5/10</td><td>Desk chair Coffee table 2</td><td>2/10</td></tr><tr><td></td><td>Easel</td><td>2/10</td><td>4/10</td><td>8/10</td><td></td><td></td><td>7/10</td></tr><tr><td colspan="2">Overall success rate</td><td>43.7%</td><td>32.7%</td><td>66.7%</td><td>73.0%</td><td></td><td>62.0%</td></tr><tr><td colspan="2">Number of objects with ≥1 success</td><td>27/30</td><td>20/30</td><td>28/30</td><td></td><td>30/30</td><td>29/30</td></tr></table>

Table 3: Real-world grasp results on HUG-BENCH. Per-object success rates on 30 test split objects in two real-world settings. HUG outperforms both baselines on the tabletop by +23% and +34% and achieves a comparable success rate in-the-wild, demonstrating both strong grasping capability and stable cross-embodiment, cross-environment generalization.

The absolute success rates are low as HUG-BENCH is challenging, spanning sizes and geometries well beyond the medium, convex objects on which dexterous grasping methods typically report high success. Dex1B is accurate on this familiar regime (e.g. pear 10/10), but drops on large or unwieldy objects (storage bin 0/10) and complex geometry (easel 2/10). CAP is limited by its hardware: its parallel jaw cannot close around objects exceeding its span (storage bin 0/10) nor precisely grasp thin objects (eraser 0/10), and grasps mainly pinchable items (glue stick 10/10).

HUG is the most robust method across the geometry and size grid. It is alone in grasping large prismatic objects where both baselines fail (storage bin 10/10 vs. 0/10), and far outperforms both on objects with handles or irregular structure that demand precise, structure-aware contact (picnic basket 9/10 vs. 1/10 and 0/10; spray bottle 9/10 vs. 4/10 and 1/10; easel 8/10 vs. 2/10 and 4/10), while remaining strong on small objects (eraser 6/10 vs. 3/10 and 0/10; strawberry 7/10 vs. 4/10 and 0/10). Common failures of HUG are due to hardware limitations of the Ability hand: the football and wipe dispenser are grasped 0/10 by all methods, as the small Ability hand cannot wrap them (Figure 10).

In-the-wild experiments. Because HUG is trained on human grasps from diverse real-world locations, we expect it to remain robust outside the lab. Figure 5 shows HUG deployed on a modified YOR mobile manipulator [75] with an AgileX NERO arm [76], the 20-DoF WUJI hand [15], and Aria Gen 2 for vision, testing transfer to a new embodiment, camera, and uncontrolled household. With no onsite tuning of the model or WUJI retargeting [18], we run all 300 HUG-BENCH test trials consecutively in one morning. Objects are placed across rooms (kitchen, living room, bedroom, office), and no two are evaluated from the same viewpoint. HUG achieves a 62.0% success rate in-the-wild, only 4.7 points below the tabletop, indicating that HUG transfers gracefully across embodiments and in-the-wild environments.

![](images/7a95f9e92e355b627f4a6e0ff84a447c3c203620fbbca08820083de9f782adc3.jpg)  
Figure 11: Failure mode breakdown. Grasp-outcome flow for the 300 HUG-BENCH test trials in each real-world setting, tracing every attempt through the pre-grasp, grasp, and lift stages into success or a specific failure mode.

Failure modes. Figure 11 traces all 300 test trials through the pre-grasp, grasp, and lift stages, with a consistent distribution across tabletop and in-the-wild. Most failures occur as the hand closes from pre-grasp to grasp, when it contacts the object or the table before the fingers settle (reporting tabletop then in-the-wild: hit object 42 and 57, hit surface 8 and 8); a smaller share misses or overreaches before pre-grasp, often on objects too large to wrap (e.g. the football); and the rest occur after a grasp is established, where the object slips during the lift (11 and 15) or is dropped once raised (8 and 16). Two directions should recover much of this gap. Motion planning beyond the open-loop trajectory would prevent the hand from striking the object or table as it closes. Force-aware closing would reduce post-grasp slips, since HUG predicts a static pose with no notion of contact force.

## 6 Conclusion

We present HUG, a framework that learns dexterous robot grasping entirely from 1M-HUGS, a large-scale egocentric dataset of 1M natural human grasps. HUG predicts a MANO hand grasp from a single RGB-D image and retargets it to multiple dexterous hands at deployment, with no perhand training. To evaluate HUG, we build HUG-BENCH, a benchmark of unseen everyday objects with metric-scale meshes that pairs simulation with real-robot trials. Across both, HUG produces stable, human-like grasps that transfer zero-shot to new stereo cameras, robot embodiments, and uncontrolled households. We hope 1M-HUGS and HUG-BENCH help move dexterous grasping toward the generality that humans achieve effortlessly.

## 7 Limitations

HUG has several limitations. Hand modeling. The model is trained on right-hand grasps only, with the MANO shape held fixed at a canonical value, so left-handed or bimanual grasping and handspecific morphology are not modeled. Because all grasps are right-handed, some object orientations are easier to grasp than others; we vary object rotation across evaluation trials so that results are not biased toward favorable poses. Retargeting. The human-to-robot retargeting can fail when the target robot hand cannot reach a feasible analog of the predicted MANO pose. Open-loop execution. Real-world rollouts run open-loop, with no closed-loop visual feedback during contact or lift, which can cause failures on objects that shift or articulate during the trajectory. Label noise under occlusion. When the hand is occluded during the grasp, Aria Gen 2 hand tracking degrades, so the grasp label can be too loose or too tight. Object scale. Accuracy drops on very small objects, limited by the 224×224 input resolution, and on large or far objects, which are rare in egocentric data; normalizing the grasp translation prediction around the 3D query point may help the latter. Singlegrasp execution. We predict and execute one grasp per trial, but generating many candidates and selecting the best is a natural extension. Scope. Our evaluation is indoor only.

## Acknowledgments

This work was supported by grants from LG, Qualcomm, Honda, Microsoft, NSF award 2339096, and ONR award N00014-22-1-2773. Lerrel Pinto is supported by the Sloan, Packard, and CIFAR Fellowships.

We thank James Fort and the Meta Project Aria team for their support using Aria Gen 2 glasses. We thank Kelly Lee, Zicheng Teng, Bowen Tan, and Blake Chang for help with data collection; and our many friends who let us collect data in their apartments. We thank Nikhil Chavan-Dafle and Jeff Cui for helpful discussions, Kanad Patel, Dhawal Kabra, and Neer Patel for robot hardware support, and Omar Rayyan, Nikhil Bhattasli, and Takuma Yoneda for MuJoCo advice.

## References

[1] R. Wang, J. Zhang, J. Chen, Y. Xu, P. Li, T. Liu, and H. Wang. Dexgraspnet: A large-scale robotic dexterous grasp dataset for general objects based on simulation, 2023. URL https: //arxiv.org/abs/2210.02697.

[2] H.-S. Fang, H. Yan, Z. Tang, H. Fang, C. Wang, and C. Lu. Anydexgrasp: General dexterous grasping for different hands with human-level learning efficiency, 2025. URL https: //arxiv.org/abs/2502.16420.

[3] J. Ye, K. Wang, C. Yuan, R. Yang, Y. Li, J. Zhu, Y. Qin, X. Zou, and X. Wang. Dex1b: Learning with 1b demonstrations for dexterous manipulation, 2025. URL https://arxiv.org/abs/ 2506.17198.

[4] Y. Xu, W. Wan, J. Zhang, H. Liu, Z. Shan, H. Shen, R. Wang, H. Geng, Y. Weng, J. Chen, T. Liu, L. Yi, and H. Wang. Unidexgrasp: Universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy, 2023. URL https://arxiv.org/ abs/2303.00938.

[5] H. Chen, Y. Yao, Y. Ye, Z. Xu, H. Bharadhwaj, J. Wang, S. Tulsiani, Z. Erickson, and J. Ichnowski. Web2grasp: Learning functional grasps from web images of hand-object interactions. arXiv preprint arXiv:2505.05517, 2025.

[6] H. Gupta, M. A. Mirzaee, and W. Yuan. Grasp to act: Dexterous grasping for tool use in dynamic settings. IEEE Robotics and Automation Letters, 11(5):6288–6295, 2026.

[7] A. Iyer, Z. Peng, Y. Dai, I. Guzey, S. Haldar, S. Chintala, and L. Pinto. Open teach: A versatile teleoperation system for robotic manipulation. In Conference on Robot Learning (CoRL), 2024.

[8] S. P. Arunachalam, I. Guzey, S. Chintala, and L. Pinto. Holo-dex: Teaching dexterity with im- ¨ mersive mixed reality. In IEEE International Conference on Robotics and Automation (ICRA), 2023.

[9] R. Ding, Y. Qin, J. Zhu, C. Jia, S. Yang, R. Yang, X. Qi, and X. Wang. Bunny-VisionPro: Realtime bimanual dexterous teleoperation for imitation learning, 2024. URL https://arxiv. org/abs/2407.03162.

[10] Y. Qin, W. Yang, B. Huang, K. Van Wyk, H. Su, X. Wang, Y.-W. Chao, and D. Fox. AnyTeleop: A general vision-based dexterous robot arm-hand teleoperation system. In Robotics: Science and Systems (RSS), 2023.

[11] Meta Reality Labs Research. Project Aria Gen 2. https://facebookresearch.github. io/projectaria\_tools/gen2/, 2026. Accessed: 2026-06-15.

[12] A. Zorin, I. Guzey, B. Yan, A. Iyer, L. Kondrich, N. X. Bhattasali, and L. Pinto. Ruka: Rethinking the design of humanoid hands with learning, 2025. URL https://arxiv.org/abs/ 2504.13165.

[13] Psyonic. Ability Hand. https://www.psyonic.io/ability-hand, 2026. Accessed: 2026- 06-15.

[14] K. Shaw, A. Agarwal, and D. Pathak. LEAP Hand: Low-cost, efficient, and anthropomorphic hand for robot learning. In Robotics: Science and Systems (RSS), 2023.

[15] WUJI Technology. WUJI Hand. https://docs.wuji.tech/docs/en/wuji-hand/v1/, 2026. Accessed: 2026-06-15.

[16] Z. Mandi, Y. Hou, D. Fox, Y. Narang, A. Mandlekar, and S. Song. DexMachina: Functional retargeting for bimanual dexterous manipulation, 2025. URL https://arxiv.org/abs/ 2505.24853.

[17] K. Li, P. Li, T. Liu, Y. Li, and S. Huang. Maniptrans: Efficient dexterous bimanual manipulation transfer via residual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6991–7003, 2025.

[18] G. He and W. Zhang. Wujihand retargeting. https://github.com/wuji-technology/ wuji-retargeting, 2026. Accessed: 2026-06-15.

[19] J. Romero, D. Tzionas, and M. J. Black. Embodied hands: modeling and capturing hands and bodies together. ACM Transactions on Graphics, 36(6):1–17, Nov. 2017. ISSN 1557-7368. doi:10.1145/3130800.3130883. URL http://dx.doi.org/10.1145/3130800.3130883.

[20] L. Pinto and A. Gupta. Supersizing self-supervision: Learning to grasp from 50k tries and 700 robot hours, 2015. URL https://arxiv.org/abs/1509.06825.

[21] H.-S. Fang, C. Wang, M. Gou, and C. Lu. Graspnet-1billion: A large-scale benchmark for general object grasping. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11441–11450, 2020. doi:10.1109/CVPR42600.2020.01146.

[22] H.-S. Fang, C. Wang, H. Fang, M. Gou, J. Liu, H. Yan, W. Liu, Y. Xie, and C. Lu. Anygrasp: Robust and efficient grasp perception in spatial and temporal domains, 2023. URL https: //arxiv.org/abs/2212.08333.

[23] A. Mousavian, C. Eppner, and D. Fox. 6-dof graspnet: Variational grasp generation for object manipulation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2901–2910, 2019.

[24] M. Sundermeyer, A. Mousavian, R. Triebel, and D. Fox. Contact-graspnet: Efficient 6-dof grasp generation in cluttered scenes. In 2021 IEEE international conference on robotics and automation (ICRA), pages 13438–13444. IEEE, 2021.

[25] N. Chavan-Dafle, S. Popovych, S. Agrawal, D. D. Lee, and V. Isler. Simultaneous object reconstruction and grasp prediction using a camera-centric object shell representation, 2022. URL https://arxiv.org/abs/2109.06837.

[26] P. Liu, Y. Orru, J. Vakil, C. Paxton, N. Shafiullah, and L. Pinto. Demonstrating ok-robot: What really matters in integrating open-knowledge models for robotics. In Robotics: Science and Systems XX, RSS2024. Robotics: Science and Systems Foundation, July 2024. doi:10.15607/ rss.2024.xx.091. URL http://dx.doi.org/10.15607/RSS.2024.XX.091.

[27] Z. J. Cui, O. Rayyan, H. Etukuru, B. Tan, Z. Andrianarivo, Z. Teng, Y. Zhou, K. Mehta, N. Wojno, K. Y. Wu, M. H. Anjaria, Z. Wu, M. Mao, G. Zhang, B. Shah, Y. Kim, S. Chintala, L. Pinto, and N. M. M. Shafiullah. Contact-anchored policies: Contact conditioning creates strong robot utility models, 2026. URL https://arxiv.org/abs/2602.09017.

[28] T. G. W. Lum, M. Matak, V. Makoviychuk, A. Handa, A. Allshire, T. Hermans, N. D. Ratliff, and K. V. Wyk. DextrAH-g: Pixels-to-action dexterous arm-hand grasping with geometric fabrics. In 8th Annual Conference on Robot Learning, 2024. URL https://openreview. net/forum?id=S2Jwb0i7HN.

[29] R. Singh, A. Allshire, A. Handa, N. Ratliff, and K. V. Wyk. Dextrah-rgb: Visuomotor policies to grasp anything with dexterous hands, 2025. URL https://arxiv.org/abs/2412.01791.

[30] S. Christen, M. Kocabas, E. Aksan, J. Hwangbo, J. Song, and O. Hilliges. D-grasp: Physically plausible dynamic grasp synthesis for hand-object interactions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20577–20586, 2022.

[31] W. Wan, H. Geng, Y. Liu, Z. Shan, Y. Yang, L. Yi, and H. Wang. Unidexgrasp++: Improving dexterous grasping policy learning via geometry-aware curriculum and iterative generalistspecialist learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3891–3902, 2023.

[32] Y. Zhong, Q. Jiang, J. Yu, and Y. Ma. Dexgrasp anything: Towards universal robotic dexterous grasping with physics awareness, 2025. URL https://arxiv.org/abs/2503.08257.

[33] J. Lu, H. Kang, H. Li, B. Liu, Y. Yang, Q. Huang, and G. Hua. UGG: Unified Generative Grasping, page 414–433. Springer Nature Switzerland, Nov. 2024. ISBN 9783031728556. doi:10.1007/978-3-031-72855-6 24. URL http://dx.doi.org/10. 1007/978-3-031-72855-6\_24.

[34] H. Etukuru, N. Naka, Z. Hu, S. Lee, J. Mehu, A. Edsinger, C. Paxton, S. Chintala, L. Pinto, and N. M. M. Shafiullah. Robot utility models: General policies for zero-shot deployment in new environments. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 8275–8283. IEEE, 2025.

[35] C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song. Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. arXiv preprint arXiv:2402.10329, 2024.

[36] M. Xu, H. Zhang, Y. Hou, Z. Xu, L. Fan, M. Veloso, and S. Song. Dexumi: Using human hand as the universal manipulation interface for dexterous manipulation. arXiv preprint arXiv:2505.21864, 2025.

[37] I. Guzey, H. Qi, J. Urain, C. Wang, J. Yin, K. Bodduluri, M. Lambeta, L. Pinto, A. Rai, J. Malik, T. Wu, A. Sharma, and H. Bharadhwaj. Dexterity from smart lenses: Multi-fingered robot manipulation with in-the-wild human demonstrations, 2025. URL https://arxiv. org/abs/2511.16661.

[38] I. Guzey, Y. Dai, G. Savva, R. Bhirangi, and L. Pinto. Bridging the human to robot dexterity gap through object-oriented rewards. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 3344–3351. IEEE, 2025.

[39] N. Karaev, I. Rocco, B. Graham, N. Neverova, A. Vedaldi, and C. Rupprecht. Cotracker: It is better to track together. In European Conference on Computer Vision (ECCV), 2024.

[40] C. Doersch, Y. Yang, M. Vecerik, D. Gokay, A. Gupta, Y. Aytar, J. Carreira, and A. Zisserman. Tapir: Tracking any point with per-frame initialization and temporal refinement, 2023. URL https://arxiv.org/abs/2306.08637.

[41] Y. Ye, P. Hebbar, A. Gupta, and S. Tulsiani. Diffusion-guided reconstruction of everyday hand-object interaction clips. In Proceedings of the IEEE/CVF international conference on computer vision, pages 19717–19728, 2023.

[42] G. Pavlakos, D. Shan, I. Radosavovic, A. Kanazawa, D. Fouhey, and J. Malik. Reconstructing hands in 3d with transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[43] Y. Ye, X. Li, A. Gupta, S. De Mello, S. Birchfield, J. Song, S. Tulsiani, and S. Liu. Affordance diffusion: Synthesizing hand-object interactions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22479–22489, 2023.

[44] S. Park, H. Bharadhwaj, and S. Tulsiani. Demodiffusion: One-shot human imitation using pre-trained diffusion policy, 2025. URL https://arxiv.org/abs/2506.20668.

[45] S. Haldar and L. Pinto. Point policy: Unifying observations and actions with key points for robot manipulation. arXiv preprint arXiv:2502.20391, 2025.

[46] C. Wang, L. Fan, J. Sun, R. Zhang, L. Fei-Fei, D. Xu, Y. Zhu, and A. Anandkumar. Mimicplay: Long-horizon imitation learning by watching human play. arXiv preprint arXiv:2302.12422, 2023.

[47] K. Grauman, A. Westbury, E. Byrne, Z. Chavis, A. Furnari, R. Girdhar, J. Hamburger, H. Jiang, M. Liu, X. Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[48] H. Bharadhwaj, R. Mottaghi, A. Gupta, and S. Tulsiani. Track2act: Predicting point tracks from internet videos enables generalizable robot manipulation. In European Conference on Computer Vision (ECCV), 2024.

[49] K. Shaw, S. Bahl, and D. Pathak. Videodex: Learning dexterity from internet videos. In Conference on Robot Learning (CoRL), 2023.

[50] J. Shi, Z. Zhao, T. Wang, I. Pedroza, A. Luo, J. Wang, J. Ma, and D. Jayaraman. Zeromimic: Distilling robotic manipulation skills from web videos. In International Conference on Robotics and Automation (ICRA), 2025.

[51] M. K. Srirama, S. Dasari, S. Bahl, and A. Gupta. Hrp: Human affordances for robotic pretraining. In Robotics: Science and Systems (RSS), 2024.

[52] T. Tao, M. K. Srirama, J. J. Liu, K. Shaw, and D. Pathak. Dexwild: Dexterous human interactions for in-the-wild robot policies. arXiv preprint arXiv:2505.07813, 2025.

[53] S. Gao, W. Liang, K. Zheng, A. Malik, S. Ye, S. Yu, W.-C. Tseng, Y. Dong, K. Mo, C.-H. Lin, et al. Dreamdojo: A generalist robot world model from large-scale human videos. arXiv preprint arXiv:2602.06949, 2026.

[54] J. Engel, K. Somasundaram, M. Goesele, A. Sun, A. Gamino, A. Turner, A. Talattof, A. Yuan, B. Souti, B. Meredith, et al. Project aria: A new tool for egocentric multi-modal ai research. arXiv preprint arXiv:2308.13561, 2023.

[55] V. Liu, A. Adeniji, H. Zhan, R. Bhirangi, P. Abbeel, and L. Pinto. Egozero: Robot learning from smart glasses, 2025. URL https://arxiv.org/abs/2505.20290.

[56] S. Kareer, D. Patel, R. Punamiya, P. Mathur, S. Cheng, C. Wang, J. Hoffman, and D. Xu. Egomimic: Scaling imitation learning via egocentric video, 2024. URL https://arxiv. org/abs/2410.24221.

[57] Y.-W. Chao, W. Yang, Y. Xiang, P. Molchanov, A. Handa, J. Tremblay, Y. S. Narang, K. Van Wyk, U. Iqbal, S. Birchfield, J. Kautz, and D. Fox. DexYCB: A benchmark for capturing hand grasping of objects. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9044–9053, 2021.

[58] N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris, C. Ryali, K. V. Alwala, H. Khedr, A. Huang, J. Lei, T. Ma, B. Guo, A. Kalla, M. Marks, J. Greer, M. Wang, P. Sun, R. Radle, T. Afouras, E. Mavroudi, K. Xu, T.-H. Wu, Y. Zhou, L. Momeni, R. Hazra, S. Ding, ¨ S. Vaze, F. Porcher, F. Li, S. Li, A. Kamath, H. K. Cheng, P. Dollar, N. Ravi, K. Saenko, ´ P. Zhang, and C. Feichtenhofer. Sam 3: Segment anything with concepts, 2026. URL https: //arxiv.org/abs/2511.16719.

[59] J. Min, Y. Jeon, J. Kim, and M. Choi. S2M2: Scalable stereo matching model for reliable depth estimation, 2025. URL https://arxiv.org/abs/2507.13229.

[60] L. Yang, X. Zhan, K. Li, W. Xu, J. Li, and C. Lu. CPF: Learning a contact potential field to model the hand-object interaction. In ICCV, 2021.

[61] Y. Zhou, C. Barnes, J. Lu, J. Yang, and H. Li. On the continuity of rotation representations in neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5745–5753, 2019.

[62] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

[63] T. Darcet, M. Oquab, J. Mairal, and P. Bojanowski. Vision transformers need registers. In International Conference on Learning Representations (ICLR), 2024.

[64] G. Qian, Y. Li, H. Peng, J. Mai, H. Hammoud, M. Elhoseiny, and B. Ghanem. PointNeXt: Revisiting PointNet++ with improved training and scaling strategies. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

[65] M. Tancik, P. P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. T. Barron, and R. Ng. Fourier features let networks learn high frequency functions in low dimensional domains. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

[66] W. Peebles and S. Xie. Scalable diffusion models with transformers. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 4195–4205, 2023.

[67] E. Todorov, T. Erez, and Y. Tassa. MuJoCo: A physics engine for model-based control. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 5026– 5033, 2012.

[68] B. Li, D. Wu, J. Li, S. Zhou, Z. Zeng, L. Li, and H. Zha. Mv-sam3d: Adaptive multi-view fusion for layout-aware 3d generation, 2026. URL https://arxiv.org/abs/2603.11633.

[69] B. Yi, C. M. Kim, J. Kerr, G. Wu, R. Feng, A. Zhang, J. Kulhanek, H. Choi, Y. Ma, M. Tancik, and A. Kanazawa. Viser: Imperative, web-based 3d visualization in python, 2025. URL https://arxiv.org/abs/2507.22885.

[70] C. Portaneri, M. Rouxel-Labbe, M. Hemmer, D. Cohen-Steiner, and P. Alliez. Alpha wrapping ´ with an offset. ACM Trans. Graph., 41(4), July 2022. ISSN 0730-0301. doi:10.1145/3528223. 3530152. URL https://doi.org/10.1145/3528223.3530152.

[71] X. Wei, M. Liu, Z. Ling, and H. Su. Approximate convex decomposition for 3d meshes with collision-aware concavity and tree search. ACM Transactions on Graphics (TOG), 41(4):1–18, 2022.

[72] Meta Reality Labs Research. Project aria gen 2 mps performance benchmarks. https: //facebookresearch.github.io/projectaria\_tools/gen2/technical-specs/ mps/benchmarks/performance, 2025. Accessed: 2026-06-14.

[73] P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, M. Y. Galliker, D. Ghosh, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, D. LeBlanc, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, A. Z. Ren, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, J. Tanner, Q. Vuong, H. Walke, A. Walling, H. Wang, L. Yu, and U. Zhilinsky. π0.5: a vision-language-action model with open-world generalization, 2025. URL https://arxiv.org/abs/2504.16054.

[74] UFACTORY. xArm 7. https://www.ufactory.us/product/ufactory-xarm-7, 2026. Accessed: 2026-06-15.

[75] M. H. Anjaria, M. E. Erciyes, V. Ghatnekar, N. Navarkar, H. Etukuru, X. Jiang, K. Patel, D. Kabra, N. Wojno, R. A. Prayage, S. Chintala, L. Pinto, N. M. M. Shafiullah, and Z. J. Cui. Yor: Your own mobile manipulator for generalizable robotics, 2026. URL https://arxiv. org/abs/2602.11150.

[76] AgileX Robotics. NERO. https://global.agilex.ai/products/nero, 2026. Accessed: 2026-06-15.

[77] B. Gyenes, E. Gospodinov, J. Frieling, E. Krohmer, N. Schreiber, X. Jia, N. Freymuth, and G. Neumann. Fourier features let agents learn high precision policies with imitation learning, 2026. URL https://arxiv.org/abs/2606.12334.

[78] S. D. Team, X. Chen, F.-J. Chu, P. Gleize, K. J. Liang, A. Sax, H. Tang, W. Wang, M. Guo, T. Hardin, X. Li, A. Lin, J. Liu, Z. Ma, A. Sagar, B. Song, X. Wang, J. Yang, B. Zhang, P. Dollar, G. Gkioxari, M. Feiszli, and J. Malik. Sam 3d: 3dfy anything in images, 2026. URL ´ https://arxiv.org/abs/2511.16624.

[79] A. Muntoni and P. Cignoni. PyMeshLab. https://doi.org/10.5281/zenodo.4438750, 2021. DOI: 10.5281/zenodo.4438750.

[80] L. Yang, K. Li, X. Zhan, F. Wu, A. Xu, L. Liu, and C. Lu. Oakink: A large-scale knowledge repository for understanding hand-object interaction, 2022. URL https://arxiv.org/abs/ 2203.15709.

## Appendix of “Human Universal Grasping”

This appendix provides additional results, analyses, and implementation details that support the main paper. We include more details about the MANO parameter optimization in § A, the 1M-HUGS curation with aria2mano in § B, the HUG model implementation details and hyperparameters in § C, the HUG-BENCH asset generation with aria2mesh in § D, the simulation grasping evaluation in § E, and the real-world grasping evaluation in § F.

A MANO Parameter Optimization 18   
B 1M-HUGS Dataset Curation with aria2mano 18   
B.1 Automated Labeling 19   
B.2 Quality Control 19   
B.3 Dataset Preparation 20   
C HUG Model Implementation Details 21   
C.1 Point Cloud Crop Radius 21   
C.2 Camera Generalization 21   
C.3 Training Details and Hyperparameters 21   
D HUG-BENCH Asset Generation with aria2mesh 22   
D.1 Metric-scale 3D Object Reconstruction 22   
D.2 Object Statistics 23   
E Simulation Grasping Evaluation 26   
E.1 Simulated MANO Hand 26   
E.2 The Human Grasp Oracle 26   
E.3 Dense Grasp Quality Metrics 26   
F Real-World Grasping Evaluation 27   
F.1 Real-World Evaluation Design 28   
F.2 Retargeting to Robot Hands . 28   
F.3 Qualitative Observations 28

## A MANO Parameter Optimization

We elaborate on the MANO [19] fitting procedure summarized in § 3. Aria Gen 2 reports the wearer’s hand only as a sparse 21- landmark skeleton in the device coordinate frame; to obtain the full articulated MANO representation (pose, shape, mesh), we run a perframe optimization that aligns the MANO joint positions with the Aria landmarks subject to anatomical constraints. Figure 12 illustrates the aria2mano fitting. This fitting procedure is released in our aria2mano package.

![](images/12f2b2ab2cac6eaf51b72e4dfc91792fbe22b147582541fce76f7fb81ca3447b.jpg)

Optimized parameters. We optimize the 15 finger-joint axis-angle parameters $\pmb { \theta } \in \mathbb { R } ^ { 1 5 \times 3 }$ (45 DoF in total). The wrist orientation and translation are fixed to the values reported by Aria, since Aria’s wrist estimate is already metric and globally consistent across frames. We also optimize the 10-dimensional MANO shape $\beta \in \mathbb { R } ^ { 1 0 }$ (later replaced by a fixed shape, § B.3) for consistency with the 1M-HUGS and HUG-BENCH.

Figure 12: aria2mano MANO fitting. aria2mano fits a full articulated MANO hand (bottom, blue) to the sparse 21-landmark Aria skeleton (top, red) via a per-frame anatomicallyconstrained optimization, recovering pose, mesh, and dense joint angles from each grasp recording.

Landmark alignment loss. Aria landmarks are first reordered to match the MANO joint convention, and the wrist (fixed) and the carpometacarpal (CMC) joint of the thumb (which Aria does not report reliably) are masked out of the loss. We then minimize a weighted MSE between the MANO-reconstructed joint positions $\hat { \mathbf { x } } _ { i }$ and the Aria landmarks ${ \bf x } _ { i } \mathrm { : }$

$$
\mathcal { L } _ { \mathrm { l m } } = \frac { \lambda _ { \mathrm { m s e } } } { \sum _ { i } w _ { i } } \sum _ { i = 1 } ^ { 2 1 } w _ { i } \| \hat { { \mathbf x } } _ { i } - { \mathbf x } _ { i } \| _ { 2 } ^ { 2 } ,\tag{3}
$$

where the five fingertip landmarks receive a weight $w _ { i } = 5$ and all other landmarks receive $w _ { i } = 1$ and $\lambda _ { \mathrm { m s e } } ~ = ~ 2 { \times } 1 0 ^ { 4 }$ balances the landmark term against the anatomical prior. The fingertip upweighting is essential because fingertips dominate downstream contact geometry, and small angular errors at proximal joints accumulate into large fingertip displacements.

Anatomical prior. In addition to our landmark loss, we attach the anatomical validity loss $\mathcal { L } _ { \mathrm { a n a t } }$ from manotorch’s AnatomyConstraintLossEE [60], which extracts per-joint Euler angles via forward kinematics on the MANO skeleton and penalizes deviations from human range-of-motion bounds. This prevents the optimizer from explaining landmarks with hyperextended or fully rotated finger poses.

Optimizer. We minimize ${ \mathcal { L } } = { \mathcal { L } } _ { \mathrm { l m } } + { \mathcal { L } } _ { \mathrm { a n a t } }$ with L-BFGS using strong-Wolfe line search, history size 10, gradient-norm tolerance $1 0 ^ { - 7 }$ , and gradient-change tolerance $1 0 ^ { - 3 }$ . To accelerate convergence we warm-start each frame from the optimized pose of the previous frame whenever temporal continuity is available, with a closure-call budget of 20; cold-started frames are capped at 50. Wrist and CMC entries are excluded from the loss by construction. We achieve an average fingertip error of less than 2 mm.

## B 1M-HUGS Dataset Curation with aria2mano

This section details the curation pipeline that turns raw Aria Gen 2 recordings into the 1M-HUGS grasp dataset, summarized in § 3. Each recording captures a single right-hand grasp of a single stationary object: the wearer moves their head around the object for 15–30 seconds with both hands behind their back, then grasps the object without lifting it. From this footage we identify the grasped object, segment it across all frames, select the grasp frame, manually verify the result, and prepare the final training entries. The full pipeline is released as part of aria2mano. Figure 13 illustrates the capture setup.

![](images/1f8fba10cdeef8c82f212a1ab22d32fd80815525795adbb2746f04622d2c6d51.jpg)  
Figure 13: Data collection. A wearer captures a grasp with Aria Gen 2 glasses.

## B.1 Automated Labeling

The first three stages run without human input: identifying the grasped object, segmenting it across the recording, and selecting the grasp frame.

Object Identification. We localize the grasped object with a vision-language model (we used Gemini 3 Flash Preview). The VLM compares one “before” frame with no hand visible against four “after” frames in which the hand grasps the object, and returns 5–10 points covering all parts of the object, including its body and any handles or appendages. A second VLM call verifies that each point lies on the grasped object and rejects the recording if at most one point does. The cost is roughly \$0.003 per recording.

Mask Propagation. The verified points prompt SAM3 [58] to segment the object and track it bidirectionally from the annotation frame across every frame of the recording, yielding a per-frame binary object mask.

Grasp-Frame Selection. We detect the grasp frame with a set of heuristics: the frame must have valid right-hand MANO tracking with all landmarks projecting in bounds, hand motion below 0.01 m per frame (stability), mean fingertip-to-mask distance below 100 px (proximity), and must fall within the last 10 seconds of the recording. Among the surviving candidates we select the frame of best MANO quality, measured as the geometric mean of fingertip error and anatomy loss, preferring frames with no visible left hand.

## B.2 Quality Control

Automatic labeling is not 100% accurate, so every recording is reviewed in a web annotation app (Figure 14). A reviewer can re-segment the mask with additional point prompts and re-propagate it, re-select the grasp frame, and either approve the recording or mark it as having no stable grasp. Approval writes a checked flag so the recording enters dataset preparation.

![](images/5cd0e57bd8d6ef8792a91b413b5d2e177d0c5077d4dbdd3242b4e81b77a90b4d.jpg)  
Figure 14: Grasp annotation app. The web interface used to verify and correct the automatic labels. The left panel refines the object mask with point prompts and SAM3 re-segmentation; the right panel steps through frames to select and verify the grasp frame. Each recording is marked checked once its mask and grasp pass review.

## B.3 Dataset Preparation

We export training entries from the checked recordings under a sequence of filters. Shared filters keep frames whose index lies in [20, grasp − 10) (skipping the SLAM warm-up and the 10 frames before the grasp), drop the annotation frame, require at least 60% valid depth-confidence pixels, and, on the center-cropped square, require a non-empty object mask with at least one grasp-hand landmark on it. Per-camera filters, applied independently to the RGB and stereo-left grayscale streams, require at least five grasp-hand landmarks inside the image, no MPS-tracked hand projecting inside the image, and a non-empty final 224×224 object mask. The grasp pose is transformed from the world frame into each per-frame camera frame. The stereo-left grayscale images are included as a free augmentation, so the full dataset contains roughly 2M entries (1M RGB plus 1M grayscale). For the scaling study we build nested video-level subsets at 25k, 50k, 100k, 250k, 500k, and 1M RGB frames, each a strict prefix of the next.

![](images/48cb0f0f486d10b8a2fdc98a3caf3a88c7bf92f0a49dac4029fad33b26269de9.jpg)  
Figure 15: RGB crop. Stereo-left depth reprojected into the RGB frame: valid reprojected depth (black), the crop (green), and parallax holes filled by nearest neighbor (red).

RGB field-of-view crop. RGB images are cropped to fit the depth field of view, since the S2M2 [59] depth is reprojected from the stereo-left

camera into the RGB frame and the two fields of view do not fully overlap (Figure 15). After the center-square crop, RGB drops the top 25% and equal side margins to a 1440×1440 window (75% of 1920), then resizes to 224 × 224; the stored intrinsics reflect the crop. After cropping, roughly 99.9% of depth is valid, and the remaining ∼0.1% parallax holes are nearest-neighbor filled with a distance transform, preserving sharp edges without smoothing across object boundaries.

Fixed MANO shape. HUG predicts only hand pose, with the MANO shape $\beta$ held fixed at a single canonical value. The data collectors have low diversity in hand size, so we set $\beta$ to the first author’s hand, which covers the data well on average and is at least as large as every hand in HUG-BENCH (Figure 10). We then recompute all grasp labels in 1M-HUGS and HUG-BENCH with this fixed β, keeping the recorded joint angles, so that a single hand size is used consistently throughout training and evaluation. For simulation we export an MJCF of this canonical hand and use it for every grasp rollout. The fixed shape can be changed and the dataset recomputed if a different hand size is desired. We detail the MJCF generation in Appendix E.1.

## C HUG Model Implementation Details

This section provides implementation details of HUG beyond the main-text method description: the point cloud crop that bounds the receptive field, how the model generalizes across cameras, the single-modality ablations and training dynamics, and the full hyperparameter list.

## C.1 Point Cloud Crop Radius

Before encoding, we crop the back-projected point cloud $\mathcal { P }$ to a ball of radius r around the 3D query point ${ \bf p } _ { q }$ ,

$$
\mathcal { P } _ { \mathrm { c r o p } } = \{ \mathbf { p } \in \mathcal { P } : \| \mathbf { p } - \mathbf { p } _ { q } \| _ { 2 } \leq r \} , \qquad r = 0 . 3 \mathrm { m } ,\tag{4}
$$

and sample $N _ { p } = 4 0 9 6$ points from ${ \mathcal { P } } _ { \mathrm { c r o p } }$ . The crop bounds the receptive field to $N _ { p }$ points (hence $N = 2 5 6$ tokens); over a full image these 256 centroids would be too sparse to resolve the target, whereas the crop concentrates them on the object. Each metric centroid is encoded with Fourier features [65], which has been shown to help point cloud encoders leverage geometric details more effectively [77] than Cartesian features. The radius sets the spatial scope of the model, $i . e .$ the largest object it can grasp, so it is fixed by the intended application rather than tuned for accuracy, and we do not ablate it. We use 0.3 m as roughly the largest object graspable with one hand. A smaller radius would likely raise metrics on our val and test sets, which are biased toward smaller objects, but at the cost of generality; a model specializing in small objects can be retrained with a smaller radius. Figure 16 visualizes the crop.

## C.2 Camera Generalization

The camera intrinsics K enter HUG only through geometric operations: back-projecting a pixel $( u , v )$ with depth d to a metric point

$$
\mathbf { p } = d \mathbf { K } ^ { - 1 } \left[ u , v , 1 \right] ^ { \top } ,\tag{5}
$$

which lifts the 2D click to the 3D query point $\mathbf { p } _ { q } .$ , and the inverse projection $[ u ^ { \prime } , v ^ { \prime } , 1 ] ^ { \top } \propto \mathbf { K } \mathbf { c } _ { i }$ that maps each centroid ci onto the image plane for point painting. No layer takes K as a learned input. The model therefore transfers across stereo cameras with different intrinsics without retraining, which is why training on Aria recordings deploys directly to a ZED stereo camera in our tabletop experiments. We have also tested HUG on Realsense D415/D435 with success despite their grayscale images.

![](images/efed57c14bb32999622711b8b12d134cfe31cb00385af39dc739556cbd36c2d2.jpg)

## C.3 Training Details and Hyperparameters

Figure 16: Point cloud crop. The 0.3 m radius crop around the 3D query point.

Table 4 lists the architecture, training, and inference hyperparameters of HUG, and Table 5 reports the per-module parameter counts. MANO-fitting hyperparameters (L-BFGS settings,

landmark weights, anatomical prior) are given in Appendix A. Figure 17 shows training curves for all ablation configurations.

Table 4: HUG full model hyperparameters.
<table><tr><td>Group</td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="9">Architecture</td><td rowspan="9">RGB encoder RGB tokens Point encoder PointNeXt width /blocks</td><td>DINOv2-Base + registers (frozen)</td></tr><tr><td>256 (16 × 16 patch grid)</td></tr><tr><td>PointNeXt U-Net (trainable)</td></tr><tr><td>c = 64(512D out),blocks [1,2,1,1] (B)</td></tr><tr><td>[0.025,0.05,0.10,0.20] m</td></tr><tr><td></td></tr><tr><td>Per-point RGB at stem yes (stem in-dim 6)</td></tr><tr><td>4096 256</td></tr><tr><td>4 layers,  $D _ { f } = 1 0 2 4 , 8$ </td></tr><tr><td rowspan="3"></td><td>Fusion transformer Flow transformer</td><td>heads 6 DiT blocks,  $D _ { m } = 5 1 2 ,$  8 heads</td></tr><tr><td>MANO target dim</td><td>99 (3 trans + 6D wrist+ 90 fingers)</td></tr><tr><td>Fourier scale</td><td>1.0</td></tr><tr><td rowspan="3"></td><td>Dropout</td><td>0.1</td></tr><tr><td>MANO shape β</td><td>[-2.37,-1.25,-2.05,-0.85,1.66,-1.35,-1.85,-0.67,-1.69,-1.21]</td></tr><tr><td>Query point</td><td>3D,Fourier-feature encoded</td></tr><tr><td rowspan="3">Conditioning</td><td>Point cloud crop radius</td><td>0.3m</td></tr><tr><td>Fusion</td><td>Point painting (bilinear DINOv2 sampling)</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td rowspan="9">Training</td><td>Base learning rate PointNeXt learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td></td><td> $4 \times 1 0 ^ { - 4 } \left( 4 \times \right)$ </td></tr><tr><td>Weight decay</td><td>1×10-3</td></tr><tr><td>Gradient clipping</td><td>1.0 (max norm)</td></tr><tr><td>Batch size</td><td>64/GPU (128 effective)</td></tr><tr><td>Steps</td><td>100K</td></tr><tr><td>Warmup</td><td>5K steps, linear</td></tr><tr><td>Precision</td><td>bf16</td></tr><tr><td>EMA Joint loss</td><td>decay 0.9999,from step 50K</td></tr><tr><td></td><td>L1</td><td></td></tr><tr><td></td><td>Loss weights</td><td> $\lambda _ { \mathrm { v } } = 1 , \lambda _ { \mathrm { 3 D } } = 2 0$ </td></tr><tr><td>Hardware</td><td></td><td>2× RTX 5090,DDP,~10 h</td></tr><tr><td></td><td>Validation cadence</td><td>every 5K steps (MuJoCo)</td></tr><tr><td>ODE solver</td><td></td><td>50-step Euler</td></tr></table>

Table 5: HUG full model parameter counts.
<table><tr><td>Module</td><td>Total</td><td>Trainable</td></tr><tr><td>RGB encoder (DINOv2, frozen)</td><td>86,583,552</td><td>0</td></tr><tr><td>Point encoder (PointNeXt)</td><td>15,587,776</td><td>15,587,776</td></tr><tr><td>Fusion transformer</td><td>67,392,256</td><td>67,390,720</td></tr><tr><td>MANO layer (frozen)</td><td>370,329</td><td>0</td></tr><tr><td>Flow transformer</td><td>37,682,374</td><td>37,682,176</td></tr><tr><td>Total</td><td>207,244,224</td><td>120,660,672</td></tr></table>

## D HUG-BENCH Asset Generation with aria2mesh

HUG-BENCH is a simulation benchmark of 90 everyday objects with metric-scale meshes and physics-ready simulation assets, reconstructed from Aria Gen 2 recordings with our released aria2mesh pipeline. This section details the asset-generation pipeline and reports per-object statistics for the full object set.

## D.1 Metric-scale 3D Object Reconstruction

Each HUG-BENCH object is reconstructed from five egocentric Aria Gen 2 [11] views with Multiview SAM3D [68], a training-free extension of SAM3D [78], into which we inject Aria camera intrinsics, extrinsics, and stereo depth. We then pose-optimize and gravity-align each mesh, manually verify and edit its scale and pose in Viser [69] against the SLAM semidense point cloud and the dense stereo depth, make it watertight with Alpha Wrap [70] implemented in PyMeshLab [79], and produce a convex decomposition with CoACD [71]. Figure 18 shows the multi-view reconstruction recovering metric-scale meshes and poses from the five Aria views. The pipeline exports a visual mesh, convex collision parts, and both a URDF and an MJCF per object, loadable directly in Py-Bullet and MuJoCo. Because reconstruction runs from a short egocentric recording, a real-world object is turned into a metric-scale, simulation-ready asset in minutes, making it practical to grow the benchmark or build task-specific object sets.

![](images/d918279b659cce99b90838f989b4f0b8c848f75ad6d5c78ba0130254c20aeacb.jpg)  
Figure 17: Training curves. Success rate and fingertip contact error on the HUG-BENCH val split for model ablations (top) and data scaling experiments (bottom).

![](images/59f7ee630fb1d5366f6a5c52923d3021c17afe5c7b51e2f42ceefff1fea3bf5c.jpg)  
Figure 18: aria2mesh asset reconstruction. Five egocentric Aria views (camera frustums) are fused with Multi-view SAM3D, then pose-optimized and gravity-aligned, and finally manually edited against the semidense point cloud and dense stereo depth points to recover per-object metricscale meshes and poses (axes).

## D.2 Object Statistics

Table 6 lists all 90 objects with their size class, geometric category, mass, and volume, a thumbnail of both the reconstructed simulation mesh and the real object, and the per-object success rates of the best checkpoint: the ground-truth (human grasp oracle) simulation SR, the HUG simulation SR, and the real-world tabletop and in-the-wild SR. The 30 test objects are listed first, followed by the val objects, which we only use for simulation evaluation.

Table 6: HUG-BENCH object statistics. All 90 objects with category, mass, volume, a real-object and simulation-mesh thumbnail, and best-checkpoint success rates: ground-truth (human grasp oracle) sim SR, HUG sim SR, HUG tabletop SR, and HUG in-the-wild SR. The latter two SRs are results from Table 3 (Tabletop: ZED + xArm + Ability; Wild: Aria + YOR + WUJI). The test objects (top) carry real-world numbers; val objects (bottom) are simulation-only.
<table><tr><td>Geometry</td><td>Size Object</td><td></td><td>Real Sim Mass(g) Vol. (cm³） Scene</td><td></td><td></td><td></td><td></td><td>GT Sim SR Sim SR Tabletop SR Wild SR</td><td></td></tr><tr><td colspan="10">HUG-BENCH test</td></tr><tr><td></td><td>S</td><td>Glue stick</td><td></td><td>42.4</td><td>64.4 test/small.1</td><td>10/10</td><td>5/10</td><td>6/10</td><td>7/10</td></tr><tr><td></td><td></td><td>Pepper shaker</td><td></td><td>112.9</td><td>222.7 test/small.2</td><td>10/10</td><td>10/10</td><td>10/10</td><td>6/10</td></tr><tr><td>Cylindrical</td><td>M</td><td>Umbrella</td><td><img src="images/a34e3da36bf3e409729dee86f725eb54545edab48fce9b3cc2144129ec4a9b6b.jpg"/></td><td>208.1</td><td>136.9 test/medium_1</td><td>10/10</td><td>10/10</td><td>5/10</td><td>6/10</td></tr><tr><td></td><td></td><td>Bowl</td><td></td><td>158.6</td><td>226.9 test/medium_2</td><td>9/10</td><td>8/10</td><td>4/10</td><td>1/10</td></tr><tr><td></td><td>L</td><td>Spray bottle</td><td></td><td>450.8</td><td>1216.1 test/large_1</td><td>9/10</td><td>10/10</td><td>9/10</td><td>9/10</td></tr><tr><td></td><td></td><td>Wine bottle</td><td></td><td>317.6</td><td>1026.1 test/large_2</td><td>10/10</td><td>10/10</td><td>3/10</td><td>10/10</td></tr><tr><td></td><td>S</td><td>Strawberry</td><td></td><td>3.1</td><td>13.4 test/small.1</td><td>10/10</td><td>4/10</td><td>7/10</td><td>6/10</td></tr><tr><td></td><td></td><td>Hacky sack</td><td></td><td>34.6</td><td>9.7 test/small.2</td><td>10/10</td><td>10/10</td><td>9/10</td><td>10/10</td></tr><tr><td>Spheroidal</td><td>M</td><td>Pear</td><td><img src="images/c5ccffae22334df4b5ff4b909516812c473db58b2b5896c6eaf2cbffc5d47278.jpg"/></td><td>16.9</td><td>246.3test/medium_1</td><td>10/10</td><td>10/10</td><td>10/10</td><td>10/10</td></tr><tr><td></td><td></td><td>Softball</td><td></td><td>202.2</td><td>539.1 test/medium_2</td><td>10/10</td><td>10/10</td><td>5/10</td><td>8/10</td></tr><tr><td></td><td>L</td><td>Pineapple</td><td></td><td>153.6</td><td>181.6 test/large_1</td><td>10/10</td><td>10/10</td><td>10/10</td><td>9/10</td></tr><tr><td></td><td></td><td>Football</td><td></td><td>421.4</td><td>3371.4 test/large_2</td><td>10/10</td><td>8/10</td><td>0/10</td><td>0/10</td></tr><tr><td></td><td>S</td><td>Eraser</td><td></td><td>24.7</td><td>30.9 test/small_1</td><td>7/10</td><td>6/10</td><td>6/10</td><td>6/10</td></tr><tr><td></td><td></td><td>Match box</td><td></td><td>10.0</td><td>40.6 test/small_2</td><td>9/10 ······</td><td>9/10</td><td>8/10</td><td>8/10 .......</td></tr><tr><td>Prismatic</td><td>M</td><td>Card deck</td><td></td><td>121.8</td><td>29.5test/medium_1</td><td>10/10</td><td>4/10</td><td>8/10</td><td>8/10</td></tr><tr><td></td><td></td><td>Sponge</td><td><img src="images/4db79470d0191e5a8d8e8ba68186ae7ee6e00020a397412d769b4622f34a4b77.jpg"/></td><td>10.0</td><td>160.1 test/medium_2</td><td>8/10</td><td>8/10</td><td>6/10</td><td>7/10</td></tr><tr><td></td><td>L</td><td>Wipe dispenser</td><td></td><td>237.3</td><td>1317.3test/large_1</td><td>10/10</td><td>3/10</td><td>0/10</td><td>4/10</td></tr><tr><td></td><td></td><td>Storage bin</td><td></td><td>237.2</td><td>2632.3 test/large-2</td><td>10/10</td><td>7/10</td><td>10/10</td><td>8/10</td></tr><tr><td></td><td>S</td><td>Nail clipper</td><td>B</td><td>43.9</td><td>5.9 test/small_1</td><td>9/10</td><td>3/10</td><td>5/10</td><td>4/10</td></tr><tr><td></td><td></td><td>Lock</td><td></td><td>111.8</td><td>30.1 test/small_2</td><td>10/10</td><td>8/10</td><td>5/10</td><td>6/10</td></tr><tr><td>Appendaged M</td><td></td><td>Dustpan</td><td>Y</td><td>41.1</td><td>243.1 test/medium_1</td><td>9/10</td><td>6/10</td><td>6/10</td><td>6/10</td></tr><tr><td></td><td></td><td>Handbell</td><td></td><td>142.7</td><td>55.8 test/medium_2</td><td>10/10</td><td>7/10</td><td>10/10</td><td>4/10</td></tr><tr><td></td><td>L</td><td>Saucepan</td><td></td><td>360.9</td><td>398.0 test/large-1</td><td>10/10</td><td>10/10</td><td>4/10</td><td>7/10</td></tr><tr><td></td><td></td><td>Picnic basket</td><td></td><td>625.8</td><td>18021.7 test/large-2</td><td>10/10</td><td>5/10</td><td>9/10</td><td>6/10</td></tr><tr><td></td><td>S</td><td>Rubber duck</td><td></td><td>10.8</td><td>41.6 test/small.1</td><td>10/10</td><td>5/10</td><td>10/10</td><td>9/10</td></tr><tr><td></td><td></td><td>Tape measure</td><td></td><td>187.2</td><td>31.2 test/small_2</td><td>10/10</td><td>9/10</td><td>8/10</td><td>5/10</td></tr><tr><td>Amorphous</td><td>M</td><td>Tape dispenser</td><td></td><td>214.9</td><td>56.2 test/medium_1</td><td>10/10</td><td>10/10</td><td>3/10</td><td>4/10</td></tr><tr><td></td><td></td><td>Grapes</td><td><img src="images/198a6adcb88af0761d0449381ccbcc21c25e2b562c914674ded40fbe4d556587.jpg"/></td><td>55.8</td><td>191.6 test/medium_2</td><td>9/10</td><td>5/10</td><td>10/10</td><td>3/10</td></tr><tr><td></td><td>L</td><td>Headphones</td><td></td><td>193.9</td><td>495.2 test/large_1</td><td>6/10</td><td>4/10</td><td>6/10</td><td>2/10</td></tr><tr><td></td><td></td><td>Easel</td><td></td><td>119.7</td><td>172.8 test/large_2</td><td>7/10</td><td>5/10</td><td>8/10</td><td>7/10</td></tr><tr><td colspan="2">HUG-BENCH val</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Battery</td><td></td><td>132.9</td><td>56.5 val/small.3</td><td>10/10</td><td>9/10</td><td></td><td></td></tr><tr><td></td><td>S</td><td>Glue tube Perfume bottle</td><td></td><td>31.2</td><td>82.8 val/small_4 142.0 val/small_5</td><td>10/10 10/10</td><td>7/10 9/10</td><td></td><td></td></tr><tr><td></td><td></td><td>Toothpaste</td><td></td><td>142.6 33.2</td><td>52.1 val/small_6</td><td>8/10</td><td>2/10</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>413.8 val/medium_3</td><td>10/10</td><td>6/10</td><td></td><td></td></tr><tr><td>Cylindrical</td><td></td><td>Almonds</td><td></td><td>209.5</td><td></td><td>9/10</td><td>9/10</td><td></td><td></td></tr><tr><td></td><td>M</td><td>Thermos</td><td><img src="images/b6275ec42aa77e00c04fe6e6c89cd659191da3413e2f6e8b99c25fd25d222ed9.jpg"/></td><td>297.3</td><td>777.9 val/medium_4</td><td>9/10</td><td>8/10</td><td></td><td></td></tr><tr><td></td><td></td><td>Coffee cup</td><td></td><td>15.7</td><td>50.1 val/medium_5</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Flowerpot</td><td></td><td>37.8</td><td>34.1 val/medium_6</td><td>9/10</td><td>10/10</td><td></td><td></td></tr><tr><td></td><td></td><td>French vanilla Mixing bowl</td><td></td><td>1109.3 245.2</td><td>1216.1 val/large_3 332.6 val/large_4</td><td>10/10 8/10</td><td>10/10 9/10</td><td></td></table>

continued on next page

<table><tr><td>Geometry</td><td>Size Object</td><td></td><td>Real Sim Mass(g) Vol. (cm³）Scene</td><td></td><td></td><td>GT Sim SR Sim SR Tabletop SR Wild SR</td><td></td><td></td><td></td></tr><tr><td rowspan="10">Spheroidal</td><td>Squash ball S</td><td></td><td></td><td>23.1 72.8</td><td>39.4 val/small.3 90.8 val/small_4</td><td>10/10 10/10</td><td>8/10</td><td></td><td></td></tr><tr><td>Apricot</td><td></td><td></td><td>85.7</td><td></td><td></td><td>8/10</td><td></td><td></td></tr><tr><td>Small onion</td><td></td><td></td><td></td><td>121.5 val/small._5</td><td>10/10</td><td>6/10</td><td></td><td></td></tr><tr><td>Garlic bulb</td><td></td><td>50.5</td><td></td><td>75.0 val/small_6</td><td>10/10</td><td>10/10</td><td></td><td>.......</td></tr><tr><td>Apple</td><td></td><td>230.4</td><td></td><td>337.1 val/medium_3</td><td>10/10</td><td>8/10</td><td></td><td></td></tr><tr><td>Bundled socks</td><td></td><td></td><td>49.6</td><td>314.4 val/medium_4</td><td>10/10</td><td>9/10</td><td></td><td></td></tr><tr><td>M Bell pepper</td><td></td><td></td><td>180.6</td><td>429.0 val/medium_5</td><td>9/10</td><td>9/10</td><td></td><td></td></tr><tr><td>Massager</td><td></td><td></td><td>174.1</td><td>372.2 val/medium_6</td><td>10/10</td><td>10/10</td><td></td><td></td></tr><tr><td>Eggplant</td><td></td><td></td><td>530.9</td><td>866.4val/large_3</td><td>10/10</td><td>8/10</td><td></td><td></td></tr><tr><td>Cabbage L</td><td></td><td></td><td>1027.1</td><td>2049.2 val/large_4</td><td>10/10</td><td>7/10</td><td></td><td></td></tr><tr><td></td><td>Small globe Squash</td><td></td><td>67.6 1005.9</td><td>922.8 val/large_5</td><td></td><td>10/10 9/10</td><td>9/10 8/10</td><td></td><td></td></tr><tr><td rowspan="9"></td><td></td><td></td><td></td><td></td><td>1184.4 val/large_6</td><td></td><td></td><td></td><td></td></tr><tr><td>Salt shaker</td><td>Wooden cube</td><td></td><td>13.2</td><td>35.0 val/small_3</td><td>10/10</td><td>8/10</td><td></td><td></td></tr><tr><td>S</td><td>Wooden bridge</td><td></td><td>87.8 12.9</td><td>136.2 val/small_4 10.7 val/small_5</td><td>10/10 10/10</td><td>9/10 9/10</td><td></td><td></td></tr><tr><td></td><td>Anchovies</td><td></td><td>67.3</td><td>76.5 val/small_6</td><td>2/10</td><td>8/10</td><td></td><td></td></tr><tr><td></td><td>Outlets</td><td></td><td>168.7</td><td>272.4val/medium_3</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>92.2</td><td>436.1 val/medium_4</td><td>9/10 7/10</td><td>8/10 0/10</td><td></td><td></td></tr><tr><td>Prismatic M</td><td>Pocky</td><td><img src="images/c041db5fc2b48cdae0d585e9d408ed8102aeef6b12d22f2103ac25eda0ae37d9.jpg"/></td><td>370.5</td><td>423.6 val/medium_5</td><td></td><td>10/10</td><td></td><td></td></tr><tr><td></td><td>Milk box</td><td></td><td>54.1</td><td>24.4 val/medium_6</td><td>10/10 10/10</td><td>7/10</td><td></td><td></td></tr><tr><td></td><td>Remote</td><td></td><td>176.4</td><td>827.2</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Keyboard</td><td></td><td>1970.0</td><td></td><td>val/large_3</td><td>10/10 8/10</td><td>8/10 4/10</td><td></td><td></td></tr><tr><td>L</td><td>Milk carton Cereal box</td><td></td><td>638.7</td><td></td><td>2052.5 val/large-4 2251.7 val/large_5</td><td>10/10</td><td>9/10</td><td></td><td></td></tr><tr><td></td><td>Picture frame</td><td></td><td></td><td>223.4</td><td>595.3 val/large_6</td><td>4/10</td><td>2/10</td><td></td><td></td></tr><tr><td></td><td>Spring clamp</td><td></td><td></td><td>7.9</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="9">Appendaged M</td><td></td><td></td><td></td><td>118.1</td><td>5.7 val/small_3 42.3 val/small_4</td><td>9/10</td><td>6/10 9/10</td><td></td><td></td></tr><tr><td>Wrench Spoon</td><td></td><td></td><td>18.5</td><td>5.0 val/small_5</td><td>8/10 5/10</td><td>3/10</td><td></td><td></td></tr><tr><td>Screwdriver</td><td></td><td></td><td>94.2</td><td>115.0 val/small_6</td><td>10/10</td><td>9/10</td><td></td><td></td></tr><tr><td>Hair brush</td><td></td><td></td><td>90.7</td><td>250.3val/medium_3</td><td>7/10</td><td>5/10</td><td></td><td></td></tr><tr><td>Hammer</td><td></td><td></td><td>390.0</td><td>237.3 val/medium_4</td><td>9/10</td><td>4/10</td><td></td><td></td></tr><tr><td>Mug</td><td></td><td></td><td>425.3</td><td>156.9 val/medium_5</td><td>9/10</td><td>5/10</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>56.7</td><td>210.7 val/medium_6</td><td></td><td></td><td></td><td></td></tr><tr><td>Cleaning brush</td><td></td><td></td><td>126.3</td><td></td><td>7/10</td><td>4/10</td><td></td><td></td></tr><tr><td>Dumbbell</td><td></td><td></td><td>1691.9</td><td>578.1val/large_3 800.9 val/large_4</td><td>8/10</td><td>7/10</td><td></td><td></td></tr><tr><td>L</td><td>Frying pan</td><td>C</td><td>95.2</td><td></td><td></td><td>10/10</td><td>10/10</td><td></td><td></td></tr><tr><td></td><td>Watering can Ab roller</td><td>&amp;</td><td>506.7</td><td></td><td>430.4 val/large_5 668.4 val/large_6</td><td>10/10 8/10</td><td>9/10 8/10</td><td></td><td></td></tr><tr><td rowspan="9">S</td><td>Windup toy</td><td></td><td>X</td><td>12.7</td><td>37.0 val/small_3</td><td>10/10</td><td>8/10</td><td></td><td></td></tr><tr><td>Broccoli Mini stapler</td><td></td><td></td><td>12.8</td><td>21.0 val/small_4</td><td>10/10</td><td>8/10</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>40.0</td><td>49.0 val/small_5</td><td>10/10</td><td>5/10</td><td></td><td></td></tr><tr><td>Jalapeno</td><td></td><td></td><td>34.1</td><td>83.0 val/small_6</td><td>9/10</td><td>3/10</td><td></td><td></td></tr><tr><td>Chips</td><td></td><td></td><td>98.3</td><td>823.5val/medium_3</td><td>10/10</td><td>6/10</td><td></td><td></td></tr><tr><td>Bird toy</td><td></td><td></td><td>11.4</td><td>212.0 val/medium_4</td><td>10/10</td><td>3/10</td><td></td><td></td></tr><tr><td>Amorphous M Turtle toy</td><td>Bear figure</td><td><img src="images/38e1ee2e22481025a86ff879f94f829b838f0f45adf3e306451d0301680cfb5b.jpg"/></td><td>116.4 24.2</td><td>49.7 val/medium_5 31.2 val/medium_6</td><td>10/10 10/10</td><td>8/10 8/10</td><td></td><td></td></tr><tr></table>

## E Simulation Grasping Evaluation

This section details the simulated MANO hand used for grasp rollouts, the realism caveats behind the human grasp oracle, and the dense metrics that complement success rate. Per-object simulation success rates for all 90 objects, the GT Sim SR and Sim SR columns, are reported in Table 6. Qualitative predicted grasps on HUG-BENCH scenes are shown in the main text (Figure 4).

## E.1 Simulated MANO Hand

To evaluate grasps in MuJoCo [67] we bake the fixed-shape MANO hand (Appendix B.3) into a physics-ready MJCF directly from the MANO model [19]. We forward the differentiable MANO layer [60] at the flat zero pose with the fixed shape $\beta ,$ assign each of the 778 mesh vertices to the MANO bone of maximum linear-blend-skinning weight, and model each phalanx as a capsule spanning its proximal-to-child joints, keeping the palm as a convex mesh. We use this capsule hand for all rollouts as its smooth normals give more stable contacts. A right and a left hand are produced from the same shape, the left by mirroring the shape basis about its x-axis.

A total mass of 0.4 kg is split across bones by hull volume, each with a uniform-density inertia. The wrist is a 6-DoF free joint (three slides plus a ball joint) and every finger joint is a ball joint, all driven by critically damped position actuators stable at the 2 ms simulation step. A softer, higherpriority contact compliance is applied to the palm, since recorded human grasps often rest the palm 1–2 cm inside the rigid object mesh and a stiff palm would eject it on lift.

## E.2 The Human Grasp Oracle

The human grasp oracle replays the recorded human grasps and does not reach 100% success. The gap is informative rather than a flaw of the model, and stems from several sources: (i) Aria Gen 2 hand-tracking error [72], where occluded fingers tend to be tracked as too open or too closed, so the recorded grasp is slightly loose or tight; (ii) the objects are rigid, neither articulated nor deformable; (iii) the pre-grasp → grasp → lift trajectory is open-loop and programmed to resemble a real-robot execution, unlike the closed-loop human grasp it is derived from. The oracle therefore measures the quality of the data and assets, and provides a realistic ceiling for the learned model.

How precise must grasps be? To quantify how much spatial precision a successful grasp demands, we displace each ground-truth grasp simultaneously along all three wrist axes before execution,

$$
\mathbf { t } ^ { \prime } = \mathbf { t } + \delta \left( \mathbf { e } _ { x } + \mathbf { e } _ { y } + \mathbf { e } _ { z } \right) , \qquad \delta \in \{ 1 , 2 , \ldots \} \mathrm { c m } ,\tag{6}
$$

![](images/206ae7450fad8ef4fd59f09b1ffea23f8665dd8f112cdd316525624fc2d8d73b.jpg)

and measure the resulting SR over all 90 HUG-BENCH objects (val and test), broken down by object size. Figure 19 reports the falloff. At zero offset the grasp is the recorded human grasp, so SR equals the oracle (∼90%); it then drops steeply, as only 2 cm per axis (a Euclidean shift of $2 \sqrt { 3 }$ ≈ 3.5 cm) already roughly halves SR for medium and large objects and cuts it below 20% for small ones. The tolerance scales with object size, as expected.

Figure 19: GT grasp offset. HUG-BENCH SR (%) vs. ground-truth grasp displacement along all three wrist axes.

Small objects collapse to near-zero SR by 3–4 cm per axis, whereas large objects, which afford larger stable contact regions, retain about 25% SR at 3 cm and 10% at 4 cm before vanishing by 6 cm. Even the most forgiving objects therefore tolerate only a few centimeters of error, showing that the benchmark rewards precise placement rather than coarse proximity to the object.

## E.3 Dense Grasp Quality Metrics

Penetration metrics. Beyond the fingertip contact error (§ 5.2, Eq. 2), we report two penetration metrics at grasp closure, both standard in the hand-object grasping literature [80]: the maximum penetration depth maxi max(0, −di) over hand surface samples i with signed surface distance $d _ { i }$ (negative inside the object), and the hand-object intersection volume |H∩O|. On their own, however, these metrics are flawed: a grasp can have low penetration depth and volume while being entirely unstable, for instance with the object merely tangent to the back of the palm. Two grasps, one 10 cm inside the object and one 10 cm outside, will average to have 0 penetration depth. Penetration volume is similarly flawed. It is normally reported as “lower is better”, but is 0 for any grasp that is entirely outside the object, regardless of distance.

HUG-BENCH val
<table><tr><td>Method</td><td>SR(%)↑</td><td>Obj.</td><td>Int. (%)</td><td>Pen. depth (mm)</td><td>）Miss dist. (mm)Pen.vol.(%)FC error (mm)↓</td><td></td><td></td></tr><tr><td>RGB + PC (full HUG)</td><td> ${ \bf 7 1 . 5 \pm 1 . 8 }$ </td><td>59/60</td><td> ${ \bf 9 0 . 5 \pm 1 . 2 }$ </td><td> ${ \bf 1 3 . 1 \pm 0 . 4 }$ </td><td> ${ \bf 1 6 . 4 \pm } 3 . 4$  </td><td> ${ \bf 1 . 9 \pm 0 . 1 }$ </td><td> ${ \bf 1 9 . 0 \pm 0 . 8 }$ </td></tr><tr><td>w/o crop</td><td> $6 1 . 2 \pm 2 . 0$ </td><td>60/60</td><td> $8 8 . 0 \pm 1 . 3$ </td><td> $1 5 . 4 \pm 0 . 5$ </td><td> $2 1 . 3 \pm 3 . 4$ </td><td> $3 . 1 \pm 0 . 2$ </td><td> $2 1 . 6 \pm 0 . 9$ </td></tr><tr><td>w/o point paint</td><td> $6 1 . 8 \pm 2 . 0$ </td><td>57/60</td><td> $8 8 . 2 \pm { 1 . 3 }$ </td><td> $1 4 . 3 \pm 0 . 4$ </td><td> $2 9 . 8 \pm 4 . 6$ </td><td> $2 . 6 \pm 0 . 2$ </td><td> $2 1 . 9 \pm 1 . 0$ </td></tr><tr><td>w/o 3D loss</td><td> $3 9 . 2 \pm 2 . 0$ </td><td>57/60</td><td> $6 8 . 3 \pm { 1 . 9 }$ </td><td> $1 4 . 4 \pm 0 . 6$ </td><td> $2 0 . 9 \pm 1 . 8$ </td><td> $3 . 5 \pm 0 . 3$ </td><td> $3 3 . 0 \pm 1 . 2$ </td></tr><tr><td>PC only</td><td> $6 4 . 2 \pm 2 . 0$ </td><td>58/60</td><td> $8 1 . 3 \pm { 1 . 6 }$ </td><td> $1 3 . 6 \pm 0 . 4$ </td><td> $3 2 . 0 \pm 3 . 1$ </td><td> $2 . 2 \pm 0 . 1$ </td><td> $2 5 . 6 \pm 1 . 2$ </td></tr><tr><td>w/o crop</td><td> $4 7 . 3 \pm 2 . 0$ </td><td>57/60</td><td> $7 2 . 2 \pm 1 . 8$ </td><td> $1 5 . 6 \pm 0 . 6$ </td><td> $3 0 . 6 \pm 3 . 3$ </td><td> $2 . 8 \pm 0 . 2$ </td><td> $3 2 . 6 \pm 1 . 5$ </td></tr><tr><td>RGB only</td><td> $2 6 . 8 \pm { 1 . 8 }$ </td><td>46/60</td><td> $4 2 . 5 \pm 2 . 0$ </td><td> $1 5 . 5 \pm 0 . 7$ </td><td> $7 4 . 9 \pm 3 . 4$ </td><td> $5 . 5 \pm 0 . 6$ </td><td> $9 5 . 4 \pm 3 . 6$ </td></tr><tr><td>Human grasp (oracle)</td><td> $9 0 . 3 \pm 1 . 2$ </td><td>60/60</td><td> $9 8 . 2 \pm 0 . 5$ </td><td> $1 3 . 1 \pm 0 . 4$ </td><td> $2 . 3 \pm 0 . 7$ </td><td> $2 . 5 \pm 0 . 1$ </td><td> $9 . 4 \pm 0 . 3$ </td></tr><tr><td colspan="8">HUG-BENCH test</td></tr><tr><td>Method</td><td> $\mathrm { S R } \left( \% \right) \uparrow$ </td><td>Obj.</td><td>Int. (%)</td><td></td><td></td><td></td><td>Pen.depth (mm)Missdist. (mm)Pen.vol.(%)FC eror (mm)↓</td></tr><tr><td> RGB + PC (full HUG)</td><td> ${ \bf 7 3 . 0 \pm 2 . 6 }$ </td><td>30/30</td><td> ${ \bf 9 3 . 0 \pm 1 . 5 }$ </td><td>1  ${ \bf 1 1 . 2 \pm 0 . 5 }$  </td><td> $1 2 . 2 \pm { 3 . 7 }$  </td><td> ${ \bf 1 . 6 \pm 0 . 1 }$ </td><td> ${ \bf 1 4 . 6 \pm 0 . 9 }$ </td></tr><tr><td>w/o crop</td><td> $5 8 . 0 \pm 2 . 8$ </td><td>29/30</td><td> $8 2 . 0 \pm 2 . 2$ </td><td> $1 3 . 8 \pm 0 . 8$ </td><td> $2 0 . 9 \pm 5 . 0$ </td><td> $2 . 4 \pm 0 . 2$ </td><td> $2 5 . 7 \pm 1 . 5$ </td></tr><tr><td>w/o point paint</td><td> $5 8 . 3 \pm 2 . 8 $ </td><td>30/30</td><td> $8 3 . 3 \pm 2 . 2$ </td><td> $1 2 . 2 \pm 0 . 7$ </td><td> $2 8 . 7 \pm 6 . 7$ </td><td> $2 . 3 \pm 0 . 2$ </td><td> $2 3 . 3 \pm 1 . 7$ </td></tr><tr><td>w/o 3D loss</td><td> $3 2 . 7 \pm 2 . 7$ </td><td>27/30</td><td> $6 5 . 0 \pm 2 . 8$ </td><td> $1 2 . 1 \pm 0 . 9$ </td><td> $2 9 . 2 \pm 4 . 5$ </td><td> $2 . 2 \pm 0 . 2$ </td><td> $3 5 . 7 \pm 2 . 2$ </td></tr><tr><td>PC only</td><td> $7 0 . 7 \pm 2 . 6$ </td><td>30/30</td><td> $8 4 . 3 \pm 2 . 1$ </td><td> $1 1 . 3 \pm 0 . 6$ </td><td> $2 2 . 5 \pm 3 . 9$ </td><td> ${ \bf 1 . 6 \pm 0 . 1 }$ </td><td> $2 2 . 1 \pm 1 . 5$ </td></tr><tr><td>w/o crop</td><td> $5 0 . 0 \pm 2 . 9$ </td><td>29/30</td><td> $7 1 . 7 \pm 2 . 6$ </td><td> $1 2 . 6 \pm 0 . 8$ </td><td> $2 8 . 8 \pm 3 . 9$ </td><td> $2 . 4 \pm 0 . 3$ </td><td> $3 2 . 8 \pm 2 . 2$ </td></tr><tr><td>RGB only</td><td> $2 9 . 7 \pm 2 . 6$ </td><td>19/30</td><td> $4 6 . 7 \pm 2 . 9$ </td><td> $1 2 . 7 \pm 0 . 9$ </td><td> $9 0 . 5 \pm 7 . 3$ </td><td> $5 . 2 \pm 0 . 6$ </td><td> $1 0 8 . 6 \pm 5 . 1$ </td></tr><tr><td>Human grasp (oracle)</td><td> $9 4 . 0 \pm 1 . 4$ </td><td>30/30</td><td> $9 9 . 3 \pm 0 . 5$ </td><td> $1 0 . 2 \pm 0 . 4$ </td><td> $1 . 5 \pm 0 . 1$ </td><td> $1 . 3 \pm 0 . 1$ </td><td> $7 . 4 \pm 0 . 3$ </td></tr></table>

Table 7: Extended ablation metrics. Additional penetration and contact metrics on HUG-BENCH for the ablation models of Table 2, at each model’s best-val-SR checkpoint, on the val (top) and test (bottom) splits. Per-grasp mean ± SE over 600 val / 300 test grasps. Obj. counts objects with $\geq 1$ successful grasp (out of 60 val / 30 test). Human grasp is an oracle upper bound.

To rectify this, we report the percentage of grasps in which the object is penetrated by the hand at grasp closure. We report penetration depth and volume averages over only grasps that are intersecting the object. For grasps not intersecting the object, we report an average miss distance, or the nearest distance from the hand to the object surface. Lower is better for all three of these metrics. Table 7 reports these metrics on our ablations, extending the main ablation (Table 2) across the same models on the val and test splits. The full HUG model performs best on these new metrics, futher evidence of the grasp quality.

Still, these static penetration metrics should be read with care. Because our grasps come from real human grasps on real objects, and the simulated objects are rigid, a grasp of a soft object reports large penetration even though it is correct; a grasp of a pillow, for example, is almost entirely in penetration against its rigid mesh. As guidance, read the table by row against its SR in Table 7 rather than minimizing penetration alone: a variant with both low penetration and high SR is cleanly accurate, whereas low penetration with low SR indicates fingertips stopping short of contact, and the oracle’s nonzero penetration sets the floor attributable to soft objects and tracking noise. We therefore treat penetration as diagnostic rather than a standalone quality score.

## F Real-World Grasping Evaluation

This section details the real-world evaluation design for the tabletop and in-the-wild settings, the retargeting of predicted MANO grasps to the Ability and WUJI robot hands, qualitative trends, and the value of dexterity over a gripper. Per-object real-world success rates are reported in the Tabletop SR and Wild SR columns of Table 6.

## F.1 Real-World Evaluation Design

We select the checkpoint with the best HUG-BENCH val SR in simulation, and without ever testing the model in the real world, run all 300 HUG-BENCH test trials consecutively, each a single grasp prediction followed by one open-loop execution, with no cuts and no retries. We do not tune our model nor the open-loop execution strategy, discussed in § F.2, to our HUG-BENCH test split objects either on the tabletop or inthe-wild. Therefore, we believe our results are a fair assessment of HUG’s generalization ability. Uncut videos of all trials are on our website.

![](images/d3ba409897963462df1290c1542022af996512f55be2524a20ceea2b422bc9d2.jpg)

We place each object at a single location for tabletop trials (Figure 20), varying only its position and rotation slightly across the 10 trials. We avoid clutter in the tabletop experiments to ensure repeatability, and to ensure fair comparison to Dex1B, which is constrained to a single-object tabletop setting. As HUG is not constrained to the tabletop, we perform in-the-wild experiments in diverse cluttered settings to demonstrate its robustness to real-world clutter.

Figure 20: Real-robot tabletop setup. Ability hand + ZED camera + xArm.

## F.2 Retargeting to Robot Hands

HUG predicts a MANO grasp, which we map to a target robot hand at deployment. We retarget to the Ability hand with AnyTeleop [10] and to the WUJI hand with WUJI retargeting [18]. Figure 10 compares the MANO hand with the robot hands.

Because robot hands differ in size from MANO, we align each robot hand’s fingertips to MANO’s using a single fixed offset, estimated from a simulation visualization. All offsets are expressed in the robot’s hand frame (different from the MANO hand frame): the origin is at the palm, the zaxis points toward the fingertips, the x-axis points outward from the palmar surface, and the y-axis points toward the thumb. We apply no correction for the WUJI hand, whereas the Ability hand uses a translation of [0.020, 0, 0.025] m together with a 10◦ rotation about the y-axis. At deployment, the hand first moves to a fixed-shape pre-grasp whose wrist is offset from the predicted grasp by [−0.05, 0, −0.02] m for WUJI and [−0.04, 0, −0.01] m for Ability. It then linearly interpolates to the grasp pose, applies a per-joint force-close, and lifts the object while holding this pose.

## F.3 Qualitative Observations

Easiest and hardest objects. The easiest objects are rounded and convex, sized to fit the hand (e.g. pear, pineapple, hacky sack), which afford many stable enveloping grasps. The hardest are objects too large for the hand to wrap (e.g. football, wipe dispenser) or irregular and articulated objects that are difficult to grasp open-loop (e.g. nail clipper, headphones).

Robustness. Because 1M-HUGS is collected across many environments and lighting conditions, HUG is robust to changes in lighting, object rotation, and viewpoint, and operates on both color and grayscale stereo cameras since the dataset includes grayscale frames. We also observe successful grasps on reflective surfaces, where stereo depth is noisier.

Hands versus gripper. Our comparison against CAP, a strong gripper policy, on identical objects shows that the benefit of dexterity is object-dependent. On large or irregular objects, an antipodal gripper finds no stable pair of opposing faces and tends to slip, whereas an enveloping multi-finger grasp cages the object. On heavy objects whose graspable feature is offset from the center of mass, a two-finger pinch approximates a single contact line that gravitational torque pivots the object out of, while a multi-finger grasp distributes contacts and resists that torque. On small, thin, or pinchable objects the gripper remains competitive. Dexterity thus pays off precisely on the large, irregular, heavy, and off-center objects that grippers handle worst.