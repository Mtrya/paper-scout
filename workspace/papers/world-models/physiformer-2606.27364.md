P HYSI F ORMER: Learning to Simulate Mechanics
in World Space

Yiming Chen

Yushi Lan

Andrea Vedaldi

Visual Geometry Group, University of Oxford

arXiv:2606.27364v1 [cs.CV] 25 Jun 2026

{yiming,yushi,vedaldi}@robots.ox.ac.uk

Abstract
We present P HYSI F ORMER, a diffusion transformer for physically-plausible 3D object motion. Unlike video world models that operate in view-dependent pixel space,
P HYSI F ORMER represents objects as 3D meshes expressed in world coordinates.
Given the initial vertex positions and velocities, as well as object material type,
rigid or elastic, the model samples future vertex trajectories. While related neural
physics approaches build on ad-hoc latent spaces or explicitly enforce rigidity and
causality, P HYSI F ORMER shows that excellent results can be obtained without any
such inductive biases, by casting vertex trajectory prediction as a single denoising
diffusion process directly in world coordinates. The probabilistic formulation
captures uncertainty in the learned dynamics, enabling diverse plausible futures
from initial conditions, making this framework potentially useful for applications
with unobserved uncertainty. The model features attention factorised over time,
space, and objects for efficiency, enabling permutation-invariant multi-object reasoning without needing explicit object encoding. Trained on over 100k simulated
trajectories, P HYSI F ORMER generates rigid and elastic mechanics, and generalises
to mixed-material settings, unseen real-world geometries, and larger object counts.
It substantially outperforms autoregressive baselines in trajectory accuracy, rigidity
preservation, and momentum-based physical consistency. Our results position
coordinate-space diffusion as a promising step toward view-invariant, geometryaware world modelling for robotics, graphics, and physical design. Visualisations,
code, and models are available at https://yimingc9.github.io/physiformer.

1

Introduction

Predicting how physical systems evolve over time is central to spatial intelligence and world modelling
and has applications to robotics, computer graphics, engineering, and more. Motivated by large-scale
video datasets and the success of diffusion transformer architectures [45], several authors [11, 44]
have sought to reduce the problem of modelling physical systems to video generation. However,
a video is a fairly indirect and ambiguous representation of the state of a physical system, as it
intertwines geometry and motion with viewpoint, lighting, and occlusion.
Here we reformulate the problem directly in the space of 3D models, similar to physics simulators
like MuJoCo [54] and PyBullet [17]. We are inspired by RenderFormer [60], which models light
transport over triangle meshes, and ask whether a neural network can similarly predict mechanical
dynamics using 3D meshes as a representation of the geometry of the system. We choose 3D meshes
because of their ubiquity in applications, making this investigation directly relevant to them.
To answer our question, we introduce P HYSI F ORMER, a diffusion transformer that takes as input 3D
mesh vertices representing one or more objects and per-vertex initial velocities. As shown in fig. 1,
Preprint.

Time
2x
rubber
rubber
+metal

t=0

Type
here.
Vertexequation
Positions

Mesh
Topology
+

2x
metal

Vertex Velocities

PHYSIFORMER

Figure 1: P HYSI F ORMER overview. Given initial per-vertex positions X0 ∈ RN ×3 and velocities
V0 ∈ RN ×3 , and material conditions of (1) rigid, (2) deformable, or (3) mixed, P HYSI F ORMER
predicts full-sequence future vertex trajectories in a single forward pass, producing physically
plausible multi-object dynamics, with mesh topology imposed at inference time. Output can be
rendered as 4D mesh motion under arbitrary conditions.
P HYSI F ORMER generates the trajectory of mesh vertices over a temporal window, approximating
how the system should behave under the laws of mechanics and capturing properties such as inertia,
gravity, and collisions. We also attach different material properties to the objects, allowing them to be
either rigid or elastic.
We build our model on top of a general-purpose Diffusion Transformer (DiT) [45], conditioned on the
initial state of the system to generate the system’s future as a sample from an underlying distribution
of physically plausible trajectories. While the architecture is mostly generic, we introduce small but
effective modifications to better suit the problem of modelling mechanics. In particular, we factorise
self-attention across time and space to improve the model’s efficiency, as well as objects, which
implicitly encodes the existence of separate objects in the scene without the need to introduce special
tokens or embeddings, and in a permutation-invariant manner. Unlike recent approaches [16], we
perform diffusion in raw 3D coordinate space—without the added complexity of learning latent
features via autoencoders—following the Just image Transformers (JiT) [33] framework.
We train our model on simulated scenes containing either all rigid or all elastic objects undergoing
collisions and complex motion. Once trained, the model generalises to novel object combinations,
geometries, dynamic and stationary configurations, mixed-materials, and larger object counts.
A key feature of our model is that it generates the entire trajectory of the system in one go. This
is a departure from the autoregressive (AR) approach often used to model physics [50, 48, 35, 65],
where the model is trained to predict the next state of the system given the current state. AR models
are motivated by mechanics, where, under mild assumptions, instantaneous vertex positions and
velocities fully characterize the physical state and thereby define a Markovian system. In fact, authors
have applied this AR assumption with some success even when the state is only partially observable
in a feature space [3, 29].
In our experiments, we compare our approach against AR alternatives. We show that P HYSI F ORMER
obtains far better performance. For one, this is due to a mismatch between training and testing statistics
for AR frameworks, which can be mitigated but not removed entirely by Diffusion Forcing [14]
and Self Forcing [25]. Moreover, this is due to irreducible error accumulation, which, for instance,
causes the shape of rigid objects to gradually deform over time. In practice, AR modelling works
well for particle systems [50, 48, 35, 65], where maintaining the long-term coherence of 3D shape is
irrelevant, or in models that hard-code assumptions such as rigidity [2, 47], which makes them far
less generalisable (for example unable to model elastic objects).
To summarise, our contributions are fourfold: (1) we introduce a unified neural model that simulates
mechanical systems with both rigid and elastic objects directly in 3D trajectory space; (2) we
demonstrate that a general-purpose diffusion transformer with factorised attention over time, space,
and objects can model such dynamics effectively; (3) we show that generative modelling captures
uncertainty in properties such as mass and friction which are not provided as conditioning variables,
enabling diverse plausible futures impossible for deterministic simulators; and (4) we find that
2

autoregressive approaches with the same input formulation are limited by error accumulation, or
require ad-hoc design choices to compensate for this. Our findings suggest that spatial intelligence
can be supported effectively on top of 3D meshes, which may have direct applications to robotic
simulation, gaming, content creation, and engineering.

2

Related Work

Traditional Physical Simulation. The world is governed by physical laws, and modelling system
behavior via physics-based simulation has long been a central goal in engineering. Substantial
progress has been made in motion and material modelling [43, 40, 56], contact resolution [10, 32],
and time integration [5], spanning discrete and continuum representations for structures such as
cloth, thin shells, and fluids. Despite their strong physical grounding and realism, these approaches
remain computationally expensive, algorithmically complex, and difficult to generalize, in part due to
domain-specific optimisation constraints.
Per-scene Optimized Physical Dynamics. Recent work couples 3D scene representations (e.g.,
NeRFs [41] and 3D Gaussian splats [30]) with learned dynamics or inverse-physics optimization,
typically overfitting per scene to recover deformable object motion from videos [65, 58, 66, 26].
While these methods enable joint reasoning over geometry, appearance, and motion, they often
require dense multi-view capture or tracking supervision, and remain limited by strong simulator
assumptions (e.g., spring-mass models, simplified contact).
Learning-based Physics Simulation. Developments in deep learning have ushered in data-driven
approaches to physical simulation. Convolutional networks have been effective for learning physics on
regular grids [22, 7]. To model more general dynamics, graph neural networks (GNNs) [49] provide
a natural framework for particle-based simulation. Particle states are represented as node features,
interactions are encoded by edges, and the system is updated by message-passing conditioned on the
previous state [6, 42, 34, 48, 35, 65, 21]. With the advent of Transformers, Transformer with Implicit
Edges (TIE) [50] replace edge-based message passing with attention. Several works further adapt
neural simulation to mesh-based representations: MeshGraphNets [46] extend message passing to
mesh discretizations, FIGNet [2] and HopNet [57] introduce face-based and higher-order interactions
for contact-rich dynamics, and HCMT [59] combines hierarchical mesh structures with Transformer
modules for flexible-body collisions. This line of work shows the value of mesh-aware inductive
biases, but often requires additional geometric machinery such as explicit connectivity, topology
preprocessing, hierarchical structures, or learned shape representations as in SDF-Sim [47]. In
contrast, P HYSI F ORMER models dynamics as diffusion over raw 3D vertex trajectories, supporting
multiple objects and materials without specialized contact modeling.
Autoregressive Prediction in Visual Feature Spaces. As noted in the previous section, deterministic
autoregressive models have become widely adopted for future prediction. This pipeline can be
applied to the Vision Foundation Model (VFM) feature space. Given four context frames, DINOForesight [29] predicts latent futures that can be decoded into interpretable outputs such as depth and
segmentation, demonstrating the practical effectiveness of autoregressive prediction.
Diffusion Models for World Simulation. Video Diffusion Models (VDMs) [11, 8] are rapidly
advancing visual content generation to convincingly simulate the physical world. However, recent
analyses find that VDM outputs frequently violate Newtonian dynamics [31] and do not learn physical
laws from videos alone [28], limiting their suitability for settings like robotics where physical fidelity
is critical. To address this, several works inject explicit physics signals into VDMs, enabling forceconditioned image-to-video generation with strong generalization [20, 37, 15]. While effective,
these approaches require objective-specific training. Moreover, diffusion in pixel space is inherently
view-dependent, which complicates enforcing viewpoint-invariant physical consistency. Motivated
by this limitation, prior work explores diffusion over 3D representations [39, 62], but these methods
are largely restricted to generating single, static objects. More recent efforts add dynamics [64, 16],
yet they remain focused on single-object settings and leverage motion in videos for animation.
3

3

Method

We consider the problem of animating a 3D scene in a physically-plausible manner. We model the
scene as a triangular mesh, with N vertices total, given by positions X(t) ∈ RN ×3 indexed by time
t ∈ R, and faces F ⊂ {1, . . . , N }3 , defined by triplets of vertex indices. A scene can contain several
objects, modelled as different connected components of the mesh. We denote the instantaneous
velocities of the vertices as V (t) = dX/dt|t ∈ RN ×3 . Given the initial scene configuration, X(0)
and V (0), our goal is to draw a sample from a stochastic process (X(t))t>0 that is consistent with
the underlying physics of the scene. We aim to learn a model of this process from data.
We also assume that, given sufficient initial information, the data can be generated deterministically
by a physics simulator (e.g., Genesis [4]). This means that there exists a function S which takes in
the state of the system (X(t), V (t)) at time t and contextual information Y independent of time to
output the future trajectory (X(t′ ), V (t′ ))t′ ≥t = S(X(t), V (t)|Y ). Here, Y specifies all properties
sufficient to carry out the simulation, such as the mesh topology, gravity, friction and restitution.
Under these assumptions, the dynamics are Markovian, which makes autoregressive next-state
prediction a natural modelling choice. However, in practice, it is challenging for a model to learn all
relevant state and contextual information, and the Markov assumption is only approximately satisfied.
In our experiments we pass to the model some material properties y ⊂ Y of the objects, but leave
other constant properties implicit, to be determined during learning.
Partially because of this, we find that one-shot trajectory generation is substantially more effective
than autoregressive rollout. Another benefit of such a model is to avoid exposure bias due to the
mismatch between teacher-forced training and autoregressive rollout at test time.
Discretizing time. To simplify modelling with little loss of generality, we discretize time, and
consider trajectories X, V ∈ RT ×N ×3 with T steps t ∈ {1, . . . , T }. We use the symbols Xt = X(t)
and Vt = V (t) to denote the t-th slices of these tensors. Our goal is to learn the conditional
trajectory distribution p(X | X0 , V0 , y). We compare that to autoregressive methods that model the
transition p(Xt+1 , Vt+1 | Xt , Vt , y) and roll it out iteratively. Note that autoregressive generation
requires predicting both X and V because velocity is part of the Markov state, whereas our one-shot
formulation predicts X directly (alternatively, Vt can be approximated as Vt ∝ Xt − Xt−1 ).
3.1

P HYSI F ORMER

In order to model the distribution p(X|X0 , V0 , y), we propose P HYSI F ORMER (fig. 2), a diffusion
model that uses a JiT [33]-style objective and a Diffusion Transformer (DiT) [45] backbone. While
we make a point to use a general-purpose architecture, we make minimal modifications to capture the
structure of our data. Instead of introducing object-identity embeddings to distinguish tokens that
belong to different objects, we interleave global and per-object spatial attention to make the model
object-aware while also being invariant to the order or identity of the objects. By denoising full
trajectories as a single prediction target, P HYSI F ORMER infers positions jointly across time, vertices,
and objects, promoting globally consistent generated motion.
Diffusion Model. We briefly summarise the diffusion framework we use. Let x ∼ p(x) be a random
vector from the target data distribution, and let ϵ ∼ N (0, I) be a normal noise vector of the same
dimension. During training, we construct a noised sample zτ = τ x + (1 − τ )ϵ by mixing data
and noise according to τ ∈ [0, 1] [36, 38, 1]. Thus, zτ is pure noise for τ = 0 and follows the
data distribution for τ = 1. The flow velocity v(zτ , x, τ ) = dzτ /dτ = (x − zτ )/(1 − τ ) is the
derivative of zτ with respect to τ (not be confused with the vertex velocity V above). The model is
trained to predict the marginal velocity v(zτ , τ ) = Ex,ϵ [v(zτ , x, τ ) | zτ ]. For this, we train a neural
network xθ (zτ , τ ) with parameters θ expressing the flow velocity as v(zτ , τ ) = v(zτ , xθ (zτ , τ ), τ )
and minimising the loss
2

xθ (τ x + (1 − τ )ϵ, τ ) − x
. (1)
1−τ
During training, we sample τ from a logit-normal distribution logit(τ ) ∼ N (µ, σ 2 ), where µ = −0.8
and σ = 0.8. At inference time, generation proceeds by integrating the corresponding ordinary
differential equation (ODE), dzτ /dτ = vθ (zτ , τ ), from τ = 0 to 1 to obtain a sample z1 ≈ x. In
practice, we solve the ODE numerically with the Heun integrator with 50 steps.
2

L(θ; τ ) = Ex,ϵ ∥v(zτ , xθ (zτ , τ ), τ ) − v(zτ , x, τ )∥ = Ex,ϵ

4

Spatial
RoPE

Material Embedding

+

Temporal
RoPE

+

Vertex Tokens

Diffusion

Temporal Attention

⋰

Object Attention

0

Temporal Attention

𝑇−1

Spatial Attention

Register
Tokens

Linear Head

N Vertex Coordinates

x6
DiT Backbone

Figure 2: P HYSI F ORMER Architecture. During training, input mesh vertex coordinates in RT ×N ×3
are projected into hidden dimension D = 1024 via a linear embedder x_embed, and diffused with
noise according to the flow-matching schedule. Each noised vertex token is additively conditioned
on first-frame position and velocity embeddings (via separate x_embed_cond and v_embed) and a
material embedding. We use 16 prepended global register tokens to aggregate context across the
factorized DiT-L backbone with 4 × 6 layers. The tokens are replicated and consolidated over time,
vertices, and objects for spatial, temporal, and object-level attention, respectively, each using its
corresponding RoPE. At inference, iterative denoising produces clean vertex trajectories, which are
projected back to coordinate space via a linear head and assembled into triangle meshes using the
provided topology for view-invariant, arbitrary-material 4D rendering.

Because the network outputs the “clean” data x while minimising the velocity prediction error, this
is called x-prediction with v-loss. As suggested by JiT [33], x belongs to a lower-dimensional data
manifold than v (see the Manifold Assumption [12]), which simplifies prediction.
3.2

Diffusion Architecture

Our goal is to model the distribution p(X|X0 , V0 , y) of vertex trajectories conditioned on initial
position and velocity, and contextual information y about the scene. We thus design a neural network
that, given X0 , V0 , y and a noised version z of the data X, denoises it to predict the “clean” data
X, i.e., X ≈ xθ (zτ , X0 , V0 , y, τ ). We base our model on general-purpose Diffusion Transformer
(DiT) [45] with modifications that capture the structure of our data, as explained below.
Encoding. The transformer requires the data to be converted to a stream of tokens. To this end, the
noisy vertices in z ∈ RT ×N ×3 are individually projected to a D-dimensional space using a linear
layer and the result is flattened to a sequence of T N D-dimensional tokens.
The network is conditioned on the initial states X0 , V0 ∈ RN ×3 , which are projected to dimension D
using separate embedding functions to distinguish initial conditions from model input. The resulting
N × D tensors are broadcast-summed to the vertex embeddings.
Additionally, we condition the model on two types of object materials by adding material embeddings
via a two-layer MLPmat . Rigid and elastic materials are embedded with inputs 0 and 1 respectively,
assigned to objects and broadcast to vertices for efficiency. The success of this simple conditioning in
P HYSI F ORMER suggests its capability to model various physical properties with explicit conditioning
in a similar fashion. The result of the encoder is thus a sequence z ′ of T N tokens, each of dimension
D, encoding z, X0 , V0 and the material properties in y.
Structured Attention. The sequence z ′ is processed by a stack of DiT blocks. By themselves, these
would operate using self-attention on the entire sequence of length T N , with cost O(T 2 N 2 ). We
5

suggest instead structuring attention for efficiency, and also use it to implicitly encode information
in the model. As done in prior works [8, 9], we employ alternating spatio-temporal attention; our
novelty is further factorising spatial attention into full spatial and object-level attention. Concretely,
for spatial attention, the T N tokens are reshaped as (T, N, D) treating T as batch dimension, so
attention is applied independently within each of the T frames. For object-level attention, padding
the K objects so that they have an equal number of vertices N/K, tokens are grouped by object as
(T K, N/K, D), so attention is applied within each object at each time step. For temporal attention,
tokens are reshaped as (N, T, D), so each vertex attends across time. This factorisation reduces the
attention cost to O(T N 2 + N T 2 ). The interplay between local and global spatial attention makes
the model aware of the different objects without using explicit object identifiers and in a manner that
is insensitive to the order of the objects in the token sequence.
Spatial and Temporal Rotational Positional Encoding. We inject spatio-temporal position information into the transformer blocks with rotary positional encodings (RoPE) [53], which naturally encode
relative information consistent with the global spatio-temporal translation invariance of physical dynamics. RoPE is applied separately in temporal and spatial attention: temporal attention uses standard
1D RoPE over time indices, while full and object-level spatial attention use coordinate-conditioned
RoPE, following RenderFormer [60], so attention depends on relative 3D vertex offsets. Specifically,
we multiply each of x, y, z by log-spaced base-2 frequencies, concatenate the resulting phases, and
convert them to sin / cos coefficients for the standard block-wise 2 × 2 RoPE rotations of query and
key vectors. If the number of coordinate-derived phases differs from the per-head rotary dimension,
we pad or truncate them; zero-padding gives identity rotations, leaving the remaining channel pairs
unrotated. Sixteen shared register tokens are replicated and consolidated across factorised attention
to aggregate global information, with details in section A.1.

4

Experiments

We evaluate the ability of P HYSI F ORMER to simulate the dynamics of rigid and deformable objects.
4.1

Dataset

We create a synthetic dataset with the Genesis physics simulator [4]. Each scene is represented as
(X0 , V0 , X, F, M ), where X0 , V0 ∈ RN ×3 are initial vertex positions and velocities, X ∈ RT ×N ×3
is the ground-truth vertex trajectory, F ∈ ZNf ×3 is the triangular face connectivity with Nf faces,
and M ∈ {0, 1}N denotes per-vertex material, rigid or elastic. Objects correspond to connected
mesh components with a shared material; the initial velocities are sampled per object and converted
to per-vertex velocities; the elastic material is defined to have a fixed Young’s modulus.
We generate 49-frame trajectories in a bounded [−1, 1]3 container with timestep ∆t = 1/240s.
Across scenes, we randomize the number of objects, object sizes, shapes, materials, and initial conditions, while keeping density and environmental parameters fixed. We choose simulator parameters
to make collisions as elastic as the simulator allows: friction is minimized, while the simulator’s
standard damping is retained for numerical stability. As a result, collisions are near-elastic but not
perfectly energy-conserving.
We generate four datasets of increasing complexity, summarized in table 1. Mesh templates used
in training are visualised in fig. 6 in the appendix. All scenes are simulated in the same bounded
environment, where objects may collide with each other and the box walls. In the floor-start rigid
settings, objects are placed on the floor without overlap and assigned random initial linear velocities,
with zero velocity for a random subset. A small 2◦ orientation jitter allows initially stationary but
unstable objects to fall under gravity. The airborne-start setting follows the same setup but with
Table 1: Summary of the four datasets used in our experiments.
D1

10k rigid floor-start scenes with 1–5 convex objects from 15 templates ((a) in fig. 6). Each object has 4–20 vertices, with at most 88
vertices per scene.

D2

15k rigid floor-start scenes with 1–5 objects from 25 convex and 10 concave templates ((a) and (b) in fig. 6). Objects have 4–88
vertices, with at most 356 vertices per scene.

D3

60k airborne-start rigid scenes: 35k with 1–5 objects and 25k with 6–10 objects. In each object count group, 10k scenes include
nonzero initial angular velocity. Objects are selected from the same mesh templates as D2 .

D4

20k elastic scenes with 1–5 objects. 10k floor-start and 10k airborne-start. Objects are selected from the same mesh templates as
D2 .

6

PHYSFORMER

t = 48

t = 30

t = 10

GT

Figure 3: Qualitative comparison of P HYSI F ORMER against autoregressive baselines on trained
10k rigid object data. At t = 10, rigidity is not preserved in ΦAR_ctx1 , but objects remain rigid
across all other models. As t increases, all autoregressive baselines diverge due to error accumulation: stationary objects fail to remain at rest, objects escape the implicit bounding box, and
object shapes deform severely, even in the strongest AR baseline TIE r=1.0 . P HYSI F ORMER consistently maintains object rigidity and produces physically plausible long-horizon motion.
objects spawning in the air. Train/validation/test splits are precomputed using a fixed-seed stratified
split to match the data distribution. All training scenes contain a single material type.
4.2

P HYSI F ORMER Implementation Details

P HYSI F ORMER is trained from scratch following JiT’s framework in section 3.1. The model predicts
vertex positions, which are combined with the original face connectivity for rendering during inference.
We sample noise as ϵ ∼ N (0, I)×noise_scale, where noise_scale = 0.1, and analyze this choice
in section 4.8. Optimization uses AdamW, and we maintain an exponential moving average (EMA)
of parameters with decay 0.9999. Training uses automatic mixed precision (AMP) with bf16. We
use PyTorch Flash SDPA for speed and compatibility with input-dependent masking. We train on 2
NVIDIA H100 GPUs with 94GB memory, with an effective batch size of 64 and lr = 4e−5.
For comparison with AR baselines, we train a model using the DiT-L backbone on D1 for 70k
iterations, denoted as P HYSI F ORMER-L-10k. We employ linear learning-rate warm-up over 780
steps, followed by a cosine decay schedule toward lr = 5e−6. We train P HYSI F ORMER by
finetuning P HYSI F ORMER-L-10k on rigid dynamics in the D1 + D2 + D3 rigid dataset for 27k
iterations, and then further finetune on elastic object motions in D4 for 12k iterations. For the latter,
we ensure that the model sees a 60/40 ratio of rigid to elastic material scenes. Inference uses EMA
weights, and we clamp first-frame positions to X0 during sampling. We use 50 Heun sampling
steps for all sampled results in the main paper, though we show that fewer steps yield comparable
performance in section B.
4.3

Baselines

Autoregressive Model ΦAR . We design and implement a transformer autoregressive framework
(ΦAR ) for next-timestep mesh vertex position prediction. We optimise performance in two ways
inspired by previous works [13, 29]. First, we explore variable context window lengths. Longer
windows should provide more anchoring information for more stable rollout. Second, to bridge the
train-test domain gap between ground-truth-conditioned training and self-conditioned inference, we
investigate noise injection in the context window. The design choices and implementation details are
in the appendix section A.2.
TIE. TIE is an autoregressive transformer that uses implicit edges defined by tokens in attention
to mimic a graph neural network for next-timestep system-state prediction. In particle-based dynamics prediction, TIE outperforms established GNN baselines [34, 48], continuous convolution
methods [55], and transformer-based models with explicit edge encoding [19]. Our training setup
is analogous to a particle-based system, as we define topology only at inference time. Therefore,
we use TIE as a strong baseline. We reproduce TIE results using its official training setup, adapted
to our mesh trajectory dataset. Specifically, for fair comparison, we treat each vertex as a separate
particle without hard-coding rigidity like in TIE’s setup as discussed in section 4.4. TIE uses radius
r to stipulate the maximum distance at which two particles are still connected by an implicit edge
7

Table 2: Comparison of P HYSI F ORMER-L-10k against baselines on 250 test samples, trained on
10k rigid dataset. MSE measures mean per-vertex position error against GT trajectories; Rigidity Loss
measures implicit rigidity preservation (Equation (2)). Momentum Drift Ratio measures inference
momentum drift from initial system momentum compared against that of GT, where values closer to
1 are better (Equation (3)). We show results for 10-frame and 49-frame inference. Best is bolded;
second best is italic. As P HYSI F ORMER-L-10k is a generative model, we show one-shot inference
and distributional statistics across 5 generations. P HYSI F ORMER-L-10k is best on average.
MSE (×10−3 )↓

Rigidity Loss (×10−4 )↓

Momentum Drift Ratio

Method

10-frame

49-frame

10-frame

49-frame

10-frame

49-frame

P HYSI F ORMER (Ours)
5 generations µ
5 generations σ
ΦAR_ctx1
ΦAR_ctx4
ΦAR_ctx4_noised
T IE r=0.4
T IE r=1.0

0.0953
0.0883
0.293
1.78
0.896
1.13
0.157
1.73

9.55
9.55
13.5
217
101
117
17.1
14.8

0.0411
0.0407
0.0210
0.928
0.0960
0.0846
0.328
0.137

0.185
0.185
1.22
143
27.6
18.5
31.0
20.6

4.45
4.42
0.049
21.7
3.82
3.69
2.64
2.06

1.91
1.94
0.027
11.9
8.37
7.95
2.91
2.29

captured through attention. Given the bounding box and range of motion in our dataset, we train the
model with r = 0.4 and r = 1.0. We show additional results with larger radii in the appendix table 5.
4.4

Evaluation Metrics

Trajectory MSE Error. Following [50, 48, 55, 34], given the ground-truth and predicted trajectories
X, X̂ ∈ RT ×N ×3 , we define the mean-square error (MSE) as LMSE (X̂|X) = ∥X̂ − X∥22 /(T N ).
Rigidity Preservation. Maintaining object rigidity is a challenge, as seen in severe object deformation across AR baselines in fig. 3. In TIE’s original Boxbath dataset [34], rigidity is hard-coded via a
single predicted rigid transformation for the rigid object (box), which is used to derive motion for all
constituent particles. Thus, we evaluate our model’s ability to implicitly learn rigidity by averaging
the deviation across all objects and frames from a rigid transformation of each object’s first-frame
position. We use the Kabsch algorithm [27] to compute a best-fit rigid transform with rotation matrix
R ∈ SO(3) and translation vector b ∈ R3 and define the rigidity error as:
T

Lrigid (X̂|X0 , F) =

1X
T t=1

X
O∈cc(F )

1
min
∥X̂t,O − X0,O R − b∥F
2 ,
|O| (R,b)∈SE(3)

(2)

where cc(F) denotes the connected components of the mesh, O ⊂ {1, . . . , N } is a connected
component (object), and X∗,O denotes the subset of vertices that belong to the component.
Momentum Prediction. We also test the ability of the model to predict the momentum of the system,
which is a fundamental physical property. We compute the momentum of object O at timestep t by
approximating the velocity of the center of
Pmass as the average of the vertex velocities computed
using finite difference Pt,O (X) =Pm(O) i∈O (Xt,i − Xt−1,i )/(O∆t). We then denote the total
momentum at time t as Pt (X) = O∈cc(F ) Pt,O (X) and define the momentum drift ratio as:
, T
T
X
X
2
2
Rmom (X̂|X, X0 , F ) =
Pt (X̂) − P0 (X̂)
∥Pt (X) − P0 (X)∥2 .
(3)
2

t=1

t=1

If the simulated and predicted trajectories are identical, this ratio equals 1.
4.5

Comparisons Against Baselines

For fairness, all baselines are trained on the 10k rigid-body dataset D1 using a stratified 9500/250/250
train/validation/test split and are compared with P HYSI F ORMER-L-10k trained on the same data.
Results are shown in table 2 and fig. 3. All AR approaches suffer from error accumulation, with mesh
8

15 Objects

Unseen Geometry

Time

Figure 4: P HYSI F ORMER generalizes to complex real-world object geometries and object counts
not seen during training. Top: Inference on 2 deformable objects (fish and teapot) plus 1 rigid
bunny, each with 100 vertices per object. Deformation is most visible for the middle-frame purple
teapot. P HYSI F ORMER allows mixed-material inference although training only saw uniform material
across all objects per scene. Bottom: Inference on 15 rigid objects although training saw at most 10.
Extension to more objects is direct owing to object-level attention, which implicitly encodes object
identity in the architecture design.
deformation and trajectory divergence over time. Short-horizon autoregressive rollouts over the first
10 steps are meaningfully better. Across ΦAR variants, a longer context window and noise injection
help with stability. For TIE, a larger radius models a broader interaction neighborhood, yielding
better results. For the momentum drift ratio, we assume constant mass for computational feasibility.
Because rigidity is largely not preserved in AR baselines, this metric should be interpreted as a rough
approximation. Importantly, we show in the appendix table 5 that all AR models perform well when
conditioned on ground-truth data at each timestep, highlighting their ability to learn in-distribution
dynamics. Note that P HYSI F ORMER-L-10k displays a large standard deviation for MSE across
samples (table 2), but MSE alone is not a good indicator of physical plausibility. Slight contact-angle
differences yield different reaction forces in collisions, causing trajectory divergence, while generated
rigid motion can differ from ground truth yet remain physically plausible.
4.6

Generalisation to Unseen Geometries and Object Numbers

P HYSI F ORMER-L-10k generalises well to object geometries and counts not seen in training as shown
in the appendix section C. P HYSI F ORMER is trained on extended data with more complex motion
and multiple material properties, and displays impressive generalisation to three settings. First, it
generalises to complex, real-world meshes, with far more vertices per object, although training used
only the simpler mesh primitives depicted in (a) and (b) in fig. 6. Second, it generalises to object
counts beyond those seen in training. Third, it generalises to mixed-material scenes, although training
saw only uniform material per scene. We illustrate these claims in fig. 4 and on the project page.
4.7

P HYSI F ORMER vs. Physics Simulators

While physics simulators provide high-fidelity trajectories by numerically integrating physical laws,
P HYSI F ORMER offers a powerful learned alternative at inference time. First, it generates physically
plausible motion from only initial positions and velocities, without requiring a fully specified physical
state such as density, friction, or material parameters as input. Second, once trained, it has a capped
inference cost, enabling efficient rollout even for deformable and contact-rich scenes that are costly to
simulate. On an 80-thread Intel Xeon Gold 6338 CPU node, Genesis rigid-body simulation averaged
1–6.5s per sample for 1–10 objects. However, elastic-body simulation averaged 20–36s per sample
for 1–5 objects excluding rendering, more than 5× the P HYSI F ORMER inference time on a single
H100 GPU for 25 denoising steps, for which we see high-quality P HYSI F ORMER outputs as detailed
in table 3. Though physics simulators can be sped up, their material-dependent timing gap highlights
a key advantage of learned rollout: after training, P HYSI F ORMER uses a fixed number of network
evaluations rather than costly per-scene simulation. Third, it generalises to complex, real-world
mesh geometries that can be difficult for simulators to handle robustly. In challenging scenes, where
9

simulators may fail due to contact-resolution artifacts or objects leaving the simulation bounding box
as shown in fig. 7 in the appendix, P HYSI F ORMER can still produce plausible samples.
4.8

Ablations

Noise Scale in Diffusion. Standard diffusion models typically assume normalized inputs, such as
VAE latents, and sample noise from a standard Gaussian. For raw coordinate signals, however,
noise scale is important for generation quality [33, 24]. As shown in table 6 in the appendix trained
on smaller DiT-B based models with 1k data, which we call P HYSI F ORMER-B-1k, we sample
ϵ ∼ N (0, I) × noise_scale and find noise_scale = 0.1 to work best: smaller values hinder
generalisation, while larger values make denoising harder and often introduce jitter. We hypothesize
that this narrower noise distribution is effective because trajectories are strongly conditioned on the
first-frame state. It also stabilizes our coordinate-derived spatial RoPE, whose inputs are noised
trajectories during training and early sampling.
Object-Level Attention vs Object-ID Embedding We compare P HYSI F ORMER-L-10k’s factorised
per-object attention with a baseline that uses alternating full spatial and temporal attention plus
learnable per-vertex object-ID embeddings in R(Nobj +1)×D , including a padding ID. On 10k rigid
trajectories over 49 frames, the two perform similarly: P HYSI F ORMER has slightly worse MSE
(9.6e-3 vs. 9.1e-3) and momentum drift ratio (1.70 vs. 1.53), but better rigidity loss (1.9e-5 vs.
4.8e-5), with comparable qualitative results. Object-ID embeddings do not extrapolate to more
objects, since unseen object indices have no learned embeddings.

5

Conclusion

We presented P HYSI F ORMER, a unified diffusion transformer that learns multi-material, multi-object
mechanics as full-trajectory coordinate diffusion in world space. By generating future mesh motion
jointly rather than autoregressively, P HYSI F ORMER avoids error accumulation while preserving object
coherence through factorised spatial, temporal, and object-level attention. It improves trajectory accuracy, rigidity preservation, and momentum-based physical consistency over autoregressive baselines,
and generalises to unseen geometries, larger object counts, and mixed-material settings. Its current
limits are fixed trajectory length and mesh resolution, motivating longer-horizon generation, spatial
compression, and physics-aware objectives for contact. P HYSI F ORMER uses a general diffusion
objective without explicit collision or object-consistency constraints, leading to occasional spurious
contacts, interpenetration, and rare orientation discontinuities. These artifacts may be reduced through
contact-focused training and tailored physical-consistency losses. Overall, P HYSI F ORMER points
toward geometry-level world models for physically plausible 3D dynamics.

Acknowledgments and Disclosure of Funding
Yiming Chen is supported by the Rhodes Scholarship. Yushi Lan and Andrea Vedaldi are supported
by the European Research Council (ERC) grant CoG 101001212-UNION. We thank Isambard-AI
and Dawn AIRR supercomputers (project code: 0261–5548–9011–1) for supporting this work.

10

References
[1] Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic
interpolants. In Proc. ICLR, 2023.
[2] Kelsey R Allen, Yulia Rubanova, Tatiana Lopez-Guevara, William Whitney, Alvaro SanchezGonzalez, Peter Battaglia, and Tobias Pfaff. Learning rigid dynamics with face interaction
graph networks. arXiv preprint arXiv:2212.03574, 2022.
[3] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili,
Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud,
Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil
Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li,
Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas
Ballas. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning.
arXiv, 2506.09985, 2025.
[4] Genesis Authors. Genesis: A generative and universal physics engine for robotics and beyond,
2024. URL https://github.com/Genesis-Embodied-AI/Genesis.
[5] David Baraff and Andrew Witkin. Large steps in cloth simulation. In Proc. SIGGRAPH, 1998.
[6] Peter W. Battaglia, Razvan Pascanu, Matthew Lai, Danilo Jimenez Rezende, and Koray
Kavukcuoglu. Interaction networks for learning about objects, relations and physics. In
Proc. NeurIPS, 2016.
[7] Saakaar Bhatnagar, Yaser Afshar, Shaowu Pan, Karthik Duraisamy, and Shailendra Kaushik.
Prediction of aerodynamic flow fields using convolutional neural networks. Computational
Mechanics, 64(2), 2019.
[8] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin
Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets.
arXiv.cs, abs/2311.15127, 2023.
[9] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja
Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent
diffusion models. In Proc. CVPR, 2023.
[10] Sofien Bouaziz, Sebastian Martin, Tiantian Liu, Ladislav Kavan, and Mark Pauly. Projective
dynamics: Fusing constraint projections for fast simulation. In Proc. SIGGRAPH, 2014.
[11] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe
Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video
generation models as world simulators. Technical report, OpenAI, 2024.
[12] Olivier Chapelle, Bernhard Schölkopf, and Alexander Zien. A discussion of semi-supervised
learning and transduction. In Semi-Supervised Learning. The MIT Press, 2006.
[13] Boyuan Chen, Yilun Du, Diego Martí, et al. Diffusion forcing: Next-token prediction meets
full-sequence diffusion. In NeurIPS, 2024.
[14] Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent
Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. arXiv,
2407.01392, 2024.
[15] Boyuan Chen, Hanxiao Jiang, Shaowei Liu, Saurabh Gupta, Yunzhu Li, Hao Zhao, and Shenlong
Wang. PhysGen3D: Crafting a miniature interactive world from a single image. CVPR, 2025.
[16] Hongyuan Chen, Xingyu Chen, Youjia Zhang, Zexiang Xu, and Anpei Chen. Motion 3-to-4:
3D motion reconstruction for 4D synthesis. arXiv, 2601.14253, 2026.
[17] Erwin Coumans and Yunfei Bai. Pybullet, a python module for physics simulation for games,
robotics and machine learning, 2016. URL http://pybullet.org.
11

[18] Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers
need registers. Proc. ICLR, 2024.
[19] Vijay Prakash Dwivedi and Xavier Bresson. A generalization of transformer networks to graphs.
Proc. AAAI Workshop, 2021.
[20] Nate Gillman, Charles Herrmann, Michael Freeman, Daksh Aggarwal, Evan Luo, Deqing
Sun, and Chen Sun. Force prompting: Video generation models can learn and generalize
physics-based control signals. In Proc. NeurIPS, volume 2505.19386, 2025.
[21] Artur Grigorev, Michael J Black, and Otmar Hilliges. Hood: Hierarchical graphs for generalized
modelling of clothing dynamics. In Proc. CVPR, 2023.
[22] Xiaoxiao Guo, Wei Li, and Francesco Iorio. Convolutional neural networks for steady flow
approximation. In Proc. SIGKDD, 2016.
[23] Alex Henry, Prudhvi Raj Dachapally, Shubham Pawar, and Yuxuan Chen. Query-key normalization for transformers. arXiv, 2020.
[24] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion
for high resolution images. In Proc. ICML, 2023.
[25] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing:
Bridging the train-test gap in autoregressive video diffusion. arXiv, 2506.08009, 2025.
[26] Hanxiao Jiang, Hao-Yu Hsu, Kaifeng Zhang, Hsin-Ni Yu, Shenlong Wang, and Yunzhu Li.
PhysTwin: Physics-informed reconstruction and simulation of deformable objects from videos.
In Proc. ICCV, 2025.
[27] Wolfgang Kabsch. A solution for the best rotation to relate two sets of vectors. Foundations of
Crystallography, 32(5), 1976.
[28] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi
Feng. How far is video generation from world model: A physical law perspective. In Proc.
ICML, 2025.
[29] Efstathios Karypidis, Ioannis Kakogeorgiou, Spyros Gidaris, and Nikos Komodakis. DINOforesight: Looking into the future with DINO. In Proc. NeurIPS, 2025.
[30] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3D Gaussian
Splatting for real-time radiance field rendering. Proc. SIGGRAPH, 42(4), 2023.
[31] Minh-Quan Le, Yuanzhi Zhu, Vicky Kalogeiton, and Dimitris Samaras. What about gravity
in video generation? post-training Newton’s laws with verifiable rewards. arXiv, 2512.00425,
2025.
[32] Minchen Li, Danny M Kaufman, and Chenfanfu Jiang. Codimensional incremental potential
contact. In Proc. SIGGRAPH, 2021.
[33] Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise. In
CVPR, 2025.
[34] Yunzhu Li, Jiajun Wu, Russ Tedrake, Joshua B Tenenbaum, and Antonio Torralba. Learning
particle dynamics for manipulating rigid bodies, deformable objects, and fluids. In Proc. ICLR,
2019.
[35] Xingyu Lin, Yufei Wang, Zixuan Huang, and David Held. Learning visible connectivity
dynamics for cloth smoothing. In Proc. CoRL, 2021.
[36] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow
matching for generative modeling. arXiv.cs, abs/2210.02747, 2022.
[37] Shaowei Liu, Zhongzheng Ren, Saurabh Gupta, and Shenlong Wang. PhysGen: Rigid-body
physics-grounded image-to-video generation. In Proc. ECCV, 2024.
12

[38] Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and
transfer data with rectified flow. In Proc. ICLR, 2023.
[39] Zhen Liu, Yao Feng, Michael J. Black, Derek Nowrouzezahrai, Liam Paull, and Weiyang Liu.
Meshdiffusion: Score-based generative 3d mesh modeling. In ICLR, 2023.
[40] Eder Miguel, Derek Bradley, Bernhard Thomaszewski, Bernd Bickel, Wojciech Matusik,
Miguel A. Otaduy, and Steve Marschner. Data-driven estimation of cloth simulation models. In
Proc. Eurographics, 2012.
[41] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi,
and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. In Proc.
ECCV, 2020.
[42] Damian Mrowca, Chengxu Zhuang, Elias Wang, Nick Haber, Fei-Fei Li, Josh Tenenbaum, and
Daniel L K Yamins. Flexible neural representation for physics prediction. In Proc. NeurIPS,
2018.
[43] Matthias Müller, David Charypar, and Markus Gross. Particle-based fluid simulation for
interactive applications. In Proc. Eurographics, 2003.
[44] Jack Parker-Holder and Shlomi Fruchter. Genie 3: A new frontier for world models, 2025. URL
https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/.
[45] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proc. ICCV,
2023.
[46] Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, and Peter W. Battaglia. Learning
mesh-based simulation with graph networks. In Proc. ICLR, 2021.
[47] Yulia Rubanova, Tatiana Lopez-Guevara, Kelsey R. Allen, William F. Whitney, Kimberly
Stachenfeld, and Tobias Pfaff. Learning rigid-body simulators over implicit shapes for largescale scenes and vision. In Proc. NeurIPS, volume k, 2024.
[48] Alvaro Sanchez-Gonzalez, Jonathan Godwin, Tobias Pfaff, Rex Ying, Jure Leskovec, and
Peter W. Battaglia. Learning to simulate complex physics with graph networks. In Proc. ICML,
2020.
[49] Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini.
The graph neural network model. IEEE Trans. on Neural Networks, 2009.
[50] Yidi Shao, Chen Change Loy, and Bo Dai. Transformer with implicit edges for particle-based
physics simulation. In Proc. ECCV, 2022.
[51] Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. Self-attention with relative position representations. In Proc. NAACL, 2018.
[52] Noam M. Shazeer. GLU variants improve transformer. In arXiv, 2020.
[53] Jianlin Su, Murtadha H. M. Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu.
RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 2024.
[54] Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based
control. In Proc.IROS, 2012.
[55] Benjamin Ummenhofer, Lukas Prantl, Nils Thuerey, and Vladlen Koltun. Lagrangian fluid
simulation with continuous convolutions. In Proc. ICLR, 2020.
[56] Pascal Volino, Nadia Magnenat-Thalmann, and Francois Faure. A simple approach to nonlinear
tensile stiffness for accurate cloth simulation. In Proc. SIGGRAPH, 2009.
[57] Amaury Wei and Olga Fink. Integrating physics and topology in neural networks for learning
rigid body dynamics. Nature Communications, 16(1), 2025.
13

[58] Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu Jiang.
PhysGaussian: Physics-integrated 3D Gaussians for generative dynamics. In Proc. CVPR, 2024.
[59] Youn-Yeol Yu, Jeongwhan Choi, Woojin Cho, Kookjin Lee, Nayong Kim, Kiseok Chang,
ChangSeung Woo, Ilho Kim, SeokWoo Lee, Joon Young Yang, et al. Learning flexible body
collision dynamics with hierarchical contact mesh transformer. In Proc. ICLR, 2024.
[60] Chong Zeng, Yue Dong, Pieter Peers, Hongzhi Wu, and Xin Tong. RenderFormer: transformerbased neural rendering of triangle meshes with global illumination. In Proc. SIGGRAPH,
2025.
[61] Biao Zhang and Rico Sennrich. Root mean square layer normalization. In Proc. NeurIPS, 2019.
[62] Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3DShape2VecSet: A 3d
shape representation for neural fields and generative diffusion models. ACM Transactions On
Graphics (TOG), 42(4):1–16, 2023.
[63] Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3DShape2VecSet: A 3D
shape representation for neural fields and generative diffusion models. In ACM Transactions on
Graphics, 2023.
[64] Bowen Zhang, Sicheng Xu, Chuxin Wang, Jiaolong Yang, Feng Zhao, Dong Chen, and Baining
Guo. Gaussian variation field diffusion for high-fidelity video-to-4D synthesis. In Proc. ICCV,
2025.
[65] Mingtong Zhang, Kaifeng Zhang, and Yunzhu Li. Dynamic 3D Gaussian tracking for graphbased neural dynamics modeling. In Proc. CoRL, 2024.
[66] Licheng Zhong, Hong-Xing Yu, Jiajun Wu, and Yunzhu Li. Reconstruction and simulation of
elastic objects with spring-mass 3D Gaussians. In Proc. ECCV, 2024.

14

P HYSI F ORMER: Learning to Simulate Mechanics in World Space
Supplementary Material

A

Method Continued

A.1

P HYSI F ORMER

Register Tokens. In input data tokenization, we further prepend Nreg = 16 shared, learnable register
tokens, yielding per-sample token embeddings in R(Nreg +T ·N )×D . The register tokens capture global
context and can potentially remove high-frequency noise in embeddings [18]. Across the temporal
and spatial attention blocks, register tokens are replicated across the corresponding dimensions (time
for full spatial blocks, objects and time for object-level spatial blocks, vertices for temporal blocks)
before attention. They are consolidated back to a single set of tokens via averaging after attention,
allowing global information to be shared consistently across factorised operations. Within temporal
blocks, temporal RoPE assigns register tokens a fixed time index of 0. For spatial blocks, each register
token uses the mean position of the vertices selected by its context mask for spatial RoPE.
DiT Details. Up to the modifications described above, the DiT blocks are standard. Each block uses
RMSNorm [61] and AdaLN [45] conditioning from the diffusion timestep embedding to produce
shift/scale modulation and gated residuals, followed by non-causal multi-head self-attention with
QK-normalization [23] and a feed-forward network with SwiGLU activations [52]. By default, we
train P HYSI F ORMER, which mirrors DiT-L in size, instantiating 24 blocks in total. Each block uses
16 attention heads and hidden size D = 1024.
A.2

Baseline ΦAR

Autoregressive Dynamics Modeling. The autoregressive predictor ΦAR outputs next-timestep
velocities for N vertices given context length L. Starting with L = 1, given inputs Xt ∈ R1×N ×3
′
and Vt ∈ R1×N ×3 , the model predicts next-step velocities Vt+1
∈ R1×N ×3 from the final contextframe velocity token. As prior work showed modeling relative offsets is more natural for attention than
′
′
absolute values [51], we predict velocity and recover positions by integration, Xt+1
= Xt + ∆t Vt+1
,
where ∆t is defined during data generation. During training, we normalize positions and velocities
using the dataset-wide global mean and standard deviation computed separately for each quantity.
′
We supervise on velocity loss L = SmoothL1(Vt+1
, Vt+1 ), with β = 1.0. At inference, we start
′
′
from the ground-truth state at t = 0, predict Vt+1 , integrate to obtain Pt+1
, and iteratively roll out by
′
′
feeding (Pt+1 , Vt+1 ) back into ΦAR .
Context Window. Intuitively, a model ΦAR that can see more of its past outputs should be able to
predict the future with higher consistency. The context window determines the dependency horizon
of the autoregressive framework. In the spirit of DINO-Foresight [29], which faithfully predicts next
timestep latent DINO features autoregressively with context window 4, we implement ΦAR_ctx4 with
L = 4 where inputs are Pt−4:t ∈ R4×N ×3 and Vt−4:t ∈ R4×N ×3 and output is Vt′ . At inference, we
roll out predictions in a sliding-window fashion, starting with the first 4 ground-truth timesteps.
Train-time Noise Injection. For our autoregressive framework on mesh vertex prediction (ΦAR ),
we employ noise injection during training to bridge the train-test domain gap. In theory, by injecting
controlled noise into the context window during training, the model learns to self-correct prediction
errors accumulated during autoregressive inference. We implement ΦAR_ctx4_noised and carefully
tune the noise level to ϵpos ∼ N (0, 0.0082 ) and ϵvel ∼ N (0, 0.082 ) according to the level of error
accumulation observed during autoregressive rollout.
ΦAR Architecture and Training Details. ΦAR is a pre-norm Transformer encoder that predicts
next-timestep vertex velocities with maximum N vertices, given context length L. We tokenize
vertex positions and velocities with shape L × N × 3, masking samples with fewer vertices. Each
3D coordinate is encoded with 8 Fourier features and projected to a C = 384-dimensional token
embedding. Tokens are augmented with a learnable shared base vertex token, temporal embeddings
for relative ordering within context window, per-vertex object-ID embeddings, and a type embedding
15

Table 3: Physics Simulator Stepping Time vs P HYSI F ORMER Inference Time. We analyse the
effects of denoising step numbers on evaluation metric performance and inference time on a single
NVIDIA H100 GPU. Rigid objects (Left): over 350 test samples with 61 vertices on average chosen
using randomised stratified selection to reflect data distribution, we observe a linear upward trend for
per-sample inference time. While MSE is best at smaller denoising steps, rigidity is best preserved at
larger denoising step numbers. Accounting for the extent to which MSE reflects physical plausibility
and taken together with qualitative renderings, 25 denoising steps performs well and is comparable
to 50 denoising steps. Elastic objects (Right): over 40 test samples with 60 vertices on average
chosen using randomised stratified selection to reflect data distribution, per-sample inference time is
similar to that of rigid object scenes and exhibits linear increase. MSE and Momentum Drift Ratio
are marginally better at 5-10 denoising steps. Qualititative results on the project page show that 5
denoising steps and above appears physically plausible.
Denoising Inference
Steps
Time

MSE

Rigidity Momentum
Loss
Drift Ratio

Denoising Inference
Steps
Time

MSE

Momentum
Drift Ratio

1
5
10
25
50

1.31e-2
1.50e-2
1.58e-2
1.70e-2
1.79e-2

1.03e-3
7.23e-5
2.69e-5
1.84e-5
2.02e-5

1
5
10
25
50

1.18e-2
1.30e-2
1.36e-2
1.45e-2
1.52e-2

1.041
1.015
1.022
1.039
1.070

0.16s
1.2s
2.5s
6.4s
12.9s

1.076
1.044
1.052
1.021
1.034

0.32s
1.36s
2.7s
6.7s
13.5s

that distinguishes position from velocity tokens. We prepend R = 4 learnable register tokens
and concatenate all position and velocity tokens. ΦAR has four layers, eight attention heads, a
feed-forward dimension of 1024, and dropout 0.1. ΦAR is trained by enumerating all context windowprediction pairs for all samples (e.g. 49–4 = 45 pairs for L = 4). We use AdamW optimizer with a
constant lr = 1e−3 schedule and effective batch size 96. We train for 300k steps on one NVIDIA
H100 GPU with 90GB memory.

B

Inference Time and Denoising Step Numbers

For reported statistics and visualisations, we apply a 50-step Heun sampler during inference by
default. P HYSI F ORMER uses a DiT-L backbone, and we report number of denoising steps against
inference time on a single NVIDIA H100 GPU and evaluation metric performance in table 3. Notably,
we observe that fewer denoising steps yields outputs of comparable quality. On 250 test samples from
the P HYSI F ORMER-L-10k training dataset, ΦAR variants follow a ViT-s structure and require 0.2s on
a single H100 machine; and TIE variants have around 770k parameters and require 0.47s on a single
Quadro RTX 6000 machine (chosen for compatibility with legacy Python and CUDA dependencies
in the original codebase). Although P HYSI F ORMER is a larger model, we show that current AR
baseline’s smaller model can learn in-distribution dynamics section E.1 with GT-conditioned rollout
and attribute failure scenarios in long-horizon autoregressive rollout. Additionally, we show with
P HYSI F ORMER that the larger model can capture significantly more nuanced dynamics on more
complex meshes with increased generalisation capabilities.

C

P HYSI F ORMER-L-10k Generalization

P HYSI F ORMER-L-10k is trained solely on convex object geometries and 1–5 objects per scene.
As shown in fig. 5 and table 4, P HYSI F ORMER-L-10k generalizes well to object geometries and counts
not seen in training. As P HYSI F ORMER-L-10k is a generative framework, we additionally include
a best-of-5 generation statistic per sample. For unseen convex shapes, although objects in training
have at most 20 vertices, P HYSI F ORMER-L-10k is able to generate physically plausible motion for
objects with up to 86 vertices, showing our framework’s capability to generalize to arbitrary topology.
For object numbers exceeding 5 (maximum seen in training), we see that P HYSI F ORMER-L-10k still
models interactions between all objects faithfully.
16

Figure 5: P HYSI F ORMER-L-10k generalizes to object geometries and counts not seen during
training, shown at t = 0, 15, 30, 48. The first row shows the best AR model (T IE r=1.0 ) on two
unseen convex objects. For the following rows, we have top: two unseen convex objects, middle:
seven objects from seen convex templates, exceeding the training maximum of five, bottom: three
objects with unseen concave geometry. P HYSI F ORMER-L-10k produces physically plausible rigidbody dynamics across all settings, demonstrating robustness to topology, vertex count, and scene
complexity beyond the training distribution.
Table 4: P HYSI F ORMER-L-10k generalization performance across in-distribution and out-ofdistribution settings on size 10k dataset. Training test set: held-out sequences from the training
distribution (1–5 objects, 15 seen convex templates). Unseen convex: 100 trajectories with 1–5
objects from 10 novel convex meshes (6–86 vertices each). 6–10 obj. seen convex: 100 trajectories
with object counts exceeding the training maximum of 5, using seen convex templates. Unseen
concave: 100 trajectories with 1–5 objects from 10 novel concave meshes (12–88 vertices each).
Metrics follow table 2.
MSE↓

D

Rigidity Loss↓

Setting

1 sample

Best-of-5

1 sample

Best-of-5

Training test set
Unseen convex
6–10 obj. seen convex
Unseen concave

9.1e-3
8.6e-3
1.1e-2
7.3e-3

5.5e-3
6.3e-3
8.6e-3
6.1e-3

4.8e-5
1.3e-4
6.4e-5
3.3e-4

5.5e-6
5.7e-5
3.7e-5
3.1e-4

P HYSI F ORMER Training and Inference Meshes

(a) in fig. 6 shows mesh templates used to train P HYSI F ORMER-L-10k and P HYSI F ORMER-B-1k; (a)
and (b) shows mesh templates used to train P HYSI F ORMER. (c) shows select unseen meshes with
significant increase in complexity that still exhibit good results in P HYSI F ORMER inference.

E

Additional Quantitative Analysis

E.1

AR Models

Both ΦAR variants and TIE follow an autoregressive inference framework. In table 5, we compare
self-conditioned autoregressive inference (the true inference setting) with ground truth-conditioned
inference (the training setting). We observe that the latter yields smaller losses and errors, as the
model can learn from in-distribution data. The inference-time failure modes of AR models are
therefore attributable to compounding errors during rollout.
Furthermore, for TIE, we explored increasing the interaction radius to 2.0 and 3.5. In a 2 × 2 × 2 box,
a radius of 2.0 models interactions among most particles, while a radius of 3.5 effectively connects
every particle to every other particle via implicit edges. We omit these radii in the main paper because
17

Table 5: Comparison between self-conditioned and ground truth-conditioned inference for AR
models. We average evaluation metrics across 250 test samples from our 10k-trajectory dataset. For
each metric, the first columns show self-conditioned rollout, which follows the true inference-time
constraints of autoregressive models, where previous model outputs are used as input for future
timestep prediction. The second columns show ground truth-conditioned rollout, in which the nexttimestep prediction uses ground-truth context as input, as during training. The low error in the ground
truth-conditioned setting indicates that the model generalizes well to in-distribution test data and
suggests that error accumulation is the primary cause of long-range instability. The last two rows
additionally report results for TIE with a larger implicit edge interaction radius, illustrating model
performance when accounting for more granular interactions between individual particles.
MSE↓
Method
ΦAR_ctx1
ΦAR_ctx4
ΦAR_ctx4_noised
T IE r=0.4
T IE r=1.0
T IE r=2.0
T IE r=3.5

Rigidity Loss↓

Self Cond.

GT Cond.

Self Cond

GT Cond.

0.22
0.10
0.12
0.017
0.015
0.017
0.016

1.9e-5
1.2e-5
1.2e-5
3.4e-6
2.9e-6
2.8e-6
2.9e-6

1.4e-2
2.8e-3
1.9e-3
3.1e-3
2.8e-3
2.0e-3
2.0e-3

9.8e-7
6.3e-7
5.8e-7
3.3e-7
2.5e-7
2.0e-7
2.1e-7

they offer only a slight advantage on certain metrics, and we aimed to match the radius setting in
TIE, which connects particles only within a small neighborhood calibrated to the particles’ range of
motion. Results are shown in table 5.

F

Evaluation Metric Discussion

We follow prior work and use MSE on the ground-truth trajectory for evaluation [50, 48, 55, 34].
However, as is apparent from our video visualizations, deviation from the ground truth does not
necessarily imply physical implausibility. In our setting, rigidity loss (when conditioned on rigid
materials), momentum drift ratio, and qualitatively visualization assessment paint a more complete
picture. To assess a model’s ability to learn physics more fully, improved and more general evaluation
metrics are needed in the field.
F.1

P HYSI F ORMER Chunked Long-Horizon Inference

Current P HYSI F ORMER inference output length is determined by the training data length of 49 frames.
Since inference only requires initial position and velocity, we explore P HYSI F ORMER’s potential for
chunked long-horizon rollout. For each continuation chunk, the final generated frame of the previous
chunk is used as the next initial position. The next initial velocity is estimated by finite differencing
the final two generated frames. Full-horizon rigidity error increases with rollout length, rising from
6.99e-05 at 1x/49 frames to 6.87e-04 at 4x/193 frames, which yields visible deformations. This
observation motivates further exploration, perhaps via more noise-robust initial-state conditioning in
training with noise injection. Visualizations are available on the project page.

G

Ablations Continued

Noise Scale in Diffusion. As explained in section 4.8, the noise scale applied in diffusion training
is significant for performance. Quantitative evaluation results for varying noise scale are shown
in table 6.
Mesh Tokenization. We also ablate a triangle tokenization analogous to RenderFormer [60], where
each token is the R3×3 vertex coordinates of a mesh triangle face, flattened to 9D. In theory, triangle
tokens offer a richer local primitive than isolated vertices by implicitly encoding edge geometry,
area, and a normal direction, which may simplify learning local surface cues. However, this induces
18

Table 6: Ablation study on diffusion noise scale and tokenization strategy. All variants use
P HYSI F ORMER-B-1k (12 total DiT blocks, D = 768) trained on a 1k-sample subset, suffixes omitted
in table for brevity. Subscript ns denotes the noise scale used to sample ϵ ∼ N (0, I) × ns. P HYSI F ORMERtri,ns=0.1 replaces per-vertex tokens with per-triangle tokens (flattened 9D coordinates),
analogous to RenderFormer [60].
MSE ↓

Rigidity Loss ↓

Method

1 sample

Best-of-5

1 sample

Best-of-5

P HYSI F ORMERns=0.05
P HYSI F ORMERns=0.1
P HYSI F ORMERns=0.25
P HYSI F ORMERns=0.5
P HYSI F ORMERtri,ns=0.1

0.0073
0.0066
0.0072
0.0072
0.015

0.0058
0.0056
0.0069
0.0070
0.014

4.5e-4
1.9e-4
3.2e-4
2.8e-4
1.5e-3

1.8e-4
1.9e-4
2.7e-4
2.8e-4
1.4e-3

redundant predictions as each true vertex appears in multiple triangles. Though we scatter-add and
average into one vertex output at inference to ensure connectivity, vertex consistency implicit during
training and contributes to the poorer performance in table 6, especially for object rigidity.

H

Limitations and Future Work

P HYSI F ORMER demonstrates a unified framework for future mechanics prediction across multiple
objects and materials. Currently, due to training dataset specifications, P HYSI F ORMER supports
generation over 49 frames and performs best with up to 356 vertices. We aim to incorporate Diffusion
Forcing [14] for autoregressive diffusion to expand the length of generation. We also want to explore
spatial compression with latent encoding [16] or learning a latent variational autoencoder like [63]
for motion to support even more complex meshes without significantly increasing compute burden.
Finally, P HYSI F ORMER is fully data-driven and trained solely with the diffusion loss. This loss can be
augmented with physical inductive biases, such as continuous collision detection (CCD), to mitigate
failures like those in fig. 8, or with tailored losses to reduce rare object-orientation discontinuities in
generated sequences.

19

(a) Mesh templates in dataset D1 used to train P HYSI F ORMER-L-10k and P HYSI F ORMER-B-1k.
Each mesh contains 4-20 vertices.

(b) Mesh templates in datasets D2 –D4 , used together with the templates from D1 to train
P HYSI F ORMER. Each mesh contains 4-88 vertices.

(c) Out-of-distribution complex meshes with far more vertices per object used in P HYSI F ORMER inference.

Figure 6: Mesh templates and real-world geometries used for dataset generation and out-ofdistribution inference.

20

Figure 7: Physics simulator failure cases occur when boundary contacts are imperfectly resolved,
allowing objects to escape the bounding box, especially at high velocities or with fewer simulation
substeps.

Figure 8: Examples of object overlap during inference

21

