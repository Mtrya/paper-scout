# 2503.14701 (from arXiv HTML; MinerU fallback)



# ARC-Calib: Autonomous Markerless Camera-to-Robot Calibration via Exploratory Robot Motions

Podshara Chanrungmaneekul

Affiliation: Department of Computer Science, Rice University, Houston, TX 77005, USA.

  
Yiting Chen

Affiliation: Department of Computer Science, Rice University, Houston, TX 77005, USA.

  
Joshua T. Grace

Affiliation: Department of Mechanical Engineering and Material Science, Yale University, New Haven, CT 06511, USA. This work was supported by the US National Science Foundation grant FRR-2133110 and FRR-2132823.

  
Aaron M. Dollar

Affiliation: Department of Mechanical Engineering and Material Science, Yale University, New Haven, CT 06511, USA. This work was supported by the US National Science Foundation grant FRR-2133110 and FRR-2132823.

  
Kaiyu Hang

Affiliation: Department of Computer Science, Rice University, Houston, TX 77005, USA.

###### Abstract

Camera-to-robot (also known as eye-to-hand) calibration is a critical component of vision-based robot manipulation.
Traditional marker-based methods often require human intervention for system setup.
Furthermore, existing autonomous markerless calibration methods typically rely on pre-trained robot tracking models that impede their application on edge devices and require fine-tuning for novel robot embodiments.
To address these limitations, this paper proposes a model-based markerless camera-to-robot calibration framework, ARC-Calib, that is fully autonomous and generalizable across diverse robots and scenarios without requiring extensive data collection or learning.
First, exploratory robot motions are introduced to generate easily trackable trajectory-based visual patterns in the camera’s image frames.
Then, a geometric optimization framework is proposed to exploit the coplanarity and collinearity constraints from the observed motions to iteratively refine the estimated calibration result.
Our approach eliminates the need for extra effort in either environmental marker setup or data collection and model training, rendering it highly adaptable across a wide range of real-world autonomous systems.
Extensive experiments are conducted in both simulation and the real world to validate its robustness and generalizability.

## I Introduction

Vision-based robot manipulation tasks rely heavily on accurate camera-to-robot calibration.
The spatial transformation between the robot frame and the camera frame is critical to connecting perception and action in task executions.
Traditional calibration methods [1, 2, 3, 4] are typically formulated as solving the classic $AX=XB$ equation and requires additional assistance from fiducial markers [5, 6, 7] or chessboards [8].
While these methods are well-established and generalizable, they often demand cumbersome manual effort in the system setup processes.
To eliminate the requirement for manual effort, markerless calibration methods have emerged as a promising alternative [9, 10, 11, 12].
These approaches develop learning-based frameworks that estimate the camera-to-robot pose directly from environmental observations without marker-aided visual features, thus enabling greater flexibility for real-world autonomous robots.

Despite the progress made, these methods still face non-negligible limitations in generalization.
First, neural networks trained for specific robots are prone to overfit and thus lack the ability to generalize across different unseen embodiments.
The required data collection and policy finetuning for new robots is labor-intensive and impractical at scale.
Second, models trained on synthetic data often require real-world finetuning to overcome the sim-to-real gap.
These limitations motivate research in autonomous markerless camera-to-robot calibration that is both accurate and easily generalizable across diverse robots with unknown geometries and scenarios.

![Refer to caption](drafts/images/arccalib-2503.14701/final_main.png)
Keypoint TrajectoriesCamera FrameRobot Frame$T_{bc}$

*Fig. 1: Our camera-to-robot calibration framework estimates the pose transformation from the camera to the robot via exploratory robot motions. The method relies on analyzing the correspondence between the visual patterns extracted from keypoint trajectories and the robot motion.*

To address this, we propose ARC-Calib, a model-based autonomous calibration framework that plans exploratory robot motions to generate trackable visual features for iterative camera-to-robot calibration.
As shown in Fig. 1, instead of
relying on marker-aided or latent space visual features, our key insight lies in examining the correspondence between the structural motion patterns of the robot and the keypoint transformation in the image frame of the camera.
Specifically, our method
consists of
1) a motion planning module that actively selects exploratory motions for the robot to generate corresponding keypoint trajectories in the camera’s image frame and
2) a calibration module that exploits coplanarity and collinearity constraints from the trajectories resulting from motion to optimize the estimated result.
Since the observed keypoint trajectories are motion-oriented and do not contain any semantic information, even traditional image processing algorithms such as optical flow [13] are sufficient to extract such visual features for robust calibration.

In summary, the primary contributions of this paper are
listed as follows:

- •

ARC-Calib is a fully autonomous framework that eliminates the need for human intervention, manual setup, or environmental markers, enabling efficient camera-to-robot calibration.
- •

The calibration framework is generalizable across diverse robots and scenarios without prior database knowledge, pre-trained models, or fine-tuning, promoting broad applicability and adaptability.
- •

ARC-Calib replaces traditional visual markers with exploratory robot motions, generating trackable trajectory-based visual patterns that can be extracted using lightweight methods like optical flow.

## II Related work

Marker-Based Calibration:
Robot-to-camera calibration is a fundamental problem in robotic manipulation.
Traditional approaches include the use of fiducial markers, where early work formulated the hand-eye calibration problem as solving the matrix equation $AX=XB$ on the Euclidean group [1, 2, 3, 4].
Chessboard patterns have also been widely adopted given their simplicity, enabling techniques such as simultaneous robot-world-hand-eye calibration [14].
Additionally, specialized calibration objects, such as spheres or customized geometries, have been used to facilitate the calibration [15, 16] along with methods utilizing RBG-D cameras [17].
While these methods are accurate, they require manual interventions, tedious setup of markers or objects, and controlled setups, thus limiting their applications in autonomous robot systems.

Markerless Calibration:
In contrast, recent markerless calibration approaches leverage advances in deep learning to eliminate the need for additional markers.
These approaches can be divided into two main types: keypoint-based and rendering-based methods.
Keypoint-based approaches [9, 18, 10, 19] utilize deep neural networks to detect keypoints on the robot and then adapt the perspective-n-point (PnP) algorithm to estimate the calibration results.
However, inconsistent keypoint detection constrains calibration accuracy.
Rendering-based methods [12, 20] often rely on domain-randomized synthetic training data to enable pose estimation in the real world.
However, the learned neural network frameworks are trained for specific robot embodiments or scenarios, which require considerable fine-tuning for additional robots.
To address these limitations, our framework utilizes self-exploratory robot motions to create trackable visual patterns, eliminating the need for physical markers or learning-based robot tracking models.

## III Problem formulation

In this work, we address the camera-to-robot calibration problem for $\mathcal{N}$-DoF serial robot manipulators.
We denote the robot’s base frame as $T_{b}\in SE(3)$ and the camera frame as $T_{c}\in SE(3)$.
The calibration goal is to estimate the transformation from the robot’s base frame to the camera frame, represented as:

|  | $$ T_{bc}=\begin{bmatrix}R_{bc}&t_{bc}\\ \boldsymbol{0}&1\end{bmatrix} $$ |  | (1) |
|---|---|---|---|

where $R_{bc}\in SO(3)$ is the rotation matrix and $t_{bc}\in\mathbb{R}^{3}$ is the translation vector.
To achieve this, the correspondence between the robot’s exploratory actions and trackable visual features is computed.
We find the 3D position of the visual features in the camera frame as follows:

|  | $\displaystyle s_{1}\begin{bmatrix}p_{x}&p_{y}&1\end{bmatrix}^{T}$ | $\displaystyle=\mathcal{K}\begin{bmatrix}u&v&1\end{bmatrix}^{T}$ |  | (2) |
|---|---|---|---|---|

where $\mathcal{K}$ is the ${3\times 3}$ intrinsic matrix of the camera’s pinhole projection model, $(p_{x},p_{y})\in\mathbb{N}^{2}$ is a pixel coordinate in the image, $s_{1}\in\mathbb{R}$ is a scaling factor, and $(u,v)\in\mathbb{R}^{2}$ are Cartesian coordinates obtained by normalizing the pixel coordinates with respect to the intrinsic parameters.
This normalization ensures that the proposed algorithm remains independent of the camera’s intrinsic matrix.
Here, we refer to $(u,v)$ as coordinates on the image plane $\pi$, which can be mapped to 3D points $(s_{2}u,s_{2}v,s_{2})$ in the homogeneous coordinates of the camera frame for any distance $s_{2}\in\mathbb{R}$ in the z-axis.

To find $T_{bc}$, we propose a markerless calibration framework, which is outlined in Algorithm 1.
The robot selects an exploratory motion, as detailed in Section IV-A, that enables the optical flow algorithm to track keypoint trajectories in the image plane coordinates.
Using the obtained keypoint trajectories, we estimate the coplanarity and collinearity constraints from the observed trajectory, as explained in Sections IV-C and IV-D.
These constraints, corresponding to the robot motion, are then used to estimate the transformation from the robot frame to the camera frame, as described in Section V.
By repeating this procedure, the robot iteratively refines the estimated transformation $T_{bc}$ until it converges, as detailed in Section V-D.

*Algorithm 1  Markerless Robot-Camera Calibration*

1:
None

2:
Estimated transformation $T_{bc}\in SE(3)$.

3:
$i\leftarrow 0$

4:
while not converged do $\triangleright$ Sec. V-D

5:
  $i\leftarrow i+1$

6:
  $\mathcal{U}_{i}\leftarrow\textsc{RobotMotionSelection}()$ $\triangleright$ Sec. IV-A

7:
  $[\mathcal{P}^{\pi}_{i,1},\dots,\mathcal{P}^{\pi}_{i,L_{i}}]\leftarrow\textsc{Execute}(\mathcal{U}_{i})$ $\triangleright$ Eq. (3)

8:
  $\mathcal{\vec{A}}^{c}_{i},\!\mathcal{\vec{D}}^{c}_{i}\!\leftarrow\!\textsc{Estimation}([\mathcal{P}^{\pi}_{i,1},\dots,\mathcal{P}^{\pi}_{i,L_{i}}])$ $\triangleright$ Sec. IV-C, IV-D

9:
  $T_{bc}\leftarrow\textsc{Calibration}(\mathcal{\vec{A}}^{c}_{1},\mathcal{\vec{D}}^{c}_{1},\dots,\mathcal{\vec{A}}^{c}_{i},\mathcal{\vec{D}}^{c}_{i})$ $\triangleright$ Sec. V

10:
end while

11:
return $T_{bc}$

## IV Visual Patterns of Exploratory Robot Motions

In this section, we define the characteristics of exploratory motions.
As the robot moves, its features are tracked, producing a distinct visual pattern on the image plane.
This pattern serves as the basis for estimating key motion parameters, specifically the rotational axis and the reference position of the center of rotation in the camera frame.
With these estimates, we can impose coplanarity and collinearity constraints, which are essential for estimating the transformation $T_{bc}$ that will be described later in Section V.

### IV-A Exploratory Robot Motions

![Refer to caption](drafts/images/arccalib-2503.14701/final_motion.png)
Joint Position $\mathcal{D}_{i}^{b}$Rotational Axis $\mathcal{\vec{A}}_{i}^{b}$

*Fig. 2: Green overlay indicates parts of the robot that are considered as the rotation of a single rigid body. The motion could be parameterized with the rotation axis $\mathcal{\vec{A}}^{b}_{i}$ and the joint position $\mathcal{D}^{b}_{i}$*

Exploratory motion consists of a single joint movement, which can be regarded as the rotation of a single rigid body, as shown in Fig. 2.
For $N$ robot motions, each exploratory motion $\mathcal{U}_{i}$ where $i=1,2,\dots,N$ is defined as a tuple $\mathcal{U}_{i}=(\mathcal{Q}_{i},\mathcal{J}_{i},\Delta_{i})$.
Here, $\mathcal{Q}_{i}\in\mathbb{R}^{\mathcal{N}}$ represents the starting robot joint configuration, $\mathcal{J}_{i}\in\{1,2,\dots,\mathcal{N}\}$ denotes the selected joint for the rotation, and $\Delta_{i}\in\mathbb{R}$ is the change in the configuration of joint $\mathcal{J}_{i}$ during the motion.
Given the environment’s collision model, the motion $U_{i}$ is selected by randomly generating $Q_{i}$ and $\mathcal{J}_{i}$, while ensuring that the $\Delta_{i}$ of the resulting motion satisfies a minimum required value while avoiding collisions.

Each robot motion, modeled as the rotation of a single rigid body, can be parameterized by its rotational axis $\mathcal{\vec{A}}^{b}_{i}\in\hat{\mathbb{R}}^{3}$ and joint position $\mathcal{D}^{b}_{i}\in\mathbb{R}^{3}$ in the robot frame.
The exploratory motions are observed in the camera frame with rotational axis $\mathcal{\vec{A}}^{c}_{i}\in\hat{\mathbb{R}}^{3}$ and reference position $\mathcal{\vec{D}}^{c}_{i}\in\hat{\mathbb{R}}^{3}$ estimated from the keypoint trajectory’s visual pattern in the camera image frame.

### IV-B Visual Patterns

During each motion $\mathcal{U}_{i}$, we track $L_{i}$ keypoints, which are identified by visual features such as strong corners in the image [21].
The value of $L_{i}$ is dynamically determined by the tracking algorithm and may vary across different motions.
These tracked keypoints generate trajectories that capture the robot’s movement.
For each keypoint $j=1,2,\dots,L_{i}$, the features are tracked across the $M_{i,j}$ consecutive frames.
This allows us to define the observation of 2D keypoint trajectories
$\mathcal{P}^{\pi}_{i,j}$ as:

|  | $$ \mathcal{P}^{\pi}_{i,j}=\{((u_{i,j,k},v_{i,j,k}),\delta_{i,j,k})\mid k=1,2,\dots,M_{i,j}\} $$ |  | (3) |
|---|---|---|---|

where $(u_{i,j,k},v_{i,j,k})$ are Cartesian coordinates on image plane $\pi$, and $\delta_{i,j,k}\in[0,\Delta_{i}]$ represents the change in configuration of joint $\mathcal{J}_{i}$ at the $k$-th frame. These trajectories provide a detailed representation of the robot’s motion as observed by the camera, as shown in Figure 3.

The 3D points corresponding to the points on tracked trajectories $\mathcal{P}^{\pi}_{i,j}$ move along circular paths in 3D space.
The normal vectors of the planes containing these 3D circles align with the rotational axis $\mathcal{\vec{A}}^{c}_{i}$.
When these 3D circular trajectories are projected onto the image plane through perspective projection, the resulting keypoint trajectories $\mathcal{P}^{\pi}_{i,j}$ typically manifest as ellipses or straight lines.
However, distinguishing between lines caused by the robot’s motion and those arising from tracking noise can be challenging.
To address this, we focus exclusively on keypoint trajectories that exhibit elliptical shapes, as they provide more reliable information for analysis.

![Refer to caption](drafts/images/arccalib-2503.14701/final_traj.png)

*Fig. 3: (left) Tracked visual features moving along 3D circles. (right) Samples of keypoint trajectories (red) and corresponding visual patterns (green) from conic functions fitting.*

These ellipses can be described as a special case of a general conic function:

|  | $\displaystyle\Xi_{i,j}^{\pi}(u,v)=$ | $\displaystyle A_{i,j}u^{2}+B_{i,j}uv+C_{i,j}v^{2}$ |  | (4) |
|---|---|---|---|---|
|  |  | $\displaystyle+D_{i,j}u+E_{i,j}v+F_{i,j}=0$ |  |

subject to the ellipse constraint:

|  | $$ 4A_{i,j}C_{i,j}-B_{i,j}^{2}=0 $$ |  | (5) |
|---|---|---|---|

Here, $A_{i,j}$, $B_{i,j}$, $C_{i,j}$, $D_{i,j}$, $E_{i,j}$ and $F_{i,j}$ are coefficients of the ellipse on the image plane $\pi$, and $(u,v)$ represents the Cartesian coordinates of a point lying on it.
The function $\Xi^{\pi}_{i,j}$ defines the algebraic distance from the point $(u,v)$ to the conic section.

Furthermore, the 3D circles corresponding to the motion $\mathcal{U}_{i}$ share a common origin on the rotational axis.
Consequently, the semi-major axes of their projected ellipses are aligned in the same direction.
For all $L_{i}$ ellipses associated with the motion $\mathcal{U}_{i}$, let $\phi_{i}$ denote their orientation. Using the Cartesian representation of the ellipse, this orientation is given by $\tan 2\phi_{i}=\frac{B_{i,j}}{C_{i,j}-A_{i,j}}$.
By introducing an additional parameter $G_{i,j}=C_{i,j}-A_{i,j}$, we can impose the constraints $B_{i,j}=B_{i}$ and $G_{i,j}=G_{i}$ for all $j=1,2,\dots,L_{i}$.
With these constraints, Equation (4) can be rewritten as:

|  | $\displaystyle\Xi^{\pi}_{i,j}(u,v)=$ | $\displaystyle A_{i,j}(u^{2}+v^{2})+B_{i}uv+G_{i}v^{2}$ |  | (6) |
|---|---|---|---|---|
|  |  | $\displaystyle+D_{i,j}u+E_{i,j}v+F_{i,j}=0$ |  |

To fit a set of points, i.e., the tracked keypoints, to conic functions, we adopt an approach similar to the method in [22], minimizing the sum of squared algebraic distances from the points to their respective conics:

|  | $\displaystyle\arg\min_{g}\sum_{j=1}^{L_{i}}\sum_{k=1}^{M_{i,j}}\Xi^{\pi}_{i,j}(u_{i,j,k},v_{i,j,k})^{2}$ |  | (7) |
|---|---|---|---|

which could be rearranged as

|  | $\displaystyle\arg\min_{g}\|Wg\|^{2}$ |  | (8) |
|---|---|---|---|---|---|

where $g$ is a vector containing all $4L_{i}+2$ coefficients and parameters of the ellipses and $W$ is a design matrix of the size $\sum_{j=1}^{L_{i}}M_{i,j}\times 4L_{i}+2$, representing the least squares minimization Equation (7).

Once the solution for $g$ is obtained, we can reconstruct the conic functions $\Xi^{\pi}_{i,j}$ for all keypoint trajectories $j=1,2,\dots,L_{i}$, ensuring that their semi-major axes share the same orientation.
An example of this reconstruction is illustrated in Figure 3.

Note that the minimization problem in Equation (8) does not explicitly enforce the ellipse constraint in Equation (5).
As a result, solutions that do not satisfy this constraint must be discarded to ensure that only valid ellipses are retained.

### IV-C Estimating Rotational Axis

2D Ellipse $\Xi^{\pi}_{i,j}$Camera Frame $c$Image Plane $\pi$3D Cone $\Xi^{c}_{i,j}$3D Circles

Candidate rotational axes for $\mathcal{\vec{A}}^{c}_{i}$

*Fig. 4: A 3D cone (green) indicates the possible 3D circle candidates (blue and yellow). When projected onto the image plane, these candidates generate the visual pattern of a 2D ellipse (red)*

For the rotational axis $\mathcal{\vec{A}}^{b}_{i}$ of the motion $\mathcal{U}_{i}$, shown in Figure 2, the corresponding rotational axis in the camera frame $\mathcal{\vec{A}}^{c}_{i}$ can be estimated using the 3D orientation (i.e., the surface normal) of the 3D circle.
A closed-form solution for this problem has been introduced in[23].

First, we define a 3D cone surface whose base is the perspective projection of the 3D circle (represented as the ellipse $\Xi^{\pi}_{i,j}$) and whose vertex is the center of the camera, as illustrated in Fig. 4.
Determining the 3D orientation of the circle involves finding a plane that intersects the cone and generates a circular curve.

The equation of the 3D cone, with the ellipse $\Xi^{\pi}_{i,j}$ as its base, can be constructed using the ellipse’s coefficients as:

|  | $\displaystyle\Xi^{c}_{i,j}(x,y,z)=$ | $\displaystyle A_{i,j}x^{2}+B_{i,j}xy+C_{i,j}y^{2}$ |  | (9) |
|---|---|---|---|---|
|  |  | $\displaystyle+D_{i,j}xz+E_{i,j}yz+F_{i,j}z^{2}=0$ |  |

where $(x,y,z)$ are Homogeneous coordinates in the camera frame $c$ of the points lying on the cone.
To simplify the representation, we introduce a symmetric matrix $Q_{i,j}$

|  | $\displaystyle Q_{i,j}$ | $\displaystyle=\begin{bmatrix}A_{i,j}&\frac{B_{i,j}}{2}&\frac{D_{i,j}}{2}\\ \frac{B_{i,j}}{2}&C_{i,j}&\frac{E_{i,j}}{2}\\ \frac{D_{i,j}}{2}&\frac{E_{i,j}}{2}&F_{i,j}\end{bmatrix}$ |  |
|---|---|---|---|
|  | $\displaystyle\Xi^{c}_{i,j}(x,y,z)$ | $\displaystyle=\begin{bmatrix}x&y&z\end{bmatrix}Q_{i,j}\begin{bmatrix}x&y&z\end{bmatrix}^{T}=0$ |  | (10) |

Let ${}^{1}\lambda_{i,j},{}^{2}\lambda_{i,j},{}^{3}\lambda_{i,j}$ denote the eigenvalues of $Q_{i,j}$, with corrsponding eigenvectors ${}^{1}e_{i,j},{}^{2}e_{i,j},{}^{3}e_{i,j}$.
The equation of the 3D cone can be further simplified as:

|  | $$ {}^{1}\lambda_{i,j}X^{2}+{}^{2}\lambda_{i,j}Y^{2}+{}^{3}\lambda_{i,j}Z^{2}=0 $$ |  | (11) |
|---|---|---|---|

where the $(X,Y,Z)$ are Homogeneous coordinates in the canonical frame of conicoids, where the principal axis of the central cone aligns with the Z axis.
Without loss of generality, we can order the eigenvalues such that ${}^{3}\lambda_{i,j}\!<\!0\!<\!{}^{1}\lambda_{i,j}\!<\!{}^{2}\lambda_{i,j}$.
The four possible solutions for the 3D circle plane normal (i.e., the rotational axis) are then given by:

|  | $\displaystyle a^{c}\!$ | $\displaystyle=\!\pm\!\sqrt{\frac{{}^{1}\lambda_{i,j}-{}^{3}\lambda_{i,j}}{{}^{2}\lambda_{i,j}-{}^{3}\lambda_{i,j}}}\!\cdot\!{}^{3}e_{i,j}\!\pm\!\sqrt{\frac{{}^{2}\lambda_{i,j}-{}^{1}\lambda_{i,j}}{{}^{2}\lambda_{i,j}-{}^{3}\lambda_{i,j}}}\!\cdot\!{}^{2}e_{i,j}$ |  | (12) |
|---|---|---|---|---|
|  | $\displaystyle\vec{a}^{c}\!$ | $\displaystyle=\!\frac{a^{c}}{\|a^{c}\|}$ |  |

These solutions represent two pairs of normals that correspond to the same 3D circle but rotate in opposite directions, as exemplified by the blue and yellow circles in Figure 4.
To resolve this ambiguity, we eliminate solutions that rotate in the opposite direction of the observed keypoint trajectory $\mathcal{P}^{\pi}_{i,j}$, resulting in two candidate rotational axes per keypoint trajectory.
This allows us to generate a set of candidate rotational axes for $\mathcal{\vec{A}}^{c}_{i}$ as $\mathcal{\vec{A}}^{\prime}_{i}=\{\vec{a}^{c}_{i,l}\mid l=1,2,\dots,2L_{i}\}$.

### IV-D Estimating the Reference Position for the Center of Rotation

Joint Position $D_{i}^{b}$3D CirclesRotational Axis $\mathcal{\vec{A}}^{b}_{i}$

Candidate Reference Position $\vec{d}^{c}_{i,l}$

Candidate Rotational Axis $\vec{a}^{c}_{i,l}$

Projection Plane $\sigma_{i,l}$

Image Plane $\pi$
2D Circles $\mathcal{C}^{\sigma_{i,l}}_{i,j}$Centers of 2D CirclesKeypoint Trajectories $\mathcal{P}^{\pi}_{i,j}$Camera Frame $c$

*Fig. 5: Perspective projection of the keypoint trajectories $\mathcal{P}^{\pi}_{i,j}$ on projection plane $\sigma_{i,l}$ create the projected trajectories $\mathcal{P}^{\sigma_{i,l}}_{i,j}$. The projected trajectories create a visual pattern of 2D circles that could be used to determine the reference positions of the exploratory motion.*

For the motion $\mathcal{U}_{i}$ with the joint position $D_{i}^{b}$ shown in Figure 2, we can use the rotational axis candidates $\mathcal{\vec{A}}^{\prime}_{i}$, calculated in Section IV-C, to estimate the corresponding reference positions $\mathcal{\vec{D}}^{\prime}_{i}=\{\vec{d}^{c}_{i,l}\mid l=1,2,\dots,2L_{i}\}$ in the camera frame $c$. Then, we could select the best candidate from $\mathcal{\vec{A}}^{\prime}_{i}$ and $\mathcal{\vec{D}}^{\prime}_{i}$as estimates $\mathcal{\vec{A}}^{c}_{i}$ and $\mathcal{\vec{D}}^{c}_{i}$ respectively.
When the trajectories $\mathcal{P}^{\pi}_{i,j}$ are perspective-projected onto a projection plane $\sigma_{i,l}$ with an arbitrary distance from the camera and a normal vector $\vec{a}^{c}_{i,l}$, the projected trajectories form 2D circles.
Simultaneously, the 3D line representing the rotational axis projects onto the plane as a 2D line $\vec{d}^{c}_{i,l}$ that passes through the centers of these 2D circles, as shown in Figure. 5.
The perspective projection of $\mathcal{P}^{\pi}_{i,j}$ could be represented as:

|  | $$ \mathcal{P}^{\sigma_{i,l}}_{i,j}\!=\!\{((u^{\prime}_{i,j,k,l},v^{\prime}_{i,j,k,l}),\delta_{i,j,k})\mid k\!=\!1,2,\dots,M_{i,j}\} $$ |  | (13) |
|---|---|---|---|

where $(u^{\prime}_{i,j,k,l},v^{\prime}_{i,j,k,l})$ are Cartesian coordinate on the projection plane $\sigma_{i,l}$.
The 2D circle $\mathcal{C}^{\sigma_{i,l}}_{i,j}$ on the projection plane $\sigma_{i,l}$ corresponding to the robot configuration $\delta_{i,j,k}$ can be described by the polar equation:

|  | $\displaystyle\mathcal{C}^{\sigma_{i,l}}_{i,j}=$ | $\displaystyle\{(u^{\prime\prime}_{i,j,k,l},v^{\prime\prime}_{i,j,k,l})\mid k=1,2,\dots,M_{i,j}\}$ |  |
|---|---|---|---|
|  | $\displaystyle u^{\prime\prime}_{i,j,k,l}=$ | $\displaystyle c^{x}_{i,j,l}+\alpha_{i,j,l}\sin(\delta_{i,j,k}+\beta_{i,j,l})$ |  | (14) |
|  | $\displaystyle v^{\prime\prime}_{i,j,k,l}=$ | $\displaystyle c^{y}_{i,j,l}+\alpha_{i,j,l}\cos(\delta_{i,j,k}+\beta_{i,j,l}))$ |  |

where $(c^{x}_{i,j,l},c^{y}_{i,j,l})$ are the 2D coordinates of the circle’s center, $\alpha_{i,j,l}$ is the radius of the circle and $\beta_{i,j,l}$ is the starting angle of the curve.

The fitting of these 2D circles can be formulated as a minimization problem, where the goal is to minimize the sum of squared distances between the projected points $\mathcal{P}^{\sigma_{i,l}}_{i,j}$ and their corresponding circles:

|  | $\displaystyle J_{i,j,l}$ | $\displaystyle=\sum_{k=1}^{M_{i,j}}\|(u^{\prime}_{i,j,k,l},v^{\prime}_{i,j,k,l})-(u^{\prime\prime}_{i,j,k,l},v^{\prime\prime}_{i,j,k,l})\|^{2}$ |  |
|---|---|---|---|---|---|
|  | $\displaystyle\arg$ | $\displaystyle\min_{c^{x}_{i,j,l},c^{y}_{i,j,l},\alpha_{i,j,l},\beta_{i,j,l}}J_{i,j,l}$ |  | (15) |

where $J_{i,j,l}$ is the cost function. To find the globally optimal solution to Equation (IV-D), we consider its first-order optimality conditions:

|  | $\displaystyle\frac{\partial J_{i,j,l}}{\partial c^{x}_{i,j,l}}=0$ |  | $\displaystyle\frac{\partial J_{i,j,l}}{\partial c^{y}_{i,j,l}}=0$ |  | $\displaystyle\frac{\partial J_{i,j,l}}{\partial\alpha_{i,j,l}}=0$ |  | $\displaystyle\frac{\partial J_{i,j,l}}{\partial\beta_{i,j,l}}=0$ |  | (16) |
|---|---|---|---|---|---|---|---|---|---|

This system of four equations can yield at most two critical points for $0\!\leq\!\beta_{i,j,l}\!<2\pi$. The global minimum is determined by substituting these solutions back into $J_{i,j,l}$ and selecting the one with the lower sum of errors.

The reference position of the rotation $\vec{d}^{c}_{i,l}$ is determined by first fitting a 2D line to the set of 2D center coordinates $(c^{x}_{i,j,l},c^{y}_{i,j,l})$ for all $j=1,2,\dots,L_{i}$. The 2D line is obtained using RANSAC linear fitting to eliminate outliers trajectories.
Once the 2D line is determined, it is converted into a 3D vector $\vec{d}^{c}_{i,l}$ in the camera frame.

After determining the reference positions for all candidate rotational axes, the best candidate is selected using a heuristic function.
This function evaluates the quality of each candidate by summing the error functions from the circle-fitting process for all keypoint trajectories projected onto the plane associated with that candidate.
To account for variations in error values caused by changes in circle size due to the projection, we normalize the error by its radius, effectively eliminating this effect.

|  | $\displaystyle J^{\prime}_{i,l}=\sum_{j=1}^{L_{i}}\frac{J_{i,j,l}}{\alpha_{i,j,l}}$ |  | $\displaystyle l^{*}=\arg\min_{l=1,\dots,2L_{i}}J^{\prime}_{i,l}$ |  | (17) |
|---|---|---|---|---|---|

where $J^{\prime}_{i,l}$ is the error function of the candidate rotational axis $\vec{a}^{c}_{i,l}$.
The candidate with the smallest total error is selected as the best rotational axis $\mathcal{\vec{A}}^{c}_{i}=\vec{a}^{c}_{i,l^{*}}$ with its corresponding reference position $\mathcal{\vec{D}}^{c}_{i}=\vec{d}^{c}_{i,l^{*}}$.
As shown in Figure 2, the estimates $\mathcal{\vec{A}}^{c}_{i}$ and $\mathcal{\vec{D}}^{c}_{i}$ correspond to the rotational axis $\mathcal{\vec{A}}^{b}_{i}$ and joint position $\mathcal{D}^{b}_{i}$, respectively, as observed from the camera frame.

## V Robot-Camera Calibration

In this section, we solve for the transformation from the camera frame to the robot frame $T_{bc}$ by imposing geometric constraints from the observation to the current robot state.
By formulating an optimization problem that enforces collinear and coplanarity constraints introduced in [24], we ensure that the computed transformation accurately aligns the robot’s movements with the camera’s observations.

### V-A Geometric Correspondences

![Refer to caption](drafts/images/arccalib-2503.14701/final_calib.png)
Camera Frame $c$Projection PlaneReference Position $\mathcal{\vec{D}}^{c}_{i}$Normal vector $\rho_{i}$Rotational Axis $\mathcal{\vec{A}}_{i}^{b}$Joint Position $\mathcal{D}_{i}^{b}$Robot Frame $b$Rotational Axis $\mathcal{\vec{A}}_{i}^{c}$

*Fig. 6: Collinearity and Coplanarity constraints corresponding to the robot’s exploratory motion.*

To ensure this alignment, we formulate geometric constraints based on the collinearity of the exploratory motion’s rotational axis and coplanarity of the exploratory motion’s joint position, as shown in Figure 6.

For each motion $\mathcal{U}_{i}$, we compute the robot’s current rotational axis $\mathcal{\vec{A}}_{i}^{b}$ and the position $\mathcal{D}_{i}^{b}$ of the rotating joint $\mathcal{J}_{i}$ using a forward kinematics function $\Gamma:(\mathbb{R}^{\mathcal{N}},\mathbb{R})\to(\hat{\mathbb{R}}^{3},\mathbb{R}^{3})$:

|  | $$ \mathcal{\vec{A}}_{i}^{b},\mathcal{D}_{i}^{b}=\Gamma(\mathcal{Q}_{i},\mathcal{J}_{i}) $$ |  | (18) |
|---|---|---|---|

These computed values, $\mathcal{\vec{A}}_{i}^{b}$ and $\mathcal{D}_{i}^{b}$
must align with the observed rotational axis $\mathcal{\vec{A}}^{c}_{i}$ and reference position $\mathcal{\vec{D}}^{c}_{i}$ obtained from the camera frame in Section IV.

Collinearity of Rotational Axes:
The rotational axis $\mathcal{\vec{A}}^{b}_{i}$ in the robot frame must be colinear with the observed rotational axis $\mathcal{\vec{A}}^{c}_{i}$ in the camera frame after applying the rotational transformation $R_{bc}$. This constraint can be expressed using the cross-product:

|  | $$ \mathcal{\vec{A}}^{c}_{i}\times(R_{bc}\mathcal{\vec{A}}^{b}_{i})=0 $$ |  | (19) |
|---|---|---|---|

where $\times$ denotes the cross product.
By representing the cross product using the skew-symmetric matrix $\lfloor\mathcal{\vec{A}}^{c}_{i}\rfloor_{\times}$ the equation can be rewritten as $\lfloor\mathcal{\vec{A}}^{c}_{i}\rfloor_{\times}R_{bc}\mathcal{\vec{A}}^{b}_{i}=0$.
To incorporate all $N$ motions, we vectorize the matrix multiplication and rearrange terms, resulting in a linear system:

|  | $$ H_{r}r_{bc}=0 $$ |  | (20) |
|---|---|---|---|

where $H_{r}$ is a $3N\times 9$ matrix constructed from the rotational axis constraints and $r_{bc}=vec(R_{bc})$ is a $9\times 1$ vector obtained by vectorizing $R_{bc}$.

Coplanarity of Reference Points:
The reference position $\mathcal{\vec{D}}^{c}_{i}$, when not colinear with $\mathcal{\vec{A}}^{c}_{i}$, forms a unique plane that intersects with the origin point in the camera frame. The normal of this plane is given by $\rho_{i}=\mathcal{\vec{D}}^{c}_{i}\times\mathcal{\vec{A}}^{c}_{i}$ where the corresponding joint position $\mathcal{D}^{b}_{i}$ must lie on this plane.
Then, We could impose a constraint using an algebraic distance from the point to the plane:

|  | $$ \rho_{i}\cdot(R_{bc}\mathcal{D}_{i}^{b}+t_{bc})=0 $$ |  | (21) |
|---|---|---|---|

Similar to the collinearity constraint, we vectorize this equation for all $N$ motions, resulting in:

|  | $$ H_{p}r_{bc}+K_{p}t_{bc}=0 $$ |  | (22) |
|---|---|---|---|

where $H_{p}$ is a $3N\times 9$ matrix, and $K_{p}$ is a $3N\times 3$ matrix, constructed from the plane normal vectors and joint position.

These constraints form the basis of a convex optimization problem, which solves the transformation $T_{bc}$ by minimizing the discrepancies between the robot’s kinematic parameters and the observed motion parameters in the camera frame.

### V-B Convex Formulation

Combining the two conditions (collinearity and coplanarity) from Section V-A, we construct the matrices $H$ and $K$ as follows:

|  | $\displaystyle H=\begin{bmatrix}H_{r}&H_{p}\end{bmatrix}^{T}$ | , | $\displaystyle K=\begin{bmatrix}\boldsymbol{0}_{3N\times 3}&K_{p}\end{bmatrix}^{T}$ |  | (23) |
|---|---|---|---|---|---|

This allows us to express the combined system as

|  | $$ Hr_{bc}+Kt_{bc}=0 $$ |  | (24) |
|---|---|---|---|

Given the optimal solution $r^{*}_{bc}$ for the rotation parameters, it is known that the optimal unconstrained solution for the translation vector $t^{*}_{bc}$ can be computed as

|  | $$ t^{*}_{bc}=-(K^{T}K)^{-1}K^{T}Hr^{*}_{bc} $$ |  | (25) |
|---|---|---|---|

Substituting this expression back into the system, we obtain a complete system $Sr_{bc}=0$ with:

|  | $\displaystyle S$ | $\displaystyle=(I_{6N}-(K^{T}K)^{-1}K^{T})H$ |  | (26) |
|---|---|---|---|---|

where $S$ is a $6N\times 9$ matrix and $I_{6N}$ is the identity matrix of size $6N$. This formulation allows us to solve for $r_{bc}$ independently of $t_{bc}$.

We can then frame the problem as a minimization of the squared norm of the residual:

|  | $$ r^{*}_{bc}=\arg\min_{r_{bc}}\|Sr_{bc}\|^{2} $$ |  | (27) |
|---|---|---|---|---|---|

This optimization problem can be solved as a minimization problem similar to the ellipse fitting in Equation 8.
However, since the constraints on the rotational matrix parameters (e.g., orthogonality and unit determinant) are non-convex, they are temporarily relaxed during the optimization.
This simplification allows us to solve the problem efficiently.
Once the optimal solution $r^{*}_{bc}$ is obtained, we recover the rotational matrix $R_{bc}^{*}$ by reshaping the vectorized form. Then, we enforce the constraints of a valid rotation matrix (orthogonality and unit determinant) by performing a singular value decomposition (SVD) on $R^{*}_{bc}$:

|  | $\displaystyle R^{*}_{bc}\!=vec^{-1}(r^{*}_{bc})$ |  | $\displaystyle U\Sigma V^{T}\!=\!svd(R^{*}_{bc})$ |  | $\displaystyle R^{\prime}_{bc}\!=\!UV^{T}$ |  | (28) |
|---|---|---|---|---|---|---|---|

This step ensures that $R^{\prime}_{bc}$ is a proper rotation matrix, satisfying all necessary geometric constraints.

Finally, we recover the optimal translation vector $t_{bc}^{*}$ by substituting the rotation matrix $R^{\prime}_{bc}$ back into Equation 25. With both the rotation matrix $R^{\prime}_{bc}$ and the translation vector $t_{bc}^{*}$ determined, we obtain the complete transformation from the robot frame to the camera frame $T_{bc}$.

### V-C Observation Pruning and Selection

During the calibration process, errors in tracking keypoints can arise due to various factors such as occlusions, lighting conditions, or other external disturbances.
These errors can lead to observations that are unsuitable for calibration, as they may introduce significant inaccuracies.
To address this, we implement a filtering mechanism to prune out unreliable observations and retain only those that meet certain quality criteria.
This process is divided into two scenarios: when no initial transformation estimate is available and when an initial transformation estimate exists.

#### V-C1 Filtering Observations Without an Initial Transformation Estimate

At the start of the calibration process, with no estimated transformation $T_{bc}$, we filter observations based on the consistency of keypoint trajectories, as described in Section IV.
Keypoint trajectories should resemble ellipses, which are projections of 3D circular motions.
Each ellipse corresponds to two possible 3D circles, and if the trajectories share a rotational axis, one solution should align with the candidate $\mathcal{\vec{A}}^{c}_{i}$.
This ensures most trajectories agree with the chosen rotational axis.
In ambiguous cases, where two candidates are very similar, we filter out observations where the difference in agreement between the top two candidates is below a certain threshold, ensuring reliability.
These strict conditions guarantee an accurate initial estimation, even without prior transformation.

#### V-C2 Filtering Observations With an Initial Transformation Estimate

Once an initial transformation estimate $T^{i}_{bc}$ with the associated rotational matrix $R^{i}_{bc}$ and translational vector $t_{bc}^{i}$
is obtained from the optimization using observations from actions $\mathcal{U}_{1},\dots,\mathcal{U}_{i}$, we can refine the filtering process. Note that only three observations satisfying the previous conditions are needed to compute this initial estimate.
With $T^{i-1}_{bc}$ available, we select observations based on their reprojection errors for the rotational axis and the reference position.

The reprojection errors are calculated as follows. For the rotational axis, the error is computed as:

|  | $$ \epsilon^{a}_{i}=\cos^{-1}\frac{\mathcal{\vec{A}}_{i}^{c}\cdot(R_{bc}^{i-1}\mathcal{\vec{A}}_{i}^{b})}{\|\mathcal{\vec{A}}_{i}^{c}\|\|(R_{bc}^{i-1}\mathcal{\vec{A}}_{i}^{b})\|} $$ |  | (29) |
|---|---|---|---|---|---|---|---|

$\mathcal{\vec{A}}_{i}^{c}$ is the observed rotational axis in the camera frame, $R_{bc}^{i-1}\mathcal{\vec{A}}_{i}^{b}$ and is the reprojected rotational axis in the camera frame using the current transformation estimate.
For the reference position, the error is computed as:

|  | $$ \epsilon^{d}_{i}=|\rho_{i}\cdot(R_{bc}^{i-1}\mathcal{D}_{i}^{b}+t^{i-1}_{bc})| $$ |  | (30) |
|---|---|---|---|---|---|

similar to Equation 21 but with the current transformation estimate.
Observations are retained only if both $\epsilon^{a}_{i}$ and $\epsilon^{d}_{i}$ are within predefined thresholds. This ensures that the selected observations are consistent with the current transformation estimate and contribute to a more accurate calibration.

### V-D Convergence Criteria

To ensure the convergence of the estimated transformation $T_{bc}$ and enable automatic termination when the algorithm is confident in the estimation, we monitor the stability of the estimates over a sliding window of $h$ robot actions.
Let $T_{bc}^{i-h:i}$ represent the sequence of estimates over the past $h$ steps, and $\gamma\in\mathbb{R}^{6}$ denote the dimension-wise range of these estimates.
If the range $\gamma$ remains below a predefined threshold for all six dimensions over the past $h$ steps, the calibration process terminates and returns the current estimate $T_{bc}^{i}$ as the final transformation from the robot to the camera frame.

## VI Experimental Evaluation

We evaluate the proposed calibration method on the Franka Research 3 robot in both real-world and simulated environments.
In the simulation, we use PyBullet [25] to generate visual feedback for the calibration process.
In the real-world setup, we record the robot using a RealSense Camera D415.
In both cases, the camera resolution is set to 1920×1080 pixels, and recordings are captured at a frame rate of 30 Hz.

### VI-A Experiment in Simulation

345678910111213141516171819202122232425$0$$0.02$$0.04$$0.06$Number of Exploratory MotionsRotational Error (rad)

345678910111213141516171819202122232425$0$$0.1$$0.2$$0.3$Number of Exploratory MotionsTranslational Error (m)

*Fig. 7: (Top) Rotational Error in rad and (Bottom) Translational Error in m of the ARC-Calib method in the simulation over the number of exploratory motions.*

Since there is no ground truth in real-world scenarios, we first conducted a simulation study to quantitatively evaluate our algorithm’s performance under various robot-camera transformations.
The intrinsic parameters of the monocular camera were set to match those of the real-world camera.
We perform the calibration process for 10 different camera poses, with 5 runs per setup, and calculate the calibration error using both translational and rotational metrics.
Figure 7 shows the accuracy of the proposed camera-to-robot calibration method.
As expected, both rotational and translational errors decrease progressively as more observations from exploratory motions are collected.
With just 3 exploratory motions, the average rotational error is 0.0225 rad, and the translational error is 0.0786 m.
By increasing the number of exploratory motions to 25, the rotational error drops to 0.0042 rad, and the translational error reduces to 0.0065 m.

$5$$10$$15$$20$$25$$30$$0.2$$0.4$$0.6$$0.8$$1$Number of Exploratory MotionsIoU

*Fig. 8: IoU values represent calibration error over the number of exploratory motions. Solid lines indicate the performance of ARC-Calib, while dashed lines show the traditional calibration results. Each camera setup is represented by matching colors for both methods.*

![Refer to caption](drafts/images/arccalib-2503.14701/real1.png)

![Refer to caption](drafts/images/arccalib-2503.14701/real2.png)

![Refer to caption](drafts/images/arccalib-2503.14701/real3.png)

![Refer to caption](drafts/images/arccalib-2503.14701/real4.png)

![Refer to caption](drafts/images/arccalib-2503.14701/real5.png)

![Refer to caption](drafts/images/arccalib-2503.14701/real6.png)

*Fig. 9: Qualitative results of our method on the real-world experiment with 6 different camera pose setups. The blue overlays are the robot mask rendered based on the estimation of traditional hand-eye calibration implemented in the hand_eye_calibration ROS package, and the red overlays are the robot mask rendered based on the estimation of our ARC-Calib.*

### VI-B Experiment in Real-World

We evaluate the proposed method in a real-world environment by setting up the camera in six different poses, as shown in 9.
The proposed calibration method is run until the estimated results converge, as described in Section V-D.
For comparison, we run the traditional AprilTag-based hand-eye calibration method implemented in the hand_eye_calibration ROS package.
The traditional method continuously collects data points until the calibration results converge, with the convergence point manually determined by the operator.

Since ground truth data is unavailable in the real world, the robot’s projection can be used as an indirect measure of calibration accuracy.
We render the Franka robot’s mask onto the camera frame using the estimated transformations from different methods and compared them with manually labeled ground truth masks at random robot configurations.
The Intersection over Union (IoU) metric is calculated between the projected mask and the ground truth, where a higher IoU value corresponds to lower calibration error and better accuracy.

The results show that our proposed method achieves an average IoU of 0.94, outperforming the traditional method with an average IoU of 0.84.
Furthermore, the calibration process converges after an average of 26.5 selected robot motions.
As shown in Figure 8, the trending of the calibration error using IoU in the real world is similar to that of translational/rotational error in the simulation.

## VII Conclusion

In this work, we present a novel framework, ARC-Calib, for autonomous markerless camera-to-robot calibration, eliminating the need for human intervention or manual setup.
By introducing an exploratory robot motion, the robot generates an associated visual feedback pattern that can be extracted using lightweight methods such as optical flow, replacing traditional visual markers.
ARC-Calib is designed to be generalizable and able to operate across diverse robots and scenarios without requiring prior database knowledge, pre-trained models, or fine-tuning.
Evaluation in both physical and simulated settings demonstrates the method’s accuracy.
This makes it a highly adaptable and efficient solution for real-world robot-to-camera calibration for vision-based robot manipulation systems.

## References

- [1]

R. Horaud and F. Dornaika, “Hand-eye calibration,” The International Journal of Robotics Research, vol. 14, no. 3, pp. 195–210, 1995.
- [2]

F. C. Park and B. J. Martin, “Robot sensor calibration: solving ax= xb on the euclidean group,” IEEE Transactions on Robotics and Automation, vol. 10, no. 5, pp. 717–721, 1994.
- [3]

I. Ali, O. Suominen, A. Gotchev, and E. R. Morales, “Methods for simultaneous robot-world-hand–eye calibration: A comparative study,” Sensors, vol. 19, no. 12, p. 2837, 2019.
- [4]

I. Fassi and G. Legnani, “Hand to sensor calibration: A geometrical interpretation of the matrix equation ax= xb,” Journal of Robotic Systems, vol. 22, no. 9, pp. 497–506, 2005.
- [5]

E. Olson, “Apriltag: A robust and flexible visual fiducial system,” in IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2011, pp. 3400–3407.
- [6]

M. Fiala, “Artag, a fiducial marker system using digital techniques,” in IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR), vol. 2. IEEE, 2005, pp. 590–596.
- [7]

S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas, and M. J. Marín-Jiménez, “Automatic generation and detection of highly reliable fiducial markers under occlusion,” Pattern Recognition, vol. 47, no. 6, pp. 2280–2292, 2014.
- [8]

S. Bennett and J. Lasenby, “Chess–quick and robust detection of chess-board features,” Computer Vision and Image Understanding, vol. 118, pp. 197–210, 2014.
- [9]

J. Lu, Z. Liang, T. Xie, F. Ritcher, S. Lin, S. Liu, and M. C. Yip, “Ctrnet-x: Camera-to-robot pose estimation in real-world conditions using a single camera,” arXiv preprint arXiv:2409.10441, 2024.
- [10]

T. E. Lee, J. Tremblay, T. To, J. Cheng, T. Mosier, O. Kroemer, D. Fox, and S. Birchfield, “Camera-to-robot pose estimation from a single image,” in IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020, pp. 9426–9432.
- [11]

J. Lu, F. Richter, and M. C. Yip, “Markerless camera-to-robot pose estimation via self-supervised sim-to-real transfer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 21 296–21 306.
- [12]

Y. Labbé, J. Carpentier, M. Aubry, and J. Sivic, “Single-view robot pose and joint angle estimation via render & compare,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 1654–1663.
- [13]

B. K. Horn and B. G. Schunck, “Determining optical flow,” Artificial intelligence, vol. 17, no. 1-3, pp. 185–203, 1981.
- [14]

L. Yang, Q. Cao, M. Lin, H. Zhang, and Z. Ma, “Robotic hand-eye calibration with depth camera: A sphere model approach,” in International Conference on Control, Automation and Robotics (ICCAR). IEEE, 2018, pp. 104–110.
- [15]

R. Y. Tsai, R. K. Lenz et al., “A new technique for fully autonomous and efficient 3 d robotics hand/eye calibration,” IEEE Transactions on Robotics and Automation, vol. 5, no. 3, pp. 345–358, 1989.
- [16]

A. Traslosheros, J. M. Sebastián, E. Castillo, F. Roberti, and R. Carelli, “A method for kinematic calibration of a parallel robot by using one camera in hand and a spherical object,” in International Conference on Advanced Robotics (ICAR). IEEE, 2011, pp. 75–81.
- [17]

A. N. Staranowicz, G. R. Brown, F. Morbidi, and G.-L. Mariottini, “Practical and accurate calibration of rgb-d cameras using spheres,” Computer Vision and Image Understanding, vol. 137, pp. 102–114, 2015.
- [18]

J. Lambrecht and L. Kästner, “Towards the usage of synthetic data for marker-less pose estimation of articulated robots in rgb images,” in International Conference on Advanced Robotics (ICAR), 2019, pp. 240–247.
- [19]

Y. Zuo, W. Qiu, L. Xie, F. Zhong, Y. Wang, and A. L. Yuille, “Craves: Controlling robotic arm with a vision-based economic system,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019, pp. 4214–4223.
- [20]

L. Chen, Y. Qin, X. Zhou, and H. Su, “Easyhec: Accurate and automatic hand-eye calibration via differentiable rendering and space exploration,” IEEE Robotics and Automation Letters, 2023.
- [21]

J. Shi and Tomasi, “Good features to track,” in 1994 Proceedings of IEEE Conference on Computer Vision and Pattern Recognition, 1994, pp. 593–600.
- [22]

R. H. oy and J. Flusser, “Numerically stable direct least squares fitting of ellipses,” 1998.
- [23]

R. Safaee-Rad, I. Tchoukanov, K. Smith, and B. Benhabib, “Three-dimensional location estimation of circular features for machine vision,” IEEE Transactions on Robotics and Automation, vol. 8, no. 5, pp. 624–640, 1992.
- [24]

S. Agostinho, J. Gomes, and A. Del Bue, “Cvxpnpl: A unified convex solution to the absolute pose estimation problem from point and line correspondences,” Journal of Mathematical Imaging and Vision, vol. 65, no. 3, pp. 492–512, 2023.
- [25]

B. Ellenberger, “Pybullet gymperium,” [https://github.com/benelot/pybullet-gym](https://github.com/benelot/pybullet-gym), 2018–2019.

