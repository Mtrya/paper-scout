<!-- page 1 of 20 -->

arXiv:2610.42096v1 [cs.AI] 30 Oct 2026

Preprint

# CAUSALWAM: LEARNING THE CAUSES OF ACTIONS IN WORLD ACTION MODELS

**Juntao Cheng**<sup>1</sup>**, Jinzhou Tang**<sup>2</sup>**, Fan Feng**<sup>2</sup>**, Henrik I. Christensen**<sup>2</sup>**, Yujia Zheng**<sup>1 1</sup>UIUC <sup>2</sup>UCSD

## ABSTRACT

World action models (WAMs) learn latent variables from observations and use these to both predict the evolution of the scene and to generate actions. The pre-diction has to be of the whole scene, whereas only a portion of the scene causes actions. WAMs fitted to demonstrations do not, in general, guarantee that they will generate actions solely from the latent causes of the demonstrated actions (the action-relevant latent variables), as the model could just as well learn to imitate in this way whilst entangling the action-relevant latent variables with the nuisance variables. We therefore separate out what causes action from what prediction depends on. We prove that under a sparsity constraint the latent variables used to generate actions are an invertible function of the action-relevant latent variables alone and the true structure is identified up to a permutation of the latent variables. This requires neither sufficient temporal dependence nor any assumptions on how the unobserved regimes evolve and holds even when the action-relevant latent variables are statistically dependent on the rest. We therefore propose CWAM (Causal WAM), which incorporates the constraint into a pretrained WAM by adding a sparse routing path to its action head, with the world modeling objective remaining unchanged. Experiments on both synthetic data and real-world robots validate the theory as well as showcase the improved efficiency in closed-loop success.

## 1 INTRODUCTION

The action a robot performs when stacking a single block atop another depends on the position of both blocks and the fact that the gripper is currently holding the uppermost block. The camera image’s pixels will change significantly if the robot removes the tablecloth or clears away the clutter located behind the table; however, this will have no effect upon the motion itself. WAMs (World Action Models) like Fast-WAM (Yuan et al., 2026), and Cosmos 3 (NVIDIA, 2026) are able to extract latent variables from images and can be used to generate actions from those latent variables, while also being capable of generating images of potential subsequent frames. Since prediction has to account for the whole scene, these latent variables also describe the tablecloth and the clutter. We refer to the latent variables that cause the demonstrated actions as action-relevant latent variables, and to the others as nuisance variables.

Fitting the demonstrations well does not in itself guarantee that the WAM is acting from the actionrelevant latent variables. For each step of the demonstration we could write $o _ { t } = f ( s _ { t } )$ and $a _ { t }   =   \pi ( s _ { t } , \eta _ { t } )$ where $s _ { t }$ is the latent state, $f$ is some unknown invertible function, π is some unknown action mechanism and $\eta _ { t }$ is exogenous noise. Separating action from prediction is then about recovering (up to certain indeterminacies) the action-relevant latent variables and which of those variables cause each action dimension. Without further assumptions, the latent causal variables cannot be identified from their observations (Hyvarinen & Pajunen, 1999; Locatello et al., 2019). ¨ Observing the actions as well doesn’t help with this ambiguity, as two models can give rise to the same distribution of $( o _ { t } , a _ { t } )$ whilst their latent variables differ by an arbitrary invertible transformation. This transformation can mix the action-relevant latent variables with the nuisance ones, so a very flexible action head could generate actions from variables which also contain information about the tablecloth (which doesn’t in fact cause the motion) and still fit the demonstrations equally well.

To make latent variables identifiable, existing works in nonlinear independent component analysis (ICA) introduce auxiliary variables, such as domain or time indices, or exploit temporal dependence

1

<!-- page 2 of 20 -->

Preprint

(Hyvarinen & Morioka, 2016; 2017; Hyv ¨ arinen et al., 2019; Khemakhem et al., 2020). For la- ¨ tent causal processes, identifiability has been established under sufficient temporal dependence (Yao et al., 2022) and conditions on the latent dynamics (Song et al., 2023). Interventions with known targets have also been introduced to pin down the latents (Lippe et al., 2022). Many of these assumptions might be violated for demonstrations in complex real-world scenarios such as robotics, where some time steps might be independent, the latent dynamics might not be Markov, and controlled interventions are either prohibitively expensive or simply impossible.

Another line of research constrains the generative process by restricting the function class or enforcing sparsity in the dependency structure (Kivva et al., 2022; Zheng et al., 2022; Lachapelle et al., 2022). Most of these studies, similar to those previously discussed, aim to recover all latent variables, requiring their conditions to hold even for nuisance variables that a robot can disregard. In contrast, results on partial identifiability focus on a subset of latent variables, determined by side information accompanying the data, such as content shared across views (von Kugelgen et al., 2021; ¨ Yao et al., 2024), components invariant across domains (Kong et al., 2022), or elements relevant to specific tasks (Zheng et al., 2026). For robots learning from demonstrations, the relevant subset consists of the latent causes underlying the demonstrated actions. Since no view, domain, or task label directly specifies this subset, the actions themselves must indicate which latent variables are pertinent. This raises the following question:

## From demonstrations alone, can we separate what causes actions from what prediction needs?

In this work, we answer this question with a constraint on the action side that leaves prediction free to use every latent variable. Since each action is generated from only part of the latent state, constraining the number of dependencies between latent variables and action dimensions can separate that part from the rest. Specifically, we prove that, without such a constraint and under nondegeneracy conditions, fitting the demonstrations only ensures that each action dimension of the estimated mechanism depends on at least as many latent variables as the true one (Prop. 1). We then incorporate a sparsity regularization on the dependency structure between latent state and action variables during estimation. Under the same conditions, the lower bound becomes an equality and the dependency structure is identified up to a permutation of the latent variables (Thm. 1).

Moreover, we prove that the estimated action-relevant variables under the proposed constraint are an invertible function of the ground-truth latent variables causing the true action (Thm. 2). The latent causes of the actions are thus identified as a block and are separated from the nuisance variables. The theoretical guarantee relies purely on observational demonstrations, without requiring any interventions, sufficient temporal dependence, or any assumptions about the latent dynamics. The framework also allows dependence between action-relevant latent variables and the nuisance variables (e.g., a tablecloth could be correlated with the positions of the cubes). According to the established theory, we propose CWAM (Causal WAM), which adds a sparse routing path from the visual and task tokens to the action head in Cosmos 3 and applies a token-level surrogate of the constraint to this path (Sec. 4). In synthetic scenarios, we validate the identifiability claims shown by our theory. On RoboCasa365, LIBERO, and a real Franka R3 robot, the routing path improves aggregate closed-loop success over Cosmos 3 without it, and in most visualized frames its routing mass falls on the object being moved and its target (Sec. 5).

## 2 PROBLEM SETUP

**Data.** Demonstration data are sequences $\{ o _ { t } , a _ { t } \} _ { t = 1 } ^ { T }$ with $o _ { t } \in \mathbb { R } ^ { d _ { o } }$ and $a _ { t } \in \mathbb { R } ^ { d _ { a } }$ . Relevance is defined by the executed action at each step, and no task, skill, or segment labels are assumed.

**Observational world action process.** We define latent states $\{ s _ { t } \} _ { t = 1 } ^ { T } , s _ { t } \in \mathbb { R } ^ { n }$ , and hidden non-parametric maps

$$
o _ { t } = f ( s _ { t } ) ,\tag{1a}
$$

$$
a _ { t } = \pi ( s _ { t } , \eta _ { t } ) ,\tag{1b}
$$

where $f : \mathbb { R } ^ { n } \to \mathbb { R } ^ { d _ { o } }$ is a diffeomorphism onto its image $( d _ { o } \gg n$ is allowed), π is an unknown action mechanism, and $\eta _ { t }$ is exogenous noise. The same mechanism applies to an H-step chunk $a _ { t : t + H - 1 }$ when the policy emits a short horizon at once (Section 4); identifiability is stated per

2

<!-- page 3 of 20 -->

Preprint

action coordinate. We do not require temporal dependence: consecutive steps may be disconnected and each $( o _ { t } , a _ { t } )$ may be i.i.d. Within a step, action-relevant latent variables are those on which π functionally depends. Since $f$ is invertible, $o _ { t }$ depends on every latent variable, whereas π depends only on the action-relevant ones. A fitted action mechanism may over-cover the support of $J _ { \pi }$ , that $\dot { \mathbf { 1 } } \mathbf { s } ,$ depend on more latent variables than π, and we ask when the latent variables used to generate actions are separated from the nuisance ones (notation below).

**Mechanism supports and the compact state.** For a differentiable structural map m of the state, let $J _ { m } ( s )   =   \partial m / \partial s$ , and let $\bar { \mathcal { I } ( J _ { m } ) } = \{ ( i , j ) \colon \operatorname* { P r } [ J _ { m } ( s ) _ { i j } \neq 0 ] > 0 \}$ be its functional support, with $\| \dot { \mathcal { I } } ( J _ { m } ) \|$ 0 its cardinality. For $\pi ,$ the Jacobian is taken with respect to $s _ { t }$ at fixed $\eta _ { t }$ . We write $I _ { i } : = \{ j : ( \stackrel { \frown } { i } , j ) \in \mathcal { I } ( J _ { \pi } ) \}$ for the coordinates that drive action dimension i, set $\begin{array} { r } { I _ { \mathrm { A } } = \bigcup _ { i } I _ { i } } \end{array}$ , and write $s _ { t , I _ { \mathrm { A } } }   =   ( s _ { t , j } ) _ { j \in I _ { \mathrm { A } } }$ for the action-relevant subvector. Relevance is functional (a perturbation of coordinate j can change a) and is not a statement about correlation. A learner returns an encoder $\hat { f } ^ { - 1 }$ with $\hat { s } _ { t }   =   \hat { f } ^ { - 1 } ( o _ { t } )$ , an action mechanism $\hat { \pi } ,$ estimated index sets $\hat { I } _ { i }   : =   \{ j   :   ( i , j )   \in   \mathcal { I } ( J _ { \hat { \pi } } ) \}$ and $\begin{array} { r } { \hat { I } _ { \mathrm { A } } = \bigcup _ { i } \hat { I } _ { i } } \end{array}$ , and the compact action-relevant state $C _ { t } : = \hat { s } _ { t , \hat { I } _ { \mathrm { A } } }$ . At deployment, $C _ { t }$ is computed from observations only; training may use $a _ { t }$ to fit $\hat{\pi},$ , but actions are not inputs to $\hat { f } ^ { - 1 }$

**Identification goal.** Write $\mathcal { L } _ { f , \pi }$ for the law of $( o , a )$ induced by equation 1a–equation 1b. Two processes $( f , \pi )$ and $( \hat { f } , \hat { \pi } )$ are observationally equivalent if

$$
\mathcal { L } _ { f , \pi } \; = \; \mathcal { L } _ { \hat { f } , \hat { \pi } } .\tag{2}
$$

Under observational equivalence on $( o , a )$ , the estimated state $\hat { s } _ { t }$ determines every latent variable, as prediction requires, so we only ask about the action side. We first ask whether the estimated action mechanism matches the true state-to-action dependency pattern. We then ask whether the compact state $C _ { t }$ used to generate actions is separated from the nuisance variables, in the sense that it carries the action-relevant block $s _ { t , I _ { \mathrm { A } } }$ <sup>up</sup> to a block-wise invertible transformation. Section 3 states sufficient conditions for both questions. Section 5.1 is the direct empirical check with known $I _ { \mathrm { A } } ,$ , and Appendix C records the synthetic generator, estimators, and metrics in full.

**Estimated process.** An estimated process is any pair $( \hat { f } , \hat { \pi } )$ that satisfies equation 1a–equation 1b with $\hat { f } \; : \; \mathbb { R } ^ { n } \; \rightarrow \; \mathbb { R } ^ { d _ { o } }$ a diffeomorphism onto its image. With $\hat { s } _ { t } \; = \; \hat { f } ^ { - 1 } ( o _ { t } )$ and $\phi \; : = \; \hat { f } ^ { - 1 } \circ$ $f ,$ observational equivalence allows $\phi$ to be an arbitrary diffeomorphism on $\dot { \mathbb { R } } ^ { n }$ compatible with equation 2. By itself, observational equivalence does not identify the full latent state $s _ { t }$ or a unique encoder $\hat { f } ^ { - 1 }$ . Section 3 asks what is identified (the compact state $C _ { t }$ and the functional support of $J _ { \pi } )$ under additional inductive bias.

**Regularity and coupling.** Throughout, $f ,   \hat { f } ,   \pi ( \cdot , \eta )$ , and $\hat { \pi } ( \cdot , \eta )$ are continuously differentiable, $f$ and $\hat { f }$ have the same image (so ϕ is a diffeomorphism of $\mathbb { R } ^ { n } )$ , and $s _ { t }$ has a density that is positive on $\mathbb { R } ^ { n }$ (hence so does $\hat { s } _ { t } = \phi ( s _ { t } ) )$ . The results below use a pathwise coupling of the two processes, in which they share the noise and share the latent state up to $\phi ,$ that is, $o _ { t }   =   f ( s _ { t } )   =   \hat { f } ( \hat { s } _ { t } )$ and $a _ { t } = \pi ( s _ { t } , \eta _ { t } ) = \hat { \pi } ( \hat { s } _ { t } , \eta _ { t } )$ with $\hat { s } _ { t } = \phi ( s _ { t } )$ . This coupling implies equation 2.

## 3 THEORY

We now study, within each time step, when fitting $( o , a )$ separates the latent causes of the action from the nuisance coordinates that only prediction needs. In equation 1, only the block $s _ { t , I _ { \mathrm { A } } }$ <sup>need</sup> enter π. Remaining coordinates may still be required for $o _ { t } = f ( s _ { t } )$ , so a latent state sufficient for observations need not isolate the coordinates that cause the action. All guarantees below concern one application of equation 1, or equivalently one pair $( o , a )$ drawn from $\mathcal { L } _ { f , \pi }$ in equation 2. We keep the subscript t when referring to demonstration tuples $( o _ { t } , a _ { t } )$ , but the argument does not use cross-time structure (Appendix A fixes a step and drops $t )$ . The within-step conclusion applies at every t, and if the steps are identically distributed, the law in equation 2 also does not depend on t.

3

<!-- page 4 of 20 -->

Preprint

## 3.1 IDENTIFIABILITY WITHOUT SPARSITY

Without a support-sparsity constraint on $\hat { \pi } ,$ observational equivalence on $( o , a )$ may still hold while πˆ depends on more latent coordinates than π. The action side then need not be separated from the nuisance coordinates, and fitting guarantees only the one-sided bound in Proposition 1.

**Proposition 1.** Consider two observationally equivalent processes $o _ { t } \; = \; f ( s _ { t } ) \; = \; \hat { f } ( \hat { s } _ { t } ) , a _ { t } \; =$ $\pi ( s _ { t } , \eta _ { t } ) \; = \; \hat { \pi } ( \hat { s } _ { t } , \eta _ { t } )$ of the form equation 1 that satisfy the regularity conditions of Section $2 .$ Assume that, for each action dimension $i \in [ d _ { a } ] ,$ there exist $\left| I _ { i } \right|$ states $\dot { s ^ { ( 1 ) } } , \ldots , s ^ { ( | I _ { i } | ) }$ at which the rows $( J _ { \pi } ( s ^ { ( l ) } ) ) _ { i , }$ · are linearly independent, and for $\phi = \hat { f } ^ { - 1 } \circ f ,   P = J _ { \phi ^ { - 1 } }$ , and a matrix M with $\mathcal { I } ( M ) { = } \mathcal { I } ( P ) , \widetilde { \mathcal { I } \big ( ( J _ { \pi } ( s ^ { ( l ) } ) M ) _ { i , \cdot } \big ) } { \subseteq } \mathcal { I } \big ( ( J _ { \hat { \pi } } ( \hat { s } ^ { ( l ) } ) ) _ { i , \cdot } \big )$ with $\hat { s } ^ { ( l ) }   =   \phi ( s ^ { ( l ) } )$ . Then, for every action dimension $i \in [ d _ { a } ]$

$$
\| \mathcal { I } \big ( ( J _ { \hat { \pi } } ) _ { i , \cdot } \big ) \| _ { 0 } \; \geq \; \| \mathcal { I } \big ( ( J _ { \pi } ) _ { i , \cdot } \big ) \| _ { 0 } .
$$

**Proof sketch.** The span condition ensures that reparameterization between s and ŝ cannot eliminate a true functional dependency on $I _ { i }$ . The inclusion condition links $J _ { \hat { \pi } }$ to $J _ { \pi }$ through $\phi$ and rules out accidental cancellations, so, up to a permutation $\rho$ of the latent coordinates that is common to all action dimensions, each row of $J _ { \hat { \pi } }$ must activate at least the coordinates required by the corresponding row of $J _ { \pi }$ . Counting these coordinates gives the bound. Details are in Appendix A.

**Discussion on assumptions.** The span condition in Proposition 1 requires the Jacobian row of each action dimension to span the coordinate subspace indexed by its true parents. This is a separate assumption on the action mechanism: smoothness and a positive state density alone do not imply it. A fuller discussion is in Appendix B.1.

**Implication.** Proposition 1 makes over-coverage precise. Without a bias toward a minimal action pathway, observational fitting need not shrink πˆ to π. It need only learn a mechanism that is no smaller in support than the true one. The estimated structure from latent coordinates to actions can then contain spurious edges, through which nuisance coordinates enter the action side. Neither matching $\mathcal { I } ( J _ { \pi } )$ nor invertible recovery of $s _ { t , I _ { \mathrm { A } } }$ <sup>in</sup> $C _ { t }$ is guaranteed.

## 3.2 IDENTIFIABILITY UNDER SUPPORT SPARSITY

The negative result points to the design choice used in our method: constrain which latent coordinates πˆ may depend on. Support sparsity requires that the estimated mechanism activate no more coordinates, counted over action dimensions, than the true one,

$$
\| \mathcal { I } ( J _ { \hat { \pi } } ) \| _ { 0 } \; \leq \; \| \mathcal { I } ( J _ { \pi } ) \| _ { 0 } .\tag{3}
$$

Combined with Proposition 1, this inequality bounds the size of the estimated support from both sides and identifies the state-to-action dependency pattern up to relabeling. The mechanism-level guarantee is as follows.

**Theorem 1.** Consider two observationally equivalent processes $o _ { t } = f ( s _ { t } ) = \hat { f } ( \hat { s } _ { t } ) , a _ { t } =$ $\pi ( s _ { t } , \eta _ { t } )   =   \hat { \pi } ( \hat { s } _ { t } , \eta _ { t } )$ of the form equation 1, and assume the regularity conditions of Section $^ { 2 , }$ the conditions in Proposition 1, and equation 3. Then there is a permutation $\rho   o f [ n ]$ such that

$$
\mathcal { I } ( J _ { \hat { \pi } } ) \; = \; \mathcal { I } ( J _ { \pi } )   P _ { \rho } ,
$$

and, for every action dimension $i   \in   [ d _ { a } ] ,   \mathcal { I } \big ( ( J _ { \hat { \pi } } ) _ { i , \cdot } \big )   =   \rho ( I _ { i } )$ . Here $P _ { \rho }$ is the permutation matrix with $( P _ { \rho } ) _ { j , k } = \mathbf { 1 } \{ \rho ( j ) = k \}$ , and supports are identified with $0 / 1$ matrices.

**Proof sketch.** By the proof of Proposition 1, reparameterization through $\phi   =   \hat { f } ^ { - 1 } \circ f$ gives a common permutation $\rho$ with $\rho ( I _ { i } ) \; \subseteq \; \mathcal { I } ( ( J _ { \hat { \pi } } ) _ { i , \cdot } )$ for every i. Sparsity equation 3 forbids extra activations, so the inclusions become equalities and assemble into $\tilde { \mathcal { I } } ( J _ { \hat { \pi } } ) \doteq \tilde { \mathcal { I } } ( J _ { \pi } ) P _ { \rho }$ (Appendix A).

**Implication.** Each action dimension i of the fitted mechanism depends on exactly the estimated coordinates $\rho ( I _ { i } )$ , which correspond to its true parents $I _ { i } ,$ and the bipartite pattern from latent coordinates to actions is matched, whereas without sparsity it can be enlarged. Sparsity turns the lower

4

<!-- page 5 of 20 -->

Preprint

bound in Proposition 1 into an exact answer to which coordinates each action dimension functionally depends on. Theorem 1 concerns the Jacobian of $\hat{\pi}.$ , that is, the bipartite pattern from latent coordinates to executed actions. However, the robot deploys the compact state $C _ { t } = \hat { s } _ { t , \hat { I } _ { \mathrm { A } } }$ , which is computed from observations without access to $a _ { t }$ . Even with matched supports, one could imagine a lossy projection that drops information within the action-relevant block or leaks nuisance coordinates into $C _ { t }$ . The next result shows that this does not happen under the coupling of Section 2 once supports align. Specifically, the reparameterization $\phi = { \hat { f } } ^ { - 1 } \circ f$ cannot mix the nuisance coordinates into the estimated coordinates $\rho ( I _ { \mathrm { A } } )$ , so $C _ { t }$ is an invertible function of $s _ { t , I _ { \mathrm { A } } }$

**Theorem 2.** Under the hypotheses of Theorem 1, including the regularity conditions of Section 2, with the same permutation $\rho ,$ there exists an invertible map h such that

$$
C _ { t } \; = \; \hat { s } _ { t , \rho ( I _ { \mathrm { A } } ) } \; = \; h \big ( s _ { t , I _ { \mathrm { A } } } \big ) .
$$

Proof sketch. For $\phi   =   \hat { f } ^ { - 1 } \circ f$ and $P   =   J _ { \phi ^ { - 1 } }$ , the aligned row supports $\mathcal { I } ( ( J _ { \hat { \pi } } ) _ { i , \cdot } )   =   \rho ( I _ { i } )$ from Theorem 1 imply that $P$ is block lower-triangular, with a zero block in rows $I _ { \mathrm { A } }$ and columns $\rho ( [ n ] \setminus I _ { \mathrm { A } } )$ . The invertibility of $P$ then yields $\hat { s } _ { t , \rho ( I _ { \mathrm { A } } ) } = h ( s _ { t , I _ { \mathrm { A } } } )$ with invertible h. By Theorem 1, $\hat { I } _ { i } = \rho ( I _ { i } )$ for every i, so $\hat { I } _ { \mathrm { A } } = \rho ( I _ { \mathrm { A } } )$ and $C _ { t } = \hat { s } _ { t , \rho ( I _ { \mathrm { A } } ) }$ . Details are in Appendix A.

**Implication.** The compact state $C _ { t }$ is an invertible re-encoding of the true action-relevant block $s _ { t , I _ { \mathrm { A } } }$ This is stronger than being correlated with pose or gripper-like factors. The conclusion is functional $( C _ { t }$ depends only on $s _ { t , I _ { \mathrm { A } } }$ and determines it), so it holds whatever the statistical dependence between $s _ { t , I _ { \mathrm { A } } }$ and the nuisance coordinates.

## 4 ESTIMATION METHOD

**Estimation principle.** Under the pathwise coupling and regularity conditions of Section 2, the nondegeneracy conditions of Proposition 1, and support sparsity equation 3, Section 3 identifies the support of $J _ { \pi }$ up to a permutation and the action-relevant latent variables in $C _ { t }$ up to a block-wise invertible transformation. We refer to a model fitted to demonstrations with a sparsity penalty on its action mechanism as CWAM (Causal WAM). Synthetic experiments impose coordinate sparsity on a linear head directly (Section 5.1). The deployed action interface extends equation 1b to an H-step chunk:

$$
\hat { a } _ { t : t + H - 1 } \; = \; \hat { \pi } \big ( o _ { t } , \epsilon _ { t } ; \thinspace C _ { t } \big ) , \qquad C _ { t } \; = \; \mathcal { S } \big ( \hat { f } ^ { - 1 } ( o _ { t } ) \big ) ,\tag{4}
$$

where $\epsilon _ { t }$ is exogenous noise for generation (an initial draw for rectified flow, in parallel with $\eta _ { t }$ in equation 1b), S selects a support-sparse subset of the estimated state, and $H$ is the chunk horizon. Figure 1 shows the split between world modeling and sparse routing to the action head.

**Backbone.** Cosmos 3 Nano (NVIDIA, 2026) co-trains video prediction and action generation from shared context. CWAM keeps that backbone and post-trains it on demonstrations $( o , a )$ with the standard joint rectified-flow recipe (Lipman et al., 2023; Liu et al., 2023b). A mixture-of-transformers (MoT) generator (Liang et al., 2024) denoises future video latents and the executed action chunk, conditioned on clean history and language. The leading frame in each window is observed, and later video tokens are noised during training. CWAM instantiates the estimation principle above as

$$
\operatorname* { m i n } _ { \hat { f } , \hat { \pi } } \; \mathcal { D } \big ( \mathcal { L } _ { f , \pi } , \: \mathcal { L } _ { \hat { f } , \hat { \pi } } \big ) \; + \; \mathcal { R } _ { \operatorname { s p a r s e } } ( \hat { \pi } ) ,\tag{5}
$$

where D penalizes mismatch between the true and estimated laws of $( o , a )$ from equation 1a– equation 1b, and $\mathcal { R } _ { \mathrm { s p a r s e } }$ biases πˆ toward a minimal action pathway as in equation 3. In Cosmos 3, $\mathcal { D }$ is the joint post-training loss $\mathcal { L } _ { \mathrm { W A M } }$ (the rectified-flow video and action co-training described above), and $\mathcal { R } _ { \mathrm { s p a r s e } }   =   \mathcal { L } _ { \mathrm { s p a r s e } }$ in equation 7. CWAM leaves $\mathcal { L } _ { \mathrm { W A M } }$ unchanged, adds the sparse router, and applies $\mathcal { L } _ { \mathrm { s p a r s e } }$ to the router readout only.

**Action-relevant router.** After the generator tower, each action hidden state $h _ { i }$ routes over clean context indices $j \in \mathcal { T } _ { t }$ (task tokens and clean vision, excluding future noised tokens):

$$
w _ { i j } \: = \: { \operatornamewithlimits { s o f t m a x } } _ { j \in \mathcal { J } _ { t } } \Big ( \frac { q _ { i } ^ { \top } k _ { j } } { \sqrt { d _ { r } } } + b _ { i j } ( \mathsf { r e l a t i o n s } ) \Big ) , \qquad \tilde { h } _ { i } \: = \: \sum _ { j \in \mathcal { J } _ { t } } w _ { i j } \: v _ { j } ,\tag{6}
$$

5

<!-- page 6 of 20 -->

Preprint

(a) Sparse action-relevant support

![Image block](doc:779ae9e/tier:basic/page:6/block:3)

Figure 1: The action depends only on a block of the latent state, and CWAM routes an estimate of this block to the action head through a sparse path. (a) In the world action process equation 1 this block is $s _ { t , I _ { \mathrm { A } } }$ , and the learner $( \hat { f } , \hat { \pi } )$ estimates it from observations as the compact state $C _ { t }$ (b) CWAM keeps the joint rectified-flow training of Cosmos 3 and adds a sparse router over clean context tokens, whose top-k routed tokens form $C _ { t }$ . Gray dashed arrows mark the dense path to the action head, which the default policy keeps.

with relation-aware biases $b _ { i j }$ on position, modality, and chunk progress $( d _ { r } = 5 1 2 , 8$ heads). The routed signal enters through a learned gate $g \in ( 0 , 1 )$ , initialized small, as $h _ { i } \leftarrow h _ { i } + g \: \tilde { h } _ { i }$ . Training adds a token-level surrogate for equation 3 on this pathway,

$$
\mathcal { L } _ { \mathrm { s p a r s e } } \; = \; \lambda   \mathbb { E } _ { i } \Big [ 1 - \sum _ { j \in T _ { k } ( i ) } w _ { i j } \Big ] ,\tag{7}
$$

where $T _ { k } ( i ) \subseteq \mathcal { J } _ { t }$ is the index set of the k largest weights $w _ { i j } ,$ so the sum is the attention mass on the top-k context tokens $( \lambda = 0 . 0 1$ , ramped in after $1 0 ^ { \widetilde { 3 } }$ steps). Post-training minimizes

$$
\mathcal { L } _ { \mathrm { t o t a l } } \; = \; \mathcal { L } _ { \mathrm { W A M } } \; + \; \mathcal { L } _ { \mathrm { s p a r s e } } ,\tag{8}
$$

with $\mathcal { L } _ { \mathrm { W A M } }$ identical to that of the matched Cosmos 3 baseline on the same demonstration windows (rectified-flow targets for noised future video tokens and the action chunk).

**Compact state.** The interface in equation 4 idealizes $C _ { t } = \mathcal { S } ( \hat { f } ^ { - 1 } ( o _ { t } ) )$ on encoder coordinates; on Cosmos 3 the same role is played by clean context tokens:

$$
C _ { t } \; = \; \Big \{ j \in \mathcal { J } _ { t } \; \Big | \; \exists \: i : \; j \in T _ { k } ( i ) \Big \} ,\tag{9}
$$

i.e. clean tokens in the top-k routed mass of some action query. The default policy retains the dense MoT visual to action path, and equation 7 regularizes the router branch of πˆ.

## 5 EXPERIMENTS

We evaluate CWAM in two settings. On synthetic world action processes with known latent variables, the compact state $C _ { t }$ used by the sparse action head recovers the action-relevant block, which is consistent with Theorem 2 (Section 5.1). On RoboCasa365, LIBERO, and a real Franka robot, the sparse routing path improves aggregate closed-loop success over the matched Cosmos 3 baseline (Section 5.2), and on the four RoboCasa365 tasks where CWAM is strongest, the routed tokens alone keep most of that success (Section 5.3). Appendix C gives the synthetic protocol in full.

## 5.1 THE COMPACT STATE RECOVERS THE ACTION-RELEVANT BLOCK

Theorem 2 predicts that $C _ { t }$ is an invertible function of the action-relevant block $s _ { t , I _ { \mathrm { A } } }$ <sup>alone</sup>. Comparing $C _ { t }$ with this prediction needs the true latent variables and $I _ { \mathrm { A } }$ , which a simulator provides.

6

<!-- page 7 of 20 -->

Preprint

Table 1: The coordinates selected by CWAM with sparsity decode the action-relevant latent variables as well as the dense readout, and the nuisance variables only about as well as the true block. Zeroing them breaks the fitted head. Held-out linear probes give the relevant $R ^ { 2 }$ and $\Delta _ { \mathrm { n u i s } } ,$ the nuisance $R ^ { \check { 2 } }$ of $C _ { t }$ minus that of $s _ { t , I _ { \mathrm { A } } }$ . Selected/full (Complement/full) is the held-out action MSE of the fitted head, not refitted, after zeroing the coordinates outside (inside) $C _ { t } ,$ relative to the full $\hat { s } _ { t }$ . Mean ± population std over five seeds, with ratios computed per seed.

| Method | \|C<sub>t</sub>\| | Relevant R<sup>2</sup> ↑ ∆nuis ↓ | Selected/full | Complement/full |
| --- | --- | --- | --- | --- |
| Dense (20D readout) | 20 | 0.9961 ± 0.0015 0.646 ± 0.115 | n/a | n/a |
| Random matched (4D) | 4 | 0.5493 ± 0.2964 0.392 ± 0.144 | 48.067 ± 31.855 | 6.36 ± 6.42 |
| CWAM without sparsity | 4 | 0.9876 ± 0.0164 -0.008 ± 0.006 | 3.456 ± 3.180 | 50.58 ± 28.81 |
| Standard MLP VAE | 4 | 0.9440 ± 0.0448 0.316 ± 0.186 | 3.259 ± 2.211 | 51.13 ± 28.54 |
| CWAM with sparsity | 4 | 0.9957 ± 0.0014 0.039 ± 0.093 | 1.704 ± 1.398 | 54.65 ± 25.78 |

**Setup.** The process follows equation 1 with $n   =   2 0$ latent variables, $| I _ { \mathrm { A } } |   =   4$ of which drive the action, and observes each latent variable through its own noisy coordinates. A hidden regime, never given to the estimator, scales the action mechanism and a transition that couples the nuisance variables to the action-relevant ones. Each estimator pairs an observation-only MLP encoder with a linear action head, whose weight matrix is its Jacobian, so CWAM with sparsity uses a group penalty on the head columns, a column-level surrogate for equation 3. The compact state $C _ { t }$ keeps the $| I _ { \mathrm { A } } |$ coordinates with the largest mean absolute head weight. The baselines are a standard MLP VAE (Zheng et al., 2026) fitted without head sparsity and three readouts of one fit without the penalty, namely the full latent state (dense), $| I _ { \mathrm { A } } |$ random coordinates (random matched), and the selected coordinates (CWAM without sparsity). Held-out linear probes give the $R ^ { 2 }$ from $C _ { t }$ to the actionrelevant latent variables. For the nuisance variables we report $\tilde { \Delta _ { \mathrm { n u i s } } } = R ^ { 2 } ( C _ { t }   \rightarrow   \mathrm { n u i s } )   - R ^ { 2 } ( s _ { t , I _ { \mathrm { A } } }   \rightarrow$ nuis), the excess of $C _ { t }$ over the oracle reference, i.e., over how well the true block predicts them.

**Block recovery.** A block that depends on $s _ { t , I _ { \mathrm { A } } }$ alone can predict the nuisance variables only through their dependence on $s _ { t , I _ { \mathrm { A } } }$ , so under Theorem 2 the nuisance excess of $C _ { t }$ should be near zero. For the coordinates selected by CWAM with sparsity it is indeed close to zero, and they decode the action-relevant latent variables about as well as the dense readout (Table 1). In contrast, the dense readout and the coordinates selected from the VAE decode the nuisance variables well beyond the true block, and four random coordinates recover the action-relevant ones poorly.

**Head ablation.** A probe measures what $C _ { t }$ contains, while the fitted head may still draw on other coordinates. Setting $C _ { t }$ to zero raises the action error by a similarly large factor for every head that selected a block, including the VAE, while setting the complement to zero costs much less (Table 1). Random matched shows the opposite pattern, and keeping only its coordinates breaks the head, so the head depends on which coordinates are kept more than on how many.

**Hidden environments.** The theorems assume nothing about how unobserved regimes evolve, so hidden variation in the nuisance dynamics should leave the conclusion of Theorem 2 unchanged. A second hidden index $e _ { t } ,$ , resampled at every step and never given to the encoder, rescales only the nuisance variables in the transition, and we increase its number of values $E$ with the action mechanism fixed. As E grows, neither readout loses accuracy on the action-relevant latent variables, and only the dense readout keeps a large nuisance excess (Appendix D, Table 4).

## 5.2 THE SPARSE ROUTING PATH IMPROVES CLOSED-LOOP SUCCESS

For a pretrained WAM, the true latent variables are unknown, so on Cosmos 3 we ask whether the sparse routing path improves closed-loop success. We compare with Cosmos 3 which is post-trained identically with matched setups but without the router and $\mathcal { L } _ { \mathrm { s p a r s e } }$ (Section 4). CWAM succeeds more often on RoboCasa365 (Nasiriany et al., 2026), LIBERO, and a real Franka robot that stacks cubes (Figure 2).

On RoboCasa365, CWAM achieves a higher mean success rate than the baseline in every foundational skill family (Figure 2(a)). The gain is small for button pressing and largest for drawer opening

7

<!-- page 8 of 20 -->

Preprint

![Image block](doc:779ae9e/tier:basic/page:8/block:2)

![Chart block](doc:779ae9e/tier:basic/page:8/block:3)

Figure 2: CWAM improves closed-loop success over the matched Cosmos 3 baseline in every Robo-Casa365 skill family, every LIBERO suite, and on the real robot. (a) RoboCasa365, mean success per skill family over 17 tasks (tasks in Table 5). (b) LIBERO, success per suite over 40 tasks. (c) Franka stack cube, 50 trials per policy. Panels (a) and (b) use 100 episodes per task. The baseline (Cosmos 3 in the legend) is post-trained on the same data and recipe, without the router and $\mathcal { L } _ { \mathrm { s p a r s e } } .$

Table 2: On the four RoboCasa365 tasks where CWAM succeeds most, keeping only the routed tokens costs far less success than keeping as many random tokens. The CWAM checkpoint is fixed at 50k steps and run on the episodes of the main evaluation. The action queries see the full visual context (CWAM), only the routed tokens that form $C _ { t }$ (C-only), or as many visual tokens drawn uniformly at random from the clean context (random matched). Neither intervention refits the policy.

| Task | CWAM | C-only | Random matched |
| --- | --- | --- | --- |
| Pick place sink to counter | 74% | 63% | 28% |
| Pick place toaster to counter | 72% | 61% | 21% |
| Open drawer | 70% | 59% | 33% |
| Open stand mixer head | 69% | 58% | 16% |
| Average (4 tasks) | 71.3% | 60.3% | 24.5% |

and closing, rack sliding, insertion, and pick-and-place tasks. Averaged across tasks, CWAM’s success rate is nearly twice that of the baseline (Table 5). On LIBERO (Liu et al., 2023a), both policies achieve high success rates, with CWAM outperforming the baseline across all suites and showing the largest gain on LIBERO-Goal (Figure 2(b)). In real-robot experiments, CWAM succeeds in a majority of trials, whereas the baseline succeeds in fewer than half (Figure 2(c)).

**Where does the router look?** In sampled frames of the real-robot sequence in Figure 3 (bottom), routing weights are concentrated on the two cubes involved in the stacking task described in the introduction. The RoboCasa365 rollout (top) shows a similar pattern, with most of the weight assigned to the manipulated object and its target. In some frames, however, the weights are more dispersed, extending to the robot arm and the wall, as in the fourth frame of the real-robot sequence.

## 5.3 ROUTED TOKENS ALONE KEEP MOST OF CWAM’S SUCCESS ON ITS BEST TASKS

Routing visualizations do not establish whether the routed tokens alone are sufficient for action pre-diction, since the default policy retains a dense visual-to-action pathway (Section 4). They also do not show whether a different set of tokens of the same size would support comparable performance. To test these questions, we evaluate a fixed CWAM checkpoint under two visual-context interventions on the four RoboCasa365 tasks with the highest success rates. C-only restricts the visual context available to the action queries to the routed tokens defining $C _ { t }$ in equation 9. Random matched instead provides the action queries with an equal number of randomly selected visual tokens.

8

<!-- page 9 of 20 -->

Preprint

RoboCasa

![Image block](doc:779ae9e/tier:basic/page:9/block:3)

Figure 3: In most frames, the routing mass falls on the object being moved and its target. (Top) A RoboCasa365 rollout. (Bottom) The real robot stacking the red cube on the green cube. Rows show the wrist and agent (third-person) views, each followed by the same view with the routing mass overlaid. Warmer colors mark more of the top-k routing mass over clean visual tokens.

As shown in Table 2, C-only retains most of the default policy’s performance, whereas Random matched results in a substantially larger drop in success rate, with the same ordering across all four tasks. These results suggest that the routed tokens retain much of the visual information needed to perform these tasks. Consistent with the synthetic-head results, the gap between the two conditions shows that token selection affects performance even when the number of retained tokens is fixed.

## 6 CONCLUSION

We show how to learn causal actions in WAM from observational decomposition alone. Specifically, we prove that under non-degeneracy conditions and without a sparsity constraint, each action dimension of any mechanism estimated to fit the demonstrations can contain action-irrelevant latent variables. Sparsity regularisation enables identification of the true causes of the actions and the causal structure from latent variables to actions (up to a permutation). The latent variables used to generate the actions are then an invertible function of the latent causes of the actions alone, while the full estimated state continues to describe the entire scene for prediction. Neither of the identifi-

9

<!-- page 10 of 20 -->

Preprint

cation results rely on interventions, temporal structure or supervision. Synthetic experiments are in agreement with the theory and CWAM does in fact assist with task completion in simulation as well as on a real robot. The main limitation is the lack of component-wise identifiability that would pin down the individual latents (which may not be required for separating actions but would be useful for precise interventions). There are many more open problems in WAMs that causality can help to address and this paper represents one of the first steps towards a causal understanding of actions.

## AI STATEMENT

In this paper, we used generative AI tools to assist coding, language editing, and figure plotting. We have reviewed all AI-assisted work. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## REFERENCES

Aapo Hyvarinen and Hiroshi Morioka. Unsupervised feature extraction by time-contrastive learning ¨ and nonlinear ICA. In Advances in Neural Information Processing Systems, volume 29, pp. 3765–3773, 2016.

Aapo Hyvarinen and Hiroshi Morioka. Nonlinear ICA of temporally dependent stationary sources. ¨ In Artificial Intelligence and Statistics, pp. 460–469. PMLR, 2017.

Aapo Hyvarinen and Petteri Pajunen. Nonlinear independent component analysis: Existence and ¨ uniqueness results. Neural Networks, 12(3):429–439, 1999.

Aapo Hyvarinen, Hiroaki Sasaki, and Richard Turner. Nonlinear ICA using auxiliary variables and ¨ generalized contrastive learning. In The 22nd International Conference on Artificial Intelligence and Statistics, pp. 859–868. PMLR, 2019.

Ilyes Khemakhem, Diederik P. Kingma, Roberto P. Monti, and Aapo Hyvarinen. Variational autoencoders and nonlinear ICA: A unifying framework. In Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics, volume 108 of Proceedings of Machine Learning Research, pp. 2207–2217, 2020. URL [https://proceedings.mlr.press/v108/khemakhem20a.html](https://proceedings.mlr.press/v108/khemakhem20a.html).

Bohdan Kivva, Goutham Rajendran, Pradeep Ravikumar, and Bryon Aragam. Identifiability of deep generative models without auxiliary information. In Advances in Neural Information Processing Systems, volume 35, pp. 15687–15701, 2022.

Lingjing Kong, Shaoan Xie, Weiran Yao, Yujia Zheng, Guangyi Chen, Petar Stojanov, Victor Akinwande, and Kun Zhang. Partial disentanglement for domain adaptation. In International Conference on Machine Learning, 2022.

Sebastien Lachapelle, Pau Rodr ´ ´ıguez Lopez, Yash Sharma, Katie E Everett, R ´ emi Le Priol, Alexan- ´ dre Lacoste, and Simon Lacoste-Julien. Disentanglement via mechanism sparsity regularization: A new principle for nonlinear ICA. In Conference on Causal Learning and Reasoning, pp. 428–484. PMLR, 2022.

Weixin Liang, Lili Yu, Liang Luo, Srinivasan Iyer, Ning Dong, Chunting Zhou, et al. Mixtureof-transformers: A sparse and scalable architecture for multi-modal foundation models. arXiv preprint arXiv:2411.04996, 2024. URL [https://arxiv.org/abs/2411.04996](https://arxiv.org/abs/2411.04996).

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

Phillip Lippe, Sara Magliacane, Sindy Lowe, Yuki M Asano, Taco Cohen, and Stratis Gavves. ¨ CITRIS: Causal identifiability from temporal intervened sequences. In International Conference on Machine Learning, pp. 13557–13603. PMLR, 2022.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. In Advances in Neural Information Processing Systems (Datasets and Benchmarks Track), 2023a.

10

<!-- page 11 of 20 -->

Preprint

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations, 2023b.

Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly, Bernhard Scholkopf, and Olivier Bachem. Challenging common assumptions in the unsupervised learning ¨ of disentangled representations. In International Conference on Machine Learning, pp. 4114–4124. PMLR, 2019.

Soroush Nasiriany, Sepehr Nasiriany, Abhiram Maddukuri, and Yuke Zhu. RoboCasa365: A largescale simulation framework for training and benchmarking generalist robots. In International Conference on Learning Representations, 2026. URL [https://arxiv.org/abs/2603.04356](https://arxiv.org/abs/2603.04356).

NVIDIA. Cosmos 3: Omnimodal world models for Physical AI. arXiv preprint arXiv:2606.02800, 2026. URL [https://arxiv.org/abs/2606.02800](https://arxiv.org/abs/2606.02800).

Xiangchen Song, Weiran Yao, Yewen Fan, Xinshuai Dong, Guangyi Chen, Juan Carlos Niebles, Eric Xing, and Kun Zhang. Temporally disentangled representation learning under unknown nonstationarity. In Advances in Neural Information Processing Systems, volume 36, 2023. URL [https://arxiv.org/abs/2310.18615](https://arxiv.org/abs/2310.18615).

Julius von Kugelgen, Yash Sharma, Luigi Gresele, Wieland Brendel, Bernhard Sch ¨ olkopf, Michel ¨ Besserve, and Francesco Locatello. Self-supervised learning with data augmentations provably isolates content from style. In Advances in Neural Information Processing Systems, volume 34, 2021.

Dingling Yao, Danru Xu, Sebastien Lachapelle, Sara Magliacane, Perouz Taslakian, Georg Martius, ´ Julius von Kugelgen, and Francesco Locatello. Multi-view causal representation learning with ¨ partial observability. In International Conference on Learning Representations, 2024.

Weiran Yao, Guangyi Chen, and Kun Zhang. Temporally disentangled representation learning. In Advances in Neural Information Processing Systems, volume 35, 2022. URL [https://arxiv.org/abs/2210.13647](https://arxiv.org/abs/2210.13647).

Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-WAM: Do world action models need test-time future imagination? arXiv preprint arXiv:2603.16666, 2026. URL [https://arxiv.org/abs/2603.16666](https://arxiv.org/abs/2603.16666).

Yujia Zheng, Ignavier Ng, and Kun Zhang. On the identifiability of nonlinear ICA: Sparsity and beyond. In Advances in Neural Information Processing Systems, volume 35, 2022.

Yujia Zheng, Fan Feng, Yuke Li, Shaoan Xie, Kevin Murphy, and Kun Zhang. From generalist to specialist representation, 2026. URL [https://arxiv.org/abs/2605.12733](https://arxiv.org/abs/2605.12733).

11

<!-- page 12 of 20 -->

Preprint

| Symbol | Meaning |
| --- | --- |
| o ∈ R<sup>do</sup> | Observation |
| s ∈ R<sup>n</sup> | Latent state |
| a ∈ R<sup>da</sup> | Action |
| f | Observation map s 7→ o, diffeomorphism onto its image |
| <sub>f</sub>ˆ | Learnerobservationmapsˆ7→ o,diffeomorphismontoitsimage;itsinverse <sub>f</sub>ˆ<sup>-1</sup> is the encoder |
| π | Action mechanism (s,η) 7→ a in equation 1b |
| πˆ | Estimated action mechanism |
| η | Exogenous action noise (held fixed in J<sub>π</sub>, J<sub>π</sub>ˆ) |
| J<sub>m</sub>(s) | Jacobian ofa vector-valued map m with respect to s |
| I(A) | Index set offunctionally nonzero entries ofmatrix or vector A |
| ∥I(A)∥0 | Cardinality ofI(A) (support size, ℓ<sub>0</sub> norm) |
| I<sub>i</sub> ⊆ [n] | Latent indices that drive action dimension i ∈ [d<sub>a</sub>] |
| I<sub>A</sub> | S<sub>di</sub>=<sub>a</sub>1<sub>Ii</sub>; action-relevant coordinate set |
| s<sub>IA</sub> | Subvector ofs on coordinates I<sub>A</sub> |
| Iˆi, IˆA, C | Learner's index sets Iˆ<sub>i</sub> = {j : (i,j) ∈ I(J<sub>π</sub>ˆ)} and Iˆ<sub>A</sub> = S<sub>iI</sub>ˆ<sub>i</sub>; compact state C = sˆ<sub>I</sub>ˆ<sub>A</sub> (denoted C<sub>t</sub> in Section 2) |
| Observational equivalence | Same law of (o,a); the results use the pathwise coupling sˆ = ϕ(s) with shared η, which implies it (Section 2) |
| ϕ | Reparameterization ϕ := <sub>f</sub>ˆ<sup>-1</sup> ◦f, sˆ= ϕ(s) |
| P(sˆ) | J<sub>ϕ</sub>-1(sˆ); chain rule links J<sub>π</sub>ˆ and J<sub>π</sub> |
| ρ | Permutation of[n] aligning true and estimated coordinates |
| P<sub>ρ</sub> | Permutation matrix, (P<sub>ρ</sub>)j,k = 1{ρ(j) = k} |

Table 3: Summary of notation for the identifiability proofs.

## A PROOFS

**Notation.** Table 3 summarizes the symbols used in the proofs below. The main text (Sections 2 and 3) indexes samples by time t as in equation 1a–equation 1b; the proofs fix one step, drop the index, and write $o = f ( s ) ,   a = \pi ( s , \eta )$ . Identifiability does not use cross-time structure, so this is equivalent to applying the argument at each t separately.

## A.1 PROOF OF PROPOSITION 1

**Proposition 1.** Consider two observationally equivalent models $o = f ( s ) = { \hat { f } } ( { \hat { s } } ) ,   a = \pi ( s , \eta ) =$ $\hat { \pi } ( \hat { s } , \eta )$ as in Table 3. Assume that, for each action dimension $i \in [ d _ { a } ]$ , there exist $\left| I _ { i } \right|$ states $s ^ { ( 1 ) } , \ldots , s ^ { ( | I _ { i } | ) }$ at which the rows $( J _ { \pi } ( s ^ { ( l ) } ) ) _ { i , }$ · are linearly independent, and for $\phi \; = \; \hat { f } ^ { - 1 } \circ f ,$ $P \; = \; J _ { \phi ^ { - 1 } }$ , and a matrix $M$ with $\mathcal { I } ( M ) \; = \; \dot { \mathcal { I } } ( P ) , \; \mathcal { I } \big ( ( J _ { \pi } ( s ^ { ( l ) } ) M ) _ { i , \cdot } \big ) \; \subseteq \; \mathcal { I } \big ( ( J _ { \hat { \pi } } ( \hat { s } ^ { ( l ) } ) ) _ { i , \cdot } \big )$ with $\hat { s } ^ { ( l ) } = \phi ( s ^ { ( l ) } )$ , and that the regularity conditions of Section 2 hold. Then, for every action dimension $i \in [ d _ { a } ]$

$$
\| \mathcal { I } \big ( ( J _ { \hat { \pi } } ) _ { i , \cdot } \big ) \| _ { 0 } \; \geq \; \| \mathcal { I } \big ( ( J _ { \pi } ) _ { i , \cdot } \big ) \| _ { 0 } .\tag{10}
$$

Proof. Step 1 (reparameterization). The pathwise coupling in the hypothesis gives, for all $\mathcal { S } _ { \mathfrak { z } }$

$$
f ( s ) = \hat { f } ( \phi ( s ) ) , \qquad \phi : = \hat { f } ^ { - 1 } \circ f , \qquad \hat { s } = \phi ( s ) ,\tag{11}
$$

and ϕ is a diffeomorphism of $\mathbb { R } ^ { n }$ by the regularity conditions of Section 2. For fixed $\eta ,$ the same coupling gives the following identity almost surely, hence for all s by continuity and full support,

$$
\pi ( s , \eta ) = { \hat { \pi } } ( \phi ( s ) , \eta ) \qquad { \mathrm { f o r \; a l l } } \; s .\tag{12}
$$

Differentiating in s and applying the chain rule,

$$
\frac { \partial \pi } { \partial s } ( s , \eta ) = \frac { \partial \hat { \pi } } { \partial \hat { s } } ( \phi ( s ) , \eta ) \thinspace \frac { \partial \phi } { \partial s } ( s ) = J _ { \hat { \pi } } ( \phi ( s ) ) \thinspace J _ { \phi } ( s ) ,\tag{13}
$$

and $J _ { \phi } ( s ) = \left( J _ { \phi ^ { - 1 } } ( \phi ( s ) ) \right) ^ { - 1 }$ , so with $P ( \hat { s } ) : = J _ { \phi ^ { - 1 } } ( \hat { s } )$

$$
J _ { \hat { \pi } } ( \phi ( s ) ) = J _ { \pi } ( s )   P ( \phi ( s ) ) .\tag{14}
$$

12

<!-- page 13 of 20 -->

Preprint

Step 2 (nondegenerate rows). Fix $i \in [ d _ { a } ]$ . Let $s ^ { ( 1 ) } , \ldots , s ^ { ( | I _ { i } | ) }$ be as in the proposition, so the rows $( J _ { \pi } ( s ^ { ( l ) } ) ) _ { i , \cdot } , l = 1 , \ldots , | I _ { i } |$ , are linearly independent. Write $\hat { s } ^ { ( l ) } = \phi ( s ^ { ( l ) } )$ and let M be the matrix in the hypothesis, with $\dot { \mathcal { I } } ( \dot { M } ) = \mathcal { I } ( P )$ . The assumed inclusion is, for each l,

$$
\mathcal { I } \big ( ( J _ { \pi } ( s ^ { ( l ) } ) M ) _ { i , \cdot } \big ) \subseteq \mathcal { I } \big ( ( J _ { \hat { \pi } } ( \hat { s } ^ { ( l ) } ) ) _ { i , \cdot } \big ) .\tag{15}
$$

Write $S _ { i } : = \mathcal { I } \big ( ( J _ { \hat { \pi } } ) _ { i , \cdot } \big )$ for the functional row support (Section 2) and $\begin{array} { r } { S ^ { ( l ) } : = \mathcal { I } \big ( ( J _ { \hat { \pi } } ( \hat { s } ^ { ( l ) } ) ) _ { i , \cdot } \big ) } \end{array}$ . If a row vector $v \in \mathbb { R } ^ { n }$ satisfies $\mathcal { I } ( v ) \subseteq T$ , then

$$
v = \sum _ { r \in T } v _ { r } e _ { r } \in \operatorname { s p a n } \{ e _ { r } : r \in T \} .\tag{16}
$$

By equation 15, $\begin{array} { r } { \mathcal { I } \big ( ( J _ { \pi } ( s ^ { ( l ) } ) M ) _ { i , \cdot } \big ) \; \subseteq \; S ^ { ( l ) } } \end{array}$ . Each $S ^ { ( l ) }$ is contained in $S _ { i } ;$ an entry of $J _ { \hat { \pi } }$ that is nonzero at $\hat { s } ^ { ( l ) }$ is nonzero on a neighborhood of $\hat { s } ^ { ( l ) }$ by continuity, and this neighborhood has positive probability because $\hat { s } = \phi ( s )$ has a positive density on $\mathbb { R } ^ { n }$ (Appendix B.1). Applying the display above with $v = ( J _ { \pi } ( s ^ { ( l ) } ) M ) _ { i }$ ,· gives

$$
( J _ { \pi } ( s ^ { ( l ) } ) M ) _ { i , \cdot } \in \operatorname { s p a n } \{ e _ { r } : r \in S _ { i } \} , \qquad l = 1 , \ldots , | I _ { i } | .\tag{17}
$$

Step 3 (from row span to every row of M on $I _ { i } ) .$ . Fix $j \in I _ { i }$ . By definition of $I _ { i } ,$ coordinate $k   \notin   I _ { i }$ implies $( \bar { J _ { \pi } } ( s ) ) _ { i , k }   =   \stackrel { \cdot } { 0 }$ almost surely, hence everywhere by continuity and full support, and in particular $( J _ { \pi } ( s ^ { ( l ) } ) ) _ { i , k }   =   0$ for every l. Thus each row $v ^ { ( l ) }   : =   ( J _ { \pi } ( s ^ { ( l ) } ) ) _ { i , }$ · lies in $V : =$ span $\{ e _ { k } : k \in I _ { i } \}$ . The $| I _ { i } |$ rows $v ^ { ( 1 ) } , \ldots , v ^ { ( | I _ { i } | ) }$ are linearly independent and contained in the $| I _ { i } | .$ dimensional space V , so they form a basis of V . In particular $e _ { j } ^ { \top } \stackrel { \star } { \in } V$ , and there exist $\beta _ { 1 } , \ldots , \dot { \beta _ { | I _ { i } | } }$ with

$$
e _ { j } ^ { \top } = \sum _ { l = 1 } ^ { | I _ { i } | } \beta _ { l } \thinspace v ^ { ( l ) } = \sum _ { l = 1 } ^ { | I _ { i } | } \beta _ { l } \thinspace ( J _ { \pi } ( s ^ { ( l ) } ) ) _ { i , \cdot } .\tag{18}
$$

Right-multiplying by M and using equation 17,

$$
M _ { j , \cdot } = e _ { j } ^ { \top } M = \sum _ { l = 1 } ^ { | I _ { i } | } \beta _ { l } \left( J _ { \pi } ( s ^ { ( l ) } ) M \right) _ { i , \cdot } \in \operatorname { s p a n } \{ e _ { r } : r \in S _ { i } \} ,\tag{19}
$$

because each term lies in span $\{ e _ { r } : r \in S _ { i } \}$ and that set is a linear subspace. Record

$$
M _ { j , \cdot } \in \operatorname { s p a n } \{ e _ { r } : r \in S _ { i } \} .\tag{20}
$$

Step 4 (a permutation common to all action dimensions). Fix a point $\hat { s } ^ { * } \in \mathbb { R } ^ { n }$ , the same for every i. The matrix $P ( \hat { s } ^ { * } )$ is invertible, so det $P ( \hat { s } ^ { * } ) \neq 0$ . By the Leibniz formula,

$$
\operatorname* { d e t } P ( \hat { s } ^ { * } ) = \sum _ { \sigma \in \mathfrak { S } _ { n } } \operatorname { s g n } ( \sigma ) \: \prod _ { k = 1 } ^ { n } P _ { k , \sigma ( k ) } ( \hat { s } ^ { * } ) ,\tag{21}
$$

where $\mathfrak { S } _ { n }$ is the set of permutations of [n]. Some summand is nonzero; fix one such $\rho \in \mathfrak { S } _ { n }$ , chosen once for all i, so that

$$
P _ { j , \rho ( j ) } ( \hat { s } ^ { * } ) \neq 0 \qquad \mathrm { f o r \; a l l } \; j \in [ n ] .\tag{22}
$$

By continuity of $P ,$ these entries are nonzero on a neighborhood of $\hat { s } ^ { * }$ , and this neighborhood has positive probability because $\hat { s }   =   \phi ( s )$ has a positive density on $\mathbb { R } ^ { n }$ . Hence $( j , \rho ( \stackrel { \circ } { j ) } )   \in   \mathcal { I } ( P )   =$ $\mathcal { I } ( M )$ , that is,

$$
M _ { j , \rho ( j ) } \neq 0 \qquad \mathrm { f o r \; a l l } \; j \in [ n ] .\tag{23}
$$

Step 5 (conclusion). Fix $j \in I _ { i }$ . Vectors in span $\{ e _ { r } \colon r \in S _ { i } \}$ vanish off $S _ { i } ,$ so equation 20 gives $M _ { j , k } = 0$ for $k \notin S _ { i } ;$ in particular, $M _ { j , \rho ( j ) } = \widetilde { 0   \mathrm { i f } \; \rho ( j ) } \notin S _ { i }$ . Step 4 gives $M _ { j , \rho ( j ) } \neq 0$ , hence

$$
\rho ( j ) \in S _ { i } = \mathcal { I } \big ( ( J _ { \hat { \pi } } ) _ { i , \cdot } \big ) .\tag{24}
$$

This holds for every $j \in I _ { i } , \mathrm { s o } \; \rho ( I _ { i } ) \subseteq S _ { i }$ and

$$
\| \mathcal { I } ( ( J _ { \hat { \pi } } ) _ { i , \cdot } ) \| _ { 0 } = | S _ { i } | \geq | \rho ( I _ { i } ) | = | I _ { i } | = \| \mathcal { I } ( ( J _ { \pi } ) _ { i , \cdot } ) \| _ { 0 } ,\tag{25}
$$

where $| \rho ( I _ { i } ) | = | I _ { i } |$ because $\rho$ is a bijection of $[ n ]$ . Step 4 does not use $i ,$ so the same $\rho$ serves every action dimension i. □

13

<!-- page 14 of 20 -->

Preprint

## A.2 PROOF OF THEOREM 1

**Theorem 1.** Consider two observationally equivalent models as in Proposition 1 and assume support sparsity equation 3. Then there is a permutation $\rho$ of [n] such that $\mathcal { I } ( J _ { \hat { \pi } } )   =   \mathcal { I } ( J _ { \pi } ) P _ { \rho }$ and $\hat { \mathcal { I } } ( ( J _ { \hat { \pi } } ) _ { i , \cdot } ) \doteq \hat { \rho ( I _ { i } ) }$ for every $i \in [ d _ { a } ]$

Proof. Step 1 (one-sided row supports). Apply Steps 1 to 5 of the proof of Proposition 1 to each $i \in [ d _ { a } ]$ , with the same $\hat { s } ^ { * }$ and the same $\rho$ in Step 4. This gives one permutation $\rho$ of $[ n ]$ , common to all $i ,$ such that

$$
\rho ( I _ { i } ) \subseteq \mathcal { I } \big ( ( J _ { \hat { \pi } } ) _ { i , \cdot } \big ) ,\tag{26}
$$

hence

$$
\| \mathcal { I } ( ( J _ { \hat { \pi } } ) _ { i , \cdot } ) \| _ { 0 } \; \geq \; | \rho ( I _ { i } ) | = | I _ { i } | = \| \mathcal { I } ( ( J _ { \pi } ) _ { i , \cdot } ) \| _ { 0 } .\tag{27}
$$

Step 2 (global one-sided bound). Each index pair $( i , j ) \in \mathcal { I } ( J _ { \pi } )$ satisfies $j \in I _ { i }$ , so

$$
\| \mathcal { I } ( J _ { \pi } ) \| _ { 0 } = \sum _ { i = 1 } ^ { d _ { a } } \| \mathcal { I } ( ( J _ { \pi } ) _ { i , \cdot } ) \| _ { 0 } = \sum _ { i = 1 } ^ { d _ { a } } | I _ { i } | .\tag{28}
$$

Similarly,

$$
\| \mathcal { I } ( J _ { \hat { \pi } } ) \| _ { 0 } = \sum _ { i = 1 } ^ { d _ { a } } \| \mathcal { I } ( ( J _ { \hat { \pi } } ) _ { i , \cdot } ) \| _ { 0 } .\tag{29}
$$

Summing equation 27 over i gives

$$
\| \mathcal { I } ( J _ { \hat { \pi } } ) \| _ { 0 } \; \geq \; \| \mathcal { I } ( J _ { \pi } ) \| _ { 0 } .\tag{30}
$$

Step 3 (sparsity forces equalities). Support sparsity equation 3 gives

$$
\| \mathcal { I } ( J _ { \hat { \pi } } ) \| _ { 0 } \leq \| \mathcal { I } ( J _ { \pi } ) \| _ { 0 } .\tag{31}
$$

Together with equation 30,

$$
\| \mathcal { I } ( J _ { \hat { \pi } } ) \| _ { 0 } = \| \mathcal { I } ( J _ { \pi } ) \| _ { 0 } .\tag{32}
$$

If some row inequality in equation 27 were strict, summing would make equation 30 strict, a contradiction. Therefore equation 27 holds with equality for every i, and with $\tilde { \rho } ( I _ { i } ) { \subseteq } { \mathcal { I } } ( ( J _ { \hat { \pi } } ) _ { i , \cdot } )$ this implies

$$
\mathcal { I } \big ( ( J _ { \hat { \pi } } ) _ { i , \cdot } \big ) = \rho ( I _ { i } ) \qquad \mathrm { f o r \; e a c h } \; i \in [ d _ { a } ] .\tag{33}
$$

Step 4 (support-level equivalence). For every $i \in [ d _ { a } ]$ and $j \in [ n ]$ , the definition of $I _ { i } ,$ the bijectivity of $\rho ,$ and equation 33 give

$$
( i , j ) \in \mathcal { I } ( J _ { \pi } ) \iff j \in I _ { i } \iff \rho ( j ) \in \rho ( I _ { i } ) = \mathcal { I } \big ( ( J _ { \hat { \pi } } ) _ { i , \cdot } \big ) \iff ( i , \rho ( j ) ) \in \mathcal { I } ( J _ { \hat { \pi } } ) .\tag{34}
$$

Step 5 (matrix support). By definition of $P _ { \rho } , ( i , k ) \in \mathcal { I } ( J _ { \pi } ) P _ { \rho }$ if and only if $( i , \rho ^ { - 1 } ( k ) ) \in \mathcal { I } ( J _ { \pi } )$ For $k = \rho ( j )$ this is equivalent to $( i , j ) \in \mathcal { I } ( \vec { J _ { \pi } } )$ , i.e. to $( \dot { i } , \dot { \rho ( j ) } ) \in \mathcal { I } ( J _ { \hat { \pi } } )$ by equation 34. Hence

$$
\mathcal { I } ( J _ { \hat { \pi } } ) = \mathcal { I } ( J _ { \pi } ) P _ { \rho } .\tag{35}
$$

□

## A.3 PROOF OF THEOREM 2

**Theorem 2.** Under the hypotheses of Theorem 1, with the same permutation $\rho ,$ there exists an invertible map h such that $\widetilde { C } = \hat { s } _ { \rho ( I _ { \mathrm { A } } ) } = h ( s _ { I _ { \mathrm { A } } } )$ (equivalently $C _ { t } = \tilde { \hat { s } } _ { t , \rho ( I _ { \operatorname { A } } ) } = \tilde { h ( } s _ { t , I _ { \operatorname { A } } } )$ in Section 2). Proof. Assume equation 33 and fix the permutation ρ from Theorem 1.

Step 1 (a zero block of P). Fix any $j \in I _ { \mathrm { A } }$ and $r \in [ n ] \setminus I _ { \mathrm { A } }$ . Since $\begin{array} { r } { I _ { \mathrm { A } } = \bigcup _ { i } I _ { i } } \end{array}$ , there exists an action dimension $i   \in   [ d _ { a } ]$ such that $j   \in   I _ { i }$ . Equation 20 from the proof of Proposition 1, together with equation 33, gives

$$
M _ { j , \cdot } \in \operatorname { s p a n } \{ e _ { k } : k \in S _ { i } \} = \operatorname { s p a n } \{ e _ { k } : k \in \rho ( I _ { i } ) \} ,
$$

14

<!-- page 15 of 20 -->

Preprint

where $S _ { i } = \mathcal { I } ( ( J _ { \hat { \pi } } ) _ { i , \cdot } )$ . Because r $\notin I _ { \mathrm { A } }$ , we have $r \notin I _ { i }$ and hence $\rho ( r ) \notin \rho ( I _ { i } )$ . Every vector in the displayed span vanishes outside $\rho ( I _ { i } )$ , so

$$
M _ { j , \rho ( r ) } = 0 .
$$

By the assumed equality of functional supports $\mathcal { I } ( M ) = \mathcal { I } ( P )$ , this implies that $P _ { j , \rho ( r ) } = 0$ almost surely. Since $P   \stackrel { \mathrm { ~ \scriptsize ~ \cdot ~ } } { = }   J _ { \phi ^ { - 1 } }$ is continuous and ŝ has a density positive on $\mathbb { R } ^ { n }$ , the entry must vanish everywhere: otherwise, continuity would make it nonzero on a neighborhood of positive probability. $\mathrm { A s } \; j$ and r were arbitrary, we obtain

$$
P _ { j , \rho ( r ) } ( \hat { s } ) = 0 \qquad \mathrm { f o r \; a l l } \; \hat { s } \in \mathbb { R } ^ { n } , \quad j \in I _ { \mathrm { A } } , \quad r \in [ n ] \setminus I _ { \mathrm { A } } .\tag{36}
$$

Step 2 (block triangular form). Order rows of P as $( I _ { \mathrm { A } } , [ n ] \backslash I _ { \mathrm { A } } )$ and columns as $( \rho ( I _ { \mathrm { A } } ) , \rho ( [ n ] \backslash I _ { \mathrm { A } } ) )$ Then equation 36 is exactly $P _ { j , \rho ( r ) } = 0$ for $j \in I _ { \mathrm { A } }$ and $r \in [ n ] \setminus I _ { \mathrm { A } }$ , i.e. the upper-right block of P vanishes:

$$
P = \left( \begin{matrix} { P _ { 1 1 } } & { 0 } \\ { P _ { 2 1 } } & { P _ { 2 2 } } \\ \end{matrix} \right) , \qquad P _ { 1 1 } \in \mathbb { R } ^ { | I _ { \Lambda } | \times | \rho ( I _ { \Lambda } ) | } , \quad P _ { 2 2 } \in \mathbb { R } ^ { ( n - | I _ { \Lambda } | ) \times ( n - | \rho ( I _ { \Lambda } ) | ) } .\tag{37}
$$

Because P is invertible,

$$
\operatorname* { d e t } P = \operatorname* { d e t } P _ { 1 1 } \operatorname* { d e t } P _ { 2 2 } \neq 0 ,\tag{38}
$$

so $P _ { 1 1 }$ and $P _ { 2 2 }$ are invertible. For an invertible block-triangular matrix,

$$
P^{-1} = \begin{pmatrix} P_{11}^{-1} & 0 \\ -P_{22}^{-1}P_{21}P_{11}^{-1} & P_{22}^{-1} \end{pmatrix},\tag{39}
$$

which is checked by direct multiplication. Since $P ^ { - 1 }   =   J _ { \phi } ( s )$ , the upper-right block of $P ^ { - 1 }$ is zero, so

$$
\frac { \partial \hat { s } _ { \rho ( j ) } } { \partial s _ { r } } = 0 \qquad \mathrm { f o r } \; j \in I _ { \mathrm { A } } , \; r \in [ n ] \setminus I _ { \mathrm { A } } ;\tag{40}
$$

equation 36 gives

$$
\frac { \partial \hat { s } _ { j } } { \partial \hat { s } _ { \rho ( r ) } } = 0 \qquad \mathrm { f o r \; t h e \; s a m e \; i n d e x \; p a i r s } .\tag{41}
$$

Step 3 (block-wise functional dependence). Because $\phi$ and $\phi ^ { - 1 }$ are diffeomorphisms on $\mathbb { R } ^ { n }$ , if $\partial \hat { s } _ { \rho ( j ) } / \partial s _ { r }   \equiv   0$ for $j   \in   I _ { \mathrm { A } }$ and $r   \in   [ n ] \setminus I _ { \mathrm { A } }$ , then $\hat { s } _ { \rho ( j ) }$ does not depend on $s _ { r }$ as a function on $\mathbb { R } ^ { n }$ ; likewise $\partial s _ { j } / \partial \hat { s } _ { \rho ( r ) }   \equiv   0$ implies $s _ { j }$ does not depend on $\hat { s } _ { \rho ( r ) }$ . Thus there exist continuously differentiable maps h and $\tilde { h }$ such that

$$
\hat { s } _ { \rho ( I _ { \operatorname { A } } ) } = h ( s _ { I _ { \operatorname { A } } } ) , \qquad s _ { I _ { \operatorname { A } } } = \tilde { h } ( \hat { s } _ { \rho ( I _ { \operatorname { A } } ) } ) .\tag{42}
$$

On the image of $h ,   \tilde { h } \circ h$ is the identity on $s _ { I _ { \mathbf { A } } }$ and $h \circ \tilde { h }$ is the identity on $\hat { s } _ { \rho ( I _ { \mathrm { A } } ) }$ , so h is invertible with inverse $\tilde { h } .$ The Jacobian of h is the upper-left block of $P ^ { - 1 }$ , hence

$$
J _ { h } = P _ { 1 1 } ^ { - 1 } .\tag{43}
$$

Step 4 (compact state). By equation 33, $\hat { I } _ { i } = \mathcal { I } ( ( J _ { \hat { \pi } } ) _ { i , \cdot } ) = \rho ( I _ { i } )$ for every i, so $\begin{array} { r } { \hat { I } _ { \mathrm { A } } = \bigcup _ { i } \rho ( I _ { i } ) = } \end{array}$ $\rho ( I _ { \mathrm { A } } )$ because ρ is a bijection. Step 3 therefore gives

$$
C = \hat { s } _ { \hat { I } _ { \mathrm { A } } } = \hat { s } _ { \rho ( I _ { \mathrm { A } } ) } = h ( s _ { I _ { \mathrm { A } } } ) ,
$$

15

(44)

which is Theorem 2 with time subscript t restored $( C _ { t } = \hat { s } _ { t , \rho ( I _ { \mathrm { A } } ) } )$

<!-- page 16 of 20 -->

Preprint

## B SUPPLEMENTARY DISCUSSIONS

## B.1 DETAILED DISCUSSION ON MAIN CONDITIONS

The assumptions are those in Proposition 1. They require that each action dimension i has enough variation in $( J _ { \pi } ( s ) ) _ { i , }$ · across $\left| I _ { i } \right|$ states to span the true parents $I _ { i } ,$ and that reparameterization through ϕ cannot hide a true dependency in πˆ by cancellation at every such state. Both parts are mild in the usual smooth, nonparametric regime.

Span. For each action dimension i, the assumption requires $| I _ { i } |$ states at which the rows $( J _ { \pi } ( s ^ { ( l ) } ) ) _ { i } ,$ are linearly independent. Equivalently, these rows must span span $\{ e _ { j } : j \in I _ { i } \}$ . This requirement concerns variation in the mechanism’s derivatives and does not follow from smoothness or from a positive state density. For example, if $\pi _ { i } ( s )   =   s _ { 1 }   +   s _ { 2 }$ , then $I _ { i } \: = \: \{ 1 , 2 \}$ but every Jacobian row equals $( 1 , 1 , 0 , \ldots , 0 )$ , so the condition fails regardless of the state distribution. More generally, mechanisms of the form $\pi _ { i } ( s ) = g ( s _ { 1 }   +   s _ { 2 } )$ have collinear Jacobian rows. The proposition therefore assumes the required spanning states explicitly. It requires their existence, rather than asserting that $\left| I _ { i } \right|$ randomly sampled states satisfy the condition with probability one.

Support inclusion. The identity equation 14 implies that rows of $J _ { \pi } ( s ) M$ with ${ \mathcal { I } } ( M )   =   { \mathcal { I } } ( P )$ typically lie inside the corresponding rows of $J _ { \hat { \pi } } ( \hat { s } )$ . The inclusion assumed in Proposition 1 must hold at the $| I _ { i } |$ states used for the span argument; it excludes measure-zero coincidences where partial derivatives cancel on all of those states simultaneously.

Functional row support. In the proof of Proposition $\begin{array} { r } { 1 ,   S ^ { ( l ) }   : =   \mathcal { I } \big ( ( J _ { \hat { \pi } } ( \hat { s } ^ { ( l ) } ) ) _ { i , \cdot } \big ) } \end{array}$ is compared with the functional row support $S _ { i }   =   \mathcal { I } \big ( ( J _ { \hat { \pi } } ) _ { i , \cdot } \big )$ from Section 2. If $( J _ { \hat { \pi } } ( \hat { s } ^ { ( l ) } ) ) _ { i , k } \neq 0$ , then, since πˆ is continuously differentiable, $J _ { \hat { \pi } } ( \hat { s } ) _ { i , k } \neq 0$ on a neighborhood of $\hat { s } ^ { ( l ) }$ , so $k \in S _ { i }$ because $\hat { s } = \phi ( s )$ has a positive density on $\mathbb { R } ^ { n }$ (Section 2).

## C EVALUATION PROTOCOL

**Estimands.** Block recovery is assessed with a ridge probe from $C _ { t }$ to the true latent variables of the synthetic process, fitted and reported on disjoint episodes, and reported as held-out $R ^ { 2 }$ for the actionrelevant and for the nuisance variables. The same probe from the true $s _ { t , I _ { \mathrm { A } } }$ <sup>to</sup> the nuisance variables gives the oracle reference. On the robot benchmarks, routing mass is visualized in Figure 3, and on four RoboCasa365 tasks, closed-loop success with only the routed tokens is measured in Table 2.

**Synthetic generator.** The simulator implements equation 1a–equation 1b and adds a latent transition and a hidden regime that scales the action mechanism and the transition, to stress nuisance coupling; those extras are not part of the identification assumptions in Section 2. Latent $s _ { t } \in \mathbb { R } ^ { 2 0 }$ with $\bar { I } _ { \mathrm { A } }   =   \{ 1 , \ldots , 4 \}$ . A hidden regime $c _ { t }   \in   \{ 0 , 1 , 2 \}$ follows a Markov chain with Dirichlet-random transition matrix and initial law. Action $a _ { t } = M _ { c _ { t } }   s _ { t , I _ { \mathrm { A } } } + 0 . 0 5   \eta _ { t }$ , where $M _ { c } = ( 1 + 0 . 1 5 ( c - 1 ) ) M$ with Gaussian $M   \in   \mathbb { R } ^ { 8 \times 4 }$ ; transition $s _ { t + 1 } =$ tanh $\left( ( 1   +   0 . 1 5   c _ { t } ) ( W _ { s } s _ { t }   +   W _ { a } a _ { t } ) \right)   +   0 . 0 5   \epsilon _ { t + 1 }$ with a dense Gaussian $W _ { s }$ (diagonal 0.85) and $W _ { a }$ feeding the action back into $I _ { \mathrm { A } }$ only, so nuisance factors are coupled to relevant ones through $W _ { s } ;$ observation $o _ { t } = \operatorname { t a n h } ( W _ { \operatorname* { m i x } } s _ { t } ) + 0 . 0 5   \xi _ { t }$ with block-diagonal Gaussian $W _ { \mathrm { m i x } }   \in   \mathbb { R } ^ { 1 0 0 \times 2 0 ^ { \circ } }$ (five observation coordinates per latent factor, no cross-block entries). The environment suite draws $e _ { t } \in [ E ]$ i.i.d. at every step and multiplies the nuisance coordinates of $s _ { t + 1 }$ by a per-environment factor in [0.5, 1.5]; the action family is unchanged across E. The regime thus modulates behavior and transition without changing the four-factor action support. It is not supplied to the estimator, and the fitted single-frame head learns a marginalized predictor rather than an explicitly recovered conditional policy. Simulator streams are seeded so that exogenous noise is paired across E.

**Synthetic estimators and metrics.** The encoder is an MLP $1 0 0 \rightarrow 2 5 6 \rightarrow 2 0$ with tanh followed by a non-affine LayerNorm, the action head is linear $2 0 \rightarrow 8 ,$ and we use Adam with batch size 512. Development seeds 900, 901 select learning rate from $\{ 3 \cdot \dot { 1 } 0 ^ { - 4 } , 1 0 ^ { - 3 } , 3 \cdot 1 0 ^ { - 3 } \}$ and epochs from {80, 160}, with 320 epochs tried only if no candidate meets the predeclared development targets. All six candidates meet the development-mean targets; minimum selected-state action MSE selects learning rate $1 0 ^ { - 3 }$ and 160 epochs. This setting is shared by all baselines and evaluated once on

16

<!-- page 17 of 20 -->

Preprint

Table 4: Block recovery by CWAM with sparsity does not degrade as hidden environments are added. At every $E$ it decodes the action-relevant block as well as the dense readout, which alone decodes nuisance variables beyond the true block. Each hidden environment rescales the nuisance variables in the transition, with the action mechanism fixed across $E ,$ and is never an encoder input. Metrics as in Table 1, whose fits give the row $E = 1$ . Mean ± population std over five seeds.

<table><tr><td rowspan="2">E</td><td colspan="2">Dense (20D readout)</td><td colspan="2">CWAM with sparsity (4D)</td></tr><tr><td> $\operatorname { R e l e v a n t } R ^ { 2 } \uparrow$ </td><td> $\Delta _ { \mathrm { n u i s } } \downarrow$ </td><td> $\mathbf { R e l e v a n t } \; R ^ { 2 } \uparrow$ </td><td> $\Delta _ { \mathrm { n u i s } } \downarrow$ </td></tr><tr><td>1</td><td> $0 . 9 9 6 1 \pm 0 . 0 0 1 5$ </td><td> $0 . 6 4 6 \pm 0 . 1 1 5$ </td><td> $0 . 9 9 5 7 \pm 0 . 0 0 1 4$ </td><td> $0 . 0 3 9 \pm 0 . 0 9 3$ </td></tr><tr><td></td><td> $2 \quad 0 . 9 9 6 1 \pm 0 . 0 0 1 6$ </td><td> $0 . 6 4 0 \pm 0 . 1 1 0$ </td><td> $0 . 9 9 6 1 \pm 0 . 0 0 1 6$ </td><td> $\ - 0 . 0 0 4 \pm 0 . 0 0 3$ </td></tr><tr><td></td><td> $3 \quad 0 . 9 9 6 1 \pm 0 . 0 0 1 7$ </td><td> $0 . 6 7 7 \pm 0 . 1 0 9$ </td><td> $0 . 9 9 6 1 \pm 0 . 0 0 1 7$ </td><td> $\ - 0 . 0 0 4 \pm 0 . 0 0 3$ </td></tr><tr><td></td><td> $5 \quad 0 . 9 9 6 1 \pm 0 . 0 0 1 7$ </td><td> $0 . 6 8 3 \pm 0 . 1 1 4$ </td><td> $0 . 9 9 6 1 \pm 0 . 0 0 1 7$ </td><td> $\ - 0 . 0 0 3 \pm 0 . 0 0 2$ </td></tr></table>

seeds 200–204, held out from development tuning. The current action is a target, never an encoder input. Dense, random and $\lambda { = } 0$ readouts share weights, and $E = 1$ is reused across tables: 70 entries come from 50 saved estimators, all reloaded for metric replay. CWAM penalty: mean over latent coordinates of the $\ell _ { 2 }$ norm of the corresponding head column, weight 0.01. VAE (Table 1): Gaussian posterior, MLP decoder, KL weight $1 \hat { 0 } ^ { - 3 }$ , head sparsity weight $0 ;$ an optional $\mathrm { V A E } + \ell _ { 1 }$ baseline (0.05 times the mean absolute head weight) is trained in the same matrix but omitted from the table. Support: the four coordinates with the largest mean absolute head weight; random matched: four uniformly random coordinates. Probes: ridge $( \alpha = 1 )$ fitted on 250 calibration episodes, $R ^ { 2 }$ averaged over target dimensions on 250 disjoint reporting episodes. Sel./full and Compl./full apply the fitted head to $\hat { s } _ { t }$ with the complement (resp. $C _ { t } )$ set to zero on the reporting episodes.

**Synthetic splits, readouts, and the oracle reference.** Each fit uses 2,000 training episodes and 500 held-out episodes of length 128. The hidden regime $c _ { t }$ enters the action mechanism and the transition in every reported run. The dense readout applies the probes to all 20 coordinates of $\hat { s } _ { t } .$ . Random matched and CWAM without sparsity are taken from the same fit, trained with head sparsity weight $\lambda = 0 .$ , and CWAM without sparsity keeps its four coordinates with the largest mean absolute head weight. CWAM with sparsity and the standard MLP VAE are separate fits, and $C _ { t }$ is selected from the head of the VAE in the same way. Every top-4 selection uses the true size $| I _ { \mathrm { A } } | = 4$ . Because the transition couples the nuisance variables to the action-relevant ones, the true block already predicts the nuisance variables with $R ^ { 2 } > 0$ . The oracle reference $R ^ { 2 } ( s _ { t , I _ { \mathrm { A } } } \to \mathrm { n u i s } )$ measures this coupling with the same linear probe and splits, and $\Delta _ { \mathrm { n u i s } }$ subtracts it from the nuisance $R ^ { 2 }$ of $C _ { t }$ seed by seed before averaging. The reference is specific to this simulator and is computed from the true coordinates $s _ { t , I _ { \mathrm { A } } }$ . Since $C _ { t } = \hat { s } _ { t , \hat { I } _ { \mathrm { A } } }$ passes through a nonlinear encoder and need not align with $s _ { t , I _ { \mathrm { A } } }$ coordinate by coordinate, a linear probe from $C _ { t }$ can land slightly above or below the reference even when block recovery is good, so $\Delta _ { \mathrm { n u i s } }$ can be slightly negative. The environment suite uses $E \; \in \; \{ 1 , 2 , 3 , 5 \}$ , and neither $c _ { t }$ nor $e _ { t }$ is an input to the encoder. Tables report the mean and the population standard deviation over seeds, with ratios computed per seed. Theorem 2 identifies the action-relevant latent variables up to a block-wise invertible transformation, so we measure block-level recovery with linear-probe $\bar { R } ^ { 2 }$ and do not report the MCC.

## D HIDDEN ENVIRONMENTS (SYNTHETIC)

The environment index $e _ { t } \in [ E ]$ in Appendix C rescales nuisance transition coordinates while the action mechanism is fixed. Table 4 reports probes of the dense readout and of CWAM with sparsity as E grows (mean ± std over five seeds); metrics match Table 1.

## E ROBOCASA365 PER-TASK SUCCESS

The RoboCasa365 panel of Figure 2 reports the mean success within each foundational skill family over the 17 evaluation tasks. Table 5 lists the per-task success rates behind those family means, together with the macro average over all tasks.

17

<!-- page 18 of 20 -->

Preprint

Table 5: CWAM raises success on 13 of the 17 RoboCasa365 tasks and the macro average from 0.24 to 0.43. Each cell is the success rate over 100 episodes. Tasks are grouped by skill family, and each point in Figure 2(a) averages the tasks of one family. Baseline and CWAM are as in Figure 2.

<table><tr><td>Foundational skill family</td><td>Task</td><td>Baseline</td><td>CWAM</td></tr><tr><td rowspan="4">Pick and place</td><td>PickPlaceCtrToCab</td><td>.31</td><td>.43</td></tr><tr><td>PickPlaceCtrToStove</td><td>.22</td><td>.45</td></tr><tr><td>PickPlaceDrawerToCtr</td><td>.15</td><td>.07</td></tr><tr><td>PickPlaceSinkToCtr</td><td>.24</td><td>.74</td></tr><tr><td rowspan="3">Opening / closing doors</td><td>PickPlaceToasterToCtr</td><td>.28</td><td>.72</td></tr><tr><td>CloseFridge</td><td>.14</td><td>.61</td></tr><tr><td>CloseToasterOvenDoor</td><td>.51</td><td>.37</td></tr><tr><td rowspan="2">Closing / opening lids</td><td>OpenCabinet</td><td>.31</td><td>.24</td></tr><tr><td>CloseBlenderLid OpenStandMixerHead</td><td>.02 .30</td><td>.03</td></tr><tr><td rowspan="3">Pressing buttons</td><td></td><td></td><td>.69</td></tr><tr><td>TurnOnMicrowave</td><td>.27</td><td>.26</td></tr><tr><td>TurnOnElectricKettle</td><td>.35</td><td>.45</td></tr><tr><td>Opening / closing drawers</td><td>OpenDrawer</td><td>.25</td><td>.70</td></tr><tr><td>Sliding racks</td><td>SlideDishwasherRack</td><td>.32</td><td>.61</td></tr><tr><td>Twisting knobs</td><td>TurnOffStove</td><td>.05</td><td>.18</td></tr><tr><td>Turning levers</td><td>TurnOnSinkFaucet</td><td>.25</td><td>.37</td></tr><tr><td colspan="2">CoffeeSetupMug Insertion Macro (unweighted mean over 17 tasks)</td><td>.10 .24</td><td>.37 .43</td></tr></table>

**Robot evaluation.** The matched Cosmos 3 baseline and CWAM are post-trained on the same demonstrations with the same recipe and optimization, and CWAM differs only by the actionrelevance router and $\mathcal { L } _ { \mathrm { s p a r s e } }$ (Section 4), so the baseline has no router. RoboCasa365 is evaluated on 17 tasks with 100 episodes per task, LIBERO on 40 tasks in four suites with 100 episodes per task, and the Franka stack-cube task with 50 trials per policy. For the context ablation in Table 2, the CWAM checkpoint is fixed, and only the visual tokens that the action queries attend to are changed at evaluation, with no refitting and with the episode keys of the main evaluation. C-only keeps the routed top-k tokens that define $C _ { t }$ in equation 9 and zeroes the other visual context to the action head. Random matched keeps the same number of visual tokens, drawn uniformly at random from the clean context. The four ablation tasks are those with the highest CWAM success in Table 5.

## F COSMOS 3 POST-TRAINING SETUP

This appendix records the matched baseline and CWAM post-training recipe used for Section 5.2. The architectural difference is the action-relevance router and $\mathcal { L } _ { \mathrm { s p a r s e } }$ (Appendix H); optimization, data, packing, and rectified-flow losses are otherwise identical (Appendix G).

## F.1 TRAINING STAGES AND SERVED CHECKPOINT

We warm-start from the public nvidia/Cosmos3-Nano checkpoint converted to distributed checkpoint (DCP) format.

**Stage 1 (router initialization).** CWAM post-trains on RoboCasa365 for 10,000 optimizer steps with the action-relevance router enabled and freshly initialized; the rectified-flow objective, data packing, and shared hyperparameters match Table 6 and Appendix G.

**Stage 2 (continue).** Training resumes from step 10,000 and runs to 50,000 steps with a flat LR multiplier 0.3871 (the learning rate reached at the end of Stage 1; Table $6 ) ;$ optimizer scheduler state is not resumed. The matched baseline follows the same two stages and data recipe without the router module (hence no $\mathcal { L } _ { \mathrm { s p a r s e } } )$

18

<!-- page 19 of 20 -->

Preprint

Table 6: Shared Cosmos 3 RoboCasa365 post-training hyperparameters (baseline and CWAM).

| Setting | Value |
| --- | --- |
| Optimizer | FusedAdamW (β<sub>1</sub>=0.9, β<sub>2</sub>=0.99, ϵ=10<sup>-8</sup>) |
| Base learning rate | 5×10<sup>-5</sup> |
| LR multipliers | 5.0 on action2llm, llm2action, action modality embed |
| Weight decay | 0.05 |
| Gradient clipping | 1.0 (force finite) |
| Precision | bf16 |
| Parallelism | FSDP, 8 GPU shard /1 replicate |
| Grad accumulation | 2 (Stages 1 and 2) |
| Stage 1 schedule | linear warmup 500 steps, cycle 16,000, max iter=10,000 |
| Stage 2 schedule | flat multiplier 0.3871 (LR at step 10,000), cycle 90,000, max iter=50,000 |
| Checkpointing | save every 500 (Stage 1) / 10,000 (Stage 2); EMA rate 0.1 |
| Rectified-flow loss scale | 10.0 on generator (RoboCasa recipe override) |
| Action loss weight | 10.0 |
| CFG dropout (train) | 0.1 |

Training runs on one 8×H200 node (FSDP shard degree 8), bf16, selective activation checkpointing on FMHA, and precomputed Wan 2.2 video latents bundled with the RoboCasa365 training shards.

## F.2 DEMONSTRATION DATA RECIPE (ROBOCASA365)

**Corpus.** Processed LeRobot-style RoboCasa365 demonstrations from the Cosmos 3 pipeline.

**Windowing and modalities.** Policy mode at 20 Hz: chunk length=16 frames per training window; the leading frame is observed and later video tokens are noised under the rectified-flow objective. Observations use camera mode=concat view (third-person and wrist, each 256×256, concatenated to 256×512 before the model canvas). Language follows caption mode=policy with JSON-formatted prompts. Actions are frame wise relative rot6d (10D: position, rotation, gripper) with quantile rot normalization and embodiment tag robocasa365. Classifier-free guidance dropout on conditioning is 0.1 on train and 0 on val.

## F.3 CLOSED-LOOP TASK PROTOCOLS

**RoboCasa365 (17 tasks).** We evaluate the target split on the 17 tasks listed in Table 5: CloseBlenderLid,CloseFridge, CloseToasterOvenDoor, CoffeeSetupMug, OpenCabinet, OpenDrawer, OpenStandMixerHead, PickPlaceCounterToCabinet, PickPlaceCounterToStove, PickPlaceDrawerToCounter, PickPlaceSinkToCounter, PickPlaceToasterToCounter, SlideDishwasherRack, TurnOffStove, TurnOnElectricKettle, TurnOnMicrowave, TurnOnSinkFaucet. Each task: 100 **episodes**, base seed 42 (paired keys across intervention conditions in Appendix E), 400 environment steps cap, 12-step open-loop action horizon (replan after each chunk).

**LIBERO (40 tasks).** Closed-loop evaluation uses the **LIBERO-post-trained** baseline and CWAM checkpoint (Appendix F.2). All LIBERO tasks are grouped into the four standard suites (40 tasks total); per-suite success rates appear in Figure 2(b). Protocol details follow the Cosmos 3 LIBERO closed-loop stack (20 Hz, quantile rotation actions, concatenated views).

## F.4 SHARED OPTIMIZATION HYPERPARAMETERS

Table 6 lists settings shared by baseline and CWAM; router-only fields are deferred to Appendix H.

19

<!-- page 20 of 20 -->

Preprint

## G COSMOS 3 BASE MODEL AND RECTIFIED-FLOW OBJECTIVE

**Backbone.** Cosmos 3-Nano: Qwen3-VL-8B-Instruct reasoner inside a mixture-of-transformers (MoT) generator with joint attn implementation=two way, video generation enabled (vision gen=True), and action generation enabled $( \arctan - \arctan = \arctan )$ Diffusion expert loads pretrained weights; RoboCasa365 post-training reinitializes action adapters (action2llm, llm2action, action modality embed, action pos embed) from scratch while keeping the public video/reasoner stack.

**Tokenization.** Text: staged local Qwen3-VL processor (QWEN3 VL TOKENIZER PATH). Video: Wan 2.2 VAE latents (48 channels) with training-time exact encode durations {17, 61, 73} frames when an online VAE is used; production runs use precomputed latents with metadata-only validation. Max packed tokens after packing: 74,000 for this recipe (override of the Nano default 45,056).

**Objective** $\mathcal { L } _ { \mathrm { W A M } \cdot }$ . Joint rectified-flow training on noised future video tokens and the executed action chunk, conditioned on clean history and language. Training-time distributions: waver on video, logitnormal on image/action/sound branches; loss scale=10.0 on the generator path in the RoboCasa365 config; action loss weight=10.0. Inference defaults (rectified flow inference config): UniPC, 1000 train timesteps, shift 1. The baseline and CWAM share this loss; CWAM adds $\mathcal { L } _ { \mathrm { s p a r s e } }$ only on the router readout (Section 4).

## H ACTION-RELEVANCE ROUTER

**Module.** When action relevance enabled=True , a residual router sits after the generator tower: relation-aware attention over clean context indices $\mathcal { I } _ { t }$ (task + clean vision; noised future vision excluded), with routing dimension $d _ { r } { = } 5 1 2 ,   8$ heads, top-k support k=32, and residual gate initialized to 0.01. Optimizer trains action relevance with LR multiplier 5.0; router weights are skipped when loading the public Nano checkpoint.

**Sparsity schedule.** λ=0.01 on $\mathcal { L } _ { \mathrm { s p a r s e } }$ (equation 7); not applicable to the matched baseline, which has no router. Sparsity loss activates at optimizer step 1,000 and anneals linearly over 4,000 steps (matches the main-text ramp). Gradients from $\mathcal { L } _ { \mathrm { s p a r s e } }$ update the shared tower and router; the dense MoT visual→action path remains active at training and default eval time (Section 4).

**Evaluation-only readouts.** Closed-loop selected (C-only) and random matched conditions (Section 5.3, Appendix E) intervene on the fixed 50k policy without refitting; they are not alternate training configs.

20