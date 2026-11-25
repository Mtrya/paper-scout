<!-- page 1 of 25 -->

arXiv:2503.12645v2 [cs.LG] 8 Apr 2025

# UNDERSTANDING GRADIENT ORTHOGONALIZATION FOR DEEP LEARNING VIA NON-EUCLIDEAN TRUST-REGION OPTIMIZATION

A PREPRINT

**Dmitry Kovalev** Yandex Research dakovalev1@gmail.com

April 9, 2025

## ABSTRACT

Optimization with matrix gradient orthogonalization has recently demonstrated impressive results in the training of deep neural networks (Jordan et al., 2024; Liu et al., 2025). In this paper, we provide a theoretical analysis of this approach. In particular, we show that the orthogonalized gradient method can be seen as a first-order trust-region optimization method, where the trust-region is defined in terms of the matrix spectral norm. Motivated by this observation, we develop the stochastic non-Euclidean trust-region gradient method with momentum, which recovers the Muon optimizer (Jordan et al., 2024) as a special case, along with normalized SGD and signSGD with momentum (Cutkosky and Mehta, 2020; Sun et al., 2023). In addition, we prove state-of-the-art convergence results for the proposed algorithm in a range of scenarios, which involve arbitrary non-Euclidean norms, constrained and composite problems, and non-convex, star-convex, first- and second-order smooth functions. Finally, our theoretical findings provide an explanation for several practical observations, including the practical superiority of Muon compared to the Orthogonal-SGDM algorithm of Tuddenham et al. (2022) and the importance of weight decay in the training of large-scale language models.

## 1 Introduction

Over the past couple of decades, a substantial amount of optimization research has been dedicated to adaptive gradient optimization algorithms (Duchi et al., 2011; Tieleman, 2012; Kingma and Ba, 2014; Gupta et al., 2018; Reddi et al., 2019), with the primary application being the training of deep neural networks (LeCun et al., 2015). One of the most notable results of this line of research is the development of the AdamW optimizer (Loshchilov and Hutter, 2017; Zhuang et al., 2022), which has become the standard algorithm of choice for training large language models (LLMs) (Achiam et al., 2023; Liu et al., 2024; Grattafiori et al., 2024; Anil et al., 2023). However, recently, Jordan et al. (2024); Liu et al. (2025) have made significant progress in the ambitious task of surpassing AdamW in training LLMs using the idea of neural network optimization with orthogonalized gradients (Tuddenham et al., 2022; Jordan et al., 2024). In our paper, we aim to establish theoretical foundations for this promising research direction. Formally speaking, we consider the following composite optimization problem:

$$
\operatorname* { m i n } _ { x \in \mathcal { X } } \left[ F ( x ) = f ( x ) + R ( x ) \right] ,\tag{1}
$$

where X is a finite-dimensional vector space endowed with the inner product $\langle \cdot , \cdot \rangle \colon { \mathcal { X } } \times { \mathcal { X } } \to \mathbb { R } ,   f ( \cdot ) \colon { \mathcal { X } } \to$ R is a bounded from below and differentiable objective function, and $R ( \cdot ) \colon \mathcal { X } \to \dot { \mathbb { R } } \dot { \cup } \left\{ + \infty \right\}$ is a proper,<sup>1</sup>closed, and convex regularizer. Our goal is to study the convergence properties of gradient methods for solving problem (1) in the stochastic non-Euclidean smooth setting. We provide a formal description of this setting below and justify our interest in this setting in the upcoming Sections 1.1 and 1.2.

**Stochastic gradient estimator.** We assume access to a stochastic estimator $g ( \cdot   ; \xi ) \colon \mathcal { X } \to \mathcal { X }$ of the gradient $\nabla f ( \cdot )$ where $\xi \sim \bar { \mathcal { D } }$ is a random variable sampled from a probability distribution D. We assume that the stochastic gradient

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">x ∈ X</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280">R(x)</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>1</sup>Function R(x) is called proper if there exists x ∈ X such that is finite.</span></small>

<!-- page 2 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

estimator $g ( \cdot ; \xi )$ is unbiased and has bounded variance, that is, the following relations hold:

$$
\mathbb { E } _ { \xi \sim \mathcal { D } } \left[ g ( x ; \xi ) \right] = \nabla f ( x ) \quad \mathsf { a n d } \quad \mathbb { E } _ { \xi \sim \mathcal { D } } \left[ \| g ( x ; \xi ) - \nabla f ( x ) \| _ { 2 } ^ { 2 } \right] \leq \sigma ^ { 2 } \quad \mathsf { f o r \; a l l \; } x \in \mathcal { X } ,\tag{A1}
$$

where $\sigma > 0$ is a positive variance parameter, and $\left\| \cdot \right\| _ { 2 }$ is the standard Euclidean norm induced by the inner product $\langle \cdot , \cdot \rangle , \mathrm { i . e . } , \| x \| _ { 2 } = { \sqrt { \langle x , x \rangle } }$ These assumptions have been widely adopted for the analysis of many stochastic gradient optimization algorithms (Ghadimi and Lan, 2013, 2016; Cutkosky and Mehta, 2020; Sun et al., 2023; Horváth et al., 2023; Gorbunov et al., 2020).

**Non-Euclidean norm setting and Lipschitz continuous gradient.** We assume that vector space $\mathcal { X }$ is equipped with a norm $\lVert \cdot \rVert \colon \mathcal { X } \to \mathbb { R } _ { + }$ , which possibly does not coincide with the Euclidean norm $\left\| \cdot \right\| _ { 2 }$ . In addition, we assume that the gradient $\overline { { \nabla f ( \cdot ) } }$ is Lipschitz continuous with respect to the norm ∥·∥, that is, the following inequality holds:

$$
\| \nabla f ( x ) - \nabla f ( x ^ { \prime } ) \| _ { * } \leq L \| x - x ^ { \prime } \| \quad \operatorname { f o r \; a l l } \; x , x ^ { \prime } \in \mathcal { X } ,\tag{A2}
$$

where $L   >   0$ is the gradient Lipschitz constant, and $\lVert \cdot \rVert _ { * } \colon { \mathcal { X } } \; \to \; \mathbb { R } _ { + }$ is the dual norm associated with ∥·∥, i.e., $\| x \| _ { * } = \operatorname { s u p } _ { \| x ^ { \prime } \| \leq 1 } \langle x , x ^ { \prime } \rangle$ for all $x \in \mathcal { X }$ . The assumption of gradient Lipschitz continuity is also widespread in the analysis of first-order optimization methods (Ghadimi and Lan, 2013; Gower et al., 2019; Cutkosky and Mehta, 2020; Horváth et al., 2023; Gorbunov et al., 2020). It is important to highlight that while Assumption (A2) uses the dual norm $\left\| \cdot \right\|$ ∗ to measure the difference between the gradients, the variance in Assumption (A1) is measured with respect to the Euclidean norm ∥·∥22, which is necessary to properly utilize the unbiasedness property of the stochastic gradient estimator $g ( \cdot ; \xi )$ . Therefore, we need to provide a connection between these norms using the following inequality:

$$
\| x \| _ { * } \leq \rho \cdot \| x \| _ { 2 } \quad \mathrm { f o r \; a l l } \; x \in { \mathcal { X } } ,\tag{A3}
$$

where $\rho > 0$ is a positive constant. Note that such a constant always exists due to the norm equivalence theorem, which always holds in the finite-dimensional space X .

## 1.1 Main Motivation: Optimization with Orthogonalized Gradients

The main motivation for this work is the recently proposed idea of using orthogonalized gradient updates for the training of deep neural networks (Tuddenham et al., $2022;$ Jordan et al., 2024). To explain the idea, we consider the problem of minimizing a differentiable matrix function $F ( \cdot ) \colon \mathbb { R } ^ { m \times n } \to \mathbb { R }$

$$
\operatorname* { m i n } _ { \mathbf { X } \in \mathbb { R } ^ { m \times n } } F ( \mathbf { X } ) ,\tag{2}
$$

which is clearly a special instance of the main problem (1), as long as we choose $\mathcal { X }$ to be the space of $m \times n$ matrices, $\mathcal { X } = \mathbb { R } ^ { m \times n }$ , and assume zero regularization, $R \equiv 0$ . The iterations of the simplest version of the orthogonalized gradient method for solving problem (2) can be written as follows:

$$
\mathbf { X } _ { k + 1 } = \mathbf { X } _ { k } - \eta \mathbf { O } _ { k } , \quad \mathbf { O } _ { k } = \operatorname { o r t h } ( \mathbf { G } _ { k } ) , \quad \mathbf { G } _ { k } = \nabla F ( \mathbf { X } _ { k } ) .\tag{3}
$$

where $\eta > 0$ is the stepsize, and orth $\mathfrak { i } ( \cdot ) \colon \mathbb { R } ^ { m \times n }   \to   \mathbb { R } ^ { m \times n }$ is the orthogonalization function, which is defined as follows:

$$
\operatorname { o r t h } ( \mathbf { G } ) = ( \mathbf { G } \mathbf { G } ^ { \top } ) ^ { \frac { \dagger } { 2 } } \mathbf { G } \quad \operatorname { f o r \: a l l } \quad \mathbf { G } \in \mathbb { R } ^ { m \times n } ,\tag{4}
$$

where $\frac { \dagger } { 2 }$ denotes the square root of the pseudoinverse matrix. The name «orthogonalized» is attributed to the fact that in the case where the gradient $\nabla F ( \mathbf { X } _ { k } )$ is a square and full-rank matrix, the update $\mathbf { O } _ { k }$ can be equivalently expressed as $\mathbf { O } _ { k } = \mathbf { U } _ { k } \mathbf { V } _ { k } ^ { \top }$ , where $\mathbf { U } _ { k }$ and $\mathbf { V } _ { k }$ are the orthogonal matrix parts of the singular value decomposition of the gradient, $\mathrm { i . e . , } \; \nabla F ( \mathbf { X } _ { k } \hat { ) } = \mathbf { U } _ { k } \mathbf { \Sigma } _ { k } \mathbf { V } _ { k } ^ { \top }$ , where $\mathbf { \Sigma } _ { k } \in \mathbb { R } ^ { m \times n }$ is a diagonal matrix with non-negative entries.

Combining iterations (3) with momentum leads to the Muon optimizer (Jordan et al., 2024) for solving the stochastic version of problem (2). It has been empirically shown that this algorithm can considerably outperform AdamW in the training of both small-scale (Jordan et al., 2024) and large-scale (Liu et al., 2025) language models. However, to the best of our knowledge, the understanding of the effectiveness of Muon is highly limited. In addition, it is worth mentioning the Orthogonal-SGDM optimizer (Tuddenham et al., 2022), which differs from Muon only in the order in which the momentum and orthogonalization are applied. Unfortunately, Tuddenham et al. (2022) were not able to outperform a well-tuned standard SGD with momentum using their algorithm, and Jordan et al. (2024) reported that Muon performs substantially better in practice than Orthogonal-SGDM. To the best of our knowledge, an explanation for this phenomenon is missing from the literature.

2

<!-- page 3 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

## 1.2 Summary of Contributions and Related Work

Motivated by the above discussion, we set the main goal of this paper to develop the theoretical foundations for optimization with orthogonalized gradients. More specifically, we provide the key contributions listed below.<sup>2</sup>

**Gradient orthogonalization as non-Euclidean trust-region optimization.** Jordan et al. (2024); Bernstein and Newhouse (2024) found an interpretation of the orthogonalized gradient method (3) as the standard gradient descent under the non-Euclidean matrix spectral norm. However, this interpretation is highly inaccurate, as further discussed in Section 2.3. In Section 2, we develop a completely new interpretation of the gradient orthogonalization as a trust-region gradient method, where the trust-region is defined in terms of the non-Euclidean spectral norm. This interpretation is accurate because it allows us to recover the iterations (3) exactly. Moreover, using this interpretation, we obtain several meaningful theoretical results, which we describe further. We also discuss the benefits of using the matrix spectral norm in the training of neural networks in Appendix A.1.

**Stochastic non-Euclidean trust-region gradient method with momentum.** Motivated by our new interpretation of gradient orthogonalization, in Section 3, we develop the stochastic non-Euclidean trust-region gradient method with momentum (Algorithm 1) for solving problem (1). This algorithm is designed to work with an arbitrary non-Euclidean norm ∥·∥, and recovers several existing optimization algorithms: normalized SGD with momentum (Cutkosky and Mehta, 2020), signSGD with momentum (Sun et al., 2023), and Muon. We also prove the $1 / \varepsilon ^ { 4 }$ iteration complexity of Algorithm 1 for non-convex functions, which yields the first convergence result for Muon and matches the existing optimal results for SGD-type methods for non-convex functions (Ghadimi and Lan, 2013; Arjevani et al., 2023). In addition, in Appendix B, we use our results to provide a theoretical explanation of why Muon outperforms its natural contender, Orthogonal-SGDM (Tuddenham et al., 2022), in practice, and justify the subtle differences in the design of these algorithms.

**Convergence analysis for star-convex functions.** In Section 4.1, we develop the stochastic non-Euclidean trust-region gradient method with weight decay (Algorithm 2) for solving problem (1) by utilizing the popular weight decay mechanism (Loshchilov and Hutter, 2017). We obtain the improved $1 / \varepsilon ^ { 3 }$ /ε3 iteration complexity of Algorithm 2 under the star-convexity assumption (Nesterov and Polyak, 2006). In addition, in Appendix C, we prove the same $1 / \varepsilon ^ { 3 }$ complexity result for Algorithm 1 without weight decay, by additionally assuming the boundedness of dom R. Our results support the practical evidence for the importance of incorporating weight decay into Muon for the training of large-scale language models, and explain why it may be less important for the training of small-scale language models, as reported by Liu et al. (2025); Jordan et al. (2024). Another important remark is that our theoretical results for star-convex functions hold substantial practical interest because there is empirical evidence (Zhou et al., 2019; Kleinberg et al., 2018) suggesting that deep neural networks may adhere to star-convexity or its variants.

**Convergence analysis for second-order smooth functions.** Inspired by the theoretical results of Cutkosky and Mehta (2020), in Section 5, we develop the stochastic non-Euclidean trust-region gradient method with extrapolation for solving problem (1) by incorporating a certain extrapolation step into Algorithm 2. We obtain the $1 / \varepsilon ^ { 3 . 5 }$ complexity for non-convex functions under the additional second-order smoothness assumption, which matches the results of Cutkosky and Mehta (2020); Sun et al. (2023) for normalized SGD and signSGD and improves upon the standard $1 / \varepsilon ^ { 4 }$ complexity for SGD-type methods for non-convex functions (Ghadimi and Lan, 2013; Arjevani et al., 2023). In addition, we provide a convergence analysis for Algorithm 3 for star-convex functions, which also shows an improvement over the results for Algorithms 1 and 2.

## 2 Non-Euclidean Trust-Region Gradient Method

## 2.1 Gradient Descent as Majorization-Minimization

It is generally known that the standard proximal gradient descent can be seen as an instance of the majorizationminimization (MM) algorithm (Hunter and Lange, 2004). In particular, we can replace the objective function $F ( x )$ in problem (1) with the following approximation at an arbitrary point $z \in { \mathcal { X } } ;$

$$
\textstyle \mathcal { U } ( x ; z ) = f ( z ) + \langle \nabla f ( z ) , x - z \rangle + \frac { 1 } { 2 \theta } \| x - z \| ^ { 2 } + R ( x ) ,\tag{5}
$$

which is constructed by replacing function $f ( x )$ with its first-order Taylor approximation at point z and adding the squared norm regularization $\textstyle { \frac { 1 } { 2 \theta } } \| { \stackrel { \circ } { x } } - z \| ^ { 2 }$ . This approximation is accurate at point z, i.e., $\bar { F ( z ) } = \mathcal { U } ( z ; z )$ , and using standard arguments, one can show that $\mathcal { U } ( x ; z )$ majorizes $F ( x ) , \mathrm { i . e . , } \; F ( x ) \leq \mathcal { \widetilde { U } } ( x ; z )$ for all $x \in \mathcal { X }$ , as long as $\theta \geq 1 / \bar { L }$

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>2</sup>In Appendix A.2 we compare our results with the concurrent work of Pethick et al. (2025), which we found only after the initial version of our paper appeared online.</span></small>

3

<!-- page 4 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

and Assumption (A2) holds. Consequently, the iterations of the MM algorithm can be written as follows:

$$
x _ { k + 1 } = \operatornamewithlimits { a r g m i n } _ { x \in \mathcal { X } } \mathcal { U } ( x ; x _ { k } ) .\tag{6}
$$

One can immediately observe that these iterations coincide with the proximal gradient method (Nesterov, 2013) as long as we assume the Euclidean norm setting, i.e., $\lVert \cdot \rVert = \lVert \cdot \rVert _ { * } = \lVert \cdot \rVert _ { 2 } ;$

$$
x _ { k + 1 } = \operatorname { p r o x } _ { \theta R ( \cdot ) } ( x _ { k } - \theta \nabla f ( x _ { k } ) ) .\tag{7}
$$

The convergence guarantees for this algorithm were established in the Euclidean setting for both non-convex and convex functions ${ \bar { f } } ( x )$ by Nesterov (2013).

## 2.2 Non-Euclidean Trust-Region Gradient Method

The trust-region optimization approach (Conn et al., 2000; Jiang et al., 2023) is a notable alternative to the MM principle discussed above. It is usually employed when access to the second-order derivatives of the objective function is assumed. This approach involves minimizing a second-order Taylor-like approximation of the objective function at a given point within a certain neighborhood of that point, which is called the trust-region.

Further, we construct a first-order, i.e., gradient, trust-region optimization algorithm for solving problem (1). More specifically, we replace the objective function $F ( x )$ with the following approximation at an arbitrary point $z \in \mathcal { X } ;$

$$
\mathcal { A } ( x ; z ) = f ( z ) + \langle \nabla f ( z ) , x - z \rangle + R ( x ) ,\tag{8}
$$

which does not necessarily majorize function $F ( x )$ due to the lack of squared norm regularization, in contrast to $\mathcal { U } ( x ; z )$ in eq. (5). Thus, the iterations of the trust-region method minimize this approximation only within the ball of radius $\eta > 0$ defined in terms of the non-Euclidean norm $\left\| \cdot \right\|$ as follows:

$$
x _ { k + 1 } = \operatornamewithlimits { a r g m i n } _ { x \in \mathcal { X } } \mathcal { A } ( x ; x _ { k } ) \quad \operatorname { s . t . } \quad \| x - x _ { k } \| \leq \eta .\tag{9}
$$

As a particular example, it is easy to verify that in the Euclidean norm setting with zero regularization, i.e., where $\lVert \cdot \rVert \stackrel { \cdot } { = } \lVert \cdot \rVert _ { * } = \lVert \cdot \rVert _ { 2 }$ and $R \equiv 0$ , these iterations coincide with normalized gradient descent (Shor, 2012):

$$
x _ { k + 1 } = x _ { k } - \frac { \eta } { \| \nabla f ( x _ { k } ) \| _ { 2 } } \nabla f ( x _ { k } ) .\tag{10}
$$

The convergence guarantees for the iterations (10) were established in the convex case by Grimmer (2019) and in some non-convex scenarios by Nesterov (1984). However, to the best of our knowledge, convergence results for the trust-region gradient method (9) with a general non-Euclidean norm ∥·∥ and an arbitrary proper closed and convex regularizer $R ( \cdot )$ are unknown for both convex and non-convex functions $f ( x )$ . We provide such results for non-convex functions in Section 2.4 and for star-convex functions in Appendix C.

## 2.3 Gradient Orthogonalization for Matrix Function Optimization

Now, we revisit the minimization problem (2) over the space of matrices $\mathcal { X } = \mathbb { R } ^ { m \times n }$ , which is a special instance of the main problem (1) with zero regularization $R \equiv 0$ . We choose the non-Euclidean norms ∥·∥ and $\left\| \cdot \right\|$ ∗ to be the spectral $\left\| \cdot \right\| _ { \mathrm { o p } }$ and the nuclear $\left\| \cdot \right\| \stackrel { \smile } { \mathrm { n u c } }$ matrix norms, respectively. That is, for an arbitrary matrix $\left[ \hat { \mathbf { X } } \in \mathbb { R } ^ { \hat { m } \hat { \times } n } \right.$ , we have $\| \mathbf { X } \| _ { \operatorname { o p } } = \overset { n } { \| } \overset { n \cdot \cdot } { \pmb { \sigma } } ( \mathbf { X } ) \| _ { \infty }$ and $\| \mathbf { X } \| _ { \mathrm { n u c } } ^ { \hat { \mathrm { ~ n u c } } } = \| \pmb { \sigma } ( \mathbf { X } ) \| _ { 1 }$ , where $\bar { \sigma } ( \mathbf { X } )$ is the vector of singular values of X. Thus, one can verify that the non-Euclidean trust-region gradient method (9) is exactly equivalent to the orthogonalized gradient method (3). This is one of the key observations of our work. It is in sharp contrast with Bernstein and Newhouse (2024), who suggested the interpretation of the orthogonalized gradient method as the gradient descent (6) in the same non-Euclidean setting. As mentioned in Section 1.2, this interpretation is highly inaccurate because the iterations (6) reduce to the following:

$$
\mathbf { X } _ { k + 1 } = \mathbf { X } _ { k } - \theta \| \mathbf { G } _ { k } \| _ { \operatorname { n u c } } \mathbf { O } _ { k } ,\tag{11}
$$

where $\mathbf { G } _ { k } = \nabla F ( \mathbf { X } _ { k } )$ and $\mathbf { O } _ { k } \in \mathbb { R } ^ { m \times n }$ is defined in eq. (3). Indeed, there is an extra nuclear norm factor $\| \mathbf { G } _ { k } \| _ { \mathrm { n u c } }$ in the update (11). More importantly, in contrast to this interpretation of Bernstein and Newhouse (2024), we can use our trust-region approach to establish convergence guarantees for Muon.

## 2.4 Convergence Analysis for Non-Convex Functions

In this section, we provide a convergence analysis for the non-Euclidean trust-region gradient method (9) for solving the non-stochastic version of problem (1) with the non-convex objective function $f ( x )$ in the following Theorem 1 and Corollary 1. The proof is available in Appendix D.1.

4

<!-- page 5 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Stochastic Non-Euclidean Trust-Region Gradient Method with Momentum
1: input: $\overline { { x _ { 0 } , m _ { 0 } \in \mathcal { X } } }$
2: parameters: stepsize $\eta &gt; 0 ,$ momentum $\alpha \in ( 0 , 1 )$ , number of iterations $K \in \{ 1 , 2 , \ldots \}$
3: for $k = 0 , 1 , \ldots , K - 1$ do
4: Sample $\xi _ { k } \sim \mathcal { D }$
5: Compute $m _ { k + 1 }$ as follows:
$m _ { k + 1 } = ( 1 - \alpha ) m _ { k } + \alpha g ( x _ { k } ; \xi _ { k } )$ (14)
6: Compute $x _ { k + 1 }$ as follows:
$x _ { k + 1 } = \operatornamewithlimits { a r g m i n } _ { x \in \mathcal { X } } [ \langle m _ { k + 1 } , x \rangle + R ( x ) ] \quad \operatornamewithlimits { s . t . } \quad \| x - x _ { k } \| \leq \eta$ (15)
7: output: $x _ { K } \in \mathcal { X }$
</div>

**Theorem 1.** Let Assumption (A2) hold and let $x _ { 0 } \in$ dom R. Then the iterations (9) satisfy the following inequality:

$$
\operatorname* { m i n } _ { k = 1 , \ldots , K } \lVert \nabla f ( x _ { k } ) + \hat { \nabla } R _ { k } \rVert _ { * } \leq \frac { \Delta _ { 0 } } { \eta K } + \frac { 3 L \eta } { 2 } ,\tag{12}
$$

where $\begin{array} { r } { \hat { \nabla } R _ { k } \in \partial R ( x _ { k } ) ,   \Delta _ { 0 } = F ( x _ { 0 } ) - \operatorname* { i n f } _ { x } F ( x ) . } \end{array}$

**Corollary 1.** To reach the precision mi $\operatorname { h } _ { k = 1 \ldots K } \lVert \nabla f ( x _ { k } ) + \hat { \nabla } R _ { k } \rVert _ { * } \leq \varepsilon$ by the iterations (9) under the conditions of Theorem 1, it is sufficient to choose the stepsize η and the number of iterations K as follows:

$$
\eta = \mathcal { O } \left( \frac { \varepsilon } { L } \right) , \qquad K = \mathcal { O } \left( \frac { L \Delta _ { 0 } } { \varepsilon ^ { 2 } } \right) .\tag{13}
$$

We make two remarks regarding Theorem 1 and Corollary 1. First, the generalized stationarity condition $\| \nabla f ( x _ { k } ) +$ $\| \hat { \nabla } R _ { k } \| _ { * } \; \leq \; \varepsilon$ is widely adopted in the literature (Nesterov, 2013). In the case of zero regularization, $R   \equiv   0 ,$ it reduces to the standard definition of an ε-stationary point, $\| \nabla f ( x _ { k } ) \| _ { * } \leq \varepsilon .$ Second, the obtained iteration complexity $K = \mathcal { O } \left( L \Delta _ { 0 } / \varepsilon ^ { 2 } \right)$ matches the standard result for gradient descent in the Euclidean setting by Nesterov (2013) and cannot be improved in general (Carmon et al., 2020).

## 3 Stochastic Non-Euclidean Trust-Region Gradient Method with Momentum

## 3.1 The Algorithm

In this section, we present Algorithm 1 for solving problem (1) in the stochastic setting. We call this algorithm the stochastic non-Euclidean trust-region gradient method with momentum. The main idea is to replace the gradient $\nabla f ( x _ { k } )$ in the trust-region gradient method (9) with the momentum term $m _ { k + 1 }$ , which is updated according to eq. (14). While the idea is inspired by the results for normalized SGD with momentum by Cutkosky and Mehta (2020), their analysis is not applicable in the case of a non-Euclidean norm ∥·∥ and a nonzero regularizer $\dot { R } \not \equiv 0$ . We provide the convergence analysis of Algorithm 1 for non-convex functions in Section 3.2 and for star-convex functions in Appendix C.

It is also important to highlight that by choosing different norms ∥·∥, Algorithm 1 can be reduced to several important special instances: **(i)** Euclidean norm, $\lVert \cdot \rVert = \lVert \vec { \cdot } \rVert _ { 2 } - \mathbf { n o r m a l i z e d } \: \ddot { \operatorname { S G D } }$ with momentum (Cutkosky and Mehta, 2020); **(ii)** infinity-norm, $\lVert \cdot \rVert = \lVert \cdot \rVert _ { \infty } - \mathrm { s i g n } \mathrm { S G D }$ with momentum (Sun et al., 2023); **(iii)** spectral norm, ∥·∥ = ∥·∥op – Muon (Jordan et al., 2024). The latter reduction is implied by the discussion in Section 2.3.

## 3.2 Convergence Analysis for Non-Convex Functions

In this section, we provide the convergence analysis for Algorithm 1 for solving problem (1) in the stochastic non-convex case in the following Theorem 2. The proof is available in Appendix D.2.

**Theorem 2.** Let Assumptions (A1) to (A3) hold, and let $x _ { 0 } \in$ dom R and $m _ { 0 }   =   g ( x _ { 0 } , \xi _ { 0 } )$ . Then the iterations of Algorithm 1 satisfy the following inequality:

$$
\mathbb { E } \left[ \operatorname* { m i n } _ { k = 1 , \ldots , K } \lVert { \nabla f ( x _ { k } ) + \hat { \nabla } R _ { k } } \rVert _ { * } \right] \leq \frac { \Delta _ { 0 } } { \eta K } + \frac { 2 \rho \sigma } { \alpha K } + 2 \sqrt { \alpha } \rho \sigma + \frac { 7 L \eta } { 2 } + \frac { 2 L \eta } { \alpha } ,\tag{16}
$$

where $\hat { \nabla } R _ { k } \in \partial R ( x _ { k } ) , \Delta _ { 0 } = F ( x _ { 0 } ) - \operatorname* { i n f } _ { x } F ( x ) .$

5

<!-- page 6 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

**Corollary 2.** To reach the precision $\mathbb { E } \left[ { \operatorname* { m i n } } _ { k = 1 \ldots K } \| \nabla f ( x _ { k } ) + \hat { \nabla } R _ { k } \| _ { * } \right] \leq \varepsilon$ by Algorithm 1 under the conditions of Theorem 2, it is sufficient to choose the parameters of Algorithm 1 as follows:

$$
\eta = \mathcal { O } \left( \operatorname* { m i n } \left\{ \frac { \varepsilon } { L } , \frac { \varepsilon ^ { 3 } } { \rho ^ { 2 } \sigma ^ { 2 } L } \right\} \right) , \qquad \alpha = \mathcal { O } \left( \operatorname* { m i n } \left\{ 1 , \frac { \varepsilon ^ { 2 } } { \rho ^ { 2 } \sigma ^ { 2 } } \right\} \right) ,\tag{17}
$$

$$
K = \mathcal { O } \left( \operatorname* { m a x } \left\{ \frac { \rho \sigma } { \varepsilon } , \frac { \rho ^ { 3 } \sigma ^ { 3 } } { \varepsilon ^ { 3 } } , \frac { L \Delta _ { 0 } } { \varepsilon ^ { 2 } } , \frac { L \Delta _ { 0 } \rho ^ { 2 } \sigma ^ { 2 } } { \varepsilon ^ { 4 } } \right\} \right) .\tag{18}
$$

Using Theorem 2, we obtain the explicit complexity result for Algorithm 1 in Corollary 2. Similarly to the convergence result in Corollary 1 for the deterministic non-Euclidean trust-region gradient method (9), Corollary 2 establishes the convergence result for Algorithm 1 in terms of the generalized expected stationarity. The iteration complexity (18) is proportional to $1 / \varepsilon ^ { 4 }$ . It matches the existing state-of-the-art results for SGD-type methods (Ghadimi and Lan, 2013; Cutkosky and Mehta, 2020; Sun et al., 2023) and cannot be improved (Arjevani et al., 2023) under Assumptions (A1) to (A3).

Another important remark is that Algorithm 1 with the spectral norm $\lVert \cdot \rVert = \lVert \cdot \rVert _ { \mathrm { o p } }$ and zero regularizer $R \equiv 0$ exactly matches Muon. As previously discussed, there is another variant of the orthogonalized gradient method with momentum, namely, Orthogonal-SGDM. While Muon demonstrated strong results in the training of small-scale language models (Jordan et al., 2024), it was reported by Tuddenham et al. (2022); Jordan et al. (2024) that Orthogonal-SGDM performs worse than Muon or a well-tuned SGD with momentum. Our theoretical results may provide a possible explanation for this practical difference. We discuss this in Appendix B.

## 4 Algorithms with Weight Decay for Star-Convex Functions

## 4.1 Non-Euclidean Trust-Region Gradient Methods with Weight Decay

In this section, we develop non-Euclidean trust-region gradient methods for solving problem (1) in the case where the objective function f(x) is star-convex, that is, the following inequality holds:

$$
f ( \beta x ^ { * } + ( 1 - \beta ) x ) \leq \beta f ( x ^ { * } ) + ( 1 - \beta ) f ( x ) \quad \mathrm { f o r \; a l l \; } x \in \mathcal { X } ,\tag{A4}
$$

where $x ^ { * } \in \mathcal { X }$ is a solution to problem (1). We start with the non-stochastic version of the problem. It turns out that under assumption (A4), it is possible to obtain improved convergence guarantees for the iterations (9) as long as these iterations are bounded. Unfortunately, proving the boundedness of these iterations requires additional assumptions, such as a bounded dom R, or bounded sublevel sets of the objective function $F ( x )$ , which often do not hold. We tackle this issue by shifting the center of the trust-region in eq. (9) by a factor of $( 1 - \beta )$ towards the zero. This idea leads to the non-Euclidean trust-region gradient method with weight decay (19).

$$
x _ { k + 1 } = \operatornamewithlimits { a r g m i n } _ { x \in \mathcal { X } } \mathcal { A } ( x ; x _ { k } ) \quad \operatorname { s . t . } \quad \| x - ( 1 - \beta ) x _ { k } \| \leq \eta .\tag{19}
$$

It is not hard to verify that the iterations (19) are bounded. On the other hand, the original and shifted trust-regions do not differ much. In particular, it is possible to show that they have a non-zero intersection. Hence, the shift modification will not hurt the convergence properties of the iterations (19), which we prove in Section 4.2.

Furthermore, we apply a similar modification to Algorithm 1. This modification leads to Algorithm 2, which we call the stochastic non-Euclidean trust-region gradient method with weight decay. Similar to the discussion in Section 3.1, one can show that by choosing different norms ∥·∥, Algorithm 2 reduces to the normalized SGD, signSGD, and Muon with momentum and weight decay.

## 4.2 Convergence Analysis for Star-Convex Functions

In this section, we provide the convergence analysis for the iterations (19) and Algorithm 2 for solving problem (1) in the star-convex case. We start with the result in Theorem 3 for the deterministic iterations (19). The proof is available in Appendix D.3. Next, we obtain the convergence result for the stochastic Algorithm 2 in Theorem 4. The proof is available in Appendix D.4.

**Theorem 3.** Let Assumptions (A2) and (A4) hold, and let $x _ { 0 } \in$ dom R. Let the parameters η and β satisfy the following inequality:

$$
\eta \geq \beta \operatorname* { m a x } \left\{ \| x _ { 0 } \| , \| x ^ { * } \| \right\} .\tag{22}
$$

Then, the iterations (19) satisfy the following inequality:

$$
F ( x _ { K } ) - F ( x ^ { * } ) \leq ( 1 - \beta ) ^ { K } ( F ( x _ { 0 } ) - F ( x ^ { * } ) ) + \frac { 4 L \eta ^ { 2 } } { \beta } .\tag{23}
$$

6

<!-- page 7 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2 Stochastic Non-Euclidean Trust-Region Gradient Method with Weight Decay
1: input: $\boxed { x _ { 0 } , m _ { 0 } \in \mathcal { X } }$
2: parameters: $\left[ \eta &gt; 0 , \alpha \in ( 0 , 1 ) , K \in \{ 1 , 2 , \ldots \} \right.$ , weight decay $\beta \in ( 0 , 1 )$
3: for $k = 0 , 1 , \ldots , K - 1$ do
4: Sample $\xi _ { k } \sim \mathcal { D }$
5: Compute $m _ { k + 1 }$ as follows:
$m _ { k + 1 } = ( 1 - \alpha ) m _ { k } + \alpha g ( x _ { k } ; \xi _ { k } )$ (20)
6: Compute $x _ { k + 1 }$ as follows:
$x _ { k + 1 } = \operatornamewithlimits { a r g m i n } _ { x \in \mathcal { X } } [ \langle m _ { k + 1 } , x \rangle + R ( x ) ] \quad \mathrm { s . t . } \quad \| x - ( 1 - \beta ) x _ { k } \| \leq \eta$ (21)
7: output: $x _ { K } \in \mathcal { X }$
</div>

**Theorem 4.** Let Assumptions (A1) to (A4) hold, and let $x _ { 0 } \in$ dom R and $m _ { 0 } = g ( x _ { 0 } , \xi _ { 0 } )$ . Let the parameters η and $\beta$ satisfy eq. (22). Then, the output of Algorithm 2 satisfies the following inequality:

$$
\mathbb { E } \left[ F ( x _ { K } ) - F ( x ^ { * } ) \right] \leq ( 1 - \beta ) ^ { K } ( F ( x _ { 0 } ) - F ( x ^ { * } ) ) + 2 \eta \rho \sigma \left( \frac { 1 } { \alpha } + \frac { \sqrt { \alpha } } { \beta } \right) + \frac { 4 L \eta ^ { 2 } } { \beta } \left( 1 + \frac { 1 } { \alpha } \right) .\tag{24}
$$

Finally, using Theorems 3 and 4, it is not hard to obtain the explicit iteration complexities for the iterations (19) and Algorithm 2 in Corollaries 3 and 4, respectively. The proofs are omitted due to their simplicity.

**Corollary 3.** Under the conditions of Theorem 3, let $\| x _ { 0 } \| \leq \| x ^ { * } \| = D$ . To reach the precision $F ( x _ { K } ) - F ( x ^ { * } ) \leq \varepsilon$ by the iterations (19), it is sufficient to choose the parameters η and $\beta$ and the number of iterations K as follows:

$$
\eta = \beta D , \qquad \beta = \mathcal { O } \left( \operatorname* { m i n } \left\{ 1 , \frac { \varepsilon } { L D } \right\} \right) , \qquad K = \tilde { \mathcal { O } } \left( \operatorname* { m a x } \left\{ 1 , \frac { L D ^ { 2 } } { \varepsilon } \right\} \right) .\tag{25}
$$

**Corollary 4.** Under the conditions of Theorem 4, let $\|x_{0}\| \leq \|x^{*}\| = D$ . To reach the precision E $\mathfrak { L } \left[ F ( x _ { K } ) - F ( x ^ { * } ) \right] \leq$ ε by Algorithm 2, it is sufficient to choose the parameters as follows:

$$
\alpha = \mathcal { O } \left( \operatorname* { m i n } \left\{ 1 , \frac { \varepsilon ^ { 2 } } { D ^ { 2 } \rho ^ { 2 } \sigma ^ { 2 } } \right\} \right) , \quad \beta = \mathcal { O } \left( \operatorname* { m i n } \left\{ 1 , \frac { \varepsilon } { L D ^ { 2 } } , \frac { \varepsilon } { D \rho \sigma } , \frac { \varepsilon ^ { 3 } } { D ^ { 3 } \rho ^ { 3 } \sigma ^ { 3 } } , \frac { \varepsilon ^ { 3 } } { L D ^ { 3 } \rho ^ { 2 } \sigma ^ { 2 } } \right\} \right) ,\tag{26}
$$

$$
K = \tilde { \mathcal { O } } \left( \operatorname* { m a x } \left\{ 1 , \frac { L D ^ { 2 } } { \varepsilon } , \frac { D \rho \sigma } { \varepsilon } , \frac { D ^ { 3 } \rho ^ { 3 } \sigma ^ { 3 } } { \varepsilon ^ { 3 } } , \frac { L D ^ { 3 } \rho ^ { 2 } \sigma ^ { 2 } } { \varepsilon ^ { 3 } } \right\} \right) .\tag{27}
$$

Both Corollaries 3 and 4 recover the standard iteration complexity $\mathcal { O } ( L D ^ { 2 } / \varepsilon )$ of gradient descent (Nesterov et al., 2018) for solving deterministic convex minimization problems, up to logarithmic factors. This complexity cannot be improved in the general non-Euclidean norm setting (Guzmán and Nemirovski, 2015). In the stochastic setting, Corollary 4 implies complexity proportional to $1 / \varepsilon ^ { 3 }$ . This improves upon the $1 / \varepsilon ^ { 3 . 5 }$ result of Cutkosky and Mehta (2020); Sun et al. (2023), which was obtained only for normalized SGD and signSGD with momentum, and for non-convex second-order smooth functions.<sup>3</sup>

Another important remark is that Algorithm 2 with the spectral norm $\lVert \cdot \rVert = \lVert \cdot \rVert _ { \mathrm { o p } }$ and zero regularizer $R \equiv 0$ exactly matches Muon with weight decay, as previously discussed in Section 4.1. It was empirically shown by Liu et al. (2025) that incorporating weight decay in Muon is crucial for outperforming AdamW in the training of large-scale language models. In particular, they reported that without weight decay, the iterations of Muon grow too large and the performance gains of Muon over AdamW diminish. Our theory suggests that combining Muon with weight decay may be the right solution for both issues. Indeed, we prove that the iterations of Algorithm 2 are bounded and obtain the improved complexity $1 / \varepsilon ^ { 3 }$ of Algorithm 2 in Corollary 4.

## 5 Algorithms with Extrapolation for Second-Order Smooth Functions

## 5.1 Stochastic Non-Euclidean Trust-Region Gradient Method with Extrapolation

In this section, we modify Algorithms 1 and 2 to solve problem (1) in the stochastic setting in the case where the objective function $f ( x )$ has an H-Lipschitz Hessian with respect to the non-Euclidean norm ∥·∥, which implies the

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>3</sup>Additional details on the correctness of this comparison are provided in Appendix A.3.</span></small>

7

<!-- page 8 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 3 Stochastic Non-Euclidean Trust-Region Gradient Method with Extrapolation
1: input: $x _ { 0 } = \overline { { x } } _ { 0 } , m _ { 0 } \in \mathcal { X }$
2: parameters: $\widetilde { \eta &gt; 0 , } \: \alpha \in ( 0 , 1 ) , \: K \in \{ 1 , 2 , \ldots \} , \: \beta \in ( 0 , 1 )$ , extrapolation $\gamma &gt; 0$
3: for $k = 0 , 1 , \ldots , K - 1$ do
4: Sample $\xi _ { k } \sim \mathcal { D }$
5: Compute $m _ { k + 1 }$ as follows:
$m _ { k + 1 } = ( 1 - \alpha ) m _ { k } + \alpha g ( \overline { { x } } _ { k } ; \xi _ { k } )$ (28)
6: Compute $x _ { k + 1 }$ as follows:
$x _ { k + 1 } = \operatornamewithlimits { a r g m i n } _ { x \in \mathcal { X } } [ \langle m _ { k + 1 } , x \rangle + R ( x ) ] \quad \operatornamewithlimits { s . t . } \quad \| x - ( 1 - \beta ) x _ { k } \| \leq \eta$ (29)
x∈X
7: Compute $\overline { { x } } _ { k + 1 }$ as follows:
$\overline { { x } } _ { k + 1 } = x ^ { k } + \gamma ( x _ { k + 1 } - x _ { k } )$ (30)
8: output: $x _ { K } \in \mathcal { X }$
</div>

following inequality:

$$
\| ( \nabla ^ { 2 } f ( x ) - \nabla ^ { 2 } f ( x ^ { \prime } ) ) ( x - x ^ { \prime } ) \| _ { * } \leq H \| x - x ^ { \prime } \| ^ { 2 } \quad \mathrm { f o r \; a l l } \; x , x ^ { \prime } \in \mathcal { X } ,\tag{A5}
$$

where $H > 0$ is the Hessian Lipschitz constant. The idea for the modification is inspired by Cutkosky and Mehta (2020). In particular, we replace the stochastic gradient $g ( x _ { k } , \xi _ { k } )$ in the update of the momentum term $m _ { k + 1 }$ in eqs. (14) and (20) with the stochastic gradient $g ( \overline { { x } } _ { k } ; \xi _ { k } )$ computed at a different point $\overline { { x } } _ { k } ,$ which is updated using the extrapolation step (30). This modification leads to Algorithm $^ { 3 , }$ which we call the stochastic non-Euclidean trust-region gradient method with extrapolation. While we borrow the idea for the analysis of the extrapolation step (30) from Cutkosky and Mehta (2020); Sun et al. (2023), their analysis is limited to normalized SGD and signSGD for non-convex functions with zero regularization $R \equiv 0$ . In contrast, in Section 5.2, we provide a convergence analysis for Algorithm 3 with an arbitrary non-Euclidean norm ∥·∥ and regularizer $R ( \cdot )$ in the non-convex and star-convex cases.

Note that, similarly to Algorithm 2 described in Section 4.1, Algorithm 3 makes use of weight decay in eq. (29) along with the extrapolation step (30) when dealing with star-convex functions. Another important remark is that Algorithm 3 with the matrix spectral norm $\left\| \cdot \right\| _ { \mathrm { o p } }$ turns into a new variant of Muon with extrapolation. We discuss this further in Section 5.2.

## 5.2 Convergence Analysis

We provide the convergence analysis for Algorithm 3 for solving problem (1) with the non-convex objective function in Theorem 5 and with the star-convex objective function in Theorem 6. The proofs of Theorems 5 and 6 are available in Appendices D.5 and D.6, respectively.

**Theorem 5.** Let Assumptions (A1) to (A3) and (A5) hold, let $x _ { 0 } \in$ dom R and $m _ { 0 } = g ( x _ { 0 } , \xi _ { 0 } )$ , and let $\beta = 0$ and $\gamma = 1 / \alpha .$ . Then the iterations of Algorithm 3 satisfy the following inequality:

$$
\mathbb { E } \left[ \operatorname* { m i n } _ { k = 1 , \ldots , K } \lVert { \nabla f ( x _ { k } ) + \hat { \nabla } R _ { k } } \rVert _ { * } \right] \leq \frac { \Delta _ { 0 } } { \eta K } + \frac { 7 L \eta } { 2 } + \frac { H \eta ^ { 2 } } { \alpha ^ { 2 } } + \frac { 2 \rho \sigma } { \alpha K } + 2 \sqrt { \alpha } \rho \sigma ,\tag{31}
$$

where $\hat { \nabla } R _ { k } \in \partial R ( x _ { k } ) , \Delta _ { 0 } = F ( x _ { 0 } ) - \operatorname* { i n f } _ { x } F ( x ) .$

**Theorem 6.** Let Assumptions (A1) to (A5) hold, and let $x _ { 0 } \in$ dom R and $m _ { 0 } = g ( x _ { 0 } , \xi _ { 0 } )$ . Let the parameters η and $\beta$ satisfy eq. (22), and let $\gamma = 1 / \alpha$ . Then the output of Algorithm 3 satisfies

$$
\mathbb { E } \left[ F ( x _ { K } ) - F ( x ^ { * } ) \right] \leq ( 1 - \beta ) ^ { K } ( F ( x _ { 0 } ) - F ( x ^ { * } ) ) + 2 \eta \rho \sigma \left( \frac { 1 } { \alpha } + \frac { \sqrt { \alpha } } { \beta } \right) + \frac { 4 L \eta ^ { 2 } } { \beta } + \frac { 4 H \eta ^ { 3 } } { \alpha ^ { 2 } \beta } .\tag{32}
$$

Using Theorems 5 and $^ { 6 , }$ it is not hard to obtain the explicit iteration complexities Algorithm 3 in Corollaries 3 and 4 for non-convex and star-convex functions, respectively. The proofs are omitted due to their simplicity.

**Corollary 5.** To reach the precision E $\left[ { \operatorname* { m i n } } _ { k = 1 \ldots K } \| \nabla f ( x _ { k } ) + \hat { \nabla } R _ { k } \| _ { * } \right] \leq \varepsilon \: b y$ Algorithm 3 under the conditions of Theorem 5, it is sufficient to choose the parameters as follows:

$$
\eta = \mathcal { O } \left( \operatorname* { m i n } \left\{ \frac { \varepsilon } { L } , \frac { \varepsilon ^ { 1 / 2 } } { H ^ { 1 / 2 } } , \frac { \varepsilon ^ { 5 / 2 } } { \rho ^ { 2 } \sigma ^ { 2 } H ^ { 1 / 2 } } \right\} \right) , \quad \alpha = \mathcal { O } \left( \operatorname* { m i n } \left\{ 1 , \frac { \varepsilon ^ { 2 } } { \rho ^ { 2 } \sigma ^ { 2 } } \right\} \right) , \quad \beta = 0 ,\tag{33}
$$

8

<!-- page 9 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

$$
K = \tilde { \mathcal { O } } \left( \operatorname* { m a x } \left\{ \frac { \rho \sigma } { \varepsilon } , \frac { \rho ^ { 3 } \sigma ^ { 3 } } { \varepsilon ^ { 3 } } , \frac { L \Delta _ { 0 } } { \varepsilon ^ { 2 } } , \frac { H ^ { 1 / 2 } \Delta _ { 0 } } { \varepsilon ^ { 3 / 2 } } , \frac { H ^ { 1 / 2 } \Delta _ { 0 } \rho ^ { 2 } \sigma ^ { 2 } } { \varepsilon ^ { 7 / 2 } } \right\} \right) .\tag{34}
$$

**Corollary 6.** Under the conditions of Theorem $^ { 6 , }$ let $\|x_{0}\| \leq \|x^{*}\| = D$ . To reach the precision $\mathbb { E } \left[ F ( x _ { K } ) - F ( x ^ { * } ) \right] \leq$ ε by Algorithm 3, it is sufficient to choose the parameters as follows:

$$
\beta = \mathcal { O } \left( \operatorname* { m i n } \left\{ 1 , \frac { \varepsilon } { L D ^ { 2 } } , \frac { \alpha \varepsilon } { D \rho \sigma } , \frac { \alpha \varepsilon ^ { 1 / 2 } } { H ^ { 1 / 2 } D ^ { 3 / 2 } } \right\} \right) , \quad \alpha = \mathcal { O } \left( \operatorname* { m i n } \left\{ 1 , \frac { \varepsilon ^ { 2 } } { D ^ { 2 } \rho ^ { 2 } \sigma ^ { 2 } } \right\} \right) ,\tag{35}
$$

$$
K = \tilde { \mathcal { O } } \left( \operatorname* { m a x } \left\{ 1 , \frac { D \rho \sigma } { \varepsilon } , \frac { D ^ { 3 } \rho ^ { 3 } \sigma ^ { 3 } } { \varepsilon ^ { 3 } } , \frac { L D ^ { 2 } } { \varepsilon } , \frac { H ^ { 1 / 2 } D ^ { 3 / 2 } } { \varepsilon ^ { 1 / 2 } } , \frac { H ^ { 1 / 2 } D ^ { 7 / 2 } \rho ^ { 2 } \sigma ^ { 2 } } { \varepsilon ^ { 5 / 2 } } \right\} \right) , \quad \eta = \beta D .\tag{36}
$$

Corollary 5 establishes an improved $1 / \varepsilon ^ { 3 . 5 }$ iteration complexity compared to the $1 / \varepsilon ^ { 4 }$ complexity of Algorithm 1 in Corollary 2, matching the results of Cutkosky and Mehta (2020); Sun et al. (2023) for normalized SGD and signSGD for non-convex second-order smooth functions.

Unfortunately, Algorithm 3 cannot achieve a major improvement over Algorithm 2 in the case of star-convex functions, since both have the term $( \rho \sigma D / \varepsilon ) ^ { 3 }$ in their complexities in Corollaries 4 and 6, respectively. However, the practical success of Muon and the discussion in appendix A.1 may indicate that the “geometric” terms, which depend on the Lipschitz constants L and H, play a more important role compared to the purely “stochastic” terms, which depend only on the variance σ. Hence, the improvement in the "mixed" term $\sqrt { \rho ^ { 4 } \sigma ^ { 4 } H D ^ { 7 } / \varepsilon ^ { 5 } }$ in Corollary 6 over the corresponding term $\rho ^ { 2 } \sigma ^ { 2 } L D ^ { 3 } / \varepsilon ^ { 3 }$ in Corollary 4 may be significant. Overall, it is an important open theoretical and practical question whether the extrapolation step (30) can improve the convergence of Muon.

9

<!-- page 10 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

## References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. (2023). Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Anil, R., Borgeaud, S., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., Millican, K., et al. (2023). Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Arjevani, Y., Carmon, Y., Duchi, J. C., Foster, D. J., Srebro, N., and Woodworth, B. (2023). Lower bounds for non-convex stochastic optimization. Mathematical Programming, 199(1):165–214.

Arjovsky, M., Chintala, S., and Bottou, L. (2017). Wasserstein generative adversarial networks. In International conference on machine learning, pages 214–223. PMLR.

Bernstein, J. and Newhouse, L. (2024). Old optimizer, new norm: An anthology. arXiv preprint arXiv:2409.20325.

Bernstein, J., Zhao, J., Meister, M., Liu, M.-Y., Anandkumar, A., and Yue, Y. (2020). Learning compositional functions via multiplicative weight updates. Advances in neural information processing systems, 33:13319–13330.

Carlson, D., Cevher, V., and Carin, L. (2015a). Stochastic spectral descent for restricted boltzmann machines. In Artificial Intelligence and Statistics, pages 111–119. PMLR.

Carlson, D. E., Collins, E., Hsieh, Y.-P., Carin, L., and Cevher, V. (2015b). Preconditioned spectral descent for deep learning. Advances in neural information processing systems, 28.

Carmon, Y., Duchi, J. C., Hinder, O., and Sidford, A. (2020). Lower bounds for finding stationary points i. Mathematical Programming, 184(1):71–120.

Conn, A. R., Gould, N. I., and Toint, P. L. (2000). Trust region methods. SIAM.

Cutkosky, A. and Mehta, H. (2020). Momentum improves normalized sgd. In International conference on machine learning, pages 2260–2268. PMLR.

Duchi, J., Hazan, E., and Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research, 12(7).

Elsayed, M., Lan, Q., Lyle, C., and Mahmood, A. R. (2024). Weight clipping for deep continual and reinforcement learning. arXiv preprint arXiv:2407.01704.

Ghadimi, S. and Lan, G. (2013). Stochastic first-and zeroth-order methods for nonconvex stochastic programming. SIAM journal on optimization, 23(4):2341–2368.

Ghadimi, S. and Lan, G. (2016). Accelerated gradient methods for nonconvex nonlinear and stochastic programming. Mathematical Programming, 156(1):59–99.

Gorbunov, E., Kovalev, D., Makarenko, D., and Richtárik, P. (2020). Linearly converging error compensated sgd. Advances in Neural Information Processing Systems, 33:20889–20900.

Gower, R. M., Loizou, N., Qian, X., Sailanbayev, A., Shulgin, E., and Richtárik, P. (2019). Sgd: General analysis and improved rates. In International conference on machine learning, pages 5200–5209. PMLR.

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al. (2024). The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Grimmer, B. (2019). Convergence rates for deterministic and stochastic subgradient methods without lipschitz continuity. SIAM Journal on Optimization, 29(2):1350–1365.

Gupta, V., Koren, T., and Singer, Y. (2018). Shampoo: Preconditioned stochastic tensor optimization. In International Conference on Machine Learning, pages 1842–1850. PMLR.

Guzmán, C. and Nemirovski, A. (2015). On lower complexity bounds for large-scale smooth convex optimization. Journal of Complexity, 31(1):1–14.

Horváth, S., Kovalev, D., Mishchenko, K., Richtárik, P., and Stich, S. (2023). Stochastic distributed learning with gradient quantization and double-variance reduction. Optimization Methods and Software, 38(1):91–106.

Hunter, D. R. and Lange, K. (2004). A tutorial on mm algorithms. The American Statistician, 58(1):30–37.

Jiang, Y., He, C., Zhang, C., Ge, D., Jiang, B., and Ye, Y. (2023). Beyond nonconvexity: A universal trust-region method with new analyses. arXiv e-prints, pages arXiv–2311.

Jordan, K., Jin, Y., Boza, V., You, J., Cesista, F., Newhouse, L., and Bernstein, J. (2024). Muon: An optimizer for hidden layers in neural networks.

Kingma, D. P. and Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

10

<!-- page 11 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

Kleinberg, B., Li, Y., and Yuan, Y. (2018). An alternative view: When does sgd escape local minima? In International conference on machine learning, pages 2698–2707. PMLR.

LeCun, Y., Bengio, Y., and Hinton, G. (2015). Deep learning. nature, 521(7553):436–444.

Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., et al. (2024). Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437.

Liu, J., Su, J., Yao, X., Jiang, Z., Lai, G., Du, Y., Qin, Y., Xu, W., Lu, E., Yan, J., et al. (2025). Muon is scalable for llm training. arXiv preprint arXiv:2502.16982.

Loshchilov, I. and Hutter, F. (2017). Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

Mokhtari, A., Hassani, H., and Karbasi, A. (2020). Stochastic conditional gradient methods: From convex minimization to submodular maximization. Journal of machine learning research, 21(105):1–49.

Nesterov, Y. (2013). Gradient methods for minimizing composite functions. Mathematical programming, 140(1):125–161.

Nesterov, Y. et al. (2018). Lectures on convex optimization, volume 137. Springer.

Nesterov, Y. and Polyak, B. T. (2006). Cubic regularization of newton method and its global performance. Mathematical programming, 108(1):177–205.

Nesterov, Y. E. (1984). Minimization methods for nonsmooth convex and quasiconvex functions. Matekon, 29(3):519–531.

Pethick, T., Xie, W., Antonakopoulos, K., Zhu, Z., Silveti-Falls, A., and Cevher, V. (2025). Training deep learning models with norm-constrained lmos. arXiv preprint arXiv:2502.07529.

Reddi, S. J., Kale, S., and Kumar, S. (2019). On the convergence of adam and beyond. arXiv preprint arXiv:1904.09237.

Rockafellar, R. T. (1997). Convex analysis, volume 28. Princeton university press.

Shor, N. Z. (2012). Minimization methods for non-differentiable functions, volume 3. Springer Science & Business Media.

Sun, T., Wang, Q., Li, D., and Wang, B. (2023). Momentum ensures convergence of signsgd under weaker assumptions. In International Conference on Machine Learning, pages 33077–33099. PMLR.

Tieleman, T. (2012). Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURS-ERA: Neural networks for machine learning, 4(2):26.

Tuddenham, M., Prügel-Bennett, A., and Hare, J. (2022). Orthogonalising gradients to speed up neural network optimisation. arXiv preprint arXiv:2202.07052.

Zhou, Y., Yang, J., Zhang, H., Liang, Y., and Tarokh, V. (2019). Sgd converges to global minimum in deep learning via star-convex path. arXiv preprint arXiv:1901.00451.

Zhuang, Z., Liu, M., Cutkosky, A., and Orabona, F. (2022). Understanding adamw through proximal methods and scale-freeness. arXiv preprint arXiv:2202.00089.

11

<!-- page 12 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

## A Additional Discussions

## A.1 Motivition for the Non-Euclidean Smoothness Assumptions

In this section, we provide a short discussion on the benefits of using the matrix spectral norm, $\lVert \cdot \rVert = \lVert \cdot \rVert _ { \mathrm { o p } } ,$ in the training of neural networks. In particular, we consider the following simple example of problem (2), which motivates the smoothness Assumptions (A2) and (A5):

$$
\operatorname* { m i n } _ { \mathbf { X } \in \mathbb { R } ^ { m \times n } } \left[ F ( \mathbf { X } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathcal { L } _ { i } ( \mathbf { X } a _ { i } ) \right] ,\tag{37}
$$

where $a _ { i } \in \mathbb { R } ^ { n }$ are some vectors, and $\mathcal { L } _ { i } ( \cdot ) \colon \mathbb { R } ^ { m } \to \mathbb { R }$ are loss functions. Our example can be seen as a generalization of the example in Proposition 6 by Bernstein and Newhouse (2024) and is also inspired by some earlier works of Carlson et al. (2015b,a). This example holds interest in the context of deep learning due to the matrix-vector multiplication term $\mathbf { X } a _ { i }$ , because similar structures appear in hidden layers of neural networks.

We show that Assumptions (A2) and (A5) are implied by the standard first- and second-order smoothness assumptions of functions $\mathcal { L } _ { i } ( \cdot )$ with respect to the standard Euclidean norm $\left\| \cdot \right\| _ { 2 }$ in the following Lemmas 1 and $^ { 2 , }$ respectively. The proofs are available in Appendices $\mathrm { A . 4 }$ and $\mathrm { A . } 5$ . The main observation here is that the smoothness constants L and H depend on the Euclidean norm $\| \mathbf { X } a _ { i } \| _ { 2 }$ , which can be naturally upper-bounded using the spectral norm as $\| \mathbf { X } a _ { i } \| _ { 2 } \leq \| \mathbf { X } \| _ { \operatorname { o p } } \| a _ { i } \| _ { 2 }$ . Note that a similar upper bound can be obtained using the standard Euclidean, aka Frobenius, norm of matrix $\bar { \mathbf { X } } ,$ which, however, would be much less tight. Also, note that besides the matrix spectral norm $\left\| \cdot \right\| _ { \mathrm { o p } } ,$ it is also possible to obtain similar upper bounds using more general operator norms. A more detailed study of this question is provided by Pethick et al. (2025).

**Lemma 1.** Let each function $\mathcal { L } _ { i } ( \cdot )$ in problem (37) be continuously differentiable and have a λ-Lipschitz gradient with respect to the standard Euclidean norm $\left\| \cdot \right\| _ { 2 }$ . Then function $F ( \cdot )$ satisfies Assumption (A2) with the matrix spectral norm, $\lVert \cdot \rVert = \lVert \cdot \rVert _ { \mathrm { o p } } ,$ and the gradient Lipschitz constant

$$
L = \lambda \cdot \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \lVert a _ { i } \rVert _ { 2 } ^ { 2 } .\tag{38}
$$

**Lemma 2.** Let each function $\mathcal { L } _ { i } ( \cdot )$ in problem (37) be twice continuously differentiable and have a λ-Lipschitz Hessian with respect to the standard Euclidean norm $\left\| \cdot \right\| _ { 2 }$ . Then function $F ( \cdot )$ satisfies Assumption (A5) with the matrix spectral norm, $\lVert \bar { \cdot } \rVert = \lVert \cdot \rVert _ { \mathrm { o p } } ,$ and the Hessian Lipschitz constant

$$
H = \lambda \cdot \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \lVert a _ { i } \rVert _ { 2 } ^ { 3 } .\tag{39}
$$

## A.2 Comparison with Pethick et al. (2025)

In this section, we provide a brief comparison of our results with the concurrent work of Pethick et al. (2025). We discovered this work only after the first version of our paper appeared online. In contrast to our trust-region approach, Pethick et al. (2025) build their analysis on the stochastic conditional gradient method of Mokhtari et al. (2020). Both these approaches are closely related, and we leave further investigation of this connection for future work. Besides, they provide analysis for arbitrary non-Euclidean norms, just as we do. However, their theoretical results have the following disadvantages:

**(i)** The analysis of Pethick et al. (2025) is limited to the case where the regularizer $R ( \cdot )$ is the indicator function of some closed and convex set. In contrast, our analysis works with an arbitrary proper, closed, and convex regularizer.

**(ii)** Pethick et al. (2025) obtain the same $1 / \varepsilon ^ { 4 }$ complexity for Muon, both with and without weight decay. In particular, they do not obtain the improved $1 / \varepsilon ^ { 3 }$ complexity under the star-convexity assumption. Besides, their theoretical results provide a limited explanation for the success of Muon with weight decay in the training of large-scale language models (Liu et al., 2025), and for the success of Muon without weight decay in the training of small-scale language models (Jordan et al., 2024). In contrast, we discuss both cases in Section 4.2 and appendix C.

**(iii)** In the case of non-convex functions, Pethick et al. (2025) establish only the complexity $1 / \varepsilon ^ { 4 }$ , which is standard for non-convex functions. In contrast, we obtain the improved $1 / \varepsilon ^ { 3 . 5 }$ complexity by utilizing an additional extrapolation step and the second-order smoothness assumption.

12

<!-- page 13 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

## A.3 Complexities for Non-Convex and Star-Convex Functions

In this paper, we often compare the complexities of finding an ε-accurate solution to problem (1) with non-convex and star-convex objective functions $f ( x )$ . It is important to highlight that we use the approximate first-order stationarity $\| \nabla f ( x _ { k } ) + \hat { \nabla } R _ { k } \| _ { * } \leq \varepsilon$ for non-convex functions, and the approximate functional suboptimality $F ( x _ { k } ) - F ( x ^ { * } ) \leq \varepsilon$ For instance, in Section 3.2, comparing the $1 / \varepsilon ^ { 3 }$ complexity result in Corollary 4 for star-convex functions with the $1 / \varepsilon ^ { 3 . 5 }$ result of Cutkosky and Mehta (2020); Sun et al. (2023) for non-convex functions may seem incorrect.

However, by slightly modifying the proof of Theorem 8, we can obtain the same improved complexity $1 / \varepsilon ^ { 3 }$ for finding an ε-stationary point under the star-convexity Assumption $( \mathbf { A } 4 )$ and bounded domain Assumption $( \mathrm { A 6 } )$ . This can be correctly compared with the results for non-convex functions, such as Corollaries 2 and 5. On the other hand, under these assumptions, we can upper-bound the functional suboptimality as follows:

$$
F ( x _ { k } ) - F ( x ^ { * } ) \leq D \| \nabla f ( x _ { k } ) + \hat { \nabla } R _ { k } \| _ { * } .\tag{40}
$$

Hence, the generalized $( \varepsilon / D )$ -stationarity $\| \nabla f ( x _ { k } ) + \hat { \nabla } R _ { k } \|$ ∗ implies the ε-approximate functional suboptimality as well. Note that, strictly speaking, these considerations are not valid for the case of zero regularization, $R \equiv 0$ . We leave further investigation for future work.

## A.4 Proof of Lemma 1

The inequality in Assumption (A2) is implied by the following inequality, which we further prove:

$$
\nabla F ( \mathbf { X } _ { 1 } ) [ \mathbf { H } ] - \nabla F ( \mathbf { X } _ { 2 } ) [ \mathbf { H } ] \leq L \| \mathbf { X } _ { 1 } - \mathbf { X } _ { 2 } \| _ { \operatorname { o p } } \| \mathbf { H } \| _ { \operatorname { o p } } ,\tag{41}
$$

where $\mathbf { X } _ { 1 } , \mathbf { H } \in \mathbb { R } ^ { m \times n }$ . We proceed as follows:

$$
\begin{aligned} { \nabla F ( \mathbf { X } _ { 1 } ) } & { { } [ \mathbf { H } ] - \nabla F ( \mathbf { X } _ { 2 } ) [ \mathbf { H } ] } \\ { } & { { } \overset { ( a ) } { \leq } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \nabla \mathcal { L } _ { i } ( \mathbf { X } _ { 1 } a _ { i } ) [ \mathbf { H } a _ { i } ] - \nabla \mathcal { L } _ { i } ( \mathbf { X } _ { 2 } a _ { i } ) [ \mathbf { H } a _ { i } ] \right) } \\ { } & { { } \overset { ( b ) } { \leq } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \lambda \| \mathbf { X } _ { 1 } a _ { i } - \mathbf { X } _ { 2 } a _ { i } \| _ { 2 } \| \mathbf { H } a _ { i } \| _ { 2 } } \\ { } & { { } \overset { ( c ) } { \leq } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \lambda \| \mathbf { X } _ { 1 } - \mathbf { X } _ { 2 } \| _ { \operatorname { o p } } \| \mathbf { H } \| _ { \operatorname { o p } } \| a _ { i } \| _ { 2 } ^ { 2 } } \\ { } & { { } \overset { ( d ) } { = } L \| \mathbf { X } _ { 1 } - \mathbf { X } _ { 2 } \| _ { \operatorname { o p } } \| \mathbf { H } \| _ { \operatorname { o p } } , } \\ \end{aligned}
$$

where (a) uses the linearity of the first differential; (b) uses the λ-Lipschitzness of the gradients $\nabla \mathcal { L } _ { i }$ with respect to the standard Euclidean norm $\left[ \left\| \cdot \right\| _ { 2 } ; \left( \mathbf { c } \right) \right.$ uses the properties of the matrix spectral norm $\| \cdot \| _ { \mathrm { o p } } ^ { - } ; ( \mathrm { d } )$ uses the definition of L in eq. (38). □

## A.5 Proof of Lemma 2

The inequality in Assumption (A5) is implied by the following inequality, which we further prove:

$$
\nabla ^ { 2 } F ( \mathbf { X } _ { 1 } ) [ \mathbf { X } _ { 1 } - \mathbf { X } _ { 2 } , \mathbf { H } ] - \nabla ^ { 2 } F ( \mathbf { X } _ { 2 } ) [ \mathbf { X } _ { 1 } - \mathbf { X } _ { 2 } , \mathbf { H } ] \leq H \| \mathbf { X } _ { 1 } - \mathbf { X } _ { 2 } \| _ { \operatorname { o p } } ^ { 2 } \| \mathbf { H } \| _ { \operatorname { o p } } ,\tag{42}
$$

where $\mathbf { X } _ { 1 } , \mathbf { X } _ { 2 } , \mathbf { H } \in \mathbb { R } ^ { m \times n }$ . We proceed as follows:

$$
\begin{aligned} { } & { { } \nabla ^ { 2 } F ( \mathbf { X } _ { 1 } ) [ \mathbf { X } _ { 1 } - \mathbf { X } _ { 2 } , \mathbf { H } ] - \nabla ^ { 2 } F ( \mathbf { X } _ { 2 } ) [ \mathbf { X } _ { 1 } - \mathbf { X } _ { 2 } , \mathbf { H } ] } \\ { } & { { } \qquad \overset { ( a ) } { \leq } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \nabla ^ { 2 } \mathcal { L } _ { i } ( \mathbf { X } _ { 1 } a _ { i } ) [ ( \mathbf { X } _ { 1 } - \mathbf { X } _ { 2 } ) a _ { i } , \mathbf { H } a _ { i } ] - \nabla ^ { 2 } \mathcal { L } _ { i } ( \mathbf { X } _ { 2 } a _ { i } ) [ ( \mathbf { X } _ { 1 } - \mathbf { X } _ { 2 } ) a _ { i } , \mathbf { H } a _ { i } ] \right) } \\ { } & { { } \qquad \overset { ( b ) } { \leq } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \lambda \| \mathbf { X } _ { 1 } a _ { i } - \mathbf { X } _ { 2 } a _ { i } \| _ { 2 } \| ( \mathbf { X } _ { 1 } - \mathbf { X } _ { 2 } ) a _ { i } \| _ { 2 } \| \mathbf { H } a _ { i } \| _ { 2 } } \\ { } & { { } \qquad \overset { ( c ) } { \leq } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \lambda \| \mathbf { X } _ { 1 } - \mathbf { X } _ { 2 } \| _ { \operatorname { o p } } ^ { 2 } \| \mathbf { H } \| _ { \operatorname { o p } } \| a _ { i } \| _ { 2 } ^ { 3 } } \\ \end{aligned}
$$

13

<!-- page 14 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

$$
\stackrel { \mathrm { ( d ) } } { = } H \| \mathbf { X } _ { 1 } - \mathbf { X } _ { 2 } \| _ { \mathrm { o p } } ^ { 2 } \| \mathbf { H } \| _ { \mathrm { o p } } ,
$$

where (a) uses the multilinearity of the second differential; (b) uses the λ-Lipschitzness of the Hessians $\nabla ^ { 2 } \mathcal { L } _ { i }$ with respect to the standard Euclidean norm $\lVert \cdot \rVert _ { 2 } ; ( \mathbf { c } )$ uses the properties of the matrix spectral norm $\lVert \cdot \rVert _ { \mathrm { o p } } ; ( \mathbf { d } )$ uses the definition of H in eq. (39). □

## B Muon vs Orthogonal-SGDM

As previously discussed in Section 3.1, one can verify that in the case of the non-Euclidean spectral norm $\lVert \cdot \rVert = \lVert \cdot \rVert _ { \mathrm { o p } }$ and zero regularizer $R \equiv 0 ,$ , Algorithm 1 exactly matches the Muon optimizer as it was originally described by Jordan et al. (2024). As mentioned in Section 1.1, there is another variant of the orthogonalized gradient method with momentum proposed by Tuddenham et al. (2022), namely, Orthogonal-SGDM. We provide pseudocode for both algorithms in Algorithm 4. While Muon can achieve impressive practical results (Jordan et al., 2024; Liu et al., 2025), Orthogonal-SGDM is unable to outperform Muon or well-tuned standard SGD with momentum (Tuddenham et al., 2022; Jordan et al., 2024). In this section, we provide a possible explanation for this phenomenon using our theoretical insights from Section 3.2.

<div class="docvortex-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 4 Muon/Orthogonal-SGDM (Jordan et al., 2024; Tuddenham et al., 2022)
1: input: $\mathbf { X } _ { 0 } , \mathbf { M } _ { 0 } \in \mathbb { R } ^ { m \times n }$
2: parameters: $\stackrel { \circ } { \eta } &gt; 0 , \alpha \in ( 0 , 1 ) , K \in \{ 1 , 2 , \ldots \}$
3: for $k = 0 , 1 , \dot { \ldots } , K - 1$ do
4: 5: Compute $\mathbf { G } _ { k } = \nabla F ( \mathbf { X } _ { k } )$
Compute $\mathbf { X } _ { k + 1 }$ and $\mathbf { M } _ { k + 1 }$ as follows:
$\mathbf { X } _ { k + 1 } = \mathbf { X } _ { k } - \eta \mathbf { O } _ { k + 1 } , \mathbf { M } _ { k + 1 } = ( 1 - \alpha ) \mathbf { M } _ { k } + \alpha \mathbf { G } _ { k } , \quad \mathbf { O } _ { k + 1 } = \operatorname { o r t h } ( \mathbf { M } _ { k + 1 } )$ (Muon)
$\mathbf { X } _ { k + 1 } = \mathbf { X } _ { k } - \eta \mathbf { M } _ { k + 1 } { , } \mathbf { M } _ { k + 1 } = ( 1 - \alpha ) \mathbf { M } _ { k } + \alpha \mathbf { O } _ { k + 1 } { , } \mathbf { O } _ { k + 1 } = \operatorname { o r t h } ( \mathbf { G } _ { k } )$ (OSGDM)
6: output: $\mathbf { X } _ { K } \in \mathbb { R } ^ { m \times n }$
</div>

The main difference between the updates (Muon) and (OSGDM) is the order in which the momentum and orthogonalization are applied. Our theory, in particular Lemmas 4 and 5 in Appendix D.2, suggests that the update rule (Muon) is preferable:

**(i)** First, by analyzing the proof of Lemma 5, one can conclude that the momentum accumulates a weighted sum of the terms $\zeta _ { k } = \nabla f ( x _ { k } ) - g ( x _ { k } , \xi _ { k } )$ . On the other hand, from Assumption (A1), it follows that the variance of $\zeta _ { k }$ is bounded. What is even more important, this assumption implies that the random variables $\zeta _ { k }$ are independent and centered, $\mathrm { i . e . } , \mathbb { E } \left[ \zeta _ { k } \right] = 0$ , which means that the total variance of their weighted sum is reduced due to the so-called batching effect. Refer to the proof of Lemma 5 in Appendix D.2.2 for details.

**(ii)** Second, recall our trust-region interpretation of the orthogonalized gradient descent. According to the discussion in Section 3.1, it is natural to feed the momentum term $m _ { k + 1 }$ to the trust-region optimization step (15) with the non-Euclidean spectral norm, which results in the «orthogonalized momentum» $\mathbf { O } _ { k + 1 }$ in eq. (Muon) rather than the «momentum of orthogonals» $\mathbf { M } _ { k + 1 }$ in eq. (OSGDM). These considerations are supported by the proof of Lemma 4 in Appendix D.2.1.

In contrast, the considerations above are not applicable to the update rule (OSGDM). Indeed, there is no reason to believe that the terms $\mathbf { O } _ { k + 1 }$ in eq. (OSGDM) are unbiased in any sense or that their accumulated variance can be reduced. Additionally, it is unclear what the possible interpretation of the momentum term $\mathbf { M } _ { k + 1 }$ , constructed from the orthogonalized parts $\mathbf { O } _ { k + 1 }$ in eq. (OSGDM), could be.

14

<!-- page 15 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

## C Algorithms with Weight Clipping for Star-Convex Functions

As mentioned in Section 4.1, it is possible to obtain improved convergence guarantees for the iterations (9) under the star-convexity Assumption (A4), provided that these iterations are bounded. Recall that the iterations (19) and Algorithm 2 use weight decay to achieve this. An alternative approach is to simply assume the boundedness of dom R, which implies the following inequality:

$$
\| x - x ^ { \prime } \| \leq D \quad { \mathrm { f o r \; a l l } } \; x , x ^ { \prime } \in \operatorname { d o m } R ,\tag{A6}
$$

where $D > 0$ is the diameter of dom R. In this section, we provide convergence guarantees for the iterations (9) and Algorithm 1 for star-convex functions under the additional Assumption (A6).

We start with the result in Theorem 7 for the deterministic iterations (9). The proof is available in Appendix D.7. Next, we obtain the convergence result for the stochastic Algorithm 1 in Theorem 8. The proof is available in Appendix D.8. Finally, using Theorems 7 and 8, it is not hard to obtain the explicit iteration complexities for the iterations (9) and Algorithm 1 in Corollaries 7 and 8, respectively. The proofs are omitted due to their simplicity.

**Theorem 7.** Let Assumptions (A2), (A4) and (A6) hold, and let $x _ { 0 } \in$ dom R. Then the iterations (9) satisfy the following inequality:

$$
F ( x _ { K } ) - F ( x ^ { * } ) \leq \left( \frac { D } { \eta + D } \right) ^ { K } \left( F ( x _ { 0 } ) - F ( x ^ { * } ) \right) + \frac { 3 L D \eta } { 2 } .\tag{43}
$$

**Theorem 8.** Let Assumptions (A1) to (A4) and (A6) hold, and let $x _ { 0 } \in$ dom R and $m _ { 0 } = g ( x _ { 0 } , \xi _ { 0 } )$ . Then the iterations of Algorithm 1 satisfy the following inequality:

$$
\mathbb { E } \left[ F ( x _ { K } ) - F ( x ^ { * } ) \right] \leq \left( \frac { D } { \eta + D } \right) ^ { K } \left( F ( x _ { 0 } ) - F ( x ^ { * } ) \right) + 2 \sqrt { \alpha } D \rho \sigma + \frac { 2 \eta \rho \sigma } { \alpha } + \frac { 3 L D \eta } { 2 } + \frac { 2 L D \eta } { \alpha } .\tag{44}
$$

**Corollary 7.** Under the conditions of Theorem 7, to reach the precision $F ( x _ { K } ) - F ( x ^ { * } ) \leq \varepsilon   b y$ the iterations (9), it is sufficient to choose the stepsize η and the number of iterations K as follows:

$$
\eta = \mathcal { O } \left( \frac { \varepsilon } { L D } \right) , \qquad K = \tilde { \mathcal { O } } \left( \operatorname* { m a x } \left\{ 1 , \frac { L D ^ { 2 } } { \varepsilon } \right\} \right) .\tag{45}
$$

**Corollary 8.** Under the conditions of Theorem 8, to reach the precision E $[ F ( x _ { K } ) - F ( x ^ { * } ) ] \leq \varepsilon$ by Algorithm 1, it is sufficient to choose the parameters as follows:

$$
\eta = \mathcal { O } \left( \operatorname* { m i n } \left\{ \frac { \varepsilon } { L D } , \frac { \varepsilon } { \rho \sigma } , \frac { \varepsilon ^ { 3 } } { D ^ { 2 } \rho ^ { 3 } \sigma ^ { 3 } } , \frac { \varepsilon ^ { 3 } } { L D ^ { 3 } \rho ^ { 2 } \sigma ^ { 2 } } \right\} \right) , \qquad \alpha = \mathcal { O } \left( \operatorname* { m i n } \left\{ 1 , \frac { \varepsilon ^ { 2 } } { D ^ { 2 } \rho ^ { 2 } \sigma ^ { 2 } } \right\} \right) ,\tag{46}
$$

$$
K = \tilde { \mathcal { O } } \left( \operatorname* { m a x } \left\{ 1 , \frac { L D ^ { 2 } } { \varepsilon } , \frac { D \rho \sigma } { \varepsilon } , \frac { D ^ { 3 } \rho ^ { 3 } \sigma ^ { 3 } } { \varepsilon ^ { 3 } } , \frac { L D ^ { 4 } \rho ^ { 2 } \sigma ^ { 2 } } { \varepsilon ^ { 3 } } \right\} \right) .\tag{47}
$$

The complexity of the iterations (9) without weight decay in Corollary 7 matches the complexity of the iterations (6) with weight decay in Corollary 3. Similarly, the complexity of Algorithm 1 without weight decay in Corollary 8 matches the result for Algorithm 2 with weight decay in Corollary 4. It is important to highlight that Assumption (A6) obviously does not hold in the case of zero regularization $R \equiv 0$ . To tackle this issue, one could choose the regularizer R(·) as the following indicator function:

$$
R ( x ) = \left\{ \begin{aligned} { } & { { } 0 } & { \| x \| \leq D } \\ { } & { { } { + \infty } } & { \| x \| > D } \\ \end{aligned} \right. , \quad \mathbf { w h e r e } \; D \geq \| x ^ { * } \| .\tag{48}
$$

In the case of the infinity norm $\lVert \cdot \rVert = \lVert \cdot \rVert _ { \infty } ,$ Algorithm 1 with the regularizer (48) can be easily implemented and reduces to signSGD with momentum and weight clipping. Weight clipping is used in various machine learning applications (Elsayed et al., 2024; Bernstein et al., 2020; Arjovsky et al., 2017) and can be seen as an alternative to weight decay used in Algorithm 2.

Unfortunately, in the case of the spectral norm $\lVert \cdot \rVert = \lVert \cdot \rVert _ { \mathrm { o p } }$ , Algorithm 1 with the regularizer (48), such a clipping procedure is not easily implementable, which justifies the use of weight decay in Muon. On the other hand, Liu et al. (2025) reported that in cases where language models are small-scale and/or the training time is not lengthy, the iterations of Muon do not grow excessively large. Hence, with a reasonable choice of the radius D, the weight clipping step in Algorithm 1 would be skipped. Thus, the fact that we obtain the same complexity results for Algorithm 1 with weight clipping and Algorithm 2 weight decay may support the practical evidence that weight decay is not necessary for the training of small-scale language models (Liu et al., 2025; Jordan et al., 2024).

15

<!-- page 16 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

## D Proofs of Theorems

## D.1 Proof of Theorem 1

We start with the following Lemma 3, which allows us to incorporate an arbitrary proper, closed, and convex regularizer $R ( x )$ into the non-Euclidean trust-region optimization step. The proof is available in Appendix D.1.1.

**Lemma 3.** Let $z _ { + } \in \mathcal { X }$ be defined as a solution to the following trust-region optimization problem:

$$
z _ { + } = \operatornamewithlimits { a r g m i n } _ { x \in \mathcal { X } } \Psi ( x ) + R ( x ) \quad { s . t . } \quad \| x - z \| \leq \eta ,\tag{49}
$$

where $\Psi ( \cdot ) \colon \mathcal { X } \to \mathbb { R }$ is a differentiable function, $z \in$ dom R is a center of the trust-region, and $\eta > 0$ is the trust-region radius. Then the following inequality holds:

$$
R ( z ) + \langle \nabla \Psi ( z _ { + } ) , z - z _ { + } \rangle \geq R ( z _ { + } ) + \eta \| \nabla \Psi ( z _ { + } ) + \hat { \nabla } R \| _ { * } ,\tag{50}
$$

where $\hat { \nabla } R \in \partial R ( z _ { + } )$

Now, we are ready to prove Theorem 1. We can upper-bound $F ( x _ { k + 1 } )$ as follows:

$$
\begin{aligned} { F ( x _ { k + 1 } ) } & { { } \overset { ( \alpha ) } { \otimes } f ( x _ { k + 1 } ) + R ( x _ { k + 1 } ) } \\ { } & { { } \overset { ( \alpha ) } { \otimes } f ( x _ { k } ) + \langle \nabla f ( x _ { k } ) , x _ { k + 1 } - x _ { k } \rangle + \textstyle \frac { 1 } { 2 } L \| x _ { k + 1 } - x _ { k } \| ^ { 2 } + R ( x _ { k + 1 } ) } \\ { } & { { } \overset { ( \alpha ) } { \otimes } f ( x _ { k } ) + R ( x _ { k } ) - \eta \| \nabla f ( x _ { k } ) + \hat { \nabla } R _ { k + 1 } \| _ { * } + \textstyle \frac { 1 } { 2 } L \| x _ { k + 1 } - x _ { k } \| ^ { 2 } } \\ { } & { { } \overset { ( \alpha ) } { = } f ( x _ { k } ) + R ( x _ { k } ) - \eta \| \nabla f ( x _ { k + 1 } ) + \hat { \nabla } R _ { k + 1 } \| _ { * } + \eta \| \nabla f ( x _ { k + 1 } ) - \nabla f ( x _ { k } ) \| _ { * } } \\ { } & { { } + \textstyle \frac { 1 } { 2 } L \| x _ { k + 1 } - x _ { k } \| ^ { 2 } } \\ { } & { { } \overset { ( \alpha ) } { \otimes } f ( x _ { k } ) + R ( x _ { k } ) - \eta \| \nabla f ( x _ { k + 1 } ) + \hat { \nabla } R _ { k + 1 } \| _ { * } + L \eta \| x _ { k + 1 } - x _ { k } \| + \textstyle \frac { 1 } { 2 } L \| x _ { k + 1 } - x _ { k } \| ^ { 2 } } \\ { } & { { } \overset { ( \alpha ) } { \otimes } f ( x _ { k } ) + R ( x _ { k } ) - \eta \| \nabla f ( x _ { k + 1 } ) + \hat { \nabla } R _ { k + 1 } \| _ { * } + \textstyle \frac { 3 } { 2 } L \eta ^ { 2 } } \\ { } & { { } \overset { ( \alpha ) } { = } F ( x _ { k } ) - \eta \| \nabla f ( x _ { k + 1 } ) + \hat { \nabla } R _ { k + 1 } \| _ { * } + \textstyle \frac { 3 } { 2 } L \eta ^ { 2 } , } \\ \end{aligned}
$$

where (a) and (g) use the definition of $F ( x )$ in problem (1); (b) and (e) use Assumption $( \mathrm { A } 2 ) ; ( \mathrm { c } )$ uses Lemma 3 with $\Psi ( x ) = \langle \nabla f ( \tilde { x _ { k } } ) , x \rangle$ and $( z , z _ { + } ) = ( x _ { k } , \widetilde { x _ { k + 1 } } ) ;$ ; (d) uses the triangle inequality; (f) uses the fact that $\left\| x _ { k + 1 } - x _ { k } \right\| \leq \eta ,$ which is implied by $\mathbf { e q . } \: ( 9 )$ □

## D.1.1 Proof of Lemma 3

The objective function in problem (49) is lower semi-continuous, because function $R ( x )$ is proper, closed and convex, and the constrained set in this problem is obviously compact. Hence, there exists at least a single solution $z ^ { + } \in \mathcal { X }$ to the problem.

Let function $I _ { \eta } ( x )$ be the indicator function of the constraint set in problem (49), i.e.,

$$
I _ { \eta } ( x ) = \left\{ \begin{aligned} & 0 & \| x - z \| \leq \eta \\ & + \infty & \| x - z \| > \eta \end{aligned} \right. ,\tag{51}
$$

and let function $R _ { \eta } ( x )$ be defined as follows:

$$
R _ { \eta } ( x ) = R ( x ) + I _ { \eta } ( x ) .\tag{52}
$$

It is easy to verify that function $R _ { \eta } ( x )$ is proper, closed, and convex, and problem (49) is equivalent to the following:

$$
\operatorname* { m i n } _ { x \in \mathcal { X } } \Psi ( x ) + R _ { \eta } ( x ) .\tag{53}
$$

Therefore, $z _ { + }$ satisfies the following inequality for all $x \in { \mathcal { X } } ;$

$$
\Psi ( x ) + R _ { \eta } ( x ) \geq \Psi ( z _ { + } ) + R _ { \eta } ( z _ { + } ) .\tag{54}
$$

Our next goal is to show that $- \nabla \Psi ( z _ { + } ) \in \partial R _ { \eta } ( z _ { + } )$ , which we prove by contradiction. Suppose that the opposite holds, $\operatorname { i . e . , } - \nabla \Psi ( z _ { + } ) \notin \partial R _ { \eta } ( z _ { + } )$ . This implies the existence of $x \in \mathcal { X }$ and $\delta > 0$ such that

$$
R _ { \eta } ( x ) - R _ { \eta } ( z _ { + } ) + \langle \nabla \Psi ( z _ { + } ) , x - z _ { + } \rangle = - \delta .\tag{55}
$$

16

<!-- page 17 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

On the other hand, by the definition of the differentiability, we have

$$
\operatorname* { l i m } _ { \tau \to 0 } \frac { \Psi ( z _ { + } + \tau ( x - z _ { + } ) ) - \Psi ( z _ { + } ) - \tau \langle \nabla \Psi ( z _ { + } ) , x - z _ { + } \rangle } { \tau } = 0 .\tag{56}
$$

Hence, there exists $\tau \in ( 0 , 1 )$ such that

$$
\Psi ( z _ { + } + \tau ( x - z _ { + } ) ) - \Psi ( z _ { + } ) - \tau \langle \nabla \Psi ( z _ { + } ) , x - z _ { + } \rangle \leq \textstyle { \frac { 1 } { 2 } } \delta \tau .\tag{57}
$$

Hence, we obtain the following:

$$
\begin{aligned} { - \delta } & { { } \stackrel { ( \mathtt { d } ) } { = } R _ { \eta } ( x ) - R _ { \eta } ( z _ { + } ) + \langle \nabla \Psi ( z _ { + } ) , x - z _ { + } \rangle } \\ { } & { { } \stackrel { ( \mathtt { d } ) } { \geq } \frac { 1 } { \tau } \big ( R _ { \eta } ( z _ { + } + \tau ( x - z _ { + } ) ) - R _ { \eta } ( z _ { + } ) + \tau \langle \nabla \Psi ( z _ { + } ) , x - z _ { + } \rangle \big ) } \\ { } & { { } \stackrel { ( \mathtt { c } ) } { \geq } - \frac { 1 } { \tau } \big ( \Psi ( z _ { + } + \tau ( x - z _ { + } ) ) - \Psi ( z _ { + } ) - \tau \langle \nabla \Psi ( z _ { + } ) , x - z _ { + } \rangle \big ) } \\ { } & { { } \stackrel { ( \mathtt { d } ) } { \geq } - \frac { 1 } { 2 } \delta , } \\ \end{aligned}
$$

where (a) uses eq. (55); (b) uses the convexity of function $R _ { \eta } ( x ) ; ( \mathrm { c } )$ uses eq. (54); (d) uses eq. (57). This implies the contradiction $\delta \leq 0$ . Thus, we have shown that $- \nabla \Psi ( z _ { + } ) \in \widehat { \partial } \dot { R } _ { \eta } ( z _ { + } )$

Our next goal is to show that ri dom R ∩ ri dom $I _ { \eta } \neq \varnothing$ . In the case where $z \in$ ri dom R, this statement is obviously true because $z \in$ ri dom $I _ { \eta } .$ Hence, it remains to consider the case $z \notin$ ri dom R. By Theorem 6.3 of Rockafellar (1997), we have cl(ri dom $\dot { R } ) = \mathrm { c l }$ dom $R \supset$ dom R. Hence, there exists a sequence of vectors $\{ x _ { k } \} _ { k = 0 } ^ { \infty } \subset$ ri dom R such that lim $\operatorname { h } _ { k \to \infty } \| { \dot { x } } _ { k } - z \| = 0 .$ . Since $I _ { \eta } ( x )$ is the indicator of the ball with the center at z, we can find $x _ { k } \in$ ri dom $I _ { \eta }$ Thus, we have $x _ { k } \in$ ri dom $I _ { \eta }$ ∩ ri dom $\dot { R } \neq \varnothing$

Using the definition of $R _ { \eta } ( x )$ in eq. (52), the fact that ri dom R ∩ ri dom $I _ { \eta } \neq \emptyset$ , and Theorem 23.8 of Rockafellar $( 1 9 9 7 )$ , we have $\partial R _ { \eta } ( z _ { + } ) \widetilde { = \partial R ( z _ { + } ) } + \partial I _ { \eta } ( z _ { + } )$ . Hence, using the fact that $\widehat { - \nabla \Psi ( z _ { + } ) } \in \partial R _ { \eta } ( z _ { + } )$ , we have

$$
\nabla \Psi ( z _ { + } ) + \hat { \nabla } R + \hat { \nabla } I _ { \eta } = 0 ,\tag{58}
$$

where $\hat { \nabla } R \in \partial R ( z _ { + } )$ , and $\hat { \nabla } I _ { \eta } \in \partial I _ { \eta } ( z _ { + } )$ . Therefore, we obtain the following:

$$
\begin{aligned} { R ( z ) + \langle \nabla \Psi ( z _ { + } ) , z - z _ { + } \rangle } & { { } \overset { ( a ) } { \geq } R ( z _ { + } ) + \langle \nabla \Psi ( z _ { + } ) + \hat { \nabla } R , z - z _ { + } \rangle } \\ { } & { { } \overset { ( b ) } { = } R ( z _ { + } ) - \langle \hat { \nabla } I _ { \eta } , z - z _ { + } \rangle } \\ { } & { { } = R ( z _ { + } ) + \langle \hat { \nabla } I _ { \eta } , z _ { + } \rangle - \langle \hat { \nabla } I _ { \eta } , z \rangle } \\ { } & { { } \overset { ( c ) } { = } R ( z _ { + } ) + I _ { \eta } ( z _ { + } ) + I _ { \eta } ^ { * } ( \hat { \nabla } I _ { \eta } ) - \langle \hat { \nabla } I _ { \eta } , z \rangle } \\ { } & { { } \overset { ( d ) } { = } R ( z _ { + } ) + \eta \| \hat { \nabla } I _ { \eta } \| _ { * } } \\ { } & { { } \overset { ( e ) } { = } R ( z _ { + } ) + \eta \| \nabla \Psi ( z _ { + } ) + \hat { \nabla } R \| _ { * } , } \\ \end{aligned}
$$

where (a) uses the convexity of function $R ( z ) ; ( \mathsf { b } )$ and (e) use eq. (58); (c) uses the Fenchel-Young equality; (d) uses the definition of $I _ { \eta } ( x )$ and standard calculations. □

17

<!-- page 18 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

## D.2 Proof of Theorem 2

We start with the following Lemma 4, which describes the update rule (15). It can be seen as an adaptation of the famous descent lemma in the context of trust-region optimization. The proof is, in many ways, based on Lemma 3 and is available in Appendix D.2.1.

**Lemma 4.** Let Assumption (A2) hold, and let $x _ { 0 } \in$ dom $R .$ Then the iterations of Algorithm 1 satisfy the following inequality:

$$
\begin{array} { l l } { F ( x _ { k + 1 } ) \leq F ( x _ { k } ) - \eta \| \nabla f ( x _ { k + 1 } ) + \hat { \nabla } R _ { k + 1 } \| _ { * } + 2 \eta \| \nabla f ( x _ { k + 1 } ) - m _ { k + 1 } \| _ { * } + \frac { 3 } { 2 } L \eta ^ { 2 } , } \\ \end{array}\tag{59}
$$

where $\hat { \nabla } R _ { k + 1 } \in \partial R ( x _ { k + 1 } )$

The next Lemma 5 describes the dynamics of the momentum term, which is updated according to eq. (14). In particular, it upper-bounds the expected distance between the momentum $m _ { k }$ and the true gradient $\overline { { \nabla } } f ( x _ { k } )$ . The proof can be seen as an adaptation of the proof by Cutkosky and Mehta (2020) to the non-Euclidean norm setting and is available in Appendix D.2.2.

**Lemma 5.** Let Assumptions (A1) to (A3) hold, and let $x _ { 0 } \in$ dom R and $m _ { 0 }   =   g ( x _ { 0 } , \xi _ { 0 } )$ . Then the iterations of Algorithm 1 satisfy the following inequality for $k \geq 0 ;$

$$
\mathbb { E } \left[ \| m _ { k + 1 } - \nabla f ( x _ { k } ) \| _ { * } \right] \leq ( 1 - \alpha ) ^ { k + 1 } \rho \sigma + \sqrt { \alpha } \rho \sigma + \frac { L \eta } { \alpha } .\tag{60}
$$

Now, we are ready to prove Theorem 2. Using Lemma $4 ,$ we obtain the following inequality:

$$
\begin{aligned} { \operatorname* { m i n } _ { k = 1 , \ldots , K } \lVert \nabla f ( x _ { k } ) + \hat { \nabla } R _ { k } \rVert _ { * } } & { { } \leq \frac { F ( x _ { 0 } ) - \operatorname* { i n f } _ { x } F ( x ) } { \eta K } + \frac { 3 L \eta } { 2 } + \frac { 2 } { K } \sum _ { k = 1 } ^ { K } \lVert \nabla f ( x _ { k } ) - m _ { k } \rVert _ { * } } \\ { } & { { } \overset { ( a ) } { \leq } \frac { F ( x _ { 0 } ) - \operatorname* { i n f } _ { x } F ( x ) } { \eta K } + \frac { 7 L \eta } { 2 } + \frac { 2 } { K } \sum _ { k = 0 } ^ { K - 1 } \lVert \nabla f ( x _ { k } ) - m _ { k + 1 } \rVert _ { * } , } \\ \end{aligned}
$$

where (a) uses the triangle inequality, Assumption (A2), and eq. (15). Using Lemma $^ { 5 , }$ we obtain

$$
\mathbb { E } \left[ \operatorname* { m i n } _ { k = 1 , \ldots , K } \lVert \nabla f ( x _ { k } ) + \hat { \nabla } R _ { k } \rVert _ { * } \right] \leq \frac { F ( x _ { 0 } ) - \operatorname* { i n f } _ { x } F ( x ) } { \eta K } + \frac { 7 L \eta } { 2 } + \frac { 2 L \eta } { \alpha } + \frac { 2 \rho \sigma } { \alpha K } + 2 \sqrt { \alpha } \rho \sigma .
$$

## D.2.1 Proof of Lemma 4

We can upper-bound $F ( x _ { k + 1 } )$ as follows:

$$
\begin{aligned} { } & { { } f ( x _ { k + 1 } ) \overset { ( a ) } { = } f ( x _ { k + 1 } ) + R ( x _ { k + 1 } ) } \\ { } & { { } \quad \overset { ( b ) } { \leq } f ( x _ { k } ) + \langle \nabla f ( x _ { k } ) , x _ { k + 1 } - x _ { k } \rangle + \frac { 1 } { 2 } L \| x _ { k + 1 } - x _ { k } \| ^ { 2 } + R ( x _ { k + 1 } ) } \\ { } & { { } \quad = f ( x _ { k } ) + \frac { 1 } { 2 } L \| x _ { k + 1 } - x _ { k } \| ^ { 2 } + R ( x _ { k + 1 } ) } \\ { } & { { } \quad + \langle m _ { k + 1 } + \nabla f ( x _ { k + 1 } ) - m _ { k + 1 } + \nabla f ( x _ { k } ) - \nabla f ( x _ { k + 1 } ) , x _ { k + 1 } - x _ { k } \rangle } \\ { } & { { } \quad \overset { ( c ) } { \leq } f ( x _ { k } ) + \frac { 1 } { 2 } L \| x _ { k + 1 } - x _ { k } \| ^ { 2 } + R ( x _ { k + 1 } ) + \langle m _ { k + 1 } , x _ { k + 1 } - x _ { k } \rangle } \\ { } & { { } \quad + \| x _ { k + 1 } - x _ { k } \| \| \nabla f ( x _ { k + 1 } ) - m _ { k + 1 } \| _ { * } + \| x _ { k + 1 } - x _ { k } \| \| \nabla f ( x _ { k } ) - \nabla f ( x _ { k + 1 } ) \| _ { * } } \\ { } & { { } \quad \overset { ( d ) } { \leq } f ( x _ { k } ) + \frac { 3 } { 2 } L \| x _ { k + 1 } - x _ { k } \| ^ { 2 } + \| x _ { k + 1 } - x _ { k } \| \| \nabla f ( x _ { k + 1 } ) - m _ { k + 1 } \| _ { * } } \\ { } & { { } \quad + R ( x _ { k + 1 } ) + \langle m _ { k + 1 } , x _ { k + 1 } - x _ { k } \rangle } \\ { } & { { } \quad \overset { ( e ) } { \leq } f ( x _ { k } ) + \frac { 3 } { 2 } L \eta ^ { 2 } + \eta \| \nabla f ( x _ { k + 1 } ) - m _ { k + 1 } \| _ { * } + R ( x _ { k + 1 } ) + \langle m _ { k + 1 } , x _ { k + 1 } - x _ { k } \rangle } \\ { } & { { } \quad \overset { ( d ) } { \leq } f ( x _ { k } ) + \frac { 3 } { 2 } L \eta ^ { 2 } + \eta \| \nabla f ( x _ { k + 1 } ) - m _ { k + 1 } \| _ { * } + R ( x _ { k } ) - \eta \| m _ { k + 1 } + \hat { \nabla } R _ { k + 1 } \| _ { * } } \\ { } & { { } \ \end{aligned}
$$

where (a) and (g) use the definition of function F(x) in problem (1); (b) and (d) use Assumption (A2); (c) uses the definition of the dual norm; (e) uses the fact that $\left\| x _ { k + 1 } - x _ { k } \right\| \leq \eta ,$ which is implied by eq. (15); (f) uses eq. (15) and Lemma 3 with $\Psi ( x ) = \langle m _ { k + 1 } , x \rangle \; \mathsf { a n d } \; ( z _ { + } , z ) = \hat { ( x _ { k + 1 } , x _ { k } ) } ; \hat { ( \mathsf { h } ) }$ uses the triangle inequality. □

18

<!-- page 19 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

## D.2.2 Proof of Lemma 5

We can express $m _ { k + 1 } - \nabla f ( x _ { k } )$ as follows using eq. (14):

$$
\begin{aligned} { m _ { k + 1 } - \nabla f ( x _ { k } ) } & { { } = ( 1 - \alpha ) m _ { k } + \alpha g ( x _ { k } ; \xi _ { k } ) - \nabla f ( x _ { k } ) } \\ { } & { { } = ( 1 - \alpha ) ( m _ { k } - \nabla f ( x _ { k - 1 } ) ) + \alpha ( g ( x _ { k } ; \xi _ { k } ) - \nabla f ( x _ { k } ) ) } \\ { } & { { } + ( 1 - \alpha ) ( \nabla f ( x _ { k - 1 } ) - \nabla f ( x _ { k } ) ) . } \\ \end{aligned}
$$

This implies the following for all $k \geq 0 ;$

$$
\begin{aligned} { m _ { k + 1 } - \nabla f ( x _ { k } ) } & { { } = ( 1 - \alpha ) ^ { k + 1 } ( m _ { 0 } - \nabla f ( x _ { 0 } ) ) + \sum _ { i = 0 } ^ { k - 1 } ( 1 - \alpha ) ^ { k - i } ( \nabla f ( x _ { i } ) - \nabla f ( x _ { i + 1 } ) ) } \\ { } & { { } + \sum _ { i = 0 } ^ { k } \alpha ( 1 - \alpha ) ^ { k - i } ( g ( x _ { i } , \xi _ { i } ) - \nabla f ( x _ { i } ) ) . } \\ \end{aligned}
$$

Using this, we can upper-bound E $[ \| m _ { k + 1 } - \nabla f ( x _ { k } ) \| _ { * } ] \: { \sf f o r \: } k \geq 0$ as follows:

$$
\begin{aligned} { \mathbb { E } \left[ \| m _ { k + 1 } - \nabla f ( x _ { k } ) \| _ { * } \right] } & { { } \overset { ( a ) } { \leq } ( 1 - \alpha ) ^ { k + 1 } \mathbb { E } \left[ \| m _ { 0 } - \nabla f ( x _ { 0 } ) \| _ { * } \right] } \\ { } & { { } + \sum _ { i = 0 } ^ { k - 1 } ( 1 - \alpha ) ^ { k - i } \| \nabla f ( x _ { i } ) - \nabla f ( x _ { i + 1 } ) \| _ { * } } \\ { } & { { } + \mathbb { E } \left[ \| \sum _ { i = 0 } ^ { k } \alpha ( 1 - \alpha ) ^ { k - i } ( g ( x _ { i } , \xi _ { i } ) - \nabla f ( x _ { i } ) ) \| _ { * } \right] } \\ { } & { { } \overset { ( b ) } { \leq } ( 1 - \alpha ) ^ { k + 1 } \mathbb { E } \left[ \| m _ { 0 } - \nabla f ( x _ { 0 } ) \| _ { * } \right] + \sum _ { i = 0 } ^ { k - i } ( 1 - \alpha ) ^ { k - i } L \eta } \\ { } & { { } + \mathbb { E } \left[ \| \sum _ { i = 0 } ^ { k } \alpha ( 1 - \alpha ) ^ { k - i } ( g ( x _ { i } , \xi _ { i } ) - \nabla f ( x _ { i } ) ) \| _ { * } \right] } \\ { } & { { } \overset { ( c ) } { \leq } ( 1 - \alpha ) ^ { k + 1 } \rho \mathbb { E } \left[ \| m _ { 0 } - \nabla f ( x _ { 0 } ) \| _ { 2 } \right] + \sum _ { i = 0 } ^ { k - 1 } ( 1 - \alpha ) ^ { k - i } L \eta } \\ { } & { { } + \rho \mathbb { E } \left[ \| \sum _ { i = 0 } ^ { k } \alpha ( 1 - \alpha ) ^ { k - i } ( g ( x _ { i } , \xi _ { i } ) - \nabla f ( x _ { i } ) ) \| _ { 2 } \right] } \\ { } & { { } \overset { ( d ) } { \leq } ( 1 - \alpha ) ^ { k + 1 } \rho \sqrt { \mathbb { E } \left[ \| m _ { 0 } - \nabla f ( x _ { 0 } ) \| _ { 2 } ^ { 2 } \right] } + \sum _ { i = 0 } ^ { k - 1 } ( 1 - \alpha ) ^ { k - i } L \eta } \\ { } & { { } + \rho \sqrt { \mathbb { E } \left[ \| \sum _ { i = 0 } ^ { k } \alpha ( 1 - \alpha ) ^ { k - i } ( g ( x _ { i } , \xi _ { i } ) - \nabla f ( x _ { i } ) ) \| _ { 2 } ^ { 2 } \right] } } \\ { } & { { } \overset { ( e ) } { \leq } ( 1 - \alpha ) ^ { k + 1 } \rho \sigma + \sum _ { i = 0 } ^ { k - 1 } ( 1 - \alpha ) ^ { k - i } L \eta + \alpha \rho \sigma \sqrt { \sum _ { i = \end{aligned}
$$

where (a) uses the triangle inequality; (b) uses Assumption (A2) and eq. (15); (c) uses Assumption (A3); (d) uses Jensen’s inequality; (e) uses Assumption (A1) and the fact that samples $\xi _ { i } \sim \mathcal { D }$ are i.i.d.. □

19

<!-- page 20 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

## D.3 Proof of Theorem 3

We start with the following Lemma 6. The proof is available in Appendix D.3.1.

**Lemma 6.** Under the conditions of Theorem 3, let $x \in \mathcal { X }$ be defined as follows:

$$
x = \beta x ^ { * } + ( 1 - \beta ) x _ { k } .\tag{61}
$$

Then, the following inequalities hold:

$$
\| x - ( 1 - \beta ) x _ { k } \| \leq \eta , \quad \| x - x _ { k } \| \leq 2 \eta , \quad \| x - x _ { k + 1 } \| \leq 2 \eta , \quad \| x _ { k + 1 } - x _ { k } \| \leq 2 \eta .\tag{62}
$$

Now, we are ready to prove Theorem 3. We can upper-bound $F ( x _ { k + 1 } )$ as follows:

$$
\begin{array} { l } { \displaystyle F ( x _ { k + 1 } ) \overset { ( a ) } { \leq } f ( x _ { k + 1 } ) + R ( x _ { k + 1 } ) } \\ { \quad \overset { ( b ) } { \leq } f ( x _ { k } ) + \langle \nabla f ( x _ { k } ) , x _ { k + 1 } - x _ { k } \rangle + \frac { 1 } { 2 } L \| x _ { k + 1 } - x _ { k } \| ^ { 2 } + R ( x _ { k + 1 } ) } \\ { \quad \overset { ( c ) } { \leq } f ( x _ { k } ) + \langle \nabla f ( x _ { k } ) , x _ { k + 1 } - x _ { k } \rangle + R ( x _ { k + 1 } ) + 2 L \eta ^ { 2 } } \\ { \quad \overset { ( d ) } { \leq } f ( x _ { k } ) + \langle \nabla f ( x _ { k } ) , x - x _ { k } \rangle + R ( x ) + 2 L \eta ^ { 2 } } \\ { \quad \overset { ( e ) } { \leq } f ( x ) + \frac { 1 } { 2 } L \| x - x _ { k } \| ^ { 2 } + R ( x ) + 2 L \eta ^ { 2 } } \\ { \quad \overset { ( o ) } { \leq } f ( x ) + R ( x ) + 4 L \eta ^ { 2 } } \\ { \quad \overset { ( g ) } { \leq } \beta ( f ( x ^ { * } ) + R ( x ^ { * } ) ) + ( 1 - \beta ) ( f ( x _ { k } ) + R ( x _ { k } ) ) + 4 L \eta ^ { 2 } } \\ { \quad \overset { ( h ) } { \leq } \beta F ( x ^ { * } ) + ( 1 - \beta ) F ( x _ { k } ) + 4 L \eta ^ { 2 } , } \\ \end{array}
$$

where (a) and (h) use the definition of function $F ( x )$ in problem (1); (b) and (e) use Assumption $( \mathrm { A } 2 ) ; ( \mathrm { c } )$ and (f) use Lemma $6 ;   ( \mathrm { d } )$ uses eq. (19) and the inequality $\| \dot { x } - ( 1 - \beta ) x _ { k } \| \: \leq \: \eta ,$ , which is implied by Lemma $6 ;   ( \mathrm { g } )$ uses assumption (A4) and the convexity if function $R ( \stackrel { \cdot \cdot } { x } )$ . After rearranging, we obtain

$$
F ( x _ { k + 1 } ) - F ( x ^ { * } ) \leq ( 1 - \beta ) ( F ( x _ { k } ) - F ( x ^ { * } ) ) + 4 L \eta ^ { 2 } ,
$$

which implies the following inequality:

$$
F ( x _ { K } ) - F ( x ^ { * } ) \leq ( 1 - \beta ) ^ { K } ( F ( x _ { 0 } ) - F ( x ^ { * } ) ) + \frac { 4 L \eta ^ { 2 } } { \beta } .
$$

## D.3.1 Proof of Lemma 6

First, we can show that $\beta \| x _ { k } \| \leq \eta$ by induction. Indeed $\beta \| x _ { 0 } \| \leq \eta$ due to eq. (22), and for all $k \in \{ 0 , 1 , \ldots , K - 1 \}$ can obtain the following:

$$
\beta \| x _ { k + 1 } \| \overset { \mathtt { ( a ) } } { \leq } \beta \| x _ { k + 1 } - ( 1 - \beta ) x _ { k } \| + ( 1 - \beta ) \beta \| x _ { k } \| \overset { \mathtt { ( b ) } } { \leq } \eta \beta + ( 1 - \beta ) \eta \leq \eta ,
$$

where (a) uses the triangle inequality; (b) uses eq. (19) and the induction hypothesis. Hence, we can prove the desired inequalities as follows:

$$
\begin{aligned} { \left\| x - ( 1 - \beta ) x _ { k } \right\| } & { { } \overset { \mathtt { ( a ) } } { = } \beta \| x ^ { * } \| \overset { \mathtt { ( b ) } } { \leq } \eta , } \\ { \left\| x - x _ { k } \right\| } & { { } \overset { \mathtt { ( c ) } } { = } \beta \| x ^ { * } - x _ { k } \| \overset { \mathtt { ( d ) } } { \leq } \beta \| x ^ { * } \| + \beta \| x _ { k } \| \overset { \mathtt { ( e ) } } { \leq } 2 \eta , } \\ { \left\| x - x _ { k + 1 } \right\| } & { { } \overset { \mathtt { ( f ) } } { \leq } \| x - ( 1 - \beta ) x _ { k } \| + \left\| x _ { k + 1 } - ( 1 - \beta ) x _ { k } \right\| \overset { \mathtt { ( g ) } } { \leq } 2 \eta , } \\ { \left\| x _ { k + 1 } - x _ { k } \right\| } & { { } \overset { \mathtt { ( h ) } } { \leq } \| x _ { k + 1 } - ( 1 - \beta ) x _ { k } \| + \beta \| x _ { k } \| \overset { \mathtt { ( i ) } } { \leq } 2 \eta , } \\ \end{aligned}
$$

where (a) and (c) use the definition of x in eq. $(61) ; ( \mathsf{b} )$ uses eq. $(22) ; ( \mathrm{d} ) , ( \mathrm{f} )$ and (h) use the triangle inequality; (e) uses eq. (22) and the previously obtained inequality $\beta \| x _ { k } \| \leq \eta ; ( \mathrm { g } )$ uses eq. (19) and the previously obtained inequality $\left\| x - (1 - \beta)x_{k} \right\| \leq \eta;$ uses eq. (19) and the previously obtained inequality $\beta \| x _ { k } \| \leq \eta .$ □

20

<!-- page 21 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

## D.4 Proof of Theorem 4

In this proof, we are going to use Lemma 6, which is valid not only for the iterations (19), but also for Algorithm 2. We also obtain the following Lemma 7. The proof is almost identical to the proof of Lemma 5 in Appendix D.2.2, with the only difference being that we use the inequality $\| x _ { k + 1 } - x _ { k } \| \leq 2 \eta$ , which holds due to Lemma 6.

**Lemma 7.** Let Assumptions (A1) to (A3) hold, and let $x _ { 0 } \in$ dom R and $m _ { 0 }   =   g ( x _ { 0 } , \xi _ { 0 } )$ . Then the iterations of Algorithm 2 satisfy the following inequality for $k \geq 0 .$

$$
\mathbb { E } \left[ \| m _ { k + 1 } - \nabla f ( x _ { k } ) \| _ { * } \right] \leq ( 1 - \alpha ) ^ { k + 1 } \rho \sigma + \sqrt { \alpha } \rho \sigma + \frac { 2 L \eta } { \alpha } .\tag{63}
$$

Now, we are ready to prove Theorem 4. We can upper-bound $F ( x _ { k + 1 } )$ as follows:

$$
\begin{aligned} { } & { { } F ( x _ { k + 1 } ) \mathop { \otimes } ^ { ( 2 ) } f ( x _ { k + 1 } ) + R ( x _ { k + 1 } ) } \\ { } & { { } \quad \mathop { \otimes } ^ { ( 2 ) } f ( x _ { k } ) + \langle \nabla f ( x _ { k } ) , x _ { k + 1 } - x _ { k } \rangle + \frac { 1 } { 2 } L \| x _ { k + 1 } - x _ { k } \| ^ { 2 } + R ( x _ { k + 1 } ) } \\ { } & { { } \quad \mathop { \otimes } ^ { ( 2 ) } f ( x _ { k } ) + \langle \nabla f ( x _ { k } ) - m _ { k + 1 } , x _ { k + 1 } - x _ { k } \rangle + 2 L \eta ^ { 2 } + \langle m _ { k + 1 } , x _ { k + 1 } - x _ { k } \rangle + R ( x _ { k + 1 } ) } \\ { } & { { } \quad \mathop { \otimes } ^ { ( 2 ) } f ( x _ { k } ) + \langle \nabla f ( x _ { k } ) - m _ { k + 1 } , x _ { k + 1 } - x _ { k } \rangle + 2 L \eta ^ { 2 } + \langle m _ { k + 1 } , x - x _ { k } \rangle + R ( x ) } \\ { } & { { } \quad \mathop { \otimes } ^ { ( 2 ) } f ( x _ { k } ) + \langle \nabla f ( x _ { k } ) , x - x _ { k } \rangle + 2 L \eta ^ { 2 } + R ( x ) + \langle m _ { k + 1 } - \nabla f ( x _ { k } ) , x - x _ { k + 1 } \rangle } \\ { } & { { } \quad \mathop { \otimes } ^ { ( 2 ) } f ( x _ { k } ) + \langle \nabla f ( x _ { k } ) , x - x _ { k } \rangle + 2 L \eta ^ { 2 } + R ( x ) + \| x - x _ { k + 1 } \| \| m _ { k + 1 } - \nabla f ( x _ { k } ) \| _ { * } } \\ { } & { { } \quad \mathop { \otimes } ^ { ( 2 ) } f ( x _ { k } ) + \langle \nabla f ( x _ { k } ) , x - x _ { k } \rangle + 2 L \eta ^ { 2 } + R ( x ) + 2 \eta \| m _ { k + 1 } - \nabla f ( x _ { k } ) \| _ { * } } \\ { } & { { } \quad \mathop { \otimes } ^ { ( 2 ) } f ( x ) + \frac { 1 } { 2 } L \| x - x _ { k } \| ^ { 2 } + 2 L \eta ^ { 2 } + R ( x ) + 2 \eta \| m _ { k + 1 } - \nabla f ( x _ { k } ) \| _ { * } } \\ { } & { { } \quad \mathop { \otimes } ^ { ( 2 ) } f ( x ) + 4 L \eta ^ { 2 } + R ( x ) + 2 \eta \| m _ { k + 1 } - \nabla f ( x _ { k } ) \| _ { * } } \\ { } & { { } \quad \mathop { \otimes } ^ { ( 2 ) } \beta ( f ( x ^ { * } ) + R ( x ^ { * } ) ) + ( 1 - \beta ) ( f  \end{aligned}
$$

where (a) and (j) use the definition of function $F ( x )$ in problem $(1) ; ( \mathsf{b} )$ and (g) use Assumption $( \mathrm { A } 2 ) ; ( \mathrm { c } ) , ( \mathrm { f } )$ and (h) use Lemma 6; (d) uses eq. (21) and the inequality $\| x - ( 1 - \beta ) x _ { k } \|   \leq   \eta ,$ which is implied by Lemma 6; (e) uses the definition of the dual norm; (i) uses assumption (A4) and the convexity if function $R ( x )$ . After rearranging, taking the expectation, and using Lemma 7, we get

$$
\begin{aligned} { \mathbb { E } \left[ F ( x _ { k + 1 } ) - F ( x ^ { * } ) \right] } & { { } \leq ( 1 - \beta ) \mathbb { E } \left[ F ( x _ { k } ) - F ( x ^ { * } ) \right] + 2 \eta \rho \sigma ( 1 - \alpha ) ^ { k } + 2 \eta \rho \sigma \sqrt { \alpha } } \\ { } & { { } + 4 L \eta ^ { 2 } + \frac { 4 L \eta ^ { 2 } } { \alpha } , } \\ \end{aligned}
$$

which implies the following inequality:

$$
\mathbb { E } \left[ F ( x _ { K } ) - F ( x ^ { * } ) \right] \leq ( 1 - \beta ) ^ { K } ( F ( x _ { 0 } ) - F ( x ^ { * } ) ) + 2 \eta \rho \sigma \left( \frac { 1 } { \alpha } + \frac { \sqrt { \alpha } } { \beta } \right) + \frac { 4 L \eta ^ { 2 } } { \beta } \left( 1 + \frac { 1 } { \alpha } \right) .
$$

21

<!-- page 22 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

## D.5 Proof of Theorem 5

In this proof, we are going to use Lemma 4, whis is valid for the iterations of Algorithm 3 as long as $\beta = 0$ . We also obtain the following Lemma 8, which describes the dynamics of the momentum $m _ { k }$ under the second-order smoothness. Similarly to the proofs of Lemma 5 and eq. (20), it can be seen as an adaptation of the proof by Cutkosky and Mehta (2020) to the non-Euclidean setting and is available in Appendix D.5.1.

**Lemma 8.** Let Assumptions (A1), (A3) and (A5) hold, and let $x _ { 0 } \in$ dom R and $m _ { 0 } = g ( x _ { 0 } , \xi _ { 0 } )$ . Then the iterations of Algorithm 3 satisfy the following inequality for $k \geq 0 ;$

$$
\mathbb { E } \left[ \| m _ { k + 1 } - \nabla f ( x _ { k } ) \| _ { * } \right] \leq ( 1 - \alpha ) ^ { k + 1 } \rho \sigma + \sqrt { \alpha } \rho \sigma + \frac { H \eta ^ { 2 } } { 2 \alpha ^ { 2 } } .\tag{64}
$$

Now, we are ready to prove Theorem 5. Using Lemma 4, we obtain the following inequality:

$$
\operatorname* { m i n } _ { k = 1 , \ldots , K } \lVert \nabla f ( x _ { k } ) + \hat { \nabla } R _ { k } \rVert _ { * } \leq \frac { F ( x _ { 0 } ) - \operatorname* { i n f } _ { x } F ( x ) } { \eta K } + \frac { 3 L \eta } { 2 } + \frac { 2 } { K } \sum _ { k = 1 } ^ { K } \lVert \nabla f ( x _ { k } ) - m _ { k } \rVert _ { * }
$$

$$
\stackrel { ( \mathsf { a } ) } { \leq } \frac { F ( x _ { 0 } ) - \operatorname* { i n f } _ { x } F ( x ) } { \eta K } + \frac { 7 L \eta } { 2 } + \frac { 2 } { K } \sum _ { k = 0 } ^ { K - 1 } \lVert \nabla f ( x _ { k } ) - m _ { k + 1 } \rVert _ { * } ,
$$

where (a) uses the triangle inequality and Assumption (A2). Using Lemma 8, we obtain the following:

$$
\mathbb { E } \left[ \operatorname* { m i n } _ { k = 1 , \ldots , K } \lVert \nabla f ( x _ { k } ) + \hat { \nabla } R _ { k } \rVert _ { * } \right] \leq \frac { F ( x _ { 0 } ) - \operatorname* { i n f } _ { x } F ( x ) } { \eta K } + \frac { 7 L \eta } { 2 } + \frac { H \eta ^ { 2 } } { \alpha ^ { 2 } } + \frac { 2 \rho \sigma } { \alpha K } + 2 \sqrt { \alpha } \rho \sigma .
$$

## D.5.1 Proof of Lemma 8

We can express $m _ { k + 1 } - \nabla f ( x _ { k } )$ as follows using eq. (28):

$$
\begin{aligned} { m _ { k + 1 } - \nabla f ( x _ { k } ) } & { { } = ( 1 - \alpha ) m _ { k } + \alpha g ( \overline { { x } } _ { k } ; \xi _ { k } ) - \nabla f ( x _ { k } ) } \\ { } & { { } = ( 1 - \alpha ) ( m _ { k } - \nabla f ( x _ { k - 1 } ) ) + \alpha ( g ( \overline { { x } } _ { k } ; \xi _ { k } ) - \nabla f ( \overline { { x } } _ { k } ) ) } \\ { } & { { } + \alpha \nabla f ( \overline { { x } } ^ { k } ) + ( 1 - \alpha ) \nabla f ( x _ { k - 1 } ) - \nabla f ( x _ { k } ) } \\ \end{aligned}
$$

This implies the following for all $k \geq 0 ;$

$$
\begin{aligned} { m _ { k + 1 } - \nabla f ( x _ { k } ) } & { { } = ( 1 - \alpha ) ^ { k + 1 } ( m _ { 0 } - \nabla f ( x _ { 0 } ) ) + \sum _ { i = 0 } ^ { k } \alpha ( 1 - \alpha ) ^ { k - i } ( g ( \overline { { x } } _ { i } , \xi _ { i } ) - \nabla f ( \overline { { x } } _ { i } ) ) } \\ { } & { { } + \sum _ { i = 0 } ^ { k - 1 } ( 1 - \alpha ) ^ { k - i - 1 } ( \alpha \nabla f ( \overline { { x } } _ { i + 1 } ) + ( 1 - \alpha ) \nabla f ( x _ { i } ) - \nabla f ( x _ { i + 1 } ) ) . } \\ \end{aligned}
$$

Using this, we can upper-bound E $[ \| m _ { k + 1 } - \nabla f ( x _ { k } ) \| _ { * } ]$ for $k \geq 0$ as follows:

$$
\begin{aligned} { \mathbb { E } \left[ \| m _ { k + 1 } - \nabla f ( x _ { k } ) \| _ { * } \right] } & { { } \overset { ( a ) } { \leq } ( 1 - \alpha ) ^ { k + 1 } \mathbb { E } \left[ \| m _ { 0 } - \nabla f ( x _ { 0 } ) \| _ { * } \right] } \\ { } & { { } + \sum _ { i = 0 } ^ { k - 1 } ( 1 - \alpha ) ^ { k - i - 1 } \| \alpha \nabla f ( \overline { { x } } _ { i + 1 } ) + ( 1 - \alpha ) \nabla f ( x _ { i } ) - \nabla f ( x _ { i + 1 } ) \| _ { * } } \\ { } & { { } + \mathbb { E } \left[ \| \sum _ { i = 0 } ^ { k } \alpha ( 1 - \alpha ) ^ { k - i } ( g ( \overline { { x } } _ { i } , \xi _ { i } ) - \nabla f ( \overline { { x } } _ { i } ) ) \| _ { * } \right] } \\ { } & { { } \overset { ( b ) } { \leq } ( 1 - \alpha ) ^ { k + 1 } \rho \mathbb { E } \left[ \| m _ { 0 } - \nabla f ( x _ { 0 } ) \| _ { 2 } \right] } \\ { } & { { } + \sum _ { i = 0 } ^ { k - 1 } ( 1 - \alpha ) ^ { k - i - 1 } \| \alpha \nabla f ( \overline { { x } } _ { i + 1 } ) + ( 1 - \alpha ) \nabla f ( x _ { i } ) - \nabla f ( x _ { i + 1 } ) \| _ { * } } \\ { } & { { } + \rho \mathbb { E } \left[ \| \sum _ { i = 0 } ^ { k } \alpha ( 1 - \alpha ) ^ { k - i } ( g ( \overline { { x } } _ { i } , \xi _ { i } ) - \nabla f ( \overline { { x } } _ { i } ) ) \| _ { 2 } \right] } \\ \end{aligned}
$$

22

<!-- page 23 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

$$
\begin{aligned} { } & { { } \sum _ { i = 0 } ^ { \infty } ( 1 - \alpha ) ^ { k + 1 } \rho \sqrt { \mathbb { E } \left[ \| m _ { 0 } - \nabla f ( x _ { 0 } ) \| _ { 2 } ^ { 2 } \right] } } \\ { } & { { } + \sum _ { i = 0 } ^ { k - 1 } ( 1 - \alpha ) ^ { k - i - 1 } \| \alpha \nabla f ( \overline { { x } } _ { i + 1 } ) + ( 1 - \alpha ) \nabla f ( x _ { i } ) - \nabla f ( x _ { i + 1 } ) \| _ { * } } \\ { } & { { } + \rho \sqrt { \mathbb { E } \left[ \| \sum _ { i = 0 } ^ { k } \alpha ( 1 - \alpha ) ^ { k - i } ( g ( \overline { { x } } _ { i } , \xi _ { i } ) - \nabla f ( \overline { { x } } _ { i } ) ) \| _ { 2 } ^ { 2 } \right] } } \\ { } & { { } \overset { ( a ) } { \leq } ( 1 - \alpha ) ^ { k + 1 } \rho \sigma + \alpha \rho \sigma \sqrt { \sum _ { i = 0 } ^ { k } ( 1 - \alpha ) ^ { 2 ( k - i ) } } } \\ { } & { { } + \sum _ { i = 0 } ^ { k - 1 } ( 1 - \alpha ) ^ { k - i - 1 } \| \alpha \nabla f ( \overline { { x } } _ { i + 1 } ) + ( 1 - \alpha ) \nabla f ( x _ { i } ) - \nabla f ( x _ { i + 1 } ) \| _ { * } } \\ { } & { { } \overset { ( b ) } { \leq } ( 1 - \alpha ) ^ { k + 1 } \rho \sigma + \sqrt { \alpha } \rho \sigma } \\ { } & { { } + \sum _ { i = 0 } ^ { k - 1 } ( 1 - \alpha ) ^ { k - i - 1 } \| \alpha \nabla ^ { 2 } f ( x _ { i + 1 } ) ( \alpha \overline { { x } } _ { i + 1 } + ( 1 - \alpha ) x _ { i } - x _ { i + 1 } ) \| _ { * } } \\ { } & { { } + \sum _ { i = 0 } ^ { k - 1 } \frac { H ( 1 - \alpha ) ^ { k - i - 1 } } { 2 } \left( \alpha \| \overline { { x } } _ { i + 1 } - x _ { i + 1 } \| ^ { 2 } + ( 1 - \alpha ) \| x _ { i } - x _ { i + 1 } \| ^ { 2 } \right) } \\ { } & { { } \overset { ( b ) } { = } ( 1 - \alpha ) ^ { k + 1 } \rho \sigma + \sqrt { \alpha } \rho \sigma + \sum _ { i = 0 } ^ { k - 1 } \frac { H ( 1 - \alpha ) ^ { k - i } } { 2 \alpha } \| x _ { i } - x _ { i + 1 } \| ^ { 2 } } \\ { } & { { } \overset { ( b ) } { \leq } ( 1 - \alpha ) ^ { k + 1 } \rho \sigma + \sqrt { \alpha } \rho \sigma + \sum _ { i = 0 } ^ { k - 1 } \frac { H \eta ^ { \end{aligned}
$$

where (a) uses the triangle inequality; (b) uses Assumption (A3); (c) uses Jensen’s inequality; (d) uses Assumption (A1) and the fact that samples $\xi _ { i } \sim \mathcal { D }$ are i.i.d.; (e) uses the triangle inequality and Assumption (A5); (f) uses eq. (30); (g) uses eq. (29). □

23

<!-- page 24 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

## D.6 Proof of Theorem 6

In this proof, we are going to use Lemma 6, which are valid for the iterations of Algorithm 3. We also obtain the following Lemma 9. The proof is almost identical to the proof of Lemma 8 in Appendix D.5.1, with the only difference being that we use the inequality $\left\| x_{k+1} - x_k \right\| \leq 2\eta$ , which holds due to Lemma 6.

**Lemma 9.** Let Assumptions $( A I ) ,   ( A 3 )$ and (A5) hold, and let $x _ { 0 } \in$ dom R and $m _ { 0 } = g ( x _ { 0 } , \xi _ { 0 } )$ . Then the iterations of Algorithm 3 satisfy the following inequality for $k \geq 0 .$

$$
\mathbb { E } \left[ \| m _ { k + 1 } - \nabla f ( x _ { k } ) \| _ { * } \right] \leq ( 1 - \alpha ) ^ { k + 1 } \rho \sigma + \sqrt { \alpha } \rho \sigma + \frac { 2 H \eta ^ { 2 } } { \alpha ^ { 2 } } .\tag{65}
$$

Now, we are ready to prove Theorem 6. Similarly to the proof of Theorem 4 in Appendix D.4, we can upper-bound $F ( x _ { k + 1 } ) - F ( x ^ { * } \dot { ) }$ as follows:

$$
F ( x _ { k + 1 } ) - F ( x ^ { * } ) \leq ( 1 - \beta ) ( F ( x _ { k } ) - F ( x ^ { * } ) ) + 4 L \eta ^ { 2 } + 2 \eta \| m _ { k + 1 } - \nabla f ( x _ { k } ) \| _ { * } ,
$$

Using Lemma 8, we obtain the following inequality:

$$
\begin{aligned} { \mathbb { E } \left[ F ( x _ { k + 1 } ) - F ( x ^ { * } ) \right] } & { { } \leq ( 1 - \beta ) \mathbb { E } \left[ F ( x _ { k } ) - F ( x ^ { * } ) \right] + 2 \eta \rho \sigma ( 1 - \alpha ) ^ { k } + 2 \eta \rho \sigma \sqrt { \alpha } } \\ { } & { { } + 4 L \eta ^ { 2 } + \frac { 4 H \eta ^ { 3 } } { \alpha ^ { 2 } } , } \\ \end{aligned}
$$

which implies the following inequality:

$$
\mathbb { E } \left[ F ( x _ { K } ) - F ( x ^ { * } ) \right] \leq ( 1 - \beta ) ^ { K } ( F ( x _ { 0 } ) - F ( x ^ { * } ) ) + 2 \eta \rho \sigma \left( \frac { 1 } { \alpha } + \frac { \sqrt { \alpha } } { \beta } \right) + \frac { 4 L \eta ^ { 2 } } { \beta } + \frac { 4 H \eta ^ { 3 } } { \alpha ^ { 2 } \beta } .
$$

## D.7 Proof of Theorem 7

We start with the following inequality obtained in the proof of Theorem 1 in Appendix D.1:

$$
\textstyle F ( x _ { k + 1 } ) \leq F ( x _ { k } ) - \eta \| \nabla f ( x _ { k + 1 } ) + \hat { \nabla } R _ { k + 1 } \| _ { * } + \frac { 3 } { 2 } L \eta ^ { 2 } .\tag{66}
$$

Using this inequality, we can upper-bound $F ( x _ { k + 1 } )$ as follows:

$$
\begin{aligned} { F ( x _ { k + 1 } ) } & { { } \overset { ( a ) } { = } f ( x _ { k + 1 } ) + R ( x _ { k + 1 } ) } \\ { } & { { } \overset { ( b ) } { \leq } f ( x ^ { * } ) + R ( x ^ { * } ) + \langle \nabla f ( x _ { k + 1 } ) + \hat { \nabla } R _ { k + 1 } , x _ { k + 1 } - x ^ { * } \rangle } \\ { } & { { } \overset { ( c ) } { \in } F ( x ^ { * } ) + \langle \nabla f ( x _ { k + 1 } ) + \hat { \nabla } R _ { k + 1 } , x _ { k + 1 } - x ^ { * } \rangle } \\ { } & { { } \overset { ( d ) } { \leq } F ( x ^ { * } ) + \| x _ { k + 1 } - x ^ { * } \| \| \nabla f ( x _ { k + 1 } ) + \hat { \nabla } R _ { k + 1 } \| _ { * } } \\ { } & { { } \overset { ( c ) } { \leq } F ( x ^ { * } ) + D \| \nabla f ( x _ { k + 1 } ) + \hat { \nabla } R _ { k + 1 } \| _ { * } } \\ { } & { { } \overset { ( t ) } { \leq } F ( x ^ { * } ) + ( D / \eta ) ( F ( x _ { k } ) - F ( x _ { k + 1 } ) ) + \frac { 3 } { 2 } L D \eta , } \\ \end{aligned}
$$

where (a) and (c) use the definition of $F ( x )$ in problem (1); (b) uses the convexity of function $R ( x )$ and Assumption (A4); (d) uses the definition of the dual norm; (e) uses Assumption (A6) and the fact that $x _ { k + 1 } , x ^ { * } \in$ dom $R ; ( \mathrm { f } )$ uses eq. (66). After rearranging, we obtain

$$
F ( x _ { k + 1 } ) - F ( x ^ { * } ) \leq \frac { D } { \eta + D } \left( F ( x _ { k } ) - F ( x ^ { * } ) + \frac { 3 L \eta ^ { 2 } } { 2 } \right) ,
$$

which implies the following inequality:

$$
F ( x _ { K } ) - F ( x ^ { * } ) \leq \left( \frac { D } { \eta + D } \right) ^ { K } \left( F ( x _ { 0 } ) - F ( x ^ { * } ) \right) + \frac { 3 L D \eta } { 2 } .
$$

24

<!-- page 25 of 25 -->

Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization A PREPRINT

## D.8 Proof of Theorem 8

Similarly to the proof of Theorem 7 in Appendix D.7, we can obtain the following inequality:

$$
F ( x _ { k + 1 } ) \leq F ( x ^ { * } ) + D \| \nabla f ( x _ { k + 1 } ) + \hat { \nabla } R _ { k + 1 } \| _ { * } .\tag{67}
$$

Combining this inequality with Lemma 4 gives the following:

$$
\begin{aligned} { F ( x _ { k + 1 } ) - F ( x ^ { * } ) } & { { } \leq \frac { D } { \eta + D } \left( F ( x _ { k } ) - F ( x ^ { * } ) + 2 \eta \| \nabla f ( x _ { k + 1 } ) - m _ { k + 1 } \| _ { * } + \frac { 3 L \eta ^ { 2 } } { 2 } \right) } \\ { } & { { } \stackrel { ( a ) } { \leq } \frac { D } { \eta + D } \left( F ( x _ { k } ) - F ( x ^ { * } ) + 2 \eta \| \nabla f ( x _ { k } ) - m _ { k + 1 } \| _ { * } + \frac { 7 L \eta ^ { 2 } } { 2 } \right) , } \\ \end{aligned}
$$

where (a) uses the triangle inequality, Assumption (A2), and eq. (15). After taking the expectation and using Lemma $5 ,$ we obtain the following:

$$
\begin{aligned} { \mathbb { E } \left[ F ( x _ { k + 1 } ) - F ( x ^ { * } ) \right] } & { { } \leq \left( \frac { D } { \eta + D } \right) \mathbb { E } \left[ F ( x _ { k } ) - F ( x ^ { * } ) \right] + \frac { 2 L \eta ^ { 2 } } { \alpha } + \frac { 3 L \eta ^ { 2 } } { 2 } } \\ { } & { { } + 2 ( 1 - \alpha ) ^ { k + 1 } \eta \rho \sigma + 2 \sqrt { \alpha } \eta \rho \sigma , } \\ \end{aligned}
$$

which implies the following inequality:

$$
\mathbb { E } \left[ F ( x _ { K } ) - F ( x ^ { * } ) \right] \leq \left( \frac { D } { \eta + D } \right) ^ { K } \left( F ( x _ { 0 } ) - F ( x ^ { * } ) \right) + 2 \sqrt { \alpha } D \rho \sigma + \frac { 2 \eta \rho \sigma } { \alpha } + \frac { 3 L D \eta } { 2 } + \frac { 2 L D \eta } { \alpha } .
$$

25