Title: Test-Time Training with KV Binding Is Secretly Linear Attention

URL Source: https://arxiv.org/html/2602.21204

Markdown Content:
1Introduction
2Related Work
3Preliminary
4Empirical Contradictions to Memorization
5TTT is secretly Linear Attention
6Practical Implications
7Conclusion
Test-Time Training with KV Binding Is Secretly Linear Attention
Junchen Liu
Sven Elflein
Or Litany
Zan Gojcic
Ruilong Li
Abstract

Test-time training (TTT) with KV binding as sequence modeling layer is commonly interpreted as a form of online meta-learning that memorizes a key–value mapping at test time. However, our analysis reveals multiple phenomena that contradict this memorization-based interpretation. Motivated by these findings, we revisit the formulation of TTT and show that a broad class of TTT architectures can be expressed as a form of learned linear attention operator. Beyond explaining previously puzzling model behaviors, this perspective yields multiple practical benefits: it enables principled architectural simplifications, admits fully parallel formulations that preserve performance while improving efficiency, and provides a systematic reduction of diverse TTT variants to a standard linear attention form. Overall, our results reframe TTT not as test-time memorization, but as learned linear attention with enhanced representational capacity.

Machine Learning, ICML
1Introduction

Test-Time Training (TTT) has emerged as a powerful paradigm for dynamic model adaptation. Initially introduced to address distribution shift by updating model parameters on unlabeled test inputs, TTT has since evolved into a distinct architectural primitive (Sun et al., 2020). Recent work has increasingly framed TTT as an alternative to standard softmax attention in transformers, offering favorable properties like linear-time compute and constant memory usage during autoregressive inference (Zhang et al., 2025; Behrouz et al., 2024). Among the various TTT formulations, this paper focuses on TTT with KV binding (Sun et al., 2025; Zhang et al., 2025; Han et al., 2025; Behrouz et al., 2024), which optimizes a self-supervised key-value association objective in the inner loop, as opposed to end-to-end methods that backpropagate from the final task loss (Tandon et al., 2025). The prevailing interpretation of TTT casts it as a form of online meta-learning or memorization (Sun et al., 2025; Finn et al., 2017; Metz et al., 2018), where the “inner loop” dynamically constructs a temporary key-value (KV) map by optimizing a neural network (e.g., an MLP) on previously observed tokens. The subsequent inference step is viewed as querying this stored knowledge. This perspective has led to increased complexity in recent architectural designs, motivating the use of sophisticated optimizers, normalization schemes, and deep inner-loop networks (Zhang et al., 2025; Han et al., 2025; Behrouz et al., 2024, 2025a; Dalal et al., 2025), all explicitly designed to improve the fidelity of this “memorization”.

The Memorization Paradox.  Yet, this interpretation of TTT as “test-time memorization” can be directly contradicted by empirical evidence. If TTT would truly function by explicitly learning and retrieving key–value associations, its behavior would conform to basic principles of memory formation and optimization dynamics. Contrary to this expectation, we identify systematic anomalies that directly contradict this hypothesis:

• 

Distributional Asymmetry. Unlike standard attention, in which queries and keys share the same semantic space, converged TTT models exhibit a significant distributional mismatch between queries and keys.

• 

Replacing queries with keys. Replacing queries with keys for TTT models has negligible effect on the task performance, suggesting that queries do not play a functional retrieval role as in standard attention.

• 

Optimization vs. Performance. Counterintuitively, improvements in the inner loop, which can be interpreted as stronger “memorization”, do not guarantee better downstream performance.

• 

The Gradient Ascent Anomaly. Most strikingly, we find that replacing inner-loop gradient descent with gradient ascent always preserves, and in some cases even improves, task performance.

These observations collectively challenge the prevailing view of TTT as an online meta-learning or key-value memorization mechanism.

TTT is Secretly Linear Attention.  Motivated by these observations, and by prior work showing that TTT reduces to linear attention in the restricted case of a single linear inner-loop layer with zero initialization (Sun et al., 2025), we revisit the mathematical formulation of TTT. We show analytically that even TTT variants with complex fast-weight parameterizations (including multi-layer MLPs and momentum) can be equivalently rewritten as a form of learned linear attention operator (Katharopoulos et al., 2020).

Under this unified view, the inner loop does not perform “meta learning” in the conventional sense. Instead, it induces a structured, history-dependent mixing of query, key, and value vectors. This perspective resolves the empirical paradoxes identified above: gradient ascent preserves performance because sign inversions are absorbed into the learned value projection, and distributional symmetry between queries and keys is unnecessary because the mechanism operates as a feature mixer rather than a similarity-based retrieval system.

Practical Implications.  Unmasking TTT as Linear Attention is not just a theoretical exercise; it unlocks significant practical benefits. By adopting this perspective, we:

• 

Simplify. We show that many components introduced in prior TTT architectures, such as weight normalization and momentum, are often redundant.

• 

Parallelize. We derive a fully parallel form of TTT that achieves up to 4.0
×
 inference throughput on attention calculation while maintaining performance.

• 

Generalize. We provide systematic reduction of diverse TTT variants to a common linear attention form, suggesting that TTT is best understood as a flexible learned linear attention mechanism with enhanced representational capacity which opens up an increased design space.

2Related Work
2.1Linear Attention

Recurrent Neural Networks (RNNs) have recently gained renewed interest as efficient alternatives to standard Transformers (Vaswani et al., 2017). The introduction of linear attention (Katharopoulos et al., 2020) has accelerated advances in RNN-based architectures, leading to the incorporation of various mechanisms such as token-dependent decay factors (Gu et al., 2021; Smith et al., 2023; Orvieto et al., 2023; Peng et al., 2023; Sun et al., 2023) and, more recently, data-dependent decay (Qin et al., 2023, 2024; Peng et al., 2024; Gu and Dao, 2024; Dao and Gu, 2024; Zhang et al., 2024; Yang et al., 2024a). The data-dependent decay factor, termed the selective mechanism in Mamba (Gu and Dao, 2024; Dao and Gu, 2024), has been highlighted as crucial for strong in-context learning performance. DeltaNet (Schlag et al., 2021) conditions the update rule on both the current token and state for improved retrieval, and chunk-parallelization (Yang et al., 2024b) has enabled its efficient deployment in many recent architectures  (Yang et al., 2024a; Peng et al., 2025a; Behrouz et al., 2024; Zhong et al., 2025; Team et al., 2025; Peng et al., 2025b; Lei et al., 2025; Liu et al., 2024). Notably, DeltaNet and its variants are equivalent to TTT with a single linear layer and MSE loss (Yang et al., 2024a).

2.2Test-Time Training

Test-time training (TTT) broadly refers to methods that continue to update model parameters during inference. This concept was first introduced to address train-test distribution shift (Sun et al., 2020; Gandelsman et al., 2022), where models adapt to test data by optimizing a self-supervised objective at inference time. Subsequently, TTT has been explored for improving inference-time performance in specific applications such as 3D reconstruction (Chen et al., 2024, 2025; Yuan et al., 2025). More recently, TTT has been developed as a sequence modeling architecture with linear complexity, serving as an alternative to softmax attention in transformers (Sun et al., 2025; Wang et al., 2025; Zhang et al., 2025; Han et al., 2025; Dalal et al., 2025; Behrouz et al., 2024).

When used as a sequence modeling layer, TTT has two main variants: (1) methods that use a key-value binding loss (e.g., dot-product or MSE loss) as the inner-loop objective (Sun et al., 2025; Zhang et al., 2025; Han et al., 2025; Behrouz et al., 2024), referred to as TTT-KVB in prior work (Tandon et al., 2025), and (2) methods that perform end-to-end backpropagation through the inner loop from the final task loss (e.g., cross-entropy in language modeling), referred to as TTT-E2E (Tandon et al., 2025). This paper focuses on the first variant.

TTT has demonstrated effectiveness across diverse tasks, including language modeling (Sun et al., 2025; Wang et al., 2025; Zhang et al., 2025), video generation (Dalal et al., 2025; Zhang et al., 2025), novel view synthesis (Zhang et al., 2025), and image classification (Han et al., 2025). In this paradigm, part of the model parameters (known as fast weights (Hinton and Plaut, 1987)) are updated online at test time using a self-supervised key-value association loss, with the goal of memorizing history associations. Since this nested optimization is performed in-context, TTT is also referred to as in-context meta-learning (Finn et al., 2017; Metz et al., 2018) and is tightly related to the concept of fast weight programming (Schlag et al., 2021) and test-time scaling (Muennighoff et al., 2025; Snell et al., 2024).

The design space of TTT is rich and has been actively explored. LaCT (Zhang et al., 2025) improves hardware utilization through large chunk sizes. Based on the key-value memorization perspective, other works have explored advanced test-time optimizers (Behrouz et al., 2024; Zhang et al., 2025; Karami et al., 2025) and alternative regression targets (Han et al., 2025; Behrouz et al., 2025b). Notably, prior work has shown that when the inner loop consists of a single linear layer, TTT can be reinterpreted as a linear attention operator (Sun et al., 2025). In this work, we demonstrate that this interpretation extends to general TTT architectures with complex multi-layer MLPs as inner loops, and show that this perspective enables practical benefits including simplified formulations and efficient parallel implementations.

Figure 1:Inner-Loop Optimization vs. Performance. Increasing inner-loop iterations improves inner-loop loss but degrades task performance, contradicting the memorization-based interpretation of TTT. Experiments are based on LaCT (Zhang et al., 2025).
3Preliminary

We provide a brief overview of the TTT mechanism that this paper focuses on. In TTT, each sequence modeling layer maintains a set of fast weights 
𝑓
𝜃
 (typically a lightweight MLP) that is updated during both training and inference. Given an input sequence, tokens are first projected into keys 
𝐾
, values 
𝑉
, and queries 
𝑄
 (similar to standard attention). The core idea is to perform online gradient descent on 
𝑓
𝜃
 using a self-supervised key-value binding objective: for each token, the key 
𝑘
 serves as input and the value 
𝑣
 serves as the regression target, i.e., 
ℒ
=
‖
𝑓
𝜃
​
(
𝑘
)
−
𝑣
‖
2
 (or a dot-product loss variant). After updating 
𝜃
 with this objective, the query 
𝑞
 is passed through the updated function 
𝑓
𝜃
 to produce the output.

This mechanism is commonly interpreted as a storage-and-retrieval system (Sun et al., 2025; Finn et al., 2017): the inner-loop optimization “memorizes” key-value associations into 
𝑓
𝜃
, which are later “retrieved” by querying the learned function. Under this view, architectural capacity, optimizer selection, and the number of inner-loop steps are all motivated by achieving more faithful memorization of key–value associations.

This variant of test-time training, which optimizes a key-value binding objective in the inner loop, is referred to as TTT-KVB in prior work (Tandon et al., 2025). Among the various TTT formulations (Sun et al., 2025; Gandelsman et al., 2022; Sun et al., 2020; Tandon et al., 2025), this paper exclusively focuses on this variant.

4Empirical Contradictions to Memorization

As we show in this section, the empirical behavior of TTT models consistently violates the properties implied by the storage-and-retrieval interpretation.

4.1Better Inner Loss Leads to Worse Performance

Under a memorization-based interpretation, inner-loop loss serves as a natural proxy for memorization quality: lower loss indicates more accurate encoding of key–value mappings and should therefore improve task performance.

To test this property, we vary the number of inner-loop gradient steps at inference time, which usually yields lower inner-loop loss in pretrained TTT models (Zhang et al., 2025; Han et al., 2025), without changing the architecture, training data, or learned parameters.

Figure 1 shows that despite improved inner-loop fitting, downstream performance degrades consistently as the number of inner-loop steps increases. This inverse relationship holds across both LLMs and novel view synthesis (NVS) tasks. Such behavior directly contradicts the memorization-based interpretation of TTT, under which more accurate key–value fitting should be beneficial, or at least not harmful for the downstream performance. Instead, these results indicate that the inner loop affects model computation in a fundamentally different manner, inconsistent with conventional notions of test-time memory.

4.2TTT with Gradient Ascent

The degradation in downstream performance with improved inner-loop fitting raises a more fundamental question: is gradient-descent-based memorization in the inner loop necessary at all?

To test this, we replace gradient descent in the inner loop with gradient ascent, effectively flipping the sign of all fast-weight gradients. Under a memorization-based interpretation, this change should have a strong negative effect, as gradient ascent explicitly worsens the fit to the key–value regression objective.

Contrary to this expectation, across all evaluated models and tasks, TTT with gradient ascent performs comparably to, and in some cases even slightly better than, standard gradient descent (Table 1). This holds despite the fact that gradient ascent consistently increases the inner-loop loss.

Notably, in methods such as LaCT (Zhang et al., 2025), where the inner-loop loss is a Frobenius inner product, flipping the gradient is equivalent to negating the loss itself. The fact that TTT remains effective even under such an inverted objective provides additional evidence that TTT does not rely on memorizing a key–value mapping.

Table 1: Observations Contradicting the Storage-and-Retrieval Interpretation of TTT. Replacing gradient descent with ascent breaks the storage interpretation, and replacing queries with keys breaks the retrieval interpretation, yet task performance remains mostly unchanged. Experiments are base on LaCT (Zhang et al., 2025) and ViTTT (Han et al., 2025).
Model	Perplexity 
↓

(LaCT-LLM)	PSNR 
↑

(LaCT-NVS)	Top-1 Acc
↑

(ViTTT)
Baseline	16.43	25.94	79.34
Gradient Ascent	16.19	25.85	79.61
Replace 
𝑄
 with 
𝐾
 	16.18	25.95	79.18
4.3Distributional Asymmetry Between 
𝑄
 and 
𝐾
Figure 2: Distributional Asymmetry Between 
𝑄
 and 
𝐾
. t-SNE visualizations of 
(
𝑄
,
𝐾
)
 and 
(
𝑉
,
𝑂
)
 features in a pretrained LaCT (Zhang et al., 2025) model on the NVS task, showing that the TTT inner loop is evaluated out of distribution and thus does not perform reliable retrieval.

As we mentioned above, TTT is commonly interpreted as a storage-and-retrieval mechanism. In the previous subsections, we showed that the storage aspect of this interpretation is empirically inconsistent with model behavior. We now examine whether the retrieval aspect provides a plausible explanation.

During the TTT inner-loop update, the parametric function (e.g., MLP) is optimized using keys 
𝐾
 as inputs and values 
𝑉
 as regression targets. After this update, queries 
𝑄
 are feed into the same function to produce outputs 
𝑂
. For such a retrieval process to be effective, the distribution of 
𝑄
 must be close to that of the 
𝐾
 used during optimization. Otherwise, the parametric function is evaluated out-of-distribution, and its outputs are not expected to reliably lie on the manifold of 
𝑉
. In other words, meaningful retrieval requires substantial distributional overlap between 
𝑄
 and 
𝐾
.

To test this assumption, we analyze the distributions of 
𝑄
 and 
𝐾
 in a pretrained LaCT (Zhang et al., 2025) model on the NVS task. For each layer, we collect 
𝑄
 and 
𝐾
 vectors across all tokens and visualize their distributions using t-SNE (Maaten and Hinton, 2008). As shown in Figure 2, we observe a pronounced and consistent mismatch between 
𝑄
/
𝐾
 and 
𝑉
/
𝑂
 across layers, indicating that the parametric function optimized on keys, is systematically evaluated on out-of-distribution inputs when applied to queries, at both training and inference time. Under such conditions, the resulting outputs cannot be interpreted as reliable retrieval of stored key–value information, and this mismatch persists regardless of how accurately the inner loop fits the key–value regression objective.

4.4Replacing 
𝑄
 with 
𝐾

The distributional asymmetry between queries and keys again raises a more direct question: is the query representation 
𝑄
 necessary at all for TTT to function?

To test this, we perform a simple experiment in which we replace the query 
𝑄
 with the key 
𝐾
 when computing the TTT output. In standard attention mechanisms, such a substitution would be highly disruptive. Because attention weights depend on query–key similarity, replacing 
𝑄
 with 
𝐾
 typically leads to degenerate behavior dominated by self-similarity and a substantial drop in performance.

Contrary to the expectation, we observe little to no degradation in TTT’s performance. For both LaCT and ViTTT formulations performance remains comparable (Table 1). This insensitivity to the query representation contradicts a retrieval-based interpretation of TTT. If the inner loop relied on query-based access to stored information, replacing 
𝑄
 with 
𝐾
 should significantly alter model behavior. Instead, the observed invariance indicates that 
𝑄
 does not function as a meaningful query into a memorized key-value map.

Summary.  Taken together, these observations challenge the interpretation of test-time training as a storage-and-retrieval mechanism. Downstream behavior is largely insensitive to both the quality and direction of inner-loop optimization, and to the presence of a meaningful query signal. These failures suggest that in practice TTT does not operate as test-time memorization.

5TTT is secretly Linear Attention

If TTT is not functioning as a memorization mechanism, how should its behavior be understood? An important clue comes from prior work, which shows that in the restricted setting of a single linear inner-loop layer with zero initialization, TTT is exactly equivalent to linear attention (Sun et al., 2025). More generally, despite their apparent complexity, existing TTT variants share the defining computational properties of linear attention: linear-time computation and a constant-size state with respect to sequence length.

Motivated by this connection, we re-examine TTT by explicitly unrolling the inner-loop updates. We show analytically that TTT induces a linear attention–like operator in a general form, even when the inner loop consists of multi-layer non-linear mappings (Section 5.1). This perspective naturally explains our empirical observations that contradict the storage-and-retrieval interpretation (Section 5.2). Finally, we rewrite two representative variants of TTT, LaCT (Zhang et al., 2025) and ViTTT (Han et al., 2025), in their linear attention form (Sections 5.3 and 5.4). Detailed derivations are deferred to the appendix.

5.1General Form
Theorem 5.1 (Linearization of Inner-Loop Updates). 

Consider a TTT model whose inner-loop function has a linear, bias-free final layer,

	
𝑓
​
(
𝑥
)
=
𝜙
​
(
𝑥
;
Θ
)
​
𝑊
,
	

where 
𝜙
​
(
𝑥
;
Θ
)
∈
ℝ
𝐷
h
 denotes the hidden representation of the inner-loop function with parameters 
Θ
, and 
𝑊
∈
ℝ
𝐷
h
×
𝐷
out
 is the weight matrix of the final layer. Suppose that at step 
𝑡
, the inner loop performs one step of gradient descent on an objective 
ℒ
 with learning rate 
𝜂
, using key input 
𝑘
, updating all trainable parameters,

	
(
𝑊
𝑡
+
1
,
Θ
𝑡
+
1
)
=
(
𝑊
𝑡
,
Θ
𝑡
)
−
𝜂
​
∇
(
𝑊
𝑡
,
Θ
𝑡
)
ℒ
​
(
𝑓
𝑡
​
(
𝑘
)
)
,
	

where 
𝜙
𝑡
​
(
⋅
)
≜
𝜙
​
(
⋅
;
Θ
𝑡
)
 and 
𝑓
𝑡
​
(
⋅
)
≜
𝜙
𝑡
​
(
⋅
)
​
𝑊
𝑡
. Then, for any query 
𝑞
, the output after the update can be written as

	
𝑜
=
𝜙
𝑡
+
1
​
(
𝑞
)
​
(
𝑊
𝑡
+
𝜙
𝑡
​
(
𝑘
)
⊤
​
𝑔
𝑡
​
(
𝑘
)
)
,
𝑔
𝑡
​
(
𝑘
)
≜
−
𝜂
​
∂
ℒ
∂
𝑓
𝑡
​
(
𝑘
)
.
	

This expression is a linear attention operator of the form

	
𝑜
=
𝑞
^
​
(
𝑆
0
+
𝑘
^
⊤
​
𝑣
^
)
,
	

where

	
𝑞
^
=
𝜙
𝑡
+
1
​
(
𝑞
)
,
𝑘
^
=
𝜙
𝑡
​
(
𝑘
)
,
𝑣
^
=
𝑔
𝑡
​
(
𝑘
)
,
𝑆
0
=
𝑊
𝑡
.
	

The complete proof is provided in Appendix B.

Theorem 5.2 (Unrolling Inner-Loop Updates). 

Given a sequence of query–key pairs 
{
(
𝑞
0
,
𝑘
0
)
,
(
𝑞
1
,
𝑘
1
)
,
…
,
(
𝑞
𝑡
,
𝑘
𝑡
)
}
, suppose the TTT model performs one gradient descent step per input in sequence. By repeated application of Theorem 5.1, the parameters after processing token 
𝑡
 are

	
(
𝑊
𝑡
+
1
,
Θ
𝑡
+
1
)
=
(
𝑊
0
,
Θ
0
)
−
𝜂
​
∑
𝑖
=
0
𝑡
∇
(
𝑊
𝑖
,
Θ
𝑖
)
ℒ
​
(
𝑓
𝑖
​
(
𝑘
𝑖
)
)
.
	

Evaluating the TTT model on query 
𝑞
𝑡
 yields

	
𝑜
𝑡
	
=
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
​
𝑊
𝑡
+
1
	
		
=
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
​
(
𝑊
0
+
∑
𝑖
=
0
𝑡
𝜙
𝑖
​
(
𝑘
𝑖
)
⊤
​
𝑔
𝑖
​
(
𝑘
𝑖
)
)
,
	

where 
𝑔
𝑖
​
(
𝑘
𝑖
)
 is defined as in Theorem 5.1. This corresponds to the extended linear attention form on sequential inputs.

	
𝑜
𝑡
=
𝑞
^
𝑡
​
(
𝑆
0
+
∑
𝑖
=
0
𝑡
𝑘
^
𝑖
⊤
​
𝑣
^
𝑖
)
.
	

The complete proof is provided in Appendix C.

Next, we extend Theorem 5.2 to the case where the inner-loop employs gradient descent with momentum.

Theorem 5.3 (Gradient Descent with Momentum). 

Given the momentum-augmented gradient accumulator defined as

	
(
Δ
​
𝑊
𝑡
,
Δ
​
Θ
𝑡
)
=
∇
(
𝑊
,
Θ
)
ℒ
​
(
𝑓
𝑡
​
(
𝑘
𝑡
)
)
+
𝛼
𝑡
​
(
Δ
​
𝑊
𝑡
−
1
,
Δ
​
Θ
𝑡
−
1
)
,
	

where 
𝛼
𝑡
 denotes the (possibly token-dependent) momentum factor at step 
𝑡
. The model parameters are then updated according to

	
(
𝑊
𝑡
+
1
,
Θ
𝑡
+
1
)
=
(
𝑊
𝑡
,
Θ
𝑡
)
−
𝜂
​
(
Δ
​
𝑊
𝑡
,
Δ
​
Θ
𝑡
)
.
	

Define the cumulative momentum coefficient as

	
𝛽
𝑖
𝑗
≜
{
∏
𝑠
=
𝑖
+
1
𝑗
𝛼
𝑠
	
if 
​
𝑖
<
𝑗
,


1
	
if 
​
𝑖
=
𝑗
.
	

Unrolling this recurrence and evaluating the TTT model on query 
𝑞
𝑡
 yields

	
𝑜
𝑡
	
=
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
​
𝑊
𝑡
+
1
	
		
=
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
​
(
𝑊
0
+
∑
𝑖
=
0
𝑡
𝜙
𝑖
​
(
𝑘
𝑖
)
⊤
​
𝑚
𝑖
​
(
𝑘
𝑖
)
)
,
	

which induces a linear-attention–like form identical to that of Theorem 5.2, with the effective value vector being a momentum-weighted sum:

	
𝑣
^
𝑖
=
𝑚
𝑖
​
(
𝑘
𝑖
)
≜
𝑔
𝑖
​
(
𝑘
𝑖
)
⋅
∑
𝑗
=
𝑖
𝑡
𝛽
𝑖
𝑗
.
	

The complete proof is provided in Appendix D.

5.2Explanation of TTT Empirical Behaviors

Having shown that TTT can be rewritten as a linear attention operator (Theorem 5.1 - 5.3), we can now revisit the empirical behaviors that appeared contradictory under the memorization-based interpretation. The linear-attention perspective provides a unified and mechanistic explanation.

More Inner-Loop Steps.  According to Theorem 5.1, the inner loop does not perform storage of key–value information, but instead defines a learnable mapping that transforms the original inputs into the effective query, key, and value representations. This mapping depends on inner-loop hyperparameters, including the number of optimization steps. Increasing the number of inner-loop iterations at inference time therefore induces an attention operator different to the one used during training, naturally leading to degraded performance due to train–test mismatch rather than improved memorization.

Gradient Ascent in the Inner Loop.  Under the same formulation, replacing gradient descent with gradient ascent simply flips the sign of the effective value vector 
𝑔
𝑡
. Since this sign is absorbed into the learned attention operator and the mapping itself is optimized under the downstream objective, the model adapts to this change. This explains why TTT remains effective under gradient ascent, despite the absence of a meaningful memorization objective.

Distributional Asymmetry Between 
𝑄
 and 
𝐾
.  The linear-attention view clarifies why similarity between 
𝑞
 and 
𝑘
 is not required. In TTT, 
𝑞
 and 
𝑘
 influence different components of the attention operator: 
𝑞
 determines the effective query via 
𝜙
𝑡
+
1
​
(
𝑞
)
, while 
𝑘
 determines the effective key and value via 
𝜙
𝑡
​
(
𝑘
)
 and 
𝑔
𝑡
​
(
𝑘
)
, respectively. They are therefore intermediate features rather than symmetric query–key representations, making distributional mismatch expected rather than pathological.

Replacing 
𝑄
 with 
𝐾
.  Finally, replacing 
𝑞
 with 
𝑘
 does not collapse the attention mechanism because the effective query and key remain distinct: 
𝜙
𝑡
+
1
​
(
𝑘
)
 versus 
𝜙
𝑡
​
(
𝑘
)
. Since 
𝜙
 is learnable and evaluated at different parameter states, the model can map the same input to different representations, preserving attention functionality.

Summary.  Viewed through the lens of linear attention, the inner loop of TTT parameterizes a structured linear-form attention operator rather than performing test-time storage and retrieval. Under this perspective, the observed empirical behaviors follow naturally from representation learning and train–test consistency considerations.

5.3Example: LaCT as Linear Attention

We now show how a representative instantiation of TTT formula, LaCT (Zhang et al., 2025), can be rewritten in the form of linear attention.

LaCT adopts a bias-free SwiGLU MLP (Shazeer, 2020) as its inner-loop mapping, parameterized by three learnable weight matrices 
𝑊
0
,
𝑊
2
∈
ℝ
𝐷
ℎ
×
𝐷
𝑘
 and 
𝑊
1
∈
ℝ
𝐷
ℎ
×
𝐷
𝑣
:

	
𝑓
​
(
𝑥
)
=
(
silu
​
(
𝑥
​
𝑊
0
)
⊙
(
𝑥
​
𝑊
2
)
)
​
𝑊
1
.
	

The inner-loop objective is defined via the Frobenius inner product

	
ℒ
​
(
𝑓
​
(
𝑘
)
,
𝑣
)
=
−
⟨
𝑓
​
(
𝑘
)
,
𝑣
⟩
.
	

At each token, LaCT performs gradient descent with per-token learning rate 
𝜂
𝑡
, momentum 
𝛼
𝑡
, and gradient orthogonalization 
ℳ
​
(
⋅
)
 inspired by Muon (Jordan et al., 2024):

	
𝑊
𝑖
,
𝑡
+
1
=
𝑊
𝑖
,
𝑡
−
𝜂
𝑡
​
ℳ
​
(
Δ
​
𝑊
𝑖
,
𝑡
)
,
	
	
Δ
​
𝑊
𝑖
,
𝑡
=
∇
𝑊
𝑖
,
𝑡
ℒ
​
(
𝑓
𝑡
​
(
𝑘
𝑡
)
,
𝑣
𝑡
)
,
	

for 
𝑖
∈
{
0
,
1
,
2
}
.

Following Theorem 5.3, the inner-loop model at step 
𝑡
 can be written as

	
𝑓
𝑡
​
(
𝑥
)
=
𝜙
𝑡
​
(
𝑥
)
​
𝑊
1
,
𝑡
,
𝜙
𝑡
​
(
𝑥
)
=
silu
​
(
𝑥
​
𝑊
0
,
𝑡
)
⊙
(
𝑥
​
𝑊
2
,
𝑡
)
.
	

Evaluating the updated model on query 
𝑞
𝑡
 yields

	
𝑜
𝑡
	
=
𝑓
𝑡
+
1
​
(
𝑞
𝑡
)
	
		
=
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
​
𝑊
1
,
𝑡
+
1
	
		
=
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
​
(
𝑊
1
,
0
+
∑
𝑖
=
0
𝑡
ℳ
​
(
𝜙
𝑖
​
(
𝑘
𝑖
)
⊤
​
𝑚
𝑖
)
)
,
		
(1)

where

	
𝑚
𝑖
​
(
𝑘
𝑖
)
≜
𝑣
𝑖
⋅
𝜂
𝑖
​
∑
𝑗
=
𝑖
𝑡
𝛽
𝑖
𝑗
.
	

This expression reveals that the inner loop of LaCT is effectively a linear attention–like operator, with 
𝜙
𝑖
​
(
𝑘
𝑖
)
 and 
𝑚
𝑖
​
(
𝑘
𝑖
)
 playing the roles of keys and values, and 
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
 acting as the query vector. Note that LaCT also applies weight normalization after each update, which we omit here for simplicity. See Appendix E for more details.

5.4Example: ViTTT as Linear Attention

We next show that another instantiation of test-time training, ViTTT (Han et al., 2025), can likewise be rewritten in the form of linear attention.

ViTTT employs fast weights consisting of two independent components: (i) a simplified gated linear unit (GLU), and (ii) a depthwise convolution layer. These components are updated independently in the inner loop. We show that each admits a linear-attention interpretation, implying that ViTTT as a whole falls within the same framework.

GLU component.  The GLU is defined as

	
𝑓
​
(
𝑥
)
=
silu
​
(
𝑥
​
𝑊
0
)
⊙
(
𝑥
​
𝑊
1
)
,
	

where 
𝑊
0
 and 
𝑊
1
 are fast weights updated via gradient descent. As in LaCT, the inner-loop loss is defined using a Frobenius inner product. Following a derivation analogous to previous sections (see Appendix F), evaluating the updated GLU on a query 
𝑞
𝑡
 yields

	
𝜙
​
(
𝑥
)
=
silu
​
(
𝑥
​
𝑊
0
)
,
	
	
𝑜
𝑡
	
=
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
⊙
(
𝑞
𝑡
​
(
𝑊
1
+
𝑘
𝑡
⊤
​
(
𝑣
𝑡
⊙
𝜙
𝑡
​
(
𝑘
𝑡
)
)
)
)
.
	

This expression takes the form of linear attention, with 
𝜙
𝑡
​
(
𝑘
𝑡
)
 acting as a multiplicative gate on values and 
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
 gating the final output.

Depthwise convolution component.  ViTTT additionally includes a 
3
×
3
 depthwise convolution layer with fast weights, updated in the inner loop. As convolution is effectively sliding window linear layer, this TTT component is equivalent to a sliding window linear attention. A formal derivation is provided in Appendix G.

Discussion.  Since both fast-weight components in ViTTT admit linear-attention formulations, their combination also induces a linear-attention–like operator. This establishes ViTTT as another concrete instance of test-time training whose behavior is more naturally understood through the lens of linear attention.

6Practical Implications
Table 2: Ablation Trajectory Reducing TTT to Standard Linear Attention. By progressively simplifying complex TTT formulations into standard linear attention (Variant 6), we quantify the contribution of each design component in TTT. Variant 1 achieves the best performance across all three tasks. Variants 2–6 admit parallel implementations. We additionally report the inference throughput of each variant’s TTT layer on the LLM task. † denotes ablations that do not apply, in which case performance matches the preceding variant. * indicates that ViTTT does not use gradient orthogonalization, so we ablate gradient normalization instead.
Alias	Description of TTT Inner-loop Designs	Perplexity 
↓

(LaCT-LLM)	PSNR 
↑

(LaCT-NVS)	Top-1 Acc
↑

(ViTTT)	Tokens Per Sec
↑

(Recurrent Impl.)	Tokens Per Sec
↑

(Parallel Impl.)
Baseline	LaCT (Zhang et al., 2025) / ViTTT (Han et al., 2025)	16.43	25.94	79.34%	4.30M	N/A
Variant 1	Baseline w/ only update the last layer parameters	15.93	25.97	79.63%	10.60M	N/A
Variant 2	Variant 1 w/ remove weight normalization	16.31	25.93	79.63%†	11.02M	30.18M
Variant 3	Variant 2 w/ multi-layer MLP 
→
 singel linear layer	16.23	25.71	79.39%	12.95M	49.69M
Variant 4	Variant 3 w/ remove per-token learnable lr	16.12	25.70	79.39%†	13.31M	53.99M
Variant 5	Variant 4 w/ remove momentum in SGD	15.97	25.70†	79.39%†	14.40M	57.28M
Variant 6	Variant 5 w/ remove gradient orthogonalization	16.80	25.73	79.54%*	89.67M	124.6M
Figure 3:Perplexity Metric for Ablation om LaCT-LLM. Evaluated on 2.5B tokens from the Book-3 dataset.

Unmasking TTT as linear attention is not merely a theoretical exercise but yields concrete practical benefits. This perspective reveals a systematic trajectory for reducing complex TTT formulations into linear attention, along which we observe that several commonly adopted design choices, such as per-token learnable learning rates, weight normalization, are not essential to final performance. By clarifying which components are truly valuable and which are unnecessary, this view enables substantial simplification of TTT formulations. Moreover, recognizing TTT as linear attention makes explicit that the seemingly recurrent inner-loop updates admit a parallel formulation, leading to significant efficiency gains in both training and inference. All experiments in this section use the official implementations of LaCT (Zhang et al., 2025) for the LLM and NVS tasks, and ViTTT (Han et al., 2025) for the image recognition task. Detailed experimental settings are provided in Appendix A.

6.1Reduce TTT to Linear Attention

Viewing TTT through the lens of linear attention reveals that several design choices that were justified under a storage-and-retrieval interpretation are in fact redundant or optional. This perspective enables a principled ablation path that progressively simplifies complex TTT variants such as LaCT and ViTTT into standard linear attention. Below, we outline this reduction trajectory and analyze the role of each component.

Step 1. Update only the last-layer parameters.  As shown in Theorem 5.1, the effective query and key vectors are 
𝜙
𝑡
+
1
​
(
𝑞
)
 and 
𝜙
𝑡
​
(
𝑘
)
, where 
𝜙
𝑡
​
(
⋅
)
≜
𝜙
​
(
⋅
;
Θ
𝑡
)
. Updating 
Θ
𝑡
 within the inner loop makes 
𝜙
𝑡
 a dynamic kernel that is difficult to unroll analytically. If the inner loop instead updates only the final-layer parameter, 
Θ
 remains fixed during TTT, and 
𝜙
​
(
⋅
)
≜
𝜙
​
(
⋅
;
Θ
)
 becomes a static function with learnable parameters. From the linear-attention perspective, 
𝜙
​
(
⋅
)
 then acts as a learnable kernel function with effective queries and keys given by 
𝜙
​
(
𝑞
)
 and 
𝜙
​
(
𝑘
)
.

Step 2. Remove weight normalization.  LaCT applies weight normalization to all learnable parameters 
Θ
𝑡
,
𝑊
𝑡
 after each inner-loop update. After Step 1, normalization on 
Θ
𝑡
 is a no-op since 
Θ
𝑡
 is fixed. Normalizing the final-layer parameter 
𝑊
𝑡
, which corresponds to the state in the linear attention view, is therefore equivalent to normalizing the state 
𝑆
𝑡
 (see Appendix E.4). As such normalization is uncommon in linear attention literature, we remove it for ablation. Notably, after this step the TTT formulation becomes fully parallelizable, as discussed in Section 6.2.

Step 3. Multi-layer MLP 
→
 single linear layer.  Several TTT variants (Han et al., 2025; Behrouz et al., 2024) employ deeper inner-loop MLPs, but report inconsistent empirical gains. From a linear attention perspective, increasing MLP depth simply induces a more complex kernel function 
𝜙
​
(
⋅
)
 over queries and keys. When 
𝑞
 and 
𝑘
 already have sufficient representational capacity, this added complexity is unlikely to help. We therefore reduce the multi-layer MLP to a single linear layer, effectively removing 
𝜙
​
(
⋅
)
 altogether. This exposes the true queries and keys as 
𝑞
^
=
𝑞
 and 
𝑘
^
=
𝑘
, bringing the formulation closer to basic linear attention.

Step 4. Remove per-token learning rates.  Many TTT methods (Behrouz et al., 2024; Zhang et al., 2025) introduce a per-token learnable learning rate 
𝜂
𝑡
. As shown in Section 5.3, with Frobenius dot product as inner loss this can be absorbed into the learnable 
𝑣
𝑡
, indicating it’s functionally redundant. Consistent with our finding, ViTTT empirically finds that a constant learning rate of 
1.0
 suffices.

Step 5. Remove momentum in SGD.  As shown in Theorem 5.3, adding momentum to the inner-loop SGD update, as done in LaCT and related methods (Behrouz et al., 2024), only alters the effective value 
𝑣
^
, from the instantaneous gradient 
𝑔
𝑡
​
(
𝑘
)
 to a momentum-weighted sum of past gradients. From the linear attention perspective, this corresponds to remixing historical key–value contributions into a single value vector. Since both keys and values are already learnable, this additional mixing is unlikely to provide meaningful benefit, thus we remove momentum for ablation. Notably, for both LaCT and ViTTT, 
𝑔
𝑡
​
(
𝑘
)
=
−
𝑣
 and removing momentum recovers 
𝑣
^
=
𝑣
, exposing the true value vector.

Step 6. Remove gradient orthogonalization.  As discussed in Section 5.3, LaCT optionally applies gradient orthogonalization 
ℳ
​
(
Δ
​
𝑊
)
. Under the linear attention reformulation, this corresponds to applying an operator to the state update 
ℳ
​
(
𝑘
^
⊤
​
𝑣
)
. We remove this operation for ablation. After this final step, Both LaCT and ViTTT reduce exactly to standard linear attention: 
𝑜
=
𝑞
​
(
𝑊
+
∑
𝑖
𝑘
𝑖
⊤
​
𝑣
𝑖
)
.

Results.  We progressively apply the above ablations to LaCT on the LLM and NVS tasks, and to ViTTT on the image recognition task, with results summarized in Table 2. Surprisingly, restricting the inner loop to update only the final MLP layer consistently yields the best overall performance across tasks, suggesting that many of the more complex design choices are unnecessary, or even detrimental. Most other components contribute only marginally to performance, with two notable exceptions: deeper MLPs are beneficial for the NVS task, while gradient orthogonalization improves performance on the LLM task. Overall, reducing the full TTT formulation to a basic linear attention operator (Variant 6) results in only minor performance degradation (
+
0.4
 perplexity on LLM and 
−
0.2
 dB on NVS). For the LLM task, Table 2 reports perplexity at 32k sequence length, with results across other lengths shown in Figure 3.

Figure 4: Training loss vs. wall-clock time on LaCT-LLM. We compare the original LaCT-TTT with both parallel and recurrent form of Variant 2. The parallel form achieves a 
1.19
×
 end-to-end speedup while maintaining comparable convergence.
6.2Parallel Form of TTT

Existing TTT variants are typically implemented in a recurrent manner, reflecting their original storage-and-retrieval interpretation. However, as we have shown that TTT can alternatively be viewed as a form of linear attention, a natural question arises: can TTT admit a parallel formulation that enables more efficient implementation? We show that under certain conditions, corresponding to Variants 2–6 in our ablation, such a parallel form indeed exists.

The key insight is that when weight normalization is removed and only the final-layer parameters are updated, the state update becomes associative. In this setting, the kernel function 
𝜙
𝑡
​
(
⋅
)
≜
𝜙
​
(
⋅
;
Θ
𝑡
)
 is static and independent of sequence history, allowing the recurrence in Theorem 5.3 to be computed via a parallel prefix scan rather than sequential token-by-token updates.

We implement this parallel formulation for LaCT on the LLM task. As shown in Table 2, switching from the recurrent to the parallel implementation improves the inference throughput of the TTT layer by up to 
4.0
×
 (measured in tokens per second, single batch). Combined with the simplifications from Step 1 and Step 2, this yields a 
1.19
×
 end-to-end training speedup without degrading model quality, as shown in Figure 4.

We provide the full parallel formulation and a proof of equivalence to the sequential recurrence in Appendix H. In Appendix I, we further show that introducing weight normalization or dynamic kernel functions breaks associativity, thereby preventing parallelization.

7Conclusion

In this work, we challenge the prevailing view of TTT as a mechanism for test-time memorization of key–value mappings. Through systematic empirical analysis, we identify several anomalies—including gradient ascent behavior, distributional asymmetry between queries and keys, and the lack of correlation between inner-loop convergence and downstream performance—that are fundamentally incompatible with a memorization-based interpretation.

We provide an alternative explanation by showing that TTT, even with complex inner loops involving multi-layer MLPs and momentum-based optimizers, can be analytically rewritten as a form of linear attention operator. Under this view, the inner loop does not perform meta learning in the conventional sense, but instead parameterizes a structured mixing of queries, keys, and values. Viewing TTT through the lens of linear attention not only resolves the observed paradoxes, but also enables principled simplifications, parallel implementations with improved efficiency, and a unified framework for understanding TTT variants.

Our analysis applies to most current TTT formulations but is limited to settings where the inner-loop final layer is linear and bias-free. Extending these insights to nonlinear final layers, and exploring deeper connections between TTT and modern linear attention mechanisms in both directions, remain important avenues for future work.

References
A. Behrouz, Z. Li, P. Kacham, M. Daliri, Y. Deng, P. Zhong, M. Razaviyayn, and V. Mirrokni (2025a)	Atlas: learning to optimally memorize the context at test time.arXiv preprint arXiv:2505.23735.Cited by: §1.
A. Behrouz, M. Razaviyayn, P. Zhong, and V. Mirrokni (2025b)	It’s all connected: a journey through test-time memorization, attentional bias, retention, and online optimization.arXiv preprint arXiv:2504.13173.Cited by: §2.2.
A. Behrouz, P. Zhong, and V. Mirrokni (2024)	Titans: learning to memorize at test time.arXiv preprint arXiv:2501.00663.Cited by: §1, §2.1, §2.2, §2.2, §2.2, §6.1, §6.1, §6.1.
X. Chen, Y. Chen, Y. Xiu, A. Geiger, and A. Chen (2025)	Ttt3r: 3d reconstruction as test-time training.arXiv preprint arXiv:2509.26645.Cited by: §2.2.
Y. Chen, J. Wang, Z. Yang, S. Manivasagam, and R. Urtasun (2024)	G3r: gradient guided generalizable reconstruction.In European Conference on Computer Vision,pp. 305–323.Cited by: §2.2.
K. Dalal, D. Koceja, J. Xu, Y. Zhao, S. Han, K. C. Cheung, J. Kautz, Y. Choi, Y. Sun, and X. Wang (2025)	One-minute video generation with test-time training.In Proceedings of the Computer Vision and Pattern Recognition Conference,pp. 17702–17711.Cited by: §1, §2.2, §2.2.
T. Dao and A. Gu (2024)	Transformers are ssms: generalized models and efficient algorithms through structured state space duality.In Proceedings of the 41st International Conference on Machine Learning,pp. 10041–10071.Cited by: §2.1.
J. Deng, W. Dong, R. Socher, L. Li, K. Li, and L. Fei-Fei (2009)	Imagenet: a large-scale hierarchical image database.In 2009 IEEE conference on computer vision and pattern recognition,pp. 248–255.Cited by: Appendix A.
C. Finn, P. Abbeel, and S. Levine (2017)	Model-agnostic meta-learning for fast adaptation of deep networks.In International conference on machine learning,pp. 1126–1135.Cited by: §1, §2.2, §3.
Y. Gandelsman, Y. Sun, X. Chen, and A. Efros (2022)	Test-time training with masked autoencoders.Advances in Neural Information Processing Systems 35, pp. 29374–29385.Cited by: §2.2, §3.
L. Gao, S. Biderman, S. Black, L. Golding, T. Hoppe, C. Foster, J. Phang, H. He, A. Thite, N. Nabeshima, et al. (2020)	The pile: an 800gb dataset of diverse text for language modeling.arXiv preprint arXiv:2101.00027.Cited by: Appendix A.
A. Gu and T. Dao (2024)	Mamba: linear-time sequence modeling with selective state spaces.In First conference on language modeling,Cited by: §2.1.
A. Gu, K. Goel, and C. Ré (2021)	Efficiently modeling long sequences with structured state spaces.arXiv preprint arXiv:2111.00396.Cited by: §2.1.
D. Han, Y. Li, T. Li, Z. Cao, Z. Wang, J. Song, Y. Cheng, B. Zheng, and G. Huang (2025)	ViT3: unlocking test-time training in vision.arXiv preprint arXiv:2512.01643.Cited by: Appendix A, Appendix F, Appendix G, §1, §2.2, §2.2, §2.2, §2.2, §4.1, Table 1, Table 1, §5.4, §5, §6.1, Table 2, §6.
G. E. Hinton and D. C. Plaut (1987)	Using fast weights to deblur old memories.In Proceedings of the ninth annual conference of the Cognitive Science Society,pp. 177–186.Cited by: §2.2.
K. Jordan, Y. Jin, V. Boza, J. You, F. Cesista, L. Newhouse, and J. Bernstein (2024)	Muon: an optimizer for hidden layers in neural networks.External Links: LinkCited by: §E.1, §5.3.
M. Karami, R. Pascanu, and V. Mirrokni (2025)	Lattice: learning to efficiently compress the memory.arXiv preprint arXiv:2504.05646.Cited by: §2.2.
A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret (2020)	Transformers are rnns: fast autoregressive transformers with linear attention.In International conference on machine learning,pp. 5156–5165.Cited by: §1, §2.1.
J. Lei, D. Zhang, and S. Poria (2025)	Error-free linear attention is a free lunch: exact solution from continuous-time dynamics.arXiv preprint arXiv:2512.12602.Cited by: §2.1.
B. Liu, R. Wang, L. Wu, Y. Feng, P. Stone, and Q. Liu (2024)	Longhorn: state space models are amortized online learners.arXiv preprint arXiv:2407.14207.Cited by: §2.1.
L. v. d. Maaten and G. Hinton (2008)	Visualizing data using t-sne.Journal of machine learning research 9 (Nov), pp. 2579–2605.Cited by: §4.3.
L. Metz, N. Maheswaranathan, B. Cheung, and J. Sohl-Dickstein (2018)	Meta-learning update rules for unsupervised representation learning.arXiv preprint arXiv:1804.00222.Cited by: §1, §2.2.
N. Muennighoff, Z. Yang, W. Shi, X. L. Li, L. Fei-Fei, H. Hajishirzi, L. Zettlemoyer, P. Liang, E. Candès, and T. B. Hashimoto (2025)	S1: simple test-time scaling.In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing,pp. 20286–20332.Cited by: §2.2.
A. Orvieto, S. L. Smith, A. Gu, A. Fernando, C. Gulcehre, R. Pascanu, and S. De (2023)	Resurrecting recurrent neural networks for long sequences.In International Conference on Machine Learning,pp. 26670–26698.Cited by: §2.1.
G. Penedo, H. Kydlíček, A. Lozhkov, M. Mitchell, C. A. Raffel, L. Von Werra, T. Wolf, et al. (2024)	The fineweb datasets: decanting the web for the finest text data at scale.Advances in Neural Information Processing Systems 37, pp. 30811–30849.Cited by: Appendix A.
B. Peng, E. Alcaide, Q. Anthony, A. Albalak, S. Arcadinho, S. Biderman, H. Cao, X. Cheng, M. Chung, L. Derczynski, et al. (2023)	RWKV: reinventing rnns for the transformer era.In Findings of the Association for Computational Linguistics: EMNLP 2023,pp. 14048–14077.Cited by: §2.1.
B. Peng, D. Goldstein, Q. Anthony, A. Albalak, E. Alcaide, S. Biderman, E. Cheah, X. Du, T. Ferdinan, H. Hou, et al. (2024)	Eagle and finch: rwkv with matrix-valued states and dynamic recurrence.arXiv preprint arXiv:2404.05892.Cited by: §2.1.
B. Peng, R. Zhang, D. Goldstein, E. Alcaide, X. Du, H. Hou, J. Lin, J. Liu, J. Lu, W. Merrill, et al. (2025a)	Rwkv-7” goose” with expressive dynamic state evolution.arXiv preprint arXiv:2503.14456.Cited by: §2.1.
L. Peng, A. Chattopadhyay, L. Zancato, E. Nunez, W. Xia, and S. Soatto (2025b)	Gated kalmanet: a fading memory layer through test-time ridge regression.arXiv preprint arXiv:2511.21016.Cited by: §2.1.
Z. Qin, S. Yang, W. Sun, X. Shen, D. Li, W. Sun, and Y. Zhong (2024)	Hgrn2: gated linear rnns with state expansion.arXiv preprint arXiv:2404.07904.Cited by: §2.1.
Z. Qin, S. Yang, and Y. Zhong (2023)	Hierarchically gated recurrent neural network for sequence modeling.Advances in Neural Information Processing Systems 36, pp. 33202–33221.Cited by: §2.1.
I. Schlag, K. Irie, and J. Schmidhuber (2021)	Linear transformers are secretly fast weight programmers.In International conference on machine learning,pp. 9355–9366.Cited by: §2.1, §2.2.
N. Shazeer (2020)	Glu variants improve transformer.arXiv preprint arXiv:2002.05202.Cited by: §5.3.
J. T. Smith, A. Warrington, and S. W. Linderman (2023)	Simplified state space layers for sequence modeling.In International Conference on Learning Representations (ICLR),Cited by: §2.1.
C. Snell, J. Lee, K. Xu, and A. Kumar (2024)	Scaling llm test-time compute optimally can be more effective than scaling model parameters.arXiv preprint arXiv:2408.03314.Cited by: §2.2.
Y. Sun, X. Li, K. Dalal, J. Xu, A. Vikram, G. Zhang, Y. Dubois, X. Chen, X. Wang, S. Koyejo, et al. (2025)	Learning to (learn at test time): rnns with expressive hidden states.In International Conference on Machine Learning,Cited by: §1, §1, §2.2, §2.2, §2.2, §2.2, §3, §3, §5.
Y. Sun, X. Wang, Z. Liu, J. Miller, A. Efros, and M. Hardt (2020)	Test-time training with self-supervision for generalization under distribution shifts.In International conference on machine learning,pp. 9229–9248.Cited by: §1, §2.2, §3.
Y. Sun, L. Dong, S. Huang, S. Ma, Y. Xia, J. Xue, J. Wang, and F. Wei (2023)	Retentive network: a successor to transformer for large language models.arXiv preprint arXiv:2307.08621.Cited by: §2.1.
A. Tandon, K. Dalal, X. Li, D. Koceja, M. Rød, S. Buchanan, X. Wang, J. Leskovec, S. Koyejo, T. Hashimoto, et al. (2025)	End-to-end test-time training for long context.arXiv preprint arXiv:2512.23675.Cited by: §1, §2.2, §3.
K. Team, Y. Zhang, Z. Lin, X. Yao, J. Hu, F. Meng, C. Liu, X. Men, S. Yang, Z. Li, et al. (2025)	Kimi linear: an expressive, efficient attention architecture.arXiv preprint arXiv:2510.26692.Cited by: §2.1.
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin (2017)	Attention is all you need.Advances in neural information processing systems 30.Cited by: §2.1.
K. A. Wang, J. Shi, and E. B. Fox (2025)	Test-time regression: a unifying framework for designing sequence models with associative memory.arXiv preprint arXiv:2501.12352.Cited by: §2.2, §2.2.
S. Yang, B. Wang, Y. Shen, R. Panda, and Y. Kim (2024a)	Gated linear attention transformers with hardware-efficient training.In Proceedings of the 41st International Conference on Machine Learning,pp. 56501–56523.Cited by: §2.1.
S. Yang, B. Wang, Y. Zhang, Y. Shen, and Y. Kim (2024b)	Parallelizing linear transformers with the delta rule over sequence length.Advances in neural information processing systems 37, pp. 115491–115522.Cited by: §2.1.
Y. Yuan, Q. Shen, S. Wang, X. Yang, and X. Wang (2025)	Test3r: learning to reconstruct 3d at test time.In The Thirty-ninth Annual Conference on Neural Information Processing Systems,Cited by: §2.2.
T. Zhang, S. Bi, Y. Hong, K. Zhang, F. Luan, S. Yang, K. Sunkavalli, W. T. Freeman, and H. Tan (2025)	Test-time training done right.arXiv preprint arXiv:2505.23884.Cited by: Appendix A, Appendix A, Appendix E, §1, Figure 1, Figure 1, §2.2, §2.2, §2.2, §2.2, Figure 2, Figure 2, §4.1, §4.2, §4.3, Table 1, Table 1, §5.3, §5, §6.1, Table 2, §6.
Y. Zhang, S. Yang, R. Zhu, Y. Zhang, L. Cui, Y. Wang, B. Wang, F. Shi, B. Wang, W. Bi, et al. (2024)	Gated slot attention for efficient linear-time sequence modeling.Advances in Neural Information Processing Systems 37, pp. 116870–116898.Cited by: §2.1.
Y. Zhang and S. Yang (2025)	Flame: flash language modeling made easyExternal Links: LinkCited by: Appendix A.
S. Zhong, M. Xu, T. Ao, and G. Shi (2025)	Understanding transformer from the perspective of associative memory.arXiv preprint arXiv:2505.19488.Cited by: §2.1.
T. Zhou, R. Tucker, J. Flynn, G. Fyffe, and N. Snavely (2018)	Stereo magnification: learning view synthesis using multiplane images.ACM Transactions on Graphics (TOG) 37 (4), pp. 1–12.Cited by: Appendix A.
Appendix AExperiment Setup

Here we outline the three tasks used to evaluate the TTT framework, along with the experimental setup.

Language Modeling  We use the 760M parameter LaCT-LLM (Zhang et al., 2025) as our baseline model. All models are trained on 100B tokens sampled from the FineWeb-Edu dataset (Penedo et al., 2024) with a batch size of 4 per GPU across 8 NVIDIA A100 GPUs for 20K iterations, which takes approximately 56 hours to complete. All other hyperparameters follow the original (Zhang et al., 2025) configuration. For evaluation, we report perplexity on 2.5B tokens from the Book-3 dataset (Gao et al., 2020). All implementations are based on the Flame (Zhang and Yang, 2025) codebase.

Novel View Synthesis  We use 12-layer 768 hidden dimension LaCT-NVS (Zhang et al., 2025) model as baseline, totaling 114M parameters. We train our model on the RealEstate10K (Zhou et al., 2018) dataset with a batch size of 128 per GPU across 4 NVIDIA A100 GPUs for 20K iterations, which takes approximately 38 hours to complete. We use 2 input views and 6 target views for training, and 2 input views and 3 target views for evaluation. All images are resized to 
128
×
128
 resolution. We train using only the MSE loss and evaluate using the Peak Signal-to-Noise Ratio (PSNR) metric. All other hyperparameters follow the original LaCT-NVS (Zhang et al., 2025) configuration.

Image Classification  We use ViTTT-B (Han et al., 2025) as our baseline model, totaling 90M parameters. Follow (Han et al., 2025), we train our model on the ImageNet-1K (Deng et al., 2009) dataset with a batch size of 256 per GPU across 2 NVIDIA H100 GPUs for 60 epochs, which takes approximately 16 hours to complete. We evaluate on the ImageNet-1K (Deng et al., 2009) validation set using the top-1 accuracy. All other hyperparameters follow the original (Han et al., 2025) configuration.

Appendix BProof of Theorem 5.1
Proof.

We explicitly unroll the single-step inner-loop update. Using the shorthand notation 
𝜙
𝑡
​
(
⋅
)
≜
𝜙
​
(
⋅
;
Θ
𝑡
)
 and 
𝑓
𝑡
​
(
⋅
)
≜
𝜙
𝑡
​
(
⋅
)
​
𝑊
𝑡
, the model output for the key input 
𝑘
 at step 
𝑡
 is

	
𝑓
𝑡
​
(
𝑘
)
=
𝜙
𝑡
​
(
𝑘
)
​
𝑊
𝑡
.
	

By the chain rule, the gradient of the last layer with respect to the loss 
ℒ
​
(
𝑓
𝑡
​
(
𝑘
)
)
 is

	
∇
𝑊
𝑡
ℒ
=
𝜙
𝑡
​
(
𝑘
)
⊤
​
∂
ℒ
∂
𝑓
𝑡
​
(
𝑘
)
.
	

Applying one step of gradient descent with learning rate 
𝜂
 yields

	
𝑊
𝑡
+
1
	
=
𝑊
𝑡
−
𝜂
​
𝜙
𝑡
​
(
𝑘
)
⊤
​
∂
ℒ
∂
𝑓
𝑡
​
(
𝑘
)
	
		
=
𝑊
𝑡
+
𝜙
𝑡
​
(
𝑘
)
⊤
​
𝑔
𝑡
​
(
𝑘
)
	

where we define

	
𝑔
𝑡
​
(
𝑘
)
≜
−
𝜂
​
∂
ℒ
∂
𝑓
𝑡
​
(
𝑘
)
∈
ℝ
𝐷
out
.
	

The kernel function parameters 
Θ
 are also updated with gradient descent, yielding the updated function 
𝜙
𝑡
+
1
​
(
⋅
)
=
𝜙
​
(
⋅
;
Θ
𝑡
+
1
)
.

Evaluating the updated model on a query 
𝑞
 gives

	
𝑜
	
=
𝜙
𝑡
+
1
​
(
𝑞
)
​
𝑊
𝑡
+
1
	
		
=
𝜙
𝑡
+
1
​
(
𝑞
)
​
(
𝑊
𝑡
+
𝜙
𝑡
​
(
𝑘
)
⊤
​
𝑔
𝑡
​
(
𝑘
)
)
	

This expression is a linear attention operator of the form

	
𝑜
=
𝑞
^
​
(
𝑆
0
+
𝑘
^
⊤
​
𝑣
^
)
,
	

where

	
𝑞
^
=
𝜙
𝑡
+
1
​
(
𝑞
)
,
𝑘
^
=
𝜙
𝑡
​
(
𝑘
)
,
𝑣
^
=
𝑔
𝑡
​
(
𝑘
)
,
𝑆
0
=
𝑊
𝑡
.
	

∎

Appendix CProof of Theorem 5.2
Proof.

We prove by induction on the number of tokens processed.

Base case.

Starting from initial parameters 
(
𝑊
0
,
Θ
0
)
, processing token 
0
 with key 
𝑘
0
 applies one gradient descent step. By Theorem 5.1:

	
𝑊
1
=
𝑊
0
+
𝜙
0
​
(
𝑘
0
)
⊤
​
𝑔
0
​
(
𝑘
0
)
,
	

where 
𝑔
0
​
(
𝑘
0
)
=
−
𝜂
​
∂
ℒ
∂
𝑓
0
​
(
𝑘
0
)
, and 
Θ
1
 is updated accordingly. This matches the claimed form with 
𝑡
=
0
.

Inductive step.

Assume that after processing tokens 
0
,
…
,
𝑡
−
1
, we have

	
𝑊
𝑡
=
𝑊
0
+
∑
𝑖
=
0
𝑡
−
1
𝜙
𝑖
​
(
𝑘
𝑖
)
⊤
​
𝑔
𝑖
​
(
𝑘
𝑖
)
.
	

Processing token 
𝑡
 with key 
𝑘
𝑡
 applies another gradient descent step. By Theorem 5.1:

	
𝑊
𝑡
+
1
=
𝑊
𝑡
+
𝜙
𝑡
​
(
𝑘
𝑡
)
⊤
​
𝑔
𝑡
​
(
𝑘
𝑡
)
.
	

Substituting the inductive hypothesis:

	
𝑊
𝑡
+
1
=
𝑊
0
+
∑
𝑖
=
0
𝑡
−
1
𝜙
𝑖
​
(
𝑘
𝑖
)
⊤
​
𝑔
𝑖
​
(
𝑘
𝑖
)
+
𝜙
𝑡
​
(
𝑘
𝑡
)
⊤
​
𝑔
𝑡
​
(
𝑘
𝑡
)
=
𝑊
0
+
∑
𝑖
=
0
𝑡
𝜙
𝑖
​
(
𝑘
𝑖
)
⊤
​
𝑔
𝑖
​
(
𝑘
𝑖
)
.
	
Output computation.

Evaluating the model on query 
𝑞
𝑡
 with the updated parameters gives:

	
𝑜
𝑡
	
=
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
​
𝑊
𝑡
+
1
	
		
=
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
​
(
𝑊
0
+
∑
𝑖
=
0
𝑡
𝜙
𝑖
​
(
𝑘
𝑖
)
⊤
​
𝑔
𝑖
​
(
𝑘
𝑖
)
)
	

This is the extended linear attention form:

	
𝑜
𝑡
=
𝑞
^
𝑡
​
(
𝑆
0
+
∑
𝑖
=
0
𝑡
𝑘
^
𝑖
⊤
​
𝑣
^
𝑖
)
,
	

where 
𝑞
^
𝑡
=
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
, 
𝑘
^
𝑖
=
𝜙
𝑖
​
(
𝑘
𝑖
)
, 
𝑣
^
𝑖
=
𝑔
𝑖
​
(
𝑘
𝑖
)
, and 
𝑆
0
=
𝑊
0
. ∎

Appendix DProof of Theorem 5.3
Proof.

We prove by induction, extending the approach of Theorem 5.2 to momentum updates.

Base case.

At 
𝑡
=
0
, the momentum accumulator is initialized as 
Δ
​
𝑊
−
1
=
0
, so

	
Δ
​
𝑊
0
=
∇
𝑊
ℒ
​
(
𝑓
0
​
(
𝑘
0
)
)
=
𝜙
0
​
(
𝑘
0
)
⊤
​
∂
ℒ
∂
𝑓
0
​
(
𝑘
0
)
.
	

Applying the update rule 
𝑊
1
=
𝑊
0
−
𝜂
​
Δ
​
𝑊
0
 gives

	
𝑊
1
=
𝑊
0
+
𝜙
0
​
(
𝑘
0
)
⊤
​
𝑔
0
​
(
𝑘
0
)
,
	

where 
𝑔
0
​
(
𝑘
0
)
=
−
𝜂
​
∂
ℒ
∂
𝑓
0
​
(
𝑘
0
)
. This matches the claimed form with 
𝑚
0
​
(
𝑘
0
)
=
𝑔
0
​
(
𝑘
0
)
.

Inductive step.

Define the cumulative momentum coefficient from step 
𝑖
 to step 
𝑗
 as:

	
𝛽
𝑖
𝑗
≜
{
∏
𝑠
=
𝑖
+
1
𝑗
𝛼
𝑠
	
if 
​
𝑖
<
𝑗
,


1
	
if 
​
𝑖
=
𝑗
.
	

Assume that after processing tokens 
0
,
…
,
𝑡
−
1
, the momentum accumulator satisfies

	
Δ
​
𝑊
𝑡
−
1
=
∑
𝑖
=
0
𝑡
−
1
𝛽
𝑖
𝑡
−
1
​
𝜙
𝑖
​
(
𝑘
𝑖
)
⊤
​
∂
ℒ
∂
𝑓
𝑖
​
(
𝑘
𝑖
)
.
	

At step 
𝑡
, the momentum update gives

	
Δ
​
𝑊
𝑡
	
=
∇
𝑊
ℒ
​
(
𝑓
𝑡
​
(
𝑘
𝑡
)
)
+
𝛼
𝑡
​
Δ
​
𝑊
𝑡
−
1
	
		
=
𝜙
𝑡
​
(
𝑘
𝑡
)
⊤
​
∂
ℒ
∂
𝑓
𝑡
​
(
𝑘
𝑡
)
+
𝛼
𝑡
​
∑
𝑖
=
0
𝑡
−
1
𝛽
𝑖
𝑡
−
1
​
𝜙
𝑖
​
(
𝑘
𝑖
)
⊤
​
∂
ℒ
∂
𝑓
𝑖
​
(
𝑘
𝑖
)
.
	

For terms 
𝑖
∈
[
0
,
𝑡
−
1
]
, we have 
𝛼
𝑡
⋅
𝛽
𝑖
𝑡
−
1
=
𝛼
𝑡
​
∏
𝑠
=
𝑖
+
1
𝑡
−
1
𝛼
𝑠
=
∏
𝑠
=
𝑖
+
1
𝑡
𝛼
𝑠
=
𝛽
𝑖
𝑡
. For the 
𝑖
=
𝑡
 term, the coefficient is 
𝛽
𝑡
𝑡
=
1
. Thus:

	
Δ
​
𝑊
𝑡
=
∑
𝑖
=
0
𝑡
𝛽
𝑖
𝑡
​
𝜙
𝑖
​
(
𝑘
𝑖
)
⊤
​
∂
ℒ
∂
𝑓
𝑖
​
(
𝑘
𝑖
)
.
	
Weight accumulation.

From 
𝑊
𝑗
+
1
=
𝑊
𝑗
−
𝜂
​
Δ
​
𝑊
𝑗
, we have 
𝑊
𝑡
+
1
=
𝑊
0
−
𝜂
​
∑
𝑗
=
0
𝑡
Δ
​
𝑊
𝑗
. Substituting the closed form for 
Δ
​
𝑊
𝑗
 and exchanging the order of summation:

	
𝑊
𝑡
+
1
	
=
𝑊
0
−
𝜂
​
∑
𝑗
=
0
𝑡
∑
𝑖
=
0
𝑗
𝛽
𝑖
𝑗
​
𝜙
𝑖
​
(
𝑘
𝑖
)
⊤
​
∂
ℒ
∂
𝑓
𝑖
​
(
𝑘
𝑖
)
	
		
=
𝑊
0
+
∑
𝑖
=
0
𝑡
𝜙
𝑖
​
(
𝑘
𝑖
)
⊤
​
(
−
𝜂
​
∂
ℒ
∂
𝑓
𝑖
​
(
𝑘
𝑖
)
​
∑
𝑗
=
𝑖
𝑡
𝛽
𝑖
𝑗
)
⏟
𝑚
𝑖
​
(
𝑘
𝑖
)
.
	

The momentum-weighted effective value is thus

	
𝑚
𝑖
​
(
𝑘
𝑖
)
=
𝑔
𝑖
​
(
𝑘
𝑖
)
⋅
∑
𝑗
=
𝑖
𝑡
𝛽
𝑖
𝑗
.
	
Output computation.

Evaluating the model on query 
𝑞
𝑡
 with the updated parameters:

	
𝑜
𝑡
	
=
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
​
𝑊
𝑡
+
1
	
		
=
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
​
(
𝑊
0
+
∑
𝑖
=
0
𝑡
𝜙
𝑖
​
(
𝑘
𝑖
)
⊤
​
𝑚
𝑖
​
(
𝑘
𝑖
)
)
.
	

This is the linear attention form with 
𝑞
^
𝑡
=
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
, 
𝑘
^
𝑖
=
𝜙
𝑖
​
(
𝑘
𝑖
)
, 
𝑣
^
𝑖
=
𝑚
𝑖
​
(
𝑘
𝑖
)
, and 
𝑆
0
=
𝑊
0
. ∎

Appendix EDerivation: LaCT as Linear Attention

We provide the full derivation showing how LaCT (Zhang et al., 2025) can be written in the form of linear attention, following the framework of Theorem 5.3.

E.1LaCT Architecture and Update Rule

LaCT adopts a bias-free SwiGLU MLP as its inner-loop mapping:

	
𝑓
​
(
𝑥
)
=
(
silu
​
(
𝑥
​
𝑊
0
)
⊙
(
𝑥
​
𝑊
2
)
)
​
𝑊
1
,
	

where 
𝑊
0
,
𝑊
2
∈
ℝ
𝐷
ℎ
×
𝐷
𝑘
 and 
𝑊
1
∈
ℝ
𝐷
ℎ
×
𝐷
𝑣
. The inner-loop objective uses the Frobenius inner product:

	
ℒ
​
(
𝑓
​
(
𝑘
)
,
𝑣
)
=
−
⟨
𝑓
​
(
𝑘
)
,
𝑣
⟩
.
	

At each token 
𝑡
, LaCT performs gradient descent with per-token learning rate 
𝜂
𝑡
, momentum 
𝛼
𝑡
, and Muon-style (Jordan et al., 2024) gradient orthogonalization 
ℳ
​
(
⋅
)
:

	
Δ
​
𝑊
𝑖
,
𝑡
=
∇
𝑊
𝑖
,
𝑡
ℒ
​
(
𝑓
𝑡
​
(
𝑘
𝑡
)
,
𝑣
𝑡
)
+
𝛼
𝑡
​
Δ
​
𝑊
𝑖
,
𝑡
−
1
,
𝑊
𝑖
,
𝑡
+
1
=
𝑊
𝑖
,
𝑡
−
𝜂
𝑡
​
ℳ
​
(
Δ
​
𝑊
𝑖
,
𝑡
)
,
	

for 
𝑖
∈
{
0
,
1
,
2
}
.

E.2Gradient Computation

The upstream gradient with respect to the output is 
∂
ℒ
∂
𝑓
𝑡
​
(
𝑘
𝑡
)
=
−
𝑣
𝑡
. Using the chain rule, the gradient for the final layer 
𝑊
1
 is:

	
∇
𝑊
1
ℒ
=
𝜙
𝑡
​
(
𝑘
𝑡
)
⊤
​
∂
ℒ
∂
𝑓
𝑡
​
(
𝑘
𝑡
)
=
−
𝜙
𝑡
​
(
𝑘
𝑡
)
⊤
​
𝑣
𝑡
,
	

where 
𝜙
𝑡
​
(
𝑥
)
=
silu
​
(
𝑥
​
𝑊
0
,
𝑡
)
⊙
(
𝑥
​
𝑊
2
,
𝑡
)
 is the kernel function.

E.3Linear Attention Form with Muon

Following the same induction as Theorem 5.3, but with the Muon orthogonalization 
ℳ
​
(
⋅
)
 applied after each gradient accumulation, the weight update becomes:

	
𝑊
1
,
𝑡
+
1
=
𝑊
1
,
0
−
∑
𝑗
=
0
𝑡
𝜂
𝑗
​
ℳ
​
(
Δ
​
𝑊
1
,
𝑗
)
.
	

Using the cumulative momentum coefficient 
𝛽
𝑖
𝑡
 defined in Theorem 5.3, the momentum accumulator for 
𝑊
1
 is:

	
Δ
​
𝑊
1
,
𝑡
=
−
𝜙
𝑡
​
(
𝑘
𝑡
)
⊤
​
𝑣
𝑡
+
𝛼
𝑡
​
Δ
​
𝑊
1
,
𝑡
−
1
=
−
∑
𝑖
=
0
𝑡
𝛽
𝑖
𝑡
​
𝜙
𝑖
​
(
𝑘
𝑖
)
⊤
​
𝑣
𝑖
.
	

Substituting into the weight update and exchanging the order of summation (as in the proof of Theorem 5.3):

	
𝑊
1
,
𝑡
+
1
=
𝑊
1
,
0
+
∑
𝑖
=
0
𝑡
ℳ
​
(
𝜙
𝑖
​
(
𝑘
𝑖
)
⊤
​
𝑚
𝑖
)
,
	

where the momentum-weighted effective value is:

	
𝑚
𝑖
≜
𝜂
𝑖
​
𝑣
𝑖
⋅
∑
𝑗
=
𝑖
𝑡
𝛽
𝑖
𝑗
.
	

Evaluating on query 
𝑞
𝑡
:

	
𝑜
𝑡
	
=
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
​
𝑊
1
,
𝑡
+
1
	
		
=
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
​
(
𝑊
1
,
0
+
∑
𝑖
=
0
𝑡
ℳ
​
(
𝜙
𝑖
​
(
𝑘
𝑖
)
⊤
​
𝑚
𝑖
)
)
.
	

This is a linear attention–like form where 
𝜙
𝑖
​
(
𝑘
𝑖
)
 and 
𝑚
𝑖
 play the roles of keys and values, and 
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
 acts as the query. The Muon orthogonalization 
ℳ
​
(
⋅
)
 is applied element-wise to each key-value outer product before accumulation.

E.4Effect of Weight Normalization

LaCT additionally applies weight normalization to 
𝑊
1
 after each update:

	
𝑊
1
,
𝑡
+
1
=
Norm
​
(
𝑊
1
,
𝑡
−
𝜂
𝑡
​
ℳ
​
(
Δ
​
𝑊
1
,
𝑡
)
)
,
	

where 
Norm
​
(
⋅
)
 applies channel-wise 
ℓ
2
 normalization.

Weight normalization does not break the linear attention perspective. Recall that in the linear attention formulation, the state matrix 
𝑆
𝑡
=
𝑊
1
,
𝑡
 accumulates key-value outer products. With weight normalization, we simply normalize this state after each update:

	
𝑆
𝑡
+
1
=
Norm
​
(
𝑆
𝑡
+
𝜙
𝑡
​
(
𝑘
𝑡
)
⊤
​
𝑚
𝑡
)
.
	

The output remains a linear function of the (normalized) state:

	
𝑜
𝑡
=
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
​
𝑆
𝑡
+
1
.
	

Thus, LaCT with weight normalization is still a linear attention mechanism—the query linearly reads from a state that accumulates key-value information.

However, the normalization prevents expressing the state as a simple sum over history. Unlike the unnormalized case where 
𝑆
𝑡
+
1
=
𝑆
0
+
∑
𝑖
=
0
𝑡
𝜙
𝑖
​
(
𝑘
𝑖
)
⊤
​
𝑚
𝑖
, the nested normalization creates a sequential dependency:

	
𝑆
𝑡
+
1
	
=
Norm
​
(
Norm
​
(
𝑆
𝑡
−
1
+
𝜙
𝑡
−
1
​
(
𝑘
𝑡
−
1
)
⊤
​
𝑚
𝑡
−
1
)
+
𝜙
𝑡
​
(
𝑘
𝑡
)
⊤
​
𝑚
𝑡
)
.
	

This has implications for parallelization, which we discuss in Appendix I.

Appendix FDerivation: ViTTT GLU as Linear Attention

We show that the simplified GLU used in ViTTT (Han et al., 2025) can be written into a linear attention form with element-wise multiplication.

F.1Architecture and Loss

In ViTTT, the GLU is defined as:

	
𝑓
𝑡
​
(
𝑥
)
=
silu
​
(
𝑥
​
𝑊
0
,
𝑡
)
⊙
(
𝑥
​
𝑊
1
,
𝑡
)
,
	

where 
𝑊
0
,
𝑡
,
𝑊
1
,
𝑡
∈
ℝ
𝐷
ℎ
×
𝐷
ℎ
 are both square matrices. We define the kernel function as

	
𝜙
𝑡
​
(
𝑥
)
≜
silu
​
(
𝑥
​
𝑊
0
,
𝑡
)
.
	

Unlike the general form where the kernel function output is followed by matrix multiplication with a final layer, here the GLU uses element-wise multiplication between the nonlinear gating 
𝜙
𝑡
​
(
𝑥
)
 and the linear projection 
𝑥
​
𝑊
1
,
𝑡
.

The inner-loop loss uses the Frobenius inner product:

	
ℒ
​
(
𝑓
𝑡
​
(
𝑘
𝑡
)
,
𝑣
𝑡
)
=
−
⟨
𝑓
𝑡
​
(
𝑘
𝑡
)
,
𝑣
𝑡
⟩
.
	
F.2Gradient Computation

The upstream gradient with respect to the GLU output is 
∂
ℒ
∂
𝑓
𝑡
​
(
𝑘
𝑡
)
=
−
𝑣
𝑡
.

Applying the chain rule, the gradient of 
𝑊
1
,
𝑡
 is:

	
∇
𝑊
1
,
𝑡
ℒ
=
𝑘
𝑡
⊤
​
(
∂
ℒ
∂
𝑓
𝑡
​
(
𝑘
𝑡
)
⊙
𝜙
𝑡
​
(
𝑘
𝑡
)
)
=
−
𝑘
𝑡
⊤
​
(
𝑣
𝑡
⊙
𝜙
𝑡
​
(
𝑘
𝑡
)
)
.
	
F.3Linear Attention Form

After one step of gradient descent with learning rate 
𝜂
:

	
𝑊
1
,
𝑡
+
1
=
𝑊
1
,
𝑡
−
𝜂
​
∇
𝑊
1
,
𝑡
ℒ
=
𝑊
1
,
𝑡
+
𝜂
​
𝑘
𝑡
⊤
​
(
𝑣
𝑡
⊙
𝜙
𝑡
​
(
𝑘
𝑡
)
)
.
	

Similarly, 
𝑊
0
,
𝑡
 is updated to 
𝑊
0
,
𝑡
+
1
, yielding the updated kernel function 
𝜙
𝑡
+
1
​
(
⋅
)
.

Evaluating the model on query 
𝑞
𝑡
 gives:

	
𝑜
𝑡
	
=
𝑓
𝑡
+
1
​
(
𝑞
𝑡
)
=
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
⊙
(
𝑞
𝑡
​
𝑊
1
,
𝑡
+
1
)
	
		
=
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
⊙
(
𝑞
𝑡
​
𝑊
1
,
𝑡
+
𝜂
​
(
𝑞
𝑡
​
𝑘
𝑡
⊤
)
​
(
𝑣
𝑡
⊙
𝜙
𝑡
​
(
𝑘
𝑡
)
)
)
	
		
=
𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
⊙
(
𝑞
𝑡
​
(
𝑊
1
,
𝑡
+
𝑘
𝑡
⊤
​
(
𝑣
𝑡
⊙
𝜙
𝑡
​
(
𝑘
𝑡
)
)
)
)
.
	

This is a linear attention form where:

• 

The second term computes a scalar attention weight 
⟨
𝑞
𝑡
,
𝑘
𝑡
⟩
 that modulates the value vector 
𝑣
𝑡

• 

𝜙
𝑡
​
(
𝑘
𝑡
)
 acts as a multiplicative gate on the values

• 

𝜙
𝑡
+
1
​
(
𝑞
𝑡
)
 gates the final output

The state matrix 
𝑆
𝑡
=
𝑊
1
,
𝑡
 accumulates outer products of keys and gated values, consistent with the general linear attention framework.

Appendix GDerivation: ViTTT Depthwise Convolution as Linear Attention

We show that the 
3
×
3
 depthwise convolution layer in ViTTT (Han et al., 2025) can be formulated as a form of spatially-local linear attention.

G.1Architecture and Loss

Let 
𝐾
,
𝑉
∈
ℝ
𝐶
×
𝐻
×
𝑊
 be the spatial key and value tensors, and 
𝑊
𝑡
∈
ℝ
𝐶
×
1
×
3
×
3
 be the depthwise convolution weights at step 
𝑡
. The forward pass computes:

	
𝑓
𝑡
​
(
𝐾
)
=
Conv
3
×
3
​
(
𝐾
;
𝑊
𝑡
)
,
	

where 
Conv
3
×
3
 denotes depthwise convolution with 
3
×
3
 kernel.

The inner-loop loss uses the Frobenius inner product:

	
ℒ
​
(
𝑓
𝑡
​
(
𝐾
)
,
𝑉
)
=
−
⟨
𝑓
𝑡
​
(
𝐾
)
,
𝑉
⟩
=
−
∑
𝑐
,
𝑖
,
𝑗
[
𝑓
𝑡
​
(
𝐾
)
]
𝑐
,
𝑖
,
𝑗
⋅
𝑉
𝑐
,
𝑖
,
𝑗
.
	
G.2Gradient Computation

The upstream gradient is 
∂
ℒ
∂
𝑓
𝑡
​
(
𝐾
)
=
−
𝑉
.

For depthwise convolution, the gradient with respect to the weight 
𝑊
𝑡
 can be written as:

	
∇
𝑊
𝑡
ℒ
=
𝐾
⋆
∂
ℒ
∂
𝑓
𝑡
​
(
𝐾
)
=
−
𝐾
⋆
𝑉
,
	

where 
⋆
 denotes cross-correlation. Specifically, for each channel 
𝑐
 and offset 
(
𝛿
𝑦
,
𝛿
𝑥
)
∈
{
−
1
,
0
,
1
}
2
:

	
[
∇
𝑊
𝑡
ℒ
]
𝑐
,
𝛿
𝑦
,
𝛿
𝑥
=
−
∑
𝑖
,
𝑗
𝐾
𝑐
,
𝑖
+
𝛿
𝑦
,
𝑗
+
𝛿
𝑥
⋅
𝑉
𝑐
,
𝑖
,
𝑗
.
	
G.3Linear Attention Form

After one step of gradient descent with learning rate 
𝜂
:

	
[
𝑊
𝑡
+
1
]
𝑐
,
𝛿
𝑦
,
𝛿
𝑥
=
[
𝑊
𝑡
]
𝑐
,
𝛿
𝑦
,
𝛿
𝑥
+
𝜂
​
∑
𝑖
,
𝑗
𝐾
𝑐
,
𝑖
+
𝛿
𝑦
,
𝑗
+
𝛿
𝑥
⋅
𝑉
𝑐
,
𝑖
,
𝑗
.
	

When evaluating on query 
𝑄
∈
ℝ
𝐶
×
𝐻
×
𝑊
, the output at position 
(
𝑖
,
𝑗
)
 is:

	
𝑂
𝑐
,
𝑖
,
𝑗
	
=
∑
𝛿
𝑦
,
𝛿
𝑥
[
𝑊
𝑡
+
1
]
𝑐
,
𝛿
𝑦
,
𝛿
𝑥
⋅
𝑄
𝑐
,
𝑖
+
𝛿
𝑦
,
𝑗
+
𝛿
𝑥
	
		
=
∑
𝛿
𝑦
,
𝛿
𝑥
[
𝑊
𝑡
]
𝑐
,
𝛿
𝑦
,
𝛿
𝑥
⋅
𝑄
𝑐
,
𝑖
+
𝛿
𝑦
,
𝑗
+
𝛿
𝑥
⏟
[
Conv
​
(
𝑄
;
𝑊
𝑡
)
]
𝑐
,
𝑖
,
𝑗
	
		
+
𝜂
​
∑
𝑖
′
,
𝑗
′
(
∑
𝛿
𝑦
,
𝛿
𝑥
𝑄
𝑐
,
𝑖
+
𝛿
𝑦
,
𝑗
+
𝛿
𝑥
⋅
𝐾
𝑐
,
𝑖
′
+
𝛿
𝑦
,
𝑗
′
+
𝛿
𝑥
)
⏟
spatial attention weight
⋅
𝑉
𝑐
,
𝑖
′
,
𝑗
′
.
	

This is a linear attention form where:

• 

The first term corresponds to the initial state 
𝑆
0
=
𝑊
𝑡

• 

The attention weight between query position 
(
𝑖
,
𝑗
)
 and key position 
(
𝑖
′
,
𝑗
′
)
 is the sum of element-wise products over the 
3
×
3
 neighborhood offsets

• 

This spatially-local attention allows each output position to attend to all key-value positions, weighted by the overlap of their local 
3
×
3
 neighborhoods

Conceptually, since convolution is effectively a sliding-window linear layer, this TTT component is equivalent to a sliding-window linear attention mechanism.

Appendix HParallel Form of TTT

We present the parallel formulation for TTT when only 
𝑊
1
 is dynamic (while 
𝑊
0
 and 
𝑊
2
 are static) and weight normalization is omitted. Under these conditions, the kernel function becomes static:

	
𝜙
​
(
𝑥
)
=
silu
​
(
𝑥
​
𝑊
0
)
⊙
(
𝑥
​
𝑊
2
)
,
	

and the state update is associative, enabling parallel computation.

H.1Parallel Formulation

The parallel formulation operates on a sequence of 
𝑁
 chunks, each of size 
𝐿
. Let 
ℚ
,
𝕂
∈
ℝ
(
𝑁
​
𝐿
)
×
𝐷
𝑘
 and 
𝕍
∈
ℝ
(
𝑁
​
𝐿
)
×
𝐷
𝑣
 denote the concatenated query, key, and value inputs across all chunks. Define the batched kernel function:

	
Φ
​
(
𝐗
)
=
silu
​
(
𝐗
​
𝑊
0
)
⊙
(
𝐗
​
𝑊
2
)
∈
ℝ
(
𝑁
​
𝐿
)
×
𝐷
ℎ
.
	

To express chunk-wise operations, we introduce:

• 

Block-diagonal matrix 
𝐁
∈
ℝ
(
𝑁
​
𝐿
)
×
(
𝑁
​
𝐿
)
: 
𝐁
=
diag
​
(
𝐼
𝐿
,
…
,
𝐼
𝐿
)
 with 
𝑁
 identity blocks

• 

Learning rate vector 
𝜼
∈
ℝ
(
𝑁
​
𝐿
)
×
1
: per-token learning rates

• 

Scaled values 
𝕍
~
=
𝕍
⊙
𝜼

• 

Momentum matrix 
𝒜
∈
ℝ
𝑁
×
𝑁
: 
𝒜
𝑖
​
𝑗
=
∏
𝑠
=
𝑗
+
1
𝑖
𝛼
𝑠
 for 
𝑖
≥
𝑗
, else 
0

• 

Causal mask 
𝐌
∈
{
0
,
1
}
𝑁
×
𝑁
: 
𝐌
𝑖
​
𝑗
=
1
 if 
𝑖
≥
𝑗

The output can be computed in parallel as:

	
𝕆
=
Φ
​
(
ℚ
)
​
𝑊
1
,
0
+
(
(
Φ
​
(
ℚ
)
​
Φ
​
(
𝕂
)
⊤
)
⊙
(
𝒜
⊙
𝐌
)
↑
𝐿
)
​
𝕍
~
,
	

where 
(
⋅
)
↑
𝐿
 denotes the Kronecker product with 
𝟏
𝐿
×
𝐿
 to expand the 
𝑁
×
𝑁
 mask to 
(
𝑁
​
𝐿
)
×
(
𝑁
​
𝐿
)
.

H.2Proof of Equivalence

We prove that this parallel formulation is equivalent to the sequential recurrence.

Proof.

Consider the sequential formulation where at each chunk 
𝑡
, the weight update is:

	
Δ
​
𝑊
𝑡
=
∇
𝑊
1
ℒ
​
(
𝑓
𝑡
​
(
𝐾
𝑡
)
)
+
𝛼
𝑡
​
Δ
​
𝑊
𝑡
−
1
,
𝑊
1
,
𝑡
+
1
=
𝑊
1
,
𝑡
−
𝜂
𝑡
​
Δ
​
𝑊
𝑡
.
	

Since 
∇
𝑊
1
ℒ
=
Φ
​
(
𝐾
𝑡
)
⊤
​
∂
ℒ
∂
𝑓
𝑡
​
(
𝐾
𝑡
)
 and using the Frobenius inner product loss, we have:

	
Δ
​
𝑊
𝑡
=
−
Φ
​
(
𝐾
𝑡
)
⊤
​
𝑉
𝑡
+
𝛼
𝑡
​
Δ
​
𝑊
𝑡
−
1
.
	
Step 1: Unroll the momentum recurrence.

Expanding 
Δ
​
𝑊
𝑡
:

	
Δ
​
𝑊
𝑡
	
=
−
∑
𝑖
=
0
𝑡
(
∏
𝑠
=
𝑖
+
1
𝑡
𝛼
𝑠
)
​
Φ
​
(
𝐾
𝑖
)
⊤
​
𝑉
𝑖
.
	
Step 2: Unroll the weight recurrence.

From 
𝑊
1
,
𝑡
+
1
=
𝑊
1
,
0
−
∑
𝑗
=
0
𝑡
𝜂
𝑗
​
Δ
​
𝑊
𝑗
, substituting and rearranging:

	
𝑊
1
,
𝑡
+
1
	
=
𝑊
1
,
0
+
∑
𝑖
=
0
𝑡
Φ
​
(
𝐾
𝑖
)
⊤
​
𝑉
𝑖
⋅
𝜂
𝑖
​
∑
𝑗
=
𝑖
𝑡
(
∏
𝑠
=
𝑖
+
1
𝑗
𝛼
𝑠
)
⏟
𝑐
𝑖
𝑡
.
	
Step 3: Compute the sequential output.

The output at chunk 
𝑡
 is:

	
𝑂
𝑡
	
=
Φ
​
(
𝑄
𝑡
)
​
𝑊
1
,
𝑡
+
1
=
Φ
​
(
𝑄
𝑡
)
​
𝑊
1
,
0
+
∑
𝑖
=
0
𝑡
Φ
​
(
𝑄
𝑡
)
​
Φ
​
(
𝐾
𝑖
)
⊤
​
𝑉
𝑖
⋅
𝑐
𝑖
𝑡
.
	
Step 4: Match with parallel formulation.

The mask 
(
𝒜
⊙
𝐌
)
↑
𝐿
 at block 
(
𝑡
,
𝑖
)
 has value 
∑
𝑗
=
𝑖
𝑡
∏
𝑠
=
𝑖
+
1
𝑗
𝛼
𝑠
 for 
𝑖
≤
𝑡
. The parallel form computes:

	
[
𝕆
]
𝑝
	
=
Φ
​
(
𝑄
𝑡
)
𝑝
​
𝑊
1
,
0
+
∑
𝑖
=
0
𝑡
Φ
​
(
𝑄
𝑡
)
𝑝
​
Φ
​
(
𝐾
𝑖
)
⊤
​
𝑉
𝑖
⋅
𝑐
𝑖
𝑡
,
	

which exactly matches the sequential output. ∎

Appendix INon-Reducible Case Analysis

In Section 6.2, we showed that TTT becomes reducible (and thus parallelizable) when only 
𝑊
1
 is dynamic while 
𝑊
0
 and 
𝑊
2
 are static. Here, we analyze two cases that break reducibility: (1) updating the kernel function parameters 
Θ
=
{
𝑊
0
,
𝑊
2
}
, and (2) applying weight normalization.

I.1Case 1: Dynamic Kernel Function (Updating 
𝑊
0
 and 
𝑊
2
)

When 
𝑊
0
 and 
𝑊
2
 are also updated, the kernel function 
𝜙
𝑡
​
(
𝑥
)
=
silu
​
(
𝑥
​
𝑊
0
,
𝑡
)
⊙
(
𝑥
​
𝑊
2
,
𝑡
)
 becomes history-dependent. Consider the gradient update for 
𝑊
0
 (the analysis for 
𝑊
2
 is analogous):

	
∇
𝑊
0
,
𝑡
ℒ
=
𝐾
𝑡
⊤
​
(
∂
ℒ
∂
𝑓
𝑡
​
(
𝐾
𝑡
)
​
𝑊
1
,
𝑡
⊤
⊙
(
𝐾
𝑡
​
𝑊
2
,
𝑡
)
⊙
silu
′
⁡
(
𝐾
𝑡
​
𝑊
0
,
𝑡
)
)
.
	

The weight update becomes:

	
𝑊
0
,
𝑡
+
1
=
𝑊
0
,
𝑡
−
𝜂
𝑡
​
∇
𝑊
0
,
𝑡
ℒ
.
	

Expanding recursively, the kernel function at step 
𝑡
 depends on 
𝑊
0
,
𝑡
:

	
𝜙
𝑡
​
(
𝐾
𝑡
)
=
silu
⁡
(
𝐾
𝑡
​
𝑊
0
,
𝑡
)
⊙
(
𝐾
𝑡
​
𝑊
2
,
𝑡
)
.
	

Substituting the recursive expression for 
𝑊
0
,
𝑡
, the kernel function involves nested nonlinearities:

	
𝜙
𝑡
​
(
𝐾
𝑡
)
=
silu
⁡
(
𝐾
𝑡
​
(
𝑊
0
,
𝑡
−
1
−
𝜂
𝑡
−
1
​
∇
𝑊
0
,
𝑡
−
1
ℒ
)
)
⊙
(
⋯
)
,
	

where 
∇
𝑊
0
,
𝑡
−
1
ℒ
 itself contains 
silu
′
⁡
(
𝐾
𝑡
−
1
​
𝑊
0
,
𝑡
−
1
)
.

The nested 
silu
 and 
silu
′
 functions create a non-linear dependency chain: computing 
𝜙
𝑡
 requires 
𝑊
0
,
𝑡
, which depends on 
silu
′
⁡
(
𝐾
𝑡
−
1
​
𝑊
0
,
𝑡
−
1
)
, which in turn depends on 
𝑊
0
,
𝑡
−
1
, and so on. This nested structure prevents expressing the output as a simple weighted sum over history, breaking the associativity required for parallel prefix scan.

I.2Case 2: Weight Normalization

Even when only 
𝑊
1
 is dynamic, applying weight normalization (as in LaCT, see Appendix E) introduces non-reducibility for the parallel formulation.

As discussed in Appendix E, weight normalization does not break the linear attention interpretation—it simply normalizes the state 
𝑆
𝑡
 after each token. However, it does prevent the parallel computation enabled by the sum form.

The key issue is that normalization is not associative:

	
Norm
​
(
𝐴
+
𝐵
)
≠
Norm
​
(
𝐴
)
+
Norm
​
(
𝐵
)
.
		
(2)

The parallel formulation in Section 6.2 relies on expressing the state as 
𝑆
𝑡
+
1
=
𝑆
0
+
∑
𝑖
=
0
𝑡
𝜙
​
(
𝐾
𝑖
)
⊤
​
𝑚
𝑖
, which can be computed via associative parallel prefix scan. With weight normalization, the state becomes:

	
𝑆
𝑡
+
1
=
Norm
​
(
Norm
​
(
𝑆
𝑡
−
1
+
𝜙
​
(
𝐾
𝑡
−
1
)
⊤
​
𝑚
𝑡
−
1
)
+
𝜙
​
(
𝐾
𝑡
)
⊤
​
𝑚
𝑡
)
.
	

This nested structure creates a strict sequential dependency: computing 
𝑆
𝑡
+
1
 requires the fully normalized 
𝑆
𝑡
, which requires 
𝑆
𝑡
−
1
, and so on. The associativity required for parallel prefix scan is broken, forcing token-by-token sequential computation even though the underlying mechanism remains linear attention.

Generated on Tue Feb 24 04:01:54 2026 by LaTeXML

