# Rewirable Circuits in Transformers with Loom Attention (2611.30634v2)

# Rewirable Circuits in Transformers with Loom Attention

Da Xiao

Affiliation: Beijing University of Posts and Telecommunications, Beijing, China

Correspondence to: xiaoda99@bupt.edu.cn

  
Qingye Meng

Affiliation: ColorfulClouds Technology Co., Ltd., Beijing, China

  
Shengping Li

Affiliation: ColorfulClouds Technology Co., Ltd., Beijing, China

  
Xingyuan Yuan

Affiliation: ColorfulClouds Technology Co., Ltd., Beijing, China

###### Abstract

 

We propose loom attention, which allows a Transformer to dynamically rewire its circuits at every token by interweaving a head’s vertical attention over the outputs of earlier heads (to obtain its query, key and value) with its horizontal attention over tokens.
We instantiate vertical attention efficiently with normalized and gated linear attention over a per-token matrix memory, which stores the compressed head outputs and serves as a second residual stream.
This yields LoomFormer, a dual-stream architecture that splits control and content between the vector and matrix streams.
Across model sizes and training token budgets in language modeling and downstream tasks, LoomFormer matches Transformers trained with over 3× compute and recent residual-stream baselines (HC/mHC, MUDDFormer, AttnRes) trained with ~1.7–2.8×.
For example, trained on the same 300B tokens with only 2% extra FLOPs, LoomPythia-2.8B achieves ~1.6× the average downstream gain of mHC-Pythia-2.8B over Pythia-2.8B, and matches (0-shot) or surpasses (5-shot) Pythia-12B.
 

###### Keywords: 
Multi-Head Attention, Transformer

 

1Beijing University of Posts and Telecommunications  2ColorfulClouds Technology

xiaoda99@bupt.edu.cn  Code: https://github.com/Caiyun-AI/LoomFormer

 

\icml@noticeprintedtrue

## 1 Introduction

FIGCAPTION: Figure 1: Conceptual sketch of loom attention. At each token, each head chooses which slots of that token’s matrix memory $M$ to read and write (i.e., vertical attention, every head does both; only one is shown per layer) and which tokens to attend to (horizontal attention), so a different circuit is woven for each token.

FIGCAPTION: Figure 2: Downstream accuracy averaged over 10 tasks.

Transformers solve tasks by composing attention heads across layers into circuits (Elhage et al., 2020; Olsson et al., 2022; Wang et al., 2023; Merullo et al., 2024):
an earlier head writes a feature into a subspace of the residual stream, and a later head reads from that subspace into its query, key or value.
Induction heads, for example, read the output of a previous-token head through their keys (K-composition) (Olsson et al., 2022).

The composition between two heads, however, is determined by the product of their output and input projections, a virtual weight that does not change after training (Elhage et al., 2020).
Attention makes the horizontal flow of information across tokens input-dependent: a head chooses which tokens to attend to.
In contrast, the vertical flow across layers is fixed: a head cannot choose which heads to read from.
A model that needs different circuits for different inputs therefore keeps all of them wired at once, lets them compete for the residual stream, and can gate them only indirectly, e.g., by sending a head’s attention to a sink token (Xiao et al., 2024b).
We ask whether the wiring itself can be chosen per token, i.e., whether circuits can be made rewirable.

Recent work has enriched the vertical flow.
Dense connections let a layer aggregate the outputs of all earlier layers (Pagliardini et al., 2024);
MUDDFormer (Xiao et al., 2025) and Attention Residuals (AttnRes) (Kimi Team, 2026a) make this aggregation input-dependent, turning it into a depth-wise (i.e., vertical) attention (separate for the query, key, value and residual inputs in MUDDFormer);
and Hyper-Connections (HC) (Zhu et al., 2025) and its manifold-constrained variant mHC (Xie et al., 2025) maintain several residual streams with learnable or dynamic mixing.
In all of them, however, the vertical channel meets horizontal attention only through the hidden state of a layer, in which the outputs of all heads have already been summed and mixed together:
a head can neither single out a particular earlier head nor be singled out by a later one.
The vertical and horizontal flows run side by side, never woven together.

To make circuits rewirable, we propose loom attention, in which a head chooses which earlier heads to read from just as it chooses which tokens to attend to.
By attending vertically over the outputs of these heads at the same token, the head obtains its query, key and value to perform the standard horizontal attention over tokens, and its output is in turn attended to vertically by later heads.
Vertical and horizontal attention are thus interwoven, each feeding the other at every head, like the warp and weft of a loom (Figure 2).

Naive vertical attention over the outputs of all earlier heads is, however, expensive.
We therefore instantiate it with normalized and gated linear attention over a per-token matrix memory, which stores the compressed head outputs and serves as a second residual stream.
We call the resulting dual-stream architecture LoomFormer, after a drawloom---a loom in which a separate stored pattern selects which warp threads to lift at each pick, rather than repeating one fixed pattern.
11
 1
 
 
 
 
 
 
 
 The automated form of the drawloom, the Jacquard loom, encoded this pattern on punched cards and inspired early programmable computers such as Babbage’s Analytical Engine.
Here the ordinary vector stream plays the pattern, carrying control—which memory slots each head writes and reads (weaving a different circuit per token) as well as which tokens to attend to—while the matrix stream carries the content moved across tokens.
The two streams are coupled through attention outputs and FFN writes.

Extensive experiments on language modeling and downstream tasks across model sizes and training token budgets show that LoomFormer adds only 2.0–3.7% FLOPs while matching Transformers trained with over 3× compute and recent residual-stream baselines (HC/mHC, MUDDFormer, AttnRes) trained with ~1.7–2.8×.
For example, trained on the same 300B tokens, LoomPythia-2.8B achieves ~1.6× the average downstream gain of mHC-Pythia-2.8B over Pythia-2.8B, and matches (0-shot) or surpasses (5-shot) Pythia-12B (Figure 2).
We also analyze the trained 2.8B models to understand how and why LoomFormer works—e.g., by rewiring head outputs to inputs per token, it keeps over three times as many deep-layer heads off attention sinks as Pythia and mHC.

## 2 Rewirable Circuits

#### Writers, readers and circuits.

Attention heads communicate through the residual stream (Elhage et al., 2020).
Consider a Transformer with $L$ layers of $n$ heads of dim $d$ , and model dim $D=nd$ .
We call the output of a head a writer and each of its inputs (query, key and value) a reader, so a head has one writer and three readers.
Writer $\mathsf{W}_{i}$ adds its output $o_{i}\in\mathbb{R}^{d}$ to the residual stream through its block $W^{O}_{i}\in\mathbb{R}^{d\times D}$ of the output projection, and reader $\mathsf{R}_{j}$ of a later head reads the stream through its input projection $W^{s}_{j}\in\mathbb{R}^{D\times d}$ ( $s\in\{Q,K,V\}$ ), so $\mathsf{W}_{i}$ reaches $\mathsf{R}_{j}$ through the virtual weight $\bar{C}_{ij}=W^{O}_{i}W^{s}_{j}\in\mathbb{R}^{d\times d}$ : the two communicate only if $\mathsf{W}_{i}$ writes into a subspace that $\mathsf{R}_{j}$ reads.
Heads composed in this way form circuits whose nodes are the writers and readers.

A task may need different circuits at different tokens, i.e., different pairs of writers and readers to be connected (Figure 3).
To describe how the wiring can change, we give it a dynamic part besides the static virtual weights: at a token, reader $\mathsf{R}_{j}$ receives

 $u_{j}=\textstyle\sum_{i}o_{i}\,c_{ij}\bar{C}_{ij},$ 

(1)

where the per-token scalars $c_{ij}$ form the wiring matrix $C\in\mathbb{R}^{w\times r}$ from the $w$ earlier writers to the $r$ readers of one layer, and $c_{ij}\bar{C}_{ij}$ is the effective wiring from $\mathsf{W}_{i}$ to $\mathsf{R}_{j}$ .
The two are connected at the token if their effective wiring is nonzero.22
 2
 
 
 
 
 
 
 
 In trained models entries are rarely exactly zero; we count those below a threshold as zero (Section A.3).
A wiring pattern $P\subseteq[w]\times[r]$ is a set of writer-reader pairs, realized when each reader in $P$ is connected to exactly its writers in $P$ .

###### Definition 2.1.

An architecture with fixed virtual weights is rewirable w.r.t. a set of wiring patterns if every pattern in the set is realized by some wiring matrix $C$ that it admits at a token.

FIGCAPTION: Figure 3: Effective wiring for three wiring patterns.
Transformers (top two rows) can only gate whole heads, i.e., rows and columns, on and off, so they connect only rectangles within the fixed support of their virtual weights.
LoomFormer (bottom row) connects exactly the required pairs.

#### Transformers are node-gated.

In a Transformer, pre-LN only rescales all writers of a token by a common factor (Lemma A.1 in Section A.1).
Within a token, nodes can thus only be gated through whole heads: a head that attends to a sink token (Xiao et al., 2024b) outputs nearly nothing, turning off both its writer and, since what its readers receive has no effect, its readers.

###### Definition 2.2.

An architecture is node-gated if, at every token, its wiring matrix has the form

 $C=g\,g^{\prime\top},$ 

(2)

with gate vectors $g\in\mathbb{R}^{w}$ and $g^{\prime}\in\mathbb{R}^{r}$ whose nonzero entries mark the writers and readers that are on.

We call the rank of $C$ the wiring rank; a node-gated architecture has wiring rank at most one, so existing connections can be gated on and off, but not rewired.

###### Proposition 2.3.

A Transformer with pre-LN is node-gated: at any token, it connects precisely the pairs with $\bar{C}_{ij}\neq 0$ whose writer and reader are both on, a rectangle within the fixed support of the virtual weights.

Gating a writer or a reader acts on its whole row or column of $C$ , so a wiring pattern is realized only if it is such a rectangle, as in the first column of Figure 3; other patterns fail in one of two ways.
With dense virtual weights, where heads share subspaces, every pair can be connected, but a non-rectangular pattern such as $\{\mathsf{W}_{1}\to\mathsf{R}_{1},\mathsf{W}_{3}\to\mathsf{R}_{3}\}$ (last column) also connects $\mathsf{W}_{1}\to\mathsf{R}_{3}$ and $\mathsf{W}_{3}\to\mathsf{R}_{1}$ : the two circuits interfere.
With sparse virtual weights, where groups of heads use separate subspaces, circuits in different groups do not interfere, but pairs across groups such as $\mathsf{W}_{3}\to\mathsf{R}_{2}$ (second column) are unreachable.

#### The price of rewiring by node gating.

Can a node-gated architecture overcome this by spending more heads?
Suppose the role of each of $r\geq 2$ readers may be played by several heads, gated on and off per token, and at every token each reader must receive exactly one of $w$ writers, chosen independently of the other readers; the last pattern in Figure 3 is the smallest instance.
Let $N(w,t)$ denote the minimum size of a ground set carrying a $t$ -cover-free family of $w$ sets, i.e., $w$ sets none of which is covered by the union of $t$ others (Kautz & Singleton, 1964).

###### Theorem 2.4.

To realize all such wiring patterns, a node-gated architecture needs at least $N(w,r-1)$ heads per reader, and this number suffices, where $N(w,1)\approx\log_{2}w$ , $N(w,t)=\Omega(t^{2}\log w/\log t)$ for $2\leq t\leq\sqrt{w}$ and $N(w,t)=\Theta(w)$ for $t>\sqrt{w}$ .

Intuitively, a head receives every writer that is on and has a nonzero virtual weight to it, so a reader must hold in reserve a head that sees its chosen writer and none of the up to $r-1$ writers chosen by the other readers (Theorem A.2 in Section A.2; bounds in Section A.3).
With the $r=3n$ readers of a layer choosing among its $w=nl$ earlier writers, $r-1>\sqrt{w}$ whenever $l\leq 9n-6$ , so at practical sizes node gating needs $\Theta(w)$ heads per reader.

#### Naive loom attention makes Transformers rewirable.

To rewire with one head per reader, the wiring matrix must be chosen entry by entry at each token.
The most direct way is to let every reader attend over the outputs of all earlier heads of the same token, with attention weights generated from the token itself:

 $u_{j}=\textstyle\sum_{i}\alpha_{ij}(x)\,o_{i},\qquad\alpha_{\cdot j}(x)=\operatorname{softmax}(xW^{\alpha}_{j}),$ 

(3)

where $W^{\alpha}_{j}\in\mathbb{R}^{D\times nl}$ for a reader in layer $l$ , which has $nl$ earlier writers.
We call this naive loom attention: its virtual weights are identities and all the wiring is left to $C=[\alpha_{ij}]$ , unlike a Transformer, which wires heads through fixed virtual weights and only gates them through a rank-one $C$ .
With one head per reader, it is rewirable w.r.t. all wiring patterns (cf. Figure 3, bottom row), but expensive: over the $3n$ readers of all $L$ layers, generating the attention weights takes about $\frac{3}{2}n^{2}L^{2}D$ parameters and multiply-adds per token, growing quadratically in the total number of heads $nL$ and rivaling the whole model at practical sizes (Section A.4).

## 3 Method

[IMG alt: Refer to caption]

FIGCAPTION: Figure 4: LoomFormer architecture: the model (left), the LoomMHA and FFN sub-layers of a block (middle), and the MRead (MR) and MWrite (MW) modules (right).
Vertical attention: reads from $M$ are in red and writes to $M$ in blue.

### 3.1 From Naive to Affordable Loom Attention

As linear attention compresses a growing key-value cache into a fixed-size state (Katharopoulos et al., 2020; Schlag et al., 2021), we compress the head outputs of each token along depth into a matrix memory $M\in\mathbb{R}^{d_{k}\times d}$ of $d_{k}$ slots of width $d$ .
Writer $\mathsf{W}_{i}$ writes its output to $M$ with a key $k^{m}_{i}\in\mathbb{R}^{d_{k}}$ , and reader $\mathsf{R}_{j}$ reads with a query $q^{m}_{j}\in\mathbb{R}^{d_{k}}$ , so a reader attends over $d_{k}$ slots instead of all earlier writers, at a cost independent of depth.
Since $M$ accumulates writes layer by layer, it is a second residual stream, the matrix stream, besides the original vector stream (Figure 4 left).

We call the interface through which a head reads from or writes to a stream a port.
The two streams have dual ports:

 $\begin{split}&\textrm{vector stream: }u_{j}=xW^{s}_{j},\quad\Delta x=o_{i}W^{O}_{i}\\
&\textrm{matrix stream: }u_{j}=q^{m}_{j}M,\quad\Delta M=k^{m}_{i}\otimes o_{i}\end{split}$ 

(4)

where $k^{m}_{i}\otimes o_{i}=(k^{m}_{i})^{\top}o_{i}$ is the outer product.
Data and ports swap shapes: a port shrinks from a $D\times d$ matrix to a vector of length $d_{k}$ , small enough to be generated per token from the vector stream, which we keep as the control path.

### 3.2 Architecture

LoomFormer keeps the backbone of the Transformer block and changes how sub-layers interface with the two streams: the vector stream carries control, i.e., where to attend and how to wire, and the matrix stream carries the content that is moved (Figure 4).
We describe the architecture bottom up, from ports to sub-layers to the model (pseudocode in Appendix C), with $x$ denoting a sub-layer’s pre-LN input.

#### MRead and MWrite.

MRead and MWrite realize the matrix-stream ports of Eq. (4).
 $\operatorname{MRead}(M,x,x^{r})$ generates a query $q^{m}\in\mathbb{R}^{d_{k}}$ to read $u\in\mathbb{R}^{d}$ from $M$ , where $x^{r}=\operatorname{GELU}(xW^{R}_{1})\in\mathbb{R}^{D_{r}}$ with $W^{R}_{1}\in\mathbb{R}^{D\times D_{r}}$ ( $D_{r}\ll D$ ) is a low-rank feature shared by all MRead modules of a sub-layer to keep port generation cheap:

 $\begin{split}&q^{m}=\operatorname{Norm}(x^{r}W^{R}_{2})\odot\sigma(xW^{R}_{g})+\bar{q}\odot\sigma(xW^{R}_{\bar{g}})\\
&u=q^{m}M\end{split}$ 

(5)

Here $x^{r}$ is projected by $W^{R}_{2}\in\mathbb{R}^{D_{r}\times d_{k}}$ , normalized with RMSNorm and gated to get a dynamic query, which is summed with a static query $\bar{q}$ (also gated) to obtain $q^{m}\in\mathbb{R}^{d_{k}}$ . The two scalar gates are computed from $x$ by projecting to scalars with $W^{R}_{g},W^{R}_{\bar{g}}\in\mathbb{R}^{D\times 1}$ and then applying sigmoid.
Similarly, with a shared $x^{w}=\operatorname{GELU}(xW^{W}_{1})\in\mathbb{R}^{D_{r}}$ , $\operatorname{MWrite}(o,x,x^{w})$ generates a key $k^{m}\in\mathbb{R}^{d_{k}}$ and writes a vector $o\in\mathbb{R}^{d}$ into $M$ as their outer product:

 $\begin{split}&k^{m}=\operatorname{Norm}(x^{w}W^{W}_{2})\odot\sigma(xW^{W}_{g})\\
&\Delta M=k^{m}\otimes o\end{split}$ 

(6)

Normalization and gating separate a port’s direction from its magnitude: the normalized vector selects which slots to read or write, and the gate alone sets how much.

#### LoomMHA.

To access the previous token’s memory without horizontal attention, MShift first mixes $M_{[t]}$ with $M_{[t-1]}$ via a per-slot gate $z_{[t]}=\sigma(x_{[t]}W^{S}_{g})\in\mathbb{R}^{d_{k}}$ ( $W^{S}_{g}\in\mathbb{R}^{D\times d_{k}}$ ), and MNorm RMS-normalizes the matrix into $\widetilde{M}_{[t]}$ .
Each head then reads four vectors from $\widetilde{M}$ with separate MRead modules: query, key, value and output bypass $o^{m}$ :

 $\begin{split}&\widetilde{M}_{[t]}=\operatorname{MNorm}\big(z_{[t]}\odot M_{[t-1]}+(1-z_{[t]})\odot M_{[t]}\big)\\
&\tilde{q},\tilde{k},v,o^{m}=\operatorname{MRead}^{Q,K,V,O}(\widetilde{M},x,x^{r})\\
&q,k=\operatorname{RoPE}(xW^{Q,K})\,\|\,(\tilde{q},\tilde{k})\end{split}$ 

(7)

where $W^{Q},W^{K}\in\mathbb{R}^{D\times d/4}$ project $x$ to $d/4$ dimensions, to which rotary position embedding (RoPE) (Su et al., 2024) is applied; the results are concatenated ( $\|$ ) with $\tilde{q}$ and $\tilde{k}$ read from $M$ .
The value is read from $M$ alone: the vector stream supplies only ports and RoPE queries/keys for slot and token routing, while all content moved by attention comes from $M$ .
Unlike MHA, LoomMHA has two outputs: the projected head outputs for the vector stream, and the writes of all heads $i\in\{1,\dots,n\}$ summed into $\Delta M$ :

 $\begin{split}&o_{i}=o^{m}_{i}+\operatorname{Attention}(q_{i},k_{i},v_{i})\\
&\Delta M=\textstyle\sum_{i=1}^{n}\operatorname{MWrite}_{i}(\operatorname{Norm}(o_{i}),x,x^{w})\\
&\operatorname{LoomMHA}(x,M)=\big(\operatorname{Concat}_{i=1}^{n}[o_{i}]W^{O},\,\Delta M\big)\end{split}$ 

(8)

where $q_{i},k_{i},v_{i},o^{m}_{i}$ are given by Eq. (7) for head $i$ , and
 $\operatorname{Attention}$ is scaled dot-product attention over tokens with a strictly causal mask that excludes the current token: a token’s own content reaches the head’s output through the bypass $o^{m}_{i}$ instead.
LoomMHA is where vertical and horizontal attention are interwoven: the vertical reads from $M$ feed the horizontal attention of head $i$ , and its output $o_{i}$ is written back to $M$ for later layers.

#### FFN with MWrite.

The feed-forward network (FFN) sub-layer reads only $x$ and, like LoomMHA, returns a pair $(\operatorname{FFN}_{x}(x),\Delta M)$ , where $\operatorname{FFN}_{x}$ is a standard FFN module.
A second, smaller $\operatorname{FFN}_{M}$ computes the values to write, which are split into $n$ head-sized vectors ( $\operatorname{split}_{i}$ ) and written to $M$ with the sub-layer’s MWrite modules.
This write is the only channel through which content of the vector stream enters $M$ , since attention moves only content read from $M$ .

 $\Delta M=\textstyle\sum_{i=1}^{n}\operatorname{MWrite}_{i}(\operatorname{split}_{i}(\operatorname{FFN}_{M}(x)),x,x^{w})$ 

(9)

#### LoomFormer.

The embedding initializes both residual streams; $M_{0}$ is written from the embedding $x_{0}$ in the same way as the FFN writes:

 $\begin{split}&x_{0}=\operatorname{Embedding}(X)\\
&M_{0}=\textstyle\sum_{i=1}^{n}\operatorname{MWrite}_{i}(\operatorname{split}_{i}(x_{0}W^{E}),x_{0},x^{w}_{0})\end{split}$ 

(10)

where $W^{E}\in\mathbb{R}^{D\times D}$ is a value projection and $x^{w}_{0}=\operatorname{GELU}(x_{0}W^{W}_{1})$ .
Each block $l$ then applies the two sub-layers with pre-LN, updating both streams residually (pairs are added element-wise):

 $\begin{split}&(x^{\prime}_{l},M^{\prime}_{l})=(x_{l},M_{l})+\operatorname{LoomMHA}(\operatorname{LN}_{1}(x_{l}),M_{l})\\
&(x_{l+1},M_{l+1})=(x^{\prime}_{l},M^{\prime}_{l})+\operatorname{FFN}(\operatorname{LN}_{2}(x^{\prime}_{l}))\end{split}$ 

(11)

for $l=0,\dots,L-1$ .
Finally, $x_{L}$ is normalized and projected to produce the final output.
As $M_{L}$ is not used, the last block does not write to $M$ .
We typically set $d_{k}=2n$ slots (so $M$ has size $2D$ ) and $D_{r}=nd_{k}/2=n^{2}$ (a $2\times$ compression bottleneck relative to $nd_{k}$ ), matching the Transformer’s parameter budget by adjusting FFN widths at an approximately $2{:}1$ ratio between $\operatorname{FFN}_{x}$ and the $\operatorname{FFN}_{M}$ write branch.

### 3.3 Rewiring Capacity

On the matrix stream without MShift, reader $\mathsf{R}_{j}$ receives $u_{j}=q^{m}_{j}M=\sum_{i}\langle q^{m}_{j},k^{m}_{i}\rangle\,o_{i}$ by Eq. (4).
The matrix stream thus keeps the division of roles of naive loom attention: its virtual weights are identities, and its wiring matrix is $C=KQ^{\top}$ , of rank at most $d_{k}$ , where the rows of $K$ and $Q$ are the keys $k^{m}_{i}$ of the writers and the queries $q^{m}_{j}$ of the readers.

###### Proposition 3.1.

(i) With ports fixed to $(K,Q)$ , LoomFormer is a standard Transformer $\operatorname{Tfm}_{K,Q}$ with residual stream $(x,\operatorname{vec}(M))\in\mathbb{R}^{D+d_{k}d}$ , whose virtual weights absorb $KQ^{\top}$ , hence node-gated.
(ii) With ports generated per token, LoomFormer is rewirable with one head per reader w.r.t. the wiring patterns with at most $d_{k}$ writers, whatever the total number of writers $w$ .

In (ii), only the writers in the pattern are on, each taking a slot through a one-hot key while each reader’s query selects the slots of its writers (Section A.6).
The $N(w,r-1)$ heads per reader that node gating needs (Theorem 2.4) are thus replaced by a per-token assignment of $d_{k}$ slots; when more than $d_{k}$ writers are on, slots are shared and writers interfere, as in linear attention.
By (i), this rewiring capacity comes from generating ports per token, not from the matrix memory itself.
Together, (i) and (ii) give a model-level view of LoomFormer’s expressiveness: within each token, LoomFormer wires heads as $\operatorname{Tfm}_{K(x),Q(x)}$ , selecting per token one Transformer from the continuum $\{\operatorname{Tfm}_{K,Q}\}$ much as a mixture of experts selects experts (Fedus et al., 2022), except that the experts are wirings, share all other weights, and cost only $d_{k}$ numbers per port to select.

FIGCAPTION: Table 1: LoomFormer among residual-stream mechanisms.
Ports $k^{m}$ , $q^{m}$ are indexed by heads ${\color[rgb]{0.8789,0,0}\boldsymbol{i}}$ , ${\color[rgb]{0.8789,0,0}\boldsymbol{j}}$ or, when shared across a layer, by layers ${\color[rgb]{0,0,0.8789}l_{i}}$ , ${\color[rgb]{0,0,0.8789}l_{j}}$ (and Q/K/V input ${\color[rgb]{0.4961,0,1}s}$ ), and marked $(x)$ or $(M)$ when computed from the vector stream or from the memory itself; $\operatorname{softmax}_{{\color[rgb]{0,0,0.8789}l_{i}}}$ normalizes over writer layers.
Typical sizes assume $d_{k}=4$ in HC and $d_{k}=2n$ in LoomFormer ( ${+}D$ for its vector stream).

Effective wiring
Wiring rank
Granularity
 

Memory: slots $\times$ width

 $=$ typical size

 

Residual

stream

Transformer
 $\bar{C}_{ij}$ 
 $1$ 
–
 $1\times D$ ${}={}$ 
 $D$ 
vector

DenseFormer
 $\langle q^{m}_{{\color[rgb]{0,0,0.8789}l_{j}}},k^{m}_{{\color[rgb]{0,0,0.8789}l_{i}}}\rangle\,\bar{C}_{ij}$ 
 $1$ 
layer
 $l\times D$ ${}={}$ 
 $lD$ 
vector

HC / mHC
 $\langle q^{m}_{{\color[rgb]{0,0,0.8789}l_{j}}}(M),k^{m}_{{\color[rgb]{0,0,0.8789}l_{i}}}(M)\rangle\,\bar{C}_{ij}$ 
 $1$ 
layer
 $d_{k}\times D$ ${}={}$ 
 $4D$ 
matrix

AttnRes (full)
 $\operatorname{softmax}_{{\color[rgb]{0,0,0.8789}l_{i}}}\langle q^{m}_{{\color[rgb]{0,0,0.8789}l_{j}}},k^{m}_{{\color[rgb]{0,0,0.8789}l_{i}}}(M)\rangle\,\bar{C}_{ij}$ 
 $1$ 
layer
 $2l\times D$ ${}={}$ 
 $2lD$ 
none

MUDDFormer
 $\langle q^{m,{\color[rgb]{0.4961,0,1}s}}_{{\color[rgb]{0,0,0.8789}l_{j}}}(x),k^{m}_{{\color[rgb]{0,0,0.8789}l_{i}}}\rangle\,\bar{C}_{ij}$ 
 $3$ 
layer $\times$ QKV
 $l\times D$ ${}={}$ 
 $lD$ 
vector

LoomFormer
 $\langle q^{m,{\color[rgb]{0.4961,0,1}s}}_{{\color[rgb]{0.8789,0,0}\boldsymbol{j}}}(x),k^{m}_{{\color[rgb]{0.8789,0,0}\boldsymbol{i}}}(x)\rangle\,I$ 
 $\min(d_{k},3n)=2n$ 
head $\times$ QKV
 $d_{k}\times d$ ${}={}$ 
 $2D\,({+}D)$ 
matrix + vector

## 4 Connections to Prior Work

Table 1 compares LoomFormer with residual-stream mechanisms (derivations in Section A.7; extended related work in Appendix B).
In DenseFormer (Pagliardini et al., 2024), HC (Zhu et al., 2025), mHC (Xie et al., 2025), AttnRes (Kimi Team, 2026a) and MUDDFormer (Xiao et al., 2025), the ports are shared by all heads of a layer: apart from node gating and the fixed projections, their wiring depends on a writer only through its layer and on a reader only through its layer (and Q/K/V input in MUDDFormer).
Their wiring rank is therefore $1$ , or $3$ in MUDDFormer, one per Q/K/V input, whereas LoomFormer generates its ports per head and per token and reaches $\min(d_{k},3n)$ , i.e., $2n$ by default, with a smaller state than HC ( $2D+D$ against $4D$ ).

#### Linear attention.

Linear attention (Katharopoulos et al., 2020; Schlag et al., 2021) and its gated and delta-rule variants (Dao & Gu, 2024; Yang et al., 2024a; Yang et al., 2024b; Yang et al., 2025; Peng et al., 2024) also read and write a fixed-size matrix state per head with keys and queries from the vector stream, but differ in axis and sharing.
Their state accumulates across tokens and its read is the head output, replacing horizontal attention, whereas $M$ accumulates along depth and its reads feed the query, key and value of horizontal attention.
Their state is also private to one head across tokens, so heads still compose through the static ports of the vector stream, whereas $M$ is shared by all heads of all layers at a token, as a channel through which heads read each other.
The two are thus orthogonal and can be combined.

## 5 Experiments

We train all models on the Pile dataset (Gao et al., 2020) in JAX, initializing dynamic port gates near zero so training starts smoothly from static ports (Section D.1).
We evaluate scaling laws (Section 5.1), large-scale pretraining and downstream performance (Section 5.2), mechanisms of the trained models (Section 5.3), computational efficiency (Section 5.4) and ablations (Section 5.5).

### 5.1 Scaling Laws

Settings.
We train three model scales—Medium ( $432\textrm{M}$ ), Large ( $778\textrm{M}$ ) and XL ( $1.4\textrm{B}$ )—with sequence length $T=4096$ , batch size $0.5\textrm{M}$ tokens ( $128$ sequences), and token budgets proportional to non-embedding parameters following Hoffmann et al. (2022) (architectural dimensions and optimizer hyperparameters in Section D.1).

Baselines.
We compare LoomFormer against recent residual-stream baselines on the same Transformer++ backbone (Touvron et al., 2023) (RoPE (Su et al., 2024), SwiGLU FFN (Shazeer, 2020), RMSNorm, no linear bias):
DHC (dynamic HC with expansion rate $4$ ; Zhu et al., 2025), mHC (Xie et al., 2025), AttnRes (full) (Kimi Team, 2026a), and MUDDFormer (Xiao et al., 2025) (details in Section D.2).

FIGCAPTION: Figure 5: Scaling curves on Pile validation loss against training compute (total parameters $\times$ training tokens).

Results.
Across the Medium, Large and XL scales (Figure 5), mHC, AttnRes and DHC improve over Transformer++, and MUDDFormer improves further, while LoomFormer achieves the lowest loss at every scale.
Interpolating log-linearly along each baseline curve across Medium and Large, LoomFormer matches Transformer++ trained with 3.19–3.25× compute (3.10× net at Large after the 3.1% FLOP overhead in Table 3) and all four residual-stream baselines at 1.73–2.76×.
At XL, where LoomFormer lies below the largest model of every baseline, it lowers validation loss from $2.101$ to $1.979$ ( $\Delta L=0.122$ , 1.8× MUDDFormer’s $0.068$ ) at 2.8% extra FLOPs (Table 3).

FIGCAPTION: Table 2: Zero/five-shot downstream evaluation results ( $\Delta$ acc is over Pythia-2.8B; best in bold within each block).

Model

 

Pile

ppl $\downarrow$ 

 

FLAN

ppl $\downarrow$ 

 

LAM

BADA

PIQA

 

Wino

Grande

 

ARC

-E

 

ARC

-C

SciQ

BoolQ

 

Hella

Swag

 

RACE

-M

 

RACE

-H

 

Avg

acc $\uparrow$ / $\Delta$ acc

Pythia-2.8B

6.63

8.16

0-shot

64.7

73.9

59.4

64.4

29.5

88.2

64.5

45.4

38.1

34.9

56.3

Pythia-6.9B

6.29

7.85

67.3

75.2

60.9

67.3

31.3

89.7

63.7

48.0

40.6

37.0

58.1

mHC-Pythia-2.8B

6.24

7.35

68.5

74.4

62.8

67.1

30.7

90.5

66.9

47.5

42.3

37.0

58.8/+2.5

Pythia-12B

6.01

7.26

70.5

76.0

63.9

70.2

31.8

90.2

67.4

50.3

40.6

38.3

59.9

LoomPythia-2.8B

6.02

6.94

71.8

74.7

63.7

68.5

31.7

92.4

65.8

49.1

42.3

39.0

59.9/+3.6

Pythia-2.8B

-

-

5-shot

60.5

73.6

60.6

67.3

32.3

94.3

65.6

45.1

38.4

35.6

57.3

Pythia-6.9B

-

-

63.8

75.5

63.7

70.2

35.6

95.1

65.7

48.1

39.0

36.5

59.3

mHC-Pythia-2.8B

-

-

62.4

74.9

64.1

70.8

35.5

95.6

69.4

47.6

40.3

38.2

59.9/+2.6

Pythia-12B

-

-

67.3

76.0

64.2

71.0

36.5

95.3

68.0

50.3

40.1

38.8

60.8

LoomPythia-2.8B

-

-

69.0

75.7

65.7

70.8

35.1

96.0

71.2

49.5

43.1

40.1

61.6/+4.3

FIGCAPTION: Figure 6: Mechanistic comparison of 2.8B models across layers $l=1,\dots,32$ on Pile validation sequences (definitions in Appendix H).
(a) Energy effective rank of the token-centered dynamic Q/K/V wiring matrix (band: 10th–90th percentiles) and per single Q, K or V input, vs. the bounds of Table 1.
(b) Cross-token cosine distance ( $1-\cos$ ) of read ports ( $\tilde{q},\tilde{k},v,o^{m}$ ) and write ports ( $k^{m}_{\mathrm{MHA}},k^{m}_{\mathrm{FFN}}$ ) in $\mathbb{R}^{d_{k}}$ (static/gated $=0$ ).
(c) Percentage of heads that are on (non-sink; legend: 32-layer mean).
(d) Within-head writer routing similarity across port pairs in layers 3–32.

### 5.2 Large-Scale Pretraining and Evaluation

Settings.
To test whether LoomFormer’s gains persist under extended training, we compare against the Pythia suite (Biderman et al., 2023) on 300B Pile tokens (over 5× the Chinchilla budget of Section 5.1; Hoffmann et al., 2022).
Specifically, we train LoomPythia-2.8B and mHC-Pythia-2.8B—the residual-stream baseline that also keeps a depth-independent state size (Table 1)---and compare them with Pythia 2.8B--12B.
Except for using serial instead of parallel sub-layers33
 3
 
 
 
 
 
 
 
 mHC requires serial sub-layers by design, and parallel vs. serial FFN placement is quality-neutral for standard Transformers, as shown by Chowdhery et al. (2023) and our own experiments., both LoomPythia and mHC-Pythia use the same architecture choices and training hyperparameters as Pythia (Biderman et al., 2023, Appendix E).
We evaluate zero- and five-shot accuracy across 10 standard benchmarks.

Results.
As shown in Table 2 and Figure 2, LoomPythia-2.8B improves average accuracy over Pythia-2.8B by $+3.6$ in 0-shot and $+4.3$ in 5-shot ( $+3.95$ on their mean), achieving 1.6× the gain of mHC-Pythia-2.8B ( $+2.5$ / $+2.6$ , mean $+2.55$ ).
Its 0-shot average ( $59.9$ ) matches Pythia-12B ( $59.9$ ), its 5-shot average ( $61.6$ ) exceeds Pythia-12B ( $60.8$ ), and its 0/5-shot mean ( $60.75$ ) matches a log-linear extrapolation of Pythia at ~4.9× compute.
On pretraining metrics, LoomPythia-2.8B reaches Pile ppl $6.02$ , on par with Pythia-12B ( $6.01$ ) and well ahead of mHC-Pythia-2.8B ( $6.24$ ).

As an out-of-domain evaluation, Table 2 also reports target-span perplexities on a randomly sampled subset of the FLAN Collection (Longpre et al., 2023), which features instruction following, chain-of-thought and in-context few-shot data.
The advantage of LoomPythia on FLAN is even larger than on Pile: FLAN ppl $6.94$ is best in the table, beating both Pythia-12B ( $7.26$ ) and mHC-Pythia-2.8B ( $7.35$ ).
Whereas mHC’s gain barely changes from 0-shot to 5-shot ( $+2.5$ vs. $+2.6$ ), LoomPythia’s widens from $+3.6$ to $+4.3$ , indicating stronger in-context learning that Section 5.3 traces to per-token rewiring freeing deep heads from attention sinks.

### 5.3 Mechanistic Analysis

We analyze Pythia-2.8B, mHC-Pythia-2.8B and LoomPythia-2.8B (Section 5.2) on held-out Pile validation sequences; metric definitions and additional visualizations are in Appendix H.

Per-token circuit rewiring.
After subtracting each pair’s token-mean static component, the dynamic wiring matrix on the $3n=96$ Q/K/V readers of layer $l$ has rank $0$ under fixed ports (Proposition 3.1(i)), at most $1$ in Transformer and mHC, at most $3$ in MUDDFormer, and up to $\min(d_{k},3n)=64$ in LoomFormer (Table 1).
In Figure 6(a), LoomPythia-2.8B’s dynamic effective rank rises with depth to $41.0$ ( $26.4$ mean), placing $71.0\%$ of dynamic wiring energy ( $90.7\%$ at layer 32) outside rank $1$ , and averages $14.3$ ( $20.9$ at layer 32) within a single Q, K or V input where MUDDFormer also has rank $\leq 1$ .
Figure 6(b) directly tests Proposition 3.1(i) vs. (ii) in $\mathbb{R}^{d_{k}}$ : whereas static or scalar-gated ports have zero cross-token cosine distance, LoomPythia’s read ports $\tilde{q},\tilde{k},v,o^{m}$ average $0.33$ – $0.46$ (up to $0.68$ ) and its attention and $\operatorname{FFN}_{M}$ write ports average $0.36$ and $0.56$ (up to $0.75$ ; Appendix H traces an in-context circuit rewiring across adjacent tokens).

Head utilization and within-head specialization.
In a node-gated model (Section 2, Theorem 2.4), heads whose fixed inputs are not needed at a token turn themselves off by attending to sink tokens (Vig & Belinkov, 2019; Xiao et al., 2024b).
Counting non-sink heads as on (Xiao et al., 2025), Figure 6(c) shows that in layers 17–32, only $22.9\%$ of heads are on in Pythia-2.8B and $17.7\%$ in mHC-Pythia-2.8B—whose four streams still merge into one shared input per layer—whereas LoomPythia-2.8B keeps $74.4\%$ on ( $3.3\times$ and $4.2\times$ ), more than doubling content-token attention mass and accounting for its 5-shot and FLAN gains (Section 5.2).
Finally, Figure 6(d) verifies $\text{head}\times\text{QKV}$ specialization (Table 1): although routing looks correlated at the layer level ( $0.92$ – $0.96$ ), per-writer correlation within the same head drops to $0.71$ – $0.84$ and top-32 writer overlap is $0.38$ – $0.49$ , so over half of the top writers read by a head’s $\tilde{q},\tilde{k},v,o^{m}$ are distinct.

FIGCAPTION: Table 3: Analytical computational overhead ( $+\textrm{FLOPs}$ ) and measured relative throughput of LoomFormer vs. matched-size Transformer at sequence length $T$ (absolute speeds in Appendix F).

Model
 

 $+\textrm{FLOPs}$ 

(%)

 

Train (%)

v5p

 

Prefill (%)

bs $1$  /  $8$ 

 

Decode (%)

bs $1$  /  $16$  /  $64$ 

Medium
 $3.70$ 
 $90.5$ 
 $87.4$  /  $90.4$ 
 $91.7$  /  $89.2$  /  $92.3$ 

Large
 $3.13$ 
 $89.2$ 
 $85.3$  /  $86.2$ 
 $85.4$  /  $87.8$  /  $84.9$ 

XL
 $2.77$ 
 $92.3$ 
 $83.8$  /  $85.3$ 
 $86.9$  /  $84.6$  /  $86.6$ 

2.8B
 $1.95$ 
 $92.4$ 
 $87.5$  /  $90.5$ 
 $86.4$  /  $87.6$  /  $88.9$ 

### 5.4 Computational Complexity and Efficiency

Analytical computational overhead.
Under a matched parameter budget, LoomFormer adds ~ $\frac{1}{d(1+\rho/12)}+\frac{1}{8(1+12/\rho)}$ relative forward FLOPs from reading/writing $M$ and wider Q/K ( $d_{k}=2n$ , $\rho=T/D$ ; derived in Appendix E), totaling $1.95\%$ – $3.70\%$ across our four scales (Table 3).

Wall-clock training and inference throughput.
Table 3 reports relative throughput for JAX training on TPU v5p pods (with fused Pallas kernels) and BF16 PyTorch inference on an NVIDIA A800 GPU (with a fused Triton kernel and torch.compile; Appendix F).
LoomFormer sustains $89\%$ – $92\%$ of Transformer training speed and $84\%$ – $92\%$ of inference speed across prefill and decode batch sizes; the gap beyond the $2.0\%$ – $3.7\%$ FLOPs reflects HBM I/O on $M$ , lower arithmetic intensity of per-token $M$ contractions, and smaller split projections and norms, and is far outweighed by the over 3× compute savings (Sections 5.1 and 5.2).

### 5.5 Ablation Studies

We ablate LoomFormer at the Medium scale (432M, 7B tokens; Section 5.1), reallocating saved parameters to $\operatorname{FFN}_{x}$ to match the parameter budget, and report Pile validation loss and the retained gain $\textrm{Gain}=(L_{\textrm{Tfm++}}-L)/(L_{\textrm{Tfm++}}-L_{\textrm{Loom}})$ (Table 4; full suite in Appendix G).

FIGCAPTION: Table 4: Ablations of LoomFormer at the Medium scale.

Variant
Loss $\downarrow$ 
 $\Delta$ Loss
Gain

LoomFormer (full)
2.266
—
100.0%

Transformer++ (baseline)
2.443
 $+$ 0.177
0.0%

Dynamic rewiring (Proposition 3.1)

Static write keys ( $k^{m}$ )
2.283
 $+$ 0.017
90.3%

Static read queries ( $q^{m}$ )
2.310
 $+$ 0.043
75.4%

Static read & write ( $q^{m},k^{m}$ )
2.344
 $+$ 0.078
56.0%

Port normalization & gating (Eqs. 5–6)

 $-$ dynamic port gates
2.277
 $+$ 0.011
93.7%

 $-$ dynamic port $\operatorname{Norm}$ 
2.289
 $+$ 0.023
87.1%

 $-$ dynamic port gates & $\operatorname{Norm}$ 
2.283
 $+$ 0.017
90.3%

Sub-layer components (Section 3.2)

 $-$ $\operatorname{FFN}_{M}$ write
2.300
 $+$ 0.034
80.7%

 $-$ output bypass $o^{m}$ 
2.299
 $+$ 0.032
81.8%

 $-$ vector-stream RoPE $W^{Q,K}$ 
2.286
 $+$ 0.020
88.7%

 $-$ $\operatorname{MShift}$ 
2.277
 $+$ 0.010
94.2%

Dynamic rewiring.
Replacing dynamic $k^{m}$ , dynamic $q^{m}$ , or both with learned static vectors (dropping their gates) tests Proposition 3.1.
Fixing both reduces LoomFormer to $\operatorname{Tfm}_{K,Q}$ and cuts the gain to 56.0%—still ahead of layer-wise dynamic baselines (Figure 5)—so the static head-level dual-stream backbone and per-token rewiring contribute roughly equal halves.
Fixing $q^{m}$ hurts far more than fixing $k^{m}$ (75.4% vs. 90.3% retained): writers can use stable slots if readers select among them per token.

Port normalization and gating.
Removing either the dynamic-branch gates or port RMSNorm hurts, yet removing both beats removing RMSNorm alone (90.3% vs. 87.1%): an unnormalized projection already varies in norm, making a separate gate redundant.
Pairing RMSNorm with gating works best by separating slot direction from read/write magnitude.

FIGCAPTION: Figure 7: Validation loss vs. per-token memory size and matrix slots $d_{k}$ at the Medium scale ( $L=18$ ).

Sub-layer components.
Removing $\operatorname{FFN}_{M}$ (Eq. 9) hurts most, as the sole post-embedding path from the vector stream into $M$ , followed closely by removing the bypass $o^{m}$ and restoring self-attention to the current token (Eq. 8).
Applying RoPE to $d/4$ of the $M$ -read coordinates instead of $xW^{Q,K}$ (Eq. 7) also hurts, supporting separate streams for positions and content, whereas removing $\operatorname{MShift}$ still retains 94.2% of the gain.

Memory capacity.
As we vary the slot count $d_{k}$ (Figure 7), validation loss drops sharply ( $-0.026$ ) from $d_{k}=16$ to $32$ with diminishing returns beyond, placing our default $d_{k}=2n$ at the knee given the linear growth of MRead/MWrite FLOPs (Appendix E) and I/O with $d_{k}$ .
Even at $d_{k}=16$ ( $2D$ total state), LoomFormer already outperforms DHC ( $4D$ ) and MUDDFormer ( $lD$ , averaging $9D$ across $L=18$ layers), showing that head-level rewiring uses state memory far more effectively than layer-wise mechanisms.

## 6 Conclusion

Viewing Transformer circuits through their dynamic wiring matrix shows that standard Transformers and layer-wise residual-stream mechanisms are restricted to low-rank gating, requiring many heads per reader to avoid circuit interference.
Loom attention lifts this bottleneck by interweaving vertical linear attention over a per-token matrix memory with horizontal attention over tokens, making circuits rewirable per token at one head per reader and $2.0\%$ – $3.7\%$ extra forward FLOPs.
Across model scales and 300B-token pretraining, LoomFormer matches Transformers trained with over 3× compute and recent residual-stream baselines with ~1.7–2.8×, while keeping over three quarters of its attention heads on and rewiring its ports from token to token.
Because the matrix memory operates along depth at each token, combining loom attention with linear or sparse attention along the sequence is a natural direction for future foundation models.

## References

 

Biderman et al. (2023)

Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O’Brien, K.,
Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., et al.

Pythia: A suite for analyzing large language models across training
and scaling.

In International Conference on Machine Learning (ICML), pp. 2397–2430. PMLR, 2023.

 

Bisk et al. (2020)

Bisk, Y., Zellers, R., Gao, J., Choi, Y., et al.

Piqa: Reasoning about physical commonsense in natural language.

In Proceedings of the AAAI conference on artificial
intelligence, volume 34, pp. 7432–7439, 2020.

 

Brandon et al. (2024)

Brandon, W., Mishra, M., Nrusimha, A., Panda, R., and Kelly, J. R.

Reducing transformer key-value cache size with cross-layer attention.

arXiv preprint arXiv:2405.12981, 2024.

 

Chowdhery et al. (2023)

Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A.,
Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., et al.

PaLM: Scaling language modeling with pathways.

Journal of Machine Learning Research, 24(240):1–113, 2023.

 

Clark et al. (2019)

Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins, M., and Toutanova,
K.

Boolq: Exploring the surprising difficulty of natural yes/no
questions.

arXiv preprint arXiv:1905.10044, 2019.

 

Clark et al. (2018)

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and
Tafjord, O.

Think you have solved question answering? try arc, the ai2 reasoning
challenge.

arXiv preprint arXiv:1803.05457, 2018.

 

Conmy et al. (2023)

Conmy, A., Mavor-Parker, A., Lynch, A., Heimersheim, S., and Garriga-Alonso, A.

Towards automated circuit discovery for mechanistic interpretability.

In Advances in Neural Information Processing Systems
(NeurIPS), volume 36, pp. 16318–16352, 2023.

 

Dai et al. (2024)

Dai, D., Deng, C., Zhao, C., Xu, R., Gao, H., Chen, D., Li, J., Zeng, W., Yu,
X., Wu, Y., et al.

Deepseekmoe: Towards ultimate expert specialization in
mixture-of-experts language models.

arXiv preprint arXiv:2401.06066, 2024.

 

Dao & Gu (2024)

Dao, T. and Gu, A.

Transformers are ssms: Generalized models and efficient algorithms
through structured state space duality.

In Proceedings of the Forty-First International Conference on
Machine Learning (ICML), 2024.

 

D’yachkov & Rykov (1982)

D’yachkov, A. G. and Rykov, V. V.

Bounds on the length of disjunctive codes.

Problemy Peredachi Informatsii, 18(3):7–13, 1982.

 

Elhage et al. (2020)

Elhage, N., Neel, N., Olsson, C., et al.

A mathematical framework for transformer circuits.

Transformer Circuits Thread, 2020.

URL https://transformer-circuits.pub/2021/framework/index.html.

 

Elhage et al. (2022)

Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S.,
Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., et al.

Toy models of superposition.

Transformer Circuits Thread, 2022.

 

ElNokrashy et al. (2022)

ElNokrashy, M., AlKhamissi, B., and Diab, M.

Depth-wise attention (dwatt): A layer fusion method for
data-efficient classification.

arXiv preprint arXiv:2209.15168, 2022.

 

Erdős et al. (1985)

Erdős, P., Frankl, P., and Füredi, Z.

Families of finite sets in which no set is covered by the union of r
others.

Israel Journal of Mathematics, 51:79–89, 1985.

 

Fang et al. (2023)

Fang, Y., Cai, Y., Chen, J., Zhao, J., Tian, G., and Li, G.

Cross-layer retrospective retrieving via layer attention.

arXiv preprint arXiv:2302.03985, 2023.

 

Fedus et al. (2022)

Fedus, W., Zoph, B., and Shazeer, N.

Switch transformers: Scaling to trillion parameter models with simple
and efficient sparsity.

The Journal of Machine Learning Research, 23(1):5232–5270, 2022.

 

Füredi (1996)

Füredi, Z.

On r-cover-free families.

Journal of Combinatorial Theory, Series A, 73(1):172–173, 1996.

 

Gao et al. (2020)

Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang,
J., He, H., Thite, A., Nabeshima, N., et al.

The pile: An 800gb dataset of diverse text for language modeling.

arXiv preprint arXiv:2101.00027, 2020.

 

Gao et al. (2023)

Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi, A., Foster, C.,
Golding, L., Hsu, J., Le Noac’h, A., Li, H., McDonell, K., Muennighoff, N.,
Ociepa, C., Phang, J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika,
L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A.

A framework for few-shot language model evaluation, 12 2023.

URL https://zenodo.org/records/10256836.

 

Gu & Dao (2024)

Gu, A. and Dao, T.

Mamba: Linear-time sequence modeling with selective state spaces.

In Proceedings of the First Conference on Language Modeling,
2024.

 

Hanna et al. (2024)

Hanna, M., Pezzelle, S., and Belinkov, Y.

Have faith in faithfulness: Going beyond circuit overlap when finding
model mechanisms.

In Proceedings of Conference on Language Modeling (COLM),
2024.

 

He et al. (2016)

He, K., Zhang, X., Ren, S., and Sun, J.

Deep residual learning for image recognition.

In Proceedings of the IEEE conference on computer vision and
pattern recognition (CVPR), pp. 770–778, 2016.

 

Hoffmann et al. (2022)

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford,
E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., et al.

An empirical analysis of compute-optimal large language model
training.

Advances in Neural Information Processing Systems,
35:30016–30030, 2022.

 

Huang et al. (2017)

Huang, G., Liu, Z., Van Der Maaten, L., and Weinberger, K. Q.

Densely connected convolutional networks.

In Proceedings of the IEEE conference on computer vision and
pattern recognition (CVPR), pp. 4700–4708, 2017.

 

Katharopoulos et al. (2020)

Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F.

Transformers are RNNs: Fast autoregressive transformers with linear
attention.

In Proceedings of the 37th International Conference on Machine
Learning (ICML), 2020.

 

Kautz & Singleton (1964)

Kautz, W. H. and Singleton, R. C.

Nonrandom binary superimposed codes.

IEEE Transactions on Information Theory, 10(4):363–377, 1964.

 

Kimi Team (2026a)

Kimi Team.

Attention residuals.

arXiv preprint arXiv:2603.15031, 2026a.

 

Kimi Team (2026b)

Kimi Team.

Kimi K3: Open frontier intelligence.

arXiv preprint arXiv:2607.24653, 2026b.

 

Lai et al. (2017)

Lai, G., Xie, Q., Liu, H., Yang, Y., and Hovy, E.

Race: Large-scale reading comprehension dataset from examinations.

arXiv preprint arXiv:1704.04683, 2017.

 

Leviathan et al. (2024)

Leviathan, Y., Kalman, M., and Matias, Y.

Selective attention improves transformer.

arXiv preprint arXiv:2410.02703, 2024.

 

Liu et al. (2024)

Liu, A., Feng, B., Wang, B., Wang, B., Liu, B., Zhao, C., Dengr, C., Ruan, C.,
Dai, D., Guo, D., et al.

Deepseek-v2: A strong, economical, and efficient mixture-of-experts
language model.

arXiv preprint arXiv:2405.04434, 2024.

 

Longpre et al. (2023)

Longpre, S., Hou, L., Vu, T., Webson, A., Chung, H. W., Tay, Y., Zhou, D., Le,
Q. V., Zoph, B., Wei, J., et al.

The flan collection: Designing data and methods for effective
instruction tuning.

arXiv preprint arXiv:2301.13688, 2023.

 

Mak & Flanigan (2025)

Mak, B. and Flanigan, J.

Residual matrix transformers: Scaling the size of the residual
stream.

In Proceedings of the 42nd International Conference on Machine
Learning (ICML), pp. 42712–42729, 2025.

 

Merullo et al. (2024)

Merullo, J., Eickhoff, C., and Pavlick, E.

Talking heads: Understanding inter-layer communication in transformer
language models.

arXiv preprint arXiv:2406.09519, 2024.

 

Ni et al. (2025)

Ni, R., Xiao, D., Meng, Q., Li, X., Zheng, S., and Liang, H.

Benchmarking and understanding compositional relational reasoning of
llms.

In Proceedings of the Thirty-Ninth AAAI Conference on
Artificial Intelligence (AAAI), 2025.

 

Olsson et al. (2022)

Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T.,
Mann, B., Askell, A., Bai, Y., Chen, A., et al.

In-context learning and induction heads.

Transformer Circuits Thread, 2022.

 

Pagliardini et al. (2024)

Pagliardini, M., Mohtashami, A., Fleuret, F., and Jaggi, M.

Denseformer: Enhancing information flow in transformers via depth
weighted averaging.

In Proceedings of the Thirty-Eighth Annual Conference on Neural
Information Processing Systems (NeurIPS), 2024.

 

Paperno et al. (2016)

Paperno, D., Kruszewski, G., Lazaridou, A., Pham, Q. N., Bernardi, R.,
Pezzelle, S., Baroni, M., Boleda, G., and Fernández, R.

The lambada dataset: Word prediction requiring a broad discourse
context.

arXiv preprint arXiv:1606.06031, 2016.

 

Peng et al. (2024)

Peng, B., Goldstein, D., Anthony, Q., Albalak, A., Alcaide, E., Biderman, S.,
Cheah, E., Du, X., Ferdinan, T., Hou, H., et al.

Eagle and finch: Rwkv with matrix-valued states and dynamic
recurrence.

arXiv preprint arXiv:2404.05892, 2024.

 

Ruszinkó (1994)

Ruszinkó, M.

On the upper bound of the size of the r-cover-free families.

Journal of Combinatorial Theory, Series A, 66(2):302–310, 1994.

 

Sakaguchi et al. (2021)

Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y.

Winogrande: An adversarial winograd schema challenge at scale.

Communications of the ACM, 64(9):99–106,
2021.

 

Schlag et al. (2021)

Schlag, I., Irie, K., and Schmidhuber, J.

Linear transformers are secretly fast weight programmers.

In Proceedings of the 38th International Conference on Machine
Learning (ICML), 2021.

 

Shazeer (2020)

Shazeer, N.

Glu variants improve transformer.

arXiv preprint arXiv:2002.05202, 2020.

 

Sperner (1928)

Sperner, E.

Ein satz über untermengen einer endlichen menge.

Mathematische Zeitschrift, 27:544–548, 1928.

 

Srivastava et al. (2015)

Srivastava, R. K., Greff, K., and Schmidhuber, J.

Training very deep networks.

Advances in neural information processing systems, 28, 2015.

 

Su et al. (2024)

Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y.

Roformer: Enhanced transformer with rotary position embedding.

Neurocomputing, 568:127063, 2024.

 

Tay et al. (2021)

Tay, Y., Dehghani, M., Aribandi, V., Gupta, J., Pham, P. M., Qin, Z., Bahri,
D., Juan, D.-C., and Metzler, D.

Omninet: Omnidirectional representations from transformers.

In International Conference on Machine Learning (ICML), pp. 10193–10202. PMLR, 2021.

 

Touvron et al. (2023)

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix,
T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al.

Llama: Open and efficient foundation language models.

arXiv preprint arXiv:2302.13971, 2023.

 

Vig & Belinkov (2019)

Vig, J. and Belinkov, Y.

Analyzing the structure of attention in a transformer language model.

arXiv preprint arXiv:1906.04284, 2019.

 

Wang et al. (2023)

Wang, K., Variengien, A., Conmy, A., Shlegeris, B., and Steinhardt, J.

Interpretability in the wild: a circuit for indirect object
identification in gpt-2 small.

In Proceedings of the Eleventh International Conference on
Learning Representations (ICLR), 2023.

 

Wang et al. (2024)

Wang, K., Xia, X., Liu, J., Yi, Z., and He, T.

Strengthening layer interaction via dynamic layer attention.

arXiv preprint arXiv:2406.13392, 2024.

 

Wang et al. (2019)

Wang, Q., Li, B., Xiao, T., Zhu, J., Li, C., Wong, D. F., and Chao, L. S.

Learning deep transformer models for machine translation.

In Proceedings of the 57th Annual Meeting of the Association
for Computational Linguistics (ACL), 2019.

 

Welbl et al. (2017)

Welbl, J., Liu, N. F., and Gardner, M.

Crowdsourcing multiple choice science questions.

arXiv preprint arXiv:1707.06209, 2017.

 

Xiao et al. (2024a)

Xiao, D., Meng, Q., Li, S., and Yuan, X.

Improving transformers with dynamically composable multi-head
attention.

2024a.

 

Xiao et al. (2025)

Xiao, D., Meng, Q., Li, S., and Yuan, X.

MUDDFormer: Breaking residual bottlenecks in transformers via
multiway dynamic dense connections.

In Proceedings of the 42nd International Conference on Machine
Learning (ICML), 2025.

 

Xiao et al. (2024b)

Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M.

Efficient streaming language models with attention sinks.

In The Twelfth International Conference on Learning
Representations (ICLR), 2024b.

 

Xie et al. (2025)

Xie, Z., Wei, Y., Cao, H., Zhao, C., Deng, C., Li, J., Dai, D., Gao, H., Chang,
J., Yu, K., Zhao, L., Zhou, S., Xu, Z., Zhang, Z., Zeng, W., Hu, S., Wang,
Y., Yuan, J., Wang, L., and Liang, W.

mHC: Manifold-constrained hyper-connections.

arXiv preprint arXiv:2512.24880, 2025.

 

Yang et al. (2024a)

Yang, S., Wang, B., Shen, Y., Panda, R., and Kim, Y.

Gated linear attention transformers with hardware-efficient training.

In Proceedings of the 41st International Conference on Machine
Learning (ICML), pp. 56501–56523, 2024a.

 

Yang et al. (2024b)

Yang, S., Wang, B., Zhang, Y., Shen, Y., and Kim, Y.

Parallelizing linear transformers with the delta rule over sequence
length.

In Proceedings of the Thirty-Eighth Annual Conference on Neural
Information Processing Systems, 2024b.

 

Yang et al. (2025)

Yang, S., Kautz, J., and Hatamizadeh, A.

Gated delta networks: Improving Mamba2 with delta rule.

In Proceedings of the Thirteenth International Conference on
Learning Representations (ICLR), 2025.

 

Ye et al. (2024)

Ye, T., Dong, L., Xia, Y., Sun, Y., Zhu, Y., Huang, G., and Wei, F.

Differential transformer.

2024.

 

Zellers et al. (2019)

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y.

Hellaswag: Can a machine really finish your sentence?

arXiv preprint arXiv:1905.07830, 2019.

 

Zhu et al. (2025)

Zhu, D., Huang, H., Huang, Z., Zeng, Y., Mao, Y., Wu, B., Min, Q., and Zhou, X.

Hyper-connections.

In Proceedings of the Thirteenth International Conference on
Learning Representations (ICLR), 2025.

 

## Appendix A Theory of Rewirable Circuits

### A.1 Normalization and proof of Proposition 2.3

###### Lemma A.1.

In a Transformer with pre-LN (LayerNorm or RMSNorm), the contribution of earlier writers to reader $\mathsf{R}_{j}$ ’s input $u_{j}=\operatorname{LN}(x)W^{s}_{j}$ is $\lambda(x)\sum_{i}o_{i}\bar{C}_{ij}$ , where the scale $\lambda(x)>0$ is shared by all writers and the virtual weights $\bar{C}_{ij}$ are fixed.

###### Proof.

Write $x=\sum_{i}o_{i}W^{O}_{i}+x_{\mathrm{other}}$ , where $x_{\mathrm{other}}$ sums the embedding and earlier FFN outputs.
LayerNorm computes $\operatorname{LN}(x)=\lambda(x)\,(x-\mu(x)\mathbf{1})\odot\gamma+\beta$ with $\lambda(x)=\sqrt{D}/\|x-\mu(x)\mathbf{1}\|>0$ (putting $\lambda(x)=0$ on the degenerate set $x\in\operatorname{span}(\mathbf{1})$ ).
Since centering is the fixed linear map $x-\mu(x)\mathbf{1}=x\big(I-\frac{1}{D}\mathbf{1}\mathbf{1}^{\top}\big)$ ,

 $u_{j}=\operatorname{LN}(x)W^{s}_{j}=\lambda(x)\textstyle\sum_{i}o_{i}\bar{C}_{ij}+\lambda(x)\,x_{\mathrm{other}}\big(I-\tfrac{1}{D}\mathbf{1}\mathbf{1}^{\top}\big)\operatorname{diag}(\gamma)W^{s}_{j}+\beta W^{s}_{j},$ 

with fixed virtual weights $\bar{C}_{ij}=W^{O}_{i}\big(I-\frac{1}{D}\mathbf{1}\mathbf{1}^{\top}\big)\operatorname{diag}(\gamma)W^{s}_{j}$ absorbing the linear maps of the normalization (any projection bias adds to $\beta W^{s}_{j}$ ).
For RMSNorm, centering is omitted, $\beta=0$ and $\lambda(x)=\sqrt{D}/\|x\|$ .
∎

###### Proof of Proposition 2.3.

By Lemma A.1, pre-LN rescales all writers of a token by the same $\lambda(x)>0$ .
Within a token, what depends on the input is only this scale and which heads are on: let $g_{i}=\lambda(x)$ if the head of $\mathsf{W}_{i}$ is on and $0$ otherwise, and $g^{\prime}_{j}=1$ if the head of $\mathsf{R}_{j}$ is on and $0$ otherwise.
Then $C=g\,g^{\prime\top}$ , which is Eq. (2), and since $\lambda(x)>0$ , $\mathsf{W}_{i}$ and $\mathsf{R}_{j}$ are connected ( $g_{i}g^{\prime}_{j}\bar{C}_{ij}\neq 0$ ) if and only if both are on and $\bar{C}_{ij}\neq 0$ .
∎

### A.2 Full statement and proof of Theorem 2.4

#### Setting.

The role of reader $\mathsf{R}_{j}$ ( $j\in[r]$ , $r\geq 2$ ) is played by a set $H_{j}$ of heads, so a reader is here a role rather than one input of one head; by Eq. (2), head $h\in H_{j}$ receives $\mathsf{W}_{i}$ when its effective wiring $g_{i}g^{\prime}_{h}\bar{C}_{ih}$ is nonzero.
A wiring task assigns to each reader a writer $\pi(j)\in[w]$ , and the tasks are all maps $\pi:[r]\to[w]$ .
Reader $\mathsf{R}_{j}$ receives exactly $\mathsf{W}_{\pi(j)}$ if some $h\in H_{j}$ receives $\mathsf{W}_{\pi(j)}$ and no $h\in H_{j}$ receives any $\mathsf{W}_{i}$ with $i\neq\pi(j)$ .
We require every head in $H_{j}$ that is on to be free of other writers because query and key inputs feed a softmax, so interference at one head cannot be cancelled downstream.
Every writer in $\pi([r])$ must therefore be on ( $g_{\pi(j)}\neq 0$ ), and turning others on can only add interference, so we may assume the writers that are on are $\pi([r])$ .
For a fixed reader $\mathsf{R}_{j}$ , let $B_{i}=\{h\in H_{j}:\bar{C}_{ih}\neq 0\}$ .
A family $\{B_{i}\}_{i=1}^{w}$ is $t$ -cover-free if $B_{i}\not\subseteq\bigcup_{i^{\prime}\in J}B_{i^{\prime}}$ for every $i\in[w]$ and every $J\subseteq[w]\setminus\{i\}$ with $|J|\leq t$ .

###### Theorem A.2 (full version of Theorem 2.4).

For $r\geq 2$ , a node-gated architecture with $w$ writers realizes all wiring tasks $\pi:[r]\to[w]$ if and only if, for every reader, $\{B_{i}\}_{i=1}^{w}$ is an $(r-1)$ -cover-free family.
Hence each reader needs at least $N(w,r-1)$ heads, and this number suffices.

###### Proof.

If. Given $\pi$ , turn on only the writers $\pi([r])$ .
For reader $\mathsf{R}_{j}$ , let $J=\pi([r])\setminus\{\pi(j)\}$ , so $|J|\leq r-1$ .
Since $\{B_{i}\}$ is $(r-1)$ -cover-free, there is a head $h\in B_{\pi(j)}\setminus\bigcup_{i\in J}B_{i}$ ; turn on $h$ and no other head of $H_{j}$ .
Then $h$ receives $\mathsf{W}_{\pi(j)}$ and none of the other writers that are on.

Only if. Fix a reader $\mathsf{R}_{j}$ , a writer $i$ and a set $J\subseteq[w]\setminus\{i\}$ with $|J|\leq r-1$ .
Choose $\pi$ with $\pi(j)=i$ that maps the other $r-1$ readers onto $J$ , which is possible since $r-1\geq 1$ , $|J|\leq r-1$ and different readers may share a writer (if $J=\emptyset$ , map them to $i$ ).
Realizing $\pi$ requires a head $h\in H_{j}$ that is on with $\bar{C}_{ih}\neq 0$ and $\bar{C}_{i^{\prime}h}=0$ for all $i^{\prime}\in J$ , i.e., $h\in B_{i}\setminus\bigcup_{i^{\prime}\in J}B_{i^{\prime}}$ .
Hence $B_{i}\not\subseteq\bigcup_{i^{\prime}\in J}B_{i^{\prime}}$ , and the family is $(r-1)$ -cover-free.

Counting. The sets $B_{i}$ are subsets of $H_{j}$ , so $|H_{j}|\geq N(w,r-1)$ .
Conversely, given an $(r-1)$ -cover-free family of $w$ sets on a ground set of size $N(w,r-1)$ for each reader, create one head per point and set $\bar{C}_{ih}=I_{d}$ if $h\in B_{i}$ and $0$ otherwise; this is realized in $D=\min\{w,rN(w,r-1)\}\,d$ dimensions by assigning orthogonal $d$ -dimensional subspaces either to the $w$ writers or to the $rN(w,r-1)$ heads.
∎

#### Remarks.

(i) Attention sinks gate heads off only approximately; under the threshold of Section 2, the counting argument is unchanged.
(ii) Sufficiency at $N(w,r-1)$ heads per reader holds in a Transformer when each of the $r$ readers belongs to a separate head (with its other two inputs fixed); when $r=3n$ counts all Q, K and V inputs of $n$ heads, coupling Q, K and V inside the same physical head (and sink gate $g^{\prime}_{h}$ ) can only require more heads, while the lower bound $N(w,r-1)$ still applies.
Likewise, if several heads of a value reader share identical attention weights, their outputs add linearly and interference could in principle cancel, whereas our clean-head requirement is necessary for query and key inputs.
(iii) The model excludes routing through FFNs, discussed in Section A.5.

### A.3 Bounds on $N(w,t)$ and remarks on Theorem 2.4

#### Bounds.

A $1$ -cover-free family is an antichain, so by Sperner’s theorem (Sperner, 1928) $N(w,1)$ is the smallest $N$ with $\binom{N}{\lfloor N/2\rfloor}\geq w$ , about $\log_{2}w$ .
For $2\leq t\leq\sqrt{w}$ , $N(w,t)=\Omega(t^{2}\log w/\log t)$ (D’yachkov & Rykov, 1982; Ruszinkó, 1994; Füredi, 1996) and $N(w,t)=\mathcal{O}(\min\{t^{2}\log w,w\})$ (Erdős et al., 1985); for $t>\sqrt{w}$ , Bassalygo’s bound $N(w,t)\geq\min\{\binom{t+2}{2},w\}$ (D’yachkov & Rykov, 1982) together with the singleton construction $N(w,t)\leq w$ gives $N(w,t)=\Theta(w)$ , reaching $N(w,t)=w$ (one head per writer) once $\binom{t+2}{2}\geq w$ .

#### Group testing.

Theorem A.2 is an instance of non-adaptive group testing: the heads of a reader are tests that must isolate the selected writer from up to $r-1$ other writers that are on.

#### Dense and sparse layouts.

With a single head per reader ( $|H_{j}|=1$ ), the two layouts of Figure 3 are degenerate codes: $B_{1}=\dots=B_{w}=H_{j}$ (dense), which is not $1$ -cover-free and makes circuits interfere, and $B_{i}=\emptyset$ for some writer $i$ (sparse), which leaves $(i,j)$ unreachable.

#### Superposition.

At layer scale ( $r=3n$ , $N(w,r-1)=\Theta(w)$ ), separating $w$ writers as above takes $\Theta(wd)$ dimensions ( $wd$ with orthogonal writer subspaces).
With $nL$ writers sharing $D=nd$ dimensions, writers across layers must overlap in superposition (Elhage et al., 2022), so virtual weights are rarely exactly zero and in practice part of the price is paid as interference.

### A.4 Cost of naive loom attention

Indexing layers by $l=0,\dots,L-1$ as in Section 3.2, every reader of layer $l$ attends over $nl$ earlier head writers.
Over the $3n$ readers of all $L$ layers, the projections $W^{\alpha}$ of Eq. (3) take $\sum_{l=0}^{L-1}3n\cdot nl\cdot D=\frac{3}{2}n^{2}L(L-1)D\approx\frac{3}{2}n^{2}L^{2}D$ parameters, and as many multiply-adds per token.
This grows quadratically in the total number of heads $nL$ and rivals the $12LD^{2}$ of the whole model once $n^{2}L\approx 8D$ (i.e., $nL\approx 8d$ ), e.g., $70\%$ of it for our XL configuration ( $L=28$ , $n=20$ , $D=1920$ ).
Factorizing the weights into keys and queries removes $W^{\alpha}$ , but still leaves $\approx\frac{3}{2}n^{2}L^{2}$ attention weights per token to compute and store for backpropagation, plus $\approx\frac{3}{2}nL^{2}D$ multiply-adds to aggregate the $o_{i}$ ’s.

### A.5 Routing through FFNs

An intermediate FFN between the writers and the reader layer can route any pattern $P\subseteq[w]\times[r]$ by copying $o_{i}$ into a $d$ -dimensional subspace private to $\mathsf{R}_{j}$ whenever $(i,j)\in P$ .
Concretely, suppose the residual stream $x\in\mathbb{R}^{D}$ carries binary indicators $p_{ij}\in\{0,1\}$ of $P$ ( $p_{ij}=1$ iff $(i,j)\in P$ ) and the writer outputs satisfy $\|o_{i}\|_{\infty}\leq\Lambda$ (ignoring pre-LN rescaling).
Since $\operatorname{ReLU}\big(z-\Lambda(1-p)\big)-\operatorname{ReLU}\big(-z-\Lambda(1-p)\big)$ equals $z$ when $p=1$ and $0$ when $p=0$ for every $z\in[-\Lambda,\Lambda]$ , a two-layer ReLU FFN can compute the gated copy $p_{ij}o_{i}\in\mathbb{R}^{d}$ coordinate-wise for each pair $(i,j)$ and sum over $i$ into $\mathsf{R}_{j}$ ’s private subspace.
However, this construction uses $2wrd$ hidden units—hence $4wrdD=\mathcal{O}(wrdD)$ parameters, or $12nlD^{2}$ when routing $w=nl$ writers to $r=3n$ readers ( $n$ times the $12lD^{2}$ parameters of all $l$ preceding blocks)—requires an intervening FFN step, and needs disjoint subspaces for the $w$ writers, $r$ readers and $wr$ control coordinates ( $D\geq(w+r)d+wr$ ).
We give this construction only as an upper bound and do not prove a lower bound on FFN routing.

### A.6 Proof of Proposition 3.1

#### Part (i).

Flatten $M\in\mathbb{R}^{d_{k}\times d}$ row by row into $\operatorname{vec}(M)\in\mathbb{R}^{d_{k}d}$ , so that entry $(a,b)$ of $M$ is entry $(a-1)d+b$ of $\operatorname{vec}(M)$ .
For a static read port $q\in\mathbb{R}^{d_{k}}$ , $(qM)_{b}=\sum_{a}q_{a}M_{ab}=\big(\operatorname{vec}(M)\,(q^{\top}\otimes I_{d})\big)_{b}$ .
For a static write port $k\in\mathbb{R}^{d_{k}}$ , $\operatorname{vec}(k^{\top}o)_{(a-1)d+b}=k_{a}o_{b}=\big(o\,(k\otimes I_{d})\big)_{(a-1)d+b}$ .
Hence every read and write of $M$ is a fixed linear map between $\operatorname{vec}(M)$ and the head, i.e., a Transformer projection with Kronecker structure, $q^{\top}\otimes I_{d}$ for reads and $k\otimes I_{d}$ for writes, and the composition weight between $\mathsf{W}_{i}$ and $\mathsf{R}_{j}$ is the constant $\langle q^{m}_{j},k^{m}_{i}\rangle$ .
The coupling $KQ^{\top}$ is therefore fixed and belongs to the virtual weights, $\bar{C}_{ij}=\langle q^{m}_{j},k^{m}_{i}\rangle I_{d}$ ; only gating heads through attention sinks (together with normalization scales) depends on the input, as in Eq. (2), so $\operatorname{Tfm}_{K,Q}$ is node-gated.
In the full architecture (Section 3.2), $\operatorname{MNorm}$ is RMSNorm over $\operatorname{vec}(M)$ and rescales all matrix-stream writers by one shared factor (Lemma A.1), while the bypass $o^{m}_{i}$ and $(\operatorname{FFN}_{x},\operatorname{FFN}_{M})$ become a fixed linear skip and a standard FFN on $(x,\operatorname{vec}(M))$ .
The per-head $\operatorname{Norm}(o_{i})$ on LoomMHA writes (Eq. 8) applies a fixed diagonal gain (absorbed into the write projection) and rescales writer $\mathsf{W}_{i}$ by $\sqrt{d}/\|o_{i}\|$ , and even if the scalar port gates $\sigma(xW^{W}_{g}),\sigma(xW^{R}_{\bar{g}})$ (Eqs. 5–6) are retained on static port directions, they multiply whole rows or columns of $C$ , leaving $C=g\,g^{\prime\top}$ of rank at most one (cf. Appendix G).
With ports generated per token, the computation at each token and layer coincides with that of $\operatorname{Tfm}_{K(x),Q(x)}$ .
The coincidence holds within a token; across tokens, attention mixes values computed by different members, so the model as a whole is not any single $\operatorname{Tfm}_{K,Q}$ .

#### Part (ii).

Let $S_{W}$ be the writers in a wiring pattern $P\subseteq[w]\times[r]$ , with $|S_{W}|\leq d_{k}$ .
Turn on only $S_{W}$ , pick an injection $\iota:S_{W}\to[d_{k}]$ , give each $i\in S_{W}$ the write key $k^{m}_{i}=e_{\iota(i)}$ ( $k^{m}_{i}=0$ for $i\notin S_{W}$ ), and give reader $\mathsf{R}_{j}$ the read query $q^{m}_{j}=\sum_{i\in S_{W}:\,(i,j)\in P}e_{\iota(i)}$ .
Then $\langle q^{m}_{j},k^{m}_{i}\rangle=1$ if $(i,j)\in P$ and $0$ otherwise for all $i\in[w]$ , so $c_{ij}\neq 0$ if and only if $(i,j)\in P$ ; since the virtual weights of the matrix stream are identities, the connected pairs are $P$ , with one head per reader and $d_{k}$ independent of $w$ .
The RMSNorm of the actual ports rescales each key and each query by a constant, and the sigmoid gates set their magnitudes (turning off writers outside $S_{W}$ ), so the pattern is realized up to per-reader scales.
In the full model, computing $\iota$ and $P$ from the vector stream is the role of the port generators on the control path.

### A.7 Wiring matrices of residual-stream mechanisms

This section derives the entries of Table 1.
As in Section 2, we consider the wiring within a token from the earlier head writers ( $o_{i}\in\mathbb{R}^{d}$ ) to the query, key and value readers ( $u_{j}$ ) of one later layer.
Other reads of the history—the FFN inputs of HC/mHC and AttnRes, the residual input of MUDDFormer and the bypass $o^{m}$ of LoomFormer—feed no head input and are not readers.
We mark a port by $(x)$ if it is computed from the vector (residual) stream and by $(M)$ if it is computed from the memory itself.
Per-token scales such as pre-LN are absorbed into the wiring matrix as in Lemma A.1, and the node-gating factors $g$ , $g^{\prime}$ of Definition 2.2 are common to all mechanisms, so we omit them from the tables.
For LoomFormer we consider the matrix stream only and ignore MShift, which mixes neighbouring tokens (the $d/4$ -dimensional RoPE slice from $x$ belongs to the node-gated vector stream of Proposition 2.3).
Ports are row vectors, as in Section 3.1.

#### A common form.

Each mechanism keeps at a token a memory $M$ of $d_{k}$ slots of width $d_{v}$ .
Writer $\mathsf{W}_{i}$ adds $k^{m}_{i}\otimes(o_{i}W^{O}_{i})$ to $M$ , reader $\mathsf{R}_{j}$ receives $u_{j}=(q^{m}_{j}M)\,W^{s}_{j}$ , and between layers $M$ may be updated linearly, $M\leftarrow\Phi_{l}M$ ; here $k^{m}_{i}$ and $q^{m}_{j}$ are the write and read ports, and $W^{O}_{i}=W^{s}_{j}=I_{d}$ in the matrix stream of LoomFormer.
Composing these steps gives

 $u_{j}=\sum_{i}\big(q^{m}_{j}\,\Phi_{l_{i}l_{j}}\,(k^{m}_{i})^{\top}\big)\,o_{i}W^{O}_{i}W^{s}_{j},$ 

(12)

where $\Phi_{l_{i}l_{j}}=\Phi_{l_{j}-1}\cdots\Phi_{l_{i}}$ is the product of the updates between the layer $l_{i}$ of $\mathsf{W}_{i}$ and the layer $l_{j}$ of $\mathsf{R}_{j}$ : the wiring matrix has entries $c_{ij}=q^{m}_{j}\Phi_{l_{i}l_{j}}(k^{m}_{i})^{\top}$ and the virtual weights are $\bar{C}_{ij}=W^{O}_{i}W^{s}_{j}$ .
This representation is not unique: a change of slot basis by an invertible $G$ maps $k^{m}_{i}\mapsto k^{m}_{i}G^{\top}$ , $q^{m}_{j}\mapsto q^{m}_{j}G^{-1}$ and $\Phi_{l}\mapsto G\Phi_{l}G^{-1}$ without changing $c_{ij}$ .
For mechanisms that keep the outputs of all earlier layers, we use the basis in which every stored output has its own slot, the write port is $e_{l_{i}}$ and $\Phi_{l}=I$ ; residual copies then become part of the read port.
Transitions $\Phi_{l_{i}l_{j}}$ can also be absorbed into the ports without adding slots: for a fixed reader layer $l_{j}$ , setting $\tilde{k}^{m}_{i}=k^{m}_{i}\Phi^{\top}_{l_{i}l_{j}}$ writes $c_{ij}=\langle q^{m}_{j},\tilde{k}^{m}_{i}\rangle$ even when $\Phi_{l}$ is singular (and when every $\Phi_{l}$ is invertible, setting $\Phi^{\times}_{l}=\Phi_{l-1}\cdots\Phi_{1}$ makes this split independent of $l_{j}$ via $\tilde{q}^{m}_{j}=q^{m}_{j}\Phi^{\times}_{l_{j}}$ and $\tilde{k}^{m}_{i}=k^{m}_{i}\Phi^{\times-\top}_{l_{i}}$ ).
Table 5 lists the parameterization of each mechanism, justified below; Table 1 omits factors equal to $1$ or $I$ .

FIGCAPTION: Table 5: Ports of residual-stream mechanisms in the common form of Eq. (12).

Write port $k^{m}_{i}$ 
Read port $q^{m}_{j}$ 
Update $\Phi_{l}$ 

Transformer
 $1$ 
 $1$ 
 $1$ 

DenseFormer
 $e_{l_{i}}$ 
static
 $I$ 

HC / mHC
 $\beta_{l_{i}}(M)$ 
 $\alpha_{l_{j}}(M)$ 
 $A_{l}(M)$ 

AttnRes (full)
 $\operatorname{Norm}(v_{l_{i}})$ 
static, softmax
 $I$ 

MUDDFormer
 $e_{l_{i}}$ 
per Q/K/V, from $x$ 
 $I$ 

LoomFormer
per head, from $x$ 
per head and Q/K/V, from $x$ 
 $I$ 

#### Transformer.

One slot of width $D$ with $k^{m}_{i}=q^{m}_{j}=\Phi=1$ gives $c_{ij}=1$ and the virtual weights $\bar{C}_{ij}=W^{O}_{i}W^{s}_{j}$ ; with the node gates, this is Eq. (2).

#### DenseFormer.

The input of each block is a static weighted sum of the embedding and the outputs of all earlier blocks, and the block adds its own attention and FFN outputs to this input through its internal residual (Pagliardini et al., 2024).
Unfolding these residual copies, the input of block $l^{\prime}$ is a static combination of the outputs of all earlier blocks, so $c_{ij}\bar{C}_{ij}=\langle q^{m}_{l_{j}},k^{m}_{l_{i}}\rangle\,W^{O}_{i}W^{s}_{j}$ with a static read port, so only node gating makes the wiring matrix input-dependent.
DenseFormer keeps one slot per block, about $l$ slots at layer $l$ (size $lD$ ).

#### HC and mHC.

HC keeps $d_{k}$ parallel streams of width $D$ (we use $d_{k}=4$ , the main configuration of Zhu et al. (2025)).
Each sub-layer reads $\alpha M$ and updates the streams as $M\leftarrow AM+\beta^{\top}h$ , with $\alpha$ , $\beta$ and $A$ all computed from $M$ in the dynamic version (Zhu et al., 2025), and $A$ projected onto the Birkhoff polytope of doubly stochastic matrices in mHC (Xie et al., 2025).
Heads read and write through the attention sub-layer, with ports $\alpha_{l}$ and $\beta_{l}$ , while the FFN and subsequent attention stream-mixing matrices between attention sub-layers $l_{i}$ and $l_{j}$ compose into $\Phi_{l_{i}l_{j}}=A_{l_{i}l_{j}}$ .
In the common form, $c_{ij}=\alpha_{l_{j}}A_{l_{i}l_{j}}\beta_{l_{i}}^{\top}$ —the unrolled weight of Eq. (27) of Zhu et al. (2025), which Kimi Team (2026a) interpret as depth-wise linear attention with queries $\alpha$ and keys $\beta$ .
Since all factors are computed from the streams $M$ , absorbing $A_{l_{i}l_{j}}$ into the write port as above (without requiring $A_{l}$ to be invertible) gives the entry $\langle q^{m}_{l_{j}}(M),k^{m}_{l_{i}}(M)\rangle$ of Table 1, with $d_{k}$ -dimensional ports and state size $d_{k}D=4D$ .

#### AttnRes.

Full AttnRes keeps the embedding and the outputs $v_{m}$ of all earlier sub-layers, without a residual connection, and gives sub-layer $m^{\prime}$ the input $\sum_{m}\operatorname{softmax}_{m}\langle w_{m^{\prime}},\operatorname{Norm}(v_{m})\rangle\,v_{m}$ with a learned query $w_{m^{\prime}}$ (Kimi Team, 2026a).
In the slot basis of Eq. (12), writers of layer $l_{i}$ write to their attention sub-layer’s slot ( $e_{l_{i}}$ , while FFN and embedding slots hold no head writer) and heads of layer $l_{j}$ read linearly with the softmax weight vector over all stored slots; Tables 1 and 5 instead display the underlying softmax query $q^{m}_{l_{j}}=w_{l_{j}}$ (static) and key $k^{m}_{l_{i}}(M)=\operatorname{Norm}(v_{l_{i}})$ (from $M$ ), normalized by $\operatorname{softmax}$ across all stored sub-layer slots.
Full AttnRes keeps one slot per sub-layer, about $2l$ slots at layer $l$ (size $2lD$ ); its block variant sums the outputs within each block into one slot.

#### MUDDFormer.

MUDDFormer keeps the embedding and the outputs of all earlier blocks, and gives the next block four inputs—for its queries, keys, values and residual—each a weighted sum of these outputs with weights generated from the latest hidden state (Xiao et al., 2025).
The residual input carries dynamic combinations of earlier outputs into the next block output; unfolding these recursions expresses the effective read port $q^{m,s}_{l_{j}}(x)$ of input $s\in\{Q,K,V\}$ as a function of the hidden states along the vector stream $x$ , giving $c_{ij}\bar{C}_{ij}=\langle q^{m,s}_{l_{j}}(x),k^{m}_{l_{i}}\rangle\,W^{O}_{i}W^{s}_{j}$ .
Like DenseFormer, MUDDFormer keeps about $l$ slots at layer $l$ (size $lD$ ).

#### LoomFormer.

The matrix stream keeps $d_{k}$ slots of width $d$ that accumulate writes ( $\Phi_{l}=I$ ) and passes content without fixed projections: absorbing the per-token RMS scales of $\operatorname{Norm}(o_{i})$ and $\operatorname{MNorm}$ into $k^{m}_{i}$ and $q^{m,s}_{j}$ (and the learned diagonal gain of $\operatorname{Norm}(o_{i})$ into the writer output $o_{i}$ ) leaves $\bar{C}_{ij}=I_{d}$ .
Indexing a reader by its head $j\in[n]$ and input $s\in\{Q,K,V\}$ as in Table 1 gives $c_{ij}=\langle q^{m,s}_{j}(x),k^{m}_{i}(x)\rangle$ , with both ports generated per head and per token from the vector stream (Proposition 3.1(ii)).
With $d_{k}=2n$ , $M$ has size $d_{k}d=2D$ , plus $D$ for the vector stream.

#### Wiring ranks.

Node gating multiplies $C$ entrywise by $g\,g^{\prime\top}$ —equivalently $C\mapsto\operatorname{diag}(g)\,C\,\operatorname{diag}(g^{\prime})$ —which cannot increase its rank.
All readers of a layer share one read port in DenseFormer, HC/mHC and AttnRes, so $C$ has a single distinct column and rank at most $1$ ; MUDDFormer has one read port per Q/K/V input, hence at most three distinct columns and rank at most $3$ (reached once at least three earlier layers are present with linearly independent Q/K/V read ports).
In LoomFormer every reader has its own query, so the rank is that of $KQ^{\top}$ restricted to the $3n$ readers of the layer, at most $\min(w,d_{k},3n)$ —hence $\min(d_{k},3n)$ once $w\geq\min(d_{k},3n)$ , reached whenever the writer keys and reader queries span a common $\min(d_{k},3n)$ -dimensional subspace.

#### Linear attention.

The same form describes linear attention along the sequence instead of depth: the writers and readers are one head at different tokens, $M$ is the head’s private state, the ports are its keys and queries, and the update $\Phi$ distinguishes the variants— $I$ for linear attention (Katharopoulos et al., 2020), a scalar decay in Mamba2 (Dao & Gu, 2024), a gated diagonal in gated linear attention (Yang et al., 2024a), $I-\beta k^{\top}k$ in DeltaNet (Yang et al., 2024b), and $\gamma(I-\beta k^{\top}k)$ in Gated DeltaNet (Yang et al., 2025).
It differs from LoomFormer in the axis and the sharing of the state (Section 4).

## Appendix B Extended Related Work

#### Cross-layer connections and residual-stream architectures.

Beyond standard residual connections (He et al., 2016), early work strengthened depth-wise signal propagation through input-dependent gating in Highway Networks (Srivastava et al., 2015) and all-to-all layer aggregation with static or learnable weights in DenseNet (Huang et al., 2017), DLCL (Wang et al., 2019) and DenseFormer (Pagliardini et al., 2024).
Subsequent architectures introduced cross-layer attention over earlier layer representations (ElNokrashy et al., 2022; Fang et al., 2023; Wang et al., 2024; Kimi Team, 2026a), multi-stream residual states with dynamic or manifold-constrained mixing in HC and mHC (Zhu et al., 2025; Xie et al., 2025), and decoupled Q/K/V/residual layer aggregation in MUDDFormer (Xiao et al., 2025).
As analyzed in Section 4 and Section A.7, all of these mechanisms operate at the granularity of whole layers (or layer $\times$ Q/K/V), summing the outputs of all heads within a layer before routing and restricting the within-token wiring rank to $1$ (or $3$ ).

The Residual Matrix Transformer (RMT) (Mak & Flanigan, 2025) replaces the vector residual stream entirely with a pure matrix residual stream, which heads and FFNs read and write through static key vectors so the stream can be widened cheaply.
LoomFormer instead keeps a dual-stream architecture that separates control on the vector stream from content on the matrix stream: the compact vector stream generates per-token ports cheaply, whereas RMT’s static ports fall under Proposition 3.1(i) and remain node-gated however wide the matrix is made ( $4D$ in RMT’s main configuration vs. $2D+D$ in LoomFormer).

#### Cross-layer token attention and KV sharing.

OmniNet (Tay et al., 2021) lets every token attend directly to all tokens across all layers, incurring attention cost quadratic in $LT$ ; loom attention instead factors cross-layer routing within each token through a fixed-size matrix memory $M$ in $\mathcal{O}(LTDd_{k})$ time and composes it with standard within-layer horizontal attention.
Cross-Layer Attention (CLA) (Brandon et al., 2024) shares key-value caches across adjacent layers to reduce inference memory at a modest quality trade-off, whereas loom attention dynamically composes earlier head outputs into distinct per-head queries, keys and values at every layer (Section 5.3).

#### Mechanistic interpretability and intra-layer advances.

Circuit-level interpretability studies (Elhage et al., 2020; Olsson et al., 2022; Conmy et al., 2023; Hanna et al., 2024; Merullo et al., 2024; Ni et al., 2025) show that Transformers implement tasks by composing specific upstream and downstream heads across layers; loom attention turns this implicit, static composition into an explicit, per-token operation over $M$ .
Orthogonal intra-layer advances—including improved horizontal attention (Liu et al., 2024; Xiao et al., 2024a; Ye et al., 2024; Leviathan et al., 2024), sequence-axis linear attention and state-space models (Gu & Dao, 2024; Dao & Gu, 2024; Peng et al., 2024; Yang et al., 2024b; Yang et al., 2025), and sparse Mixture-of-Experts FFNs (Fedus et al., 2022; Dai et al., 2024)—modify horizontal mixing or FFN capacity within a layer and can be combined directly with loom attention along depth.

## Appendix C PyTorch-Style Pseudocode for LoomFormer

   1  

# b = batch size, t = seq len, n = num_heads, d = head_dim
2  

# k = num_slots d_k, r = RoPE dim d // 4. M is of shape btkd
3  4  

def MRead(M, x, xr):
5  

 qm = rearrange(xr @ W_r2, 'bt(nk)->btnk')
6  

 qm = rmsnorm(qm) * sigmoid(rearrange(x @ W_rg, 'btn->btn1'))
7  

 qm += q_bar * sigmoid(rearrange(x @ W_rg_bar, 'btn->btn1')) # nk,btn1->btnk
8  

 u = einsum('btnk,btkd->btnd', qm, M)
9  

 return u
10  11  

def MWrite(o, x, xw):
12  

 km = rearrange(xw @ W_w2, 'bt(nk)->btnk')
13  

 km = rmsnorm(km) * sigmoid(rearrange(x @ W_wg, 'btn->btn1'))
14  

 dM = einsum('btnk,btnd->btkd', km, o)
15  

 return dM
16  17  

def MShift(M, x):
18  

 M_prev = shift_right(M, 1, dim=1) # shift t dim of btkd
19  

 z = sigmoid(rearrange(x @ W_sg, 'btk->btk1'))
20  

 return z * M_prev + (1 - z) * M
21  22  

def loom_mha(x, M, write_M=True):
23  

 M = RMSNorm(MShift(M, x), dim=(-2, -1))
24  

 xr, xw = gelu(x @ W_r1), gelu(x @ W_w1)
25  

 q, k, v, o = [MRead[s](M, x, xr) for s in ['q', 'k', 'v', 'o']]
26  

 qr = rope(rearrange(x @ W_q, 'bt(nr)->btnr'))
27  

 kr = rope(rearrange(x @ W_k, 'bt(nr)->btnr'))
28  

 q, k = concat([qr, q], dim=-1), concat([kr, k], dim=-1)
29  30  

 allow_mask = tril(ones(T, T), diagonal=-1) # strictly causal mask
31  

 o = o + MHA(q, k, v, allow_mask=allow_mask)
32  

 dx = rearrange(o, 'btnd->bt(nd)') @ W_o
33  

 dM = MWrite(RMSNorm(o, dim=-1), x, xw) if write_M else 0.
34  

 return dx, dM
35  36  

def ffn(x, write_M=True):
37  

 xw = gelu(x @ W_w1)
38  

 dx = _ffn_x(x)
39  

 dM = MWrite(rearrange(_ffn_m(x), 'bt(nd)->btnd'), x, xw) if write_M else 0.
40  

 return dx, dM
41  42  

def loomformer(input_ids):
43  

 x = embedding(input_ids)
44  

 o = rearrange(x @ W_e, 'bt(nd)->btnd')
45  

 M = MWrite(o, x, gelu(x @ W_w1))
46  47  

 for l in range(L):
48  

 write_M = l < L - 1
49  

 # pairs are added element-wise: (x, M) + (dx, dM) = (x + dx, M + dM)
50  

 (x, M) = (x, M) + loom_mha[l](ln_1[l](x), M, write_M=write_M)
51  

 (x, M) = (x, M) + ffn[l](ln_2[l](x), write_M=write_M)
52  

 return unembedding(ln_f(x))
   

## Appendix D Experimental Configurations, Baselines and Benchmarks

### D.1 Model Configurations and Hyperparameters

Table 6 lists the architectural dimensions, peak learning rates and token budgets across all four LoomFormer scales (Sections 5.1 and 5.2).
In blocks $0,\dots,L-2$ , $\operatorname{FFN}_{x}$ and the $\operatorname{FFN}_{M}$ write branch ( $\operatorname{FFN}_{M}$ plus its MWrite module) follow a $2{:}1$ parameter ratio (rounded to multiples of $128$ at 2.8B, giving $2.2{:}1$ ), while block $L-1$ (last) omits unused writes to $M_{L}$ and widens $\operatorname{FFN}_{x}$ to match the baseline Transformer’s total parameter count.
Table 7 lists the training hyperparameters for the scaling-law experiments (Section 5.1; weight decay is excluded on normalization scales and biases, including gate biases), while the 2.8B models (Section 5.2) follow Pythia-2.8B (Biderman et al., 2023, Appendix E).
To improve training stability at the XL scale, LoomFormer-XL replaces SwiGLU in $\operatorname{FFN}_{x}$ with Kimi K3’s bounded SiTU-GLU activation (Kimi Team, 2026b) with $\beta_{1}=4$ , $\beta_{2}=7$ .

#### Port and gate initialization.

Each gate projection in Eqs. (5)–(7) includes a learnable bias (omitted in Section 3.2 for brevity) that sets its initial opening.
Specifically, the gate projection weights are initialized to zero, with biases setting initial gate values to $0.05$ on the dynamic read branch, $0.9$ on the static read branch, $0.1$ on the write branch, and $0.5$ on MShift ( $z_{[t]}$ ).
For the static queries $\bar{q}\in\mathbb{R}^{d_{k}}$ , value reads $v$ are initialized from $\mathcal{N}(0,1/d_{k})$ , whereas $\tilde{q},\tilde{k}$ and $o^{m}$ start at $\bar{q}=0$ so memory-read Q/K coordinates and the output bypass grow in gradually.

FIGCAPTION: Table 6: Model configurations across scales. $d_{k}=2n$ ; shared port-feature width $D_{r}=nd_{k}/2=n^{2}$ , capped at $D/4=640$ at 2.8B.

Model
Params
 $L$ 
 $D$ 
 $D/L$ 
 $n$ 
 $d$ 
 $D_{r}$ 
Tfm FFN
 $\operatorname{FFN}_{x}$ (last)
 $\operatorname{FFN}_{M}$ 
LR
Tokens

Medium
432M
18
1200
66.7
16
75
256
3200
2510 (3904)
1128
3.0e-4
7B

Large
778M
22
1536
69.8
16
96
256
4096
3317 (5083)
1539
2.5e-4
14B

XL
1.4B
28
1920
68.6
20
96
400
5120
4048 (6277)
1827
2.0e-4
26B

2.8B
2.8B
32
2560
80
32
80
640
10240
7808 (9984)
2944
1.6e-4
300B

FIGCAPTION: Table 7: Training hyperparameters for the scaling-law experiments (Section 5.1).

Hyperparameter
Value    
Hyperparameter
Value

Optimizer
AdamW    
Sequence length
 $4096$ 

 $(\beta_{1},\beta_{2})$ 
 $(0.9,0.95)$     
Batch size (seqs)
 $128$ 

 $\epsilon$ 
 $10^{-8}$     
Vocab size
 $50{,}432$ 

Weight decay
 $0.1$     
Tied embeddings
False

Gradient clipping
 $1.0$     
Linear bias
False

LR schedule
Cosine    
Normalization
RMSNorm ( $\epsilon=10^{-6}$ )

Min LR ratio
 $0.1$     
FFN activation
SwiGLU

Warmup fraction
 $0.01$     
Positional emb.
RoPE ( $\theta=10{,}000$ )

Dropout
 $0.0$     
Precision
BF16

### D.2 Scaling-Law Baselines

All baselines in Section 5.1 are trained in JAX/MaxText on the identical data order, tokenizer and schedule as LoomFormer:

• 

Transformer++: Llama-style pre-norm Transformer (Touvron et al., 2023) with RoPE (Su et al., 2024), SwiGLU FFN (Shazeer, 2020), RMSNorm and no linear biases.

• 

DHC: Dynamic Hyper-Connections with expansion rate $d_{k}=4$ and $\tanh$ -bounded dynamic routing following Zhu et al. (2025).

• 

mHC: Manifold-constrained Hyper-Connections ( $d_{k}=4$ streams) projecting the stream-mixing matrix onto the Birkhoff polytope via Sinkhorn normalization (Xie et al., 2025).

• 

AttnRes (full): Full Attention Residuals aggregating all preceding sub-layer outputs per sub-layer via learned depth-wise softmax queries (Kimi Team, 2026a).

• 

MUDDFormer: Multiway Dynamic Dense connections generating per-token depth-aggregation weights separately for the query, key, value and residual inputs of each block (Xiao et al., 2025).

### D.3 Downstream Evaluation Benchmarks

In addition to the benchmarks used by Pythia (Biderman et al., 2023)—LAMBADA (Paperno et al., 2016), PIQA (Bisk et al., 2020), WinoGrande (Sakaguchi et al., 2021), ARC-Easy/Challenge (Clark et al., 2018) and SciQ (Welbl et al., 2017)—we evaluate BoolQ (Clark et al., 2019) and HellaSwag (Zellers et al., 2019) for commonsense reasoning, and RACE-Middle/High (Lai et al., 2017) for reading comprehension (Table 2), all under LM Evaluation Harness (Gao et al., 2023) in zero-shot and five-shot settings.

## Appendix E Details of Complexity Analysis

We derive the forward-FLOP overhead in Table 3 under a matched parameter budget, following the counting convention of DCFormer (Xiao et al., 2024a) and MUDDFormer (Xiao et al., 2025).
Let $L$ be the number of layers, $T$ the sequence length, $D$ the model width, $n$ the number of heads, $d=D/n$ the head dimension, and $M\in\mathbb{R}^{d_{k}\times d}$ the matrix memory.
We report FLOPs per sequence at two FLOPs per multiply-add, omitting the shared token embedding and unembedding as well as normalization, nonlinearities, biases and other elementwise operations (such as gate activations and the elementwise mixing in MShift).
Causal attention has approximately $T^{2}/2$ query–key pairs; using a strictly rather than inclusively causal mask changes only the lower-order diagonal terms.

#### Parameter budget and tokenwise projections.

The reference Transformer has $4D^{2}$ attention and $8D^{2}$ FFN weights per layer, for a total of $P_{0}=12LD^{2}$ (applying to both a GELU FFN of hidden width $4D$ and a SwiGLU FFN of hidden width $8D/3$ ).
LoomFormer replaces the full Q/K/V projections with smaller RoPE projections and matrix reads, redistributing the saved parameter budget among the port generators, $\operatorname{FFN}_{x}$ and $\operatorname{FFN}_{M}$ , as well as the initial value projection $W^{E}$ and embedding MWrite (Eq. 10), so that $P_{\mathrm{Loom}}=P_{0}$ .

Every tokenwise dense matrix $W$ costs $2T|W|$ FLOPs, where $|W|$ is its number of entries.
Consequently, to leading order, the combined projection and FFN cost is $2TP_{0}$ in both models.
This includes both stages of the low-rank port generators, all projected gates (including $W^{S}_{g}$ in MShift), $W^{E}$ , $\operatorname{FFN}_{x}$ and $\operatorname{FFN}_{M}$ ; a shared projection is counted once.
Adding the port-generation or $\operatorname{FFN}_{M}$ FLOPs without subtracting the $\operatorname{FFN}_{x}$ FLOPs that paid for their weights would double-count them.
Static ports, learned normalization scales and biases give only lower-order corrections to this dense-weight relation, and for the same reason the bottleneck width $D_{r}$ affects parameter allocation without entering the leading net FLOP overhead.

#### Reference Transformer FLOPs.

In addition to the tokenwise projections and FFNs, each layer computes $QK^{\top}$ and attention-weighted values.
Each costs $2nd\cdot T^{2}/2=DT^{2}$ FLOPs under the causal approximation.
The reference forward cost is therefore

 $F_{0}=2TP_{0}+2LDT^{2}=2LDT(12D+T).$ 

(13)

#### Reading and writing the matrix memory.

A single MRead contracts a $d_{k}$ -dimensional port with $M\in\mathbb{R}^{d_{k}\times d}$ , at a cost of $2d_{k}d$ FLOPs per token.
Its dynamic and static ports are summed before this contraction (Eq. 5), so they do not require two separate reads.
With four reads per head and $n$ heads, the $L$ layers cost

 $F_{\mathrm{read}}=4\cdot 2Tndd_{k}\cdot L=8LTDd_{k}.$ 

(14)

For each writing sub-layer, MWrite sums $n$ outer products of shape $d_{k}\times d$ , costing $2Tndd_{k}=2TDd_{k}$ FLOPs.
There are two such sub-layers per block, attention and $\operatorname{FFN}_{M}$ , except in the last block, which does not write to $M$ .
The embedding adds one initial write of the same size (Eq. 10).
Hence

 $\begin{split}F_{\mathrm{write}}&=4(L-1)TDd_{k}+2TDd_{k},\\
F_{\mathrm{M}}&=F_{\mathrm{read}}+F_{\mathrm{write}}=(12L-2)TDd_{k}\simeq 12LTDd_{k}.\end{split}$ 

(15)

The last approximation neglects the small endpoint factor $1-1/(6L)$ multiplying the memory-overhead term.
Since $M$ has a fixed size, the read/write cost per layer does not increase with depth.

#### Wider query and key.

In the reference Transformer, Q, K and V each have $d$ coordinates per head.
LoomFormer concatenates $d/4$ RoPE coordinates to the $d$ coordinates of Q/K read from $M$ (Eq. 7), while V retains width $d$ .
Thus only the $QK^{\top}$ contraction is wider; the attention-weighted value contraction is unchanged.
The extra cost is

 $\Delta F_{\mathrm{QK}}=2Ln\frac{d}{4}\cdot\frac{T^{2}}{2}=\frac{LDT^{2}}{4}.$ 

(16)

#### Total overhead and scaling.

Combining Eqs. (13)–(16) gives

 $R_{\Delta\mathrm{FLOPs}}\simeq\frac{12LTDd_{k}+LDT^{2}/4}{2LDT(12D+T)}=\frac{d_{k}}{2D(1+\rho/12)}+\frac{1}{8(1+12/\rho)},$ 

(17)

where $\rho=T/D$ .
With $D=nd$ and $d_{k}=2n$ , we have $d_{k}/(2D)=1/d$ , giving the expression in Section 5.4:

 $\boxed{R_{\Delta\mathrm{FLOPs}}\simeq\underbrace{\frac{1}{d(1+\rho/12)}}_{\text{MRead/MWrite}}+\underbrace{\frac{1}{8(1+12/\rho)}}_{\text{wider Q/K}}.}$ 

(18)

At fixed $\rho$ , the matrix read/write term scales as $1/d$ while the wider-Q/K term is independent of $d$ ; as $\rho$ grows, the first term falls from $1/d$ toward $0$ and the second rises toward $1/8$ , with both independent of $L$ at leading order.
In Table 3 ( $d=75,96,96,80$ and $\rho=T/D=3.41,2.67,2.13,0.80$ for Medium, Large, XL and 2.8B, using $T=4096$ at Medium/Large/XL and $T=2048$ at 2.8B; Table 6), Large, XL and 2.8B have integer $d/4$ and follow Eq. (18) directly ( $3.13\%$ , $2.77\%$ and $1.95\%$ ), whereas Medium rounds its RoPE width $d/4=75/4$ to $\lfloor d/4\rfloor=18$ in Eq. (16), giving $3.70\%$ instead of Eq. (18)’s unrounded $3.81\%$ (retaining $12L-2$ in Eq. 15 changes every row by $Appendix F.

## Appendix F Training and Inference Throughput Details

#### Training throughput.

Table 8 reports paired training benchmarks in JAX/MaxText on TPU v5p pods at each model’s training configuration (Sections 5.1 and 5.2), with both Transformer and LoomFormer using Splash attention ( $1024$ -token blocks and fused backward).
LoomFormer executes custom Pallas kernels (forward and backward) that fuse MShift, MNorm, port normalization and gating, and the four MReads of all $n$ heads into one per-token $(4n\times d_{k})\times(d_{k}\times d)$ MXU matrix multiplication in VMEM, and likewise fuse MWrite’s $n$ -head outer-product accumulation, avoiding intermediate HBM traffic on $M$ .
We compare median XPlane device step time under full activation rematerialization on both sides (Table 3).
At 2.8B, LoomFormer uses $512$ -token cross-entropy chunking to fit v5p-128 HBM whereas Pythia-2.8B does not, making the 2.8B ratio slightly conservative.

FIGCAPTION: Table 8: TPU v5p training step time and relative throughput of LoomFormer (Loom) vs. Transformer (Tfm).

Model
Pod
 $T$ 
Batch (seqs)
Tfm (ms)
Loom (ms)
Relative speed

Medium
v5p-16
4096
128
 $1224$ 
 $1352$ 
 $90.5\%$ 

Large
v5p-32
4096
128
 $962$ 
 $1078$ 
 $89.2\%$ 

XL
v5p-32
4096
128
 $1590$ 
 $1723$ 
 $92.3\%$ 

2.8B
v5p-128
2048
1024
 $2988$ 
 $3233$ 
 $92.4\%$ 

#### Inference throughput on NVIDIA A800.

Table 9 reports BF16 inference throughput on a single NVIDIA A800-80GB PCIe GPU (torch 2.14 / triton 3.8 / cuDNN 9.24) using trained checkpoints at all four scales, taking the median over five measurements.
LoomFormer executes one fused Triton kernel per layer covering the preceding write, the MShift read view, the four MReads, RoPE and cache updates, together with grouped decode GEMVs for the small port projections; prefill uses each model’s fastest SDPA backend (FlashAttention for Transformer; cuDNN SDPA for LoomFormer, whose Q/K head width $1.25d$ differs from V width $d$ ).
Prefill measures one uncached forward pass of $B\times T$ tokens, and decode measures single-token generation steps after a $(T-64)$ -token prompt under torch.compile(mode="max-autotune") with CUDA graphs.

FIGCAPTION: Table 9: BF16 inference throughput on one NVIDIA A800-80GB GPU, reported as relative throughput and absolute tokens/s (Loom / Tfm); $T=4096$ for Medium, Large and XL, and $T=2048$ for 2.8B.

Model
Prefill bs $1$ 
Prefill bs $8$ 
Decode bs $1$ 
Decode bs $16$ 
Decode bs $64$ 

Medium
 $87.4\%$ ( $148.9\textrm{k}/170.3\textrm{k}$ )
 $90.4\%$ ( $166.0\textrm{k}/183.5\textrm{k}$ )
 $91.7\%$ ( $565/616$ )
 $89.2\%$ ( $2109/2365$ )
 $92.3\%$ ( $2499/2709$ )

Large
 $85.3\%$ ( $95.3\textrm{k}/111.7\textrm{k}$ )
 $86.2\%$ ( $104.0\textrm{k}/120.6\textrm{k}$ )
 $85.4\%$ ( $438/513$ )
 $87.8\%$ ( $1916/2183$ )
 $84.9\%$ ( $2311/2721$ )

XL
 $83.8\%$ ( $50.8\textrm{k}/60.6\textrm{k}$ )
 $85.3\%$ ( $55.3\textrm{k}/64.9\textrm{k}$ )
 $86.9\%$ ( $272/313$ )
 $84.6\%$ ( $1245/1472$ )
 $86.6\%$ ( $1452/1677$ )

2.8B
 $87.5\%$ ( $30.4\textrm{k}/34.7\textrm{k}$ )
 $90.5\%$ ( $33.1\textrm{k}/36.6\textrm{k}$ )
 $86.4\%$ ( $171/198$ )
 $87.6\%$ ( $1248/1425$ )
 $88.9\%$ ( $1836/2066$ )

## Appendix G Extended Ablation Studies

Table 10 reports the complete 18-variant ablation suite at the Medium scale ( $L=18$ , $D=1200$ , $n=16$ , $d=75$ , $d_{k}=32$ , $D_{r}=256$ , RoPE dim $\lfloor d/4\rfloor=18$ , $T=4096$ , 7B tokens), extending Table 4.
In full LoomFormer, $\operatorname{FFN}_{M}$ has SwiGLU hidden width $1128$ , and $\operatorname{FFN}_{x}$ has width $2510$ in blocks $0$ – $16$ and $3904$ in block $17$ (which omits unused writes to $M_{L}$ ).
Whenever an ablation removes parameters, we widen $\operatorname{FFN}_{x}$ uniformly across blocks $0$ – $16$ and absorb any embedding or integer-rounding residual in block $17$ , matching the $432.11\textrm{M}$ parameter budget within $\pm 1.4\textrm{K}$ ( $

FIGCAPTION: Table 10: Complete Medium-scale (432M) ablation suite of LoomFormer under a matched parameter budget ( $432.11\textrm{M}\pm 1.4\textrm{K}$ ).
†Included in Table 4.

Variant
Specification (vs. full LoomFormer)
Loss $\downarrow$ 
 $\Delta$ Loss
Gain

LoomFormer (full)†
—
2.266
—
100.0%

Transformer++ (baseline)†
Standard MHA + SwiGLU FFN, no matrix stream $M$ 
2.443
 $+$ 0.177
0.0%

Dynamic rewiring & write-port gating (Proposition 3.1)

Static LoomMHA $k^{m}$ , gated
Static $k^{m}$ in LoomMHA; write gate $\sigma(xW^{W}_{g})$ kept
2.266
 $0.000$ 
100.0%

Static LoomMHA $k^{m}$ 
Static $k^{m}$ in LoomMHA; write gate dropped ( $\equiv 1$ )
2.282
 $+$ 0.016
91.1%

Static $\operatorname{FFN}_{M}$ $k^{m}$ , gated
Static $k^{m}$ in $\operatorname{FFN}_{M}$ ; write gate $\sigma(xW^{W}_{g})$ kept
2.268
 $+$ 0.001
99.3%

Static $\operatorname{FFN}_{M}$ $k^{m}$ 
Static $k^{m}$ in $\operatorname{FFN}_{M}$ ; write gate dropped ( $\equiv 1$ )
2.272
 $+$ 0.006
96.6%

Static write keys ( $k^{m}$ ), gated
Static $k^{m}$ in emb., LoomMHA, $\operatorname{FFN}_{M}$ ; write gates kept
2.274
 $+$ 0.008
95.6%

Static write keys ( $k^{m}$ )†
Static $k^{m}$ in emb., LoomMHA, $\operatorname{FFN}_{M}$ ; write gates dropped
2.283
 $+$ 0.017
90.3%

Static read queries ( $q^{m}$ )†
Only static $\bar{q}$ in $\operatorname{MRead}^{Q,K,V,O}$ ; all read gates dropped
2.310
 $+$ 0.043
75.4%

Static read & write ( $q^{m},k^{m}$ ), gated
Static $q^{m}$ (no read gates), all $k^{m}$ ; write gates kept
2.327
 $+$ 0.060
65.8%

Static read & write ( $q^{m},k^{m}$ )†
Static $q^{m}$ , all $k^{m}$ ; port gates dropped ( $\operatorname{Tfm}_{K,Q}$ + $\operatorname{MShift}$ )
2.344
 $+$ 0.078
56.0%

Static read & write ( $q^{m},k^{m}$ ), $-$ $\operatorname{MShift}$ 
Static $q^{m}$ , all $k^{m}$ ; port gates, $\operatorname{MShift}$ dropped (Prop. 3.1(i))
2.348
 $+$ 0.082
53.6%

Port normalization & gating (Eqs. 5–6)

 $-$ dynamic port gates†
 $\sigma(xW^{R}_{g}),\sigma(xW^{W}_{g})\equiv 1$ ; $\sigma(xW^{R}_{\bar{g}})$ and $\operatorname{MShift}$ kept
2.277
 $+$ 0.011
93.7%

 $-$ dynamic port $\operatorname{Norm}$ †
Drop $\operatorname{Norm}$ on dynamic $q^{m},k^{m}$ ; $\operatorname{Norm}(o_{i})$ , $\operatorname{MNorm}$ kept
2.289
 $+$ 0.023
87.1%

 $-$ dynamic port gates & $\operatorname{Norm}$ †
Drop both $\sigma(xW^{R}_{g}),\sigma(xW^{W}_{g})$ and port $\operatorname{Norm}$ 
2.283
 $+$ 0.017
90.3%

Sub-layer components (Section 3.2)

 $-$ $\operatorname{FFN}_{M}$ write†
Remove $\operatorname{FFN}_{M}$ and its $\operatorname{MWrite}$ (Eq. 9)
2.300
 $+$ 0.034
80.7%

 $-$ output bypass $o^{m}$ †
Drop $\operatorname{MRead}^{O}$ ; restore self-attention to current token (Eq. 8)
2.299
 $+$ 0.032
81.8%

 $-$ vector-stream RoPE $W^{Q,K}$ †
Drop $W^{Q,K}$ ; RoPE on $18$ of $75$ dims of $\tilde{q},\tilde{k}$ (Eq. 7)
2.286
 $+$ 0.020
88.7%

 $-$ $\operatorname{MShift}$ †
Set $z_{[t]}\equiv 0$ in Eq. (7)
2.277
 $+$ 0.010
94.2%

 $-$ LoomMHA write $\operatorname{Norm}(o_{i})$ 
Drop $\operatorname{Norm}$ on $o_{i}$ in LoomMHA writes (Eq. 8)
2.268
 $+$ 0.002
98.8%

Write-port gating vs. slot selection, and reduction to $\operatorname{Tfm}_{K,Q}$ .
Fixing write keys $k^{m}$ to learned static vectors while keeping their dynamic gates $\sigma(xW^{W}_{g})$ preserves nearly the full gain ( $100.0\%$ for LoomMHA, $99.3\%$ for $\operatorname{FFN}_{M}$ , and $95.6\%$ across all three write sites including the embedding).
Dropping the gate on static $k^{m}$ , however, hurts LoomMHA over three times as much as $\operatorname{FFN}_{M}$ ( $100.0\%\to 91.1\%$ vs. $99.3\%\to 96.6\%$ ): LoomMHA normalizes its head outputs $\operatorname{Norm}(o_{i})$ before writing (Eq. 8), leaving the gate as the sole control of write magnitude (and whether a head is on), whereas $\operatorname{FFN}_{M}$ writes unnormalized slices (Eq. 9) whose norms can still modulate write strength per token.
When both $q^{m}$ and $k^{m}$ are static, retaining dynamic write gates gives a node-gated circuit on fixed slots ( $65.8\%$ ), dropping those gates leaves $\operatorname{Tfm}_{K,Q}$ with $\operatorname{MShift}$ ( $56.0\%$ ), and further removing $\operatorname{MShift}$ yields the exact $\operatorname{Tfm}_{K,Q}$ of Proposition 3.1(i) ( $53.6\%$ )—which has no token-dependent port or shift gates on $M$ , yet still outperforms all layer-wise dynamic baselines at the same scale (Figure 5, where MUDDFormer reaches $48.5\%$ of LoomFormer’s gain).

Attention write-value normalization.
Removing $\operatorname{Norm}(o_{i})$ from LoomMHA writes (Eq. 8) loses only $1.2\%$ of the gain ( $98.8\%$ retained), as whole-matrix pre-normalization $\operatorname{MNorm}$ before reads (Eq. 7) already bounds the scale of $M$ .

## Appendix H Mechanistic Analysis Details and Visualizations

#### Evaluation protocol and linear attribution.

The statistics in Section 5.3 and this section are computed on held-out Pile validation sequences of length $T=2048$ at the 300B-token checkpoints of Pythia-2.8B, mHC-Pythia-2.8B and LoomPythia-2.8B (evaluating token positions $t\geq 64$ to avoid sequence-boundary transients).
Because MShift and MNorm act linearly on the accumulated writes at each layer (Equation 7), absorbing the per-slot shift gate $z_{[t]}$ (or $1-z_{[t]}$ ) and the scalar MNorm factor into $c_{ij}=\langle q^{m}_{j},k^{m}_{i}\rangle$ decomposes every MRead output as $u_{j}=\sum_{i}c_{ij}o_{i}$ over all earlier writers (embedding, LoomMHA and $\operatorname{FFN}_{M}$ writes at $t$ and $t-1$ ; $0.4\%$ median RMS difference from BF16 execution across layers).
Projecting each term onto $u_{j}$ gives the additive writer attribution

 $a_{ij}=\frac{c_{ij}\langle o_{i},u_{j}\rangle}{\|u_{j}\|^{2}},\qquad\textstyle\sum_{i}a_{ij}=1.$ 

(19)

#### Definitions for Figure 6.

In Figure 6(a), to isolate dynamic port rewiring from both static virtual weights (Proposition 3.1(i)) and per-token scalar gate or writer-norm fluctuations, let $\hat{q}^{m}_{j}(t)=q^{m}_{j}(t)/\|q^{m}_{j}(t)\|_{2}$ and $\hat{k}^{m}_{i}(t)=k^{m}_{i}(t)/\|k^{m}_{i}(t)\|_{2}\in\mathbb{S}^{d_{k}-1}$ denote the unit port directions of reader $j$ and same-token writer $i$ , weighted by each writer’s corpus RMS norm $\bar{\omega}_{i}=\sqrt{\mathbb{E}_{t}[\|o_{i}(t)\|^{2}]}$ : $\widehat{C}_{ij}(t)=\bar{\omega}_{i}\langle\hat{q}^{m}_{j}(t),\hat{k}^{m}_{i}(t)\rangle$ .
We center across tokens, $\Delta\widehat{C}(t)=\widehat{C}(t)-\mathbb{E}_{t^{\prime}}[\widehat{C}(t^{\prime})]$ , and report the energy effective rank $\exp\!\big({-}\sum_{k}p_{k}\log p_{k}\big)$ with $p_{k}=\sigma_{k}^{2}/\sum_{k^{\prime}}\sigma_{k^{\prime}}^{2}$ over the $3n=96$ Q/K/V readers of layer $l$ (or the $n=32$ readers of a single Q, K or V input); under fixed-port $\operatorname{Tfm}_{K,Q}$ with arbitrary positive scalar gates, $\Delta\widehat{C}(t)\equiv 0$ .
In Figure 6(b), for each read port $q^{m}_{j}(t)\in\mathbb{R}^{d_{k}}$ ( $\tilde{q},\tilde{k},v,o^{m}$ ) and write port $k^{m}_{i}(t)\in\mathbb{R}^{d_{k}}$ (LoomMHA and $\operatorname{FFN}_{M}$ ), we average $1-\cos(p(t_{1}),p(t_{2}))$ over distinct token pairs $t_{1}\neq t_{2}$ ( $t\geq 64$ ) and over the $n=32$ heads of layer $l$ (embedding write ports remain near-static at $0.005$ , providing a sanity check).
In Figure 6(c), following Xiao et al. (2025), a head is on at query position $t$ if $\arg\max_{s}\alpha_{ts}$ falls neither in $\{0,1\}$ nor on a sink token (<|endoftext|>, “.”, “ .”, “\n”, “\n\n”); in layers 17–32, first-two-token sink mass drops from $26.4\%$ (Pythia) and $27.2\%$ (mHC) to $1.5\%$ in LoomPythia while content-token attention mass rises from $22.7\%$ and $31.2\%$ to $65.4\%$ , and masking the self-diagonal $s=t$ in Pythia and mHC to match LoomMHA’s strict causal mask (Equation 8) leaves the gap unchanged ( $74.4\%$ vs. $21.2\%$ and $15.7\%$ ).
In Figure 6(d), for each head $h$ in layers 3–32 and each pair among its four ports $\{\tilde{q}_{h},\tilde{k}_{h},v_{h},o^{m}_{h}\}$ , we compute from token-averaged positive attributions $\bar{a}^{+}_{ij}=\mathbb{E}_{t}[\max(a_{ij}(t),0)]$ over same-token writers: (i) Pearson correlation after summing attributions by writer group (Emb, each earlier LoomMHA layer, each earlier $\operatorname{FFN}_{M}$ layer), (ii) Pearson correlation across individual head/module writers, and (iii) $|S_{A}\cap S_{B}|/32$ for the top-32 writer sets $S_{A},S_{B}$ , reporting the median across heads (in slot space $\mathbb{R}^{d_{k}}$ , same-token cosine similarity between ports of the same head averages $0.01$ – $0.07$ outside Q–K and $0.27$ on Q–K).

#### Writer composition and depth-to-depth routing.

Averaged over layers 2–32 in LoomPythia-2.8B, positive attribution mass $\max(a_{ij},0)$ is shared across LoomMHA writes ( $53.4\%$ – $62.5\%$ across $\tilde{q},\tilde{k},v,o^{m}$ ), $\operatorname{FFN}_{M}$ writes ( $27.5\%$ – $35.2\%$ ) and embedding writes ( $10.0\%$ – $11.7\%$ ), while previous-token MShift reads account for $13.7\%$ – $15.2\%$ of $\sum_{i}|a_{ij}|$ (Figure 8(a)).
 $\operatorname{FFN}_{M}$ reaches its largest share ( $35.2\%$ ) on value reads $v$ , consistent with $\operatorname{FFN}_{M}$ serving as the sole post-embedding channel from the vector stream into $M$ (Equation 9) and with its large effect in Table 4.
Plotted with writer sources on the vertical axis (bottom to top) and reader layers on the horizontal axis, the column-max normalized routing map (Figure 8(b)) combines local relay along the diagonal with persistent horizontal bands from hub writer sources—most prominently the embedding at early layers and writer layers 13–14 and 19 across all subsequent layers.

#### Adjacent-token rewiring on an in-context prompt.

Figure 9 shows the head-writer attribution submatrix $[a_{ij}]$ at hub layer 19 (selecting the top 6 readers per MRead arm and top 18 earlier LoomMHA head writers by peak $|a_{ij}|$ across the two tokens) on "France: Paris\nJapan: Tokyo\nChina: Beijing\nSpain: Madrid\nGermany:", where LoomPythia-2.8B predicts " Berlin" top-1 with probability $0.65$ .
At "Germany" (Figure 9(a)), writers route selectively to distinct MRead arms—L13.H11 feeds key reads ( $\tilde{k}$ : K14, K23, K25) without feeding value or bypass reads, whereas L13.H12 and L08.H25 feed value and bypass reads (V10, O09, O22)—forming a non-rectangular within-token pattern (Proposition 2.3) that prepares "Germany"’s horizontal keys, values and residual state.
At the immediately following delimiter ":" (Figure 9(b)), LoomMHA’s strict causal mask ( $sEquation 8) means that only $\tilde{q}$ (attending back to "Germany" and the few-shot capitals) and $o^{m}$ affect the " Berlin" prediction, whereas $\tilde{k}$ and $v$ at ":" serve only future tokens.
Correspondingly, L13.H11 and L13.H12 switch off on layer 19, while layer 19 rewires its causal query reads $\tilde{q}$ onto L18.H25, L15.H32 and L14.H01 (Q06, Q07, Q17, Q22) and its bypass outputs $o^{m}$ onto L14.H06, L16.H29 and L14.H19 (O16, O21); causally freezing layer 19’s dynamic read ports at ":" to those at "Germany" alone drops [math] from $0.65$ to $0.54$ ( $0.41$ when frozen across all layers).

[IMG alt: Refer to caption]

FIGCAPTION: Figure 8: Matrix-stream routing in LoomPythia-2.8B across Pile validation sequences.
(a) Share of positive attribution from Embedding, LoomMHA and $\operatorname{FFN}_{M}$ writes, and share of $|a_{ij}|$ from previous-token MShift ( $M_{[t-1]}$ ).
(b) Column-max normalized positive attribution from each writer source (vertical, bottom to top) to each reader layer $l$ (horizontal; gray upper triangle: causally unreachable).

[IMG alt: Refer to caption]

FIGCAPTION: Figure 9: Attribution submatrix $[a_{ij}]$ (rows: top-18 earlier head writers L $l$ .H $h$ ; columns: top-6 layer-19 MRead outputs per arm $\tilde{q},\tilde{k},v,o^{m}$ ) of LoomPythia-2.8B at two consecutive tokens of an in-context prompt (next-token top-1: " Berlin", $p=0.65$ ), showing non-rectangular within-token wiring at "Germany" (a) and per-token rewiring onto causal $\tilde{q}$ and $o^{m}$ pathways at ":" (b; hatched $\tilde{k},v$ columns affect only future tokens under $s