<!-- page 1 of 23 -->

arXiv:2611.30634v2 [cs.LG] 24 Nov 2026

# Rewirable Circuits in Transformers with Loom Attention

**Da Xiao** <sup>1</sup> **Qingye Meng** <sup>2</sup> **Shengping Li** <sup>2</sup> **Xingyuan Yuan** <sup>2</sup>

<sup>1</sup>Beijing University of Posts and Telecommunications <sup>2</sup>ColorfulClouds Technology

xiaoda99@bupt.edu.cn **Code:** [https://github.com/Caiyun-AI/LoomFormer](https://github.com/Caiyun-AI/LoomFormer)

## Abstract

We propose loom attention, which allows a Transformer to dynamically rewire its circuits at every token by interweaving a head’s vertical attention over the outputs of earlier heads (to obtain its query, key and value) with its horizontal attention over tokens. We instantiate vertical attention efficiently with normalized and gated linear attention over a per-token matrix memory, which stores the compressed head outputs and serves as a second residual stream. This yields Loom-Former, a dual-stream architecture that splits control and content between the vector and matrix streams. Across model sizes and training token budgets in language modeling and downstream tasks, LoomFormer matches Transformers trained with over 3× compute and recent residual-stream baselines (HC/mHC, MUDDFormer, AttnRes) trained with \~1.7–2.8×. For example, trained on the same 300B tokens with only 2% extra FLOPs, LoomPythia-2.8B achieves \~1.6× the average downstream gain of mHC-Pythia-2.8B over Pythia-2.8B, and matches (0-shot) or surpasses (5-shot) Pythia-12B.

## 1. Introduction

Transformers solve tasks by composing attention heads across layers into circuits (Elhage et al., 2020; Olsson et al., 2022; Wang et al., 2023; Merullo et al., 2024): an earlier head writes a feature into a subspace of the residual stream, and a later head reads from that subspace into its query, key or value. Induction heads, for example, read the output of a previous-token head through their keys (K-composition) (Olsson et al., 2022).

The composition between two heads, however, is determined by the product of their output and input projections, a virtual weight that does not change after training (Elhage et al., 2020). Attention makes the horizontal flow of information across tokens input-dependent: a head chooses which tokens to attend to. In contrast, the vertical flow across layers

![Chart block](doc:734c5bd/tier:basic/page:1/block:11)

slots of M at token t slots of M at token t· · · ′

Figure 1. Conceptual sketch of loom attention. At each token, each head chooses which slots of that token’s matrix memory M to read and write (i.e., vertical attention, every head does both; only one is shown per layer) and which tokens to attend to (horizontal attention), so a different circuit is woven for each token.

![Chart block](doc:734c5bd/tier:basic/page:1/block:14)

Figure 2. Downstream accuracy averaged over 10 tasks.

is fixed: a head cannot choose which heads to read from. A model that needs different circuits for different inputs therefore keeps all of them wired at once, lets them compete for the residual stream, and can gate them only indirectly, e.g., by sending a head’s attention to a sink token (Xiao et al., 2024b). We ask whether the wiring itself can be chosen per token, i.e., whether circuits can be made rewirable.

Recent work has enriched the vertical flow. Dense connections let a layer aggregate the outputs of all earlier layers (Pagliardini et al., 2024); MUDDFormer (Xiao et al., 2025) and Attention Residuals (AttnRes) (Kimi Team, 2026a) make this aggregation input-dependent, turning it into a depth-wise (i.e., vertical) attention (separate for the query,

1

<!-- page 2 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

key, value and residual inputs in MUDDFormer); and Hyper-Connections (HC) (Zhu et al., 2025) and its manifoldconstrained variant mHC (Xie et al., 2025) maintain several residual streams with learnable or dynamic mixing. In all of them, however, the vertical channel meets horizontal attention only through the hidden state of a layer, in which the outputs of all heads have already been summed and mixed together: a head can neither single out a particular earlier head nor be singled out by a later one. The vertical and horizontal flows run side by side, never woven together.

To make circuits rewirable, we propose loom attention, in which a head chooses which earlier heads to read from just as it chooses which tokens to attend to. By attending vertically over the outputs of these heads at the same token, the head obtains its query, key and value to perform the standard horizontal attention over tokens, and its output is in turn attended to vertically by later heads. Vertical and horizontal attention are thus interwoven, each feeding the other at every head, like the warp and weft of a loom (Figure 1).

Naive vertical attention over the outputs of all earlier heads is, however, expensive. We therefore instantiate it with normalized and gated linear attention over a per-token matrix memory, which stores the compressed head outputs and serves as a second residual stream. We call the resulting dual-stream architecture LoomFormer, after a drawloom—a loom in which a separate stored pattern selects which warp threads to lift at each pick, rather than repeating one fixed pattern. <sup>1</sup> Here the ordinary vector stream plays the pattern, carrying control—which memory slots each head writes and reads (weaving a different circuit per token) as well as which tokens to attend to—while the matrix stream carries the content moved across tokens. The two streams are coupled through attention outputs and FFN writes.

Extensive experiments on language modeling and downstream tasks across model sizes and training token budgets show that LoomFormer adds only 2.0–3.7% FLOPs while matching Transformers trained with over 3× compute and recent residual-stream baselines (HC/mHC, MUDDFormer, AttnRes) trained with \~1.7–2.8×. For example, trained on the same 300B tokens, LoomPythia-2.8B achieves \~1.6× the average downstream gain of mHC-Pythia-2.8B over Pythia-2.8B, and matches (0-shot) or surpasses (5-shot) Pythia-12B (Figure 2). We also analyze the trained 2.8B models to understand how and why LoomFormer works—e.g., by rewiring head outputs to inputs per token, it keeps over three times as many deep-layer heads off attention sinks as Pythia and mHC.

## 2. Rewirable Circuits

**Writers, readers and circuits.** Attention heads communicate through the residual stream (Elhage et al., 2020). Consider a Transformer with L layers of n heads of dim $d ,$ and model dim $D = n d .$ We call the output of a head a writer and each of its inputs (query, key and value) a reader, so a head has one writer and three readers. Writer W<sub>i</sub> adds its output $o _ { i } \in \mathbb { R } ^ { d }$ to the residual stream through its block $W _ { i } ^ { O }   \stackrel { \cdot } { \in }   \mathbb { R } ^ { d \times D }$ of the output projection, and reader $\mathsf { R } _ { j }$ of a later head reads the stream through its input projection $W _ { j } ^ { s } \in \mathbb { R } ^ { D \times d } \; ( s \in \{ Q , K , V \} )$ , so $\mathsf { W } _ { i }$ reaches $\mathsf { R } _ { j }$ through the virtual weight $\bar { C } _ { i j }   =   W _ { i } ^ { O } W _ { j } ^ { s } \in \mathbb { R } ^ { d \times d }$ : the two communicate only if W<sub>i</sub> writes into a subspace that $\mathsf { R } _ { j }$ reads. Heads composed in this way form circuits whose nodes are the writers and readers.

A task may need different circuits at different tokens, i.e., different pairs of writers and readers to be connected (Figure 3). To describe how the wiring can change, we give it a dynamic part besides the static virtual weights: at a token, reader $\mathsf { R } _ { j }$ receives

$$
\begin{array} { r } { u _ { j } = \sum _ { i } o _ { i }   c _ { i j } \bar { C } _ { i j } , } \end{array}\tag{1}
$$

where the per-token scalars $c _ { i j }$ form the wiring matrix $C \in$ $\mathbb { R } ^ { w \times r }$ from the w earlier writers to the r readers of one layer, and $c _ { i j } \bar { C } _ { i j }$ is the effective wiring from $\mathsf { W } _ { i }$ to $\mathsf { R } _ { j }$ . The two are connected at the token if their effective wiring is nonzero.<sup>2</sup> A wiring pattern $P \subseteq [ w ] \times [ r ]$ is a set of writerreader pairs, realized when each reader in $P$ is connected to exactly its writers in P.

**Definition 2.1.** An architecture with fixed virtual weights is rewirable w.r.t. a set of wiring patterns if every pattern in the set is realized by some wiring matrix C that it admits at a token.

**Transformers are node-gated.** In a Transformer, pre-LN only rescales all writers of a token by a common factor (Theorem A.1 in Section A.1). Within a token, nodes can thus only be gated through whole heads: a head that attends to a sink token (Xiao et al., 2024b) outputs nearly nothing, turning off both its writer and, since what its readers receive has no effect, its readers.

**Definition 2.2.** An architecture is node-gated if, at every token, its wiring matrix has the form

$$
C = g   g ^ { \prime \top } ,\tag{2}
$$

with gate vectors $g \; \in \; \mathbb { R } ^ { w }$ and $g ^ { \prime } \; \in \; \mathbb { R } ^ { r }$ whose nonzero entries mark the writers and readers that are on.

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>1</sup>The automated form of the drawloom, the Jacquard loom, encoded this pattern on punched cards and inspired early programmable computers such as Babbage’s Analytical Engine.</span></small>

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>2</sup>In trained models entries are rarely exactly zero; we count those below a threshold as zero (Section A.3).</span></small>

2

<!-- page 3 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

![Image block](doc:734c5bd/tier:basic/page:3/block:2)

Figure 3. Effective wiring for three wiring patterns. Transformers (top two rows) can only gate whole heads, i.e., rows and columns, on and off, so they connect only rectangles within the fixed support of their virtual weights. LoomFormer (bottom row) connects exactly the required pairs.

We call the rank of C the wiring rank; a node-gated architecture has wiring rank at most one, so existing connections can be gated on and off, but not rewired.

**Proposition 2.3.** A Transformer with pre-LN is node-gated: at any token, it connects precisely the pairs with $\bar { C } _ { i j } \neq 0$ whose writer and reader are both on, a rectangle within the fixed support of the virtual weights.

Gating a writer or a reader acts on its whole row or column of C, so a wiring pattern is realized only if it is such a rectangle, as in the first column of Figure 3; other patterns fail in one of two ways. With dense virtual weights, where heads share subspaces, every pair can be connected, but a non-rectangular pattern such as $\{ \mathsf { W } _ { 1 } \to \mathsf { R } _ { 1 } , \mathsf { W } _ { 3 } \to \mathsf { R } _ { 3 } \}$ (last column) also connects $\mathsf { W } _ { 1 } \to \mathsf { R } _ { 3 }$ and $\mathsf { W } _ { 3 } \to \mathsf { R } _ { 1 } ;$ : the two circuits interfere. With sparse virtual weights, where groups of heads use separate subspaces, circuits in different groups do not interfere, but pairs across groups such as $\mathsf { W } _ { 3 } \to \mathsf { R } _ { 2 }$ (second column) are unreachable.

**The price of rewiring by node gating.** Can a node-gated architecture overcome this by spending more heads? Suppose the role of each of $r \geq 2$ readers may be played by several heads, gated on and off per token, and at every token each reader must receive exactly one of w writers, chosen independently of the other readers; the last pattern in Figure 3 is the smallest instance. Let $N ( w , t )$ denote the minimum size of a ground set carrying a t-cover-free family of w sets, i.e., w sets none of which is covered by the union of t others (Kautz & Singleton, 1964).

**Theorem 2.4.** To realize all such wiring patterns, a nodegated architecture needs at least $N ( w , r - 1 )$ heads per reader, and this number suffices, where $N ( w , 1 )$ ≈ log<sub>2</sub> w, $N ( w , t ) = \Omega ( t ^ { 2 } \log w / \log t )$ for $2 \leq t \leq { \sqrt { w } }$ and $N ( w , t ) = \Theta ( w ) \mathit { f o r } \: t > \sqrt { w } .$

Intuitively, a head receives every writer that is on and has a nonzero virtual weight to it, so a reader must hold in reserve a head that sees its chosen writer and none of the up to $r - 1$ writers chosen by the other readers (Theorem $\mathrm { A } . 2$ in Section $\mathrm { A . 2 ; }$ bounds in Section A.3). With the $r = 3 n$ readers of a layer choosing among its w = nl earlier writers, $r - 1 > { \sqrt { w } }$ whenever $l \leq 9 n - 6 ,$ , so at practical sizes node gating needs $\Theta ( w )$ heads per reader.

**Naive loom attention makes Transformers rewirable.** To rewire with one head per reader, the wiring matrix must be chosen entry by entry at each token. The most direct way is to let every reader attend over the outputs of all earlier heads of the same token, with attention weights generated from the token itself:

$$
\begin{array} { c c } { u _ { j } = \sum _ { i } \alpha _ { i j } ( x ) \thinspace o _ { i } , \qquad \alpha _ { \cdot j } ( x ) = \operatorname { s o f t m a x } ( x W _ { j } ^ { \alpha } ) , } \\ \end{array}\tag{3}
$$

where $W _ { j } ^ { \alpha } \; \in \; \mathbb { R } ^ { D \times n l }$ for a reader in layer l, which has nl earlier writers. We call this naive loom attention: its virtual weights are identities and all the wiring is left to $C = [ \alpha _ { i j } ] ,$ , unlike a Transformer, which wires heads through fixed virtual weights and only gates them through a rankone C. With one head per reader, it is rewirable w.r.t. all wiring patterns (cf. Figure 3, bottom row), but expensive: over the 3n readers of all L layers, generating the attention weights takes about $\textstyle { \frac { 3 } { 2 } } n ^ { 2 } L ^ { 2 } D$ parameters and multiply-adds per token, growing quadratically in the total number of heads nL and rivaling the whole model at practical sizes (Section A.4).

## 3. Method

## 3.1. From Naive to Affordable Loom Attention

As linear attention compresses a growing key-value cache into a fixed-size state (Katharopoulos et al., 2020; Schlag et al., 2021), we compress the head outputs of each token along depth into a matrix memory $M \in \mathbb { R } ^ { d _ { k } \times d }$ of $d _ { k }$ slots of width d. Writer $\mathsf { W } _ { i }$ writes its output to M with a key $k _ { i } ^ { m } \in \mathbb { R } ^ { d _ { k } }$ , and reader $\mathsf { R } _ { j }$ reads with a query $q _ { j } ^ { m } \in \mathbb { R } ^ { d _ { k } }$ , so a reader attends over $d _ { k }$ slots instead of all earlier writers, at a cost independent of depth. Since M accumulates writes layer by layer, it is a second residual stream, the matrix stream, besides the original vector stream (Figure 4 left).

We call the interface through which a head reads from or writes to a stream a port. The two streams have dual ports:

$$
\begin{aligned} { } & { { } \mathsf { v e c t o r \; s t r e a m } \colon u _ { j } = x W _ { j } ^ { s } , \quad \Delta x = o _ { i } W _ { i } ^ { O } } \\ { } & { { } \mathsf { m a t r i x \; s t r e a m } \colon u _ { j } = q _ { j } ^ { m } M , \quad \Delta M = k _ { i } ^ { m } \otimes o _ { i } } \\ \end{aligned}\tag{4}
$$

3

<!-- page 4 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

![Image block](doc:734c5bd/tier:basic/page:4/block:2)

Figure 4. LoomFormer architecture: the model (left), the LoomMHA and FFN sub-layers of a block (middle), and the MRead (MR) and MWrite (MW) modules (right). Vertical attention: reads from M are in red and writes to M in blue.

where $\begin{array} { r } { k _ { i } ^ { m } \otimes o _ { i } = ( k _ { i } ^ { m } ) ^ { \top } o _ { i } } \end{array}$ is the outer product. Data and ports swap shapes: a port shrinks from a $D \times d$ matrix to a vector of length $d _ { k }$ , small enough to be generated per token from the vector stream, which we keep as the control path.

## 3.2. Architecture

LoomFormer keeps the backbone of the Transformer block and changes how sub-layers interface with the two streams: the vector stream carries control, i.e., where to attend and how to wire, and the matrix stream carries the content that is moved (Figure 4). We describe the architecture bottom up, from ports to sub-layers to the model (pseudocode in Section C), with x denoting a sub-layer’s pre-LN input.

**MRead and MWrite.** MRead and MWrite realize the matrix-stream ports of Eq. (4). $\mathrm { M R e a d } ( M , x , x ^ { r } )$ generates a query $q ^ { m }   \in   \mathbb { R } ^ { d _ { k } }$ to read $u \in \mathbb { R } ^ { d }$ from M, where $x ^ { r } =$ $\widetilde{\mathrm{GELU}(xW_1^R)} \in \mathbb{R}^{D_r}$ with $W _ { 1 } ^ { R }   \in   \mathbb { R } ^ { D \times D _ { r } } \; ( D _ { r }   \ll   D )$ is a low-rank feature shared by all MRead modules of a sub-layer to keep port generation cheap:

$$
\begin{aligned} &q^{m} = \mathrm{Norm}(x^{r}W_{2}^{R}) \odot \sigma(xW_{g}^{R}) + \bar{q} \odot \sigma(xW_{\bar{g}}^{R}) \\&u = q^{m}M\\ \end{aligned}\tag{5}
$$

Here $x ^ { r }$ is projected by $W _ { 2 } ^ { R } \in \mathbb { R } ^ { D _ { r } \times d _ { k } }$ , normalized with RMSNorm and gated to get a dynamic query, which is summed with a static query q¯ (also gated) to obtain $\boldsymbol { q } ^ { m } \in$

$\mathbb { R } ^ { d _ { k } }$ . The two scalar gates are computed from x by projecting to scalars with $\tilde { W } _ { g } ^ { R } , W _ { \bar { g } } ^ { R } \in \mathbb { R } ^ { D \times 1 }$ and then applying sigmoid. Similarly, with a shared $x ^ { w } = \mathrm { G E L U } ( x W _ { 1 } ^ { W } ) \in$ $\mathbb { R } ^ { \bar { D } _ { \tau } }$ , $\mathrm { M W r i t e } ( o , x , x ^ { w } )$ generates a key $k ^ { m } \; \in \; \mathbb { R } ^ { d _ { k } }$ and writes a vector $o \in \mathbb { R } ^ { d }$ into M as their outer product:

$$
\begin{aligned} &k^{m} = \mathrm{Norm}(x^{w}W_{2}^{W}) \odot \sigma(xW_{g}^{W}) \\&\Delta M = k^{m} \otimes o\\ \end{aligned}\tag{6}
$$

Normalization and gating separate a port’s direction from its magnitude: the normalized vector selects which slots to read or write, and the gate alone sets how much.

**LoomMHA.** To access the previous token’s memory without horizontal attention, MShift first mixes $M _ { [ t ] }$ with $M _ { [ t - 1 ] }$ via a per-slot gate $z _ { [ t ] } = \sigma ( x _ { [ t ] } W _ { g } ^ { S } ) \in \dot { \mathbb { R } } ^ { d _ { k } } ( W _ { g } ^ { \dot { S } } \in$ $\mathbb { R } ^ { D \times d _ { k } } )$ , and MNorm RMS-normalizes the matrix into $\widetilde { M } _ { [ t ] }$ Each head then reads four vectors from $\widetilde { M }$ with separate MRead modules: query, key, value and output bypass $o ^ { m }$

$$
\begin{aligned} &\widetilde{M}_{[t]} = \mathrm{MNorm}\left(z_{[t]} \odot M_{[t-1]} + (1 - z_{[t]}) \odot M_{[t]}\right) \\&\tilde{q}, \tilde{k}, v, o^m = \mathrm{MRend}^{Q, K, V, O}(\widetilde{M}, x, x^r) \\&q, k = \mathrm{RoPE}(xW^{Q, K}) \left\| \left(\tilde{q}, \tilde{k}\right)\right.\\ \end{aligned}\tag{7}
$$

where $W ^ { Q } , W ^ { K } \in \mathbb { R } ^ { D \times d / 4 }$ project x to $d / 4$ dimensions, to which rotary position embedding (RoPE) (Su et al., 2024)

4

<!-- page 5 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

is applied; the results are concatenated (∥) with $\tilde { q }$ and $\tilde { k }$ read from M. The value is read from M alone: the vector stream supplies only ports and RoPE queries/keys for slot and token routing, while all content moved by attention comes from M. Unlike MHA, LoomMHA has two outputs: the projected head outputs for the vector stream, and the writes of all heads $i \in \{ 1 , \ldots , n \}$ summed into $\Delta M ;$

$$
\begin{aligned} { } & { { } o _ { i } = o _ { i } ^ { m } + \operatorname { A t t e n t i o n } ( q _ { i } , k _ { i } , v _ { i } ) } \\ { } & { { } \Delta M = \textstyle \sum _ { i = 1 } ^ { n } \operatorname { M W r i t e } _ { i } ( \operatorname { N o r m } ( o _ { i } ) , x , x ^ { w } ) } \\ { } & { { } \operatorname { L o o m M H A } ( x , M ) = \big ( \operatorname { C o n c a t } _ { i = 1 } ^ { n } [ o _ { i } ] W ^ { O } , \thinspace \Delta M \big ) } \\ \end{aligned}\tag{8}
$$

where $q _ { i } , k _ { i } , v _ { i } , o _ { i } ^ { m }$ are given by Eq. (7) for head i, and Attention is scaled dot-product attention over tokens with a strictly causal mask that excludes the current token: a token’s own content reaches the head’s output through the bypass $o _ { i } ^ { m }$ instead. LoomMHA is where vertical and horizontal attention are interwoven: the vertical reads from $M$ feed the horizontal attention of head $i ,$ and its output $o _ { i }$ is written back to M for later layers.

**FFN with MWrite.** The feed-forward network (FFN) sublayer reads only x and, like LoomMHA, returns a pair $\left( \mathrm{FFN}_{x}(x), \Delta M \right)$ , where $\mathrm { F F N } _ { x }$ is a standard FFN module. A second, smaller $\mathrm { F F N } _ { M }$ computes the values to write, which are split into n head-sized vectors $( \mathrm { s p l i t } _ { i } )$ and written to M with the sub-layer’s MWrite modules. This write is the only channel through which content of the vector stream enters M, since attention moves only content read from M.

$$
\begin{array} { r } { \Delta M = \sum _ { i = 1 } ^ { n } \mathrm { M W r i t e } _ { i } ( \mathrm { s p l i t } _ { i } ( \mathrm { F F N } _ { M } ( x ) ) , x , x ^ { w } ) } \end{array}\tag{9}
$$

**LoomFormer.** The embedding initializes both residual streams; $M _ { 0 }$ is written from the embedding $x _ { 0 }$ in the same way as the FFN writes:

$$
\begin{aligned} { } & { { } x _ { 0 } = \operatorname { E m b e d d i n g } ( X ) } \\ { } & { { } M _ { 0 } = \textstyle \sum _ { i = 1 } ^ { n } \operatorname { M W r i t e } _ { i } ( \operatorname { s p l i t } _ { i } ( x _ { 0 } W ^ { E } ) , x _ { 0 } , x _ { 0 } ^ { w } ) } \\ \end{aligned}\tag{10}
$$

where $W ^ { E } \in \mathbb { R } ^ { D \times D }$ is a value projection and $x _ { 0 } ^ { w } ~ =$ $\operatorname { G E L U } ( x _ { 0 } W _ { 1 } ^ { W } )$ . Each block l then applies the two sublayers with pre-LN, updating both streams residually (pairs are added element-wise):

$$
\begin{aligned} { } & { { } ( x _ { l } ^ { \prime } , M _ { l } ^ { \prime } ) = ( x _ { l } , M _ { l } ) + \operatorname { L o o m M H A } ( \operatorname { L N } _ { 1 } ( x _ { l } ) , M _ { l } ) } \\ { } & { { } ( x _ { l + 1 } , M _ { l + 1 } ) = ( x _ { l } ^ { \prime } , M _ { l } ^ { \prime } ) + \operatorname { F F N } ( \operatorname { L N } _ { 2 } ( x _ { l } ^ { \prime } ) ) } \\ \end{aligned}\tag{11}
$$

for $l = 0 , \ldots , L { - } 1$ . Finally, x<sub>L</sub> is normalized and projected to produce the final output. As $M _ { L }$ is not used, the last block does not write to M. We typically set $d _ { k } = 2 n$ slots (so M has size 2D) and $D _ { r }   =   n d _ { k } / 2   =   n ^ { 2 } \; ( \mathrm { a } \; 2 \times$ compression bottleneck relative to $n d _ { k } )$ , matching the Transformer’s parameter budget by adjusting FFN widths at an approximately 2:1 ratio between $\mathrm { F F N } _ { x }$ and the $\mathrm { F F N } _ { M }$ write branch.

## 3.3. Rewiring Capacity

On the matrix stream without MShift, reader $\mathsf { R } _ { j }$ receives $\begin{array} { r } { u _ { j } \; = \; q _ { j } ^ { m } M \; = \; \sum _ { i } \langle q _ { j } ^ { m } , k _ { i } ^ { m } \rangle   o _ { i } } \end{array}$ by Eq. (4). The matrix stream thus keeps the division of roles of naive loom attention: its virtual weights are identities, and its wiring matrix is $C = K Q ^ { \top }$ , of rank at most $d _ { k }$ , where the rows of K and $Q$ are the keys $k _ { i } ^ { m }$ of the writers and the queries $q _ { j } ^ { m }$ of the readers.

**Proposition 3.1.** (i) With ports fixed to $( K , Q )$ , Loom-Former is a standard Transformer $\mathrm { T f m } _ { K , Q }$ with residual stream $( x , \operatorname { v e c } ( M ) ) \in \mathbb { R } ^ { D + d _ { k } d } ,$ , whose virtual weights absorb $K \dot { Q } ^ { \top }$ , hence node-gated. (ii) With ports generated per token, LoomFormer is rewirable with one head per reader w.r.t. the wiring patterns with at most $d _ { k }$ writers, whatever the total number of writers w.

In (ii), only the writers in the pattern are on, each taking a slot through a one-hot key while each reader’s query selects the slots of its writers (Section A.6). The $N ( w , r   -   1 )$ heads per reader that node gating needs (Theorem 2.4) are thus replaced by a per-token assignment of $d _ { k }$ slots; when more than $d _ { k }$ writers are on, slots are shared and writers interfere, as in linear attention. By (i), this rewiring capacity comes from generating ports per token, not from the matrix memory itself. Together, (i) and (ii) give a model-level view of LoomFormer’s expressiveness: within each token, Loom-Former wires heads as Tfm $K ( x ) { , } Q ( x )$ , selecting per token one Transformer from the continuum $\{ \mathrm { T f m } _ { K , Q } \}$ much as a mixture of experts selects experts (Fedus et al., 2022), except that the experts are wirings, share all other weights, and cost only $d _ { k }$ numbers per port to select.

## 4. Connections to Prior Work

Table 1 compares LoomFormer with residual-stream mechanisms (derivations in Section A.7; extended related work in Section B). In DenseFormer (Pagliardini et al., 2024), HC (Zhu et al., 2025), mHC (Xie et al., 2025), AttnRes (Kimi Team, 2026a) and MUDDFormer (Xiao et al., 2025), the ports are shared by all heads of a layer: apart from node gating and the fixed projections, their wiring depends on a writer only through its layer and on a reader only through its layer (and Q/K/V input in MUDDFormer). Their wiring rank is therefore 1, or 3 in MUDDFormer, one per Q/K/V input, whereas LoomFormer generates its ports per head and per token and reaches min $( d _ { k } , 3 n )$ , i.e., 2n by default, with a smaller state than HC $( 2 D + D$ against 4D).

**Linear attention.** Linear attention (Katharopoulos et al., 2020; Schlag et al., 2021) and its gated and delta-rule variants (Dao & Gu, 2024; Yang et al., 2024a;b; 2025; Peng et al., 2024) also read and write a fixed-size matrix state per head with keys and queries from the vector stream, but

5

<!-- page 6 of 23 -->

6

Rewirable Circuits in Transformers with Loom Attention

Table 1. LoomFormer among residual-stream mechanisms. Ports $k ^ { m } , q ^ { m }$ are indexed by heads i, j or, when shared across a layer, by layers $l _ { i } , l _ { j }$ (and Q/K/V input s), and marked (x) or (M) when computed from the vector stream or from the memory itself; softmaxl normalizes over writer layers. Typical sizes assume $d _ { k } = 4$ <sub>i</sub>n HC and $d _ { k } = 2 n$ in LoomFormer (+D for its vector stream).

|  | Effective wiring | Wiring rank | Granularity | Memory: slots × width = typical size | Residual stream |
| --- | --- | --- | --- | --- | --- |
| Transformer | C¯ij | 1 | - | 1×D = D | vector |
| DenseFormer | ⟨qlmj,klmi ⟩C¯ij | 1 | layer | l ×D = lD | vector |
| HC / mHC | ⟨qlmj(M),klmi (M)⟩C¯ij | 1 | layer | d<sub>k</sub> ×D = 4D | matrix |
| AttnRes (full) | softmaxli⟨qlmj,klmi (M)⟩C¯ij | 1 | layer | 2l ×D = 2lD | none |
| MUDDFormer | ⟨qlmj,s(x),klmi ⟩C¯ij | 3 | layer × QKV | l ×D = lD | vector |
| LoomFormer | ⟨qjm,s(x),kim(x)⟩I | min(d<sub>k</sub>,3n) = 2n | head × QKV | d<sub>k</sub> ×d = 2D(+D) | matrix + vector |

differ in axis and sharing. Their state accumulates across tokens and its read is the head output, replacing horizontal attention, whereas M accumulates along depth and its reads feed the query, key and value of horizontal attention. Their state is also private to one head across tokens, so heads still compose through the static ports of the vector stream, whereas M is shared by all heads of all layers at a token, as a channel through which heads read each other. The two are thus orthogonal and can be combined.

## 5. Experiments

We train all models on the Pile dataset (Gao et al., 2020) in JAX, initializing dynamic port gates near zero so training starts smoothly from static ports (Section D.1). We evaluate scaling laws (Section 5.1), large-scale pretraining and downstream performance (Section 5.2), mechanisms of the trained models (Section 5.3), computational efficiency (Section 5.4) and ablations (Section 5.5).

## 5.1. Scaling Laws

**Settings.** We train three model scales—Medium (432M), Large (778M) and XL (1.4B)—with sequence length $T =$ 4096, batch size 0.5M tokens (128 sequences), and token budgets proportional to non-embedding parameters following Hoffmann et al. (2022) (architectural dimensions and optimizer hyperparameters in Section D.1).

**Baselines.** We compare LoomFormer against recent residual-stream baselines on the same Transformer++ backbone (Touvron et al., 2023) (RoPE (Su et al., 2024), SwiGLU FFN (Shazeer, 2020), RMSNorm, no linear bias): DHC (dynamic HC with expansion rate 4; Zhu et al., 2025), mHC (Xie et al., 2025), AttnRes (full) (Kimi Team, 2026a), and MUDDFormer (Xiao et al., 2025) (details in Section D.2).

**Results.** Across the Medium, Large and XL scales (Figure 5), mHC, AttnRes and DHC improve over Transformer++, and MUDDFormer improves further, while LoomFormer achieves the lowest loss at every scale. In-

![Chart block](doc:734c5bd/tier:basic/page:6/block:12)

Figure 5. Scaling curves on Pile validation loss against training compute (total parameters × training tokens).

terpolating log-linearly along each baseline curve across Medium and Large, LoomFormer matches Transformer++ trained with 3.19–3.25× compute (3.10× net at Large after the 3.1% FLOP overhead in Table 3) and all four residualstream baselines at 1.73–2.76×. At XL, where LoomFormer lies below the largest model of every baseline, it lowers validation loss from 2.101 to 1.979 $( \Delta L   =   0 . 1 2 2$ , 1.8× MUDDFormer’s 0.068) at 2.8% extra FLOPs (Table 3).

## 5.2. Large-Scale Pretraining and Evaluation

**Settings.** To test whether LoomFormer’s gains persist under extended training, we compare against the Pythia suite (Biderman et al., 2023) on 300B Pile tokens (over 5× the Chinchilla budget of Section 5.1; Hoffmann et al., 2022). Specifically, we train LoomPythia-2.8B and mHC-Pythia-2.8B—the residual-stream baseline that also keeps a depthindependent state size (Table 1)—and compare them with Pythia 2.8B–12B. Except for using serial instead of parallel sub-layers , both LoomPythia and mHC-Pythia use the same architecture choices and training hyperparameters as

<small><span class="docvortex-page-footnote" data-block-type="page_footnote" style="color:#6b7280"><sup>3</sup>mHC requires serial sub-layers by design, and parallel vs. serial FFN placement is quality-neutral for standard Transformers, as shown by Chowdhery et al. (2023) and our own experiments.</span></small>

<!-- page 7 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

Table 2. Zero/five-shot downstream evaluation results (∆acc is over Pythia-2.8B; best in bold within each block).

<table><tr><td>Model</td><td>Pile ppl.↓</td><td>FLAN ppl.↓</td><td></td><td>LAM BADA</td><td>PIQA</td><td>Wino Grande</td><td>ARC -E</td><td> $\underset { - C } { \mathrm { A R C } }$ </td><td>SciQ BoolQ</td><td>Hella Swag</td><td>RACE -M</td><td>RACE -H</td><td></td><td>Avg acc↑/Δacc</td></tr><tr><td>Pythia-2.8B</td><td>6.63</td><td>8.16</td><td rowspan="6">[0-shot</td><td>64.7</td><td>73.9</td><td>59.4</td><td>64.4</td><td>29.5</td><td>88.2</td><td>64.5</td><td>45.4</td><td>38.1</td><td>34.9</td><td>56.3</td></tr><tr><td>Pythia-6.9B</td><td>6.29</td><td>7.85</td><td>67.3</td><td>75.2</td><td>60.9</td><td>67.3</td><td>31.3</td><td>89.7</td><td>63.7</td><td>48.0</td><td>40.6</td><td>37.0</td><td>58.1</td></tr><tr><td>mHC-Pythia-2.8B</td><td>6.24</td><td>7.35</td><td>68.5</td><td>74.4</td><td>62.8</td><td>67.1</td><td>30.7</td><td>90.5</td><td>66.9</td><td>47.5</td><td>42.3</td><td>37.0</td><td>58.8/+2.5</td></tr><tr><td>Pythia-12B</td><td>6.01</td><td>7.26</td><td>70.5</td><td>76.0</td><td>63.9</td><td>70.2</td><td>31.8</td><td>90.2</td><td>67.4</td><td>50.3</td><td>40.6</td><td>38.3</td><td>59.9</td></tr><tr><td>LoomPythia-2.8B</td><td>6.02</td><td>6.94</td><td>71.8</td><td>74.7</td><td>63.7</td><td>68.5</td><td>31.7</td><td>92.4</td><td>65.8</td><td>49.1</td><td>42.3</td><td>39.0</td><td>59.9/+3.6</td></tr><tr><td>Pythia-2.8B</td><td></td><td></td><td>60.5</td><td>73.6</td><td>60.6</td><td></td><td>32.3</td><td>94.3</td><td>65.6</td><td>45.1</td><td>38.4</td><td>35.6</td><td>57.3</td></tr><tr><td>Pythia-6.9B</td><td></td><td></td><td rowspan="5">5-shot</td><td></td><td>75.5</td><td>63.7</td><td>67.3 70.2</td><td>35.6</td><td>95.1</td><td>65.7</td><td>48.1</td><td>39.0</td><td>36.5</td><td>59.3</td></tr><tr><td>mHC-Pythia-2.8B</td><td></td><td></td><td>63.8 62.4</td><td>74.9</td><td>64.1</td><td>70.8</td><td>35.5</td><td>95.6</td><td>69.4</td><td>47.6</td><td>40.3</td><td>38.2</td><td>59.9/+2.6</td></tr><tr><td></td><td></td><td></td><td>67.3</td><td>76.0</td><td>64.2</td><td>71.0</td><td>36.5</td><td>95.3</td><td>68.0</td><td>50.3</td><td>40.1</td><td>38.8</td><td>60.8</td></tr><tr><td>Pythia-12B</td><td></td><td></td><td>69.0</td><td></td><td>65.7</td><td>70.8</td><td>35.1</td><td>96.0</td><td>71.2</td><td></td><td></td><td></td><td>61.6/+4.3</td></tr><tr><td>LoomPythia-2.8B</td><td></td><td></td><td></td><td>75.7</td><td></td><td></td><td></td><td></td><td></td><td>49.5</td><td>43.1</td><td>40.1</td><td></td></tr></table>

Pythia (Biderman et al., 2023, Appendix E). We evaluate zero- and five-shot accuracy across 10 standard benchmarks.

**Results.** As shown in Table 2 and Figure 2, LoomPythia-2.8B improves average accuracy over Pythia-2.8B by +3.6 in 0-shot and +4.3 in 5-shot (+3.95 on their mean), achieving 1.6× the gain of mHC-Pythia-2.8B (+2.5 / +2.6, mean +2.55). Its 0-shot average (59.9) matches Pythia-12B (59.9), its 5-shot average (61.6) exceeds Pythia-12B (60.8), and its 0/5-shot mean (60.75) matches a log-linear extrapolation of Pythia at \~4.9× compute. On pretraining metrics, LoomPythia-2.8B reaches Pile ppl 6.02, on par with Pythia-12B (6.01) and well ahead of mHC-Pythia-2.8B (6.24).

As an out-of-domain evaluation, Table 2 also reports targetspan perplexities on a randomly sampled subset of the FLAN Collection (Longpre et al., 2023), which features instruction following, chain-of-thought and in-context fewshot data. The advantage of LoomPythia on FLAN is even larger than on Pile: FLAN ppl 6.94 is best in the table, beating both Pythia-12B (7.26) and mHC-Pythia-2.8B (7.35). Whereas mHC’s gain barely changes from 0-shot to 5-shot (+2.5 vs. +2.6), LoomPythia’s widens from +3.6 to +4.3, indicating stronger in-context learning that Section 5.3 traces to per-token rewiring freeing deep heads from attention sinks.

## 5.3. Mechanistic Analysis

We analyze Pythia-2.8B, mHC-Pythia-2.8B and LoomPythia-2.8B (Section 5.2) on held-out Pile validation sequences; metric definitions and additional visualizations are in Section H.

**Per-token circuit rewiring.** After subtracting each pair’s token-mean static component, the dynamic wiring matrix on the $3 n = 9 6   \mathrm { Q / K / V }$ readers of layer l has rank 0 under fixed ports (Theorem 3.1(i)), at most 1 in Transformer and mHC, at most 3 in MUDDFormer, and up to min $( d _ { k } , 3 n ) = 6 4$ in LoomFormer (Table 1). In Figure 6(a), LoomPythia-2.8B’s dynamic effective rank rises with depth to 41.0 (26.4 mean), placing 71.0% of dynamic wiring energy (90.7% at layer 32) outside rank 1, and averages 14.3 (20.9 at layer 32) within a single Q, K or V input where MUDDFormer also has rank $\leq 1$ . Figure 6(b) directly tests Theorem 3.1(i) vs. (ii) in $\mathbb { R } ^ { d _ { k } }$ : whereas static or scalar-gated ports have zero cross-token cosine distance, LoomPythia’s read ports $\tilde { q } , \tilde { k } , v , o ^ { m }$ average 0.33–0.46 (up to 0.68) and its attention and $\mathrm { F F N } _ { M }$ write ports average 0.36 and 0.56 (up to 0.75; Section H traces an in-context circuit rewiring across adjacent tokens).

(b) Port Direction Rewiring

7

(a) Dynamic Wiring Rank

**Head utilization and within-head specialization.** In a node-gated model (Section 2, Theorem 2.4), heads whose

![Chart block](doc:734c5bd/tier:basic/page:7/block:14)

![Chart block](doc:734c5bd/tier:basic/page:7/block:15)

(c) Non-Sink Head Utilization (d) Per-Head Q/K/V/O Specialization

![Chart block](doc:734c5bd/tier:basic/page:7/block:17)

![Chart block](doc:734c5bd/tier:basic/page:7/block:18)

Figure 6. Mechanistic comparison of 2.8B models across layers $l = 1 , \ldots , 3 2$ on Pile validation sequences (definitions in Section H). (a) Energy effective rank of the token-centered dynamic Q/K/V wiring matrix (band: 10th–90th percentiles) and per single Q, K or V input, vs. the bounds of Table 1. (b) Cross-token cosine distance (1 − cos) of read ports $( \tilde { q } , \tilde { k } , v , o ^ { m } )$ and write ports $( k _ { \mathrm { M H A } } ^ { m } , k _ { \mathrm { F F N } } ^ { m } )$ in $\mathbb { R } ^ { d _ { k } }$ (static/gated = 0). (c) Percentage of heads that are on (non-sink; legend: 32-layer mean). (d) Within-head writer routing similarity across port pairs in layers 3–32.

<!-- page 8 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

Table 3. Analytical computational overhead (+FLOPs) and measured relative throughput of LoomFormer vs. matched-size Transformer at sequence length T (absolute speeds in Section F).

| Model | +F(L%O)Ps | Travin5p(%) | Prbesfi1ll/(8%) bDse1co/d1e6(/%64) |
| --- | --- | --- | --- |
| Medium | 3.70 | 90.5 | 87.4/90.4 91.7/89.2/92.3 |
| Large | 3.13 | 89.2 | 85.3/86.2 85.4/87.8/84.9 |
| XL | 2.77 | 92.3 | 83.8/85.3 86.9/84.6/86.6 |
| 2.8B | 1.95 | 92.4 | 87.5/90.5 86.4/87.6/88.9 |

fixed inputs are not needed at a token turn themselves off by attending to sink tokens (Vig & Belinkov, 2019; Xiao et al., 2024b). Counting non-sink heads as on (Xiao et al., 2025), Figure 6(c) shows that in layers 17–32, only 22.9% of heads are on in Pythia-2.8B and 17.7% in mHC-Pythia-2.8B— whose four streams still merge into one shared input per layer—whereas LoomPythia-2.8B keeps 74.4% on (3.3× and 4.2×), more than doubling content-token attention mass and accounting for its 5-shot and FLAN gains (Section 5.2). Finally, Figure 6(d) verifies head × QKV specialization (Table 1): although routing looks correlated at the layer level (0.92–0.96), per-writer correlation within the same head drops to 0.71–0.84 and top-32 writer overlap is 0.38–0.49, so over half of the top writers read by a head’s $\tilde { q } , \tilde { k } , v , o ^ { m }$ are distinct.

## 5.4. Computational Complexity and Efficiency

**Analytical computational overhead.** Under a matched parameter budget, LoomFormer adds $\scriptstyle \sim { \frac { 1 } { d ( 1 + \rho / 1 2 ) } } + { \frac { 1 } { 8 ( 1 + 1 2 / \rho ) } }$ relative forward FLOPs from reading/writing M and wider Q/K $( d _ { k } = 2 n ,   \rho = T / D ;$ derived in Section E), totaling 1.95%–3.70% across our four scales (Table 3).

**Wall-clock training and inference throughput.** Table 3 reports relative throughput for JAX training on TPU v5p pods (with fused Pallas kernels) and BF16 PyTorch inference on an NVIDIA A800 GPU (with a fused Triton kernel and torch.compile; Section F). LoomFormer sustains 89%–92% of Transformer training speed and 84%–92% of inference speed across prefill and decode batch sizes; the gap beyond the 2.0%–3.7% FLOPs reflects HBM I/O on M, lower arithmetic intensity of per-token M contractions, and smaller split projections and norms, and is far outweighed by the over 3× compute savings (Sections 5.1 and 5.2).

## 5.5. Ablation Studies

We ablate LoomFormer at the Medium scale (432M, 7B tokens; Section 5.1), reallocating saved parameters to FFN<sub>x</sub> to match the parameter budget, and report Pile validation loss and the retained gain $\mathrm{Gain} = (L_{\mathrm{Tfm}++} - L) / (L_{\mathrm{Tfm}++} -$ $L _ { \mathrm { L o o m } } )$ (Table 4; full suite in Section G).

**Dynamic rewiring.** Replacing dynamic $k ^ { m }$ , dynamic $q ^ { m }$ or both with learned static vectors (dropping their gates)

Table 4. Ablations of LoomFormer at the Medium scale.

<table><tr><td>Variant</td><td>Loss↓</td><td>∆Loss</td><td>Gain</td></tr><tr><td>LoomFormer (full) Transformer++ (baseline)</td><td>2.266 2.443</td><td>+0.177</td><td>100.0% 0.0%</td></tr><tr><td colspan="4">Dynamic rewiring (Theorem 3.1)</td></tr><tr><td>Static write keys  $( k ^ { m } )$ </td><td>2.283</td><td>+0.017</td><td>90.3%</td></tr><tr><td> $( \boldsymbol { q } ^ { m } )$  Static read queries</td><td>2.310</td><td>+0.043</td><td>75.4%</td></tr><tr><td> $\widetilde { ( q ^ { m } , k ^ { m } ) }$  Static read &amp; write</td><td>2.344</td><td>+0.078</td><td>56.0%</td></tr><tr><td colspan="4">Port normalization &amp; gating (Eqs. 5–6)</td></tr><tr><td>dynamic port gates 一 – dynamic port Norm</td><td>2.277 2.289</td><td>+0.011</td><td>93.7%</td></tr><tr><td>dynamic port gates &amp; Norm</td><td>2.283</td><td>+0.023 +0.017</td><td>87.1% 90.3%</td></tr><tr><td colspan="4">Sub-layer components (Section 3.2)</td></tr><tr><td> $- \operatorname { F F N } _ { M }$  write</td><td>2.300</td><td>+0.034</td><td>80.7%</td></tr><tr><td> $\mathrm { ~ - ~ o u t p u t \: b y p a s s \: } o ^ { m }$ </td><td>2.299</td><td>+0.032</td><td>81.8%</td></tr><tr><td> $\operatorname { R o P E } W ^ { Q , K }$  — vector-stream</td><td>2.286</td><td>+0.020</td><td>88.7%</td></tr><tr><td>- MShift</td><td>2.277</td><td>+0.010</td><td>94.2%</td></tr></table>

tests Theorem 3.1. Fixing both reduces LoomFormer to Tfm $\cdot K { , } Q$ and cuts the gain to 56.0%—still ahead of layerwise dynamic baselines (Figure 5)—so the static head-level dual-stream backbone and per-token rewiring contribute roughly equal halves. Fixing $q ^ { m }$ hurts far more than fixing $k ^ { m }$ (75.4% vs. 90.3% retained): writers can use stable slots if readers select among them per token.

**Port normalization and gating.** Removing either the dynamic-branch gates or port RMSNorm hurts, yet removing both beats removing RMSNorm alone (90.3% vs. 87.1%): an unnormalized projection already varies in norm, making a separate gate redundant. Pairing RMSNorm with gating works best by separating slot direction from read/write magnitude.

**Sub-layer components.** Removing $\mathrm { F F N } _ { M } ~ ( \mathrm { E q . } ~ 9 )$ hurts most, as the sole post-embedding path from the vector stream into M, followed closely by removing the bypass $o ^ { m }$ and restoring self-attention to the current token (Eq. 8). Applying RoPE to d/4 of the M-read coordinates instead of $\widetilde { x W ^ { Q , K } } \left( \mathrm { E q . } \; 7 \right)$ also hurts, supporting separate streams

![Chart block](doc:734c5bd/tier:basic/page:8/block:16)

Per-Token Memory Size

Figure 7. Validation loss vs. per-token memory size and matrix slots $d _ { k }$ at the Medium scale (L = 18).

8

<!-- page 9 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

for positions and content, whereas removing MShift still retains 94.2% of the gain.

**Memory capacity.** As we vary the slot count $d _ { k }$ (Figure 7), validation loss drops sharply (−0.026) from $d _ { k } = 1 6$ to 32 with diminishing returns beyond, placing our default $d _ { k } =$ 2n at the knee given the linear growth of MRead/MWrite FLOPs (Section E) and I/O with $d _ { k }$ . Even at $d _ { k } \; = \; 1 6$ (2D total state), LoomFormer already outperforms DHC (4D) and MUDDFormer (lD, averaging 9D across $L = 1 8$ layers), showing that head-level rewiring uses state memory far more effectively than layer-wise mechanisms.

## 6. Conclusion

Viewing Transformer circuits through their dynamic wiring matrix shows that standard Transformers and layer-wise residual-stream mechanisms are restricted to low-rank gating, requiring many heads per reader to avoid circuit interference. Loom attention lifts this bottleneck by interweaving vertical linear attention over a per-token matrix memory with horizontal attention over tokens, making circuits rewirable per token at one head per reader and 2.0%–3.7% extra forward FLOPs. Across model scales and 300B-token pretraining, LoomFormer matches Transformers trained with over 3× compute and recent residual-stream baselines with \~1.7–2.8×, while keeping over three quarters of its attention heads on and rewiring its ports from token to token. Because the matrix memory operates along depth at each token, combining loom attention with linear or sparse attention along the sequence is a natural direction for future foundation models.

## References

Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O’Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning (ICML), pp. 2397–2430. PMLR, 2023.

Bisk, Y., Zellers, R., Gao, J., Choi, Y., et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 7432–7439, 2020.

Brandon, W., Mishra, M., Nrusimha, A., Panda, R., and Kelly, J. R. Reducing transformer key-value cache size with cross-layer attention. arXiv preprint arXiv:2405.12981, 2024.

Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., et al. PaLM: Scaling language modeling

with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023.

Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins, M., and Toutanova, K. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044, 2019.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Conmy, A., Mavor-Parker, A., Lynch, A., Heimersheim, S., and Garriga-Alonso, A. Towards automated circuit discovery for mechanistic interpretability. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, pp. 16318–16352, 2023.

Dai, D., Deng, C., Zhao, C., Xu, R., Gao, H., Chen, D., Li, J., Zeng, W., Yu, X., Wu, Y., et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. arXiv preprint arXiv:2401.06066, 2024.

Dao, T. and Gu, A. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. In Proceedings of the Forty-First International Conference on Machine Learning (ICML), 2024.

D’yachkov, A. G. and Rykov, V. V. Bounds on the length of disjunctive codes. Problemy Peredachi Informatsii, 18 (3):7–13, 1982.

Elhage, N., Neel, N., Olsson, C., et al. A mathematical framework for transformer circuits. Transformer Circuits Thread, 2020. URL [https://transformer-circuits.pub/2021/framework/index.html](https://transformer-circuits.pub/2021/framework/index.html).

Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., et al. Toy models of superposition. Transformer Circuits Thread, 2022.

ElNokrashy, M., AlKhamissi, B., and Diab, M. Depth-wise attention (dwatt): A layer fusion method for data-efficient classification. arXiv preprint arXiv:2209.15168, 2022.

Erdos, P., Frankl, P., and F ˝ uredi, Z. Families of finite sets in ¨ which no set is covered by the union of r others. Israel Journal of Mathematics, 51:79–89, 1985.

Fang, Y., Cai, Y., Chen, J., Zhao, J., Tian, G., and Li, G. Cross-layer retrospective retrieving via layer attention. arXiv preprint arXiv:2302.03985, 2023.

Fedus, W., Zoph, B., and Shazeer, N. Switch transformers: Scaling to trillion parameter models with simple

9

<!-- page 10 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

and efficient sparsity. The Journal of Machine Learning Research, 23(1):5232–5270, 2022.

Furedi, Z. On r-cover-free families. ¨ Journal of Combinatorial Theory, Series A, 73(1):172–173, 1996.

Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., et al. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.

Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., Le Noac’h, A., Li, H., McDonell, K., Muennighoff, N., Ociepa, C., Phang, J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika, L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A. A framework for few-shot language model evaluation, 12 2023. URL [https://zenodo.org/records/10256836](https://zenodo.org/records/10256836).

Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. In Proceedings of the First Conference on Language Modeling, 2024.

Hanna, M., Pezzelle, S., and Belinkov, Y. Have faith in faithfulness: Going beyond circuit overlap when finding model mechanisms. In Proceedings of Conference on Language Modeling (COLM), 2024.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), pp. 770–778, 2016.

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., et al. An empirical analysis of compute-optimal large language model training. Advances in Neural Information Processing Systems, 35: 30016–30030, 2022.

Huang, G., Liu, Z., Van Der Maaten, L., and Weinberger, K. Q. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), pp. 4700–4708, 2017.

Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F. Transformers are RNNs: Fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

Kautz, W. H. and Singleton, R. C. Nonrandom binary superimposed codes. IEEE Transactions on Information Theory, 10(4):363–377, 1964.

Kimi Team. Attention residuals. arXiv preprint arXiv:2603.15031, 2026a.

Kimi Team. Kimi K3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026b.

Lai, G., Xie, Q., Liu, H., Yang, Y., and Hovy, E. Race: Large-scale reading comprehension dataset from examinations. arXiv preprint arXiv:1704.04683, 2017.

Leviathan, Y., Kalman, M., and Matias, Y. Selective attention improves transformer. arXiv preprint arXiv:2410.02703, 2024.

Liu, A., Feng, B., Wang, B., Wang, B., Liu, B., Zhao, C., Dengr, C., Ruan, C., Dai, D., Guo, D., et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434, 2024.

Longpre, S., Hou, L., Vu, T., Webson, A., Chung, H. W., Tay, Y., Zhou, D., Le, Q. V., Zoph, B., Wei, J., et al. The flan collection: Designing data and methods for effective instruction tuning. arXiv preprint arXiv:2301.13688, 2023.

Mak, B. and Flanigan, J. Residual matrix transformers: Scaling the size of the residual stream. In Proceedings of the 42nd International Conference on Machine Learning (ICML), pp. 42712–42729, 2025.

Merullo, J., Eickhoff, C., and Pavlick, E. Talking heads: Understanding inter-layer communication in transformer language models. arXiv preprint arXiv:2406.09519, 2024.

Ni, R., Xiao, D., Meng, Q., Li, X., Zheng, S., and Liang, H. Benchmarking and understanding compositional relational reasoning of llms. In Proceedings of the Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI), 2025.

Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, A., et al. In-context learning and induction heads. Transformer Circuits Thread, 2022.

Pagliardini, M., Mohtashami, A., Fleuret, F., and Jaggi, M. Denseformer: Enhancing information flow in transformers via depth weighted averaging. In Proceedings of the Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS), 2024.

Paperno, D., Kruszewski, G., Lazaridou, A., Pham, Q. N., Bernardi, R., Pezzelle, S., Baroni, M., Boleda, G., and Fernandez, R. The lambada dataset: Word prediction ´ requiring a broad discourse context. arXiv preprint arXiv:1606.06031, 2016.

Peng, B., Goldstein, D., Anthony, Q., Albalak, A., Alcaide, E., Biderman, S., Cheah, E., Du, X., Ferdinan, T., Hou, H., et al. Eagle and finch: Rwkv with matrixvalued states and dynamic recurrence. arXiv preprint arXiv:2404.05892, 2024.

10

<!-- page 11 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

Ruszinko, M. On the upper bound of the size of the r-cover- ´ free families. Journal of Combinatorial Theory, Series A, 66(2):302–310, 1994.

Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021.

Schlag, I., Irie, K., and Schmidhuber, J. Linear transformers are secretly fast weight programmers. In Proceedings of the 38th International Conference on Machine Learning (ICML), 2021.

Shazeer, N. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.

Sperner, E. Ein satz uber untermengen einer endlichen ¨ menge. Mathematische Zeitschrift, 27:544–548, 1928.

Srivastava, R. K., Greff, K., and Schmidhuber, J. Training very deep networks. Advances in neural information processing systems, 28, 2015.

Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Tay, Y., Dehghani, M., Aribandi, V., Gupta, J., Pham, P. M., Qin, Z., Bahri, D., Juan, D.-C., and Metzler, D. Omninet: Omnidirectional representations from transformers. In International Conference on Machine Learning (ICML), pp. 10193–10202. PMLR, 2021.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E., \` Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Vig, J. and Belinkov, Y. Analyzing the structure of attention in a transformer language model. arXiv preprint arXiv:1906.04284, 2019.

Wang, K., Variengien, A., Conmy, A., Shlegeris, B., and Steinhardt, J. Interpretability in the wild: a circuit for indirect object identification in gpt-2 small. In Proceedings of the Eleventh International Conference on Learning Representations (ICLR), 2023.

Wang, K., Xia, X., Liu, J., Yi, Z., and He, T. Strengthening layer interaction via dynamic layer attention. arXiv preprint arXiv:2406.13392, 2024.

Wang, Q., Li, B., Xiao, T., Zhu, J., Li, C., Wong, D. F., and Chao, L. S. Learning deep transformer models for machine translation. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), 2019.

Welbl, J., Liu, N. F., and Gardner, M. Crowdsourcing multiple choice science questions. arXiv preprint arXiv:1707.06209, 2017.

Xiao, D., Meng, Q., Li, S., and Yuan, X. Improving transformers with dynamically composable multi-head attention. 2024a.

Xiao, D., Meng, Q., Li, S., and Yuan, X. MUDDFormer: Breaking residual bottlenecks in transformers via multiway dynamic dense connections. In Proceedings of the 42nd International Conference on Machine Learning (ICML), 2025.

Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations (ICLR), 2024b.

Xie, Z., Wei, Y., Cao, H., Zhao, C., Deng, C., Li, J., Dai, D., Gao, H., Chang, J., Yu, K., Zhao, L., Zhou, S., Xu, Z., Zhang, Z., Zeng, W., Hu, S., Wang, Y., Yuan, J., Wang, L., and Liang, W. mHC: Manifold-constrained hyperconnections. arXiv preprint arXiv:2512.24880, 2025.

Yang, S., Wang, B., Shen, Y., Panda, R., and Kim, Y. Gated linear attention transformers with hardware-efficient training. In Proceedings of the 41st International Conference on Machine Learning (ICML), pp. 56501–56523, 2024a.

Yang, S., Wang, B., Zhang, Y., Shen, Y., and Kim, Y. Parallelizing linear transformers with the delta rule over sequence length. In Proceedings of the Thirty-Eighth Annual Conference on Neural Information Processing Systems, 2024b.

Yang, S., Kautz, J., and Hatamizadeh, A. Gated delta networks: Improving Mamba2 with delta rule. In Proceedings of the Thirteenth International Conference on Learning Representations (ICLR), 2025.

Ye, T., Dong, L., Xia, Y., Sun, Y., Zhu, Y., Huang, G., and Wei, F. Differential transformer. 2024.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019.

Zhu, D., Huang, H., Huang, Z., Zeng, Y., Mao, Y., Wu, B., Min, Q., and Zhou, X. Hyper-connections. In Proceedings of the Thirteenth International Conference on Learning Representations (ICLR), 2025.

11

<!-- page 12 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

## A. Theory of Rewirable Circuits

## A.1. Normalization and proof of Proposition 2.3

**Lemma A.1.** In a Transformer with pre-LN (LayerNorm or RMSNorm), the contribution of earlier writers to reader $\mathsf { R } _ { j }   ^ { \prime } s$ input $\begin{array} { r } { u _ { j } = \mathrm { L N } ( x ) W _ { j } ^ { s } \; \mathrm { i s } \; \lambda ( x ) \sum _ { i } o _ { i } \bar { C } _ { i j } } \end{array}$ , where the scale $\lambda ( x ) > 0$ is shared by all writers and the virtual weights $\bar { C } _ { i j }$ are fixed.

Proof. Write $\begin{array} { r } { x = \sum _ { i } o _ { i } W _ { i } ^ { O } + x _ { \mathrm { o t h e r } } } \end{array}$ , where $x _ { \mathrm { o t h e r } }$ sums the embedding and earlier FFN outputs. LayerNorm computes $\mathrm { L N } ( x )   =   \lambda ( x )   ( x   -   \mu ( x ) \mathbf { 1 } ) \odot \gamma + \beta$ with $\lambda ( x )   =   \sqrt { D } / \| x   -   \mu ( x ) \mathbf { 1 } \|   >   0$ (putting $\lambda ( x )   =   0$ on the degenerate set $x \in \mathrm { s p a n } ( \mathbf { 1 } ) )$ . Since centering is the fixed linear map $x - \hat { \mu ( x ) } \mathbf { 1 } = x \left( I - \frac { 1 } { D } \mathbf { 1 } \mathbf { 1 } ^ { \top } \right)$

$$
\begin{array} { r } { u _ { j } = \mathrm { L N } ( x ) W _ { j } ^ { s } = \lambda ( x ) \sum _ { i } o _ { i } \bar { C } _ { i j } + \lambda ( x )   x _ { \mathrm { o t h e r } } \big ( I - \frac { 1 } { D } \mathbf { 1 } \mathbf { 1 } ^ { \top } \big )   \mathrm { d i a g } ( \gamma ) W _ { j } ^ { s } + \beta W _ { j } ^ { s } , } \end{array}
$$

with fixed virtual weights $\begin{array} { r } { \bar { C } _ { i j } \; = \; W _ { i } ^ { O } \big ( I   -   \frac { 1 } { D } \mathbf { 1 } \mathbf { 1 } ^ { \top } \big ) \operatorname { d i a g } ( \gamma ) W _ { j } ^ { s } } \end{array}$ absorbing the linear maps of the normalization (any projection bias adds to $\beta W _ { j } ^ { s } )$ . For RMSNorm, centering is omitted, $\beta = 0$ and $\lambda ( x ) = \sqrt { D } / \| x \|$ □

Proof of Proposition 2.3. By Theorem A.1, pre-LN rescales all writers of a token by the same $\lambda ( x ) > 0$ . Within a token, what depends on the input is only this scale and which heads are on: let $g _ { i } = \lambda ( x )$ if the head of $\mathsf { W } _ { i }$ is on and 0 otherwise, and $g _ { j } ^ { \prime } = 1$ if the head of $\mathsf { R } _ { j }$ is on and 0 otherwise. Then $C = g   g ^ { \prime \top }$ , which is Eq. (2), and since $\lambda ( x ) > 0 , \mathsf { W } _ { i }$ and $\mathsf { R } _ { j }$ are connected $( g _ { i } g _ { j } ^ { \prime } \bar { C } _ { i j } \neq 0 )$ if and only if both are on and $\bar { C } _ { i j } \neq 0 .$ □

## A.2. Full statement and proof of Theorem 2.4

**Setting.** The role of reader $\mathsf { R } _ { j } \: ( j \in [ r ] ,   r \geq 2 )$ is played by a set $H _ { j }$ of heads, so a reader is here a role rather than one input of one head; by Eq. (2), head $h \in H _ { j }$ receives $\mathsf { W } _ { i }$ when its effective wiring $g _ { i } g _ { h } ^ { \prime } \bar { C } _ { i h }$ is nonzero. A wiring task assigns to each reader a writer $\pi ( j ) \in [ w ]$ , and the tasks are all maps $\pi : [ r ] \to [ w ]$ . Reader $\mathsf { R } _ { j }$ receives exactly $\mathsf { W } _ { \pi ( j ) }$ if some $h \in H _ { j }$ receives $\mathsf { W } _ { \pi ( j ) }$ and no $h \in H _ { j }$ receives any $W _ { i }$ with $i \neq \pi ( j )$ . We require every head in $H _ { j }$ that is on to be free of other writers because query and key inputs feed a softmax, so interference at one head cannot be cancelled downstream. Every writer in $\pi ( [ r ] )$ must therefore be on $( g _ { \pi ( j ) } \neq 0 )$ , and turning others on can only add interference, so we may assume the writers that are on are $\pi ( [ r ] )$ . For a fixed reader $\mathsf { R } _ { j } ,$ let $B _ { i } = \{ h \in H _ { j } : \bar { C } _ { i h } \neq 0 \}$ . A family $\{ B _ { i } \} _ { i = 1 } ^ { w }$ is t-cover-free if $\textstyle B _ { i } \not \subseteq \bigcup _ { i ^ { \prime } \in J } B _ { i ^ { \prime } }$ for every $i \in [ w ]$ and every $J \subseteq [ w ] \mathbin { \mathring { \setminus } } \{ i \}$ with $| J | \leq t$

**Theorem A.2** (full version of Theorem 2.4). $For   r \geq 2,$ , a node-gated architecture with w writers realizes all wiring tasks $\pi : [ r ] \to [ w ] \; i f$ and only if, for every reader, $\{ B _ { i } \} _ { i = 1 } ^ { w }$ is an $( r - 1 )$ -cover-free family. Hence each reader needs at least $N ( w , r - 1 )$ heads, and this number suffices.

Proof. If. Given π, turn on only the writers $\pi ( [ r ] )$ . For reader $\mathsf { R } _ { j } ,$ let $J = \pi ( [ r ] ) \setminus \{ \pi ( j ) \}$ }, so $| J | \leq r - 1$ . Since $\{ B _ { i } \}$ is $( r - 1 )$ -cover-free, there is a head $h \in B _ { \pi ( j ) } \setminus \textstyle \bigcup _ { i \in J } B _ { i }$ ; turn on h and no other head of $H _ { j }$ . Then h receives $\mathsf { W } _ { \pi ( j ) }$ and none of the other writers that are on.

Only if. Fix a reader $\mathsf { R } _ { j }$ , a writer i and a set $J \subseteq [ w ] \setminus \{ i \}$ with $| J | \leq r - 1$ . Choose π with $\pi ( j ) = i$ that maps the other $r - 1$ readers onto $J ,$ which is possible since $r - 1 \geq 1 , | J | \leq r - 1$ and different readers may share a writer (if $J = \emptyset$ , map them to i). Realizing π requires a head $h \in H _ { j }$ that is on with $\bar { C } _ { i h } \neq 0$ and $\bar { C } _ { i ^ { \prime } h } = 0$ for all $\begin{array} { r } { i ^ { \prime } \in J , \mathrm { i . e . } , h \in B _ { i } \setminus \bigcup _ { i ^ { \prime } \in J } B _ { i ^ { \prime } } } \end{array}$ Hence $\textstyle B _ { i } \not \subseteq \bigcup _ { i ^ { \prime } \in J } B _ { i ^ { \prime } }$ , and the family is $( r - 1 )$ -cover-free.

Counting. The sets $B _ { i }$ are subsets of $H _ { j } , \mathrm { s o } \; | H _ { j } | \geq N ( w , r - 1 )$ . Conversely, given an $( r - 1 )$ -cover-free family of w sets on a ground set of size $N ( w , r - 1 )$ for each reader, create one head per point and set $\bar { C } _ { i h } = I _ { d } \; \mathrm { i f } \; h \in B _ { i }$ and 0 otherwise; this is realized in $D = \operatorname* { m i n } \{ w , r N ( w , r - 1 ) \}$ d dimensions by assigning orthogonal d-dimensional subspaces either to the w writers or to the $r N ( w , r - 1 )$ heads. □

**Remarks.** (i) Attention sinks gate heads off only approximately; under the threshold of Section 2, the counting argument is unchanged. (ii) Sufficiency at $N ( w , r - 1 )$ heads per reader holds in a Transformer when each of the r readers belongs to a separate head (with its other two inputs fixed); when $r = 3 n$ counts all Q, K and V inputs of n heads, coupling Q, K and V inside the same physical head (and sink gate $g _ { h } ^ { \prime } )$ can only require more heads, while the lower bound $N ( w , r - 1 )$ still applies. Likewise, if several heads of a value reader share identical attention weights, their outputs add linearly and

12

<!-- page 13 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

interference could in principle cancel, whereas our clean-head requirement is necessary for query and key inputs. (iii) The model excludes routing through FFNs, discussed in Section A.5.

## A.3. Bounds on $N ( w , t )$ and remarks on Theorem 2.4

**Bounds.** A 1-cover-free family is an antichain, so by Sperner’s theorem (Sperner, 1928) $N ( w , 1 )$ is the smallest N with $\binom { N } { \lfloor N / 2 \rfloor }   \geq   w$ , about $\log _ { 2 } w$ . For $2   \leq   t   \leq   \sqrt { w } ,   N ( w , t )   =   \Omega ( t ^ { 2 } \log w / \log t )     ( \mathrm { D } )$ yachkov & Rykov, 1982; Ruszinkó, 1994; Furedi ¨ , 1996) and $N ( w , t ) = \mathbb { O } ( \operatorname* { m i n } \{ t ^ { 2 }$ log w, w}) (Erdos et al. ˝ , 1985); for $t > \sqrt { w }$ , Bassalygo’s bound $N ( w , t ) \geq$ min $\{ \binom { t + 2 } { 2 } , w \}$ (D’yachkov & Rykov, 1982) together with the singleton construction $N ( w , t ) \leq u$ gives $N ( w , t ) = \Theta ( w )$ reaching $N ( w , t ) = w$ (one head per writer) once ${ \binom { t + 2 } { 2 } } \geq w$

**Group testing.** Theorem $\mathrm { A } . 2$ is an instance of non-adaptive group testing: the heads of a reader are tests that must isolate the selected writer from up to $r - 1$ other writers that are on.

**Dense and sparse layouts.** With a single head per reader $( | H _ { j } | = 1 )$ , the two layouts of Figure 3 are degenerate codes: $B _ { 1 } = \cdots = B _ { w } = H _ { j }$ (dense), which is not 1-cover-free and makes circuits interfere, and $B _ { i } = \emptyset$ for some writer i (sparse), which leaves $( i , j )$ unreachable.

**Superposition.** At layer scale $( r = 3 n , N ( w , r - 1 ) = \Theta ( w ) )$ , separating w writers as above takes $\Theta ( w d )$ dimensions (wd with orthogonal writer subspaces). With nL writers sharing $D = n d$ dimensions, writers across layers must overlap in superposition (Elhage et al., 2022), so virtual weights are rarely exactly zero and in practice part of the price is paid as interference.

## A.4. Cost of naive loom attention

Indexing layers by $l = 0 , \ldots , L - 1$ as in Section 3.2, every reader of layer l attends over nl earlier head writers. Over the 3n readers of all L layers, the projections $W ^ { \alpha }$ of Eq. (3) take $\begin{array} { r } { \sum _ { l = 0 } ^ { L - 1 } 3 n \dot { \boldsymbol { \cdot } \boldsymbol { n } } l \cdot \boldsymbol { D } = \frac { 3 } { 2 } n ^ { 2 } L ( L - 1 ) \boldsymbol { D } \approx \frac { 3 } { 2 } n ^ { 2 } L ^ { 2 } \boldsymbol { D } } \end{array}$ parameters, and as many multiply-adds per token. This grows quadratically in the total number of heads nL and rivals the $1 2 L D ^ { 2 }$ of the whole model once $n ^ { 2 } L \approx 8 D   ( \mathrm { i . e . , }   n L \approx 8 d ) , \mathrm { e . g . }$ , 70% of it for our XL configuration $( L = 2 8 ,   n = 2 0 ,   D = 1 9 2 0 )$ Factorizing the weights into keys and queries removes $W ^ { \alpha }$ , but still leaves ≈ $\textstyle { \frac { 3 } { 2 } } n ^ { 2 } L ^ { 2 } \vec { }$ attention weights per token to compute and store for backpropagation, plus ≈ ${ \textstyle { \frac { 3 } { 2 } } } n L ^ { 2 } D$ multiply-adds to aggregate the ${ o _ { i } } ^ { \prime } \mathbf { S }$

## A.5. Routing through FFNs

An intermediate FFN between the writers and the reader layer can route any pattern $P \subseteq [ w ] \times [ r ]$ by copying $o _ { i }$ into a d-dimensional subspace private to $\mathsf { R } _ { j }$ whenever $( i , j ) \in P$ . Concretely, suppose the residual stream $\stackrel { \frown } { x \in \mathbb { R } ^ { D } }$ carries binary indicators $p _ { i j } \in \{ 0 , 1 \}$ of $P \left( p _ { i j } = 1 \; { \mathrm { i f f } } \; ( i , j ) \in P \right)$ and the writer outputs satisfy $\| o _ { i } \| _ { \infty } \leq \Lambda$ (ignoring pre-LN rescaling). Since $\mathrm{ReLU}\left(z-\Lambda(1-p)\right)-\mathrm{ReLU}\left(-z-\Lambda(1-p)\right)$ equals z when $p = 1$ and 0 when $p = 0$ for every $z   \in   [ - \Lambda , \Lambda ] ,$ a two-layer ReLU FFN can compute the gated copy $p _ { i j } o _ { i }   \in   \mathbb { R } ^ { d }$ coordinate-wise for each pair $( i , j )$ and sum over i into $\mathsf { R } _ { j } \dot { \mathsf { \Sigma } } \mathbf { s }$ private subspace. However, this construction uses 2wrd hidden units—hence $4 w r d D = \mathbb { O } ( w r d D )$ parameters, or $\dot { 1 2 n l D ^ { 2 } }$ when routing $w = n l$ writers to $r = 3 n$ readers (n times the $1 2 l D ^ { 2 }$ parameters of all l preceding blocks)—requires an intervening FFN step, and needs disjoint subspaces for the w writers, r readers and wr control coordinates $( D \geq ( w + r ) d + w r )$ . We give this construction only as an upper bound and do not prove a lower bound on FFN routing.

## A.6. Proof of Proposition 3.1

**Part (i).** Flatten $M   \in   \mathbb { R } ^ { d _ { k } \times d }$ row by row into vec $\mathbf { \mathit { c } } ( M )   \in   \mathbb { R } ^ { d _ { k } d } ,$ , so that entry $( a , b )$ of M is entry $( a - 1 ) d + b$ of vec(M). For a static read port $\textstyle q \in \mathbb { R } ^ { d _ { k } } , ( q M ) _ { b } = \sum _ { a } \widehat { q _ { a } M _ { a b } } = \big ( \operatorname { v e c } ( M ) \left( q ^ { \top } \otimes \widetilde { I _ { d } } \right) \big ) _ { b } .$ For a static write port $k \in \mathbb { R } ^ { d _ { k } }$ $\operatorname { v e c } ( k ^ { \top } o ) _ { ( a - 1 ) d + b } \: = \: k _ { a } o _ { b } \: = \: \big ( o \left( k { \otimes } I _ { d } \right) \big ) _ { ( a - 1 ) d + b } .$ Hence every read and write of M is a fixed linear map between $\operatorname { v e c } ( M )$ and the head, i.e., a Transformer projection with Kronecker structure, $q ^ { \top } \otimes I _ { d }$ for reads and $k \otimes I _ { d }$ for writes, and the composition weight between $\mathsf { W } _ { i }$ and $\mathsf { R } _ { j }$ is the constant $\langle q _ { j } ^ { m } , k _ { i } ^ { m } \rangle$ . The coupling $K Q ^ { \top }$ is therefore fixed and belongs to the virtual weights, $\bar { C } _ { i j } = \langle q _ { j } ^ { m } , k _ { i } ^ { m } \rangle I _ { d } ;$ only gating heads through attention sinks (together with normalization scales) depends on the input, as in Eq. (2), so $\mathrm { T f m } _ { K , Q }$ is node-gated. In the full architecture (Section 3.2), MNorm is

13

<!-- page 14 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

Table 5. Ports of residual-stream mechanisms in the common form of Eq. (12).

|  | Write port kim | Read port qjm | Update Φ<sub>l</sub> |
| --- | --- | --- | --- |
| Transformer | 1 | 1 | 1 |
| DenseFormer | e<sub>li</sub> | static | I |
| HC / mHC | β<sub>li</sub>(M) | α<sub>lj</sub>(M) | A<sub>l</sub>(M) |
| AttnRes (full) | Norm(v<sub>li</sub>) | static, softmax | I |
| MUDDFormer | e<sub>li</sub> | per Q/K/V, from x | I |
| LoomFormer | per head, from x | per head and Q/K/V, from x | I |

RMSNorm over vec(M) and rescales all matrix-stream writers by one shared factor (Theorem A.1), while the bypass $o _ { i } ^ { m }$ and $( \operatorname { F F N } _ { x } , \operatorname { F F N } _ { M } )$ become a fixed linear skip and a standard FFN on $( x , \operatorname { v e c } ( M ) )$ . The per-head $\mathrm { N o r m } ( o _ { i } )$ on LoomMHA writes (Eq. 8) applies a fixed diagonal gain (absorbed into the write projection) and rescales writer $\mathsf { W } _ { i }$ by $\sqrt { d } / \| o _ { i } \|$ , and even if the scalar port gates $\sigma ( x W _ { g } ^ { W } ) , \sigma ( x W _ { \bar { g } } ^ { R } )$ (Eqs. 5–6) are retained on static port directions, they multiply whole rows or columns of C, leaving $C = g   g ^ { \bar { \bar { \top } } }$ of rank at most one (cf. Section G). With ports generated per token, the computation at each token and layer coincides with that of Tfm $K ( x ) { , } Q ( x )$ . The coincidence holds within a token; across tokens, attention mixes values computed by different members, so the model as a whole is not any single $\mathrm { T f m } _ { K , Q }$

**Part (ii).** Let $S _ { W }$ be the writers in a wiring pattern $P \subseteq [ w ] \times [ r ]$ , with $| S _ { W } | \leq d _ { k }$ . Turn on only $S _ { W }$ , pick an injection $\iota : S _ { W } \to [ d _ { k } ]$ , give each $i \in S _ { W }$ the write key $k _ { i } ^ { m }   =   e _ { \iota ( i ) } \; ( k _ { i } ^ { m }   =   0$ for $i \notin S _ { W } )$ , and give reader $\mathsf { R } _ { j }$ the read query $\textstyle q _ { j } ^ { m } = \sum _ { i \in S _ { W } : \: ( i , j ) \in P } e _ { \iota ( i ) }$ Then $\langle q _ { j } ^ { m } , k _ { i } ^ { m } \rangle = 1 \; \mathrm { i f } \; \dot { ( } i , j ) \in \dot { P }$ and 0 otherwise for all $i   \in   [ w ]$ , so $c _ { i j } \neq 0$ if and only if $( i , j ) \in P ;$ since the virtual weights of the matrix stream are identities, the connected pairs are $P ,$ with one head per reader and $d _ { k }$ independent of $w .$ The RMSNorm of the actual ports rescales each key and each query by a constant, and the sigmoid gates set their magnitudes (turning off writers outside $S _ { W } )$ , so the pattern is realized up to per-reader scales. In the full model, computing ι and P from the vector stream is the role of the port generators on the control path.

## A.7. Wiring matrices of residual-stream mechanisms

This section derives the entries of Table 1. As in Section 2, we consider the wiring within a token from the earlier head writers $( o _ { i } \in \mathbb { R } ^ { d } )$ to the query, key and value readers $( u _ { j } )$ of one later layer. Other reads of the history—the FFN inputs of HC/mHC and AttnRes, the residual input of MUDDFormer and the bypass $o ^ { m }$ of LoomFormer—feed no head input and are not readers. We mark a port by (x) if it is computed from the vector (residual) stream and by (M) if it is computed from the memory itself. Per-token scales such as pre-LN are absorbed into the wiring matrix as in Theorem $\mathrm { A . 1 }$ , and the node-gating factors $g , g ^ { \prime }$ of Theorem 2.2 are common to all mechanisms, so we omit them from the tables. For LoomFormer we consider the matrix stream only and ignore MShift, which mixes neighbouring tokens (the $d / \cdot$ 4-dimensional RoPE slice from x belongs to the node-gated vector stream of Theorem 2.3). Ports are row vectors, as in Section 3.1.

**A common form.** Each mechanism keeps at a token a memory M of $d _ { k }$ slots of width $d _ { v }$ . Writer $\mathsf { W } _ { i }$ adds $k _ { i } ^ { m } \otimes ( o _ { i } W _ { i } ^ { O } )$ to M, reader $\mathsf { R } _ { j }$ receives $u _ { j } = \left( q _ { j } ^ { m } M \right) W _ { j } ^ { s }$ , and between layers M may be updated linearly, $M \leftarrow \Phi _ { l } M ;$ here $k _ { i } ^ { m }$ km and $q _ { j } ^ { m }$ are the write and read ports, and $\dot { W } _ { i } ^ { O } = \dot { W } _ { j } ^ { s } = I _ { d }$ in the matrix stream of LoomFormer. Composing these steps gives

$$
u _ { j } = \sum _ { i } \left( q _ { j } ^ { m } \: \Phi _ { l _ { i } l _ { j } } \: ( k _ { i } ^ { m } ) ^ { \top } \right) o _ { i } W _ { i } ^ { O } W _ { j } ^ { s } ,\tag{12}
$$

14

where $\Phi _ { l _ { i } l _ { j } } = \Phi _ { l _ { j } - 1 } \cdots \Phi _ { l _ { i } }$ is the product of the updates between the layer $l _ { i }$ of $\mathbb { W } _ { i }$ and the layer $l _ { j }$ of $\mathsf { R } _ { j } ;$ the wiring matrix has entries $c _ { i j } = q _ { j } ^ { m } \Phi _ { l _ { i } l _ { j } } ( k _ { i } ^ { m } ) ^ { \top }$ and the virtual weights are $\bar { C } _ { i j } = W _ { i } ^ { O } W _ { j } ^ { s }$ . This representation is not unique: a change of slot basis by an invertible G maps $k _ { i } ^ { m } \mapsto k _ { i } ^ { m } G ^ { \top } , q _ { j } ^ { m } \mapsto q _ { j } ^ { m } G ^ { - 1 }$ and $\vec { \Phi _ { l } } \mapsto G \Phi _ { l } G ^ { - 1 }$ without changing $c _ { i j }$ . For mechanisms that keep the outputs of all earlier layers, we use the basis in which every stored output has its own slot, the write port is $e _ { l _ { i } }$ and $\Phi _ { l } = I ;$ residual copies then become part of the read port. Transitions $\Phi _ { l _ { i } l _ { j } }$ can also be absorbed into the ports without adding slots: for a fixed reader layer $l _ { j } ,$ setting $\tilde { k } _ { i } ^ { m } = k _ { i } ^ { m } \Phi _ { l _ { i } l } ^ { \top }$ writes $c _ { i j } = \dot { \langle q _ { j } ^ { m } , \tilde { k } _ { i } ^ { m } \rangle }$ even when $\Phi _ { l }$ is singular (and when every $\Phi _ { l }$ is invertible, setting $\Phi _ { l } ^ { \times } = \Phi _ { l - 1 } \cdots \Phi _ { 1 }$ makes this split independent of $l _ { j }$ via $\tilde { q } _ { j } ^ { m } = q _ { j } ^ { m } \Phi _ { l _ { j } } ^ { \times }$ and $\tilde { k } _ { i } ^ { m } = k _ { i } ^ { m } \Phi _ { l _ { i } } ^ { \times - \top } )$ . Table 5 lists the parameterization of each mechanism, justified below; Table 1 omits factors equal to 1 or I.

<!-- page 15 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

15

**Transformer.** One slot of width $D$ with $k _ { i } ^ { m } = q _ { j } ^ { m } = \Phi = 1 \; \mathrm { g i v e s } \; c _ { i j } = 1$ and the virtual weights $\bar { C } _ { i j } = W _ { i } ^ { O } W _ { j } ^ { s }$ ; with the node gates, this is Eq. (2).

**DenseFormer.** The input of each block is a static weighted sum of the embedding and the outputs of all earlier blocks, and the block adds its own attention and FFN outputs to this input through its internal residual (Pagliardini et al., 2024). Unfolding these residual copies, the input of block $l ^ { \prime }$ is a static combination of the outputs of all earlier blocks, so $c _ { i j } \bar { C } _ { i j } =$ $\langle q _ { l _ { j } } ^ { m } , k _ { l _ { i } } ^ { m } \rangle   W _ { i } ^ { O } W _ { j } ^ { s }$ with a static read port, so only node gating makes the wiring matrix input-dependent. DenseFormer keeps one slot per block, about l slots at layer l (size lD).

**HC and mHC.** HC keeps $d _ { k }$ parallel streams of width D (we use $d _ { k } = 4 ,$ the main configuration of Zhu et al. (2025)). Each sub-layer reads $\alpha M$ and updates the streams as $M \gets A M + \beta ^ { \top } h$ , with $\alpha , \beta$ and A all computed from M in the dynamic version (Zhu et al., 2025), and A projected onto the Birkhoff polytope of doubly stochastic matrices in mHC (Xie et al., 2025). Heads read and write through the attention sub-layer, with ports $\alpha _ { l }$ and $\beta _ { l } ,$ while the FFN and subsequent attention stream-mixing matrices between attention sub-layers $l _ { i }$ and $l _ { j }$ compose into $\Phi _ { l _ { i } l _ { j } } = A _ { l _ { i } l _ { j } }$ . In the common form, $c _ { i j } = \alpha _ { l _ { j } } A _ { l _ { i } l _ { j } } \beta _ { l _ { i } } ^ { \top }$ the unrolled weight of Eq. (27) of Zhu et al. (2025), which Kimi Team (2026a) interpret as depth-wise linear attention with queries α and keys $\beta .$ Since all factors are computed from the streams $M$ , absorbing $A _ { l _ { i } l _ { j } }$ into the write port as above (without requiring $A _ { l }$ to be invertible) gives the entry $\langle q _ { l _ { j } } ^ { m } ( M ) , k _ { l _ { i } } ^ { m } ( M ) \rangle$ of Table 1, with $d _ { k }$ -dimensional ports and state size $d _ { k } D = 4 D$

**AttnRes.** Full AttnRes keeps the embedding and the outputs $v _ { m }$ of all earlier sub-layers, without a residual connection, and gives sub-layer m′the input $\Sigma _ { m }$ sof $\operatorname { \dot { t } m a x } _ { m } \langle w _ { m ^ { \prime } } , \operatorname { N o r m } ( v _ { m } ) \rangle \: v _ { m }$ with a learned query $w _ { m ^ { \prime } }$ (Kimi Team, 2026a). In the slot basis of Eq. (12), writers of layer $l _ { i }$ write to their attention sub-layer’s slot $( e _ { l _ { i } }$ , while FFN and embedding slots hold no head writer) and heads of layer $l _ { j }$ read linearly with the softmax weight vector over all stored slots; Tables 1 and 5 instead display the underlying softmax query $q _ { l _ { j } } ^ { m } = w _ { l _ { j } }$ (static) and key $\begin{array} { r } { k _ { l _ { i } } ^ { m } ( M ) = \mathrm { N o r m } ( v _ { l _ { i } } ) } \end{array}$ (from M), normalized by softmax across all stored sub-layer slots. Full AttnRes keeps one slot per sub-layer, about 2l slots at layer l (size 2lD); its block variant sums the outputs within each block into one slot.

**MUDDFormer.** MUDDFormer keeps the embedding and the outputs of all earlier blocks, and gives the next block four inputs—for its queries, keys, values and residual—each a weighted sum of these outputs with weights generated from the latest hidden state (Xiao et al., 2025). The residual input carries dynamic combinations of earlier outputs into the next block output; unfolding these recursions expresses the effective read port $q _ { l _ { j } } ^ { m , s } ( x )$ of input $s \in \{ Q , K , V \}$ as a function of the hidden states along the vector stream x, giving $c _ { i j } \bar { C } _ { i j } = \langle q _ { l _ { j } } ^ { m , s } ( x ) , k _ { l _ { i } } ^ { m } \rangle ^ { \mathring { } } W _ { i } ^ { O } W _ { j } ^ { s }$ . Like DenseFormer, MUDDFormer keeps about l slots at layer l (size lD).

**LoomFormer.** The matrix stream keeps $d _ { k }$ slots of width d that accumulate writes $( \Phi _ { l } = I )$ and passes content without fixed projections: absorbing the per-token RMS scales of Norm $( o _ { i } )$ and MNorm into $k _ { i } ^ { m }$ and $q _ { j } ^ { m , s }$ (and the learned diagonal gain of $\mathrm { N o r m } ( o _ { i } )$ into the writer output $o _ { i } )$ leaves $\bar { C } _ { i j } = I _ { d } .$ Indexing a reader by its head $j \in [ n ]$ and input $s \in \{ Q , K , V \}$ as in Table 1 gives $c _ { i j }   =   \langle q _ { j } ^ { m , s } ( x ) , k _ { i } ^ { m } ( x ) \rangle$ , with both ports generated per head and per token from the vector stream (Theorem 3.1(ii)). With $d _ { k } = 2 n ,$ M has size $d _ { k } d = 2 D$ , plus D for the vector stream.

**Wiring ranks.** Node gating multiplies C entrywise by $g   g ^ { \prime \top }$ —equivalently $C \mapsto \operatorname { d i a g } ( g )   C   \operatorname { d i a g } ( g ^ { \prime } )$ —which cannot increase its rank. All readers of a layer share one read port in DenseFormer, HC/mHC and AttnRes, so C has a single distinct column and rank at most $^ { 1 ; }$ MUDDFormer has one read port per Q/K/V input, hence at most three distinct columns and rank at most 3 (reached once at least three earlier layers are present with linearly independent Q/K/V read ports). In LoomFormer every reader has its own query, so the rank is that of $K Q ^ { \top }$ restricted to the 3n readers of the layer, at most min $\mathbf { \ell } ( w , d _ { k } , 3 n )$ —hence min $( d _ { k } , 3 n )$ once $w \geq \operatorname* { m i n } ( d _ { k } , 3 n )$ , reached whenever the writer keys and reader queries span a common min $( d _ { k } , 3 n )$ -dimensional subspace.

**Linear attention.** The same form describes linear attention along the sequence instead of depth: the writers and readers are one head at different tokens, M is the head’s private state, the ports are its keys and queries, and the update Φ distinguishes the variants—I for linear attention (Katharopoulos et al., 2020), a scalar decay in Mamba2 (Dao & Gu, 2024), a gated diagonal in gated linear attention (Yang et al., 2024a), $I - \beta k ^ { \top }$ k in DeltaNet (Yang et al., 2024b), and $\gamma ( I - \beta k ^ { \top } k )$ in Gated DeltaNet (Yang et al., 2025). It differs from LoomFormer in the axis and the sharing of the state (Section 4).

<!-- page 16 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

## B. Extended Related Work

**Cross-layer connections and residual-stream architectures.** Beyond standard residual connections (He et al., 2016), early work strengthened depth-wise signal propagation through input-dependent gating in Highway Networks (Srivastava et al., 2015) and all-to-all layer aggregation with static or learnable weights in DenseNet (Huang et al., 2017), DLCL (Wang et al., 2019) and DenseFormer (Pagliardini et al., 2024). Subsequent architectures introduced cross-layer attention over earlier layer representations (ElNokrashy et al., 2022; Fang et al., 2023; Wang et al., 2024; Kimi Team, 2026a), multi-stream residual states with dynamic or manifold-constrained mixing in HC and mHC (Zhu et al., 2025; Xie et al., 2025), and decoupled Q/K/V/residual layer aggregation in MUDDFormer (Xiao et al., 2025). As analyzed in Section 4 and Section A.7, all of these mechanisms operate at the granularity of whole layers (or layer × Q/K/V), summing the outputs of all heads within a layer before routing and restricting the within-token wiring rank to 1 (or 3).

The Residual Matrix Transformer (RMT) (Mak & Flanigan, 2025) replaces the vector residual stream entirely with a pure matrix residual stream, which heads and FFNs read and write through static key vectors so the stream can be widened cheaply. LoomFormer instead keeps a dual-stream architecture that separates control on the vector stream from content on the matrix stream: the compact vector stream generates per-token ports cheaply, whereas RMT’s static ports fall under Theorem 3.1(i) and remain node-gated however wide the matrix is made (4D in RMT’s main configuration vs. 2D + D in LoomFormer).

**Cross-layer token attention and KV sharing.** OmniNet (Tay et al., 2021) lets every token attend directly to all tokens across all layers, incurring attention cost quadratic in LT; loom attention instead factors cross-layer routing within each token through a fixed-size matrix memory M in O(LT Dd<sub>k</sub>) time and composes it with standard within-layer horizontal attention. Cross-Layer Attention (CLA) (Brandon et al., 2024) shares key-value caches across adjacent layers to reduce inference memory at a modest quality trade-off, whereas loom attention dynamically composes earlier head outputs into distinct per-head queries, keys and values at every layer (Section 5.3).

**Mechanistic interpretability and intra-layer advances.** Circuit-level interpretability studies (Elhage et al., 2020; Olsson et al., 2022; Conmy et al., 2023; Hanna et al., 2024; Merullo et al., 2024; Ni et al., 2025) show that Transformers implement tasks by composing specific upstream and downstream heads across layers; loom attention turns this implicit, static composition into an explicit, per-token operation over M. Orthogonal intra-layer advances—including improved horizontal attention (Liu et al., 2024; Xiao et al., 2024a; Ye et al., 2024; Leviathan et al., 2024), sequence-axis linear attention and state-space models (Gu & Dao, 2024; Dao & Gu, 2024; Peng et al., 2024; Yang et al., 2024b; 2025), and sparse Mixture-of-Experts FFNs (Fedus et al., 2022; Dai et al., 2024)—modify horizontal mixing or FFN capacity within a layer and can be combined directly with loom attention along depth.

## C. PyTorch-Style Pseudocode for LoomFormer

```python
1 # b = batch size, t = seq len, n = num_heads, d = head_dim
2 # k = num_slots d_k, r = RoPE dim d // 4. M is of shape btkd
3
4 def MRead(M, x, xr):
5 qm = rearrange(xr @ W_r2, 'bt(nk)->btnk')
6 qm = rmsnorm(qm) * sigmoid(rearrange(x @ W_rg, 'btn->btn1'))
7 qm += q_bar * sigmoid(rearrange(x @ W_rg_bar, 'btn->btn1')) # nk,btn1->btnk
8 u = einsum('btnk,btkd->btnd', qm, M)
9 return u
10
11 def MWrite(o, x, xw):
12 km = rearrange(xw @ W_w2, 'bt(nk)->btnk')
13 km = rmsnorm(km) * sigmoid(rearrange(x @ W_wg, 'btn->btn1'))
14 dM = einsum('btnk,btnd->btkd', km, o)
15 return dM
16
17 def MShift(M, x):
18 M_prev = shift_right(M, 1, dim=1) # shift t dim of btkd
```

16

<!-- page 17 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

```python
19 z = sigmoid(rearrange(x @ W_sg, 'btk->btk1'))
20 return z * M_prev + (1 - z) * M
21
22 def loom_mha(x, M, write_M=True):
23 M = RMSNorm(MShift(M, x), dim=(-2, -1))
24 xr, xw = gelu(x @ W_r1), gelu(x @ W_w1)
25 q, k, v, o = [MRead[s](M, x, xr) for s in ['q', 'k', 'v', 'o']]
26 qr = rope(rearrange(x @ W_q, 'bt(nr)->btnr'))
27 kr = rope(rearrange(x @ W_k, 'bt(nr)->btnr'))
28 q, k = concat([qr, q], dim=-1), concat([kr, k], dim=-1)
29
30 allow_mask = tril(ones(T, T), diagonal=-1) # strictly causal mask
31 o = o + MHA(q, k, v, allow_mask=allow_mask)
32 dx = rearrange(o, 'btnd->bt(nd)') @ W_o
33 dM = MWrite(RMSNorm(o, dim=-1), x, xw) if write_M else 0.
34 return dx, dM
35
36 def ffn(x, write_M=True):
37 xw = gelu(x @ W_w1)
38 dx = _ffn_x(x)
39 dM = MWrite(rearrange(_ffn_m(x), 'bt(nd)->btnd'), x, xw) if write_M else 0.
40 return dx, dM
41
42 def loomformer(input_ids):
43 x = embedding(input_ids)
44 o = rearrange(x @ W_e, 'bt(nd)->btnd')
45 M = MWrite(o, x, gelu(x @ W_w1))
46
47 for l in range(L):
48 write_M = l < L - 1
49 # pairs are added element-wise: (x, M) + (dx, dM) = (x + dx, M + dM)
50 (x, M) = (x, M) + loom_mha[l](ln_1[l](x), M, write_M=write_M)
51 (x, M) = (x, M) + ffn[l](ln_2[l](x), write_M=write_M)
52 return unembedding(ln_f(x))
```

## D. Experimental Configurations, Baselines and Benchmarks

## D.1. Model Configurations and Hyperparameters

Table 6 lists the architectural dimensions, peak learning rates and token budgets across all four LoomFormer scales (Sections 5.1 and 5.2). In blocks $0 , \ldots , L - 2 , \mathrm { F F N } _ { x }$ and the $\mathrm { F F N } _ { M }$ write branch $\mathrm { ( F F N } _ { M }$ plus its MWrite module) follow a 2:1 parameter ratio (rounded to multiples of 128 at 2.8B, giving 2.2:1), while block $L - 1$ (last) omits unused writes to $M _ { L }$ and widens $\mathrm { F F N } _ { x }$ to match the baseline Transformer’s total parameter count. Table 7 lists the training hyperparameters for the scaling-law experiments (Section 5.1; weight decay is excluded on normalization scales and biases, including gate biases), while the 2.8B models (Section 5.2) follow Pythia-2.8B (Biderman et al., 2023, Appendix E). To improve training stability at the XL scale, LoomFormer-XL replaces SwiGLU in $\mathrm { F F N } _ { x }$ with Kimi K3’s bounded SiTU-GLU activation (Kimi Team, 2026b) with $\beta _ { 1 } = 4 , \beta _ { 2 } = 7$

**Port and gate initialization.** Each gate projection in Eqs. (5)–(7) includes a learnable bias (omitted in Section 3.2 for brevity) that sets its initial opening. Specifically, the gate projection weights are initialized to zero, with biases setting initial gate values to 0.05 on the dynamic read branch, 0.9 on the static read branch, 0.1 on the write branch, and 0.5 on MShift $( z _ { [ t ] } )$ . For the static queries $\bar { q } \in \mathbb { R } ^ { d _ { k } }$ , value reads v are initialized from $\mathcal { N } ( 0 , 1 / d _ { k } )$ , whereas $\tilde { q } , \tilde { k }$ and $o ^ { m }$ start at $\bar { q } = 0$ so memory-read $\mathrm { Q / K }$ coordinates and the output bypass grow in gradually.

17

<!-- page 18 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

Table 6. Model configurations across scales. $d _ { k } = 2 n ;$ shared port-feature width $D _ { r } = n d _ { k } / 2 = n ^ { 2 } ,$ capped at $D / 4 = 6 4 0 \: \mathrm { a t } \: 2 . 8 \mathrm { B }$

| Model | Params | L D | D/L | n d D<sub>r</sub> | Tfm FFN | FFN<sub>x</sub> (last) | FFN<sub>M</sub> | LR | Tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Medium | 432M | 18 1200 | 66.7 | 16 75 256 | 3200 | 2510 (3904) | 1128 | 3.0e-4 | 7B |
| Large | 778M | 22 1536 | 69.8 | 16 96 256 | 4096 | 3317 (5083) | 1539 | 2.5e-4 | 14B |
| XL | 1.4B | 28 1920 | 68.6 | 20 96 400 | 5120 | 4048 (6277) | 1827 | 2.0e-4 | 26B |
| 2.8B | 2.8B | 32 2560 | 80 | 32 80 640 | 10240 | 7808 (9984) | 2944 | 1.6e-4 | 300B |

Table 7. Training hyperparameters for the scaling-law experiments (Section 5.1).

| Hyperparameter | Value | Hyperparameter | Value |
| --- | --- | --- | --- |
| Optimizer | AdamW | Sequence length | 4096 |
| (β<sub>1</sub>,β<sub>2</sub>) | (0.9,0.95) | Batch size (seqs) | 128 |
| ϵ | 10<sup>-8</sup> | Vocab size | 50,432 |
| Weight decay | 0.1 | Tied embeddings | False |
| Gradient clipping | 1.0 | Linear bias | False |
| LR schedule | Cosine | Normalization | RMSNorm (ϵ = 10<sup>-6</sup>) |
| Min LR ratio | 0.1 | FFN activation | SwiGLU |
| Warmup fraction | 0.01 | Positional emb. | RoPE (θ = 10,000) |
| Dropout | 0.0 | Precision | BF16 |

## D.2. Scaling-Law Baselines

All baselines in Section 5.1 are trained in JAX/MaxText on the identical data order, tokenizer and schedule as LoomFormer:

• **Transformer++**: Llama-style pre-norm Transformer (Touvron et al., 2023) with RoPE (Su et al., 2024), SwiGLU FFN (Shazeer, 2020), RMSNorm and no linear biases.

• **DHC**: Dynamic Hyper-Connections with expansion rate $d _ { k } = 4$ and tanh-bounded dynamic routing following Zhu et al. (2025).

• **mHC**: Manifold-constrained Hyper-Connections $( d _ { k }   =   4 \mathrm { s t r e a m s } )$ projecting the stream-mixing matrix onto the Birkhoff polytope via Sinkhorn normalization (Xie et al., 2025).

• **AttnRes (full)**: Full Attention Residuals aggregating all preceding sub-layer outputs per sub-layer via learned depthwise softmax queries (Kimi Team, 2026a).

• **MUDDFormer**: Multiway Dynamic Dense connections generating per-token depth-aggregation weights separately for the query, key, value and residual inputs of each block (Xiao et al., 2025).

## D.3. Downstream Evaluation Benchmarks

In addition to the benchmarks used by Pythia (Biderman et al., 2023)—LAMBADA (Paperno et al., 2016), PIQA (Bisk et al., 2020), WinoGrande (Sakaguchi et al., 2021), ARC-Easy/Challenge (Clark et al., 2018) and SciQ (Welbl et al., 2017)—we evaluate BoolQ (Clark et al., 2019) and HellaSwag (Zellers et al., 2019) for commonsense reasoning, and RACE-Middle/High (Lai et al., 2017) for reading comprehension (Table 2), all under LM Evaluation Harness (Gao et al., 2023) in zero-shot and five-shot settings.

## E. Details of Complexity Analysis

We derive the forward-FLOP overhead in Table 3 under a matched parameter budget, following the counting convention of DCFormer (Xiao et al., 2024a) and MUDDFormer (Xiao et al., 2025). Let L be the number of layers, T the sequence length, D the model width, n the number of heads, $d = D / n$ the head dimension, and $M \in \mathbb { R } ^ { d _ { k } \times d }$ the matrix memory. We report FLOPs per sequence at two FLOPs per multiply-add, omitting the shared token embedding and unembedding as well as normalization, nonlinearities, biases and other elementwise operations (such as gate activations and the elementwise mixing in MShift). Causal attention has approximately $T ^ { 2 } / 2$ query–key pairs; using a strictly rather than inclusively causal mask changes only the lower-order diagonal terms.

18

<!-- page 19 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

**Parameter budget and tokenwise projections.** The reference Transformer has $4 D ^ { 2 }$ attention and $8 D ^ { 2 }$ FFN weights per layer, for a total of $P _ { 0 } = 1 2 L D ^ { 2 }$ (applying to both a GELU FFN of hidden width $4 D$ and a SwiGLU FFN of hidden width $8 D / 3 )$ . LoomFormer replaces the full Q/K/V projections with smaller RoPE projections and matrix reads, redistributing the saved parameter budget among the port generators, $\mathrm { F F N } _ { x }$ and $\mathrm { F F N } _ { M }$ , as well as the initial value projection $W ^ { E }$ and embedding MWrite (Eq. 10), so that $P _ { \mathrm { L o o m } } = P _ { 0 }$

Every tokenwise dense matrix $W$ costs $2 T | W |$ FLOPs, where $| W |$ is its number of entries. Consequently, to leading order, the combined projection and FFN cost is 2T $P _ { 0 }$ in both models. This includes both stages of the low-rank port generators, all projected gates (including $W _ { g } ^ { S }$ in MShift), $W ^ { E }$ , FFNx and $\mathrm{FFN}_{M};$ a shared projection is counted once. Adding the port-generation or $\mathrm { F F N } _ { M }$ FLOPs without subtracting the $\mathrm { F F N } _ { x }$ FLOPs that paid for their weights would double-count them. Static ports, learned normalization scales and biases give only lower-order corrections to this dense-weight relation, and for the same reason the bottleneck width $D _ { r }$ affects parameter allocation without entering the leading net FLOP overhead.

**Reference Transformer FLOPs.** In addition to the tokenwise projections and FFNs, each layer computes $Q K ^ { \top }$ and attention-weighted values. Each costs 2nd · $T ^ { 2 } / 2 = D T ^ { 2 }$ FLOPs under the causal approximation. The reference forward cost is therefore

$$
F _ { 0 } = 2 T P _ { 0 } + 2 L D T ^ { 2 } = 2 L D T ( 1 2 D + T ) .\tag{13}
$$

**Reading and writing the matrix memory.** A single MRead contracts a $d _ { k }$ -dimensional port with $M \in \mathbb { R } ^ { d _ { k } \times d }$ , at a cost of $2 d _ { k } d$ FLOPs per token. Its dynamic and static ports are summed before this contraction $( \mathrm { E q . }   5 )$ , so they do not require two separate reads. With four reads per head and n heads, the L layers cost

$$
F _ { \mathrm { r e a d } } = 4 \cdot 2 T n d d _ { k } \cdot L = 8 L T D d _ { k } .\tag{14}
$$

For each writing sub-layer, MWrite sums n outer products of shape $d _ { k } \times d ,$ costing $2 T n d d _ { k } = 2 T D d _ { k }$ FLOPs. There are two such sub-layers per block, attention and FF $\mathrm { N } _ { M }$ , except in the last block, which does not write to M. The embedding adds one initial write of the same size (Eq. 10). Hence

$$
\begin{aligned} { F _ { \operatorname { w r i t e } } } & { { } = 4 ( L - 1 ) T D d _ { k } + 2 T D d _ { k } , } \\ { F _ { \operatorname { M } } } & { { } = F _ { \operatorname { r e a d } } + F _ { \operatorname { w r i t e } } = ( 1 2 L - 2 ) T D d _ { k } \simeq 1 2 L T D d _ { k } . } \\ \end{aligned}\tag{15}
$$

The last approximation neglects the small endpoint factor $1 - 1 / ( 6 L )$ multiplying the memory-overhead term. Since M has a fixed size, the read/write cost per layer does not increase with depth.

**Wider query and key.** In the reference Transformer, Q, K and V each have d coordinates per head. LoomFormer concatenates $d / 4$ RoPE coordinates to the d coordinates of Q/K read from M (Eq. 7), while V retains width d. Thus only the $Q K ^ { \top }$ contraction is wider; the attention-weighted value contraction is unchanged. The extra cost is

$$
\Delta F _ { \mathrm { Q K } } = 2 L n { \frac { d } { 4 } } \cdot { \frac { T ^ { 2 } } { 2 } } = { \frac { L D T ^ { 2 } } { 4 } } .\tag{16}
$$

**Total overhead and scaling.** Combining Eqs. (13)–(16) gives

$$
R _ { \Delta \operatorname { F L O P s } } \simeq \frac { 1 2 L T D d _ { k } + L D T ^ { 2 } / 4 } { 2 L D T ( 1 2 D + T ) } = \frac { d _ { k } } { 2 D ( 1 + \rho / 1 2 ) } + \frac { 1 } { 8 ( 1 + 1 2 / \rho ) } ,\tag{17}
$$

where $\rho = T / D$ . With D = nd and $d _ { k } = 2 n$ , we have $d _ { k } / ( 2 D ) = 1 / d ,$ giving the expression in Section 5.4:

$$
\boxed { R _ { \Delta \operatorname { F L O P s } } \simeq \underbrace { \frac { 1 } { d ( 1 + \rho / 1 2 ) } } _ { \operatorname { M R e a d / M W r i t e } } + \underbrace { \frac { 1 } { 8 ( 1 + 1 2 / \rho ) } } _ { \operatorname { w i d e r } \operatorname { Q / K } } . }
$$

19

(18)

At fixed $\rho ,$ the matrix read/write term scales as $1 / d$ while the wider-Q/K term is independent of $d ;$ as $\rho$ grows, the first term falls from $1 / d$ toward 0 and the second rises toward $1 / 8 ,$ with both independent of L at leading order. In Table $3 \; ( d = 7 5 , 9 6 , 9 6$ , 80 and $\rho = T / D = 3 . 4 1 , 2 . 6 7 , 2 . 1 3 , 0 . 8 0$ for Medium, Large, XL and 2.8B, using $T = 4 0 9 6 :$ at

<!-- page 20 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

Medium/Large/XL and T = 2048 at 2.8B; Table 6), Large, XL and 2.8B have integer $d / 4$ and follow Eq. (18) directly (3.13%, 2.77% and 1.95%), whereas Medium rounds its RoPE width $d / 4 = 7 5 / 4 \; \mathrm { t o } \; \lfloor d / 4 \rfloor = 1 8$ in Eq. (16), giving 3.70% instead of Eq. (18)’s unrounded 3.81% (retaining 12L − 2 in Eq. 15 changes every row by $< 0 . 0 1$ percentage points). For a backbone with a different FFN parameter ratio, one can retain the general denominator $2 T P _ { 0 } + 2 L D T ^ { 2 }$ instead of substituting $P _ { 0 } = 1 2 L D ^ { 2 }$ . Wall-clock training and inference speeds are detailed in Section F.

## F. Training and Inference Throughput Details

**Training throughput.** Table 8 reports paired training benchmarks in JAX/MaxText on TPU v5p pods at each model’s training configuration (Sections 5.1 and 5.2), with both Transformer and LoomFormer using Splash attention (1024-token blocks and fused backward). LoomFormer executes custom Pallas kernels (forward and backward) that fuse MShift, MNorm, port normalization and gating, and the four MReads of all n heads into one per-token $( 4 n \times d _ { k } ) \times ( d _ { k } \times d )$ MXU matrix multiplication in VMEM, and likewise fuse MWrite’s n-head outer-product accumulation, avoiding intermediate HBM traffic on M. We compare median XPlane device step time under full activation rematerialization on both sides (Table 3). At 2.8B, LoomFormer uses 512-token cross-entropy chunking to fit v5p-128 HBM whereas Pythia-2.8B does not, making the 2.8B ratio slightly conservative.

Table 8. TPU v5p training step time and relative throughput of LoomFormer (Loom) vs. Transformer (Tfm).

| Model | Pod | T | Batch (seqs) | Tfm (ms) | Loom (ms) | Relative speed |
| --- | --- | --- | --- | --- | --- | --- |
| Medium | v5p-16 | 4096 | 128 | 1224 | 1352 | 90.5% |
| Large | v5p-32 | 4096 | 128 | 962 | 1078 | 89.2% |
| XL | v5p-32 | 4096 | 128 | 1590 | 1723 | 92.3% |
| 2.8B | v5p-128 | 2048 | 1024 | 2988 | 3233 | 92.4% |

**Inference throughput on NVIDIA A800.** Table 9 reports BF16 inference throughput on a single NVIDIA A800-80GB PCIe GPU (torch 2.14 / triton 3.8 / cuDNN 9.24) using trained checkpoints at all four scales, taking the median over five measurements. LoomFormer executes one fused Triton kernel per layer covering the preceding write, the MShift read view, the four MReads, RoPE and cache updates, together with grouped decode GEMVs for the small port projections; prefill uses each model’s fastest SDPA backend (FlashAttention for Transformer; cuDNN SDPA for LoomFormer, whose Q/K head width 1.25d differs from V width d). Prefill measures one uncached forward pass of $B \times T$ tokens, and decode measures single-token generation steps after a $( T - 6 4 )$ -token prompt under torch.comp $\mathtt { i l e \; ( m o d e { = } m a x { - } a u t o t u n e { ~ } ) }$ with CUDA graphs.

Table 9. BF16 inference throughput on one NVIDIA A800-80GB GPU, reported as relative throughput and absolute tokens/s (Loom / Tfm); T = 4096 for Medium, Large and XL, and T = 2048 for 2.8B.

| Model | Prefill bs 1 | Prefill bs 8 | Decode bs 1 Decode bs 16 Decode bs 64 |
| --- | --- | --- | --- |
| Medium | 87.4% (148.9k/170.3k) | 90.4% (166.0k/183.5k) | 91.7% (565/616) 89.2% (2109/2365) 92.3% (2499/2709) |
| Large | 85.3% (95.3k/111.7k) | 86.2% (104.0k/120.6k) | 85.4% (438/513) 87.8% (1916/2183) 84.9% (2311/2721) |
| XL | 83.8% (50.8k/60.6k) | 85.3% (55.3k/64.9k) | 86.9% (272/313) 84.6% (1245/1472) 86.6% (1452/1677) |
| 2.8B | 87.5% (30.4k/34.7k) | 90.5% (33.1k/36.6k) | 86.4% (171/198) 87.6% (1248/1425) 88.9% (1836/2066) |

## G. Extended Ablation Studies

Table 10 reports the complete 18-variant ablation suite at the Medium scale $( L = 1 8 ,   D = 1 2 0 0 ,   n = 1 6 ,   d = 7 5 ,   d _ { k } = 3 2 ,$ $D _ { r } = 2 5 6$ , RoPE dim $\lfloor d / 4 \rfloor = 1 8 , T = 4 0 9 6 , 7 \mathrm { B }$ tokens), extending Table 4. In full LoomFormer, $\mathrm { F F N } _ { M }$ has SwiGLU hidden width 1128, and $\mathrm { F F N } _ { x }$ has width 2510 in blocks 0–16 and 3904 in block 17 (which omits unused writes to $M _ { L } )$ Whenever an ablation removes parameters, we widen FFN<sub>x</sub> uniformly across blocks 0–16 and absorb any embedding or integer-rounding residual in block 17, matching the 432.11M parameter budget within $\pm 1.4  K  \left( < 0.0004\% \right)$ .

**Write-port gating vs. slot selection, and reduction to** $\mathrm { T f m } _ { K , Q } .$ . Fixing write keys $k ^ { m }$ to learned static vectors while keeping their dynamic gates $\sigma ( x W _ { g } ^ { W } )$ preserves nearly the full gain (100.0% for LoomMHA, 99.3% for $\mathrm { F F N } _ { M }$ , and 95.6% across all three write sites including the embedding). Dropping the gate on static $k ^ { m }$ , however, hurts LoomMHA over three times as much as $\mathrm{FFN}_{M} \; (100.0\% \rightarrow 91.1\% \;  vs.  \; 99.3\% \rightarrow 96.6\%)$ : LoomMHA normalizes its head outputs $\mathrm { N o r m } ( o _ { i } )$ before writing (Eq. 8), leaving the gate as the sole control of write magnitude (and whether a head is on), whereas $\mathrm { F F N } _ { M }$

20

<!-- page 21 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

Table 10. Complete Medium-scale (432M) ablation suite of LoomFormer under a matched parameter budget (432.11M±1.4K). †Included in Table 4.

<table><tr><td>Variant</td><td>Specification (vs. full LoomFormer)</td><td>Loss↓</td><td>∆Loss</td><td>Gain</td></tr><tr><td> $( \mathrm { f u l l } ) ^ { \dagger }$  LoomFormer</td><td></td><td>2.266</td><td></td><td>100.0%</td></tr><tr><td>Transformer++ (baseline)†</td><td>Standard MHA + SwiGLU FFN, no matrix stream M</td><td>2.443</td><td>+0.177</td><td>0.0%</td></tr><tr><td colspan="5">Dynamic rewiring &amp; write-port gating (Theorem 3.1)</td></tr><tr><td> $k ^ { m } .$  Static LoomMHA , gated</td><td> $\sigma ( x W _ { g } ^ { W } )$  kept in LoomMHA; write gate  $k ^ { m }$  Static</td><td>2.266</td><td>0.000</td><td>100.0%</td></tr><tr><td>Static LoomMHA  $k ^ { m }$ </td><td>Static  $k ^ { m }$  in LoomMHA; write gate dropped  $( \equiv 1 )$ </td><td>2.282</td><td>+0.016</td><td>91.1%</td></tr><tr><td>, gated  $\mathrm { F F N } _ { M } \; k ^ { m }$  Static</td><td> $\widehat { \sigma ( x W _ { g } ^ { \hat { W } } ) }$  Static  $k ^ { m }$  kept  $\mathrm{FFN}_{M};$  write gate in</td><td>2.268</td><td>+0.001</td><td>99.3%</td></tr><tr><td>Static  $\mathrm { F F N } _ { M } \; k ^ { m }$ </td><td>Static in  $k ^ { m }$   $\mathrm{FFN}_{M};$  write gate dropped  $( \equiv 1 )$ </td><td>2.272</td><td>+0.006</td><td>96.6%</td></tr><tr><td>gated  $( k ^ { m } ) ,$  Static write keys</td><td>Static  $k ^ { m }$   $\hat { \mathrm { F F N } _ { M } } ;$  in emb., LoomMHA, write gates kept</td><td>2.274</td><td>+0.008</td><td>95.6%</td></tr><tr><td>Static write keys  $( k ^ { m } ) ^ { \dagger }$ </td><td> $k ^ { m }$   $\mathrm{FFN}_{M};$  in emb., LoomMHA, Static write gates dropped</td><td>2.283</td><td>+0.017</td><td>90.3%</td></tr><tr><td> $( q ^ { m } ) ^ { \dagger }$  Static read queries</td><td>Only static ī in  $\mathrm { M R e a d } ^ { Q , K , V , O } ;$  all read gates dropped</td><td>2.310</td><td>+0.043</td><td>75.4%</td></tr><tr><td>, gated  $( q ^ { m } , k ^ { m } ) ,$  Static read &amp; write</td><td>(no read gates), all Static  $k ^ { m }$   $q ^ { m }$  ; write gates kept</td><td>2.327</td><td>+0.060</td><td>65.8%</td></tr><tr><td> $( q ^ { m } , k ^ { m } ) ^ { \dagger }$  Static read &amp; write</td><td>, all Static  $q ^ { m }$   $k ^ { m } ;$  port gates dropped  $( \mathrm { T f m } _ { K , Q } + \mathrm { M S h i f t } )$ </td><td>2.344</td><td>+0.078</td><td>56.0%</td></tr><tr><td> $\dot { ( q ^ { m } , k ^ { m } ) } , \mathrm { - M S h i f t }$  Static read &amp; write</td><td>port gates, MŠhift dropped (Prop. 3.1(i)) Static , all  $k ^ { m } ;$   $q ^ { m }$ </td><td>2.348</td><td>+0.082</td><td>53.6%</td></tr><tr><td colspan="5">Port normalization &amp; gating (Eqs. 5–6)</td></tr><tr><td>dynamic port gates†</td><td> $\sigma ( x W _ { g } ^ { R } ) , \sigma ( x W _ { g } ^ { W } ) \equiv 1 ; \sigma ( x W _ { \bar { g } } ^ { R } )$  and MShift kept</td><td>2.277</td><td>+0.011</td><td>93.7%</td></tr><tr><td> $\mathrm { N o r m } ^ { \dagger }$  dynamic port</td><td>, MNorm kept  $q ^ { m } , k ^ { m } ;$   $( o _ { i } )$  Drop Norm on dynamic ; Norm</td><td>2.289</td><td>+0.023</td><td>87.1%</td></tr><tr><td>dynamic port gates &amp; Norm†</td><td> $\sigma ( x \dot { W _ { g } ^ { R } } ) , \sigma ( \dot { x W _ { g } ^ { W } } )$  Drop both and port Norm</td><td>2.283</td><td>+0.017</td><td>90.3%</td></tr><tr><td colspan="5">Sub-layer components (Section 3.2)</td></tr><tr><td>write†  $\mathrm { F F N } _ { M }$ </td><td> $\mathrm { F F N } _ { M }$   $( \mathrm { E q . }   9 )$  Remove and its MWrite</td><td>2.300</td><td>+0.034</td><td>80.7%</td></tr><tr><td> $o ^ { m \dagger }$  output bypass</td><td> $\mathrm { M R e a d } ^ { O } ;$  Drop ; restore self-attention to current token (Eq. 8)</td><td>2.299</td><td>+0.032</td><td>81.8%</td></tr><tr><td> $W ^ { Q , K \dagger }$  vector-stream RoPE</td><td> $W ^ { Q , K }$   $\tilde { q } , \tilde { k }   ( \mathrm { E q . }   7 )$  Drop ; RoPE on 18 of 75 dims of</td><td>2.286</td><td>+0.020</td><td>88.7%</td></tr><tr><td> $- \mathrm { M S h i f t ^ { \dagger } }$ </td><td>Set  $z _ { [ t ] } \equiv 0$  in Eq. (7)</td><td>2.277</td><td>+0.010</td><td>94.2%</td></tr><tr><td> $\mathrm { N o r m } ( o _ { i } )$  — LoomMHA write</td><td> $( \mathrm { E q . ~ 8 } )$  Drop Norm on in LoomMHA writes  $o _ { i }$ </td><td>2.268</td><td>+0.002</td><td>98.8%</td></tr></table>

writes unnormalized slices (Eq. 9) whose norms can still modulate write strength per token. When both $q ^ { m }$ and $k ^ { m }$ are static, retaining dynamic write gates gives a node-gated circuit on fixed slots (65.8%), dropping those gates leaves Tfm<sub>K</sub>,Q with MShift (56.0%), and further removing MShift yields the exact $\mathrm { T f m } _ { K , Q }$ of Theorem 3.1(i) (53.6%)—which has no token-dependent port or shift gates on M, yet still outperforms all layer-wise dynamic baselines at the same scale (Figure 5, where MUDDFormer reaches 48.5% of LoomFormer’s gain).

**Attention write-value normalization.** Removing Norm(o<sub>i</sub>) from LoomMHA writes (Eq. 8) loses only 1.2% of the gain (98.8% retained), as whole-matrix pre-normalization MNorm before reads (Eq. 7) already bounds the scale of M.

## H. Mechanistic Analysis Details and Visualizations

**Evaluation protocol and linear attribution.** The statistics in Section 5.3 and this section are computed on held-out Pile validation sequences of length $T   =   2 0 4 8$ at the 300B-token checkpoints of Pythia-2.8B, mHC-Pythia-2.8B and LoomPythia-2.8B (evaluating token positions $t \geq 6 4$ to avoid sequence-boundary transients). Because MShift and MNorm act linearly on the accumulated writes at each layer (Equation (7)), absorbing the per-slot shift gate $z _ { [ t ] } \: ( \mathrm { o r } \: 1 - z _ { [ t ] } )$ and the scalar MNorm factor into $c _ { i j } = \langle q _ { j } ^ { m } , k _ { i } ^ { m } \rangle$ decomposes every MRead output as $\begin{array} { r } { u _ { j } = \sum _ { i } c _ { i j } o _ { i } } \end{array}$ over all earlier writers (embedding, LoomMHA and FFN<sub>M</sub> writes at t and $t - 1 ;   0 . 4 \%$ median RMS difference from BF16 execution across layers). Projecting each term onto $u _ { j }$ gives the additive writer attribution

$$
a _ { i j } = \frac { c _ { i j } \langle o _ { i } , u _ { j } \rangle } { \| u _ { j } \| ^ { 2 } } , \qquad \textstyle \sum _ { i } a _ { i j } = 1 .\tag{19}
$$

**Definitions for Figure 6.** In Figure 6(a), to isolate dynamic port rewiring from both static virtual weights (Theorem 3.1(i)) and per-token scalar gate or writer-norm fluctuations, let $\hat { q } _ { j } ^ { m } ( t )   =   q _ { j } ^ { m } ( t ) / \| q _ { j } ^ { m } ( t ) \| _ { 2 }$ and $\hat { k } _ { i } ^ { m } ( t )   =   k _ { i } ^ { m } ( t ) / \| k _ { i } ^ { m } ( t ) \| _ { 2 }   \in$ $\mathbb { S } ^ { d _ { k } - 1 }$ denote the unit port directions of reader $j$ and same-token writer $i ,$ weighted by each writer’s corpus RMS norm $\bar { \omega } _ { i } = \sqrt { \mathbb { E } _ { t } [ \| o _ { i } ( t ) \| ^ { 2 } ] } { : } \widehat { C } _ { i j } ( t ) = \bar { \omega } _ { i } \langle \hat { q } _ { j } ^ { m } ( t ) , \hat { k } _ { i } ^ { m } ( t ) \rangle$ . We center across tokens, $\Delta \widehat { \widehat { C } } ( t ) = \widehat { \widehat { C } } ( t ) - \mathbb { E } _ { t ^ { \prime } } [ \widehat { C } ( t ^ { \prime } ) ]$ , and report the energy effective rank exp $\textstyle { \mathrm { b } } { \big ( } { - \sum _ { k } p _ { k } \operatorname { l o g } p _ { k } } { \big ) }$ with $\begin{array} { r } { p _ { k }   =   \sigma _ { k } ^ { 2 } / \sum _ { k ^ { \prime } } \sigma _ { k ^ { \prime } } ^ { 2 } } \end{array}$ over the $3 n   =   9 6 \; \mathrm { Q / K / V }$ readers of layer l (or the $n = 3 2$ readers of a single $\mathbf { Q } ,$ K or V input); under fixed-port $\mathrm { T f m } _ { K , Q }$ with arbitrary positive scalar gates, $\Delta \widehat { C } ( t ) \equiv 0$ . In

21

<!-- page 22 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

Figure 6(b), for each read port $\begin{array} { r } { q _ { j } ^ { m } ( t ) \in \mathbb { R } ^ { d _ { k } } \left( \tilde { q } , \tilde { k } , v , o ^ { m } \right) } \end{array}$ and write port $\begin{array} { r } { k _ { i } ^ { m } ( t ) \in \mathbb { R } ^ { d _ { k } } } \end{array}$ (LoomMHA and $\mathrm { F F N } _ { M } )$ , we average $1 - \operatorname { c o s } ( p ( t _ { 1 } ) , p ( t _ { 2 } ) )$ over distinct token pairs $t _ { 1 } \neq t _ { 2 } \; ( t \geq 6 4 )$ and over the $n = 3 2$ heads of layer l (embedding write ports remain near-static at 0.005, providing a sanity check). In Figure 6(c), following Xiao et al. (2025), a head is on at query position t if arg $\operatorname* { m a x } _ { s } \alpha _ { t s }$ falls neither in {0, 1} nor on a sink token (<|endoftext $$\vert > , `` \because '' , `` \cdots , '' \because \backslash \backslash \backslash \backslash '' , `` \backslash \backslash \backslash \backslash \backslash \backslash '' \rangle ; $$ in layers 17–32, first-two-token sink mass drops from 26.4% (Pythia) and 27.2% (mHC) to 1.5% in LoomPythia while content-token attention mass rises from 22.7% and 31.2% to 65.4%, and masking the self-diagonal $s = t$ in Pythia and mHC to match LoomMHA’s strict causal mask (Equation (8)) leaves the gap unchanged (74.4% vs. 21.2% and 15.7%). In Figure 6(d), for each head h in layers 3–32 and each pair among its four ports $\{ \tilde { q } _ { h } , \tilde { k } _ { h } , v _ { h } , o _ { h } ^ { m } \}$ , we compute from tokenaveraged positive attributions $\bar { a } _ { i j } ^ { + } = \mathbb { E } _ { t } [ \operatorname* { m a x } ( a _ { i j } ( t ) , 0 ) ]$ over same-token writers: (i) Pearson correlation after summing attributions by writer group (Emb, each earlier LoomMHA layer, each earlier $\mathrm { F F N } _ { M }$ layer), (ii) Pearson correlation across individual head/module writers, and (iii) $| S _ { A } \cap S _ { B } | / 3 2$ for the top-32 writer sets $S _ { A } , S _ { B } ,$ reporting the median across heads (in slot space $\mathbb { R } ^ { d _ { k } }$ , same-token cosine similarity between ports of the same head averages 0.01–0.07 outside Q–K and 0.27 on Q–K).

**Writer composition and depth-to-depth routing.** Averaged over layers 2–32 in LoomPythia-2.8B, positive attribution mass max $( a _ { i j } , 0 )$ is shared across LoomMHA writes (53.4%–62.5% across $\tilde { q } , \tilde { k } , v , o ^ { m } )$ , FFN<sub>M</sub> writes (27.5%–35.2%) and embedding writes (10.0%–11.7%), while previous-token MShift reads account for 13.7%–15.2% of $\textstyle \sum _ { i } | a _ { i j } |$ (Figure 8(a)). $\mathrm { F F N } _ { M }$ reaches its largest share (35.2%) on value reads $v ,$ consistent with $\mathrm { F F N } _ { M }$ serving as the sole post-embedding channel from the vector stream into M (Equation (9)) and with its large effect in Table 4. Plotted with writer sources on the vertical axis (bottom to top) and reader layers on the horizontal axis, the column-max normalized routing map (Figure 8(b)) combines local relay along the diagonal with persistent horizontal bands from hub writer sources—most prominently the embedding at early layers and writer layers 13–14 and 19 across all subsequent layers.

**Adjacent-token rewiring on an in-context prompt.** Figure 9 shows the head-writer attribution submatrix $\left[ a _ { i j } \right]$ at hub layer 19 (selecting the top 6 readers per MRead arm and top 18 earlier LoomMHA head writers by peak $| a _ { i j } |$ across the two tokens) on "France: Paris\nJapan: Tokyo\nChina: Beijing\nSpain: Madrid\nGermany:", where LoomPythia-2.8B predicts " Berlin" top-1 with probability 0.65. At "Germany" (Figure 9(a)), writers route selectively to distinct MRead arms—L13.H11 feeds key reads (˜k: K14, K23, K25) without feeding value or bypass reads, whereas L13.H12 and L08.H25 feed value and bypass reads (V10, O09, O22)—forming a non-rectangular within-token pattern (Theorem 2.3) that prepares "Germany"’s horizontal keys, values and residual state. At the immediately following delimiter ":" (Figure 9(b)), LoomMHA’s strict causal mask $( s < t ,$ Equation (8)) means that only q˜ (attending back to "Germany" and the few-shot capitals) and $o ^ { m }$ affect the " Berlin" prediction, whereas ˜k and v at $\quad \because \quad$ serve only future tokens. Correspondingly, L13.H11 and L13.H12 switch off on layer 19, while layer 19 rewires its causal query reads q˜ onto L18.H25, L15.H32 and L14.H01 (Q06, Q07, Q17, Q22) and its bypass outputs $o ^ { m }$ onto L14.H06, L16.H29 and L14.H19 (O16, O21); causally freezing layer 19’s dynamic read ports at $\quad \because$ to those at "Germany" alone drops $P( \text{" }  Ber1i n  )$ from 0.65 to 0.54 (0.41 when frozen across all layers).

![Chart block](doc:734c5bd/tier:basic/page:22/block:5)

![Chart block](doc:734c5bd/tier:basic/page:22/block:6)

Figure 8. Matrix-stream routing in LoomPythia-2.8B across Pile validation sequences. (a) Share of positive attribution from Embedding, LoomMHA and $\mathrm { F F N } _ { M }$ writes, and share of $| a _ { i j } |$ from previous-token MShift $\bar { ( } M _ { [ t - 1 ] } ) .$ . (b) Column-max normalized positive attribution from each writer source (vertical, bottom to top) to each reader layer l (horizontal; gray upper triangle: causally unreachable).

22

<!-- page 23 of 23 -->

Rewirable Circuits in Transformers with Loom Attention

(a) Layer 19 Wiring at "Germany" (t 1)

![Chart block](doc:734c5bd/tier:basic/page:23/block:3)

(b) Layer 19 Rewired at ":" $\left[ \begin{matrix} { ( t _ { \pmb { \mathscr { h } } } ) } \\ { \in O [ \bar { h } ] } \end{matrix} \right]$ future-token cach

![Chart block](doc:734c5bd/tier:basic/page:23/block:5)

Figure 9. Attribution submatrix $\left[ a _ { i j } \right]$ (rows: top-18 earlier head writers Ll.Hh; columns: top-6 layer-19 MRead outputs per arm $\tilde { q } , \tilde { k } , v , o ^ { m } )$ of LoomPythia-2.8B at two consecutive tokens of an in-context prompt (next-token top-1: " $\mathtt { B e r l i n  , } p = 0 . 6 5 ) ,$ , showing non-rectangular within-token wiring at $\text{" } \mathtt{G}  \text{e} \mathtt{r}  \text{m} \mathtt{a} \mathtt{n} \mathtt{y}  \text{" } \mathtt{(a)}$ and per-token rewiring onto causal $\tilde { q }$ and $o ^ { m }$ pathways at $n :$ (b; hatched $\tilde { k } , v$ columns affect only future tokens under $s < t )$

23