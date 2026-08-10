# Addressable Memory for Video World Models

Xindi Wu<sup>1</sup> <sup>2</sup> Sven Elflein<sup>1</sup> <sup>3</sup> <sup>4</sup> James Lucas<sup>1</sup> Olga Russakovsky<sup>2</sup> Laura Leal-Taixé<sup>1</sup> Despoina Paschalidou<sup>1</sup> Jonathan Lorraine<sup>1</sup> Aljosa Osep<sup>1</sup>

<sup>1</sup>NVIDIA <sup>2</sup>Princeton University <sup>3</sup>University of Toronto <sup>4</sup>Vector Institute https://research.nvidia.com/labs/sil/projects/WorldTrace/

## Abstract

We study visual persistence in interactive video world models. These models rely on a Key-Value (KV) cache as a growing visual memory to carry forward previously generated frames. However, we find that models can no longer reliably address stored content once rollouts extend beyond the training horizon, because temporal Rotary Positional Embeddings (RoPE) ofsets then fall outside the range seen during training and the model struggles to retrieve the relevant visual information through attention. Moreover, naively compressing the cache in the RoPE-rotated space corrupt memory by averaging together incompatible positional phases. To address this, we propose , a training-free memory framework for long-horizon visual persistence. W T keeps compressed memory addressable by assigning each summary slot a distinct, in-distribution virtual position. Within this addressable cache, we study two memory compression approaches: W T F compresses history for temporal coherence, while stores verbatim scene traces at detected transitions for episodic recall. We further introduce L B , a benchmark evaluating whether a compressed cache can reconstruct a previously visited scene after a long detour. W T F improves temporal consistency by +15.5%, and W T L improves episodic recall by +19.5% on LoopBench, extending visually persistent generation without retraining.

## 1 Introduction

Memory is what makes a generated world persistent. A world model should not only predict what comes next, but also preserve what has already been seen. World models aim to simulate worlds we can explore [29, 30, 25], and recent autoregressive video world models [82, 32, 20] pursue this vision by generating video scenes chunk-by-chunk, with each chunk attending over a Key-Value (KV) cache of prior context. These models promise interactive applications such as next-generation game engines [84, 20, 32] and closed-loop robot simulators [1, 3], in which users can move freely through a visual world and revisit prior locations. Such revisits demand visual persistence, meaning that when an agent returns, the generated results should be consistent with the scene’s original appearance, not a plausible-looking alternative.

In practice, visual persistence degrades rapidly once generation exceeds the training context length. A natural way to address this issue is to compress the linearly growing KV cache into a fixed-size memory [115, 53, 9, 42] equal to the size of the context seen during training. However, we find that the bottleneck is not simply whether past content is stored in the cache, but whether it remains addressable and how it is compressed. As the generation horizon extends beyond the training context length, the positional encoding these models rely on (e.g., RoPE [78]) is queried at position indices outside its training-bounded range, so the model needs to attend over relative distances that are out-of-distribution (OOD), degrading attention-based memory retrieval. In other words, even if past memories are stored in the KV cache, the model cannot reliably retrieve them. A summary placed at such an ofset can therefore become hard to reach. This reveals that long-horizon failure is fundamentally a problem of addressability, since no memory scheme can improve visual persistence if past observations cannot be reliably accessed once they fall outside the context window.

WorldTrace. To tackle this problem, we propose <sup>W</sup>orld<sup>T</sup>race, a training-free approach for addressable compressed memory over long horizons. We assign each memory entry a fixed, in-distribution temporal position relative to the current frame, keeping past observations within the temporal positional encoding range the model was trained on so that distant tokens remain retrievable. The cache is organized into a recent window and summary slots for the distant horizon. Within this addressable cache, the remaining challenge is what the world model should choose to remember. We view this as a structured projection of a growing visual history into a fixed set of retrievable traces. We propose two complementary memory writers: WorldTrace-Field which preserves the coarse field of past information in rotation-invariant space for temporal coherence, and WorldTrace-Landmark which preserves sparse scene-entry traces for long-horizon recall.

![](images/a6370e0f9334e4c8aff27ec1f0a6a6d4e3d0f0294b6dcc6710051804b9b57aec.jpg)  
Figure 1: Addressable memory maintains long-horizon visual persistence. (Left) Loop topology A→B→C→D→A. (Rig<sup>h</sup>t) Representative frames on Matrix-Game-2 along the path, covering the initial scene A (blue border), waypoints B, C, and D, and the return to A. Sliding window (top) mismatches the reference appearance at the return (red border). Our WorldTrace with frozen landmark keys (green border) matches scene A, confirming addressable long-horizon recall.

Our experiments show that addressable memory extends the efective memory horizon. To evaluate episodic recall under controlled scene revisits, we propose LoopBench, a memory benchmark that tests whether the compressed KV cache can retrieve a previously visited scene after a long detour. While the sliding-window baseline begins to forget scene structure after only a few seconds and loses structural fidelity when revisiting the same location, our WorldTrace maintains consistent scene reconstruction over long rollouts and recovers the original scene appearance (Fig. 1).

## Our contributions are:

1. We identify two coupled challenges for long-horizon generation in autoregressive video world models, w<sup>h</sup>ere memory is placed and w<sup>h</sup>at it stores. Out-of-distribution temporal ofsets make cached tokens hard to address, and RoPE phase cancellation makes naively compressed summaries uninformative (Sec. 2).

2. We propose WorldTrace, a training-free approach that keeps compressed memory both addressable and useful without retraining. It assigns each summary an in-distribution virtual position (Sec. 3.1) and organizes the growing visual history into a fixed set of retrievable traces, realized by two complementary operators: WorldTrace-Field for temporal coherence (Sec. 3.3) and WorldTrace-Landmark for episodic recall (Sec. 3.4).

3. We introduce LoopBench, a memory benchmark of controlled scene revisits that measures whether a compressed cache can recover a previously visited scene (Sec. 4). We show that WorldTrace extends the efective memory horizon over the default sliding-window cache, with WorldTrace-Field improving temporal consistency by +15.5% on long rollouts and WorldTrace-Landmark improving episodic recall by +19.5% on LoopBench ABA revisits, both without retraining.

## 2 Rethinking Memory in Video World Models

Autoregressive video world models retain prior context as cached keys and values during generation. Longhorizon memory therefore depends both on whether relevant past context remains inside the cache and on whether the model can efectively attend to it during future generation. Two coupled bottlenecks arise. First, cached tokens need to remain addressable (Sec. 2.1), because attending to a distant token requires a temporal RoPE ofset that may lie outside the training distribution, which leaves the token unreadable even when stored. Second, compressed summaries need to remain informative (Sec. 2.2), because naively compressing old keys lets mismatched RoPE phases cancel out the signal the summary is supposed to carry.

## 2.1 Position Determines Whether Memory Is Addressable

A cached token helps only if the current query can still attend to it. Because a token’s temporal RoPE ofset from the current query grows with how far back it lies, entries eventually carry ofsets beyond the range seen during training, at which point the model can no longer read them even though they still sit in the cache. We summarize the notations in App. A.

Concretely, during training the query only attends over a bounded range of relative ofsets $\delta _ { q , k }$ between the current frame and each cached frame:

$$
0 \leq \delta _ { q , k } = q - k \leq \Delta t _ { \mathrm { t r a i n } } ,
$$

where $q$ is the absolute position of the current query frame, � that of a cached key frame, with $k \leq q$ under causal autoregressive attention, and $\Delta t _ { \mathrm { t r a i n } }$ the largest ofset seen during training.

For temporal RoPE pair $f$ at angular frequency $\theta _ { f }$ , its contribution $a _ { q , k } ^ { f }$ to the query-key attention score is:

$$
\begin{array} { r } { a _ { q , k } ^ { f } = \operatorname { R e } \left( A _ { q , k } ^ { f } e ^ { i \theta _ { f } \delta _ { q , k } } \right) , } \end{array}
$$

where $\operatorname { R e } ( \cdot )$ is the real part and $A _ { q , k } ^ { f }$ is the content-dependent inner product in canonical coordinates between the query at $q$ and the key at �. During long-horizon inference, these ofsets grow ever larger, and $\delta _ { q , k }$ can exceed $\Delta t _ { \mathrm { t r a i n } }$ . The afected RoPE components then take on rotation angles the model never saw during training. This efect is frequency-dependent, with the fastest temporal components degrading into positional noise while slower ones stay close to their trained range and keep carrying usable signal [67, 2]. We quantify this per frequency in App. C.

Several concurrent methods investigate related positional issues. Infinity-RoPE [107] proposes Block-relative RoPE which reindexes each cached frame relative to the current AR chunk so that $\delta _ { q , k }$ always stays within the trained range, keeping generation in-distribution without a fixed memory budget. It targets infinite video generation rather than memory compression. Applied to a compressed cache, however, this reindexing caps every ofset at $\Delta t _ { \mathrm { t r a i n } }$ , so distinct summary slots collapse onto the same position and become indistinguishable (Rem. 1). MemRoPE [42] compresses history into two EMA memory tokens, a long-term and a short-term stream, which avoids this collapse simply because only two summaries need distinct positions. The design is fixed to two streams and does not extend to an �-slot cache, where capped Block-relative positions would again collide. Unlike these, our WorldTrace targets an �-slot compressed cache directly by assigning every slot a distinct, in-distribution slot-rank position, so an arbitrary number of summaries stay individually addressable at any horizon. We provide a detailed discussion of related work in App. D.

## 2.2 Content Compression Determines Whether Memory Is Informative

Even a summary placed at an in-distribution temporal position is only worth reading if compression preserved what it encodes. A common way to compress cached keys is averaging [6, 71]. With RoPE-rotated keys, at frequency � this corresponds to:

$$
\bar { K } _ { \mathrm { n a i v e } } ^ { f } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } R ( \theta _ { f } t _ { m } ) { K } _ { m } ^ { f } ,
$$

where $K _ { m } ^ { f }$ is the canonical (unrotated) key of frame � at frequency � and $R ( \theta _ { f } t _ { m } )$ is its RoPE rotation. Each frame is rotated by a diferent angle $\theta _ { f } t _ { m }$ before averaging, so $\bar { K } _ { \mathrm { n a i v e } } ^ { f }$ sums vectors that point in diferent directions. If the frames come from nearby timestamps, their rotation angles are close, so averaging largely preserves the signal. If they come from distant timestamps, their angles can point in opposite directions and partially cancel, weakening that frequency’s contribution to the summary regardless of what it encoded. Naive averaging in RoPE-rotated space thus corrupts the compressed memory.

The two failures are coupled. Most prior methods fix only one of the two, with one line of work adjusting w<sup>h</sup>ere memories are placed, reparameterizing or capping ofsets (e.g. Block-relative [107]), while another decides w<sup>h</sup>at each summary stores, whether by smoothing past keys with dual-rate EMA [42], merging them by temporal averaging [6, 71], or evicting entries [115, 53, 9]. Addressable positions cannot help if compression cancels the signal a summary should carry, and an informative summary cannot help if it sits at an out-of-distribution position. Thus we need to keep compressed memory both addressable and informative once generation exceeds the training window. Visual persistence in a video world model demands exactly this, because an agent that explores and later revisits a location needs the cache to keep those distant scenes readable across long detours (addressability) and to retain enough of their content to regenerate them faithfully (informativeness). Our WorldTrace is built to satisfy both, which is what lets a generated world stay consistent over long, interactive rollouts.

![](images/b577a31915da93cb9c1f038cd536e1cf7510df53b9f2436fddabe893104a9a99.jpg)  
Figure 2: W T overview. <sup>W</sup>orld<sup>T</sup>race partitions the context window into a compressed summary cache � for the distant past and a verbatim recent window ℛ (top). Each slot occupies one latent-frame position in the cache, with recent slots holding one verbatim frame and summary slots holding compressed history. WorldTrace-Field (bottom left): Keys are unrotated and averaged in canonical space, then re-rotated at each slot’s virtual position. WorldTrace-Landmark (bottom right): Scene-entry landmarks store frozen canonical keys and are re-rotated to the current virtual position.

## 3 WorldTrace

We propose WorldTrace, a training-free approach for addressable memory in autoregressive video world models. WorldTrace maintains a bounded trace of distant history in fixed summary slots via canonical-space key assignment, while keeping each slot addressable (Sec. 3.1). We formulate this cached memory compression for visual persistence as structured sparse attention (Sec. 3.2, App. B). Its two variants, WorldTrace-Field (Sec. 3.3) and WorldTrace-Landmark (Sec. 3.4), target temporal coherence and episodic recall, respectively.

## 3.1 Virtual Position Assignment: WorldTrace Slot Indexing

Cache structure. As shown in Fig. 2, we partition the local attention window $L _ { \mathrm { a t t n } }$ into a recent window ℛ of $N _ { r }$ verbatim latent frames and a summary cache � of $N _ { s }$ slots storing compressed history, so that $N _ { s } + N _ { r } = L _ { \mathrm { a t t n } }$

Given this cache structure, the remaining question is how to position each summary slot relative to the query. A memory position scheme assigns a virtual position to summary slot � for a query at timestamp �, and to be usable by the attention mechanism, it should satisfy four properties: (i) linearity in $q$ and � for a simple predictable mapping, (ii) in-distribution positions so the model can attend, (iii) horizon-stability so virtual positions stay in-distribution at any generation length, and (iv) distinct positions for each slot so attention can tell them apart. Summary slot positions must stay within the range seen during training:

$$
t _ { \mathrm { m i n } } ^ { v } = \mathrm { m a x } \big ( 0 , q - \Delta t _ { \mathrm { t r a i n } } \big ) , \quad t _ { \mathrm { m a x } } ^ { v } = q - N _ { r } .\tag{1}
$$

Within those bounds, each summary slot takes the position fixed by its rank.

Definition 1 (WorldTrace Slot Indexing). Given current query position $q ,$ local attention window $L _ { \mathrm { a t t n } }$ $( = N _ { s } + N _ { r } )$ , and summary cache size $N _ { s }$ , the virtual position of summary slot � is:

$$
t _ { s } ^ { v } \ = \ q - ( L _ { \mathrm { a t t n } } - 1 - s ) , \quad s = 0 , \ldots , N _ { s } - 1 .\tag{2}
$$

The assignment covers the in-distribution range with fixed slot-rank ofsets relative to the current frame. By construction, $t _ { s } ^ { v } \in [ t _ { \operatorname* { m i n } } ^ { v } , t _ { \operatorname* { m a x } } ^ { v } ]$ at any generation length, ofsets depend only on slot rank and $q ,$ and importantly, not the absolute horizon �.

Remark 1 (Block-relative collapse). Bloc<sup>k</sup>-relative positions [107] cap all KV-cac<sup>h</sup>e ofsets at $\Delta t _ { \mathrm { t r a i n } }$ (Sec. 2.1). As <sub>g</sub>eneration <sub>g</sub>rows<sub>,</sub> summar<sub>y</sub> slots s<sub>p</sub>annin<sub>g</sub> histor<sub>y</sub> be<sub>y</sub>ond $\Delta t _ { \mathrm { t r a i n } }$ <sub>s</sub>t<sub>eps</sub> b<sub>ac</sub>k <sub>a</sub>ll <sub>s</sub>h<sub>are</sub> th<sub>e same m</sub>i<sub>n</sub>i<sub>mum v</sub>i<sub>r</sub>t<sub>ua</sub>l <sub>pos</sub>iti<sub>o</sub>n $t _ { \mathrm { m i n } } ^ { v } ,$ makin<sub>g</sub> the com<sub>p</sub>ressed slots <sub>p</sub>ositionall<sub>y</sub> indistin<sub>g</sub>uishable. Our WorldTrace slot indexin<sub>g</sub> (Def. 1) <sub>avo</sub>id<sub>s</sub> thi<sub>s</sub> b<sub>y anc</sub>h<sub>or</sub>i<sub>ng eac</sub>h <sub>v</sub>i<sub>r</sub>t<sub>ua</sub>l <sub>pos</sub>iti<sub>on</sub> t<sub>o s</sub>l<sub>o</sub>t <sub>ran</sub>k th<sub>roug</sub>h $t _ { s } ^ { v } = q - ( L _ { \mathrm { a t t n } } - 1 - s ) .$ , whose ofset $\left( L _ { \mathrm { a t t n } } - 1 - s \right)$ is fixed by slot rank alone and therefore keeps all $N _ { s }$ slots distinct at any generation horizon �.

Why canonical keys. Each slot’s virtual position $t _ { s } ^ { v }$ is set by slot rank (Def. 1), independently of the original timestamps $\{ t _ { m } \}$ of its source frames, so placing compressed content at $t _ { s } ^ { v }$ requires first unrotating each key to remove its original RoPE rotation and then re-rotating to the target phase at $t _ { s } ^ { v }$ . Compressing in the rotated domain fails for two reasons: keys carrying distinct rotations $R ( \theta _ { f } t _ { m } )$ point in diferent directions at each frequency and partially cancel when combined, and the compressed key would still carry the phase of its source timestamps rather than that of $t _ { s } ^ { v }$ . Storing keys canonically resolves both. Concurrent memory-token work adopts the same mechanism [42] and what we add is its pairing with slot-rank positions (Def. 1). With addressable summary slots, the remaining design choice is the content writer. Both WorldTrace variants use the same canonical (un-rotated) key domain.

## 3.2 Memory Compression as Structured Sparse Attention

With slot positions fixed, the remaining choice is w<sup>h</sup>at to store in each slot. A bounded cache can take neither extreme, since storing all past frames verbatim would let the KV cache grow linearly with generation length, while evicting them, as a sliding window does, is precisely what destroys visual persistence (Sec. 2). What remains is to compress distant history rather than discard it. We view this memory compression as structured sparse attention because that view states the design target explicitly, asking the compressed cache to approximate full attention over all $T$ past frames with a fixed-� structured sparse pattern in which each of the � cache slots summarizes a segment of the full history. Framing the problem this way turns the choice of slot content into a quantifiable approximation problem. The projection matrix $P \in \mathbb { R } ^ { L \times T }$ defines this pattern by giving slot � the compressed key $P _ { i } K _ { m } ^ { f }$ , rotated to the slot’s virtual position. The recent window uses verbatim identity rows for the last $N _ { r }$ frames, and the $N _ { s }$ summary rows carry the compressed information.

Approximating full attention. Attending over all $T$ past frames would produce a softmax-weighted combination of their values, with weights $\alpha _ { q }$ determined by query-key similarity. In order to keep the cache bounded, memory compression replaces this with projected attention over � compressed keys, where row � of $P$ defines which past frames contribute to compressed key �, given by $P _ { i } K _ { m } ^ { f }$ . Re-encoding each compressed key at $t _ { s } ^ { v }$ keeps the approximation in-distribution at any horizon. We approximate the error with the distribution mismatc<sup>h</sup> between the full attention weights $\alpha _ { q }$ and the weights induced by the compressed cache (Prop. 1, App. B). The choice of � therefore determines which parts of history are preserved under a fixed slot budget.

Choosing the projection matrix. This view reduces memory compression design to the choice of $P ,$ with a good $P$ minimizing the distribution mismatch above for the queries encountered during rollout. Relaxing the entries of $P$ from fixed patterns to arbitrary nonnegative values turns this minimization into a nonnegative matrix factorization (NMF) problem [44], approximately factoring the matrix of full attention weights into the product of two nonnegative low-rank factors (App. B). The optimal row structure of $P$ then depends on these queries. Near-optimal rows average contiguous temporal groups when future attention spreads smoothly over history, and select single frames verbatim when it concentrates instead on a few salient past frames. These two attention patterns motivate the two variants we introduce next, WorldTrace-Field (Sec. 3.3) and WorldTrace-Landmark (Sec. 3.4).

## 3.3 W T F : Canonical Key Averaging

In order to maintain temporal coherence when future attention spreads smoothly over history, we propose WorldTrace-Field, which instantiates the averaging case of Sec. 3.2 by having each summary row of $P$ uniformly average the contiguous temporal group of frames assigned to its slot, compressing distant history into rotation-invariant slot traces. To avoid the phase-cancellation failure mentioned in Sec. 2.2, we fix the compression domain by aligning all source keys to a shared phase before averaging, i.e., the canonical (unrotated) representation:

Definition 2 (Canonical Key Averaging (WorldTrace-Field operator)). For each temporal head-dimension pair $f ,$ the compressed key at virtual position $t ^ { v }$ is:

$$
K _ { \mathrm { f i e l d } } ^ { f } ( t ^ { v } ) ~ = ~ R ( \theta _ { f } t ^ { v } ) ~ { \frac { 1 } { M } } \sum _ { m = 1 } ^ { M } R ( - \theta _ { f } t _ { m } ) ~ K _ { t _ { m } } ^ { f } ,\tag{3}
$$

i.e. we unrotate each key to its canonical content $R ( - \theta _ { f } t _ { m } ) ~ K _ { t _ { m } } ^ { f }$ , average in that space, and then re-encode at virtual position $t ^ { v }$ . Values are not rotated, as only keys carry RoPE. Slot values are computed as the mean of the source-frame values assigned to the slot. The count � follows from the grouping rather than being a separate hyperparameter. The $T - N _ { \tau }$ <sub>�</sub> frames that have left the recent window are split into $N _ { s }$ contiguous temporal groups (oldest to newest, slot � receives group $s )$ , giving $M \approx T / N _ { s }$ per slot, a ratio that grows linearly with generation length �.

Every new chunk pushes another frame out of the recent window and changes which frames each group covers, so the slot averages have to be maintained as the rollout grows. Frames are never averaged away irreversibly. The writer keeps the canonical key of each evicted frame in host memory and recomputes every slot mean from its own source set at each step, so a moving boundary always acts on source frames and never on an already averaged quantity. Peak GPU memory therefore stays equal to the sliding-window baseline because the retained keys live outside the cache (App. E.6).

Remark 2 (Mean attention preservation, informal). Since a summary slot stands in for its � source frames, a query should attend to it roughly as it would have attended to those frames, which makes any compression that silently changes these attention scores a source of bias toward or away from distant history. Def. 2 provides exactly thi<sub>s g</sub>u<sub>a</sub>r<sub>a</sub>nt<sub>ee</sub> b<sub>y co</sub>n<sub>s</sub>tru<sub>c</sub>tin<sub>g a co</sub>m<sub>p</sub>r<sub>esse</sub>d k<sub>ey</sub> th<sub>a</sub>t <sub>p</sub>r<sub>ese</sub>rv<sub>es</sub> th<sub>e</sub> m<sub>ea</sub>n <sub>a</sub>tt<sub>e</sub>nti<sub>o</sub>n <sub>sco</sub>r<sub>e</sub> th<sub>e so</sub>ur<sub>ce</sub> k<sub>eys</sub> $\{ K _ { t _ { m } } ^ { f } \} _ { m = 1 } ^ { M }$ would receive if each were reassigned the shared virtual position $t ^ { v }$ (formal statement in A<sub>pp</sub>. $C ,$ Pro<sub>p</sub>. 2). Avera<sub>g</sub>in<sub>g</sub> in the rotated domain ofers no such guarantee, as phase cancellation shrinks the compressed key and suppresses the score (Sec. 2.2). The <sub>g</sub>uarantee is stated on <sub>p</sub>re-softmax scores, and it does not extend to the wei<sub>g</sub>hts themselves, since after softmax each attention weight also depends on every other cached key.

## 3.4 WorldTrace-Landmark: Landmark Traces with Frozen Keys

To optimize for episodic recall, WorldTrace-Landmark fills the �<sub>�</sub> slots with verbatim high-value past frames rather than averaged summaries. This is akin to Landmark Attention [63]. However, unlike token-level landmark insertion in trained transformers, our method uses scene-entry detection, frozen canonical keys, and slot-rank virtual positions.

Landmark selection. We identify recall-relevant frames from the canonical-K representation of Eq. (3), computing for each incoming frame the cosine distance between the canonical keys of consecutive frames and marking any frame whose distance spikes above threshold � as a scene-entry event. WorldTrace-Landmark fills the $N _ { s }$ summary slots with the most recent scene-entry frames verbatim, and while fewer than $N _ { s }$ sceneentry events have been detected, the remaining slots repeat the oldest available landmark. Once more than $N _ { s }$ scene-entry events have been detected over a long rollout, the oldest is evicted to make room for the newest. Landmark slots use the same position assignment as Eq. (2), so each landmark frame inherits the slot-rank position of its summary slot rather than its absolute timestamp.

Frozen landmark keys. Slots are ordered from oldest (�=0) to newest, so each new chunk shifts every cached frame one slot toward the older end, moving a landmark frame from slot � at chunk � to slot �−1 at chunk �+1. Under standard summary updates, each shift unrotates the cached Key to its canonical form, then re-rotates it to its new virtual position. Over many shifts, these unrotate-rerotate cycles accumulate floating-point errors. This is exacerbated in practice by bfloat16 precision, as shown by Wang et al. [88]. WorldTrace-Landmark removes this drift by freezing each selected key in canonical form, storing that key once at landmark time and applying a single fresh rotation to the current virtual position at every subsequent shift:

$$
K _ { \mathrm { l a n d } } ^ { f } ( t _ { s } ^ { v } ) = R ( \theta _ { f } t _ { s } ^ { v } ) R ( - \theta _ { f } t _ { \ell ^ { * } } ) K _ { t _ { \ell ^ { * } } } ^ { f } ,\tag{4}
$$

where $t _ { \ell ^ { * } }$ is the original timestamp of the selected landmark frame. Compared to Eq. (3), the form is identical (unrotate to canonical, re-rotate to virtual position), with �=1 and no averaging. The canonical key $R ( - \theta _ { f } t _ { \ell ^ { * } } ) \ K _ { t _ { \ell ^ { * } } } ^ { f }$ is computed once at landmark time and reused across all subsequent shifts. The canonicalcaching mechanism is shared with concurrent work that stores RoPE keys canonically and rotates at attention time (e.g. MemRoPE [42]). WorldTrace-Landmark difers in (i) applying it selectively at detected sceneentry events rather than to every cached key, and (ii) combining it with the slot-rank position assignment, so each frozen landmark occupies a distinct in-distribution virtual position at any horizon.

A unified view. Viewed through the structured sparse attention lens in Sec. 3.2, WorldTrace-Field and WorldTrace-Landmark difer only in the choice of the projection matrix $P ,$ each matched to a family of future queries. Under WorldTrace-Field (Sec. 3.3), each summary row uniformly averages keys over a disjoint temporal group, yielding exactly the canonical key averaging of Eq. (3). This is near-optimal when attention varies smoothly within each compressed slot, so that future queries care about group-level history rather than individual frame identity (Eq. (6), App. B). Under WorldTrace-Landmark, each summary row instead selects a single detected scene-entry frame verbatim, recovering its frozen canonical key as in Eq. (4). This is near-optimal when recall queries concentrate on a small set of salient past scenes (Eq. (8), App. B). Together the two variants cover the query families that dominate long-horizon rollouts, smooth continuation and episodic recall, and both realize the same slot-rank, canonical-store design, difering only in the rows of �.

## 3.5 Position-Content Coupling

As noted in Sec. 2, position assignment and content compression are coupled. Concretely, the virtual positions assigned to summary slots can diverge from the ofset range under which their keys were compressed, so the cached content may no longer match what the query attends to. Block-relative collapse makes compressed entries efectively unaddressable, reducing compression to sliding-window eviction once addressability collapses (Rem. 1). Averaging position-conditioned keys under one ofset distribution and querying them under another changes the efective phase of cached tokens, breaking the mean-attention preservation that motivates Def. 2.

WorldTrace resolves this coupling by combining two mechanisms. Slot indexing (Def. 1) fixes a consistent virtual position scheme, while canonical key averaging (WorldTrace-Field) and frozen landmark keys (WorldTrace-Landmark) absorb the source-to-virtual shift by unrotating keys to canonical space and re-encoding at slot-rank positions. Within this framework, the choice of slot content, averaged frames or frozen landmarks, becomes a content-only design choice. We provide the detailed algorithm in App. G.

## 4 Experiments

We organize our experiments around our three claims. (Q1, Sec. 4.2) Is position, not content, the binding constraint? (Q2, Sec. 4.2) Does WorldTrace-Field improve co<sup>h</sup>erence over naive averaging and a slidingwindow cache? (Q3, Sec. 4.3) Can WorldTrace-Landmark improve episodic recall at extended horizons?

## 4.1 Setup

Models. Our evaluation uses Matrix-Game-2 (MG2-1.3B) [32], a distilled 1.3B-parameter autoregressive game world model based on Wan 1.3B T2V [87] with 3D-RoPE, training-time KV-cache extent $L _ { \mathrm { t r a i n } } = 6$ AR chunks (�=3 latent frames per chunk), and a local attention window of 2 chunks, i.e. $\Delta t _ { \mathrm { t r a i n } } + 1 { = } 6$ latent frames. We provide additional experiments for LingBot-World [82] in App. E, an autoregressive world model built on a 14B backbone with Plücker camera conditioning. We use $N _ { s } { = } 2 / N _ { r } { = } 4$ for coherence (Sec. 4.2) and ${ N _ { s } } \mathrm { { = } } 4 / { N _ { r } } { = } 2$ for episodic recall (Sec. 4.3) (latent-frame slots, with $N _ { s } + N _ { r } = L _ { \mathrm { a t t n } } { = } 6 )$

![](images/a7e0387f1dfec8975a39a073ce89bb366cf95c0ad08cc793c9dca7150bde7d6a.jpg)  
Figure 3: Three LoopBench geometries. We propose <sup>L</sup>oop<sup>B</sup>ench, a memory benchmark with diverse geometries, including ABA (straight reversal), ABCA (approximate L-triangle with diagonal $5 \sqrt { 2 } \approx 7 )$ , and ABCDA (square). � is rollout length in AR chunks. Each path returns to scene A. We provide a full gallery of edge-length, orientation, and multi-revisit settings in App. F (Fig. 8).

Metrics. Our evaluation focuses on two key aspects relevant to world models: the temporal co<sup>h</sup>erence of generated outputs and the ability to correctly recall revisited locations. We evaluate co<sup>h</sup>erence with TempSSIM (↑, SSIM [91] between consecutive decoded frames) and Local Scene Drift (↓, mean per-chunk CLIP feature distance to the preceding chunk). For recall, we use Position-Aligned CLIP (PAC ↑, CLIP-ViT-H/14 [70] cosine similarity between geometrically paired return- and forward-leg frames in loops).

Baselines. We compare against MG2’s default sliding-window cache throughout, which evicts the oldest tokens first-in-first-out with no position correction or content compression, plus controlled variants that isolate position assignment (Block-relative, Centroid-linear, WorldTrace) and content compression (naive vs. canonical averaging, Sec. 4.2, Sec. 4.4). Episodic recall compares three retention tiers (compression, canonical-K anchoring via Latent re-anc<sup>h</sup>or, and verbatim landmarks) on LoopBench and in the PAC sweep (Tab. 3, Tab. 10 in App. E.3). MemRoPE [42], YaRN [67], and Landmark+Block-relative are evaluated in App. E.2.

Benchmark. We introduce LoopBench, a memory benchmark for scene revisits in AR world models. The model traces waypoints before returning to a previously visited location, and the regenerated return frame is scored against the original scene appearance at geometrically matched positions, requiring no external reference. LoopBench varies four properties of the path: waypoint count (�, Fig. 3), rollout length (�), camera orientation, and multi-revisit depth (�). The full benchmark gallery with all evaluated configurations is in Fig. 8 (App. F).

## 4.2 Coherence: W T F

Virtual Position Assignment. WorldTrace-Field averages keys in the canonical domain, then assigns each averaged slot a virtual position, maintaining scene coherence under compression. We compare it against canonical-K averaging variants that apply this same key averaging but difer in position assignments. Bloc<sup>k</sup>-relative [107] clamps ofsets beyond the training horizon, so distinct older slots collapse onto a single capped position,

Table 1: Position assignment. Canonical averaging stays the same so that only positions vary. Values are TempSSIM (↑).
<table><tr><td>Position Assignment</td><td>N=8 N=16</td></tr><tr><td>Block-relative</td><td>0.390 0.530</td></tr><tr><td>Centroid-linear</td><td>0.377 0.479</td></tr><tr><td>WORLDTRACE (ours)</td><td>0.413 0.545</td></tr></table>

whereas Centroid-linear positions each slot by the average timestamp of the frames it summarizes, linearly mapped into $[ t _ { \operatorname* { m i n } } ^ { v } , t _ { \operatorname* { m a x } } ^ { v } ]$ , so slots stay distinct but their positions shift as the rollout grows. Relative to the four position properties of Sec. 3.1, Centroid-linear satisfies (i) linearity and (ii) in-distribution placement but violates (iii) horizon-stability, since its virtual positions depend on �. Instead, our WorldTrace-Field assigns positions from slot rank alone, so every summary slot stays in-distribution and individually addressable no matter how long the rollout runs.

With the same compression approach but diferent position assignment shows that compression helps only when the summary slots remain addressable. With Block-relative positions, ofsets beyond the training window receive

![](images/c07a3a3c2fae4a3be5d331bb06907e66a600d2b45178aecf461a5111c3f50cb7.jpg)  
Figure 4: W T F at number of chunks �=48. Four KV-cache approaches conditioned on the same initial image and camera trajectory, including three baselines (sliding window, Block-relative, Centroid-linear) and WorldTrace-Field (ours). Baseline approaches start drifting at �=18 (red border). WorldTrace-Field remains coherent across the horizon. Camera path plotted in Fig. 11 (App. G).

Table 2: achieves the highest TempSSIM at both horizons and the lowest Scene Drift. At �=32, WorldTrace-Field has the highest TempSSIM (0.613) though centroid variants have lower Drift. At �=48, WorldTrace-Field is best on both metrics.
<table><tr><td rowspan="2">Method</td><td colspan="2">N=32</td><td colspan="2">N=48</td></tr><tr><td>TempSSIM ↑</td><td>Scene Drift ↓</td><td>TempSSIM ↑</td><td>Scene Drift↓</td></tr><tr><td>Sliding window</td><td>0.571</td><td>0.0229</td><td>0.472</td><td>0.0305</td></tr><tr><td>Canonical averaging + Block-relative</td><td>0.585</td><td>0.0215</td><td>0.530</td><td>0.0339</td></tr><tr><td>Canonical averaging + Centroid-linear</td><td>0.573</td><td>0.0211</td><td>0.479</td><td>0.0297</td></tr><tr><td>WORLDTRACE-FIELD (ours)</td><td>0.613</td><td>0.0250</td><td>0.545</td><td>0.0295</td></tr></table>

zero softmax weight under local attention mask, so canonical averaging reduces to a plain sliding-window cache. Centroid-linear avoids this saturation by keeping positions within the attention mask, but its �-dependent assignment shifts old summaries into positional ranges the model was not trained to use as the horizon grows. Our WorldTrace instead assigns positions by slot rank alone and keeps summary slots in-distribution at all horizons, outperforming Block-relative by +5.9% and +2.8% TempSSIM and Centroid-linear by +9.5% and +13.8% at �=8 and �=16 (Tab. 1).

We show this qualitatively in Fig. 4, where by �=18 the baselines drift from the initial scene context and generate incoherent geometry, while our WorldTrace-Field stays coherent. We quantify it at two horizons (�=32, 48) in Tab. 2, under the same canonical averaging setting. At �=32, WorldTrace-Field achieves higher TempSSIM (0.613 vs. 0.571), whereas Centroid-linear reaches the lowest Drift but at 0.04 lower TempSSIM. At �=48, WorldTrace-Field is best on bot<sup>h</sup> dimensions (+15.5% relative gain on TempSSIM, lowest Drift), while a position assignment that depends on � (Centroid-linear at 0.479) ranks inconsistently across horizons, as its slot indices shift out-of-distribution once � exceeds the training window. These results show that horizon-independent position assignment is the bottleneck for long-horizon generation.

## 4.3 Episodic Recall: WorldTrace-Landmark

While Sec. 4.2 evaluates WorldTrace-Field on open-ended rollouts, episodic recall asks whether the cache can restore a specific past scene at revisit points. We therefore test WorldTrace-Landmark (Sec. 3.4) on LoopBench (Fig. 3, Sec. 4.1) and score each return frame against scene A at matched poses with PAC. We show the qualitative results in Fig. 5, where the sliding-window baseline fails to reconstruct scene A at revisits, whereas WorldTrace-Landmark matches it across all three geometries.

We evaluate WorldTrace-Landmark across four dificulty tiers in Tab. 3: topology (ABA, ABCA, ABCDA), rollout length (ABA at �∈{8, 16, 32}), camera-orientation (pan 90<sup>∘</sup>/180<sup>∘</sup>/360<sup>∘</sup> with the agent staying at A), and

View at A on return  
![](images/e5c4200a2421524b4138ee9d7e582d05365d3bad06307036f660bb6eb25a9220.jpg)  
Figure 5: L B qualitative results on the three revisit geometries in Fig. 3 (ABA, ABCA, and ABCDA). The sliding-window baseline fails to reconstruct scene A when revisiting. WorldTrace-Landmark matches scene A in all three, keeping memory addressable across waypoints.

Table 3: WorldTrace-Landmark improves episodic recall across topology, edge length, camera orientation, and multi-revisit depth. Loop geometries and dificulty axes are in Fig. 8 (App. F). Each tier groups methods (rows) by loop configuration (columns), and entries are PAC / TempSSIM. � is rollout length in AR chunks, and $n { = } 1 0 0$ is the number of distinct initial scenes per condition (Tab. 6). In Tier 4 (�=2, row 4 of Fig. 8), ABABA revisits scene A, while ABCBA and ABCDBA revisit waypoint B on palindrome/shortcut paths.
<table><tr><td colspan="4">Tier 1: varying topology</td></tr><tr><td></td><td> ${ \tt A B A } \left( K { = } 1 , N { = } 1 6 \right)$ </td><td> $\mathrm { A B C A } \left( K { = } 2 , N { = } 1 7 \right)$ </td><td> $\mathrm { A B C D A } \left( K { = } 3 , N { = } 1 6 \right)$ </td></tr><tr><td>Sliding window</td><td> $0 . 7 2 3 _ { \pm 0 . 0 1 3 } / 0 . 7 6 1$ </td><td> $0 . 6 7 3 { \scriptstyle \pm 0 . 0 1 4 } / 0 . 7 4 0$ </td><td> $0 . 6 6 6 { \scriptstyle \pm 0 . 0 1 3 } / 0 . 7 4 8$ </td></tr><tr><td>WORLDTRACE-LANDMARK</td><td> $\mathbf { 0 . 8 6 4 { \scriptstyle \pm 0 . 0 0 9 } } / \mathbf { 0 . 7 8 6 }$ </td><td> $\mathbf { 0 . 7 9 2 { \scriptstyle \pm 0 . 0 1 1 } } / \mathbf { 0 . 7 5 8 }$ </td><td> $\mathbf { 0 . 7 9 9 { \scriptstyle \pm 0 . 0 1 1 } } / \mathbf { 0 . 7 7 1 }$ </td></tr><tr><td colspan="2">Tier 2: varying rollout length (N, ABA)</td><td></td><td></td></tr><tr><td></td><td>ABA short (N=8)</td><td> $\mathrm { A B A } \left( N { = } 1 6 \right)$ </td><td>ABA long (N=32)</td></tr><tr><td>Sliding window</td><td> $0 . 8 5 9 { \scriptstyle \pm 0 . 0 0 7 } / 0 . 7 8 0$ </td><td> $0 . 7 2 3 { \scriptstyle \pm 0 . 0 1 3 } / 0 . 7 6 1$ </td><td> $0 . 6 2 7 { \scriptstyle \pm 0 . 0 1 3 } / 0 . 7 7 1$ </td></tr><tr><td>WORLDTRACE-LANDMARK 0.922±0.004 / 0.789</td><td></td><td> $\mathbf { 0 . 8 6 4 { \scriptstyle \pm 0 . 0 0 9 } } / \mathbf { 0 . 7 8 6 }$ </td><td> $\mathbf { 0 . 8 2 5 { \scriptstyle \pm 0 . 0 0 9 } } / \mathbf { 0 . 7 8 7 }$ </td></tr><tr><td colspan="2">Tier 3: camera-orientation (agent fixed at A)</td><td></td><td></td></tr><tr><td></td><td> $\operatorname { P a n } 9 0 ^ { \circ } \ ( N { = } 4 )$ </td><td> $\mathrm { P a n } 1 8 0 ^ { \circ } \ ( N { = } 8 )$ </td><td> $\mathrm { P a n 3 6 0 ^ { \circ } } \left( N { = } 8 \right)$ </td></tr><tr><td>Sliding window</td><td> $0 . 8 2 9 _ { \pm 0 . 0 0 8 } / 0 . 4 8 0$ </td><td> $0 . 6 7 1 { \scriptstyle \pm 0 . 0 1 4 } / \ 0 . 4 5 8$ </td><td> $0 . 5 5 9 { \scriptstyle \pm 0 . 0 1 5 } / 0 . 4 5 5$ </td></tr><tr><td>WORLDTRACE-LANDMARK</td><td> $\mathbf { 0 . 8 6 1 { \scriptstyle \pm 0 . 0 0 7 } } / \mathbf { 0 . 4 9 3 }$ </td><td> $\mathbf { 0 . 7 8 1 { \scriptstyle \pm 0 . 0 1 0 } / 0 . 4 8 6 }$ </td><td> $\mathbf { 0 . 5 7 7 { \scriptstyle \pm 0 . 0 1 5 } / 0 . 4 6 7 }$ </td></tr><tr><td>Tier 4: multi-revisit</td><td></td><td></td><td></td></tr><tr><td colspan="2"> $\mathrm { A B A B A } \left( N { = } 3 2 \right)$ </td><td> $\operatorname { A B C B A } \left( N { = } 2 0 \right)$ </td><td>ABCDBA (N=20)</td></tr><tr><td>Sliding window</td><td> $0 . 8 9 2 _ { \pm 0 . 0 0 4 } / 0 . 7 6 5$ </td><td></td><td></td></tr><tr><td></td><td></td><td> $0 . 8 2 5 { \scriptstyle \pm 0 . 0 1 0 } / 0 . 7 5 1$ </td><td> $0 . 8 4 2 _ { \pm 0 . 0 1 0 } / 0 . 7 5 8$ </td></tr><tr><td>WORLDTRACE-LANDMARK</td><td> $\mathbf { 0 . 9 4 1 { \scriptstyle \pm 0 . 0 0 5 } } / \mathbf { 0 . 7 8 9 }$ </td><td> $\mathbf { 0 . 8 6 3 _ { \pm 0 . 0 0 9 } / 0 . 7 7 1 }$ </td><td> $\mathbf { 0 . 8 7 6 { \scriptstyle \pm 0 . 0 0 9 } } / \mathbf { 0 . 7 7 5 }$ </td></tr></table>

multi-revisit (ABABA, ABCBA, ABCDBA). All runs use $ N _ { s } { = } 4 , N _ { r } { = } 2 .$

Across tiers, WorldTrace-Landmark improves PAC over the sliding-window baseline in every reported condition, with the largest relative gains when the KV gap to scene A is longest (Tier 2) and when multiple waypoints separate departure and return (Tier 1). Tier 3 is the hardest setting, where gains shrink at a full 360<sup>∘</sup> pan, and Tier 4 shows a narrower margin on palindrome returns that revisit waypoint B before scene A. At extended horizons, we report the PAC sweep in Tab. 10 (App. E.3), where only verbatim landmark recall sustains PAC≈0.99 at $N { = } 2 5 6$ , against 0.610 for canonical-K anchoring. We provide further comparisons with MemRoPE and YaRN, history-selection ablations, and slot-count sensitivity in App. E.

## 4.4 Ablation

We isolate two design axes. (1) Position assignment (WorldTrace vs. Block-relative vs. Centroid-linear), where holding the content operator fixed and varying only positions isolates whether position, not content, is

the binding constraint for WorldTrace-Field. (2) Canonical vs. naive key compression, where canonicaldomain averaging avoids phase cancellation (Sec. 2.2). Its benefit surfaces on coherence (TempSSIM, LatentDif) rather than recall. We use the ABA recall protocol with a fixed cache budget.  
Table 4: Position assignment & canonical compression ablation (Axes 1 & 2). PAC (↑) and TempSSIM (↑) on ABA recall.
<table><tr><td rowspan="2">Method</td><td colspan="4">PAC ↑</td><td rowspan="2">TempSSIM (N=16) ↑</td></tr><tr><td>N=16</td><td>N=64</td><td>N=128</td><td>N=256</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Axes 1 &amp; 2: position assignment &amp; canonical compression Sliding window (baseline)</td><td>0.540</td><td>0.412</td><td>0.504</td><td>0.631</td><td>0.9945</td></tr><tr><td>+ Naive averaging (Block-relative)</td><td>0.570</td><td>0.443</td><td>0.486</td><td>0.598</td><td>0.9944</td></tr><tr><td>+ Field averaging (Block-relative)</td><td>0.540</td><td>0.412</td><td>0.504</td><td>0.631</td><td>0.9945</td></tr><tr><td>+ WORLDTRACE (WORLDTRACE-FIELD, Ours)</td><td>0.555</td><td>0.442</td><td>0.495</td><td>0.602</td><td>0.9948</td></tr></table>

(1) Slot-rank positions keep summary slots addressable (Tab. 4). Naive averaging in RoPE-rotated space scores marginally higher on PAC at �=16 (0.570 vs. 0.540), but under the same Block-relative positions it cannot reflect addressable recall, and both stay in the compression tier (≈0.4 to 0.6 PAC) far below anchoring or verbatim recall (≈0.95+, Tab. 10). Only WorldTrace keeps every summary slot in-distribution, and WorldTrace-Field accordingly recovers PAC=0.555 against 0.540 for the sliding-window baseline (+0.015), a margin that grows to +0.030 at $N { = } 6 4$ , while achieving the best TempSSIM (0.9948 vs. 0.9945).

(2) Canonical key averaging avoids phase cancellation and improves coherence (Tab. 5). Holding positions fixed at Block-relative, switching from naive to canonical averaging reduces LatentDif from 0.312 to 0.233 at $N _ { s } { = } 4 \ ( - 2 5 . 3 \% )$ . Under naive averaging, LatentDif grows with $N _ { s }$ (from 0.257 at $N _ { s } { = } 1$ to 0.312 at $N _ { s } { = } 4 )$ as each additional summary slot mixes keys from frames at more diverse timestamps amplifying RoPE phase cancellation, while canonical

Table 5: Phase cancellation: Naive vs. canonical key averaging. LatentDif $\left( \downarrow \right)$ at �=16 rollout chunks for varying numbers of summary-cache slots $N _ { s } .$ Canonical averaging avoids the phase cancellation that corrupts low frequencies under naive RoPE-space averaging.
<table><tr><td> $N _ { s }$ </td><td>Naive LatentDiff ↓</td><td>Canonical LatentDiff↓</td></tr><tr><td>1</td><td>0.257</td><td>0.224</td></tr><tr><td>2</td><td>0.261</td><td>0.257</td></tr><tr><td>4</td><td>0.312</td><td>0.233</td></tr></table>

averaging is unafected (0.224 to 0.257). Averaging in canonical (unrotated) space is therefore necessary to preserve the low-frequency content of compressed summary Keys.

## 5 Conclusion

Long-horizon failure in autoregressive video world models is not only a matter of retaining more past content, but of keeping that content addressable. We show that temporal RoPE extrapolation can make compressed memories efectively unreadable once rollouts move beyond the training horizon, and introduce WorldTrace, a training-free cache mechanism that keeps distant traces at fixed in-distribution virtual positions. Within this addressable cache, WorldTrace-Field provides a coherence-oriented summary of history, while WorldTrace-Landmark preserves verbatim traces for episodic recall. Empirically, WorldTrace-Field improves Temporal Consistency by +15.5% at $N = 4 8$ , and WorldTrace-Landmark raises Scene Consistency by +19.5% on LoopBench ABA loops, extending visually persistent generation from seconds toward minute-scale return-toorigin rollouts.

More broadly, WorldTrace suggests that memory for video world models should be designed around two questions: how to ensure past content remains addressable, and what kind of visual traces should be stored for future queries. The two variants studied here represent diferent points in this design space, with smooth summaries supporting temporal coherence and sparse landmarks supporting recall of specific places. This perspective also clarifies the limits of the current method. WorldTrace is designed for temporal-RoPE autoregressive models with a fixed KV-cache budget, and its current content writers instantiate only two simple structured projections of history, so that WorldTrace-Field favors coherence but can blur specific scene details, while WorldTrace-Landmark preserves specific places but depends on detecting and retaining the right scene entries. Future memory systems may need to choose these projections more adaptively, especially important in interactive settings where agents move freely, revisit locations, and query the past from changing viewpoints. In this view, WorldTrace-Field and WorldTrace-Landmark are not endpoints, but two structured approximations to the same underlying problem of projecting a long visual history into a small addressable cache.

## Acknowledgements

We thank our collaborators at NVIDIA’s Spatial Intelligence Lab and Princeton University for their helpful discussions and feedback.

## References

[1] Niket Agarwal, Arslan Ali, Jon Allen, Martin Antolini, Adeline Aubame, Alisson Azzolini, Junjie Bai, Maciej Bala, Yogesh Balaji, Josh Bapst, et al. Cosmos 3: Omnimodal world models for physical ai. arXiv preprint arXiv:2606.02800, 2026.

[2] Federico Barbero, Alex Vitvitskyi, Christos Perivolaropoulos, Razvan Pascanu, and Petar Veličković. Round and round we go! what makes rotary positional encodings useful? In International Conference on Learning Representations (ICLR), 2025. arXiv:2410.06205.

[3] Aarti Basant, Amlan Kar, Despoina Paschalidou, Fangyin Wei, Francesco Ferroni, Guillermo Garcia Cobo, Haithem Turki, Huan Ling, Jaewoo Seo, James Lucas, et al. Nvidia omnidreams: Real-time generative world model for closed-loop autonomous vehicle simulation. arXiv preprint arXiv:2606.03159, 2026.

[4] Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.

[5] Jesse Bettencourt, Xindi Wu, Matan Atzmon, James Lucas, and Jonathan Lorraine. Variance reduction for expectations with difusion teachers. arXiv preprint arXiv:2605.21489, 2026.

[6] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hofman. Token merging: Your ViT but faster. In International Conference on Learning Representations (ICLR), 2023.

[7] Aydar Bulatov, Yuri Kuratov, and Mikhail S. Burtsev. Recurrent memory transformer. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

[8] Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, Maneesh Agrawala, Lu Jiang, and Gordon Wetzstein. Mixture of contexts for long video generation. In International Conference on Learning Representations (ICLR), 2026. arXiv:2508.21058, OpenReview: https://openreview.net/forum?id=y6XJZlEC2x.

[9] Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Yucheng Li, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Junjie Hu, and Wen Xiao. PyramidKV: Dynamic KV cache compression based on pyramidal information funneling. In Conference on Language Modeling (COLM), 2025.

[10] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Difusion forcing: Next-token prediction meets full-sequence difusion. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

[11] Hanmo Chen, Chenghao Xu, Xu Yang, Xuan Chen, and Cheng Deng. Past- and future-informed kv cache policy with salience estimation in autoregressive video difusion. arXiv preprint arXiv:2601.21896, 2026.

[12] Jintao Chen, Chengyu Bai, Junjun Hu, Xinda Xue, and Mu Xu. Grounded forcing: Bridging time-independent semantics and proximal dynamics in autoregressive video synthesis. arXiv preprint arXiv:2604.06939, 2026.

[13] Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595, 2023.

[14] Shuo Chen, Cong Wei, Sun Sun, Ping Nie, Kai Zhou, Ge Zhang, Ming-Hsuan Yang, and Wenhu Chen. Context forcing: Consistent autoregressive video generation with long context. arXiv preprint arXiv:2602.06028, 2026.

[15] Taiye Chen, Xun Hu, Zihan Ding, and Chi Jin. Learning world models for interactive video generation. arXiv preprint ar<sup>Xi</sup>v:<sup>2505</sup>.<sup>21</sup>99<sup>6</sup>, 2025. Project page https://sites.google.com/view/vrag.

[16] Giulio Corallo and Paolo Papotti. FINCH: Prompt-guided key-value cache compression for large language models. Transactions of the Association for Computational Linguistics, 12:1517–1532, 2024.

[17] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Lol: Longer than longer, scaling video generation to hour. arXiv preprint arXiv:2601.16914, 2026.

[18] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Self-Forcing++: Towards minute-scale high-quality video generation. In International Conference on Learning Representations (ICLR), 2026. arXiv:2510.02283.

[19] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, and Ruslan Salakhutdinov. Transformer-XL: Attentive language models beyond a fixed-length context. In Annual Meeting of t<sup>h</sup>e Associationfor Computational Linguistics (ACL), pages 2978–2988, 2019. arXiv:1901.02860.

[20] Decart, Julian Quevedo, Quinn McIntyre, Spruce Campbell, Xinlei Chen, and Robert Wachen. Oasis: A universe in a transformer. https://oasis-model.github.io, 2024.

[21] Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. WorldScore: A unified evaluation benchmark for world generation. In International Conference on Computer Vision (ICCV), 2025.

[22] Dan Feldman and Michael Langberg. A unified framework for approximating and clustering data. In Proceedings of t<sup>h</sup>e 43rd Annual ACM Symposium on T<sup>h</sup>eory of Computing (STOC), pages 569–578, 2011.

[23] Xinhang Gao, Junlin Guan, Shuhan Luo, Wenzhuo Li, Guanghuan Tan, and Jiacheng Wang. Memcam: Memoryaugmented camera control for consistent video generation. arXiv preprint arXiv:2603.26193, 2026.

[24] Olga Golovneva, Tianlu Wang, Jason Weston, and Sainbayar Sukhbaatar. Contextual position encoding: Learning to count what’s important. arXiv preprint arXiv:2405.18719, 2024.

[25] Google DeepMind. Genie 3: A new frontier for world models. DeepMind Blog, https://deepmind.google/blog/ genie-3-a-new-frontier-for-world-models, 2025.

[26] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In Conference on Language Modeling (COLM), 2024.

[27] Yuchao Gu, Weijia Mao, and Mike Zheng Shou. Long-context autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325, 2025.

[28] Yanjun Guo, Zhengqiang Zhang, Pengfei Wang, Xinyue Liang, Zhiyuan Ma, and Lei Zhang. Memorize when needed: Decoupled memory control for spatially consistent long-horizon video generation. arXiv preprint arXiv:2604.18215, 2026.

[29] David Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. In Advances in Neural Information Processing Systems (NeurIPS), 2018.

[30] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature, 640:647–653, 2025.

[31] Junhui He, Junna Xing, Nan Wang, Rui Xu, Shangyu Wu, Peng Zhou, Qiang Liu, Chun Jason Xue, and Qingan Li. A<sup>2</sup>ATS: Retrieval-based KV cache reduction via windowed rotary position embedding and query-aware vector quantization. In Findings of t<sup>h</sup>e Association for Computational Linguistics (ACL Findings), 2025. arXiv:2502.12665.

[32] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, Baixin Xu, Hao-Xiang Guo, Kaixiong Gong, Size Wu, Wei Li, Xuchen Song, Yang Liu, Yangguang Li, and Yahui Zhou. Matrix-game 2.0: An open-source, real-time, and streaming interactive world model. arXiv preprint arXiv:2508.13009, 2025.

[33] Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. StreamingT2V: Consistent, dynamic, and extendable long video generation from text. In Conference on Computer Vision and Pattern Recognition (CVPR), 2025. arXiv:2403.14773.

[34] Byeongho Heo, Song Park, Dongyoon Han, and Sangdoo Yun. Rotary position embedding for vision transformer. In European Conference on Computer Vision (ECCV), 2024. arXiv:2403.13298.

[35] Yicong Hong, Yiqun Mei, Chongjian Ge, Yiran Xu, Yang Zhou, Sai Bi, Yannick Hold-Geofroy, Mike Roberts, Matthew Fisher, Eli Shechtman, Kalyan Sunkavalli, Feng Liu, Zhengqi Li, and Hao Tan. RELIC: Interactive video world model with long-horizon memory. arXiv preprint arXiv:2512.04040, 2025.

[36] Ermo Hua, Che Jiang, Xingtai Lv, Kaiyan Zhang, Youbang Sun, Yuchen Fan, Xuekai Zhu, Biqing Qi, Ning Ding, and Bowen Zhou. Fourier position embedding: Enhancing attention’s periodic extension for length generalization. In International Conference on Machine Learning (ICML), 2025. arXiv:2412.17739.

[37] Siqiao Huang, Jialong Wu, Qixing Zhou, Shangchen Miao, and Mingsheng Long. Vid2World: Crafting video difusion models to interactive world models. In International Conference on Learning Representations (ICLR), 2026. arXiv:2505.14357.

[38] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video difusion. In Advances in Neural Information Processing Systems (NeurIPS), 2025. Spotlight; arXiv:2506.08009.

[39] DeLesley Hutchins, Imanol Schlag, Yuhuai Wu, Ethan Dyer, and Behnam Neyshabur. Block-recurrent transformers. In Advances in Neural Information Processing Systems (NeurIPS), 2022. arXiv:2203.07852.

[40] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are RNNs: Fast autoregressive transformers with linear attention. In International Conference on Mac<sup>h</sup>ine Learning (ICML), 2020.

[41] Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, and Siva Reddy. The impact of positional encoding on length generalization in transformers. In Advances in Neural Information Processing Systems (NeurIPS), 2023. arXiv:2305.19466.

[42] Youngrae Kim, Qixin Hu, C.-C. Jay Kuo, and Peter A. Beerel. MemRoPE: Training-free infinite video generation via evolving memory tokens. arXiv preprint arXiv:2603.12513, 2026.

[43] Deguang Kong, Chris Ding, and Heng Huang. Robust nonnegative matrix factorization using l21-norm. In Proceedings of the 20th ACM International Conference on Information and Knowledge Management, pages 673–682, 2011.

[44] Daniel D. Lee and H. Sebastian Seung. Learning the parts of objects by non-negative matrix factorization. Nature, 401(6755):788–791, 1999.

[45] Jiaqi Leng, Xiang Hu, Junxiong Wang, Jianguo Li, Wei Wu, and Yucheng Lu. Understanding and improving length generalization in hierarchical sparse attention models. arXiv preprint arXiv:2510.17196, 2025.

[46] Haodong Li, Shaoteng Liu, Zhe Lin, and Manmohan Chandraker. Rolling sink: Bridging limited-horizon training and open-ended testing in autoregressive video difusion. arXiv preprint arXiv:2602.07775, 2026.

[47] Jia Li, Xiaomeng Fu, Xurui Peng, Weifeng Chen, Youwei Zheng, Tianyu Zhao, Jiexi Wang, Fangmin Chen, Xing Wang, and Hayden Kwok-Hay So. Train short, inference long: Training-free horizon extension for autoregressive video generation. arXiv preprint arXiv:2602.14027, 2026.

[48] Kunyang Li, Mubarak Shah, and Yuzhang Shang. PackCache: A training-free acceleration method for unified autoregressive video generation via compact KV-cache. arXiv preprint arXiv:2601.04359, 2026.

[49] Ruibin Li, Tao Yang, Fangzhou Ai, Tianhe Wu, Shilei Wen, Bingyue Peng, and Lei Zhang. Long-horizon streaming video generation via hybrid attention with decoupled distillation. arXiv preprint arXiv:2604.10103, 2026.

[50] Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, and Angjoo Kanazawa. Cameras as relative positional encoding. Advances in Neural Information Processing Systems, 38:15984–16009, 2026.

[51] Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. VMem: Consistent interactive video scene generation with surfel-indexed view memory. In International Conference on Computer Vision (ICCV), 2025. arXiv:2506.18903.

[52] Wuyang Li, Wentao Pan, Po-Chien Luan, Yang Gao, and Alexandre Alahi. Stable video infinity: Infinite-length video generation with error recycling. arXiv preprint arXiv:2510.09212, 2025.

[53] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. SnapKV: LLM knows what you are looking for before generation. In Advances in Neural Information Processing Systems (NeurIPS), 2024. arXiv:2404.14469.

[54] Kewei Lian, Shaofei Cai, Yilun Du, and Yitao Liang. Toward memory-aided world models: Benchmarking via spatial consistency. arXiv preprint arXiv:2505.22976, 2025. Loop-based Minecraft navigation benchmark for spatial consistency in world models.

[55] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video difusion in real time. In International Conference on Learning Representations (ICLR), 2026. arXiv:2509.25161.

[56] Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. KIVI: A tuning-free asymmetric 2-bit quantization for KV cache. In International Conference on Mac<sup>h</sup>ine Learning (ICML), 2024. arXiv:2402.02750.

[57] Jonathan Lorraine and Safwan Hossain. JacNet: Learning functions with structured Jacobians. In ICML Wor<sup>k</sup>s<sup>h</sup>op on Invertible Neural Networks and Normalizing Flows (INNF), 2019.

[58] Jonathan Lorraine, Nihesh Anderson, Chansoo Lee, Quentin De Laroussilhe, and Mehadi Hassen. Task selection for AutoML system evaluation. arXiv preprint arXiv:2208.12754, 2022.

[59] Jonathan Lorraine, Kevin Xie, Xiaohui Zeng, Chen-Hsuan Lin, Towaki Takikawa, Nicholas Sharp, Tsung-Yi Lin, Ming-Yu Liu, Sanja Fidler, and James Lucas. ATT3D: Amortized text-to-3D object synthesis. In International Conference on Computer Vision (ICCV), 2023.

[60] Yuexiao Ma, Xuzhe Zheng, Jing Xu, Xiwei Xu, Feng Ling, Xiawu Zheng, Huafeng Kuang, Huixia Li, Xing Wang, Xuefeng Xiao, Fei Chao, and Rongrong Ji. Flow caching for autoregressive video generation. arXiv preprint arXiv:2602.10825, 2026.

[61] Weian Mao, Xi Lin, Wei Huang, Yuxin Xie, Tianfu Fu, Bohan Zhuang, Song Han, and Yukang Chen. TriAttention: Eficient long reasoning with trigonometric KV compression. arXiv preprint arXiv:2604.04921, 2026.

[62] Xiaofeng Mao, Shaohao Rui, Kaining Ying, Bo Zheng, Chuanhao Li, Mingmin Chi, and Kaipeng Zhang. PackForcing: Short video training sufices for long video sampling and long context inference. arXiv preprint arXiv:2603.25730, 2026.

[63] Amirkeivan Mohtashami and Martin Jaggi. Landmark attention: Random-access infinite context length for transformers. In Advances in Neural Information Processing Systems (NeurIPS), 2023. arXiv:2305.16300.

[64] Tsendsuren Munkhdalai, Manaal Faruqui, and Siddharth Gopal. Leave no context behind: Eficient infinite context transformers with Infini-attention. arXiv preprint arXiv:2404.07143, 2024.

[65] NVIDIA. KVPress: A compression library for transformer KV caches. https://github.com/NVIDIA/kvpress, 2024.

[66] Yuta Oshima, Yusuke Iwasawa, Masahiro Suzuki, Yutaka Matsuo, and Hiroki Furuta. WorldPack: Compressed memory improves spatial consistency in video world modeling. arXiv preprint arXiv:2512.02473, 2025.

[67] Bowen Peng, Jefrey Quesnelle, Honglu Fan, and Enrico Shippole. YaRN: Eficient context window extension of large language models. In International Conference on Learning Representations (ICLR), 2024. arXiv:2309.00071.

[68] Ryan Po, Yotam Nitzan, Richard Zhang, Berlin Chen, Tri Dao, Eli Shechtman, Gordon Wetzstein, and Xun Huang. Long-context state-space video world models. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 8733–8744. IEEE, 2025.

[69] Ofir Press, Noah A. Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. In International Conference on Learning Representations (ICLR), 2022.

[70] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Mac<sup>h</sup>ine Learning (ICML), 2021.

[71] Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, and Timothy P. Lillicrap. Compressive transformers for long-range sequence modelling. In International Conference on Learning Representations (ICLR),

2020.

[72] Jessie Richter-Powell, Antonio Torralba, and Jonathan Lorraine. Score distillation sampling for audio: Source separation, synthesis, and beyond. In ICML Wor<sup>k</sup>s<sup>h</sup>o on AI Heard T<sup>h</sup>at!, 2025. arXiv:2505.04621.

[73] Dvir Samuel, Issar Tzachor, Matan Levy, Michael Green, Gal Chechik, and Rami Ben-Ari. Fast autoregressive video difusion and world models with temporal cache compression and sparse attention. arXiv preprint arXiv:2602.01801, 2026.

[74] Ning Shang, Li Lyna Zhang, Siyuan Wang, Gaokai Zhang, Gilsinia Lopez, Fan Yang, Weizhu Chen, and Mao Yang. LongRoPE2: Near-lossless LLM context window scaling. In International Conference on Mac<sup>h</sup>ine Learning (ICML), 2025. arXiv:2502.20082.

[75] Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. History-guided video difusion. In International Conference on Mac<sup>h</sup>ine Learning (ICML), 2025. arXiv:2502.06764.

[76] Yanke Song, Jonathan Lorraine, Weili Nie, Karsten Kreis, and James Lucas. Multi-student difusion distillation for better one-step generators. In ICML Wor<sup>k</sup>s<sup>h</sup>op on Eficient Systems for Foundation Models (ES-FoMo), 2025. arXiv:2410.23274.

[77] Sebastian Stapf, Pablo Acuaviva, Aram Davtyan, and Paolo Favaro. Composition of memory experts for difusion world models. In International Conference on Learning Representations (ICLR), 2026. OpenReview: https:// openreview.net/forum?id=sUEdpZCHdp.

[78] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

[79] Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. WorldPlay: Towards long-term geometric consistency for real-time interactive world modeling. arXiv preprint arXiv:2512.14614, 2025.

[80] Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, and Carlos Guestrin. Learning to (learn at test time): RNNs with expressive hidden states. In International Conference on Mac<sup>h</sup>ine Learning (ICML), 2025.

[81] Yutao Sun, Li Dong, Barun Patra, Shuming Ma, Shaohan Huang, Alon Benhaim, Vishrav Chaudhary, Xia Song, and Furu Wei. A length-extrapolatable transformer. In Annual Meeting of t<sup>h</sup>e Association for Computational Linguistics (ACL), 2023. arXiv:2212.10554.

[82] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, Yihang Chen, Jie Liu, Yansong Cheng, Yao Yao, Jiayi Zhu, Yihao Meng, Kecheng Zheng, Qingyan Bai, Jingye Chen, Zehong Shen, Yue Yu, Xing Zhu, Yujun Shen, and Hao Ouyang. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.

[83] Yuxuan Tian, Zihan Wang, Yebo Peng, Aomufei Yuan, Zhiming Wang, Bairen Yi, Xin Liu, Yong Cui, and Tong Yang. KeepKV: Achieving periodic lossless KV cache compression for eficient LLM inference. In AAAI Conference on Artificial Intelligence, 2026. arXiv:2504.09936.

[84] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Difusion models are real-time game engines. In International Conference on Learning Representations (ICLR), 2025.

[85] Pavlo Vasylenko, Hugo Pitorro, André F. T. Martins, and Marcos Treviso. Long-context generalization with sparse attention. arXiv preprint arXiv:2506.16640, 2025.

[86] Apoorv Vyas, Angelos Katharopoulos, and François Fleuret. Fast transformers with clustered attention. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

[87] Wan Team. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

[88] Haonan Wang, Qian Liu, Chao Du, Tongyao Zhu, Cunxiao Du, Kenji Kawaguchi, and Tianyu Pang. When precision meets position: BFloat16 breaks down RoPE in long-context training. Transactions on Mac<sup>h</sup>ine Learning Researc<sup>h</sup> (TMLR), 2025. arXiv:2411.13476.

[89] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020.

[90] Zhengyi Wang, Jonathan Lorraine, Yikai Wang, Hang Su, Jun Zhu, Sanja Fidler, and Xiaohui Zeng. LLaMA-Mesh: Unifying 3D mesh generation with language models. arXiv preprint arXiv:2411.09595, 2024.

[91] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004.

[92] Zile Wang, Zexiang Liu, Jiaxing Li, Kaichen Huang, Baixin Xu, Fei Kang, Mengyin An, Peiyu Wang, Biao Jiang, Yichen Wei, Yidan Xietian, Jiangbo Pei, Liang Hu, Boyi Jiang, Hua Xue, Zidong Wang, Haofeng Sun, Wei Li, Wanli Ouyang, Xianglong He, Yang Liu, Yangguang Li, and Yahui Zhou. Matrix-game 3.0: Real-time and streaming interactive world model with long-horizon memory. arXiv preprint arXiv:2604.08995, 2026.

[93] Xilin Wei, Xiaoran Liu, Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Jian Tong, Haodong Duan, Qipeng Guo, Jiaqi Wang, Xipeng Qiu, and Dahua Lin. VideoRoPE: What makes for good video rotary position embedding? In International Conference on Mac<sup>h</sup>ine Learning (ICML), 2025. Oral; arXiv:2502.05173.

[94] Tong Wu, Shuai Yang, Ryan Po, Yinghao Xu, Ziwei Liu, Dahua Lin, and Gordon Wetzstein. Video world models with long-term spatial memory. In Advances in Neural Information Processing Systems (NeurIPS), 2025. arXiv:2506.05284,

project page https://spmem.github.io.

[95] Xindi Wu, Uriel Singer, Zhaojiang Lin, Andrea Madotto, Xide Xia, Yifan Xu, Paul Crook, Xin Luna Dong, and Seungwhan Moon. Corgi: Cached memory guided video generation. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 4585–4594. IEEE, 2025.

[96] Xindi Wu, Despoina Paschalidou, Jun Gao, Antonio Torralba, Laura Leal-Taixé, Olga Russakovsky, Sanja Fidler, and Jonathan Lorraine. Motion attribution for video generation. In International Conference on Mac<sup>h</sup>ine Learning (ICML), 2026. arXiv:2601.08828.

[97] Yuhuai Wu, Markus N. Rabe, DeLesley Hutchins, and Christian Szegedy. Memorizing transformers. In International Conference on Learning Representations (ICLR), 2022.

[98] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Eficient streaming language models with attention sinks. In International Conference on Learning Representations (ICLR), 2024.

[99] Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Junxian Guo, Shang Yang, Haotian Tang, Yao Fu, and Song Han. DuoAttention: Eficient long-context LLM inference with retrieval and streaming heads. In International Conference on Learning Representations (ICLR), 2025. arXiv:2410.10819.

[100] Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. WorldMem: Long-term consistent world simulation with memory. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

[101] Kevin Xie, Jonathan Lorraine, Tianshi Cao, Jun Gao, James Lucas, Antonio Torralba, Sanja Fidler, and Xiaohui Zeng. LATTE3D: Large-scale amortized text-to-enhanced 3D synthesis. In European Conference on Computer Vision (ECCV), 2024.

[102] Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, and Vikas Singh. Nyströmformer: A Nyström-based algorithm for approximating self-attention. In AAAI Conference on Artificial Intelligence (AAAI), 2021.

[103] Boxun Xu, Yuming Du, Zichang Liu, Siyu Yang, Ziyang Jiang, Siqi Yan, Rajasi Saha, Albert Pumarola, Wenchen Wang, and Peng Li. Sparse forcing: Native trainable sparse attention for real-time autoregressive difusion video generation. arXiv preprint arXiv:2604.21221, 2026.

[104] Tian-Xing Xu, Zi-Xuan Wang, Guangyuan Wang, Li Hu, Zhongyi Zhang, Peng Zhang, Bang Zhang, and Song-Hai Zhang. UCM: Unifying camera control and memory with time-aware positional encoding warping for world models. arXiv preprint arXiv:2602.22960, 2026.

[105] Yang Yang, Tianyi Zhang, Wei Huang, Jinwei Chen, Boxi Wu, Xiaofei He, Deng Cai, Bo Li, and Peng-Tao Jiang. Anchor forcing: Anchor memory and tri-region RoPE for interactive streaming video difusion. arXiv preprint arXiv:2603.13405, 2026.

[106] Yixuan Ye, Xuanyu Lu, Yuxin Jiang, Yuchao Gu, Rui Zhao, Qiwei Liang, Jiachun Pan, Fengda Zhang, Weijia Wu, and Alex Jinpeng Wang. MIND: Benchmarking memory consistency and action control in world models. arXiv preprint arXiv:2602.08025, 2026.

[107] Hidir Yesiltepe, Tuna Han Salih Meral, Adil Kaan Akan, Kaan Oktay, and Pinar Yanardag. ∞-RoPE: Action-controllable infinite video generation emerges from autoregressive self-rollout. In Conference on Computer Vision and Pattern Recognition (CVPR), 2026. arXiv:2511.20649.

[108] Jung Yi, Wooseok Jang, Paul Hyunbin Cho, Jisu Nam, Heeji Yoon, and Seungryong Kim. Deep forcing: Training-free long video generation with deep sink and participative compression. arXiv preprint arXiv:2512.05081, 2025.

[109] Tianwei Yin, Qiang Zhang, Richard Zhang, William T. Freeman, Frédo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video difusion models. In Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[110] Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. arXiv preprint arXiv:2506.03141, 2025.

[111] Wei Yu, Runjia Qian, Yumeng Li, Liquan Wang, Songheng Yin, Sri Siddarth P. Chakaravarthy, Dennis Anthony, Yang Ye, Yidi Li, Weiwei Wan, and Animesh Garg. MosaicMem: Hybrid spatial memory for controllable video world models. arXiv preprint arXiv:2603.17117, 2026.

[112] Yifei Yu, Xiaoshan Wu, Xinting Hu, Tao Hu, Yang-Tian Sun, Xiaoyang Lyu, Bo Wang, Lin Ma, Yuewen Ma, Zhongrui Wang, and Xiaojuan Qi. VideoSSM: Autoregressive long video generation with hybrid state-space memory. arXiv preprint arXiv:2512.04519, 2025.

[113] Lvmin Zhang, Shengqu Cai, Muyang Li, Gordon Wetzstein, and Maneesh Agrawala. Frame context packing and drift prevention in next-frame-prediction video difusion models. In Advances in Neural Information Processing Systems (NeurIPS), 2025. Spotlight; arXiv:2504.12626.

[114] Xuan Zhang, Cunxiao Du, Chao Du, Tianyu Pang, Wei Gao, and Min Lin. SimLayerKV: A simple framework for layer-level KV cache reduction. arXiv preprint arXiv:2410.13846v1, 2024.

[115] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, and Beidi Chen. H<sub>2</sub>O: Heavy-hitter oracle for eficient generative inference of large language models. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

[116] Min Zhao, Guande He, Yixiao Chen, Hongzhou Zhu, Chongxuan Li, and Jun Zhu. RIFLEx: A free lunch for length extrapolation in video difusion transformers. In International Conference on Mac<sup>h</sup>ine Learning (ICML), 2025. arXiv:2502.15894.

[117] Zengqun Zhao, Yanzuo Lu, Ziquan Liu, Jifei Song, Jiankang Deng, and Ioannis Patras. Relax forcing: Relaxed KV-memory for consistent long video generation. arXiv preprint arXiv:2603.21366, 2026.

[118] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu. VBench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.

[119] Yuhao Zhou, Sirui Song, Boyang Liu, Zhiheng Xi, Senjie Jin, Xiaoran Fan, Zhihao Zhang, Wei Li, and Xuanjing Huang. EliteKV: Scalable KV cache compression via RoPE frequency selection and joint low-rank projection. arXiv preprint arXiv:2503.01586, 2025.

[120] Haoyi Zhu, Haozhe Liu, Yuyang Zhao, Tian Ye, Junsong Chen, Jincheng Yu, Tong He, Song Han, and Enze Xie. SANA-WM: Eficient minute-scale world modeling with hybrid linear difusion transformer. arXiv preprint arXiv:2605.15178, 2026.

## Appendices

Notation 19   
W T as Structured Sparse Attention 20   
B.1 Full and Projected Attention 20   
B.2 Structured Attention Approximations Unified by � 20   
B.3 Approximation Quality and Error Decomposition 21   
B.4 Reduction to Nonnegative Matrix Factorization 22   
B.5 When Are WorldTrace Projections Close to Optimal? 22   
C Additional Properties 23   
D Related Work 25   
D.1 Memory-Augmented Transformers 25   
D.2 KV Cache Compression for Large Language Models 25   
D.3 Position Extrapolation for RoPE . 25   
D.4 Autoregressive Video World Models 26   
D.5 Memory and KV Cache for Video World Models 26   
E Additional Experimental Results 27   
E.1 Unified Cache Allocation 27   
E.2 Training-Free Positional Baselines 27   
E.3 Scene Consistency Episodic Recall Sweep 28   
E.4 Cross-Architecture Experiments 28   
E.5 Slot Allocation . 29   
E.6 Memory Accounting 29   
E.7 Streaming Summary Writer . 29   
: Loop Memory Benchmark 30   
Implementation Details 30   
G.1 Model Architecture and Self-Forcing Background . 30   
G.2 WorldTrace Algorithm and Baselines 32   
G.3 Hyperparameters, Compute, and Evaluation Protocol 32   
H Discussion 34   
H.1 Limitations . 34   
H.2 Future Directions 34   
H.3 Broader Impact 34

Table 6: Glossary and notation.
<table><tr><td>Symbol</td><td>Description</td></tr><tr><td colspan="2">Cache Structure</td></tr><tr><td> $N$ </td><td>Total number of AR chunks in a rollout (generation length,  $F$  latent frames per chunk)</td></tr><tr><td> $N _ { s }$ </td><td>Number of summary slots in the KV cache (one latent frame each)</td></tr><tr><td> $N _ { r }$ </td><td>Number of recent-window slots (one verbatim latent frame each)</td></tr><tr><td> $L _ { \mathrm { a t t } }$ </td><td>Local attention / inference KV window in latent frames  $( N _ { s } + N _ { r } = L _ { \mathrm { a t t n } } )$ </td></tr><tr><td> $L _ { \mathrm { t r a i n } }$ </td><td>Training-time KV-cache extent in AR chunks during Self-Forcing rollouts (distinct from</td></tr><tr><td></td><td>the local attention window  $L _ { \mathrm { a t t r a } } )$ </td></tr><tr><td> $F$   $M$ </td><td>Latent frames per AR chunk Source frames compressed into one summary slot under WoRLDTRACE-FIELD</td></tr><tr><td> $_ T$ </td><td>Total past latent frames at the current chunk (grows linearly with N). The newest  $N _ { r }$ </td></tr><tr><td></td><td>stay verbatim, and the rest are compressed into the  $N _ { s }$  summary slots</td></tr><tr><td> $s$   $\mathcal { R }$ </td><td>Summary slot index  $\mathfrak { c } , s = 0 , \ldots , N _ { s } - 1 ( 0 = \mathrm { o l d e s t } )$ </td></tr><tr><td> $s$ </td><td>Recent-window cache: verbatim KVs of the newest  $N _ { r }$  latent frames</td></tr><tr><td></td><td>Summary cache:  $N _ { s }$  compressed slots indexed by s</td></tr><tr><td>Indices</td><td></td></tr><tr><td> $f$ </td><td>RoPE frequency-pair index</td></tr><tr><td> $m$   $_ n$ </td><td>Source-frame iteration index (inside  $\Sigma _ { m = 1 } ^ { M } )$ </td></tr><tr><td colspan="2">AR-chunk index Virtual Position Assignment (Slot Indexing)</td></tr><tr><td></td><td></td></tr><tr><td>q</td><td>Absolute timestamp of the current query frame</td></tr><tr><td> $k$ </td><td>Absolute timestamp of a cached key frame</td></tr><tr><td> $t _ { \mathrm { m i n } } ^ { v }$ </td><td>In-distribution lower bound: max(0,  $q - \Delta t _ { \mathrm { t r a i n } } )$ </td></tr><tr><td> $t _ { \mathrm { m a x } } ^ { v }$ </td><td>In-distribution upper bound for summary slots:  $q - N _ { r }$ </td></tr><tr><td> $t ^ { v }$ </td><td>Virtual position</td></tr><tr><td> $t _ { s } ^ { v }$ </td><td>Virtual position of slot s:  $q - ( L _ { \mathrm { a t t n } } - 1 - s )$ </td></tr><tr><td colspan="2">LoOPBENCH Benchmark</td></tr><tr><td> $K$ </td><td>Number of intermediate waypoints between departure and return to scene A</td></tr><tr><td> $R$ </td><td>Multi-revisit depth: number of times the repeated waypoint is visited in one path  $( e . g .$ </td></tr><tr><td colspan="2">ABABA revisits scene A twice) RoPE Parameters</td></tr><tr><td> $\theta$ </td><td>RoPE base frequency  $\scriptstyle ( \theta = 1 0 0 0 0$  for MG2)</td></tr><tr><td> $\theta _ { f }$ </td><td>RoPE angular frequency for temporal head-dimension pair  $f \ ( \theta _ { f } = \theta ^ { - f / c _ { t } } )$ </td></tr><tr><td> $c _ { t } , c _ { h } , c _ { w }$ </td><td>Number of temporal, height, and width RoPE complex pairs per head</td></tr><tr><td> $n _ { \ell }$ </td><td>Number of transformer layers</td></tr><tr><td> $R ( \alpha )$ </td><td>Rotation matrix by angle α (applied per frequency pair)</td></tr><tr><td colspan="2"> $K e y / Q u e r y$  Variants</td></tr><tr><td> $K _ { + } ^ { f }$ </td><td>frequency pair f</td></tr><tr><td> $\smash { \cdot } \to _ { t } ^ { t }$ </td><td>RoPE-rotated key at absolute position  $t _ { m } ,$ </td></tr><tr><td> $K _ { m } ^ { f }$ </td><td>Canonical (unrotated) key content:  $R ( - \theta _ { f } t _ { m } ) \ K _ { t _ { m } } ^ { f }$ </td></tr><tr><td> $\bar { K } ^ { f }$ </td><td>Canonical mean across M source frames</td></tr><tr><td> $\bar { K } _ { \mathrm { n a i v e } } ^ { f }$ </td><td>Naive RoPE-space average of M rotated keys</td></tr><tr><td> $K _ { \mathrm { f i e l d } } ^ { f } ( t ^ { v } )$ </td><td>WoRLDTRACE-FIELD compressed key at virtual position  $t ^ { v }$ </td></tr><tr><td> $K _ { \mathrm { l a n d } } ^ { f } ( t ^ { v } )$ </td><td>WoRLDTRACE-LANDMARK frozen canonical key at virtual position  $t ^ { v }$ </td></tr><tr><td> $t _ { \ell ^ { * } }$ </td><td>Original timestamp of a selected landmark frame</td></tr><tr><td> $K _ { \ell ^ { * } } ^ { f }$ </td><td>Landmark source key at  $t _ { \ell ^ { * } }$  , frequency pair f</td></tr><tr><td> $Q _ { q } ^ { f }$ </td><td>RoPE-rotated query at q, frequency pair f</td></tr><tr><td colspan="2">Attention Quantities</td></tr><tr><td></td><td>Query-key temporal offset:  $q - k$ </td></tr><tr><td> $\delta _ { q , k }$   $\Delta t _ { \mathrm { t r a i n } }$ </td><td>Largest trained temporal offset  $( \Delta t _ { \mathrm { t r a i n } } { = } 5 $  for MG2,  $L _ { \mathrm { a t t n } } { = } \Delta t _ { \mathrm { t r a i n } } { + } 1 )$ </td></tr><tr><td> $a _ { q _ { \mathfrak { s } } k } ^ { f }$ </td><td>Attention-score contribution from frequency pair  $f$  at offset  $\delta _ { q , k }$ </td></tr><tr><td> $A _ { q , k } ^ { J }$ </td><td>Query-key content term in canonical coordinates for  $a _ { q , k } ^ { f }$ </td></tr><tr><td> $\alpha _ { q }$ </td><td>Full attention weights for query  $, q$  (softmax over all T past frames):  $\alpha _ { q }$  ∈</td></tr><tr><td> $\mathcal { Q }$ </td><td> $\Delta _ { T }$  Query set in the aggregate mismatch objective (App. B)</td></tr></table>

## B WorldTrace as Structured Sparse Attention

Under autoregressive decoding with full attention over the history, each new chunk attends to all $T$ past frames, so both the per-step attention cost and the KV cache grow as $O ( T )$ with generation length. Video world models typically cap this with a fixed-size sliding window over the most recent frames. Our WorldTrace instead introduces structured sparse approximations that aim to compress the full �-frame history to $L _ { \mathrm { a t t n } } \ll T$ efective tokens via a projection matrix $P \in \mathbb { R } ^ { L _ { \mathrm { a t t n } } \times T }$ , where each row defines which frames contribute to one compressed slot. Throughout this section, � denotes the canonical (unrotated) key matrix, which is the domain in which the canonical-store operator applies $P .$ . The slot-rank re-encoding of Def. 1 is applied afterwards at attention time, which is the same fixed, in-distribution rotation on the compressed cache.

## B.1 Full and Projected Attention

For a single query $q ,$ the standard attention output is $\begin{array} { r } { A ( q , K , V ) = \operatorname { s o f t m a x } ( q K ^ { \top } ) V = \alpha _ { q } V } \end{array}$ , a weighted sum over all $T$ value vectors with $\alpha _ { q } \in \Delta _ { T }$ , where $\Delta _ { T }$ is the probability simplex over the $T$ past frames (non-negative weights summing to one). The projected attention approximation replaces the �-token sequence with $L _ { \mathrm { a t t n } } \ll T$ prototype tokens (one per compressed slot) defined by a row-stochastic matrix $P \in \mathbb { R } ^ { L _ { \mathrm { a t t n } } \times T }$ meaning each row of $P$ is itself a distribution over past frames (this property is what makes the bounds of Sec. B.3 non-vacuous):

$$
\begin{array} { r } { \widehat { A } ( q , K , V ; P ) = \operatorname { s o f t m a x } ( q ( P K ) ^ { \top } ) P V , } \end{array}\tag{5}
$$

where $\hat { \alpha } _ { q } = \mathrm { s o f t m a x } ( q ( P K ) ^ { \top } ) \in \Delta _ { L _ { \mathrm { a t t r } } }$ are the prototype attention weights, now a distribution over the $L _ { \mathrm { a t t n } }$ compressed slots rather than the � frames, and $P _ { i } V$ is the compressed value for slot �.

Relation to prior structured-attention approximations. Eq. (5) is the low-rank factorization of attention studied by eficient transformer architectures, whose projections are learned or trained end-to-end [89, 102, 86, 6, 85, 45]. WorldTrace difers mainly in that the projection is applied training-free at inference time to a pretrained model’s KV cache. Beyond this, two design choices follow from the RoPE-based world-model setting rather than marking a contrast with these methods. First, � acts in the canonical key domain, so merging avoids the phase cancellation (Sec. 2.2). Second, the compressed slots are re-encoded at slot-rank virtual positions (Def. 1), keeping them addressable beyond the training horizon. The row structure of $P$ (uniform averaging for WorldTrace-Field, one-hot selection for WorldTrace-Landmark) then determines whether compression favors coherence or recall, as formalized below.

## B.2 Structured Attention Approximations Unified by �

The four cache structures (Fig. 6) compared in our experiments correspond to four structured choices of $P .$ Sliding window attention. Here $P = \left[ 0 \quad I _ { L \mathrm { a t t n } } \right]$ makes each row a one-hot vector selecting one of the most recent $L _ { \mathrm { a t t n } }$ frames, discarding all earlier history (Fig. 6a). Adding an attention sin<sup>k</sup> anc<sup>h</sup>or is a simple modification (Fig. 6b) in which row 0 is pinned to the first (sink) frame while the remaining $L _ { \mathrm { a t t n } } { - } 1$ rows select the most recent $L _ { \mathrm { a t t n } } { - } 1$ frames verbatim, always keeping a global context token in cache regardless of generation length. This anchor mitigates the collapse of the attention distribution onto only local tokens at long generation horizons [98].

WorldTrace-Field. Rows corresponding to the $N _ { r }$ recent slots are identity rows as above. Each remaining row is a uniform average over the � frames assigned to its summary slot (Fig. 6c), setting $P _ { i , t } = 1 / M$ if frame � is assigned to slot � and 0 otherwise.

W T L . Summary rows are one-hot selectors at the detected scene-entry frames (Fig. 6d), so the row storing the landmark at timestamp $t _ { \ell ^ { * } }$ has $P _ { i , t } = 1$ for $t = t _ { \ell } ,$ \* and 0 otherwise, simply copying that stored key, $P _ { i } K = K [ t _ { \ell ^ { * } } ]$ , matching the frozen key of Eq. (4). The rows update as generation proceeds, with a new scene-entry frame overwriting a summary row and, once all rows are full, the oldest landmark evicted to make room for the newest. These choices span a spectrum of $P \mathrm { \cdot }$ row structures, from sparse (one-hot selection) to dense (uniform averaging).

![](images/06557a9e329be63d1f372e51ca3e49c25ad321e4d2924b4406d2bca0291242a4.jpg)  
� frames

![](images/b0d667d719ed50e9e0c0e1de0ecf9e77b8844a120e93546d4b7903889f3e6965.jpg)  
(a) Sliding window  
� frames  
(b) + attention sink

![](images/77a9b878b8824bd54edb794be5334eec3f60f0e2d444cd435d9459ef5c8f174c.jpg)  
� frames  
(c) WorldTrace-Field

![](images/daa1e30fd12e5f1d0d798a1acb98720cf4175789b2a6c0321ecdffd60a13daf3.jpg)  
(d) WorldTrace-Landmark

Figure 6: Projection matrices � for the four cache structures. (a) Sliding window: last $L _ { \mathrm { a t t n } }$ frames verbatim.   
(b) Sliding window wit<sup>h</sup> attention sin<sup>k</sup>: initial frame held as an attention sink, remaining slots verbatim recent.   
(c) WorldTrace-Field: recent rows verbatim newest, summary slots each average a group of earlier frames.   
(d) WorldTrace-Landmark: summary rows are one-hot at scene-entry frames, recent rows verbatim.

## B.3 Approximation Quality and Error Decomposition

The cache contents, and hence �, change as generation proceeds, since the recent window slides and summary rows are filled or overwritten. All statements in this and the following subsections therefore refer to one generation step at a time, where � and � denote the history length and the projection realized by the cache at the current step, and � denotes the queries of the chunk being generated (one attention row per query token and head). Every bound holds at each step separately, and no single static � is claimed to serve all steps.

Two sources of error. The compressed cache difers from an unbounded cache in two independent ways: keys are repositioned (re-encoded at slot-rank virtual positions rather than their original timestamps), and keys are compressed (merged through �). Writing $A _ { \mathrm { o r i g } }$ for full attention with keys at their original positions, � for full attention with keys re-encoded at virtual positions, and $\widehat { A }$ for the projected attention of Eq. (5) on the re-encoded cache, the total error decomposes as:

$$
\begin{array} { r } { \| \widehat { \mathcal { A } } - \mathcal { A } _ { \mathrm { o r i g } } \| \ \leq \underbracket { \| \mathcal { A } - \mathcal { A } _ { \mathrm { o r i g } } \| } _ { \mathrm { r e p o s i t i o n i n g } } + \underbracket { \| \widehat { \mathcal { A } } - \mathcal { A } \| } _ { \mathrm { c o m p r e s s i o n } } . } \end{array}
$$

The repositioning term is not an artifact of compression but the trade-of for addressability. Beyond the training horizon the original ofsets are out-of-distribution and attention degrades outright (Sec. 2.1), so slot-rank re-encoding trades an uncontrolled OOD failure for a bounded, in-distribution position shift. We do not bound this term analytically, but it is measured empirically by the position ablation (Tab. 1), which varies only the position assignment with the writer held fixed. Prop. 1 below bounds the compression term. All statements in this section are relative to the repositioned reference �.

For a single query �, let $\hat { \alpha } _ { q } = \mathrm { s o f t m a x } ( q ( P K ) ^ { \top } ) \in \Delta _ { L _ { \mathrm { a t t n } } }$ be the prototype attention weights under the compressed cache, and define the output error $r _ { q } ( P ) = \widehat { \mathcal { A } } ( q , K , V ; P ) - \mathcal { A } ( q , K , V )$ Proposition 1 (Approximation bound).

$$
\| r _ { q } ( P ) \| _ { 2 } \ \leq \ \underbrace { \| \hat { \alpha } _ { q } P - \alpha _ { q } \| _ { 1 } } _ { d i s t r i b u t i o n \ m i s m a t c h } \cdot \| V \| _ { \infty , \mathrm { r o w } } ,
$$

<sub>w</sub>h<sub>ere</sub> $\| V \| _ { \infty , \operatorname { r o w } } = \operatorname* { m a x } _ { t } \| V [ t ] \| _ { 2 } .$

Proof. By associativity $\begin{array} { r } { \widehat { \mathcal { A } } = \widehat { \alpha } _ { q } ( P V ) = ( \widehat { \alpha } _ { q } P ) V , { s o } r _ { q } ( P ) = ( \widehat { \alpha } _ { q } P - \alpha _ { q } ) V = \sum _ { t } c _ { t } V [ t ] } \end{array}$ with $c = \hat { \alpha } _ { q } P - \alpha _ { q } .$ . The vector-valued Hölder inequality $( \ell _ { 1 } / \ell _ { \infty }$ duality) on the rows of � then gives $\| r _ { q } ( P ) \| _ { 2 } \leq \| c \| _ { 1 } \| V \| _ { \infty , \mathrm { r o w } }$

The bound motivates minimizing distribution mismatch $\| \hat { \alpha } _ { q } P - \alpha _ { q } \| _ { 1 }$ independently of value geometry. For WorldTrace-Field, Prop. 2 (stated later in App. C) ensures the compressed key $K _ { \mathrm { f i e l d } } ^ { f } ( t ^ { v } )$ reproduces the mean attention score of its source keys at the shared virtual position, limiting per-slot mismatch when query preferences vary smoothly across a group. For WorldTrace-Landmark, a verbatim scene-entry key keeps the mismatch small for queries concentrated on that event, the recall condition of Eq. (8) in Sec. B.5.

## B.4 Reduction to Nonnegative Matrix Factorization

Nonnegative matrix factorization (NMF) approximates a nonnegative matrix as the product of two low-rank nonnegative factors [44], which form a compact summary of the original data. Compressing attention through $P$ is an instance of this problem. Treat $\hat { \alpha } _ { q } \in \Delta _ { L _ { \mathrm { a t t n } } }$ as a free variable (decoupling it from the softmax in Eq. (5)), and define $\mathbf { A } \in \mathbb { R } ^ { | \mathcal { Q } | \times T }$ as the matrix whose rows are the full attention distributions $\alpha _ { q } ,$ one row per query in $\mathcal { Q } ,$ and $\mathbf { B } \in \mathbb { R } ^ { | \mathcal { Q } | \times L _ { \mathrm { a t t n } } }$ as the matrix whose rows are the corresponding prototype weights $\hat { \alpha } _ { q }$ . The aggregate distribution-mismatch objective then becomes:

$$
\operatorname* { m i n } _ { \mathbf { B } \geq 0 , \cal P \geq 0 } \| \mathbf { B } \boldsymbol { P } - \mathbf { A } \| _ { 1 } \quad \mathrm { s u b j e c t ~ t o ~ r o w { \cdot } s u m ~ c o n s t r a i n t s ~ o n ~ } \mathbf { B } \mathrm { ~ a n d ~ } \boldsymbol { P } ,
$$

which is NMF with an $\ell _ { 1 }$ fidelity term, in the spirit of robust NMF formulations [43]. This relaxation serves the analysis only, because at inference $\hat { \alpha } _ { q }$ is not free but determined by $P$ through the softmax, $\hat { \alpha } _ { q } = \operatorname { s o f t m a x } ( q ( P K ) ^ { \top } )$ for each query $q$ and key matrix $K _ { \ast }$ , and Sec. B.5 accounts for the gap through the lower-bound relation.

Each structural constraint on $P$ then recovers a WorldTrace variant and a known clustering algorithm, spanning the sparse-to-dense spectrum. The uniform-partition rows of WorldTrace-Field are a fixedassignment, $\ell _ { 1 }$ (total-variation) analog of �-means with cluster-mean prototypes, with the time-ordered partition replacing the learned assignment step. The $0 / 1$ rows of WorldTrace-Landmark, with $\hat { \alpha } _ { q }$ constrained to hard assignment, reduce to �-medoids on attention distributions, for which the canonical-key cosine-distance rule is a lightweight single-pass $O ( 1 )$ -per-chunk surrogate.

## B.5 When Are WorldTrace Projections Close to Optimal?

The NMF objective above is deliberately broad, since without assumptions on the query distribution no fixed $L _ { \mathrm { a t t n } } { \mathrm { - r o w } }$ projection can be uniformly optimal. Any reachable compressed distribution has the form $\gamma P$ , where the slot-weight vector $\gamma \in \Delta _ { L _ { \mathrm { a t t r } } }$ mixes the $L _ { \mathrm { a t t n } }$ rows of $P .$ Since each row is a distribution over the � frames, the $L _ { \mathrm { a t t n } }$ rows together place total weight $L _ { \mathrm { a t t n } }$ across the $T$ columns. Averaged over the columns, some frame � therefore receives at most max $P _ { i , t } \leq L _ { \mathrm { a t t n } } / T$ , so a query concentrating on that one frame has mismatch $\| \gamma P - \mathbf { e } _ { t } \| _ { 1 } = 2 \big ( 1 - ( \gamma P ) _ { t } \big ) \geq 2 ( 1 - L _ { \mathrm { a t t n } } / T )$ for every �. Intuitively, the compressed cache can place at most an $L _ { \mathrm { a t t n } } / T$ fraction of its attention on any single frame, so it cannot reproduce a query that focuses on one frame, and the error grows as the compression ratio $L _ { \mathrm { a t t n } } / T$ shrinks. A compressed cache therefore needs to choose which parts of history to preserve. We define one target query family per variant, a definition rather than an assumption about the pretrained model, and show each variant near-optimal on its own family under the following relaxed mismatch objective, the total $\ell _ { 1 }$ gap between each query’s true attention $\alpha _ { q }$ and the closest distribution reachable from the $L _ { \mathrm { a t t n } }$ compressed slots:

$$
\mathcal { I } ( P ) = \sum _ { q \in \mathcal { Q } } \operatorname* { m i n } _ { \gamma _ { q } \in \Delta _ { L _ { \mathrm { a t t n } } } } \| \gamma _ { q } P - \alpha _ { q } \| _ { 1 } .
$$

Here $\gamma _ { q }$ is chosen freely per query. At inference the slot weights are instead fixed by the softmax, $\hat { \alpha } _ { q } =$ softmax $( q ( P K ) ^ { \top } )$ , so $\mathcal { I } ( P )$ is a lower bound on the true inference objective, and the bounds we derive are for this relaxed version.

Recent-window mass. Split the attention as $\alpha _ { q } = ( \alpha _ { q } ^ { \mathrm { \scriptsize ~ o l d } } , \alpha _ { q } ^ { \mathrm { \scriptsize ~ r e c } } )$ , the weights on old and on recent frames. Both projections keep the $N _ { r }$ most recent frames as identity rows, which reproduce ${ \alpha _ { q } } ^ { \mathrm { { ~ r e c } } }$ exactly, so compression error arises only on old history. The conditions below therefore involve only the old-history attention, normalized to the distribution $\bar { \alpha } _ { q } ^ { \mathrm { o l d } } = { \alpha _ { q } } ^ { \mathrm { o l d } } / \| { \alpha _ { q } } ^ { \mathrm { o l d } } \| _ { 1 }$

Coherent queries: group-level attention. Partition the old history into $N _ { s }$ contiguous temporal groups, one per summary slot, and let $u _ { i }$ be the uniform distribution over the frames in group �. Call a query $q \ \varepsilon _ { q } - c o h e r e n t$ if its old-history attention depends on groups rather than individual frames, meaning there exists a distribution

$a _ { q } \in \Delta _ { N _ { s } }$ such that:

$$
{ \left\| \bar { \alpha } _ { q } ^ { \mathrm { o l d } } - \sum _ { i } { a _ { q , i } u _ { i } } \right\| } _ { 1 } \le \varepsilon _ { q } ,\tag{6}
$$

which is the coherence condition. For a query set $\mathcal { Q }$ of $\varepsilon _ { q }$ -coherent queries, the WorldTrace-Field projection $P _ { \mathrm { f i e l d } }$ satisfies:

$$
\mathcal { I } ( P _ { \mathrm { f i e l d } } ) \leq \sum _ { \boldsymbol { q } \in \mathcal { Q } } \| { \boldsymbol { \alpha } } _ { \boldsymbol { q } } ^ { \mathrm { o l d } } \| _ { 1 } \varepsilon _ { \boldsymbol { q } } \leq \sum _ { \boldsymbol { q } \in \mathcal { Q } } \varepsilon _ { \boldsymbol { q } } .\tag{7}
$$

Choosing $\gamma _ { q } = \left( \lVert \alpha _ { q } ^ { \mathrm { \tiny ~ o l d } } \rVert _ { 1 } a _ { q } , \ \alpha _ { q } ^ { \mathrm { \tiny ~ r e c } } \right)$ copies ${ \alpha _ { q } } ^ { \mathrm { { ~ r e c } } }$ exactly and leaves old-history error at most $\| \boldsymbol { \alpha _ { q } } ^ { \mathrm { ~ o l d } } \| _ { 1 } \varepsilon _ { q }$ . Since $\mathcal { I } ( P ) \ge 0$ for any projection, WorldTrace-Field is within $\textstyle \sum _ { q } \varepsilon _ { q }$ of the best possible projection for this query family. The canonical averaging produces these uniform rows in a RoPE-invariant domain, keeping each group mean addressable at its virtual position.

Recall queries: sparse scene-entry attention. For episodic recall, a return query wants one specific past scene rather than a smooth summary. At most $N _ { s }$ landmarks are stored. Call a query $q \delta _ { q } .$ -covered if the landmark frame it targets, written $m ( q )$ , satisfies:

$$
\| \bar { \alpha } _ { q } ^ { \mathrm { o l d } } - \mathbf { e } _ { m ( q ) } \| _ { 1 } \leq \delta _ { q } ,\tag{8}
$$

where ${ \bf e } _ { m \left( q \right) }$ is the one-hot distribution on that frame. Eqn. (8) is the recall condition. For a query set $\mathcal { Q }$ of $\delta _ { q }$ -covered queries, the WorldTrace-Landmark projection $P _ { \mathrm { l a n d m a r k } }$ satisfies:

$$
\mathcal { I } ( P _ { \mathrm { l a n d m a r k } } ) \leq \sum _ { \boldsymbol { q } \in \mathcal { Q } } \| { \boldsymbol { \alpha _ { \boldsymbol { q } } } } ^ { \mathrm { o l d } } \| _ { 1 } \delta _ { \boldsymbol { q } } \leq \sum _ { \boldsymbol { q } \in \mathcal { Q } } \delta _ { \boldsymbol { q } } .\tag{9}
$$

The construction is the same with ${ \bf e } _ { s ( q ) }$ in place of $a _ { q } .$ , where $s ( q )$ is the landmark slot whose row copies frame $m ( q )$ , so WorldTrace-Landmark is close to the best one-hot projection whenever the detector forms a �-cover of the recall targets. The oldest-out scene-entry rule of our experiments is a streaming approximation to this cover, reliable when at most $N _ { s }$ scenes are simultaneously relevant.

What these bounds do and do not claim. Eq. (7) and Eq. (9) are conditional characterizations, true by construction for the query families that Eq. (6) and Eq. (8) define, not unconditional optimality claims. We do not measure $\varepsilon _ { q }$ or $\delta _ { q }$ on the pretrained model, since that would require uncompressed rollouts past the training horizon, where Sec. 2.1 shows the reference attention itself degrades. Each condition is instead tested through the behavior it predicts.

Future directions. The NMF framing points to several natural extensions, including online �-means or streaming coresets [22] that adapt partition boundaries at inference time, soft-sparse rows that interpolate between averaging and verbatim selection, and a directly optimized � that balances coherence and recall across all queries in a rollout.

## C Additional Properties

Temporal RoPE at long horizons. We provide temporal-RoPE background for the addressability failure in Sec. 2.1 and expand Eq. (3), summarized at the end of Sec. 3.3. We analyze which temporal RoPE frequencies stay in-distribution at long horizons in MG2-1.3B, and which reach rotation angles the model never saw in training, making their contribution to the attention score efectively random. MG2-1.3B inherits the 3D RoPE split of the Wan 2.1 backbone [87]. The 128-dimensional per-head embedding is divided across three position axes: $2 c _ { t } { = } 4 4$ dimensions for temporal position and $2 c _ { h } { = } 2 c _ { w } { = } 4 2$ dimensions each for spatial height and width, with shared base $\theta = 1 0 0 0 0$ . The �-th temporal frequency is $\theta _ { f } = \theta ^ { - f / c _ { t } }$ for $f = 0 , \ldots , c _ { t } { - } 1$ , ranging from $\theta _ { f } { = } 1 . 0$ at �=0 (fastest rotating) down to $\theta _ { f } \approx 1 . 5 \times 1 0 ^ { - 4 }$ (slowest).

Frequency � enters the attention score through $\cos ( \theta _ { f } | \delta _ { q , k } | )$ , where $\delta _ { q , k } = q - k$ is the query-key ofset. When $\theta _ { f } | \delta _ { q , k } | \ll \pi _ { ; }$ , the cosine stays near 1, and the component is position-invariant and carries semantic content. When $\theta _ { f } | \delta _ { q , k } | \gg \pi _ { \cdot }$ , the phase wraps past � many times, so the cosine takes an essentially arbitrary value at each ofset and the component stops signaling how far away the key is. The consequence is instability in the fastest components, where moving a key by a single ofset step shifts the phase by up to 1 rad, so the contribution these components make to the score can change substantially, even in sign, between neighboring ofsets, and recall degrades steadily over a long rollout. This wavelength-vs-context view follows YaRN’s NTK-by-parts framing [67], which likewise groups RoPE dimensions by how their wavelength compares to the context length and treats the short-wavelength (fast) and long-wavelength (slow) groups diferently.

![](images/f32170352bc1411423464ae51b5b45052c91e2cb184726c2ed542b8da3d25250.jpg)  
Figure 7: Fast RoPE components go out of distribution at long horizons. Per-frequency RoPE phase $\theta _ { f } | \delta _ { q , k } |$ on MG2-1.3B over frequency index $f$ and inference distance $\lvert \delta _ { q , k } \rvert$ . The log color scale marks the severity thresholds, and the dashed line shows the training max $\Delta t _ { \mathrm { t r a i n } } { = } 5$ . The plotted horizon is for illustration only, since $| \delta _ { q , k } |$ keeps growing in an unbounded rollout and the phases only increase.

Concretely, at the training max ofset $\Delta t _ { \mathrm { t r a i n } } = 5$ , components $f \geq 1 0$ satisfy $\theta _ { f } \times 5 < 0 . 0 8$ rad and act as near-semantic carriers. At a representative long-horizon ofset $| \delta _ { q , k } | = 3 0$ , the three fastest components $\left( f \leq 2 \right)$ all exceed $3 \pi \mathrm { r a d } _ { \cdot }$ , with $f { = } 3 \operatorname { a t } 8 . 5$ rad and $f { = } 5 \mathrm { a t } 3 . 7 \mathrm { r a d }$ . We show the phase $\theta _ { f } | \delta _ { q , k } |$ across all frequencies and inference distances in Fig. 7, over a plotted distance range chosen for illustration only, since in an unbounded rollout the ofset grows past it and pushes ever more components beyond $\pi .$ . The long-horizon failure is therefore driven by the fast components alone, while the slow components remain reliable semantic carriers even at large ofsets. This motivates the position-side intervention of WorldTrace, which places every compressed slot at an in-distribution virtual ofset and thereby keeps the fast components within their trained phase range at any horizon.

Mean attention preservation. The WorldTrace-Field compression preserves the mean attention score, so that a single summary token $K _ { \mathrm { f i e l d } } ^ { f } ( t ^ { v } )$ reproduces exactly the average of the scores its � source keys would produce if all were relocated to the shared virtual position $t ^ { v }$ , for every query and every temporal RoPE frequency. This matters because attention scores are the only channel through which the model reads the cache, so compression that silently changed them would bias the model toward or away from distant history, precisely the failure the pretrained attention never learned to correct.

Proposition 2 (Mean attention preservation). Fix a temporal RoPE pair � and a virtual position $t ^ { v } .$ . For any que<sup>r</sup>y $Q _ { q } ^ { f } ,$

$$
\big \langle Q _ { q } ^ { f } , K _ { \mathrm { f i e l d } } ^ { f } ( t ^ { v } ) \big \rangle = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \big \langle Q _ { q } ^ { f } , R ( \theta _ { f } t ^ { v } ) K _ { m } ^ { f } \big \rangle .
$$

Proof. By Eq. (3), $K _ { \mathrm { f i e l d } } ^ { f } ( t ^ { v } ) = R ( \theta _ { f } t ^ { v } ) \bar { K } ^ { f }$ with $\begin{array} { r } { \bar { K } ^ { f } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } K _ { m } ^ { f } } \end{array}$ . Since the rotation $R ( \theta _ { f } t ^ { v } )$ is linear and the inner product is linear in its second argument,

$$
\left. Q _ { q } ^ { f } , K _ { \mathrm { f i e l d } } ^ { f } ( t ^ { v } ) \right. = \left. Q _ { q } ^ { f } , R ( \theta _ { f } t ^ { v } ) \left( \frac { 1 } { M } \sum _ { m = 1 } ^ { M } K _ { m } ^ { f } \right) \right. = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \langle Q _ { q } ^ { f } , R ( \theta _ { f } t ^ { v } ) K _ { m } ^ { f } \rangle .
$$

Consequently, the compressed representation separates content from position by design, storing only the canonical content average $\bar { K } ^ { f }$ in a slot whose temporal identity comes entirely from its virtual position $t ^ { v } .$ . This separation is exactly what slot-rank addressing needs, because the stored content carries no leftover phase from its source timestamps, letting WorldTrace place a slot at any virtual position (Def. 1) where a query reads the same content. Prop. 2 also pins down this design by showing that any linear operator preserving the mean attention score for every query must output the same canonical average $\bar { K } ^ { f }$ , leaving only the virtual position $t ^ { v }$ free, which WorldTrace sets by slot rank (Sec. 3.1). Constraining an operator through the structure of the RoPE rotation it inverts has antecedents in structured-Jacobian parameterizations [57]. One side efect of averaging is a norm reduction, since mutually uncorrelated canonical keys average to roughly $1 / \sqrt { M }$ times the norm of a single key, so a summary slot enters the softmax with a smaller logit than a verbatim key. We keep this behavior rather than restoring the norm with a $\sqrt { M }$ rescale, both because the smaller logit softly downweights heavily compressed history relative to the verbatim recent window and because the rescale did not help at long horizons.

## D Related Work

## D.1 Memory-Augmented Transformers

Memory-augmented transformers pair a recent verbatim window with a compressed record of older context, realized as segment-level recurrence with relative positions [19], a compressed memory of older tokens [71], external �NN memory [97], learned memory tokens passed between segments [7, 39], and a fused sliding window plus compressive memory [64]. WorldTrace adopts the same two-tier structure, with a recent verbatim window plus compressed summary slots. Landmark Attention [63] shares vocabulary with WorldTrace-Landmark, but its landmarks are trained representatives that gate attention to of-cache blocks at 1D position, whereas WorldTrace-Landmark is training-free, stores verbatim canonical-frame keys inside the $O ( 1 )$ cache, and freezes them against unrotate-and-rerotate drift in 3D RoPE. The broader spectrum toward compressed state spans linear attention [40], Mamba [26], and Test-Time Training [80]. We target the fixed-budget, training-free setting in AR video world models.

## D.2 KV Cache Compression for Large Language Models

LLM KV-cache methods fall into three families: window/eviction [4, 98, 115, 53, 9], merging [6, 83], and quantization [56], the last orthogonal to the position-OOD failure we target. DuoAttention [99] splits heads into retrieval and streaming families, parallel to our recent-window and summary-slot split. These methods change content only, picking which tokens to keep or merge rather than which virtual position a summary spanning several merged frames should occupy. The closest exceptions touch per-token positional structure [119, 31] or unrotate and rerotate individually retained keys [16, 65], but none reassign summary-slot positions. Our setting also difers from LLM compression in that keys accumulate across independently denoised AR video chunks, so importance must be ranked under denoising-timestep-conditioned queries. Under such queries, angular-concentration scores [61] lose the stable pre-RoPE key geometry they assume, and layer-budget reallocations [9, 114] tune which layers keep tokens rather than how summary slots are addressed at read time. WorldTrace-Field instead averages canonical keys and pairs the result with slot-rank positions.

## D.3 Position Extrapolation for RoPE

RoPE [78] degrades past trained relative ofsets. Existing fixes span index rescaling [13, 67, 74], alternative distance biases [69, 81], position-free training [41], and frequency- or content-aware encodings [24, 36], while vision and video extensions add 2D/3D splits [34, 93], camera-relative encodings [50], and training-free difusion extrapolation [116]. All of this remaps positions for individually retained tokens or fixed-length inputs. Our cache instead needs to be compressed, since a fixed $O ( 1 )$ budget cannot keep every past frame verbatim. Each merged summary then needs an explicit virtual position, a quantity that does not arise when tokens keep separate identities, and WorldTrace assigns slot-rank ofsets for that role without retraining.

Wang et al. [88] tie position to numerical precision, consistent with our coupling thesis (Sec. 3.5). UCM [104] and MosaicMem [111] warp 3D positions for camera-controlled worlds as geometry-driven, trained operators.

## D.4 Autoregressive Video World Models

Interactive AR video world models [25, 32, 20, 84] generate via chunk-wise difusion prediction, dominantly under the Self-Forcing paradigm [38, 10, 75, 109], which aligns training rollouts with the model’s own KV cache and inherits a local attention window not sized for arbitrarily long inference. Training-time horizon extensions [18, 37] are covered with memory-specific methods in Sec. D.5. Foundation backbones [87] and the world-model lineage from Ha and Schmidhuber [29] through Dreamer [30] to closed-loop driving simulators [3] motivate interactive rollouts beyond passive long-form generation.

## D.5 Memory and KV Cache for Video World Models

Inference-time KV cache. Closest to WorldTrace, inference-time methods modify cache read/write without retraining. Position-side fixes remap or cap temporal ofsets via block-relative RoPE with aggressive flushing [107], inference-time position adjustment [17, 47], or per-token virtual positions with cross-frame decay [48]. Content-side writers decide what to merge, prune, or retain, through re-aligned sink tokens [108], dual-rate EMA summaries [42], importance pruning [11, 46], and temporal-correspondence merging [73]. FlowCache [60] targets per-step compute rather than long-horizon recall, but its recent-summary structure parallels ours. These methods adjust the position side, the content side, or pair the two without designing them jointly. To our knowledge, none combines a position assignment that keeps an arbitrary number of compressed slots distinct at any horizon with summary writes performed in the canonical pre-RoPE domain. WorldTrace couples exactly these two (WorldTrace-Field compresses, WorldTrace-Landmark freezes), keeping every slot addressable inside a fixed �(1) cache.

Training-time extensions. Training-time methods finetune the generator itself for longer efective context. One line pairs partitioned caches or memory hierarchies with RoPE re-alignment or long-context teacher supervision [12, 105, 14, 49, 52, 117, 55]. Memory-decomposition approaches assign diferent parts of history to dedicated modules [77, 94, 8]. Architectural long-horizon models train for extended rollouts rather than patching a fixed checkpoint [27, 66, 79, 112, 120], and Sparse Forcing [103] learns block-sparse attention with persistent anchors end-to-end. Po et al. [68] extend the efective context of a video world model with a trained state-space memory and evaluate it on Memory Maze with a spatial retrieval task, where the model is given a trajectory as context and then backtracks the exact action sequence to the starting position. WorldTrace needs no retraining, restoring addressability by changing cache read/write on a fixed checkpoint.

Geometry-conditioned memory. Trained methods decide what to keep in memory based on camera pose, actions, or scene geometry, drawing on pose- and action-annotated compressed latents [35], warped positions or latents for camera-controlled worlds [111, 104, 62], co-visibility-based selection [23], pose-tagged frame re-injection [100], and camera-aware memory gating [28]. WorldTrace is geometry-free, replacing pose-, action-, or co-visibility-driven memory selection with slot-rank positions and canonical cache writes.

External memory banks and retrieval. Rather than compressing inside the KV cache, these systems store or retrieve frames outside it and re-inject them via auxiliary attention or re-encoding, through FOV-based frame selection [92], importance-weighted frame contexts [113], first-chunk appearance anchoring [33], cached latent banks [95, 51, 110], and global-state retrieval [15]. WorldTrace keeps memory in the KV cache under one �(1) recent-plus-summary budget, with no external bank, re-encoding, or retrieval.

Comparison. In Tab. 7, we qualitatively compare memory approaches on four axes: whether positional OOD is fixed, whether memory is bounded, whether intermediate history is preserved, and whether retraining is required. For evaluation, WorldScore [21], MIND [106], and VBench-2.0 [118] assess long-horizon quality, and Lian et al. [54] targets revisit consistency in Minecraft. The Memory Maze retrieval task of Po et al. [68] is the closest published revisit test to ours and scores a backtracked trajectory against the recorded ground-truth frames, whereas LoopBench sweeps loop topology, rollout length, camera orientation, and multi-revisit depth and scores each return against the model’s own earlier generation at matched poses, so it needs no ground-truth footage of the revisited scene.

Table 7: Memory approach comparison. Representative baselines and inference-time cache methods on four axes.
<table><tr><td>Method</td><td>Fixes pos. OOD?</td><td>Bounded memory?</td><td>Preserves history?</td><td>Training-free?</td></tr><tr><td>Full KV</td><td>No</td><td>No (O(N))</td><td>Yes (exact)</td><td>Yes</td></tr><tr><td>Sliding window</td><td>N/A (no long-range reads)</td><td>Yes (O(1))</td><td>Recent only</td><td>Yes</td></tr><tr><td>KV Flush + Block-relative [107]</td><td>Yes</td><td>Yes (O(1))</td><td>No</td><td>Yes</td></tr><tr><td>Naive avg. + Block-relative</td><td>Partial (slots collapse)</td><td>Yes (O(1))</td><td>No (corrupted)</td><td>Yes</td></tr><tr><td>MemRoPE [42]</td><td>Yes</td><td>Yes (O(1))</td><td>EMA-diluted</td><td>Yes</td></tr><tr><td>Deep Forcing [108]</td><td>Partial</td><td>Yes (O(1))</td><td>Sink plus pruned recent</td><td>Yes</td></tr><tr><td>WORLDTRACE-FIELD (ours)</td><td>Yes</td><td>Yes (O(1))</td><td>Yes (compressed field)</td><td>Yes</td></tr><tr><td>WORLDTRACE-LANDMARK (ours)</td><td>Yes</td><td>Yes (O(1))</td><td>Yes (verbatim landmarks)</td><td>Yes</td></tr></table>

Table 8: Slot allocation between WorldTrace-Field and WorldTrace-Landmark at a fixed cache budget. All rows use $N _ { s } { = } 4$ summary slots and $N _ { r } { = } 2$ recent slots on ABA revisits at a fixed seed. F counts slots written by canonical key averaging and L counts slots holding verbatim landmarks. Recall rises with the landmark share while the remaining WorldTrace-Field slots hold coherence at or above the sliding-window baseline.
<table><tr><td rowspan="2">Allocation</td><td colspan="3"> $\mathrm { T e m p S S I M } \uparrow$ </td><td colspan="3"> $\mathrm { P A C \uparrow }$ </td></tr><tr><td> $N { = } 1 6$ </td><td> $N { = } 3 2$ </td><td> $N { = } 4 8$ </td><td> $N { = } 1 6$ </td><td> $N { = } 3 2$ </td><td> $N { = } 4 8$ </td></tr><tr><td>Sliding window</td><td>0.754</td><td>0.778</td><td>0.776</td><td>0.837</td><td>0.815</td><td>0.792</td></tr><tr><td>4F+OL (WORLDTRACE-FIELD)</td><td>0.762</td><td>0.783</td><td>0.767</td><td>0.828</td><td>0.808</td><td>0.804</td></tr><tr><td> $3 \mathrm { F } { + } 1 \mathrm { L }$ </td><td>0.765</td><td>0.770</td><td>0.769</td><td>0.829</td><td>0.803</td><td>0.799</td></tr><tr><td> $2 \mathrm { F } { + } 2 \mathrm { L }$ </td><td>0.766</td><td>0.762</td><td>0.770</td><td>0.843</td><td>0.793</td><td>0.796</td></tr><tr><td> $1 \mathrm { F } { + } 3 \mathrm { L }$ </td><td>0.767</td><td>0.780</td><td>0.768</td><td>0.841</td><td>0.809</td><td>0.780</td></tr><tr><td>0F+4L (WORLDTRACE-LANDMARK)</td><td>0.760</td><td>0.770</td><td>0.797</td><td>0.866</td><td>0.827</td><td>0.817</td></tr></table>

## E Additional Experimental Results

## E.1 Unified Cache Allocation

Sec. 4.2 and Sec. 4.3 evaluate the two content writers separately, while a single long rollout typically asks for both coherent continuation and reliable revisits. The slot-rank cache of Def. 1 is a shared interface, so the two writers can occupy the same cache and the only decision is how to divide the fixed slot budget between them. Every summary slot that does not hold a landmark stays a WorldTrace-Field average.

We sweep the full allocation from all-WorldTrace-Field to all-WorldTrace-Landmark at the fixed budget of $N _ { s } { = } 4$ summary slots and report both metrics in Tab. 8. The all-landmark end of the sweep is the strongest on recall at every horizon and reaches $\mathrm { P A C = 0 . 8 6 6 , 0 . 8 2 7 }$ , and 0.817 at �=16, 32, and 48, above the sliding-window baseline in each case. The intermediate allocations do not interpolate monotonically between the two ends, and at �=32 and �=48 several of them sit slightly below the baseline, so on these runs the gain comes from dedicating slots to landmarks rather than from any particular mixture. The practical conclusion is that one cache serves both workloads with no retraining and no additional memory, and that the split follows from which behavior a given rollout needs rather than from tuning.

## E.2 Training-Free Positional Baselines

We compare MemRoPE [42] and YaRN [67] on the Sec. 4.3 ABA loops in Tab. 9, factoring the position scheme (Block-relative vs. WorldTrace) from the content writer at two horizons.

MemRoPE [42] (Block-relative + dual-rate EMA with $\alpha _ { \mathrm { l o n g } } { = } 0 . 0 1$ and $\alpha _ { \mathrm { s h o r t } } { = } 0 . 1$ , matching the original release) survives OOD positions via EMA decay but is still beaten by Landmark+Block-relative, because verbatim canonical keys retain stronger query-key similarity than smoothed averages, and WorldTrace-Landmark adds in-distribution positions on top. MemRoPE+WorldTrace (Block-relative swapped for WorldTrace, EMA kept) degrades (�<0.001) because its write and read ofsets disagree.

YaRN [67] rescales temporal RoPE frequencies to bring ofsets back in-distribution, gaining +22% over the sliding-window baseline at �=32 (�<0.001) and confirming position is the primary bottleneck. However, it needs �(�) memory (OOM by ${ \sim } N { = } 1 0 0 )$ , and the rescale diverges with horizon. WorldTrace-Landmark

Table 9: Concurrent training-free baselines. PAC (↑) at two horizons (ABA). YaRN [67] keeps an �(�) cache and improves over the sliding window at �=32. MemRoPE+WorldTrace replaces Block-relative positions with WorldTrace positions while leaving the dual-rate EMA content unchanged.
<table><tr><td>Method</td><td>Position</td><td>Content</td><td>N=32 (16×)</td><td>N=48 (24×)</td></tr><tr><td>O(N) cache:</td><td></td><td></td><td></td><td></td></tr><tr><td>Sliding window (baseline)</td><td>actual</td><td>Sliding window</td><td>0.401</td><td>0.388</td></tr><tr><td>YaRN [67]</td><td>NTK</td><td>Sliding window</td><td>0.490</td><td>0.412</td></tr><tr><td>O(1) cache, Block-relative positions:</td><td></td><td></td><td></td><td></td></tr><tr><td>MemRoPE [42]</td><td>Block-relative</td><td>dual EMA</td><td>0.651</td><td>0.706</td></tr><tr><td>Landmark + Block-relative</td><td>Block-relative</td><td>verbatim</td><td>0.929</td><td>0.934</td></tr><tr><td>O(1) cache, WoRLDTRACE positions:</td><td></td><td></td><td></td><td></td></tr><tr><td>MemRoPE + WORLDTRACE positions</td><td>WORLDTRACE</td><td>dual EMA</td><td>0.592</td><td>0.662</td></tr><tr><td>WORLDTRACE-LANDMARK (ours)</td><td>WORLDTRACE</td><td>verbatim</td><td>0.964</td><td>0.972</td></tr></table>

Table 10: Episodic recall splits into three tiers. Only verbatim recall scales. ABA loops, PAC (↑). Bold = best per column.
<table><tr><td>Method</td><td>N=16  $N { = } 3 2$ </td><td></td><td> $N { = } 4 8$ </td><td> $N { = } 6 4$ </td><td>N=128</td><td>N=256</td></tr><tr><td>Compression only:</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Sliding window (baseline)</td><td>0.540</td><td>0.401</td><td>0.388</td><td>0.412</td><td>0.504</td><td>0.631</td></tr><tr><td>Naive + Block-relative</td><td>0.570</td><td>0.434</td><td>0.433</td><td>0.443</td><td>0.486</td><td>0.598</td></tr><tr><td>WORLDTRACE-FIELD (ours)</td><td>0.555</td><td>0.434</td><td>0.436</td><td>0.442</td><td>0.495</td><td>0.602</td></tr><tr><td>Canonical-K anchoring:</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Latent re-anchor</td><td>0.955</td><td>0.952</td><td>0.938</td><td>0.913</td><td>0.782</td><td>0.610</td></tr><tr><td>WoRLDTRACE-FIELD with scene anchoring</td><td>0.955</td><td>0.951</td><td>0.935</td><td>0.911</td><td>0.777</td><td>0.624</td></tr><tr><td>Verbatim recall:</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>WORLDTRACE-LANDMARK (ours)</td><td>0.959</td><td>0.964</td><td>0.972</td><td>0.976</td><td>0.986</td><td>0.989</td></tr></table>

exceeds it by a wide margin at �=32 (0.964 vs. 0.490) in �(1) memory and stays near-constant through �=512.

## E.3 Scene Consistency Episodic Recall Sweep

We report the full PAC breakdown across � ∈ {16, 32, 48, 64, 128, 256} on ABA loops $( N _ { s } { = } 4 , N _ { r } { = } 2 )$ in Tab. 10 and partition methods by retained past content into three tiers: compression-only (sliding window, Naive, WorldTrace-Field), canonical-K anc<sup>h</sup>oring (Latent re-anchor, WorldTrace-Field with scene anchoring), and verbatim recall (WorldTrace-Landmark). Within compression, WorldTrace-Field beats the slidingwindow baseline at �=32 to 64 (�<0.001, paired �-test). Anchoring lifts recall by 40 to 50 points at moderate horizons. Only verbatim recall sustains it at �=256. At �=256 the canonical-K tier’s PAC falls to 0.610 while WorldTrace-Landmark holds at 0.989.

## E.4 Cross-Architecture Experiments

In Tab. 11, we report PAC at 2× to 8× training horizon on LingBot-World [82] with WorldTrace applied. LingBot-World is a mixture-of-experts difusion transformer built on a 14B backbone with Plücker camera conditioning, and MG2 is a dense 1.3B model without it, so this comparison varies both scale and architecture. The recall gain that WorldTrace-Landmark delivers on both backbones indicates that the positional bottleneck we identify follows the trained ofset range rather than parameter count. WorldTrace-Field tracks the sliding-window baseline closely, above it at 2×, 6×, and 8× and 0.005 below it at 4×, since Plücker camera conditioning already supplies the recall signal that KV-cache content provides on MG2, leaving no room for canonical averaging to add. WorldTrace-Landmark improves over sliding-window retention from 4× onward (+8.9%, +14.1%, +7.3% at $4 { \times } / 6 { \times } / 8 { \times } )$ , with no significant gain at 2× (−0.006 absolute, �>0.1), since verbatim key injection provides a complementary recall signal that strengthens as camera-pose priors alone become insuficient at longer horizons.

Table 11: LingBot-World [82] PAC at extended horizons. WorldTrace-Landmark improves over the sliding-window baseline from 4× onward. WorldTrace-Field stays close to the sliding window, consistent with Plücker conditioning supplying the primary recall signal.
<table><tr><td> $N$ </td><td>Horizon</td><td>Sliding window</td><td>WORLDTRACE-FIELD</td><td>WORLDTRACE-LANDMARK</td></tr><tr><td>14</td><td>2×</td><td>0.657</td><td>0.668</td><td>0.651</td></tr><tr><td>28</td><td>4×</td><td>0.624</td><td>0.619</td><td>0.680</td></tr><tr><td>42</td><td>6×</td><td>0.591</td><td>0.620</td><td>0.674</td></tr><tr><td>56</td><td>8×</td><td>0.632</td><td>0.648</td><td>0.678</td></tr></table>

## E.5 Slot Allocation

Summary/recent split sensitivity. The main experiments use $N _ { s } { = } 4$ summary slots and $N _ { r } { = } 2$ recent-window slots (total $N _ { s } { + } N _ { r } { = } 6 ,$ , matching the training context size). To test sensitivity to this split, we swept five allocations $( N _ { s } + N _ { r } { = } 6$ , ABA loops at �=32 and �=64). Results are in Tab. 12. $N _ { s } { \le } 2$ collapses toward the sliding-window baseline, since with one slot the B→A detector overwrites the scene-A landmark. With two slots the B-side traversal evicts it before the return. $N _ { s } { = } 3$ is intermediate (0.652/0.693 at $N { = } 3 2 / 6 4 )$ but still evicts the landmark. $N _ { s } { = } 4$ retains the landmark through the B-side return. $N _ { s } { = } 5$ adds only $+ 0 . 0 0 9 / + 0 . 0 0 5$ confirming a four-slot plateau.

Table 12: Slot Allocation. PAC at $N { = } 3 2$ and 64 as a function of $N _ { s }$ (total budget fixed at $N _ { s } + N _ { r } { = } 6 ) . \ N _ { s } { \leq } 2$ collapses toward the sliding-window baseline. Four slots is the critical boundary for retaining the scene-A landmark through the B-side traversal.
<table><tr><td> $N _ { s }$ </td><td> $N _ { r }$ </td><td>PAC N=32</td><td>PAC N=64</td></tr><tr><td>Sliding window</td><td></td><td>0.401</td><td>0.412</td></tr><tr><td>1</td><td>5</td><td>0.413</td><td>0.437</td></tr><tr><td>2</td><td>4</td><td>0.419</td><td>0.426</td></tr><tr><td>3</td><td>3</td><td>0.652</td><td>0.693</td></tr><tr><td>4</td><td>2</td><td>0.964</td><td>0.976</td></tr><tr><td>5</td><td>1</td><td>0.973</td><td>0.981</td></tr></table>

## E.6 Memory Accounting

WorldTrace never holds a full-attention KV cache. At every step the model attends over the same $N _ { s } { + N _ { r } }$ entries, so attention cost is constant per step and peak GPU memory is nearly flat in the rollout length. What a summary slot stores does evolve as generation proceeds, since each chunk that leaves the recent window updates the slot it belongs to, while the query position afects only the in-distribution slot-rank ofset at which the slot is read through a single �(1) rotation.

We report peak GPU memory at �=64, 256, and 512 in Tab. 13. WorldTrace-Field matches the slidingwindow baseline exactly at all three horizons because it adds nothing to the GPU-resident cache. Its recompute writer instead ofloads the canonical key of each frame that leaves the recent window to host memory, which grows by 5.4 MB per generated latent frame and stays of the GPU. WorldTrace-Landmark keeps no history at all beyond the frozen scene-entry keys in its slots and adds a constant scene-anchor overhead of about 0.6 GB over the baseline at every horizon.

## E.7 Streaming Summary Writer

The content writer of WorldTrace-Field is a recompute writer, which retains evicted canonical keys and recomputes each slot mean when group boundaries move. A strictly streaming alternative keeps one running sum and count per slot instead. Each evicted frame is folded into its slot once and then discarded, and a boundary shift is handled by an exact merge of adjacent buckets, so the writer state is $O ( N _ { s } )$ and constant in rollout length.

We compare the two writers at $N { = } 3 2$ and $N { = } 4 8$ in Tab. 14. The streaming writer keeps most of the recompute writer’s coherence gain and reaches the lowest Scene Drift at �=32, and both writers stay above the slidingwindow baseline at both horizons. Since the streaming writer retains no history outside the cache, this

Table 13: Peak GPU memory and writer-side state. Measured on one A100 80 GB at batch size 1. WorldTrace-Field matches the sliding-window baseline at every horizon and keeps its evicted canonical keys in host memory, and WorldTrace-Landmark adds a constant scene-anchor cost that does not grow with �.
<table><tr><td rowspan="2">Method</td><td colspan="3">Peak GPU memory (MB)</td><td rowspan="2">State outside the cache</td></tr><tr><td>N=64</td><td>N=256</td><td>N=512</td></tr><tr><td>Sliding window</td><td>1838</td><td>1903</td><td>1993</td><td>none</td></tr><tr><td>WORLDTRACE-FIELD</td><td>1838</td><td>1903</td><td>1993</td><td>host memory, +5.4 MB per latent frame</td></tr><tr><td>WORLDTRACE-LANDMARK</td><td>2470</td><td>2536</td><td>2626</td><td>none</td></tr></table>

Table 14: Recompute and streaming W T F writers. The streaming writer folds each evicted frame into a running per-slot accumulator once and then discards it, so its state is constant in rollout length. It keeps most of the coherence gain of the recompute writer and both stay above the sliding-window baseline.
<table><tr><td rowspan="2">Method</td><td colspan="2">N=32</td><td colspan="2">N=48</td></tr><tr><td>TempSSIM ↑</td><td>Scene Drift ↓</td><td>TempSSIM ↑</td><td>Scene Drift ↓</td></tr><tr><td>Sliding window</td><td>0.571</td><td>0.0229</td><td>0.472</td><td>0.0305</td></tr><tr><td>WORLDTRACE-FIELD (recompute)</td><td>0.613</td><td>0.0250</td><td>0.545</td><td>0.0295</td></tr><tr><td>WORLDTRACE-FIELD (streaming)</td><td>0.606</td><td>0.0223</td><td>0.538</td><td>0.0298</td></tr></table>

comparison isolates the efect of the slot-rank positions of Def. 1 from any benefit of the extra statistics the recompute writer happens to keep.

## F L B : Loop Memory Benchmark

We propose L B , a benchmark for episodic memory in autoregressive world models. Each task is a navigation path that departs from a location, visits other locations, and returns. The frames generated on the first visit serve as the reference, and we measure how closely the frames generated on the return reproduce them. LoopBench therefore needs no external reference video, as the rollout itself produces both the target and the prediction. Four properties of the path control task dificulty. (1) Waypoint count (�): the number of intermediate locations between departure and return to A (ABA: �=1, ABCDA: �=3). (2) Rollout length (�): the total number of AR chunks in the path. Longer rollouts put more chunks between the first visit and the return, so the revisited scene sits deeper in the compressed cache. (3) Camera orientation: the agent stays at A while the camera pans away by 90<sup>∘</sup>, 180<sup>∘</sup>, or 360<sup>∘</sup> and then returns, testing whether the initial view is recalled after the pan. (4) Multi-revisit depth (�): the number of times the repeated waypoint is visited in one path (ABABA returns to A twice, ABCBA and ABCDBA pass through B twice). Each axis varies one property while the others stay unchanged, following the principle that benchmark tasks should be picked to produce distinguishable rankings between candidate systems [58]. We show all twelve configurations in Fig. 8 and evaluate every one of them in Tab. 3. We show qualitative reconstructions on the ABA (axis 1) and pan 90<sup>∘</sup> (axis 3) settings in Fig. 9 and Fig. 10.

## G Implementation Details

## G.1 Model Architecture and Self-Forcing Background

We evaluate on MG2-1.3B [32], a 1.3B-parameter autoregressive video world model based on Wan 1.3B T2V [87], with 30 transformer layers, 12 attention heads, and a head dimension of 128. Video is generated in AR chunks of 3 latent frames at latent resolution 44 × 80 (352 × 640 pixels), giving 880 tokens per latent frame. Each new chunk is denoised by a distilled flow-matching sampler conditioned on the KV cache of prior chunks. The model is trained with Self-Forcing [38], which rolls out on the student’s own KV cache rather than on teacher-forced context (related training schemes include Difusion Forcing [10], CausVid [109], and one-step difusion distillation [76]). Attention is restricted to a local window of local\_attn\_size= 6 latent frames, so cross-frame temporal ofsets stay $\leq \Delta t _ { \mathrm { t r a i n } } { = } 5$ . At inference the rolling KV cache would grow without bound. Our method keeps it at the constant $L _ { \mathrm { a t t n } ^ { - } }$ latent-frame budget of the sliding window, which comes to $6 \times 8 8 0 \times 1 2 \times ( 1 2 8 \times 2 ) \times 3 0 \times 2 \approx 0 . 9 7 \mathrm { G B }$ in fp16 per generated video at batch size 1 (880 tokens/frame,

## Row 1: varying topology (� intermediate waypoints)

![](images/b678452de03ce99c3c3dee719473daf1c735dceef96e30f6174e1fa4908e2e79.jpg)  
ABA �=16

![](images/19f319d6d61a4ff1d8c4dc4146a03242680da9b3323e462d8c258b1412c04ada.jpg)  
ABCA �=17

![](images/e3a574a84314264aa447eadde8cb5cb860addad80fba652e3d9c89d7510021b6.jpg)  
ABCDA �=16

## Row 2: varying rollout length (� AR chunks in total, ABA topology)

![](images/1452829201eec8102d6359a7950c3be9184a1850995e2107678e7cfea014a55c.jpg)  
short �=8

![](images/a5250c1990f874abfc77dfadaa658190ff0019e136dbc46d713e063c54a66241.jpg)  
medium �=16

![](images/5afd876fc6672fd0913d6d7b8df4f08dde86680558a7b8ba1046f396b815512b.jpg)  
long �=32

## Row 3: camera-orientation pans (solid = pan away, dashed = return)

![](images/678bcbcf4cbfb335274790208590684844e34ba197935040b511f3beae6b8ed0.jpg)  
90° yaw pan

![](images/7450bbe7699b530f52624445ec1037e62d624190a67d6b0da8ad394204eb4894.jpg)  
180° look-back

![](images/beffaaa2df6cb05f508ac2645b5d6fac6c7e00ac841956b7c7c92689ab81526c.jpg)  
360° full pan

## Row 4: multi-revisit patterns (� > 1, solid/lighter arcs = 1st/2nd traversal)

![](images/c88c7b128d8ca9503659ec9b446ebf1e6c55ccc4977dfd90111097f1da8d5d8b.jpg)  
ABABA �=2, 4 edges

![](images/ebf9a8122875fd6dc8b8c83683e8b5bc5c168421e92d3345a41db51b19303c30.jpg)  
ABCBA �=2, 4 edges

![](images/577e857e402bbb537a23baf3cb0b2b8316d562573dbdd386791ecebd7e15aba1.jpg)  
ABCDBA �=2, 5 edges

Figure 8: L B benchmark gallery. Solid and dashed blue arrows mark the outbound and return legs, lighter arcs mark a second traversal, and dotted arrows mark shortcuts that skip intermediate waypoints. Row 1: topology (ABA, ABCA, ABCDA). Row 2: rollout length on the ABA topology, where longer rollouts push cached ofsets further out of distribution. Row 3: camera orientation, where the agent stays at A while the camera pans away (solid) and returns (dashed). Row 4: multi-revisit patterns, where � counts visits to the repeated waypoint (A in ABABA, B in the ABCBA palindrome and the ABCDBA shortcut).

![](images/facc61a967a74ca2ea5efaa6b05aec0b530efe2e1f6bc1cca0c97611afd4dd5b.jpg)  
Figure 9: ABA qualitative results. Each sample group of three frames shows trajectory keyframes at chunks 0 (A) / 7 (B) / 15 (A return). The two rows compare the sliding-window baseline (top) vs. WorldTrace-Landmark (bottom). The sliding window drifts away from the scene-A appearance on the return leg across the four samples. WorldTrace-Landmark brings the return frames back to the original scene appearance.

![](images/bcc85bb8713df89b89fbae1e766468d57eb6476af1dc21d6f318926a6d070d0a.jpg)  
Figure 10: Pan $9 0 °$ qualitative results. In camera-orientation Tier 3, the agent stays at A while the camera pans right by ${ \sim } 9 0 ^ { \circ }$ and then pans back. Each sample group of three frames shows the start (0<sup>∘</sup>), the pan peak $( \sim 9 0 ^ { \circ } )$ and the return to $0 ^ { \circ }$ . The two rows compare the sliding-window baseline (top) and WorldTrace-Landmark (bottom). WorldTrace-Landmark restores the initial-view appearance on the return. The sliding window does not.

12 heads, head dim 128, $\times 2$ for K and ${ \mathrm { V } } ,$ 30 layers, 2 B per value), independent of generation length. The long-horizon failure is therefore not one of missing memories, since keys are stored verbatim and the model was simply never trained to read them through the temporal RoPE rotations they accumulate beyond $\Delta t _ { \mathrm { t r a i n } }$ (App. C).

## G.2 WorldTrace Algorithm and Baselines

W T cache update. After each new chunk is generated, its keys and values enter the verbatim recent window. When the recent window exceeds $N _ { r }$ chunks, the oldest chunk is evicted and folded into the summary cache, by canonical key averaging over its temporal group (WorldTrace-Field, Eq. (3)) or by storing it as a canonical landmark when a scene entry is detected (WorldTrace-Landmark, Eq. (4)). We list these update steps in Alg. 1, together with the shared final step that re-rotates each canonical summary key to its virtual position when attention scores are computed.

Contrast with MG3. Matrix-Game-3 [92] addresses long-horizon memory from the training side. It retrains a bidirectional difusion backbone whose memory is built in during training, so that past frames are retrieved by camera pose and field-of-view overlap, injected as additional conditioning latents alongside the recent history, and learned through error-aware training and a modified temporal RoPE, followed by multi-segment distillation for real-time inference. We do not compare against MG3 because its memory is trained into

![](images/8c0ee048c07fb8881469af48e12713c00eed271e64b8db854734c2cd6d480402.jpg)  
Figure 11: Camera trajectory used for Fig. 4. Bird’s-eye view of the camera trajectory from start (�=1) to end (�=48), with ticks at the chunks shown as columns in Fig. 4.

the generator rather than implemented in the cache. There is no fixed pretrained checkpoint whose cache read/write WorldTrace could modify, so the comparison would conflate a retrained memory mechanism with our training-free, inference-time setting.

## G.3 Hyperparameters, Compute, and Evaluation Protocol

Camera trajectory for the coherence qualitative panel. The qualitative comparison in Fig. 4 (Sec. 4.2) is generated along a single camera path on Matrix-Game 2, played out over �=48 AR chunks (∼57 s of decoded video). We show that path from a bird’s-eye view in Fig. 11, where the camera follows one continuous trajectory from the initial scene (start, �=1) to a distinct end pose $\scriptstyle ( n = 4 8 )$ , and the marked chunks $n \in \{ 8 , 1 8 , 2 4 , 3 2 , 4 0 \}$ give the waypoints shown as columns in $\mathrm { F i g . }$ 4. All methods in ${ \mathrm { F i g . } }$ 4 follow this same trajectory.

Algorithm 1 WorldTrace cache update. � stores canonical (unrotated) keys. The rotation $R ( \theta _ { f } t _ { s } ^ { v } )$ is applied   
per slot when attention scores are computed, using the virtual positions of Def. 1, so positions recompute   
automatically as � advances.   
Require: recent cache ${ \mathcal { R } } ,$ summary cache �, new chunk’s KVs, mode (WorldTrace-Field or WorldTrace-  
Landmark)   
Ensure: updated $( \mathcal { R } , S )$   
1: Append new KVs to $\mathcal { R }$ ◁ recent window is verbatim   
2: if $| \mathcal { R } | > N _ { r }$ then   
3: Evict the oldest entry from $\mathcal { R }$   
4: if mode is WorldTrace-Field then   
5: Find the summary slot whose source bucket now contains the evicted entry ◁ uniform temporal   
grouping   
6: Update that slot to the canonical mean of its source frames ◁ Def. 2   
7: else if mode is WorldTrace-Landmark then   
8: if any frame of the evicted entry has canonical-key cosine distance $> \tau$ to its predecessor then ◁   
scene-entry detection, Sec. 3.4   
9: Shift � left by one slot ◁ evict oldest landmark   
10: Store the evicted entry’s canonical keys in $S [ N _ { s } { - } 1 ]$ ◁ canonical landmark, Eq. (4)   
11: end if   
12: If fewer than $N _ { s }$ landmarks are stored, fill empty slots with the oldest.   
13: end if   
14: end if   
15: At attention: for $s { = } 0 , \ldots , N _ { s } { - } 1 _ { \mathrm { { i } } }$ , apply $R ( \theta _ { f } t _ { s } ^ { v } )$ to �[�] with $t _ { s } ^ { v }$ from Def. 1.

Inference hyperparameters. MG2-1.3B [32] uses 3 distilled denoising steps $( t \in \{ 1 0 0 0 , 6 6 6 , 3 3 3 \}$ with timestep shift 5.0), no classifier-free guidance at inference, a single conditioning image, and $F { = } 3$ latent frames per AR chunk. Each chunk is additionally conditioned on per-frame keyboard and mouse actions through the model’s action module, which is how the camera trajectories are controlled. In our setting, the first chunk is generated with an empty KV cache, and every later chunk attends to the accumulated compressed cache.

W T F hyperparameters. Short-horizon experiments $\left( N { = } 8 \right)$ use $N _ { s } { = } 2$ summary slots and $N _ { r } { = } 4$ recent-window slots $( N _ { s } + N _ { r } = L _ { \mathrm { a t t n } } { = } 6 )$ . Longer horizon, LoopBench, and ablation experiments (Secs. 4.3–4.4) use $ N _ { s } { = } 4 , N _ { r } { = } 2$ (same total capacity) for both WorldTrace-Field and WorldTrace-Landmark. Compression uses uniform temporal grouping, splitting the $T - N _ { r }$ frames outside the recent window into $N _ { s }$ equal groups. Each group’s keys are unrotated to canonical space, averaged, and re-rotated at the virtual position.

Evaluation protocol. Each method is evaluated on 100 videos generated from distinct initial frames. Multiseed experiments use seeds {0, 42, 123, 456, 789}. All experiments run on a single NVIDIA A100 80 GB GPU.

Compute. WorldTrace-Field generates at ∼0.9 s per AR chunk at batch size 1, and VAE decode adds ∼2.5 s per chunk. The full experiment set required approximately 100 GPU-hours on single A100 80 GB GPUs. We report wall-clock time per AR chunk on one A100 80 GB at batch size 1 in Tab. 15. All methods stay within 6% of the sliding-window baseline, since the forward pass dominates the cache update, so the overhead of WorldTrace is negligible in practice.

Table 15: Runtime per chunk. Wall-clock time at �=8 with VAE decode, on one A100 80 GB (batch 1, 3 distilled denoising steps, 352×640).
<table><tr><td>Method</td><td>s/chunk</td></tr><tr><td>Sliding window</td><td>0.95</td></tr><tr><td>WORLDTRACE-FIELD</td><td>0.99</td></tr><tr><td>WORLDTRACE-LANDMARK</td><td>1.00</td></tr></table>

## H Discussion

## H.1 Limitations

WorldTrace operates entirely at inference time, leaving the pretrained weights frozen and changing only what information the fixed-size KV cache retains (compressed summaries in place of evicted frames) and the virtual RoPE positions at which those entries are read, with the goal of improving long-horizon memory. Everything else, including per-chunk visual quality, motion dynamics, and action following, is inherited unchanged from the pretrained backbone. The method also assumes an autoregressive generator with temporal RoPE applied to keys and a known local attention window. Compression into a fixed budget of $N _ { s }$ slots is lossy by construction, since each WorldTrace-Field slot averages roughly $T / N _ { s }$ frames, a ratio that grows with horizon, so frame-level detail inside a group is progressively blurred. Under WorldTrace-Landmark, a scene can be recalled later only if the scene-entry detector stored it as a landmark. Moreover, the cache holds only $N _ { s }$ landmarks, so a rollout that enters more scenes than that loses the oldest ones and can reliably revisit at most $N _ { s }$ distinct places. Future work could relax this cap, for example by merging evicted landmarks into a coarse residual summary slot so that discarded scenes leave a recoverable trace, or by selecting which landmark to evict based on predicted revisit likelihood.

## H.2 Future Directions

Geometry-aware canonical keys. Coupling the unrotate/rerotate primitive with camera-pose warping (e.g. the Plücker and Warped-RoPE writers of MosaicMem [111] and UCM [104]) would align frames to a shared scene coordinate system and then average, so each summary token blends observations of similar scene content.

Learned scene-entry policies. A small policy trained on action discontinuities, agent-pose changes, or scene-segmentation logits could let WorldTrace-Landmark commit landmarks during continuous motion and decide w<sup>h</sup>ic<sup>h</sup> landmark to select. The fixed split between $N _ { s }$ and $N _ { r }$ could become an autotunable schedule and adapt per backbone or horizon without the sweep of App. E.5.

Fine-tuning extensions. WorldTrace stays within the attention window $L _ { \mathrm { a t t n } } ,$ where the budget $N _ { s } + N _ { r } =$ $L _ { \mathrm { a t t n } }$ keeps every summary slot in-distribution. Two light fine-tuning paths relax this budget. (i) Contextextensionfine-tunin on synthetic long rollouts would let WorldTrace allocate more recent slots for coherence or more summary slots for longer recall at the same in-distribution cost. (ii) Position-aware fine-tuning that trains the model on the slot-rank ofsets of Eq. (2) would tighten the canonical-mean approximation of WorldTrace-Field.

Training memory policies from revisit supervision. The memory policy that decides what to store, compress, and evict could itself be trained, and revisit tasks supply the training signal for free, since whether the model reproduces the original scene on its return is directly measurable. Closed-loop simulators can collect such revisit episodes at scale, supporting supervised fine-tuning on rollouts where recall succeeds as well as reinforcement learning with the revisit score as reward. In both cases the backbone stays frozen and WorldTrace keeps every stored summary in-distribution, so only the lightweight policy over the cache is trained.

Scaling L B . We propose <sup>L</sup>oop<sup>B</sup>ench (App. F) with four dificulty tiers, and it can grow beyond them, since longer paths, more waypoints, and richer camera trajectories are all generated by the same recipe, and evaluating other world models on it would show how well their memory mechanisms support revisits, complementing WorldScore [21], MIND [106], and VBench-2.0 [118].

## H.3 Broader Impact

WorldTrace is an inference-time cache mechanism that alters no training data, modifies no weights, and expands none of the generative capabilities of the underlying video model, so the relevant considerations are those already attached to pretrained video world models. By keeping long-horizon generation within a constant memory budget, it also lowers the compute cost of building on pretrained generators, a cost that shapes the wider ecosystem reusing them, from score-distilled 3D synthesis [59, 101] with compute-aware gradient estimators [5, 72] to LLM-conditioned mesh generation [90] and motion attribution for video models [96]. We release no new datasets, and downstream use should follow the base models’ content policies.