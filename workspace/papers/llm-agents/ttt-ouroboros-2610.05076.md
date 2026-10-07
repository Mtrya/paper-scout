# 2610.05076 (from arXiv HTML)

##### Report GitHub Issue

×

Title:

Content selection saved. Describe the issue below:

Description:

Submit without GitHub Submit in GitHub

![](/static/base/1.0.1/images/icons/smileybones-small.svg) arXiv is now an independent nonprofit! [Learn more](https://info.arxiv.org/about) ×

[![arXiv logo](/static/base/1.0.1/images/arxiv-logo-primary-light.svg) Back to arXiv ](/)

[Why HTML?](https://info.arxiv.org/about/accessible_HTML.html) Report Issue [ Back to Abstract ](/abs/2610.05076v1 "Back to abstract page") [ Download PDF](/pdf/2610.05076v1 "Download PDF") [ ](javascript:toggleNavTOC\(\); "Toggle navigation") [ ](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")

  1. Abstract
  2. 1 Introduction
  3. 2 Preliminaries: Persistent Test-Time Training
     1. TTT-E2E learns during inference.
     2. Reading, writing, and the feedback loop.
  4. 3 Experimental Setup
     1. Models and updates.
     2. Streams and independent evaluation.
     3. Primary comparison and outcomes.
  5. 4 Long-Horizon Persistent Self-Writing Degrades Prediction
     1. Frequent external text interrupts the feedback loop.
     2. Beyond native TTT: Qwen3-4B shows the same failure.
     3. Real-text writes can be useful.
  6. 5 A Causal Decomposition of the Feedback Path
     1. 5.1 Keep learning, but fix the generator
     2. 5.2 Recorded Replay: process the same text with and without writing
     3. 5.3 One update can fit its source and hurt another input
        1. Gradient conflict predicts which updates are more harmful.
        2. What the causal decomposition establishes.
  7. 6 Which Updates Should Be Retained?
     1. 6.1 Damage amplifies in degraded states but remains heterogeneous
     2. 6.2 External validation measures whether an update transfers
        1. When is external validation useful beyond Source Masking?
  8. 7 Related Work
     1. Writable inference-time state.
     2. Test-time adaptation and stability.
     3. Generated-data feedback and update selection.
  9. 8 Discussion and Conclusion
     1. What causes the failure?
     2. How does the causal decomposition support this explanation?
     3. What reduces the damage?
  10. References
  11. A Experimental scope and implementation
     1. Stable book identities.
     2. Protocol map.
     3. Model and training.
     4. Conversion and numerical checks.
     5. Streaming evaluation.
     6. Write strength, retention, and state restoration.
     7. A.1 Generator isolation and chunk boundaries
     8. A.2 Feedback and displacement control
     9. A.3 Model-specific contamination screening
  12. B Replication and boundaries of the failure regime
     1. B.1 Fixed Generation confirmation at 125M and 760M
     2. B.2 Decoder and reset controls change the rate of damage
     3. B.3 External-text exposure bounds the failure regime
     4. B.4 Teacher-forced real-text payoff
  13. C Causal decomposition: feedback, attention, and persistent weights
     1. C.1 Fixed Generation and Recorded Replay
     2. C.2 3B fixed-text replay confirmation
     3. C.3 Paired one-update comparison
     4. C.4 Gradient conflict and one-update transfer cost
        1. Purpose.
        2. Quantities computed for each candidate.
        3. Two pre-update signals.
        4. Correlation analysis.
        5. Results.
     5. C.5 In-place Adam: dose and history are separate considerations
  14. D From heterogeneous harm to partial repairs
     1. D.1 A minority of source passages drives the late mean
     2. D.2 Repetition weighting mitigates but does not remove damage
     3. D.3 Smaller writes expose a stability–adaptation trade-off
     4. D.4 Real read-only context helps because of its content
  15. E Evaluating transfer with independent evidence
     1. E.1 Sequential validation and commitment
     2. E.2 Cross-scale outcomes
     3. E.3 Component ablation
     4. E.4 Admission and adaptation in one all-real stream
     5. E.5 Preserving adaptation and validating the committed state
  16. F External validation when source labels are corrupted
  17. G WebShop feedback and task-metric validation
  18. H ALFWorld: keeping the agent able to complete tasks
     1. Tasks and learning.
     2. Three ways to handle an update.
     3. What changes when updates are checked?



[ License: CC BY-SA 4.0 ](https://info.arxiv.org/help/license/index.html#licenses-available)

arXiv:2610.05076v1 [cs.CL] 04 Oct 2026

# Self-Generated Feedback Destabilizes Test-Time Training:   
A Causal Decomposition of Long-Horizon Adaptation

Cheng Luo  Bing Li ††thanks: Corresponding author. Bernard Ghanem  Affiliation: King Abdullah University of Science and Technology (KAUST) 

###### Abstract

Test-time training (TTT) lets a model store information in its weights during inference. When the model learns from its own output, however, each update also changes the model that generates the next training example. Across 128K-token streams, retaining generated-text updates worsens prediction on independent human-written text with three TTT-E2E model configurations (labeled 125M, 760M, and 3B). The same failure occurs when Adam updates Qwen3-4B’s existing weights. The same update mechanisms can improve on real text, so writing itself is not the failure. Three matched comparisons trace the causal pathway. _Fixed Generation_ removes over 98%98\% of the damage at 125M and 760M by using a frozen model to generate training chunks. _Recorded Replay_ separates the loss caused by reading degraded text from the additional loss stored by updating on it. A paired one-update comparison then shows the local conflict: an update predicts its source better but new real text worse. This cost grows after Closed Loop adaptation, with a few trajectories accounting for most large failures. Finally, _Settlement_ evaluates the candidate state on independent real text before commitment. It leaves mean endpoint gaps of .07.07 and −.02-.02 nats at 125M and 760M while retaining real-text adaptation. These results motivate checking prediction on independent evidence before retaining an update.11 1 The project repository is publicly available at <https://github.com/lingjivoo/ttt-ouroboros>.  
  
---  
  
## 1 Introduction

Test-time training (TTT) lets part of a model continue to learn during inference (Sun et al., 2020; Liu et al., 2021; Sun et al., 2025). In long-context language models, these changing parameters—the _fast weights_ —can retain information after the source text leaves attention (Schmidhuber, 1992; Ba et al., 2016; Schlag et al., 2021; Tandon et al., 2025; Behrouz et al., 2025). This makes TTT attractive for long documents, streams, and long-running agents, where retaining the full history in attention is costly (Tandon et al., 2025; Behrouz et al., 2025; Wang et al., 2026). We call processing text without changing weights _reading_ , and retaining its update _writing_.

Writing becomes risky when the model also supplies its future training data. Fast weights WtW_{t} generate chunk xtx_{t}, learning from it produces Wt+1W_{t+1}, and the new weights generate the next chunk. Lower loss on xtx_{t} shows that the update fits its source, not that it predicts other text better. We ask which part of this feedback loop makes persistent self-writing fail.

Over 128K-token streams, matched models retain or discard each generated-text update and are evaluated on separate human-written passages. In End-to-End Test-Time Training (TTT-E2E) (Tandon et al., 2025), retaining generated-text writes worsens real-text prediction at 125M, 760M, and 3B. The effect emerges beyond the original evaluation horizon and also occurs when Adam updates Qwen3-4B’s existing feed-forward weights (Kingma and Ba, 2015; Yang et al., 2025); it is not specific to TTT-E2E’s update module. The same update rules improve prediction when trained on real text, so the question is why closed-loop generation reverses their effect.

We trace the causal pathway through three matched comparisons. First, Fixed Generation removes feedback to future training text: a frozen model generates each chunk while a separate model learns from it, removing over 98%98\% of the damage at 125M and 760M. Second, Recorded Replay holds the text fixed: degraded tokens hurt through attention, and retaining their updates adds a persistent weight cost, including at 3B. Third, paired one-update branches start from identical weights, attention, and text: retaining the update improves its source prediction but worsens the next real passage. This cost grows after Closed Loop adaptation, and a few trajectories produce most large failures.

A repetition penalty slows the feedback trajectory, reducing the 128K endpoint gap from 3.013.01 to .88.88 nats, but repetition does not identify which passage is safe to write. _Settlement_ instead evaluates the candidate state on independent real text. It preserves real-text adaptation and leaves mean endpoint gaps of .07.07 and −.02-.02 nats at 125M and 760M, respectively. It also remains stable when source labels are noisy.

Recursive synthetic-data studies track feedback across successive model generations (Briesch et al., 2023; Shumailov et al., 2024). aTTT studies feedback within an agent episode and downweights repeated update tokens (Wang et al., 2026); VANE validates updates on later visual observations (Ji et al., 2026). Here we isolate feedback in long language-model streams, using matched comparisons to distinguish changes to future training text from the effects of reading and updating on fixed text. Our contributions are: (1) long-horizon evidence that self-generated feedback damages independent real-text prediction across three TTT-E2E scales and under ordinary Adam, together with an external-text exposure boundary; (2) a causal decomposition using Fixed Generation, Recorded Replay, and paired one-update branches to separate changes to future training text, attention, and persistent weights, revealing a state-dependent failure concentrated in a few trajectories; and (3) a transfer criterion showing why source fit and repetition cannot justify retaining an update, and how independent text can evaluate the complete candidate state.

## 2 Preliminaries: Persistent Test-Time Training

#### TTT-E2E learns during inference.

An ordinary language model keeps the same parameters throughout inference. Test-time training (TTT) instead updates part of the model on the input it is currently processing, allowing information to remain in the weights after the corresponding tokens leave the attention window. Applying such updates only at inference creates a mismatch with pretraining. TTT-E2E addresses this mismatch by applying the same next-token updates during training and meta-learning an initialization that works well after those updates (Tandon et al., 2025). At inference, it updates MLP blocks in the final quarter of a sliding-window Transformer (Vaswani et al., 2017; Tandon et al., 2025). Following the fast-weight literature (Schmidhuber, 1992; Ba et al., 2016), we denote these adaptable parameters by WtW_{t}: W0W_{0} is their initial value and WtW_{t} is their value after tt chunks.

#### Reading, writing, and the feedback loop.

Every chunk can affect later predictions by remaining in attention; we call this _reading_. If the model also retains the gradient update learned from that chunk, the chunk changes WtW_{t}; we call this _writing_. When the model learns from its own output, the current state WtW_{t} generates chunk xtx_{t}, the update on xtx_{t} produces Wt+1W_{t+1}, and that new state helps generate the next chunk. This recurring dependence is the _Closed Loop_. Our main control, _Generated Writes Off_ (hereafter _Writes Off_), reads the same kind of generated chunks but discards the update after each one. Both policies keep updates from the initial real-text prefix. Their difference is therefore whether learning from generated text becomes persistent, not whether the model sees generated text at all.

## 3 Experimental Setup

#### Models and updates.

TTT-E2E pretrains on DCLM and continues long-context training on books (Li and others, 2024; Tandon et al., 2025). We follow that domain setting with PG-19 (Rae et al., 2020): we train models labeled 125M and 760M by their TTT-E2E configurations and use the released 3B model. Most causal experiments use the 125M model. It has an 8192-token attention window, and one clipped gradient step follows each 1024-token chunk, updating feed-forward modules in the final three layers. To check that the phenomenon is not specific to TTT-E2E, we also apply Adam (Kingma and Ba, 2015) directly to existing Qwen3-4B weights (Yang et al., 2025). The main cross-scale comparison uses the same six books, five seeds, 128K horizon, and batch width eight at every scale. Appendix A gives the complete configuration map.

#### Streams and independent evaluation.

Each 128K stream starts with 8K tokens of real text. The model then generates 105 chunks, with 15 later real-text evaluations plus an initial reference. In the canonical comparison, these passages come from later, disjoint regions of the same PG-19 book. After each evaluation, we restore the preceding weights and attention state, so evaluation cannot change the continuing stream. Throughout the paper, _independent real text_ means text outside the generated training stream; it need not come from a different book. Its NLL asks whether an update transfers beyond the text that produced it. It does not measure whether the model remembers its own earlier output. K denotes 1024 tokens. Generation uses temperature one and top-p=0.95p=0.95 nucleus sampling (Holtzman et al., 2020). Appendix A gives complete schedules and state-restoration checks.

#### Primary comparison and outcomes.

We want to measure deterioration caused by retaining generated-text updates, not the change that occurs even when those updates are disabled. Both policies therefore start from identical weights and an identical real-text prefix. The Closed Loop retains each generated-text update; Writes Off discards it, while both policies continue to read generated tokens. For condition cc, let Lc,firstL_{c,\mathrm{first}} and Lc,lastL_{c,\mathrm{last}} be NLL on its first and last independent real-text evaluations. Lower NLL is better. We define

| Dc=Lc,last−Lc,first,Hc=Dc−Doff.D_{c}=L_{c,\mathrm{last}}-L_{c,\mathrm{first}},\qquad H_{c}=D_{c}-D_{\mathrm{off}}. |  | (1)  
---|---|---|---  
  
DcD_{c} is the first-to-last change within condition cc: positive values mean that real-text prediction became worse. DoffD_{\mathrm{off}} is the same change under Writes Off. Their difference, HcH_{c}, is the additional deterioration caused by the full policy relative to disabling generated-text writes. Thus Hc>0H_{c}>0 means worse than Writes Off, and Hc<0H_{c}<0 means better.

When both policies receive identical prerecorded tokens, HcH_{c} isolates the effect of retaining their updates. During online generation, it includes both the updates and the later text that those updates cause the model to generate; this total feedback effect is our primary outcome. Intervention suites additionally report the endpoint gap, Ec=Lc,last−Loff,lastE_{c}=L_{c,\mathrm{last}}-L_{\mathrm{off},\mathrm{last}}. Positive EcE_{c} means that damage remains at the end; zero means that the final NLL matches Writes Off. When initial losses match, Ec=HcE_{c}=H_{c}. Unless specified otherwise, 95% intervals resample books after averaging seeds within each book. Appendix A maps every comparison to its configuration and baseline.

## 4 Long-Horizon Persistent Self-Writing Degrades Prediction

Table 1: Generated-text writes worsen real-text prediction. The same six books, five seeds, and 128K horizon are used at every scale. Model | HH (↑\uparrow worse) | 95% Interval  
---|---|---  
125M TTT-E2E | +3.01+3.01 | [+1.86,+4.20][+1.86,+4.20]  
760M TTT-E2E | +6.00+6.00 | [+4.27,+7.73][+4.27,+7.73]  
3B TTT-E2E | +0.40+0.40 | [+0.30,+0.51][+0.30,+0.51]  
  
We first ask whether persistent self-writing eventually makes the model worse at predicting human-written text. Closed Loop and Writes Off both generate for 128K tokens and keep the generated tokens in attention. Closed Loop alone retains the update after each chunk. We periodically pause both streams and evaluate the current weights on the same independent real passages. The difference therefore measures the consequence of making generated-text updates persistent, beyond merely reading the generated text.

Table 1 uses the same six books at every scale. Retaining the updates adds 3.013.01, 6.006.00, and .40.40 nats over Writes Off at 125M, 760M, and 3B, respectively. Every interval excludes zero, establishing the same directional failure at all three evaluated scales. The magnitude differs across models; the result is a cross-scale replication of direction rather than a monotone relation with parameter count.

Figure 1a shows why the long horizon matters. Within the earlier 8K+8K evaluation window, Closed Loop and Writes Off remain close. Only after many additional self-writes does Closed Loop rise while Writes Off stays nearly flat. The failure is therefore cumulative rather than a large penalty from the first update.

Could decoding alone explain the result? It does not. Full-support sampling reduces the 128K endpoint gap from 3.0103.010 to 1.4521.452 [.708,2.180][.708,2.180], but late loss is still rising. Lower temperature, typical sampling, and a repetition penalty change the speed or severity of degradation without providing a reliable boundary between safe and harmful writes. Periodically resetting the fast weights is stronger: an eight-chunk reset leaves a .175.175 gap [.084,.321][.084,.321], but also preserves only 44.1% of the benefit obtained from real-text adaptation. The full decoder and reset comparisons appear in Appendix B.2.

#### Frequent external text interrupts the feedback loop.

Can occasional real text stop the damage from accumulating? We replace 0, 5, 11, 21, or 33 of the stream’s 105 generated chunks with real chunks while retaining every update. Figure 1b reports the remaining harm relative to no replacements. Evenly spaced real chunks reduce it to approximately 87%, 37%, 7%, and 6%, respectively.

Timing also matters. With 33 evenly spaced real chunks, the model generates at most three chunks consecutively. Grouping the same 33 chunks into bursts permits stretches of 12 and leaves about 60% of the original harm rather than 6%. Real text works best when it repeatedly interrupts self-writing (Appendix B.3).

Figure 1: Long horizons and sparse real-text exposure expose the failure. (a) Real-text NLL across a 128K stream; shading marks the earlier 8K+8K horizon (Tandon et al., 2025). (b) Harm relative to a separate long-book suite’s 0% baseline. Even spacing interrupts the loop; grouping the same 31% real-text budget into bursts restores much of the harm. Panel (b) is normalized within a separate matched suite; error bars scale its 95% intervals by the fixed 0%-exposure estimate (Appendix B.3). Table 2: Qwen3-4B with Adam. The same weights benefit from real text but fail under persistent self-writing. Sampling units and complete settings: Appendix C.5. Adam LR |  Persistent self-writing HH 95% CI; ↑\uparrow worse |  Real-text BB state range; ↑\uparrow better  
---|---|---  
10−510^{-5} | .026​[−.071,.122].026\;[-.071,.122] | .139​[.115,.163].139\;[.115,.163]  
10−410^{-4} | 1.231​[.580,2.350]\mathbf{1.231\;[.580,2.350]} | .168​[.132,.204].168\;[.132,.204]  
  
#### Beyond native TTT: Qwen3-4B shows the same failure.

We use Adam (Kingma and Ba, 2015) to update Qwen3-4B’s existing feed-forward weights (Yang et al., 2025), without adding a TTT module. At learning rate 10−410^{-4}, persistent self-writing raises real-text NLL by 1.2311.231 nats relative to Writes Off; at 10−510^{-5}, the interval includes zero (Table 2). Yet the same 10−410^{-4} update lowers NLL by .168.168 nats when it learns from real text. Ordinary Adam updates can therefore help on real text and fail in a self-generated loop: the failure is not specific to TTT-E2E’s learned update mechanism.

#### Real-text writes can be useful.

These results do not imply that all inference-time learning should be disabled. On eleven identical teacher-forced real-text streams, retaining updates lowers mean NLL by .0340.0340 nats relative to _No Updates_ (Appendix B.4). Because the input text is fixed, the gain comes from learning rather than from changing future inputs. Writing can help; the next section asks why it becomes damaging when the learner also supplies the text used for later updates.

## 5 A Causal Decomposition of the Feedback Path

Why does Closed Loop fail? There are three distinct possibilities. Generated text may be poor training data even when it comes from a fixed model; degraded tokens may hurt while they remain in attention; and learning from those tokens may store additional harm in the weights. Figure 2 separates these possibilities by changing one dependency at a time. In Fixed Generation (a), frozen W0W_{0} generates every chunk for a separate learner, so learning cannot change later training text. In Recorded Replay (b), a new receiver processes the same saved degraded tokens with and without updates. In the Closed Loop (c), current WtW_{t} generates xtx_{t}, learns from it, and becomes Wt+1W_{t+1} before generating again. We call this sequence of controlled comparisons a _causal decomposition_ : it locates the path to damage, rather than assigning additive shares of one total loss. Every comparison evaluates the adapting model on independent real text.

### 5.1 Keep learning, but fix the generator

We first ask whether generated text is harmful merely because it is synthetic. In panel (a), the learner still updates after every chunk, but an independently cached W0W_{0} supplies all future chunks. We evaluate the learner, not W0W_{0}.

Figure 2: Three controlled comparisons separate generation, content, and updating. (a) The initial state W0W_{0} generates while a separate WtW_{t} learns, breaking feedback. (b) Saved tokens are processed read-only or with updates, holding content fixed. (c) Current WtW_{t} both generates and learns, closing the loop. Every write-enabled condition evaluates the adapted WtW_{t}; Appendix A.1 verifies generator isolation.

Fixed Generation leaves only .052.052 and .073.073 nats of excess change at 125M and 760M, removing 98.3%98.3\% and 98.8%98.8\% of their matched Closed Loop gaps (Table 3). The learner still sees generated text and still updates after every chunk. The only removed link is its ability to change the text from which it will learn next. Generated origin alone is therefore insufficient; feedback to future training text accounts for nearly all of the measured gap in these comparisons.

Table 3: Breaking feedback removes most of the excess change at both evaluated scales. Fixed Generation residuals and removal fractions use matched within-scale comparisons on the same six canonical books. Complete endpoints are in Appendix B. Model | Fixed Generation HH | Closed Loop harm removed  
---|---|---  
125M | .052.052 | 98.3%98.3\%  
760M | .073.073 | 98.8%98.8\%  
  
Update magnitude does not explain the result. Fixed Generation moves farther from the initial weights than Closed Loop (.01387.01387 versus .00857.00857) but causes far less damage. Real-Text Learning moves farther still (.01759.01759) and improves NLL by .3544.3544 nats [.1236,.6048][.1236,.6048]. Generated origin is also insufficient: Fixed Generation learns from generated text without closing the feedback loop. The distinguishing feature is therefore that the updated weights determine the text used for later updates.

Fixed Generation necessarily changes the generated sequence, so it cannot tell us what happens after a degraded sequence has already been produced. Closed Loop ends almost entirely repetitive (repeated-4 =.9902=.9902), unlike Writes Off (.0217.0217) and Fixed Generation (.0082.0082), but that correlation does not separate the effect of reading the degraded text from the effect of learning from it. Recorded Replay holds the tokens fixed to separate the two.

### 5.2 Recorded Replay: process the same text with and without writing

A receiver adapted on different books processes a saved degraded stream that it did not generate. From the same starting state, it first reads the recording without updating. We then restore the state and process the identical tokens again while retaining their updates. The first comparison measures the cost of keeping degraded tokens in attention; the difference between the two passes measures the additional cost stored in the weights.

Table 4: Degraded content harms through both attention and weights. The 125M read-only excess shows that the recording is harmful in attention; the paired read-and-write difference measures the additional persistent cost. Scale | Receiver operation on the same recorded text | Real-text NLL above control  
---|---|---  
125M | Read only | 1.6061.606  
125M | Read and write | 3.8743.874  
125M | Additional cost of writing | 2.268\mathbf{2.268}  
3B | Additional cost of writing | .193​[.104,.327]\mathbf{.193\;[.104,.327]}  
  
At 125M, reading alone adds 1.6061.606 nats over the receiver’s control, so the degraded content is already harmful while it remains in attention. Reading and writing adds 3.8743.874 nats; the extra 2.2682.268 nats is the cost stored by the weight updates. A 3B replication measures the same write-minus-read contrast across six source and six disjoint receiver books. Writing adds .1927.1927 [.1043,.3272][.1043,.3272] nats, and every source- and receiver-book aggregate is positive (Appendix C.2).

Figure 3: Feedback, not parameter drift, amplifies the cost of writing. (a) Matched 125M control: the same four policies are ordered across relative drift and final clean-text NLL; horizontal bars are 95% book intervals. (b) Identical recorded tokens with updates disabled or retained; only this panel holds text fixed. Both panels use eight books and five seeds. (c) In a separate paired one-update comparison, the real-text cost of retaining one update rises sharply over a Closed Loop history, but stays near zero when earlier generated writes were discarded or came from Fixed Generation.

Figure 3 links the three results. Panel (a) shows that parameter movement does not rank damage. Panel (b) holds degraded text fixed and isolates the added harm from writing it. Panel (c) follows one-update cost through the stream: it rises sharply only after Closed Loop adaptation and stays near zero under Writes Off and Fixed Generation. Feedback therefore changes both future training text and the cost of later writes.

### 5.3 One update can fit its source and hurt another input

The preceding endpoints combine many writes, so they do not show what one update itself changes. We isolate that effect by copying the same weights and attention cache immediately before a generated chunk. Both copies process the same chunk; one keeps its update and the other discards it. Any later difference is therefore caused by retaining that single update. We measure whether it improves prediction of its source chunk, whether that improvement transfers to the next independent real passage, and whether later generation becomes more repetitive. Table 5 reports _keep minus skip_ for the three outcomes.

The three rows describe the state before this paired update. _Writes Off History_ means all earlier generated-text writes were discarded; _Closed Loop_ means they were retained; and _Fixed Generation_ means earlier updates were learned from chunks generated by frozen W0W_{0}. Negative training-chunk NLL means that keeping the update fits its source better; positive next-real-text NLL means that the same update harms the new passage.

Table 5: An update can improve source fit while harming new text. 125M; keep minus discard, with 95% book intervals. Preceding History |  Training-Chunk NLL (↓\downarrow better fit) |  Next Real-Text NLL (↑\uparrow worse) |  Future Repeated-4 (↑\uparrow more repetition)  
---|---|---|---  
Writes Off History | −.1374​[−.1453,−.1285]-.1374\;[-.1453,-.1285] | +.0045​[.0037,.0053]+.0045\;[.0037,.0053] | +.0049​[−.0079,.0248]+.0049\;[-.0079,.0248]  
Closed Loop | −.0354​[−.0384,−.0317]-.0354\;[-.0384,-.0317] | +.0381​[.0098,.0718]+.0381\;[.0098,.0718] | +.0086​[.0020,.0184]+.0086\;[.0020,.0184]  
Fixed Generation | −.1449​[−.1482,−.1407]-.1449\;[-.1482,-.1407] | +.0018​[.0014,.0023]+.0018\;[.0014,.0023] | +.0017​[−.0034,.0063]+.0017\;[-.0034,.0063]  
  
In every row, retaining the update lowers mean source-chunk NLL but raises mean NLL on the next real passage. Better source fit therefore does not guarantee a benefit on new text. Fixed Generation has the largest source-fit improvement (0.14490.1449) and the smallest real-text cost (0.00180.0018), but even that cost is positive. Closed Loop has a smaller source improvement (0.03540.0354) and a much larger real-text cost (0.03810.0381). Within Closed Loop, the mean cost also grows from 0.0060.006 nats at the earliest measured position to 0.1130.113 at the latest. Both the model state and its generated text change over this history; the comparison measures their combined effect on later writes.

Table 6: Two gradient signals rank harmful updates. Entries are Spearman correlations with the measured increase in next-real-text NLL after keeping rather than discarding one update. (a) Negative ρ\rho means that more-negative gradient cosine predicts greater harm. (b) Positive ρ\rho means that a larger first-order predicted loss increase matches greater measured harm. Each model-history setting contains 96 paired updates.

(a) Cosine vs. measured harm

History | 125M ρ\rho | 760M ρ\rho  
---|---|---  
Closed Loop | −.502​[−.642,−.305]-.502\;[-.642,-.305] | −.633​[−.808,−.383]-.633\;[-.808,-.383]  
Writes Off | −.788​[−.826,−.720]-.788\;[-.826,-.720] | −.731​[−.810,−.666]-.731\;[-.810,-.666]  
  
(b) Predicted vs. measured harm

History | 125M ρ\rho | 760M ρ\rho  
---|---|---  
Closed Loop | +.704+.704 | +.632+.632  
Writes Off | +.794+.794 | +.751+.751  
  
#### Gradient conflict predicts which updates are more harmful.

Table 6 asks a simple question: before keeping an update, can we tell whether learning the generated chunk will conflict with predicting new real text? We score each candidate before applying it, then use the controlled comparison from Table 5: two identical copies keep or discard the update and predict the same real passage. The Keep copy’s extra loss is the measured harm.

For panel (a), we compute the cosine between the generated-chunk gradient and the real-passage gradient for each candidate, then correlate the 96 cosine scores with their 96 measured harms. The correlations are −.502/−.788-.502/-.788 after Closed Loop/Writes Off history at 125M and −.633/−.731-.633/-.731 at 760M; all four intervals exclude zero. Thus greater gradient opposition accompanies greater harm. For panel (b), we compute greal⊤​Δ​Wg_{\rm real}^{\top}\Delta W for each candidate and correlate these predicted loss changes with the same measured harms. The correlations are +.704/+.794+.704/+.794 at 125M and +.632/+.751+.632/+.751 at 760M. These are rank correlations, not NLL increases. Both calculations consistently identify the same pattern: candidates with stronger generated–real conflict cause more real-text harm. This explains poor transfer at one update; it is not a selection rule in these experiments: it uses the real passage on which harm is measured (Appendix C.4).

#### What the causal decomposition establishes.

The comparisons establish: (1) Feedback: Fixed Generation prevents updates from changing later text and removes most damage. (2) Reading and writing: Replay holds tokens fixed and separates harm from reading them from the additional harm stored in weights. (3) Transfer: one update can fit its source yet harm new real text, and gradient conflict ranks that harm. The last comparison also shows that average one-update cost grows after Closed Loop adaptation. We next ask whether most late writes become harmful or whether a small subset drives that increase.

## 6 Which Updates Should Be Retained?

### 6.1 Damage amplifies in degraded states but remains heterogeneous

The average one-update cost rises later in Closed Loop. Does this increase affect most passages, or come from a small harmful subset?

Figure 4: Damage concentrates in a few late passages. (a) Mean and median passage-level write cost; (b) number of costs above 0.50.5 nats among 72 paired comparisons at each position.

For each passage, identical receivers process the same tokens as Read Only or Read+Write, then predict the same real text. We define _write cost_ =NLL(Read+Write)−-NLL(Read Only); a positive value is harm caused by retaining the updates, beyond reading the same tokens. Each position has 72 paired costs (Appendix D).

At the first position, mean and median costs are similar (0.04390.0439 and 0.04220.0422), with no value above 0.50.5 nats. At the final position, the mean rises to 0.75120.7512 but the median falls to 0.00640.0064: the typical cost does not increase with the mean. Only 12 of 72 late costs exceed 0.50.5. All twelve come from four of 24 source sequences. After a source–receiver pair first exceeds the threshold, its later sampled passages remain above it. Thus a few source sequences repeatedly produce high-cost passages and raise the late mean; position alone cannot identify a safe write.

Within repetition bins, the estimated early-to-late cost change is +.0587+.0587 [−.0382,.3286][-.0382,.3286]; this comparison does not resolve a change. The intervention results provide more direct evidence: repetition weighting and read-only real context leave 54%54\% and 48%48\% of the Closed Loop gap, whereas an eight-chunk reset removes 94.2%94.2\% of the canonical gap but retains only 44.1%44.1\% of useful adaptation. These results expose a trade-off: milder controls leave substantial damage, whereas the stronger reset also removes most of the adaptation benefit. Neither evaluates whether a particular update helps prediction beyond its source. Section 6.2 evaluates that effect directly (Appendices B.2 and D.2).

### 6.2 External validation measures whether an update transfers

Settlement checks an update before making it permanent. From current fast weights WW, it forms a temporary candidate W+δW+\delta and compares both states on the same independent real-text passage qq, without changing attention:

| A⁡(δ,W,q)=L⁡(q,W)−L⁡(q,W+δ).A(\delta;W,q)=L(q;W)-L(q;W+\delta). |  | (2)  
---|---|---|---  
  
Positive AA means that the candidate predicts qq better, so Settlement commits δ\delta only when A≥0A\geq 0. Figure 4 locates large costs accumulated over an eight-chunk passage; Settlement makes the finer, candidate-level decision: an update must not increase loss on qq. Under this rule, 22/936 and 18/312 candidates are accepted at 125M and 760M. Each candidate is evaluated on top of earlier accepted updates, because changes that appear safe separately can be harmful when combined (Appendix E.5). Settlement uses the next arriving real passage, or retained prefix text when none arrives. _The rule does not use the candidate’s source label_ , but it requires identifiable real text for the comparison. Algorithm 1 in Appendix E.1 gives the full procedure.

Settlement leaves endpoint gaps of .07.07 nats at 125M and −.02-.02 at 760M (Figure 5a); both intervals include zero. The improvement comes from validation rather than merely smaller updates or better context. Without validation, quarter-sized writes, read-only real text, and their combination leave .90.90, 1.041.04, and .28.28 nats; adding Settlement puts every variant within .04.04 nats of Writes Off (Appendix E.3). On a matched all-real stream, Settlement accepts 89 of 113 candidates. Relative to Reject All, it reduces the first-to-last NLL change by .0534.0534 nats [.0299,.0721][.0299,.0721]; its paired advantage over Write All is .0083.0083 [.0020,.0161][.0020,.0161]. Thus validation retains measurable real-text learning (Appendix E.4).

Figure 5: Independent evidence evaluates transfer. (a) Settlement leaves little endpoint residual. (b) Source Masking deteriorates under corrupted labels; Settlement remains stable. Lower is better in (a–b). (c) Task-metric Settlement uses WebShop reward to select updates; the plot shows exact success (mean ±\pm SD; five paired seeds; shared Writes Off). Higher is better. Details are in Appendices E–F and G.

#### When is external validation useful beyond Source Masking?

Source Masking keeps real-text writes and rejects writes labeled as generated; Settlement instead evaluates the proposed state. With correct labels, Source Masking is simpler and has slightly lower NLL (3.80723.8072 versus 3.81963.8196). We then corrupt the observed source labels while leaving the underlying stream unchanged. Source Masking begins to accept generated writes and reject real ones, so its NLL rises. Settlement instead evaluates the candidate’s prediction. Its mean NLL is lower from 21.2%21.2\% observed error; paired intervals exclude zero at the evaluated error rates of 38.9%38.9\% and above (Figure 5b). Thus Source Masking is simpler with reliable labels; validation is more robust to label errors in this experiment. Full results appear in Appendix F.

Source Masking cannot choose among updates learned from an agent’s own actions: every candidate has the same “generated” label. Rejecting them all is Writes Off; retaining them all is Closed Loop. WebShop (Yao et al., 2022) provides this setting through prompt–action updates with model-generated action targets. Settlement instead checks validation reward, accepts 10 of 180 candidate blocks, and reaches .1853.1853 exact success, compared with .1053.1053 for Closed Loop and .1600.1600 for the shared Writes Off reference; the adapting policies use five paired seeds (Figure 5c). The task metric therefore distinguishes updates that their common source label cannot (Appendix G). In ALFWorld household tasks (Shridhar et al., 2021), two of three Closed Loop seeds complete none of the 134 unseen tasks at each update scale. Settlement instead achieves 88.1%88.1\%–94.0%94.0\% unseen-task success across seeds, retaining 19%19\%–33%33\% of update blocks across the two scales (Appendix H).

## 7 Related Work

#### Writable inference-time state.

Fast weights provide memory that changes during sequence processing (Schmidhuber, 1992; Ba et al., 2016). Linear attention interprets recurrent state as a fast-weight matrix (Schlag et al., 2021), while dynamic evaluation updates parameters on recent context (Krause et al., 2018). TTT layers, TTT-E2E, Titans, and In-Place TTT bring learned update rules or writable memory to language models (Sun et al., 2025; Tandon et al., 2025; Behrouz et al., 2025; Feng et al., 2026). We study the case in which this state also generates its next training example.

#### Test-time adaptation and stability.

Test-time training adapts on deployment inputs through self-supervised losses, entropy minimization, or retrieved neighbors (Sun et al., 2020; Liu et al., 2021; Wang et al., 2021; Hardt and Sun, 2024). Continual methods limit erroneous pseudo-label accumulation through teacher averaging, restoration, or reset (Arazo et al., 2020; Wang et al., 2022; Niu et al., 2023; Press et al., 2023). Our setting differs because the adapted model produces later inputs, a within-stream form of performative prediction (Perdomo et al., 2020). Fixed Generation and Recorded Replay separate changes to future text, attention, and weights.

#### Generated-data feedback and update selection.

Recursive synthetic-data studies show that training successive model generations on generated samples can reduce quality or diversity, while mixing in real data can slow or prevent collapse (Briesch et al., 2023; Shumailov et al., 2024; Alemohammad et al., 2024; Marchi et al., 2024; Gerstgrasser et al., 2024). Neural decoding has a related but distinct failure: locally likely tokens can produce repetitive continuations, motivating nucleus sampling and repetition-aware objectives (Holtzman et al., 2020; Welleck et al., 2020; Xu et al., 2022). Our object of study is one adapting model in one stream, where weights change later text and learning from that text changes the weights again. Decoder and replay controls show why diversity and synthetic origin alone do not determine persistent cost.

Methods for continual learning constrain gradients or preserve performance on stored examples (Lopez-Paz and Ranzato, 2017; Chaudhry et al., 2019); validation-based reweighting favors examples whose gradients improve held-out data (Ren et al., 2018). Self-improvement methods likewise learn from generated rationales, feedback, or successful trajectories (Zelikman et al., 2022; Huang et al., 2023; Shinn et al., 2023). More closely related, aTTT studies within-episode feedback from online updates and downweights repeated update tokens (Wang et al., 2026); VANE validates visual updates on later observations (Ji et al., 2026). Our controlled comparisons isolate the feedback pathway in long text streams; Settlement evaluates the accumulated fast-weight state on independent real text before retaining an update.

## 8 Discussion and Conclusion

#### What causes the failure?

With a frozen generator, the learner updates on synthetic text yet stays near Writes Off. Loss grows when adapted weights also generate later training chunks: each update can change both the model and its future data. Late harm is concentrated: four of 24 source sequences account for every fixed-receiver write cost above 0.50.5 nats. Affected source–receiver pairs remain above that threshold at later sampled positions. The loop makes some trajectories persistently harmful, not every late passage.

#### How does the causal decomposition support this explanation?

Fixed Generation keeps the generated source and update count but breaks the link from updates to later training text, removing 98.3%98.3\% and 98.8%98.8\% of the matched Closed Loop gap at 125M and 760M. Recorded Replay then gives identical receivers the same degraded tokens: reading raises real-text loss, and retaining updates raises it further, including at 3B. Finally, one update fits its generated chunk better but predicts the next real passage worse; that cost grows after Closed Loop adaptation. The controls locate the failure in feedback and poor transfer, not synthetic origin alone. In WebShop, breaking feedback likewise raises mean exact success from .1053.1053 to .1800.1800 across five paired seeds.

#### What reduces the damage?

Real text works best when it repeatedly interrupts self-writing: with 31%31\% real chunks, even spacing leaves .0694.0694 nats of excess harm versus .7155.7155 for bursts. Decoder changes reduce harm without identifying which update helps. Settlement directly compares current and proposed states on independent real text before commitment. In separate experiments, it nearly matches Writes Off on generated streams and retains useful updates on real-text streams. Source Masking is simpler with reliable labels; at tested error rates of 38.9%38.9\% and above, Settlement has lower NLL when real validation text is available. The criterion is whether the complete state to be retained, including combined parallel updates, improves performance beyond its source.

## References

  * Alemohammad et al. (2024) S. Alemohammad, J. Casco-Rodriguez, L. Luzi, A. I. Humayun, H. Babaei, D. LeJeune, A. Siahkoohi, and R. G. Baraniuk Self-consuming generative models go MAD.  In International Conference on Learning Representations (ICLR),  Cited by: §7. 
  * Arazo et al. (2020) E. Arazo, D. Ortego, P. Albert, N. E. O’Connor, and K. McGuinness Pseudo-labeling and confirmation bias in deep semi-supervised learning.  In International Joint Conference on Neural Networks (IJCNN),  Cited by: §7. 
  * Ba et al. (2016) J. Ba, G. E. Hinton, V. Mnih, J. Z. Leibo, and C. Ionescu Using fast weights to attend to the recent past.  In Advances in Neural Information Processing Systems (NeurIPS),  Cited by: §1, §2, §7. 
  * Behrouz et al. (2025) A. Behrouz, P. Zhong, and V. Mirrokni Titans: learning to memorize at test time.  In Advances in Neural Information Processing Systems (NeurIPS),  Vol. 38.  Cited by: §1, §7. 
  * Briesch et al. (2023) M. Briesch, D. Sobania, and F. Rothlauf Large language models suffer from their own output: an analysis of the self-consuming training loop.  arXiv preprint arXiv:2311.16822.  Cited by: §1, §7. 
  * Chaudhry et al. (2019) A. Chaudhry, M. Ranzato, M. Rohrbach, and M. Elhoseiny Efficient lifelong learning with A-GEM.  In International Conference on Learning Representations (ICLR),  Cited by: §7. 
  * Feng et al. (2026) G. Feng, S. Luo, K. Hua, G. Zhang, W. Huang, D. He, and T. Cai In-place test-time training.  In International Conference on Learning Representations (ICLR),  Cited by: §7. 
  * Gerstgrasser et al. (2024) M. Gerstgrasser, R. Schaeffer, A. Dey, R. Rafailov, H. Sleight, J. Hughes, T. Korbak, R. Agrawal, D. Pai, A. Gromov, et al. Is model collapse inevitable? breaking the curse of recursion by accumulating real and synthetic data.  arXiv preprint arXiv:2404.01413.  Cited by: §7. 
  * Hardt and Sun (2024) M. Hardt and Y. Sun Test-time training on nearest neighbors for large language models.  In International Conference on Learning Representations (ICLR),  Cited by: §7. 
  * Holtzman et al. (2020) A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi The curious case of neural text degeneration.  In International Conference on Learning Representations (ICLR),  Cited by: §3, §7. 
  * Hu et al. (2022) E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen LoRA: low-rank adaptation of large language models.  In International Conference on Learning Representations (ICLR),  Cited by: Appendix G, Appendix H. 
  * Huang et al. (2023) J. Huang, S. S. Gu, L. Hou, Y. Wu, X. Wang, H. Yu, and J. Han Large language models can self-improve.  In Conference on Empirical Methods in Natural Language Processing (EMNLP),  Cited by: §7. 
  * Ji et al. (2026) H. Ji, G. Xia, L. Sun, F. Feng, and L. Ren VANE: reliable test-time training for vision-language-action models via future visual representation prediction.  arXiv preprint arXiv:2608.09448.  Cited by: §1, §7. 
  * Kingma and Ba (2015) D. P. Kingma and J. Ba Adam: a method for stochastic optimization.  In International Conference on Learning Representations (ICLR),  Cited by: §1, §3, §4. 
  * Krause et al. (2018) B. Krause, E. Kahembwe, I. Murray, and S. Renals Dynamic evaluation of neural sequence models.  In International Conference on Machine Learning (ICML),  Proceedings of Machine Learning Research, Vol. 80, pp. 2766–2775.  Cited by: §7. 
  * Li et al. (2024) J. Li et al. DataComp-lm: in search of the next generation of training sets for language models.  In Advances in Neural Information Processing Systems (NeurIPS),  Vol. 37.  Cited by: Appendix A, §3. 
  * Liu et al. (2024) X. Liu, H. Yu, H. Zhang, Y. Xu, X. Lei, H. Lai, Y. Gu, H. Ding, K. Men, K. Yang, S. Zhang, X. Deng, A. Zeng, Z. Du, C. Zhang, S. Shen, T. Zhang, Y. Su, H. Sun, M. Huang, Y. Dong, and J. Tang AgentBench: evaluating LLMs as agents.  In International Conference on Learning Representations (ICLR),  Cited by: Appendix H. 
  * Liu et al. (2021) Y. Liu, P. Kothari, B. van Delft, B. Bellot-Gurlet, T. Mordan, and A. Alahi TTT++: when does self-supervised test-time training fail or thrive?.  In Advances in Neural Information Processing Systems (NeurIPS),  Vol. 34, pp. 21808–21820.  Cited by: §1, §7. 
  * Lopez-Paz and Ranzato (2017) D. Lopez-Paz and M. Ranzato Gradient episodic memory for continual learning.  In Advances in Neural Information Processing Systems (NeurIPS),  Cited by: §7. 
  * Marchi et al. (2024) M. Marchi, S. Soatto, P. Chaudhari, and P. Tabuada Heat death of generative models in closed-loop learning.  In IEEE Conference on Decision and Control (CDC),  pp. 1524–1530.  External Links: [Document](https://dx.doi.org/10.1109/CDC56724.2024.10886816) Cited by: §7. 
  * Niu et al. (2023) S. Niu, J. Wu, Y. Zhang, Z. Wen, Y. Chen, P. Zhao, and M. Tan Towards stable test-time adaptation in dynamic wild world.  In International Conference on Learning Representations (ICLR),  Cited by: §7. 
  * Perdomo et al. (2020) J. Perdomo, T. Zrnic, C. Mendler-Dünner, and M. Hardt Performative prediction.  In International Conference on Machine Learning (ICML),  pp. 7599–7609.  Cited by: §7. 
  * Press et al. (2023) O. Press, S. Schneider, M. Kümmerer, and M. Bethge RDumb: a simple approach that questions our progress in continual test-time adaptation.  In Advances in Neural Information Processing Systems (NeurIPS),  Cited by: §7. 
  * Qwen Team (2026) Qwen Team Qwen3.8-27B.  Note: Official model card External Links: [Link](https://huggingface.co/Qwen/Qwen3.8-27B) Cited by: Appendix H. 
  * Rae et al. (2020) J. W. Rae, A. Potapenko, S. M. Jayakumar, and T. P. Lillicrap Compressive transformers for long-range sequence modelling.  In International Conference on Learning Representations (ICLR),  Cited by: Appendix A, §3. 
  * Ren et al. (2018) M. Ren, W. Zeng, B. Yang, and R. Urtasun Learning to reweight examples for robust deep learning.  In International Conference on Machine Learning (ICML),  Cited by: §7. 
  * Schlag et al. (2021) I. Schlag, K. Irie, and J. Schmidhuber Linear transformers are secretly fast weight programmers.  In International Conference on Machine Learning (ICML),  Cited by: §1, §7. 
  * Schmidhuber (1992) J. Schmidhuber Learning to control fast-weight memories: an alternative to dynamic recurrent networks.  Neural Computation 4 (1), pp. 131–139.  Cited by: §1, §2, §7. 
  * Shinn et al. (2023) N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao Reflexion: language agents with verbal reinforcement learning.  In Advances in Neural Information Processing Systems (NeurIPS),  Cited by: §7. 
  * Shridhar et al. (2021) M. Shridhar, X. Yuan, M. Côté, Y. Bisk, A. Trischler, and M. Hausknecht ALFWorld: aligning text and embodied environments for interactive learning.  In International Conference on Learning Representations (ICLR),  Cited by: Appendix H, §6.2. 
  * Shumailov et al. (2024) I. Shumailov, Z. Shumaylov, Y. Zhao, N. Papernot, R. Anderson, and Y. Gal AI models collapse when trained on recursively generated data.  Nature 631, pp. 755–759.  Cited by: §1, §7. 
  * Sun et al. (2025) Y. Sun, X. Li, K. Dalal, J. Xu, A. Vikram, G. Zhang, Y. Dubois, X. Chen, X. Wang, S. Koyejo, T. Hashimoto, and C. Guestrin Learning to (learn at test time): rnns with expressive hidden states.  In International Conference on Machine Learning (ICML),  Proceedings of Machine Learning Research, Vol. 267, pp. 57503–57522.  Cited by: §1, §7. 
  * Sun et al. (2020) Y. Sun, X. Wang, Z. Liu, J. Miller, A. A. Efros, and M. Hardt Test-time training with self-supervision for generalization under distribution shifts.  In International Conference on Machine Learning (ICML),  pp. 9229–9248.  Cited by: §1, §7. 
  * Tandon et al. (2025) A. Tandon, K. Dalal, X. Li, D. Koceja, M. Rød, S. Buchanan, X. Wang, J. Leskovec, S. Koyejo, T. Hashimoto, C. Guestrin, J. McCaleb, Y. Choi, and Y. Sun End-to-end test-time training for long context.  arXiv preprint arXiv:2512.23675.  Cited by: §1, §1, §2, §3, Figure 1, §7. 
  * Vaswani et al. (2017) A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin Attention is all you need.  In Advances in Neural Information Processing Systems (NeurIPS),  Cited by: §2. 
  * Wang et al. (2021) D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T. Darrell Tent: fully test-time adaptation by entropy minimization.  In International Conference on Learning Representations (ICLR),  Cited by: §7. 
  * Wang et al. (2022) Q. Wang, O. Fink, L. Van Gool, and D. Dai Continual test-time domain adaptation.  In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),  Cited by: §7. 
  * Wang et al. (2026) Y. Wang, J. Hao, Y. Shi, K. Yuan, and M. Sun No time like the present: agentic test-time training for llm agents.  arXiv preprint arXiv:2607.03441.  Cited by: §1, §1, §7. 
  * Welleck et al. (2020) S. Welleck, I. Kulikov, S. Roller, E. Dinan, K. Cho, and J. Weston Neural text generation with unlikelihood training.  In International Conference on Learning Representations (ICLR),  Cited by: §7. 
  * Xu et al. (2022) J. Xu, X. Liu, J. Yan, D. Cai, H. Li, and J. Li Learning to break the loop: analyzing and mitigating repetitions for neural text generation.  In Advances in Neural Information Processing Systems (NeurIPS),  Vol. 35.  Cited by: §7. 
  * Yang et al. (2025) A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. Qwen3 technical report.  arXiv preprint arXiv:2505.09388.  Cited by: Appendix G, §1, §3, §4. 
  * Yao et al. (2022) S. Yao, H. Chen, J. Yang, and K. Narasimhan WebShop: towards scalable real-world web interaction with grounded language agents.  In Advances in Neural Information Processing Systems (NeurIPS),  Cited by: Appendix G, Appendix G, §6.2. 
  * Zelikman et al. (2022) E. Zelikman, Y. Wu, J. Mu, and N. Goodman STaR: bootstrapping reasoning with reasoning.  In Advances in Neural Information Processing Systems (NeurIPS),  Cited by: §7. 



## Appendix A Experimental scope and implementation

The experiments answer three questions: whether persistent self-writing causes long-horizon damage, which part of the feedback loop causes it, and whether independent evidence can screen a proposed update. Every conclusion comes from a treatment and control with the shared settings listed below; online generation may produce different text under different policies. Unless stated otherwise, language-model intervals are 95% bootstrap intervals over books after averaging seeds within each book.

#### Stable book identities.

The cross-scale, Fixed Generation, and decoder experiments use the same six PG-19 documents, denoted _canonical books 2–7_ , with five seeds and a 128K horizon. Exposure and validation require longer uninterrupted documents, so each uses one fixed long-book set shared by every policy in that comparison. The manifests record document hashes, selection thresholds, and scripts.

#### Protocol map.

Table 7 maps each paper claim to its shared configuration and controlled comparison. Long-horizon HH and EE use Writes Off as the reference; local comparisons hold the received text fixed and compare retaining versus discarding its updates. HH compares first-to-last changes; EE compares final endpoints and is used for intervention policies. They coincide when the initial losses match.

Table 7: Main experiment configurations and why they differ. Each row compares policies under one shared configuration; books are the independent units for language-model intervals. Question |  Shared configuration |  Controlled comparison  
---|---|---  
Long horizon |  125M, 760M, 3B; canonical six books; five seeds; 128K; width 8 |  Closed Loop versus Writes Off  
Feedback path |  125M and 760M; canonical six books; five seeds; 128K; Fixed Generation added |  Closed Loop versus Fixed Generation  
Read vs. write |  Fixed recorded tokens; 125M: eight receiver books, five seeds; 3B: six source and six disjoint receiver books |  Read Only versus Read + Write  
Single update |  Eight books and four stream positions; identical state and source text in each branch |  Keep versus discard one update  
Decoder/exposure |  Decoder: canonical setup; exposure: one eight-book long-stream set, five seeds |  Decoder endpoints versus a common Writes Off reference; evenly spaced versus bursty real text  
Validation |  Scale-specific long-book sets; each scale has its own paired Writes Off trajectory |  Closed Loop versus Settlement  
Agents (WebShop) |  Qwen3-4B LoRA; five paired seeds; fixed test-goal evaluation |  Closed Loop, Fixed Generation, Settlement, Writes Off  
Agents (ALFWorld) |  Qwen3.8-27B LoRA; three paired 274-task streams; prequential scoring; update scales 8 and 16 |  Closed Loop, Settlement, Writes Off  
  
#### Model and training.

The 125M TTT-E2E model has 12 layers and width 768, with an 8192-token sliding attention window in the first nine layers and a SwiGLU fast-weight MLP in each of the final three layers. Each sequence has its own fast weights. One clipped inner-loop SGD step follows each 1024-token chunk. Unless a decoder control is specified, temperature is one and top-pp is .95.95. Training follows the official 4800-step recipe on 2.5B DCLM tokens (Li and others, 2024), followed by the 32K length-extension schedule on PG-19 (Rae et al., 2020) in place of Books. The experiment manifests record complete SHA-256 digests for the 125M model file and the PG-19 validation array.

#### Conversion and numerical checks.

The PyTorch and official-format models have identical parameter counts. Conversion changes the sequence loss by 3.6×10−53.6\times 10^{-5} over eight inner-loop chunks; every chunk loss differs by less than 10−410^{-4}. A rematerialized inner-loop backward agrees with a naive second-order implementation to 6×10−166\times 10^{-16} in fp64. For the released 3B/128K model, the converted training forward changes sequence loss by 1.05×10−41.05\times 10^{-4} and gives token-level NLL correlation 0.99960.9996. This verifies the converted training forward, not exact parity of the 3B streaming decoder with the official decoder.

#### Streaming evaluation.

The standard 128K schedule starts with eight real-text chunks, followed by generated chunks and periodic real-text evaluations. The original schedule has 15 branch evaluations and 105 generated chunks; the new cross-scale suite reports 16 NLL values including its initial reference. Evaluation never updates weights. Evaluations snapshot and restore the complete carried state, so evaluation text does not become later generation context. The state contains fast weights, attention caches, positions, held updates, and partial-chunk buffers. We distinguish a condition’s first-to-last NLL change DcD_{c} from the paired excess Hc=Dc−DoffH_{c}=D_{c}-D_{\mathrm{off}}. When initial losses are matched exactly, this equals the final-evaluation difference. Other tables explicitly use the endpoint gap Ec=Lc,last−Loff,lastE_{c}=L_{c,\mathrm{last}}-L_{\mathrm{off},\mathrm{last}}.

#### Write strength, retention, and state restoration.

The write-strength multiplier scales the clipped update, not the loss before clipping. For w∈{1/8,1/4,1/2,1,2,4}w\in\\{1/8,1/4,1/2,1,2,4\\}, a single-step norm obeys ‖Δ​W​(w)‖/(w​‖Δ​W​(1)‖)=1\|\Delta W(w)\|/(w\|\Delta W(1)\|)=1 to a worst relative error of 1.15×10−71.15\times 10^{-7}; w=0w=0 gives exactly zero update. Loss scaling before a saturated norm clip would not implement the same controlled operation. Over 12 chunks, net drift is not linear in ww because update directions partly cancel. Retention applies W←W0+λ⁡(W−W0)W\leftarrow W_{0}+\lambda(W-W_{0}) once per chunk, including chunks whose proposals are rejected. With no accepted updates, six chunks at λ=.9\lambda=.9 give exactly the expected displacement factor .531441.531441. Snapshot–restore returns all 39 carried tensors bitwise identical and four scalars equal. A teacher-forced 12-chunk regression check with interleaved evaluations matches the losses obtained without evaluation exactly. These checks do not establish bitwise reproducibility of all stochastic generation trajectories.

### A.1 Generator isolation and chunk boundaries

The Fixed Generation control uses a separate cache advanced only by the frozen generator. On two books, perturbing receiver weights, KV, positions, and buffers leaves this generator unchanged. Keyed per-row sampling is invariant to batch reordering, and a writes-off null comparison is bitwise identical.

The canonical implementation starts each generated chunk with a real boundary token, and every paired policy uses the same rule. Teacher-forcing recorded tokens is not bitwise equivalent to online decoding; the comparison gives a maximum logit difference .1875.1875 and mean KL .000287.000287.

### A.2 Feedback and displacement control

This suite uses the 125M model, eight receiver books, five seeds, 128 chunks, and the canonical boundary rule. All four policies use the same code path. Real-text evaluations snapshot and restore state. The analysis first averages seeds within book, then uses a 20,000-draw paired bootstrap over the eight books. CUDA sampling and the disabled token-level entropy observer are recorded in the experiment manifest.

Table 8: Matched 125M feedback and displacement control. Drift is ‖Wt−W0‖/‖W0‖\|W_{t}-W_{0}\|/\|W_{0}\|. Real-Text Learning contains no generated tail, so token-diversity statistics are not applicable. Policy | Final clean NLL | Final relative drift | Tail Distinct-2 | Tail Repeated-4  
---|---|---|---|---  
Closed Loop | 6.50846.5084 | .00857.00857 | .0084.0084 | .9902.9902  
Writes Off | 3.67763.6776 | .00500.00500 | .8493.8493 | .0217.0217  
Fixed Generation | 3.74833.7483 | .01387.01387 | .8779.8779 | .0082.0082  
Real-Text Learning | 3.32323.3232 | .01759.01759 | — | —  
  
Paired endpoint contrasts are: Closed Loop minus Writes Off 2.83082.8308 [1.8171,3.8598][1.8171,3.8598]; Fixed Generation minus Writes Off .0707.0707 [.0510,.0924][.0510,.0924]; Real-Text Learning minus Writes Off −.3544-.3544 [−.6048,−.1236][-.6048,-.1236]; and Closed Loop minus Fixed Generation 2.76012.7601 [1.7555,3.7881][1.7555,3.7881]. Fixed Generation and Real-Text Learning move farther from initialization than Closed Loop while causing far less harm. Parameter displacement is therefore not a sufficient explanation of the closed-loop loss. The independent Fixed Generation suite in Appendix B remains a replication under a separate protocol.

### A.3 Model-specific contamination screening

The first two candidate books have anomalously low loss under the larger released models. We compare unadapted first-chunk loss against the 125M reproduction. The larger ratios on books 0–1 are consistent with model-specific familiarity, but cannot establish the exact training membership of any book. Table 9 defines the ratio explicitly. The screened 3B set uses canonical books 2–7. Screening removes a visible anomaly; it does not prove absence of all contamination. The 760M and 3B training lineages also differ, so their damage magnitudes do not establish a scaling law.

Table 9: Unadapted loss ratio L125​M/LmodelL_{125M}/L_{\mathrm{model}} on the same book. A large ratio means unusually low loss under the larger model. Model | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7  
---|---|---|---|---|---|---|---|---  
760M, DCLM | 3.69 | 1.44 | 1.15 | 1.23 | 1.18 | 1.19 | 1.14 | 1.18  
3B, DCLM | 10.35 | 4.12 | 1.30 | 1.33 | 1.29 | 1.32 | 1.30 | 1.28  
3B, Books finetune | 8.83 | 2.67 | 1.44 | 1.69 | 1.44 | 1.41 | 1.37 | 1.46  
  
## Appendix B Replication and boundaries of the failure regime

This appendix follows Section 4. It first confirms Fixed Generation at a second model scale, then asks which properties of the stream change the severity of the failure. The evidence gives two clear boundaries: decoder choice changes both when harm becomes visible and how fast it grows, whereas regular real-text writes sharply suppress it.

### B.1 Fixed Generation confirmation at 125M and 760M

The three-condition suite compares Closed Loop, Writes Off, and Fixed Generation. Table 10 evaluates both scales on the same six canonical books with five seeds per book. Initial evaluations agree across policies within each scale, so the endpoint contrasts are directly comparable within a row and use the same first-to-last definition as Table 1.

Table 10: Fixed Generation confirmation on the same six receiver books. Intervals resample books after averaging seeds. Model | Condition | Initial NLL | Final NLL | Final excess over Writes Off [95% CI]  
---|---|---|---|---  
125M | Writes Off | 3.3469 | 3.3845 | 0  
| Fixed Generation | 3.3469 | 3.4361 | .0516 [.0248,.0772]  
| Closed Loop | 3.3469 | 6.3942 | 3.0097 [1.8575,4.1972]  
760M | Writes Off | 2.7913 | 2.9383 | 0  
| Fixed Generation | 2.7913 | 3.0116 | .0733 [.0538,.0939]  
| Closed Loop | 2.7913 | 8.9405 | 6.0021 [4.2695,7.7347]  
  
At both scales, fixing the generator leaves only a small fraction of the Closed Loop endpoint gap. The independent 3B confirmation in Appendix C.2 instead holds the received text fixed and measures the additional cost of retaining its updates.

### B.2 Decoder and reset controls change the rate of damage

Six decoder policies and two reset schedules use the same 125M model, canonical books 2–7, five seeds, and a 128K schedule with 105 generated chunks. All rows use the same book-blocked estimator as Table 1. The repetition penalty is applied to tokens already generated within the current chunk; typical-p=.95p=.95 replaces rather than composes with top-pp sampling. A reset restores fast weights to their initial values every KK chunks but leaves the attention cache intact.

Table 11: Canonical decoder and reset controls. The endpoint gap uses the same canonical Writes Off endpoint for every row; late slope is the mean change per evaluation over the final four intervals. Lower is better in both columns. Control |  Endpoint gap to canonical Writes Off |  Late slope (nats/evaluation)  
---|---|---  
top-p=.95p=.95 | 3.00973.0097 | +.1339+.1339  
top-p=.99p=.99 | 3.35313.3531 | +.1497+.1497  
typical-p=.95p=.95 | 2.92922.9292 | +.0948+.0948  
full support | 1.45201.4520 | +.1979+.1979  
T=.7T=.7, top-p=.95p=.95 | 2.22462.2246 | +.1704+.1704  
repetition penalty 1.11.1 | .8794\mathbf{.8794} | +.0027\mathbf{+.0027}  
reset every 8 chunks | .1749\mathbf{.1749} | +.0056\mathbf{+.0056}  
reset every 32 chunks | 1.92181.9218 | +.0623+.0623  
  
Endpoint and late slope answer different questions. Full-support sampling has a smaller 128K endpoint than the default decoder, but its late slope is the largest in the table: it delays the visible failure without flattening the curve. Fixed Generation, reset every eight chunks, and the repetition penalty all have late slopes below .006.006; the remaining continuing controls have late slopes above .062.062. Reset frequency matters sharply. Resetting every eight chunks leaves an endpoint gap of .1749.1749 [.084,.321][.084,.321] (5.8% of the canonical Closed Loop damage), whereas resetting every 32 leaves a gap of 1.92181.9218 [.949,3.158][.949,3.158] (63.9%). Thus a short reset window is an effective safety baseline, but it also limits how long adaptation can persist.

We measure that trade-off on real-text streams. Ordinary persistent writing improves NLL by .0850.0850 [.0444,.1260][.0444,.1260] over Writes Off. Resetting every eight chunks preserves a .0375.0375 [.0195,.0526][.0195,.0526] benefit, or 44.1% of the persistent adaptation gain. Reset therefore controls runaway accumulation by shortening the lifetime of every update, including useful ones. The repetition penalty also changes the entire future stream; neither control decides whether a particular fixed passage should be written.

For intuition, consider an idealized SGD step at a fixed context. Sampling from qW​(x)∝pW​(x)1/Tq_{W}(x)\propto p_{W}(x)^{1/T} on fixed support AA and applying η​∇W​log⁡pW​(x)\eta\nabla_{W}\log p_{W}(x) gives expected increment η​T​∇W​log​∑x∈ApW​(x)1/T\eta T\nabla_{W}\log\sum_{x\in A}p_{W}(x)^{1/T}. At T=1T=1 with full support, this mean is zero; truncation instead increases retained probability mass to first order. The derivation assumes an infinitesimal unclipped step and a fixed context and therefore does not predict a zero finite-horizon effect. The paired experiment above supplies the relevant empirical result: truncation amplifies damage but is not required for it.

### B.3 External-text exposure bounds the failure regime

We vary the number and placement of real-text write slots while keeping the 125M model, the dedicated long-book set [0,1,27,28,29,43,44,46][0,1,27,28,29,43,44,46], five seeds, 8K real prefix, 105 live slots, temperature one, and top-p=.95p=.95 fixed. _All Writes_ retains updates from both real and generated slots. _Writes Off_ retains real-text updates but discards generated-text writes while still reading those tokens. Thus HH is the excess NLL change caused by retaining generated-text writes at each exposure density. The exposure sweep needs longer uninterrupted documents than the canonical comparison, so every exposure policy instead uses the same eight long books. The main-text panel reports H/H0%H/H_{0\%} within this set; the table below gives the corresponding absolute values.

Table 12: Real-text cadence controls excess NLL change. HH is the excess NLL change of retaining generated writes; BB is the benefit of retaining real-text writes. Intervals use a paired book bootstrap after averaging seeds. Real fraction | Schedule | Real slots | Max self burst | HH [95% CI] | BB [95% CI]  
---|---|---|---|---|---  
0% | Even | 0 | 105 | 1.1858;[.5326,1.9430]1.1858;[.5326,1.9430] | .0000;[.0000,.0000].0000;[.0000,.0000]  
5% | Even | 5 | 17 | 1.0275;[.5679,1.6195]1.0275;[.5679,1.6195] | .1350;[.0557,.2578].1350;[.0557,.2578]  
10% | Even | 11 | 8 | .4389;[.2517,.6465].4389;[.2517,.6465] | .0933;[.0729,.1155].0933;[.0729,.1155]  
20% | Even | 21 | 4 | .0806;[.0657,.0981].0806;[.0657,.0981] | .0802;[.0585,.1052].0802;[.0585,.1052]  
31% | Even | 33 | 3 | .0694;[.0534,.0843].0694;[.0534,.0843] | .1051;[.0668,.1431].1051;[.0668,.1431]  
31% | Bursty | 33 | 12 | .7155;[.3730,1.1578].7155;[.3730,1.1578] | .1040;[.0502,.1527].1040;[.0502,.1527]  
  
Increasing evenly spaced real-text exposure reduces excess NLL change while preserving a positive real-text adaptation benefit. Density is not the whole story: at 31%, grouping the same 33 real slots together increases the longest uninterrupted stretch of self-generated chunks from 3 to 12 and increases HH by an order of magnitude. The controlled quantity is therefore real-text cadence: both how much real text arrives and the longest interval without it.

### B.4 Teacher-forced real-text payoff

The matched 125M payoff experiment uses eleven fixed teacher-forced real-text books. Benefit is the loss under _No Updates_ minus the loss under the evaluated writing policy; positive values mean that adaptation helps. Ordinary sequential writing gives .0340.0340 nats of benefit. Because the text is fixed, the improvement cannot come from changing future training tokens. The same suite evaluates whether validation preserves this gain, including the failure of individually validated but jointly committed updates; the full comparison appears in Table 26.

## Appendix C Causal decomposition: feedback, attention, and persistent weights

This appendix follows the causal order of Section 5. We first separate the generator from the learner, then replay identical text with and without updates, and finally branch before a single update. Each step removes one source of variation, moving from the full online loop to a paired local comparison.

### C.1 Fixed Generation and Recorded Replay

The separate eight-book displacement suite is reported in Appendix A.2: Closed Loop minus Writes Off is 2.83082.8308 [1.8171,3.8598][1.8171,3.8598], while Fixed Generation minus Writes Off is .0707.0707 [.0510,.0924][.0510,.0924]. The matched 760M result is reported in Appendix B. These controls show that continuing to update on generated text need not reproduce the large closed-loop loss. They change subsequent text as well as its dependence on adapting weights, so they do not hold content fixed.

The Recorded Replay comparison instead supplies the same recorded tokens to the receiver with writing enabled or disabled. Relative to the suite’s Writes Off control, Recorded Replay without writes gives 1.6061.606 nats of excess and replay with writes gives 3.8743.874, an additional 2.2682.268 nats. This paired comparison isolates the contribution of writing that replayed text. Comparing live closed generation with replay does not isolate an additional feedback effect, because they contain different text. The 3B experiment below is the large-model confirmation.

### C.2 3B fixed-text replay confirmation

This experiment evaluates whether retaining updates from a fixed recorded stream adds cost beyond reading exactly the same tokens. Six source books are disjoint from six receiver books. Two cyclic source–receiver mappings and five independently sampled recordings supply the matched read/write comparisons. For each pair,

| J=(Lwrite,last−Lwrite,first)−(Lread,last−Lread,first).J=(L_{\rm write,last}-L_{\rm write,first})-(L_{\rm read,last}-L_{\rm read,first}). |   
---|---|---  
  
The paired trajectories have exactly equal initial NLL and identical received token hashes. A 20,000-draw two-way source/receiver cluster bootstrap gives J=0.1927J=0.1927 [0.1043,0.3272][0.1043,0.3272]; clustering only by source gives 0.19270.1927 [0.1181,0.2941][0.1181,0.2941]. Both analyses account for shared source books rather than treating trajectories as independent samples.

Table 13: The 3B passage-level write cost is positive across every source and receiver aggregate. The statistical interval is clustered by source and receiver, not by individual trajectory.

Source book | Mean JJ | Leave-one-out JJ  
---|---|---  
12 | .1862 | .1940  
13 | .1953 | .1922  
14 | .0617 | .2189  
15 | .1331 | .2046  
16 | .1602 | .1992  
17 | .4196 | .1473  
  
Receiver book | Mean JJ  
---|---  
2 | .2872  
3 | .1824  
4 | .1302  
5 | .1042  
6 | .1343  
7 | .3178  
  
### C.3 Paired one-update comparison

The 125M experiment branches at generated positions 1, 33, 65, and 97 from Closed Loop, Writes Off, or Fixed Generation histories. The branches share the starting state and current text; one retains that text’s update and the other skips it. Four draws of four future chunks use paired random numbers. All 15 history trajectories and 60 paired configurations completed, giving 480 book-level observations. We average seeds and positions within each book before a 20,000-draw paired book bootstrap. Each row in Table 14 is a paired keep-minus-discard comparison within its starting states; differences between rows also change the text and history, and do not isolate receiver susceptibility alone.

Table 14: Keep minus discard for one update: negative source NLL improves fitting; positive real NLL harms independent prediction. Intervals are 95% book-bootstrap intervals. History | Δ\Delta source NLL | Δ\Delta real NLL | Δ\Delta future repeated-4  
---|---|---|---  
Closed Loop | −.0354​[−.0384,−.0317]-.0354\;[-.0384,-.0317] | .0381​[.0098,.0718].0381\;[.0098,.0718] | .0086​[.0020,.0184].0086\;[.0020,.0184]  
Writes Off | −.1374​[−.1453,−.1285]-.1374\;[-.1453,-.1285] | .0045​[.0037,.0053].0045\;[.0037,.0053] | .0049​[−.0079,.0248].0049\;[-.0079,.0248]  
Fixed Generation | −.1449​[−.1482,−.1407]-.1449\;[-.1482,-.1407] | .0018​[.0014,.0023].0018\;[.0014,.0023] | .0017​[−.0034,.0063].0017\;[-.0034,.0063]  
Table 15: Mean one-update real-text NLL effects by position, 125M. Positions differ in both current text and receiver state. History | Position 1 | 33 | 65 | 97  
---|---|---|---|---  
Closed Loop | .0062 | .0087 | .0250 | .1126  
Writes Off | .0063 | .0044 | .0038 | .0037  
Fixed Generation | .0042 | .0008 | .0011 | .0012  
  
The paired comparison uses matched random numbers and measures update norms from the applied parameter increment. Reimplementation checks reproduce the reported aggregate outcomes; the analysis concerns paired outcome differences rather than bitwise execution traces.

### C.4 Gradient conflict and one-update transfer cost

#### Purpose.

The paired experiment in Section 5.3 shows that retaining one generated-text update can worsen prediction of new real text. Here we ask whether that harm is related to a conflict already present before the update: does the generated chunk ask the weights to move in a direction that opposes what the next real passage needs?

#### Quantities computed for each candidate.

Let WW be the shared weights before branching, xx the generated chunk, and qq the next independent real passage. We compute both gradients at WW:

| gself=∇WL​(x,W),greal=∇WL​(q,W).g_{\rm self}=\nabla_{W}L(x;W),\qquad g_{\rm real}=\nabla_{W}L(q;W). |   
---|---|---  
  
The optimizer uses xx to produce the actual candidate change Δ​W\Delta W, including the same clipping and normalization used in the main experiment. The Keep branch applies W+Δ​WW+\Delta W; the Discard branch remains at WW. With attention held fixed, their difference on qq is

| Δ​Lreal=L⁡(q,W+Δ​W)−L⁡(q,W).\Delta L_{\rm real}=L(q;W+\Delta W)-L(q;W). |   
---|---|---  
  
This is the measured one-update cost. Positive Δ​Lreal\Delta L_{\rm real} means that retaining the update made the real passage harder to predict. We also report Δ​Lself=L⁡(x,W+Δ​W)−L⁡(x,W)\Delta L_{\rm self}=L(x;W+\Delta W)-L(x;W); a negative value confirms that the same update improved its generated source.

#### Two pre-update signals.

The first signal is gradient cosine,

| scos=gself⊤​greal∥gself∥​∥greal∥.s_{\rm cos}=\frac{g_{\rm self}^{\top}g_{\rm real}}{\lVert g_{\rm self}\rVert\lVert g_{\rm real}\rVert}. |   
---|---|---  
  
When scos<0s_{\rm cos}<0, the two gradients point in opposing directions. A step that reduces the generated-chunk loss then tends to increase real-text loss. The second signal uses the actual proposed change,

| sfirst=greal⊤​Δ​W≈Δ​Lreal.s_{\rm first}=g_{\rm real}^{\top}\Delta W\approx\Delta L_{\rm real}. |   
---|---|---  
  
It is the first-order prediction of the real-text loss change: larger positive values predict greater harm.

#### Correlation analysis.

For each model and preceding history, eight books, three seeds, and four stream positions give 8×3×4=968\times 3\times 4=96 candidates. We compute Spearman correlation across these candidates. If conflict predicts harm, scoss_{\rm cos} should have negative correlation with Δ​Lreal\Delta L_{\rm real}, whereas sfirsts_{\rm first} should have positive correlation. Spearman correlation evaluates whether each signal ranks the more harmful candidates; its value is not an NLL increase. The reported intervals use book-level resampling.

Table 16: Gradient conflict predicts the measured cost of retaining one generated-text update. Δ​Lreal\Delta L_{\rm real} is keep minus discard on the next real passage; negative cosine means the source and real-text gradients oppose one another. Negative mean Δ​Lself\Delta L_{\rm self} indicates improved source fit on average. Model / history | ρ⁡(cos,Δ​Lreal)\rho(\cos,\Delta L_{\rm real}) | ρ⁡(greal⊤​Δ​W,Δ​Lreal)\rho(g_{\rm real}^{\top}\Delta W,\Delta L_{\rm real}) | Δ​Lself\Delta L_{\rm self} | Mean cos\cos  
---|---|---|---|---  
125M / Closed Loop | −.502​[−.642,−.305]-.502\;[-.642,-.305] | .704​[.521,.829].704\;[.521,.829] | −.033-.033 | −.093-.093  
125M / Writes Off | −.788​[−.826,−.720]-.788\;[-.826,-.720] | .794​[.721,.838].794\;[.721,.838] | −.136-.136 | −.005-.005  
760M / Closed Loop | −.633​[−.808,−.383]-.633\;[-.808,-.383] | .632​[.449,.781].632\;[.449,.781] | −.051-.051 | −.103-.103  
760M / Writes Off | −.731​[−.810,−.666]-.731\;[-.810,-.666] | .751​[.685,.828].751\;[.685,.828] | −.166-.166 | −.004-.004  
  
#### Results.

All four cosine correlations are negative (−.502-.502 to −.788-.788), and all four first-order correlations are positive (.632.632 to .794.794); all eight confidence intervals exclude zero. Thus, in both models and both preceding histories, the candidate updates with stronger generated–real conflict cause larger measured real-text costs.

The mean cosine adds a state-level observation. Generated and real gradients are nearly orthogonal after Writes Off history (−.005-.005 at 125M and −.004-.004 at 760M), but become more opposing after Closed Loop history (−.093-.093 and −.103-.103). This is not a generic property of any two gradients: two real-text gradients have positive mean cosine in the same states (.125/.481.125/.481 at 125M and .097/.481.097/.481 at 760M, Writes Off/Closed Loop). The analysis therefore identifies a specific conflict between fitting generated text and transferring to real text. It explains which single updates are harmful; Settlement instead evaluates the complete accumulated state before committing it.

### C.5 In-place Adam: dose and history are separate considerations

Qwen3-4B freezes all parameters except the final four mlp.down_proj matrices and takes one Adam step per chunk, with gradient norm clipped at one. It uses no outer meta-learning or added modules. Eight books, three seeds, batches of four, eight real warm-up chunks, and 128 total chunks supply Table 17. The latest 8192 tokens are re-encoded each chunk; naive KV truncation raised evaluation loss from about 3.04 to 8.85, while rebuilding restored it to 3.02. This numerical check motivates the reported window handling. The comparison isolates ordinary Adam updates to existing model parameters rather than an added TTT module.

Table 17: In-place Adam long-horizon control. The larger-rate excess is 1.2311.231 [.580,2.350]; the smaller-rate excess is .026.026 [-.071,.122]. Condition | NLL change DD | Relative drift | Distinct-2 | Repeated-4  
---|---|---|---|---  
Writes Off | -.060 | .00000 | .140 | .831  
Adam 10−510^{-5} | -.034 | .00627 | .187 | .775  
Adam 10−410^{-4} | 1.171 | .02399 | .008 | .991  
  
To verify that the larger learning rate is not simply unusable, we also adapt the same parameter subset on teacher-forced real text. The future tokens are fixed and therefore cannot be changed by the update. Positive B=Lno​update,last−Lreal​update,lastB=L_{\rm no\ update,last}-L_{\rm real\ update,last} means that writing improves prediction on independent real text.

Table 18: The Qwen3-4B update configuration also learns useful real text. Two independently adapted states each contain four books; the bracket is their range, not a book-level confidence interval. Adam LR | BB [range of two state means] | State 0 | State 1  
---|---|---|---  
10−510^{-5} | .1392​[.1150,.1634].1392\;[.1150,.1634] | .1634.1634 | .1150.1150  
10−410^{-4} | .1680​[.1317,.2043].1680\;[.1317,.2043] | .2043.2043 | .1317.1317  
  
Both learning rates improve independent real-text prediction in both adapted states. Together with Table 17, this shows that the 10−410^{-4} configuration can help on real text while harming prediction when its future training stream is self-generated.

## Appendix D From heterogeneous harm to partial repairs

Section 6 shows that late damage is heavy-tailed rather than uniform. This appendix first gives the underlying distribution, then evaluates three natural responses: weighting repeated tokens less, shrinking every write, and inserting real text without writing it. Each response reduces harm, but none directly determines whether a particular update transfers.

### D.1 A minority of source passages drives the late mean

The fixed-receiver sweep replays eight-chunk passages from 24 source rows into three receiver offsets at seven stream positions. For each passage, write damage is subsequent real-text NLL after read-and-write minus NLL after read-only processing from the same receiver state. Receiver offsets show whether a source effect transfers across books; they do not create additional independent source trajectories.

Table 19: Fixed-receiver stage replay. The increasing mean is produced by a growing right tail; the median remains near zero. Source stage | Median | Mean | Standard deviation | Cases above .5 nats  
---|---|---|---|---  
1 | .0422 | .0439 | .025 | 0/72  
17 | .0040 | .0898 | .385 | 3/72  
33 | .0039 | .1399 | .620 | 3/72  
49 | .0042 | .3135 | 1.009 | 6/72  
65 | .0047 | .4855 | 1.274 | 9/72  
81 | .0105 | .4858 | 1.267 | 9/72  
97 | .0064 | .7512 | 1.731 | 12/72  
  
All cases above .5.5 nats come from four of the 24 source rows. The same rows produce large effects at all three receiver offsets. There are 24×3×7=50424\times 3\times 7=504 paired costs in total. After first exceeding .5.5 nats, those source–receiver pairs remain above it at later sampled positions. Receivers are reset between comparisons; this measures the harm of later source passages, not a receiver’s recovery. A separate repetition-bin comparison gives an unresolved early-to-late change of .0587.0587 [−.0382,.3286][-.0382,.3286]. Stream position and repetition therefore indicate risk but do not identify every harmful passage.

### D.2 Repetition weighting mitigates but does not remove damage

The expanded policy suite uses sixteen books and five seeds. Random and uniform controls reduce update weight without using token identity; the comparison asks whether repetition-aware weighting helps beyond simply learning less.

Table 20: Repetition weighting and real context reduce the endpoint gap but do not reach Writes Off. Lower EE is better. Policy | Final NLL | EE [95% interval] | Distinct-2  
---|---|---|---  
Writes Off | 3.5767 | 0 | .8425  
Closed Loop | 6.4620 | 2.8853 [2.0072,3.8310] | .0082  
Repetition weighting | 5.1404 | 1.5637 [1.1858,1.9825] | .0469  
Random dose | 5.6325 | 2.0558 [1.6339,2.4946] | .0086  
Uniform dose | 6.0628 | 2.4861 [1.9510,3.0939] | .0155  
Read-only real context | 4.9746 | 1.3979 [1.0988,1.7140] | .0730  
  
Repetition weighting improves on random and uniform dose assignment by .4921.4921 and .9224.9224 nats, respectively, while leaving 1.56371.5637 nats above Writes Off. The result supports repetition as a useful symptom, not a sufficient admission rule. Because these are online policies, later generated text also differs.

### D.3 Smaller writes expose a stability–adaptation trade-off

We scale each post-clipping parameter increment by α∈{0,1/16,1/8,1/4,1/2,1,2}\alpha\in\\{0,1/16,1/8,1/4,1/2,1,2\\}. The same scale is applied to teacher-forced real-text writes. Generated-stream harm HH and real-text benefit BB are both measured relative to α=0\alpha=0.

Table 21: Uniformly smaller writes reduce both harm and useful adaptation. Every evaluated nonzero dose has positive values for both quantities. α\alpha | Excess NLL change HH [95% CI] | Real-text benefit BB [95% CI]  
---|---|---  
00 | .0000;[.0000,.0000].0000;[.0000,.0000] | .0000;[.0000,.0000].0000;[.0000,.0000]  
1/161/16 | .3856;[.1737,.6432].3856;[.1737,.6432] | .0412;[.0249,.0579].0412;[.0249,.0579]  
1/81/8 | .8583;[.5480,1.2276].8583;[.5480,1.2276] | .0543;[.0328,.0758].0543;[.0328,.0758]  
1/41/4 | .9248;[.6525,1.3374].9248;[.6525,1.3374] | .0682;[.0413,.0942].0682;[.0413,.0942]  
1/21/2 | 1.0319;[.6424,1.4540]1.0319;[.6424,1.4540] | .0812;[.0507,.1106].0812;[.0507,.1106]  
11 | 1.1867;[.8512,1.5056]1.1867;[.8512,1.5056] | .0922;[.0579,.1248].0922;[.0579,.1248]  
22 | 1.9374;[1.0941,2.9775]1.9374;[1.0941,2.9775] | .1009;[.0626,.1373].1009;[.0626,.1373]  
  
Reducing α\alpha delays and reduces damage, but it also reduces the payoff on real text. Uniform scaling therefore moves along a stability–adaptation frontier rather than separating useful and harmful writes.

### D.4 Real read-only context helps because of its content

Every eighth generation slot is replaced by a read-only slot, leaving 92 generated-text writes in every condition. The model receives no update in that slot. A matched no-op condition skips the same slots, so the comparison separates the content read in attention from the pause in generation.

Table 22: Read-only content at a matched write schedule. Negative differences improve prediction relative to skipping the slot. Read-only content | Final NLL | Difference from no-op [95% CI]  
---|---|---  
No-op | 6.4042 | 0  
Self-generated | 6.7214 | .3172 [-.4775,1.1963]  
Random tokens | 7.3969 | .9927 [-.2699,2.0992]  
Shuffled real | 5.5680 | -.8362 [-1.7643,.0324]  
Fixed Generation text | 5.7912 | -.6130 [-1.5127,.2200]  
Real text | 4.8349 | -1.5693 [-2.3011,-.8433]  
  
Real text lowers final NLL by 1.56931.5693 nats relative to the no-op schedule and outperforms random, shuffled-real, and Fixed Generation anchors in direct paired comparisons. Thus real text can interrupt the loop through attention without being stored in weights. In the separate policy comparison, read-only real context ends 1.39791.3979 nats above Writes Off (Table 20), so it is a partial repair rather than a complete selection rule.

## Appendix E Evaluating transfer with independent evidence

The mechanism experiments show that source fit does not certify transfer. This appendix turns that observation into a controlled question: if a candidate state predicts independent real text better, can it be retained without recreating the closed loop? We define the sequential rule, report its evidence requirements and scale results, and then separate validation from weaker updates and better context. The hybrid policy evaluates a pending candidate on arriving real text before learning from that text, and uses retained real text when no new external text arrives. Each candidate is compared with the currently accepted state using the same validation context. Accepted candidates update that reference before the next candidate is evaluated. The decision does not inspect the candidate’s source label, but selecting trusted real validation text does require knowing which evidence is real. Evaluation passages are separate from admission text. The policy uses external validation; it does not claim that validation itself is a new learning principle.

### E.1 Sequential validation and commitment

Algorithm 1 describes the decision rule for an ordered list of pending fast-weight increments. Each increment was computed from its source chunk before the validation passage became eligible; the passage must not be the text used to produce that increment. In the hybrid policy, new external text is preferred and retained real text supplies a fallback. Outcome-evaluation passages are not used to choose updates.

1: Accepted fast weights WW; ordered pending increments PP; eligible new real text qnewq_{\rm new}; retained real-text bank BB; fixed validation context CC

2: q←qnewq\leftarrow q_{\rm new} if available; otherwise eligible text from BB

3: if no eligible qq is available then

4: return (W,P)(W,P) ⊳\triangleright Wait; buffer capacity is configuration-specific 

5: end if

6: S←WS\leftarrow W

7: for each δ\delta in PP, in proposal order do

8: ℓ0←ReadOnlyNLL​(q,S,C)\ell_{0}\leftarrow\textsc{ReadOnlyNLL}(q,S,C)

9: ℓ1←ReadOnlyNLL​(q,S+δ,C)\ell_{1}\leftarrow\textsc{ReadOnlyNLL}(q,S+\delta,C)

10: if ℓ1≤ℓ0\ell_{1}\leq\ell_{0} then

11: S←S+δS\leftarrow S+\delta ⊳\triangleright Next candidate is judged against accepted changes 

12: end if

13: Remove this candidate from PP; record losses and decision 

14: end for

15: Commit W←SW\leftarrow S

16: return (W,P)(W,P)

Algorithm 1 Settlement processes pending fast-weight updates sequentially.

ReadOnlyNLL restores the same attention context, positions, buffers, and other carried state for both evaluations and leaves the live stream unchanged. It does not train on qq. Only accepted increments change the committed fast weights; ordinary reading of arriving input occurs outside this procedure. The next increment is evaluated against the accumulated accepted state, not the original WW. Candidate increments are not jointly summed after separate approval against an unchanged reference.

For Quarter Dose, the proposed increment is one quarter of the clipped generated-text update before it enters PP. The validation comparison thus judges the same increment that would be committed. This differs from scaling the loss before clipping, accepting a full update and scaling it afterwards, or decaying previously accepted weights. Retention is not part of this ablation’s validation rule.

Evidence scheduling and pending-buffer capacity are implementation choices; the reported experiments use the schedules stated with each comparison.

### E.2 Cross-scale outcomes

Table 23: Completed Settlement outcomes and later admission-log extraction. The two scales use separate matched suites and Writes Off trajectories. Model / books | Closed Loop minus Writes Off | Settlement minus Writes Off | Generated admitted  
---|---|---|---  
125M / 12 | 3.183 [2.106,4.253] | .067 [-.013,.169] | 22/936  
760M / 4 | 1.263 [.799,1.618] | -.021 [-.058,.014] | 18/312  
  
At 125M, Settlement improves on Closed Loop on 11 of 12 books; at 760M, it improves all four books. The residual intervals include zero but are not equivalence bounds and do not establish superiority to Writes Off. Real-text acceptance is 83% and 100% in the respective log summaries.

### E.3 Component ablation

This four-book, three-seed comparison separates external validation from two weaker controls: quarter-sized updates and periodic read-only real context. Table 24 reports final NLL above Writes Off. Without validation, each control leaves a visible gap. With Settlement, every combination finishes within .04.04 nats of Writes Off. Adding read-only real context to Settlement lowers NLL by a further .057.057 nats [.024,.089][.024,.089]; the incremental effect of Quarter Dose is not resolved.

Table 24: Completed component ablation, 125M, four books and three seeds. Rounded interval endpoints at zero are not equivalence statements. Without validation | Excess [95% interval] | With Settlement | Excess [95% interval]  
---|---|---|---  
Closed Loop | 3.06 [.67,7.02] | Settlement | .04 [.00,.09]  
Quarter Dose | .90 [.32,1.97] | \+ Quarter Dose | .02 [-.01,.04]  
Read-Only Real Context | 1.04 [.19,2.59] | \+ Real Context | -.02 [-.03,.00]  
Both controls | .28 [.13,.53] | \+ Both controls | .00 [-.01,.02]  
  
### E.4 Admission and adaptation in one all-real stream

A rule can match Writes Off by rejecting every proposal. We therefore measure admission and adaptation together on an all-real stream. The three policies process the same six long-stream books with three seeds. _Reject All_ supplies the no-write baseline, _Write All_ retains every real-text proposal, and Settlement applies the same sequential rule used above. Because the first evaluation occurs after adaptation has begun, benefit is the policy’s first-to-last NLL improvement relative to Reject All.

Table 25: Settlement admits most real-text candidates and preserves their adaptation benefit in the same matched experiment. Positive benefit is better; intervals resample books after averaging seeds. Policy | Real admitted | Benefit [95% interval]  
---|---|---  
Reject All | 0/1130/113 | 00  
Write All | 113/113113/113 | .0450​[.0235,.0609].0450\;[.0235,.0609]  
Settlement | 𝟖𝟗/𝟏𝟏𝟑\mathbf{89/113} | .0534​[.0299,.0721]\mathbf{.0534\;[.0299,.0721]}  
  
Settlement admits 78.8% of the real-text proposals. Its paired benefit over Write All is .0083.0083 [.0020,.0161][.0020,.0161]. The low admission rate on self-generated streams is therefore a property of those candidate updates, not a fixed tendency to reject all learning.

### E.5 Preserving adaptation and validating the committed state

The scheduled gates below alternate admission and evaluation passages so the same passage does not both authorize and evaluate an update. Harm and benefit in Table 26 come from separate streams, not a joint experiment such as Table 25. Harm uses four books (five seeds for ordinary/scheduled gates; three for Settlement). Real-text benefit uses eleven teacher-forced books and _No Updates_ as its baseline.

Table 26: Validate the state that will be committed. Joint commit checks each candidate at WW but deploys their unevaluated sum; sequential policies judge the accumulated state SiS_{i}. Policy | Candidate judged against | State committed | HH (↓\downarrow) | BB (↑\uparrow)  
---|---|---|---|---  
Ordinary writing | — | W+δtW+\delta_{t} | +1.56+1.56 | +.034+.034  
Joint commit | WW (each δ\delta) | W+∑iδiW+\sum_{i}\delta_{i} | ≈0\approx 0 | −1.093-1.093  
Sequential | Si−1S_{i-1} | SiS_{i} | +.10+.10 | +.019+.019  
Settlement | Si−1S_{i-1} on next qq | SiS_{i} | ≈0\approx 0 | +.037+.037  
Figure 6: Four ways to apply persistent updates. (a) Ordinary writing immediately changes the live state. (b) Joint commit validates candidates separately at WW but deploys their unevaluated sum. (c) Sequential validation judges each candidate against the accumulated accepted state. (d) Settlement keeps a candidate pending until arriving external text qq can evaluate the state that would be committed.

Table 26 reports the numerical comparison. Positive benefit improves on No Updates; missing intervals are not inferred from other rows. The individual gate accepts 113/113 real candidates computed against a shared base, then commits their combination. This committed state matches unconditional deferred commitment and has terminal drift .0862, versus .0176 for ordinary sequential writing. Sequential validation evaluates each next increment against the accumulated accepted state, retains about 56% of the ordinary real-text benefit, and avoids that large loss. This identifies a failure of this joint application; it does not rule out alternative batch normalization, increment recomputation, or learning-rate corrections. Settlement accepts 98–101 of 112 real candidates. Its point benefit is close to ordinary writing. Table 25 provides the matched admission–benefit comparison.

## Appendix F External validation when source labels are corrupted

Source Masking uses a real/generated label to decide whether a write should be retained. Settlement uses the observed labels only to locate real validation text; admission is determined by the candidate state’s validation advantage AA. We keep the underlying 125M mixed real/generated stream fixed and corrupt an increasing fraction of the observed labels. Each condition uses three seeds. All Writes retains every update, and Oracle Source Masking uses the uncorrupted labels.

Table 27: Corrupting source labels harms Source Masking but has little effect on Settlement. Lower final NLL is better. The middle columns show the incorrect decisions made by Source Masking. Bold marks comparisons whose paired 95% interval excludes zero in favor of Settlement. Observed error | Generated kept | Real discarded | Source Masking NLL | Settlement NLL  
---|---|---|---|---  
.000 | 0 | 0 | 3.8072 | 3.8196  
.053 | 4 | 2 | 3.8116 | 3.8230  
.124 | 9 | 5 | 3.8129 | 3.8248  
.212 | 15 | 9 | 3.8377 | 3.8286  
.389 | 26 | 18 | 3.8584 | 3.8215  
.619 | 41 | 29 | 3.9242 | 3.8270  
.876 | 64 | 35 | 4.0204 | 3.8243  
1.000 | 73 | 40 | 4.2544 | 3.8244  
  
With accurate labels, Source Masking has the lower endpoint NLL. As errors increase, it retains generated writes and discards real writes; its NLL rises from 3.8072 to 4.2544. Settlement stays between 3.8196 and 3.8286 and becomes lower in mean NLL from 21.2% observed error. The paired Settlement-minus- Masking intervals exclude zero at the evaluated error rates .389, .619, .876, and 1.000; the mean difference at .212 is not resolved by its interval.

The stable result is not caused by a constant admission policy. At 100% observed-label error, regrouping Settlement decisions by true source gives 120/120 real candidates admitted and 0/219 self-generated candidates admitted; regrouping the same decisions by the corrupted labels reverses these counts. The validation advantage AA therefore continues to separate the candidate states even when the labels used to locate validation text are wrong.

## Appendix G WebShop feedback and task-metric validation

The main experiments evaluate independent real-text prediction. WebShop (Yao et al., 2022) asks whether the same feedback path also changes task completion. The experiment uses persistent rank-16 LoRA updates (Hu et al., 2022) to Qwen3-4B (Yang et al., 2025) rather than TTT-E2E. The candidate LoRA update is optimized on prompt–action pairs whose action targets are generated by the agent. All candidates therefore receive the same source label. A source-only rule can either reject all of them, giving Writes Off, or retain all of them, giving Closed Loop; it cannot select among them. Other agent TTT methods can instead learn from environment observations or trajectory summaries, as discussed in Section 7. _Writes Off_ uses the initial model and retains no action updates. _Closed Loop_ retains updates, so the adapted model chooses the actions that form later training episodes. _Fixed Generation_ uses the same update rule, but a frozen copy W0W_{0} supplies every action for the learner. _Settlement_ accumulates updates in a temporary model while the committed model acts. It retains the candidate only when its mean validation reward is strictly higher. Final evaluations use the learner with writing disabled.

Table 28: WebShop outcomes across five paired seeds. Exact success is mean ±\pm SD; reward is the mean score. Writes Off is a shared deterministic reference. Retained counts pool 25-episode blocks across seeds. Policy | Reward | Exact success | Retained blocks  
---|---|---|---  
Writes Off | .3309.3309 | .1600.1600 | 00  
Closed Loop | .4356.4356 | .1053±.0597.1053\pm.0597 | 180/180180/180  
Fixed Generation | .3724.3724 | .1800±.0089.1800\pm.0089 | 180/180180/180  
Settlement | .3823.3823 | .1853±.0202\mathbf{.1853\pm.0202} | 10/18010/180  
  
The base model is frozen, and LoRA is applied to down_proj in the final quarter of layers (about 1.8M trainable parameters). Each written episode takes two Adam steps at 10−510^{-5} on at most 16 prompt–action pairs. Each policy processes 900 episodes and is evaluated on the same 150 goals from the test partition (Yao et al., 2022). Closed Loop, Fixed Generation, and Settlement use the same five seeds; Writes Off is the same deterministic reference in each comparison. The environment uses a fixed product catalogue, BM25 retrieval, and model-scored candidate actions, including template-generated search queries. Exact success means reward ≥1\geq 1, not merely making a purchase.

Fixed Generation raises exact success from .1053±.0597.1053\pm.0597 to .1800±.0089.1800\pm.0089 and beats Closed Loop in every paired seed. Settlement evaluates a candidate every 25 training episodes on 10 validation episodes. Validation and final evaluation use seeds 4321 and 1234, respectively, to select tasks from the test partition; the runner does not enforce disjoint goal sets. It accepts 10 of 180 candidate blocks (36 per seed) and reaches exact success .1853±.0202.1853\pm.0202, again beating Closed Loop in every paired seed. The feedback path therefore changes task behavior as well as language-model loss. These results compare policies within this task protocol; a common source label cannot make the distinctions supplied by validation reward.

## Appendix H ALFWorld: keeping the agent able to complete tasks

In ALFWorld (Shridhar et al., 2021), an agent reads room descriptions and uses text commands to complete household tasks. We ask whether learning from its own actions harms later task completion, and whether checking updates reduces that harm.

#### Tasks and learning.

The Qwen3.8-27B agent (Qwen Team, 2026) attempts 140 _seen_ tasks followed by 134 _unseen_ tasks, each with a 30-action limit. Each task is scored before its update; learned weights persist between tasks. We use LoRA (Hu et al., 2022), an AgentBench-style harness (Liu et al., 2024), and the official ALFWorld action parser. Scales 8 and 16 multiply the clipped weight change. Each policy uses seeds 1234, 2345, and 3456; both scales share Writes Off.

#### Three ways to handle an update.

_Writes Off_ keeps the original weights. _Closed Loop_ keeps every update and uses the changed model for the next task. _Settlement_ accumulates proposed updates in a temporary copy. Every 25 tasks, the current model and this copy attempt the same 20 validation tasks. Settlement keeps the copy only if it completes more tasks; otherwise it keeps the current model. It also checks at task 140 (validation seed 8765). A _block_ is a batch of updates accepted or rejected together: 12 per seed, or 36 per scale.

Figure 7: ALFWorld unseen-task success by update policy. Each open circle shows one seed’s success over 134 tasks; filled diamonds and the right-hand numbers show the three-seed mean. Points are vertically offset to separate overlapping values. Writes Off is shared across update scales. Table 29: ALFWorld success rates and update acceptance. Success rates are percentages (mean ±\pm sample SD across three seeds). Bold marks the highest mean per scale, including shared Writes Off, not statistical significance. Block counts pool seeds. |  | Success rate (%) | Accepted  
---|---|---|---  
Scale | Policy | Overall (274) | Seen (140) | Unseen (134) | blocks  
– | Writes Off | 82.182.1 ±\pm 0.40.4 | 77.6\mathbf{77.6} ±\pm 0.40.4 | 86.886.8 ±\pm 0.90.9 | –  
8 | Closed Loop | 43.943.9 ±\pm 36.336.3 | 59.559.5 ±\pm 25.625.6 | 27.627.6 ±\pm 47.847.8 | All updates  
Settlement | 83.8\mathbf{83.8} ±\pm 5.05.0 | 76.976.9 ±\pm 6.96.9 | 91.0\mathbf{91.0} ±\pm 3.03.0 | 12/3612/36  
16 | Closed Loop | 30.730.7 ±\pm 35.335.3 | 42.142.1 ±\pm 38.638.6 | 18.718.7 ±\pm 32.332.3 | All updates  
Settlement | 83.2\mathbf{83.2} ±\pm 2.22.2 | 76.776.7 ±\pm 4.64.6 | 90.0\mathbf{90.0} ±\pm 1.11.1 | 7/367/36  
  
#### What changes when updates are checked?

Halving the update scale still leaves two Closed Loop seeds with zero unseen success. Settlement completes 88.1%88.1\%–94.0%94.0\% and retains 19.4%19.4\%–33.3%33.3\% of update blocks. Across all 274 tasks, its mean success is 83.2%83.2\%–83.8%83.8\%, versus 82.1%82.1\% for Writes Off. These three-seed results show retained updates without the observed severe failures, but do not establish a reliable overall gain over Writes Off. Validation adds 20 tasks per model state and decision.

Experimental support, please [view the build logs](./2610.05076v1/__stdout.txt) for errors. Generated by [ L A T E xml ![\[LOGO\]](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==) ](https://math.nist.gov/~BMiller/LaTeXML/). 

## Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

  * Click the "Report Issue" ( ) button, located in the page header.



**Tip:** You can select the relevant text first, to include it in your report.

Our team has already identified [the following issues](https://github.com/arXiv/html_feedback/issues). We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a [list of packages that need conversion](https://github.com/brucemiller/LaTeXML/wiki/Porting-LaTeX-packages-for-LaTeXML), and welcome [developer contributions](https://github.com/brucemiller/LaTeXML/issues).

We gratefully acknowledge support from our **major funders** , [**member institutions**](https://info.arxiv.org/about/ourmembers.html) , ****, and all contributors.

[About](https://info.arxiv.org/about) * [Help](https://info.arxiv.org/help) * [Contact](https://info.arxiv.org/help/contact.html) * [Subscribe](https://info.arxiv.org/help/subscribe) * [Copyright](https://info.arxiv.org/help/license/index.html) * [Privacy](https://info.arxiv.org/help/policies/privacy_policy.html) * [Accessibility](https://info.arxiv.org/help/web_accessibility.html) * [Operational Status (opens in new tab)](https://status.arxiv.org)

Major funding support from

[ ![Simons Foundation](/static/base/1.0.1/images/funders/simons-foundation.png) ](https://www.simonsfoundation.org/) [ ![Simons Foundation International](/static/base/1.0.1/images/funders/simons-foundation-international.png) ](https://www.sfi.org.bm/) [ ![Schmidt Sciences](/static/base/1.0.1/images/funders/schmidt-sciences.png) ](https://www.schmidtsciences.org/)

[ ](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")
