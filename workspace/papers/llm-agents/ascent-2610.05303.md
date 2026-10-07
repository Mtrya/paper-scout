# 2610.05303 (from arXiv HTML)

##### Report GitHub Issue

×

Title:

Content selection saved. Describe the issue below:

Description:

Submit without GitHub Submit in GitHub

![](/static/base/1.0.1/images/icons/smileybones-small.svg) arXiv is now an independent nonprofit! [Learn more](https://info.arxiv.org/about) ×

[![arXiv logo](/static/base/1.0.1/images/arxiv-logo-primary-light.svg) Back to arXiv ](/)

[Why HTML?](https://info.arxiv.org/about/accessible_HTML.html) Report Issue [ Back to Abstract ](/abs/2610.05303v1 "Back to abstract page") [ Download PDF](/pdf/2610.05303v1 "Download PDF") [ ](javascript:toggleNavTOC\(\); "Toggle navigation") [ ](javascript:toggleReadingMode\(\); "Disable reading mode, show header and footer")

  1. Abstract
  2. 1 Introduction
  3. 2 Related Work
  4. 3 Method
     1. 3.1 Online Agentic Test-Time Training (OaTTT)
     2. 3.2 OaTTT via Direct Imitation or Reinforcement of Experience
     3. 3.3 ASCENT: OaTTT via Self-Distillation of Verified Experience
        1. 3.3.1 On-Policy Self-Distillation Objective
        2. 3.3.2 Distilling Verified and Filtered Experience Trajectories
  5. 4 Experiments
     1. 4.1 Experimental Setting
     2. 4.2 Main Experimental Results
     3. 4.3 Analysis and Discussion
     4. 4.4 Ablation Studies
  6. 5 Conclusion
  7. References
  8. 6 Additional Method Details
     1. 6.1 What Verifier Selection and Distillation Identify
        1. 6.1.1 Verifier Selection Reweights Completed Trajectories
        2. 6.1.2 Forward KL Fits Teacher Predictions at Student Prefixes
        3. 6.1.3 Verification Does Not Identify Shorter Paths
  9. 7 Additional Experimental Details
     1. 7.1 Evaluation Protocol and Comparison Controls
     2. 7.2 Baseline Details
     3. 7.3 Implementation Details
     4. 7.4 Full Benchmark Results
     5. 7.5 Additional Evaluation on AppWorld
  10. 8 Additional Analyses and Results
     1. 8.1 Teacher Evolution in OaTTT
     2. 8.2 Turn Efficiency under Success Composition
     3. 8.3 Retrieval and Execution for In-Context Adaptation



[ License: CC BY-NC-SA 4.0 ](https://info.arxiv.org/help/license/index.html#licenses-available)

arXiv:2610.05303v1 [cs.LG] 04 Oct 2026

\emails

, \metadata[Project Page][Artificer-AI-Lab/ASCENT](https://artificer-ai-lab.github.io/ASCENT/)

#  ASCENT: Online Test-Time Training of Long-Horizon Agents via Self-Distillation of Verified Experience

Haodong Lu  Dong Gong  Affiliation: University of New South Wales (UNSW Sydney)  Email: [haodong.lu@unsw.edu.au](mailto:haodong.lu@unsw.edu.au) Email: [dong.gong@unsw.edu.au](mailto:dong.gong@unsw.edu.au)

###### Abstract

A large language model (LLM) agent solves long-horizon tasks through many reasoning-action turns, with one verification signal at termination. Deployed agents face streams of related tasks, making their own trajectories a natural resource for improvement. In-context adaptation agents store reflections, memories, or skills as text at the harness level, so reuse depends on retrieving the right experience and on a frozen policy executing it. We study _Online Agentic Test-Time Training_ (OaTTT), which trains the LLM’s weights on the agent’s own execution trajectories during deployment. The agent executes each task once in one pass over the task stream, and the executed trajectory with its verification result is the only and immediate learning signal for weight updates that persist across tasks. Directly imitating or reinforcing the generated tokens of this single attempt destabilizes the policy. We introduce ASCENT (_Agentic Self-distillation for Cross-task EvolutioN at Test-time_), which instead self-distills verified experience. A stable version of the LLM, its frozen initial copy, receives the verifier-accepted trajectory as privileged information and predicts next-token distributions along it with this hindsight. Distilling them into persistent LoRA fast weights updates the agent for later tasks, without an external reference solution or stronger teacher. By further removing invalid-action turns from the trajectories, ASCENT can distill enhanced privileged experience for more effective and efficient execution. We characterize the population target of this distillation and the limits of sparse outcome selection. Across ALFWorld, WebShop, and AppWorld on varied model scales, ASCENT improves task success and interaction efficiency as experience accumulates, outperforms the compared online adaptation methods, and transfers to held-out scenes, showing that a deployed agent can consolidate its own verified experience into its policy without a separate training phase or memory retrieval at inference.

## 1 Introduction

A large language model (LLM) agent interacts with its environment to accomplish a task, iteratively reasoning about the current state and taking actions such as tool calls [51]. The LLM serves as the core of the agent’s decision policy. Agent tasks are often long-horizon, requiring multi-turn interactions with complex environments. Each completed interaction forms one episode, a trajectory of reasoning, actions, and observations.

Deployed agents often operate in persistent, long-running roles, encountering a continuous stream of tasks and interactions with users and environments. Because real-world tasks are complex and diverse, an agent built on a frozen LLM may not handle every situation it faces. The trajectories the agent gathers in the task stream are a natural resource for improving the agent at test time, as experience from earlier tasks can enhance performance on related tasks encountered later.

Relying on the in-context learning capability of LLMs, one line of work stores reflections on completed episodes, memories, and reusable skills as text at the harness level and reuses them in later interactions [32, 59]. Online variants accumulate this experience across tasks during deployment [26, 57, 10, 58]. Such in-context adaptation requires no parameter updates and changes only the model’s input, keeping the model itself frozen. Its gains can therefore be limited by how well relevant experience or skill is retrieved and how reliably the frozen policy executes it over a long interaction. Efforts are thus also taken to improve the core LLM for the agentic task process, via parametric adaptation on model weights. Some test-time training (TTT) methods adapt the LLM’s parameters within a single episode and reset them afterward, training on self-generated supervision (aTTT [45]) or QA pairs from the episode’s accumulated context (TMEM [28]). Other TTT methods require pre-deployment training of the adaptation behavior [62] or multiple sampled attempts per problem [61, 40]. Reinforcement learning (RL) post-training also improves the agentic capability of LLMs, which optimizes the policy before deployment on a training task set, typically with many attempts per task [30, 11, 38, 48, 23] or single-attempt RL with a learned critic [15]. Thus, none of them adapts a long-horizon agent’s weights at test time, in a single pass over the task stream with one attempt per task, while carrying updates across tasks.

To explore the potential of evolving LLMs through test-time training in deployment, we study _Online Agentic Test-Time Training_ (OaTTT), which trains the LLM’s weights directly on the agent’s own execution trajectories to improve task success. Tasks arrive in a single-pass stream, and the agent executes the task once, receiving verification from environment. Each executed trajectory and its verification result drive weight updates that persist across following tasks. Like in-context adaptation agents [26, 57], the agent learns from each trajectory once, without multiple trails. It accumulates learning in LLM weights, whereas those agents use textual harness-level memory. Learning occurs entirely at test time, without task-specific or offline training. We ask:

_Can a deployed agent turn its own verified execution trajectories, one per task, into persistent weight updates that improve its success on subsequent tasks?_

We propose ASCENT (_Agentic Self-distillation for Cross-task EvolutioN at Test-time_), which enables agents to learn during deployment, turning each verified experience trajectory into a persistent weight update. Training directly on these streaming trajectories is straightforward but fragile, because each task yields only one attempt by the current policy. Directly imitating or reinforcing the verified trajectories, with direct imitation or single-attempt policy gradient (Sec. 3.2), destabilizes the policy, erodes executable behavior (Fig. 1), and lowers task success. ASCENT instead _self-distills from the verified experience_ to enable stable online test-time training. A stable version of the LLM, its frozen initial copy, receives the verifier-accepted trajectory as privileged information and, at every token position of that trajectory, predicts a full next-token distribution with this hindsight. Environment verification acts as rejection sampling, providing self-distillation positive guidance over the full horizon. Distilling it into persistent LoRA fast weights absorbs selected experience without treating every generated token as a direct target [16, 2, 60]. ASCENT thus distills hindsight from verified experience into the LLM weights, using this stable version of the LLM as the teacher for stable training. Beyond raw verified experiences, ASCENT can further refine accepted trajectories by removing invalid-action _turns_ , while retaining the reasoning-action content of valid turns as privileged information. Filtering changes only the teacher input. The agent is still trained at removed-turn positions. The filtered trajectories provide the agent with more effective and efficient execution paths. The execution of the next task is based on the updated agent, enabling verification-gated online self-evolution. Across tasks and model scales, ASCENT improves task success and interaction efficiency as experience accumulates, while transferring learned weights to held-out scenes in the stream (Fig. 2). Our contributions are as follows:

  * •

We study OaTTT, in which a deployed agent executes each task once in one pass over its task stream, and the executed trajectory with its verification result is the only data for weight updates that persist across tasks. In OaTTT, we identify the instability of directly imitating or reinforcing these trajectories.

  * •

We propose ASCENT, which self-distills verified experience for stable online test-time training. A stable version of the LLM, its frozen initial copy, conditioned on the verifier-accepted trajectory as privileged information, with invalid-action turns further removed, supplies hindsight-informed full-distribution targets that are distilled into persistent LoRA fast weights. We characterize the population target of this distillation and the limits of sparse outcome selection.

  * •

We validated ASCENT across different types of tasks/steams, across ALFWorld, WebShop, and AppWorld on varied model scales. ASCENT successfully improves task success and interaction efficiency with experience, outperforms the compared online adaptation baselines, and transfers to held-out scenes with effective continued test-time training. The influence of prevailed information and distillation divergence are investegated. We also analyzed in-context adaptation failure modes, and present a preliminary in-context and parametric co-evolution study with ASCENT.




## 2 Related Work

Self-evolving agents. In-context methods store reflections, memories, or reusable skills and retrieve them at inference [32, 59, 42, 37, 26, 57, 10, 49, 58], so reuse depends on retrieval and on the frozen policy executing it over a long interaction. Parametric methods internalize experience in weights [62, 28]. ASCENT consolidates verifier-selected interactions into persistent LoRA fast weights that carry across tasks.

Online learning and test-time training. Online learning updates a model from sequentially arriving data [29]. Test-time training adapts parameters on inference-time signals [35, 41, 44, 13, 18], though Agentic TTT and TT-SI reset parameters after each episode or instance [45, 1]. Existing LLM self-training and RL methods require multiple solutions, correctness-selected generations, or pre-deployment attempts [61, 40, 55, 53, 12, 34, 30, 20, 15]. ASCENT persists updates across tasks from one attempt each, selecting by the environment’s verification outcome and distilling frozen-teacher soft targets.

On-policy distillation and self-distillation. On-policy distillation queries a teacher on student-sampled sequences, reducing train–generation mismatch [2]. OPSD conditions a self-teacher on reference solutions while the student receives only the problem [60]. SDFT conditions the teacher on demonstrations, and SDPO on environment feedback or the model’s own successful attempts [31, 19]. Recent agentic methods use privileged or hindsight-conditioned teacher guidance, mostly before deployment and often combined with RL [43, 23, 48, 46, 21, 52]. CLaaS studies online self-distillation with asynchronous replay [9]. Like OPSD, ASCENT keeps the teacher fixed at the initial model and distills at test time, without replay, immediately after each verifier-accepted interaction of the student.

Figure 1: Imitation and reinforcement lose executable behavior. Online per-episode updates (Sec. 3.2) of Qwen3.5-4B on the ALFWorld seen stream, with std shading: (a) first-response-token entropy, (b) KL to the initial policy, (c) valid-action rate, and (d) generated tokens per turn.

## 3 Method

### 3.1 Online Agentic Test-Time Training (OaTTT)

Self-evolving agent-environment interaction. At deployment, the agent encounters a one-pass task stream 𝒳1:N=(x1,…,xN)\mathcal{X}_{1:N}=(x_{1},\ldots,x_{N}), where each xix_{i} specifies a goal in a partially observed interactive environment ℰ\mathcal{E}. We call the agent’s single attempt at task xix_{i} episode ii. At turn tt of episode ii, the agent conditions on

| Hi,t=(xi,oi,1,wi,1,oi,2,…,wi,t−1,oi,t),H_{i,t}=(x_{i},o_{i,1},w_{i,1},o_{i,2},\ldots,w_{i,t-1},o_{i,t}), |  | (1)  
---|---|---|---  
  
where oi,to_{i,t} is the current observation and wi,tw_{i,t} is the generated reasoning-action response [51]. An action parser ρ\rho extracts the action ai,t=ρ⁡(wi,t)a_{i,t}=\rho(w_{i,t}) from each response, and no action results when parsing fails. Let ℳi\mathcal{M}_{i} denote an optional, evolving cross-task-episode memory, such as reflections, memories, or skills accumulated from earlier episodes [26, 57, 10]. The deployed policy is

| πi(⋅∣Hi,t):=pθi(⋅∣g(Hi,t;ℳi)),wi,t∼πi(⋅∣Hi,t).\pi_{i}(\cdot\mid H_{i,t}):=p_{\theta_{i}}\bigl(\cdot\mid g(H_{i,t};\mathcal{M}_{i})\bigr),\qquad w_{i,t}\sim\pi_{i}(\cdot\mid H_{i,t}). |  | (2)  
---|---|---|---  
  
Here gg renders the instructions, history, and content from ℳi\mathcal{M}_{i}. The environment executes ai,ta_{i,t} and returns oi,t+1∼ℰ(⋅∣Hi,t,ai,t)o_{i,t+1}\sim\mathcal{E}(\cdot\mid H_{i,t},a_{i,t}), and episode ii yields its complete trajectory τi\tau_{i} at termination.

OaTTT as agent evolution. In online evolution, there is no separate task-specific training split or pre-deployment adaptation. We aim to update the policy LLM model from its own interaction experience during deployment. The current policy πi\pi_{i} makes a single attempt at each task xix_{i}, producing one episode:

| τi∼ℙπi,ℰ(⋅∣xi),vi=𝒱(xi,τi)∈{0,1},\tau_{i}\sim\mathbb{P}_{\pi_{i},\mathcal{E}}(\cdot\mid x_{i}),\qquad v_{i}=\mathcal{V}(x_{i},\tau_{i})\in\\{0,1\\}, |  | (3)  
---|---|---|---  
  
where viv_{i} is the sparse, episode-level verification outcome, returned when episode ii ends and recorded before any update. The trajectory τi\tau_{i} and outcome viv_{i} then update the policy:

| (θi+1,ℳi+1)=𝖤𝗏𝗈𝗅𝗏𝖾⁡(θi,ℳi,xi,τi,vi).(\theta_{i+1},\mathcal{M}_{i+1})=\mathsf{Evolve}(\theta_{i},\mathcal{M}_{i};x_{i},\tau_{i},v_{i}). |  | (4)  
---|---|---|---  
  
The update operator 𝖤𝗏𝗈𝗅𝗏𝖾\mathsf{Evolve} is specific to each method. In-context adaptation methods evolve the agent by updating its memory ℳi\mathcal{M}_{i} while keeping the model fixed. Because the policy in Eq. (2) conditions on ℳi\mathcal{M}_{i}, this memory update also changes the policy. Parametric methods evolve θi\theta_{i} [62], joint methods may evolve both, and frozen methods evolve neither. This ordering follows a prequential online protocol [7, 5]: each reported outcome is computed before its own trajectory can influence the policy. In OaTTT, that scored trajectory and its verification result are also the only learning data, with no retries or additional attempts. No replay buffer is used in this paper. Reported success follows each benchmark’s criterion, which can be stricter than the one defining viv_{i}. Success on later tasks thus reflects how much earlier experience helps. Observations describe the resulting states. They give no turn-level reward or causal credit for the verified outcome [36, 56, 38]. We call this protocol the OaTTT protocol, and all online methods, in-context or parametric, follow it. In OaTTT, the LLM’s weights adapt at test time, in one pass over the task stream, from a single multi-turn attempt per task and its sparse episode-level verification, and the updated weights carry across tasks.

### 3.2 OaTTT via Direct Imitation or Reinforcement of Experience

A straightforward way to perform OaTTT is to directly imitate or reinforce the agent’s own interaction trajectories. We formulate five such straightforward baseline methods in two families: direct imitation of the generated tokens (Online ungated imitation, Online RFT, and Online RFT+KL), and single-attempt policy gradient, which reinforces or suppresses them with a signed advantage. Since group-relative advantages [30] need several attempts per task, Online REINFORCE uses REINFORCE [47] with a cross-task baseline, and Online REINFORCE++ adapts REINFORCE++ [17] to normalize advantages over the stream.

Figure 2: Online Agentic Test-Time Training of ASCENT. The student makes a single attempt per task. A verified trajectory becomes privileged information for the frozen teacher, whose soft targets update the LoRA fast weights for the next task. Charts on the right show ALFWorld seen-stream means of 4B and 9B runs (per-scale in Fig. 5(a) and Table 1). Episode time includes the update.

Imitation and reinforcement on the generated trajectories. In a controlled study of OaTTT, we examine the learning dynamics of these five methods along the task stream. For the agent being evolved, every method uses fast-weight LoRA updates [16]: θ0\theta_{0} stays frozen and a LoRA adapter is carried across episodes, so the policy entering episode ii is pθ0,ϕip_{\theta_{0},\phi_{i}}. Let yi,t,1:Li,ty_{i,t,1:L_{i,t}} be the Li,tL_{i,t} tokens of response wi,tw_{i,t} and H^i,t=g⁡(Hi,t,ℳi)\widehat{H}_{i,t}=g(H_{i,t};\mathcal{M}_{i}) the rendered context of the attempt, including any retrieved context. The student prefix ci,t,j=H^i,t⊕yi,t,<jc_{i,t,j}=\widehat{H}_{i,t}\mathbin{\oplus}y_{i,t,<j} is taken from the single attempt at xix_{i} and reused unchanged during the update. Let ℐi\mathcal{I}_{i} index the turns with nonempty responses, and let αi,t,j=(|ℐi|​Li,t)−1\alpha_{i,t,j}=(|\mathcal{I}_{i}|L_{i,t})^{-1} average tokens within each turn and then across turns. Starting from ϕ=ϕi\phi=\phi_{i}, all five methods update the adapter ϕ\phi with a common loss that puts a signed weight ωi,t,j\omega_{i,t,j} on each generated token:

| ℒi(ϕ)=−∑t∈ℐi∑j=1Li,tαi,t,jωi,t,jlogpθ0,ϕ(yi,t,j∣ci,t,j).\mathcal{L}_{i}(\phi)=-\sum_{t\in\mathcal{I}_{i}}\sum_{j=1}^{L_{i,t}}\alpha_{i,t,j}\,\omega_{i,t,j}\log p_{\theta_{0},\phi}\\!\left(y_{i,t,j}\mid c_{i,t,j}\right). |  | (5)  
---|---|---|---  
  
We detail the online per-episode update of each method used in this study:

  1. 1.

Online ungated imitation, as a naive baseline, sets ωi,t,j=1\omega_{i,t,j}=1, so every trajectory, including failed ones, becomes a target.

  2. 2.

Online RFT [53, 30] sets ωi,t,j=vi\omega_{i,t,j}=v_{i}, so only verifier-accepted trajectories become training targets and failed episodes contribute no gradient.

  3. 3.

Online RFT+KL adds a weighted DKL(pθ0(⋅∣ci,t,j)∥pθ0,ϕ(⋅∣ci,t,j))D_{\mathrm{KL}}(p_{\theta_{0}}(\cdot\mid c_{i,t,j})\,\|\,p_{\theta_{0},\phi}(\cdot\mid c_{i,t,j})) to each term of Online RFT, anchoring the policy to the frozen base. The other methods add no KL term to the loss.

  4. 4.

Online REINFORCE sets ωi,t,j=vi−v¯≤i\omega_{i,t,j}=v_{i}-\bar{v}_{\leq i}, with v¯≤i\bar{v}_{\leq i} the running mean verification outcome up to episode ii, so after a first success, failed episodes lower the likelihood of their tokens.

  5. 5.

Online REINFORCE++ sets ωi,t,j\omega_{i,t,j} to a running-normalized token advantage (reward +1+1 for success, −1-1 for failure, minus KL penalties to the base from that token onward) and uses the clipped surrogate min⁡(ρ​ωi,t,j,clip⁡(ρ,1±ϵ)​ωi,t,j)\min(\rho\,\omega_{i,t,j},\operatorname{clip}(\rho,1\pm\epsilon)\,\omega_{i,t,j}), with ρ\rho the ratio to the policy pθ0,ϕip_{\theta_{0},\phi_{i}} that made the attempt.




All five methods learn only from the tokens the agent generated. Their weights make each of these tokens more or less likely. They say nothing about the other tokens at that position. Lowering a token simply moves its probability to tokens the policy already favors, and the KL anchor of Online RFT+KL pulls toward a base model that has not seen the episode. Every update therefore stays centered on the generated tokens, including incidental reasoning, formatting, and detours.

Direct online imitation or reinforcement degrades agent execution. An online update rule can become _unstable_ when, along the stream, its policy drifts ever further from the initial model and degrades while its task success falls below that of the base. On the ALFWorld stream (Fig. 1), Online ungated imitation and Online RFT become nearly deterministic at the first response token while drifting from the initial model. Online RFT+KL drifts more slowly but still substantially (a,b). For the policy-gradient methods, successes soon stop, so nearly every update comes from a failed episode. Online REINFORCE ends collapsed onto first response tokens that the base finds very unlikely, while the first-response-token entropy of Online REINFORCE++ swings above and below the base as it drifts (a,b). All five increasingly produce invalid actions (c), mostly in long responses (d), and over the second half of the stream they succeed less often than the base. These straightforward baselines degrades the agent policy and collapse to lower success rate than the un-trained model.

In OaTTT, we aim to enable single-attempt, per-sample online learning (batch size 1) of LLM weights, analogous to in-context adaptation based agent evolving [26, 51]. In the straightforward online learning baselines, each update relies on a single execution trajectory, so sample-specific characteristics and noise are not summarized or averaged. As subsequent attempts come from the updated policy, with the gradient from direct imitation and reinforcement the generated tokens, the issue can be amplified, causing more overfitting, policy drift, and eventual degradation. ASCENT addresses this by distilling experience with an OPSD objective, using distributions from an anchored self-teacher conditioned on hindsight from verified trajectories as targets.

### 3.3 ASCENT: OaTTT via Self-Distillation of Verified Experience

Overview. ASCENT stably distills verified experience from hindsight-informed predictions of the fixed initial model pθ0p_{\theta_{0}} into LoRA fast weights ϕi\phi_{i} [16], evolving the agent pθ0,ϕip_{\theta_{0},\phi_{i}} during OaTTT. In episode ii, the agent executes xix_{i} once to produce trajectory τi\tau_{i} and verification viv_{i}. If vi=1v_{i}=1, the current agent is distilled into ϕi+1\phi_{i+1} from the frozen base model conditioned on privileged information ziz_{i} from the filtered trajectory. Otherwise, the update is skipped. The updated agent produces better trajectories enabling online self-evolution along the stream (Fig. 2).

#### 3.3.1 On-Policy Self-Distillation Objective

On-policy distillation trains on student-generated prefixes using a teacher’s next-token distributions [2]. OPSD conditions a self-teacher on privileged information unavailable to the student [60]. ASCENT applies this asymmetry to each verifier-accepted episode, with the frozen base model pθ0p_{\theta_{0}} as the teacher.

The verification outcome viv_{i} selects trajectories for adaptation. It assigns no credit to individual turns. Updating only on accepted episodes (vi=1v_{i}=1) acts as rejection sampling [53, 30], so the teacher’s hindsight always comes from a trajectory that passed verification. With binary rewards, maximizing successful-trajectory likelihood gives an unbiased policy-gradient estimate. ASCENT thus learns only from verified successes, requiring no value model or critic under sparse rewards. From the trajectory, without an external reference solution, we construct the privileged information ziz_{i} (Sec. 3.3.2) and obtain the following next-token distributions at each shared student prefix ci,t,jc_{i,t,j}:

| qi,t,j​(⋅)\displaystyle q_{i,t,j}(\cdot) | =stopgrad[pθ0(⋅∣zi⊕ci,t,j)],\displaystyle=\operatorname{stopgrad}\\!\left[p_{\theta_{0}}\\!\left(\cdot\mid z_{i}\mathbin{\oplus}c_{i,t,j}\right)\right], |  | (6)  
---|---|---|---|---  
| pi,t,jϕ​(⋅)\displaystyle p_{i,t,j}^{\phi}(\cdot) | =pθ0,ϕ(⋅∣ci,t,j).\displaystyle=p_{\theta_{0},\phi}\\!\left(\cdot\mid c_{i,t,j}\right). |  | (7)  
  
The turn set ℐi\mathcal{I}_{i}, token lengths Li,tL_{i,t}, and weights αi,t,j\alpha_{i,t,j} are those of Eq. (5), so ASCENT trains at the same student prefixes as direct imitation but replaces its target. Starting from ϕ=ϕi\phi=\phi_{i}, ASCENT minimizes the forward KL to the detached teacher distributions:

| ℒiASCENT​(ϕ)=∑t∈ℐi∑j=1Li,tαi,t,j​DKL​(qi,t,j∥pi,t,jϕ).\mathcal{L}_{i}^{\mathrm{ASCENT}}(\phi)=\sum_{t\in\mathcal{I}_{i}}\sum_{j=1}^{L_{i,t}}\alpha_{i,t,j}D_{\mathrm{KL}}\\!\left(q_{i,t,j}\,\middle\|\,p_{i,t,j}^{\phi}\right). |  | (8)  
---|---|---|---  
  
The fast weights entering episode i+1i+1 are

| ϕi+1={𝖤𝗏𝗈𝗅𝗏𝖾⁡(ϕi,ℒiASCENT),vi=1,ϕi,vi=0,\phi_{i+1}=\begin{cases}\mathsf{Evolve}(\phi_{i};\mathcal{L}_{i}^{\mathrm{ASCENT}}),&v_{i}=1,\\\ \phi_{i},&v_{i}=0,\end{cases} |  | (9)  
---|---|---|---  
  
where 𝖤𝗏𝗈𝗅𝗏𝖾\mathsf{Evolve} is the operator from Eq. (4), here a few optimizer steps on Eq. (8) (Appendix 7.3). The teacher stays fixed at pθ0p_{\theta_{0}}, so self-distillation injects each verified trajectory’s experience into the student’s fast weights through a stable version of the LLM. Student updates never change the teacher’s targets for a given trajectory, which helps mitigate student drift during online learning. Because ziz_{i} covers the full horizon of the verified trajectory, the teacher sees at each turn how the episode continued to verified success, giving the student better-informed turn-level guidance. We next describe how the verified trajectory supplies this hindsight and the student prefixes.

#### 3.3.2 Distilling Verified and Filtered Experience Trajectories

Hindsight from the complete trajectory. Episode ii, with TiT_{i} turns, yields

| τi=(xi,oi,1,(wi,t,oi,t+1)t=1Ti).\tau_{i}=\bigl(x_{i},o_{i,1},(w_{i,t},o_{i,t+1})_{t=1}^{T_{i}}\bigr). |  | (10)  
---|---|---|---  
  
It contains the task, each generated response, and the observation returned after its action. At a student prefix ci,t,jc_{i,t,j}, the student had not yet seen the rest of wi,tw_{i,t}, the later turns and observations, or the verification outcome. We call this later part, together with its verified outcome, _hindsight_ : it shows how the episode continued from each earlier decision to verified success [14, 38]. Although it does not identify which actions caused success, it provides the student with the successful experience as privileged information via the self-teacher.

Action-validity filter on the privileged information. Even an accepted trajectory can contain turns whose response yields no parseable action or whose action the environment does not execute, such as an unavailable command or code that raises an error. Invalid-action signals arise naturally during execution, requiring neither additional cost nor credit assignment. ASCENT further marks them with a validity indicator read from the environment’s response at each turn:

| di,t={1,if the environment executes the action ρ⁡(wi,t),0,otherwise, including when no action is parsed.d_{i,t}=\begin{cases}1,&\text{if the environment executes the action $\rho(w_{i,t})$},\\\ 0,&\text{otherwise, including when no action is parsed}.\end{cases} |  | (11)  
---|---|---|---  
  
The filter records only whether an action was executed. It does not judge whether a valid action helps the task, gives no reward, and assigns no credit.

The teacher’s privileged information is then

| zi=hASCENT(xi,(wi,t)t:di,t=1),z_{i}=h_{\mathrm{ASCENT}}\\!\left(x_{i},\,(w_{i,t})_{t:\,d_{i,t}=1}\right), |  | (12)  
---|---|---|---  
  
where the fixed map hASCENTh_{\mathrm{ASCENT}} serializes the task goal, a success statement, and the reasoning–action responses of valid turns, with actions parsed by ρ\rho. By default, ziz_{i} omits observation text. The verified trajectory thus plays two roles. Its valid turns build ziz_{i}, while all its nonempty responses, including those of invalid-action turns, supply the student prefixes ci,t,jc_{i,t,j}. Sec. 4.4 ablates the content of ziz_{i} and the filter.

ASCENT distills experience online stably through full-vocabulary distribution matching. In comparison to the baseline methods of Sec. 3.2, ASCENT updates the same student distribution pϕp^{\phi} at the same student prefixes, with a different target. With position indices suppressed, for the generated token yy (one-hot vector 𝐞y\mathbf{e}_{y}) and student logits ℓϕ\bm{\ell}^{\phi}, a term of Eq. (5) (of the baseline) and the forward KL of Eq. (8) (of ASCENT) have gradients

| ∇ℓϕ[−ωlogpyϕ]=ω(pϕ−𝐞y),and∇ℓϕDKL(q∥pϕ)=pϕ−q,respectively.\nabla_{\bm{\ell}^{\phi}}\bigl[-\omega\log p^{\phi}_{y}\bigr]=\omega\,\bigl(p^{\phi}-\mathbf{e}_{y}\bigr),\penalty\ \penalty\ \penalty\ \penalty\ \penalty\ \text{and}\penalty\ \penalty\ \penalty\ \penalty\ \penalty\ \nabla_{\bm{\ell}^{\phi}}D_{\mathrm{KL}}\bigl(q\,\|\,p^{\phi}\bigr)=p^{\phi}-q,\penalty\ \penalty\ \penalty\ \text{respectively}. |  | (13)  
---|---|---|---  
  
Minimizing the first term pushes yy’s logit by ω⁡(1−pyϕ)\omega(1-p^{\phi}_{y}) and every other logit kk by −ω​pkϕ-\omega p^{\phi}_{k}, in proportion to its current probability. The methods of Sec. 3.2 thus perform hard-target matching on the generated token, where a positive weight (direct imitation, and policy gradient on successful episodes) makes the student more confident in yy, a negative weight (policy gradient, mainly on failed episodes) hands yy’s probability to tokens the student already favors, and both tend to sharpen the distribution around the student’s own outputs. Minimizing the second in ASCENT pushes each logit kk by qk−pkϕq_{k}-p^{\phi}_{k} and stops only when pϕ=qp^{\phi}=q. ASCENT thus performs full-vocabulary distribution matching, so alternatives the hindsight teacher prefers rise, and yy falls when the teacher disagrees, for example at an invalid-action turn omitted from ziz_{i}. As a stable LLM copy (i.e., the initial checkpoint), the teacher does not track the student, preventing the target from sharpening with it. While a stable yet refreshed teacher could better follow the evolving policy with a more precisely balanced trade-off, we find the frozen initial model (the default in [60]), simple and effective for online streaming adaptation (Appendix 8.1). In Fig. 1, ASCENT stays closest to the initial model while keeping its valid-action rate, with a relatively more stable dynamic behavior. Under a fixed collection distribution, the population target averages the teacher distributions of accepted trajectories through each prefix (Appendix 6.1.2).

Potential turn efficiency in long-horizon tasks. Under a turn budget, an unnecessary turn uses up part of the budget, so a response that starts a detour can be less likely to reach verified success. Verifier selection gives such responses less weight (Appendix 6.1.1), and the hindsight teacher, which sees how the accepted episode reached success, can potentially lower them at earlier student prefixes. We empirically show that ASCENT has the fewest mean turns in Tables 1, 2, 4, and 5, and fewer turns than the base along the stream (Fig. 5(a)), including on tasks that both the un-evolved base model and ASCENT solve (Sec. 4.3). Without turn-level credit, an accepted trajectory can still contain unnecessary turns, and one attempt cannot show what an alternative action would have achieved (Appendix 6.1.3). We leave a theoretical understanding and complete proof of this mechanism for future work.

## 4 Experiments

Method | Pick | Look | Clean | Heat | Cool | Pick2 | Avg. SR | Turns |  Time GPU·h / s  
---|---|---|---|---|---|---|---|---|---  
Qwen3.5-4B  
Base | 91.4 | 46.2 | 40.7 | 6.2 | 8.0 | 54.2 | 46.4 | 35.1 | 97  
Offline  
MemP | 94.3 | 53.8 | 77.8 | 0.0 | 36.0 | 66.7 | 61.4±\pm6.5 | 28.6 | 5.2 / 61  
ACE | 85.7 | 46.2 | 70.4 | 6.2 | 60.0 | 58.3 | 60.7±\pm9.2 | 31.1 | 6.3 / 81  
EvoSkill | 93.3 | 17.9 | 35.8 | 6.2 | 32.0 | 66.7 | 49.8±\pm4.9 | 34.4 | 5.9 / 123  
GEPA | 100.0 | 23.1 | 22.2 | 37.5 | 64.0 | 66.7 | 58.6±\pm3.4 | 34.5 | 6.3 / 91  
TextGrad | 88.6 | 46.2 | 22.2 | 6.2 | 12.0 | 50.0 | 42.1±\pm4.9 | 37.5 | 8.3 / 104  
Trace2Skill | 94.3 | 69.2 | 29.6 | 0.0 | 16.0 | 75.0 | 51.4±\pm6.1 | 33.0 | 5.2 / 92  
Online  
MemP | 94.3 | 61.5 | 71.6 | 12.5 | 36.0 | 73.6 | 63.6±\pm5.9 | 29.7 | 58  
ReasoningBank | 94.3 | 30.8 | 33.3 | 14.6 | 22.7 | 61.1 | 49.0±\pm1.5 | 35.6 | 89  
ACE | 90.5 | 17.9 | 49.4 | 22.9 | 42.7 | 41.7 | 51.2±\pm13.8 | 34.6 | 92  
A-Mem | 88.6 | 30.8 | 35.8 | 10.4 | 20.0 | 58.3 | 46.7±\pm5.4 | 34.5 | 79  
MemRL | 85.7 | 23.1 | 33.3 | 12.5 | 16.0 | 62.5 | 45.0±\pm5.2 | 35.4 | 89  
Dynamic Cheatsheet | 91.4 | 23.1 | 44.4 | 6.2 | 28.0 | 54.2 | 48.6±\pm4.1 | 36.9 | 94  
TT-OPSD | 95.2 | 46.2 | 29.6 | 2.1 | 25.3 | 72.2 | 51.0±\pm3.7 | 32.9 | 87  
ASCENT | 99.0 | 69.2 | 69.1 | 20.8 | 54.7 | 75.0 | 69.5±\pm5.6 | 26.3 | 115  
Qwen3.5-9B  
Base | 94.3 | 61.5 | 44.4 | 0.0 | 8.0 | 91.7 | 55.0 | 32.6 | 115  
Offline  
MemP | 100.0 | 92.3 | 74.1 | 18.8 | 28.0 | 83.3 | 69.3±\pm2.4 | 26.3 | 5.9 / 83  
ACE | 94.3 | 53.8 | 51.9 | 37.5 | 28.0 | 79.2 | 61.4±\pm5.5 | 27.9 | 7.1 / 129  
EvoSkill | 97.1 | 71.8 | 54.3 | 18.8 | 37.3 | 90.3 | 65.7±\pm3.5 | 29.0 | 4.7 / 145  
GEPA | 100.0 | 76.9 | 66.7 | 25.0 | 10.0 | 91.7 | 65.4±\pm4.6 | 30.1 | 6.7 / 115  
TextGrad | 91.4 | 61.5 | 59.3 | 18.8 | 44.0 | 54.2 | 59.3±\pm4.5 | 32.6 | 7.0 / 147  
Trace2Skill | 94.3 | 84.6 | 44.4 | 31.2 | 60.0 | 79.2 | 67.9±\pm6.4 | 28.4 | 6.4 / 132  
Online  
MemP | 95.2 | 74.4 | 75.3 | 16.7 | 38.7 | 80.6 | 67.9±\pm2.9 | 27.4 | 85  
ReasoningBank | 95.2 | 59.0 | 75.3 | 22.9 | 36.0 | 76.4 | 66.0±\pm2.4 | 29.0 | 93  
ACE | 93.3 | 51.3 | 76.5 | 29.2 | 57.3 | 75.0 | 69.3±\pm6.3 | 28.3 | 113  
A-Mem | 95.7 | 69.2 | 53.7 | 9.4 | 22.0 | 85.4 | 60.4±\pm0.4 | 30.7 | 101  
MemRL | 97.1 | 53.8 | 66.7 | 12.5 | 20.0 | 95.8 | 63.6±\pm2.5 | 29.5 | 110  
Dynamic Cheatsheet | 97.1 | 61.5 | 48.1 | 0.0 | 20.0 | 83.3 | 57.1±\pm4.8 | 32.2 | 126  
TT-OPSD | 98.1 | 79.5 | 54.3 | 12.5 | 5.3 | 95.8 | 61.2±\pm1.8 | 29.3 | 99  
ASCENT | 100.0 | 89.7 | 76.5 | 37.5 | 48.0 | 95.8 | 77.4±\pm1.8 | 21.0 | 133  
Table 1: ALFWorld seen-stream results. Success rate (SR, %) by task type and over the full stream, mean turns per episode, and runtime, given as training GPU-hours / evaluation seconds per episode for offline methods and seconds per episode otherwise. Bold indicates the highest SR and fewest mean turns within each backbone.

### 4.1 Experimental Setting

Benchmarks. We evaluate on ALFWorld [33], a text-based household environment, and WebShop [50], a simulated web store in which the agent searches for and buys a product matching an instruction. The ALFWorld seen split (140 tasks across six task families) is the main stream, and the unseen split (134 tasks in held-out rooms and layouts) tests frozen transfer and continued OaTTT. AppWorld [39], which tests interactive coding across simulated apps, is reported in Appendix 7.5. More details in Appendix 7.

OaTTT protocol. All online methods follow Section 3.1 on the same task order: one pass over the stream, a single scored attempt per task (50-turn budget on ALFWorld and WebShop), and each task scored before its trajectory can update the method. They receive the same environment verification signal. Offline baselines instead build memories, skills, or prompts from the same fixed-size training split before evaluation. They fall outside this setting and serve as reference points.

Baselines. We compare ASCENT with the base model, offline and online self-evolution methods (Appendix 7.2), and TT-OPSD, our test-time (TT) adaptation of OPSD [60], which was designed for static, single-turn reasoning. Its teacher receives only the current turn’s response and returned observation as privileged information. All other configurations are the same as ASCENT.

Evaluation metrics. We report ALFWorld exact success overall and by task family, WebShop exact success (task score =1=1) and mean score, and mean per-episode turns and wall-clock time. On WebShop, ASCENT’s adaptation gate accepts a score of at least 0.90.9. All results are reported as mean±\pmstd over 3 independent runs.

Implementation details. We use Qwen3.5-4B and Qwen3.5-9B [27]. The base, all compared methods, and ASCENT run with the same ReAct template [51] and greedy decoding. More details in Appendix 7.3.

| Qwen3.5-4B | Qwen3.5-9B  
---|---|---  
Method | Strict-SR | Mean-Score | Turns |  Time GPU·h / s | Strict-SR | Mean-Score | Turns |  Time GPU·h / s  
Base | 17.8 | 25.9 | 39.5 | 87 | 17.2 | 27.8 | 37.5 | 166  
Offline  
MemP (offline) | 33.2±\pm1.3 | 43.7±\pm2.0 | 30.0 | 6.9 / 74 | 26.6±\pm1.3 | 40.6±\pm2.8 | 31.9 | 12.3 / 148  
ACE (offline) | 5.8±\pm0.5 | 21.4±\pm1.1 | 42.0 | 9.4 / 122 | 13.2±\pm0.6 | 22.4±\pm1.4 | 41.6 | 16.4 / 229  
EvoSkill | 20.0±\pm1.6 | 29.3±\pm1.9 | 38.0 | 2.06 / 84 | 20.2±\pm2.1 | 33.4±\pm1.4 | 38.2 | 2.33 / 169  
GEPA | 23.4±\pm1.2 | 37.5±\pm2.4 | 34.0 | 8.4 / 75 | 35.4±\pm3.2 | 48.6±\pm1.6 | 30.4 | 10.1 / 135  
TextGrad | 10.4±\pm2.1 | 30.5±\pm2.5 | 35.9 | 13.3 / 79 | 18.4±\pm2.5 | 34.7±\pm2.8 | 34.5 | 14.7 / 153  
Trace2Skill | 14.2±\pm0.6 | 21.7±\pm1.2 | 41.6 | 7.7 / 92 | 10.0±\pm4.2 | 21.0±\pm3.6 | 40.4 | 10.2 / 179  
Online adaptation  
MemP | 36.8±\pm3.4 | 47.7±\pm4.8 | 28.7 | 74 | 22.4±\pm2.5 | 38.9±\pm4.1 | 32.9 | 152  
ReasoningBank | 10.6±\pm0.7 | 21.6±\pm1.0 | 42.1 | 176 | 12.6±\pm2.1 | 21.8±\pm1.5 | 41.5 | 226  
ACE | 10.6±\pm3.2 | 23.6±\pm4.3 | 40.6 | 107 | 16.0±\pm1.4 | 32.0±\pm2.6 | 36.1 | 144  
A-Mem | 23.2±\pm3.1 | 36.4±\pm2.9 | 33.9 | 98 | 15.4±\pm0.6 | 32.5±\pm4.3 | 35.6 | 176  
MemRL | 21.8±\pm1.2 | 32.3±\pm2.5 | 37.1 | 106 | 14.0±\pm4.2 | 27.4±\pm1.7 | 38.9 | 166  
Dynamic Cheatsheet | 16.2±\pm1.5 | 28.4±\pm2.1 | 39.4 | 133 | 13.4±\pm2.9 | 27.4±\pm3.0 | 39.1 | 211  
TT-OPSD | 25.6±\pm0.5 | 37.1±\pm1.7 | 34.6 | 100 | 24.2±\pm2.4 | 35.9±\pm2.3 | 34.8 | 108  
ASCENT | 41.9±\pm0.3 | 62.4±\pm1.0 | 17.8 | 76 | 41.5±\pm1.8 | 56.8±\pm1.7 | 26.7 | 164  
Table 2: WebShop results. Strict-SR (%) denotes task_score=1=1, and Mean-Score (%) averages task_score over each 500-task stream. Offline runtime is training GPU-hours / evaluation seconds per episode, and other entries are seconds per episode. Bold indicates the highest Strict-SR and Mean-Score and the fewest turns within each backbone.

(a) Qwen3.5-4B

(b) Qwen3.5-9B

Figure 3: Cross-scene transfer and continued OaTTT on ALFWorld unseen. Transfer freezes the seen-stream adaptation state, whereas continued OaTTT resumes online updates. More in Tables 4–5.

### 4.2 Main Experimental Results

ASCENT consistently improves long-horizon task success. As shown in Tables 1 and 2, ASCENT improves exact success over the base by more than 2222 points across ALFWorld and WebShop at both scales and exceeds the strongest online comparator. It also achieves the highest WebShop mean task score and outperforms every offline in-context adaptation method on ALFWorld. ASCENT outperforms TT-OPSD (our test-time adaptation of OPSD [60]), which uses the same self-distillation but whose teacher sees only the current turn, showing the benefit of hindsight from the complete verified trajectory. These consistent gains show that cross-episode parameter updates can extract reusable behavior from the same sparse environment verification available to offline and online in-context adaptation methods. As discussed in Sec. 3.2, we omit the direct imitation and policy-gradient baselines from these tables because they severely degrade under OaTTT (Fig. 1).

Cross-episode gains extend to difficult tasks. Table 1 shows that the base rarely solves Heat or Cool, whereas ASCENT improves both task families at each scale. Fig. 5(a) shows that ASCENT’s gain over the base is much larger on later 4B tasks, while the base remains flat. ASCENT derives this improvement from its own verified deployment interactions, without a stronger teacher or a pre-deployment adaptation stage.

![Refer to caption](2610.05303v1/g135_case_study_v8_arxiv.png) Figure 4: Quantitative analysis on Alfworld. Columns show the base, two representative in-context adaptation failures (irrelevant or contradictory retrieved memory, or relevant guidance left unfollowed), and ASCENT, which carries LoRA updates from two earlier cool-then-place tasks and retrieves no memory. Numbers are turn indices. Checks mark valid actions, crosses mark invalid or non-action outputs, and ellipses omit turns.

(a) Seen-stream diagnostics

(b) Co-evolution

(c) Distillation divergence

Figure 5: Analysis of online gains, preliminary co-evolution, and divergence ablation. (a) Mean success-rate gain and turns saved by ASCENT over the base, by task range and over all tasks (lighter bars 4B, darker 9B, with turns on tasks solved by both the un-evolved base model and ASCENT in Table 8). (b) Preliminary in-context and parametric co-evolution. (c) Distillation-divergence ablation, pre-update SR (%) at both scales, with forward KL as default.

Fewer turns and modest online cost. Tables 1 and 2 show that ASCENT reduces capped all-episode mean turns at both scales on both benchmarks. On tasks that both the un-evolved base model and ASCENT solve, ALFWorld trajectories are also shorter (Sec. 4.3). Episode time includes the attempt and LoRA update after verified successes. It is modestly higher than the base on ALFWorld and lower on WebShop.

The evolved policy transfers and continues learning under scene shift. As shown in Fig. 3 and Tables 4–5, ASCENT leads the online comparators in both frozen transfer and continued adaptation at both scales. At 4B, continued OaTTT substantially exceeds frozen transfer. At 9B, continuation can preserve the already strong transfer result. These results show that seen-stream experience is retained in the evolved policy and that the self-evolution protocol remains effective.

### 4.3 Analysis and Discussion

|  | Qwen3.5-4B | Qwen3.5-9B  
---|---|---|---  
Privileged information ziz_{i} | Filter | SR (%) | Turns | Valid (%) | SR (%) | Turns | Valid (%)  
Base | – | 46.446.4 | 35.135.1 | 70.170.1 | 55.055.0 | 32.632.6 | 78.778.7  
| ✗ | 48.848.8 | 32.932.9 | 64.464.4 | 42.942.9 | 34.134.1 | 66.266.2  
Action only: ai,ta_{i,t} | ✓ | 48.348.3 | 33.533.5 | 61.261.2 | 48.648.6 | 32.332.3 | 71.471.4  
| ✗ | 62.962.9 | 28.028.0 | 74.674.6 | 77.9\mathbf{77.9} | 21.121.1 | 86.686.6  
Reasoning ++ action ++ observation: (wi,t,oi,t+1)(w_{i,t},o_{i,t+1}) | ✓ | 64.564.5 | 28.328.3 | 79.5\mathbf{79.5} | 77.177.1 | 23.823.8 | 88.2\mathbf{88.2}  
| ✗ | 60.060.0 | 30.630.6 | 74.174.1 | 75.375.3 | 21.621.6 | 81.881.8  
Reasoning ++ action: wi,tw_{i,t} | ✓ | 69.5\mathbf{69.5} | 26.3\mathbf{26.3} | 78.078.0 | 77.477.4 | 21.0\mathbf{21.0} | 85.985.9  
Table 3: Privileged-information ablation on ALFWorld seen, with the action-validity filter off (✗) or on (✓). The shaded row is ASCENT’s default (reasoning ++ action with the filter).

Case study on long-horizon task completion. As shown in Fig. 4, the base and five in-context adaptation variants exhaust the turn budget on the task “put a cool egg in microwave,” whereas ASCENT completes it in 20 turns. The in-context variants fail either by retrieving irrelevant or contradictory experience or by not following relevant guidance. ASCENT instead internalizes earlier same-family experience in its weights and transfers it across objects and destinations. Across whole streams, retrieving available same-type prior experience does not reliably lead to task completion. Appendix 8.3 gives this retrieval analysis and the provenance of each trace.

Interaction efficiency. In Fig. 5(a), ASCENT’s turn savings over the base grow later in the stream. They also persist on _shared successes_ , the tasks that both the un-evolved base model and ASCENT solve: ASCENT uses fewer turns than the base on 64%64\% (4B) and 67%67\% (9B) of them (Appendix 8.2, Table 8).

Preliminary study on in-context and parametric co-evolution. ASCENT stores experience in fast weights, whereas in-context adaptation stores it in the context. In Fig. 5(b), we combine ASCENT with ACE [57] and MemP [10] on ALFWorld seen. We find that the two are complementary: adding ASCENT improves both in-context methods at both scales. Whether retrieved memory also helps on top of ASCENT depends on the model. On the smaller 4B model, ASCENT alone performs best, so the extra retrieved context may distract a weaker model, in line with its preference for shorter privileged information in Table 3. On the relatively stronger 9B model, both combinations exceed ASCENT alone, maybe because the model can use the retrieved memory and its evolved weights together. Co-evolving context and weights is thus promising for stronger agents.

### 4.4 Ablation Studies

Ablation on privileged information. In Table 3, we study the effect of each component of the privileged information ziz_{i} (actions ai,ta_{i,t}, reasoning–action responses wi,tw_{i,t}, and observations oi,t+1o_{i,t+1}), with and without our proposed action-validity filter (Sec. 3.3.2). We find that generally reasoning is the key component: action-only ziz_{i} (ai,ta_{i,t}) performs near or below the base, while adding reasoning (wi,tw_{i,t}) consistently improves success and turn efficiency. This may be because reasoning explains why each action was taken, such as which subgoal it serves. It also makes up most of the tokens the student generates, while an action is only a short command, so action-only hindsight informs few of the positions the student is trained on.

Smaller models benefit more from shorter, filtered privileged information. On the smaller 4B model, shorter (wi,tw_{i,t}, without observations) and carefully handled (filtered) privileged information performs better, with higher turn efficiency (Table 3). With reasoning in ziz_{i}, the filter helps most at this scale. On the relatively stronger 9B model, providing the full trajectory (wi,tw_{i,t}, oi,t+1o_{i,t+1}) performs well, maybe due to the model’s stronger general language capability. Here the filter changes success slightly while still raising the valid-action rate, possibly because the stronger model is less affected by invalid-action turns. Without the filter, observations may help by showing which actions the environment flagged as invalid, so they add little once the filter removes those turns. Our default setting (wi,tw_{i,t} with the filter) on 9B also performs comparably, with slightly better turn efficiency.

Ablation on distillation divergence. As shown in Fig. 5(c), ASCENT still improves on the base at both scales when reverse KL or Jensen–Shannon (JS) divergence replaces forward KL in Eq. (8). Reverse KL trails forward KL at both scales, most at 9B. JS matches reverse KL at 4B but exceeds forward KL at 9B. Forward KL favors coverage of the teacher distribution, whereas reverse KL is mode-seeking [2].

## 5 Conclusion

In our Online Agentic TTT study, directly imitating or reinforcing generated tokens destabilizes the policy. ASCENT instead self-distills verified experience into persistent LoRA fast weights through a frozen teacher. On ALFWorld and WebShop at two model scales, it outperforms the base and the compared baselines, with gains that grow later in the stream and transfer to held-out scenes.

Limitations and future work. ASCENT needs full next-token distributions from its frozen copy, so it applies only to open-weight models. It gates updates on sparse episode-level verification and weights every turn equally. Future work could approximate finer-grained, turn-level credit. Because the teacher’s privileged context contains the student’s response text, it may partially reconstruct generated tokens, which would limit the independence of its hindsight. Isolating this effect is an open question. The online evolution also shows potential for environment- or user-specific agents by plugging in the matching evolved fast weights.

## References

  * [1] E. C. Acikgoz, C. Qian, H. Ji, D. Hakkani-Tür, and G. Tur (2026) TT-SI: Self-Improving LLM Agents with Test-Time Training.  In Findings of the Association for Computational Linguistics: ACL 2026,  San Diego, California, United States, pp. 9483–9508.  External Links: [Document](https://dx.doi.org/10.18653/v1/2026.findings-acl.462), [Link](https://aclanthology.org/2026.findings-acl.462/), 2510.07841 Cited by: §2. 
  * [2] R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. Ramos Garea, M. Geist, and O. Bachem (2024) On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes.  In International Conference on Learning Representations,  pp. 21246–21263.  External Links: [Link](https://openreview.net/forum?id=3zKtaqxLhW) Cited by: §1, §2, §3.3.1, §4.4. 
  * [3] L. A. Agrawal, S. Tan, D. Soylu, N. Ziems, R. Khare, K. Opsahl-Ong, A. Singhvi, H. Shandilya, M. J. Ryan, M. Jiang, C. Potts, K. Sen, A. G. Dimakis, I. Stoica, D. Klein, M. Zaharia, and O. Khattab (2026) GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning.  In International Conference on Learning Representations,  pp. 8479–8565.  External Links: [Link](https://proceedings.iclr.cc/paper_files/paper/2026/file/0e9e708b6f48e14fd0ac29e167413f76-Paper-Conference.pdf) Cited by: §7.2. 
  * [4] S. Alzubi, N. Provenzano, J. Bingham, W. Chen, and T. Vu (2026) EvoSkill: Automated Skill Discovery for Multi-Agent Systems.  arXiv preprint arXiv:2603.02766.  Cited by: §7.2. 
  * [5] A. Blum, A. Kalai, and J. Langford (1999) Beating the Hold-Out: Bounds for K-Fold and Progressive Cross-Validation.  In Proceedings of the Twelfth Annual Conference on Computational Learning Theory,  pp. 203–208.  External Links: [Document](https://dx.doi.org/10.1145/307400.307439) Cited by: §3.1. 
  * [6] D. Brandfonbrener, A. Bietti, J. Buckman, R. Laroche, and J. Bruna (2022) When Does Return-Conditioned Supervised Learning Work for Offline Reinforcement Learning?.  In Advances in Neural Information Processing Systems,  Vol. 35, pp. 1542–1553.  External Links: [Document](https://dx.doi.org/10.52202/068431-0113) Cited by: §6.1.1. 
  * [7] A. P. Dawid (1984) Present Position and Potential Developments: Some Personal Views: Statistical Theory: The Prequential Approach.  Journal of the Royal Statistical Society: Series A (General) 147 (2), pp. 278–290.  External Links: [Document](https://dx.doi.org/10.2307/2981683) Cited by: §3.1. 
  * [8] B. Eysenbach, S. Udatha, R. Salakhutdinov, and S. Levine (2022) Imitating Past Successes Can Be Very Suboptimal.  In Advances in Neural Information Processing Systems,  Vol. 35, pp. 6047–6059.  External Links: [Document](https://dx.doi.org/10.52202/068431-0438) Cited by: §6.1.1. 
  * [9] K. Fallah, S. Naihin, B. Widawsky, and Q. Mao (2026) CLaaS: Continual learning as a service for sample efficient online learning.  arXiv preprint arXiv:2606.05559.  Cited by: §2. 
  * [10] R. Fang, Y. Liang, X. Wang, J. Wu, S. Qiao, P. Xie, F. Huang, H. Chen, and N. Zhang (2026) Memp: Exploring Agent Procedural Memory.  In Findings of the Association for Computational Linguistics: ACL 2026,  San Diego, California, United States, pp. 17490–17502.  External Links: [Document](https://dx.doi.org/10.18653/v1/2026.findings-acl.866), [Link](https://aclanthology.org/2026.findings-acl.866/) Cited by: §1, §2, §3.1, §4.3, §7.2, §7.2. 
  * [11] L. Feng, Z. Xue, T. Liu, and B. An (2025) Group-in-Group Policy Optimization for LLM Agent Training.  In Advances in Neural Information Processing Systems,  Vol. 38, pp. 46375–46408.  External Links: [Document](https://dx.doi.org/10.52202/085713-1544), [Link](https://proceedings.neurips.cc/paper_files/paper/2025/file/420c9f777c0b4f78d515e53cf74d58b2-Paper-Conference.pdf) Cited by: §1. 
  * [12] C. Gulcehre, T. L. Paine, S. Srinivasan, K. Konyushkova, L. Weerts, A. Sharma, A. Siddhant, A. Ahern, M. Wang, C. Gu, W. Macherey, A. Doucet, O. Firat, and N. de Freitas (2023) Reinforced Self-Training (ReST) for Language Modeling.  arXiv preprint arXiv:2308.08998.  Cited by: §2. 
  * [13] M. Hardt and Y. Sun (2024) Test-Time Training on Nearest Neighbors for Large Language Models.  In International Conference on Learning Representations,  Vol. 2024, pp. 54625–54640.  External Links: [Link](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f02f1185b97518ab5bd7ebde466992d3-Abstract-Conference.html) Cited by: §2. 
  * [14] A. Harutyunyan, W. Dabney, T. Mesnard, M. Gheshlaghi Azar, B. Piot, N. Heess, H. van Hasselt, G. Wayne, S. Singh, D. Precup, and R. Munos (2019) Hindsight Credit Assignment.  In Advances in Neural Information Processing Systems,  Vol. 32.  External Links: [Link](https://proceedings.neurips.cc/paper_files/paper/2019/file/195f15384c2a79cedf293e4a847ce85c-Paper.pdf) Cited by: §3.3.2. 
  * [15] Z. Hou, Y. Li, J. Tang, and Y. Dong (2026) Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning.  arXiv preprint arXiv:2607.07508.  Cited by: §1, §2. 
  * [16] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen (2022) LoRA: Low-Rank Adaptation of Large Language Models.  In International Conference on Learning Representations,  External Links: [Link](https://openreview.net/forum?id=nZeVKeeFYf9) Cited by: §1, §3.2, §3.3. 
  * [17] J. Hu, J. K. Liu, H. Xu, and W. Shen (2025) REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization.  arXiv preprint arXiv:2501.03262.  Cited by: §3.2, §7.2. 
  * [18] J. Hu, Z. Zhang, G. Chen, X. Wen, C. Shuai, W. Luo, B. Xiao, Y. Li, and M. Tan (2025) Test-Time Learning for Large Language Models.  In Proceedings of the 42nd International Conference on Machine Learning,  Proceedings of Machine Learning Research, Vol. 267, pp. 24823–24849.  External Links: [Link](https://proceedings.mlr.press/v267/hu25z.html) Cited by: §2. 
  * [19] J. Hübotter, F. Lübeck, L. D. Behric, A. Baumann, M. Bagatella, D. Marta, I. Hakimi, I. Shenfeld, T. Kleine Buening, C. Guestrin, and A. Krause (2026) Reinforcement Learning via Self-Distillation.  In Proceedings of the 43rd International Conference on Machine Learning,  Proceedings of Machine Learning Research, Vol. 306, pp. 48677–48721.  External Links: 2601.20802, [Link](https://proceedings.mlr.press/v306/hubotter26a.html) Cited by: §2. 
  * [20] H. Kim, Y. Lee, G. Lee, C. Finn, and K. Lee (2026) WHALE: A Simple Recipe for Joint Harness-Weight Optimization.  arXiv preprint arXiv:2609.00196.  Cited by: §2. 
  * [21] J. Liu, W. Li, J. Ling, and P. Wang (2026) When Privileged Guidance Misaligns: State-Matched Routing and Contextualized Self-Distillation for Multi-Turn Agents.  arXiv preprint arXiv:2608.05219.  Cited by: §2. 
  * [22] I. Loshchilov and F. Hutter (2019) Decoupled Weight Decay Regularization.  In International Conference on Learning Representations,  External Links: [Link](https://openreview.net/forum?id=Bkg6RiCqY7) Cited by: §7.3. 
  * [23] Z. Lu, Z. Yao, Z. Han, Z. Wang, J. Wu, Q. Gu, X. Cai, W. Lu, J. Xiao, Y. Zhuang, and Y. Shen (2026) Self-Distilled Agentic Reinforcement Learning.  arXiv preprint arXiv:2605.15155.  Cited by: §1, §2. 
  * [24] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis (2015) Human-Level Control through Deep Reinforcement Learning.  Nature 518 (7540), pp. 529–533.  External Links: [Document](https://dx.doi.org/10.1038/nature14236) Cited by: §8.1. 
  * [25] J. Ni, Y. Liu, X. Liu, Y. Sun, M. Zhou, P. Cheng, D. Wang, E. Zhao, X. Jiang, and G. Jiang (2026) Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills.  arXiv preprint arXiv:2603.25158.  Cited by: §7.2. 
  * [26] S. Ouyang, J. Yan, I. Hsu, Y. Chen, K. Jiang, Z. Wang, R. Han, L. Le, S. Daruki, X. Tang, V. Tirumalashetty, G. Lee, M. Rofouei, H. Lin, J. Han, C. Lee, and T. Pfister (2026) ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory.  In International Conference on Learning Representations,  Vol. 2026, pp. 94327–94354.  External Links: [Link](https://proceedings.iclr.cc/paper_files/paper/2026/file/980ea04d23d1f6908964eba2a74afe45-Paper-Conference.pdf) Cited by: §1, §1, §2, §3.1, §3.2, §7.2. 
  * [27] Qwen Team (2026) Qwen3.5: Towards Native Multimodal Agents.  External Links: [Link](https://qwen.ai/blog?id=qwen3.5) Cited by: §4.1. 
  * [28] T. Ren, W. Luo, H. Yang, R. Zhu, X. Huang, Y. Wu, B. Chou, J. Ye, J. Liang, Y. Li, and Y. Peng (2026) Scaling Self-Evolving Agents via Parametric Memory.  arXiv preprint arXiv:2606.04536.  Cited by: §1, §2. 
  * [29] S. Shalev-Shwartz (2012) Online Learning and Online Convex Optimization.  Foundations and Trends in Machine Learning 4 (2), pp. 107–194.  Cited by: §2. 
  * [30] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo (2024) DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.  arXiv preprint arXiv:2402.03300.  Cited by: §1, §2, item 2, §3.2, §3.3.1, §7.2. 
  * [31] I. Shenfeld, M. Damani, J. Hübotter, and P. Agrawal (2026) Self-Distillation Enables Continual Learning.  In Proceedings of the 43rd International Conference on Machine Learning,  Proceedings of Machine Learning Research, Vol. 306, pp. 111542–111558.  External Links: 2601.19897, [Link](https://proceedings.mlr.press/v306/shenfeld26a.html) Cited by: §2. 
  * [32] N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao (2023) Reflexion: Language Agents with Verbal Reinforcement Learning.  In Advances in Neural Information Processing Systems,  Vol. 36, pp. 8634–8652.  External Links: [Document](https://dx.doi.org/10.52202/075280-0377) Cited by: §1, §2. 
  * [33] M. Shridhar, X. Yuan, M. Côté, Y. Bisk, A. Trischler, and M. Hausknecht (2021) ALFWorld: Aligning Text and Embodied Environments for Interactive Learning.  In International Conference on Learning Representations,  Cited by: §4.1. 
  * [34] A. Singh, J. D. Co-Reyes, R. Agarwal, A. Anand, P. Patil, X. Garcia, P. J. Liu, J. Harrison, J. Lee, K. Xu, A. T. Parisi, A. Kumar, A. A. Alemi, A. Rizkowsky, A. Nova, B. Adlam, B. Bohnet, G. F. Elsayed, H. Sedghi, I. Mordatch, I. Simpson, I. Gur, J. Snoek, J. Pennington, J. Hron, K. Kenealy, K. Swersky, K. Mahajan, L. A. Culp, L. Xiao, M. Bileschi, N. Constant, R. Novak, R. Liu, T. Warkentin, Y. Bansal, E. Dyer, B. Neyshabur, J. Sohl-Dickstein, and N. Fiedel (2024) Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models.  Transactions on Machine Learning Research.  External Links: [Link](https://openreview.net/forum?id=lNAyUngGFK) Cited by: §2. 
  * [35] Y. Sun, X. Wang, Z. Liu, J. Miller, A. A. Efros, and M. Hardt (2020) Test-Time Training with Self-Supervision for Generalization under Distribution Shifts.  In Proceedings of the 37th International Conference on Machine Learning,  Proceedings of Machine Learning Research, Vol. 119, pp. 9229–9248.  External Links: [Link](https://proceedings.mlr.press/v119/sun20b.html) Cited by: §2. 
  * [36] R. S. Sutton and A. G. Barto (2018) Reinforcement Learning: An Introduction.  Second edition, The MIT Press.  Cited by: §3.1. 
  * [37] M. Suzgun, M. Yuksekgonul, F. Bianchi, D. Jurafsky, and J. Zou (2026) Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory.  In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers),  Rabat, Morocco, pp. 7080–7106.  External Links: [Document](https://dx.doi.org/10.18653/v1/2026.eacl-long.333), [Link](https://aclanthology.org/2026.eacl-long.333/) Cited by: §2, §7.2. 
  * [38] H. Tan, X. Yang, H. Chen, J. Shao, Y. Wen, Y. Shen, W. Luo, X. Du, L. Guo, and Y. Li (2026) Hindsight Credit Assignment for Long-Horizon LLM Agents.  arXiv preprint arXiv:2603.08754.  External Links: 2603.08754 Cited by: §1, §3.1, §3.3.2. 
  * [39] H. Trivedi, T. Khot, M. Hartmann, R. Manku, V. Dong, E. Li, S. Gupta, A. Sabharwal, and N. Balasubramanian (2024) AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents.  In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),  Bangkok, Thailand, pp. 16022–16076.  External Links: [Document](https://dx.doi.org/10.18653/v1/2024.acl-long.850), [Link](https://aclanthology.org/2024.acl-long.850/) Cited by: §4.1, §7.5. 
  * [40] A. Wang, Z. Lu, J. Wang, S. Lv, Y. Liu, W. Lu, J. Xiao, Y. Zhuang, H. Yang, Q. Chen, and Y. Shen (2026) TTPO: Test-Time Policy Optimization.  arXiv preprint arXiv:2608.27448.  Cited by: §1, §2. 
  * [41] D. Wang, E. Shelhamer, S. Liu, B. Olshausen, and T. Darrell (2021) Tent: Fully Test-Time Adaptation by Entropy Minimization.  In International Conference on Learning Representations,  Cited by: §2. 
  * [42] G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar (2024) Voyager: An Open-Ended Embodied Agent with Large Language Models.  Transactions on Machine Learning Research.  Cited by: §2. 
  * [43] H. Wang, G. Wang, H. Xiao, Y. Zhou, Y. Pan, J. Wang, K. Xu, Y. Wen, X. Ruan, X. Chen, and H. Qi (2026) Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents.  arXiv preprint arXiv:2604.10674.  Cited by: §2. 
  * [44] Q. Wang, O. Fink, L. Van Gool, and D. Dai (2022) Continual Test-Time Domain Adaptation.  In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),  pp. 7191–7201.  External Links: [Document](https://dx.doi.org/10.1109/CVPR52688.2022.00706) Cited by: §2. 
  * [45] Y. Wang, J. Hao, Y. Shi, K. Yuan, and M. Sun (2026) No Time Like the Present: Agentic Test-Time Training for LLM Agents.  arXiv preprint arXiv:2607.03441.  Cited by: §1, §2. 
  * [46] Z. Wang, Z. Lu, Z. Yao, J. Wu, J. Wu, Z. Cai, Y. Sun, Z. Ye, L. Hao, Q. Gu, X. Cai, Y. Shen, and Y. Yang (2026) AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning.  arXiv preprint arXiv:2608.05987.  Cited by: §2. 
  * [47] R. J. Williams (1992) Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.  Machine Learning 8 (3–4), pp. 229–256.  External Links: [Document](https://dx.doi.org/10.1007/BF00992696) Cited by: §3.2, §7.2. 
  * [48] J. Wu, S. Yang, Z. Lu, F. Zhang, Y. Shen, L. Feng, H. Luo, Z. Lian, S. Zhang, Z. Wen, and J. Tao (2026) SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning.  arXiv preprint arXiv:2607.14777.  Cited by: §1, §2. 
  * [49] W. Xu, Z. Liang, K. Mei, H. Gao, J. Tan, and Y. Zhang (2025) A-Mem: Agentic Memory for LLM Agents.  In Advances in Neural Information Processing Systems,  Vol. 38, pp. 17577–17604.  External Links: [Document](https://dx.doi.org/10.52202/085713-0593), [Link](https://proceedings.neurips.cc/paper_files/paper/2025/file/19909c36f51abc4856b4560aff3d36d6-Paper-Conference.pdf) Cited by: §2, §7.2. 
  * [50] S. Yao, H. Chen, J. Yang, and K. Narasimhan (2022) WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents.  In Advances in Neural Information Processing Systems,  Vol. 35, pp. 20744–20757.  External Links: [Document](https://dx.doi.org/10.52202/068431-1508) Cited by: §4.1. 
  * [51] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao (2023) ReAct: Synergizing Reasoning and Acting in Language Models.  In International Conference on Learning Representations,  Cited by: §1, §3.1, §3.2, §4.1, §7.2. 
  * [52] W. Yeo, Y. Choi, T. Ki, and S. J. Hwang (2026) HINT-SD: Targeted Hindsight Self-Distillation for Long-Horizon Agents.  arXiv preprint arXiv:2605.17873.  Cited by: §2. 
  * [53] Z. Yuan, H. Yuan, C. Li, G. Dong, K. Lu, C. Tan, C. Zhou, and J. Zhou (2023) Scaling Relationship on Learning Mathematical Reasoning with Large Language Models.  arXiv preprint arXiv:2308.01825.  Cited by: §2, item 2, §3.3.1, §7.2. 
  * [54] M. Yuksekgonul, F. Bianchi, J. Boen, S. Liu, P. Lu, Z. Huang, C. Guestrin, and J. Zou (2025) Optimizing Generative AI by Backpropagating Language Model Feedback.  Nature 639 (8055), pp. 609–616.  External Links: [Document](https://dx.doi.org/10.1038/s41586-025-08661-4) Cited by: §7.2. 
  * [55] E. Zelikman, Y. Wu, J. Mu, and N. D. Goodman (2022) STaR: Bootstrapping Reasoning With Reasoning.  In Advances in Neural Information Processing Systems,  Vol. 35, pp. 15476–15488.  External Links: [Document](https://dx.doi.org/10.52202/068431-1126) Cited by: §2. 
  * [56] S. Zeng, Q. Wei, W. Brown, O. Frunza, Y. Nevmyvaka, and M. Hong (2025) Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment.  arXiv preprint arXiv:2505.11821v1.  External Links: 2505.11821v1 Cited by: §3.1. 
  * [57] Q. Zhang, C. Hu, S. Upasani, B. Ma, F. Hong, V. Kamanuru, J. Rainton, C. Wu, M. Ji, H. Li, U. Thakker, J. Y. Zou, and K. Olukotun (2026) Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models.  In International Conference on Learning Representations,  Vol. 2026, pp. 86069–86100.  External Links: [Link](https://proceedings.iclr.cc/paper_files/paper/2026/file/8a94ff6f922d995d7d3f4ebf4143e442-Paper-Conference.pdf) Cited by: §1, §1, §2, §3.1, §4.3, §7.2, §7.2. 
  * [58] S. Zhang, J. Wang, R. Zhou, J. Liao, Y. Feng, Z. Li, Y. Zheng, W. Zhang, Y. Wen, Z. Li, F. Xiong, Y. Qi, B. Tang, and M. Wen (2026) MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory.  arXiv preprint arXiv:2601.03192.  Cited by: §1, §2, §7.2. 
  * [59] A. Zhao, D. Huang, Q. Xu, M. Lin, Y. Liu, and G. Huang (2024) ExpeL: LLM Agents Are Experiential Learners.  In Proceedings of the AAAI Conference on Artificial Intelligence,  Vol. 38, pp. 19632–19642.  External Links: [Document](https://dx.doi.org/10.1609/aaai.v38i17.29936) Cited by: §1, §2. 
  * [60] S. Zhao, Z. Xie, M. Liu, J. Huang, G. Pang, F. Chen, and A. Grover (2026) Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models.  In Proceedings of the 43rd International Conference on Machine Learning,  Proceedings of Machine Learning Research, Vol. 306, pp. 162433–162448.  External Links: 2601.18734, [Link](https://proceedings.mlr.press/v306/zhao26be.html) Cited by: §1, §2, §3.3.1, §3.3.2, §4.1, §4.2, §7.2. 
  * [61] Y. Zuo, K. Zhang, L. Sheng, S. Qu, G. Cui, X. Zhu, H. Li, Y. Zhang, X. Long, E. Hua, B. Qi, Y. Sun, Z. Ma, L. Yuan, N. Ding, and B. Zhou (2025) TTRL: Test-Time Reinforcement Learning.  In Advances in Neural Information Processing Systems,  Vol. 38, pp. 131459–131483.  External Links: [Document](https://dx.doi.org/10.52202/085713-4376), [Link](https://proceedings.neurips.cc/paper_files/paper/2025/file/be690ea16f005c174f6c4102a5970e67-Paper-Conference.pdf) Cited by: §1, §2. 
  * [62] A. Zweiger, J. Pari, H. Guo, Y. Kim, and P. Agrawal (2025) Self-Adapting Language Models.  In Advances in Neural Information Processing Systems,  Vol. 38, pp. 74084–74115.  External Links: [Document](https://dx.doi.org/10.52202/085713-2483), [Link](https://proceedings.neurips.cc/paper_files/paper/2025/file/6b41e04c41726e2a60e456d0a2b961ab-Paper-Conference.pdf) Cited by: §1, §2, §3.1. 



\beginappendix

## 6 Additional Method Details

### 6.1 What Verifier Selection and Distillation Identify

We separate what the verifier selects from what the distillation objective fits. Fix a task xx and let R⁡(τ)=𝒱⁡(x,τ)∈{0,1}R(\tau)=\mathcal{V}(x,\tau)\in\\{0,1\\} be the verification outcome that gates an update (viv_{i} in Eq. (3)). Reported benchmark success can use a stricter criterion. The results below concern the population of possible trajectories under a fixed collection policy, whereas ASCENT makes a single attempt per task in one pass over the task stream.

#### 6.1.1 Verifier Selection Reweights Completed Trajectories

Let π\pi be the response policy that collects the trajectory, including its decoding rule, and let ℙπ​(τ∣x)\mathbb{P}_{\pi}(\tau\mid x) be the resulting trajectory distribution in the environment. If the acceptance probability Jx​(π):=Prπ,ℰ⁡(R=1∣x)>0J_{x}(\pi):=\Pr_{\pi,\mathcal{E}}(R=1\mid x)>0, the trajectories selected for an update follow

| ℙπ+​(τ∣x)=R⁡(τ)​ℙπ​(τ∣x)Jx​(π).\mathbb{P}_{\pi}^{+}(\tau\mid x)=\frac{R(\tau)\mathbb{P}_{\pi}(\tau\mid x)}{J_{x}(\pi)}. |  | (14)  
---|---|---|---  
  
The selected distribution is ℙπ\mathbb{P}_{\pi} restricted to accepted trajectories and renormalized. At a history S=HtS=H_{t} just before the agent responds, let Vπ​(S)=Prπ⁡(R=1∣S)V^{\pi}(S)=\Pr_{\pi}(R=1\mid S), and let Qπ​(S,u)Q^{\pi}(S,u) be the probability of acceptance when the agent produces the complete response uu and then follows π\pi. When Vπ​(S)>0V^{\pi}(S)>0, Bayes’ rule gives

| π+​(u∣S):=Prπ⁡(u∣S,R=1)=π⁡(u∣S)​Qπ​(S,u)Vπ​(S).\pi^{+}(u\mid S):=\Pr_{\pi}(u\mid S,R=1)=\pi(u\mid S)\frac{Q^{\pi}(S,u)}{V^{\pi}(S)}. |  | (15)  
---|---|---|---  
  
Verifier selection thus reweights each response by its relative chance of leading to acceptance [6, 8]. It assigns no causal credit to individual turns of an attempt and does not reveal the outcome of an untried response. With only one collected response at a history, selection cannot rank alternatives there. The hindsight-conditioned teacher supplies soft targets at the student prefixes, but whether these targets improve later tasks must be tested empirically.

#### 6.1.2 Forward KL Fits Teacher Predictions at Student Prefixes

The student fits the teacher at the student prefixes of selected trajectories, with the collection policy π\pi still fixed for task xx. For a selected trajectory τ\tau, let ℐ⁡(τ)\mathcal{I}(\tau) index the turns with nonempty responses, which enter the student loss, and let Lτ​tL_{\tau t} be the number of generated tokens at turn tt. Assume |ℐ⁡(τ)|>0|\mathcal{I}(\tau)|>0 for selected trajectories (otherwise condition the selected distribution on this event). Averaging tokens within each turn and then across turns, as in Eq. (8), gives the training measure

| νπ(dτ,t,j∣x,R=1)=ℙπ+(dτ∣x)𝟏{t∈ℐ(τ)}|ℐ⁡(τ)|𝟏{1≤j≤Lτ​t}Lτ​t.\nu_{\pi}(\mathrm{d}\tau,t,j\mid x,R=1)=\mathbb{P}_{\pi}^{+}(\mathrm{d}\tau\mid x)\frac{\mathbf{1}\\{t\in\mathcal{I}(\tau)\\}}{|\mathcal{I}(\tau)|}\frac{\mathbf{1}\\{1\leq j\leq L_{\tau t}\\}}{L_{\tau t}}. |  | (16)  
---|---|---|---  
  
This measure draws an accepted trajectory from ℙπ+\mathbb{P}_{\pi}^{+}, then a turn uniformly from ℐ⁡(τ)\mathcal{I}(\tau), then a token position uniformly within that turn. Let Z=(τ,t,j)Z=(\tau,t,j) be an index drawn from it, cZ=cτ,t,jc_{Z}=c_{\tau,t,j} its student prefix (defined in Sec. 3.2), and z⁡(τ)z(\tau) the privileged information (defined in Eq. (12)). At this prefix the frozen teacher supplies qZ(⋅)=pθ0(⋅∣z(τ)⊕cZ)q_{Z}(\cdot)=p_{\theta_{0}}(\cdot\mid z(\tau)\mathbin{\oplus}c_{Z}), and the population objective for a candidate adapter ϕ\phi is

| ℒx(ϕ;π)=𝔼Z∼νπ[DKL(qZ∥pθ0,ϕ(⋅∣cZ))].\mathcal{L}_{x}(\phi;\pi)=\mathbb{E}_{Z\sim\nu_{\pi}}\left[D_{\mathrm{KL}}\\!\left(q_{Z}\,\middle\|\,p_{\theta_{0},\phi}(\cdot\mid c_{Z})\right)\right]. |  | (17)  
---|---|---|---  
  
Let q¯(⋅∣c)=𝔼νπ[qZ(⋅)∣cZ=c]\bar{q}(\cdot\mid c)=\mathbb{E}_{\nu_{\pi}}[q_{Z}(\cdot)\mid c_{Z}=c] denote the mean teacher distribution at a student prefix cc, which averages the teacher distributions of the accepted trajectories through cc, weighted as in Eq. (8).

###### Lemma 6.1 (Forward-KL population target).

For any student next-token distributions at the student prefixes,

| ℒx​(ϕ,π)\displaystyle\mathcal{L}_{x}(\phi;\pi) | =𝔼Z∼νπ[DKL(qZ∥q¯(⋅∣cZ))]\displaystyle=\mathbb{E}_{Z\sim\nu_{\pi}}\left[D_{\mathrm{KL}}\\!\left(q_{Z}\,\middle\|\,\bar{q}(\cdot\mid c_{Z})\right)\right] |   
---|---|---|---  
|  | +𝔼Z∼νπ[DKL(q¯(⋅∣cZ)∥pθ0,ϕ(⋅∣cZ))].\displaystyle\quad+\mathbb{E}_{Z\sim\nu_{\pi}}\left[D_{\mathrm{KL}}\\!\left(\bar{q}(\cdot\mid c_{Z})\,\middle\|\,p_{\theta_{0},\phi}(\cdot\mid c_{Z})\right)\right]. |  | (18)  
  
###### Proof.

Condition on cZ=cc_{Z}=c, expand the KL terms over next tokens, and use q¯(⋅∣c)=𝔼[qZ(⋅)∣cZ=c]\bar{q}(\cdot\mid c)=\mathbb{E}[q_{Z}(\cdot)\mid c_{Z}=c]. The terms involving the student distribution then agree on both sides, and averaging over student prefixes gives Eq. (18). ∎

The first term in Eq. (18) does not depend on ϕ\phi, so an unrestricted minimizer matches q¯\bar{q} at the student prefixes covered by the training measure. The student’s responses determine which prefixes appear and how they are weighted, while the hindsight-conditioned teacher supplies their soft targets. The identity characterizes this target only. It does not show that the teacher’s predictions help, that a finite adapter reaches the target, or that the fitted behavior transfers to later tasks.

#### 6.1.3 Verification Does Not Identify Shorter Paths

An accepted trajectory can contain valid but unnecessary turns. Suppose _finish_ succeeds now, while _wait_ leads to success on the next turn. Both responses can have Qπ​(S,u)=1Q^{\pi}(S,u)=1, so verifier selection prefers neither, and the action-validity filter keeps both when the environment executes them.

Let TT be the episode length under π\pi. Selecting accepted episodes shifts its mean by

| 𝔼π​[T∣R=1]−𝔼π​[T]=Covπ⁡(T,R)Jx​(π).\mathbb{E}_{\pi}[T\mid R=1]-\mathbb{E}_{\pi}[T]=\frac{\operatorname{Cov}_{\pi}(T,R)}{J_{x}(\pi)}. |  | (19)  
---|---|---|---  
  
The sign of this shift follows the covariance between length and acceptance, so verification alone does not imply shorter accepted trajectories. With a turn cap, failed episodes may run to the limit, so a drop in capped all-episode mean turns can reflect more successes even when successful paths are no shorter. Appendix 8.2 also compares turns on shared successes, the tasks that both the un-evolved base model and ASCENT solve. This comparison describes turn differences on those tasks. It does not identify which individual turns the updated policy learned to avoid.

## 7 Additional Experimental Details

### 7.1 Evaluation Protocol and Comparison Controls

All methods in a paired comparison run on the same task order. After each scored episode, every online method receives the same environment verification signal and may use it to update its memory, skills, or model weights. All in-context adaptation methods start from the ReAct template that ASCENT uses to collect its attempts. Offline self-evolution methods instead build their adaptation state from the same fixed-size training split before evaluation. These controls match the task stream, prompt template, and offline data volume, but leave each method free to learn in its own way. On WebShop, exact success requires a task score of 11, whereas ASCENT’s adaptation gate accepts a score of at least 0.90.9.

### 7.2 Baseline Details

Base. The base is the initial LLM with the shared ReAct interaction template [51] and makes no cross-task update.

Offline in-context adaptation. MemP [10], ACE [57], EvoSkill [4], GEPA [3], TextGrad [54], and Trace2Skill [25] build memories, skills, or prompts from the fixed training split before evaluation.

Online in-context adaptation. MemP [10], ReasoningBank [26], ACE [57], A-Mem [49], MemRL [58], and Dynamic Cheatsheet [37] update memory or skills after a scored episode, so the new context can first affect the next task.

Figure 6: Teacher prompt template. For a verifier-accepted episode, the privileged information ziz_{i} is prepended to each shared student prefix. The empty think block reflects disabled thinking mode. Each response’s reasoning is the visible ReAct rationale before the action tag.

Online parametric adaptation. TT-OPSD is our test-time (TT) adaptation of OPSD [60] to the one-pass task stream. Its teacher receives the turn-local privileged information zi,tTT=(xi,wi,t,oi,t+1)z^{\mathrm{TT}}_{i,t}=(x_{i},w_{i,t},o_{i,t+1}) together with each student prefix ci,t,jc_{i,t,j}, so the teacher’s extra information covers only the current response and returned observation, without the completed trajectory. Its verifier gate, student prefixes, forward-KL objective, and optimizer match ASCENT, but it keeps invalid-action turns in accepted episodes. Sec. 3.2 also studies direct imitation (Online ungated imitation, Online RFT [53, 30], and Online RFT+KL) and single-attempt policy gradient (Online REINFORCE [47] and Online REINFORCE++ [17]).

Single-attempt policy gradient. Online REINFORCE and Online REINFORCE++ use ASCENT’s LoRA configuration and optimizer and update after each episode. Online REINFORCE++ uses a per-token KL penalty to the frozen base with β=0.01\beta=0.01 and ratio clipping with ϵ=0.2\epsilon=0.2. Where REINFORCE++ uses batch statistics, Online REINFORCE++ normalizes each token advantage by the mean and standard deviation of all token advantages in the stream so far, including the current episode.

### 7.3 Implementation Details

Turn budget and action validity. Every ALFWorld and WebShop attempt, in evaluation and in online training, has a 50-turn budget. An attempt still unfinished at turn 50 stops there and is scored as a failure. Each environment decides validity from its own response to the action. ALFWorld executes an action only if it belongs to its current available-action list and otherwise returns “Nothing happens.” We set di,t=1d_{i,t}=1 for listed actions and di,t=0d_{i,t}=0 for unparsed or unavailable ones, so a listed action that has no effect still counts as valid. WebShop executes search[⋅\cdot] only on a page with a search bar and click[⋅\cdot] only on an element of the current page. We set di,t=0d_{i,t}=0 for other or unparsed actions. In AppWorld, di,t=0d_{i,t}=0 when the turn contains no parseable code or its execution raises an error.

Method | Pick | Look | Clean | Heat | Cool | Pick2 | Avg.SR |  Mean turns (all) |  Time s/ep  
---|---|---|---|---|---|---|---|---|---  
Qwen3.5-4B  
Base | 87.5 | 38.9 | 29.0 | 8.7 | 28.6 | 76.5 | 43.3 | 36.2 | 107  
Offline  
MemP (offline) | 91.7 | 72.2 | 64.5 | 8.7 | 52.4 | 76.5 | 60.4±\pm4.5 | 30.0 | 69  
ACE (offline) | 79.2 | 33.3 | 41.9 | 8.7 | 42.9 | 23.5 | 39.6±\pm3.2 | 38.5 | 106  
EvoSkill | 83.3 | 40.7 | 29.0 | 5.8 | 47.6 | 58.8 | 43.0±\pm4.6 | 37.4 | 146  
GEPA | 95.8 | 38.9 | 38.7 | 47.8 | 76.2 | 70.6 | 60.4±\pm4.9 | 32.8 | 95  
TextGrad | 79.2 | 66.7 | 22.6 | 8.7 | 19.0 | 47.1 | 38.8±\pm6.4 | 38.6 | 112  
Trace2Skill | 79.2 | 50.0 | 16.1 | 4.3 | 19.0 | 76.5 | 38.1±\pm5.3 | 39.4 | 127  
Online-built artifacts (frozen on unseen)  
MemP | 87.5 | 88.9 | 80.6 | 4.3 | 42.9 | 29.4 | 57.5±\pm6.9 | 31.1 | 66  
ReasoningBank | 87.5 | 16.7 | 41.9 | 21.7 | 52.4 | 47.1 | 45.5±\pm7.6 | 36.0 | 91  
ACE | 70.8 | 11.1 | 3.2 | 0.0 | 4.8 | 35.3 | 20.1±\pm7.4 | 45.2 | 134  
A-Mem | 91.7 | 38.9 | 41.9 | 8.7 | 42.9 | 58.8 | 47.0±\pm9.0 | 34.5 | 84  
MemRL | 79.2 | 38.9 | 45.2 | 17.4 | 33.3 | 70.6 | 47.0±\pm6.1 | 35.6 | 88  
Dynamic Cheatsheet | 79.2 | 27.8 | 12.9 | 13.0 | 38.1 | 82.4 | 39.6±\pm7.9 | 38.7 | 102  
TT-OPSD | 98.6 | 66.7 | 35.5 | 5.8 | 33.3 | 70.6 | 50.0±\pm4.8 | 33.9 | 67  
ASCENT | 86.1 | 46.3 | 72.0 | 20.3 | 69.8 | 78.4 | 62.7±\pm7.5 | 28.4 | 94  
Qwen3.5-9B  
Base | 91.7 | 88.9 | 38.7 | 4.3 | 0.0 | 88.2 | 49.3 | 34.3 | 138  
Offline  
MemP (offline) | 100.0 | 100.0 | 71.0 | 21.7 | 33.3 | 82.4 | 67.2±\pm6.7 | 27.8 | 94  
ACE (offline) | 83.3 | 66.7 | 41.9 | 39.1 | 42.9 | 88.2 | 58.2±\pm7.3 | 30.9 | 141  
EvoSkill | 95.8 | 70.4 | 46.2 | 23.2 | 38.1 | 90.2 | 58.7±\pm4.3 | 31.9 | 163  
GEPA | 100.0 | 83.3 | 58.1 | 23.9 | 11.9 | 82.4 | 59.0±\pm9.0 | 30.8 | 139  
TextGrad | 100.0 | 77.8 | 64.5 | 17.4 | 85.7 | 64.7 | 67.9±\pm4.7 | 30.3 | 173  
Trace2Skill | 100.0 | 94.4 | 41.9 | 34.8 | 57.1 | 70.6 | 64.2±\pm5.1 | 32.4 | 157  
Online-built artifacts (frozen on unseen)  
MemP | 100.0 | 83.3 | 71.0 | 13.0 | 76.2 | 82.4 | 70.1±\pm4.3 | 26.3 | 84  
ReasoningBank | 95.8 | 66.7 | 67.7 | 26.1 | 28.6 | 64.7 | 59.0±\pm8.1 | 32.8 | 120  
ACE | 83.3 | 66.7 | 19.4 | 17.4 | 71.4 | 82.4 | 53.0±\pm5.9 | 35.1 | 151  
A-Mem | 95.8 | 72.2 | 45.2 | 26.1 | 47.6 | 88.2 | 60.4±\pm7.1 | 32.0 | 111  
MemRL | 100.0 | 88.9 | 54.8 | 30.4 | 57.1 | 88.2 | 67.9±\pm6.6 | 29.5 | 107  
Dynamic Cheatsheet | 95.8 | 61.1 | 19.4 | 21.7 | 9.5 | 88.2 | 46.3±\pm5.2 | 35.4 | 151  
TT-OPSD | 100.0 | 72.2 | 48.4 | 8.7 | 4.8 | 96.1 | 53.2±\pm5.0 | 31.8 | 89  
ASCENT | 97.2 | 81.5 | 90.3 | 72.5 | 96.8 | 94.1 | 88.8±\pm6.6 | 17.0 | 74  
Table 4: ALFWorld held-out transfer. Success rate (SR, %) by task type and over the full held-out stream, mean turns per episode, and runtime (s/episode). Seen-stream artifacts remain fixed. Bold indicates the highest SR (including ties) and fewest mean turns within each backbone.  Method | Pick | Look | Clean | Heat | Cool | Pick2 |  Avg.SR ( Δ\Delta vs Transfer) |  Mean turns (all) |  Time s/ep  
---|---|---|---|---|---|---|---|---|---  
Qwen3.5-4B  
Base | 87.5 | 38.9 | 29.0 | 8.7 | 28.6 | 76.5 | 43.3 | 36.2 | 107  
Online adaptation on unseen stream  
MemP | 100.0 | 94.4 | 74.2 | 21.7 | 33.3 | 64.7 | 64.9±\pm4.8 (++7.4) | 28.2 | 58  
ReasoningBank | 87.5 | 33.3 | 48.4 | 26.1 | 52.4 | 52.9 | 50.7±\pm7.3 (++5.2) | 35.0 | 89  
ACE | 75.0 | 11.1 | 3.2 | 0.0 | 9.5 | 35.3 | 21.6±\pm5.2 (++1.5) | 45.5 | 225  
A-Mem | 87.5 | 55.6 | 51.6 | 21.7 | 28.6 | 52.9 | 50.0±\pm3.9 (++3.0) | 33.5 | 85  
MemRL | 79.2 | 38.9 | 51.6 | 21.7 | 42.9 | 47.1 | 47.8±\pm8.7 (++0.8) | 34.3 | 86  
Dynamic Cheatsheet | 79.2 | 61.1 | 29.0 | 0.0 | 23.8 | 35.3 | 37.3±\pm6.1 (−-2.3) | 37.5 | 99  
TT-OPSD | 97.2 | 81.5 | 39.8 | 15.9 | 28.6 | 90.2 | 56.2±\pm3.1 (++6.2) | 31.5 | 90  
ASCENT | 90.3 | 83.3 | 83.9 | 53.6 | 77.8 | 82.4 | 78.6±\pm5.1 (++15.9) | 21.1 | 98  
Qwen3.5-9B  
Base | 91.7 | 88.9 | 38.7 | 4.3 | 0.0 | 88.2 | 49.3 | 34.3 | 138  
Online adaptation on unseen stream  
MemP | 100.0 | 100.0 | 83.9 | 13.0 | 76.2 | 94.1 | 76.9±\pm5.2 (++6.8) | 24.9 | 74  
ReasoningBank | 95.8 | 66.7 | 58.1 | 17.4 | 38.1 | 70.6 | 57.5±\pm4.8 (−-1.5) | 33.1 | 124  
ACE | 91.7 | 55.6 | 38.7 | 13.0 | 66.7 | 76.5 | 55.2±\pm7.1 (++2.2) | 33.2 | 142  
A-Mem | 83.3 | 72.2 | 64.5 | 21.7 | 33.3 | 82.4 | 59.0±\pm3.7 (−-1.4) | 31.7 | 116  
MemRL | 95.8 | 94.4 | 74.2 | 21.7 | 52.4 | 94.1 | 70.9±\pm7.2 (++3.0) | 27.0 | 99  
Dynamic Cheatsheet | 95.8 | 61.1 | 54.8 | 17.4 | 47.6 | 88.2 | 59.7±\pm8.6 (++13.4) | 29.1 | 121  
TT-OPSD | 97.2 | 85.2 | 46.2 | 13.0 | 4.8 | 96.1 | 54.7±\pm0.9 (++1.5) | 30.3 | 111  
ASCENT | 98.6 | 92.6 | 88.2 | 72.5 | 95.2 | 96.1 | 90.0±\pm3.6 (++1.2) | 15.7 | 107  
Table 5: ALFWorld continued OaTTT. Pre-update success rate (SR, %) by task type and over the held-out stream, mean turns per episode, and runtime (s/episode). Δ\Delta is the change from frozen transfer in Table 4. Bold indicates the highest SR (including ties) and fewest mean turns within each backbone. 

Teacher input. The shared student prefix ci,t,jc_{i,t,j} includes any retrieved context used during the attempt. The student reads ci,t,jc_{i,t,j}, whereas the teacher reads zi⊕ci,t,jz_{i}\oplus c_{i,t,j}, with teacher token positions shifted by |zi||z_{i}|. The trajectory text in ziz_{i} conditions the teacher. It is not itself a prediction target. Fig. 6 shows this arrangement for reasoning–action responses.

Optimization. The teacher is a frozen copy of the initial LLM, pθ0p_{\theta_{0}}, so teacher inference simply disables the student’s LoRA adapter and returns detached full-vocabulary logits. LoRA adapts the down_proj modules with rank 16 and scale 32. After each verifier-accepted episode, ASCENT takes two optimizer steps on the forward-KL loss of Eq. (8) with AdamW [22] and learning rate 1.5×10−41.5\times 10^{-4}. The adapter and optimizer state carry across tasks, while the trajectory is discarded after its update. All experiments run on a single NVIDIA H100 GPU.

### 7.4 Full Benchmark Results

Tables 4–5 report the full ALFWorld unseen results for frozen transfer and continued OaTTT, which Fig. 3 summarizes in the main paper. The full ALFWorld seen and WebShop results, including offline methods, are in the main paper (Tables 1 and 2).

### 7.5 Additional Evaluation on AppWorld

| Test-Normal | Test-Challenge  
---|---|---  
Method | TGC | SGC | TGC | SGC  
Base | 20.2 | 5.4 | 15.8 | 5.8  
MemP | 22.0±\pm2.1 | 1.8±\pm0.2 | 16.1±\pm3.3 | 5.8±\pm0.7  
ReasoningBank | 19.6±\pm1.9 | 3.6±\pm0.7 | 19.2±\pm2.8 | 4.3±\pm1.1  
ACE | 22.0±\pm4.1 | 3.6±\pm1.2 | 14.4±\pm3.9 | 2.2±\pm2.0  
A-Mem | 14.3±\pm0.4 | 8.9±\pm1.2 | 11.3±\pm1.1 | 2.2±\pm0.4  
Dynamic Cheatsheet | 18.5±\pm2.3 | 1.8±\pm0.6 | 14.9±\pm1.4 | 2.9±\pm0.5  
TT-OPSD | 19.1±\pm3.2 | 5.7±\pm0.8 | 16.2±\pm2.0 | 3.8±\pm1.2  
ASCENT | 26.2±\pm1.3 | 9.5±\pm0.9 | 20.0±\pm2.5 | 5.3±\pm0.9  
Table 6: AppWorld online results. Task-goal completion (TGC, %) and scenario-goal completion (SGC, %) on Test-Normal and Test-Challenge with Qwen3.5-4B.

AppWorld [39] tests interactive code generation across simulated applications. We evaluate Qwen3.5-4B on Test-Normal (168 tasks in 56 scenarios) and Test-Challenge (417 tasks in 139 scenarios). Each task has one scored attempt with a 30-interaction budget, and online methods receive the episode’s verification outcome only after that attempt. Task Goal Completion (TGC) counts tasks that pass every state-based test, and Scenario Goal Completion (SGC) counts scenarios in which every task variant succeeds. Table 6 reports these official metrics under the OaTTT protocol.

Task-level gains extend to interactive coding. As shown in Table 6, ASCENT improves TGC over the base on both splits and leads every online comparator in the table on Test-Normal TGC and SGC. On Test-Challenge, its task-level gain does not translate into a scenario-level lead.

## 8 Additional Analyses and Results

### 8.1 Teacher Evolution in OaTTT

ASCENT keeps the teacher fixed while the student changes. To analyze this choice, we study teacher evolution in OaTTT: the teacher gets its own adapter ϕ¯i\bar{\phi}_{i}, which follows the student. Let Qϕ¯(⋅∣z,c)=pθ0,ϕ¯(⋅∣z⊕c)Q_{\bar{\phi}}(\cdot\mid z,c)=p_{\theta_{0},\bar{\phi}}(\cdot\mid z\mathbin{\oplus}c) denote this teacher at a student prefix cc with privileged information zz, and let Qϕ¯i​(τi)Q_{\bar{\phi}_{i}}(\tau_{i}) collect its targets at the student prefixes of an accepted episode. With token indices suppressed, episode ii gives

| τi∼Pϕi,qi=Qϕ¯i​(τi),ϕi+1=𝖤𝗏𝗈𝗅𝗏𝖾⁡(ϕi,ℒiASCENT​[qi]).\tau_{i}\sim P_{\phi_{i}},\qquad q_{i}=Q_{\bar{\phi}_{i}}(\tau_{i}),\qquad\phi_{i+1}=\mathsf{Evolve}\bigl(\phi_{i};\mathcal{L}_{i}^{\mathrm{ASCENT}}[q_{i}]\bigr). |  | (20)  
---|---|---|---  
  
Here PϕiP_{\phi_{i}} is the trajectory distribution of the student pθ0,ϕip_{\theta_{0},\phi_{i}} on task xix_{i}, 𝖤𝗏𝗈𝗅𝗏𝖾\mathsf{Evolve} is the operator in Eq. (9), and ℒiASCENT​[qi]\mathcal{L}_{i}^{\mathrm{ASCENT}}[q_{i}] is Eq. (8) with targets qiq_{i}. ASCENT sets ϕ¯i=0\bar{\phi}_{i}=0, so its teacher is always pθ0p_{\theta_{0}}.

EMA and snapshot teachers. The evolving teachers keep the hindsight context, loss, and optimizer of ASCENT and differ from it only in letting ϕ¯\bar{\phi} follow the student. An exponential moving average (EMA) teacher applies ϕ¯←β​ϕ¯+(1−β)​ϕ\bar{\phi}\leftarrow\beta\bar{\phi}+(1-\beta)\phi after every optimizer step. A periodic snapshot teacher starts at ϕ¯=0\bar{\phi}=0 and copies the student’s adapter after every NN-th update (one update per accepted episode), like the target network of deep Q-learning [24]. A fixed teacher still gives different targets across episodes, because ziz_{i} and the student prefixes change. An evolving teacher also changes the rule that turns this evidence into supervision. We run these variants with Qwen3.5-4B on the ALFWorld seen stream (Table 7).

Teacher | Schedule | SR (%) |  Early→\rightarrowlate SR (%)  
---|---|---|---  
Fixed  
ASCENT (frozen π0\pi_{0}) | fixed | 69.5 | 59.0→\rightarrow75.2  
EMA of the student  
EMA | β=0.999\beta=0.999 | 61.4 | 48.6→\rightarrow77.1  
EMA | β=0.99\beta=0.99 | 57.1 | 54.3→\rightarrow57.1  
EMA | β=0.90\beta=0.90 | 24.3 | 48.6→\rightarrow2.9  
Periodic student snapshot  
Snapshot | N=30N=30 | 55.7 | 51.4→\rightarrow68.6  
Snapshot | N=20N=20 | 63.6 | 57.1→\rightarrow62.9  
Snapshot | N=10N=10 | 45.7 | 60.0→\rightarrow25.7  
Table 7: Analysis of teacher evolution in OaTTT on ALFWorld seen. The snapshot teacher is a copy of the student that is refreshed every NN updates (the frozen π0\pi_{0} before the first refresh). Early and late refer to tasks 1–35 and 106–140, respectively.

(a) EMA teachers

(b) Periodic snapshots

(c) Success vs. teacher distance

Figure 7: Dynamics of teacher evolution in OaTTT on ALFWorld seen. Qwen3.5-4B with evolving teachers: (a,b) trailing 30-episode success against the frozen teacher in the same setup, with triangles at snapshot refreshes, and (c) stream success against the mean teacher–student adapter distance ‖ϕ¯−ϕ‖2\|\bar{\phi}-\phi\|_{2} before each update, with the frozen teacher as the dashed line.

Faster teacher evolution strengthens the feedback loop. An evolving teacher returns student drift as supervision on later tasks. Stop-gradient blocks this path only within an update, leaving it open across tasks. As shown in Table 7, the fastest-evolving teachers (EMA with β=0.90\beta=0.90 and snapshots with N=10N=10) lose success along the stream, whereas slower ones hold or improve. The slowest EMA (β=0.999\beta=0.999) ends the stream level with the frozen teacher in the same setup (Fig. 7(a)). No evolving teacher matches that teacher over the whole stream. Teacher evolution thus need not lose success along the stream, and here it brings no net gain. These results order the schedules without locating the evolution rate at which success starts to fall, because β\beta or NN alone does not set the lag in episodes. An EMA remembers about (1−β)−1(1-\beta)^{-1} optimizer steps, about half as many accepted updates, and because the verifier gates updates, the lag of either teacher in episodes also depends on the success rate.

Success falls when the teacher stays close to the student. Fig. 7(c) relates stream success to how far the teacher’s adapter stays from the student’s. The two teachers closest to the student, EMA with β=0.90\beta=0.90 and snapshots refreshed every 10 updates, succeed least. Once the mean distance exceeds about 1.51.5, success levels off between 56%56\% and 64%64\%, below the 65%65\% of the frozen teacher in the same setup. The snapshots show this loop over time (Fig. 7(b)): each refresh hands the teacher the student’s latest weights, and with frequent refreshes success declines through the second half of the stream.

Gating and filtering do not remove the second feedback path of an evolving teacher. The verification outcome selects episodes and assigns no credit to individual turns. The action-validity filter removes invalid-action turns from ziz_{i} and cannot judge the remaining valid turns. With an evolving teacher, those turns also shape the model that supplies later targets, a second feedback path that neither gating nor filtering removes.

Frozen and evolving teachers define different targets. Let μϕ+​(z∣c)\mu_{\phi}^{+}(z\mid c) be the distribution of privileged information from verifier-accepted trajectories of the student with adapter ϕ\phi at a student prefix cc. Under a fixed collection distribution (Appendix 6.1.2), an unrestricted student fitted by forward KL to the frozen teacher Q0Q_{0} (ϕ¯=0\bar{\phi}=0) has the centroid solution

| p∗(⋅∣c)=𝔼z∼μϕ+(⋅∣c)[Q0(⋅∣z,c)].p^{*}(\cdot\mid c)=\mathbb{E}_{z\sim\mu_{\phi}^{+}(\cdot\mid c)}\left[Q_{0}(\cdot\mid z,c)\right]. |  | (21)  
---|---|---|---  
  
The selected zz may shift as the student evolves, but the same frozen teacher pθ0p_{\theta_{0}} always interprets it. When the teacher evolves with the student, both the selected zz and its interpretation depend on the student, and a self-consistent solution satisfies

| pϕ∗(⋅∣c)=𝔼z∼μϕ∗+(⋅∣c)[Qϕ∗(⋅∣z,c)].p_{\phi^{*}}(\cdot\mid c)=\mathbb{E}_{z\sim\mu_{\phi^{*}}^{+}(\cdot\mid c)}\left[Q_{\phi^{*}}(\cdot\mid z,c)\right]. |  | (22)  
---|---|---|---  
  
This self-consistency condition does not establish convergence. A policy can satisfy it without succeeding in the environment, because the policy being trained determines both the selected information and its interpretation. Direct imitation is the extreme case, with one-hot targets that give no mass to alternative tokens.

Low distillation loss need not mean task success. Forward KL measures agreement with the teacher and ignores task success. It can also cover only the alternatives its teacher retains. In Online ungated imitation, the zero-lag extreme in which the student imitates its own generated tokens, the student’s entropy on fixed prompts approaches zero as drift from the initial policy grows. Teacher entropy was not logged, so these observations show that the student collapses and leave open whether teacher entropy collapse passes to the student.

A KL anchor limits drift and supplies no episode-conditioned targets. Online RFT+KL limits some drift, but its success stays well below that of ASCENT with the frozen privileged teacher. Its KL anchor keeps the student closer to the base, whereas the privileged teacher also supplies episode-conditioned soft targets.

Collapsed behavior does not require a small adapter. At an endpoint, the collapsed direct imitation checkpoint has a lower adapter stable rank and a next-token gradient about 25×25\times larger than the frozen-teacher checkpoint. Online ungated imitation also ends with a larger adapter norm than the frozen teacher, while its valid-action rate falls. Student entropy approaches zero, repeated actions become more common, and executable actions become rare. These are behavioral and endpoint associations: stable rank does not establish the cause of collapse, and adapter norm or probe gradient neither measures the realized update nor identifies a LoRA capacity limit.

ASCENT changes the evidence and keeps the teacher fixed. ASCENT’s targets vary with ziz_{i} and ci,t,jc_{i,t,j}, while the teacher pθ0p_{\theta_{0}} that interprets them stays the same. Slowly evolving teachers can hold their success and even improve late in the stream. None exceeds the frozen teacher overall. Updating both models after every episode lets drift compound as in Sec. 3.2, now through the targets as well. Evolving the teacher only in stages, after a verified gain of the student, could let the supervision improve without this loop.

### 8.2 Turn Efficiency under Success Composition

In the ALFWorld seen-stream runs, failed episodes use all Tmax=50T_{\max}=50 turns, so the capped all-episode mean decomposes as

| 𝔼⁡[T]=Pr⁡(W)​𝔼​[T∣W]+(1−Pr⁡(W))​Tmax,\mathbb{E}[T]=\Pr(W)\,\mathbb{E}[T\mid W]+\bigl(1-\Pr(W)\bigr)T_{\max}, |  | (23)  
---|---|---|---  
  
where WW denotes episode success. A policy with higher success therefore has fewer mean turns even if its successful trajectories are no shorter. To compare turn use apart from this success composition, we pair each of the 3 ASCENT runs per scale in Table 1 with the base on the same task order.

Among successful episodes, the base and ASCENT average 17.917.9 and 15.815.8 turns at 4B, and 18.418.4 and 12.612.6 at 9B. These means still cover different sets of solved tasks, so Table 8 compares turn counts only on tasks that both the un-evolved base model and ASCENT solve (shared successes).

Scale | Both won | ASCENT only | Base only |  Mean Δ​T\Delta T |  Median Δ​T\Delta T |  ASCENT uses fewer turns (%)  
---|---|---|---|---|---|---  
4B | 61.761.7±\pm1.2 | 35.735.7±\pm6.6 | 3.33.3±\pm1.2 | −4.8-4.8±\pm0.9 | −2.3-2.3±\pm0.5 | 63.863.8±\pm1.8  
9B | 74.374.3±\pm0.5 | 34.034.0±\pm2.2 | 2.72.7±\pm0.5 | −7.0-7.0±\pm0.3 | −3.5-3.5±\pm0.4 | 67.367.3±\pm4.7  
Table 8: Shared-success turn comparison on ALFWorld seen. Δ​T=TASCENT−TBase\Delta T=T_{\mathrm{ASCENT}}-T_{\mathrm{Base}} on tasks solved by both the un-evolved base model and ASCENT, so negative values favor ASCENT. The first three columns count the tasks in the 140-task stream solved by both, by ASCENT only, and by the base only. The last column is the percentage of shared successes on which ASCENT uses fewer turns than the base. Entries are mean±\pmstd of per-run statistics over the 3 paired runs per scale.

ASCENT uses fewer turns on most shared successes. Table 8 shows that, on these shared successes, the mean and median Δ​T\Delta T are negative at both scales, and ASCENT uses fewer turns than the base on most of these tasks. The paired comparison thus supports an interaction-efficiency gain beyond the change in the success/failure mix. It identifies neither the individual turns avoided nor the cause of each shorter path.

### 8.3 Retrieval and Execution for In-Context Adaptation

Qwen3.5-4B  Qwen3.5-9B

(a) Retrieval hit when a prior same-type success exists

(b) Success conditioned on memory state

Figure 8: Memory retrieval and success on ALFWorld seen. For the retrieval-based in-context adaptation baseline, bars show means with std error bars: (a) retrieval of an available prior same-type success, and (b) success by memory availability and retrieval.

We analyze a retrieval-based in-context adaptation baseline on three 140-task ALFWorld seen streams per scale. Before task ii, let Ei=1E_{i}=1 when memory holds an item from an earlier verified success of the same task type, and let Ri(k)=1R_{i}^{(k)}=1 when at least one such item appears among the top-kk retrieved memories. Fig. 8(a) reports the retrieval hit rate Pr⁡(Ri(k)=1∣Ei=1)\Pr(R_{i}^{(k)}=1\mid E_{i}=1). Fig. 8(b) splits episode success by whether a prior same-type success is absent (Ei=0E_{i}=0), available and missed by retrieval (Ei=1,Ri(5)=0E_{i}=1,R_{i}^{(5)}=0), or retrieved (Ri(5)=1R_{i}^{(5)}=1). Logged and reconstructed retrieval counts agree for all 840 episodes.

Retrieval often misses available experience and does not ensure success. As shown in Fig. 8(a,b), the top five results often miss an available prior same-type success, and retrieval is associated with higher success. Retrieved episodes can still fail. Same task type is only a proxy for relevance, so a retrieved item may not supply the instruction the current task needs. A failure after retrieval therefore cannot by itself show that the agent failed to execute useful guidance. The case study in Sec. 4.3 (Fig. 4) examines concrete traces behind these aggregate patterns. The paragraphs below record the provenance of each trace.

Case-study protocol. The task in Fig. 4 is ALFWorld seen-stream episode game_idx=135: “put a cool egg in microwave.” Each method uses Qwen3.5-4B and has one scored attempt with a shared 50-turn budget. Online methods follow the same task order, whereas offline in-context adaptation variants use fixed preconstructed banks. We keep these scored outcomes separate from the later trace diagnostics shown in the figure.

Prior in-context experience. Online MemP acquires a cool-then-place workflow at episode 138, so this workflow is not yet available at episode 135. The retrieval baseline analyzed above has 408 memories in its post-episode bank. Restricting the bank to entries with src_idx<135 leaves 405 memories, and retrieval from this pre-task bank returns the contradictory top item shown in Fig. 4.

Prior parametric updates. Before episode 135, ASCENT updates on two verifier-accepted episodes from the pick_cool_then_place_in_recep family: episode 98 (cool a tomato and place it in a garbage can) and episode 100 (cool a plate and place it in a cabinet). The fast weights entering episode 135 may reflect these updates, while episode 135 itself is scored before its own trajectory can update them.

Experimental support, please [view the build logs](./2610.05303v1/__stdout.txt) for errors. Generated by [ L A T E xml ![\[LOGO\]](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==) ](https://math.nist.gov/~BMiller/LaTeXML/). 

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
