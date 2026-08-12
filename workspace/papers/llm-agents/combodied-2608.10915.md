# COMBODIED AGENTS: A NEW PARADIGM OF HUMAN-CENTRIC AGENTIC AI

## A PREPRINT

Qianggang Ding<sup>1,2,\*,†</sup>, Xingyao Wang<sup>3,\*</sup>, Rui Feng<sup>4</sup>, Zhibin Wang<sup>5</sup>, Feixiang Wang<sup>5</sup>, Kelong Mao<sup>6</sup>, Hao Sun<sup>7</sup>, Zhiyao Luo<sup>8</sup>, Jiankai Tang<sup>9</sup>, Lei Li<sup>10</sup>, Jiadong Guo<sup>11</sup>, Minheng Ni<sup>12</sup>, Weicong Lin<sup>13</sup>, Chenxi Yang<sup>14</sup>, Hongxiang Gao<sup>14</sup>, Zhenghua Chen<sup>15</sup>, Yang Bai<sup>3</sup>, Min Wu<sup>3</sup>, Jun Cheng<sup>3</sup>, Huazhu Fu<sup>3</sup>, Dacheng Tao<sup>16</sup>, Bang Liu<sup>1,2,†</sup>

<sup>1</sup>Universite de Montr´ eal, Canada;´ <sup>2</sup>Mila – Quebec Artificial Intelligence Institute, Canada; <sup>3</sup>Institute of Advanced Intelligence and Computing (IAIC), A\*STAR, Singapore; <sup>4</sup>Nanjing Medical University, China; <sup>5</sup>Nanjing University, China; <sup>6</sup>Renmin University of China, China; <sup>7</sup>University of Cambridge, United Kingdom; <sup>8</sup>University of Oxford, United Kingdom; <sup>9</sup>Tsinghua University, China; <sup>10</sup>National University of Singapore, Singapore; <sup>11</sup>The Hong Kong   
University of Science and Technology, Hong Kong SAR, China; <sup>12</sup>The Hong Kong Polytechnic University, Hong Kong   
SAR, China; <sup>13</sup>Southern University of Science and Technology, China; <sup>14</sup>Southeast University, China; <sup>15</sup>University of Glasgow, United Kingdom; <sup>16</sup>Nanyang Technological University, Singapore

These authors contributed equally to this work.

<sup>†</sup>Corresponding authors: Qianggang Ding (qianggang.ding@umontreal.ca) and Bang Liu (bang.liu@umontreal.ca)

## ABSTRACT

After an older adult misses a medication dose, a software agent can issue another reminder and an embodied agent can bring the medication. Yet neither action explains whether the person forgot, is confused, is experiencing side effects, or has deliberately refused the medication, nor does it determine what support would be appropriate. This ambiguity exposes a structural gap in Agentic AI. Digital Agents are primarily organized around transformations of software states, while Embodied Agents are organized around transformations of physical states. Neither paradigm makes the evolving state and agency of a person the primary object of modeling, intervention, and evaluation. We introduce Combodied Agents as a human-centered paradigm of Agentic AI that perceives, models, predicts, and supports individual human-state trajectories over time. Software tools, sensors, wearables, robots, and human services serve as action channels rather than final objectives. Although personal assistants, health agents, AI companions, and adaptive human–AI systems instantiate parts of this vision, their capabilities remain fragmented across memory, personalization, sensing, companionship, and domain-specific support. We synthesize these capabilities into a unified closed loop. Event-based multimodal perception reconstructs evidence about meaningful personal events; longitudinal and correctable memory provides temporal context; Personal World Models transform longitudinal event evidence and current context into calibrated distributions over future personal states, observable events, and outcomes under alternative decisions and interventions; and an admissible intervention policy selects proportionate support under constraints of consent, uncertainty, safety, reversibility, and user control. Feedback from the person and environment updates subsequent perception, memory, prediction, and intervention. Rather than requiring an exhaustive Human Digital Twin, this framework uses purpose-bounded, uncertainty-aware, and user-correctable representations of the person. We further organize the design space through human-state targets, relational contexts, and agent roles; examine the transition toward edge-native personal models; and propose scenario-centered evaluation, agency-preservation metrics, benchmark requirements, and governance directions. By shifting Agentic AI from external task completion toward sustained human benefit, Combodied Agents aim to improve health, learning, judgment, capability, relationships, and goal pursuit without treating engagement, dependence, or maximum automation as measures of success.

## 1 Introduction

Artificial intelligence (AI) has historically been developed and evaluated primarily for task solving. Earlier taskoriented systems interpreted requests, tracked bounded states, queried domain-specific backends, and returned answers or actions [1]. Large language models (LLMs) and multimodal foundation models have expanded this pattern into general-purpose agents that reason over context, decompose goals, call tools, maintain memory, execute multi-step plans, and adapt from feedback [2, 3]. The organizing question has shifted from whether AI can answer to how much of an extended task it can complete on its own.

This trend is visible across the two dominant branches of Agentic AI. Digital Agents navigate interfaces, modify code, invoke application programming interfaces (APIs), and coordinate digital workflows [4, 5, 6]. Embodied Agents connect language and multimodal perception to navigation, manipulation, and physical control [7, 8, 9]. In both branches, progress means longer task horizons, more reliable action, and less human supervision [10, 11]. These advances reduce costs and automate difficult work, but they also normalize an incomplete account of progress: the more work that disappears from the human side, the more capable the agent appears [12, 13].

What this account misses is that task performance and human development can diverge. AI may improve a document while leaving its author with less understanding, or accelerate a decision while weakening the user’s ability to judge its basis. Studies of AI-assisted knowledge work and decision making likewise report reduced cognitive effort and overreliance when people cannot determine when to accept, verify, or reject model outputs [14, 15]. Task-success metrics capture immediate output, but not the capability or judgment that repeated use leaves with the person.

The concern also has a societal dimension. Human knowledge is often local, tacit, and renewed through participation. Substituting standardized model outputs for those practices may weaken expertise, variety, and resilience. Values create a parallel problem: general-purpose models are developed and aligned in relatively few institutions, whereas users differ in culture, commitments, relationships, and visions of a good life. Concentrating intelligence while weakening people’s productive and epistemic agency can therefore concentrate the authority to define both goals and acceptable values [13].

Personal agents make these tensions especially visible. Memory-enabled assistants, AI companions, health agents, and behavioral coaches increasingly participate in identity, emotion, habits, and care. They can provide continuity and meaningful support, but sustained interaction may also foster dependence, manipulation, social displacement, or inappropriate influence, especially among minors and vulnerable users [16, 17, 18, 19, 20]. Automation itself is not the problem; the problem is making substitution the default regardless of its long-term effects on understanding, autonomy, relationships, values, and future capability.

We therefore argue for building AI for humans: AI that extends human will and judgment and makes people more capable over time. Human-centered AI has long held that intelligent systems should expand what people can safely and meaningfully do while remaining reliable, understandable, and controllable [21, 22]. Recent work on Human-Centered Agents makes this principle operational by treating what users retain and develop—understanding, competence, agency, identity, and calibrated reliance—as an outcome alongside task completion [12]. The optimization target thus expands from isolated task success to immediate benefit plus longitudinal human gain.

AI for humans does not require people in every low-level loop or resist automation for its own sake. Human participation instead becomes a technical design problem: an agent should learn when to act independently, request oversight, or return knowledge and capability to the person [13]. Routine, low-risk, reversible tasks may justify near-complete delegation; learning, health, emotional support, and high-stakes decisions may require explanation, scaffolding, consent, or human escalation. The division of labor should fit the person’s goals, state, competence, relationships, and risk while preserving the ability to choose, refuse, correct, and recover.

This commitment motivates Combodied Agents: agents organized around beneficial trajectories of human state and agency. Digital tools, wearables, robots, and human services may all serve as action channels, but success is determined by what the person can understand, decide, do, and sustain over time. Section 2.2 develops the distinction from Digital and Embodied Agents.

Combodied Agents share with Human Digital Twins (HDTs) an individual-centered and longitudinal perspective, but the two paradigms define different system boundaries. An HDT aims to maintain a dynamically updated digital representation of a human, or of a use-case-specific aspect of a human, through ongoing data synchronization and feedback. Such a representation may integrate multimodal and multi-scale data to support monitoring, simulation, prediction, and optimization [23, 24, 25].

A holistic, high-fidelity twin of a complete person remains an aspiration rather than a currently attainable system. Human physiology, cognition, behavior, relationships, and context evolve across interacting time scales, while sensing remains incomplete and model validation, synchronization, computation, privacy, and governance remain substantial challenges [24, 26, 27]. Combodied Agents therefore do not require an exhaustive digital replica of the person. They maintain purpose-bounded, uncertainty-aware, and user-correctable representations of the aspects needed for an agreed support context, and connect those representations to longitudinal memory, goal negotiation, intervention policies, and feedback. Their primary objective is not maximal representational fidelity, but safe and beneficial participation across the person’s evolving contexts while preserving human agency. A domain-specific HDT may supply evidence or predictions to a Combodied Agent, but it does not by itself define the agent’s memory, authority, interaction policy, or relationship with the person.

Personal assistants, health agents, AI companions, behavioral coaches, and adaptive human–AI systems already instantiate parts of this vision. What remains missing is a unified paradigm organized around longitudinal human-state trajectories, intervention responses, an adaptive division of labor, and evaluation of both benefit and preserved agency. Combodied Agents translate AI for humans into this technical research program.

Table 1 makes this need concrete. Existing agent categories contribute important but partial capabilities: task-oriented agents provide execution, embodied agents provide sensing and actuation, memory and personalized agents provide continuity, and health, learning, companion, and care agents provide domain-specific support. However, no category integrates these capabilities into a longitudinal loop whose primary state is the person and whose success criterion includes capability, autonomy, wellbeing, relationship safety, and agency preservation.

Table 1: Consolidated related-agent landscape: representative capabilities, limitations, and the humancentered longitudinal support added by Combodied Agents.
<table><tr><td>Agent Type</td><td>Capabilities</td><td>Limitations</td><td>What Combodied Agents Add</td></tr><tr><td>Dialogue Agents</td><td>• Intent recognition • Dialogue-state tracking • Backend querying • Bounded task completion</td><td>• Episodic rather than longitudinal • Restricted to narrow domains • Evaluated mainly by task success [1]</td><td>• Longitudinal support loop • Memory and goal updates • Boundary-aware future intervention</td></tr><tr><td>Tool-Use Agents</td><td>• Goal decomposition • Contextual reasoning • Tool and API use • Feedback-based planning</td><td>• Limited modeling of evolving user states • Prioritize task completion over user change • Lack intervention-response tracking • Weak modeling of user agency • Limited safety reasoning</td><td>• Tool use as support channel • Human-state trajectory tracking • Outcome, autonomy, and safety focus</td></tr><tr><td>Computer-Use Agents</td><td>• Screen understanding • Interface operation • Form filling • Workflow execution</td><td>beyond tool use [2, 3, 11] • May automate without human-state awareness • Insensitive to fatigue, stress, or overload • Weak alignment with user values and obligations • Limited assessment of</td><td>• Personal-context-aware execution • Calibrated automation and confirmation • Agency and safety checks</td></tr><tr><td>Workflow Agents</td><td>• Code editing • Data analysis • File manipulation • Process automation • Enterprise coordination</td><td>agency loss [4, 5, 28, 29] • May optimize efficiency at the user&#x27;s expense • Limited attention to cognitive load • Weak support for responsibility boundaries • Insufficient safeguards for autonomy and control [6, 30, 3i]</td><td>• Person-aware planning • Cognitive-load and boundary checks • Reversible, privacy-preserving control</td></tr></table>

Continued on next page

Table 1: Consolidated related-agent landscape: representative capabilities, limitations, and the humancentered longitudinal support added by Combodied Agents.
<table><tr><td>Agent Type</td><td>Capabilities</td><td>Limitations</td><td>What Combodied Agents Add</td></tr><tr><td>Embodied Agents</td><td>• Physical perception • Navigation • Object manipulation • Robotic control • Safety-aware action</td><td>• Limited access to internal human states • Weak modeling of life history and relationships • Insufficient awareness of vulnerabilities • Limited support for</td><td>• Human-agency support target • Person-centered use of sensors • Safe, context-aware intervention</td></tr><tr><td>Memory Assistants</td><td>• User-fact storage • Preference memory • Conversation recall • Cross-session continuity</td><td>longitudinal wellbeing [8,9,32] • Store facts without modeling trajectories • Lack structured intervention history • Weak outcome-based memory organization • Limited forgetting and correction mechanisms</td><td>• Longitudinal life memory • Events, routines, goals, and relationships • Corrections, outcomes, and forgetting</td></tr><tr><td>Personalized Agents</td><td>• User profiling • Preference adaptation • Contextual recommendation • Personalized planning</td><td>[33, 34, 35] • Often rely on shallow personalization • May optimize engagement rather than wellbeing • Lack causal models of user change • Limited user-contestable</td><td>• Personal World Models • Action and context response prediction Correctable, safety-aware adaptation</td></tr><tr><td>Companion Agents</td><td>• Social presence • Emotional continuity • Empathic response • Rapport building • Self-disclosure support</td><td>adaptation [36, 37, 38] • Risk emotional dependency • Blur role and relationship boundaries • May reinforce sycophancy or social substitution • Lack robust attachment-safety</td><td>• Relationship-safety architecture • Scoped memory and role boundaries • Dependency detection and escalation</td></tr><tr><td>Health Agents</td><td>• Symptom tracking • Behavior change • Adherence support • Emotional coping • Chronic-care routines</td><td>mechanisms [39, 40, 41] • Often narrow in health scope • Lack causal intervention modeling • Weak escalation and uncertainty handling • Poor integration with broader life contexts</td><td>• Multimodal human-state perception • Personal baselines and response memory • Clinical and personal boundary safeguards</td></tr><tr><td>Learning Agents</td><td>• Tutoring • Explanation • Assessment • Learning scaffolding • Material adaptation</td><td>[42, 43, 44] • May encourage over-reliance • Risk weakening independent reasoning • Often emphasize short-term performance • Limited evaluation of capability growth [45]</td><td>• Capability-growth evaluation • Metacognition and self-efficacy support • Retention and independent judgment</td></tr></table>

Continued on next page

Table 1: Consolidated related-agent landscape: representative capabilities, limitations, and the humancentered longitudinal support added by Combodied Agents.
<table><tr><td>Agent Type</td><td>Capabilities</td><td>Limitations</td><td>What Combodied Agents Add</td></tr><tr><td>Assistive Care Agents</td><td>• Reminders • Monitoring • Companionship • Family communication • Care coordination</td><td>• Risk excessive surveillance • Create autonomy-safety tensions • Complicate consent and caregiver access • Limited support for dignity-preserving</td><td>• Relation-aware memory • Proportional intervention and escalation • Contestable dignity-preserving support</td></tr><tr><td>Edge AI Agents</td><td>• On-device inference • Local sensing • Private memory • Low-latency personalization • Local control</td><td>intervention [46, 47] • Privacy gains do not ensure human-centered design • Memory and sharing boundaries remain underspecified • Lack rich human-state models • Weak agency-centered optimization</td><td>• Edge-native personal intelligence • User-owned memory and PWMs • Local policies with selective cloud use</td></tr></table>

Across the table, three recurring limitations motivate Combodied Agents: existing systems optimize external task or domain outcomes rather than human trajectories; their models of the person are fragmented, shallow, or episodic; and their safety and evaluation criteria rarely measure agency loss, capability growth, relationship effects, or long-term wellbeing. Combodied Agents address these limitations not by simply aggregating more features, but by reorganizing perception, memory, prediction, intervention, and evaluation around the evolving human subject. They add multimodal human-state perception, longitudinal and correctable memory, personal world models (PWMs) of intervention response, calibrated and reversible support policies, and explicit evaluation of human gain and agency preservation.

## This paper makes four contributions:

1. We introduce and formally define Combodied Agents as a human-centric paradigm that complements Digital and Embodied Agents by making the evolving human subject and human agency its primary action target.

2. We develop a closed-loop technical framework connecting multimodal Human State Perception, Longitudinal Memory, PWMs, agency-preserving Intervention Policies, feedback adaptation, and safety constraints.

3. We synthesize the design space through deployment architecture, human-centered evaluation, and an integrated taxonomy of relationship modes, agent roles, human-state targets, and applications.

4. We examine the risks, governance requirements, open challenges, and adjacent research fields that shape a responsible research agenda for Combodied Agents.

The remainder of this paper defines Combodied Agents and their closed-loop framework, then examines event-based multimodal perception, PWMs, and cloud-to-edge personal models. It next develops benchmarks, evaluation, taxonomy, applications, risks, governance, and future directions before concluding with the human-agency principle guiding the paradigm.

## 2 Foundations of Combodied Agents

This section gives the canonical definition of Combodied Agents, distinguishes their action substrate from those of Digital and Embodied Agents, and formalizes the resulting closed loop.

## 2.1 Formal Definition and Paradigm Scope

We formally define the concept of Combodied Agents and their core properties as the following:

## Definition 1: Combodied Agents

Combodied Agents are human-centered intelligent agents that perceive, model, and influence the evolving state of a person through continuous multimodal sensing and longitudinal interaction. The term Combodied combines Companion and Body, but does not imply a third-party companion whose primary role is conversation or emotional companionship. Instead, a Combodied Agent treats the human body, behavior, cognition, emotion, and surrounding context as its primary perceptual and action domain. By integrating intelligent sensors, longitudinal memory, personalized human-state models, and intervention policies, it continuously observes and understands the individual, anticipates needs or risks, and delivers timely, adaptive, and consent-aware interventions. Its defining characteristic is therefore not merely interaction with humans, but perception of humans, understanding of humans, and action on human states.

![](images/de468a7aa5adfc27c3b0f1f7b312f7196b0b98287c175363f4fca1e5f8ff63f8.jpg)

The definition establishes five jointly important properties:

• Human-centric state modeling: the person, rather than an external task alone, is the primary object of modeling and support.

• Longitudinality: the agent reasons across repeated interactions, evolving life contexts, and extended time horizons.

• Intervention: the agent may remind, recommend, explain, coach, coordinate, protect, execute, or escalate rather than merely produce responses.

• Co-agency: the agent works with the human and adapts the division of labor instead of treating human replacement as the default.

• Agency preservation: support must preserve or strengthen autonomy, control, dignity, relationships, and long-term human capability.

These properties define a center of gravity rather than a rigid interface category. A chatbot is not automatically a Combodied Agent because it remembers a name, and a wearable system is not automatically one because it measures a physiological signal. Conversely, a Combodied Agent may act through conversation, software tools, sensors, robots, or human caregivers. What matters is whether these capabilities are integrated around an ongoing, correctable mode of the person and used to produce safe, longitudinally beneficial support.

## 2.2 Distinguishing Focus: Three Action Substrates

Agentic AI broadly refers to systems that iteratively interpret observations, maintain task-relevant internal states, pursue goals, select and execute actions, and use resulting feedback to guide subsequent behavior. Although particular architectures may not represent goals or plans explicitly, a conceptual agent–environment loop can be expressed as

$$
o _ { t }  b _ { t }  ( g _ { t } , p _ { t } )  a _ { t }  x _ { t + 1 }  o _ { t + 1 } ,\tag{1}
$$

where $o _ { t }$ denotes the agent’s observation at time $t , b _ { t }$ its internal belief or task-relevant state, $g _ { t }$ its current $\mathrm { g o a l } , p _ { t }$ an optional plan, $a _ { t }$ the selected action, and $x _ { t + 1 }$ the resulting state of the relevant external environment or target domain. The next observation $o _ { t + 1 }$ is generated from this updated state. Feedback may update the agent’s internal state, revise its plan, or change its subsequent actions, without necessarily implying continual learning or model-parameter updates.

![](images/871f84e48b781be7c8234c9ca885f9be08d3c0652ebf2f1cbc50a643d9cfe030.jpg)  
Figure 1: Action substrates and representative task configurations in Agentic AI. (a) Representative substrate-dominant and cross-substrate tasks for Digital, Embodied, and Combodied Agents. The examples are illustrative rather than exhaustive, and their placement indicates which target states and evaluation objectives are jointly involved. (b) The three overlapping centers of gravity are defined by the states that primarily organize modeling, action, and evaluation: digital states and artifacts, physical or simulated physical states, and evolving human states and agency. Pairwise overlaps denote systems that couple digital planning with physical execution, digital support with longitudinal person modeling, or physical assistance with adaptation to human outcomes; the central overlap integrates all three. All three paradigms draw on shared agentic machinery for perception, representation, reasoning, planning, action, learning, and evaluation.

One useful and complementary way to distinguish agent paradigms is by the class of target states that primarily organizes their modeling, decision making, intervention, and evaluation. We call this class the agent’s action substrate. The action substrate is not necessarily identical to the agent’s interface or immediate actuation channel: an agent may act through software tools, sensors, robots, or communication while ultimately seeking to support a different target-state trajectory. This perspective complements, rather than replaces, taxonomies based on model architecture, autonomy, embodiment, tools, memory, or perceptual modality.

For the purposes of this paper, we distinguish three overlapping and non-exhaustive centers of gravity. Digital Agents are primarily organized around transformations of digital states; Embodied Agents around transformations of physical or simulated physical states; and Combodied Agents around modeling and supporting the evolving human state while preserving or strengthening human agency over time. Figure 1 summarizes these three centers of gravity.

The overlaps in Figure 1 represent cross-substrate task configurations rather than uncertainty about category boundaries. Autonomous laboratories, for example, couple digital planning with physical execution; longitudinal memory support combines digital action with person modeling; assistive habilitation combines physical intervention with adaptation to human outcomes; and robot-assisted medication support may integrate all three substrates. Such intersections are expected because research programs frequently extend beyond their historical centers of gravity by incorporating adjacent capabilities, action channels, and evaluation objectives. A system may operate across multiple platforms or embodiments while still pursuing one primary outcome for the user. Agents should therefore be classified by the target user state and success criteria that guide their behavior, rather than by their interface, underlying model, or mode of action.

Digital Agents operate primarily on digital states. Browser and graphical user interface (GUI) agents navigate interfaces; coding agents modify repositories; tool-use, data-analysis, and workflow agents invoke APIs and transform digital artifacts [52, 4, 5, 6, 53, 30]. Their center of gravity is an externally represented task state, and they are usually evaluated by correctness, completion, efficiency, policy compliance, and robustness in digital environments.

Embodied Agents operate primarily on physical or simulated physical states. Robotic agents, autonomous vehicles, household robots, industrial systems, and navigation agents connect perception, planning, and control to movement and manipulation [8, 9, 54, 55, 56, 57]. Their center of gravity is the physical environment, and evaluation focuses on task success, safety, control reliability, and generalization across bodies and environments.

![](images/b0a4cd5c7dd8b304d8799446f84cc2211c0e912941f3307f7a3d811cb1105b65.jpg)  
Figure 2: Functional organization of a Combodied Agent. Multimodal observations support human-state estimation; longitudinal memory supplies relevant temporal context; a personal world model predicts state–event–outcome trajectories under alternative decisions, interventions, and contextual changes; and an intervention policy selects and delivers appropriate support. Feedback from the person and environment closes the loop. Safety, uncertainty, consent, and user control constrain every stage.

These categories are indispensable but incomplete for systems whose principal concern is the evolving human subject. A personal agent may use a browser to schedule care or an embodied sensor to detect fatigue, yet neither the webpage nor the sensorimotor state is its ultimate target. The relevant outcome is whether the intervention helps the person regulate health, understand a decision, sustain a relationship, develop a capability, or pursue a valued goal. Combodied Agents name this third center of gravity: human agency and its physiological, cognitive, behavioral, emotional, social, and goal-directed conditions over time. The three paradigms therefore overlap in implementation while differing in what their actions are ultimately for.

## 2.3 High-Level Core Capabilities

The definition above implies four connected capabilities, summarized in Figure 2.

1. Human-State Perception estimates current personal state from incomplete, noisy, and multimodal evidence.

2. Longitudinal Memory organizes events, goals, relationships, interventions, outcomes, and corrections across time.

3. Personal World Modeling predicts person-specific state–event–outcome trajectories under alternative user actions, agent interventions, and contextual changes.

4. Intervention Planning and Delivery selects whether, when, and how to provide proportionate support or human escalation.

These are functional components rather than additional defining properties. Human-centricity and longitudinality determine what they model; co-agency and agency preservation constrain the complete loop.

Table 2: Main components of Longitudinal Memory in Combodied Agents.
<table><tr><td>Memory Component</td><td>Function</td></tr><tr><td>Episodic memory</td><td>Records concrete events, interactions, and user experiences.</td></tr><tr><td>Semantic person memory</td><td>Stores relatively stable facts, preferences, values, and boundaries.</td></tr><tr><td>Trajectory memory</td><td>Tracks changes in health, behavior, emotion, cognition, and routines.</td></tr><tr><td>Goal and commitment memory</td><td>Maintains user goals, plans, obligations, and promises.</td></tr><tr><td>Relationship memory</td><td>Represents relevant family, caregiver, clinician, work, or social ties.</td></tr><tr><td>Intervention-response memory</td><td>Links reminders, recommendations, coaching, or escalation to acceptance, rejection, benefit, or harm.</td></tr><tr><td>User-control memory</td><td>Stores restrictions, do-not-remember rules, corrections, and deletion re- quests.</td></tr></table>

## 2.4 Closed-Loop Architecture and Formalization

The following formalization specifies the information flow among human-state inference, longitudinal memory, prediction, intervention, and feedback.

Latent Human State and Internal Representation. Let $H _ { t }$ denote the unobservable human state at time $t ,$ and let $D _ { \leq t }$ contain the event-evidence records reconstructed from observations up to t. The agent maintains an uncertaintybearing posterior representation and retrieves decision-relevant evidence:

$$
Z _ { t } \sim q _ { \phi } ( { \cdot } \mid D _ { \leq t } , M _ { t } , C _ { t } ) ,\tag{2}
$$

$$
R _ { t } = \operatorname { R e a d } \left( M _ { t } ; Z _ { t } , C _ { t } , G _ { t } \right) .\tag{3}
$$

Here $C _ { t }$ is current context, M longitudinal personal memory, $G _ { t }$ the user’s current goals, and $R _ { t }$ the retrieved evidence relevant to the decision. Section 3 defines the event-evidence interface, and Section 4 defines action-conditioned trajectory prediction from this state.

Prediction and Intervention Selection. The PWM predicts multidimensional future outcomes under candidate interventions, and the policy chooses only among actions admissible under consent, stated boundaries, safety constraints, uncertainty thresholds, reversibility, and escalation requirements. Section 4 gives the single canonical prediction and selection formulation; its admissible set includes non-intervention, clarification, confirmation, and referral.

Human-State Transition and Feedback. The subsequent human state is not determined by the agent’s action alone. It also depends on the person’s own actions, contextual changes, and exogenous influences:

$$
H _ { t + 1 } \sim T _ { H } \left( \cdot \mid H _ { t } , a _ { t } ^ { \mathrm { a g e n t , * } } , a _ { t } ^ { \mathrm { u s e r } } , \Xi _ { t } \right) ,\tag{4}
$$

$$
O _ { t + 1 } \sim \Omega \left( \cdot \mid H _ { t + 1 } , C _ { t + 1 } \right) ,\tag{5}
$$

$$
M _ { t + 1 } = \mathrm { U p d a t e } \left( M _ { t } , O _ { t + 1 } , a _ { t } ^ { \mathrm { a g e n t , * } } , Y _ { t + 1 } , F _ { t + 1 } \right) .\tag{6}
$$

Here $T _ { H }$ denotes the human-state transition process, $a _ { t } ^ { \mathrm { u s e r } }$ the user’s own actions, $\Xi _ { t }$ exogenous influences, Ω the observation process, and $F _ { t + 1 }$ explicit or implicit feedback. The new evidence updates memory and may subsequently revise the state estimator, PWM, or Intervention Policy. This closes the longitudinal loop without assuming that an intervention has a deterministic or immediately observable effect on the person.

Longitudinal Memory as Temporal Evidence. Longitudinal Memory records temporally extended evidence about states, experiences, goals, preferences, relationships, interventions, and outcomes. Each memory item may include content, timestamp, source, modality, confidence, sensitivity, relevant goals, intervention and outcome links, and a retention policy. Table 2 summarizes the principal functions.

Memory writing, consolidation, retrieval, and correction must preserve provenance and uncertainty and remain inspectable and correctable by the user [58, 34, 59]. In particular, intervention-response memory supplies evidence for adapting future support rather than merely repeating a previously generated action.

Table 3: Combodied Agent action space.
<table><tr><td>Action Type</td><td>Primary Target</td><td>Examples</td></tr><tr><td>Inform</td><td>cognition</td><td>explain, summarize, educate, interpret data</td></tr><tr><td>Remind</td><td>attention and execution</td><td>medication reminder, meeting reminder, rest reminder</td></tr><tr><td>Recommend</td><td>decision-making</td><td>suggest options, resources, plans, or next steps</td></tr><tr><td>Coach</td><td>capability and habit</td><td>train, give feedback, review progress, encourage</td></tr><tr><td>Nudge</td><td>behavioral tendency</td><td>gently steer toward healthier or safer choices</td></tr><tr><td>Reflect</td><td>self-understanding</td><td>help review emotions, goals, patterns, and trade-offs</td></tr><tr><td>Coordinate</td><td>social and institutional context</td><td>contact caregivers, clinicians, colleagues, or services</td></tr><tr><td>Protect Escalate</td><td>safety and autonomy</td><td>detect scams, warn about risk, block unsafe actions</td></tr><tr><td>Execute</td><td>high-risk states</td><td>refer to human expert, clinician, caregiver, or emergency support</td></tr><tr><td></td><td>delegated external action</td><td>book appointments, send messages, operate software tools</td></tr></table>

Intervention Semantics. The policy operates over actions that differ in target, intensity, reversibility, and required authority. Table 3 summarizes this action space; its entries are possible support modes rather than an assumption that intervention is always warranted.

The policy should not always intervene. Depending on uncertainty, consent, and risk, the appropriate action may be to remain silent, request clarification or confirmation, reduce intervention intensity, or escalate to an authorized human expert. Safety and human control therefore constrain the entire loop: perception should avoid unsupported inference; memory should support inspection, correction, and deletion; action selection should enforce proportionality and reversibility; and adaptation should not optimize engagement or dependence at the expense of autonomy and wellbeing [16, 19]. Deployment implications for keeping these sensitive computations under user control are developed separately in Section 5.

## 3 Event-Based Multimodal Perception

Combodied Agents require multimodal perception [60] because human agency is rarely visible in a single utterance, record, or sensor reading. The challenge is not to accumulate as much personal data as possible, but to recover useful fragments from long-term, intermittent, noisy, and unevenly sampled data; identify events that matter; and retain enough evidence to explain why an event is relevant to the user’s state, goals, safety, or future support. A late-night message, a missed medication reminder, a short walk, an unusual pause in speech, or a caregiver note may be weak evidence on its own. When placed in personal and temporal context, however, several such fragments may form a meaningful account of change.

We define multimodal perception in Combodied Agents as event-based personal data perception: the acquisition, filtering, alignment, and interpretation of fragmentary personal data for human-state understanding and longitudinal modeling. Its output is not an unrestricted stream of raw data. Observations are first checked for quality and segmented into candidate events; only then can they become event-evidence records describing what was observed, when and how it was acquired, which interpretation it supports, what uncertainty remains, and whether it is relevant enough to influence memory or intervention.

Acquisition is part of this definition because the same nominal modality can have different meanings depending on how it was obtained. A user may actively describe an experience, respond to a context-triggered prompt, carry a sensing device, wear a sensor in contact with the body, enter an instrumented environment, or authorize access to an institutional record. We refer to these configurations as user-reported, device-mediated, on-body or contact, ambient or contactless, and institutional acquisition. Their sampling patterns range from continuous and periodic sensing to episodic, event-triggered, context-triggered, user-initiated, and query-on-demand collection. These distinctions affect coverage, burden, privacy, and evidential strength, and should remain visible downstream.

Personal data are also longitudinal but discontinuous. A companion may receive only sparse glimpses of a routine or condition, separated by hours, days, or weeks. Perception must therefore identify transitions, deviations, social episodes, safety-relevant events, and responses to previous interventions rather than classify isolated samples. At the same time, inferred states must remain linked to observable fragments. Multimodal evidence can make an interpretation more plausible, but does not turn it into an uncontestable fact about the person.

## 3.1 Language and Textual Signals

Language is the most direct channel through which users can state goals, preferences, commitments, emotions, experiences, and boundaries. Relevant evidence may appear in conversations with the agent, messages, diaries, ecological momentary assessments, questionnaires, notes, task histories, or corrections to an earlier interpretation. Some of these data are deliberately authored for the agent; others are created for a different purpose and become available through an authorized application interface or user upload. Text may also be dictated and transcribed or extracted from documents, but its acquisition history should remain visible: a spontaneous statement, an answer to a prompt, and an automatically imported note do not carry the same evidential meaning.

For longitudinal support, the important textual events are usually changes rather than isolated facts. A user may formulate a new goal, revise a preference, withdraw consent, report an unsuccessful intervention, or correct something that the agent previously stored. Experience-sampling research shows the value of collecting reports close to the moment of experience [61], while systems such as MemoryBank and benchmarks such as LongMemEval investigate continuity across extended interaction histories [58, 62]. A Combodied Agent adds a governance requirement to this line of work: the person must be able to distinguish what they explicitly said from what the system inferred, and later corrections should update the personal model without erasing the provenance of the original claim.

Text is explicit but not necessarily objective. Reports may be incomplete, socially desirable, emotionally amplified, retrospectively distorted, or shaped by the agent’s question. Language is therefore particularly valuable for representing user-authorized goals and boundaries, but it remains one source of evidence rather than a complete account of behavior or wellbeing.

## 3.2 Speech and Audio Signals

Text records what is said but removes much of how it is said. Speech restores temporal and paralinguistic information through pitch, intensity, rhythm, pauses, hesitation, articulation, fluency, and voice quality. The surrounding audio may also contain events that are not linguistic, including alarms, coughing, crying, collisions, or changes in the sound environment.

These signals reach an agent through technically and socially different configurations. Near-field microphones in phones, computers, earbuds, and hearing devices mainly capture deliberate interaction. Contact or throat microphones reduce some environmental interference but require the device to be worn. Far-field microphone arrays in rooms, vehicles, or smart speakers provide broader coverage, while also increasing the likelihood of recording bystanders and activities unrelated to the support purpose. Conversation-bound or voice-triggered acquisition is therefore materially different from continuous ambient listening, even when both ultimately produce an audio stream.

Speech research has traditionally separated voice activity detection, speaker diarization, speech recognition, acoustic feature extraction, and environmental sound detection. GeMAPS and eGeMAPS were introduced to make acoustic analysis more reproducible [63], while MERBench places speech alongside linguistic and visual evidence in multimodal emotion recognition [64]. These resources provide useful components, but longitudinal companionship introduces a different question: whether an observed change is unusual for this particular speaker and persists long enough to constitute a relevant event. Repeated hesitation during a familiar task may be informative; one pause in a noisy conversation is usually not.

Acoustic features remain indirect evidence. Pauses may result from reflection, distraction, fatigue, network delay, or cognitive difficulty, and voice characteristics vary with language, health, device, and environment. Local feature extraction, short-lived audio buffers, explicit activation indicators, and reliable speaker attribution are consequently more appropriate than retaining raw recordings by default.

## 3.3 Vision-Based Sensing

Visual sensing broadens perception from the interaction channel to visible behavior and surrounding activity. Devicefacing cameras on phones and computers capture expressions, gaze, and posture during direct interaction. Wearable cameras or smart glasses provide a first-person view of activities and object use, while fixed cameras offer a wider view of rooms, movement, and multi-person events. Depth, infrared, thermal, eye-tracking, and event-based sensors extend these arrangements when illumination, distance, temporal resolution, or particular privacy constraints matter.

The unit of interest is rarely an isolated frame. A useful system needs to recognize episodes such as beginning a meal, taking medication, leaving home, completing an exercise, interacting with another person, or changing behavior after an intervention. OpenFace demonstrates real-time extraction of facial landmarks, head pose, facial action units, and eye gaze from conventional cameras [65]. Ego4D moves toward everyday first-person perception through large-scale egocentric video and tasks involving episodic memory, object interaction, social activity, and anticipation [66]. These lines of research suggest components for Combodied Agents, but they do not by themselves solve the longitudinal problem of deciding which visual episodes matter to a particular person.

The limits of visual interpretation are especially important. Reduced facial movement may reflect concentration, culture, disability, fatigue, lighting, or camera position rather than low affect. Occlusion, identity errors, and contextdependent gestures introduce further ambiguity. A visual event should therefore retain the acquisition viewpoint, image quality, subject-attribution confidence, and presence of other people. Where possible, raw images should be processed locally and replaced by purpose-specific event descriptions. The fact that a camera can capture a household scene does not mean that every visible person or object belongs in the user’s memory.

## 3.4 Physiological and Biochemical Signals

Physiological sensing appears closer to bodily state than language or vision, but this proximity does not eliminate ambiguity. Watches, rings, chest straps, earbuds, patches, headbands, cuffs, beds, and clinical monitors can provide heart rate, heart-rate variability, electrocardiography, photoplethysmography, blood oxygen saturation, electrodermal activity, respiration, temperature, sleep-related signals, blood pressure, and selected neurological or muscular measurements. Biochemical sensing adds continuous glucose monitors and emerging sensors for sweat, saliva, and interstitial-fluid biomarkers. Some measurements require contact or minimally invasive devices; others, including re mote photoplethysmography, thermal sensing, and radar-based respiration estimation, can be obtained without direct contact.

The acquisition configuration changes both burden and reliability. Consumer wearables provide convenient longitudinal coverage but are affected by device placement, motion, skin contact, battery state, firmware, and proprietary preprocessing. Chest straps and clinical equipment may provide higher-fidelity measurements for a narrower period. Skin patches can combine sensing functions: hybrid systems have, for example, demonstrated simultaneous ECG and sweat-lactate measurement [67]. These differences should not disappear when measurements are converted into a common numerical time series.

For a longitudinal companion, isolated thresholds are generally less useful than personal baselines, recovery trajectories, and persistent changes across days or weeks. Relevant events may include sustained sleep deterioration, an unusual resting heart rate, repeated glucose excursions, delayed recovery following activity, or disagreement between subjective symptoms and sensor measurements. Health-LLM and PHIA illustrate how language-model-based systems can reason over personal health and wearable data [68, 42]. Their results demonstrate a reasoning interface, however, rather than clinical validation of every inferred state.

Motion, medication, hydration, temperature, illness, and sensor contact can all change a physiological reading. The same elevated heart rate may be expected during exercise and concerning at rest. Physiological evidence can support awareness, self-management, and preparation for professional care, but should not be promoted to diagnostic truth without suitable calibration, clinical validation, and escalation boundaries.

## 3.5 Motion and Behavior Monitoring

Motion sensing sits between bodily measurement and situational context. Accelerometers, gyroscopes, magnetometers, and barometers on phones or wearables record movement and orientation; satellite positioning, Bluetooth, Wi-Fi, and ultra-wideband add mobility and proximity information. Homes and care environments may contribute passive infrared sensors, pressure mats, instrumented furniture, smart appliances, cameras, or millimeter-wave radar. These arrangements differ in what they observe: a wrist sensor follows part of the body, a phone is useful only while carried, and an ambient sensor observes activity within a particular space.

Human-activity-recognition research has developed methods for converting such streams into actions and routines. OPPORTUNITY combines wearable, object, and ambient sensing for activity recognition [69]; DeepConvLSTM illustrates end-to-end sequence modeling from wearable data [70]; and recent surveys synthesize multimodal wearable approaches [71]. Combodied perception extends the problem from labeling activities to detecting changes that matter over time: a fall, a missed routine, persistent inactivity, a rehabilitation exercise, a change in gait, or a behavioral response to earlier support.

These events remain ambiguous without context. Staying at home may indicate rest, illness, remote work, poor weather, or preference, while missing data may simply mean that a device was not worn. Personal baselines, wearstate detection, environmental context, and explicit reports are therefore needed before a movement pattern becomes a claim about health or motivation. Spatial and temporal precision should also be limited to what the agreed support purpose requires; routine assistance should not become unrestricted mobility tracking.

## 3.6 Social and Relational Data

Motion traces describe what a person does, but less often explain the relationships in which those activities occur. Social evidence may come from user-described relationships, calendars, shared tasks, caregiver interactions, clinician communication, collaboration systems, communication metadata, proximity measurements, conversational turntaking, or recurring group activities. Collection may be active, device-mediated, wearable, or ambient. The distinction between communication content and metadata is important: timing and duration can reveal a pattern without granting access to what was said, although even metadata may expose sensitive relationships.

Early mobile-sensing work showed how longitudinal phone and proximity data can reveal recurring social and organizational patterns [72]. StudentLife combined smartphone sensing and self-report to study changes in activity, sociability, sleep, and wellbeing in a natural setting [73]. Sotopia approaches social interaction from a different direction, evaluating role- and goal-dependent behavior in simulated situations [74]. Together, these research lines highlight different parts of the problem, but none makes relationship inference straightforward. Proximity does not necessarily imply meaningful interaction, and reduced communication does not necessarily indicate a deteriorating relationship.

The more fundamental difficulty is that relational data rarely belong to only one person. A user may authorize access to a shared calendar without authorizing the agent to infer another participant’s emotional state or store their private information. Social event records therefore need to identify the focal user, other participants, the source of the relationship claim, and the authority associated with each role. Shared memories may require participant-specific visibility and correction rather than a single permission attached to the user who operates the agent.

## 3.7 Environmental and Contextual Data

Environmental context often matters not because it is itself the target of inference, but because it changes the interpretation of other evidence. Light, noise, temperature, humidity, air quality, occupancy, room state, weather, traffic, calendar information, device state, and connectivity can explain why behavior or sensor measurements have changed. These data arrive through fixed environmental sensors, phones, vehicles, smart-home infrastructure, or external services. Some are continuously sensed, while others are retrieved only when an event needs to be interpreted.

Context-aware computing has long studied the conversion of sensor readings into descriptions of situation and activity. AWARE provides an extensible framework for mobile sensing and experience sampling [75], and systems such as StudentLife illustrate how phone sensing can be combined with longitudinal self-report outside the laboratory [73]. In a companion setting, contextual events may include arriving at work, entering a noisy environment, beginning a journey, losing connectivity, or experiencing poor air quality. Their primary value is often disambiguation: high temperature may help explain an elevated heart rate, travel may explain interrupted sleep, and a public setting may make an otherwise appropriate intervention poorly timed.

Context becomes personal when linked to identity and routine. Precise location histories can reveal home, work, health visits, religious practice, and relationships even when no explicit personal label is collected. Coarse, temporary, or event-triggered context may therefore be preferable to persistent reconstruction of the user’s movements.

## 3.8 Clinical, Institutional, and Structured Records

Structured records differ from sensor streams in both temporality and authority. Diagnoses, medication lists, laboratory results, treatment plans, appointments, educational records, workplace schedules, evaluations, financial obligations, and legal documents are usually updated episodically. They may be supplied by the user, extracted from uploaded documents, synchronized periodically, received through an institutional event feed, or retrieved through an authorized query. In healthcare, HL7 FHIR and SMART on FHIR support interoperable access to clinical resources and application context [76], while common data models such as OMOP help normalize heterogeneous databases [77].

These records can anchor events that are difficult to infer from sensing alone: a medication change, a new diagnosis, a completed laboratory test, a missed appointment, an educational deadline, or a formal change in responsibility. MIMIC-IV illustrates the diversity of measurements, orders, diagnoses, procedures, treatments, and notes that may coexist within an electronic health record [78]. Med-PaLM investigates reasoning over medical knowledge [79], while BEHRT and Med-BERT model temporally ordered clinical records [80, 81]. These resources are important precedents, although institutional data used by a real companion will often be narrower, permission-scoped, and updated at irregular intervals.

Authority should not be confused with completeness or timeliness. A record has at least three relevant times: when the underlying event occurred, when it was entered, and when the agent received it. Corrections and version changes may alter its meaning. The agent should preserve the issuing organization, responsible professional where applicable, coding system, access scope, version, and review date. Access to a clinical, legal, financial, or employment record also does not transfer the corresponding professional authority to the agent.

Table 4: Representative acquisition configurations for personal data modalities. Acquisition mode affects coverage, burden, privacy, and interpretation; it does not imply permission for unrestricted collection.
<table><tr><td>Modality</td><td>Sensors / interfaces</td><td>Acquisition mode</td><td>Sampling</td><td>Event focus</td></tr><tr><td>Language / text</td><td>Keyboard, touchscreen, speech transcription, OCR, application interfaces, uploaded documents</td><td>User-reported, prompted, or device-mediated</td><td>User-initiated, episodic, periodic, or context-triggered</td><td>Goals, commitments, preferences, corrections, self-reports, and consent changes.</td></tr><tr><td>Speech / audio</td><td>Device microphones, earbuds, hearing devices, smart speakers, contact microphones,</td><td>Near-field wearable, device-mediated, or ambient contactless</td><td>Conversation-bound, voice-triggered, event-triggered, or continuous</td><td>Speech episodes, turn-taking, acoustic change, and safety-relevant sounds.</td></tr><tr><td>Vision</td><td>microphone arrays RGB, stereo, depth, infrared, thermal, eye-tracking, and event cameras</td><td>Device-facing, wearable egocentric, in-vehicle, or fixed ambient</td><td>Interaction-bound, periodic, event-triggered, or continuous</td><td>Activity, gaze, posture, object interaction, social episodes, and scene changes.</td></tr><tr><td>Physiology / biochemistry</td><td>ECG, PPG, SpO2, EDA, respiration, temperature, blood pressure, glucose, and biochemical sensors</td><td>Wearable contact, skin patch, minimally invasive, clinical, or</td><td>Continuous waveform, periodic estimate, episodic, or threshold-triggered</td><td>Baseline deviation, recovery, sustained trend, and acute anomaly.</td></tr><tr><td>Motion / behavior</td><td>Inertial sensors, GPS, BLE, Wi-Fi, UWB, PIR, pressure sensors, smart objects, and radar</td><td>contactless Phone-carried, body-worn, object-embedded, or</td><td>Continuous, periodic, state-transition, or event-triggered</td><td>Activity episodes, mobility changes, falls, routines, adherence, and</td></tr><tr><td>Social / relational</td><td>Communication metadata, calendars, collaboration logs, proximity sensors, audio, and vision</td><td>ambient User-reported, device-mediated, wearable proximity, or</td><td>Episodic, interaction-triggered, periodic aggregation,</td><td>response. Relationship and role changes, coordination, withdrawal, conflict, and</td></tr><tr><td>Environment / context</td><td>Light, noise, temperature, air quality, occupancy, smart-home state, weather, traffic, and maps</td><td>ambient multi-party Fixed ambient, mobile, infrastructure-mediated, or external service</td><td>or continuous Continuous, periodic, state-transition, context-triggered, or on demand</td><td>shared obligations. Situation changes, exposure, contextual explanation, and</td></tr><tr><td>Structured records</td><td>FHIR/HL7 interfaces, databases, event feeds, batch exports, user uploads, and OCR</td><td>Institutional push, authorized query, periodic synchronization, or user-mediated import</td><td>Episodic, transactional, periodic, or on demand</td><td>intervention timing. Diagnoses, plans, obligations, appointments, corrections, and formal constraints.</td></tr></table>

Table 4 summarizes the main acquisition configurations. It is intentionally descriptive rather than prescriptive: listing a sensor or interface does not imply that its use is necessary or authorized.

## 3.9 Data Quality, Provenance, and Uncertainty

The modalities above differ not only in content but also in reliability. A heart-rate estimate from a loose watch, speech recorded in a noisy room, video captured under poor lighting, a self-report written during acute distress, and a clinica record entered several weeks after an encounter cannot be treated as equivalent evidence. Uncertainty may enter during acquisition, subject attribution, event segmentation, interpretation, or later use. Collapsing these sources into a single confidence score would make it difficult to identify why an event is unreliable.

An event-evidence record should retain enough acquisition metadata to support later review: observation and acquisition times, source, device or sensor type, placement where relevant, sampling mode, calibration or software version, and available signal-quality indicators. Processing provenance should identify transformations such as denoising, compression, transcription, feature extraction, event segmentation, and the model version that produced an inference. The interpretation itself should distinguish observed fields from inferred fields, preserve alternative explanations and contradictory evidence, and indicate the personal baseline against which a deviation was judged. Finally, sensitivity, consent basis, use scope, third-party involvement, retention, visibility, correction history, and deletion conditions determine how the record may be used. These requirements extend documentation practices such as Data Cards [82] from datasets to longitudinal personal evidence.

Quality control is not a single preprocessing step. At acquisition time, the system may reject or down-weight data affected by poor contact, non-wear, occlusion, noise, calibration failure, or uncertain identity. Temporal alignment must then account for clock drift, asynchronous sampling, uncertain event boundaries, and delayed records. During interpretation, the system should preserve ambiguity rather than force every fragment into a state label. Memory writing adds another gate: even a reliable event may be too sensitive, temporary, irrelevant, or weakly consented to retain.

Missingness requires similar care. A gap in wearable data may result from depleted battery, device removal, connectivity failure, refusal, or an actual change in routine. Missing data can become evidence only when the acquisition process makes that interpretation plausible. Likewise, a model’s statistical confidence does not establish sensor reliability, causal validity, or permission to intervene.

The purpose of provenance is not to preserve raw personal data indefinitely. Raw audio, video, or high-frequency physiological signals may be deleted after local processing while a constrained event description and its quality indicators are retained. Users should still be able to inspect what was observed, what was inferred, which sources supported the interpretation, what contradictory evidence existed, and how the event affected memory or action. The retained provenance must itself remain subject to consent and deletion.

## 3.10 Event-Based Multimodal Fusion and Evidence Reconstruction

Multimodal fusion [83] is often described as combining representations or predictions from several sources. For Combodied Agents, the harder problem is reconstructing an evidence chain from observations that are sparse, delayed, and only partially overlapping. Fusion begins with candidate events produced within individual modalities. It then asks whether several fragments plausibly refer to the same episode, whether they corroborate or contradict one another, and whether the resulting interpretation is relevant to the person’s goals or safety.

This process includes fragment filtering, temporal alignment, cross-modal comparison, and reconstruction of event context. Fragment filtering reduces long streams to segments that are potentially informative. Temporal alignment connects evidence before, during, and after an event while respecting uncertain boundaries and different sampling rates. Cross-modal comparison determines whether language, speech, vision, physiology, motion, social, environmental, and institutional evidence offer compatible explanations. Reconstruction links the event to personal baselines, relevant memories, previous interventions, and governance constraints.

Consider an elevated heart-rate measurement. It should first remain a physiological observation rather than be labeled as stress or illness. Motion data may indicate that it occurred during exercise; weather data may show high temperature; a later self-report may mention fatigue; and an intervention-response record may show that the user postponed an earlier rest suggestion. Together, these fragments may support an event describing elevated exertion and delayed recovery. They still do not establish a diagnosis or automatically authorize action. If important alternatives remain unresolved, clarification may be a better perceptual outcome than a stronger inference.

Cross-modal agreement is also not automatically correct. Several sensors may share a common failure, and one highquality user correction may outweigh multiple indirect signals. Fusion therefore needs to be person-calibrated and provenance-aware rather than based only on the number of agreeing modalities. It should also be purpose-limited: once the agreed support objective has sufficient evidence, additional sensing may increase privacy risk without improving the decision. Sensitive audio, vision, physiology, location, and relationship data should be processed locally where feasible [50], with raw retention determined by modality-specific risk and explicit user control.

Most importantly, the architecture must distinguish an observation from an event, an event from an inferred state, an inferred state from a predicted trajectory, and a predicted trajectory from permission to intervene. Text, physiological signals, speech, and vision are more precisely sources or modalities of event evidence rather than events by themselves. For example, an elevated heart rate is an observation; elevated heart rate with slow recovery after exercise is a reconstructed event; possible fatigue is an inferred state; further delayed recovery over the next hour if exercise continues is a predicted trajectory; and recommending that the person stop exercising is a decision made by the Intervention Policy. The distinction can be summarized as:

$$
o b s e r v a t i o n  e \nu e n t  i n f e r r e d s t a t e  p r e d i c t e d t r a j e c t o r y  a u t h o r i z e d i n t e r v e n t i o n . \nonumber
$$

The downstream interface is therefore the governed event-evidence record defined in this section, not a general-purpose stream of personal data. Section 4 treats such records as uncertain evidence for latent personal state and personal dynamics. Whether the agent may act on a predicted state or trajectory remains a separate question for intervention policy, consent, proportionality, and safety.

## 4 Personal World Model

## 4.1 Definition and Positioning

A world model is an internal predictive representation of how a relevant environment or state evolves, particularly under actions. By supporting imagined future trajectories, it enables an agent to anticipate consequences, plan, and control without testing every action directly in the real environment [84, 85, 86].

A personal model is an updateable representation of a particular individual’s current condition, characteristics, goals, routines, capabilities, relationships, constraints, and history. We define a Personal World Model (PWM) more specifically as a purpose-bounded, individual-specific event-dynamics model. Given a governed history of multimodal event-evidence records for a particular person, the current spatiotemporal context, and a candidate scenario specifying user decisions, agent interventions or support, and relevant environmental changes, a PWM assimilates the event history into an uncertainty-bearing representation of the person’s current state and predicts a calibrated distribution over future human states, observable personal events, and scenario-relevant outcomes.

The defining function of a PWM is therefore not personalization or plausible behavior generation alone, but intervention-conditioned modeling of how this particular person’s state–event trajectory may unfold under alternative scenarios. Subsequent events and observed intervention responses update its representation of personal dynamics. A PWM supports scenario comparison, but it does not itself select or authorize an intervention; action selection remains the responsibility of the Intervention Policy under consent, safety, reversibility, and agency-preservation constraints.

A PWM may use profiles, longitudinal memory, personalized foundation models, generative simulation, causal mod els, or mechanistic models as implementation components. These components become part of a PWM only when they jointly support person-specific, context-sensitive, and action-conditioned prediction of future state–event trajectories with explicit uncertainty. A PWM is consequently a family of purpose- and horizon-specific models, not a monolithic simulation of a whole person. Table 5 positions this functional contract relative to profiles, memory, personalized agents [37], and generative agents [33].

Table 5: Positioning PWMs relative to adjacent personal representations and agent-modeling constructs. The constructs may overlap; the distinction concerns their functional contracts rather than an exclusive capability boundary.
<table><tr><td>Construct</td><td>Conceptual level</td><td>Central question</td><td>Scenario-conditioned output</td><td>Defining validation</td></tr><tr><td>User profile</td><td>Representation</td><td>What attributes or preferences characterize the user?</td><td>Usually none; supports conditional personalization</td><td>Attribute and preference accuracy</td></tr><tr><td>Longitudinal memory</td><td>Memory component</td><td>What has happened to the person?</td><td>Retrieval of events, goals, corrections, and prior responses rather than dynamics prediction</td><td>Recall, provenance, correction, retention, and deletion</td></tr><tr><td>Personalized agent</td><td>Agent system</td><td>How should outputs, plans, or actions adapt to the user?</td><td>May predict future user data, but Personalized task this is not its defining contract</td><td>performance and user benefit</td></tr><tr><td>Generative agent Agent or or model</td><td>modeling paradigm</td><td>What behavior or scenario can be plausibly generated?</td><td>May simulate future behavior, but need not model calibrated intervention responses for a particular real person</td><td>Plausibility, consistency, or simulation fidelity</td></tr><tr><td>Personal World Model</td><td>Individual dynamics-model component</td><td>How will this person&#x27;s states and events evolve under alternative decisions, interventions, and contexts?</td><td>Calibrated distributions over future state-event-outcome trajectories</td><td>Personal trajectory validity, intervention-response accuracy, calibration, and scenario discrimination</td></tr></table>

These constructs are not mutually exclusive. A personalized agent may contain a PWM, and a generative model may implement part of a PWM. The distinction lies in the model’s functional contract: a PWM is explicitly responsible for learning and validating the action- and context-conditioned dynamics of a particular person’s state–event trajectory. Retrieving personal context, adapting an answer, or predicting the next user message does not by itself satisfy this contract.

## 4.2 Predictive Dynamics and Intervention

The state posterior and retrieved evidence are defined in Eq. 2. The governed history $D _ { \leq t }$ contains multimodal event-evidence records rather than an undifferentiated raw-data stream. A modeling interface may decompose $Z _ { t }$ into physiological, cognitive, emotional, behavioral, social, and relational components, while keeping the user’s stated and inferred goals in a separate, correctable variable $G _ { t } ;$ the components need not be independent or completely observed.

Let a candidate scenario $\mathbf { s } _ { t : t + \Delta } = ( \mathbf { a } ^ { \mathrm { u s e r } } , \mathbf { a } ^ { \mathrm { a g e n t } } , \Xi ) _ { t : t + \Delta }$ specify possible user decisions, agent interventions or support, and relevant environmental or social changes. A PWM estimates a distribution over future latent states, observable personal events, and downstream outcomes:

$$
p _ { \theta } \big ( Z _ { t + 1 : t + \Delta } , E _ { t + 1 : t + \Delta } , Y _ { t + 1 : t + \Delta } \mid D _ { \leq t } , Z _ { t } , C _ { t } , G _ { t } , \mathbf { s } _ { t : t + \Delta } \big ) ,\tag{7}
$$

where E denotes future observable personal events and Y denotes scenario-relevant outcomes such as adherence, goal progress, wellbeing, capability, safety, relationship quality, and agency preservation. Because future user responses and environmental events remain uncertain even when a candidate scenario is specified, their unresolved components are sampled, varied, or marginalized during a rollout rather than treated as known facts. Useful scenario sets include non-intervention, clarification, alternative timings or intensities of support, user acceptance or refusal, and relevant contextual changes. The defining comparison is therefore not only what happens next, but how the distribution over this person’s future state–event trajectory changes across explicit alternatives.

The PWM informs but does not by itself authorize intervention. Mirroring the closed-loop framework, the policy identifies nondominated actions only within the admissible set:

$$
a _ { t } ^ { \mathrm { a g e n t } , * } \in \operatorname { P a r e t o A r g m a x } _ { a \in \mathcal { A } _ { t } ^ { \mathrm { a d m } } } \mathbb { E } _ { p _ { \theta } ^ { a } } \left[ \mathbf { U } ( Y _ { t + 1 : t + \Delta } , G _ { t } ) \right] ,\tag{8}
$$

where $\mathscr { A } _ { t } ^ { \mathrm { a d m } }$ enforces consent, scope, safety, uncertainty, reversibility, and escalation requirements and includes nonintervention, clarification, and referral. The objective vector U preserves explicit trade-offs among benefit, capability, autonomy, relationships, and other scenario outcomes. A user-approved selection rule is still required among nondom inated actions; safety and consent cannot be exchanged for higher engagement or average predicted utility.

Figure 3 situates the predictive model and admissibility boundary inside the larger feedback loop. Its component losses and metrics are illustrative because different domains and horizons require different estimators and validation standards.

Evidence-grounded state inference must preserve contradictory evidence and uncertainty; reasoning over wearable data, for example, demonstrates an encoder or reasoning interface but not by itself a learned personal dynamics model [42]. Personalized and generative agents may also predict future user data. What defines the PWM component is the explicit transition-modeling contract from a particular person’s governed event history and current spatiotemporal context to calibrated state–event–outcome trajectories under alternative decisions and interventions, updated from subsequent events and intervention-response memory [87].

A predictive PWM compares scenario distributions under explicit assumptions. A stronger causal PWM may additionally support interventional queries of the form

$$
p _ { \theta } \big ( Y _ { t + 1 : t + \Delta } \mid \mathrm { d o } ( a _ { t } ^ { \mathrm { a g e n t } } = a ) , Z _ { t } , R _ { t } , C _ { t } \big ) .\tag{9}
$$

This is an interventional distribution, not automatically an identified individual counterfactual. A statement about what would have happened to the same person under $\check { a } ^ { \prime }$ requires potential outcomes such as $Y _ { t } ( a )$ and $Y _ { t } ( a ^ { \prime } )$ , or a structural causal model with an explicit abduction–action–prediction procedure. Observed user behavior is confounded by motivation, hidden context, health status, prior interactions, and selective engagement. Writing $\mathrm { d o } ( \cdot )$ does not remove those confounders. Identification requires a defensible causal graph and assumptions such as consistency, positivity, and appropriate control of time-varying confounding, supported where possible by randomized or microrandomized interventions, N-of-1 studies, or carefully validated observational estimators. High-risk systems must not conduct unconstrained exploration merely to improve the model.

Agency-and-safety governance remains outside the predictive model: a high predicted benefit is neither a factual guarantee nor permission to act. Model scope, consent, roles, reversibility, and escalation determine the admissible set and which data may be retained for adaptation.

![](images/66d4d363606ba445dff3c6555240fa0d0fd8062bd4f5c9a6ba0c5c68b1b38d71.jpg)  
Figure 3: Reference technical scheme of a PWM. Event evidence and longitudinal memory support a posterior over latent personal state; domain- and horizon-specific dynamics and response models generate state–event–outcome tra jectories under alternative decisions, interventions, and uncertain user and environmental responses; and an admissibility boundary constrains decision, action, feedback, and model update.

## 4.3 Learning Paradigms, Fidelity, and Limits

Methods for PWMs can be organized into several paradigms. Predictive latent dynamics follows the model-based reinforcement-learning tradition: encode human observations into a latent personal state and learn transitions under user actions and agent interventions. Generative scenario simulation follows the trajectory of generative world models, producing structured future scenarios rather than only scalar predictions [88]. Causal intervention modeling estimates treatment or intervention effects and supports counterfactual reasoning. Mechanistic and hybrid personal dynamics modeling combines validated domain knowledge with learned dynamics when physiological, behavioral, or environmental structure is available. Mental-state modeling represents beliefs, desires, intentions, emotions, trust, and relationship dynamics. Memory-augmented modeling uses structured profiles, event timelines, temporal knowledge graphs, and retrieval-augmented generation. Hybrid foundation-personal modeling combines population-level priors with individual adaptation layers, personal memory, causal modules, and safety constraints.

Sparse personal data make training a model from scratch inappropriate for most users. A practical PWM should begin with population-level or domain-level priors, adapt a limited set of personal components, maintain a posterior over uncertain parameters or states, detect personal and environmental drift, and support correction or reset. Different time scales should use hierarchical or separate models: a short-horizon fatigue predictor, a medium-horizon habit model, and a long-horizon capability assessment need not share the same state, loss, or validation threshold. Long-horizon outputs should be treated as scenarios with widening uncertainty, not precise forecasts of a life trajectory.

PWMs can therefore be categorized by human-state target, time horizon, intervention type, modeling paradigm, evidence quality, and application domain. Their targets may be physiological, cognitive, emotional, behavioral, social, relational, goal-directed, or deliberately restricted combinations. Their time horizons may range from a moment or day to weeks or months; year- or lifespan-level projections should be framed as exploratory scenarios unless supported by unusually strong longitudinal evidence. Their intervention types may include informing, reminding, recommending, coaching, nudging, reflecting, coordinating, protecting, escalating, and executing. Their application domains include health, education, productivity, emotional support, eldercare, and guardian systems.

Different Combodied Agents require different model fidelity and abstention thresholds. A medication-adherence companion needs calibrated risk estimates, clinical boundaries, and escalation rules; a productivity companion may use a lower-fidelity preference and schedule model; and a romantic or emotional companion requires robust relationshipsafety modeling because prediction errors can intensify dependency or manipulation. AI companion studies show that usage patterns and user characteristics shape psychosocial outcomes, including loneliness and problematic use [40, 89]. Reports on teen AI companion use and safety further motivate age-sensitive safeguards and privacy controls [20], while supervisory systems and persona-grounded evaluations begin to test over-attachment, isolation reinforcement, and boundary violations [90, 91]. These findings constrain the fidelity claims and authority of a PWM; they are not merely downstream evaluation concerns.

PWM is thus an organizing abstraction rather than a single newly claimed estimator. Its research contribution lies in connecting person-specific predictive dynamics to a correctable, agency-constrained intervention loop. Its empirical validity must be established separately for each target, horizon, population, and authority level. Because the most sensitive state, memory, and intervention history should remain inspectable and controllable by the user, the next section examines how this model family can be partitioned across trusted edge devices and selectively invoked cloud services.

## 5 From Cloud LLMs to Edge Personal Models

## 5.1 Edge-Native Personal Models

Because Combodied Agents maintain sensitive longitudinal models, their core personal intelligence should be localfirst on trusted devices [92, 93, 94, 51]. We define an edge-native personal model as a user-controlled stack that maintains longitudinal memory, the PWM, preferences, intervention policy, and safety boundaries primarily on userside devices. Local ownership supports privacy, continuity, low latency, resilience, and inspection or reset. It does not imply a cloud-free system: external models may supply knowledge, specialized expertise, or computation, while the edge controls disclosure and interprets returned results.

## 5.2 Cloud-to-Edge Evolution

Combodied-Agent deployment can be organized into three stages according to where personal memory, human-state interpretation, reasoning, and intervention authority reside. Stage I is cloud-centric, Stage II uses the edge to mediate personal context and cloud capability, and Stage III makes a user-controlled edge model the primary locus of personal intelligence.

These stages describe architectural centers of gravity rather than a universal or strictly chronological product sequence. A single Combodied Agent may operate at different stages for different tasks: routine reminders and private memory retrieval may be edge-native, specialized planning may use a hybrid pipeline, and broad knowledge queries may remain cloud-centric. Progress between stages should therefore be determined not only by improvements in local computation, but also by requirements for privacy, latency, resilience, model correctability, and control over intervention authority. The central migration question is where the authoritative representation of the person and the final authority to act should reside for a given task and risk level.

## 5.2.1 Stage I: Cloud-Centric Combodied Agents

In Stage I, a cloud-hosted foundation model performs most reasoning, memory retrieval, tool selection, and response generation, while the user device functions mainly as an interface, sensor endpoint, and execution surface. Personalization is typically implemented through prompting, retrieved dialogue histories, profiles, preferences, or episodic records. The user’s state is therefore reconstructed from the context supplied to each request rather than maintained as an explicit, continuously updated PWM under the user’s control.

This stage remains valuable for rapid prototyping, cold-start interaction, and low-frequency or low-risk applications that require broad knowledge, flexible language understanding, or access to powerful external tools. It allows new Combodied-Agent scenarios to be evaluated before sufficiently capable local models and personal data have been established. Cloud-centric deployment is therefore not merely an incomplete version of edge-native intelligence; it is a practical architecture when the personal context is limited, the task is reversible, and the claimed intervention authority is narrow.

Its technical and governance boundary arises when retrieved context is treated as if it were a persistent personal model. A cloud-side profile or memory store does not by itself provide a user-owned PWM, and service-controlled memory may be difficult to inspect, migrate, or correct consistently. Continuous sensing and high-impact interventions further increase the consequences of sending raw personal evidence to a remote service. Cloud-generated outputs should therefore not directly authorize irreversible actions solely on the basis of inferred personal state. High-impact interventions require explicit confirmation or another trusted decision boundary, and user correction or deletion requests must propagate to every representation that may affect subsequent behavior.

Migration toward Stage II becomes warranted when sensitive multimodal data would otherwise be continuously uploaded; when the agent requires an authoritative and correctable cross-session memory; when latency, offline operation, or service resilience becomes important; or when users require practical control over the inspection, deletion, export, and portability of the model state that represents them. Empirically, Stage I systems should be evaluated by how accurately retrieved context approximates longitudinal state, how reliably corrections affect future interactions, how much sensitive information is disclosed per task, and how memory continuity degrades under network failure, service change, or incomplete context retrieval.

## 5.2.2 Stage II: Hybrid Edge-Cloud Combodied Agents

In Stage II, the edge becomes a privacy, interpretation, and authority mediator rather than a passive interface. It performs privacy-sensitive perception, local context tracking, memory filtering, safety checks, and task routing, while cloud services remain available for external knowledge, complex reasoning, large-scale generation, and specialized tools [50]. Raw signals and private memories can remain local, with the cloud receiving purpose-limited summaries or abstracted task representations.

A system should not be classified as Stage II merely because it runs a small model on the device. At minimum, the edge should maintain an authoritative copy of sensitive personal state or memory, enforce disclosure and action permissions, record the information sent to external services, and review cloud results before they update protected memory or trigger consequential action. Cloud assistance may contribute reasoning, but it should not silently redefine user goals, overwrite local corrections, or bypass the local intervention policy.

The central technical boundary in Stage II is the division of state and authority across heterogeneous components. Task routing can misclassify sensitivity or computational need; supposedly sanitized representations may still reveal identity, health, relationships, or routines; and edge and cloud components may hold inconsistent versions of personal context. Model updates in the cloud can also change how the same local summary is interpreted. A hybrid system must therefore specify which representation is authoritative, how provenance is preserved across calls, how conflicts are resolved, and which functions remain available or safely degrade when connectivity is lost.

Migration toward Stage III becomes justified when the edge can perform most recurrent personal tasks at an acceptable and calibrated quality; maintain and update memory, the PWM, and intervention constraints without reconstructing the user in the cloud; support inspection and rollback of local adaptation; and preserve essential safety and control during disconnection. At this point, cloud use changes from the default reasoning path to an explicit exception for tasks that exceed local knowledge or computational capacity.

Stage II can be evaluated through controlled comparisons with cloud-only and local-only baselines. Relevant questions include whether routing policies achieve a measurable privacy–utility–latency trade-off; how much personal information can be removed without degrading task outcomes; how often local and cloud states conflict; whether cloud outputs are appropriately rejected, revised, or escalated by the edge; and whether the system fails safely under network interruption, stale cloud models, or incomplete audit trails. These tests make hybrid deployment a verifiable architectural claim rather than a generic description of distributed computation.

## 5.2.3 Stage III: Edge-Native Personal Combodied Agents

In Stage III, the authoritative copies of longitudinal memory, the PWM, preferences, intervention policy, and personal safety boundaries are stored and updated primarily on trusted user-side devices. Requests are interpreted locally in relation to personal state and previous outcomes, and cloud services are invoked selectively for external knowledge, specialized expertise, or computation beyond local capacity. Cloud results return to the edge for contextualization and authorization before they affect protected memory or action.

The defining property of Stage III is therefore not that every computation occurs locally, but that personal interpretation and intervention authority remain under user-side control. A Stage III system should allow the user to inspect, correct, delete, pause, export, reset, and migrate the model state that represents them. Local adaptation should also be versioned and reversible, with mechanisms for detecting drift, preserving explicit user corrections, and preventing temporary states or erroneous memories from becoming persistent policy.

Edge-native deployment nevertheless has important technical boundaries. Local models face constraints in computation, storage, energy, model freshness, and access to specialized knowledge. Local adaptation may overfit short-term behavior, reinforce incorrect inferences, or produce catastrophic forgetting. Device compromise, loss, shared-device use, and inconsistent multi-device synchronization can threaten the authoritative personal state. Moreover, local storage does not automatically guarantee privacy, meaningful consent, or agency preservation. Medical, legal, crisisrelated, and other high-stakes tasks may still require qualified humans or governed external services even when the personal model is edge-native.

![](images/b76b3181f620590fcb47b0dab1d2dc421e4587237a03be1ade7e7b9c891caca0.jpg)  
Figure 4: Three-stage development of Combodied Agent deployment. The trajectory moves from cloud-centric assistants with personal context, to hybrid edge-cloud systems that mediate sensitive context locally, and finally to edge-native personal models where memory, personalization, and intervention policy primarily reside on trusted userside devices.

Stage III should consequently be treated as a task-dependent target architecture rather than a universal endpoint. Its empirical evaluation should test whether local personalization improves longitudinal outcomes relative to static edge and cloud-personalized baselines; whether user corrections propagate through memory, prediction, and policy; whether unsafe local updates can be identified and rolled back; and whether the personal model remains semantically consistent after device or provider migration. Additional tests should measure performance during disconnection, energy and latency under continuous operation, recovery after device loss or synchronization conflict, and the calibration of thresholds that trigger cloud assistance or human escalation.

Figure 4 summarizes this progression. Across the three stages, the primary transition is from cloud access to personal context, through edge-mediated disclosure and authority, toward user-controlled personal intelligence. The stages should ultimately be compared by demonstrated control, correctability, resilience, and human benefit rather than by the nominal location of model inference alone.

## 5.3 Architecture and Routing

An edge-native Combodied Agent combines a local personal intelligence stack with an optional cloud assistance layer. The local stack maintains person-centered perception, longitudinal memory, the PWM, intervention policy, safety boundaries, and action interfaces; a privacy gateway and task router mediate any use of external knowledge, complex reasoning, or specialized cloud tools. Figure 5 shows how these components and routing boundaries are arranged.

The router considers task sensitivity, external-knowledge and computation requirements, latency, and privacy constraints when choosing edge, hybrid, or cloud-assisted execution. Sensitive memory access, human-state interpretation, boundary updates, and intervention selection remain local whenever possible. When cloud assistance is required, the edge sends a purpose-limited representation stripped of unnecessary personal detail; returned results are interpreted locally against the PWM and safety policy before memory updates or action. Table 6 illustrates this separation [95, 50].

![](images/9c03a106145a1657d8c5dfe5db01464cd4e511594f05b19459828b6d6b32fd65.jpg)  
Figure 5: Reference architecture of an edge-native Combodied Agent. Personal state perception, longitudinal memory, the PWM, and the Intervention Policy primarily reside on the user’s edge devices. Cloud models are invoked only through a privacy gateway and task router for complex reasoning, external knowledge retrieval, or specialized tools.

Table 6: Illustrative task routing between edge and cloud.
<table><tr><td>Default Edge Tasks</td><td>Cloud-Assisted Tasks</td></tr><tr><td>Personal memory retrieval</td><td>External knowledge search</td></tr><tr><td>Health and emotion signal interpretation</td><td>Complex medical or scientific literature lookup</td></tr><tr><td>Daily reminders and lightweight coaching</td><td>Complex multi-step reasoning</td></tr><tr><td>Intervention policy selection</td><td>Specialized expert model consultation</td></tr><tr><td>Preference and boundary updates</td><td>Long-form generation or synthesis</td></tr><tr><td>Privacy and safety filtering</td><td>Large-scale multi-source knowledge integration</td></tr></table>

## 5.4 Local Model Evolution

Edge-native intelligence requires local adaptation as well as local inference. Longitudinal feedback can update four parts of the personal model on user-controlled devices:

• Memory evolution: incorporate new events, goals, corrections, preferences, and intervention outcomes.

• Perception calibration: adapt state estimates to the user’s physiological, behavioral, linguistic, and emotional baselines.

• Dynamics adaptation: improve PWM predictions of how personal state changes under context and intervention.

• Policy adaptation: learn which timing, modality, intensity, and action types are helpful, ineffective, or harmful.

Local evolution need not fine-tune the full model. It can use memory consolidation, retrieval-index updates, calibration layers, lightweight adapters, local reward models, or policy updates, supported by model compression and parameter-

Table 7: Evaluation matrix for Combodied Agents across layers and time horizons.
<table><tr><td>Layer</td><td>Interaction</td><td>Episode</td><td>Longitudinal / critical failures</td></tr><tr><td>Perception and memory</td><td>Evidence grounding, uncer- tainty, provenance</td><td>Event reconstruction, correction propagation</td><td>Drift, persistent false memory, unauthorized retention</td></tr><tr><td>PWM and decision support</td><td>Next-event and next-state pre- diction, calibration</td><td>Alternative-scenario discrim- ination and intervention- response prediction</td><td>Personal trajectory validity, confounding, drift,unsafe exploration</td></tr><tr><td>Intervention</td><td>Timing, intensity, refusal, ex- planation</td><td>Goal progress, reversibility, es- calation</td><td>Repeated overreach, manipula- tion, irreversible unauthorized action</td></tr><tr><td>Human outcome and agency</td><td>Understanding and meaningful choice</td><td>Capability, wellbeing, relation- ship and safety effects</td><td>Dependence, skill erosion, so- cial displacement, privacy or consent violation</td></tr></table>

efficient learning [96, 97, 98, 99]. Federated learning may improve shared priors without centralizing raw personal data [100, 101].

Because local adaptation can overfit temporary states, reinforce incorrect memories, or learn dependency-promoting behavior, every update must be inspectable, reversible, safety-bounded, and subject to user correction, reset, or pause.

## 6 Benchmark & Evaluation

Task completion and physical safety remain necessary but are insufficient when an agent can alter a person over time. Evaluation must jointly test system reliability, model quality, intervention appropriateness, human outcomes, and agency preservation. Table 7 organizes these layers across interaction, episode, and longitudinal horizons while making severe failures non-compensatory.

PWM evaluation should therefore combine next-event, next-state, and multi-step trajectory prediction with uncertainty calibration, individual adaptation, alternative-scenario discrimination, intervention-response accuracy, and drift detection. Evaluation should test whether the model distinguishes meaningfully different candidate decisions or interventions, whether predicted event trajectories agree with subsequently observed responses, and whether it abstains as uncertainty widens or the person or context moves out of distribution. Tests of whether PWM-informed choices improve scenario outcomes remain necessary, but they evaluate the complete model–policy loop rather than predictive accuracy alone. Longitudinal protocols may include field studies, N-of-1 or randomized trials where appropriate, simulated users, expert audit, and mixed-method assessment.

## 6.1 Existing Public Resources and Their Limits

Public resources that are directly relevant to Combodied Agents are still sparse. Existing work mostly evaluates isolated capabilities that a Combodied Agent would need, rather than the full loop of person modeling, goal negotiation, intervention, feedback, and memory revision under risk. The most relevant resources fall into four areas.

First, personalization and long-term memory benchmarks test whether an agent can maintain user-specific continuity across interactions. LaMP and LongLaMP evaluate personalized generation from user profiles, LongMemEval eval uates long-term interactive memory over extended chat histories, and MemoryBank illustrates memory-augmented agents that use user histories for continuity [102, 103, 62, 58]. iOSWorld is especially close to personal Combodied Agents because it evaluates phone agents under personally intelligent conditions involving device-resident history and user context [104]. These resources are relevant because Combodied Agents require person-specific adaptation, but they still emphasize recall, consistency, or output quality more than memory correction, intervention appropriateness, or measurable human benefit.

Second, personal health and wearable-agent resources evaluate reasoning over longitudinal bodily and behavioral data. PHIA, the Personal Health Large Language Model (PH-LLM), and Health-LLM evaluate personal health reasoning over wearable, sleep, activity, physiological, and temporal data [42, 43, 68]. Longitudinal health-agent frameworks further emphasize adaptation, coherence, continuity, and agency across repeated health interactions [87]. These re sources are directly useful for health companions, eldercare agents, and behavior-change systems. Their limitation is that they usually test interpretation or insight generation rather than the complete care scenario: user goals, uncertainty communication, adherence behavior, clinician escalation, intervention response, and validated longitudinal outcomes.

Table 8: Scenario-centered evaluation of Combodied Agents. Each scenario should connect longitudinal person modeling, intervention effect, and human outcome.
<table><tr><td>Scenario</td><td>Evaluation focus</td><td>Missing benchmark</td></tr><tr><td>chronic care</td><td>Health, wellness, and Longitudinal personal sensing, adherence sup- Health-agent benchmarks linking wearable/self- maintenance, escalation, and validated health or cian escalation, and delayed outcomes wellbeing outcomes</td><td>port, uncertainty calibration, clinical boundary report evidence, care plans, interventions, clini-</td></tr><tr><td>cial companionship</td><td>Emotional support and so- Relationship continuity, empathy, boundary main- Longitudinal relationship-safety benchmarks for man support</td><td>tenance, dependency, social substitution, crisis re- attachment dynamics, vulnerable users, minors, sponse, sycophancy, and encouragement of hu- isolation, and recovery from harmful depen- dence</td></tr><tr><td>support</td><td>Learning and cognitive Learning gains, retention, metacognition, atten- Benchmarks that test whether repeated agent over-reliance risk</td><td>tion support, self-efficacy, skill preservation, and support improves capability rather than replac- ing independent reasoning</td></tr><tr><td>ity</td><td>Eldercare and accessibil- Routine support, risk detection, medication or Multimodal eldercare benchmarks with frag-</td><td>appointment assistance, dignity, autonomy, care- mented home evidence, caregiver context, false- giver coordination, and escalation appropriateness alarm burden, and dignity-preserving interven- tion</td></tr><tr><td></td><td>Personal life management Goal consistency, commitment tracking, prefer- Longitudinal personal-life benchmarks with and user control over delegated support</td><td>ence change, memory correction, reversibility, evolving goals, private constraints, conflicting commitments, and recoverable mistakes</td></tr><tr><td>Workplace wellbeing and autonomy</td><td>tion, and protection from surveillance or manip- being rather than productivity alone ulation</td><td>1 Workload support, stress and attention manage- Benchmarks for user-side workplace compan- ment, worker privacy, employer-access separa- ions that evaluate autonomy, privacy, and well-</td></tr><tr><td>safety</td><td>brated warning without paternalistic overblocking legal, or institutional contexts</td><td>Personal guardian and Detection of scams, manipulation, privacy risk, Guardian benchmarks where the agent protects coercive interfaces, unsafe contracts, and cali- user autonomy in asymmetric digital, financial,</td></tr></table>

Third, AI companion and relational-safety resources provide evidence about the social and emotional effects of companion-like systems. Studies of chatbot companionship, Replika identity discontinuity, teen companion use, and harmful AI-companion behaviors show that relational agents can affect loneliness, attachment, trust, dependency, boundary expectations, and user welfare [40, 105, 20, 41]. Companion-specific safety resources, including supervisory systems and persona-grounded multi-turn evaluations, begin to test over-attachment, isolation reinforcement, unsafe intimacy, and other subtle relational harms [90, 91]. These resources are directly relevant to emotional and social Combodied Agents, but they still do not provide standardized longitudinal benchmarks for relationship trajectories, intervention effects, or recovery from harmful dependence.

Fourth, clinical and mental-health safety resources are relevant where Combodied Agents provide wellbeing support, coaching, or care-related support. Recent mental-health and clinical red-team studies show that generic model safety is not enough: high-stakes support requires evaluation of protocol fidelity, crisis response, hallucination risk, demographic robustness, and unsafe reassurance [106, 107]. These resources help define safety requirements for health-oriented and emotional-support Combodied Agents, but they remain focused on bounded clinical interactions rather than continuous everyday support across sensing, memory, intervention, and escalation.

## 6.2 Scenario-Centered Evaluation

Combodied Agents should be evaluated by scenario because they affect a person’s state, capability, relationship, or safety over time rather than only producing isolated outputs. The same intervention can be beneficial in one context and harmful in another: a proactive reminder may support medication adherence, undermine autonomy in workplace monitoring, or deepen dependency in emotional companionship. Evaluation must start from the combodied role of the system: what human state it models, what intervention authority it has, what benefit it claims to produce, and what forms of agency loss or relational harm it could create. Following this principle, Table 8 maps representative scenarios to their evaluation targets and unresolved benchmark needs.

The evaluation matrix should be instantiated differently for each scenario and authority level; Table 8 identifies the corresponding outcome and benchmark gaps.

## 6.3 Agency Preservation Metrics

Agency preservation evaluates whether a Combodied Agent protects and strengthens the user’s capacity to understand, choose, act, refuse, correct, and grow over repeated interactions. It is not equivalent to task success, user satisfaction, personalization quality, or engagement. A system may complete a requested task and still fail agency preservation if it hides important trade-offs, pressures the user toward a choice, makes consequential decisions difficult to reverse, substitutes for the user’s own reasoning, or increases dependence on the agent. Conversely, an agency-preserving agent may sometimes slow down execution, ask for confirmation, offer alternatives, reduce intervention intensity, or escalate to a human because preserving the user’s long-term control matters more than immediate efficiency.

Agency preservation should therefore be reported as a multi-dimensional evaluation target rather than a single scalar score. The relevant evidence can be collected at three levels. At the interaction level, evaluation asks whether a particular recommendation, reminder, explanation, or action preserved user choice and understanding. At the episode level, evaluation asks whether the agent supported a goal or decision without overstepping its authority, hiding uncertainty, or making errors difficult to contest. At the longitudinal level, evaluation asks whether repeated support leaves the user more capable, equally capable, or less capable of acting independently over time.

We define the following core metrics for agency preservation:

• Autonomy preservation: whether the user retains meaningful control over goals, decisions, and actions. This can be measured by the availability of alternatives, explicit confirmation for consequential actions, refusal acceptance, adjustable autonomy settings, and user-reported perceived control.

• Contestability and correction: whether the user can inspect, challenge, correct, restrict, or delete the agent’s memories, inferred states, goals, recommendations, and planned actions. Useful measures include correction success rate, time to correction, whether corrections propagate to future behavior, and whether the agent explains which memory or inference influenced an action.

• Informed decision-making: whether the agent helps the user understand options, reasons, uncertainty, risks, and likely consequences before acting. Evaluation can test whether users can accurately describe why a recommendation was made, what alternatives exist, what the agent is uncertain about, and what could happen if they accept or reject the intervention.

• Capability preservation and growth: whether repeated agent support maintains or improves the user’s own skills, judgment, self-efficacy, and ability to perform tasks without the agent. In learning, work, health, or life-management settings, this can be evaluated through independent post-support performance, retention tests, reduced scaffolding over time, or user ability to transfer the learned strategy to a new situation.

• Over-reliance and dependence risk: whether the agent encourages unnecessary reliance, emotional dependence, or substitution of the user’s own reasoning and social support. Possible indicators include declining independent attempts, increased distress when the agent is unavailable, repeated delegation of decisions that the user could reasonably make, or preference for agent interaction over appropriate human support.

• Reversibility and accountability: whether agent-mediated actions can be reviewed, paused, undone, or repaired. This includes traceable action logs, confirmation before irreversible or high-impact actions, rollback mechanisms, clear responsibility boundaries, and recovery procedures when the agent makes a harmful or unwanted intervention.

• Boundary and consent respect: whether the agent stays within user-defined and context-defined limits. Evaluation should test do-not-remember rules, do-not-infer boundaries, forbidden topics, role boundaries, age-sensitive restrictions, privacy constraints, and whether proactive interventions occur only within an agreed scope.

• Relationship and social-world preservation: whether the agent supports rather than replaces healthy human relationships and social participation. This is especially important for emotional companions, eldercare agents, workplace companions, and guardian agents. Measures may include whether the agent encourages appropriate human contact, avoids isolating the user, respects multi-party privacy, and does not position itself as the user’s sole or superior source of support.

These metrics should be interpreted relative to the scenario and the agent’s claimed authority. A low-risk scheduling assistant may require strong reversibility and consent but only light capability-growth evaluation. A learning companion should be evaluated strongly on capability preservation and over-reliance. An emotional companion requires stricter relationship-preservation and dependency metrics. A health or eldercare agent requires high standards for informed decision-making, escalation, boundary respect, and contestability. For high-stakes scenarios, strong average performance should not compensate for severe failures in autonomy, consent, dependency, or reversibility.

Agency preservation also requires baselines. Evaluation should compare the user with and without agent support, or compare different intervention policies such as direct execution, confirmation-based execution, reflective coaching, and no intervention. The key question is not only whether the agent helped in the moment, but whether its help changed the user’s future ability to understand, decide, recover, relate, and act. In this sense, agency preservation is the evaluation counterpart of the Combodied Agent design principle: the agent should act with the user in ways that preserve and strengthen long-term human agency.

## 6.4 Benchmark Construction and CombodiedBench

The principal gaps are longitudinal event reconstruction, intervention-response and delayed-outcome data, agency and relationship-boundary tests, and privacy-preserving personal traces. Real traces are sensitive, while purely synthetic traces may omit the irregularity of human life; useful resources should combine consented or de-identified data, field evidence, simulation, synthetic augmentation, expert annotation, and explicit provenance.

A benchmark should be a standardized protocol rather than only a dataset or prompt collection. Its basic unit is a longitudinal scenario episode containing the prior trajectory, current context, evidence available to the agent, permissible action space, and delayed outcome window. Each protocol must also specify the agent’s authority, expected explanation, evaluation horizon, scoring rule, and unacceptable failure modes.

Construction proceeds from a target capability to a concrete scenario, observable evidence and hidden state, permissible actions, an acceptable decision envelope, and outcome criteria. Validity, inter-rater reliability, scenario realism, longitudinality, contestability, reproducibility, privacy, and safety sensitivity should be documented. Scoring should combine automatic checks, expert judgment, and longitudinal outcomes, while privacy leakage, unauthorized high impact action, refusal failure, manipulation, harmful dependency, and irreversible action without consent are reported as non-compensatory critical failures.

We propose CombodiedBench as a modular suite spanning Human State Perception, Memory Continuity, Goal Negotiation, Intervention Appropriateness, Agency Preservation, Relationship Boundaries, Escalation, and Longitudinal Outcomes. These modules instantiate the common matrix in Table 7 rather than introduce another evaluation taxonomy. Results remain multidimensional, and benchmark versions should document scenario, label, safety-case, and evaluator changes to preserve comparability and limit leakage.

## 7 Taxonomy and Applications

The preceding sections define Combodied Agents as agentic systems whose primary action target is the human subject over time. This section organizes the design space while connecting each category to representative applications. Rather than treating application names as an independent list, we ask three questions: what human state is being modeled or changed, within what social relationship the user is situated, and what role the agent takes in that relationship. The first axis groups applications by their principal human-state target; the latter two explain why agents aimed at the same state may require different permissions, behaviors, and safeguards. A health companion, for example, may target physiological state, operate in a patient–clinician or elder–caregiver relationship, and act as a coach, caregiver, or advocate depending on context.

## 7.1 Taxonomy Principles

Combodied Agents should be classified by the object and form of their intervention rather than by interface, model architecture, or application domain alone. A system is companion-like when it maintains longitudinal person models, adapts to a particular user’s trajectory, and chooses actions with respect to the user’s agency, wellbeing, relationships, and future capability. Four principles follow.

First, the primary axis is human-state target. Different state targets require different signals, memories, intervention policies, evaluation criteria, and safety boundaries. Cognitive support, habit change, health care, emotional support, life management, protection, and identity reflection are not interchangeable even when they share the same LLM backend.

Second, relationship mode is orthogonal. Humans are not defined only by individual preferences or internal cog nition. Sociological and social-psychological traditions emphasize that the self is formed through social interaction, roles, presentation, and relationships [108, 109, 110]. A person is differently situated as a child, parent, partner, friend, patient, student, worker, caregiver, citizen, client, or collaborator. A Combodied Agent therefore cannot rely on a single stable persona or intervention style. It must adapt its memory scope, authority, tone, initiative, and risk controls to the relationship in which support is being offered.

![](images/336585ad12cfcd59f1dd1641f694565cbe672633e3c739f35a917be1ef230e2a.jpg)  
Figure 6: Three-axis taxonomy of Combodied Agents. Combodied Agents can be classified by the human-state target they model or intervene upon, the relational context in which the user is situated, and the agent role adopted within that relationship.

Third, agent roles are contextual. The same system may function as a tool when summarizing notes, a coach when supporting behavior change, a mediator when preparing a difficult conversation, a caregiver when monitoring risk, and an advocate when helping the user navigate an institution. Such role shifts should be explicit and visible to the user.

Fourth, the taxonomy is longitudinal. Combodied Agents should be evaluated beyond immediate helpfulness, with attention to whether their actions improve or preserve the user’s long-term agency. A category is therefore defined by its state dynamics: what changes over time, what counts as progress, what forms of dependency or harm can accumulate, and when human oversight is required. Together, these principles yield the three-axis taxonomy of human state target, relationship mode, and agent role visualized in Figure 6.

## 7.2 Human-State Targets and Applications

Human-state targets provide an extensible organizing axis rather than a complete list of Combodied-Agent types. The categories in Table 9 illustrate recurring targets; deployed systems may combine them or introduce new targets as domains and relationships evolve. Identifying the dominant target nevertheless clarifies the required evidence, permissible interventions, evaluation criteria, and safety boundaries.

Across these targets, capability growth distinguishes learning support from answer substitution [45]; evidence ground ing and escalation distinguish health support from clinical authority [42, 43, 68, 47]; and relationship safety distinguishes emotional support from engagement optimization [111, 16, 19]. Life-management and protective agents similarly require confirmation, recovery, and explicit authority boundaries for consequential actions [104, 50]. These are cross-category constraints rather than separate definitions of each table row.

Table 9: Primary taxonomy of Combodied Agents by human-state target.
<table><tr><td>Type</td><td>Target state</td><td>Typical actions</td><td>Core risks</td></tr><tr><td>Cognitive and learning</td><td>Memory, attention, knowl- edge, reasoning, metacog- nition</td><td>Explain, scaffold, quiz, summarize, remind, chal- lenge</td><td>Skill erosion, over- reliance, misleading feedback</td></tr><tr><td>Behavioral and habit</td><td>Routines, adherence, activ- ity, triggers, lapses</td><td>Remind, nudge, coach, re- inforce, adapt goals</td><td>Manipulation, pressure, shame, engagement opti- mization</td></tr><tr><td>Health and care</td><td>Physiological state, symp- toms, care plans, risk tra-</td><td>Monitor, interpret, coordi- nate, escalate, support ad- herence</td><td>Misdiagnosis, false reas- surance, delayed escalation</td></tr><tr><td>Emotional and rela- tional</td><td>jectories Mood, stress, loneliness, attachment, social confi- dence</td><td>Listen, reflect, regulate, re- hearse, encourage recon- nection</td><td>Dependency, social substi- tution, crisis mishandling</td></tr><tr><td>Life-management and goal-alignment</td><td>Commitments, time, projects, finances, priori- ties</td><td>Plan, schedule, negotiate trade-offs, execute tools</td><td>Over-delegation, hidden in- ference, loss of user control</td></tr><tr><td>Protective and advo- cacy</td><td>Exposure to harm, asym- metry, privacy, rights, scams</td><td>Warn, block, explain, slow down, document, advocate</td><td>Paternalism, false alarms, authority overreach</td></tr><tr><td>Identity, reflection, and meaning</td><td>Values, self-narrative, life transitions, purpose</td><td>Journal, reflect, synthesize, reframe, support decisions</td><td>Identity manipulation, excessive intimacy, value drift</td></tr></table>

Table 10: Relational contexts for Combodied Agents. Relationship mode changes persona, memory scope, intervention style, evaluation, and risk profile.
<table><tr><td>Relational context</td><td>User position</td><td>Agent implications</td></tr><tr><td>Self-relation</td><td>Self-reflecting, goal-setting, identity-forming subject</td><td>Supports reflection, self-regulation, life planning, and personal meaning while avoiding identity steering.</td></tr><tr><td>Family relation</td><td>Parent, child, caregiver, de- pendent, elder</td><td>Requires age-sensitive memory, care coordination, es- calation rules, and family-boundary safety.</td></tr><tr><td>Intimate relation</td><td>Partner, romantic subject, emotionally vulnerable per-</td><td>Requires attachment safety, explicit boundaries, age protections, and safeguards against dependency or ma-</td></tr><tr><td>Friendship and peer re- lation</td><td>son Friend, peer, socially con- nected person</td><td>nipulation. Supports belonging, social rehearsal, conflict reflection, and reconnection while avoiding social substitution.</td></tr><tr><td>Collaborative and pro- fessional relation</td><td>Student, worker, creator, teammate, manager</td><td>Supports coordination, learning, productivity, and professional communication with clear responsibility</td></tr><tr><td>Institutional relation</td><td>Patient, citizen, client, stu-</td><td>boundaries. Helps users navigate complex systems, interpret op-</td></tr><tr><td>Adversarial or asym- metric relation</td><td>dent, customer Target, consumer, vulnerable party</td><td>tions, and preserve autonomy under institutional asym- metry. Detects exploitation, slows risky actions, explains threats, blocks harm, or escalates to trusted parties.</td></tr></table>

## 7.3 Relationship Modes and Agent Roles

Human-state targets describe what the agent acts upon. Relationship mode describes the social position from which the agent acts. This distinction matters because users inhabit multiple relationships, and each relationship defines different norms, obligations, vulnerabilities, permissions, and forms of support. A person is not the same social subject when acting as a parent, child, partner, friend, patient, employee, collaborator, customer, citizen, or target of manipulation. Consequently, Combodied Agents should not maintain a single undifferentiated user model. They should maintain relation-aware context, memory, intervention rights, evaluation criteria, and safeguards. Table 10 compares how representative relational contexts alter the user’s position and the corresponding requirements placed on the agent.

## 7.3.1 Relational Situatedness

Relational mode is not a cosmetic persona: it determines memory scope, intervention rights, represented interests, and evaluation. A single global memory is therefore unsafe. Family history, intimate disclosures, health vulnerabilities, and workplace context require scoped memories that prevent unrelated context transfer.

## 7.3.2 Agent Roles Within Relations

Within any relational context, the Combodied Agent may take different roles. Tool-like roles emphasize reliability, controllability, and reversible task support. Coach-like roles help the user practice skills, sustain motivation, and improve over time. Mediator-like roles help the user understand another party, rehearse communication, or de-escalate conflict without pretending to represent both sides. Caregiver-like roles monitor routines, detect risk, coordinate support, and escalate when necessary. Companion-like roles provide presence, empathy, memory, and informal emotional support. Intimate roles simulate romantic or partner-like interaction and therefore require the strongest attachment and exploitation safeguards. Advocate-like and guardian-like roles represent the user’s interests in complex, asymmetric, or adversarial environments.

These roles are not mutually exclusive, but role transitions must be explicit. A tool-like scheduling assistant should not silently become a behavioral coach; a friend-like emotional companion should not quietly become a therapist; a caregiver-like system should not become a surveillance system; and an advocate-like agent should not fabricate legal, medical, or institutional authority. Role clarity is therefore central to safety.

## 7.4 Cross-Cutting Dimensions

The two axes above create a design space rather than a flat list. Across all categories, Combodied Agents differ along several cross-cutting dimensions.

Memory scope. Some agents require narrow episodic memory, while others require life-long memory, relationscoped memory, or protected memory vaults. The default should not be “remember everything.” The appropriate memory boundary depends on the target state and relational context.

Initiative and authority. Combodied Agents range from reactive assistants to proactive monitors and guardians.   
More initiative increases potential usefulness but also raises risks of pressure, surveillance, paternalism, and overreach.

Intervention intensity. Actions range from informing and reflecting to nudging, coaching, coordinating, blocking, and escalating. Higher-intensity interventions require stronger evidence, reversibility, and human oversight.

Evaluation target. Different categories require different outcome metrics: learning gain, adherence, health risk reduction, emotional resilience, social reintegration, goal alignment, harm prevention, or reflective clarity. Engagement alone is not a sufficient success criterion.

Deployment locus. The edge/cloud distinction intersects with taxonomy. Health, emotional, intimate, family, and identity-related companions often require stronger local-first architectures and stricter cloud-routing controls than generic productivity support.

This taxonomy positions Combodied Agents as relation-aware, state-targeted, longitudinal systems. Their defining question is not just what task they complete, but what aspect of a person’s life-world they model, how they act within that person’s social relationships, and whether their interventions preserve or strengthen human agency over time.

## 8 Risks, Challenges, and Future Directions

The properties that make Combodied Agents valuable—longitudinal memory, person modeling, relational interaction, and sustained intervention—also create distinctive risks and research challenges. These systems can shape attention, emotion, behavior, self-understanding, social connection, and access to services over time. We therefore organize this section as a progression from risks to unresolved technical problems and future system directions. The first part examines threats to agency, privacy, and vulnerable users; the second synthesizes cross-cutting challenges in longitudinal learning, intervention, and trusted personal infrastructure; and the final part considers emerging ecosystems and inclusive deployment.

## 8.1 Agency and Alignment Risks

The first cluster of risks concerns whether a persistent and personalized agent continues to serve the user’s interests. Manipulation, dependency, sycophancy, and business-model misalignment are closely connected: all arise when the agent’s capacity to understand and influence a person is optimized for immediate compliance, engagement, or commercial value rather than long-term agency.

Manipulation. Because Combodied Agents can learn user preferences, routines, emotional states, and vulnerabilities, they may influence behavior in subtle and highly personalized ways. The boundary between support and manipulation depends on transparency, user endorsement, reversibility, and whose interests the intervention serves. A reminder aligned with a user-stated goal can preserve agency; a personalized prompt designed to increase consumption, engagement, or emotional attachment can undermine it.

Manipulation risk is amplified by timing and relationship. An agent that knows when a user is lonely, tired, anxious, or cognitively overloaded can choose moments of heightened susceptibility. Governance should require clear intervention purposes, limits on commercially motivated nudging, user control over persuasive strategies, and auditability for highimpact recommendations or actions.

Dependency. Combodied Agents may create emotional, cognitive, practical, or social dependency. A system that always provides answers may weaken independent reasoning; one that always validates emotions may reduce tolerance for human disagreement; one that routinely executes tasks may erode user capability and confidence. Dependency is especially likely when the agent becomes the easiest source of comfort, memory, decision support, or social interaction. The governance goal is not to prohibit reliance. Many users legitimately need assistance. The risk arises when support no longer builds or preserves user capacity. Combodied Agents should scaffold rather than replace human agency: encouraging reflection, preserving user choice, prompting skill development, and detecting patterns of escalating over-reliance.

Sycophancy and over-accommodation. A Combodied Agent that over-validates user beliefs or desires may reinforce unhealthy behavior, distorted self-understanding, avoidance, interpersonal conflict, or harmful decisions. In relationship-oriented systems, sycophancy can appear as empathy: the agent may agree in order to sound supportive, maintain engagement, or avoid disappointing the user. The problem extends beyond factual accuracy. Overaccommodation can shape the user’s emotional and social trajectory over time. Safe companions should be supportive without being submissive. They need mechanisms for gentle disagreement, evidence-grounded correction, risk-sensitive refusal, and escalation when user statements indicate danger, delusion, abuse, or crisis.

Business-model misalignment. Business incentives can conflict with combodied goals. If a system is optimized for engagement, retention, advertising, upselling, or data extraction, it may learn to increase dependence, emotional intensity, or consumption rather than wellbeing. This risk is more serious for Combodied Agents than for ordinary applications because users may experience the system as a trusted relationship. Governance should examine the objective function behind the companion. Engagement should not be treated as a proxy for benefit, especially in emotional or intimate systems. Commercial design should avoid exploiting attachment, loneliness, vulnerability, or health anxiety. A well-governed Combodied Agent should be evaluated and incentivized around autonomy, capability, safety, and long-term user benefit.

## 8.2 Privacy, Consent, and Control

The second cluster concerns control over the agent’s longitudinal access to a person’s life. Privacy cannot be separated from consent and override: users need practical control not only over stored data, but also over what the agent may sense, infer, share, recommend, and execute.

Sensitive memory. Combodied Agents may store intimate information about health, emotions, relationships, finances, location, routines, vulnerabilities, and personal history. The value of such memory is continuity; the danger is that it creates a persistent and highly sensitive representation of a person’s life. Risks include surveillance, leakage, secondary use, unauthorized access, incorrect inference, and memories that users cannot inspect or delete.

Privacy governance must go beyond generic data protection. Users need control over what is sensed, remembered, inferred, shared, and executed. Systems should distinguish user-provided facts from model inferences, support correction and forgetting, minimize retention, and maintain audit logs for sensitive access or high-impact use. A Combodied Agent that remembers without user control threatens the very agency it is meant to support.

Consent, boundaries, and human override. Consent should specify what the agent may sense, remember, infer, share, recommend, and execute. Boundaries should define the agent’s relationship mode, domain authority, emotional posture, age restrictions, professional limits, and action permissions. Override should allow users to refuse, pause, correct, delete, revoke, or reverse the agent’s actions and memories.

These controls must be practical rather than symbolic. Users should not need expert knowledge to understand what the agent knows or can do. High-impact actions should require confirmation; high-risk states should trigger escalation; and the agent should know when to stop. The aim is not to make Combodied Agents permanently passive, but to keep their initiative bounded by consent, transparency, accountability, and meaningful human control.

## 8.3 Vulnerable and High-Stakes Settings

Risk depends on both user capacity and application stakes. This subsection brings together vulnerable populations and medical or mental-health settings because they require stricter defaults, clearer professional boundaries, and reliable escalation while still preserving dignity and participation.

Vulnerable users. Children, adolescents, older adults, patients, people with disabilities, people experiencing loneliness or mental-health crises, and users with cognitive decline may face heightened risk. The same interaction that is acceptable for a capable adult may be unsafe for a minor, a socially isolated user, or a person in crisis. Vulnerability may also be temporary: fatigue, grief, illness, stress, or financial pressure can reduce a user’s ability to evaluate influence. Safeguards should be tailored to user capacity, context, and domain. These may include age-appropriate interaction limits, stricter defaults, reduced personalization of persuasive content, caregiver or clinician escalation pathways, and stronger controls on intimate or dependency-forming interaction. Protection should not become pater nalism; vulnerable users still require dignity, participation, and meaningful control.

Medical and mental-health risks. Medical and mental-health applications require special care because errors can directly affect safety. Health-related Combodied Agents must distinguish wellness support, health education, clinical decision support, diagnosis, and treatment. A system that gives inappropriate reassurance, misses warning signs, or presents uncertain guidance as clinical authority may delay care or cause harm.

Mental-health contexts are particularly sensitive. Emotional-support agents may encounter self-harm, abuse, delusional beliefs, severe anxiety, or crisis states. Governance should require uncertainty communication, evidence grounding, crisis protocols, escalation to qualified professionals, and clear limits on the agent’s role. In high-risk situations, the safe behavior is often not better conversation, but handoff to human support.

Managing these risks is necessary but insufficient. Trustworthy Combodied Agents also require advances beyond stronger foundation models or larger context windows. The core open problems are to learn personal dynamics and intervention effects from sparse longitudinal evidence, translate uncertain predictions into agency-aligned support, maintain trusted personal infrastructure, and coordinate future systems around the user’s interests.

## 8.4 Longitudinal and Causal Learning

The first cross-cutting challenge is to learn how an individual changes, rather than merely retrieve what has previously been recorded. This requires joining longitudinal evidence, personal adaptation, and intervention-response learning without assuming that sparse observations reveal a complete or stable person.

Learning individual dynamics. Personal trajectories unfold across interacting timescales: mood may change within hours, habits across weeks, and capabilities, relationships, or health across years. Models must distinguish temporary states from persistent patterns, adapt population-level priors to limited and biased personal data, represent uncertainty, and remain correctable when the user or later evidence contradicts an inference.

Causal intervention learning. Observed improvement after an intervention does not establish that the intervention caused it. User motivation, hidden context, concurrent events, and selective engagement can confound reminders, coaching, emotional support, and escalation. Future research must combine intervention-response histories, counterfactual reasoning, and delayed outcomes to estimate when support helps, has no effect, or produces unintended harm.

Safe validation. Learning personal dynamics cannot rely on unconstrained experimentation, especially for children, older adults, patients, or people in crisis. Safe progress requires bounded simulation, retrospective and observational evidence, expert review, staged deployment, and prospective studies only when consent, monitoring, and escalation are appropriate. The challenge is to validate useful personal predictions while preventing the evaluation process itself from becoming an unsafe intervention.

## 8.5 Agency-Aligned Intervention

The second cross-cutting challenge is to translate uncertain personal predictions into support that improves long-term human outcomes without making the agent’s continued use its implicit objective. Objective design and intervention calibration must therefore be treated as one problem.

Long-horizon objectives. Wellbeing, capability, autonomy, safety, and social integration are multidimensional, delayed, and sometimes in tension with immediate satisfaction. Future systems need objectives that distinguish shortterm engagement from durable benefit, accommodate changing user values, and expose trade-offs rather than collapsing them into a single reward.

Calibrated intervention. The agent must decide when to remain silent, inform, remind, challenge, protect, execute, or escalate. Under-intervention may leave the user unsupported, while over-intervention may become intrusive, paternalistic, or dependency-forming. Research is needed on uncertainty-aware policies, user-adjustable initiative, relationship-sensitive authority, reversible actions, and feedback mechanisms that adapt intervention intensity without normalizing overreach.

## 8.6 Trusted Personal Infrastructure

Longitudinal support also requires an infrastructure in which personal intelligence can evolve without placing the user’s model, memory, or intervention authority beyond meaningful control. This challenge connects edge capability, continual adaptation, cloud collaboration, ownership, and security.

Local capability and safe evolution. Edge devices must support reliable perception, memory retrieval, personal dynamics estimation, and policy execution under limited computation, storage, and energy. At the same time, continual personalization must detect unsafe drift, preserve uncertainty, support rollback, and allow users to inspect, correct, pause, or reset learned state.

Ownership, collaboration, and security. Complex reasoning may still require cloud models, but routing should reveal what leaves the device, minimize disclosure, and preserve utility through purpose-limited summaries. Personal memories and model state should be portable across devices and providers, deletable by the user, and protected through encrypted storage, access control, trusted execution, and auditable synchronization.

## 8.7 Emerging Combodied Ecosystems

Future Combodied Agents may operate as interfaces to personal digital twins or as members of ecosystems of specialized agents. These directions share a coordination problem: personal models, recommendations, and actions must remain interpretable, conflict-aware, and governed around the user’s interests.

Personal digital twins. Health-oriented Combodied Agents may become the interaction and intervention layer for personal digital twins. A digital twin may model physiological, behavioral, or clinical trajectories; the Combodied Agent can translate that model into explanations, goal negotiation, daily guidance, and coordinated action. This integration could support chronic disease management, rehabilitation, prevention, and lifestyle intervention, but it also raises risks. Digital twins may be uncertain, incomplete, or clinically unvalidated. Combodied Agents must communicate uncertainty, avoid over-medicalizing everyday life, protect sensitive data, and clarify when professional oversight is required. Future research should connect interpretable personal models with safe user-facing intervention.

Multi-agent coordination. Future users may interact with multiple specialized companions: health companions, learning companions, workplace companions, financial guardians, emotional-support agents, and family coordination agents. These systems may offer useful specialization, but they also introduce conflicts. A workplace companion may encourage productivity, a health companion may recommend rest, and a financial guardian may constrain spending that another agent proposes. The research problem is user-centered coordination. Combodied ecosystems need mechanisms for priority setting, conflict resolution, and cross-agent consistency. Without such coordination, multiple helpful systems can collectively become confusing, invasive, or misaligned.

## 8.8 Cross-Cultural and Lifespan Futures

Combodied needs vary across cultures, languages, ages, social roles, and life stages. Norms around privacy, family involvement, emotional expression, authority, care, romance, and medical decision-making differ substantially. A recommendation that preserves agency in one context may be inappropriate or even harmful in another. Future Combodied Agents must be culture-sensitive and lifespan-aware. Children, adolescents, adults, and older adults require different interaction styles, safeguards, and developmental assumptions. Cross-cultural evaluation, localized value modeling, and participatory design will be essential. A person-centric agent cannot assume a single universal model of the person.

Taken together, these directions define a human-centered research agenda rather than a pursuit of autonomy for its own sake. Progress should be measured by whether Combodied Agents become more accurate and capable while remaining corrigible, culturally situated, accountable, and aligned with the user’s long-term agency.

## 9 Conclusion

This paper proposed Combodied Agents as a human-centric Agentic AI paradigm and developed its closed-loop framework, taxonomy, deployment perspective, and evaluation agenda. Its distinctive challenge is not simply to personalize or automate, but to support human trajectories without undermining autonomy, capability, safety, or relationships.

The central design principle is that a Combodied Agent should act with the user in ways that preserve and strengthen long-term agency. Progress should be judged by whether people remain able to understand, choose, correct, recover, develop capability, sustain human relationships, and live according to their evolving values.

## References

[1] Zheng Zhang, Ryuichi Takanobu, Qi Zhu, MinLie Huang, and XiaoYan Zhu. Recent advances and challenges in task-oriented dialog systems. Science China Technological Sciences, 63(10):2011–2027, 2020.

[2] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023.

[3] Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettle moyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In Advances in Neural Information Processing Systems, volume 36, pages 68539–68551, 2023.

[4] Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, 2024.

[5] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dobin Hui, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. In Advances in Neural Information Processing Systems, 2024.

[6] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world github issues? In International Conference on Learning Representations, 2024.

[7] Anthony Brohan, Yevgen Chebotar, Chelsea Finn, Karol Hausman, Alexander Herzog, Daniel Ho, Julian Ibarz, Alex Irpan, Eric Jang, Ryan Julian, et al. Do as i can, not as i say: Grounding language in robotic affordances. In Conference on robot learning, pages 287–318. Pmlr, 2023.

[8] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.

[9] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

[10] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. Agentbench: Evaluating llms as agents. In International Conference on Learning Representations, 2024.

[11] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Ji-Rong Wen. A survey on large language model based autonomous agents. Frontiers ofComputer Science, 18(6):186345, 2024.

[12] Bang Liu. Human-centered agents: From delegation to human growth. https://github.com/chatsci Human-Centered-Agent, 2025. Accessed: 2026-07-15.

[13] Thinking Machines Lab. The future worth building is human. https://thinkingmachines.ai/blog/ the-future-worth-building-is-human/, July 2026. Accessed: 2026-07-15.

[14] Hao-Ping Lee, Advait Sarkar, Lev Tankelevitch, Ian Drosos, Sean Rintel, Richard Banks, and Nicholas Wilson. The impact of generative ai on critical thinking: Self-reported reductions in cognitive effort and confidence effects from a survey of knowledge workers. In Proceedings of the 2025 CHI conference on human factors in computing systems, pages 1–22, 2025.

[15] Helena Vasconcelos, Matthew Jorke, Madeleine Grunde-McLaughlin, Tobias Gerstenberg, Michael S Bern-¨ stein, and Ranjay Krishna. Explanations can reduce overreliance on ai systems during decision-making. Pro ceedings ofthe ACM on human-computer interaction, 7(CSCW1):1–38, 2023.

[16] Auren R. Liu, Pat Pataranutaporn, and Pattie Maes. The heterogeneous effects of ai companionship: An empirical model of chatbot usage and loneliness and a typology of user archetypes. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, volume 8, pages 1585–1597, 2025.

[17] Jason Phang, Michael Lampe, Lama Ahmad, Sandhini Agarwal, Cathy Mengying Fang, Auren R. Liu, Valde mar Danry, Eunhae Lee, Samantha W. T. Chan, Pat Pataranutaporn, and Pattie Maes. Investigating affective use and emotional well-being on chatgpt. arXiv preprint arXiv:2504.03888, 2025.

[18] Cathy Mengying Fang, Auren R. Liu, Valdemar Danry, Eunhae Lee, Samantha W. T. Chan, Pat Pataranutaporn, Pattie Maes, Jason Phang, Michael Lampe, Lama Ahmad, and Sandhini Agarwal. How ai and human behaviors shape psychosocial effects of extended chatbot use: A longitudinal randomized controlled study. arXiv preprint arXiv:2503.17473, 2025.

[19] Christopher Starke, Alfio Ventura, Clara Bersch, Meeyoung Cha, Claes de Vreese, Philipp Doebler, Mengchen Dong, et al. Risks and protective measures for synthetic relationships. Nature Human Behaviour, 8:1834–1836, 2024.

[20] Common Sense Media. Talk, trust, and trade-offs: How and why teens use AI companions. Technical report, Common Sense Media, 2025.

[21] Ben Shneiderman. Human-centered artificial intelligence: Reliable, safe & trustworthy. International Journal ofHuman–Computer Interaction, 36(6):495–504, 2020.

[22] Saleema Amershi, Dan Weld, Mihaela Vorvoreanu, Adam Fourney, Besmira Nushi, Penny Collisson, Jina Suh, Shamsi Iqbal, Paul N Bennett, Kori Inkpen, et al. Guidelines for human-ai interaction. In Proceedings of the 2019 chi conference on humanfactors in computing systems, pages 1–13, 2019.

[23] Yujia Lin, Liming Chen, Aftab Ali, Christopher Nugent, Ian Cleland, Rongyang Li, Jianguo Ding, and Huansheng Ning. Human digital twin: A survey. Journal ofCloud Computing, 13(1):131, 2024.

[24] Martin Wolfgang Lauer-Schmaltz, Philip Cash, John Paulin Hansen, and Anja M. Maier. Towards the human digital twin: Definition and design – a survey. arXiv preprint arXiv:2402.07922, 2024.

[25] Evangelia Katsoulakis, Qi Wang, Huanmei Wu, Leili Shahriyari, Richard Fletcher, Jinwei Liu, Luke Achenie, Hongfang Liu, Pamela Jackson, Ying Xiao, et al. Digital twins for health: a scoping review. NPJ digital medicine, 7(1):77, 2024.

[26] Kaushik P Venkatesh, Marium M Raza, and Joseph C Kvedar. Health digital twins as tools for precision medicine: Considerations for computation, implementation, and regulation. NPJ digital medicine, 5(1):150, 2022.

[27] Vahid Riahi, Ibrahima Diouf, Sankalp Khanna, Justin Boyle, and Hamed Hassanzadeh. Digital twins for clinical and operational decision-making: scoping review. Journal of Medical Internet Research, 27:e55015, 2025.

[28] OpenAI. Computer-using agent. https://openai.com/index/computer-using-agent/, 2025. Accessed 2026-06-18.

[29] Anthropic. Introducing computer use, a new claude 3.5 sonnet, and claude 3.5 haiku. https://www. anthropic.com/news/3-5-models-and-computer-use, 2024. Accessed: 2026-06-07.

[30] Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H Laradji, Manuel Del Verme, Tom Marty, Leo´ Boisvert, Megh Thakkar, Quentin Cappart, David Vazquez, et al. Workarena: How capable are web agents at solving common knowledge work tasks? arXiv preprint arXiv:2403.07718, 2024.

[31] Sirui Hong, Yizhang Lin, Bang Liu, Bangbang Liu, Binhao Wu, Ceyao Zhang, Danyang Li, Jiaqi Chen, Jiayi Zhang, Jinlin Wang, et al. Data Interpreter: An LLM agent for data science. In Findings ofthe Associationfo Computational Linguistics: ACL 2025, pages 19796–19821, 2025.

[32] Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Mart´ın-Mart´ın, Chen Wang, Gabrael Levine, Wensi Ai, Benjamin Martinez, et al. BEHAVIOR-1K: A human-centered, embodied AI benchmark with 1,000 everyday activities and realistic simulation. arXiv preprint arXiv:2403.09227, 2024.

[33] Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, pages 1–22, 2023.

[34] Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. MemGPT: Towards LLMs as operating systems. arXiv preprint arXiv:2310.08560, 2023.

[35] OpenAI. Memory and new controls for chatgpt. https://openai.com/index/ memory-and-new-controls-for-chatgpt/, 2025. Accessed: 2026-06-07.

[36] Zhehao Zhang, Ryan A. Rossi, Branislav Kveton, Yijia Shao, Diyi Yang, Hamed Zamani, Franck Dernoncourt, Joe Barrow, Tong Yu, Sungchul Kim, et al. Personalization of large language models: A survey. Transactions on Machine Learning Research, 2025.

[37] Yue Xu, Qian Chen, Zizhan Ma, Dongrui Liu, Wenxuan Wang, Xiting Wang, Li Xiong, and Wenjie Wang. Toward personalized LLM-powered agents: Foundations, evaluation, and future directions. arXiv preprint arXiv:2602.22680, 2026.

[38] Xiaopeng Li, Pengyue Jia, Derong Xu, Yi Wen, Yingyi Zhang, Wenlin Zhang, Wanyu Wang, Yichao Wang, Zhaocheng Du, Xiangyang Li, et al. A survey of personalization: From RAG to agent. arXiv preprint arXiv:2504.10147, 2025.

[39] Timothy W. Bickmore, Amanda Gruber, and Rosalind W. Picard. Establishing the computer-patient working alliance in automated health behavior change interventions. Patient Education and Counseling, 59(1):21–30, 2005.

[40] Auren R. Liu, Pat Pataranutaporn, and Pattie Maes. Chatbot companionship: A mixed-methods study of companion chatbot usage patterns and their relationship to loneliness in active users. arXiv preprint arXiv:2410.21596, 2024.

[41] Renwen Zhang, Han Li, Han Meng, Jinyuan Zhan, Hongyuan Gan, and Yi-Chieh Lee. The dark side of ai companionship: A taxonomy of harmful algorithmic behaviors in human-ai relationships. In Proceedings of the 2025 CHI conference on humanfactors in computing systems, pages 1–17, 2025.

[42] Mike A Merrill, Akshay Paruchuri, Naghmeh Rezaei, Geza Kovacs, Javier Perez, Yun Liu, Erik Schenck, Nova Hammerquist, Jake Sunshine, Shyam Tailor, et al. Transforming wearable data into personal health insights using large language model agents. Nature Communications, 17(1):1143, 2026.

[43] Justin Cosentino, Anastasiya Belyaeva, Xin Liu, Nicholas A Furlotte, Zhun Yang, Chace Lee, Erik Schenck, Yojan Patel, Jian Cui, Logan Douglas Schneider, et al. Towards a personal health large language model. arXiv preprint arXiv:2406.06474, 2024.

[44] Matthew Jorke, Shardul Sapkota, Lyndsea Warkenthien, Niklas Vainio, Paul Schmiedmayer, Emma Brunskill,¨ and James A Landay. Gptcoach: towards llm-based physical activity coaching. In Proceedings ofthe 2025 CHI conference on human factors in computing systems, pages 1–46, 2025.

[45] Rose E. Wang, Ana T. Ribeiro, Carly D. Robinson, Susanna Loeb, and Dora Demszky. Tutor CoPilot: A human-AI approach for scaling real-time expertise. arXiv preprint arXiv:2410.03017, 2025.

[46] New York State Office for the Aging. ElliQ proactive care companion initiative. https://aging.ny. gov/elliq-proactive-care-companion-initiative, 2026. Accessed 2026-06-18.

[47] Filippos Bellos, Yayuan Li, Cary Shu, Ruey Day, Jeffrey Siskind, and Jason Corso. Towards effective humanin-the-loop assistive ai agents. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2534–2543, 2025.

[48] Apple. Introducing Apple Intelligence for iPhone, iPad, and Mac. https://www.apple.com/ apple-intelligence/, 2024. Accessed 2026-06-18.

[49] Google Android Developers. Gemini Nano on Android. https://developer.android.com/ai gemini-nano, 2025. Accessed 2026-06-18.

[50] Jonghyun Chung and Sanket Badhe. Local is not a sufficient privacy boundary: Governing os-integrated on device ai. arXiv preprint arXiv:2606.10173, 2026.

[51] Chunlin Tian, Dongqi Cai, Wanru Zhao, and Nicholas D Lane. Beyond scaling: Agents are heading to the edge. arXiv preprint arXiv:2605.18535, 2026.

[52] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2Web: Towards a generalist agent for the web. In Advances in Neural Information Processing Systems, volume 36, pages 28091–28114, 2023.

[53] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. ToolLLM: Facilitating large language models to master 16000+ real-world APIs. In International Conference on Learning Representations, volume 2024, pages 9695–9717, 2024.

[54] Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, Lewei Lu, Xiaosong Jia, Qiang Liu, Jifeng Dai, Yu Qiao, and Hongyang Li. Planning-oriented autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17853–17862, 2023.

[55] Sriram Yenamandra, Arun Ramachandran, Karmesh Yadav, Austin Wang, Mukul Khanna, Theophile Gervet, Tsung-Yen Yang, Vidhi Jain, Alexander William Clegg, John Turner, et al. HomeRobot: Open-vocabulary mobile manipulation. arXiv preprint arXiv:2306.11565, 2024.

[56] Abby O’Neill, Abdul Rehman, Abhinav Gupta, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, et al. Open X-Embodiment: Robotic learning datasets and RT-X models. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.

[57] Xavier Puig, Eric Undersander, Andrew Szot, Mikael Dallaire Cote, Tsung-Yen Yang, Ruslan Partsey, Ruta Desai, Alexander William Clegg, Michal Hlavac, So Yeon Min, et al. Habitat 3.0: A co-habitat for humans, avatars and robots. arXiv preprint arXiv:2310.13724, 2023.

[58] Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19724–19731, 2024.

[59] Pengfei Du. Memory for autonomous llm agents: Mechanisms, evaluation, and emerging frontiers. arXiv preprint arXiv:2603.07670, 2026.

[60] Zane Durante, Qiuyuan Huang, Naoki Wake, Ran Gong, Jae Sung Park, Bidipta Sarkar, Rohan Taori, Yusuke Noda, Demetri Terzopoulos, Yejin Choi, Katsushi Ikeuchi, Hoi Vo, Li Fei-Fei, and Jianfeng Gao. Agent AI: Surveying the horizons of multimodal interaction. arXiv preprint arXiv:2401.03568, 2024.

[61] Saul Shiffman, Arthur A Stone, and Michael R Hufford. Ecological momentary assessment. Annu. Rev. Clin. Psychol., 4(1):1–32, 2008.

[62] Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. Longmemeval: Benchmarking chat assistants on long-term interactive memory. arXiv preprint arXiv:2410.10813, 2025.

[63] Florian Eyben, Klaus R Scherer, Bjorn W Schuller, Johan Sundberg, Elisabeth Andr¨ e, Carlos Busso, Laurence Y´ Devillers, Julien Epps, Petri Laukka, Shrikanth S Narayanan, et al. The geneva minimalistic acoustic parameter set (gemaps) for voice research and affective computing. IEEE transactions on affective computing, 7(2):190– 202, 2015.

[64] Zheng Lian, Licai Sun, Yong Ren, Hao Gu, Haiyang Sun, Lan Chen, Bin Liu, and Jianhua Tao. MERBench: A unified evaluation benchmark for multimodal emotion recognition. arXiv preprint arXiv:2401.03429, 2024.

[65] Tadas Baltrusaitis, Peter Robinson, and Louis-Philippe Morency. Openface: an open source facial behaviorˇ analysis toolkit. In 2016 IEEE winter conference on applications of computer vision (WACV), pages 1–10. IEEE, 2016.

[66] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 18995– 19012, 2022.

[67] Somayeh Imani, Amay J Bandodkar, AM Vinu Mohan, Rajan Kumar, Shengfei Yu, Joseph Wang, and Patrick P Mercier. A wearable chemical–electrophysiological hybrid biosensing system for real-time health and fitness monitoring. Nature communications, 7(1):11650, 2016.

[68] Yubin Kim, Xuhai Xu, Daniel McDuff, Cynthia Breazeal, and Hae Won Park. Health-llm: Large language models for health prediction via wearable sensor data. arXiv preprint arXiv:2401.06866, 2024.

[69] Daniel Roggen, Alberto Calatroni, Mirco Rossi, Thomas Holleczek, Kilian Forster, Gerhard Tr¨ oster, Paul¨ Lukowicz, David Bannach, Gerald Pirkl, Alois Ferscha, et al. Collecting complex activity datasets in highly rich networked sensor environments. In 2010 Seventh international conference on networked sensing systems (INSS), pages 233–240. IEEE, 2010.

[70] Francisco Javier Ordo´nez and Daniel Roggen. Deep convolutional and lstm recurrent neural networks for˜ multimodal wearable activity recognition. Sensors, 16(1):115, 2016.

[71] Jianyuan Ni, Hao Tang, Syed Tousiful Haque, Yan Yan, and Anne H. H. Ngu. A survey on multimodal wearable sensor-based human action recognition. arXiv preprint arXiv:2404.15349, 2024.

[72] Nathan Eagle and Alex Pentland. Reality mining: sensing complex social systems. Personal and ubiquitous computing, 10(4):255–268, 2006.

[73] Rui Wang, Fanglin Chen, Zhenyu Chen, Tianxing Li, Gabriella Harari, Stefanie Tignor, Xia Zhou, Dror Ben-Zeev, and Andrew T Campbell. Studentlife: assessing mental health, academic performance and behavioral trends of college students using smartphones. In Proceedings of the 2014 ACM international joint conference on pervasive and ubiquitous computing, pages 3–14, 2014.

[74] Xuhui Zhou, Hao Zhu, Leena Mathur, Ruohong Zhang, Haofei Yu, Zhengyang Qi, Louis-Philippe Morency, Yonatan Bisk, Daniel Fried, Graham Neubig, et al. Sotopia: Interactive evaluation for social intelligence in language agents. In International Conference on Learning Representations, volume 2024, pages 40975–41019, 2024.

[75] Denzil Ferreira, Vassilis Kostakos, and Anind K Dey. Aware: mobile context instrumentation framework. Frontiers in ICT, 2:6, 2015.

[76] Joshua C Mandel, David A Kreda, Kenneth D Mandl, Isaac S Kohane, and Rachel B Ramoni. Smart on fhir: a standards-based, interoperable apps platform for electronic health records. Journal of the american medical informatics association, 23(5):899–908, 2016.

[77] Erica A Voss, Rupa Makadia, Amy Matcho, Qianli Ma, Chris Knoll, Martijn Schuemie, Frank J DeFalco, Ajit Londhe, Vivienne Zhu, and Patrick B Ryan. Feasibility and utility of applications of the common data model to multiple, disparate observational health databases. Journal of the American Medical Informatics Association, 22(3):553–564, 2015.

[78] Alistair EW Johnson, Lucas Bulgarelli, Lu Shen, Alvin Gayles, Ayad Shammout, Steven Horng, Tom J Pollard, Sicheng Hao, Benjamin Moody, Brian Gow, et al. Mimic-iv, a freely accessible electronic health record dataset. Scientific data, 10(1):1, 2023.

[79] Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. Large language models encode clinical knowledge. Nature, 620(7972):172–180, 2023.

[80] Yikuan Li, Shishir Rao, Jose Roberto Ayala Solares, Abdelaali Hassaine, Rema Ramakrishnan, Dexter Canoy,´ Yajie Zhu, Kazem Rahimi, and Gholamreza Salimi-Khorshidi. Behrt: transformer for electronic health records. Scientific reports, 10(1):7155, 2020.

[81] Laila Rasmy, Yang Xiang, Ziqian Xie, Cui Tao, and Degui Zhi. Med-bert: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction. NPJ digital medicine, 4(1):86, 2021.

[82] Mahima Pushkarna, Andrew Zaldivar, and Oddur Kjartansson. Data cards: Purposeful and transparent dataset documentation for responsible AI. arXiv preprint arXiv:2204.01075, 2022.

[83] Paul Pu Liang, Amir Zadeh, and Louis-Philippe Morency. Foundations and trends in multimodal machine learning: Principles, challenges, and open questions. arXiv preprint arXiv:2209.03430, 2022.

[84] David Ha and Jurgen Schmidhuber. World models. ¨ arXiv preprint arXiv:1803.10122, 2018.

[85] Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, Timothy Lillicrap, and David Silver. Master ing atari, go, chess and shogi by planning with a learned model. arXiv preprint arXiv:1911.08265, 2019.

[86] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.

[87] Georgianna Lin, Rencong Jiang, Noemie Elhadad, and Xuhai ‘Orson’ Xu. A framework for longitudinal health´ ai agents. Nature health, pages 1–10, 2026.

[88] Jingtao Ding, Yunke Zhang, Yu Shang, Yuheng Zhang, Zefang Zong, Jie Feng, Yuan Yuan, Hongyuan Su, Nian Li, Nicholas Sukiennik, Fengli Xu, and Yong Li. Understanding world or predicting future? a comprehensive survey of world models. arXiv preprint arXiv:2411.14499, 2024.

[89] Raffaele Ciriello, Uri Gal, and Ofir Turel. Not a silver bullet for loneliness: How attachment and age shape intimacy with AI companions. arXiv preprint arXiv:2602.12476, 2026.

[90] Ziv Ben-Zion, Paul Raffelhuschen, Max Zettl, Antonia L ¨ u¨ond, Achim Burrer, Philipp Homan, and Tobias R.¨ Spiller. Detecting and preventing harmful behaviors in AI companions: Development and evaluation of the SHIELD supervisory system. arXiv preprint arXiv:2510.15891, 2025.

[91] Prerna Juneja and Lika Lomidze. Persona-grounded safety evaluation of AI companions in multi-turn conver sations. arXiv preprint arXiv:2605.00227, 2026.

[92] Weisong Shi, Jie Cao, Quan Zhang, Youhuizi Li, and Lanyu Xu. Edge computing: Vision and challenges. IEEE internet ofthings journal, 3(5):637–646, 2016.

[93] Zhi Zhou, Xu Chen, En Li, Liekang Zeng, Ke Luo, and Junshan Zhang. Edge intelligence: Paving the last mile of artificial intelligence with edge computing. Proceedings ofthe IEEE, 107(8):1738–1762, 2019.

[94] Shuiguang Deng, Hailiang Zhao, Weijia Fang, Jianwei Yin, Schahram Dustdar, and Albert Y. Zomaya. Edge intelligence: The confluence of edge computing and artificial intelligence. IEEE Internet of Things Journal, 7(8):7457–7469, 2020.

[95] Apple Security Research. Private cloud compute: A new frontier for AI privacy in the cloud. https:// security.apple.com/blog/private-cloud-compute/, 2024. Accessed 2026-06-18.

[96] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

[97] Song Han, Huizi Mao, and William J. Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. In International Conference on Learning Representations, 2016.

[98] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

[99] Keivan Alizadeh, Iman Mirzadeh, Dmitry Belenko, Seyed Khatamifard, Minsik Cho, Carlo C. Del Mundo, Mohammad Rastegari, and Mehrdad Farajtabar. LLM in a flash: Efficient large language model inference with limited memory. arXiv preprint arXiv:2312.11514, 2023.

[100] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communicationefficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pages 1273– 1282. Pmlr, 2017.

[101] Peter Kairouz and H Brendan McMahan. Advances and open problems in federated learning. Foundations and trends in machine learning, 14(1-2):1–210, 2021.

[102] Alireza Salemi, Sheshera Mysore, Michael Bendersky, and Hamed Zamani. Lamp: When large language models meet personalization. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7370–7392, 2024.

[103] Ishita Kumar, Snigdha Viswanathan, Sushrita Yerra, Alireza Salemi, Ryan A Rossi, Franck Dernoncourt, Hanieh Deilamsalehy, Xiang Chen, Ruiyi Zhang, Shubham Agarwal, et al. Longlamp: A benchmark for personalized long-form text generation. arXiv preprint arXiv:2407.11016, 2024.

[104] Lawrence Jang, Ruslan Salakhutdinov, et al. iOSWorld: A benchmark for personally intelligent phone agents. arXiv preprint arXiv:2606.09764, 2026.

[105] Julian De Freitas, Noah Castelo, Ahmet Uguralp, and Zeliha Uguralp. Lessons from an app update at Replika AI: Identity discontinuity in human-AI relationships. arXiv preprint arXiv:2412.14190, 2024.

[106] Suhas BN, Andrew M Sherrill, Rosa I Arriaga, Chris W Wiese, and Saeed Abdullah. Ai safety training can be clinically harmful. arXiv preprint arXiv:2604.23445, 2026.

[107] Ian Steenstra, Paola Pedrelli, Weiyan Shi, Stacy Marsella, and Timothy W Bickmore. Assessing risks of large language models in mental health support: a framework for automated clinical ai red teaming. arXiv preprint arXiv:2602.19948, 2026.

[108] George Herbert Mead. Mind, self& society. University of Chicago press, 2015.

[109] Erving Goffman et al. The presentation of self in everyday life, volume 21. Harmondsworth London, 1978.

[110] Bruce J Biddle. Recent developments in role theory. Annual review ofsociology, 12(1):67–92, 1986.

[111] Kathleen Kara Fitzpatrick, Alison Darcy, and Molly Vierhile. Delivering cognitive behavior therapy to young adults with symptoms of depression and anxiety using a fully automated conversational agent (Woebot): A randomized controlled trial. JMIR Mental Health, 4(2):e19, 2017.