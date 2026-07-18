#!/usr/bin/env python3
"""
classify_landscape.py
=====================
Triangulation of arXiv:2607.14076 ("From Pixels to States") against recent
world-model papers covered in this workspace (papers/world-models/*.md).

The position paper organizes interactive game world modeling along FOUR
dimensions, grounded in the recurrent action->state->observation loop of
conventional game engines:

  D1 player action control        how player intent is represented
                                  (geometric trajectories | motor signals | semantic events)
  D2 game state dynamics          how world state is represented & updated
                                  (entangled in observations | learned latents | explicit descriptions)
  D3 state-observation persistence how consequences persist over long horizons
                                  (memory as stored observations | memory as estimates of the present)
  D4 real-time interactive generation  control latency vs consequence latency
                                  (reducing generation latency | reducing conditioning latency)

This script encodes our per-paper classification judgments (with evidence
quotes pulled from the local markdown caches) as structured data and emits:
  - classification.json   (machine-readable)
  - classification.md     (human-readable table)
  - taxonomy_matrix.png   (figure: paper x dimension, colored by family)

The judgments are the content; the script only formats them. Each entry was
made by reading the full cached text and locating concrete evidence
(action space, memory mechanism, fps numbers, state handling).
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Family color codes for the matrix figure (position-paper families + "none")
FAMILIES = {
    # D1
    "geometric trajectories": "#8ecae6",
    "motor signals": "#90be6d",
    "semantic events": "#f9c74f",
    "motor + semantic (hybrid)": "#b5838d",
    # D2
    "implicit (state entangled in observations)": "#e76f51",
    "learned latent state (recurrent)": "#2a9d8f",
    "explicit symbolic/textual state": "#6a4c93",
    # D3
    "stored observations (past-copy memory)": "#577590",
    "estimates of the present (updating memory)": "#f4a261",
    "running-summary latent memory": "#43aa8b",
    "none (rolling window only)": "#d62828",
    # D4
    "real-time (generation-latency reduction)": "#606c38",
    "near/offline (not play-rate)": "#9d6b53",
}

PAPERS = [
    {
        "paper": "MIRA (Multiplayer Interactive WM)",
        "arxiv": "2607.05352",
        "domain": "game (Rocket League, 4-player)",
        "D1": {
            "family": "motor signals",
            "summary": "Per-player controller action streams (9 discrete high-frequency actions); multiplayer conditioning must attribute scene changes to the correct agent.",
            "evidence": "'conditions on the action streams of multiple agents, learning to attribute changes in the scene to the correct player'; action probe on frozen DINOv3 features reaches 0.84 mAP over the nine actions.",
        },
        "D2": {
            "family": "implicit (state entangled in observations)",
            "summary": "5B latent diffusion over a fixed rolling window of T=20 latents (~2 s at 10 Hz). No separate state variable. Privileged game state (ball/car, 50-dim) is logged ONLY as probe ground truth.",
            "evidence": "'the model holds a fixed-size window of T = 20 latents, rolling the window by 1 each step'; 'we also log privileged game state... instead it serves as ground truth for evaluation, letting us probe'.",
        },
        "D3": {
            "family": "none (rolling window only)",
            "summary": "No persistent memory beyond the rolling context. Hour-scale stability comes from diffusion-forcing-style training (noised history), not from any carried state. Probe shows rolled-out car/ball positions track true physics LOCALLY.",
            "evidence": "'a game-state probe further confirms that the rolled-out car and ball positions track the true physics'; 'distributional quality holds steady out to five minutes' — quality, not state persistence.",
        },
        "D4": {
            "family": "real-time (generation-latency reduction)",
            "summary": "20 fps on a single B200: few-step diffusion distillation + streaming KV-cache + rolling context.",
            "evidence": "'generating 20 frames per second on a single Nvidia B200 GPU'.",
        },
        "note": "Purest modern example of the state-free design the position paper critiques — and the authors know it: the probe machinery exists precisely because state is implicit and must be verified post-hoc.",
    },
    {
        "paper": "ActWorld",
        "arxiv": "2606.17730",
        "domain": "general indoor scenes (synthetic interaction data)",
        "D1": {
            "family": "motor + semantic (hybrid)",
            "summary": "Per-frame WASD/arrow keyboard-mouse control PLUS per-chunk semantic captions (40 action categories, 6 interaction-phase labels via CoT VLM annotation).",
            "evidence": "'under per-frame keyboard and mouse control (WASD + arrow keys)'; 'annotate each chunk offline with a dedicated description and the structured labels'.",
        },
        "D2": {
            "family": "implicit (state entangled in observations)",
            "summary": "No explicit state; object state lives only in video latents + memory tokens. The paper itself diagnoses 'action-forgetting' as a memory-design failure, not a missing state variable.",
            "evidence": "'recency-biased history compression in existing world models discards the event-transition frames that causally determine subsequent object states'.",
        },
        "D3": {
            "family": "estimates of the present (updating memory)",
            "summary": "Hierarchical action-aware memory: local bank routes frames by interaction importance (EAFR, contact-phase prior beats recency); persistent bank carries event-update and object-identity (DINOv3 anchor) tokens beyond the eviction horizon.",
            "evidence": "'a persistent memory bank that maintains event-update and object-identity tokens across long rollouts'; pixels-to-states itself cites ActWorld in its 'memory as estimates of the present' family.",
        },
        "D4": {
            "family": "real-time (generation-latency reduction)",
            "summary": "Chunk-AR, 3-step adversarial DMD distillation (Helios recipe); 33-frame chunks (~1.4 s @24 fps) 'land in a fraction of a second'. Real-time claimed; exact FPS not reported.",
            "evidence": "'an adversarial DMD-style distillation that reduces the 50-step teacher to a 3-step generator'.",
        },
        "note": "Strongest counterexample-adjacent system: it moves memory from past-copy toward present-estimate — but the 'state' is still unverifiable latent tokens, and update decisions are left to learned saliency, exactly the ungrounded-update concern the position paper raises.",
    },
    {
        "paper": "Kairos",
        "arxiv": "2606.16533",
        "domain": "physical AI / robot + game inputs",
        "D1": {
            "family": "motor + semantic (hybrid)",
            "summary": "Joint World-Action MoT: Video DiT + smaller Action DiT generates future action tokens; inputs listed as camera control, natural language, keyboard/mouse, trajectory directives.",
            "evidence": "'The Action DiT predicts future action tokens... approximately one-fifth of the Video DiT'; 'camera control, natural language commands, keyboard/mouse operations, or trajectory directives'.",
        },
        "D2": {
            "family": "learned latent state (recurrent)",
            "summary": "Gated Linear Attention state matrix S_t acts as an explicit-architecture, continuously carried latent state with gated delta update; formal bounds claim limited error accumulation. But the state is uninterpretable and not rule-grounded.",
            "evidence": "'The Gated Linear Attention (GLA) state matrix S_t acts as a compressed latent memory'; 'the GLA mechanism, with its gated delta updates, serves as a persistent memory that tracks the world state over long horizons'.",
        },
        "D3": {
            "family": "running-summary latent memory",
            "summary": "SWA (local) + dilated SWA (mid) + GLA (global contractive memory); claims object permanence and 'Necessity of Persistent Latent States' (App. B.2).",
            "evidence": "'gated linear attention maintains persistent global memory... mathematically guaranteeing state propagation across extended horizons'.",
        },
        "D4": {
            "family": "real-time (generation-latency reduction)",
            "summary": "4-step distilled 480P model reaches real-time on A800; 720P 5 s clip costs 43 s on 1 GPU / 9 s on 4 GPUs — deployment co-designed but not 720p play-rate.",
            "evidence": "'480P video generation on the Nvidia A800 achieves real-time performance'; '43 seconds on 1 GPU and 9 seconds on 4 GPUs' (720P, 5 s).",
        },
        "note": "The most serious challenge to 'state stays implicit': Kairos carries a persistent recurrent state and even proves bounds about it. Yet it matches the position paper's 'learned latents' family verdict: no interpretability, and unreliable for visually-invisible rule variables (remaining HP) since the state is trained by visual prediction only.",
    },
    {
        "paper": "DreamX-World 1.0",
        "arxiv": "2606.16993",
        "domain": "photorealistic + game-style + stylized",
        "D1": {
            "family": "motor + semantic (hybrid)",
            "summary": "Camera trajectories via E-PRoPE (projective positional encoding, ~30% latency cut vs PRoPE) + composable multi-entity Event Instruction Tuning.",
            "evidence": "'Event Instruction Tuning adds composable event control'; 'E-PRoPE... retaining comparable trajectory-following performance while reducing inference latency by approximately 30%'.",
        },
        "D2": {
            "family": "implicit (state entangled in observations)",
            "summary": "Wan2.2-based few-step autoregressive generator; no separate state representation anywhere in the stack.",
            "evidence": "'initialized from Wan2.2... converted into a few-step autoregressive world model using causal forcing, DMD-style distillation, and long-rollout training'.",
        },
        "D3": {
            "family": "stored observations (past-copy memory)",
            "summary": "Memory-Conditioned Scene Persistence: retrieves earlier views by camera geometry; residual recycling tolerates imperfect memory latents. Exactly the past-copy family — good for revisits of static layout, silent on irreversible change.",
            "evidence": "'Memory-Conditioned Scene Persistence retrieves earlier views through camera-geometry-based retrieval, while residual recycling makes the conditioning path less sensitive to imperfect memory latents'.",
        },
        "D4": {
            "family": "real-time (generation-latency reduction)",
            "summary": "Up to 16 FPS on 8x RTX 5090 via mixed-precision DiT, residual reuse, 75%-pruned VAE decode, async pipeline parallelism.",
            "evidence": "'reaches up to 16 FPS on eight RTX 5090 GPUs'.",
        },
        "note": "Textbook illustration of the position paper's stored-observation critique: geometry-retrieved memory guarantees the past reappears faithfully — including, potentially, a past that should no longer exist (the 'skill demolishes a building, memory restores it intact' failure).",
    },
    {
        "paper": "GigaWorld-1",
        "arxiv": "2607.02642",
        "domain": "robot manipulation (policy evaluator)",
        "D1": {
            "family": "geometric trajectories",
            "summary": "Explicit spatially-aligned control maps — EE-pose map (head view) + ray maps (wrist views) — channel-concatenated with the noisy latent; best of 4 action encodings in controlled comparison.",
            "evidence": "'the strongest result comes from channel-concatenated control maps... a unified pixel-aligned representation derived from calibrated robot and camera geometry'.",
        },
        "D2": {
            "family": "implicit (state entangled in observations)",
            "summary": "No symbolic state; headline finding is that evaluator quality is dominated by long-horizon action-faithful consistency, i.e., implicit-state fidelity over rollouts.",
            "evidence": "'evaluator quality is dominated by long-horizon, action-faithful rollout consistency rather than short-term visual realism'.",
        },
        "D3": {
            "family": "stored observations (past-copy memory)",
            "summary": "Hierarchical history buffer: a NEVER-EVICTED first-frame anchor + short/mid/long-range memories. The anchor freezes initial scene identity — the polar opposite of an updating state.",
            "evidence": "'the anchor is never evicted during memory updates, each generation step retains access to the original appearance statistics'; 'reliable evaluators require persistent memory for long-horizon rollout' (Finding 10).",
        },
        "D4": {
            "family": "near/offline (not play-rate)",
            "summary": "Offline closed-loop evaluator; throughput not a design goal (324k simulated rollouts run offline).",
            "evidence": "'closed-loop rollout in world models... until task termination' — no real-time claim anywhere.",
        },
        "note": "Independently confirms, with 324k rollouts of evidence, the position paper's core empirical premise: long-horizon STATE consistency (not visual realism) is what makes a world model useful — yet its own answer is past-copy memory, not an explicit evolving state.",
    },
    {
        "paper": "BadWAM (attack analysis)",
        "arxiv": "2607.15207",
        "domain": "robot WAMs (LIBERO, RoboTwin)",
        "D1": {
            "family": "motor signals",
            "summary": "WAMs output action chunks a_{t:t+H-1} conditioned on observation + instruction (+ imagined future).",
            "evidence": "'its world-prediction module first imagines a future, and its action module then produces an action sequence conditioned on that imagination'.",
        },
        "D2": {
            "family": "learned latent state (recurrent)",
            "summary": "The 'state' is the imagined latent/decoded future z_{t+1:t+K} — and the paper shows it can be adversarially desynchronized from the executed action.",
            "evidence": "'the model may still produce plausible future imaginations, yet execute actions that cause the task to fail' (96.5% -> 43.1% success under action-only attack).",
        },
        "D3": {
            "family": "none (rolling window only)",
            "summary": "Not a persistence mechanism paper; chunk-horizon imagination only.",
            "evidence": "n/a",
        },
        "D4": {
            "family": "near/offline (not play-rate)",
            "summary": "Runs at policy-eval rate; real-time generation not the subject.",
            "evidence": "n/a",
        },
        "note": "Complements the position paper from the safety side: even when a model carries an explicit imagined future, nothing BINDS action outcomes to that state — the action-state-observation loop is open. Implicit state is not just forgetful; its coupling to action is unverifiable.",
    },
]


def to_json():
    return {
        "lens_source": "arXiv:2607.14076 From Pixels to States (Alaya Lab, 2026-07-15)",
        "dimensions": {
            "D1": "player action control",
            "D2": "game state dynamics",
            "D3": "state-observation persistence",
            "D4": "real-time interactive generation",
        },
        "papers": PAPERS,
    }


def to_md():
    lines = [
        "# WAM landscape classified on the Pixels-to-States four-dimension lens",
        "",
        "Lens: arXiv:2607.14076. D1 = player action control, D2 = game state dynamics,",
        "D3 = state-observation persistence, D4 = real-time interactive generation.",
        "",
        "| Paper (arXiv) | D1 action control | D2 state dynamics | D3 persistence | D4 real-time |",
        "|---|---|---|---|---|",
    ]
    for p in PAPERS:
        row = [f"**{p['paper']}** ({p['arxiv']})"]
        for d in ("D1", "D2", "D3", "D4"):
            row.append(p[d]["family"])
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    for p in PAPERS:
        lines.append(f"## {p['paper']} — arXiv:{p['arxiv']} ({p['domain']})")
        for d, name in (("D1", "action control"), ("D2", "state dynamics"),
                        ("D3", "persistence"), ("D4", "real-time")):
            e = p[d]
            lines.append(f"- **{d} {name}** — *{e['family']}*: {e['summary']}")
            lines.append(f"  - evidence: {e['evidence']}")
        lines.append(f"- **verdict note**: {p['note']}")
        lines.append("")
    return "\n".join(lines)


def to_png(outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    dims = ("D1", "D2", "D3", "D4")
    dim_titles = [
        "D1 action\ncontrol",
        "D2 state\ndynamics",
        "D3 persistence\n(memory)",
        "D4 real-time\ngeneration",
    ]
    n_p, n_d = len(PAPERS), len(dims)
    fig, ax = plt.subplots(figsize=(13.5, 0.95 * n_p + 2.2))
    ax.set_xlim(0, n_d)
    ax.set_ylim(0, n_p + 1)
    ax.invert_yaxis()

    for j, t in enumerate(dim_titles):
        ax.text(j + 0.5, 0.5, t, ha="center", va="center", fontsize=11, weight="bold")

    for i, p in enumerate(PAPERS):
        ax.text(-0.06, i + 1.5, f"{p['paper']}\n{p['arxiv']}", ha="right", va="center", fontsize=10)
        for j, d in enumerate(dims):
            fam = p[d]["family"]
            color = FAMILIES[fam]
            ax.add_patch(Rectangle((j + 0.03, i + 1.08), 0.94, 0.84, facecolor=color,
                                   edgecolor="white", lw=1.5))
            short = fam if len(fam) <= 34 else fam.replace(" (", "\n(")
            ax.text(j + 0.5, i + 1.5, short, ha="center", va="center", fontsize=7.6,
                    color="white", weight="bold", wrap=True)

    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # legend
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=l, ) for l, c in FAMILIES.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Recent world-action models on the Pixels-to-States four-dimension lens\n"
                 "(classification by this investigation; colors = position-paper families)",
                 fontsize=12, y=0.99)
    fig.savefig(outpath, dpi=170, bbox_inches="tight")
    print("wrote", outpath)


if __name__ == "__main__":
    with open(os.path.join(HERE, "classification.json"), "w") as f:
        json.dump(to_json(), f, indent=2)
    with open(os.path.join(HERE, "classification.md"), "w") as f:
        f.write(to_md())
    to_png(os.path.join(HERE, "taxonomy_matrix.png"))
    print("wrote classification.json / classification.md")
