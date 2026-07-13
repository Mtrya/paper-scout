# MIRA — Multiplayer Interactive World Models with Representation Autoencoders

**Paper:** arXiv:2607.05352v2 [cs.CV], 7 Jul 2026  
**Thread question:** How does MIRA build a real-time, multi-agent world model for a fast 3D physics game, and which design choices actually matter for stability and controllability?

## What was done

1. **Read the paper end-to-end**, focusing on the representation autoencoder, the latent diffusion world model, multiplayer conditioning, and the physical-understanding evaluations.
2. **Cloned the public repo** (`https://github.com/mira-wm/mira`, HEAD `db25448`) into `code/mira` and inspected the data pipeline, codec, world-model architecture, training configs, and inference path.
3. **Wrote a small config probe** (`code/inspect_mira_codec.py`) that reconstructs the published codec architecture and action vocabulary from the released YAML configs and source without downloading weights.
4. **Compiled quantitative findings** from the paper’s ablations into tables and noted where claims need more evidence (e.g., "hours" of stability, human-play generalization, downstream RL validation).

## How to rerun the probe

```bash
cd code/mira
python3 ../../runs/2026-07-13-lingbot-kinematic-memory/multiplayer-world-models/code/inspect_mira_codec.py .
```

## Key evidence preserved

- `memo.md` — full research memo with mechanism, code pointers, numbers, novelty analysis, and limitations.
- `code/inspect_mira_codec.py` — runnable probe summarizing the RAEv2 codec config and action vocabulary.

## Bottom line

MIRA is a strong engineering demonstration that pretrained-feature representation autoencoders, diffusion forcing, and tiled multi-agent conditioning combine into a stable real-time multiplayer simulator. The most important open questions are whether the recipe transfers beyond Rocket League, how it behaves under human (not bot) action distributions, and whether it can train agents that transfer to the real game.
