# Qwen-VLA Research Artifacts

First-principle probes of the paper's claims, run without access to the model weights (the GitHub repository is a README-only placeholder).

- `qwen-vla-researcher-checks.py` — PyTorch/sklearn tests for:
  - T2A masked flow-matching loss (Eq. 1–2)
  - Sigmoid-Normal vs Beta(1,3) timestep mass ratio
  - Prompt-brittleness toy probe on rephrased embodiment templates
  - Zero-padding action representation parameter savings
- `qwen-vla-researcher-checks-output.txt` — captured output of the above script.

Outputs are summarized in `../deep-dive.md` under "D4-RC: Researcher Checks".
