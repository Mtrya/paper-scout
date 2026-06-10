# Qwen-VLA Evidence

This thread preserves first-principles probes for the Qwen-VLA deep dive in the June 7 report. The paper did not provide usable implementation artifacts at the time of the run, so the checks reconstruct and stress-test claims from the paper.

- `code/qwen-vla-researcher-checks.py` contains PyTorch and scikit-learn probes for the T2A masked flow-matching loss, timestep sampling mass, prompt brittleness, and zero-padding parameter savings.
- `code/qwen-vla-researcher-checks-output.txt` is the captured output from the script.
