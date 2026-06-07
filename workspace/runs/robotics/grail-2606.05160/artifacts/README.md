# GRAIL Research Artifacts

Standalone verification of the reconstruction loss math and config audit, run without the full Docker environment.

- `run_checks.py` — reimplements loss-term math from `grail/optimization/loss_terms.py`, parses `configs/recon_4dhoi/pickup_smplx.yaml`, reproduces filtering logic, and loads real trajectory samples from the HuggingFace dataset.
- `check_output.txt` — captured output of the above script.

Outputs are summarized in `../deep-dive.md` under "D4.5: Researcher Checks".
