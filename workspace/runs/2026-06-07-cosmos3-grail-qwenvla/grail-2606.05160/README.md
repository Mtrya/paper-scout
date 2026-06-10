# GRAIL Evidence

This thread preserves researcher-check scripts and captured output for the GRAIL deep dive in the June 7 report.

- `code/run_checks.py` reimplements loss-term math from `grail/optimization/loss_terms.py`, parses `configs/recon_4dhoi/pickup_smplx.yaml`, reproduces filtering logic, and loads trajectory samples from the Hugging Face dataset.
- `code/check_output.txt` is the captured output from `code/run_checks.py`.
