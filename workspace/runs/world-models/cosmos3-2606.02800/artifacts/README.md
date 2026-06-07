# Cosmos 3 Research Artifacts

Scripts run to test claims in the paper and the released code.

- `check_mot_params.py` — stand-in `PackedAttentionMoT` + `MoTDecoderLayer` using the real Qwen3-VL-8B config to quantify the parameter overhead of the dual-pathway design.
- `check_domain_aware_linear.py` — verifies `DomainAwareLinear` dispatch and parameter overhead for per-embodiment action projections.
- `check_action_representation.py` — round-trip test of the repo's `pose_utils.py` SE(3) → rot6d relative action vector → SE(3).

Outputs are summarized in `../deep-dive.md` under "Researcher Checks".
