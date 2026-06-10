# Cosmos 3 Evidence

This thread preserves researcher-check scripts for the Cosmos 3 deep dive in the June 7 report.

- `code/check_mot_params.py` uses a stand-in `PackedAttentionMoT` and `MoTDecoderLayer` with the released Qwen3-VL-8B configuration to quantify the dual-pathway parameter overhead.
- `code/check_domain_aware_linear.py` verifies `DomainAwareLinear` dispatch and parameter overhead for per-embodiment action projections.
- `code/check_action_representation.py` round-trips the repository's SE(3) action representation through relative action vectors and back.
