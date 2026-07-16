"""
Sanity-check TerraZero's throughput claim against its reported training configuration.

Paper reports:
  - 1.3M agent-steps/sec on a single server GPU
  - 512 total agents per GPU (Table 14)
  - 32 environments per GPU, vectorized batch size 4 (so 32*4 = 128 controlled agents?)
  - Episode length 256 steps (25.6 s)
  - Policy timestep 0.1 s

We cannot verify the actual throughput without the code, but we can check whether
1.3M SPS is plausible given the observation/action sizes and a modest forward-pass budget.
"""

# Observation dimensions (from Appendix C)
EGO_DIM = 1 + 2 + 1 + 1 + 1 + 2 + 1 + 1 + 2 + 3 + 19 + 4  # ~38
PARTNER_DIM = 8
ROAD_DIM = 7
TRAFFIC_DIM = 12
MAX_PARTNERS = 20
MAX_ROAD = 200
MAX_TRAFFIC = 16

obs_bytes_fp16 = (
    EGO_DIM +
    PARTNER_DIM * MAX_PARTNERS +
    ROAD_DIM * MAX_ROAD +
    TRAFFIC_DIM * MAX_TRAFFIC
) * 2  # fp16

# If buffers are sized to actual controlled agents, not padded to max, effective bytes lower.
# Paper says "sizes its buffers to the controlled agents actually present".
# Default training: 512 agents per GPU total; assume ~1/3 are policy-controlled in a dense scene? Hard to know.
# We just report the max theoretical observation bytes per step.

print(f"Max theoretical obs bytes per agent (all budgets full, fp16): {obs_bytes_fp16:,}")
print(f"  = {obs_bytes_fp16/1024:.1f} KB/agent/step")

# At 1.3M agent-steps/sec, observation bandwidth from CPU to GPU:
bandwidth_gbps = obs_bytes_fp16 * 1.3e6 / 1e9
print(f"\nIf every agent carried the full obs budget:")
print(f"  H2D observation bandwidth @ 1.3M SPS: {bandwidth_gbps:.1f} GB/s")
print(f"  PCIe4 x16 bidirectional ~64 GB/s, so bandwidth is well within envelope.")

# Compute: 1.3M SPS / 512 agents/GPU ≈ 2538 environment steps/sec/GPU
steps_per_sec_per_gpu = 1.3e6 / 512
print(f"\nSteps per simulated second per GPU: {steps_per_sec_per_gpu:.0f} env-s/s")
print(f"  i.e. one 25.6s episode every {25.6/steps_per_sec_per_gpu:.2f} wall seconds")

# With bf16 mixed precision and a 3.5M-param MLP, a forward pass is sub-millisecond at batch 512.
# Back-of-envelope: 3.5M params * 2 FLOP/param ≈ 7 MFLOP per forward; at 1.3M/sec = 9 TFLOP/s,
# well within A100 FP16 tensor-core capacity (~300 TFLOP).
params = 3.5e6
flop_per_forward = params * 2  # rough
flop_per_sec = flop_per_forward * 1.3e6
print(f"\nRough policy compute for planner (3.5M params):")
print(f"  ~{flop_per_forward/1e6:.1f} MFLOP/forward")
print(f"  @ 1.3M SPS ≈ {flop_per_sec/1e12:.1f} TFLOP/s (A100 FP16 ~300 TFLOP/s)")
print("\nConclusion: 1.3M SPS is physically plausible on a server GPU; cannot be independently verified without code.")
