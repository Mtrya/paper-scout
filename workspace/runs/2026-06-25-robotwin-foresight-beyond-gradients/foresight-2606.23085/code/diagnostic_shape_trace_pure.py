"""
Pure-Python diagnostic tracing Foresight tensor shapes without PyTorch.

Runs with the standard library only, so it can be executed in any environment
to sanity-check the paper's dimensional claims.
"""


def trace_shapes():
    B, T = 2, 16                       # batch, policy calls
    H, W = 16, 16                      # patches for 256x256 / 16
    patches_per_frame = H * W          # 256
    embed_dim = 1408                   # ViT-Giant token dim
    pred_dim = 1024                    # predictor internal dim
    action_dim = 7                     # EEF pose + gripper

    # Frozen encoder output: batch, time, patches, embed
    z_h = (B, T, patches_per_frame, embed_dim)
    print(f"z_h (hidden latent)       : {z_h}")

    # Predictor input flattens time and patches
    predictor_in = (B, T * patches_per_frame, embed_dim)
    print(f"predictor input           : {predictor_in}")

    actions = (B, T - 1, action_dim)
    states = (B, T, action_dim)
    print(f"actions A_t               : {actions}")
    print(f"states s_t                : {states}")

    # Predictor projects to pred_dim internally; action/state tokens prepended
    cond_tokens = 2                    # action + state
    tokens_per_timestep = cond_tokens + patches_per_frame  # 258
    total_tokens = T * tokens_per_timestep                 # 4128
    print(f"tokens per timestep       : {tokens_per_timestep}")
    print(f"total tokens in predictor : {total_tokens}")

    # Output projected back to embed_dim and stripped of conditioning tokens
    z_p = (B, T * patches_per_frame, embed_dim)
    print(f"z_p (predicted latent)    : {z_p}")

    # Mean pool spatially per timestep
    z_p_pooled = (B, T, embed_dim)
    print(f"z_p mean-pooled           : {z_p_pooled}  <- 1408-d timestep token")

    # Detector input after learned projection W and positional encoding
    u_seq = (B, T, embed_dim)
    print(f"u_t sequence              : {u_seq}")

    # Detector outputs one score per timestep
    scores = (B, T)
    print(f"s_t failure scores        : {scores}")

    # Conformal threshold: one value per timestep from successful cal rollouts
    delta = (T,)
    print(f"delta_t threshold         : {delta}")

    # Sanity checks
    assert z_p[1] == T * patches_per_frame
    assert total_tokens == T * (cond_tokens + patches_per_frame)
    print("\nShape checks passed.")


if __name__ == "__main__":
    trace_shapes()
