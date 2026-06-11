"""
Shape probe for World Pilot fusion modules.
Verifies tensor dimensions through Latent Steering and Action Steering.
"""
import torch
from WorldPilot.model.modules.cosmos_fusion import CosmosImageFuser, CosmosActionProjector

def test_latent_steering():
    B, L, H = 2, 256, 768          # batch, VLM seq len, hidden dim
    N_cam, C, Hp, Wp = 2, 16, 28, 28  # 2 cameras, Cosmos VAE latent shape

    fuser = CosmosImageFuser(hidden_size=H, cosmos_latent_channels=C,
                             cosmos_latent_spatial=Hp, num_cameras=N_cam)
    vl_hidden = torch.randn(B, L, H)
    future_latents = torch.randn(B, N_cam, C, Hp, Wp)

    out = fuser(vl_hidden, future_latents)
    assert out.shape == (B, L, H), f"Expected {(B, L, H)}, got {out.shape}"
    print("[Latent Steering] OK  — input:", vl_hidden.shape,
          "latents:", future_latents.shape, "output:", out.shape)

def test_action_steering():
    B, chunk, action_dim = 2, 16, 7
    abot_horizon = 10
    output_dim = 768

    projector = CosmosActionProjector(action_dim=action_dim, cosmos_chunk_size=chunk,
                                      abot_action_horizon=abot_horizon, output_dim=output_dim)
    cosmos_actions = torch.randn(B, chunk, action_dim)
    out = projector(cosmos_actions)
    assert out.shape == (B, 1, output_dim), f"Expected {(B, 1, output_dim)}, got {out.shape}"
    print("[Action Steering] OK — input:", cosmos_actions.shape,
          "output:", out.shape)

if __name__ == "__main__":
    test_latent_steering()
    test_action_steering()
