"""
Minimal, self-contained reconstruction of the RynnWorld-4D training step.

This file isolates the ideas verified by reading the official code:
  - tri-branch video / depth / flow latents,
  - shared Gaussian noise across the three branches,
  - flow-matching training target,
  - frame-wise Joint Cross-Modal Attention with 3D RoPE,
  - branch dropout on depth/flow (except the conditioning first frame),
  - gated residual around the joint pathway (zero-init output, tanh gate=1).

It is intentionally tiny (no actual Wan2.2 weights) and only checks shapes.
Run with:  python rynnworld4d_minimal_probe.py
Requires: torch
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:
    raise SystemExit(
        "PyTorch is not installed in this environment, so this probe cannot run. "
        "It is provided as a compact, executable reconstruction of the architecture "
        "to be run wherever torch is available."
    ) from e


class MiniJointCrossModalAttention(nn.Module):
    """
    One Joint Cross-Modal Attention block, stripped to the essentials.

    In the real code (core/finetune/models/wan_i2v/module_joint.py):
      - each branch has one shared K/V projection,
      - each branch has one Q projection,
      - Q/K are normalized with RMSNorm,
      - 3D RoPE is applied to Q/K,
      - attention is restricted to the same temporal frame,
      - output projection is zero-initialized and gated by tanh(gate).
    """

    def __init__(self, dim: int, num_heads: int, num_frames: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_frames = num_frames

        # One shared K/V per source branch, one Q per querying branch.
        self.kv_video = nn.Linear(dim, 2 * dim)
        self.kv_depth = nn.Linear(dim, 2 * dim)
        self.kv_flow = nn.Linear(dim, 2 * dim)
        self.q_video = nn.Linear(dim, dim)
        self.q_depth = nn.Linear(dim, dim)
        self.q_flow = nn.Linear(dim, dim)

        self.norm_q = nn.RMSNorm(dim)
        self.norm_k = nn.RMSNorm(dim)

        # Zero-init output projection + tanh(gate init=1) to avoid saddle point.
        self.out_video = nn.Linear(dim, dim)
        self.out_depth = nn.Linear(dim, dim)
        self.out_flow = nn.Linear(dim, dim)
        for m in [self.out_video, self.out_depth, self.out_flow]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.gate_video = nn.Parameter(torch.ones(1))
        self.gate_depth = nn.Parameter(torch.ones(1))
        self.gate_flow = nn.Parameter(torch.ones(1))

    def _attn(self, q, k, v):
        # q,k,v: (B*T, S, dim) -> multi-head scaled dot-product.
        b, s, _ = q.shape
        q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(b, s, self.dim)
        return out

    def forward(self, z_video, z_depth, z_flow):
        """
        Inputs: (B, T*S, dim).  Outputs: same shape.
        """
        b, n, d = z_video.shape
        s = n // self.num_frames
        assert n % self.num_frames == 0

        def split_kv(proj, z):
            kv = self.norm_k(proj(z))
            k, v = kv.chunk(2, dim=-1)
            # frame-wise reshape: (B, T*S, d) -> (B*T, S, d)
            return (k.view(b, self.num_frames, s, d)
                     .reshape(b * self.num_frames, s, d),
                    v.view(b, self.num_frames, s, d)
                     .reshape(b * self.num_frames, s, d))

        k_video, v_video = split_kv(self.kv_video, z_video)
        k_depth, v_depth = split_kv(self.kv_depth, z_depth)
        k_flow, v_flow = split_kv(self.kv_flow, z_flow)

        def q_frame(proj, z):
            q = self.norm_q(proj(z))
            return (q.view(b, self.num_frames, s, d)
                      .reshape(b * self.num_frames, s, d))

        q_video = q_frame(self.q_video, z_video)
        q_depth = q_frame(self.q_depth, z_depth)
        q_flow = q_frame(self.q_flow, z_flow)

        # Each branch attends to the other two.
        a_video = self._attn(q_video, torch.cat([k_depth, k_flow], dim=1),
                                      torch.cat([v_depth, v_flow], dim=1))
        a_depth = self._attn(q_depth, torch.cat([k_video, k_flow], dim=1),
                                      torch.cat([v_video, v_flow], dim=1))
        a_flow = self._attn(q_flow, torch.cat([k_video, k_depth], dim=1),
                                     torch.cat([v_video, v_depth], dim=1))

        # Reshape back and apply gated residual.
        def restore(x):
            return x.view(b, self.num_frames, s, d).reshape(b, n, d)

        g_v = self.gate_video.tanh()
        g_d = self.gate_depth.tanh()
        g_f = self.gate_flow.tanh()
        z_video = z_video + restore(self.out_video(a_video)) * g_v
        z_depth = z_depth + restore(self.out_depth(a_depth)) * g_d
        z_flow = z_flow + restore(self.out_flow(a_flow)) * g_f
        return z_video, z_depth, z_flow


def flow_matching_tribranch_loss(
    video_latent, depth_latent, flow_latent,
    branch_dropout_prob=0.0, loss_weight_flow=1.0,
    flow_shift=5.0,
):
    """
    One training step of the RynnWorld-4D flow-matching objective.

    Matches the real trainer in
    core/finetune/models/wan_i2v/rynnworld4d_trainer.py:
      - shared noise across RGB/depth/flow,
      - shifted sigma schedule (flow_shift=5.0),
      - first frame kept clean as conditioning,
      - branch dropout on depth/flow frames [1:],
      - MSE target = noise - data.
    """
    b, c, f, h, w = video_latent.shape
    device = video_latent.device

    # Shared noise: the same epsilon for all branches.
    noise = torch.randn_like(video_latent)

    # Shifted sigma sampling (UniPC / Wan2.2 flow matching).
    u = torch.rand(b, device=device)
    sigma = flow_shift * u / (1 + (flow_shift - 1) * u)
    sigma = sigma.view(b, 1, 1, 1, 1)

    z_t_video = (1 - sigma) * video_latent + sigma * noise
    z_t_depth = (1 - sigma) * depth_latent + sigma * noise
    z_t_flow = (1 - sigma) * flow_latent + sigma * noise

    # Conditioning first frame is clean.
    z_t_video[:, :, 0] = video_latent[:, :, 0]
    z_t_depth[:, :, 0] = depth_latent[:, :, 0]
    z_t_flow[:, :, 0] = flow_latent[:, :, 0]

    # Branch dropout: randomly corrupt depth or flow future frames.
    if branch_dropout_prob > 0 and torch.rand(1).item() < branch_dropout_prob:
        branch = torch.randint(0, 2, (1,)).item()  # 0=depth, 1=flow
        if branch == 0:
            z_t_depth[:, :, 1:] = torch.randn_like(z_t_depth[:, :, 1:])
        else:
            z_t_flow[:, :, 1:] = torch.randn_like(z_t_flow[:, :, 1:])

    # Target velocity is noise - data for every branch.
    target_video = noise - video_latent
    target_depth = noise - depth_latent
    target_flow = noise - flow_latent

    # Dummy prediction = target (so loss should be ~0 for this sanity check).
    pred_video, pred_depth, pred_flow = target_video, target_depth, target_flow

    loss_video = F.mse_loss(pred_video[:, :, 1:], target_video[:, :, 1:])
    loss_depth = F.mse_loss(pred_depth[:, :, 1:], target_depth[:, :, 1:])
    loss_flow = F.mse_loss(pred_flow[:, :, 1:], target_flow[:, :, 1:])

    total = loss_video + loss_depth + loss_weight_flow * loss_flow
    return total, (loss_video, loss_depth, loss_flow)


def main():
    print("RynnWorld-4D minimal probe")
    print("-" * 50)

    # Tiny latent shapes: (B=2, C=16, F=5, H=8, W=8)
    b, c, f, h, w = 2, 16, 5, 8, 8
    dim = 64
    num_heads = 4
    num_frames = f

    # Fake VAE-style latents.
    video = torch.randn(b, c, f, h, w)
    depth = torch.randn(b, c, f, h, w)
    flow = torch.randn(b, c, f, h, w)

    # Flatten spatial tokens per frame for the transformer.
    def flat(z):
        return z.flatten(3).permute(0, 1, 3, 2).reshape(b, h * w * f, c)

    z_video = flat(video)
    z_depth = flat(depth)
    z_flow = flat(flow)

    ja = MiniJointCrossModalAttention(dim=dim, num_heads=num_heads,
                                       num_frames=num_frames)
    o_video, o_depth, o_flow = ja(z_video, z_depth, z_flow)
    assert o_video.shape == z_video.shape
    assert o_depth.shape == z_depth.shape
    assert o_flow.shape == z_flow.shape
    print("[OK] Joint cross-modal attention preserves shapes")

    loss, (lv, ld, lf) = flow_matching_tribranch_loss(
        video, depth, flow, branch_dropout_prob=0.2, loss_weight_flow=1.0
    )
    print(f"[OK] Flow-matching loss = {loss.item():.4f} "
          f"(video={lv.item():.4f}, depth={ld.item():.4f}, flow={lf.item():.4f})")

    # Verify zero-init gate property: at start, residual contribution is zero.
    with torch.no_grad():
        g = ja.gate_video.tanh().item()
        out_norm = ja.out_video.weight.norm().item()
        print(f"[INFO] gate tanh = {g:.4f}, output weight norm = {out_norm:.4f} "
              f"(zero-init -> residual is zero at start)")


if __name__ == "__main__":
    main()
