"""
Minimal stand-alone reproduction of DomainAwareLinear.

The class is copied verbatim from:
  cosmos_framework/model/vfm/mot/domain_aware_linear.py

We verify:
1. It correctly dispatches per-sample domain IDs for both 2D and 3D inputs.
2. The parameter overhead vs a shared linear for multiple embodiments.
"""

import torch
from torch import nn


class DomainAwareLinear(nn.Module):
    """Copied verbatim from the Cosmos 3 codebase."""

    def __init__(self, input_size: int, output_size: int, num_domains: int = 50) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_domains = num_domains
        self.fc = nn.Embedding(num_domains, output_size * input_size)
        self.bias = nn.Embedding(num_domains, output_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, domain_id: torch.LongTensor) -> torch.Tensor:
        B = domain_id.shape[0]
        W = self.fc(domain_id).view(B, self.input_size, self.output_size)
        b = self.bias(domain_id).view(B, self.output_size)
        if x.dim() == 2:
            return torch.bmm(x.unsqueeze(1), W).squeeze(1) + b
        else:
            return torch.bmm(x, W) + b.unsqueeze(1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def fmt(n: int) -> str:
    return f"{n:,}"


if __name__ == "__main__":
    INPUT_SIZE = 64
    OUTPUT_SIZE = 4096
    NUM_DOMAINS = 32
    BATCH = 4

    layer = DomainAwareLinear(INPUT_SIZE, OUTPUT_SIZE, NUM_DOMAINS)
    shared = nn.Linear(INPUT_SIZE, OUTPUT_SIZE, bias=True)

    print("=== DomainAwareLinear sanity check ===")
    print(f"input={INPUT_SIZE}, output={OUTPUT_SIZE}, num_domains={NUM_DOMAINS}")
    print()

    # 2D input: one vector per sample, each sample has its own domain.
    x2d = torch.randn(BATCH, INPUT_SIZE)
    domain_ids = torch.tensor([0, 5, 5, 31], dtype=torch.long)
    y2d = layer(x2d, domain_ids)
    assert y2d.shape == (BATCH, OUTPUT_SIZE)

    # Verify dispatch is actually domain-specific: compare with explicit loop.
    y2d_loop = torch.stack([
        torch.matmul(x2d[i], layer.fc(domain_ids[i]).view(INPUT_SIZE, OUTPUT_SIZE))
        + layer.bias(domain_ids[i]).view(OUTPUT_SIZE)
        for i in range(BATCH)
    ])
    max_err_2d = (y2d - y2d_loop).abs().max().item()
    print(f"2D dispatch max error vs explicit loop: {max_err_2d:.2e}")
    assert max_err_2d < 1e-5

    # 3D input: [B, T, I] (e.g. a temporal action sequence per sample).
    T = 7
    x3d = torch.randn(BATCH, T, INPUT_SIZE)
    y3d = layer(x3d, domain_ids)
    assert y3d.shape == (BATCH, T, OUTPUT_SIZE)

    # Same per-sample weight should be applied across time.
    y3d_loop = torch.stack([
        torch.matmul(x3d[i], layer.fc(domain_ids[i]).view(INPUT_SIZE, OUTPUT_SIZE))
        + layer.bias(domain_ids[i]).view(OUTPUT_SIZE)
        for i in range(BATCH)
    ])
    max_err_3d = (y3d - y3d_loop).abs().max().item()
    print(f"3D dispatch max error vs explicit loop: {max_err_3d:.2e}")
    assert max_err_3d < 1e-5

    # Parameter-count comparison.
    dal_params = count_params(layer)
    shared_params = count_params(shared)
    print()
    print(f"DomainAwareLinear params:  {fmt(dal_params)}")
    print(f"Shared nn.Linear params:   {fmt(shared_params)}")
    print(f"Overhead factor:           {dal_params / shared_params:.1f}x (= num_domains)")
    print(f"As fraction of Nano MoT:   {dal_params / 15_136_811_008 * 100:.4f}%")

