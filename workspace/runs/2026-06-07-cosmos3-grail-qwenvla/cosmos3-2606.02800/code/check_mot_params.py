"""
Lightweight parameter-count ablation for Cosmos 3's Mixture-of-Transformers.

We replicate the layer shapes of the Nano model (Qwen3-VL-8B backbone,
hidden_size=4096, intermediate_size=12288, 32 query heads, 8 KV heads,
head_dim=128, 36 layers) and compare:

A. A single-pathway decoder layer (standard Qwen3-VL).
B. The MoT decoder layer (PackedAttentionMoT + dual MLPs/norms).

The script only needs torch; it defines tiny stand-in modules that capture
exact parameter shapes (no HF/transformers dependency).
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        return x  # not used


class QwenStyleMLP(nn.Module):
    """Qwen3-VL dense MLP: gate_proj, up_proj, down_proj (SwiGLU)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class SinglePathAttention(nn.Module):
    """Standard GQA self-attention (one set of projections)."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x):
        return x


class SinglePathDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.self_attn = SinglePathAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.mlp = QwenStyleMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

    def forward(self, x):
        return x


class MoTAttention(nn.Module):
    """PackedAttentionMoT: two complete GQA projection sets."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        # Understanding (reasoner) pathway
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        # Generation pathway
        self.q_proj_moe_gen = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj_moe_gen = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj_moe_gen = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj_moe_gen = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm_moe_gen = RMSNorm(head_dim)
        self.k_norm_moe_gen = RMSNorm(head_dim)

    def forward(self, x):
        return x


class MoTDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.self_attn = MoTAttention(hidden_size, num_heads, num_kv_heads, head_dim)
        self.mlp = QwenStyleMLP(hidden_size, intermediate_size)
        self.mlp_moe_gen = QwenStyleMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size)
        self.input_layernorm_moe_gen = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm_moe_gen = RMSNorm(hidden_size)

    def forward(self, x):
        return x


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def fmt(n: int) -> str:
    return f"{n:,}"


if __name__ == "__main__":
    # Nano / Qwen3-VL-8B text backbone
    H = 4096
    I = 12288
    NUM_HEADS = 32
    NUM_KV_HEADS = 8
    HEAD_DIM = 128
    NUM_LAYERS = 36
    VOCAB_SIZE = 151936

    single_layer = SinglePathDecoderLayer(H, I, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM)
    mot_layer = MoTDecoderLayer(H, I, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM)

    single_layer_p = count_params(single_layer)
    mot_layer_p = count_params(mot_layer)

    # Model-level: shared token embedding + lm_head + final norms
    embed_tokens = VOCAB_SIZE * H
    lm_head = VOCAB_SIZE * H
    # MoT has two final RMSNorms (norm + norm_moe_gen); single has one.
    single_final_norms = H
    mot_final_norms = 2 * H

    single_total = embed_tokens + lm_head + single_final_norms + NUM_LAYERS * single_layer_p
    mot_total = embed_tokens + lm_head + mot_final_norms + NUM_LAYERS * mot_layer_p

    print("=== Cosmos 3 Nano (Qwen3-VL-8B backbone) parameter ablation ===")
    print(f"Config: hidden={H}, intermediate={I}, heads={NUM_HEADS}, kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}, layers={NUM_LAYERS}, vocab={VOCAB_SIZE}")
    print()
    print(f"Single-pathway decoder layer params: {fmt(single_layer_p)}")
    print(f"MoT decoder layer params:           {fmt(mot_layer_p)}")
    print(f"  -> MoT layer / single layer ratio: {mot_layer_p / single_layer_p:.3f}x")
    print()
    print(f"Single-pathway model total: {fmt(single_total)} ({single_total/1e9:.2f}B)")
    print(f"MoT model total:           {fmt(mot_total)} ({mot_total/1e9:.2f}B)")
    print(f"  -> MoT total / single total ratio: {mot_total / single_total:.3f}x")
    print()
    print("Per-layer breakdown (MoT vs single):")
    for name, p in single_layer.named_parameters():
        print(f"  single  {name:40s} {fmt(p.numel()):>14s}")
    for name, p in mot_layer.named_parameters():
        # find corresponding single param if any
        single_name = name.replace("_moe_gen", "")
        try:
            single_p = dict(single_layer.named_parameters())[single_name].numel()
        except KeyError:
            single_p = 0
        marker = "NEW" if single_p == 0 else f"{p.numel()/single_p:.0f}x"
        print(f"  mot     {name:40s} {fmt(p.numel()):>14s}  ({marker})")

