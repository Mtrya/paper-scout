"""Typed-token Logic Engine: decoder-only Transformer.

Width 256, 6 layers, 8 heads, MLP x4, tied token embeddings, learned
positions — matches the MASS matched Logic Engine (~5.6M params).
Includes KV-cache incremental decoding for autoregressive rollout.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from codec import VOCAB_SIZE


class CausalSelfAttention(nn.Module):
    def __init__(self, d=256, heads=8):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def _split(self, x, B, T, C):
        return x.view(B, T, self.h, C // self.h).transpose(1, 2)

    def forward(self, x, past=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split(q, B, T, C), self._split(k, B, T, C), self._split(v, B, T, C)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        causal = past is None
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C)), (k, v)


class Block(nn.Module):
    def __init__(self, d=256, heads=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x, past=None):
        a, present = self.attn(self.ln1(x), past)
        x = x + a
        return x + self.mlp(self.ln2(x)), present


class LogicEngine(nn.Module):
    def __init__(self, vocab=VOCAB_SIZE, d=256, layers=6, heads=8, max_len=1024):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # tied
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x, _ = b(x)
        return self.head(self.ln(x))

    @torch.no_grad()
    def step(self, idx, past=None, offset=0):
        """Incremental decode: idx (B,T_new) -> logits (B,T_new,V), new past."""
        B, T = idx.shape
        pos = torch.arange(offset, offset + T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        new_past = []
        for i, b in enumerate(self.blocks):
            x, present = b(x, None if past is None else past[i])
            new_past.append(present)
        return self.head(self.ln(x)), new_past


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = LogicEngine()
    print(f"params: {count_params(m)/1e6:.2f}M")
