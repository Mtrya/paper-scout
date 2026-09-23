"""Tiny decoder-only transformer with LoRA on every attention/MLP projection.

Toy-scale stand-in for the ComposeCL setup (arXiv:2609.06986): a dense base
model (pretrained on filler text, then frozen) plus per-matrix LoRA adapters
that are the only trainable parameters during continual SFT.

LoRA follows the paper's notation: effective weight W_eff = W + rho * B @ A
with rho = alpha / r (Section 3.3, Eq. 6).
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = nn.Linear(d_in, d_out, bias=False)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r  # rho
        self.lora_A = nn.Linear(d_in, r, bias=False)
        self.lora_B = nn.Linear(r, d_out, bias=False)
        self.reset_adapter()

    def reset_adapter(self):
        """Kaiming A, zero B — the merged-LoRA reinit rule (App. B.6.2)."""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @torch.no_grad()
    def merge_adapter(self):
        """Fold rho * B @ A into the dense weight, then reinit (App. B.6.2)."""
        self.base.weight.add_(self.scaling * (self.lora_B.weight @ self.lora_A.weight))
        self.reset_adapter()

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(x))


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, r: int, alpha: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = LoRALinear(d_model, d_model, r, alpha)
        self.k = LoRALinear(d_model, d_model, r, alpha)
        self.v = LoRALinear(d_model, d_model, r, alpha)
        self.o = LoRALinear(d_model, d_model, r, alpha)

    def forward(self, x, key_mask=None):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        att = att.masked_fill(causal, float("-inf"))
        if key_mask is not None:
            att = att.masked_fill(~key_mask[:, None, None, :].bool(), float("-inf"))
        att = torch.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, D)
        return self.o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, r: int, alpha: int):
        super().__init__()
        self.fc1 = LoRALinear(d_model, d_ff, r, alpha)
        self.fc2 = LoRALinear(d_ff, d_model, r, alpha)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, r: int, alpha: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, r, alpha)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, r, alpha)

    def forward(self, x, key_mask=None):
        x = x + self.attn(self.ln1(x), key_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab: int, d_model: int = 128, n_heads: int = 4,
                 d_ff: int = 512, n_layers: int = 2, max_len: int = 64,
                 r: int = 8, alpha: int = 16):
        super().__init__()
        self.wte = nn.Embedding(vocab, d_model)
        self.wpe = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, r, alpha) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.apply(self._init)
        # apply() overwrote the adapter init; restore Kaiming-A / zero-B.
        for m in self.modules():
            if isinstance(m, LoRALinear):
                m.reset_adapter()

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear,)):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, key_mask=None):
        T = ids.shape[1]
        h = self.wte(ids) + self.wpe.weight[:T]
        for blk in self.blocks:
            h = blk(h, key_mask)
        return self.head(self.ln_f(h))

    def lora_named_params(self):
        return [(n, p) for n, p in self.named_parameters() if ".lora_" in n]

    def freeze_dense_train_lora(self):
        for p in self.parameters():
            p.requires_grad_(False)
        for _, p in self.lora_named_params():
            p.requires_grad_(True)

    @torch.no_grad()
    def merge_all_adapters(self):
        for m in self.modules():
            if isinstance(m, LoRALinear):
                m.merge_adapter()
