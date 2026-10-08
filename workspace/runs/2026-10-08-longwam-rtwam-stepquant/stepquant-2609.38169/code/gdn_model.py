"""Pure-torch CPU re-implementation of the fla GatedDeltaNet (flame checkpoint layout).

Semantics verified against fla 0.5.2 (fla/layers/gated_deltanet.py,
fla/ops/gated_delta_rule/{naive,gate,chunk}.py, fla/modules/fused_norm_gate.py,
fla/modules/l2norm.py) and the checkpoint `m-a-p/340M-20B-GatedDeltaNet-pure-baseline`.

Per layer (hidden=1024, 4 heads, head_dim=256):
  q,k,v = silu(causal_depthwise_conv4(W_{q,k,v} x))         # no conv bias
  q,k   = l2norm over head dim (eps inside sqrt: 1e-6)
  beta  = sigmoid(W_b x)                # [T, H]
  g     = -exp(A_log) * softplus(W_a x + dt_bias)           # [T, H], log gate
  S     = S * exp(g_t); S += beta_t * k_t (v_t - S^T k_t)^T # state [H, K, V]
  o_t   = (q_t / sqrt(256))^T S
  o     = RMSNorm_gated(o, W_o_norm; gate=silu(W_g x))      # per head, eps=1e-6
  out   = W_o o.flatten
The checkpoint's `attn.D` (per-head) is exactly 0.0 in all 24 layers (verified);
it is a dead flame-era parameter and is dropped.
Block: pre-norm (RMSNorm, eps=1e-6) residual; MLP = SwiGLU(1024 -> 2816 -> 1024).
Final RMSNorm + untied lm_head.
"""
import math

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

H, DK, DV, HID, LAYERS, INTER, VOCAB, EPS = 4, 256, 256, 1024, 24, 2816, 32000, 1e-6
QSCALE = DK ** -0.5


def rmsnorm(x, w, eps=EPS):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * w


def causal_conv_silu(x, w):
    # x [T, C]; w [C, 1, K]. Left pad K-1 zeros, depthwise conv, silu.
    # Manual shift-and-add: grouped F.conv1d is pathologically slow on CPU.
    K = w.shape[-1]
    T, C = x.shape
    xp = F.pad(x, (0, 0, K - 1, 0))  # left pad on time
    wk = w.squeeze(1)  # [C, K]
    y = torch.zeros_like(x)
    for i in range(K):
        y = y.addcmul_(xp[i:i + T], wk[:, i].unsqueeze(0))
    return F.silu(y)


class GDNWeights:
    def __init__(self, sd, i):
        p = f"model.layers.{i}."
        self.attn_norm = sd[p + "attn_norm.weight"]
        self.mlp_norm = sd[p + "mlp_norm.weight"]
        a = p + "attn."
        self.q_proj, self.k_proj, self.v_proj = sd[a + "q_proj.weight"], sd[a + "k_proj.weight"], sd[a + "v_proj.weight"]
        self.q_conv, self.k_conv, self.v_conv = sd[a + "q_conv1d.weight"], sd[a + "k_conv1d.weight"], sd[a + "v_conv1d.weight"]
        self.a_proj, self.b_proj = sd[a + "a_proj.weight"], sd[a + "b_proj.weight"]
        self.A_log, self.dt_bias = sd[a + "A_log.weight" if a + "A_log.weight" in sd else a + "A_log"], sd[a + "dt_bias"]
        self.g_proj, self.o_norm_w, self.o_proj = sd[a + "g_proj.weight"], sd[a + "o_norm.weight"], sd[a + "o_proj.weight"]
        m = p + "mlp."
        self.down_proj = sd[m + "down_proj.weight"]
        if m + "up_proj.weight" in sd:
            self.gate_proj, self.up_proj = sd[m + "gate_proj.weight"], sd[m + "up_proj.weight"]
        else:  # fused gate_up layout (linear-moe-hub): [2*I, H] = [gate; up]
            gw = sd[m + "gate_proj.weight"]
            self.gate_proj, self.up_proj = gw[: INTER], gw[INTER:]


class GDNModel:
    def __init__(self, ckpt_path):
        sd = {k: v.float() for k, v in load_file(ckpt_path).items()}
        self.embed = sd["model.embeddings.weight"]
        self.layers = [GDNWeights(sd, i) for i in range(LAYERS)]
        self.norm = sd["model.norm.weight"]
        self.lm_head = sd["lm_head.weight"]
        n = sum(v.numel() for v in sd.values())
        print(f"loaded {n/1e6:.1f}M params")

    def gdn_io(self, lw, x):
        """Projections + conv + gates for one layer. x [T, HID] -> q,k,v [T,H,D], beta,g [T,H]."""
        q = causal_conv_silu(x @ self._t(lw.q_proj), lw.q_conv)
        k = causal_conv_silu(x @ self._t(lw.k_proj), lw.k_conv)
        v = causal_conv_silu(x @ self._t(lw.v_proj), lw.v_conv)
        T = x.shape[0]
        q = q.view(T, H, DK)
        k = k.view(T, H, DK)
        v = v.view(T, H, DV)
        q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
        beta = torch.sigmoid(x @ self._t(lw.b_proj))
        g = -lw.A_log.exp() * F.softplus(x @ self._t(lw.a_proj) + lw.dt_bias)
        return q, k, v, beta, g

    @staticmethod
    def _t(w):
        return w.t()

    def gdn_recur(self, lw, x, quantize=None, capture=None):
        """Full GDN layer recurrence. quantize(S, layer_head_bits) applied after each update.
        Returns o [T, H*DV]. capture dict gets q,k,v,beta,g (post-norm/conv, pre-scale)."""
        q, k, v, beta, g = self.gdn_io(lw, x)
        T = x.shape[0]
        if capture is not None:
            capture.update(q=q.clone(), k=k.clone(), v=v.clone(), beta=beta.clone(), g=g.clone())
        S = torch.zeros(H, DK, DV)
        o = torch.empty(T, H, DV)
        for t in range(T):
            S = S * g[t].exp().view(H, 1, 1)
            dv = v[t] - (S * k[t].unsqueeze(-1)).sum(1)
            S = S + beta[t].view(H, 1, 1) * k[t].unsqueeze(-1) * dv.unsqueeze(-2)
            o[t] = ((q[t] * QSCALE).unsqueeze(1) * S).sum(1)
            if quantize is not None:
                S = quantize(S)
        z = (x @ self._t(lw.g_proj)).view(T, H, DV)
        o = rmsnorm(o, lw.o_norm_w) * F.silu(z)
        return o.reshape(T, H * DV) @ self._t(lw.o_proj)

    def forward(self, ids, quantize=None, capture_layers=None):
        """ids [T] -> logits [T, VOCAB]. If capture_layers is a set, returns (logits, caps)
        where caps[i] holds per-step q,k,v,beta,g for layer i."""
        x = self.embed[ids]
        caps = {}
        for i, lw in enumerate(self.layers):
            cap = {} if (capture_layers and i in capture_layers) else None
            h = rmsnorm(x, lw.attn_norm)
            x = x + self.gdn_recur(lw, h, quantize=(lambda S, _i=i: quantize(S, _i)) if quantize else None, capture=cap)
            if cap is not None:
                caps[i] = cap
            h = rmsnorm(x, lw.mlp_norm)
            x = x + (F.silu(h @ self._t(lw.gate_proj)) * (h @ self._t(lw.up_proj))) @ self._t(lw.down_proj)
        logits = rmsnorm(x, self.norm) @ self._t(self.lm_head)
        return (logits, caps) if capture_layers is not None else logits


def quant_rowwise(S, bits):
    """Symmetric per-key-row quantization exactly as paper App. B.5:
    scale = absmax(row)/qmax clamped to [2^-14, 65504], nearest codes in [-qmax, qmax]."""
    qmax = 2 ** (bits - 1) - 1
    s = (S.abs().amax(-1, keepdim=True) / qmax).clamp(2 ** -14, 65504.0)
    return torch.round(S / s).clamp(-qmax, qmax) * s


def quant_mixed(S, bits_per_head):
    """bits_per_head: list/Tensor of per-head bit width."""
    out = torch.empty_like(S)
    for h in range(S.shape[0]):
        out[h] = quant_rowwise(S[h], int(bits_per_head[h]))
    return out
