"""Cross-check: fla official GatedDeltaNetForCausalLM (all triton ops monkeypatched to
naive torch) vs my gdn_model.GDNModel, on the same tokens."""
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(6)

import fla.layers.gated_deltanet as fla_gdn
import fla.modules.fused_norm_gate as fla_fng
import fla.modules.layernorm as fla_ln
import fla.modules.mlp as fla_mlp
from fla.modules.conv.short_conv import ShortConvolution


def naive_chunk(q, k, v, g, beta, A_log=None, dt_bias=None, scale=None,
                initial_state=None, output_final_state=False,
                use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
                state_v_first=False, cu_seqlens=None, **kw):
    # q,k [B,T,H,K]; v [B,T,HV,V]; g,beta [B,T,HV]
    B, T, Hn, K = q.shape
    V = v.shape[-1]
    if use_qk_l2norm_in_kernel:
        q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
    if use_beta_sigmoid_in_kernel:
        beta = torch.sigmoid(beta.float()) * (2.0 if allow_neg_eigval else 1.0)
    if use_gate_in_kernel:
        g = -A_log.float().exp() * F.softplus(g.float() + dt_bias.float())
    if scale is None:
        scale = K ** -0.5
    S = torch.zeros(B, Hn, K, V)
    o = torch.empty(B, T, Hn, V)
    for t in range(T):
        S = S * g[:, t].exp().view(B, Hn, 1, 1)
        kk = k[:, t]
        dv = v[:, t] - (S * kk.unsqueeze(-1)).sum(2)
        S = S + beta[:, t].view(B, Hn, 1, 1) * kk.unsqueeze(-1) * dv.unsqueeze(-2)
        o[:, t] = ((q[:, t] * scale).unsqueeze(2) * S).sum(2)
    return o, None


def naive_rmsnorm_fwd(self, x, residual=None, prenorm=False, residual_in_fp32=False):
    def norm(y):
        return y.float() * torch.rsqrt(y.float().pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
    if residual is None:
        return norm(x)
    res = residual + x
    out = norm(res)
    return (out, res) if prenorm else (res, out) if False else (out, res)


def naive_frng_fwd(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
    y = y * self.weight.float()
    return (y * F.silu(g.float())).to(x.dtype)


def naive_conv_fwd(self, x, mask=None, cache=None, output_final_state=False, cu_seqlens=None, **kw):
    # x [B, T, C]
    B, T, C = x.shape
    K = self.kernel_size[0] if hasattr(self.kernel_size, '__len__') else self.kernel_size
    w = self.weight.squeeze(1)  # [C, K]
    xp = F.pad(x, (0, 0, K - 1, 0))
    y = torch.zeros_like(x)
    for i in range(K):
        y = y + xp[:, i:i + T] * w[:, i].unsqueeze(0)
    if self.bias is not None:
        y = y + self.bias
    if self.activation == 'silu':
        y = F.silu(y)
    return y, None


def naive_swiglu_linear(self, x, y, weight, bias):
    return F.linear(F.silu(x) * y, weight, bias)


fla_gdn.chunk_gated_delta_rule = naive_chunk
fla_gdn.fused_recurrent_gated_delta_rule = naive_chunk
fla_ln.RMSNorm.forward = naive_rmsnorm_fwd
fla_fng.FusedRMSNormGated.forward = naive_frng_fwd
ShortConvolution.forward = naive_conv_fwd
fla_mlp.SwiGLULinear.forward = naive_swiglu_linear

import fla.models  # noqa: registers gated_deltanet
from transformers import AutoModelForCausalLM

import glob
import os

MD = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--m-a-p--340M-20B-GatedDeltaNet-pure-baseline/snapshots/*")))[0]
model = AutoModelForCausalLM.from_pretrained(MD, torch_dtype=torch.float32)
model.eval()

segs = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wt2_test_segs.npy"))
ids = torch.tensor(segs[0][:64]).unsqueeze(0)
with torch.no_grad():
    out = model(ids)
logits_ref = out.logits[0]
tgt = torch.tensor(segs[0][1:64])
lp = F.log_softmax(logits_ref.float(), -1)
nll = -lp[:-1].gather(1, tgt.unsqueeze(1)).mean().item()
print("FLA-reference NLL(64 tok):", round(nll, 4), "PPL", round(float(np.exp(nll)), 1))

from gdn_model import GDNModel
mine = GDNModel(MD + "/model.safetensors")
with torch.no_grad():
    logits_mine = mine.forward(torch.tensor(segs[0][:64]))
diff = (logits_ref - logits_mine).abs()
print("logit diff: max", diff.max().item(), "mean", diff.mean().item())
lpm = F.log_softmax(logits_mine.float(), -1)
nllm = -lpm[:-1].gather(1, tgt.unsqueeze(1)).mean().item()
print("mine NLL(64 tok):", round(nllm, 4))
