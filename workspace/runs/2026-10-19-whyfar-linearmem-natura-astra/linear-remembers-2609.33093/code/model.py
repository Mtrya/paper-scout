"""Batched pure-torch Gated DeltaNet (GDN-340M) with causal-intervention hooks.

Semantics follow the fla 0.5.2 naive recurrence, cross-checked to 1e-5 logits in the
2026-10-08 STEPQuant thread (same checkpoint, pure-torch port). Per head:

    S_t = S_{t-1} * exp(g_t);  dv_t = v_t - S_t^T k_t
    S_t = S_t + beta_t k_t dv_t^T
    o_t = (q_t / sqrt(D))^T S_t

The current token's write is included in its own readout. Prefix / query-token
processing is split so that a pre-query state can be replaced per (layer, head)
without rerunning the prefix.
"""
import glob
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

H, DK, DV, HID, LAYERS, INTER, VOCAB, EPS = 4, 256, 256, 1024, 24, 2816, 32000, 1e-6
QSCALE = DK ** -0.5


def rmsnorm(x, w, eps=EPS):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * w


def conv_silu(x, w):
    """Causal depthwise conv width K then SiLU. x [B,T,C], w [C,1,K]."""
    K = w.shape[-1]
    T = x.shape[1]
    xp = F.pad(x, (0, 0, K - 1, 0))
    wk = w.squeeze(1)  # [C,K]
    y = torch.zeros_like(x)
    for i in range(K):
        y = y.addcmul_(xp[:, i:i + T], wk[:, i].view(1, 1, -1))
    return F.silu(y)


def conv_step(x, cache, w):
    """One-token conv. x [B,C], cache [B,K-1,C] (oldest first), w [C,1,K]."""
    K = w.shape[-1]
    xs = torch.cat([cache, x.unsqueeze(1)], dim=1)  # [B,K,C]
    y = (xs * w.squeeze(1).t().view(1, K, -1)).sum(1)
    return F.silu(y), xs[:, 1:]


class GDNModel:
    def __init__(self, ckpt_path=None, device="cuda"):
        if ckpt_path is None:
            ckpt_path = default_ckpt()
        sd = {k: v.to(device).float() for k, v in load_file(ckpt_path).items()}
        self.sd = sd
        self.device = device
        self.embed = sd["model.embeddings.weight"]
        self.norm = sd["model.norm.weight"]
        self.lm_head = sd["lm_head.weight"]
        self.p = [f"model.layers.{i}.attn." for i in range(LAYERS)]
        self.an = [f"model.layers.{i}.attn_norm.weight" for i in range(LAYERS)]
        self.mn = [f"model.layers.{i}.mlp_norm.weight" for i in range(LAYERS)]
        self.gp = [f"model.layers.{i}.attn.g_proj.weight" for i in range(LAYERS)]
        self.op = [f"model.layers.{i}.attn.o_proj.weight" for i in range(LAYERS)]
        self.on = [f"model.layers.{i}.attn.o_norm.weight" for i in range(LAYERS)]
        self.gate = [f"model.layers.{i}.mlp.gate_proj.weight" for i in range(LAYERS)]
        self.up = [f"model.layers.{i}.mlp.up_proj.weight" for i in range(LAYERS)]
        self.down = [f"model.layers.{i}.mlp.down_proj.weight" for i in range(LAYERS)]
        self.mp = [f"model.layers.{i}.mlp." for i in range(LAYERS)]
        n = sum(v.numel() for v in sd.values())
        print(f"loaded {n/1e6:.1f}M params on {device}")

    # ---- projections ----
    def _qkv(self, x, l):
        p = self.p[l]
        return (conv_silu(x @ self.sd[p + "q_proj.weight"].t(), self.sd[p + "q_conv1d.weight"]),
                conv_silu(x @ self.sd[p + "k_proj.weight"].t(), self.sd[p + "k_conv1d.weight"]),
                conv_silu(x @ self.sd[p + "v_proj.weight"].t(), self.sd[p + "v_conv1d.weight"]))

    def _qkv_step(self, xn, conv_cache, l):
        p = self.p[l]
        q, cq = conv_step(xn @ self.sd[p + "q_proj.weight"].t(), conv_cache[0], self.sd[p + "q_conv1d.weight"])
        k, ck = conv_step(xn @ self.sd[p + "k_proj.weight"].t(), conv_cache[1], self.sd[p + "k_conv1d.weight"])
        v, cv = conv_step(xn @ self.sd[p + "v_proj.weight"].t(), conv_cache[2], self.sd[p + "v_conv1d.weight"])
        return q, k, v, (cq, ck, cv)

    @staticmethod
    def _norm_qk(q, k, B, T):
        q = q.view(B, T, H, DK)
        k = k.view(B, T, H, DK)
        q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
        return q, k

    def _gates(self, xn, l):
        p = self.p[l]
        beta = torch.sigmoid(xn @ self.sd[p + "b_proj.weight"].t())
        g = -self.sd[p + "A_log"].exp() * F.softplus(xn @ self.sd[p + "a_proj.weight"].t()
                                                     + self.sd[p + "dt_bias"])
        return beta, g

    # ---- full prefill ----
    @torch.no_grad()
    def prefill(self, ids, cut=None, logits_pos=None, intervene=None,
                capture_states=False, capture_g=False, capture_writes=None):
        """ids [B,T]. If cut is set, process ids[:, :cut] and return per-layer state
        and conv caches at that boundary (query-token processing is separate).
        intervene: dict(layer, head, start, end, repl) where repl is None (block) or
        a tensor [B, end-start, DK, DV] of replacement write terms.
        """
        B, T = ids.shape
        if cut is not None:
            T = cut
            ids = ids[:, :cut]
        x = self.embed[ids]  # [B,T,HID]
        states, caches = {}, {}
        gsum = torch.zeros(LAYERS, H, device=self.device)
        gcnt = 0
        writes_out = None
        for l in range(LAYERS):
            xn = rmsnorm(x, self.sd[self.an[l]])
            qp, kp, vp = self._qkv(xn, l)
            q, k = self._norm_qk(qp, kp, B, T)
            v = vp.view(B, T, H, DV)
            beta, g = self._gates(xn, l)
            if capture_g:
                gsum[l] = g[:, 64:].mean(dim=(0, 1)) if T > 64 else g.mean(dim=(0, 1))
                gcnt += 1
            S = torch.zeros(B, H, DK, DV, device=self.device)
            o = torch.empty(B, T, H, DV, device=self.device)
            cap_w = None
            cap_spans = None
            if capture_writes is not None and capture_writes["layer"] == l:
                cap_spans = capture_writes["spans"]
                cap_total = sum(b - a for a, b in cap_spans)
                cap_w = torch.empty(B, cap_total, DK, DV, device=self.device)
                cap_pos = {t: i for i, (a, b) in enumerate(cap_spans) for t in range(a, b)}
                cap_idx = {t: sum(bb - aa for aa, bb in cap_spans[:i]) + (t - a)
                           for i, (a, b) in enumerate(cap_spans) for t in range(a, b)}
            iv = intervene if (intervene is not None and intervene["layer"] == l) else None
            iv_seg = None
            if iv is not None:
                iv_seg = {t: (rep, t - a) for (a, b, rep) in iv["segments"] for t in range(a, b)}
            for t in range(T):
                S = S * g[:, t].exp().view(B, H, 1, 1)
                dv = v[:, t] - (S * k[:, t].unsqueeze(-1)).sum(2)
                w = beta[:, t].view(B, H, 1, 1) * k[:, t].unsqueeze(-1) * dv.unsqueeze(-2)
                if cap_w is not None and t in cap_pos:
                    cap_w[:, cap_idx[t]] = w[:, capture_writes['head']]
                if iv_seg is not None and t in iv_seg:
                    h = iv["head"]
                    rep, j = iv_seg[t]
                    w = w.clone()
                    w[:, h] = 0.0 if rep is None else rep[:, j]
                S = S + w
                o[:, t] = ((q[:, t] * QSCALE).unsqueeze(2) * S).sum(2)
            if capture_states:
                states[l] = S
                # pre-conv projections of the last K-1 prefix positions
                xlast = xn[:, -3:]
                caches[l] = ((xlast @ self.sd[self.p[l] + "q_proj.weight"].t()).contiguous(),
                             (xlast @ self.sd[self.p[l] + "k_proj.weight"].t()).contiguous(),
                             (xlast @ self.sd[self.p[l] + "v_proj.weight"].t()).contiguous())
            if cap_w is not None:
                writes_out = cap_w
            z = (xn @ self.sd[self.gp[l]].t()).view(B, T, H, DV)
            o = rmsnorm(o, self.sd[self.on[l]]) * F.silu(z)
            attn_out = o.reshape(B, T, H * DV) @ self.sd[self.op[l]].t()
            x = x + attn_out
            hn = rmsnorm(x, self.sd[self.mn[l]])
            gate = self.sd[self.gate[l]]
            up = self.sd[self.up[l]]
            x = x + (F.silu(hn @ gate.t()) * (hn @ up.t())) @ self.sd[self.down[l]].t()
        if cut is not None:
            out = {"x": x, "states": states, "caches": caches}
            if capture_g:
                out["gmean"] = gsum / max(gcnt, 1)
            return out
        if logits_pos is None:
            logits_pos = T - 1
        if isinstance(logits_pos, (list, tuple)):
            xs = torch.stack([x[:, p] for p in logits_pos], dim=1)
            logits = rmsnorm(xs, self.norm) @ self.lm_head.t()
        else:
            logits = rmsnorm(x[:, logits_pos], self.norm) @ self.lm_head.t()
        out = {"logits": logits}
        if capture_g:
            out["gmean"] = gsum / max(gcnt, 1)
        if writes_out is not None:
            out["writes"] = writes_out
        return out

    # ---- single query token through all layers ----
    @torch.no_grad()
    def query_token(self, token_ids, pre, swap=None, return_trace=False, return_pre=False, return_x=False):
        """token_ids [B] -> logits [B,V]. pre = prefill(cut=...) output.
        swap: dict(layer, head, donor_states[layer] [B,H,DK,DV])."""
        B = token_ids.shape[0]
        x = self.embed[token_ids]
        trace = []
        new_states, new_caches = {}, {}
        for l in range(LAYERS):
            xn = rmsnorm(x, self.sd[self.an[l]])
            q, k, v, caches = self._qkv_step(xn, pre["caches"][l], l)
            q = q.view(B, H, DK)
            k = k.view(B, H, DK)
            q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6)
            k = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
            v = v.view(B, H, DV)
            beta, g = self._gates(xn, l)
            S = pre["states"][l].clone()
            if swap is not None and swap["layer"] == l:
                S[:, swap["head"]] = swap["donor_states"][l][:, swap["head"]]
            S = S * g.exp().view(B, H, 1, 1)
            dv = v - (S * k.unsqueeze(-1)).sum(2)
            S = S + beta.view(B, H, 1, 1) * k.unsqueeze(-1) * dv.unsqueeze(-2)
            new_states[l] = S
            new_caches[l] = caches
            o = ((q * QSCALE).unsqueeze(2) * S).sum(2)
            if return_trace:
                trace.append({"x_in": x, "q": q, "k": k, "v": v, "beta": beta, "g": g, "S": S, "o": o})
            z = (xn @ self.sd[self.gp[l]].t()).view(B, H, DV)
            o = rmsnorm(o, self.sd[self.on[l]]) * F.silu(z)
            x = x + o.reshape(B, H * DV) @ self.sd[self.op[l]].t()
            hn = rmsnorm(x, self.sd[self.mn[l]])
            gate = self.sd[self.gate[l]]
            up = self.sd[self.up[l]]
            x = x + (F.silu(hn @ gate.t()) * (hn @ up.t())) @ self.sd[self.down[l]].t()
        logits = rmsnorm(x, self.norm) @ self.lm_head.t()
        if return_trace:
            return logits, trace
        if return_pre:
            return logits, {"states": new_states, "caches": new_caches}
        if return_x:
            return logits, x
        return logits

    @torch.no_grad()
    def query_token_from(self, l0, x_in, token_ids, pre, swap=None):
        """Run the query token starting at layer l0 with a given block input."""
        B = token_ids.shape[0]
        x = x_in
        for l in range(l0, LAYERS):
            xn = rmsnorm(x, self.sd[self.an[l]])
            q, k, v, _ = self._qkv_step(xn, pre["caches"][l], l)
            q = q.view(B, H, DK)
            k = k.view(B, H, DK)
            q = q / torch.sqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6)
            k = k / torch.sqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
            v = v.view(B, H, DV)
            beta, g = self._gates(xn, l)
            S = pre["states"][l].clone()
            if swap is not None and swap["layer"] == l:
                S[:, swap["head"]] = swap["donor_states"][l][:, swap["head"]]
            S = S * g.exp().view(B, H, 1, 1)
            dv = v - (S * k.unsqueeze(-1)).sum(2)
            S = S + beta.view(B, H, 1, 1) * k.unsqueeze(-1) * dv.unsqueeze(-2)
            o = ((q * QSCALE).unsqueeze(2) * S).sum(2)
            z = (xn @ self.sd[self.gp[l]].t()).view(B, H, DV)
            o = rmsnorm(o, self.sd[self.on[l]]) * F.silu(z)
            x = x + o.reshape(B, H * DV) @ self.sd[self.op[l]].t()
            hn = rmsnorm(x, self.sd[self.mn[l]])
            gate = self.sd[self.gate[l]]
            up = self.sd[self.up[l]]
            x = x + (F.silu(hn @ gate.t()) * (hn @ up.t())) @ self.sd[self.down[l]].t()
        return rmsnorm(x, self.norm) @ self.lm_head.t()


def default_ckpt():
    HF = os.path.expanduser("~/.cache/huggingface/hub")
    pat = os.path.join(HF, "models--m-a-p--340M-20B-GatedDeltaNet-pure-baseline/snapshots/*/model.safetensors")
    return sorted(glob.glob(pat))[0]
