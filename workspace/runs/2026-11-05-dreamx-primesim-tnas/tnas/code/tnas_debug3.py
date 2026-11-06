#!/usr/bin/env python3
"""Debug 3: magnitude-response curve for tnas vs rand on example 0."""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

work = "/inspire/qb-ilm/project/exploration-topic/ky26021/tnas-1105"
sys.path.insert(0, os.path.join(work, "repo"))
sys.path.insert(0, work)
import tnas_probe
from tnas_probe import (HookedPass, LAYERS, build_action_pass, differentiable_actions,
                        generate_response, response_span, text_losses, tf_kl)
from vla.instructvla_eagle_dual_sys_v2_meta_query_v2 import load_vla, DEFAULT_SYSTEM_MESSAGE
tnas_probe.DEFAULT_SYSTEM_MESSAGE = DEFAULT_SYSTEM_MESSAGE

vla = load_vla(os.path.join(work, "vla_ckpt", "checkpoints", "model.pt"), stage="stage2")
vla = vla.eval().to(torch.bfloat16).to("cuda")
for p in vla.parameters():
    p.requires_grad_(True)

ex = json.load(open(os.path.join(work, "eval", "language_evaluation_with_gt.json")))[0]
image = Image.open(os.path.join(work, "eval", "language_evaluation", ex["hashkey"] + "_first_frame.jpg")).convert("RGB")
instruction = ex["annotation"]["CR"][0]["question"]
response, _ = generate_response(vla, image, instruction)
input_ids, pixel_values = build_action_pass(vla, image, instruction, response)
empty_ids, _ = build_action_pass(vla, image, instruction, response, empty=True)
ids = input_ids[0].tolist()
resp_pos, _ = response_span(vla, input_ids, empty_ids, response)

hooked = HookedPass(vla)
logits, hs = hooked.run(input_ids, pixel_values, resp_pos)
layer_states = [hs[l + 1] for l in LAYERS]

def pooled_dir_grad(loss):
    grads = torch.autograd.grad(loss, layer_states, retain_graph=True)
    return torch.cat([gr[0, resp_pos].sum(dim=0) for gr in grads]).detach().float()

nll, margin = text_losses(logits, ids, resp_pos)
g_nll = pooled_dir_grad(nll)
g_margin = pooled_dir_grad(margin)

g = torch.Generator(device="cuda").manual_seed(42)
w0 = torch.randn(112, generator=g, device="cuda", dtype=torch.float32)
w0 = w0 / w0.norm()
meta_mask = (input_ids >= vla.min_meta_token) & (input_ids <= vla.max_meta_token)
g_noise = torch.Generator(device="cuda").manual_seed(420)
noise0 = torch.randn((1, 16, 7), generator=g_noise, device="cuda", dtype=torch.float32)
acts = differentiable_actions(vla, hs[-1], input_ids, image, noise=noise0)
v = pooled_dir_grad(acts.float() @ w0)
d_act = v / v.norm()

e1 = g_nll / g_nll.norm()
e2v = g_margin - (g_margin @ e1) * e1
e2 = e2v / e2v.norm()
P = lambda u: (u @ e1) * e1 + (u @ e2) * e2
rho = (P(d_act).norm() / d_act.norm()).item()
r = d_act - P(d_act)
d_tnas = r / r.norm()
h = torch.cat([st[0, resp_pos].mean(dim=0) for st in layer_states]).float()
c = (h @ d_tnas).item()
print("rho=%.4f  h_norm=%.2f  c=%.4f  hidden_rms_span=%.4f" %
      (rho, h.norm().item(), c,
       torch.cat([st[0, resp_pos] for st in layer_states]).float().pow(2).mean().sqrt().item()))

A0 = acts.detach().float()
logits_clean = logits.detach()
rng = np.random.default_rng(42)
dirs = {"tnas": d_tnas}
for rs in range(2):
    rv = torch.from_numpy(rng.standard_normal(h.shape[0]).astype(np.float32)).to(h.device)
    dirs[f"rand{rs}"] = rv / rv.norm()
dirs["nll"] = e1

for mag in [0.005, 0.02, 0.05, 0.1, 0.3, 1.0, 3.62]:
    line = f"mag={mag:6.3f} "
    for name, u in dirs.items():
        D = h.shape[0] // len(LAYERS)
        deltas = {l: (mag * u[i * D:(i + 1) * D]) for i, l in enumerate(LAYERS)}
        with torch.no_grad():
            lg2, hs2 = hooked.run(input_ids, pixel_values, resp_pos, deltas=deltas)
            acts2 = differentiable_actions(vla, hs2[-1], input_ids, image, noise=noise0)
        da = (acts2.detach().float() - A0).reshape(16, 7)
        shift = da.norm(dim=-1).mean().item()
        kl = tf_kl(logits_clean, lg2.detach(), resp_pos)
        line += "%s=%.4f/%.1e  " % (name, shift, kl)
    print(line, flush=True)
