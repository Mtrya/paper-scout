"""Probe C: full-model excess NLL with quantized recurrent states.

The FP32 forward is monkeypatched at the recurrence: after every state update the state
is quantized (per-key-row symmetric, scale=absmax/(2^(b-1)-1) clamped [2^-14, 65504]).
Variants:
  uniform6   - every head INT6 (paper's collapsing baseline)
  lifetime   - simplified lifetime-aware allocation: heads ranked by gate half-life
               (from probe A, same WikiText-2 text), longest-lived quartile -> 8 bit,
               middle half -> 6 bit, shortest quartile -> 4 bit (mean = 6 bit).
               SIMPLIFICATION vs STEPQuant: no key-row-aware dual-axis fitting
               (plain per-row absmax scales), no FP16 pivots, no distortion-weighted
               optimization - rank-based three-level assignment only.
Excess NLL = NLL(quantized) - NLL(FP32), teacher-forced on the same segments.
"""
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(6)
from gdn_model import GDNModel, H, LAYERS, quant_mixed, quant_rowwise

MD = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--m-a-p--340M-20B-GatedDeltaNet-pure-baseline/snapshots/*")))[0]
HERE = os.path.dirname(os.path.abspath(__file__))
N_SEG = 3

tau = np.load(os.path.join(HERE, "probe_a_tau.npy"))  # [LAYERS, H]
flat = tau.ravel()
order = np.argsort(flat)
q = len(order) // 4
bits_flat = np.full(len(order), 6, dtype=int)
bits_flat[order[-q:]] = 8
bits_flat[order[:q]] = 4
bits_map = bits_flat.reshape(LAYERS, H)
assert bits_map.mean() == 6.0

m = GDNModel(MD + "/model.safetensors")
segs = np.load(os.path.join(HERE, "wt2_test_segs.npy"))

variants = {
    "uniform6": lambda S, i: quant_rowwise(S, 6),
    "lifetime_8_6_4": lambda S, i: quant_mixed(S, bits_map[i]),
}

out = {"n_segments": N_SEG, "bits_map_mean": float(bits_map.mean()),
       "bits_counts": {"4bit": int((bits_map == 4).sum()), "6bit": int((bits_map == 6).sum()), "8bit": int((bits_map == 8).sum())}}
nll_curves = {}
for name, fn in variants.items():
    nlls = []
    for si in range(N_SEG):
        ids = torch.tensor(segs[si])
        t0 = time.time()
        with torch.no_grad():
            logits = m.forward(ids, quantize=fn)
        lp = F.log_softmax(logits.float(), -1)
        nll = -lp[:-1].gather(1, torch.tensor(segs[si][1:]).unsqueeze(1)).squeeze(1).numpy()
        nlls.append(nll)
        print(f"{name} seg{si}: {time.time()-t0:.1f}s NLL {nll.mean():.4f}", flush=True)
    nll_curves[name] = np.stack(nlls)
    out[name] = {"nll_mean_per_seg": [float(x.mean()) for x in nlls], "nll_mean": float(np.mean([x.mean() for x in nlls]))}

refs = []
for si in range(N_SEG):
    c = dict(np.load(os.path.join(HERE, f"caps/seg{si}.npz")))
    refs.append(c["nll_ref"])
nll_curves["fp32_ref"] = np.stack(refs)
out["fp32_ref"] = {"nll_mean": float(np.mean([x.mean() for x in refs]))}
for name in variants:
    out[name]["excess_nll_per_seg"] = [float((nll_curves[name][si] - refs[si]).mean()) for si in range(N_SEG)]
    out[name]["excess_nll"] = float(np.mean(out[name]["excess_nll_per_seg"]))

for _name, _arr in nll_curves.items():
    np.save(os.path.join(HERE, f"probe_c_nll_{_name}.npy"), _arr)
with open(os.path.join(HERE, "probe_c_summary.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
