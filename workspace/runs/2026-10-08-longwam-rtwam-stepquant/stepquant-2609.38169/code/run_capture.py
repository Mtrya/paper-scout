"""FP32 teacher-forced capture: for each WikiText-2 test segment, run the model and
cache per-layer per-step GDN inputs (q,k,v after conv+norm, beta, log-gate g) plus the
FP32 per-position NLL (reference for probe C). Saves caps/seg{i}.npz."""
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(6)
from gdn_model import GDNModel, LAYERS

MD = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--m-a-p--340M-20B-GatedDeltaNet-pure-baseline/snapshots/*")))[0]
HERE = os.path.dirname(os.path.abspath(__file__))
N_SEG = 3

m = GDNModel(MD + "/model.safetensors")
segs = np.load(os.path.join(HERE, "wt2_test_segs.npy"))
os.makedirs(os.path.join(HERE, "caps"), exist_ok=True)

for si in range(N_SEG):
    ids = torch.tensor(segs[si])
    t0 = time.time()
    with torch.no_grad():
        logits, caps = m.forward(ids, capture_layers=set(range(LAYERS)))
    lp = F.log_softmax(logits.float(), -1)
    nll = -lp[:-1].gather(1, torch.tensor(segs[si][1:]).unsqueeze(1)).squeeze(1).numpy()
    out = {f"nll_ref": nll}
    for i in range(LAYERS):
        c = caps[i]
        for name in ["q", "k", "v", "beta", "g"]:
            out[f"l{i}_{name}"] = c[name].numpy()
    np.savez(os.path.join(HERE, f"caps/seg{si}.npz"), **out)
    print(f"seg{si}: {time.time()-t0:.1f}s NLL {nll.mean():.4f} PPL {float(np.exp(nll.mean())):.1f}", flush=True)
print("CAPTURE DONE")
