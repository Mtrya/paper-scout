"""Reproduce ST-WAM's frame-triplet diagnosis (protocol not released by authors).

Triplet: (clean_init from physical-intelligence/libero, shifted_init from
lerobot/libero_plus with matched scene layout, clean_final from libero).
Feature conventions (ours, paper does not specify):
  DINOv3: mean of patch tokens (excl. CLS + register tokens), cosine.
  Wan2.2 VAE: flattened latent of the single-frame encode, cosine.
Metrics: S_same = cos(clean_init, shifted_init);
         disc   = S_same > cos(shifted_init, clean_final).
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

W = Path("/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/embodied-research")
T = W / "libero-triplet"
FRAMES = T / "frames"
DINO_DIR = T / "dinov3"
ABOT = W / "ABot-World"
VAE_PTH = W / "ABot-World-0-5B-LF" / "Wan2.2_VAE.pth"

spec = json.load(open(T / "pairs.json"))
pairs, controls = spec["pairs"], spec["controls"]

need = sorted({p[k] for p in pairs for k in ("clean_init", "shifted_init", "clean_final")}
              | {c[k] for c in controls for k in ("a", "b")})
print("unique frames:", len(need))

device = "cuda"

# ---------------- DINOv3 ----------------
from transformers import AutoModel, AutoImageProcessor

proc = AutoImageProcessor.from_pretrained(DINO_DIR)
dino = AutoModel.from_pretrained(DINO_DIR, torch_dtype=torch.bfloat16).to(device).eval()
nreg = getattr(dino.config, "num_register_tokens", 0)
print("dinov3 register tokens:", nreg)

feats_dino = {}
with torch.no_grad():
    for i in range(0, len(need), 16):
        batch = [Image.open(FRAMES / n).convert("RGB") for n in need[i:i + 16]]
        inputs = proc(images=batch, return_tensors="pt").to(device, torch.bfloat16)
        out = dino(**inputs).last_hidden_state.float()
        patch = out[:, 1 + nreg:, :].mean(dim=1)
        for n, v in zip(need[i:i + 16], patch):
            feats_dino[n] = v.cpu()
print("dino feats:", len(feats_dino))
del dino
torch.cuda.empty_cache()

# ---------------- Wan2.2 VAE ----------------
import sys
sys.path.insert(0, str(ABOT))
from utils.wan_wrapper import WanVAEWrapper

vae = WanVAEWrapper(pretrained_path=str(VAE_PTH), z_dim=48).to(device).eval()

feats_vae = {}
with torch.no_grad():
    for n in need:
        img = Image.open(FRAMES / n).convert("RGB").resize((256, 256))
        x = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
        x = x.reshape(256, 256, 3).permute(2, 0, 1).float() / 127.5 - 1.0
        x = x.unsqueeze(0).unsqueeze(2).to(device)  # [1,3,1,256,256]
        z = vae.encode_to_latent(x).float().flatten()
        feats_vae[n] = z.cpu()
print("vae feats:", len(feats_vae))
del vae
torch.cuda.empty_cache()

def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

rows = []
for p in pairs:
    r = dict(p)
    for space, feats in (("dino", feats_dino), ("vae", feats_vae)):
        s_same = cos(feats[p["clean_init"]], feats[p["shifted_init"]])
        s_final = cos(feats[p["shifted_init"]], feats[p["clean_final"]])
        r[f"{space}_s_same"] = s_same
        r[f"{space}_s_final"] = s_final
        r[f"{space}_disc"] = s_same > s_final
    rows.append(r)

ctrl_rows = []
for c in controls:
    r = dict(c)
    for space, feats in (("dino", feats_dino), ("vae", feats_vae)):
        r[f"{space}_cos"] = cos(feats[c["a"]], feats[c["b"]])
    ctrl_rows.append(r)

def agg(rs, space):
    return {
        "mean_s_same": float(np.mean([r[f"{space}_s_same"] for r in rs])),
        "disc_rate": float(np.mean([r[f"{space}_disc"] for r in rs])),
        "mean_s_final": float(np.mean([r[f"{space}_s_final"] for r in rs])),
        "n": len(rs),
    }

summary = {}
for space in ("dino", "vae"):
    summary[space] = {"overall": agg(rows, space)}
    by_shift = {}
    for r in rows:
        by_shift.setdefault(r["shift"], []).append(r)
    summary[space]["by_shift"] = {k: agg(v, space) for k, v in sorted(by_shift.items())}
    clean_rows = [r for r in rows if r["shift"] == "clean"]
    pert_rows = [r for r in rows if r["shift"] != "clean"]
    summary[space]["clean_only"] = agg(clean_rows, space)
    summary[space]["perturbed_only"] = agg(pert_rows, space)

ctrl_summary = {}
for kind in ("cross_task", "same_task_init_final"):
    rs = [c for c in ctrl_rows if c["kind"] == kind]
    for space in ("dino", "vae"):
        ctrl_summary.setdefault(kind, {})[space] = float(np.mean([c[f"{space}_cos"] for c in rs]))

out = {"summary": summary, "controls": ctrl_summary, "pairs": rows, "control_rows": ctrl_rows}
json.dump(out, open(T / "triplet_results.json", "w"), indent=1)

print(json.dumps(summary, indent=1))
print(json.dumps(ctrl_summary, indent=1))
print("TRIPLET_DONE")
