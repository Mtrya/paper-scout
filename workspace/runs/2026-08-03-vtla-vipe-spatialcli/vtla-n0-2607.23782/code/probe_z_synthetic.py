"""Synthetic z-sensitivity probe on the RELEASED N0-VTLA base checkpoint.

Paper: N0-VTLA (arXiv:2607.23782) conditions a flow-matching action expert on latent
tactile tokens z, predicted from the current tactile difference in VL context. The
paper's Section 5.5 perturbation probe reports a tactile-to-VL sensitivity ratio of
4.3 after Stage-1 predictor training, dropping to ~1.4 after end-to-end joint training
(the VL-shortcut concern). The repo ships the same methodology as
scripts/probe_z_tactile_dependence.py + docs/TACTILE_CAUSAL_PROBE.md, but that probe
needs a data loader over NeoData, which is NOT released.

This script therefore re-implements the probe with a synthetic batch and runs it on
the released n0-vtla-base weights (which are arch C "tactile_kv", n_latent=5 -- NOT the
arch A, 10-token configuration the paper text describes; see config.json on the hub).

Batch: 8 samples with distinct scenes (Physics-IQ conditioning frames as base RGB),
distinct prompts, and synthetic gel readings: baseline frame + a bright contact blob
at a per-sample location (each finger view gets its own blob). This makes the tactile
perturbation a REAL semantic change of the contact signal, while the VL perturbation
swaps scene+prompt to a different sample -- mirroring the shipped probe's
z_shuffle vs z_vlswap comparison.

Groups (per their doc): z_real | z_null (diff zeroed) | z_shuffle (tactile rolled)
| z_vlswap (RGB+prompt rolled) | z_padpert (extra tail tokens masked).
Metric: centered cosine vs z_real, relL2, R = (1-cos_cent(shuffle))/(1-cos_cent(vlswap)),
plus the z_xsample collapse indicator and the learned z_gate value.

Runs on the Inspire notebook, cwd = n0-vtla repo root, PYTHONPATH=$PWD.
  ../vtla-venv/bin/python probe_z_synthetic.py --ckpt /path/to/model.safetensors \
      --frames-dir /path/to/init_frames --out-json probe_synth.json
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image

from n0vtla.models.model import Observation
from n0vtla.models_pytorch.n0vtla_policy import N0VTLAConfig, N0VTLAPolicy

TACTILE_KEYS = (
    "left_wrist_left_tactile",
    "left_wrist_right_tactile",
    "right_wrist_left_tactile",
    "right_wrist_right_tactile",
)
BASELINE_SUFFIX = ".baseline"

PROMPTS = [
    "insert the plug into the socket",
    "fold the towel on the table",
    "pack the bag with the items",
    "fold the cardboard box",
    "stand the bottle upright",
    "place the gears on the peg",
    "stack the bowls neatly",
    "unstack the cup carefully",
]

B = 8
TAC_RES = 224


def load_rgb_frames(frames_dir: str) -> list[torch.Tensor]:
    """8 distinct scene frames as float NCHW in [-1, 1]."""
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))[:B]
    assert len(paths) == B, f"need {B} frames, found {len(paths)} in {frames_dir}"
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((224, 224))
        a = np.asarray(im, dtype=np.float32) / 255.0 * 2.0 - 1.0
        imgs.append(torch.from_numpy(a).permute(2, 0, 1))
    return imgs


def synth_tactile(seed: int) -> dict[str, torch.Tensor]:
    """One sample's 4 finger views + baselines: dark gel background + per-view bright
    contact blob at a seeded position. Returns float NCHW tensors in [-1, 1]."""
    rng = np.random.default_rng(seed)
    out = {}
    for v, key in enumerate(TACTILE_KEYS):
        base = -0.85 + 0.05 * rng.standard_normal((3, TAC_RES, TAC_RES)).astype(np.float32)
        cur = base.copy()
        # contact blob: bright ellipse at a sample-specific location per view
        cx = int(rng.integers(40, TAC_RES - 40))
        cy = int(rng.integers(40, TAC_RES - 40))
        r = int(rng.integers(14, 30))
        yy, xx = np.mgrid[0:TAC_RES, 0:TAC_RES]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
        for c in range(3):
            cur[c][mask] = 0.6 + 0.3 * rng.standard_normal(int(mask.sum())).astype(np.float32)
        out[key] = torch.from_numpy(np.clip(cur, -1, 1))
        out[key + BASELINE_SUFFIX] = torch.from_numpy(np.clip(base, -1, 1))
    return out


def build_obs(scenes, prompts_tokens, prompts_masks, tactile_list, order_rgb, order_tac):
    """Assemble an Observation; order_rgb/order_tac permute sample indices for swaps."""
    images, image_masks = {}, {}
    for i in range(B):
        sc = scenes[order_rgb[i]]
        for cam in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
            images.setdefault(cam, []).append(sc)
            image_masks.setdefault(cam, []).append(True)
        for k, v in tactile_list[order_tac[i]].items():
            images.setdefault(k, []).append(v)
            image_masks.setdefault(k, []).append(True)
    images = {k: torch.stack(v) for k, v in images.items()}
    image_masks = {k: torch.tensor(v) for k, v in image_masks.items()}
    tok = torch.stack([prompts_tokens[order_rgb[i]] for i in range(B)])
    tok_m = torch.stack([prompts_masks[order_rgb[i]] for i in range(B)])
    return Observation(
        images=images,
        image_masks=image_masks,
        state=torch.zeros(B, 32),
        tokenized_prompt=tok,
        tokenized_prompt_mask=tok_m,
    )


@torch.no_grad()
def compute_z(model, obs, device, pad_extra: int = 0):
    obs = Observation(
        images={k: v.to(device) for k, v in obs.images.items()},
        image_masks={k: v.to(device) for k, v in obs.image_masks.items()},
        state=obs.state.to(device),
        tokenized_prompt=obs.tokenized_prompt.to(device),
        tokenized_prompt_mask=obs.tokenized_prompt_mask.to(device),
    )
    if pad_extra > 0:
        m = obs.tokenized_prompt_mask.clone()
        # mask N extra tokens at the tail of the valid span per sample
        for i in range(B):
            valid = m[i].nonzero().max().item()
            lo = max(0, valid - pad_extra + 1)
            m[i, lo : valid + 1] = False
        obs = obs.replace(tokenized_prompt_mask=m)
    images, img_masks, lang_tokens, lang_masks, _state, _ei = model._preprocess_observation(
        obs, train=False
    )
    vl_ctx, _pe, prefix_pad_masks, _pa, _pkv = model._prefix_forward(
        images, img_masks, lang_tokens, lang_masks, use_cache=False
    )
    z, _g, _has = model._compute_z(vl_ctx, prefix_pad_masks)
    return z.float().cpu()


def centered_cos(z_ref, z_var):
    """Mean per-sample centered cosine (batch mean of z_ref subtracted), per their doc."""
    c = z_ref.mean(dim=0, keepdim=True)
    a = (z_ref - c).flatten(1)
    b = (z_var - c).flatten(1)
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


def rel_l2(z_ref, z_var):
    return ((z_var - z_ref).flatten(1).norm(dim=-1) / z_ref.flatten(1).norm(dim=-1)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--tokenizer", default="models/paligemma_tokenizer.model")
    ap.add_argument("--out-json", default="probe_synth.json")
    ap.add_argument("--seed", type=int, default=1000, help="base seed for synthetic tactile blobs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_proto=open(args.tokenizer, "rb").read())
    max_len = 200  # pi05 default (pi0_config post-init)
    toks, masks = [], []
    for p in PROMPTS:
        t = sp.encode(p.strip().replace("\n", " "), add_bos=True) + sp.encode("\n")
        m = [True] * len(t)
        t = t + [0] * (max_len - len(t))
        m = m + [False] * (max_len - len(m))
        toks.append(torch.tensor(t[:max_len], dtype=torch.long))
        masks.append(torch.tensor(m[:max_len], dtype=torch.bool))

    scenes = load_rgb_frames(args.frames_dir)
    tactile_list = [synth_tactile(args.seed + i) for i in range(B)]

    config = N0VTLAConfig(
        pi05=True,
        action_dim=32,
        action_horizon=50,
        pytorch_compile_mode=None,
        tactile_predictor_enabled=True,
        tactile_mode="latent",
        n_latent=5,
        predictor_arch="tactile_kv",
        z_gate_zero_init=True,
        vl_dropout_prob=0.0,
        tactile_image_keys=TACTILE_KEYS,
    )
    model = N0VTLAPolicy(config)
    import safetensors.torch as st

    missing, unexpected = st.load_model(model, args.ckpt, strict=False)
    print(f"ckpt load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  MISSING:", missing[:10])
    if unexpected:
        print("  UNEXPECTED:", unexpected[:10])
    # NB: no global .to(bf16)! PaliGemmaWithExpertModel handles precision internally
    # (precision=config.dtype); the tactile predictor must stay fp32 (its LayerNorms
    # would otherwise receive fp32 activations from _compute_z's explicit casts).
    model = model.to(device).eval()

    gate = getattr(model, "z_gate", None)
    print(f"z_gate value: {None if gate is None else gate.detach().float().cpu().item():.4f}")

    ident = list(range(B))
    roll = list(np.roll(np.arange(B), 1))

    z = {}
    z["real"] = compute_z(model, build_obs(scenes, toks, masks, tactile_list, ident, ident), device)
    # z_null: tactile current == baseline (diff exactly zero)
    null_tac = []
    for t in tactile_list:
        nt = {}
        for k in TACTILE_KEYS:
            nt[k] = t[k + BASELINE_SUFFIX].clone()
            nt[k + BASELINE_SUFFIX] = t[k + BASELINE_SUFFIX]
        null_tac.append(nt)
    z["null"] = compute_z(model, build_obs(scenes, toks, masks, null_tac, ident, ident), device)
    # z_shuffle: tactile rolled by one, VL untouched
    z["shuffle"] = compute_z(model, build_obs(scenes, toks, masks, tactile_list, ident, roll), device)
    # z_vlswap: RGB+prompt rolled by one, tactile untouched
    z["vlswap"] = compute_z(model, build_obs(scenes, toks, masks, tactile_list, roll, ident), device)
    # z_padpert: mask 8 extra tail tokens
    z["padpert"] = compute_z(
        model, build_obs(scenes, toks, masks, tactile_list, ident, ident), device, pad_extra=8
    )
    # self-check: recompute z_real, must be ~identical
    z2 = compute_z(model, build_obs(scenes, toks, masks, tactile_list, ident, ident), device)
    print(f"self-check max |dz_real|: {(z2 - z['real']).abs().max().item():.2e}")

    ref = z["real"]
    out = {"groups": {}}
    for name in ("null", "shuffle", "vlswap", "padpert"):
        out["groups"][name] = {
            "cos_cent": centered_cos(ref, z[name]),
            "one_minus_cos_cent": 1.0 - centered_cos(ref, z[name]),
            "relL2": rel_l2(ref, z[name]),
        }
        print(f"{name:8s} cos_cent={out['groups'][name]['cos_cent']:.4f} "
              f"relL2={out['groups'][name]['relL2']:.4f}")
    # collapse indicator: centered cos between z_real[i], z_real[i+1]
    xs = centered_cos(ref, torch.roll(ref, 1, dims=0))
    out["z_xsample_cos_cent"] = xs
    print(f"z_xsample cos_cent={xs:.4f} (near 1.0 => collapse)")
    num = out["groups"]["shuffle"]["one_minus_cos_cent"]
    den = out["groups"]["vlswap"]["one_minus_cos_cent"]
    out["R_tactile_over_vl"] = num / den if den > 1e-9 else float("nan")
    out["z_gate"] = None if gate is None else gate.detach().float().cpu().item()
    out["ckpt_missing"] = sorted(missing)
    out["ckpt_unexpected"] = sorted(unexpected)
    print(f"R = tactile/VL sensitivity = {out['R_tactile_over_vl']:.3f}")

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("PROBE_DONE")


if __name__ == "__main__":
    main()
