#!/usr/bin/env python3
"""Load the official LeWM cube checkpoint (quentinll/lewm-cube) without stable-worldmodel.

Uses stable_pretraining's ViT factory + le-wm's jepa/module classes.
Config mirrors quentinll/lewm-cube config.json (ViT-tiny patch14 @224,
Predictor 6x192, action Embedder 25->192, two BatchNorm MLP projectors).
"""
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lewm_swm import LeWM  # noqa: E402
from lewm_swm_module import Predictor, Embedder, MLP  # noqa: E402


def load_lewm(weights=None, device="cuda"):
    if weights is None:
        weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "lewm_cube_weights.pt")
    from transformers import ViTConfig, ViTModel
    cfg = ViTConfig(hidden_size=192, num_hidden_layers=12, num_attention_heads=12,
                    intermediate_size=768, patch_size=14, image_size=224)
    encoder = ViTModel(cfg)
    predictor = Predictor(num_frames=3, input_dim=192, hidden_dim=192, output_dim=192,
                          depth=6, heads=16, mlp_dim=2048, dim_head=64,
                          dropout=0.1, emb_dropout=0.0)
    action_encoder = Embedder(input_dim=25, emb_dim=192)
    projector = MLP(input_dim=192, output_dim=192, hidden_dim=2048,
                    norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=192, output_dim=192, hidden_dim=2048,
                    norm_fn=torch.nn.BatchNorm1d)
    model = LeWM(encoder=encoder, predictor=predictor, action_encoder=action_encoder,
                 projector=projector, pred_proj=pred_proj)
    sd = torch.load(weights, map_location="cpu", weights_only=False)
    # quentinll's release uses transformers 4.x ViT naming; local transformers 5.x
    # uses the timm-style names. Remap 4.x -> 5.x.
    ren = {}
    for k, v in sd.items():
        nk = k.replace("encoder.encoder.layer.", "encoder.layers.")
        nk = nk.replace("attention.attention.query", "attention.q_proj")
        nk = nk.replace("attention.attention.key", "attention.k_proj")
        nk = nk.replace("attention.attention.value", "attention.v_proj")
        nk = nk.replace("attention.output.dense", "attention.o_proj")
        nk = nk.replace("intermediate.dense", "mlp.fc1")
        nk = nk.replace("output.dense", "mlp.fc2")
        ren[nk] = v
    model.load_state_dict(ren, strict=False)
    model.requires_grad_(False).eval().to(device)
    return model


if __name__ == "__main__":
    import numpy as np
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    model = load_lewm()
    eps = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cube_smoke.pt"), weights_only=False)
    pix = eps[0]["states"][0]["pixels"].astype(np.float32) / 255.0
    x = (pix - MEAN) / STD
    x = torch.from_numpy(x).permute(2, 0, 1)[None, None].cuda()
    with torch.no_grad():
        out = model.encode({"pixels": x})
    print("emb", out["emb"].shape, float(out["emb"].norm()))
