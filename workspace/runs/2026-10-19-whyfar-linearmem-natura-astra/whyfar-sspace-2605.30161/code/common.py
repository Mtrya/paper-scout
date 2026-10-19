"""Shared utilities for the WFLU / S-Space cross-probe on Qwen2.5-VL-3B.

All probes use prefill-only forwards (batch size 1).  The WFLU official code
calls ``generate`` and hooks the prefill hidden states; a single forward pass
produces exactly the same last-token states, without spending time decoding.
"""
import base64
import json
import os
import random
import re
import sys
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

WORKSPACE = os.environ.get(
    "PAPER_SCOUT_WS",
    os.path.dirname(os.path.abspath(__file__)) + "/../../../..",
)
REPO = os.path.join(WORKSPACE, "code", "contrastive-probing")
DATA_TSV = os.path.join(WORKSPACE, "code", "data", "EmbSpatial-Bench.tsv")
MODEL_PATH = os.path.join(WORKSPACE, "code", "models", "Qwen2.5-VL-3B-Instruct")
OUT_DIR = os.path.join(WORKSPACE, "code", "probe_out")

sys.path.insert(0, REPO)
from probing import (  # noqa: E402
    CATEGORY_ORDER,
    OPPOSITE_MAP,
    SHORT_TEMPLATES,
    load_swap_pairs,
    extract_objects,
)

CATS = CATEGORY_ORDER
GROUPS = {"left": "horizontal", "right": "horizontal",
          "above": "vertical", "below": "vertical",
          "far": "distance", "close": "distance"}
POSITIVE_WORD = {"horizontal": "right", "vertical": "above", "distance": "close"}
NEGATIVE_WORD = {"horizontal": "left", "vertical": "below", "distance": "far"}


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def decode_image(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


def load_pairs():
    """All contrastive pairs, with a unified query/reference naming."""
    pairs = load_swap_pairs(DATA_TSV, seed=42)
    for p in pairs:
        if p["group"] == "distance":
            p["obj_q"] = p["target_object"]
            p["obj_r"] = p["reference_object"]
            # q is the object named by the correct answer: far or close.
            p["sign_q_is_positive"] = 1 if p["category"] == "close" else -1
        else:
            p["obj_q"] = p["obj1"]
            p["obj_r"] = p["obj2"]
            p["sign_q_is_positive"] = 0  # sign depends on category, not fixed
        p["group_name"] = p["group"]
    return pairs


def sample_per_category(pairs, n, seed, exclude_indices=None):
    rng = random.Random(seed)
    excl = set(exclude_indices or [])
    grouped = defaultdict(list)
    for p in pairs:
        if p["index"] not in excl:
            grouped[p["category"]].append(p)
    out = []
    for cat in CATS:
        s = grouped[cat]
        if n and len(s) > n:
            s = rng.sample(s, n)
        out.extend(s)
    return out


# ---------------------------------------------------------------------------
# Model / processor
# ---------------------------------------------------------------------------

def load_model(max_pixels: int = 1280 * 28 * 28, min_pixels: int = 3136,
               freeze_vision: bool = True, freeze_all: bool = False):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    # The vision tower is below every text layer we probe; freezing it keeps
    # its activations out of the autograd graph during gradient identification.
    if freeze_all:
        for p in model.parameters():
            p.requires_grad_(False)
    elif freeze_vision:
        for p in model.visual.parameters():
            p.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, max_pixels=max_pixels, min_pixels=min_pixels
    )
    return model, processor


def build_inputs(processor, image, question):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return inputs


def _ids(tokenizer, s):
    return tokenizer(s, add_special_tokens=False).input_ids


def find_name_position(input_ids: List[int], tokenizer, name: str) -> Optional[int]:
    """Return the last token index of *name* in the expanded input_ids."""
    name = str(name).strip()
    variants = [name, " " + name, name.lower(), " " + name.lower(),
                name.capitalize(), " " + name.capitalize()]
    # longest first, so multi-token names are matched as a whole
    for v in sorted(set(variants), key=len, reverse=True):
        sub = _ids(tokenizer, v)
        if not sub:
            continue
        n, m = len(input_ids), len(sub)
        hits = [i for i in range(n - m + 1) if input_ids[i:i + m] == sub]
        if hits:
            return hits[-1] + m - 1
    # fallback: first token of the name, still searching the full sequence
    sub = _ids(tokenizer, " " + name)
    if sub:
        n, m = len(input_ids), len(sub)
        hits = [i for i in range(n - m + 1) if input_ids[i:i + m] == sub]
        if hits:
            return hits[-1] + m - 1
    return None


def get_lm_layers(model):
    sys.path.insert(0, REPO)
    from probing import _find_llm_layers
    layers = _find_llm_layers(model, [
        ["model", "layers"],
        ["model", "language_model", "layers"],
        ["model", "language_model", "model", "layers"],
    ], hint="Qwen2.5-VL")
    return layers


@torch.no_grad()
def forward_last_and_obj(model, lm_layers, inputs, obj_positions, layer_ids):
    """Forward once; return last-token hidden states and object-token states.

    Returns:
      last: [n_layers, d]
      obj:  dict name -> [n_layers, d]
    """
    captured = {}

    def make_hook(layer_idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = h[0]  # [seq, d]
        return hook

    handles = [lm_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
               for layer_idx in layer_ids]
    try:
        model(**inputs)
    finally:
        for h in handles:
            h.remove()

    last = torch.stack([captured[l][-1] for l in layer_ids], dim=0).float().cpu()
    obj = {}
    for name, pos in obj_positions.items():
        if pos is None:
            obj[name] = None
        else:
            obj[name] = torch.stack(
                [captured[l][pos] for l in layer_ids], dim=0
            ).float().cpu()
    return last, obj


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cos(a, b, eps=1e-12):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def mean_pairwise_cos(deltas):
    if len(deltas) < 2:
        return float("nan")
    x = np.asarray(deltas, dtype=np.float64)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    sim = x @ x.T
    iu = np.triu_indices(len(x), k=1)
    return float(sim[iu].mean())


def axis_coherence(deltas_by_cat, group, layer=None):
    """WFLU Eq. 2: sign-correct and pool both directions of an axis.

    ``deltas_by_cat`` maps category -> list of delta vectors (np arrays).
    """
    canon = {"horizontal": "left", "vertical": "above", "distance": "far"}[group]
    opp = OPPOSITE_MAP[canon]
    vecs = []
    for cat in (canon, opp):
        for d in deltas_by_cat.get(cat, []):
            vecs.append(d if cat == canon else -d)
    return mean_pairwise_cos(vecs), len(vecs)


def vd_ei_eq3(mean_by_cat):
    """Paper Eq. 3, computed from category mean delta vectors."""
    m = mean_by_cat
    return 0.25 * (
        cos(m["above"], m["far"]) + cos(m["below"], m["close"])
        - cos(m["above"], m["close"]) - cos(m["below"], m["far"])
    )


def vd_ei_repo(mean_by_cat):
    """Official repo implementation: cos(mean sign-corrected vertical, same for distance)."""
    v = 0.5 * (mean_by_cat["above"] - mean_by_cat["below"])
    d = 0.5 * (mean_by_cat["far"] - mean_by_cat["close"])
    return cos(v, d)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
