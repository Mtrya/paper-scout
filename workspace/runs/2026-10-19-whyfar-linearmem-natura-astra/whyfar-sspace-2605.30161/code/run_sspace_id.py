"""S-Space axis identification on Qwen2.5-VL-3B.

Paper method (S-Space Eq. 2): for a relative spatial question, take the
answer-contrast scalar s_g = logit(positive) - logit(negative), differentiate
w.r.t. the query and reference object-token activations, form the
role-antisymmetric difference (grad_q - grad_r)/2, and average across
templates, object orders, and examples.  The normalized mean is the axis.

We use EmbSpatial pairs in the same binary-question format as the WFLU probe,
with an identification split disjoint from the WFLU 200/category sample.
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    CATS, GROUPS, NEGATIVE_WORD, OUT_DIR, POSITIVE_WORD, build_inputs,
    decode_image, find_name_position, get_lm_layers, load_model, load_pairs,
    sample_per_category, save_json, set_seed,
)

TEMPLATES = {
    "horizontal": [
        "Is the {q} to the left or right of the {r}? Answer with only one word.",
        "In the image, is the {q} to the left or right of the {r}? Answer with only one word.",
    ],
    "vertical": [
        "Is the {q} above or below the {r}? Answer with only one word.",
        "In the image, is the {q} above or below the {r}? Answer with only one word.",
    ],
    "distance": [
        "Compared to {r}, is {q} far or close from you? Answer with only one word.",
        "Is the {q} farther from or closer to the camera than the {r}? Answer with far or close.",
    ],
}


def word_token_id(tok, word):
    ids = tok(" " + word, add_special_tokens=False).input_ids
    if len(ids) != 1:
        # fall back to the bare word; log its first token
        ids = tok(word, add_special_tokens=False).input_ids
    return ids[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-category", type=int, default=50)
    ap.add_argument("--wflu-n", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "sspace_qwen25"))
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    ap.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--layers", type=int, nargs="+",
                    default=[0, 4, 8, 12, 15, 18, 21, 24, 28, 32, 35])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(0)

    all_pairs = load_pairs()
    wflu = sample_per_category(all_pairs, args.wflu_n, seed=42)
    wflu_idx = {p["index"] for p in wflu}
    pairs = sample_per_category(all_pairs, args.n_per_category, seed=7,
                                exclude_indices=wflu_idx)
    print(f"identification pairs: {len(pairs)}; "
          f"{ {c: sum(p['category'] == c for p in pairs) for c in CATS} }")

    model, processor = load_model(max_pixels=args.max_pixels,
                                  min_pixels=args.min_pixels, freeze_all=True)
    lm_layers = get_lm_layers(model)
    layer_ids = args.layers
    tok = processor.tokenizer
    word_ids = {g: (word_token_id(tok, POSITIVE_WORD[g]),
                    word_token_id(tok, NEGATIVE_WORD[g]))
                for g in POSITIVE_WORD}
    print("answer token ids:", word_ids)

    # hooks save the layer output tensor (graph-connected, not detached)
    saved = {}
    handles = []
    for li in layer_ids:
        def make_hook(layer_idx):
            def hook(module, inp, out):
                value = out[0] if isinstance(out, tuple) else out
                value.requires_grad_(True)
                saved[layer_idx] = value
            return hook
        handles.append(lm_layers[li].register_forward_hook(make_hook(li)))

    sums = np.zeros((len(layer_ids), 3, model.config.text_config.hidden_size),
                    dtype=np.float64)
    counts = np.zeros((len(layer_ids), 3), dtype=np.int64)
    records = []
    fail = 0
    done = set()
    partial_path = os.path.join(args.out, "partial.npz")
    if os.path.exists(partial_path):
        z = np.load(partial_path)
        sums = z["sums"]
        counts = z["counts"]
        done = set(int(x) for x in z["done"])
        print(f"resuming: {len(done)} samples already done")

    n_since_save = 0
    for pair in tqdm(pairs, desc="S-Space identification"):
        if int(pair["index"]) in done:
            continue
        image = decode_image(pair["image_base64"])
        g = pair["group"]
        gi = {"horizontal": 0, "vertical": 1, "distance": 2}[g]
        q_name, r_name = pair["obj_q"], pair["obj_r"]
        for t_i, template in enumerate(TEMPLATES[g]):
            for order in (0, 1):
                qq, rr = (q_name, r_name) if order == 0 else (r_name, q_name)
                question = template.format(q=qq, r=rr)
                inp = build_inputs(processor, image, question).to("cuda")
                ids = inp["input_ids"][0].tolist()
                pos_q = find_name_position(ids, tok, qq)
                pos_r = find_name_position(ids, tok, rr)
                if pos_q is None or pos_r is None:
                    fail += 1
                    continue
                saved.clear()
                out = model(**inp)
                logits = out.logits[0, -1]
                pos_id, neg_id = word_ids[g]
                s = logits[pos_id] - logits[neg_id]
                targets = [saved[li] for li in layer_ids]
                grads = torch.autograd.grad(
                    s, targets, retain_graph=False, allow_unused=True)
                a = np.zeros_like(sums[:, gi, :])
                for li, gr in enumerate(grads):
                    if gr is None:
                        continue
                    gq = gr[0, pos_q].float().cpu().numpy()
                    gr_ = gr[0, pos_r].float().cpu().numpy()
                    a[li] = 0.5 * (gq - gr_)
                    sums[li, gi] += a[li]
                    counts[li, gi] += 1
                records.append({
                    "index": int(pair["index"]), "category": pair["category"],
                    "group": g, "template": t_i, "order": order,
                    "s": float(s.item()), "a": a.astype(np.float32),
                })
                del out, logits, s, grads, targets, saved
                saved = {}
                torch.cuda.empty_cache()
        done.add(int(pair["index"]))
        n_since_save += 1
        if n_since_save >= 5:
            np.savez(partial_path, sums=sums, counts=counts,
                     done=np.array(sorted(done), dtype=np.int64))
            n_since_save = 0

    np.savez(partial_path, sums=sums, counts=counts,
             done=np.array(sorted(done), dtype=np.int64))

    for h in handles:
        h.remove()

    means = sums / np.maximum(counts, 1)[:, :, None]
    norms = np.linalg.norm(means, axis=2)
    axes = means / np.maximum(norms, 1e-12)[:, :, None]

    # diagnostics: angle between S-Space axes (entanglement inside the basis)
    diag = {}
    for li, L in enumerate(layer_ids):
        diag[str(L)] = {
            "cos_x_y": float(np.dot(axes[li, 0], axes[li, 1])),
            "cos_x_z": float(np.dot(axes[li, 0], axes[li, 2])),
            "cos_y_z": float(np.dot(axes[li, 1], axes[li, 2])),
            "norm_x": float(norms[li, 0]), "norm_y": float(norms[li, 1]),
            "norm_z": float(norms[li, 2]), "n": counts[li].tolist(),
        }

    np.save(os.path.join(args.out, "axes.npy"), axes.astype(np.float32))
    np.save(os.path.join(args.out, "axis_norms.npy"), norms)
    np.save(os.path.join(args.out, "layer_ids.npy"), np.array(layer_ids))
    np.savez_compressed(
        os.path.join(args.out, "grad_records.npz"),
        indices=np.array([r["index"] for r in records]),
        categories=np.array([r["category"] for r in records]),
        groups=np.array([r["group"] for r in records]),
        templates=np.array([r["template"] for r in records]),
        orders=np.array([r["order"] for r in records]),
        scores=np.array([r["s"] for r in records], dtype=np.float32),
        avecs=np.stack([r["a"] for r in records]).astype(np.float32),
    )
    save_json(os.path.join(args.out, "axis_diagnostics.json"), diag)
    print(f"saved axes to {args.out}; failures={fail}")
    for L in layer_ids:
        d = diag[str(L)]
        print(f"L{L:02d} cos(y,z)={d['cos_y_z']:+.3f} "
              f"cos(x,y)={d['cos_x_y']:+.3f} cos(x,z)={d['cos_x_z']:+.3f}")


if __name__ == "__main__":
    main()
