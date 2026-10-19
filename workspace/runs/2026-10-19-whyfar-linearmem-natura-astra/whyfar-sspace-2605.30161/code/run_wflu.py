"""Extract WFLU contrastive-probe features from Qwen2.5-VL-3B.

Two prefill forwards per sample (original and object-swapped question) give:
  - last-token hidden states at every transformer layer -> WFLU delta vectors
  - object-name-token hidden states in the original prompt -> S-Space readout

Raw arrays are written to the ignored lab bench (code/probe_out/).
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    CATS, GROUPS, OUT_DIR, build_inputs, decode_image, find_name_position,
    forward_last_and_obj, get_lm_layers, load_model, load_pairs, sample_per_category,
    set_seed, save_json,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-category", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test limit (0 = all)")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "wflu_qwen25"))
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    ap.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(0)

    pairs = load_pairs()
    pairs = sample_per_category(pairs, args.n_per_category, seed=42)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"{len(pairs)} pairs; categories: "
          f"{ {c: sum(p['category'] == c for p in pairs) for c in CATS} }")

    model, processor = load_model(max_pixels=args.max_pixels,
                                  min_pixels=args.min_pixels)
    lm_layers = get_lm_layers(model)
    layer_ids = list(range(len(lm_layers)))
    tok = processor.tokenizer
    print(f"{len(layer_ids)} transformer layers, hidden dim "
          f"{model.config.text_config.hidden_size}")

    deltas, hs_q, hs_r, metas = [], [], [], []
    failures = 0

    for pair in tqdm(pairs, desc="WFLU pairs"):
        image = decode_image(pair["image_base64"])
        inp_o = build_inputs(processor, image, pair["original_question"])
        inp_s = build_inputs(processor, image, pair["swapped_question"])
        ids = inp_o["input_ids"][0].tolist()
        pos_q = find_name_position(ids, tok, pair["obj_q"])
        pos_r = find_name_position(ids, tok, pair["obj_r"])
        if pos_q is None or pos_r is None:
            failures += 1
            if failures <= 20:
                print(f"[warn] object not found: {pair['obj_q']!r} / "
                      f"{pair['obj_r']!r} in {pair['original_question']!r}")
            continue
        inp_o = inp_o.to("cuda")
        inp_s = inp_s.to("cuda")
        last_o, obj_o = forward_last_and_obj(
            model, lm_layers, inp_o, {"q": pos_q, "r": pos_r}, layer_ids)
        last_s, _ = forward_last_and_obj(
            model, lm_layers, inp_s, {}, layer_ids)

        deltas.append((last_s - last_o).to(torch.float16).numpy())
        hs_q.append(obj_o["q"].to(torch.float16).numpy())
        hs_r.append(obj_o["r"].to(torch.float16).numpy())
        metas.append({
            "index": int(pair["index"]),
            "category": pair["category"],
            "group": GROUPS[pair["category"]],
            "obj_q": pair["obj_q"],
            "obj_r": pair["obj_r"],
            "pos_q": int(pos_q),
            "pos_r": int(pos_r),
            "question": pair["original_question"],
            "swapped_question": pair["swapped_question"],
            "sign_q_is_positive": int(pair["sign_q_is_positive"]),
        })

    print(f"kept {len(metas)} / {len(pairs)} pairs ({failures} object-locate failures)")
    np.save(os.path.join(args.out, "delta.npy"), np.stack(deltas))
    np.save(os.path.join(args.out, "hs_q.npy"), np.stack(hs_q))
    np.save(os.path.join(args.out, "hs_r.npy"), np.stack(hs_r))
    np.save(os.path.join(args.out, "layer_ids.npy"), np.array(layer_ids))
    save_json(os.path.join(args.out, "meta.json"), metas)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
