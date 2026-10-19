"""Smoke test: model load, object-token localization, answer logits, gradients."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    CATS, build_inputs, decode_image, find_name_position, get_lm_layers,
    load_model, load_pairs,
)
from run_sspace_id import TEMPLATES, word_token_id  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    args = ap.parse_args()
    pairs = load_pairs()
    model, processor = load_model(max_pixels=args.max_pixels)
    print("after load allocated GB", torch.cuda.memory_allocated() / 1e9,
          "reserved", torch.cuda.memory_reserved() / 1e9)
    lm_layers = get_lm_layers(model)
    tok = processor.tokenizer
    print("layers", len(lm_layers), "hidden",
          model.config.text_config.hidden_size)
    print("words", {w: tok(" " + w, add_special_tokens=False).input_ids
                    for w in ["left", "right", "above", "below", "far", "close"]})

    seen = set()
    for p in pairs:
        if p["category"] in seen:
            continue
        seen.add(p["category"])
        image = decode_image(p["image_base64"])
        inp = build_inputs(processor, image, p["original_question"])
        ids = inp["input_ids"][0].tolist()
        pq = find_name_position(ids, tok, p["obj_q"])
        pr = find_name_position(ids, tok, p["obj_r"])
        print(f"\n[{p['category']}] {p['original_question']}")
        print(f"  q={p['obj_q']!r}@{pq} r={p['obj_r']!r}@{pr}")
        print("  ids", len(ids), "image tokens",
              sum(tok.convert_ids_to_tokens(i) == "<|image_pad|>" for i in ids))
        print("  mem before forward GB", round(torch.cuda.memory_allocated() / 1e9, 2))
        if pq is not None:
            print("  q context:", tok.decode(ids[max(0, pq - 3):pq + 3]))
        if pr is not None:
            print("  r context:", tok.decode(ids[max(0, pr - 3):pr + 3]))
        inp = inp.to("cuda")
        with torch.no_grad():
            out = model(**inp)
        logits = out.logits[0, -1]
        print("  answer logits:", {
            w: round(float(logits[word_token_id(tok, w)]), 3)
            for w in ["left", "right", "above", "below", "far", "close"]})
        del out, logits
        torch.cuda.empty_cache()

    # one gradient pass
    p = next(p for p in pairs if p["category"] == "far")
    image = decode_image(p["image_base64"])
    q = TEMPLATES["distance"][0].format(q=p["obj_q"], r=p["obj_r"])
    inp = build_inputs(processor, image, q).to("cuda")
    ids = inp["input_ids"][0].tolist()
    pq = find_name_position(ids, tok, p["obj_q"])
    pr = find_name_position(ids, tok, p["obj_r"])
    saved = {}
    handles = [lm_layers[i].register_forward_hook(
        lambda m, i, o, L=i: saved.__setitem__(
            L, o[0] if isinstance(o, tuple) else o)) for i in [28]]
    out = model(**inp)
    s = out.logits[0, -1, word_token_id(tok, "close")] - \
        out.logits[0, -1, word_token_id(tok, "far")]
    grads = torch.autograd.grad(s, [saved[28]])
    g = grads[0][0]
    print("\ngradient pass:", float(s), "norm q", float(g[pq].norm()),
          "norm r", float(g[pr].norm()))
    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
