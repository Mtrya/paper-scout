"""Memory probe: one forward+backward at a given max_pixels."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (build_inputs, decode_image, find_name_position,
                    get_lm_layers, load_model, load_pairs)
from run_sspace_id import word_token_id  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pixels", type=int, default=1003520)
    ap.add_argument("--min-area", type=int, default=0)
    args = ap.parse_args()
    pairs = load_pairs()
    p = None
    for cand in pairs:
        if cand["category"] != "far":
            continue
        im = decode_image(cand["image_base64"])
        if im.width * im.height >= args.min_area:
            p = cand
            break
    if p is None:
        raise SystemExit("no matching image")
    image = decode_image(p["image_base64"])
    q = f"Compared to {p['obj_r']}, is {p['obj_q']} far or close from you? Answer with only one word."
    model, processor = load_model(max_pixels=args.max_pixels, freeze_all=True)
    lm = get_lm_layers(model)
    tok = processor.tokenizer
    inp = build_inputs(processor, image, q).to("cuda")
    ids = inp["input_ids"][0].tolist()
    pq = find_name_position(ids, tok, p["obj_q"])
    pr = find_name_position(ids, tok, p["obj_r"])
    print("seq", len(ids), "img tokens",
          sum(tok.convert_ids_to_tokens(i) == "<|image_pad|>" for i in ids))
    saved = {}

    def hook(module, inp_, out):
        value = out[0] if isinstance(out, tuple) else out
        value.requires_grad_(True)
        saved[28] = value

    handles = [lm[28].register_forward_hook(hook)]
    out = model(**inp)
    s = out.logits[0, -1, word_token_id(tok, "close")] - \
        out.logits[0, -1, word_token_id(tok, "far")]
    grads = torch.autograd.grad(s, [saved[28]])
    print("grad norm", float(grads[0][0, pq].norm()),
          "max mem GB", torch.cuda.max_memory_allocated() / 1e9)
    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
