"""Extract frozen Qwen3-VL-2B-Instruct hidden states for every frame of selected episodes.

Per frame the slow branch stores, for four routed depths (25/50/75/100% of the backbone):
  * ``vis``: hidden states of the visual tokens of both cameras (agentview, wrist), fp16;
  * ``txt``: mean-pooled hidden state of the non-image tokens (the instruction, which attends to the images), fp16.
The raw uint8 frames of both cameras are stored too, so the trainable fresh-vision encoder never re-decodes JPEG.

Run: python extract_features.py --split train|val --out <dir> [--limit N] [--tasks ...]
Resumable: an episode whose .npz already exists is skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dashdata as D  # noqa: E402

LAYER_FRACS = (0.25, 0.5, 0.75, 1.0)


def layer_indices(n_layers: int):
    return [max(1, int(round(f * n_layers))) for f in LAYER_FRACS]


def build_batch(processor, frames, task, image_token_id, pad_id):
    """frames: list of (img_agent, img_wrist) uint8 arrays.  One batched processor call: the jinja template and the
    image preprocessing are the CPU bottleneck of this pipeline, and batching them is ~6x faster than per-sample."""
    convs = [[{"role": "user", "content": [
        {"type": "image"}, {"type": "image"}, {"type": "text", "text": D.PROMPTS[task]}]}] for _ in frames]
    texts = processor.apply_chat_template(convs, tokenize=False, add_generation_prompt=False)
    images = [[a, w] for a, w in frames]
    try:
        enc = processor(text=texts, images=images, return_tensors="pt", padding=True)
        if "mm_token_type_ids" not in enc:
            raise KeyError("mm_token_type_ids")
        return dict(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    mm_token_type_ids=enc["mm_token_type_ids"],
                    pixel_values=enc["pixel_values"], image_grid_thw=enc["image_grid_thw"])
    except Exception as e:  # noqa: BLE001
        print(f"[build_batch] batched processor failed ({e!r}); falling back to per-sample", flush=True)
    pv, gthw, ids, mm = [], [], [], []
    for t, im in zip(texts, images):
        enc = processor(text=[t], images=im, return_tensors="pt")
        pv.append(enc["pixel_values"])
        gthw.append(enc["image_grid_thw"])
        ids.append(enc["input_ids"][0])
        mm.append(enc["mm_token_type_ids"][0] if "mm_token_type_ids" in enc else torch.zeros_like(enc["input_ids"][0]))
    L = max(x.numel() for x in ids)
    input_ids = torch.full((len(ids), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(ids), L), dtype=torch.long)
    mm_ids = torch.zeros((len(ids), L), dtype=torch.long)
    for i, (x, m) in enumerate(zip(ids, mm)):
        input_ids[i, : x.numel()] = x
        attn[i, : x.numel()] = 1
        mm_ids[i, : x.numel()] = m
    return dict(input_ids=input_ids, attention_mask=attn, mm_token_type_ids=mm_ids,
                pixel_values=torch.cat(pv, 0), image_grid_thw=torch.cat(gthw, 0))


@torch.inference_mode()
def run_model(model, batch, layers, image_token_id):
    batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    out = model(**batch, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple of L+1
    ids = batch["input_ids"]
    am = batch["attention_mask"].bool()
    vis_mask = (ids == image_token_id) & am
    txt_mask = am & ~vis_mask
    n_vis = int(vis_mask.sum(1)[0].item())
    vis = torch.stack([hs[l][vis_mask].reshape(len(ids), n_vis, -1) for l in layers], dim=1)
    txt = torch.stack([(hs[l] * txt_mask.unsqueeze(-1)).sum(1)
                       / txt_mask.sum(1, keepdim=True).clamp_min(1) for l in layers], dim=1)
    return vis.float().cpu().numpy().astype(np.float16), txt.float().cpu().numpy().astype(np.float16), n_vis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only the first N episodes per task (smoke)")
    ap.add_argument("--tasks", nargs="+", default=D.TASKS)
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--n-val", type=int, default=12)
    args = ap.parse_args()

    data_root, out = Path(args.data_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split = D.split_episodes(data_root, args.n_train, args.n_val)
    (out / f"split_{args.split}.json").write_text(json.dumps(
        {t: [p.name for p in split[t][args.split]] for t in D.TASKS}, indent=1))

    proc = AutoProcessor.from_pretrained(args.model)
    # keep the 128x128 native frames at native resolution (no upsampling: 16 visual tokens per camera)
    proc.image_processor.size = {"shortest_edge": 128 * 128, "longest_edge": 128 * 128}
    print(f"[proc] size={proc.image_processor.size}", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    n_layers = model.config.text_config.num_hidden_layers
    layers = layer_indices(n_layers)
    image_token_id = getattr(model.config, "image_token_id", None) or proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    pad_id = proc.tokenizer.pad_token_id or 0
    print(f"[model] layers={n_layers} routed={layers} image_token_id={image_token_id} "
          f"dtype={next(model.parameters()).dtype}", flush=True)
    meta = dict(model=args.model, n_layers=n_layers, layers=layers, layer_fracs=LAYER_FRACS,
                prompt_template="user: <image><image><instruction>", split=args.split)
    (out / f"meta_{args.split}.json").write_text(json.dumps(meta, indent=1))

    t0 = time.time()
    n_frames_done = 0
    for task in args.tasks:
        files = split[task][args.split]
        if args.limit:
            files = files[: args.limit]
        for ep in files:
            dst = out / task / (ep.stem + ".npz")
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            d = D.load_episode(ep)
            T = d["T"]
            imgs = np.stack([np.stack([D.decode_frame(d, c, i) for c in D.CAMS], 0) for i in range(T)], 0)
            vis_all, txt_all, n_vis = None, None, None
            for s in range(0, T, args.batch):
                e = min(T, s + args.batch)
                b = build_batch(proc, [(imgs[i, 0], imgs[i, 1]) for i in range(s, e)],
                                task, image_token_id, pad_id)
                vis, txt, nv = run_model(model, b, layers, image_token_id)
                if vis_all is None:
                    n_vis = nv
                    vis_all = np.empty((T, len(layers), n_vis, vis.shape[-1]), np.float16)
                    txt_all = np.empty((T, len(layers), txt.shape[-1]), np.float16)
                vis_all[s:e] = vis
                txt_all[s:e] = txt
            np.savez(dst, vis=vis_all, txt=txt_all, state=d["proprio"].astype(np.float32),
                     act=d["actions13"].astype(np.float32), img=imgs, n_vis=n_vis,
                     seed=np.int32(int(ep.stem.split("_")[1])), task=task, T=np.int32(T))
            n_frames_done += T
            if (n_frames_done // 5000) != ((n_frames_done - T) // 5000):
                print(f"[extract] {n_frames_done} frames, {(time.time()-t0)/max(1,n_frames_done)*1e3:.1f} ms/frame",
                      flush=True)
    print(f"[done] {args.split}: {n_frames_done} new frames in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
