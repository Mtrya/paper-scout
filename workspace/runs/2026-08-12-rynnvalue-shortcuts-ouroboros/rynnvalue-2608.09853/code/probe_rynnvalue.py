"""RynnValue shortcut stress probe.

One source video, six controlled perturbations, official prefix protocol.
Tests whether predicted remaining-time v_t is grounded in visual evidence
(paper's central claim) or in sequence position / presentation order.

Conditions:
  forward   control: chronological video
  reversed  frames reversed: grounded v should INCREASE over time
  frozen    first 40% then repeat frame: grounded v should plateau
  rewind    0->60%, back to 30%, forward to end: v bump during regression
  shuffle   uniform permutation: per-prefix v should track LAST FRAME's true
            timestamp, not prefix length
  truncate  first 90% only: near-completion sensitivity (v_end should stay >0)
Plus mismatch analysis: forward video + wrong instruction -> Match/Success No.
"""
import argparse
import json
import os

import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor


def load_video_frames(video_path):
    reader = imageio.get_reader(video_path)
    try:
        frames = [Image.fromarray(f).convert("RGB") for f in reader]
    finally:
        reader.close()
    return frames


def resize_frames(frames, max_side):
    w, h = frames[0].size
    if max(w, h) <= max_side:
        return frames
    scale = max_side / max(w, h)
    new = (int(round(w * scale)), int(round(h * scale)))
    return [f.resize(new, resample=Image.BICUBIC) for f in frames]


def sample_frame_indices(total, num_steps):
    if num_steps <= 0 or num_steps >= total:
        return list(range(total))
    step = (total - 1) / (num_steps - 1)
    return sorted(set(int(round(j * step)) for j in range(num_steps)))


def build_conditions(frames):
    T = len(frames)
    cond = {}
    cond["forward"] = (frames, list(range(T)))
    rev_idx = list(range(T - 1, -1, -1))
    cond["reversed"] = ([frames[i] for i in rev_idx], rev_idx)
    cut = int(T * 0.4)
    froz_idx = list(range(cut)) + [cut - 1] * (T - cut)
    cond["frozen"] = ([frames[i] for i in froz_idx], froz_idx)
    cut80 = int(T * 0.8)
    froz80_idx = list(range(cut80)) + [cut80 - 1] * (T - cut80)
    cond["frozen80"] = ([frames[i] for i in froz80_idx], froz80_idx)
    a, b = int(T * 0.45), int(T * 0.55)
    loop_idx = list(range(b)) + list(range(a, b)) * 2 + list(range(b, T))
    cond["loop"] = ([frames[i] for i in loop_idx], loop_idx)
    lo, hi = int(T * 0.35), int(T * 0.76)
    cond["loopdense"] = ([frames[i] for i in loop_idx[lo:hi]], loop_idx[lo:hi])
    m, k = int(T * 0.6), int(T * 0.3)
    rew_idx = list(range(m)) + list(range(m - 1, k - 1, -1)) + list(range(k + 1, T))
    cond["rewind"] = ([frames[i] for i in rew_idx], rew_idx)
    rng = np.random.RandomState(0)
    shuf_idx = list(rng.permutation(T))
    cond["shuffle"] = ([frames[i] for i in shuf_idx], list(shuf_idx))
    tr_idx = list(range(int(T * 0.9)))
    cond["truncate"] = ([frames[i] for i in tr_idx], tr_idx)
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--video_path", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--wrong_instruction", default="Put the bread into the basket")
    ap.add_argument("--robot_description", default=None)
    ap.add_argument("--camera_description", default=None)
    ap.add_argument("--output_path", default="./probe_out")
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--num_steps", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_image_side", type=int, default=640)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--conditions", default=None,
                    help="comma-separated subset of conditions to run")
    ap.add_argument("--skip_analysis", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    hf_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    hf_config._attn_implementation = "pred_slot_isolated_eager"
    model = AutoModel.from_pretrained(
        args.model_path, config=hf_config, trust_remote_code=True, torch_dtype=dtype
    ).to(device=device, dtype=dtype)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    print(f"Loading video: {args.video_path}", flush=True)
    raw = resize_frames(load_video_frames(args.video_path), args.max_image_side)
    T = len(raw)
    fps = 30.0
    print(f"{T} frames @ {raw[0].size}", flush=True)

    conditions = build_conditions(raw)
    if args.conditions:
        keep = set(args.conditions.split(","))
        conditions = {k: v for k, v in conditions.items() if k in keep}

    def run_prefixes(frame_list, instruction, eval_indices):
        """Official prefix protocol: prefix frames[0:end+1] resampled to
        num_frames, read last-slot absolute value."""
        def build(end_idx):
            fidx = np.linspace(0, end_idx, args.num_frames, dtype=int)
            return processor.process_episode(
                instruction=instruction, images=[frame_list[j] for j in fidx],
                robot_description=args.robot_description,
                camera_description=args.camera_description,
            )

        def run_batch(samples):
            kw = dict(
                input_ids=torch.cat([s["input_ids"] for s in samples], 0).to(device).long(),
                attention_mask=torch.cat([s["attention_mask"] for s in samples], 0).to(device).long(),
                pixel_values=torch.cat([s["pixel_values"].flatten(0, 1) for s in samples], 0).to(device),
                image_grid_thw=torch.cat([s["image_grid_thw"].flatten(0, 1) for s in samples], 0).to(device).long(),
            )
            with torch.inference_mode():
                out = model(**kw)
            pred = out.value.pred_value
            if pred.dim() == 2 and pred.shape[0] == 1:
                pred = pred.reshape(len(samples), -1)
            if pred.dim() == 3:
                pred = pred.mean(dim=0)
            if pred.dim() == 2 and pred.shape[-1] > 1:
                pred = pred[:, -1]
            elif pred.dim() == 2:
                pred = pred[:, 0]
            return pred.float().reshape(-1).tolist()

        values, batch = [], []
        for step, end_idx in enumerate(eval_indices):
            batch.append(build(end_idx))
            if len(batch) >= args.batch_size or step == len(eval_indices) - 1:
                values.extend(run_batch(batch))
                batch = []
        return values

    def run_analysis(frame_list, instruction):
        fidx = np.linspace(0, len(frame_list) - 1, args.num_frames, dtype=int)
        s = processor.process_episode(
            instruction=instruction, images=[frame_list[j] for j in fidx],
            robot_description=args.robot_description,
            camera_description=args.camera_description,
        )
        with torch.inference_mode():
            gen = model.generate(
                input_ids=s["input_ids"].to(device).long(),
                attention_mask=s["attention_mask"].to(device).long(),
                pixel_values=s["pixel_values"].flatten(0, 1).to(device),
                image_grid_thw=s["image_grid_thw"].flatten(0, 1).to(device).long(),
                max_new_tokens=args.max_new_tokens,
                do_sample=False, num_beams=1,
                eos_token_id=eos_token_id, pad_token_id=eos_token_id, use_cache=True,
            )
        return tokenizer.decode(gen[0, s["input_ids"].shape[1]:], skip_special_tokens=True)

    results = {"video": args.video_path, "instruction": args.instruction,
               "total_frames": T, "fps": fps, "conditions": {}, "analysis": {}}

    for name, (flist, true_idx) in conditions.items():
        n = len(flist)
        eval_idx = sample_frame_indices(n, args.num_steps)
        print(f"[{name}] {n} frames, {len(eval_idx)} prefixes", flush=True)
        vals = run_prefixes(flist, args.instruction, eval_idx)
        # true timestamp (seconds) of the last frame of each prefix, in source video
        true_last_ts = [true_idx[e] / fps for e in eval_idx]
        results["conditions"][name] = {
            "eval_positions": [int(e) for e in eval_idx],
            "values": [float(v) for v in vals],
            "true_last_frame_ts": [float(t) for t in true_last_ts],
            "true_last_frame_idx": [int(true_idx[e]) for e in eval_idx],
        }
        print(f"  v[0]={vals[0]:.2f} v[mid]={vals[len(vals)//2]:.2f} v[-1]={vals[-1]:.2f}", flush=True)

    if not args.skip_analysis:
        print("[analysis] forward + right instruction", flush=True)
        results["analysis"]["right"] = run_analysis(raw, args.instruction)
        print(results["analysis"]["right"], flush=True)
        print("[analysis] forward + wrong instruction", flush=True)
        results["analysis"]["wrong"] = run_analysis(raw, args.wrong_instruction)
        print(results["analysis"]["wrong"], flush=True)

    out_file = os.path.join(args.output_path, "probe_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved {out_file}", flush=True)


if __name__ == "__main__":
    main()
