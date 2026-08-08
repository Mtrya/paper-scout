"""Offline batch inference runner using plain transformers (andromeda, no vLLM).

Same CLI/output contract as remote_runner.py: reads data/tasks.jsonl, appends
raw responses to results/raw/<model_name>.jsonl, idempotent (skips done task_ids).

Usage:
  python runner_hf.py --model-path <weights> --model-name qwen3vl8b \
      --data data --out results/raw/qwen3vl8b.jsonl [--load int8|bf16] \
      [--batch-size 4] [--max-tokens 4096] [--probes p1,p2,p3] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os

import torch


def load_done(out_path: str) -> set[str]:
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["task_id"])
                except Exception:
                    pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--probes", default="p1,p2,p3")
    ap.add_argument("--modes", default="", help="comma list: text,vision (empty = all)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--load", choices=["int8", "nf4", "bf16"], default="int8")
    ap.add_argument("--max-mem-gib", type=float, default=0,
                    help="cap GPU memory for device_map (0 = library default)")
    ap.add_argument("--skip-modules", default="",
                    help="comma list of module names to keep unquantized (e.g. visual)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    tasks = []
    with open(os.path.join(args.data, "tasks.jsonl")) as f:
        for line in f:
            t = json.loads(line)
            if t["probe"] in args.probes.split(","):
                if args.modes and t["mode"] not in args.modes.split(","):
                    continue
                tasks.append(t)
    done = load_done(args.out)
    tasks = [t for t in tasks if t["task_id"] not in done]
    # text tasks and vision tasks batched separately (heterogeneous inputs)
    text_tasks = [t for t in tasks if not t["image"]]
    vis_tasks = [t for t in tasks if t["image"]]
    tasks = text_tasks + vis_tasks
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[runner] {len(tasks)} tasks ({len(text_tasks)} text, {len(vis_tasks)} vision); "
          f"skipped {len(done)} done", flush=True)
    if not tasks:
        return

    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    kwargs = dict(device_map="auto", dtype=torch.bfloat16)
    if args.max_mem_gib:
        kwargs["max_memory"] = {0: f"{args.max_mem_gib}GiB", "cpu": "12GiB"}
    skip = args.skip_modules.split(",") if args.skip_modules else []
    if args.load == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_skip_modules=skip or None)
        kwargs.pop("dtype")
    elif args.load == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=skip or None)
        kwargs.pop("dtype")
    model = AutoModelForImageTextToText.from_pretrained(args.model_path, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    tok = processor.tokenizer
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    gen_kwargs = dict(max_new_tokens=args.max_tokens)
    gc = model.generation_config
    if getattr(gc, "do_sample", False):
        gen_kwargs.update(do_sample=True,
                          temperature=getattr(gc, "temperature", None),
                          top_p=getattr(gc, "top_p", None),
                          top_k=getattr(gc, "top_k", None))
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    print(f"[runner] load={args.load} gen={gen_kwargs}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    img_cache: dict[str, Image.Image] = {}

    def get_img(rel: str):
        if rel not in img_cache:
            img_cache[rel] = Image.open(os.path.join(args.data, rel)).convert("RGB")
        return img_cache[rel]

    def build_inputs(chunk):
        texts, images = [], []
        for t in chunk:
            if t["image"]:
                content = [{"type": "image"}, {"type": "text", "text": t["prompt"]}]
            else:
                content = [{"type": "text", "text": t["prompt"]}]
            msgs = [{"role": "system", "content": t["system"]},
                    {"role": "user", "content": content}]
            texts.append(processor.apply_chat_template(msgs, add_generation_prompt=True,
                                                       tokenize=False))
            if t["image"]:
                images.append(get_img(t["image"]))
        kw = dict(text=texts, padding=True, return_tensors="pt")
        if images:
            kw["images"] = images
        return processor(**kw).to(model.device)

    n_done = 0
    for i in range(0, len(tasks), args.batch_size):
        chunk = tasks[i:i + args.batch_size]
        inputs = build_inputs(chunk)
        n_in = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        recs = []
        for j, t in enumerate(chunk):
            gen = out[j][inputs["input_ids"].shape[1]:]
            text = tok.decode(gen, skip_special_tokens=True)
            n_out = int((gen != tok.pad_token_id).sum())
            recs.append({
                "task_id": t["task_id"], "model": args.model_name,
                "response": text.strip(),
                "finish_reason": "length" if n_out >= args.max_tokens else "stop",
                "n_prompt_tokens": int((inputs["input_ids"][j] != tok.pad_token_id).sum()),
                "n_output_tokens": n_out,
            })
        with open(args.out, "a") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_done += len(chunk)
        print(f"[runner] {n_done}/{len(tasks)} done (prompt {n_in} tok)", flush=True)
        del inputs, out
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
