"""Remote inference runner (executes on the Inspire GPU instance).

Reads data/tasks.jsonl, runs a VLM via vLLM offline batch inference, appends
raw responses to results/raw/<model_name>.jsonl. Idempotent: task_ids already
present in the output file are skipped, so restarts/resumes are safe.

Usage:
  python remote_runner.py --model-path <local weights dir> --model-name qwen3vl8b \
      --data data --out results/raw/qwen3vl8b.jsonl [--probes p1,p2,p3] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys


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
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=32)
    args = ap.parse_args()

    tasks = []
    with open(os.path.join(args.data, "tasks.jsonl")) as f:
        for line in f:
            t = json.loads(line)
            if t["probe"] in args.probes.split(","):
                tasks.append(t)
    done = load_done(args.out)
    tasks = [t for t in tasks if t["task_id"] not in done]
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[runner] {len(tasks)} tasks to run (skipped {len(done)} done)", flush=True)
    if not tasks:
        return

    from PIL import Image
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model_path, max_model_len=16384,
              limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=0.90,
              enforce_eager=False, seed=1234)
    sp = llm.get_default_sampling_params()
    sp.max_tokens = args.max_tokens
    if sp.temperature is None:
        sp.temperature = 0.7
    print(f"[runner] sampling: T={sp.temperature} top_p={sp.top_p} top_k={sp.top_k}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    img_cache: dict[str, Image.Image] = {}

    def get_img(rel: str):
        if rel not in img_cache:
            img_cache[rel] = Image.open(os.path.join(args.data, rel)).convert("RGB")
        return img_cache[rel]

    n_done = 0
    for i in range(0, len(tasks), args.chunk):
        chunk = tasks[i:i + args.chunk]
        convs = []
        for t in chunk:
            if t["image"]:
                content = [
                    {"type": "image_pil", "image_pil": get_img(t["image"])},
                    {"type": "text", "text": t["prompt"]},
                ]
            else:
                content = [{"type": "text", "text": t["prompt"]}]
            convs.append([
                {"role": "system", "content": t["system"]},
                {"role": "user", "content": content},
            ])
        outs = llm.chat(convs, sp)
        with open(args.out, "a") as f:
            for t, o in zip(chunk, outs):
                rec = {
                    "task_id": t["task_id"],
                    "model": args.model_name,
                    "response": o.outputs[0].text,
                    "finish_reason": o.outputs[0].finish_reason,
                    "n_prompt_tokens": len(o.prompt_token_ids),
                    "n_output_tokens": len(o.outputs[0].token_ids),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_done += len(chunk)
        print(f"[runner] {n_done}/{len(tasks)} done", flush=True)


if __name__ == "__main__":
    main()
