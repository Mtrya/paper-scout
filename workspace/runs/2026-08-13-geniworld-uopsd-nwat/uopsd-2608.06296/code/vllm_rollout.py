"""U-OPSD probe — vLLM rollout worker (G=8 per prompt) + saved JSON."""
import argparse
import json
import os
import re
import time

from vllm import LLM, SamplingParams

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def extract_answer(text):
    m = BOXED_RE.findall(text)
    if not m:
        return None
    return normalize_answer(m[-1].strip())


def normalize_answer(s):
    s = s.strip().lower()
    s = s.replace("\\ ", "").replace(" ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("{", "").replace("}", "")
    s = s.replace(",", "")
    s = re.sub(r"\\frac(\d+)(\d+)", r"\1/\2", s)
    s = s.replace("\\%", "%").replace("\\pi", "pi")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/cache/models/Qwen3-4B")
    ap.add_argument("--prompts", default="prompts.json")  # JSON list, one string per prompt
    ap.add_argument("--out", default="rollouts.json")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.45)
    args = ap.parse_args()

    prompts = json.load(open(args.prompts))
    print(f"[vllm] loading model, {len(prompts)} prompts x {args.n}")
    t0 = time.time()
    llm = LLM(model=args.model, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_mem, max_model_len=8192,
              enforce_eager=True, trust_remote_code=True)
    sp = SamplingParams(n=args.n, temperature=1.0, top_p=0.95, top_k=20,
                        max_tokens=args.max_new)
    outs = llm.generate(prompts, sp)
    res = []
    for o in outs:
        entries = []
        for seq in o.outputs:
            entries.append({"text": seq.text, "answer": extract_answer(seq.text)})
        res.append(entries)
    with open(args.out, "w") as f:
        json.dump(res, f)
    n_parsable = sum(1 for r in res for e in r if e["answer"] is not None)
    print(f"[vllm] done in {time.time()-t0:.0f}s, {n_parsable}/{len(res)*args.n} parsable")


if __name__ == "__main__":
    main()
