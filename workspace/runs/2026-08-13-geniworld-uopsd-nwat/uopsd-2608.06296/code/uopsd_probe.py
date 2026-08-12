"""U-OPSD trust probe (vLLM sampling + HF LoRA training).

Phases:
  prep    : load MATH-500, write prompts.txt + golds.json (train 200 / held 100)
  train   : build U-OPSD distillation set from rollouts.json, LoRA train 150 steps
  metrics : per-class metrics from rollouts (no sampling); used for base (train set
            reuses the frozen-teacher rollouts) and checkpoints (vLLM-fresh samples)

Sampling is done by vllm_rollout.py (train rollouts + held/fresh evals).
"""
import argparse
import json
import os
import re
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def extract_answer(text):
    m = BOXED_RE.findall(text)
    return normalize_answer(m[-1].strip()) if m else None


def normalize_answer(s):
    s = s.strip().lower()
    s = s.replace("\\ ", "").replace(" ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("{", "").replace("}", "")
    s = s.replace(",", "")
    s = re.sub(r"\\frac(\d+)(\d+)", r"\1/\2", s)
    s = s.replace("\\%", "%").replace("\\pi", "pi")
    return s


def majority_vote(answers):
    counts = {}
    for a in answers:
        if a is not None:
            counts[a] = counts.get(a, 0) + 1
    if not counts:
        return None, counts
    return max(counts, key=lambda a: counts[a]), counts


def classify(majority, counts, gold):
    parsable = sum(counts.values())
    if parsable < 2 or majority is None:
        return "low_signal"
    if majority == gold:
        return "maj_correct"
    if counts[majority] == parsable:
        return "unanimous_wrong"
    return "split_wrong"


def build_training_set(prompts, rollouts, golds, tau=0.5):
    items, meta = [], []
    for x, rs, g in zip(prompts, rollouts, golds):
        answers = [r["answer"] for r in rs]
        maj, counts = majority_vote(answers)
        cls = classify(maj, counts, g)
        c = counts.get(maj, 0) / max(1, len(answers))
        meta.append({"prompt": x, "gold": g, "majority": maj, "class": cls,
                     "consensus": c, "counts": {k: v for k, v in counts.items()}})
        if cls not in ("maj_correct", "split_wrong") or c < tau:
            continue
        agreeing = [r for r in rs if r["answer"] == maj]
        disagreeing = [r for r in rs if r["answer"] not in (maj, None)]
        if not agreeing or not disagreeing:
            continue
        y_plus = max(agreeing, key=lambda r: len(r["text"]))["text"]
        for y_minus in disagreeing:
            items.append({"prompt": x, "y_plus": y_plus, "y_minus": y_minus["text"]})
    return items, meta


def train_step(model, tokenizer, items, optimizer, batch=4, yplus_max=1024, yminus_max=1536):
    idx = torch.randperm(len(items))[:batch].tolist()
    total_loss, n = 0.0, 0
    model.train()
    for i in idx:
        it = items[i]
        yp_tok = tokenizer(it["y_plus"], truncation=True, max_length=yplus_max).input_ids[0]
        ym_tok = tokenizer(it["y_minus"], truncation=True, max_length=yminus_max).input_ids[0]
        yp_text = tokenizer.decode(yp_tok, skip_special_tokens=True)
        ym_text = tokenizer.decode(ym_tok, skip_special_tokens=True)
        s_enc = tokenizer(it["prompt"] + ym_text, return_tensors="pt", truncation=True,
                          max_length=4096).to(model.device)
        t_enc = tokenizer(it["prompt"] + yp_text + ym_text, return_tensors="pt", truncation=True,
                          max_length=4096).to(model.device)
        s_out = model(**s_enc)
        with torch.no_grad(), model.disable_adapter():
            t_out = model(**t_enc)
        ym_tok_n = len(tokenizer(ym_text).input_ids)
        s_logits = s_out.logits[0, -ym_tok_n:, :].float()
        t_logits = t_out.logits[0, -ym_tok_n:, :].float()
        t_dist = torch.softmax(t_logits, dim=-1)
        logp_s = torch.log_softmax(s_logits, dim=-1)
        loss = -(t_dist * logp_s).sum(dim=-1).mean()
        (loss / batch).backward()
        total_loss += float(loss.detach()); n += 1
        del s_enc, t_enc, s_out, t_out, s_logits, t_logits, t_dist, logp_s
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    return total_loss / max(n, 1)


def metrics_from_rollouts(rollouts, golds):
    per_class = {c: {"n": 0, "maj8_correct": 0, "pass1_correct": 0,
                     "wrong_agree": 0.0, "unique": []}
                 for c in ["maj_correct", "split_wrong", "unanimous_wrong", "low_signal"]}
    overall_maj = overall_pass1 = 0
    for r, g in zip(rollouts, golds):
        answers = [e["answer"] for e in r]
        maj, counts = majority_vote(answers)
        cls = classify(maj, counts, g)
        d = per_class[cls]
        d["n"] += 1
        d["maj8_correct"] += int(maj == g)
        d["pass1_correct"] += int(answers[0] == g)
        d["unique"].append(len({a for a in answers if a is not None}))
        if cls in ("split_wrong", "unanimous_wrong") and maj is not None:
            valid = [a for a in answers if a is not None]
            d["wrong_agree"] += counts[maj] / len(valid)
        overall_maj += int(maj == g)
        overall_pass1 += int(answers[0] == g)
    res = {"overall_maj8": overall_maj / len(rollouts),
           "overall_pass1": overall_pass1 / len(rollouts)}
    for c, d in per_class.items():
        if d["n"]:
            res[c] = {"n": d["n"], "maj8_acc": d["maj8_correct"] / d["n"],
                      "pass1_acc": d["pass1_correct"] / d["n"],
                      "wrong_agree": d["wrong_agree"] / d["n"] if c != "low_signal" else 0.0,
                      "mean_unique": sum(d["unique"]) / d["n"]}
    return res


def prep(out, train_n=200, held_n=100):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = [d["problem"] for d in ds]
    golds = [normalize_answer(str(d["answer"])) for d in ds]
    json.dump(problems[:train_n], open(os.path.join(out, "prompts_train.json"), "w"), indent=1)
    json.dump(problems[train_n:train_n + held_n], open(os.path.join(out, "prompts_held.json"), "w"), indent=1)
    json.dump({"train": golds[:train_n], "held": golds[train_n:train_n + held_n]},
              open(os.path.join(out, "golds.json"), "w"), indent=2)
    print(f"[prep] {train_n} train + {held_n} held prompts written")


def train(out, model_path, steps=150, batch=4, tau=0.5):
    prompts = json.load(open(os.path.join(out, "prompts_train.json")))
    golds = json.load(open(os.path.join(out, "golds.json")))["train"]
    rollouts = json.load(open(os.path.join(out, "train_rollouts.json")))
    assert len(prompts) == len(rollouts) == len(golds), (len(prompts), len(rollouts), len(golds))
    items, meta = build_training_set(prompts, rollouts, golds, tau=tau)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2, ensure_ascii=False)
    print(f"[train] distillation items: {len(items)}")
    print("[train] class distribution:",
          {c: sum(1 for m in meta if m["class"] == c) for c in
           ["maj_correct", "split_wrong", "unanimous_wrong", "low_signal"]})
    base_metrics = metrics_from_rollouts(rollouts, golds)
    json.dump({"train": base_metrics},
              open(os.path.join(out, "eval_base.json"), "w"), indent=2)
    print("[train] base maj8=%.3f pass1=%.3f" %
          (base_metrics["overall_maj8"], base_metrics["overall_pass1"]))

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                device_map="auto", trust_remote_code=True)
    base.gradient_checkpointing_enable()
    base.eval()
    lora = LoraConfig(r=64, lora_alpha=128, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, task_type="CAUSAL_LM")
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-6)
    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        loss = train_step(model, tok, items, opt, batch=batch)
        losses.append(loss)
        if step % 25 == 0:
            ckpt = os.path.join(out, f"ckpt_{step}")
            model.save_pretrained(ckpt)
            print(f"[train] step {step} loss={loss:.4f} ({time.time()-t0:.0f}s)")
            t0 = time.time()
            if step in (75, 150):
                # also save merged for vLLM eval
                merged = PeftModel.from_pretrained(base, ckpt).merge_and_unload()
                merged.save_pretrained(os.path.join(out, f"merged_{step}"))
                del merged
                torch.cuda.empty_cache()
    json.dump(losses, open(os.path.join(out, "train_loss.json"), "w"))


def metrics(out, rollouts_file, golds_key):
    rollouts = json.load(open(os.path.join(out, rollouts_file)))
    golds = json.load(open(os.path.join(out, "golds.json")))[golds_key]
    return metrics_from_rollouts(rollouts, golds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["prep", "train", "metrics"])
    ap.add_argument("--out", default="uopsd_results")
    ap.add_argument("--model", default="/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/cache/models/Qwen3-4B")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--rollouts", default="train_rollouts.json")
    ap.add_argument("--golds-key", default="train")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.phase == "prep":
        prep(args.out)
    elif args.phase == "train":
        train(args.out, args.model, steps=args.steps, batch=args.batch, tau=args.tau)
    elif args.phase == "metrics":
        res = metrics(args.out, args.rollouts, args.golds_key)
        out_name = args.rollouts.replace(".json", "_metrics.json")
        json.dump(res, open(os.path.join(args.out, out_name), "w"), indent=2)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
