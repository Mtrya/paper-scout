"""Verify AnyS1-4B calibration claims on the released Any-DP test split (held-out skills).

Replicates the released serve_s1.py loading/scoring path, but points the base model at a
local download. Reports: top-1 accuracy, coverage/agreement at tau_cold=0.918, and the
agreement-vs-tau operating curve.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from anys1.model.head import HEADS  # noqa: E402

SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
import re  # noqa: E402
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def _user_tokens(tok, text):
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--max-state", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta = torch.load(Path(args.ckpt) / "head.pt", map_location="cpu", weights_only=False)
    print("head.pt meta:", {k: v for k, v in meta.items() if k != "head"}, flush=True)
    tok = AutoTokenizer.from_pretrained(args.ckpt)
    base_lm = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).model
    lm = PeftModel.from_pretrained(base_lm, args.ckpt).to("cuda").eval()
    head = HEADS[meta.get("head_type", "oc")](lm.config.hidden_size, dp=int(meta.get("head_dim", 256)))
    head.load_state_dict(meta["head"])
    head.temperature = float(meta.get("temperature", 1.0))
    head.to(device="cuda", dtype=torch.bfloat16).eval()

    records = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    random.Random(7).shuffle(records)
    used, skipped_label = [], 0
    for r in records:
        opts = [str(o) for o in r["options"]]
        lab = str(r["label"]).strip()
        if lab not in opts and lab.lower() not in [o.lower() for o in opts]:
            skipped_label += 1
            continue
        used.append(r)
        if len(used) >= args.n:
            break
    print(f"evaluating {len(used)} records (skipped {skipped_label} label-not-in-options)", flush=True)

    q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL[1:])
    pfx = tok.convert_tokens_to_ids(SPECIAL[0])
    rows = []
    for i, r in enumerate(used):
        opts = [str(o) for o in r["options"]]
        lab = str(r["label"]).strip()
        lab_idx = opts.index(lab) if lab in opts else [o.lower() for o in opts].index(lab.lower())
        s_toks = _user_tokens(tok, r["state"])[: args.max_state - 1]
        s_ids = [pfx] + s_toks
        instr_ids = [q_id] + _user_tokens(tok, r["question"])
        spans = [[o_id] + _user_tokens(tok, o) + [c_id] for o in opts]
        br = instr_ids + [t for sp in spans for t in sp] + [d_id]
        ends, cursor = [], len(s_ids) + len(instr_ids)
        for sp in spans:
            cursor += len(sp)
            ends.append(cursor - 1)
        decide_idx = len(s_ids) + len(br) - 1
        input_ids = torch.tensor([s_ids + br], dtype=torch.long, device="cuda")
        hidden = lm(input_ids=input_ids).last_hidden_state[0]
        logits = head(hidden[decide_idx], hidden[ends]).float()
        probs = F.softmax(logits, dim=-1).cpu().tolist()
        best = max(range(len(probs)), key=probs.__getitem__)
        rows.append({"id": r["id"], "judgment": r["judgment"], "n_options": len(opts),
                     "p_max": probs[best], "correct": best == lab_idx, "probs": probs})
        if (i + 1) % 50 == 0:
            acc = sum(x["correct"] for x in rows) / len(rows)
            print(f"  {i+1}/{len(rows)} running acc {acc:.4f}", flush=True)

    n = len(rows)
    acc = sum(x["correct"] for x in rows) / n
    curve = []
    for tau in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.918, 0.95, 0.97, 0.99]:
        acc_sel = [x for x in rows if x["p_max"] >= tau]
        cov = len(acc_sel) / n
        agr = sum(x["correct"] for x in acc_sel) / len(acc_sel) if acc_sel else None
        curve.append({"tau": tau, "coverage": round(cov, 4), "agreement": round(agr, 4) if agr is not None else None})
    tau918 = [x for x in rows if x["p_max"] >= 0.918]
    result = {
        "n_eval": n, "skipped_label_not_in_options": skipped_label,
        "top1_accuracy": round(acc, 4),
        "tau_cold_0918": {"coverage": round(len(tau918) / n, 4),
                           "agreement": round(sum(x["correct"] for x in tau918) / len(tau918), 4) if tau918 else None,
                           "n_accepted": len(tau918)},
        "curve": curve,
        "by_judgment": {j: {"n": sum(1 for x in rows if x['judgment'] == j),
                             "acc": round(sum(x['correct'] for x in rows if x['judgment'] == j) / max(1, sum(1 for x in rows if x['judgment'] == j)), 4)}
                         for j in {x["judgment"] for x in rows}},
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
