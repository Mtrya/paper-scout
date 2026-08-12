"""Compute U-OPSD probe per-class metrics locally (stdlib only)."""
import json
import re

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


def main():
    out = "uopsd_results"
    golds = json.load(open(f"{out}/golds.json"))
    train_rollouts = json.load(open(f"{out}/train_rollouts.json"))
    merged_all = json.load(open(f"{out}/all_merged150_rollouts.json"))
    merged_train, merged_held = merged_all[:200], merged_all[200:]

    base_train = metrics_from_rollouts(train_rollouts, golds["train"])
    fin_train = metrics_from_rollouts(merged_train, golds["train"])
    res = {"base": {"train": base_train}, "merged150": {"train": fin_train}}

    held_path = f"{out}/held_base_rollouts.json"
    if not __import__("os").path.exists(held_path):
        print("[warn] held_base_rollouts.json missing — held metrics deferred")
    else:
        held_base = json.load(open(held_path))
        base_held = metrics_from_rollouts(held_base, golds["held"])
        fin_held = metrics_from_rollouts(merged_held, golds["held"])
        res["base"]["held"] = base_held
        res["merged150"]["held"] = fin_held

    json.dump(res, open(f"{out}/final_metrics.json", "w"), indent=2)

    print("=== overall (maj@8 / pass@1) ===")
    for name, d in [("base-train", base_train), ("merged-train", fin_train)]:
        print(f"{name:13s} {d['overall_maj8']:.4f} / {d['overall_pass1']:.4f}")
    if "held" in res["base"]:
        print(f"{'base-held':13s} {res['base']['held']['overall_maj8']:.4f} / {res['base']['held']['overall_pass1']:.4f}")
        print(f"{'merged-held':13s} {res['merged150']['held']['overall_maj8']:.4f} / {res['merged150']['held']['overall_pass1']:.4f}")
    print("\n=== per-class (train prompts): base -> merged150 ===")
    for c in ["maj_correct", "split_wrong", "unanimous_wrong", "low_signal"]:
        b, f = base_train.get(c, {}), fin_train.get(c, {})
        print(f"{c:16s} n={b.get('n', 0):3d} maj8 {b.get('maj8_acc', 0):.3f} -> {f.get('maj8_acc', 0):.3f}"
              f" | wrong_agree {b.get('wrong_agree', 0):.3f} -> {f.get('wrong_agree', 0):.3f}"
              f" | unique {b.get('mean_unique', 0):.2f} -> {f.get('mean_unique', 0):.2f}"
              f" | pass1 {b.get('pass1_acc', 0):.3f} -> {f.get('pass1_acc', 0):.3f}")
    if "held" in res["base"]:
        print("\n=== per-class (held): base -> merged150 ===")
        for c in ["maj_correct", "split_wrong", "unanimous_wrong", "low_signal"]:
            b, f = res["base"]["held"].get(c, {}), res["merged150"]["held"].get(c, {})
            print(f"{c:16s} n={b.get('n', 0):3d} maj8 {b.get('maj8_acc', 0):.3f} -> {f.get('maj8_acc', 0):.3f}"
                  f" | wrong_agree {b.get('wrong_agree', 0):.3f} -> {f.get('wrong_agree', 0):.3f}"
                  f" | unique {b.get('mean_unique', 0):.2f} -> {f.get('mean_unique', 0):.2f}")


if __name__ == "__main__":
    main()
