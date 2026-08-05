#!/usr/bin/env python3
"""Probe and causally localize answer pre-commitment in a real LLM.

The paper "Post-Hoc Reasoning in Chain of Thought" steers the residual
stream at every decoding position, including the final answer token.  This
script separates that intervention into four temporal scopes:

* prefill: only the final prompt token, before any written reasoning;
* prefill_all: every prompt token at the selected layer, then stop;
* early_reasoning_4: only the first four generated-token states;
* reasoning: generated tokens before the literal ``FINAL`` marker;
* answer: generated tokens after that marker;
* all_decode: every generated token, matching the paper's broad scope.

It uses the public BIG-Bench Sports Understanding task and predicts the
model's own semantic final answer, not the gold label.  A norm-matched random
orthogonal direction is included as an off-manifold control.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    LogitsProcessor,
    LogitsProcessorList,
)


TASK_URL = (
    "https://raw.githubusercontent.com/google/BIG-bench/main/"
    "bigbench/benchmark_tasks/sports_understanding/task.json"
)
FINAL_RE = re.compile(r"FINAL\s*:\s*\(?([AB])\)?", re.IGNORECASE)


@dataclass
class DecodeState:
    prompt_length: int
    tokenizer: Any
    generated_ids: list[int] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.tokenizer.decode(self.generated_ids, skip_special_tokens=True)


class TrackGeneratedTokens(LogitsProcessor):
    def __init__(self, state: DecodeState) -> None:
        self.state = state

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.state.generated_ids = input_ids[0, self.state.prompt_length :].tolist()
        return scores


def find_decoder_layers(model: torch.nn.Module) -> tuple[str, torch.nn.ModuleList]:
    expected = int(model.config.text_config.num_hidden_layers)
    matches: list[tuple[str, torch.nn.ModuleList]] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) == expected and name.endswith("layers"):
            matches.append((name, module))
    if not matches:
        raise RuntimeError(f"could not find a {expected}-layer decoder ModuleList")
    matches.sort(key=lambda pair: ("language_model" not in pair[0], len(pair[0])))
    return matches[0]


def load_task(path: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    if not path.exists():
        response = requests.get(TASK_URL, timeout=60)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    payload = json.loads(path.read_text())
    examples = payload["examples"]
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:limit]


def semantic_gold(example: dict[str, Any]) -> str:
    scores = example["target_scores"]
    return "yes" if scores["plausible"] > scores["implausible"] else "no"


def build_prompt(tokenizer: Any, example: dict[str, Any], index: int, seed: int) -> tuple[str, dict[str, str]]:
    rng = random.Random(seed + 7919 * index)
    yes_first = rng.random() < 0.5
    mapping = {"A": "yes", "B": "no"} if yes_first else {"A": "no", "B": "yes"}
    labels = {
        letter: "Yes, the sentence is plausible" if meaning == "yes" else "No, the sentence is implausible"
        for letter, meaning in mapping.items()
    }
    user = (
        f'Is the following sports sentence plausible? "{example["input"]}"\n\n'
        f"(A) {labels['A']}\n"
        f"(B) {labels['B']}\n\n"
        "Give exactly one short sentence of reasoning (at most 30 words), then a new line "
        "with exactly FINAL: (A) or FINAL: (B). Do not add anything after that line."
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return rendered + "Let's think step by step:", mapping


def parsed_semantic(text: str, mapping: dict[str, str]) -> str | None:
    matches = FINAL_RE.findall(text)
    return mapping[matches[-1].upper()] if matches else None


def replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden


def should_steer(mode: str, hidden: torch.Tensor, state: DecodeState) -> tuple[bool, bool]:
    is_prefill = hidden.shape[1] > 1
    if mode == "prefill":
        return is_prefill, True
    if mode == "prefill_all":
        return is_prefill, False
    if is_prefill:
        return False, False
    marker_seen = "FINAL" in state.text.upper()
    if mode == "early_reasoning_4":
        return not marker_seen and len(state.generated_ids) < 4, False
    if mode == "reasoning":
        return not marker_seen, False
    if mode == "answer":
        return marker_seen, False
    if mode in {"all_decode", "orthogonal"}:
        return True, False
    if mode == "none":
        return False, False
    raise ValueError(mode)


def generate(
    model: Any,
    tokenizer: Any,
    layers: torch.nn.ModuleList,
    prompt: str,
    max_new_tokens: int,
    capture: bool = False,
    layer_index: int | None = None,
    direction: torch.Tensor | None = None,
    alpha: float = 0.0,
    mode: str = "none",
) -> tuple[str, np.ndarray | None]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    state = DecodeState(prompt_length=inputs.input_ids.shape[1], tokenizer=tokenizer)
    tracker = TrackGeneratedTokens(state)
    captured: list[torch.Tensor | None] = [None] * len(layers)
    handles: list[Any] = []

    if capture:
        for idx, layer in enumerate(layers):
            def capture_hook(_module: Any, _args: Any, output: Any, layer_no: int = idx) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden.shape[1] > 1 and captured[layer_no] is None:
                    captured[layer_no] = hidden[0, -1].detach().float().cpu()
            handles.append(layer.register_forward_hook(capture_hook))

    if layer_index is not None and direction is not None and alpha != 0:
        delta = (alpha * direction).to(model.device)

        def steer_hook(_module: Any, _args: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            active, last_only = should_steer(mode, hidden, state)
            if not active:
                return output
            edited = hidden.clone()
            if last_only:
                edited[:, -1, :] = edited[:, -1, :] + delta.to(edited.dtype)
            else:
                edited = edited + delta.to(edited.dtype)
            return replace_hidden(output, edited)

        handles.append(layers[layer_index].register_forward_hook(steer_hook))

    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                logits_processor=LogitsProcessorList([tracker]),
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
    finally:
        for handle in handles:
            handle.remove()

    new_ids = generated[0, inputs.input_ids.shape[1] :]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    if capture:
        if any(item is None for item in captured):
            missing = [idx for idx, item in enumerate(captured) if item is None]
            raise RuntimeError(f"failed to capture prefill activations at layers {missing}")
        matrix = torch.stack([item for item in captured if item is not None]).numpy()
    else:
        matrix = None
    return text, matrix


def difference_of_means(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0)


def cosine_scores(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (x @ w) / (np.linalg.norm(x, axis=1) * np.linalg.norm(w) + 1e-12)


def orthogonal_direction(direction: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_vector = torch.randn(direction.shape, generator=generator, dtype=torch.float32)
    unit = direction.float() / direction.float().norm()
    random_vector = random_vector - torch.dot(random_vector, unit) * unit
    return random_vector / random_vector.norm() * direction.float().norm()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--task-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--train-size", type=int, default=160)
    parser.add_argument("--intervention-size", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--alphas", type=float, nargs="+", default=(2, 4, 8, 12))
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()

    if args.train_size >= args.limit:
        raise ValueError("train-size must be smaller than limit")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples = load_task(args.task_json, args.limit, args.seed)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    ).eval()
    layer_path, layers = find_decoder_layers(model)
    print(json.dumps({"decoder_layers": layer_path, "count": len(layers)}), flush=True)

    baseline_path = args.output_dir / "baseline.jsonl"
    existing_baselines: dict[int, dict[str, Any]] = {}
    if baseline_path.exists():
        for line in baseline_path.read_text().splitlines():
            if line.strip():
                payload = json.loads(line)
                existing_baselines[int(payload["index"])] = payload
    records: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        if index in existing_baselines:
            records.append(existing_baselines[index])
            continue
        prompt, mapping = build_prompt(tokenizer, example, index, args.seed)
        response, activations = generate(
            model, tokenizer, layers, prompt, args.max_new_tokens, capture=True
        )
        answer = parsed_semantic(response, mapping)
        record = {
            "index": index,
            "input": example["input"],
            "gold": semantic_gold(example),
            "mapping": mapping,
            "response": response,
            "answer": answer,
            "activations": activations.tolist() if activations is not None else None,
        }
        records.append(record)
        with baseline_path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({"baseline": index, "answer": answer, "gold": record["gold"]}), flush=True)

    parsed = [record for record in records if record["answer"] in {"yes", "no"}]
    train = [record for record in parsed if record["index"] < args.train_size]
    test = [record for record in parsed if record["index"] >= args.train_size]
    x_train = np.asarray([record["activations"] for record in train], dtype=np.float32)
    x_test = np.asarray([record["activations"] for record in test], dtype=np.float32)
    y_train = np.asarray([record["answer"] == "yes" for record in train], dtype=np.int64)
    y_test = np.asarray([record["answer"] == "yes" for record in test], dtype=np.int64)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise RuntimeError("probe split does not contain both model-answer classes")

    aucs: list[float] = []
    directions: list[np.ndarray] = []
    for layer_no in range(x_train.shape[1]):
        direction = difference_of_means(x_train[:, layer_no], y_train)
        directions.append(direction)
        aucs.append(float(roc_auc_score(y_test, cosine_scores(x_test[:, layer_no], direction))))
    best_layer = int(np.argmax(aucs))
    best_direction = torch.from_numpy(directions[best_layer])

    rng = np.random.default_rng(args.seed)
    permutation_max_aucs: list[float] = []
    for _ in range(200):
        permuted = rng.permutation(y_train)
        layer_aucs = []
        for layer_no in range(x_train.shape[1]):
            direction = difference_of_means(x_train[:, layer_no], permuted)
            layer_aucs.append(roc_auc_score(y_test, cosine_scores(x_test[:, layer_no], direction)))
        permutation_max_aucs.append(float(max(layer_aucs)))

    balanced: list[dict[str, Any]] = []
    for semantic in ("yes", "no"):
        candidates = [record for record in test if record["answer"] == semantic and record["gold"] == semantic]
        balanced.extend(candidates[: args.intervention_size // 2])
    if len(balanced) < args.intervention_size:
        extras = [record for record in test if record not in balanced]
        balanced.extend(extras[: args.intervention_size - len(balanced)])

    for record in records:
        record.pop("activations", None)
    del x_train, x_test, directions
    gc.collect()

    intervention_path = args.output_dir / "interventions.jsonl"
    completed_interventions: set[tuple[int, str, float]] = set()
    if intervention_path.exists():
        for line in intervention_path.read_text().splitlines():
            if line.strip():
                payload = json.loads(line)
                completed_interventions.add(
                    (int(payload["index"]), str(payload["mode"]), float(payload["alpha"]))
                )
    modes = (
        "prefill",
        "prefill_all",
        "early_reasoning_4",
        "reasoning",
        "answer",
        "all_decode",
        "orthogonal",
    )
    for record in balanced:
        example = examples[record["index"]]
        prompt, mapping = build_prompt(tokenizer, example, record["index"], args.seed)
        sign = -1.0 if record["answer"] == "yes" else 1.0
        orthogonal = orthogonal_direction(best_direction, args.seed + record["index"])
        for alpha_abs in args.alphas:
            for mode in modes:
                key = (record["index"], mode, sign * alpha_abs)
                if key in completed_interventions:
                    continue
                direction = orthogonal if mode == "orthogonal" else best_direction
                response, _ = generate(
                    model,
                    tokenizer,
                    layers,
                    prompt,
                    args.max_new_tokens,
                    layer_index=best_layer,
                    direction=direction,
                    alpha=sign * alpha_abs,
                    mode=mode,
                )
                payload = {
                    "index": record["index"],
                    "input": record["input"],
                    "gold": record["gold"],
                    "baseline_answer": record["answer"],
                    "baseline_response": record["response"],
                    "target_answer": "no" if record["answer"] == "yes" else "yes",
                    "alpha": sign * alpha_abs,
                    "mode": mode,
                    "answer": parsed_semantic(response, mapping),
                    "response": response,
                }
                with intervention_path.open("a") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                print(json.dumps({key: payload[key] for key in ("index", "mode", "alpha", "answer")}), flush=True)

    summary = {
        "task_url": TASK_URL,
        "model": str(args.model),
        "n_total": len(records),
        "n_parsed": len(parsed),
        "n_train": len(train),
        "n_test": len(test),
        "model_accuracy": sum(record["answer"] == record["gold"] for record in parsed) / len(parsed),
        "model_yes_rate": sum(record["answer"] == "yes" for record in parsed) / len(parsed),
        "layer_aucs": aucs,
        "best_layer": best_layer,
        "best_auc": aucs[best_layer],
        "probe_direction_norm": float(best_direction.norm()),
        "permutation_max_auc_mean": float(np.mean(permutation_max_aucs)),
        "permutation_max_auc_p95": float(np.quantile(permutation_max_aucs, 0.95)),
        "permutation_max_auc_exceedances": sum(value >= aucs[best_layer] for value in permutation_max_aucs),
        "permutation_max_auc_empirical_p": (
            1 + sum(value >= aucs[best_layer] for value in permutation_max_aucs)
        ) / (1 + len(permutation_max_aucs)),
        "n_intervention_examples": len(balanced),
        "alphas": list(args.alphas),
        "modes": list(modes),
    }
    (args.output_dir / "probe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
