#!/usr/bin/env python3
"""MF-1 text-latent round-trip probe (clean, noisy, error-budget, and controls).

Question
--------
Multimodal Flow (arXiv 2609.40362) represents text as continuous T5-small latents
in 8-token blocks and reads it back with a separately trained text decoder. How
much of the original text survives that channel? The paper never reports it.

Pipeline replicated here (the released code is authoritative; the two vendored
modules in this directory are byte-identical copies of the clone's files):

  1. T5 sentencepiece tokenization, add_special_tokens=False, stripping EOS/PAD
     -> mf/data/text.py:_content_ids
  2. append EOS, EOS-fill the terminal block to a multiple of 8
     -> mf/data/text.py:tokenize_block_aligned_document_ids, selected by
        flow.text_block_causal.packing = block_aligned_eos with the default
        whole_record_defer_pad policy (mf/data/physical.py)
  3. T5-small encoder applied to each 8-token block independently, attention
     inside the block only
     -> mf/codecs/online.py:_encode_target_rows (target_encoding = block_local)
  4. per-position normalization (h - 0.0) / 0.2
     -> mf/latents/stats.py:LatentStatsRegistry.normalize_text with
        latent_stats.text_normal {mean: 0.0, std: 0.2} (configs/pretrain.yaml)
  5. LatentTextDecoder: 6 bidirectional layers, width 512, 8 heads of dim 64,
     RMSNorm, RoPE, SwiGLU (int(4/3 * 4 * 512) = 1365), 512 -> 128 -> 512
     bottleneck, tanh-GELU head over the 32128-way T5 vocabulary, one prediction
     per input position, no attention mask at inference
     -> mf/codecs/text_decoder.py:LatentTextDecoder, configs/pretrain.yaml
        model.text_decoder {max_length: 8, input_space: normalized}
  6. argmax per position -> token ids -> T5 detokenization

Weights (public, no token required)
----------------------------------
  * hustvl/Multimodal-Flow, file "Text Decoder/text_decoder.safetensors",
    71,580,384 bytes, safetensors metadata {source: pretrain-step-160000,
    weights: ema, format: multimodal-flow-hf}; 80 BF16 tensors loaded into
    float32 parameters. The key set matches LatentTextDecoder.state_dict()
    exactly, so load_state_dict(strict=True) succeeds.
  * google-t5/t5-small, frozen encoder + sentencepiece tokenizer.

Experiments
-----------
  A  clean round trip: encode -> normalize -> decode -> detokenize
  B  lambda sweep, lambda = sigmoid(0.8 + 0.8 z), the released corruption rule
  B2 additive error: h_hat = h + sigma_e * eps, the flow's error budget
  C  context-enriched encoder input (whole-record T5) fed to the frozen decoder
  D  controls: position-permuted latents and foreign-block latents (leak check)
  E  leave-one-out latent swap: how much of a token survives on the other
     seven positions of its own block (intra-block redundancy of the coding)

Run
---
    code/.venv-mmf/bin/python roundtrip.py

Defaults resolve weights from <workspace>/code/mmf-assets and text from
<workspace>/papers; override with the CLI flags. Outputs land next to this
script in results/roundtrip.json plus fig-roundtrip.png.

Note on threads: the machine's CPU threading is heavily oversubscribed, and a
tiny T5 forward takes 1.6 s at 8 threads against 17 ms at 1 thread. The probe
therefore pins torch to a single thread by default (--threads).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

BLOCK_SIZE = 8
TEXT_LATENT_DIM = 512
TEXT_STATS_MEAN = 0.0
TEXT_STATS_STD = 0.2
DECODER_P_MEAN = 0.8
DECODER_P_STD = 0.8
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1
MAX_RECORD_TOKENS = 512
DECODE_CHUNK = 128

HERE = Path(__file__).resolve()
WORKSPACE = HERE.parents[4]

DEFAULT_DECODER = WORKSPACE / "code/mmf-assets/hustvl-mf/Text Decoder/text_decoder.safetensors"
DEFAULT_T5 = WORKSPACE / "code/mmf-assets/t5-small"

# Prose sources, relative to the papers root: one paper markdown per area, so
# the sample set is not dominated by a single writing style. Chosen for dense
# English prose, not for content.
PAPER_SOURCES = (
    "agents/harness-handbook-2607.13285.md",
    "robotics/apt-action-expert-pretraining-2606.12366.md",
    "vla/eventvla-2606.20092.md",
    "perception/invisible-shortcuts-2608.05424.md",
    "control/pir2-2607.26055.md",
    "embodied-data/ace-data-0-2607.28625.md",
)

# Edge-case sentences: punctuation, digits, proper nouns, hyphenation, quotes,
# abbreviations, symbols, capitalization, non-ASCII, and short fragments.
HANDMADE = (
    "The quick brown fox jumps over the lazy dog.",
    "Wait — did she really say that?!",
    "GPT-4o, Llama-3.1-405B, and Qwen2.5-72B are all decoder-only models.",
    "In 2026, the benchmark reported 82.8 on GenEval and 75.3 on VQAv2.",
    "The loss dropped from 3.14 to 0.0027 after 100,000 steps.",
    "e.g. the model's input (x, y) maps to z = f(x) + g(y).",
    "He said: \"I'll be back by 5 p.m., OK?\"",
    "A 512-dimensional latent holds 8 tokens; 8 x 512 = 4,096 floats.",
    "BERT-Base, RoBERTa-Large, and DeBERTa-v3 are encoder models.",
    "Is it 3.5% or 35%? The table says 0.035.",
    "T5-small's sentencepiece vocabulary has 32,128 entries including padding.",
    "Section 4.2.1 describes the ablation; see Table 3 and Figure 5(b).",
    "The authors' 0.6B, 1.2B, and 1.6B variants were trained for 150B tokens.",
    "OK.",
    "Yes,",
    "No!",
    "The URL is https://huggingface.co/hustvl/Multimodal-Flow.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "COVID-19, SARS-CoV-2, and mRNA vaccines: a 2020-2021 retrospective.",
    "The GPU (NVIDIA H100, 80 GB) ran at 95% utilization for 12 hours.",
    "Einstein's E = mc^2 is famous, but F = ma matters more here.",
    "Numbers like 1,234,567.89 and dates like 2026-10-12 must survive.",
    "Siamese twins; Oxford commas, and semicolons; oh my.",
    "The prefix 'un-' and the suffix '-ness' combine in 'unhappiness'.",
    "a a a a a a a a a a a a a a a a",
    "Repeat after me: token, token, token, token, token.",
    "Despite the noise, the signal — weak as it was — remained measurable.",
    "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16",
    "The model is frozen; the backbone is trained; the decoder is frozen again.",
    "Multimodal Flow: unified flow modeling of language and vision in embedding spaces.",
    "A man walks into a bar and orders a beer for his robot.",
    "中文句子也应该被这个编码器处理。",
    "日本語の文章もテストします。",
    "Emoji check: the rocket launched without issues.",
    "Mixed 中英 text with numbers 123 and symbols.",
    "Trailing whitespace aside, this sentence ends with a period.",
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？])|(?<=[.!?])\s+")


def display_path(path: Path) -> str:
    """Workspace-relative where possible: run artifacts must not carry machine paths."""

    try:
        return str(path.resolve().relative_to(WORKSPACE))
    except ValueError:
        return f"<external>/{path.name}"


def load_vendored(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import vendored module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


text_decoder_module = load_vendored(HERE.parent / "mf_text_decoder.py", "mf_text_decoder")
decoder_noise_module = load_vendored(HERE.parent / "mf_decoder_noise.py", "mf_decoder_noise")
LatentTextDecoder = text_decoder_module.LatentTextDecoder
corrupt_text_decoder_latents = decoder_noise_module.corrupt_text_decoder_latents


# --------------------------------------------------------------------------- #
# text preparation (mirrors mf/data/text.py)
# --------------------------------------------------------------------------- #
def split_complete_text_units(text: str) -> tuple[str, ...]:
    units: list[str] = []
    for paragraph in re.split(r"[\r\n]+", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        units.extend(
            sentence
            for sentence in (part.strip() for part in _SENTENCE_BOUNDARY.split(paragraph))
            if sentence
        )
    return tuple(units)


def is_prose(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 30 or len(stripped) > 1200:
        return False
    if stripped.startswith(("#", "|", ">", "-", "*", "+", "!", "```", "<!--")):
        return False
    if re.match(r"^(Figure|Fig\.|Table|Equation|Algorithm|Appendix|Listing)\b", stripped):
        return False
    if re.search(r"(\$|\\[a-zA-Z]+|\\\(|_\{|\^\{|&\w+;|<[a-z/]+>)", stripped):
        return False
    letters = sum(character.isalpha() or character.isspace() for character in stripped)
    return letters / len(stripped) > 0.75


def content_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return [token_id for token_id in encoded if token_id not in (EOS_TOKEN_ID, PAD_TOKEN_ID)]


def build_record(ids: list[int], *, block_size: int = BLOCK_SIZE) -> list[int]:
    """Append EOS and EOS-fill the terminal block, as the block_aligned_eos packer does."""

    if not ids:
        return []
    record = [*ids, EOS_TOKEN_ID]
    record.extend([EOS_TOKEN_ID] * ((-len(record)) % block_size))
    return record


def collect_samples(tokenizer, papers_root: Path, seed: int, max_paper_samples: int):
    import random

    rng = random.Random(seed)
    pool: list[tuple[int, str, str]] = []
    for relative in PAPER_SOURCES:
        path = papers_root / relative
        if not path.is_file():
            continue
        for unit in split_complete_text_units(path.read_text(encoding="utf-8")):
            if not is_prose(unit):
                continue
            n_tokens = len(content_ids(tokenizer, unit))
            if 4 <= n_tokens <= 200:
                pool.append((n_tokens, unit, relative))
    if not pool:
        raise RuntimeError(
            f"no prose units found under {papers_root}; check PAPER_SOURCES and --papers-root"
        )
    pool.sort(key=lambda item: (item[0], item[1]))
    # Length-stratified draw: bucket the length-sorted pool and walk each bucket in
    # equal strides, so the sample set spans short, medium, and long records
    # instead of collapsing onto the modal sentence length.
    bands = ((4, 12), (13, 24), (25, 40), (41, 70), (71, 120), (121, 200))
    per_band = max(1, max_paper_samples // len(bands))
    picked: list[tuple[int, str, str]] = []
    for low, high in bands:
        members = [item for item in pool if low <= item[0] <= high]
        if not members:
            continue
        stride = max(1, len(members) // per_band)
        picked.extend(members[::stride][:per_band])
    if len(picked) < max_paper_samples:
        chosen = {(item[0], item[1]) for item in picked}
        for item in pool:
            if len(picked) >= max_paper_samples:
                break
            if (item[0], item[1]) not in chosen:
                picked.append(item)
    picked = picked[:max_paper_samples]
    print(
        f"paper prose pool={len(pool)} units, picked={len(picked)} "
        f"(token lengths {min(item[0] for item in picked)}.."
        f"{max(item[0] for item in picked)})",
        flush=True,
    )
    rng.shuffle(picked)
    samples = [
        {"group": "paper", "source": relative, "text": text}
        for _, text, relative in picked
    ]
    samples.extend(
        {"group": "handmade", "source": "handmade", "text": text} for text in HANDMADE
    )
    return samples


# --------------------------------------------------------------------------- #
# model side
# --------------------------------------------------------------------------- #
class Probe:
    def __init__(self, decoder_weights: Path, t5_path: Path, device: str = "cpu"):
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, T5EncoderModel

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(t5_path)
        self.encoder = T5EncoderModel.from_pretrained(t5_path).to(self.device).eval()
        self.decoder = LatentTextDecoder(input_dim=TEXT_LATENT_DIM, max_length=BLOCK_SIZE)
        state = load_file(str(decoder_weights))
        self.decoder.load_state_dict(state, strict=True)
        self.decoder = self.decoder.to(self.device).eval()
        self.special_ids = set(self.tokenizer.all_special_ids)
        self.decoder_state_keys = len(state)

    @torch.no_grad()
    def encode_blocks(self, blocks: torch.Tensor) -> torch.Tensor:
        """Block-local T5 encoding: each 8-token block is encoded independently."""

        latents = []
        for chunk in blocks.split(4096):
            out = self.encoder(
                input_ids=chunk, attention_mask=torch.ones_like(chunk)
            ).last_hidden_state
            latents.append(out)
        return torch.cat(latents).float()

    @torch.no_grad()
    def encode_records(self, records: list[list[int]], batch_size: int = 16) -> list[torch.Tensor]:
        """Whole-record T5 encoding, used only for the global-context control.

        Records are padded to the batch maximum; padding is masked out, so the
        rows match the per-record encodings (verified once in main()).
        """

        latents: list[torch.Tensor] = []
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            width = max(len(record) for record in batch)
            ids = torch.full((len(batch), width), PAD_TOKEN_ID, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row, record in enumerate(batch):
                ids[row, : len(record)] = torch.tensor(record, dtype=torch.long)
                mask[row, : len(record)] = 1
            out = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state.float()
            latents.extend(out[row, : len(record)] for row, record in enumerate(batch))
        return latents

    @torch.no_grad()
    def evaluate(self, latents: torch.Tensor, dataset: "Dataset", stats: bool = True) -> dict:
        """Decode latents and accumulate predictions plus cross-entropy statistics."""

        predictions = []
        ce_sum = 0.0
        top1_sum = 0.0
        top1_correct_sum = 0.0
        count = 0
        correct_count = 0
        for start in range(0, latents.shape[0], DECODE_CHUNK):
            block = latents[start : start + DECODE_CHUNK]
            targets = dataset.block_tokens[start : start + DECODE_CHUNK]
            real = dataset.block_real_mask[start : start + DECODE_CHUNK]
            logits = self.decoder(block).float()
            prediction = logits.argmax(dim=-1)
            correct = prediction == targets
            if stats:
                log_probs = torch.log_softmax(logits, dim=-1)
                nll = -log_probs.gather(
                    -1, targets.clamp(0, logits.shape[-1] - 1).unsqueeze(-1)
                ).squeeze(-1)
                top1 = log_probs.exp().max(dim=-1).values
                ce_sum += float(nll[real].sum())
                top1_sum += float(top1[real].sum())
                top1_correct_sum += float(top1[real][correct[real]].sum())
            count += int(real.sum())
            correct_count += int(correct[real].sum())
            predictions.append(prediction)
        stacked = torch.cat(predictions)
        return {
            "predictions": stacked,
            "ce_nats": ce_sum / max(1, count),
            "ppl": math.exp(ce_sum / max(1, count)),
            "mean_top1_prob": top1_sum / max(1, count),
            "mean_top1_prob_correct": top1_correct_sum / max(1, correct_count),
            "count": count,
        }

    def render(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


def normalize(latents: torch.Tensor) -> torch.Tensor:
    return (latents - TEXT_STATS_MEAN) / TEXT_STATS_STD


def corrupted_with_z(latents_norm: torch.Tensor, z: float, eps: torch.Tensor) -> torch.Tensor:
    """lambda = sigmoid(0.8 + 0.8 z), x~ = lambda x + (1 - lambda) eps (released rule)."""

    weight = float(torch.sigmoid(torch.tensor(z * DECODER_P_STD + DECODER_P_MEAN)))
    return weight * latents_norm + (1.0 - weight) * eps


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class Dataset:
    """Flat block tensors plus the bookkeeping needed to rebuild per-sample text."""

    def __init__(self, probe: Probe, samples: list[dict]):
        self.samples = samples
        self.records: list[list[int]] = []
        self.real_ids: list[list[int]] = []
        blocks: list[list[int]] = []
        real_mask: list[list[bool]] = []
        block_owner: list[int] = []
        block_index_in_sample: list[int] = []
        counters = [0] * len(samples)
        for index, sample in enumerate(samples):
            ids = content_ids(probe.tokenizer, sample["text"])
            record = build_record(ids)
            sample["n_real_tokens"] = len(ids)
            sample["n_blocks"] = len(record) // BLOCK_SIZE
            self.records.append(record)
            self.real_ids.append(ids)
            for block_index in range(len(record) // BLOCK_SIZE):
                start = block_index * BLOCK_SIZE
                blocks.append(record[start : start + BLOCK_SIZE])
                real_mask.append(
                    [position < len(ids) for position in range(start, start + BLOCK_SIZE)]
                )
                block_owner.append(index)
                block_index_in_sample.append(counters[index])
                counters[index] += 1
        self.block_tokens = torch.tensor(blocks, dtype=torch.long)
        self.block_real_mask = torch.tensor(real_mask, dtype=torch.bool)
        self.block_owner = torch.tensor(block_owner, dtype=torch.long)
        self.block_index_in_sample = block_index_in_sample

    def predictions_to_real_ids(self, predictions: torch.Tensor) -> list[list[int]]:
        per_sample: list[list[int]] = [[] for _ in self.samples]
        rows, columns = self.block_real_mask.nonzero(as_tuple=True)
        for row, column in zip(rows.tolist(), columns.tolist()):
            per_sample[int(self.block_owner[row])].append(int(predictions[row, column]))
        return per_sample

    def score(self, probe: Probe, predictions: torch.Tensor) -> tuple[dict, list[dict]]:
        correct = predictions == self.block_tokens
        real = self.block_real_mask
        per_sample_real = self.predictions_to_real_ids(predictions)
        rows: list[dict] = []
        for index, sample in enumerate(self.samples):
            true_ids = self.real_ids[index]
            pred_ids = per_sample_real[index]
            matches = sum(int(a == b) for a, b in zip(true_ids, pred_ids))
            original = sample["text"].strip()
            baseline = probe.render(true_ids)
            reconstruction = probe.render(pred_ids)
            denominator = max(1, len(original))
            rows.append(
                {
                    "group": sample["group"],
                    "n_real_tokens": len(true_ids),
                    "n_blocks": sample["n_blocks"],
                    "matches": matches,
                    "token_acc": matches / max(1, len(true_ids)),
                    "exact": bool(matches == len(true_ids)),
                    "edit_norm": levenshtein(original, reconstruction) / denominator,
                    "edit_norm_baseline": levenshtein(original, baseline) / denominator,
                    "special_predictions": sum(1 for t in pred_ids if t in probe.special_ids),
                    "text": original[:200],
                    "reconstruction": reconstruction[:200],
                }
            )
        per_position = [
            float(correct[:, column][real[:, column]].float().mean())
            if real[:, column].any()
            else float("nan")
            for column in range(self.block_tokens.shape[1])
        ]
        summary = {
            "n_samples": len(rows),
            "n_blocks": int(self.block_tokens.shape[0]),
            "n_real_tokens": int(real.sum()),
            "token_acc_real": float(correct[real].float().mean()),
            "token_acc_all_targets": float(correct.float().mean()),
            "block_exact": float(correct.all(dim=1).float().mean()),
            "sample_exact": float(np.mean([row["exact"] for row in rows])),
            "sample_token_acc_mean": float(np.mean([row["token_acc"] for row in rows])),
            "sample_token_acc_median": float(np.median([row["token_acc"] for row in rows])),
            "edit_norm_mean": float(np.mean([row["edit_norm"] for row in rows])),
            "edit_norm_median": float(np.median([row["edit_norm"] for row in rows])),
            "edit_norm_baseline_mean": float(
                np.mean([row["edit_norm_baseline"] for row in rows])
            ),
            "per_position_acc": per_position,
            "special_predictions": int(sum(row["special_predictions"] for row in rows)),
            "by_group": {},
        }
        for group in sorted({row["group"] for row in rows}):
            group_rows = [row for row in rows if row["group"] == group]
            summary["by_group"][group] = {
                "n_samples": len(group_rows),
                "token_acc_real": float(np.mean([row["token_acc"] for row in group_rows])),
                "sample_exact": float(np.mean([row["exact"] for row in group_rows])),
                "edit_norm_mean": float(np.mean([row["edit_norm"] for row in group_rows])),
            }
        return summary, rows


def exemplars(
    dataset: Dataset, probe: Probe, predictions: torch.Tensor, indices: list[int]
) -> list[dict]:
    per_sample_real = dataset.predictions_to_real_ids(predictions)
    rows = []
    for index in indices:
        true_ids = dataset.real_ids[index]
        pred_ids = per_sample_real[index]
        matches = sum(int(a == b) for a, b in zip(true_ids, pred_ids))
        rows.append(
            {
                "index": index,
                "group": dataset.samples[index]["group"],
                "n_real_tokens": len(true_ids),
                "token_acc": matches / max(1, len(true_ids)),
                "text": dataset.samples[index]["text"].strip()[:200],
                "reconstruction": probe.render(pred_ids)[:200],
            }
        )
    return rows


def pick_exemplar_indices(samples: list[dict]) -> list[int]:
    markers = (
        "In 2026, the benchmark",
        "GPT-4o",
        "The URL is",
        "中文句子",
        "Wait — did she",
        "The GPU (NVIDIA",
    )
    indices: list[int] = []
    for marker in markers:
        for position, sample in enumerate(samples):
            if marker in sample["text"]:
                indices.append(position)
                break
    paper = [position for position, sample in enumerate(samples) if sample["group"] == "paper"]
    for fraction in (0.0, 0.3, 0.6, 0.95):
        if paper:
            indices.append(paper[int(fraction * (len(paper) - 1))])
    return sorted(set(indices))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-weights", type=Path, default=DEFAULT_DECODER)
    parser.add_argument("--t5", type=Path, default=DEFAULT_T5)
    parser.add_argument("--papers-root", type=Path, default=WORKSPACE / "papers")
    parser.add_argument("--results-dir", type=Path, default=HERE.parent / "results")
    parser.add_argument("--figure", type=Path, default=HERE.parent / "fig-roundtrip.png")
    parser.add_argument("--max-paper-samples", type=int, default=200)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument(
        "--figure-only",
        action="store_true",
        help="redraw the figure from <results-dir>/roundtrip.json without touching the models",
    )
    args = parser.parse_args()

    if args.figure_only:
        payload = json.loads((args.results_dir / "roundtrip.json").read_text(encoding="utf-8"))
        make_figure(payload, args.figure)
        return 0

    started = time.time()
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)

    probe = Probe(args.decoder_weights, args.t5)
    samples = collect_samples(probe.tokenizer, args.papers_root, args.seed, args.max_paper_samples)
    dataset = Dataset(probe, samples)
    print(
        f"samples={len(samples)} blocks={dataset.block_tokens.shape[0]} "
        f"real_tokens={int(dataset.block_real_mask.sum())}",
        flush=True,
    )

    # --- raw latent statistics of the frozen T5 encoder -------------------- #
    latents_raw = probe.encode_blocks(dataset.block_tokens)
    latents_norm = normalize(latents_raw)
    per_dim_std = latents_raw.reshape(-1, TEXT_LATENT_DIM).std(dim=0)
    latent_stats = {
        "raw_mean": float(latents_raw.mean()),
        "raw_std": float(latents_raw.std()),
        "raw_std_per_dim_min": float(per_dim_std.min()),
        "raw_std_per_dim_median": float(per_dim_std.median()),
        "raw_std_per_dim_max": float(per_dim_std.max()),
        "configured_text_normal_std": TEXT_STATS_STD,
        "normalized_std": float(latents_norm.std()),
        "normalized_std_per_dim_min": float(per_dim_std.min() / TEXT_STATS_STD),
        "normalized_std_per_dim_median": float(per_dim_std.median() / TEXT_STATS_STD),
        "normalized_std_per_dim_max": float(per_dim_std.max() / TEXT_STATS_STD),
    }

    results: dict[str, object] = {
        "probe": "mf1-text-latent-roundtrip",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "provenance": {
            "decoder_weights": display_path(args.decoder_weights),
            "decoder_weights_bytes": args.decoder_weights.stat().st_size,
            "decoder_weights_sha256": "86ba2911fad4b18e497ae4135d5841d6ecee9dbee7e6b4abe47c5c1e06a0a922",
            "decoder_weights_metadata": {
                "source": "pretrain-step-160000",
                "weights": "ema",
                "format": "multimodal-flow-hf",
            },
            "decoder_state_keys": probe.decoder_state_keys,
            "decoder_params": 35786364,
            "t5": display_path(args.t5),
            "paper_sources": list(PAPER_SOURCES),
            "block_size": BLOCK_SIZE,
            "max_record_tokens": MAX_RECORD_TOKENS,
            "torch": torch.__version__,
            "threads": args.threads,
        },
        "latent_stats": latent_stats,
    }

    # --- corruption replication check against the released function --------- #
    generator = torch.Generator().manual_seed(args.seed)
    released = corrupt_text_decoder_latents(latents_norm, generator)
    replay_generator = torch.Generator().manual_seed(args.seed)
    z = torch.randn(latents_norm.shape[:-1], generator=replay_generator)
    eps = torch.randn(latents_norm.shape, generator=replay_generator)
    weight = torch.sigmoid(z * DECODER_P_STD + DECODER_P_MEAN).unsqueeze(-1)
    replay = weight * latents_norm + (1.0 - weight) * eps
    exact = bool(torch.equal(released.corrupted, replay))
    results["corruption_replication_exact"] = exact
    if not exact:
        raise AssertionError("lambda-corruption replication does not match released code")
    results["lambda_mean_training_draw"] = float(released.lambda_.mean())

    # --- experiment A: clean round trip ------------------------------------ #
    clean = probe.evaluate(latents_norm, dataset)
    summary_clean, rows_clean = dataset.score(probe, clean["predictions"])
    summary_clean["ce_nats"] = clean["ce_nats"]
    summary_clean["ppl"] = clean["ppl"]
    summary_clean["mean_top1_prob"] = clean["mean_top1_prob"]
    summary_clean["mean_top1_prob_correct"] = clean["mean_top1_prob_correct"]
    results["experiment_a_clean"] = summary_clean
    print(f"A clean: acc={summary_clean['token_acc_real']:.4f} "
          f"ppl={summary_clean['ppl']:.4f}", flush=True)

    # control: wrong input space (raw, unnormalized latents)
    raw_eval = probe.evaluate(latents_raw, dataset)
    summary_raw, _ = dataset.score(probe, raw_eval["predictions"])
    summary_raw["ppl"] = raw_eval["ppl"]
    results["control_raw_input_space"] = summary_raw

    # --- experiment B: fixed-z lambda sweep -------------------------------- #
    eps_fixed = torch.randn(
        latents_norm.shape, generator=torch.Generator().manual_seed(args.seed + 1)
    )
    z_grid = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    lambda_sweep = []
    sweep_predictions: dict[float, torch.Tensor] = {}
    sweep_sample_acc: dict[str, list[float]] = {}
    for z_value in z_grid:
        corrupted = corrupted_with_z(latents_norm, z_value, eps_fixed)
        evaluation = probe.evaluate(corrupted, dataset)
        sweep_predictions[z_value] = evaluation["predictions"]
        summary, rows = dataset.score(probe, evaluation["predictions"])
        sweep_sample_acc[str(z_value)] = [row["token_acc"] for row in rows]
        lambda_sweep.append(
            {
                "z": z_value,
                "lambda": float(
                    torch.sigmoid(torch.tensor(z_value * DECODER_P_STD + DECODER_P_MEAN))
                ),
                "token_acc_real": summary["token_acc_real"],
                "block_exact": summary["block_exact"],
                "sample_exact": summary["sample_exact"],
                "edit_norm_mean": summary["edit_norm_mean"],
                "ppl": evaluation["ppl"],
                "per_position_acc": summary["per_position_acc"],
            }
        )
        print(f"B lambda z={z_value:+.1f} acc={summary['token_acc_real']:.4f} "
              f"ppl={evaluation['ppl']:.2f}", flush=True)
    results["experiment_b_lambda_sweep"] = lambda_sweep
    results["sweep_sample_acc"] = sweep_sample_acc

    # training-distribution reference: z ~ N(0, 1) drawn per token position
    training_draws = []
    for repeat in range(3):
        generator = torch.Generator().manual_seed(args.seed + 100 + repeat)
        corruption = corrupt_text_decoder_latents(latents_norm, generator)
        evaluation = probe.evaluate(corruption.corrupted, dataset)
        summary, _ = dataset.score(probe, evaluation["predictions"])
        training_draws.append(
            {
                "repeat": repeat,
                "token_acc_real": summary["token_acc_real"],
                "block_exact": summary["block_exact"],
                "sample_exact": summary["sample_exact"],
                "ppl": evaluation["ppl"],
                "lambda_mean": float(corruption.lambda_.mean()),
            }
        )
    results["experiment_b_training_distribution"] = {
        "draws": training_draws,
        "token_acc_real_mean": float(np.mean([d["token_acc_real"] for d in training_draws])),
        "token_acc_real_std": float(np.std([d["token_acc_real"] for d in training_draws])),
        "lambda_mean": float(np.mean([d["lambda_mean"] for d in training_draws])),
    }

    # --- experiment B2: additive error budget ------------------------------ #
    additive = []
    additive_predictions: dict[float, torch.Tensor] = {}
    for scale in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        noise = torch.randn(
            latents_norm.shape, generator=torch.Generator().manual_seed(args.seed + 7)
        )
        evaluation = probe.evaluate(latents_norm + scale * noise, dataset)
        additive_predictions[scale] = evaluation["predictions"]
        summary, _ = dataset.score(probe, evaluation["predictions"])
        additive.append(
            {
                "error_std": scale,
                "error_std_raw_t5_units": scale * TEXT_STATS_STD,
                "token_acc_real": summary["token_acc_real"],
                "block_exact": summary["block_exact"],
                "sample_exact": summary["sample_exact"],
                "edit_norm_mean": summary["edit_norm_mean"],
                "ppl": evaluation["ppl"],
            }
        )
        print(f"B2 additive std={scale:.2f} acc={summary['token_acc_real']:.4f} "
              f"ppl={evaluation['ppl']:.2f}", flush=True)
    results["experiment_b_additive_noise"] = additive

    # --- experiment C: context-enriched encoder input ---------------------- #
    record_latents = probe.encode_records(dataset.records)
    single = probe.encode_records(dataset.records[:1], batch_size=1)[0]
    padding_check = bool(
        torch.allclose(
            record_latents[0],
            single,
            atol=1e-4,
        )
    )
    results["batched_record_encoding_matches_single"] = padding_check
    if not padding_check:
        raise AssertionError("batched padded record encoding does not match single-record encoding")
    blocks_from_records = []
    for owner, block_index in zip(
        dataset.block_owner.tolist(), dataset.block_index_in_sample
    ):
        record = record_latents[owner]
        blocks_from_records.append(record[block_index * BLOCK_SIZE : (block_index + 1) * BLOCK_SIZE])
    global_latents = normalize(torch.stack(blocks_from_records))
    global_eval = probe.evaluate(global_latents, dataset)
    summary_global, _ = dataset.score(probe, global_eval["predictions"])
    summary_global["ppl"] = global_eval["ppl"]
    results["experiment_c_global_context_encoder"] = summary_global
    print(f"C global-context: acc={summary_global['token_acc_real']:.4f}", flush=True)

    # --- experiment D: controls -------------------------------------------- #
    permutation = torch.roll(torch.arange(BLOCK_SIZE), 1)
    permuted_eval = probe.evaluate(latents_norm[:, permutation], dataset)
    summary_perm, _ = dataset.score(probe, permuted_eval["predictions"])
    summary_perm["ppl"] = permuted_eval["ppl"]
    results["control_position_permuted"] = summary_perm

    foreign_order = torch.randperm(
        latents_norm.shape[0], generator=torch.Generator().manual_seed(args.seed + 3)
    )
    foreign_eval = probe.evaluate(latents_norm[foreign_order], dataset)
    summary_foreign, _ = dataset.score(probe, foreign_eval["predictions"])
    summary_foreign["ppl"] = foreign_eval["ppl"]
    results["control_foreign_block"] = summary_foreign
    print(f"D permuted acc={summary_perm['token_acc_real']:.4f} "
          f"foreign acc={summary_foreign['token_acc_real']:.4f}", flush=True)

    # --- experiment E: leave-one-out latent swap --------------------------- #
    # Replace the latent at one position with the same position of a foreign
    # block, keep the other seven intact: how much of a token is recoverable
    # from the rest of its own block (intra-block redundancy of the coding)?
    leave_one_out = []
    for position in range(BLOCK_SIZE):
        swapped = latents_norm.clone()
        swapped[:, position, :] = latents_norm[foreign_order, position, :]
        evaluation = probe.evaluate(swapped, dataset, stats=False)
        correct = evaluation["predictions"] == dataset.block_tokens
        real = dataset.block_real_mask
        per_position = [
            float(correct[:, column][real[:, column]].float().mean())
            if real[:, column].any()
            else float("nan")
            for column in range(BLOCK_SIZE)
        ]
        others = [
            per_position[column]
            for column in range(BLOCK_SIZE)
            if column != position
        ]
        leave_one_out.append(
            {
                "swapped_position": position,
                "swapped_position_acc": per_position[position],
                "intact_positions_acc_mean": float(np.mean(others)),
                "overall_token_acc": float(correct[real].float().mean()),
            }
        )
        print(f"E leave-one-out position {position}: swapped acc="
              f"{per_position[position]:.4f} intact={np.mean(others):.4f}", flush=True)
    results["experiment_e_leave_one_out"] = {
        "per_position": leave_one_out,
        "swapped_token_acc_mean": float(
            np.mean([entry["swapped_position_acc"] for entry in leave_one_out])
        ),
        "intact_token_acc_mean": float(
            np.mean([entry["intact_positions_acc_mean"] for entry in leave_one_out])
        ),
    }

    # --- exemplars: what the reconstructions actually look like ------------- #
    exemplar_indices = pick_exemplar_indices(samples)
    results["exemplar_indices"] = exemplar_indices
    results["exemplars"] = {
        "clean": exemplars(dataset, probe, clean["predictions"], exemplar_indices),
        "lambda_z-1.5": exemplars(dataset, probe, sweep_predictions[-1.5], exemplar_indices),
        "lambda_z-2.0": exemplars(dataset, probe, sweep_predictions[-2.0], exemplar_indices),
        "additive_std_1.0": exemplars(
            dataset, probe, additive_predictions[1.0], exemplar_indices
        ),
        "foreign_block_control": exemplars(
            dataset, probe, foreign_eval["predictions"], exemplar_indices
        ),
    }

    # --- failure cases and per-sample table -------------------------------- #
    worst = sorted(rows_clean, key=lambda row: (row["token_acc"], -row["n_real_tokens"]))[:25]
    results["failure_cases"] = worst
    results["samples"] = [
        {
            "group": row["group"],
            "n_real_tokens": row["n_real_tokens"],
            "n_blocks": row["n_blocks"],
            "token_acc": row["token_acc"],
            "exact": row["exact"],
            "edit_norm": row["edit_norm"],
            "edit_norm_baseline": row["edit_norm_baseline"],
            "text": row["text"],
            "reconstruction": row["reconstruction"],
        }
        for row in rows_clean
    ]
    results["runtime_seconds"] = round(time.time() - started, 1)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.results_dir / "roundtrip.json"
    json_path.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {json_path}", flush=True)

    if not args.no_figure:
        make_figure(results, args.figure)

    print(f"clean token acc (real tokens) = {summary_clean['token_acc_real']:.4f}")
    print(f"clean sample exact            = {summary_clean['sample_exact']:.4f}")
    print(f"clean edit distance (norm)    = {summary_clean['edit_norm_mean']:.4f}")
    print(f"tokenizer-only edit floor     = {summary_clean['edit_norm_baseline_mean']:.4f}")
    return 0


def make_figure(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_clean = results["samples"]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), facecolor="white")
    sweep = results["experiment_b_lambda_sweep"]
    lambdas = [entry["lambda"] for entry in sweep]

    axis = axes[0][0]
    axis.plot(lambdas, [entry["token_acc_real"] for entry in sweep], "o-", color="#1f77b4",
              label="token accuracy (content tokens)")
    axis.plot(lambdas, [entry["block_exact"] for entry in sweep], "s--", color="#d62728",
              label="exact block (8/8 tokens)")
    axis.axvspan(
        float(torch.sigmoid(torch.tensor(-1.0 * DECODER_P_STD + DECODER_P_MEAN))),
        float(torch.sigmoid(torch.tensor(1.0 * DECODER_P_STD + DECODER_P_MEAN))),
        color="#cccccc", alpha=0.35, label=r"$\lambda$ for $z\in[-1,1]$"
    )
    axis.axvline(float(torch.sigmoid(torch.tensor(DECODER_P_MEAN))), color="#2ca02c",
                 linestyle=":", label=r"$\lambda=\sigma(0.8)=0.69$ ($z=0$)")
    axis.set_xlabel(r"$\lambda$ in $\tilde h=\lambda h+(1-\lambda)\epsilon$")
    axis.set_ylabel("fidelity")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.3)
    axis.set_title("(a) Decoder fidelity under the released corruption rule\n(z fixed per sweep point)")
    axis_ce = axis.twinx()
    axis_ce.plot(lambdas, [entry["ppl"] for entry in sweep], color="#7f7f7f", lw=1.0,
                 label="decoder perplexity (right axis)")
    axis_ce.set_ylabel("decoder perplexity (nats-teacher-forced)", fontsize=9, color="#7f7f7f")
    axis_ce.set_yscale("log")
    handles, labels = axis.get_legend_handles_labels()
    handles2, labels2 = axis_ce.get_legend_handles_labels()
    axis.legend(handles + handles2, labels + labels2, fontsize=8, loc="lower right")

    axis = axes[0][1]
    additive = results["experiment_b_additive_noise"]
    errors = [entry["error_std"] for entry in additive]
    axis.plot(errors, [entry["token_acc_real"] for entry in additive], "o-", color="#1f77b4",
              label="token accuracy (content tokens)")
    axis.plot(errors, [entry["block_exact"] for entry in additive], "s--", color="#d62728",
              label="exact block (8/8 tokens)")
    axis.axhline(0.99, color="#888888", linestyle=":", linewidth=1)
    axis.text(0.02, 0.905, "99% ", fontsize=8, color="#888888")
    axis.set_xlabel(r"additive error std, normalized space ($\hat h = h + \sigma_e\epsilon$)")
    axis.set_ylabel("fidelity")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.3)
    axis.set_title("(b) Error budget of the frozen decoder\n(isotropic error the flow may leave behind)")
    axis.legend(fontsize=8, loc="lower left")

    axis = axes[1][0]
    stressed = results["sweep_sample_acc"]["-1.5"]
    clean_values = [row["token_acc"] for row in rows_clean]
    axis.hist(stressed, bins=np.linspace(0.0, 1.0, 21), color="#dd8452", edgecolor="white",
              label=rf"$\lambda=0.40$ ($z=-1.5$), mean={np.mean(stressed):.3f}")
    axis.axvline(1.0, color="#4c72b0", linestyle="--", linewidth=1.2,
                 label=f"clean latents: all {len(clean_values)} samples at 1.0")
    axis.set_xlabel("per-sample content-token accuracy")
    axis.set_ylabel("samples")
    axis.set_title("(c) Spread of per-sample fidelity at a stressed\noperating point vs. the clean round trip")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="upper left")

    axis = axes[1][1]
    positions = list(range(BLOCK_SIZE))
    stressed = next(entry for entry in sweep if abs(entry["z"] + 1.5) < 1e-9)
    swapped = [
        entry["swapped_position_acc"]
        for entry in results["experiment_e_leave_one_out"]["per_position"]
    ]
    axis.bar([p - 0.2 for p in positions], stressed["per_position_acc"], width=0.4,
             color="#dd8452", label=rf"$\lambda={stressed['lambda']:.2f}$ ($z=-1.5$) corruption")
    axis.bar([p + 0.2 for p in positions], swapped, width=0.4, color="#55a868",
             label="own latent replaced by a foreign block's")
    axis.axhline(1.0, color="#4c72b0", linestyle="--", linewidth=1.2, label="clean latents (all 1.0)")
    axis.set_xlabel("token position inside the 8-token block")
    axis.set_ylabel("accuracy")
    axis.set_ylim(0, 1.08)
    axis.grid(alpha=0.3, axis="y")
    axis.set_title("(d) Positional weakness: mild noise vs. losing\nthe token's own latent slot")
    axis.legend(fontsize=8, loc="lower right")

    figure.suptitle(
        "MF-1 text latent round trip: 8-token T5-small blocks through the frozen 6-layer decoder",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=160, facecolor="white")
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
