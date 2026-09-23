"""Character-level data for the toy ComposeCL reproduction.

Vocabulary (~40 tokens): PAD, EOS, a replay seed token <|replay|> (the paper's
s, App. B.3.1), then punctuation, digits and lowercase letters.

Task stream: T tasks of 100 key->value associations each. Keys are random
6-char strings, values random 4-char strings, keys unique across ALL tasks
(Symbol-QA style, paper App. C.2). Format: "key: xxxxxx\nvalue: xxxx\n".
Every training sequence is prefixed with the replay token (App. B.3.1), so
p(. | s) is the distribution the data anchor samples from.
"""

import numpy as np

PAD, EOS, STOK = 0, 1, 2
_CHARS = "\n :" + "0123456789" + "abcdefghijklmnopqrstuvwxyz"
ID2CH = {PAD: "<pad>", EOS: "<eos>", STOK: "<|replay|>"}
CH2ID = {}
for i, ch in enumerate(_CHARS):
    ID2CH[3 + i] = ch
    CH2ID[ch] = 3 + i
VOCAB = len(ID2CH)  # 42

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def encode(text: str):
    return [CH2ID[c] for c in text]


def decode(ids, stop_at_eos: bool = False):
    out = []
    for i in ids:
        if i == EOS and stop_at_eos:
            break
        if i in (PAD, EOS, STOK):
            continue
        out.append(ID2CH[i])
    return "".join(out)


def encode_train_example(key: str, value: str):
    """[s] + "key: xxxxxx\nvalue: xxxx\n" + [EOS]  -> always 26 tokens."""
    return [STOK] + encode(f"key: {key}\nvalue: {value}\n") + [EOS]


def encode_prompt(key: str):
    """[s] + "key: xxxxxx\nvalue: "  -> always 20 tokens."""
    return [STOK] + encode(f"key: {key}\nvalue: ")


def make_filler(seed: int = 999, n_chars: int = 50_000) -> str:
    """Random filler text for base pretraining (covers every vocab char)."""
    rng = np.random.default_rng(seed)
    parts, total = [], 0
    word_count = 0
    while total < n_chars:
        r = rng.random()
        if r < 0.08:
            w = "".join(rng.choice(list("0123456789"), size=rng.integers(1, 5)))
        else:
            w = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"),
                                   size=rng.integers(1, 9)))
        if rng.random() < 0.05:
            w += ":"
        parts.append(w)
        total += len(w) + 1
        word_count += 1
        if word_count % int(rng.integers(8, 20)) == 0:
            parts.append("\n")
            total += 1
    return " ".join(parts)[:n_chars]


def make_filler_qa(seed: int = 999, n_chars: int = 50_000,
                   exclude_keys=frozenset(), key_pool_size: int = 300) -> str:
    """Random QA pairs in task format, keys from a fixed pool of 300.

    Two from-scratch constraints shape this choice (measured, see README):
    1. Pretraining on uniform random text never needs sharp predictions, so
       the output head stays diffuse and a rank-8 LoRA cannot move the
       softmax at all (acquisition ~0 at any budget).
    2. Pretraining on QA pairs with UNLIMITED random keys teaches the base
       "keys are uniform over 36^6", and task SFT then never concentrates
       p(.|s) on the trained keys: unconditional replay samples hit a true
       key in <1% of generations and the data anchor rehearses noise.
    A fixed pool of 300 filler keys (disjoint from the eval stream) with
    fresh random values teaches the format, a confident head, and a
    concentrated key prior — so after SFT the model's unconditional
    generations actually revisit trained associations (~60% exact kv at
    tau_G=1.5), which is what the paper's replay relies on.
    """
    rng = np.random.default_rng(seed)
    pool = set()
    while len(pool) < key_pool_size:
        pool.add("".join(rng.choice(list(ALPHABET), size=6)))
    pool = sorted(pool - set(exclude_keys))
    lines, total = [], 0
    while total < n_chars:
        key = pool[rng.integers(0, len(pool))]
        value = "".join(rng.choice(list(ALPHABET), size=4))
        line = f"key: {key}\nvalue: {value}\n"
        lines.append(line)
        total += len(line)
    return "".join(lines)[:n_chars]


def make_tasks(seed: int = 12345, n_tasks: int = 25, n_examples: int = 100):
    """T tasks x 100 (key, value) pairs; every key globally unique."""
    rng = np.random.default_rng(seed)
    tasks, seen = [], set()
    for _ in range(n_tasks):
        task = []
        while len(task) < n_examples:
            key = "".join(rng.choice(list(ALPHABET), size=6))
            if key in seen:
                continue
            seen.add(key)
            value = "".join(rng.choice(list(ALPHABET), size=4))
            task.append((key, value))
        tasks.append(task)
    return tasks
