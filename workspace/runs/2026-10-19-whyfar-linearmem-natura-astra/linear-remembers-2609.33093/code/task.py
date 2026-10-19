"""WikiText-103 associative-recall task builder.

Layout of one source (paper App. B): 64-token natural prefix, K key-value bindings
each presented three times, 512 filler tokens, and a one-token query (":").
Binding surface form is " {key}: {value}." (4 tokens); the query is " {key}:".
The pre-query boundary is immediately before the final ":" token, i.e. the key has
already been read into the state and only the one-token read remains.
"""
import glob
import os

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
VALUE4 = ["red", "blue", "green", "yellow"]
KEYS = ["apple", "table", "river", "mountain", "window", "garden", "castle", "forest",
        "island", "doctor", "letter", "bottle", "engine", "market", "planet", "temple",
        "bridge", "candle", "mirror", "pillow", "desert", "valley", "summit", "harbor"]


def _ckpt():
    HF = os.path.expanduser("~/.cache/huggingface/hub")
    return sorted(glob.glob(os.path.join(
        HF, "models--m-a-p--340M-20B-GatedDeltaNet-pure-baseline/snapshots/*")))[0]


def tokenizer():
    return AutoTokenizer.from_pretrained(_ckpt())


def load_stream(split="validation"):
    path = os.path.join(HERE, "data", f"wt103_{split}.parquet")
    tbl = pq.read_table(path)
    text = "\n\n".join(t for t in tbl["text"].to_pylist() if t.strip())
    tok = tokenizer()
    return np.array(tok(text, add_special_tokens=False)["input_ids"], dtype=np.int64)


def bind_ids(key, value, tok):
    ids = tok.encode(f" {key}: {value}.", add_special_tokens=False)
    assert len(ids) == 4, (key, value, ids)
    return ids


def query_key_ids(key, tok):
    ids = tok.encode(f" {key}", add_special_tokens=False)
    assert len(ids) == 1, (key, ids)
    return ids


def query_colon_id(tok):
    ids = tok.encode(":", add_special_tokens=False)
    assert len(ids) == 1, ids
    return ids[0]


def value_token_ids(tok):
    out = {}
    for v in VALUE4:
        ids = tok.encode(f" {v}", add_special_tokens=False)
        assert len(ids) == 1, (v, ids)
        out[v] = ids[0]
    return out


def make_source(stream, offset, K, tok, target_idx=0, donor_value=None, seed=0,
                n_filler=512, fixed_total=None, distance=None, target_value=None):
    """Return dict with ids, fact token spans, target key/value, candidates.

    fixed_total: if set, shrink filler so len(ids) == fixed_total (load sweep).
    distance: if set, use exactly this many filler tokens after the bindings
              (distance sweep); overrides n_filler.
    """
    rng = np.random.default_rng(seed)
    prefix = stream[offset:offset + 64].tolist()
    keys = KEYS[:K] if K <= len(KEYS) else None
    if keys is None:
        raise ValueError(K)
    perm = rng.permutation(VALUE4).tolist()
    tgt_val = target_value if target_value is not None else perm[0]
    others = [v for v in VALUE4 if v != tgt_val]
    rng.shuffle(others)
    vals = [others[i % len(others)] for i in range(K)]
    vals[target_idx] = tgt_val
    if donor_value is None:
        donor_value = rng.choice(others)

    def build(value_override=None):
        ids = list(prefix)
        spans = []
        for rep in range(3):
            for j in range(K):
                val = vals[j]
                if j == target_idx and value_override is not None:
                    val = value_override
                b = bind_ids(keys[j], val, tok)
                if j == target_idx:
                    spans.append((len(ids), len(ids) + len(b)))
                ids.extend(b)
        if distance is not None:
            nf = distance
        elif fixed_total is not None:
            nf = fixed_total - len(ids) - 2
        else:
            nf = n_filler
        assert nf >= 0, (fixed_total, len(ids), nf)
        ids.extend(stream[offset + 64: offset + 64 + nf].tolist())
        qstart = len(ids)
        ids.extend(query_key_ids(keys[target_idx], tok))
        colon = query_colon_id(tok)
        ids.append(colon)
        return ids, spans, qstart

    ids, spans, qstart = build()
    donor_ids, _, _ = build(donor_value)
    assert len(ids) == len(donor_ids)
    return {
        "ids": ids, "donor_ids": donor_ids,
        "spans": spans, "query_key_pos": qstart, "colon_pos": len(ids) - 1,
        "keys": keys, "values": vals, "target_idx": target_idx,
        "target_key": keys[target_idx], "target_value": tgt_val, "donor_value": donor_value,
    }


def make_query_variant(src, key_idx, tok):
    """Same context, query another fact (for all-fact accuracy)."""
    ids = list(src["ids"][:src["query_key_pos"]])
    ids.extend(query_key_ids(src["keys"][key_idx], tok))
    ids.append(query_colon_id(tok))
    return ids
