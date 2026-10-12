"""Prepare the three input classes for the CyFA probes, all chunked to 2048 tokens with BOS=1.

Classes:
- natural:    WikiText-2 test split (Salesforce/wikitext, wikitext-2-raw-v1), tokenized as one
              stream, split into consecutive non-overlapping chunks (paper D.2 style).
- code:       Python sources of the cloned CyFA repository (cyclic_flow_attention/**.py,
              excluding eval/vendor), concatenated.
- repetitive: synthetic periodic texts (repeated sentence, periodic table rows, single-token),
              fixed seed.
Outputs: data/{natural,code,repetitive}_ids.npy int64 [N, 2048]; data/meta.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cyfa_cpu import load_tokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
SEQ = 2048
BOS = 1


def chunk(ids: np.ndarray) -> np.ndarray:
    n = (len(ids)) // (SEQ - 1)
    arr = ids[: n * (SEQ - 1)].reshape(n, SEQ - 1)
    bos = np.full((n, 1), BOS, dtype=np.int64)
    return np.concatenate([bos, arr], axis=1)


def main():
    import glob
    import os
    ckpt = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--cyxxxxxxxxxx--cyfa-400M-15B/snapshots/*")))[0]
    tok = load_tokenizer(ckpt)

    # natural
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    ids = np.array(tok(text, add_special_tokens=False).input_ids, dtype=np.int64)
    natural = chunk(ids)
    np.save(DATA / "natural_ids.npy", natural)

    # code
    repo = HERE.parent / "CyclicFlowAttention" / "cyclic_flow_attention"
    files = sorted(str(p) for p in repo.rglob("*.py"))
    code_text = "\n\n".join(Path(f).read_text() for f in files)
    cids = np.array(tok(code_text, add_special_tokens=False).input_ids, dtype=np.int64)
    code = chunk(cids)
    np.save(DATA / "code_ids.npy", code)

    # repetitive
    rng = np.random.default_rng(0)
    sent = "The quick brown fox jumps over the lazy dog. "
    rep_a = sent * 4000
    rows = []
    names = ["alpha", "beta", "gamma"]
    for i in range(6000):
        j = i % 3
        rows.append(f"Row {i:04d}: {names[j]}={rng.uniform(0, 10):.2f} status=ok\n")
    rep_b = "".join(rows)
    rep_c = "the " * 6000
    reps = []
    for t in (rep_a, rep_b, rep_c):
        r = np.array(tok(t, add_special_tokens=False).input_ids, dtype=np.int64)
        reps.append(chunk(r))
    repetitive = np.concatenate(reps, axis=0)
    np.save(DATA / "repetitive_ids.npy", repetitive)

    meta = {
        "seq_len": SEQ, "bos": BOS,
        "natural": {"source": "Salesforce/wikitext wikitext-2-raw-v1 test", "chunks": int(len(natural))},
        "code": {"source": "workspace/code/CyclicFlowAttention/cyclic_flow_attention/**.py",
                 "files": len(files), "chunks": int(len(code))},
        "repetitive": {"variants": ["sentence x4000", "periodic rows x6000", "single token"],
                       "chunks": int(len(repetitive))},
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
