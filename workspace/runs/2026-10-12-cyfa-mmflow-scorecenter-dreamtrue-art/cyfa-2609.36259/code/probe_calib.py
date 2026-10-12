"""Calibration: WikiText-2 token NLL and word-level perplexity of the 400M checkpoint
under the CPU harness, compared against paper Table 2 (Wiki. ppl 25.85) and the
D.2 held-out NLL (2.6105). Non-overlapping 2048-token chunks, BOS-prepended; the paper
uses lm-eval rolling windows, so our ppl is expected slightly higher.

Outputs: results/calib.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import cyfa_cpu

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RES.mkdir(exist_ok=True)


def main():
    natural = np.load(HERE / "data" / "natural_ids.npy")
    model, ckpt = cyfa_cpu.load_model(op_impl="chunk")
    tok = cyfa_cpu.load_tokenizer(ckpt)
    torch.set_num_threads(16)

    N = 16
    ids = torch.tensor(natural[:N], dtype=torch.long)
    t0 = time.time()
    nlls, logits = cyfa_cpu.forward_batch_nll(model, ids, want_logits=True)
    dt = time.time() - t0
    lp = torch.log_softmax(logits, dim=-1)
    nll_sum = float(-lp[:, :-1].gather(2, ids[:, 1:].unsqueeze(2)).sum().item())
    n_tok = ids.numel() - ids.shape[0]

    # word count of the evaluated text (decode each chunk, count whitespace-separated words)
    texts = [tok.decode(row[1:].tolist()) for row in natural[:N]]
    n_words = sum(len(t.split()) for t in texts)

    token_nll = nll_sum / n_tok
    word_ppl = float(np.exp(nll_sum / n_words))
    out = {
        "checkpoint": ckpt,
        "n_chunks": N, "n_tokens": int(n_tok), "n_words": n_words,
        "tokens_per_word": n_tok / n_words,
        "token_nll": token_nll,
        "token_ppl": float(np.exp(token_nll)),
        "word_ppl_nonoverlap": word_ppl,
        "paper_wiki_ppl": 25.85,
        "paper_d2_heldout_nll": 2.6105,
        "time_s": dt,
    }
    (RES / "calib.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
