"""Tokenize WikiText-2 test into 2048-token segments for the STEPQuant probes."""
import glob
import os

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
HF = os.path.expanduser("~/.cache/huggingface/hub")
MODEL_DIR = sorted(glob.glob(os.path.join(
    HF, "models--m-a-p--340M-20B-GatedDeltaNet-pure-baseline/snapshots/*")))[0]
WT = sorted(glob.glob(os.path.join(
    HF, "datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/test-00000-of-00001.parquet")))[0]

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
tbl = pq.read_table(WT)
text = "\n\n".join(t for t in tbl["text"].to_pylist())
ids = np.array(tok(text, add_special_tokens=False)["input_ids"], dtype=np.int64)
print("total test tokens:", len(ids))
T = 2048
segs = [ids[i * T:(i + 1) * T] for i in range(3)]
out = np.stack(segs)
np.save(os.path.join(HERE, "wt2_test_segs.npy"), out)
print("saved segments:", out.shape)
