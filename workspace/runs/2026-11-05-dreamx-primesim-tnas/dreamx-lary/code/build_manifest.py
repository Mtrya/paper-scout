#!/usr/bin/env python3
"""Build sample manifests for the LARYBench regression probe.

Intersects the official metadata CSVs (code/LARYBench/data/) with the files
actually extracted under code/lary-probe/, so a partially downloaded dataset
yields a valid manifest.

Output CSV columns: sample_id,src_img,tgt_img,action  (paths absolute)
"""
import argparse
import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(ROOT, "..", "LARYBench", "data")

DATASET_DIR = {
    ("calvin", "train"): ("calvin/train_stride5", "calvin_metadata_train.csv"),
    ("calvin", "val"): ("calvin/val_stride5", "calvin_metadata_val.csv"),
    ("vlabench", "train"): ("vlabench", "vlabench_metadata_train.csv"),
    ("vlabench", "val"): ("vlabench", "vlabench_metadata_val.csv"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["calvin", "vlabench"])
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    subdir, csv_name = DATASET_DIR[(args.dataset, args.split)]
    data_root = os.path.join(ROOT, subdir)
    df = pd.read_csv(os.path.join(META, csv_name))

    rows = []
    for _, r in df.iterrows():
        src = os.path.join(data_root, str(r["src_img"]))
        tgt = os.path.join(data_root, str(r["tgt_img"]))
        act = os.path.join(data_root, str(r["action"]))
        if os.path.exists(src) and os.path.exists(tgt) and os.path.exists(act):
            rows.append((str(r["src_img"]), src, tgt, act))
        if args.limit and len(rows) >= args.limit:
            break

    out = pd.DataFrame(rows, columns=["sample_id", "src_img", "tgt_img", "action"])
    os.makedirs(os.path.join(ROOT, "manifests"), exist_ok=True)
    out_path = os.path.join(ROOT, "manifests", f"{args.dataset}_{args.split}.csv")
    out.to_csv(out_path, index=False)
    print(f"{args.dataset}/{args.split}: {len(out)} usable samples -> {out_path}")


if __name__ == "__main__":
    main()
