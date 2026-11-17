#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def native(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with h5py.File(args.input, "r") as h5:
        data = h5["data"]
        demo_names = sorted(data.keys())
        demos = {}
        first_datasets = {}
        for name in demo_names:
            demo = data[name]
            datasets = {}
            demo.visititems(lambda key, item: datasets.update({
                key: {"shape": list(item.shape), "dtype": str(item.dtype)}
            }) if isinstance(item, h5py.Dataset) else None)
            if name == demo_names[0]:
                first_datasets = datasets
            demos[name] = {
                "attrs": {key: native(value) for key, value in demo.attrs.items()},
                "dataset_first_dims": {key: value["shape"][0] for key, value in datasets.items()},
            }
        result = {
            "file": str(args.input),
            "bytes": args.input.stat().st_size,
            "data_attrs": {key: native(value) for key, value in data.attrs.items()},
            "demo_count": len(demo_names),
            "demo_names": demo_names,
            "first_demo_datasets": first_datasets,
            "demos": demos,
        }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "demo_count": result["demo_count"],
        "data_attrs": result["data_attrs"],
        "output": str(args.output),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
