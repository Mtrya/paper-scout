# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download only the HaMeR checkpoint and configs needed by V2D."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPOSITORY = "Edson125/mesh_hamer"
FILES = (
    "_DATA/hamer_ckpts/checkpoints/hamer.ckpt",
    "_DATA/hamer_ckpts/dataset_config.yaml",
    "_DATA/hamer_ckpts/model_config.yaml",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_download(weights_dir):
    destination = Path(weights_dir)
    destination.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    revision = HfApi(endpoint=endpoint).model_info(REPOSITORY).sha
    records = []
    for filename in FILES:
        local_path = Path(hf_hub_download(
            repo_id=REPOSITORY,
            filename=filename,
            revision=revision,
            endpoint=endpoint,
            local_dir=destination,
        ))
        records.append({
            "path": str(local_path.relative_to(destination)),
            "bytes": local_path.stat().st_size,
            "sha256": sha256(local_path),
        })
        print(json.dumps(records[-1]), flush=True)
    checkpoint = destination / FILES[0]
    if checkpoint.stat().st_size < 1_000_000_000:
        raise RuntimeError(f"HaMeR checkpoint is unexpectedly small: {checkpoint}")
    manifest = {
        "repository": REPOSITORY,
        "revision": revision,
        "endpoint": endpoint,
        "source_kind": "third-party mirror of the HaMeR checkpoint layout",
        "files": records,
    }
    (destination / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    cache = destination / ".cache"
    if cache.exists():
        import shutil
        shutil.rmtree(cache)
    print(f"HaMeR runtime weights ready: {checkpoint}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights_dir", required=True)
    args = parser.parse_args()
    run_download(args.weights_dir)
