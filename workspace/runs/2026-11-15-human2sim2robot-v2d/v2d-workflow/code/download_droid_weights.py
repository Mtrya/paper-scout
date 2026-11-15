# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download the DROID-SLAM checkpoint from a small public mirror."""

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

MIRROR = "https://raw.githubusercontent.com/youmi-zym/GO-SLAM/main/pretrained/droid.pth"


def download_droid(output_dir):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "droid.pth"
    if not checkpoint.exists():
        temporary = checkpoint.with_suffix(".download")
        request = urllib.request.Request(MIRROR, headers={"User-Agent": "v2d-weight-download"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
            while chunk := response.read(4 * 1024 * 1024):
                stream.write(chunk)
        os.replace(temporary, checkpoint)
    if checkpoint.stat().st_size < 1_000_000:
        raise RuntimeError(f"DROID-SLAM checkpoint is unexpectedly small: {checkpoint}")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "source": MIRROR,
        "source_kind": "third-party mirror of the DROID-SLAM checkpoint",
        "path": checkpoint.name,
        "bytes": checkpoint.stat().st_size,
        "sha256": digest,
    }
    (destination / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    download_droid(args.output_dir)
