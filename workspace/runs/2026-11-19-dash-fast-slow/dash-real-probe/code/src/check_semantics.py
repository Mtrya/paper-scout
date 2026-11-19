"""Semantics check for the two visual channels, run against the packed val split.

Fresh-vision arm: the ResNet must read the frame at the anchor t, while the frozen VLM context reads t-Δ.  The bug this
guards against (both channels falling back to t-Δ) silently removes the whole point of the fresh pathway.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from packed import PackedSplit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", default="packed")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    ds = PackedSplit(Path(args.packed), args.split)
    anchors = ds.anchors(stride=4)[:16]
    for delta in (0, 2, 8):
        b = ds.gather(anchors, delta, need_img=True)
        img_ok = np.array_equal(b["img"], np.asarray(ds.img[anchors]))
        img_stale = np.array_equal(b["img"], np.asarray(ds.img[anchors - delta]))
        vis_ok = np.array_equal(b["vis"], np.asarray(ds.vis[anchors - delta]))
        vis_fresh = np.array_equal(b["vis"], np.asarray(ds.vis[anchors]))
        st_ok = np.array_equal(b["state"], np.asarray(ds.state[anchors]))
        print(f"delta={delta}: img@t={img_ok} img@t-d={img_stale} vis@t-d={vis_ok} vis@t={vis_fresh} state@t={st_ok}")
        assert img_ok, "fresh image is not taken at the anchor"
        assert vis_ok, "slow context is not taken at t-delta"
        assert st_ok, "state is not taken at the anchor"
        if delta > 0:
            assert not img_stale, "fresh image fell back to the stale frame"
            assert not vis_fresh, "slow context accidentally taken at t"
    b = ds.gather(anchors, 8, need_img=True, img_delta=8)
    assert np.array_equal(b["img"], np.asarray(ds.img[anchors - 8])), "stalefresh control not honoured"
    print("delta=8 stalefresh control: img@t-d=True")
    print("SEMANTICS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
