"""Compute normalisation statistics from the train split only (state / action per dim, VLM feature scale per layer)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from packed import PackedSplit, H  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=6000, help="frames sampled for the feature scale")
    ap.add_argument("--scene", action="store_true", help="also write norm_leak.json with the scene-vector dims")
    args = ap.parse_args()
    ds = PackedSplit(Path(args.packed), "train")
    anchors = ds.anchors(stride=1)
    rng = np.random.default_rng(0)

    state = np.asarray(ds.state[:])
    act = np.asarray(ds.act[:])
    scene = np.asarray(ds.scene[:]) if args.scene else None

    sub = rng.choice(anchors, size=min(args.frames, anchors.size), replace=False)
    vis = np.asarray(ds.vis[sub]).astype(np.float32)          # (n, L, n_vis, dim)
    txt = np.asarray(ds.txt[sub]).astype(np.float32)          # (n, L, dim)
    vis_scale = [float(np.std(vis[:, l])) for l in range(vis.shape[1])]
    txt_scale = [float(np.std(txt[:, l])) for l in range(txt.shape[1])]

    chunks = np.asarray(ds.act[np.sort(sub)[:, None] + np.arange(H)[None, :]])
    act_chunks = chunks.reshape(-1, 13)
    norm = dict(
        state_mean=state.mean(0).tolist(), state_std=(state.std(0) + 1e-6).tolist(),
        act_mean=act_chunks.mean(0).tolist(), act_std=(act_chunks.std(0) + 1e-6).tolist(),
        vis_scale=vis_scale, txt_scale=txt_scale,
        n_train_frames=int(ds.n), n_anchors=int(anchors.size),
        vis_scale_abs_mean=[float(np.abs(vis[:, l]).mean()) for l in range(vis.shape[1])],
    )
    Path(args.out).write_text(json.dumps(norm, indent=1))
    if scene is not None and scene.shape[1]:
        full = np.concatenate([state, scene], 1)
        leak = dict(norm)
        leak.update(state_dim=int(full.shape[1]), state_mean=full.mean(0).tolist(),
                    state_std=(full.std(0) + 1e-6).tolist(), leak=True, scene_dim=int(scene.shape[1]))
        Path(args.out).with_name("norm_leak.json").write_text(json.dumps(leak, indent=1))
        print("wrote", Path(args.out).with_name("norm_leak.json"), "state_dim", full.shape[1])
    print(json.dumps({k: v for k, v in norm.items() if not isinstance(v, list)}, indent=1))
    print("act_raw_mean", np.round(act_chunks.mean(0), 4).tolist())
    print("act_raw_std", np.round(act_chunks.std(0), 4).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
