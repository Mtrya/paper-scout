"""Data/feature sanity checks: decode equivalence, action/state statistics, feature drift, image samples."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dashdata as D  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--feats", default="feats")
    ap.add_argument("--out", default="out/inspect")
    ap.add_argument("--n-eps", type=int, default=2)
    ap.add_argument("--save-frames", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split = D.split_episodes(Path(args.data_root), 32, 12)
    rep = {}
    # 1) decode equivalence + action stats on a few episodes
    for task in D.TASKS:
        stats = []
        for ep in split[task]["train"][:args.n_eps]:
            d = D.load_episode(ep)
            T = d["T"]
            img = D.decode_frame(d, "agentview", T // 2)
            stats.append(dict(file=ep.name, T=T, img_shape=list(img.shape), img_mean=float(img.mean()),
                              act_min=d["actions13"].min(0).tolist(), act_max=d["actions13"].max(0).tolist(),
                              kp_unique=int(np.unique(d["kp"]).size),
                              state0=d["proprio"][0].round(4).tolist(),
                              scene=[o.get("name") for o in d["meta"].get("scene0", [])]))
            if args.save_frames:
                Image.fromarray(img).save(out / f"{task}_{ep.stem}_mid.png")
                Image.fromarray(D.decode_frame(d, "robot0_eye_in_hand", T // 2)).save(
                    out / f"{task}_{ep.stem}_wrist_mid.png")
        rep[task] = stats
    # 2) feature check
    fdir = Path(args.feats) / split["weight_sort"]["train"][0].parent.name
    f = sorted(fdir.glob("*.npz"))[0]
    with np.load(f) as z:
        vis, txt = z["vis"], z["txt"]
        rep["features"] = dict(file=str(f), vis_shape=list(vis.shape), vis_dtype=str(vis.dtype),
                               n_vis=int(z["n_vis"]),
                               vis_abs_mean=[float(np.abs(vis[:, l].astype(np.float32)).mean()) for l in range(vis.shape[1])],
                               txt_abs_mean=[float(np.abs(txt[:, l].astype(np.float32)).mean()) for l in range(txt.shape[1])],
                               txt_std_across_frames=[float(txt[:, l].astype(np.float32).std(0).mean()) for l in range(txt.shape[1])],
                               vis_std_across_frames=[float(vis[:, l].astype(np.float32).std(0).mean()) for l in range(vis.shape[1])])
        v = vis.astype(np.float32)
        for dlt in (1, 2, 4, 8):
            a, b = v[dlt:], v[:-dlt]
            cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)
            rep["features"][f"vis_cos_d{dlt}"] = float(cos.mean())
        rep["features"]["txt_cos_d8"] = float(
            (txt[8:].astype(np.float32) * txt[:-8].astype(np.float32)).sum(-1).mean()
            / (np.linalg.norm(txt[8:].astype(np.float32), axis=-1).mean() * np.linalg.norm(txt[:-8].astype(np.float32), axis=-1).mean()))
    (out / "inspect.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
