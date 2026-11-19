"""Pack per-episode feature files into flat memmaps and serve training/eval batches from them."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

H = 8
DELTA_MAX = 8


class PackedSplit:
    """Flat per-frame arrays for one split, addressed by frame index.

    Frames of an episode are contiguous, so a chunk at anchor i is simply rows i..i+H-1 and a stale lookup is row i-Δ.
    """

    def __init__(self, root: Path, split: str, mmap: bool = True):
        self.root = Path(root)
        self.split = split
        idx = json.loads((self.root / f"index_{split}.json").read_text())
        self.episodes = idx["episodes"]
        self.n = idx["n_frames"]
        self.n_vis = idx["n_vis"]
        self.n_layers = idx["n_layers"]
        self.dim = idx["dim"]
        mode = "r" if mmap else None
        self.vis = np.load(self.root / f"vis_{split}.mm", mmap_mode=mode)
        self.txt = np.load(self.root / f"txt_{split}.mm", mmap_mode=mode)
        self.img = np.load(self.root / f"img_{split}.mm", mmap_mode=mode)
        self.state = np.load(self.root / f"state_{split}.mm", mmap_mode=mode)
        self.scene = np.load(self.root / f"scene_{split}.mm", mmap_mode=mode)
        self.act = np.load(self.root / f"act_{split}.mm", mmap_mode=mode)

    def to_ram(self):
        """Copy the memmaps into RAM (vis is ~0.5 MB per frame; 42k frames is ~23 GB, fine on a 1 TB node)."""
        for k in ("vis", "txt", "img", "state", "act", "scene"):
            a = np.array(getattr(self, k), copy=True)   # np.asarray would keep the memmap itself
            setattr(self, k, a)
        return self

    def anchors(self, stride: int = 1, horizon: int = H, delta_max: int = DELTA_MAX) -> np.ndarray:
        out = []
        for ep in self.episodes:
            s, T = ep["start"], ep["length"]
            ts = np.arange(delta_max, T - horizon, stride, dtype=np.int64)
            out.append(ts + s)
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    def episode_of(self, idx: np.ndarray) -> np.ndarray:
        starts = np.array([e["start"] for e in self.episodes])
        return np.searchsorted(starts, idx, side="right") - 1

    def gather(self, idx: np.ndarray, delta, horizon: int = H, need_img: bool = True,
               img_delta=None):
        """Return slow context at ``idx - delta`` and fresh images at ``idx`` by default.

        ``delta`` may be a scalar or one integer per anchor. ``img_delta`` is
        reserved for the staleness control, not for the fresh-vision input.
        """
        idx = np.asarray(idx, np.int64)
        delta = np.broadcast_to(np.asarray(delta, np.int64), idx.shape)
        vis = np.asarray(self.vis[idx - delta])
        txt = np.asarray(self.txt[idx - delta])
        state = np.asarray(self.state[idx])
        chunks = np.asarray(self.act[idx[:, None] + np.arange(horizon)[None, :]])
        img_delta = np.zeros_like(idx) if img_delta is None else np.broadcast_to(np.asarray(img_delta, np.int64), idx.shape)
        img = np.asarray(self.img[idx - img_delta]) if need_img else None
        scene = np.asarray(self.scene[idx]) if self.scene.shape[1] else None
        return dict(vis=vis, txt=txt, state=state, act=chunks, img=img, scene=scene)


def scene_vector(scene0, keys=None):
    objs = {o["name"]: o for o in (scene0 or []) if isinstance(o, dict) and "name" in o and "pos" in o}
    if keys is None:
        keys = sorted(objs)
    out = []
    for k in keys:
        o = objs.get(k)
        p = list(o["pos"])[:3] if o is not None else [0.0, 0.0, 0.0]
        p += [0.0] * (3 - len(p))
        yaw = float(o.get("yaw_deg", 0.0)) if o is not None else 0.0
        out += [*p, float(np.sin(np.radians(yaw))), float(np.cos(np.radians(yaw)))]
    return np.asarray(out, np.float32), keys


def pack(ep_dir: Path, out_dir: Path, split: str, tasks, raw_root: Path | None = None) -> dict:
    """Concatenate per-episode npz files into memmaps.  Deterministic and rerunnable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    split_json = json.loads((ep_dir / f"split_{split}.json").read_text())
    files, episodes, n = [], [], 0
    for task in tasks:
        for name in split_json[task]:
            p = ep_dir / task / (name if name.endswith(".npz") else name + ".npz")
            if not p.exists():
                print(f"[pack] missing {p}, skipped", flush=True)
                continue
            with np.load(p) as z:
                T = int(z["T"])
            episodes.append(dict(task=task, seed=int(p.stem.split("_")[1]), start=n, length=T, file=p.name))
            files.append(p)
            n += T
    if not files:
        raise SystemExit("no episode files to pack")
    with np.load(files[0]) as z0:
        n_layers, n_vis, dim = z0["vis"].shape[1:]
        state_dim = z0["state"].shape[1]
    # scene vector: ground-truth object poses per episode (leak-control arm only)
    scene = np.zeros((n, 0), np.float32)
    scene_keys, off = {}, 0
    svs = []
    for task in tasks:
        keys = None
        for ep in [e for e in episodes if e["task"] == task]:
            with np.load((raw_root or Path("data/raw")) / task / f"episode_{ep['seed']:05d}.npz") as z:
                meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
            sv, k = scene_vector(meta.get("scene0"), keys)
            keys = keys or k
            svs.append(sv)
        scene_keys[task] = keys
    scene_dim = max(v.size for v in svs) if svs else 0
    scene = np.zeros((n, scene_dim), np.float32)
    for i, v in enumerate(svs):
        scene[i] = 0  # filled below per episode
    if not files:
        raise SystemExit("no episode files to pack")
    with np.load(files[0]) as z0:
        pass
    vis = np.lib.format.open_memmap(out_dir / f"vis_{split}.mm", mode="w+", dtype=np.float16,
                                    shape=(n, n_layers, n_vis, dim))
    txt = np.lib.format.open_memmap(out_dir / f"txt_{split}.mm", mode="w+", dtype=np.float16,
                                    shape=(n, n_layers, dim))
    img = np.lib.format.open_memmap(out_dir / f"img_{split}.mm", mode="w+", dtype=np.uint8,
                                    shape=(n, 2, 128, 128, 3))
    state = np.lib.format.open_memmap(out_dir / f"state_{split}.mm", mode="w+", dtype=np.float32,
                                      shape=(n, state_dim))
    act = np.lib.format.open_memmap(out_dir / f"act_{split}.mm", mode="w+", dtype=np.float32, shape=(n, 13))
    scene_mm = np.lib.format.open_memmap(out_dir / f"scene_{split}.mm", mode="w+", dtype=np.float32,
                                         shape=(n, scene_dim))
    off = 0
    for p in files:
        with np.load(p) as z:
            T = int(z["T"])
            task = str(z["task"])
            with np.load((raw_root or Path("data/raw")) / task / f"episode_{int(z['seed']):05d}.npz") as zr:
                meta = json.loads(bytes(zr["meta_json"]).decode("utf-8"))
            sv, _ = scene_vector(meta.get("scene0"), scene_keys[task])
            vis[off:off + T] = z["vis"]
            txt[off:off + T] = z["txt"]
            img[off:off + T] = z["img"]
            state[off:off + T] = z["state"]
            act[off:off + T] = z["act"]
            scene_mm[off:off + T, :sv.size] = sv
            off += T
    for m in (vis, txt, img, state, act, scene_mm):
        m.flush()
    idx = dict(split=split, n_frames=int(n), n_vis=int(n_vis), n_layers=int(n_layers), dim=int(dim),
               state_dim=int(state_dim), scene_dim=int(scene_dim), scene_keys=scene_keys, episodes=episodes)
    (out_dir / f"index_{split}.json").write_text(json.dumps(idx, indent=1))
    return idx
