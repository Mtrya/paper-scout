import json, glob, io
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import imageio_ffmpeg
from PIL import Image

W = Path("/inspire/qb-ilm/project/cq-scientific-cooperation-zone/ky26021/embodied-research/libero-triplet")
OUT = W / "frames"
OUT.mkdir(exist_ok=True)

TARGET_TASKS = [
 "put the white mug on the left plate and put the yellow and white mug on the right plate",
 "put the white mug on the plate and put the chocolate pudding to the right of the plate",
 "turn on the stove and put the moka pot on it",
 "put both the alphabet soup and the cream cheese box in the basket",
 "put the bowl on the plate",
 "open the middle drawer of the cabinet",
 "pick up the milk and place it in the basket",
 "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
 "pick up the black bowl on the ramekin and place it on the plate",
]

# --- pi/libero: frames embedded as PNG bytes in parquet ---
pi_done = 0
for p in sorted(glob.glob(str(W / "pi_eps" / "*.parquet"))):
    i = int(p.split("_")[-1].split(".")[0])
    t = pq.read_table(p, columns=["image", "frame_index"]).to_pylist()
    t.sort(key=lambda r: r["frame_index"])
    for tag, row in (("init", t[0]), ("final", t[-1])):
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        img.save(OUT / f"pi_{i:05d}_{tag}.jpg", quality=90)
    pi_done += 1
print("pi episodes extracted:", pi_done)

# --- libero_plus: frames from mp4 by timestamp ---
eps = pq.read_table(W / "lp_episodes.parquet").to_pylist()
sel = [e for e in eps if e["videos/observation.images.front/chunk_index"] == 0
       and e["videos/observation.images.front/file_index"] == 0
       and (e["tasks"][0] if isinstance(e["tasks"], list) else e["tasks"]) in TARGET_TASKS]
gen = imageio_ffmpeg.read_frames(str(W / "lp_front_000.mp4"), pix_fmt="rgb24")
meta = next(gen)
fps = meta["fps"]
del gen
need = {}
for e in sel:
    idx = e["episode_index"]
    f0 = int(e["videos/observation.images.front/from_timestamp"] * fps) + 1
    f1 = int(e["videos/observation.images.front/to_timestamp"] * fps) - 1
    need[idx] = (f0, f1)

def grab(path, indices):
    gen = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    m = next(gen); w, h = m["size"]
    want = sorted(indices); out = {}; wi = 0; target = want[0]
    for i, buf in enumerate(gen):
        if i == target:
            out[i] = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            wi += 1
            if wi >= len(want):
                break
            target = want[wi]
    return out

all_idx = sorted({f for v in need.values() for f in v})
frames = grab(W / "lp_front_000.mp4", all_idx)
inv = {f0: ("init", idx) for idx, (f0, f1) in need.items()}
inv.update({f1: ("final", idx) for idx, (f0, f1) in need.items()})
n = 0
for fi, arr in frames.items():
    tag, idx = inv[fi]
    Image.fromarray(arr).save(OUT / f"lp_{idx:05d}_{tag}.jpg", quality=90)
    n += 1
print("lp frames:", n, "for", len(need), "episodes:", sorted(need))
print("EXTRACT2_DONE")
