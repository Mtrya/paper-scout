"""Download exactly the episodes in the episode-disjoint split, from hf-mirror, with validation.

The mirror rate-limits resolver requests, so: low concurrency, retry with backoff, and every file is validated as a
readable NPZ archive with the expected keys before it counts as present.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dashdata as D  # noqa: E402

BASES = ["https://hf-mirror.com/datasets/Jack97/ForceHarness-Data/resolve/main/",
         "https://huggingface.co/datasets/Jack97/ForceHarness-Data/resolve/main/"]
KEYS = {k + ".npy" for k in ("action", "kp", "t", "eef_pos", "eef_quat", "gripper_qpos", "force_raw",
                            "meta_json", "img_agentview_jpeg", "img_agentview_off")}


SIZES = {}


def valid(p: Path, rel: str) -> bool:
    if not p.exists():
        return False
    want = SIZES.get(rel)
    if want is not None and p.stat().st_size != want:
        return False
    if p.stat().st_size < 1_000_000:
        return False
    try:
        with zipfile.ZipFile(p) as z:
            if z.testzip() is not None:
                return False
            return KEYS.issubset(set(z.namelist()))
    except Exception:
        return False


def fetch(rel: str, root: Path, tries: int = 8) -> str:
    dst = root / rel
    if valid(dst, rel):
        return "skip"
    if not rel.endswith(".npz"):
        dst.parent.mkdir(parents=True, exist_ok=True)
        for b in BASES:
            try:
                with urllib.request.urlopen(b + rel, timeout=60) as r:
                    dst.write_bytes(r.read())
                return "ok"
            except Exception as e:  # noqa: BLE001
                print(f"[aux] {rel}: {e}", flush=True)
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(".part")
    if valid(part, rel):
        part.rename(dst)
        return "ok"
    for i in range(tries):
        try:
            base = BASES[i % len(BASES)] if i % 3 else BASES[0]
            with urllib.request.urlopen(base + rel, timeout=180) as r:
                data = r.read()
            if data[:2] != b"PK":
                raise RuntimeError(f"bad body {len(data)}B: {data[:160]!r}")
            want = SIZES.get(rel)
            if want is not None and len(data) != want:
                raise RuntimeError(f"size {len(data)} != {want}")
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(data)
            if not valid(tmp, rel):
                raise RuntimeError("zip validation failed")
            tmp.rename(dst)
            return "ok"
        except Exception as e:  # noqa: BLE001
            wait = 3 * (i + 1) if i < 4 else 30
            print(f"[retry {i}] {rel}: {e} (sleep {wait}s)", flush=True)
            time.sleep(wait)
    return "FAIL"


def main() -> int:
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    n_val = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    root = Path("data")
    if (root / "sizes.json").exists():
        SIZES.update(json.loads((root / "sizes.json").read_text()))
    split = D.split_episodes(root, n_train, n_val)
    rels = []
    for t in D.TASKS:
        rels.append(f"raw/{t}/index.jsonl")
        for k in ("train", "val"):
            rels += [f"raw/{t}/{p.name}" for p in split[t][k]]
    print(f"[fetch] {len(rels)} files, workers={workers}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda r: fetch(r, root), rels))
    print(f"[fetch] ok={res.count('ok')} skip={res.count('skip')} fail={res.count('FAIL')} "
          f"in {time.time()-t0:.0f}s", flush=True)
    for r, s in zip(rels, res):
        if s == "FAIL":
            print("  FAILED:", r)
    return 0 if "FAIL" not in res else 1


if __name__ == "__main__":
    sys.exit(main())
