import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from packed import pack

ap = argparse.ArgumentParser()
ap.add_argument("--ep-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--tasks", nargs="+", default=["weight_sort", "com_sort"])
ap.add_argument("--raw-root", default="data/raw")
a = ap.parse_args()
idx = pack(Path(a.ep_dir), Path(a.out), a.split, a.tasks, Path(a.raw_root))
print({k: v for k, v in idx.items() if k not in ("episodes", "scene_keys")})
