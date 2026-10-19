"""Run the official WFLU probing.py on a tiny subset for fidelity checking.

The only patch is pointing the qwen25 registry at the local checkpoint so the
script does not re-download weights.  Everything else (generate + prefill hook,
pair construction, metrics) is the released code path.
"""
import os
import sys

WS = os.environ.get(
    "PAPER_SCOUT_WS",
    os.path.dirname(os.path.abspath(__file__)) + "/../../../..",
)
REPO = os.path.join(WS, "code", "contrastive-probing")
sys.path.insert(0, REPO)
import probing  # noqa: E402

spec = probing.MODEL_REGISTRY["qwen25"]
spec.checkpoints["vanilla"] = os.path.join(
    WS, "code", "models", "Qwen2.5-VL-3B-Instruct")
spec.base_processor_id = spec.checkpoints["vanilla"]

out = os.path.join(WS, "code", "probe_out", "official_smoke")
os.makedirs(out, exist_ok=True)
sys.argv = [
    "probing.py",
    "--model_type", "qwen25",
    "--scales", "vanilla",
    "--data_path", os.path.join(WS, "code", "data", "EmbSpatial-Bench.tsv"),
    "--max-samples-per-category", "2",
    "--output_dir", out,
]
probing.main()
