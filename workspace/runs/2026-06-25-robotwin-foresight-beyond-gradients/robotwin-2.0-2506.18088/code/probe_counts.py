"""Small probe that counts concrete artifacts directly from the released code.
Run from the repo root with the assets present; counts that do not need sim.
"""
import json, os, re
from pathlib import Path

REPO = Path("/home/betelgeuse/Documents/paper-scout/workspace/code/robotwin-repo")

# 1. Tasks
task_dir = REPO / "description" / "task_instruction"
tasks = sorted([p.stem for p in task_dir.glob("*.json")])
print(f"Tasks in task_instruction/: {len(tasks)}")
print("  ", ", ".join(tasks[:10]), "...")

# 2. Object categories present in the description index
obj_dir = REPO / "description" / "objects_description"
obj_cats = sorted([p.name for p in obj_dir.iterdir() if p.is_dir()])
print(f"\nObject-description categories locally indexed: {len(obj_cats)}")
print("  first 10:", ", ".join(obj_cats[:10]))

# 3. Domain-randomization axes from the config template
sample_cfg = REPO / "task_config" / "demo_randomized.yml"
print("\nDomain-randomization block in demo_randomized.yml:")
with open(sample_cfg) as f:
    import yaml
    cfg = yaml.safe_load(f)
    for k, v in cfg["domain_randomization"].items():
        print(f"  {k}: {v}")

# 4. Embodiments
emb_cfg = REPO / "task_config" / "_embodiment_config.yml"
with open(emb_cfg) as f:
    print("\nEmbodiments:")
    for name in yaml.safe_load(f):
        print(f"  {name}")
