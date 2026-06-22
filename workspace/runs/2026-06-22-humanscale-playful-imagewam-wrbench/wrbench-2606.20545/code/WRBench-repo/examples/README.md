# WRBench Examples

Equivalent CLI commands for the three actions shown in `quickstart.py`.

---

## (a) List supported models

```bash
# Plain table (active models only)
wrbench models

# Include deferred models
wrbench models --deferred

# Full JSON output
wrbench models --json
```

---

## (b) Compile the `yaw_LR` preset for `wan22-fun-5b-cam`

Use one of the bundled Natural-25 first frames:

```bash
python - <<'PY'
from wrbench.datasets import natural25_first_frame_path
print(natural25_first_frame_path("bedroom_cat_bed_jump"))
PY
```

```bash
IMAGE="$(python - <<'PY'
from wrbench.datasets import natural25_first_frame_path
print(natural25_first_frame_path("bedroom_cat_bed_jump"))
PY
)"

wrbench generate \
  --model wan22-fun-5b-cam \
  --camera preset:yaw_LR \
  --image "$IMAGE" \
  --out /tmp/wrbench_demo/yaw_lr_demo.mp4

# With custom peak angle and frame count:
wrbench generate \
  --model wan22-fun-5b-cam \
  --camera preset:yaw_LR \
  --peak-deg 45 \
  --frames 81 \
  --image "$IMAGE" \
  --out /tmp/wrbench_demo/yaw_lr_demo.mp4
```

---

## (c) Compile an arbitrary sweep script

```bash
# Using the raw script grammar directly:
wrbench generate \
  --model wan22-fun-5b-cam \
  --camera "yaw:left:37@49" \
  --image "$IMAGE" \
  --out /tmp/wrbench_demo/sweep_demo.mp4

# Inspect / validate the script before generating:
wrbench actions --camera "yaw:left:37@49"
```

---

## Other useful commands

```bash
# List preset names with default expansion
wrbench presets

# Validate registry and adapter wiring for all models
wrbench doctor --all

# Validate a specific model
wrbench doctor --model wan22-fun-5b-cam
```
