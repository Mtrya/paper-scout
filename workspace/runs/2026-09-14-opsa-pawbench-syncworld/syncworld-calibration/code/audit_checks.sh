#!/bin/bash
# SyncWorld release-vs-paper audit checks (run from the repo root).
# Each check prints PASS/FAIL against the claims recorded in
# docs/paper_differences.md. Verified 2026-09-14 on commit @HEAD.
set -u
cd "$(dirname "$0")/../../../../code/syncworld" 2>/dev/null || {
  echo "clone code/syncworld first (git clone --depth 1 https://github.com/UMass-Embodied-AGI/SyncWorld.git)"
  exit 1
}
PASS=0; FAIL=0
ck() { # ck <label> <grep-pattern> <file>
  if grep -q "$2" "$3" 2>/dev/null; then echo "PASS $1"; PASS=$((PASS+1));
  else echo "FAIL $1 (pattern '$2' not in $3)"; FAIL=$((FAIL+1)); fi
}
# 1. backbone: released checkpoint is Cosmos3 (paper used Wan2.2)
ck "backbone=Cosmos3-Nano (released)" "Cosmos3-Nano" docs/paper_differences.md
ck "backbone doc mentions Wan2.2 paper model" "Wan2.2" docs/paper_differences.md
# 2. action units: cm + degrees in released recipe
ck "action_translation_scale=100" "action_translation_scale = 100.0" docs/paper_differences.md
# 3. conditioning: joint self-attention via action tokens (MoT)
ck "action token stream (action2llm)" "action2llm" docs/paper_differences.md
# 4. calibration segments: 6 in released (12 in paper)
ck "calib_segments=6" "calib_segments = 6" docs/paper_differences.md
# 5. config schema carries the flags
ck "gripperhead config has calib_segments" "calib_segments" cosmos_framework/configs/toml_config/sft_config.py
ck "gripperhead config has action scales" "action_translation_scale" cosmos_framework/configs/toml_config/sft_config.py
# 6. eval script: default fast preset vs reference sampler params
ck "eval fast preset num-steps 20" "num-steps.*20" examples/eval_gripperhead_fdm_rollout.py
ck "eval reference params documented" "num-steps 35 --action-cfg-scale 5.0" examples/eval_gripperhead_fdm_rollout.py
# 7. calibration design: gripper opened off-camera (unrecorded warmup)
ck "unrecorded warmup" "UNRECORDED warmup" docs/visual_calibration.md
ck "randomized sweep signs" "x_sign.*rng_py.random" docs/visual_calibration.md
ck "move_range.pkl records realized dirs" "move_range.pkl" docs/visual_calibration.md
echo "---"
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
