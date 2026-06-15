# Hy-Embodied-0.5-VLA (arXiv 2606.14409) — Code Inspection Notes

Repo: `code/hyvla-0.5` (clone of `github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA`, commit `fdf0645`).
Paper: `/home/betelgeuse/Documents/paper-scout/workspace/papers/vla/hy-embodied-0.5-vla-2606.14409.md`.
Copied artifacts: `code/` subdirectory in this run directory.

---

## (a) Repo structure

```
hy_vla/
  __init__.py                     # Public API: HyVLA, HyVLAConfig, HyDualTower
  configuration_hy_vla.py         # HyVLAConfig dataclass
  modeling_hy_vla.py              # HyVLA + HyVLAFlowMatching
  modeling_dual_tower.py          # HyDualTower (VLM + action-expert shared attention)
  space_time_attention.py         # MEM / compact memory encoder patch
  train.py                        # Hydra + accelerate training entry point
  data/
    vla_dataset.py                # VLADataset / VLADataCollator
    lance_dataset.py              # LanceTableReader + LanceVLADataset (UMI pretraining)
    hdf5_dataset.py               # HDF5VLADataset (RoboTwin SFT)
  config/
    base.yaml                     # Hydra root config
    dataset/umi_lance.yaml        # Pretraining data config (K=1)
    dataset/robotwin_hdf5.yaml    # SFT data config (K=6, MEM on, relabs)
    zero2.json                    # DeepSpeed ZeRO-2
  hunyuan_vl_mot/                 # Vendor fallback for upstream MoT VLM
scripts/
  quick_start.py                  # HF smoke test
  train_umi_vlm.sh                # 64-GPU UMI pretraining launch
  train_robotwin_umi.sh           # RoboTwin SFT launch
  train_table_vlm.sh              # Single-table fast iteration
  eval_robotwin_test.sh / full.sh
  compute_norm_lance.py / hdf5.py
  vis_umi_episode.py
robotwin_eval/                    # RoboTwin policy adapter
  deploy_policy.py
  policy_wrapper.py
  transforms.py
  deploy_policy.yml
```

---

## (b) Where the key components live

### 1. Hy-Embodied-0.5-MoT backbone

* The canonical classes live in the upstream `transformers` fork (`model_type=hunyuan_vl_mot`).
* The repo carries a verbatim vendor fallback under `hy_vla/hunyuan_vl_mot/`, registered at import time into the HuggingFace `Auto*` registries (`_register_hunyuan_vl_mot`, `hy_vla/hunyuan_vl_mot/__init__.py:58`).
* `HyVLA` resolves tokenizer + VLM config in `modeling_hy_vla.py:98-276`; self-contained checkpoints embed `vlm_config_dict` so they load without contacting the upstream VLM repo.
* `HyVLAFlowMatching.__init__` builds the actual VLM instance:
  ```python
  # modeling_hy_vla.py:930-951
  vlm_inner_config = _load_vlm_autoconfig(self.config)
  expert_inner_config = _copy.deepcopy(vlm_inner_config)
  expert_inner_config.hidden_size = self.config.proj_width
  expert_inner_config.intermediate_size = 2048
  dual_tower_config = HyDualTowerConfig(
      vlm_config=vlm_inner_config,
      expert_config=expert_inner_config,
      ...
  )
  self.dual_tower = HyDualTower(dual_tower_config)
  ```

### 2. Dual-tower / action expert

Implemented in `hy_vla/modeling_dual_tower.py`.

* `HyDualTower` owns `self.vlm` and `self.expert` (`modeling_dual_tower.py:201-209`).
* The expert is `_HunYuanVLMoTTextForCausalLM` with `hidden_size=1024` and `intermediate_size=2048`, ~370M params as claimed.
* Shared-attention forward: VLM + action-expert sequences are concatenated, Q/K/V are produced through modality-aware projections (`mask_apply` routes text vs. vision params), RoPE is applied jointly, attention is computed once, then outputs are split back to each tower (`modeling_dual_tower.py:373-560`).
* Training toggles: `freeze_vision_encoder=True`, `train_expert_only=True` by default; full-model training is supported via config.

### 3. Flow-matching action head

Implemented in `hy_vla/modeling_hy_vla.py` inside `HyVLAFlowMatching`.

* Timestep sampled from Beta(1.5, 1.0), shifted to `[0.001, 0.999]` (`modeling_hy_vla.py:977-980`):
  ```python
  def sample_time(self, bsize, device):
      time_beta = sample_beta(1.5, 1.0, bsize, device)
      time = time_beta * 0.999 + 0.001
      return time.to(dtype=torch.float32, device=device)
  ```
* Linear interpolant and velocity target (`modeling_hy_vla.py:1271-1273`):
  ```python
  x_t = time_expanded * noise + (1 - time_expanded) * actions
  u_t = noise - actions
  ```
* Suffix embedding fuses state, noisy actions, and sinusoidal timestep through an MLP (`modeling_hy_vla.py:1110-1178`).
* Training loss: MSE between `u_t` and predicted velocity (`modeling_hy_vla.py:1308`).
* Inference: 10 Euler steps from `τ=1` to `0` (`modeling_hy_vla.py:1372-1389`):
  ```python
  dt = -1.0 / self.config.num_steps
  while time >= -dt / 2:
      v_t, att_vis_output = self.denoise_step(...)
      x_t += dt * v_t
      time += dt
  ```
* KV-cache: prefix (image + language) is cached in `sample_actions` (`fill_kv_cache=True`), then each denoising step reuses it and only recomputes the action suffix (`denoise_step`, `modeling_hy_vla.py:1407-1447`).

### 4. Compact memory encoder (MEM)

Implemented in `hy_vla/space_time_attention.py`.

* `SpaceTimeBlock` wraps a standard ViT block by **reference**, adopting its submodules so `state_dict` keys are preserved and no new parameters are added (`space_time_attention.py:60-119`).
* Time embedding is a fixed sinusoid with `e(0)=0`, rebuilt on-device every forward; optional `learnable_time_embed` exists but defaults to `False` (`space_time_attention.py:121-134`).
* Factorized causal temporal attention over frames followed by spatial attention over patches, reusing the same QKV / `W_O` projections (`space_time_attention.py:148-205`).
* Patched at runtime via `apply_video_encoder_patch`:
  * wraps every `spacetime_layer_stride` block (default 4);
  * optionally drops past-frame tokens after `past_drop_layer`;
  * accepts 5-D `(B, K, C, H, W)` tensors and returns current-frame tokens only.
* Enabled by `HyVLAConfig.use_video_encoder` and `policy.enable_video_encoder_if_needed()` (`modeling_hy_vla.py:626-653`).

### 5. Delta-chunk action representation

* State and action normalization: `HyVLA.prepare_state` / `prepare_action` pad to `max_state_dim` / `max_action_dim` (`modeling_hy_vla.py:883-891`).
* Raw dual-arm action space is 20-d per frame: `xyz(3) + rot6d(6) + gripper(1)` per arm.
* Relative (RT-relative) chunk is computed in `hy_vla/utils/transform_utils.py:71-101`:
  ```python
  delta_T_all = T0_inv @ Ti_all
  rotation_6d = delta_T_all[:, :2, :3].reshape(N, -1)
  translation_3d = delta_T_all[:, :3, 3]
  ```
* RoboTwin wrapper decodes back to 16-d dual-arm PosQuat via `relative_to_dual_arm_poses` (`robotwin_eval/transforms.py:185-207`).
* Optional `relative_chunk_ee_RT_with_absolute` doubles the action chunk in time: first half RT-relative, second half absolute PosRotMat; decoded with blend modes `rel_only | abs_only | rel_abs` (`robotwin_eval/policy_wrapper.py:246-283`).

### 6. FlowPRO RL post-training

**Not present in this codebase.** The README explicitly marks FlowPRO as “*(under review)*” and the paper Sec. 4 says “code coming soon.” A repo-wide grep for `FlowPRO`, `RPRO`, `preference`, `intervention`, `rollback` returns only the README/paper text, not implementation code. This is the largest gap between the paper and the released code.

---

## (c) Runnable entry points

### Inference / smoke test

```bash
python scripts/quick_start.py
```

Loads `tencent/Hy-VLA-RoboTwin`, runs one forward pass on dummy `(B=1, K=6, 3, 224, 224)` images + zero state + text task.

### Training

Pretraining on UMI Lance data (needs `CHIEF_IP`, `INDEX`, `NORM_PATH`):

```bash
export CHIEF_IP=<ip> INDEX=0
bash scripts/train_umi_vlm.sh
```

RoboTwin SFT from the UMI checkpoint:

```bash
export CHIEF_IP=<ip> INDEX=0
bash scripts/train_robotwin_umi.sh
```

Hydra command (single-GPU or `accelerate launch`):

```bash
python hy_vla/train.py \
  model.pretrain_source=vla \
  model.vla_model_path=tencent/Hy-VLA-UMI \
  dataset=robotwin_hdf5 \
  dataset.hdf5_dir=/path/to/robotwin/hdf5 \
  dataset.mean_std_path=/path/to/norm_stats.pkl \
  training.batch_size=8
```

### Evaluation

```bash
export ROBOTWIN_DIR=/path/to/RoboTwin
export CKPT_PATH=tencent/Hy-VLA-RoboTwin
bash scripts/eval_robotwin_test.sh       # 6 tasks x 10 rollouts
bash scripts/eval_robotwin_full.sh       # 50 tasks x 100 rollouts
```

### Data / norm prep

```bash
python scripts/compute_norm_lance.py \
  --lance-source tencent/Hy-Embodied-0.5-VLA-Data \
  --output norm_stats.pkl

python scripts/vis_umi_episode.py -t table_000 -e 666
```

### Released artifacts (HF Hub)

| Artifact | HF repo |
|---|---|
| UMI pretrained VLA | `tencent/Hy-Embodied-0.5-VLA-UMI` |
| RoboTwin post-train | `tencent/Hy-Embodied-0.5-VLA-RoboTwin` |
| Dataset (2K+ h) | `tencent/Hy-Embodied-0.5-VLA-Data` |
| Upstream VLM | `tencent/HY-Embodied-0.5` |

---

## (d) Notable implementation details, discrepancies, blockers

1. **FlowPRO is not shipped.** The paper’s Sec. 4 (RL post-training) is not implemented in this release. The README caveats it as “under review,” so the reported real-robot FlowPRO results cannot be reproduced from this repo alone.

2. **VLM fork dependency.** The model requires a pinned `transformers` fork (`git+https://github.com/huggingface/transformers@9293856...`) because `model_type=hunyuan_vl_mot` is not in upstream `transformers`. A vendor fallback exists, but the intended path is the fork.

3. **Flash Attention hard dependency for MEM.** `space_time_attention.py` imports `flash_attn_func` / `flash_attn_varlen_func` at module load. Without flash-attn installed the MEM path cannot be imported; the single-frame path in `modeling_hy_vla` would still work if flash-attn is unavailable, but training scripts import `space_time_attention` indirectly.

4. **Visual-segment attention mask has two modes.** `HyVLAConfig.visual_segment_isolation`:
   * `False` (default, patch-only): cross-image patch visibility is cleared, split rows stay causal.
   * `True` (required to reproduce the released RoboTwin checkpoint): full image segment is bidirectional and isolated from other segments (`modeling_hy_vla.py:1180-1250`). The paper figure implies full-segment isolation; the code exposes it as a flag.

5. **Mixed-precision policy is opinionated.** The model casts VLM + layer params to `bfloat16` but keeps embeddings/projections in `float32` (`modeling_dual_tower.py:318-335`; `to_bfloat16_like_physical_intelligence`).

6. **Action chunk layout for relabs checkpoints.** When `act_type=relative_chunk_ee_RT_with_absolute`, the model outputs `2 * chunk_size` rows (rel then abs) and the dataset concatenates mean/std accordingly (`hdf5_dataset.py:267-269`). The RoboTwin wrapper decodes them with optional slerp blending.

7. **State / action padding to 32 dims.** The network always sees `max_state_dim=32` and `max_action_dim=32`; the extra 12 dims per tensor are zero-padded and ignored on output (`modeling_hy_vla.py:954-959`, `prepare_state`/`prepare_action`).

8. **No built-in async Bézier smoother in the released code.** The paper Sec. 5.2–5.3 describes an asynchronous producer-consumer execution loop with cubic-Bézier chunk stitching. The released repo only contains the training policy and a synchronous RoboTwin wrapper (`policy_wrapper.get_action` caches the chunk and serves it sequentially). The deployment smoother is **not present** in this codebase.

9. **Real-robot platform mapper (Track A/B) not present.** The paper Sec. 5.1 describes fixed-base / humanoid platform mapping; only the RoboTwin simulation wrapper is included.

10. **Co-training / auxiliary NTP objective is wired.** `HyVLAFlowMatching.forward` supports `lang_token_labels` and computes a cross-entropy NTP loss on the VLM prefix (`modeling_hy_vla.py:1311-1338`), but the default configs set it to `λ_ntp = 0` (action-only). The paper mentions optional co-training tasks `D_ct`.

11. **Deterministic dataloader required for the iter-based loop.** `train.py` relies on `dataset.deterministic=True` and `len(dataloader)` being finite; `umi_lance.yaml` has `deterministic=False`, which is fine for pretraining because `max_training_steps` is the stopping criterion, but the logged “epoch” becomes a non-epoch cycle.

12. **Dataset note.** The public dataset is ~2K hours (one-fifth of the full 10K-hour corpus used in the paper), split into Lance tables compatible with LeRobot v3.0.

---

## Bottom line

The released code faithfully implements the model architecture described in the paper: MoT VLM backbone, 370M-parameter flow-matching action expert, parameter-free MEM video encoder, delta-chunk action representation, and KV-cached flow sampling. The UMI pretraining and RoboTwin SFT pipelines are runnable end-to-end against public checkpoints and data. However, the two deployment/last-mile components most emphasized in the system narrative — **FlowPRO post-training** and the **asynchronous Bézier deployment stack** — are **not included** in this release, and neither is the real-robot platform mapper. Treat the repo as a strong research training+inference baseline, not a complete real-world deployment stack.
