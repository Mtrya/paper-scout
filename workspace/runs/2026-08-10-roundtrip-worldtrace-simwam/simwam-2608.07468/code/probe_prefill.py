"""SimWAM 推理探针:视频塔 prefill 截断敏感性 + prefill/去噪延迟分解。

问题:论文与 README 声称"视频生成分支在推理时被移除",代码审计发现
30 层视频 DiT 的 prefill 仍在推理路径上(infer_action → mot.prefill_video_cache)。
本探针回答两个后续问题:

1. 截断敏感性:action 输出对视频塔深层处理的依赖有多强?
   做法:prefill 只跑前 k 层视频 block,第 k 层之后冻结视频 token 流,
   层 j>=k 的 K/V 用 block j 的投影作用于冻结的 depth-k token 计算
   (即"每层 video block 看到停止演化于 depth k 的视频流")。
   k=30 即原始模型。比较同一初始噪声(seed 固定)下 action 轨迹的偏移。

2. 延迟分解:一次性视频 prefill vs 20 步 action DiT 去噪循环各占多少,
   核验论文图 1 "低延迟"的账。

输入绕过 NAVSIM 数据管线:infer_action 直接吃原始图像张量 [1,3,H,W]([-1,1]),
proprio=[vx,vy,ax,ay,cmd_onehot(4)],prompt 用数据配置 use_dynamic_prompt=False
的静态模板(见 navsim_dataset.build_prompt_fixed)。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

# navsim_dataset.build_prompt_fixed(use_dynamic_prompt=False) 的静态模板原文
STATIC_PROMPT = (
    "A high-quality, photorealistic dashboard camera view of autonomous driving. "
    "Based on the past 2 seconds videos, "
    "predict and generate the next 4 seconds of realistic driving continuation, "
    "Maintain temporal consistency, stable camera perspective, natural motion flow without jitter or artifacts, "
    "clear details, and realistic physics. "
)

# navsim_dataset.norm_odo 的反变换(absolute trajectory_mode)
def denorm_odo(traj: np.ndarray) -> np.ndarray:
    x = (traj[..., 0:1] + 1) / 2 * 66.74 - 1.57
    y = (traj[..., 1:2] + 1) / 2 * 42 - 19.68
    heading = (traj[..., 2:3] + 1) / 2 * 3.53 - 1.67
    return np.concatenate([x, y, heading], axis=-1)


def load_image_tensor(path: Path, size=(384, 672)) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size[1], size[0]), Image.BICUBIC)
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 127.5 - 1.0
    return x.unsqueeze(0)  # [1,3,H,W]


def build_model(cfg_path: Path):
    from hydra.utils import instantiate

    cfg = OmegaConf.load(cfg_path)
    cfg.proprio_dim = 8
    cfg.load_text_encoder = True
    cfg.skip_dit_load_from_pretrain = True  # DiT 权重由 SimWAM.pt 覆盖,省下基座下载
    cfg.action_dit_pretrained_path = None
    cfg.mot_checkpoint_mixed_attn = False
    # yaml 里的 ${model.mot_checkpoint_mixed_attn} 插值在单独加载时无法解析,直接覆写
    cfg.video_dit_config.use_gradient_checkpointing = False
    cfg.action_dit_config.use_gradient_checkpointing = False
    model = instantiate(cfg)
    return model


def load_checkpoint(model, ckpt_path: Path):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    info = {"payload_keys": sorted(payload.keys())}
    if "mot" in payload:
        inc = model.mot.load_state_dict(payload["mot"], strict=False)
        info["mot_missing"] = list(inc.missing_keys)
        info["mot_unexpected"] = list(inc.unexpected_keys)
    if "proprio_encoder" in payload and getattr(model, "proprio_encoder", None) is not None:
        model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    info["step"] = payload.get("step")
    return info


def make_truncated_prefill(mot, k: int, timing: dict):
    """prefill 只推进前 k 层视频 block;层 j>=k 的 K/V 由 block j 投影冻结的 depth-k token。"""

    def prefill(video_tokens, video_freqs, video_t_mod, video_context_payload, video_attention_mask):
        t0 = time.perf_counter()
        expert = mot.mixtures["video"]
        x = video_tokens
        kv_cache = []
        for layer_idx in range(mot.num_layers):
            block = expert.blocks[layer_idx]
            (q, kk, vv, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, _use_gc) = (
                mot._build_expert_attention_io(
                    expert=expert, block=block, x=x, freqs=video_freqs, t_mod=video_t_mod
                )
            )
            if layer_idx < k:
                mixed = mot._mixed_attention(
                    q_cat=q, k_cat=kk, v_cat=vv, attention_mask=video_attention_mask
                )
                x = mot._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=residual_x,
                    gate_msa=gate_msa,
                    shift_mlp=shift_mlp,
                    scale_mlp=scale_mlp,
                    gate_mlp=gate_mlp,
                    use_gradient_checkpointing=False,
                    mixed_slice=mixed,
                    context_payload=video_context_payload,
                )
            kv_cache.append({"k": kk, "v": vv})
        torch.cuda.synchronize()
        timing["prefill_s"] = time.perf_counter() - t0
        return kv_cache

    return prefill


def run_scenario(model, image, proprio, k_list, steps, seed):
    """对一个场景扫 k,返回 {k: traj_normalized} 与逐 k 计时。"""
    mot = model.mot
    orig_noise_pred = model._predict_action_noise_with_cache
    trajs, timing = {}, {}
    for k in k_list:
        tdict = {"denoise_s": 0.0}
        mot.prefill_video_cache = make_truncated_prefill(mot, k, tdict)

        def timed_noise_pred(*args, **kw):
            t0 = time.perf_counter()
            out = orig_noise_pred(*args, **kw)
            torch.cuda.synchronize()
            tdict["denoise_s"] += time.perf_counter() - t0
            return out

        model._predict_action_noise_with_cache = timed_noise_pred
        t0 = time.perf_counter()
        out = model.infer_action(
            prompt=STATIC_PROMPT,
            input_image=image,
            action_horizon=8,
            proprio=proprio,
            negative_prompt="",
            text_cfg_scale=1.0,
            num_inference_steps=steps,
            seed=seed,
            rand_device="cpu",
            tiled=False,
        )
        torch.cuda.synchronize()
        tdict["total_s"] = time.perf_counter() - t0
        model._predict_action_noise_with_cache = orig_noise_pred
        trajs[k] = out["action"].numpy()  # [8,3] normalized
        timing[k] = tdict
    return trajs, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, default=Path("configs/model/simwam_navsim.yaml"))
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--scenes", type=Path, default=Path("probe_assets"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k-list", type=str, default="0,3,6,10,15,20,25,30")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    k_list = [int(x) for x in args.k_list.split(",")]
    commands = {
        "straight": [0.0, 1.0, 0.0, 0.0],
        "left": [1.0, 0.0, 0.0, 0.0],
        "right": [0.0, 0.0, 1.0, 0.0],
    }
    scene_files = sorted(args.scenes.glob("scene*.png"))

    model = build_model(args.cfg)
    ckpt_info = load_checkpoint(model, args.ckpt)
    print("checkpoint:", json.dumps(ckpt_info, default=str)[:500])
    model.eval()

    # warmup(k=30 全流程一遍,丢弃)
    img0 = load_image_tensor(scene_files[0])
    proprio0 = torch.tensor([8.0, 0.0, 0.0, 0.0] + commands["straight"], dtype=torch.float32)
    mot = model.mot
    _t = {}
    mot.prefill_video_cache = make_truncated_prefill(mot, 30, _t)
    with torch.no_grad():
        model.infer_action(
            prompt=STATIC_PROMPT, input_image=img0, action_horizon=8, proprio=proprio0,
            negative_prompt="", text_cfg_scale=1.0, num_inference_steps=args.steps,
            seed=args.seed, rand_device="cpu", tiled=False,
        )
    print("warmup done")

    results = {"ckpt_info": {k: v for k, v in ckpt_info.items() if k != "mot_missing"},
               "k_list": k_list, "steps": args.steps, "seed": args.seed, "scenarios": {}}
    with torch.no_grad():
        for scene_path in scene_files:
            image = load_image_tensor(scene_path)
            for cmd_name, cmd_onehot in commands.items():
                name = f"{scene_path.stem}-{cmd_name}"
                proprio = torch.tensor([8.0, 0.0, 0.0, 0.0] + cmd_onehot, dtype=torch.float32)
                trajs, timing = run_scenario(model, image, proprio, k_list, args.steps, args.seed)
                ref = trajs[max(k_list)]
                ref_m = denorm_odo(ref)
                entry = {"timing": {str(k): timing[k] for k in k_list}, "delta": {}, "trajs_m": {}}
                for k in k_list:
                    d = trajs[k] - ref
                    rel = float(np.linalg.norm(d) / (np.linalg.norm(ref) + 1e-12))
                    d_m = denorm_odo(trajs[k]) - ref_m
                    entry["delta"][str(k)] = {
                        "rel_l2": rel,
                        "max_xy_err_m": float(np.abs(d_m[..., :2]).max()),
                        "mean_xy_err_m": float(np.abs(d_m[..., :2]).mean()),
                        "max_heading_err_deg": float(np.degrees(np.abs(d_m[..., 2]).max())),
                    }
                    entry["trajs_m"][str(k)] = denorm_odo(trajs[k]).round(3).tolist()
                results["scenarios"][name] = entry
                d0 = entry["delta"]["0"]
                print(f"{name}: k=0 rel_l2={d0['rel_l2']:.4f} "
                      f"max_xy={d0['max_xy_err_m']:.3f}m "
                      f"prefill={timing[max(k_list)]['prefill_s']:.2f}s "
                      f"denoise={timing[max(k_list)]['denoise_s']:.2f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print("saved", args.out)


if __name__ == "__main__":
    main()
