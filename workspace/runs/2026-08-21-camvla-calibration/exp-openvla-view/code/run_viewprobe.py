"""
OpenVLA-7B × LIBERO-Spatial 相机视角扰动分解实验主脚本
========================================================
条件:
  baseline : θ=0,不扰动
  raw      : 相机绕基座 z 轴旋转 θ(回合 reset+set_init_state 后静态设定)
  rescue   : 同 raw,但把 OpenVLA 输出的 7 维 EE delta 动作的平移与旋转部分
             用 Rz(±θ) 后乘补偿(rescue_sign 决定 ±,先探符号)

推导(CamVLA 式):相机绕基座 z 转 +θ ⇔ 场景在视野里转 -θ;策略若"跟着场景
几何走",输出 = R_z(-θ)·真实动作,故 rescue = R_z(+θ)·预测。

用法:
  python run_viewprobe.py --task-suite libero_spatial --tasks 0 1 \
    --checkpoint <local ckpt dir> --num-trials 20 \
    --mode raw|rescue|baseline --theta-deg 15 --rescue-sign +1 \
    --out-json results/x.json --tag smoke --gpu-id 0
幂等:已记录 (task,theta,mode,rescue_sign,episode) 的条目自动跳过。
"""

import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

# ---- paths -----------------------------------------------------------------
OPENVLA_ROOT = os.environ["OPENVLA_ROOT"]  # 仓库根目录
sys.path.insert(0, OPENVLA_ROOT)
sys.path.insert(0, os.path.join(OPENVLA_ROOT, "experiments", "robot", "libero"))

from libero.libero import benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_env,
    get_libero_image,
    get_libero_dummy_action,
    quat2axisangle,
)
from experiments.robot.openvla_utils import get_processor  # noqa: E402
from experiments.robot.robot_utils import (  # noqa: E402
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("LIBERO_ENV_GPU_ID", "0")


# ---- 预处理加速补丁(TF → PIL;TF 逐 op 开销大,JPEG+resize 是每步 CPU 瓶颈)----
# 语义等价:encode_jpeg(q=75)→decode→lanczos3 resize ≈ PIL JPEG(q=75)+LANCZOS;
# center_crop 的 tf.image.crop_and_resize(0.9 中心裁剪,bilinear 回原尺寸)≈ PIL crop+resize。
# 注意:此补丁让"补丁后"的条件与"补丁前"的 TF 版 baseline 在预处理上有微小差异,
# sweep 阶段会另跑一组 PIL 版 θ=0 作为一致锚点(见 README)。
import io  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

import experiments.robot.libero.libero_utils as _lu  # noqa: E402
import experiments.robot.openvla_utils as _ou  # noqa: E402
import experiments.robot.robot_utils as _ru  # noqa: E402


def _pil_resize_image(img, resize_size):
    """等价原 resize_image:JPEG 编解码往返 + lanczos3 缩放,但不依赖 TF。"""
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    p = PILImage.fromarray(img)
    buf = io.BytesIO()
    p.save(buf, format="JPEG", quality=75)
    p = PILImage.open(io.BytesIO(buf.getvalue()))
    p = p.resize(resize_size, PILImage.LANCZOS)
    return np.asarray(p)


def _pil_center_crop(img_pil, scale=0.9):
    """0.9 中心裁剪后 bilinear 放大回原尺寸(等价 tf crop_and_resize 默认 bilinear)。"""
    w, h = img_pil.size
    nw, nh = int(round(w * math.sqrt(scale))), int(round(h * math.sqrt(scale)))
    left, top = (w - nw) // 2, (h - nh) // 2
    return img_pil.crop((left, top, left + nw, top + nh)).resize((w, h), PILImage.BILINEAR)


def _pil_get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    image = PILImage.fromarray(obs["full_image"]).convert("RGB")
    if center_crop:
        image = _pil_center_crop(image)
    if "openvla-v01" in base_vla_name:
        prompt = (
            f"{_ou.OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to "
            f"{task_label.lower()}? ASSISTANT:"
        )
    else:
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs = processor(prompt, image).to(_ru.DEVICE, dtype=torch.bfloat16)
    return vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)


_lu.resize_image = _pil_resize_image
_ou.get_vla_action = _pil_get_vla_action
_ru.get_vla_action = _pil_get_vla_action  # robot_utils 在 import 时已持有引用,需覆盖


# ---- geometry helpers -------------------------------------------------------
def rz(theta_deg):
    """世界系绕 z 轴旋转矩阵(右手系,逆时针为正)。"""
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_multiply(q1, q2):
    """(w,x,y,z) Hamilton 积 q1⊗q2。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def rotate_camera_about_base(env, theta_deg, base_xy, cam_name="agentview"):
    """把相机(位置+朝向)绕基座竖直轴(过 base_xy)整体旋转 theta_deg。
    等价于场景在视野中绕该轴旋转 -theta_deg(相机固定)。"""
    sim = env.sim
    cam_id = sim.model.camera(cam_name).id  # mujoco >=3.0 命名索引
    pos = sim.model.cam_pos[cam_id].copy()
    quat = sim.model.cam_quat[cam_id].copy()  # (w,x,y,z)
    R = rz(theta_deg)
    center = np.array([base_xy[0], base_xy[1], pos[2]])
    new_pos = R @ (pos - center) + center
    t = math.radians(theta_deg)
    q_rot = np.array([math.cos(t / 2), 0.0, 0.0, math.sin(t / 2)])  # wxyz,Rz(θ)
    new_quat = quat_multiply(q_rot, quat)  # R' = Rz(θ) @ R
    sim.model.cam_pos[cam_id] = new_pos
    sim.model.cam_quat[cam_id] = new_quat
    return new_pos, new_quat


def rescue_action(action, theta_deg, sign):
    """rescue = R_z(sign·θ) 作用在平移与 axis-angle 旋转分量上,夹爪不变。"""
    a = np.array(action, dtype=np.float64).copy()
    R = rz(sign * theta_deg)
    a[:3] = R @ a[:3]
    a[3:6] = R @ a[3:6]
    return a


# ---- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-suite", default="libero_spatial")
    ap.add_argument("--tasks", nargs="+", type=int, required=True, help="任务索引")
    ap.add_argument("--checkpoint", required=True, help="本地权重目录")
    ap.add_argument("--num-trials", type=int, default=20)
    ap.add_argument("--mode", choices=["baseline", "raw", "rescue"], required=True)
    ap.add_argument("--theta-deg", type=float, default=0.0)
    ap.add_argument("--rescue-sign", type=float, default=1.0, help="±1")
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--center-crop", type=int, default=1)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed_everywhere(args.seed)
    cfg = SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=args.checkpoint,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=bool(args.center_crop),
        task_suite_name=args.task_suite,
        unnorm_key=args.task_suite,
        num_steps_wait=args.num_steps_wait,
        local_log_dir="./experiments/logs",
    )

    # ---- 模型与 processor(每进程一次)----
    model = get_model(cfg)
    if hasattr(model, "norm_stats") and cfg.unnorm_key not in model.norm_stats:
        cand = f"{cfg.unnorm_key}_no_noops"
        if cand in model.norm_stats:
            cfg.unnorm_key = cand
    print(f"[*] unnorm_key = {cfg.unnorm_key}", flush=True)
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)

    # ---- 任务套件 ----
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    max_steps_map = {"libero_spatial": 220, "libero_object": 280,
                     "libero_goal": 300, "libero_10": 520, "libero_90": 400}
    max_steps = max_steps_map[args.task_suite]

    # ---- 结果文件(幂等)----
    out_dir = os.path.dirname(os.path.abspath(args.out_json))
    os.makedirs(out_dir, exist_ok=True)
    results = []
    if os.path.exists(args.out_json):
        with open(args.out_json) as f:
            results = json.load(f)
    done_keys = {(r["task"], r["theta_deg"], r["mode"], r["rescue_sign"], r["episode"], r.get("tag", ""))
                 for r in results}

    cam_info_written = False
    summary = {"task_suite": args.task_suite, "mode": args.mode, "theta_deg": args.theta_deg,
               "rescue_sign": args.rescue_sign, "tasks": {}, "tag": args.tag}

    for task_id in args.tasks:
        task = task_suite.get_task(task_id)
        task_description = task.language
        initial_states = task_suite.get_task_init_states(task_id)
        env, _ = get_libero_env(task, cfg.model_family, resolution=256)

        # 打印默认相机位姿与机器人基座位置(几何核验)
        sim = env.sim
        cam_id = sim.model.camera("agentview").id
        default_cam_pos = sim.model.cam_pos[cam_id].copy()
        default_cam_quat = sim.model.cam_quat[cam_id].copy()
        body_names = [sim.model.body(i).name for i in range(sim.model.nbody)]
        base_body = [n for n in body_names if "base" in n.lower()]
        base_xy = None
        if base_body:
            base_xy = sim.data.body_xpos[sim.model.body(base_body[0]).id][:2]
        if base_xy is None:
            base_xy = np.array([0.0, 0.0])  # 兜底;smoke 时核对几何
            print("[WARN] no base body found; rotating about (0,0)", flush=True)
        cam_info = {"task": task_id, "camera": "agentview",
                    "default_cam_pos": default_cam_pos.tolist(),
                    "default_cam_quat": default_cam_quat.tolist(),
                    "base_body": base_body[0] if base_body else None,
                    "base_xy": base_xy.tolist() if base_xy is not None else None}
        if not cam_info_written:
            cam_info_written = True
            with open(os.path.join(out_dir, "cam_info.json"), "w") as f:
                json.dump(cam_info, f, indent=1)
        print(f"[cam] {json.dumps(cam_info, indent=1)}", flush=True)

        task_res = {"task": task_id, "description": task_description,
                    "episodes": [], "successes": 0}
        for ep in range(args.num_trials):
            if (task_id, args.theta_deg, args.mode, args.rescue_sign, ep, args.tag) in done_keys:
                print(f"[skip] task {task_id} ep {ep} (done)", flush=True)
                continue

            env.reset()
            obs = env.set_init_state(initial_states[ep])

            # 相机扰动:整集静态
            if args.mode in ("raw", "rescue") and args.theta_deg != 0:
                rotate_camera_about_base(env, args.theta_deg, base_xy)

            t, success, done = 0, False, False
            steps_taken = 0
            while t < max_steps + cfg.num_steps_wait:
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue
                img = get_libero_image(obs, resize_size)
                observation = {
                    "full_image": img,
                    "state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                }
                action = get_action(cfg, model, observation, task_description, processor=processor)
                if os.environ.get("VIEWPROBE_DEBUG_ACTIONS"):
                    _dbg = {"mode": args.mode, "theta": args.theta_deg, "sign": args.rescue_sign,
                            "action_pre": np.round(action, 3).tolist()}
                if args.mode == "rescue" and args.theta_deg != 0:
                    action = rescue_action(action, args.theta_deg, args.rescue_sign)
                    if os.environ.get("VIEWPROBE_DEBUG_ACTIONS"):
                        _dbg["action_post"] = np.round(action, 3).tolist()
                if os.environ.get("VIEWPROBE_DEBUG_ACTIONS"):
                    print("[dbg-act]", json.dumps(_dbg), flush=True)
                action = normalize_gripper_action(action, binarize=True)
                if cfg.model_family == "openvla":
                    action = invert_gripper_action(action)
                obs, reward, done, info = env.step(action.tolist())
                t += 1
                steps_taken = t - cfg.num_steps_wait
                if done:
                    success = True
                    break

            ep_rec = {
                "task": task_id, "theta_deg": args.theta_deg, "mode": args.mode,
                "rescue_sign": args.rescue_sign, "episode": ep, "tag": args.tag,
                "success": bool(success), "steps": int(steps_taken),
                "final_eef_pos": obs["robot0_eef_pos"].tolist(),
                "final_eef_quat": obs["robot0_eef_quat"].tolist(),
                "final_gripper_qpos": float(np.asarray(obs["robot0_gripper_qpos"]).mean()),
                "ts": time.time(),
            }
            results.append(ep_rec)
            with open(args.out_json, "w") as f:
                json.dump(results, f, indent=1)
            if success:
                task_res["successes"] += 1
            task_res["episodes"].append(ep_rec["episode"])
            print(f"[t{task_id} ep{ep}] mode={args.mode} θ={args.theta_deg} sign={args.rescue_sign} "
                  f"success={success} steps={steps_taken}", flush=True)

        print(f"[task {task_id} done] success {task_res['successes']}/{args.num_trials}", flush=True)
        # 汇总只统计本进程实际跑的(去掉 skip 的旧条目口径由 README 说明;此处用最新 JSON 重算)
        fresh = [r for r in results if r["task"] == task_id and r["mode"] == args.mode
                 and r["theta_deg"] == args.theta_deg and r["rescue_sign"] == args.rescue_sign
                 and r.get("tag", "") == args.tag]
        sr = sum(r["success"] for r in fresh) / len(fresh) if fresh else None
        summary["tasks"][str(task_id)] = {"description": task_description,
                                          "success_rate": sr, "n": len(fresh)}

    with open(args.out_json + ".summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[SUMMARY] {json.dumps(summary, indent=1)}", flush=True)
    print("VIEWPROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
