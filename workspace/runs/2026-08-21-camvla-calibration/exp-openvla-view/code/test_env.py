"""LIBERO env 冒烟:创建任务 0 env,reset+set_init_state,打印相机/基座几何,渲染一帧。"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import sys

OPENVLA_ROOT = os.environ["OPENVLA_ROOT"]
sys.path.insert(0, OPENVLA_ROOT)
sys.path.insert(0, os.path.join(OPENVLA_ROOT, "experiments", "robot", "libero"))

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env

benchmark_dict = benchmark.get_benchmark_dict()
ts = benchmark_dict["libero_spatial"]()
task = ts.get_task(0)
initial_states = ts.get_task_init_states(0)
env, desc = get_libero_env(task, "openvla", resolution=256)
print("desc:", desc)
obs = env.reset()
obs = env.set_init_state(initial_states[0])
print("obs keys:", sorted(obs.keys()))

sim = env.sim
cam = sim.model.camera("agentview")
print("cam id:", cam.id)
print("cam pos:", sim.model.cam_pos[cam.id])
print("cam quat(wxyz):", sim.model.cam_quat[cam.id])
body_names = [sim.model.body(i).name for i in range(sim.model.nbody)]
print("bodies:", body_names[:25])
base = [n for n in body_names if "base" in n.lower()]
print("base bodies:", base)
if base:
    bid = sim.model.body(base[0]).id
    print("base pos:", sim.data.body_xpos[bid])

# 渲染一帧 agentview
img = obs["agentview_image"]
print("agentview img shape:", img.shape, "dtype:", img.dtype, "mean:", float(np.mean(img)))
# 扰动测试:绕基座转 15°,再渲染,确认图像变化
import math
from run_viewprobe import rotate_camera_about_base
base_xy = sim.data.body_xpos[sim.model.body(base[0]).id][:2] if base else np.array([0.0, 0.0])
rotate_camera_about_base(env, 15.0, base_xy)
img2 = env.env._get_observations(force_update=True)["agentview_image"]
print("perturbed img mean:", float(np.mean(img2)), "diff:", float(np.mean(np.abs(img2.astype(float) - img.astype(float)))))
print("ENV_SMOKE_DONE")
