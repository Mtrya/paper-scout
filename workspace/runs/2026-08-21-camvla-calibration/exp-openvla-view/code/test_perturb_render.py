"""直接对比 sim.render 在扰动前后的输出,并检查 obs 缓存行为。"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import sys

OPENVLA_ROOT = os.environ["OPENVLA_ROOT"]
sys.path.insert(0, OPENVLA_ROOT)
sys.path.insert(0, os.path.join(OPENVLA_ROOT, "experiments", "robot", "libero"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env
from viewprobe_geom import rotate_camera_about_base

ts = benchmark.get_benchmark_dict()["libero_spatial"]()
task = ts.get_task(0)
env, desc = get_libero_env(task, "openvla", resolution=256)
obs = env.reset()
obs = env.set_init_state(ts.get_task_init_states(0)[0])

sim = env.sim
img0 = sim.render(256, 256, camera_name="agentview")
print("render0 mean:", float(np.mean(img0)), img0.shape)

base_xy = sim.data.body_xpos[sim.model.body("robot0_base").id][:2]
print("base_xy:", base_xy)
new_pos, new_quat = rotate_camera_about_base(env, 15.0, base_xy)
print("new cam pos:", new_pos, "quat:", new_quat)

img1 = sim.render(256, 256, camera_name="agentview")
print("render1 mean:", float(np.mean(img1)))
print("direct render diff:", float(np.mean(np.abs(img1.astype(float) - img0.astype(float)))))

obs2 = env.env._get_observations(force_update=True)
print("force_update obs diff:", float(np.mean(np.abs(obs2["agentview_image"].astype(float) - img0.astype(float)))))

# 复位相机
rotate_camera_about_base(env, -15.0, base_xy)
img2 = sim.render(256, 256, camera_name="agentview")
print("restore diff vs img0:", float(np.mean(np.abs(img2.astype(float) - img0.astype(float)))))
print("PERTURB_RENDER_DONE")
