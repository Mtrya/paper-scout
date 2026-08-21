"""验证:改 cam_pos/cam_quat 后调用 mj_forward 是否让渲染反映扰动。"""
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
from viewprobe_geom import rz, quat_multiply

ts = benchmark.get_benchmark_dict()["libero_spatial"]()
env, desc = get_libero_env(ts.get_task(0), "openvla", resolution=256)
obs = env.reset()
obs = env.set_init_state(ts.get_task_init_states(0)[0])
sim = env.sim
cid = sim.model.camera("agentview").id
base_xy = sim.data.body_xpos[sim.model.body("robot0_base").id][:2]

img0 = sim.render(256, 256, camera_name="agentview")
pos0 = sim.model.cam_pos[cid].copy()
quat0 = sim.model.cam_quat[cid].copy()

R = rz(15.0)
center = np.array([base_xy[0], base_xy[1], pos0[2]])
new_pos = R @ (pos0 - center) + center
t = np.radians(15.0)
q_rot = np.array([np.cos(t / 2), 0, 0, np.sin(t / 2)])
new_quat = quat_multiply(q_rot, quat0)

sim.model.cam_pos[cid] = new_pos
sim.model.cam_quat[cid] = new_quat
# 关键:mj_forward 把 model.cam_* 同步到 data.cam_xpos/cam_xmat
sim.forward()
img1 = sim.render(256, 256, camera_name="agentview")
print("after forward+render diff:", float(np.mean(np.abs(img1.astype(float) - img0.astype(float)))))

# obs 管线里同样生效
obs2 = env.env._get_observations(force_update=True)
print("obs diff:", float(np.mean(np.abs(obs2["agentview_image"].astype(float) - img0.astype(float)))))
print("DEBUG_CAM2_DONE")
