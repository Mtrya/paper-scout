"""调试:确认 model.cam_pos/cam_quat 写入是否生效,以及哪种写入方式能改变渲染。"""
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
print("base_xy:", base_xy)

img0 = sim.render(256, 256, camera_name="agentview")
pos0 = sim.model.cam_pos[cid].copy()
quat0 = sim.model.cam_quat[cid].copy()

R = rz(15.0)
center = np.array([base_xy[0], base_xy[1], pos0[2]])
new_pos = R @ (pos0 - center) + center
t = np.radians(15.0)
q_rot = np.array([np.cos(t / 2), 0, 0, np.sin(t / 2)])
new_quat = quat_multiply(q_rot, quat0)

# 方式1: robosuite sim.model 的 numpy item 写入
sim.model.cam_pos[cid] = new_pos
sim.model.cam_quat[cid] = new_quat
print("after write via sim.model:")
print("  cam_pos:", sim.model.cam_pos[cid], "== new?", np.allclose(sim.model.cam_pos[cid], new_pos))
print("  cam_quat:", sim.model.cam_quat[cid])
img1 = sim.render(256, 256, camera_name="agentview")
print("  render diff:", float(np.mean(np.abs(img1.astype(float) - img0.astype(float)))))

# 方式2: 直接写 mujoco 原生 MjModel(绕过 robosuite metaclass)
raw = sim._model  # robosuite MjSim 的 _model 是 mujoco.MjModel?
print("raw type:", type(raw))
raw.cam_pos[cid] = new_pos
raw.cam_quat[cid] = new_quat
print("after write via raw _model: cam_pos:", sim.model.cam_pos[cid])
img2 = sim.render(256, 256, camera_name="agentview")
print("  render diff:", float(np.mean(np.abs(img2.astype(float) - img0.astype(float)))))

# 方式3: 用 sim.model 的 property 整体赋值
try:
    sim.model.cam_pos = np.zeros_like(sim.model.cam_pos) + new_pos
    print("whole-property set OK; cam_pos[0]:", sim.model.cam_pos[cid])
    img3 = sim.render(256, 256, camera_name="agentview")
    print("  render diff:", float(np.mean(np.abs(img3.astype(float) - img0.astype(float)))))
except Exception as e:
    print("whole-property set failed:", e)
print("DEBUG_CAM_DONE")
