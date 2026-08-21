"""视角扰动实验的纯几何工具(不依赖 openvla/模型栈)。"""
import math
import numpy as np


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
    等价于场景在视野中绕该轴旋转 -theta_deg(相机固定)。
    注意:mujoco>=3.x 相机位姿渲染时读 data.cam_xpos/cam_xmat(mj_forward 从
    model.cam_* 同步),改完必须 env.sim.forward() 才会生效。"""
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
    sim.forward()
    return new_pos, new_quat


def rescue_action(action, theta_deg, sign):
    """rescue = R_z(sign·θ) 作用在平移与 axis-angle 旋转分量上,夹爪不变。"""
    a = np.array(action, dtype=np.float64).copy()
    R = rz(sign * theta_deg)
    a[:3] = R @ a[:3]
    a[3:6] = R @ a[3:6]
    return a
