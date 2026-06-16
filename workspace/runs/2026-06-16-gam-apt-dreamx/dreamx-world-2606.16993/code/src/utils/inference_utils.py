import math
import time
import torch
import random
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
from .pose_utils import interpolate_camera_poses

# from videox_fun.data.util import _invert_SE3, ucm_unproject_grid_intrinsics, world_to_ray_mats, compute_up_lat_map

ACTION_DICT = {"w": "forward", "a": "left", "d": "right", "s": "backward", "j":"left_rot", "l":"right_rot", "i":"up_rot", "k":"down_rot",}
            
def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    # if pver.parse(torch.__version__) < pver.parse('1.10'):
    #     return torch.meshgrid(*args)
    # else:
    return torch.meshgrid(*args, indexing='ij')
    
def get_relative_pose(cam_params, scale_factor=1):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    source_cam_c2w = abs_c2ws[0]
    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    for pose in ret_poses:
        pose[:3, -1:] *= scale_factor
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses

def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker

def get_c2w(w2cs, transform_matrix, relative_c2w):
    if relative_c2w:
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ np.linalg.inv(w2c) for w2c in w2cs[1:]]
        for pose in ret_poses:
            pose[:3, -1:] *= 2
        # ret_poses = [poses[:, :3]*2 for poses in ret_poses]
        # ret_poses[:, :, :3] *= 2
    else:
        ret_poses = [np.linalg.inv(w2c) for w2c in w2cs]
    ret_poses = [transform_matrix @ x for x in ret_poses]
    return np.array(ret_poses, dtype=np.float32)

def _compute_translation_step(motion_type, current_pose, translation_value, duration):
    """Compute per-frame translation step in **world coordinates**.

    The camera forward direction in world space is ``R_w2c^T @ [0,0,1]``
    (OpenCV convention: +Z is the camera's viewing direction).

    ``current_pose['position']`` accumulates world-space displacement;
    the conversion to w2c translation ``t = -R @ pos`` is done later
    when building the extrinsic matrix.
    """
    if motion_type in ['forward', 'backward']:
        yaw_rad = np.radians(current_pose['rotation'][1])
        pitch_rad = np.radians(current_pose['rotation'][0])
        forward_vec = np.array([
            -math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            math.cos(yaw_rad) * math.cos(pitch_rad)
        ])
        direction = 1 if motion_type == 'forward' else -1
        total_move = forward_vec * translation_value * direction
        return total_move / duration

    elif motion_type in ['left', 'right']:
        yaw_rad = np.radians(current_pose['rotation'][1])
        right_vec = np.array([math.cos(yaw_rad), 0, math.sin(yaw_rad)])
        direction = -1 if motion_type == 'left' else 1
        total_move = right_vec * translation_value * direction
        return total_move / duration

    return np.zeros(3)

def _compute_rotation_step(motion_type, rotation_value, duration):
    """Compute per-frame rotation step vector for a single rotation motion type.

    rotation layout: [pitch (X-axis, up/down look), yaw (Y-axis, left/right turn), roll (Z-axis)]
    """
    if motion_type.endswith('rot'):
        axis = motion_type.split('_')[0]
        total_rotation = np.zeros(3)
        if axis == 'left':
            total_rotation[1] = rotation_value
        elif axis == 'right':
            total_rotation[1] = -rotation_value
        elif axis == 'up':
            total_rotation[0] = -rotation_value
        elif axis == 'down':
            total_rotation[0] = rotation_value
        return total_rotation / duration

    return np.zeros(3)


def generate_composite_motion_segment(current_pose,
                                      motion_types,
                                      translation_value: float,
                                      rotation_value: float,
                                      duration: int = 30):
    """Generate a trajectory that combines multiple motions simultaneously.

    Unlike ``generate_motion_segment`` which accepts a single motion type,
    this function accepts a list of motion types and blends them together
    so that, e.g., "forward" + "right_rot" produces a forward-moving arc.

    Parameters:
        current_pose: dict with 'position' (np.array[3]) and 'rotation' (np.array[3])
        motion_types: list of str, each one of
            ('forward', 'backward', 'left', 'right',
             'left_rot', 'right_rot', 'up_rot', 'down_rot')
        translation_value: Translation magnitude (m)
        rotation_value: Rotation magnitude (degree)
        duration: Number of frames

    Return:
        positions:    list of np.array(x, y, z)
        rotations:    list of np.array(pitch, yaw, roll)
        current_pose: updated pose dict after the motion
    """
    if isinstance(motion_types, str):
        motion_types = [motion_types]

    positions = []
    rotations = []

    translation_step = np.zeros(3)
    rotation_step = np.zeros(3)

    for motion_type in motion_types:
        translation_step += _compute_translation_step(
            motion_type, current_pose, translation_value, duration
        )
        rotation_step += _compute_rotation_step(
            motion_type, rotation_value, duration
        )

    for i in range(1, duration + 1):
        new_pos = current_pose['position'] + translation_step * i
        new_rot = current_pose['rotation'] + rotation_step * i
        positions.append(new_pos.copy())
        rotations.append(new_rot.copy())

    current_pose['position'] = positions[-1].copy()
    current_pose['rotation'] = rotations[-1].copy()

    return positions, rotations, current_pose

def euler_to_quaternion(angles):
    """Convert Euler angles (pitch, yaw, roll) to quaternion.

    Uses ZYX intrinsic rotation order (roll around Z, then pitch around X',
    then yaw around Y'') which matches the w2c + OpenCV convention used by
    the translation computation in generate_motion_segment.

    Args:
        angles: [pitch, yaw, roll] in degrees.
    """
    pitch, yaw, roll = np.radians(angles)
    
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    qw = cy * cp * cr + sy * sp * sr
    qx = cy * sp * cr + sy * cp * sr
    qy = sy * cp * cr - cy * sp * sr
    qz = cy * cp * sr - sy * sp * cr
    
    return [qw, qx, qy, qz]

def quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    

TRANSLATION_BASE_UNIT = 1.0   
ROTATION_BASE_UNIT = 10.0     

def ActionToPoseFromID(action_ids, action_speed_list, duration=33):
    """Convert a sequence of action segments into camera pose trajectories.

    """
    all_positions = []
    all_rotations = []
    current_pose = {
        'position': np.array([0.0, 0.0, 0.0]),  # XYZ
        'rotation': np.array([0.0, 0.0, 0.0])   # (pitch, yaw, roll)
    }
    intrinsic = [0.8, 0.5, 0.5, 0.5]

    for idx, action_id in enumerate(action_ids):
        # Normalise action_id into a list of individual keys
        if isinstance(action_id, str):
            keys = list(action_id)  # "wl" -> ["w", "l"]
        else:
            keys = list(action_id)  # already a list / tuple

        motion_types = [ACTION_DICT[key] for key in keys]
        speed = action_speed_list[idx]

        positions, rotations, current_pose = generate_composite_motion_segment(
            current_pose,
            motion_types=motion_types,
            translation_value=speed * TRANSLATION_BASE_UNIT,
            rotation_value=speed * ROTATION_BASE_UNIT,
            duration=duration,
        )
        all_positions.extend(positions)
        all_rotations.extend(rotations)

    pose_list = []

    row = [0] + intrinsic + [0, 0] + [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    first_row = " ".join(map(str, row))
    pose_list.append(first_row)

    for i, (pos, rot) in enumerate(zip(all_positions, all_rotations)):
        quat = euler_to_quaternion(rot)
        R = quaternion_to_rotation_matrix(quat)
        # pos is world-space camera position; w2c translation is t = -R @ pos
        t = -R @ pos
        extrinsic = np.hstack([R, t.reshape(3, 1)])

        row = [i] + intrinsic + [0, 0] + extrinsic.flatten().tolist()
        pose_list.append(" ".join(map(str, row)))
    return pose_list

class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def align_to(value, alignment):
    return int(math.ceil(value / alignment) * alignment)


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def GetPoseEmbedsFromPosesPrope(
    poses, h, w, target_length, 
    flip=False, start_index=0, 
    cam_method='prope',
    dtype=torch.float32, device='cpu',
):
   
    poses = [pose.split(' ') for pose in poses]

    start_idx = start_index
    sample_id = [start_idx + i for i in range(target_length)]
    poses = [poses[i] for i in sample_id]
    
    cam_params = [[float(x) for x in pose] for pose in poses]
    assert len(cam_params) == target_length
    cam_params = [Camera(cam_param) for cam_param in cam_params]
    

    # Align to VAE temporal downsampling (1+4k pattern):
    # Frame 0 is kept, then every 4th frame starting from frame 1
    # latent_frame_count = 1 + (N-1) // 4
    n_frames = len(cam_params)
    latent_frame_count = 1 + (n_frames - 1) // 4

    src_indices = np.arange(n_frames, dtype=np.float64)
    tgt_indices = np.linspace(0, n_frames - 1, latent_frame_count)
    cam_params = interpolate_camera_poses(cam_params, src_indices, tgt_indices)

    
    c2w_poses_aligned = get_relative_pose(cam_params)
    c2ws = torch.as_tensor(c2w_poses_aligned, dtype=dtype, device=device)


    T_latent = c2ws.shape[0]
    viewmats = _invert_SE3(c2ws)  # [T_latent, 4, 4]

    default_intrinsic = [
        [969.6969696969696, 0.0, 960.0],
        [0.0, 969.6969696969696, 540.0],
        [0.0, 0.0, 1.0],
    ]
    fx_norm = default_intrinsic[0][0] / (default_intrinsic[0][2] * 2)
    fy_norm = default_intrinsic[1][1] / (default_intrinsic[1][2] * 2)

    Ks = torch.zeros((T_latent, 3, 3), device=device, dtype=dtype)
    Ks[:, 0, 0] = fx_norm
    Ks[:, 1, 1] = fy_norm
    Ks[:, 0, 2] = 0
    Ks[:, 1, 2] = 0
    Ks[:, 2, 2] = 1.0

    camera_condition = {
        'viewmats': viewmats,
        'K': Ks,
    }
    return camera_condition, poses
    
   
