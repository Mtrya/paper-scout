import numpy as np
import torch
import math
from PIL import Image
from einops import rearrange
from packaging import version as pver

# ========================================================================================
# 1. Helper Functions & Classes (Copied from videox_fun/data/utils.py)
# ========================================================================================

def custom_meshgrid(*args):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


class Camera(object):
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py
    """
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


def get_relative_pose(cam_params):
    """Calculates camera poses relative to the first frame's pose.
    """
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    
    target_cam_c2w = abs_c2ws[0]
    abs2rel = np.linalg.inv(target_cam_c2w)
    
    ret_poses = [abs2rel @ abs_c2w for abs_c2w in abs_c2ws]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def ray_condition(K, c2w, H, W, device):
    """Generates Plücker embeddings from camera intrinsics (K) and camera-to-world poses (c2w).
    """
    B = K.shape[0]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5

    fx, fy, cx, cy = K.chunk(4, dim=-1)

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)
    rays_o = c2w[..., :3, 3]
    rays_o = rays_o[:, :, None].expand_as(rays_d)

    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)
    return plucker


def process_trajectory_file_to_plucker(
    pose_file_path, 
    width=1280, 
    height=704, 
    original_pose_width=1280, 
    original_pose_height=704, 
    device='cpu'
    ):
    """
    Reads a trajectory file, processes it, and returns the Plücker embedding tensor.
    """
    with open(pose_file_path, 'r') as f:
        poses = f.readlines()

    poses = [pose.strip().split(' ') for pose in poses[1:]]
    cam_params = [[float(x) for x in pose] for pose in poses]
    
    return process_poses_to_plucker(
        cam_params, width, height, original_pose_width, original_pose_height, device
    )


def euler_to_rotation_matrix(roll, pitch, yaw):
    """Converts Euler angles (in radians) to a 3x3 rotation matrix."""
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll), math.cos(roll)]])
    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])
    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]])
    return R_z @ R_y @ R_x


def generate_keyboard_controlled_trajectory(
    motion_sequence,
    num_frames_per_action=4,
    speed=1.0,
    fx=0.8, fy=0.8, cx=0.5, cy=0.5
):
    """
    Generates camera trajectory based on keyboard control sequence.
    
    Args:
        motion_sequence (list): List of keyboard control strings, e.g., ["w", "wi", "sj"]
            Movement keys (WASD):
                w: forward (move along camera's forward direction)
                s: backward (move along camera's backward direction)
                a: left (move along camera's left direction)
                d: right (move along camera's right direction)
            View keys (IKJL):
                i: look up (pitch up)
                k: look down (pitch down)
                j: look left (yaw left)
                l: look right (yaw right)
        num_frames_per_action (int): Number of frames for each action in the sequence.
        speed (float): Controls the intensity of the motion.
        fx, fy, cx, cy: Camera intrinsic parameters.
    
    Returns:
        list of lists: A list where each inner list contains 19 pose parameters for a frame.
    """
    trajectories = []
    
    # Current camera state in world coordinates
    current_position = np.zeros(3)  # [x, y, z]
    current_rotation = np.eye(3)    # 3x3 rotation matrix
    
    # Movement and rotation increments per frame
    move_step = speed * 0.05  # Movement distance per frame
    rotate_step = speed * math.pi / 180  # Rotation angle per frame (in radians)
    
    frame_idx = 0
    
    for action in motion_sequence:
        # Parse the action string
        move_forward = 'w' in action
        move_backward = 's' in action
        move_left = 'a' in action
        move_right = 'd' in action
        look_up = 'i' in action
        look_down = 'k' in action
        look_left = 'j' in action
        look_right = 'l' in action
        
        for _ in range(num_frames_per_action):
            # Apply view rotations first (modify current_rotation)
            pitch_delta = 0
            yaw_delta = 0
            
            if look_up:
                pitch_delta += rotate_step
            if look_down:
                pitch_delta -= rotate_step
            if look_left:
                yaw_delta += rotate_step
            if look_right:
                yaw_delta -= rotate_step
            
            # Apply rotation
            if pitch_delta != 0 or yaw_delta != 0:
                rotation_delta = euler_to_rotation_matrix(pitch_delta, yaw_delta, 0)
                current_rotation = current_rotation @ rotation_delta
            
            # Calculate movement in camera's local coordinate system
            local_movement = np.zeros(3)
            if move_forward:
                local_movement[2] += move_step  # Forward in camera space
            if move_backward:
                local_movement[2] -= move_step  # Backward in camera space
            if move_left:
                local_movement[0] -= move_step  # Left in camera space
            if move_right:
                local_movement[0] += move_step  # Right in camera space
            
            # Transform local movement to world coordinates
            world_movement = current_rotation @ local_movement
            current_position += world_movement
            
            # Build camera-to-world (c2w) matrix
            c2w_rotation = current_rotation
            c2w_translation = current_position
            
            # Convert to world-to-camera (w2c) matrix
            w2c_rotation = c2w_rotation.T
            w2c_translation = -w2c_rotation @ c2w_translation
            w2c_mat = np.hstack((w2c_rotation, w2c_translation.reshape(3, 1)))
            
            # Create the 19-parameter entry for the frame
            frame_params = [frame_idx, fx, fy, cx, cy, 0, 0] + list(w2c_mat.flatten())
            trajectories.append(frame_params)
            frame_idx += 1
    
    return trajectories


def generate_random_balanced_trajectory(
    num_frames,
    width=1280,
    height=704,
    original_pose_width=1280,
    original_pose_height=704,
    fx=0.8,
    fy=0.8,
    cx=0.5,
    cy=0.5,
    chunk_size = -1,
    device='cpu',
    seed=None,
    return_cam_params=False
):
    """
    Generates a random but balanced camera trajectory and returns the Plücker embedding.
    
    Args:
        num_frames (int): Number of frames to generate.
        width (int): Target video width.
        height (int): Target video height.
        original_pose_width (int): Original pose width for intrinsic adjustment.
        original_pose_height (int): Original pose height for intrinsic adjustment.
        speed (float): Controls the intensity of the motion.
        fx, fy, cx, cy: Camera intrinsic parameters.
        device (str): Device for tensor operations ('cpu' or 'cuda').
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        torch.Tensor: Plücker embedding tensor of shape [1, num_frames, height, width, 6].
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Define motion primitives for balanced trajectory
    # Single motion primitives (70% probability)
    single_motion_primitives = [
        'w',   # forward
        's',   # backward
        'a',   # left
        'd',   # right
        'i',   # look up
        'k',   # look down
        'j',   # look left
        'l',   # look right
        ' ',   # stay still
    ]
    
    # Combined motion primitives (30% probability)
    combined_motion_primitives = [
        'wi',  # forward + look up
        'wj',  # forward + look left
        'wl',  # forward + look right
        'wk',  # forward + look down
        'si',  # backward + look up
        'sj',  # backward + look left
        'sl',  # backward + look right
        'sk',  # backward + look down
        'ai',  # left + look up
        'aj',  # left + look left
        'al',  # left + look right
        'ak',  # left + look down
        'di',  # right + look up
        'dj',  # right + look left
        'dl',  # right + look right
        'dk'   # right + look down
    ]
    
    # Create weighted motion primitives list
    # 70% single, 30% combined
    num_single = int(len(single_motion_primitives) * 0.7 / (0.7 / len(single_motion_primitives) + 0.3 / len(combined_motion_primitives)))
    num_combined = int(len(combined_motion_primitives) * 0.3 / (0.7 / len(single_motion_primitives) + 0.3 / len(combined_motion_primitives)))
    
    # Build motion primitives with proper probability distribution
    motion_primitives = []
    for _ in range(7):  # 70% weight for single motions
        motion_primitives.extend(single_motion_primitives)
    for _ in range(3):  # 30% weight for combined motions
        motion_primitives.extend(combined_motion_primitives)
    
    # Decide whether to use simple trajectory (50% chance)
    use_simple_trajectory = np.random.random() < 0.7
    
    # Generate a balanced motion sequence
    motion_sequence = []
    remaining_frames = num_frames
    
    if use_simple_trajectory:
        # Simple trajectory: maximum 3 actions
        num_actions = np.random.randint(1, 4)  # 1, 2, or 3 actions
        selected_primitives = np.random.choice(motion_primitives, size=num_actions, replace=False)
        frames_per_action = num_frames // num_actions
        
        for i, primitive in enumerate(selected_primitives):
            if i == num_actions - 1:
                # Last action gets all remaining frames
                action_frames = remaining_frames
            else:
                # Randomly vary the number of frames per action (±30%)
                action_frames = max(1, int(frames_per_action * np.random.uniform(0.7, 1.3)))
                action_frames = min(action_frames, remaining_frames)
            
            motion_sequence.append((primitive, action_frames))
            remaining_frames -= action_frames
    else:
        # Complex trajectory: original logic with all primitives
        frames_per_action = max(2, num_frames // len(motion_primitives))
        
        # Shuffle motion primitives for randomness
        shuffled_primitives = motion_primitives.copy()
        np.random.shuffle(shuffled_primitives)
        
        for primitive in shuffled_primitives:
            if remaining_frames <= 0:
                break
            
            # Randomly vary the number of frames per action (±50%)
            action_frames = max(1, int(frames_per_action * np.random.uniform(0.5, 1.5)))
            action_frames = min(action_frames, remaining_frames)
            
            motion_sequence.append((primitive, action_frames))
            remaining_frames -= action_frames
        
        # If we still have remaining frames, distribute them randomly
        while remaining_frames > 0:
            primitive = np.random.choice(motion_primitives)
            action_frames = min(remaining_frames, max(1, int(frames_per_action * 0.5)))
            motion_sequence.append((primitive, action_frames))
            remaining_frames -= action_frames
    
    # Generate trajectory using keyboard control
    trajectories = []
    current_position = np.zeros(3)
    current_rotation = np.eye(3)
    
    translation_speed = np.random.uniform(0.5, 1.0) #(1.0, 1.5)
    move_step = translation_speed * 0.05

    rotation_speed = np.random.uniform(1.0, 1.5)
    rotate_step = rotation_speed * math.pi / 180
    
    frame_idx = 0
    
    for action, num_action_frames in motion_sequence:
        # Parse the action string
        move_forward = 'w' in action
        move_backward = 's' in action
        move_left = 'a' in action
        move_right = 'd' in action
        look_up = 'i' in action
        look_down = 'k' in action
        look_left = 'j' in action
        look_right = 'l' in action
        
        for _ in range(num_action_frames):
            # Apply view rotations
            pitch_delta = 0
            yaw_delta = 0
            
            if look_up:
                pitch_delta += rotate_step
            if look_down:
                pitch_delta -= rotate_step
            if look_left:
                yaw_delta += rotate_step
            if look_right:
                yaw_delta -= rotate_step
            
            if pitch_delta != 0 or yaw_delta != 0:
                rotation_delta = euler_to_rotation_matrix(pitch_delta, yaw_delta, 0)
                current_rotation = current_rotation @ rotation_delta
            
            # Calculate movement
            local_movement = np.zeros(3)
            if move_forward:
                local_movement[2] += move_step
            if move_backward:
                local_movement[2] -= move_step
            if move_left:
                local_movement[0] -= move_step
            if move_right:
                local_movement[0] += move_step
            
            world_movement = current_rotation @ local_movement
            current_position += world_movement
            
            # Build w2c matrix
            c2w_rotation = current_rotation
            c2w_translation = current_position
            w2c_rotation = c2w_rotation.T
            w2c_translation = -w2c_rotation @ c2w_translation
            w2c_mat = np.hstack((w2c_rotation, w2c_translation.reshape(3, 1)))
            
            frame_params = [frame_idx, fx, fy, cx, cy, 0, 0] + list(w2c_mat.flatten())
            trajectories.append(frame_params)
            frame_idx += 1
    
    # Convert to Plücker embedding
    plucker_embedding, plucker_embedding_relative = process_poses_to_plucker(
        trajectories,
        width=width,
        height=height,
        original_pose_width=original_pose_width,
        original_pose_height=original_pose_height,
        chunk_size=chunk_size,
        device=device
    )
    
    if return_cam_params:
        return plucker_embedding, plucker_embedding_relative, np.array(trajectories, dtype=np.float32)
    return plucker_embedding, plucker_embedding_relative

def generate_fixed_balanced_trajectory(
    num_frames,
    width=1280,
    height=704,
    original_pose_width=1280,
    original_pose_height=704,
    speed=1.0,
    fx=0.8,
    fy=0.8,
    cx=0.5,
    cy=0.5,
    chunk_size=-1,
    device='cpu',
    return_cam_params=False
):
    """
    Generates a fixed balanced camera trajectory where each of the 8 directions
    (w, a, s, d, i, j, k, l) is traversed once, with frames evenly distributed.

    Args:
        num_frames (int): Total number of frames to generate.
        width (int): Target video width.
        height (int): Target video height.
        original_pose_width (int): Original pose width for intrinsic adjustment.
        original_pose_height (int): Original pose height for intrinsic adjustment.
        speed (float): Controls the intensity of the motion.
        fx, fy, cx, cy: Camera intrinsic parameters.
        device (str): Device for tensor operations ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Plücker embedding tensor of shape [1, num_frames, height, width, 6].
    """
    if num_frames > 81:
        fixed_primitives = ['w', 'a', 's', 'd', 'i', 'j', 'k', 'l']
    else:
        if np.random.random() < 0.5:
            fixed_primitives = ['w', 'a', 's', 'd']
        else:
            fixed_primitives = ['i', 'j', 'k', 'l']
    num_primitives = len(fixed_primitives)

    base_frames_per_action = num_frames // num_primitives
    remainder = num_frames % num_primitives

    motion_sequence = []
    for idx, primitive in enumerate(fixed_primitives):
        action_frames = base_frames_per_action + (1 if idx < remainder else 0)
        if action_frames > 0:
            motion_sequence.append((primitive, action_frames))

    trajectories = []
    current_position = np.zeros(3)
    current_rotation = np.eye(3)

    move_step = speed * 0.05
    rotate_step = speed * math.pi / 180

    frame_idx = 0

    for action, num_action_frames in motion_sequence:
        move_forward = 'w' in action
        move_backward = 's' in action
        move_left = 'a' in action
        move_right = 'd' in action
        look_up = 'i' in action
        look_down = 'k' in action
        look_left = 'j' in action
        look_right = 'l' in action

        for _ in range(num_action_frames):
            pitch_delta = 0
            yaw_delta = 0

            if look_up:
                pitch_delta += rotate_step
            if look_down:
                pitch_delta -= rotate_step
            if look_left:
                yaw_delta -= rotate_step
            if look_right:
                yaw_delta += rotate_step

            if pitch_delta != 0 or yaw_delta != 0:
                rotation_delta = euler_to_rotation_matrix(pitch_delta, yaw_delta, 0)
                current_rotation = current_rotation @ rotation_delta

            local_movement = np.zeros(3)
            if move_forward:
                local_movement[2] += move_step
            if move_backward:
                local_movement[2] -= move_step
            if move_left:
                local_movement[0] -= move_step
            if move_right:
                local_movement[0] += move_step

            world_movement = current_rotation @ local_movement
            current_position += world_movement

            c2w_rotation = current_rotation
            c2w_translation = current_position
            w2c_rotation = c2w_rotation.T
            w2c_translation = -w2c_rotation @ c2w_translation
            w2c_mat = np.hstack((w2c_rotation, w2c_translation.reshape(3, 1)))

            frame_params = [frame_idx, fx, fy, cx, cy, 0, 0] + list(w2c_mat.flatten())
            trajectories.append(frame_params)
            frame_idx += 1

    plucker_embedding, plucker_embedding_relative = process_poses_to_plucker(
        trajectories,
        width=width,
        height=height,
        original_pose_width=original_pose_width,
        original_pose_height=original_pose_height,
        chunk_size=chunk_size,
        device=device
    )

    if return_cam_params:
        return plucker_embedding, plucker_embedding_relative, np.array(trajectories, dtype=np.float32)
    return plucker_embedding, plucker_embedding_relative

def process_poses_to_plucker(
    cam_params,
    width=1280, 
    height=704, 
    original_pose_width=1280, 
    original_pose_height=704, 
    chunk_size=-1,
    device='cpu'
):
    """
    Takes a list of camera pose parameters, processes it, and returns the Plücker embedding.
    This is the core logic, now decoupled from file reading.
    """
    cam_params = [Camera(cam_param) for cam_param in cam_params]

    sample_wh_ratio = width / height
    pose_wh_ratio = original_pose_width / original_pose_height

    if pose_wh_ratio > sample_wh_ratio:
        resized_ori_w = height * pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fx = resized_ori_w * cam_param.fx / width
    else:
        resized_ori_h = width / pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fy = resized_ori_h * cam_param.fy / height

    intrinsic = np.asarray([[cam_param.fx * width,
                            cam_param.fy * height,
                            cam_param.cx * width,
                            cam_param.cy * height]
                            for cam_param in cam_params], dtype=np.float32)
    
    K = torch.as_tensor(intrinsic, device=device)[None]

    c2ws = get_relative_pose(cam_params)
    c2ws = torch.as_tensor(c2ws, device=device)[None]

    plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[0].permute(0, 3, 1, 2).contiguous()
    plucker_embedding = plucker_embedding[None]
    plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b f h w c")[0]


    if chunk_size == -1:
        plucker_embedding_relative = None
    else:
        vae_compression_ratio = 4
        chunk_size_list = [ (chunk_size-1) * vae_compression_ratio + 1] + [ chunk_size * vae_compression_ratio ]*(((len(cam_params)-1)//vae_compression_ratio+1)//chunk_size-1)
        assert sum(chunk_size_list) == len(cam_params)
        start_index = 0
        for c in chunk_size_list:
            if start_index == 0:
                c2ws_chunk = get_relative_pose(cam_params[start_index:start_index+c])
                c2ws_relative = c2ws_chunk
            else:
                # using last chunk as reference
                c2ws_chunk = get_relative_pose(cam_params[start_index-1:start_index+c])
                c2ws_relative = np.concatenate([c2ws_relative, c2ws_chunk[1:] ], axis=0)
            start_index = start_index + c
        c2ws_relative = torch.as_tensor(c2ws_relative, device=device)[None]

        plucker_embedding_relative = ray_condition(K, c2ws_relative, height, width, device=device)[0].permute(0, 3, 1, 2).contiguous()
        plucker_embedding_relative = plucker_embedding_relative[None]
        plucker_embedding_relative = rearrange(plucker_embedding_relative, "b f c h w -> b f h w c")[0]
    
    return plucker_embedding, plucker_embedding_relative


def cam_params_to_y_camera(cam_params, width=1280, height=704, chunk_size=-1, device='cpu'):
    """
    Convert raw camera pose parameters to y_camera tensor ready for model input.

    Args:
        cam_params: numpy array of shape [num_frames, 19], each row is
                    [frame_idx, fx, fy, cx, cy, 0, 0, w2c_mat(12 floats)]
        width: target video width
        height: target video height
        device: torch device

    Returns:
        y_camera: tensor of shape [C=24, F=21, H, W] (without batch dimension)
                  where C=24 is the Plücker embedding channels (6 * 4 temporal compression)
                  and F=21 is the number of latent frames ((81+3)/4 = 21)
                  
        Note: After DataLoader batching, shape becomes [B, C=24, F=21, H, W]
              which matches control_adapter's expected input [bs, c, f, h, w]
    """
    if isinstance(cam_params, np.ndarray):
        cam_params = cam_params.tolist()

    plucker_embedding, plucker_embedding_relative = process_poses_to_plucker(
        cam_params, width=width, height=height, 
        chunk_size=chunk_size, device=device,
    )

    if chunk_size > 0:
        plucker_embedding = plucker_embedding_relative

    # plucker_embedding: [num_frames, H, W, 6]
    # permute to [6, num_frames, H, W]
    control_camera_video = plucker_embedding.permute([3, 0, 1, 2])

    # Repeat first frame 4 times and concat: [6, num_frames+3, H, W]
    control_camera_latents = torch.cat(
        [
            control_camera_video[:, 0:1].repeat(1, 4, 1, 1),
            control_camera_video[:, 1:]
        ], dim=1
    )

    # [6, num_frames+3, H, W] -> [num_frames+3, 6, H, W]
    control_camera_latents = control_camera_latents.transpose(0, 1)

    # Reshape: [num_frames+3, 6, H, W] -> [(num_frames+3)//4, 4, 6, H, W]
    num_frames_padded, channels, frame_height, frame_width = control_camera_latents.shape
    control_camera_latents = control_camera_latents.contiguous().view(
        num_frames_padded // 4, 4, channels, frame_height, frame_width
    ).transpose(1, 2)

    # [(num_frames+3)//4, 6, 4, H, W] -> [(num_frames+3)//4, 24, H, W]
    control_camera_latents = control_camera_latents.contiguous().view(
        num_frames_padded // 4, channels * 4, frame_height, frame_width
    )

    # [21, 24, H, W] -> [24, 21, H, W] = [C, F, H, W]
    # Transpose to match control_adapter's expected [C, F, H, W] format
    # where C=24 (Plücker channels) and F=21 (latent frames)
    # After DataLoader batching: [B, C=24, F=21, H, W]
    control_camera_latents = control_camera_latents.transpose(0, 1)

    return control_camera_latents


def _invert_SE3(transforms):
    """Invert a batch of 4x4 SE(3) matrices."""
    assert transforms.shape[-2:] == (4, 4)
    rotation_inv = transforms[..., :3, :3].transpose(-1, -2)
    result = torch.zeros_like(transforms)
    result[..., :3, :3] = rotation_inv
    result[..., :3, 3] = -torch.einsum('...ij,...j->...i', rotation_inv, transforms[..., :3, 3])
    result[..., 3, 3] = 1.0
    return result


def cam_params_to_prope_dict(cam_params, device='cpu', dtype=torch.float32):
    """
    Convert raw camera pose parameters to a PRoPE camera condition dict.

    Unlike plucker embeddings which encode camera info into spatial feature maps,
    PRoPE passes view matrices and intrinsics directly to the attention layers.

    Args:
        cam_params: list of Camera objects, or numpy array of shape [num_frames, 19]
                    where each row is [frame_idx, fx, fy, cx, cy, 0, 0, w2c_mat(12 floats)]
        device: torch device
        dtype: torch dtype

    Returns:
        dict with:
            'viewmats': tensor of shape [T_latent, 4, 4] (w2c matrices)
            'K': tensor of shape [T_latent, 3, 3] (normalized intrinsics)
    """
    if isinstance(cam_params, np.ndarray):
        cam_params = cam_params.tolist()
    if isinstance(cam_params, list) and not isinstance(cam_params[0], Camera):
        cam_params = [Camera(cp) for cp in cam_params]

    # Align to VAE temporal downsampling (1+4k pattern):
    # Frame 0 is kept, then every 4th frame starting from frame 1
    num_frames = len(cam_params)
    latent_frame_count = 1 + (num_frames - 1) // 4
    aligned_indices = [0] + [1 + 4 * i for i in range(latent_frame_count - 1)]
    cam_params_sub = [cam_params[i] for i in aligned_indices]

    # Compute relative c2w poses for aligned frames
    c2w_poses_aligned = get_relative_pose(cam_params_sub)
    c2ws_sub = torch.as_tensor(c2w_poses_aligned, dtype=dtype, device=device)

    # Invert c2w to get w2c (viewmats)
    viewmats = _invert_SE3(c2ws_sub)  # [T_latent, 4, 4]

    # Use fixed default intrinsics (normalized), matching training config
    num_latent_frames = viewmats.shape[0]
    default_fx_norm = 969.6969696969696 / (960.0 * 2)
    default_fy_norm = 969.6969696969696 / (540.0 * 2)

    k_matrices = torch.zeros((num_latent_frames, 3, 3), dtype=dtype, device=device)
    k_matrices[:, 0, 0] = default_fx_norm
    k_matrices[:, 1, 1] = default_fy_norm
    k_matrices[:, 0, 2] = 0.5
    k_matrices[:, 1, 2] = 0.5
    k_matrices[:, 2, 2] = 1.0

    return {
        'viewmats': viewmats,
        'K': k_matrices,
    }



def _invert_SE3(transforms):
    """Invert a batch of 4x4 SE(3) matrices."""
    assert transforms.shape[-2:] == (4, 4)
    rotation_inv = transforms[..., :3, :3].transpose(-1, -2)
    result = torch.zeros_like(transforms)
    result[..., :3, :3] = rotation_inv
    result[..., :3, 3] = -torch.einsum("...ij,...j->...i", rotation_inv, transforms[..., :3, 3])
    result[..., 3, 3] = 1.0
    return result


def cam_params_to_prope_dict(cam_params, device='cpu', dtype=torch.float32):
    """
    Convert raw camera pose parameters to a PRoPE camera condition dict.

    Unlike plucker embeddings which encode camera info into spatial feature maps,
    PRoPE passes view matrices and intrinsics directly to the attention layers.

    Args:
        cam_params: list of Camera objects, or numpy array of shape [num_frames, 19]
                    where each row is [frame_idx, fx, fy, cx, cy, 0, 0, w2c_mat(12 floats)]
        device: torch device
        dtype: torch dtype

    Returns:
        dict with:
            'viewmats': tensor of shape [T_latent, 4, 4] (w2c matrices)
            'K': tensor of shape [T_latent, 3, 3] (normalized intrinsics)
    """
    if isinstance(cam_params, np.ndarray):
        cam_params = cam_params.tolist()
    if isinstance(cam_params, list) and not isinstance(cam_params[0], Camera):
        cam_params = [Camera(cp) for cp in cam_params]

    # Compute relative c2w poses (all relative to first frame)
    c2w_poses = get_relative_pose(cam_params)
    c2ws = torch.as_tensor(c2w_poses, dtype=dtype, device=device)

    # Align to VAE temporal downsampling (1+4k pattern):
    # Frame 0 is kept, then every 4th frame starting from frame 1
    num_frames = len(cam_params)
    latent_frame_count = 1 + (num_frames - 1) // 4
    aligned_indices = [0] + [1 + 4 * i for i in range(latent_frame_count - 1)]
    cam_params_sub = [cam_params[i] for i in aligned_indices]

    # Recompute relative pose for aligned frames
    c2w_poses_aligned = get_relative_pose(cam_params_sub)
    c2ws_sub = torch.as_tensor(c2w_poses_aligned, dtype=dtype, device=device)

    # Invert c2w to get w2c (viewmats)
    viewmats = _invert_SE3(c2ws_sub)  # [T_latent, 4, 4]

    # Use fixed default intrinsics (normalized), matching training config
    num_latent_frames = viewmats.shape[0]
    default_fx_norm = 969.6969696969696 / (960.0 * 2)
    default_fy_norm = 969.6969696969696 / (540.0 * 2)

    k_matrices = torch.zeros((num_latent_frames, 3, 3), dtype=dtype, device=device)
    k_matrices[:, 0, 0] = default_fx_norm
    k_matrices[:, 1, 1] = default_fy_norm
    k_matrices[:, 0, 2] = 0.5
    k_matrices[:, 1, 2] = 0.5
    k_matrices[:, 2, 2] = 1.0

    return {
        'viewmats': viewmats,
        'K': k_matrices,
    }


def generate_trajectory_from_json(
    trajectory_spec,
    num_frames,
    width=1280,
    height=704,
    original_pose_width=1280,
    original_pose_height=704,
    speed=1.5,
    fx=0.8,
    fy=0.8,
    cx=0.5,
    cy=0.5,
    device='cpu',
    return_cam_params=False
):
    """
    Generates a camera trajectory from a JSON specification.

    The specification is a list of [direction, proportion] pairs, where:
        - direction (str): Motion key(s), e.g. 'w', 'i', 'wj', 'sl'.
            Movement keys: w(forward), s(backward), a(left), d(right)
            View keys: i(look up), k(look down), j(look left), l(look right)
            ' '(stay still)
        - proportion (int/float): Relative weight of this segment in the total sequence.

    Example: [["w", 3], ["i", 5]] means 3/(3+5)=37.5% forward, 5/(3+5)=62.5% look up.

    Args:
        trajectory_spec (list): List of [direction, proportion] pairs.
        num_frames (int): Total number of frames to generate.
        width, height: Target video dimensions.
        original_pose_width, original_pose_height: Original pose dimensions.
        speed (float): Controls the intensity of the motion.
        fx, fy, cx, cy: Camera intrinsic parameters.
        device (str): Device for tensor operations.
        return_cam_params (bool): Whether to return raw camera parameters.

    Returns:
        plucker_embedding: Plücker embedding tensor.
        cam_params (optional): Raw camera parameters as numpy array.
        per_frame_keys (optional): Per-frame motion key list.
    """
    total_weight = sum(item[1] for item in trajectory_spec)

    motion_sequence = []
    allocated_frames = 0
    for idx, (direction, weight) in enumerate(trajectory_spec):
        if idx == len(trajectory_spec) - 1:
            action_frames = num_frames - allocated_frames
        else:
            action_frames = max(1, round(num_frames * weight / total_weight))
            action_frames = min(action_frames, num_frames - allocated_frames)
        if action_frames > 0:
            motion_sequence.append((direction, action_frames))
            allocated_frames += action_frames

    trajectories = []
    current_position = np.zeros(3)
    current_rotation = np.eye(3)

    move_step = speed * 0.05
    rotate_step = speed * math.pi / 180

    frame_idx = 0

    for action, num_action_frames in motion_sequence:
        move_forward = 'w' in action
        move_backward = 's' in action
        move_left = 'a' in action
        move_right = 'd' in action
        look_up = 'i' in action
        look_down = 'k' in action
        look_left = 'j' in action
        look_right = 'l' in action

        for _ in range(num_action_frames):
            pitch_delta = 0
            yaw_delta = 0

            if look_up:
                pitch_delta += rotate_step
            if look_down:
                pitch_delta -= rotate_step
            if look_left:
                yaw_delta -= rotate_step
            if look_right:
                yaw_delta += rotate_step

            if pitch_delta != 0 or yaw_delta != 0:
                rotation_delta = euler_to_rotation_matrix(pitch_delta, yaw_delta, 0)
                current_rotation = current_rotation @ rotation_delta

            local_movement = np.zeros(3)
            if move_forward:
                local_movement[2] += move_step
            if move_backward:
                local_movement[2] -= move_step
            if move_left:
                local_movement[0] -= move_step
            if move_right:
                local_movement[0] += move_step

            world_movement = current_rotation @ local_movement
            current_position += world_movement

            c2w_rotation = current_rotation
            c2w_translation = current_position
            w2c_rotation = c2w_rotation.T
            w2c_translation = -w2c_rotation @ c2w_translation
            w2c_mat = np.hstack((w2c_rotation, w2c_translation.reshape(3, 1)))

            frame_params = [frame_idx, fx, fy, cx, cy, 0, 0] + list(w2c_mat.flatten())
            trajectories.append(frame_params)
            frame_idx += 1

    per_frame_keys = []
    for action, num_action_frames in motion_sequence:
        for _ in range(num_action_frames):
            per_frame_keys.append(action)

    plucker_embedding, _ = process_poses_to_plucker(
        trajectories,
        width=width,
        height=height,
        original_pose_width=original_pose_width,
        original_pose_height=original_pose_height,
        device=device
    )

    if return_cam_params:
        return plucker_embedding, np.array(trajectories, dtype=np.float32), per_frame_keys
    return plucker_embedding