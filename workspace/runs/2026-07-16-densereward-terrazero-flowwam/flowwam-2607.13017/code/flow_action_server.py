"""
WebSocket inference server for the dual-stream flow+RGB world model with an IDM
action expert on RoboTwin.

Per observation the server runs three stages:
  1. Denoise RGB + Flow from noise with the dual-stream video DiT.
  2. Capture the video DiT's per-layer [rgb, flow] hidden states at t=0.
  3. Denoise the action chunk with ``ActionExpertIDM``, whose block i
     cross-attends to captured video layer ``map(i)``.

Protocol (msgpack + numpy):
  1. Server -> Client : metadata dict
  2. Client -> Server : observation dict (images, instruction, qpos)
     Server -> Client : {"actions": ndarray(N, action_dim)}
  3. Client -> Server : {"__reset__": True} between episodes
"""

import asyncio
import argparse
import logging
import os
import sys
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import websockets.asyncio.server
import websockets.frames

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # inference/
REPO_ROOT = os.path.dirname(SCRIPT_DIR)                          # repo root (diffsynth)
for _p in (REPO_ROOT, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from action_dit import ActionExpertIDM, build_action_expert_idm  # noqa: E402
from dual_stream_capture import (  # noqa: E402
    capture_video_layer_features, concat_layer_feats,
)
from dataset_action_robotwin import (  # noqa: E402
    ActionNormStats, tshape_tile, RoboTwinActionFlowDataset,
)

from diffsynth.models.wan_video_dit_dual_stream import FlowStreamModule  # noqa: E402
from diffsynth.pipelines.wan_video_new import WanVideoPipeline  # noqa: E402
from diffsynth.pipelines.wan_video_dual_stream import (  # noqa: E402
    model_fn_wan_video_dual_stream,
)
from diffsynth.schedulers.flow_match import FlowMatchScheduler  # noqa: E402
from diffsynth.data.video import save_video  # noqa: E402

from pipeline_loader import build_pipeline  # noqa: E402
import msgpack_numpy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("flow_action_server")


def load_action_expert_idm(
    checkpoint_path: str,
    action_dim: int = 14,
    num_frames: int = 49,
    text_context_dim: int = 4096,
    video_dim: int = 3072,
    device: torch.device = torch.device("cuda"),
    joint_state_dim: int = 14,
    num_action_layers: int = 30,
    action_expert_dim: int = 1024,
    action_expert_heads: int = 16,
    action_expert_ffn_dim: int = 4096,
    pred_target: str = "velocity",
    use_rope: bool = True,
    proprio_mode: str = "text",
) -> ActionExpertIDM:
    model = build_action_expert_idm(
        action_dim=action_dim,
        num_frames=num_frames,
        video_dim=video_dim,
        text_context_dim=text_context_dim,
        joint_state_dim=joint_state_dim,
        num_layers=num_action_layers,
        dim=action_expert_dim,
        num_heads=action_expert_heads,
        ffn_dim=action_expert_ffn_dim,
        pred_target=pred_target,
        use_rope=use_rope,
        proprio_mode=proprio_mode,
    )
    from diffsynth.models.utils import load_state_dict
    state_dict = load_state_dict(checkpoint_path)
    action_keys = {
        k.replace("action_expert.", ""): v
        for k, v in state_dict.items()
        if k.startswith("action_expert.")
    }
    if not action_keys:
        action_keys = state_dict
    missing, unexpected = model.load_state_dict(action_keys, strict=False)
    log.info(
        f"ActionExpertIDM: loaded {len(action_keys) - len(unexpected)} keys, "
        f"{len(missing)} missing, {len(unexpected)} unexpected"
    )
    return model.to(device=device, dtype=torch.bfloat16).eval()


@torch.no_grad()
def rollout_and_predict_actions(
    pipe: WanVideoPipeline,
    flow_stream: FlowStreamModule,
    action_expert: ActionExpertIDM,
    camera_frames: Dict[str, np.ndarray],
    cameras: List[str],
    instruction: str,
    action_norm: ActionNormStats,
    current_qpos: Optional[np.ndarray] = None,
    num_frames: int = 49,
    num_video_frames: Optional[int] = None,
    size: Tuple[int, int] = (320, 256),
    video_inference_steps: int = 25,
    sigma_shift: float = 5.0,
    action_inference_steps: int = 10,
    action_snr_shift: float = 3.0,
    action_cond_sigma: float = 0.0,
    cond_layer_stride: int = 1,
    seed: int = 1,
    video_save_path: Optional[str] = None,
    cached_text: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[np.ndarray, Tuple[torch.Tensor, torch.Tensor]]:
    vae_z_dim = getattr(pipe.vae, "z_dim", 16)
    w, h = size
    device = pipe.device
    dtype = pipe.torch_dtype

    action_len = num_frames
    video_frames = num_video_frames if num_video_frames is not None else num_frames

    # ---- T-shape tile the latest frame ----
    head = camera_frames[cameras[0]]
    if head.shape[:2] != (h, w):
        head = np.array(Image.fromarray(head).resize((w, h), Image.BICUBIC))
    if len(cameras) >= 3:
        left = camera_frames[cameras[1]]
        right = camera_frames[cameras[2]]
        if left.shape[:2] != (h, w):
            left = np.array(Image.fromarray(left).resize((w, h), Image.BICUBIC))
        if right.shape[:2] != (h, w):
            right = np.array(Image.fromarray(right).resize((w, h), Image.BICUBIC))
        tiled = tshape_tile(head, left, right)
    else:
        tiled = head

    tiled_h, tiled_w = tiled.shape[:2]
    tiled_h, tiled_w, video_frames = pipe.check_resize_height_width(
        tiled_h, tiled_w, video_frames)
    tiled_pil = Image.fromarray(tiled).resize((tiled_w, tiled_h), Image.BICUBIC)
    flow_h, flow_w = tiled_h, tiled_w

    # ---- Text ----
    camera_prefix = RoboTwinActionFlowDataset.CAMERA_PREFIX
    video_prompt = camera_prefix + instruction
    if cached_text is not None:
        context, action_context = cached_text
    else:
        pipe.load_models_to_device(["text_encoder"])
        context = pipe.prompter.encode_prompt(video_prompt, positive=True, device=device)
        action_context = pipe.prompter.encode_prompt(instruction, positive=True, device=device)
    fresh_text_pair = (context, action_context)

    # ---- VAE encode the single conditioning frame ----
    pipe.load_models_to_device(["vae"])
    upscale = pipe.vae.upsampling_factor
    T_lat = (video_frames - 1) // 4 + 1
    rgb_H_lat = tiled_h // upscale
    rgb_W_lat = tiled_w // upscale
    flow_H_lat = flow_h // upscale
    flow_W_lat = flow_w // upscale

    rgb_vid = pipe.preprocess_video([tiled_pil])
    rgb_prefix = pipe.vae.encode(rgb_vid, device=device).to(dtype=dtype, device=device)

    zero_flow_pil = Image.new("RGB", (flow_w, flow_h), (255, 255, 255))
    flow_vid = pipe.preprocess_video([zero_flow_pil])
    flow_prefix = pipe.vae.encode(flow_vid, device=device).to(dtype=dtype, device=device)

    rgb_noise_shape = (1, vae_z_dim, T_lat, rgb_H_lat, rgb_W_lat)
    flow_noise_shape = (1, vae_z_dim, T_lat, flow_H_lat, flow_W_lat)
    rgb_noise = pipe.generate_noise(rgb_noise_shape, seed=seed, rand_device="cpu").to(dtype=dtype, device=device)
    rgb_noise[:, :, :1] = rgb_prefix
    flow_noise = pipe.generate_noise(
        flow_noise_shape, seed=(seed + 1) if seed is not None else None, rand_device="cpu").to(dtype=dtype, device=device)
    flow_noise[:, :, :1] = flow_prefix

    rgb_latents = rgb_noise.clone()
    flow_latents = flow_noise.clone()

    # ---- Stage 1: dual-stream video denoising ----
    pipe.scheduler.set_timesteps(video_inference_steps, shift=sigma_shift)
    pipe.load_models_to_device(pipe.in_iteration_models)
    for progress_id, timestep in enumerate(
        tqdm(pipe.scheduler.timesteps, desc="Video DiT Denoising")
    ):
        t_tensor = timestep.unsqueeze(0).to(dtype=dtype, device=device)
        rgb_pred, flow_pred = model_fn_wan_video_dual_stream(
            dit=pipe.dit,
            flow_stream=flow_stream,
            latents=rgb_latents,
            flow_latents=flow_latents,
            timestep=t_tensor,
            context=context,
            fuse_vae_embedding_in_latents=True,
            use_gradient_checkpointing=False,
        )
        rgb_latents = pipe.scheduler.step(rgb_pred, pipe.scheduler.timesteps[progress_id], rgb_latents)
        flow_latents = pipe.scheduler.step(flow_pred, pipe.scheduler.timesteps[progress_id], flow_latents)
        rgb_latents[:, :, :1] = rgb_prefix
        flow_latents[:, :, :1] = flow_prefix

    if video_save_path is not None:
        os.makedirs(video_save_path, exist_ok=True)
        pipe.load_models_to_device(["vae"])
        save_video(pipe.vae_output_to_video(pipe.vae.decode(rgb_latents, device=device)),
                   os.path.join(video_save_path, "rgb_tiled.mp4"), fps=12)
        save_video(pipe.vae_output_to_video(pipe.vae.decode(flow_latents, device=device)),
                   os.path.join(video_save_path, "flow_head.mp4"), fps=12)
        pipe.load_models_to_device(pipe.in_iteration_models)

    # ---- Stage 2: capture per-layer video features at t=0 (IDM cond) ----
    cond_ts_vec = torch.full((1,), float(action_cond_sigma), dtype=dtype, device=device)
    feats = capture_video_layer_features(
        dit=pipe.dit,
        flow_stream=flow_stream,
        latents=rgb_latents,
        flow_latents=flow_latents,
        timestep=cond_ts_vec,
        context=context,
        fuse_vae_embedding_in_latents=True,
    )
    video_layer_feats = concat_layer_feats(feats, cond_layer_stride)

    # ---- Stage 3: action denoising conditioned on per-layer features ----
    action_device = next(action_expert.parameters()).device
    action_dtype = next(action_expert.parameters()).dtype
    video_layer_feats = [f.to(device=action_device, dtype=action_dtype) for f in video_layer_feats]

    text_ctx = None
    if action_expert.text_context_dim > 0:
        text_ctx = action_context.to(device=action_device, dtype=action_dtype)

    action_scheduler = FlowMatchScheduler(
        num_inference_steps=action_inference_steps, shift=action_snr_shift)
    action_dim = action_expert.action_dim
    action_latents = torch.randn(1, action_len, action_dim, device=action_device, dtype=action_dtype)

    if current_qpos is not None:
        cur = np.asarray(current_qpos, dtype=np.float32).reshape(1, -1)
        if cur.shape[1] != action_dim:
            raise ValueError(f"current_qpos dim {cur.shape[1]} != {action_dim}")
        cur_norm = torch.from_numpy(action_norm.normalize(cur).astype(np.float32)).to(
            device=action_device, dtype=action_dtype)
    else:
        cur_norm = torch.zeros(1, action_dim, device=action_device, dtype=action_dtype)

    action_anchor = cur_norm
    action_latents[:, 0] = action_anchor
    uses_proprio = (
        getattr(action_expert, "joint_state_encoder", None) is not None
        or getattr(action_expert, "proprio_to_text", None) is not None
    )
    joint_state_input = cur_norm if uses_proprio else None
    cond_ts = torch.full((1,), float(action_cond_sigma), device=action_device, dtype=action_dtype)

    for progress_id, timestep in enumerate(
        tqdm(action_scheduler.timesteps, desc="Action Denoising")
    ):
        t_tensor = timestep.unsqueeze(0).to(dtype=action_dtype, device=action_device)
        sigma_t = action_scheduler.sigmas[progress_id].to(
            device=action_device, dtype=action_dtype).reshape(1)
        velocity = action_expert(
            noisy_actions=action_latents,
            timestep=t_tensor.squeeze(),
            video_layer_feats=video_layer_feats,
            text_context=text_ctx,
            joint_state=joint_state_input,
            sigma=sigma_t,
            cond_timestep=cond_ts,
            return_x0=False,
        )
        action_latents = action_scheduler.step(
            velocity, action_scheduler.timesteps[progress_id], action_latents)
        action_latents[:, 0] = action_anchor

    normalized_actions = action_latents[0].float().cpu().numpy()
    raw_actions = action_norm.denormalize(normalized_actions)
    pipe.load_models_to_device([])
    return raw_actions.astype(np.float32), fresh_text_pair


class ConnectionState:
    def __init__(self, pipe, flow_stream, action_expert, action_norm, cameras,
                 num_frames, num_video_frames, size, action_dim, action_snr_shift,
                 action_inference_steps, action_chunk_size, video_inference_steps,
                 sigma_shift, action_cond_sigma=0.0, cond_layer_stride=1,
                 video_save_dir=None, save_videos_mode="off", save_videos_every=20,
                 server_id=None):
        self.pipe = pipe
        self.flow_stream = flow_stream
        self.action_expert = action_expert
        self.action_norm = action_norm
        self.cameras = cameras
        self.num_frames = num_frames
        self.num_video_frames = num_video_frames
        self.size = size
        self.action_dim = action_dim
        self.action_snr_shift = action_snr_shift
        self.action_inference_steps = action_inference_steps
        self.action_chunk_size = action_chunk_size
        self.video_inference_steps = video_inference_steps
        self.sigma_shift = sigma_shift
        self.action_cond_sigma = action_cond_sigma
        self.cond_layer_stride = cond_layer_stride
        self.video_save_dir = video_save_dir
        self.save_videos_mode = save_videos_mode
        self.save_videos_every = max(1, int(save_videos_every))
        self.server_id = server_id
        self._episode_id = 0
        self._step_id = 0
        self._task_name = None
        self._text_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def reset(self, task_name=None):
        self._episode_id += 1
        self._step_id = 0
        if task_name is not None:
            self._task_name = task_name
        self._text_cache.clear()

    def _should_save_videos(self):
        if self.video_save_dir is None or self.save_videos_mode == "off":
            return False
        if self.save_videos_mode == "all":
            return True
        return (self._step_id == 0) and (self._episode_id % self.save_videos_every == 0)

    def infer(self, obs: dict) -> dict:
        images = obs["images"]
        instruction = obs["instruction"]
        current_qpos = obs.get("qpos", None)
        if current_qpos is not None:
            current_qpos = np.asarray(current_qpos, dtype=np.float32)

        video_save_path = None
        if self._should_save_videos():
            parts = [self.video_save_dir]
            if self._task_name:
                parts.append(self._task_name)
            if self.server_id:
                parts.append(self.server_id)
            parts.append(f"episode_{self._episode_id}")
            parts.append(f"step_{self._step_id:04d}")
            video_save_path = os.path.join(*parts)

        cached_text = self._text_cache.get(instruction)
        raw_actions, fresh_text = rollout_and_predict_actions(
            pipe=self.pipe,
            flow_stream=self.flow_stream,
            action_expert=self.action_expert,
            camera_frames=images,
            cameras=self.cameras,
            instruction=instruction,
            action_norm=self.action_norm,
            current_qpos=current_qpos,
            num_frames=self.num_frames,
            num_video_frames=self.num_video_frames,
            size=self.size,
            video_inference_steps=self.video_inference_steps,
            sigma_shift=self.sigma_shift,
            action_inference_steps=self.action_inference_steps,
            action_snr_shift=self.action_snr_shift,
            action_cond_sigma=self.action_cond_sigma,
            cond_layer_stride=self.cond_layer_stride,
            video_save_path=video_save_path,
            cached_text=cached_text,
        )
        if cached_text is None and fresh_text is not None:
            self._text_cache[instruction] = fresh_text
        self._step_id += 1
        return {"actions": raw_actions.astype(np.float32)}


class FlowActionServer:
    def __init__(self, state_kwargs: dict, host: str, port: int):
        self._state_kwargs = state_kwargs
        self._host = host
        self._port = port
        self._metadata = {
            "model": "FlowWAM-DualStream-FlowAction-IDM",
            "action_dim": state_kwargs["action_dim"],
            "num_frames": state_kwargs["num_frames"],
            "cameras": state_kwargs["cameras"],
            "action_chunk_size": state_kwargs["action_chunk_size"],
            "video_inference_steps": state_kwargs["video_inference_steps"],
            "action_inference_steps": state_kwargs["action_inference_steps"],
            "size": list(state_kwargs["size"]),
        }

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        async with websockets.asyncio.server.serve(
            self._handler, self._host, self._port,
            compression=None, max_size=None, ping_interval=None, ping_timeout=None,
        ) as server:
            log.info(f"Serving on ws://{self._host}:{self._port}")
            await server.serve_forever()

    async def _handler(self, websocket):
        log.info(f"Connection from {websocket.remote_address}")
        packer = msgpack_numpy.Packer()
        state = ConnectionState(**self._state_kwargs)
        await websocket.send(packer.pack(self._metadata))
        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                if obs.get("__reset__", False):
                    state.reset(task_name=obs.get("task_name"))
                    await websocket.send(packer.pack({"status": "reset_ok"}))
                    continue
                result = state.infer(obs)
                await websocket.send(packer.pack(result))
            except websockets.ConnectionClosed:
                break
            except Exception:
                tb = traceback.format_exc()
                log.error(tb)
                await websocket.send(tb)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error.")
                raise


def parse_args():
    p = argparse.ArgumentParser(description="Dual-stream flow-action IDM inference server")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--action_norm_path", type=str, default=None)
    p.add_argument("--local_model_path", type=str, default=None,
                   help="Directory containing Wan-AI/ base models. "
                        "Defaults to <inference>/models.")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--checkpoint_mode", type=str, default="full", choices=["lora", "full"])
    p.add_argument("--action_dim", type=int, default=14)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--num_video_frames", type=int, default=None)
    p.add_argument("--num_action_layers", type=int, default=30)
    p.add_argument("--action_expert_dim", type=int, default=1024)
    p.add_argument("--action_expert_heads", type=int, default=16)
    p.add_argument("--action_expert_ffn_dim", type=int, default=4096)
    # The following must match training_config.json of the checkpoint.
    p.add_argument("--action_pred_target", type=str, default="velocity",
                   choices=["velocity", "x0"])
    p.add_argument("--action_pos_mode", type=str, default="rope",
                   choices=["rope", "learned"])
    p.add_argument("--proprio_mode", type=str, default="text",
                   choices=["text", "state_token"])
    p.add_argument("--cameras", type=str, nargs="+",
                   default=["head_camera", "left_camera", "right_camera"])
    p.add_argument("--text_context_dim", type=int, default=4096)
    p.add_argument("--size", type=int, nargs=2, default=[320, 256], metavar=("W", "H"))
    p.add_argument("--video_inference_steps", type=int, default=25)
    p.add_argument("--sigma_shift", type=float, default=5.0)
    p.add_argument("--action_inference_steps", type=int, default=10)
    # Must match training's --action_snr_shift.
    p.add_argument("--action_snr_shift", type=float, default=5.0)
    p.add_argument("--action_cond_sigma", type=float, default=0.0,
                   help="Noise-level label of the (t=0) cond video fed to the "
                        "action expert. 0 treats the generated latent as clean.")
    p.add_argument("--cond_layer_stride", type=int, default=1,
                   help="Must match training's --cond_layer_stride.")
    p.add_argument("--action_chunk_size", type=int, default=49)
    p.add_argument("--save_videos_mode", type=str, default="off",
                   choices=["off", "sample", "all"])
    p.add_argument("--save_videos_every", type=int, default=20)
    p.add_argument("--video_save_dir", type=str, default=None)
    p.add_argument("--server_id", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda":
        args.device = "cuda:0"
    device = torch.device(args.device)

    if args.action_norm_path is None:
        args.action_norm_path = os.path.join(
            os.path.dirname(args.checkpoint), "action_norm_stats.npz")
    if args.local_model_path is None:
        args.local_model_path = os.path.join(SCRIPT_DIR, "models")

    lora_path = args.checkpoint if args.checkpoint_mode == "lora" else None
    full_path = args.checkpoint if args.checkpoint_mode == "full" else None

    log.info("Loading dual-stream pipeline (VAE + T5 + DiT + FlowStream) ...")
    pipe, flow_stream = build_pipeline(
        local_model_path=args.local_model_path,
        device=device, lora_path=lora_path, full_path=full_path,
    )
    video_dim = int(pipe.dit.dim)

    log.info(f"Loading ActionExpertIDM from {args.checkpoint} ...")
    action_expert = load_action_expert_idm(
        checkpoint_path=args.checkpoint,
        action_dim=args.action_dim,
        num_frames=args.num_frames,
        text_context_dim=args.text_context_dim,
        video_dim=video_dim,
        device=device,
        joint_state_dim=args.action_dim,
        num_action_layers=args.num_action_layers,
        action_expert_dim=args.action_expert_dim,
        action_expert_heads=args.action_expert_heads,
        action_expert_ffn_dim=args.action_expert_ffn_dim,
        pred_target=args.action_pred_target,
        use_rope=(args.action_pos_mode == "rope"),
        proprio_mode=args.proprio_mode,
    )

    action_norm = ActionNormStats.load(args.action_norm_path)
    chunk_size = min(args.action_chunk_size, args.num_frames)

    video_save_dir = None
    if args.save_videos_mode != "off":
        epoch_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        video_save_dir = args.video_save_dir or os.path.join(
            os.path.dirname(args.checkpoint), "gen_videos_idm", epoch_stem)
        os.makedirs(video_save_dir, exist_ok=True)

    state_kwargs = dict(
        pipe=pipe, flow_stream=flow_stream, action_expert=action_expert,
        action_norm=action_norm, cameras=args.cameras, num_frames=args.num_frames,
        num_video_frames=args.num_video_frames, size=tuple(args.size),
        action_dim=args.action_dim, action_snr_shift=args.action_snr_shift,
        action_inference_steps=args.action_inference_steps, action_chunk_size=chunk_size,
        video_inference_steps=args.video_inference_steps, sigma_shift=args.sigma_shift,
        action_cond_sigma=args.action_cond_sigma, cond_layer_stride=args.cond_layer_stride,
        video_save_dir=video_save_dir, save_videos_mode=args.save_videos_mode,
        save_videos_every=args.save_videos_every, server_id=args.server_id,
    )
    server = FlowActionServer(state_kwargs=state_kwargs, host=args.host, port=args.port)
    log.info(
        f"Server ready - action_dim={args.action_dim}, num_frames={args.num_frames}, "
        f"num_action_layers={args.num_action_layers}, video_steps={args.video_inference_steps}, "
        f"action_steps={args.action_inference_steps}, cond_sigma={args.action_cond_sigma}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
