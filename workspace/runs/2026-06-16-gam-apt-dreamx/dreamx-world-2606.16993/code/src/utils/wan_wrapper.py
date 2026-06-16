import types
from typing import List, Optional
import torch
import os

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.t5 import umt5_xxl
from wan.modules.vae_2_2 import _video_vae as _video_vae_2_2


class WanTextEncoder(torch.nn.Module):
    def __init__(self, model_name: str = "Wan2.2-TI2V-5B-Camera", model_root_path: str = "",
                 text_encoder_path: str = None, tokenizer_path: str = None):
        super().__init__()
        if text_encoder_path is None:
            text_encoder_path = os.path.join(model_root_path, f"wan_models/{model_name}/models_t5_umt5-xxl-enc-bf16.pth")
        if tokenizer_path is None:
            tokenizer_path = os.path.join(model_root_path, f"wan_models/{model_name}/google/umt5-xxl/")

        self.text_encoder = umt5_xxl(
            encoder_only=True, return_tokenizer=False,
            dtype=torch.float32, device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(text_encoder_path, map_location='cpu', weights_only=False)
        )
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=512, clean='whitespace')

    @property
    def device(self):
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0
        return {"prompt_embeds": context}


class WanVAEWrapper(torch.nn.Module):
    def __init__(self, model_root_path="", vae_path: str = None):
        super().__init__()
        mean = [
            -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
            -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
            -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
            -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
            -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
            0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
        ]
        std = [
            0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
            0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
            0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
            0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
            0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
            0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        if vae_path is None:
            vae_path = os.path.join(model_root_path, "wan_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
        self.model = _video_vae_2_2(
            pretrained_path=vae_path,
            z_dim=48, temperal_downsample=[False, True, True]
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        output = [self.model.encode(u.unsqueeze(0), scale).float().squeeze(0) for u in pixel]
        output = torch.stack(output, dim=0)
        return output.permute(0, 2, 1, 3, 4)

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        zs = latent.permute(0, 2, 1, 3, 4)
        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        decode_fn = self.model.cached_decode if use_cache else self.model.decode
        output = []
        with torch.autocast(device_type=device.type, dtype=dtype):
            for u in zs:
                output.append(decode_fn(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        return output.permute(0, 2, 1, 3, 4)


class WanDiffusionCameraWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.2-TI2V-5B-Camera",
            timestep_shift=5.0,
            is_causal=True,
            local_attn_size=12,
            sink_size=3,
            model_root_path="",
            model_config_path: str = None,
            **kwargs,
    ):
        super().__init__()

        from wan.modules.causal_camera_model_2_2_prope_infinity import CausalWanModel

        num_output_frames = kwargs.get('num_output_frames', 21)
        eprope = kwargs.get('eprope', False)
        attn_compress = 4 if eprope else 1

        if model_config_path is None:
            model_config_path = os.path.join(model_root_path, f"wan_models/{model_name}/config.json")
        self.model = CausalWanModel.from_config(
            model_config_path,
            local_attn_size=local_attn_size, sink_size=sink_size, attn_compress=attn_compress)
        self.model.eval()

        self.scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.seq_len = 880 * num_output_frames
        self._bind_scheduler_methods()

    def _bind_scheduler_methods(self):
        self.scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, self.scheduler)
        self.scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, self.scheduler)
        self.scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, self.scheduler)

    def get_scheduler(self) -> SchedulerInterface:
        return self.scheduler

    def _convert_flow_pred_to_x0(self, flow_pred, xt, timestep):
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps])
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        return (xt - sigma_t * flow_pred).to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        y_camera,
        timestep: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None,
        cache_update_policy: str = "commit_detached",
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]
        skip_length = noisy_image_or_video.shape[-1] * noisy_image_or_video.shape[-2] // 4
        original_timestep = timestep[:, ::skip_length]

        y_camera_input = y_camera if (y_camera is None or isinstance(y_camera, dict)) else y_camera.permute(0, 2, 1, 3, 4)

        flow_pred = self.model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4),
            t=timestep,
            context=prompt_embeds,
            y=y.permute(0, 2, 1, 3, 4) if y is not None else None,
            y_camera=y_camera_input,
            seq_len=self.seq_len,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            cache_start=cache_start,
            cache_update_policy=cache_update_policy,
        ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred.flatten(0, 1),
            noisy_image_or_video.flatten(0, 1),
            original_timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0
