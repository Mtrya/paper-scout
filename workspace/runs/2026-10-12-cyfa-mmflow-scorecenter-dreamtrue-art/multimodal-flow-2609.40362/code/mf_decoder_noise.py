# Vendored verbatim (body unchanged) from hustvl/Multimodal-Flow, commit
# c84caf50fc3da57e30d5984a4d99dee81f8b0c43, file src/mf/latents/decoder_noise.py,
# sha256 d2b471f53b0c075e5bca7f555eaffbded7959da0b79dfc1be132e05346f136e3.
# Only this header comment was added so that the round-trip probe is runnable
# without a local clone of the repository. Upstream license: Apache-2.0.
from typing import NamedTuple

import torch
from torch import Tensor

DECODER_P_MEAN = 0.8
DECODER_P_STD = 0.8
DECODER_NOISE_SCALE = 1.0


class DecoderCorruption(NamedTuple):
    """Decoder input plus the random state needed to verify its construction."""

    corrupted: Tensor
    lambda_: Tensor
    noise: Tensor


def corrupt_text_decoder_latents(
    raw: Tensor,
    generator: torch.Generator,
    *,
    p_mean: float = DECODER_P_MEAN,
    p_std: float = DECODER_P_STD,
    noise_scale: float = DECODER_NOISE_SCALE,
) -> DecoderCorruption:
    """Corrupt raw T5 latents with ELF-compatible tokenwise lambda and unit noise."""
    if not isinstance(raw, Tensor):
        raise ValueError("raw must be a torch.Tensor")
    if raw.ndim < 1:
        raise ValueError("raw must have a batch dimension")
    if not torch.is_floating_point(raw):
        raise ValueError(f"raw must have a floating-point dtype; got {raw.dtype}")
    if not isinstance(generator, torch.Generator):
        raise ValueError("generator must be a torch.Generator")

    detached_raw = raw.detach()
    compute_dtype = (
        torch.float32 if raw.dtype in (torch.float16, torch.bfloat16) else raw.dtype
    )
    lambda_ = torch.randn(
        raw.shape[:-1],
        device=raw.device,
        dtype=compute_dtype,
        generator=generator,
    )
    lambda_ = torch.sigmoid(lambda_ * p_std + p_mean).to(dtype=raw.dtype)
    noise = torch.randn(
        raw.shape,
        device=raw.device,
        dtype=compute_dtype,
        generator=generator,
    ).to(dtype=raw.dtype)
    if raw.dtype in (torch.float16, torch.bfloat16):
        zero = torch.zeros((), device=raw.device, dtype=raw.dtype)
        one = torch.ones((), device=raw.device, dtype=raw.dtype)
        lambda_ = torch.clamp(
            lambda_,
            min=torch.nextafter(zero, one),
            max=torch.nextafter(one, zero),
        )

    weight = lambda_.to(dtype=compute_dtype).unsqueeze(-1)
    corrupted = (
        weight * detached_raw.to(dtype=compute_dtype)
        + (1.0 - weight) * noise.to(dtype=compute_dtype) * noise_scale
    ).to(dtype=raw.dtype)
    return DecoderCorruption(corrupted=corrupted, lambda_=lambda_, noise=noise)
