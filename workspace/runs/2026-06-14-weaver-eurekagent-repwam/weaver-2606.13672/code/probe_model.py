"""
WEAVER architecture probe.

Goals:
1. Reconstruct the released model architecture from the HF config and count
   parameters by component.
2. Run a tiny CPU-synthetic forward pass to make the flow-matching objective,
   reward/critic losses, and latent-rollout generation concrete.
3. Inspect the diffusion-forcing pyramid schedule and the ReFlow NFE reduction.
"""

import json
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Disable torch.compile inside weaver.wm.model so the probe stays fast on CPU.
# ---------------------------------------------------------------------------
torch.compile = lambda f=None, *args, **kwargs: f if f is not None else (lambda x: x)

import yaml

# Add the checked-out repo to the path so we can import `weaver`.
REPO_ROOT = Path(__file__).resolve().parents[4] / "code" / "weaver-2606.13672"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weaver.utils.config import dict_to_namespace
from weaver.wm.model import WEAVER


# ---------------------------------------------------------------------------
# Dummy encoders: keep the VAE/CLIP weights off the critical path.
# ---------------------------------------------------------------------------
class DummyImageEncoder(torch.nn.Module):
    def __init__(self, spatial_size: int, image_size):
        super().__init__()
        self.spatial_size = spatial_size
        self.image_size = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        # SD3 VAE has 16 latent channels.
        self._feature_dim = 16 * spatial_size * spatial_size

    @property
    def feature_dim(self):
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use precomputed features for this probe.")

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Return a blank RGB image of the right resolution."""
        bt = latents.shape[0]
        return torch.zeros(bt, 3, self.image_size[0], self.image_size[1])


class DummyTaskEncoder(torch.nn.Module):
    def __init__(self, feature_dim: int = 768):
        super().__init__()
        self._feature_dim = feature_dim

    @property
    def feature_dim(self):
        return self._feature_dim

    def forward(self, x):
        if isinstance(x, dict):
            return x["features"]
        if isinstance(x, torch.Tensor):
            return x
        raise ValueError(type(x))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_config(path: Path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_model(cfg_dict, device="cpu", image_encoder=None, task_encoder=None):
    cfg = dict_to_namespace(deepcopy(cfg_dict))
    img_keys = cfg.dataset.img_keys
    image_size = tuple(cfg.dataset.image_size)

    if image_encoder is None:
        image_encoder = DummyImageEncoder(
            spatial_size=cfg.im_encoder.spatial_size,
            image_size=image_size,
        )
    if task_encoder is None:
        task_encoder = DummyTaskEncoder()

    n_states = getattr(cfg.dataset, "n_states", 8)
    n_actions = getattr(cfg.dataset, "n_actions", 8)

    model = WEAVER(
        img_keys=img_keys,
        im_encoder=image_encoder,
        train_decoder=False,
        task_encoder=task_encoder,
        n_history=cfg.n_history,
        n_horizon=cfg.horizon,
        config=cfg.model,
        n_states=n_states,
        n_actions=n_actions,
        use_precomputed_features=True,
        image_size=image_size,
        device=device,
        n_memory_frames=cfg.n_memory_frames,
        t_memory=cfg.t_memory,
        inference_config=cfg.inference,
    )
    return model.to(device), cfg


def count_params(model: torch.nn.Module):
    total = 0
    breakdown = {}
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        breakdown[name] = n
        total += n
    return total, breakdown


def make_synthetic_batch(
    B: int,
    T: int,
    img_keys,
    n_states: int,
    n_actions: int,
    image_size,
    spatial_size: int,
    device="cpu",
):
    latent_h = image_size[0] // 8 // spatial_size
    latent_w = image_size[1] // 8 // spatial_size
    patches_per_img = latent_h * latent_w
    feature_dim = 16 * spatial_size * spatial_size

    obs = {"states": torch.randn(B, T, n_states, device=device)}
    for key in img_keys:
        obs[f"{key}_features"] = torch.randn(B, T, patches_per_img, feature_dim, device=device)

    actions = torch.randn(B, T, n_actions, device=device)
    tasks = {"features": torch.randn(B, 768, device=device)}
    gt_rewards = torch.rand(B, T, device=device) * 2.0 - 1.0
    return obs, actions, tasks, gt_rewards


def nfe_for_schedule(model, horizon: int):
    """Number of function evaluations under the current pyramid settings."""
    schedule = model._build_pyramid_schedule(horizon)
    return schedule.shape[0] - 1


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------
def main():
    results = {}
    device = "cpu"

    # ---- 1. Released-model parameter count (no VAE download) ----------------
    released_cfg_path = REPO_ROOT / "hf_configs" / "WEAVER" / "config.yaml"
    if not released_cfg_path.exists():
        # Fall back to the default repo config if the HF config was not fetched.
        released_cfg_path = REPO_ROOT / "weaver" / "config.yaml"

    released_cfg = load_config(released_cfg_path)
    released_model, released_ns = build_model(released_cfg, device=device)
    released_total, released_breakdown = count_params(released_model)

    results["released_model"] = {
        "config_path": str(released_cfg_path),
        "image_size": released_ns.dataset.image_size,
        "n_embed": released_ns.model.n_embed,
        "n_layers": released_ns.model.n_layers,
        "n_heads": released_ns.model.n_heads,
        "n_spatial": released_ns.model.n_spatial,
        "diff_forcing": released_ns.model.diff_forcing,
        "use_sprint": released_ns.model.use_sprint,
        "sprint_drop_ratio": released_ns.model.sprint_drop_ratio,
        "n_memory_frames": released_ns.n_memory_frames,
        "t_memory": released_ns.t_memory,
        "total_params": released_total,
        "breakdown": {k: int(v) for k, v in released_breakdown.items()},
    }
    print("\n=== Released WEAVER architecture ===")
    print(f"Total trainable parameters: {released_total:,}")
    for k, v in released_breakdown.items():
        print(f"  {k:20s}: {v:>14,} ({100 * v / released_total:.1f}%)")

    # ---- 2. Tiny CPU-synthetic forward pass ---------------------------------
    tiny_cfg = deepcopy(released_cfg)
    tiny_cfg["dataset"]["image_size"] = [96, 160]
    tiny_cfg["im_encoder"]["spatial_size"] = 4
    tiny_cfg["n_history"] = 2
    tiny_cfg["horizon"] = 2
    tiny_cfg["n_memory_frames"] = 0
    tiny_cfg["model"]["n_embed"] = 384
    tiny_cfg["model"]["n_layers"] = 4
    tiny_cfg["model"]["n_heads"] = 4
    tiny_cfg["model"]["n_hidden"] = 512
    tiny_cfg["model"]["use_sprint"] = False
    tiny_cfg["model"]["diff_forcing"] = False
    tiny_cfg["model"]["loss_target"] = "v-pred"
    tiny_cfg["model"]["val_steps"] = 4
    tiny_cfg["inference"]["pyramid_schedule"] = "linear"
    tiny_cfg["inference"]["pyramid_stagger_width"] = 0

    tiny_model, tiny_ns = build_model(tiny_cfg, device=device)
    tiny_total, tiny_breakdown = count_params(tiny_model)

    B, T = 1, tiny_ns.n_history + tiny_ns.horizon
    n_states = getattr(tiny_ns.dataset, "n_states", 8)
    n_actions = getattr(tiny_ns.dataset, "n_actions", 8)
    obs, actions, tasks, gt_rewards = make_synthetic_batch(
        B=B,
        T=T,
        img_keys=tiny_ns.dataset.img_keys,
        n_states=n_states,
        n_actions=n_actions,
        image_size=tiny_ns.dataset.image_size,
        spatial_size=tiny_ns.im_encoder.spatial_size,
        device=device,
    )

    tiny_model.train()
    t0 = time.time()
    total_loss, log_dict = tiny_model(
        obs=obs,
        actions=actions,
        tasks=tasks,
        gt_rewards=gt_rewards,
        update_rm=True,
    )
    train_time = time.time() - t0

    results["tiny_forward"] = {
        "total_params": tiny_total,
        "batch": B,
        "time_steps": T,
        "train_time_sec": train_time,
        "losses": {k: round(v.item(), 6) for k, v in log_dict.items()},
    }
    print("\n=== Tiny synthetic training forward ===")
    print(f"Tiny params: {tiny_total:,}")
    print(f"Forward time: {train_time:.3f}s")
    for k, v in log_dict.items():
        print(f"  {k:30s}: {v.item():.6f}")

    # ---- 3. Pyramid / diffusion-forcing schedule inspection -----------------
    # Use the released model's settings but evaluate on a small horizon.
    released_model.eval()
    for horizon, val_steps, stagger, schedule, label in [
        (8, 16, 1, "cosine", "released_50nfe_setting"),
        (8, 16, 1, "linear", "released_16nfe_setting"),
        (8, 4, 0, "cosine", "reflow_4nfe_setting"),
    ]:
        released_model._inference_steps = val_steps
        released_model._pyramid_stagger_width = stagger
        released_model._pyramid_schedule_type = schedule
        released_model._pyramid_schedule_cache.clear()
        sched = released_model._build_pyramid_schedule(horizon)
        nfe = sched.shape[0] - 1
        results.setdefault("schedules", {})[label] = {
            "horizon": horizon,
            "val_steps": val_steps,
            "stagger_width": stagger,
            "schedule": schedule,
            "schedule_shape": list(sched.shape),
            "nfe": nfe,
            "first_row": sched[0].tolist(),
            "last_row": sched[-1].tolist(),
        }
        print(f"\n=== Schedule {label} ===")
        print(f"shape: {tuple(sched.shape)} -> NFE = {nfe}")
        print(f"first row t: {sched[0].tolist()}")
        print(f"last  row t: {sched[-1].tolist()}")

    # ---- 5. Save evidence ---------------------------------------------------
    out_dir = Path(__file__).resolve().parent / "probe_outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "probe_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved probe results to {out_path}")


if __name__ == "__main__":
    main()
