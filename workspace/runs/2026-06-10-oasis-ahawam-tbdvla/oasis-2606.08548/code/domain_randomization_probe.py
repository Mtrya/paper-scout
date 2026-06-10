"""
Diagnostic probe for OASIS domain randomization parameters (Appendix B, Table 4).

Purpose:
- Quantify the magnitude of visual diversity injected during offline rendering
- Assess whether the randomization ranges are sufficient to cover real-world variation
- Compute effective sample expansion factor and statistical coverage

Paper claims:
- Each trajectory is rendered into 20 randomized environments
- Lighting contributes the most to sim-to-real transfer (Table 2)
- Success rate saturates around 15-20 renderings per trajectory
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class RandomizationRange:
    name: str
    low: float
    high: float
    unit: str
    distribution: str = "uniform"

    def sample(self, n: int = 1, rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.low, self.high, size=n)

    def relative_range(self) -> float:
        """Normalized span relative to typical operating point (approximate)."""
        if self.unit == "K":
            return (self.high - self.low) / 5500.0  # normalize by mid-range
        if self.unit == "m":
            return (self.high - self.low) / 0.5
        if self.unit == "deg":
            return (self.high - self.low) / 45.0
        return self.high - self.low


# Table 4 from Appendix B
OASIS_RANDOMIZATIONS = {
    "background": {
        "wall_texture": ["Concrete", "Wood", "Terrazzo", "Metal"],
        "floor_texture": ["Concrete", "Wood", "Terrazzo"],
    },
    "material": [
        RandomizationRange("roughness", 0.1, 0.65, ""),
        RandomizationRange("metallic", 0.25, 1.0, ""),
        RandomizationRange("texture_rotation", 0, 45, "deg"),
        RandomizationRange("texture_translation", 0.1, 1.0, ""),
        RandomizationRange("uvw_projection", 0.0, 1.0, ""),  # Bernoulli(0.9) -> treat as prob
    ],
    "lighting": [
        RandomizationRange("dome_intensity", 1000, 3000, "K"),
        RandomizationRange("dome_color_temp", 4500, 6500, "K"),
        RandomizationRange("dome_color_r", 0.85, 1.0, ""),
        RandomizationRange("dome_color_g", 0.85, 1.0, ""),
        RandomizationRange("dome_color_b", 0.85, 1.0, ""),
        RandomizationRange("indoor_intensity", 20000, 200000, "lux"),
        RandomizationRange("indoor_color_temp", 4500, 6500, "K"),
        RandomizationRange("indoor_color_r", 0.85, 1.0, ""),
        RandomizationRange("indoor_color_g", 0.85, 1.0, ""),
        RandomizationRange("indoor_color_b", 0.85, 1.0, ""),
    ],
    "camera": [
        RandomizationRange("pos_offset_x", -0.01, 0.01, "m"),
        RandomizationRange("pos_offset_y", -0.01, 0.01, "m"),
        RandomizationRange("pos_offset_z", -0.01, 0.01, "m"),
        RandomizationRange("rot_offset_roll", -1.5, 1.5, "deg"),
        RandomizationRange("rot_offset_pitch", -1.5, 1.5, "deg"),
        RandomizationRange("rot_offset_yaw", -1.5, 1.5, "deg"),
    ],
}


def estimate_diversity_coverage(renderings_per_traj: int = 20, num_trials: int = 1000) -> dict:
    """
    Estimate how many unique visual configurations are effectively covered.
    Uses birthday-problem style collision estimate for discrete textures
    and volume-fraction for continuous parameters.
    """
    rng = np.random.default_rng(42)

    # Discrete textures
    wall_opts = len(OASIS_RANDOMIZATIONS["background"]["wall_texture"])
    floor_opts = len(OASIS_RANDOMIZATIONS["background"]["floor_texture"])
    discrete_combos = wall_opts * floor_opts  # 4 * 3 = 12

    # Continuous parameters (lighting + material + camera)
    continuous_params: List[RandomizationRange] = []
    continuous_params.extend(OASIS_RANDOMIZATIONS["material"])
    continuous_params.extend(OASIS_RANDOMIZATIONS["lighting"])
    continuous_params.extend(OASIS_RANDOMIZATIONS["camera"])

    total_span = sum(r.high - r.low for r in continuous_params)
    avg_resolution = 0.01  # assume 1% discrimination threshold
    continuous_volume = np.prod([
        (r.high - r.low) / avg_resolution for r in continuous_params
    ])

    return {
        "renderings_per_traj": renderings_per_traj,
        "discrete_texture_combinations": discrete_combos,
        "num_continuous_params": len(continuous_params),
        "total_param_span": total_span,
        "effective_continuous_volume_log10": np.log10(continuous_volume),
        "collision_prob_20_samples": 1 - np.prod([
            (discrete_combos - i) / discrete_combos for i in range(renderings_per_traj)
        ]) if renderings_per_traj <= discrete_combos else 1.0,
    }


def lighting_dominance_analysis() -> dict:
    """
    Paper says lighting is the single most important factor (Table 2).
    Quantify its relative range compared to camera and material randomization.
    """
    lighting_ranges = OASIS_RANDOMIZATIONS["lighting"]
    camera_ranges = OASIS_RANDOMIZATIONS["camera"]
    material_ranges = OASIS_RANDOMIZATIONS["material"]

    def aggregate_importance(ranges: List[RandomizationRange]) -> float:
        # Heuristic: sum of normalized ranges
        return sum(r.relative_range() for r in ranges)

    return {
        "lighting_importance_score": aggregate_importance(lighting_ranges),
        "camera_importance_score": aggregate_importance(camera_ranges),
        "material_importance_score": aggregate_importance(material_ranges),
        "lighting_vs_camera_ratio": aggregate_importance(lighting_ranges) / max(
            aggregate_importance(camera_ranges), 1e-6
        ),
    }


def compare_with_viral_randomization() -> dict:
    """
    VIRAL (He et al., 2025) randomizes lighting, materials, camera parameters,
    image quality, and sensor delays. OASIS does not randomize image quality
    or sensor delays, focusing purely on rendering conditions.
    """
    return {
        "oasis_factors": ["texture", "lighting", "camera_extrinsics"],
        "viral_factors": ["lighting", "materials", "camera", "image_quality", "sensor_delays"],
        "oasis_missing": ["image_quality", "sensor_delays", "domain_randomization_during_rl"],
        "note": "VIRAL trains with online DAgger + domain randomization during RL; "
                "OASIS uses fixed teleop trajectories + offline rendering randomization.",
    }


if __name__ == "__main__":
    print("=" * 60)
    print("OASIS Domain Randomization Diagnostic")
    print("=" * 60)

    div = estimate_diversity_coverage(renderings_per_traj=20)
    print("\n1. Diversity Coverage Estimate")
    for k, v in div.items():
        print(f"   {k}: {v:.3f}" if isinstance(v, float) else f"   {k}: {v}")

    imp = lighting_dominance_analysis()
    print("\n2. Factor Importance (heuristic normalized spans)")
    for k, v in imp.items():
        print(f"   {k}: {v:.3f}" if isinstance(v, float) else f"   {k}: {v}")

    comp = compare_with_viral_randomization()
    print("\n3. Comparison with VIRAL")
    for k, v in comp.items():
        print(f"   {k}: {v}")

    print("\n" + "=" * 60)
    print("Key finding: OASIS's lighting randomization spans 4500-6500K")
    print("(color temp) and 1000-3000K (dome intensity), which covers")
    print("typical indoor-to-warm-lighting variation. Camera extrinsics")
    print("are very small (+/- 1 cm position, +/- 1.5 deg rotation),")
    print("suggesting the policy is expected to be robust to minor")
    print("mounting tolerances rather than large viewpoint changes.")
    print("=" * 60)
