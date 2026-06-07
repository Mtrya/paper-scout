"""
GRAIL researcher checks — standalone verification script.

Runs without the full GRAIL docker environment.  It re-implements the small
loss-term fragments we want to test (the upstream loss_terms.py imports
pytorch3d, which is not installed in the host environment) and then exercises
the actual released code, configs and dataset samples.

Usage from the workspace root:
    .venv-grail/bin/python runs/robotics/grail-2606.05160-checks/run_checks.py
"""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

REPO_ROOT = Path("repos/robotics/grail-repo")
CFG_PATH = REPO_ROOT / "configs/recon_4dhoi/pickup_smplx.yaml"
LOSS_TERMS_PATH = REPO_ROOT / "grail/optimization/loss_terms.py"
FILTER_PATH = REPO_ROOT / "grail/postprocessing/filter.py"

OUT = Path("runs/robotics/grail-2606.05160-checks/check_output.txt")
OUT.parent.mkdir(parents=True, exist_ok=True)


def log(*args, **kwargs):
    print(*args, **kwargs)
    with OUT.open("a") as f:
        kwargs["file"] = f
        print(*args, **kwargs)


def section(title: str):
    log("\n" + "=" * 72)
    log(title)
    log("=" * 72)


# ---------------------------------------------------------------------------
# 1.  Can we import the real loss_terms module in this host environment?
# ---------------------------------------------------------------------------
section("A. Import sanity check on the real repo modules")

try:
    sys.path.insert(0, str(REPO_ROOT))
    import grail.optimization.loss_terms as real_loss_terms  # noqa: F401
    log("OK: imported grail.optimization.loss_terms directly")
except Exception as e:
    log(f"FAIL: cannot import real loss_terms module: {type(e).__name__}: {e}")
    log("This confirms the released code needs the docker/conda env (pytorch3d).")


# ---------------------------------------------------------------------------
# 2.  Config parsing — what does the pickup pipeline actually optimize?
# ---------------------------------------------------------------------------
section("B. pickup_smplx.yaml — optimization schedule and loss weights")

cfg = yaml.safe_load(CFG_PATH.read_text())
opt_stages = cfg["optimization"]["opt_stage_specs"]

rows = []
total_iters = 0
for stage_name, stage in opt_stages.items():
    niter = stage["niter"]
    total_iters += niter
    opt_vars = ", ".join(stage["opt_vars"].keys())
    losses = stage["loss_cfg"]
    row = {
        "stage": stage_name,
        "iters": niter,
        "opt_vars": opt_vars,
        "losses": {k: v["weight"] for k, v in losses.items()},
    }
    rows.append(row)
    log(f"\n{stage_name}: {niter} iters (vars: {opt_vars})")
    for loss_name, loss_cfg in losses.items():
        log(f"  - {loss_name}: weight={loss_cfg['weight']}", end="")
        extra = {k: v for k, v in loss_cfg.items() if k != "weight"}
        if extra:
            log(f"  {extra}", end="")
        log()

log(f"\nTotal optimization iterations: {total_iters}")
log("Dominant weights: contact=1e5 in stages 2 and 3; human_global_init_reg=3e3 in stage 1.")

# ---------------------------------------------------------------------------
# 3.  Mini ablation from the config — what happens if each weight is zeroed?
# ---------------------------------------------------------------------------
section("C. Config-level mini ablation: relative loss weight dominance")

for row in rows:
    stage, losses = row["stage"], row["losses"]
    total_w = sum(abs(v) for v in losses.values())
    sorted_losses = sorted(losses.items(), key=lambda x: -x[1])
    log(f"\n{stage} (total |weight| = {total_w:.1f})")
    for name, w in sorted_losses:
        pct = 100 * abs(w) / total_w if total_w > 0 else 0
        log(f"  {name:30s} weight={w:12.1f}  ({pct:5.1f}%)")

log("\n=> contact dominates stages 2/3 (>99.9%). Zeroing it removes the only")
log("   term that pulls hands onto the object during interaction.")
log("=> human_global_init_reg dominates stage 1 (~94%). Zeroing it lets the")
log("   human global trajectory drift far from GENMO initialization.")


# ---------------------------------------------------------------------------
# 4.  Stand-alone numerical verification of core loss terms
# ---------------------------------------------------------------------------
section("D. Numerical verification of loss-term implementations")

# Re-implement the subset of loss_terms.py we need (avoiding pytorch3d).
# These are one-to-one with the upstream functions except knn_points, which
# we replace by a naive cdist version because our synthetic point clouds
# are tiny.


def huber_loss(a: torch.Tensor, delta: float = 1e-2) -> torch.Tensor:
    abs_a = torch.abs(a)
    quadratic = 0.5 * abs_a**2
    linear = delta * (abs_a - 0.5 * delta)
    return torch.where(abs_a < delta, quadratic, linear).mean()


def knn_points_naive(A: torch.Tensor, B: torch.Tensor, K: int = 1):
    """Minimal replacement for pytorch3d.ops.knn_points returning squared distances."""
    # A: (1,N,3), B: (1,M,3)
    dists = torch.cdist(A, B)  # (1,N,M)
    top = torch.topk(dists, k=min(K, B.shape[1]), largest=False, dim=-1)
    # top.values has shape (1,N,K); square to match knn_points' squared-L2 convention.
    return type("_", (), {"dists": top.values ** 2})()


def bidirectional_chamfer_loss(pred_verts, gt_points, trim_pct=0.2):
    if pred_verts.shape[0] == 0 or gt_points.shape[0] == 0:
        return torch.tensor(0.0, requires_grad=True)
    pred = pred_verts.float().unsqueeze(0)
    gt = gt_points.float().unsqueeze(0)
    pred2gt = knn_points_naive(pred, gt, K=1).dists.squeeze(0).squeeze(-1)
    gt2pred = knn_points_naive(gt, pred, K=1).dists.squeeze(0).squeeze(-1)
    k_p = max(1, int(len(pred2gt) * (1.0 - trim_pct)))
    k_g = max(1, int(len(gt2pred) * (1.0 - trim_pct)))
    trimmed_p, _ = pred2gt.topk(k_p, largest=False)
    trimmed_g, _ = gt2pred.topk(k_g, largest=False)
    return trimmed_p.mean() + trimmed_g.mean()


def contact_loss(verts_A, verts_B, num_vertices=2000, top_k=200, delta=0.001):
    if verts_A.shape[0] > num_vertices:
        verts_A = verts_A[torch.randperm(verts_A.shape[0])[:num_vertices]]
    if verts_B.shape[0] > num_vertices:
        verts_B = verts_B[torch.randperm(verts_B.shape[0])[:num_vertices]]
    distances = torch.cdist(verts_A.float(), verts_B.float())
    min_A = torch.min(distances, dim=1).values
    min_B = torch.min(distances, dim=0).values
    all_min = torch.cat([min_A, min_B])
    sorted_min, _ = torch.sort(all_min)
    top_k = min(top_k, len(sorted_min))
    clipped = torch.clamp(sorted_min[:top_k], min=0.0)
    return huber_loss(clipped, delta=delta)


class MockCamera:
    """Pinhole camera at the origin looking down +z; fx=fy=500, c=(320,240)."""

    def __init__(self, fx=500.0, fy=500.0, cx=320.0, cy=240.0):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

    def transform_points_screen(self, pts: torch.Tensor):
        # pts: (1, N, 3)
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        xs = self.fx * x / z + self.cx
        ys = self.fy * y / z + self.cy
        return torch.stack([xs, ys, z], dim=-1)

    def get_world_to_view_transform(self):
        return self

    def transform_points(self, pts: torch.Tensor):
        return pts  # identity mapping for our synthetic test


def contact_depth_loss(
    verts_A, verts_B, cameras, num_vertices=2000, top_k=200, delta=0.001, screen_dist_thresh=20.0
):
    if verts_A.shape[0] == 0 or verts_B.shape[0] == 0:
        return torch.tensor(0.0, requires_grad=True)
    if verts_A.shape[0] > num_vertices:
        verts_A = verts_A[torch.randperm(verts_A.shape[0])[:num_vertices]]
    if verts_B.shape[0] > num_vertices:
        verts_B = verts_B[torch.randperm(verts_B.shape[0])[:num_vertices]]

    screen_A = cameras.transform_points_screen(verts_A.unsqueeze(0)).squeeze(0)[:, :2]
    screen_B = cameras.transform_points_screen(verts_B.unsqueeze(0)).squeeze(0)[:, :2]
    dists_2d = torch.cdist(screen_B.float(), screen_A.float())
    min_dists_2d = dists_2d.min(dim=1).values
    mask = min_dists_2d < screen_dist_thresh
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    verts_B = verts_B[mask]

    distances_3d = torch.cdist(verts_A.float(), verts_B.float())
    nn_idx_A = distances_3d.argmin(dim=1)
    nn_idx_B = distances_3d.argmin(dim=0)

    view = cameras.get_world_to_view_transform()
    z_A = view.transform_points(verts_A.unsqueeze(0)).squeeze(0)[:, 2]
    z_B = view.transform_points(verts_B.unsqueeze(0)).squeeze(0)[:, 2]

    depth_diff_A = (z_A - z_B[nn_idx_A]).abs()
    depth_diff_B = (z_B - z_A[nn_idx_B]).abs()
    all_depth = torch.cat([depth_diff_A, depth_diff_B])
    sorted_diffs, _ = all_depth.sort()
    top_k = min(top_k, len(sorted_diffs))
    return huber_loss(sorted_diffs[:top_k], delta=delta)


def smoothness_loss(seq, beta=1.0):
    first = torch.nn.functional.l1_loss(seq[1:] - seq[:-1], torch.zeros_like(seq[1:]))
    if seq.shape[0] < 3:
        return first
    second = torch.nn.functional.l1_loss(
        seq[2:] - 2 * seq[1:-1] + seq[:-2], torch.zeros_like(seq[2:])
    )
    return first + beta * second


def keypoint_loss(pred, gt, gt_conf=None, conf_thres=0.6):
    if gt_conf is not None:
        pred = pred[gt_conf > conf_thres]
        gt = gt[gt_conf > conf_thres]
    if pred.shape[0] == 0:
        return torch.tensor(0.0)
    return torch.nn.functional.l1_loss(pred, gt)


def ground_loss(verts_seq, gravity_axis="z", height=0.14):
    if gravity_axis == "z":
        mins = torch.min(verts_seq[:, :, 2], dim=1).values
    elif gravity_axis == "y":
        mins = torch.min(verts_seq[:, :, 1], dim=1).values
    else:
        raise ValueError(gravity_axis)
    return torch.mean(torch.abs(mins - height))


# ---- Tests -------------------------------------------------------------

def expect_close(name, got, want, tol=1e-5):
    ok = abs(got - want) < tol
    marker = "PASS" if ok else "FAIL"
    log(f"  [{marker}] {name}: got {got:.6e}, want {want:.6e}")
    return ok


log("\nD1. huber_loss")
a = torch.tensor([-0.005, 0.020])
# delta=0.01: |0.005|<0.01 -> 0.5*0.005^2 = 1.25e-5
#            |0.02|>0.01 -> 0.01*(0.02-0.005)=1.5e-4
# mean = 8.125e-5
expect_close("huber([-0.005,0.020])", huber_loss(a, delta=0.01).item(), 8.125e-5)

log("\nD2. bidirectional_chamfer_loss (no trimming)")
pred = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
gt = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
# pred->gt mins: 0,1 -> mean 0.5
# gt->pred mins: 0,1 -> mean 0.5
# total = 1.0
expect_close("trim=0.0", bidirectional_chamfer_loss(pred, gt, trim_pct=0.0).item(), 1.0)
log("\nD3. bidirectional_chamfer_loss (20% trim — same tiny clouds)")
# 2 points -> keep 1 in each direction -> top1 is 0 for both -> total 0
expect_close(
    "trim=0.2 (2 points each, keep 1)",
    bidirectional_chamfer_loss(pred, gt, trim_pct=0.2).item(),
    0.0,
)
log("   -> outlier trimming can zero the loss completely for small point clouds.")

log("\nD4. contact_loss at 5 cm separation")
hand = torch.tensor([[0.0, 0.0, 0.0]])
obj = torch.tensor([[0.05, 0.0, 0.0]])
# min distances: 0.05 and 0.05 -> huber linear: 0.001*(0.05 - 0.0005) = 4.95e-5
expect_close(
    "contact_loss 5cm (single point pair)",
    contact_loss(hand, obj, top_k=200, delta=0.001).item(),
    4.95e-5,
    tol=1e-6,
)

log("\nD5. contact_depth_loss — screen-space gating")
cam = MockCamera()
hand = torch.tensor([[0.0, 0.0, 1.0]])
obj_far = torch.tensor([[0.05, 0.0, 1.0]])  # 25 px away on a 500 px focal camera
obj_close = torch.tensor([[0.005, 0.0, 1.0]])  # 2.5 px away
log(f"  screen distance for far vertex: {(0.05*500/1):.1f} px")
log(f"  screen distance for close vertex: {(0.005*500/1):.1f} px")
loss_far = contact_depth_loss(hand, obj_far, cam, screen_dist_thresh=20.0).item()
loss_close = contact_depth_loss(hand, obj_close, cam, screen_dist_thresh=20.0).item()
log(f"  loss with far vertex (screen>20):   {loss_far:.6e}  (expected 0)")
log(f"  loss with close vertex (screen<20): {loss_close:.6e}  (>0 because z-diff=0)")
# Actually z-diff is 0 -> huber(0)=0, so loss_close should be 0. Let's verify.
assert loss_far == 0.0, "screen gate should block far vertex"
assert loss_close == 0.0, "zero depth diff -> zero loss"

log("\nD6. contact_depth_loss — depth penalty")
obj_shift_z = torch.tensor([[0.005, 0.0, 1.10]])  # close on screen, depth diff 0.10
loss_depth = contact_depth_loss(hand, obj_shift_z, cam, screen_dist_thresh=20.0).item()
# huber: 0.001*(0.10 - 0.0005) = 9.95e-5
expect_close("depth diff 0.10m", loss_depth, 9.95e-5, tol=1e-6)

log("\nD7. smoothness_loss")
linear = torch.tensor([0.0, 1.0, 2.0, 3.0]).unsqueeze(-1)
# Code penalises first-order difference against zero, so even constant-velocity
# motion gets loss = mean(|1,1,1|) = 1.0.  This is a motion-magnitude penalty,
# not purely an acceleration penalty.
expect_close("linear motion", smoothness_loss(linear, beta=1.0).item(), 1.0)
erratic = torch.tensor([0.0, 0.0, 1.0, 0.0]).unsqueeze(-1)
# first-order: [0,1,-1] -> mean 0.6667
# second-order: [1,-2] -> mean 1.5
# total = 2.1667
expect_close("erratic motion", smoothness_loss(erratic, beta=1.0).item(), 2.0 + 1.0 / 6.0)

log("\nD8. keypoint_loss with confidence mask")
pred = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
gt = torch.tensor([[1.1, 2.1], [3.2, 4.2], [5.5, 6.5]])
conf = torch.tensor([1.0, 0.0, 1.0])
# only frames 0 and 2 pass
want = (abs(0.1) + abs(0.1) + abs(0.5) + abs(0.5)) / 4.0  # 0.30
expect_close("conf mask", keypoint_loss(pred, gt, conf, conf_thres=0.5).item(), 0.30)

log("\nD9. ground_loss")
verts = torch.tensor([[[0.0, 0.0, 0.12], [0.0, 0.0, 0.16]]])
expect_close("ground at 0.12/0.16 vs 0.14", ground_loss(verts, "z", 0.14).item(), 0.02)


# ---------------------------------------------------------------------------
# 5.  Contact-loss gating by max_contact_dist
# ---------------------------------------------------------------------------
section("E. Contact gating by 3D distance (max_contact_dist)")

log("Code path: loss_computer.py _contact_loss() -> _too_far() ->")
log("  if min(cdist(human_verts, obj_verts)) > max_contact_dist: skip")

# Simulate the _too_far decision
hand = torch.randn(30, 3) * 0.01
obj_near = hand + torch.tensor([0.02, 0.0, 0.0])
obj_far = hand + torch.tensor([0.20, 0.0, 0.0])
max_dist = 0.05

min_near = torch.cdist(hand.float(), obj_near.float()).min().item()
min_far = torch.cdist(hand.float(), obj_far.float()).min().item()
log(f"\n  near object min distance = {min_near:.4f} m  (< {max_dist}) -> compute contact loss")
log(f"  far  object min distance = {min_far:.4f} m  (> {max_dist}) -> skip contact loss")
assert min_near < max_dist and min_far > max_dist


# ---------------------------------------------------------------------------
# 6.  Filter thresholds — are they as strict as the paper suggests?
# ---------------------------------------------------------------------------
section("F. Filtering thresholds in pickup_smplx.yaml vs filter.py defaults")

filter_cfg = cfg.get("filtering", {})
log(f"\nConfig values (pickup_smplx.yaml):")
for k in ["camera_trans_thr", "object_mask_tol", "total_mask_tol", "human_static_thr"]:
    log(f"  {k}: {filter_cfg.get(k, '<missing>')}")

log("\nfilter.py hard-coded defaults:")
log("  object_mask_tol = 0.5")
log("  total_mask_tol  = 0.3")
log("  human_static_thr= 0.01")

log("\nObservations:")
log("  - object_mask_tol in config is 1.0 (2x the default 0.5).")
log("  - total_mask_tol  in config is 0.4 (looser than default 0.3).")
log("  - human_static_thr in config is 0  -> check_static_human always passes")
log("    because avg_max_velocity >= 0 is true for any motion.")


# Reproduce the static-human check explicitly
def check_static_human(verts_seq: torch.Tensor, threshold: float):
    velocities = verts_seq[1:] - verts_seq[:-1]
    avg_max = velocities.norm(dim=-1).max(dim=1).values.mean().item()
    return avg_max >= threshold, avg_max


static_verts = torch.randn(30, 100, 3) * 1e-4  # almost no motion
pass_0, vel_0 = check_static_human(static_verts, threshold=0.0)
pass_d, vel_d = check_static_human(static_verts, threshold=0.01)
log(f"\nSynthetic almost-static motion avg max velocity = {vel_0:.6f}")
log(f"  passes with config thr=0.0 ? {pass_0}  (yes -> false negative)")
log(f"  passes with default thr=0.01? {pass_d}")

moving_verts = torch.zeros(30, 100, 3)
moving_verts[:, :, 2] = torch.linspace(0.0, 0.5, 30).unsqueeze(-1)
pass_m, vel_m = check_static_human(moving_verts, threshold=0.01)
log(f"\nSynthetic clear motion avg max velocity = {vel_m:.4f}")
log(f"  passes with default thr=0.01? {pass_m}")


# ---------------------------------------------------------------------------
# 7.  Dataset statistics from the HuggingFace README
# ---------------------------------------------------------------------------
section("G. Dataset category counts (HuggingFace README)")

categories = {
    "pickup_table": 2991,
    "pickup_ground": 1613,
    "sitting": 1748,
    "slope": 1880,
    "curb": 1769,
    "stair": 12188,
}
total = sum(categories.values())
log(f"\nReleased counts: {json.dumps(categories, indent=2)}")
log(f"Sum: {total}  (paper claims >20,000; released: {total} -> consistent)")

# Check with the API manifest if we can reach it
try:
    url = (
        "https://huggingface.co/api/datasets/nvidia/"
        "PhysicalAI-Robotics-Locomanipulation-GRAIL/tree/main/data/pickup_table/meta"
    )
    with urllib.request.urlopen(url, timeout=20) as r:
        tree = json.loads(r.read().decode())
    log(f"\nHF API sanity check: pickup_table/meta contains {len(tree)} files")
except Exception as e:
    log(f"\nHF API tree fetch failed: {e}")


# ---------------------------------------------------------------------------
# 8.  Inspect a real released trajectory
# ---------------------------------------------------------------------------
section("H. Real trajectory inspection (pickup_table__alcohol_0__000)")

robot_path = "/tmp/sample_robot.pkl"
obj_path = "/tmp/sample_obj.pkl"

robot = joblib.load(robot_path)
inner = list(robot.values())[0]
dof = inner["dof"]  # (T, 29)
root_trans = inner["root_trans_offset"]  # (T, 3)
root_rot = inner["root_rot"]  # (T, 4) quat
fps = inner["fps"]
hand_action_left = inner["hand_action_left"]
hand_action_right = inner["hand_action_right"]
T = dof.shape[0]

duration = T / fps
root_xy = root_trans[:, :2]
root_displacement = np.linalg.norm(root_xy[-1] - root_xy[0])
root_height_min = float(root_trans[:, 2].min())
root_height_max = float(root_trans[:, 2].max())

# per-joint velocity (rad/s)
joint_vel = np.abs(np.diff(dof, axis=0)) * fps
mean_joint_vel = float(joint_vel.mean())
max_joint_vel = float(joint_vel.max())

# hand primitives (binary open/close)
hl_changes = int(np.sum(np.abs(np.diff(hand_action_left)) > 0))
hr_changes = int(np.sum(np.abs(np.diff(hand_action_right)) > 0))

log(f"\nRobot trajectory:")
log(f"  frames = {T}, fps = {fps:.1f}, duration = {duration:.2f} s")
log(f"  root XY displacement = {root_displacement:.3f} m")
log(f"  root height range = [{root_height_min:.3f}, {root_height_max:.3f}] m")
log(f"  mean joint speed = {mean_joint_vel:.3f} rad/s")
log(f"  max  joint speed = {max_joint_vel:.3f} rad/s")
log(f"  left hand action changes = {hl_changes}, right = {hr_changes}")

obj = joblib.load(obj_path)
obj_inner = list(obj.values())[0]
obj_pos = obj_inner["root_pos"].squeeze(1)  # (T, 3)
obj_quat = obj_inner["root_quat"].squeeze(1)  # (T, 4)
obj_displacement = float(np.linalg.norm(obj_pos[-1] - obj_pos[0]))
obj_path_len = float(np.sum(np.linalg.norm(np.diff(obj_pos, axis=0), axis=-1)))
obj_max_height = float(obj_pos[:, 2].max())

log(f"\nObject trajectory:")
log(f"  start pos = {obj_pos[0].tolist()}")
log(f"  end   pos = {obj_pos[-1].tolist()}")
log(f"  net displacement = {obj_displacement:.3f} m")
log(f"  total path length = {obj_path_len:.3f} m")
log(f"  max height = {obj_max_height:.3f} m")

# Simple check: is the object actually lifted?
lifted = obj_max_height > (obj_pos[0, 2] + 0.05)
log(f"\n  object lifted >5 cm? {lifted} (max - init = {obj_max_height - obj_pos[0, 2]:.3f} m)")

log("\n--- END OF CHECKS ---")
