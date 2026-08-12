"""Synthetic 2-link planar arm manipulation videos for the GeniWorld interface probe.

Generates scripted "reach -> grasp -> lift -> transport -> place" clips with a
2-link arm + gripper and a colored cube, on 64x64 RGB frames, T=16 frames.

Design goals:
- deterministic, cheap, no physics engine needed (scripted kinematics)
- full control over train vs OOD scenes (background patterns)
- three action representations emitted per clip:
    * numeric : (T, 3) joint angles + gripper, Ctrl-World style
    * render   : per-frame arm-only render (GeniWorld "visual action")
    * render_static : arm at mean pose, repeated (spatial grounding, no motion)
    * render_shuffle: per-frame renders with frame order permuted (marginal
      statistics match, temporal motion destroyed)

The cube is scripted: it sticks to the gripper while "grasped". Ground-truth
cube positions per frame are saved for functional evaluation (task success).
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw


def draw_arm_into(img, theta1, theta2, gripper, color=(255, 255, 255), width=3):
    """Draw the 2-link arm on a copy of `img`. Returns the new image array."""
    H, W = img.shape[:2]
    base = np.array([W * 0.5, H - 6])
    L1, L2 = 16.0, 13.0
    # link1: from base at angle theta1 (measured from vertical up, CCW)
    a1 = theta1
    j1 = base + L1 * np.array([np.sin(a1), -np.cos(a1)])
    a2 = theta1 + theta2
    ee = j1 + L2 * np.array([np.sin(a2), -np.cos(a2)])
    pts = [tuple(base), tuple(j1), tuple(ee)]
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    d.line([pts[0], pts[1]], fill=color, width=width)
    d.line([pts[1], pts[2]], fill=color, width=width)
    d.ellipse([pts[0][0]-3, pts[0][1]-3, pts[0][0]+3, pts[0][1]+3], fill=color)
    d.ellipse([pts[1][0]-2, pts[1][1]-2, pts[1][0]+2, pts[1][1]+2], fill=color)
    # gripper: small rectangle at end effector
    g = 4 if gripper > 0.5 else 8
    d.line([(ee[0]-g, ee[1]), (ee[0]+g, ee[1])], fill=color, width=2)
    d.line([(ee[0]-g, ee[1]-3), (ee[0]-g, ee[1]+3)], fill=color, width=2)
    d.line([(ee[0]+g, ee[1]-3), (ee[0]+g, ee[1]+3)], fill=color, width=2)
    return np.asarray(pil)


def draw_scene(background, cube_pos, cube_size=7, cube_color=(255, 0, 0)):
    H, W = background.shape[:2]
    img = background.copy()
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    # table
    d.rectangle([0, H-10, W, H-4], fill=(90, 90, 90))
    # cube
    cx, cy = cube_pos
    d.rectangle([cx-cube_size//2, cy-cube_size//2, cx+cube_size//2, cy+cube_size//2],
                fill=cube_color)
    return np.asarray(pil)


def background(kind, H=64, W=64, rng=None):
    rng = rng or np.random
    if kind == "plain":    # train
        c = rng.integers(30, 120)
        return np.full((H, W, 3), c, dtype=np.uint8)
    if kind == "checker":  # train
        base = rng.integers(30, 120)
        sq = 8
        img = np.full((H, W, 3), base, dtype=np.uint8)
        off = (rng.integers(0, 255), rng.integers(0, 255), rng.integers(0, 255))
        for i in range(0, H, sq):
            for j in range(0, W, sq):
                if (i // sq + j // sq) % 2 == 0:
                    img[i:i+sq, j:j+sq] = np.clip(np.array(off) * 0.7, 0, 255).astype(np.uint8)
        return img
    if kind == "grad":     # train
        base = rng.integers(30, 120)
        grad = np.linspace(base, np.clip(base + rng.integers(-60, 60), 5, 200), W)
        return np.stack([grad]*H, axis=0)[:, :, None].repeat(3, axis=-1).astype(np.uint8)
    if kind == "stripes":  # OOD
        img = np.zeros((H, W, 3), dtype=np.uint8)
        c1 = rng.integers(40, 200); c2 = rng.integers(40, 200)
        for j in range(0, W, 4):
            img[:, j:j+2] = c1
            img[:, j+2:j+4] = c2
        return img
    if kind == "dots":     # OOD
        base = rng.integers(100, 220)
        img = np.full((H, W, 3), base, dtype=np.uint8)
        for _ in range(40):
            x, y = rng.integers(0, W), rng.integers(0, H)
            img[y, x] = (0, 0, 0)
        return img
    raise ValueError(kind)


def gen_clip(T, W, H, bg_kind, rng):
    """Returns dict with frames, action numeric, renders, cube positions."""
    L1, L2 = 16.0, 13.0

    def fk(t1, t2):
        base = np.array([W * 0.5, H - 6])
        j1 = base + L1 * np.array([np.sin(t1), -np.cos(t1)])
        ee = j1 + L2 * np.array([np.sin(t1 + t2), -np.cos(t1 + t2)])
        return j1, ee

    # sample cube start / target on the table
    start_x = rng.uniform(W*0.22, W*0.40); start_y = H - 16
    targ_x = rng.uniform(W*0.60, W*0.78); targ_y = H - 16

    # scripted phases (indices within T)
    T_reach, T_grasp, T_lift, T_move, T_place, T_back = 4, 1, 2, 6, 1, 2
    # solve an approximate IK for start/target via simple numerical search
    def ik(px, py):
        base = np.array([W * 0.5, H - 6])
        d = np.clip(np.linalg.norm(np.array([px, py]) - base) / (L1 + L2), 0.05, 0.95)
        r = d * (L1 + L2)
        # pick orientation "elbow up"
        cos_t2 = np.clip((r**2 - L1**2 - L2**2) / (2*L1*L2), -1, 1)
        t2 = np.arccos(cos_t2)  # positive (elbow up)
        ang = np.arctan2(px - base[0], base[1] - py)  # from vertical
        # t1 = ang - atan2(L2 sin t2, L1 + L2 cos t2)
        t1 = ang - np.arctan2(L2 * np.sin(t2), L1 + L2 * np.cos(t2))
        return t1, t2

    t1s, t2s, grips = [], [], []
    cxs, cys = [], []
    cube = np.array([start_x, start_y])
    target = np.array([targ_x, targ_y])
    ik1 = ik(start_x, start_y)
    ik2 = ik(targ_x, targ_y)
    rest = (0.0, -0.5)
    for t in range(T):
        if t < T_reach:            # rest -> start
            u = (t + 1) / T_reach
            t1 = rest[0] + (ik1[0] - rest[0]) * u
            t2 = rest[1] + (ik1[1] - rest[1]) * u
            grip = 0.0
        elif t < T_reach + T_grasp:  # grasp
            t1, t2 = ik1; grip = 1.0
        elif t < T_reach + T_grasp + T_lift:  # lift
            t1, t2 = ik1; grip = 1.0
            cube = np.array([start_x, start_y - 4])
        elif t < T_reach + T_grasp + T_lift + T_move:  # transport (linear interp)
            u = (t - (T_reach + T_grasp + T_lift)) / T_move
            t1 = ik1[0] + (ik2[0] - ik1[0]) * u
            t2 = ik1[1] + (ik2[1] - ik1[1]) * u
            grip = 1.0
            cube = np.array([start_x, start_y - 4]) + (target - np.array([start_x, start_y - 4])) * u
        elif t < T_reach + T_grasp + T_lift + T_move + T_place:  # place
            t1, t2 = ik2; grip = 1.0
            cube = np.array([targ_x, targ_y - 4])
        elif t < T_reach + T_grasp + T_lift + T_move + T_place + T_back:  # release+back
            t1, t2 = ik2; grip = 0.0
            cube = np.array([targ_x, targ_y - 3])
        else:
            u = (t - (T_reach + T_grasp + T_lift + T_move + T_place + T_back)) / max(T_back, 1)
            t1 = ik2[0] + (rest[0] - ik2[0]) * min(u, 1.0)
            t2 = ik2[1] + (rest[1] - ik2[1]) * min(u, 1.0)
            grip = 0.0
        t1s.append(t1); t2s.append(t2); grips.append(grip)
        cxs.append(cube[0]); cys.append(cube[1])

    bg = background(bg_kind, H, W, rng)
    frames, renders = [], []
    for t in range(T):
        scene = draw_scene(bg, (cxs[t], cys[t]))
        arm_only = draw_arm_into(np.zeros_like(bg), t1s[t], t2s[t], grips[t],
                                 color=(255, 255, 255), width=3)
        frames.append(draw_arm_into(scene, t1s[t], t2s[t], grips[t],
                                    color=(200, 200, 200), width=3))
        renders.append(arm_only)
    frames = np.stack(frames)          # (T,H,W,3)
    renders = np.stack(renders)        # (T,H,W,3)
    # end-effector trajectory (spatial numeric action, Ctrl-World-ish)
    base = np.array([W * 0.5, H - 6])
    ees = []
    for t in range(T):
        j1 = base + L1 * np.array([np.sin(t1s[t]), -np.cos(t1s[t])])
        ee = j1 + L2 * np.array([np.sin(t1s[t] + t2s[t]), -np.cos(t1s[t] + t2s[t])])
        ees.append([ee[0] / W, ee[1] / H])
    numeric = np.stack([np.array(ees)[:, 0], np.array(ees)[:, 1], np.array(grips)], axis=-1)  # (T,3)
    joints = np.stack([np.array(t1s), np.array(t2s), np.array(grips)], axis=-1)  # (T,3)

    mean_i = T // 2
    static = np.stack([renders[mean_i]] * T)
    order = rng.permutation(T)
    shuffled = renders[order]

    cube_pos = np.stack([np.array(cxs), np.array(cys)], axis=-1)  # (T,2)
    return dict(frames=frames, renders=renders, numeric=numeric, joints=joints,
                static=static, shuffled=shuffled, cube=cube_pos)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "arm_data"
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    T, H, W = 16, 64, 64

    splits = {
        "train": (["plain", "checker", "grad"], 2400),
        "val": (["plain", "checker", "grad"], 200),
        "ood_stripes": (["stripes"], 300),
        "ood_dots": (["dots"], 300),
    }
    for split, (bgs, n) in splits.items():
        d = os.path.join(out_dir, split)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            bg = bgs[rng.integers(0, len(bgs))]
            clip = gen_clip(T, W, H, bg, rng)
            np.savez_compressed(os.path.join(d, f"{i:05d}.npz"),
                                frames=clip["frames"], renders=clip["renders"],
                                numeric=clip["numeric"], joints=clip["joints"],
                                static=clip["static"],
                                shuffled=clip["shuffled"], cube=clip["cube"])
        print(f"{split}: {n} clips done")
    print("total:", sum(v[1] for v in splits.values()))


if __name__ == "__main__":
    main()
