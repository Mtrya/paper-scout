"""Synthetic 2-link planar arm transport clips with TWO strategy modes.

Extends the GeniWorld-probe generator with:
- mode 0: direct transport (low lift, linear move)  [like the morning data]
- mode 1: high-arc detour (lift 10px higher, move via midpoint at height)
Both modes share the same rest pose, same start cube, same target, and the
same first frame (mode only diverges after frame 0) -> the future video is
bimodal given the observation, which is exactly what the FM-vs-L2 and the
plan-injection probes need.

Outputs per clip (npz):
  frames  (16,64,64,3) uint8   full expert episode, frame 0 = observation
  joints  (16,3) float32       (t1,t2,grip) keyframes, 1 per frame
  action  (32,3) float32       expert action chunk: 2 control steps per frame
  cube    (16,2) float32       scripted cube position per frame
  target  (2,)  float32        goal cube position
  mode    ()    int
The planprobe split stores pairs of clips (mode 0, mode 1) with identical
first frames and targets.
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw


def draw_arm_into(img, theta1, theta2, gripper, color=(255, 255, 255), width=3):
    H, W = img.shape[:2]
    base = np.array([W * 0.5, H - 6])
    L1, L2 = 16.0, 13.0
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
    d.rectangle([0, H-10, W, H-4], fill=(90, 90, 90))
    cx, cy = cube_pos
    d.rectangle([cx-cube_size//2, cy-cube_size//2, cx+cube_size//2, cy+cube_size//2],
                fill=cube_color)
    return np.asarray(pil)


def background(kind, H=64, W=64, rng=None):
    rng = rng or np.random
    if kind == "plain":
        c = rng.integers(30, 120)
        return np.full((H, W, 3), c, dtype=np.uint8)
    if kind == "checker":
        base = rng.integers(30, 120)
        sq = 8
        img = np.full((H, W, 3), base, dtype=np.uint8)
        off = (rng.integers(0, 255), rng.integers(0, 255), rng.integers(0, 255))
        for i in range(0, H, sq):
            for j in range(0, W, sq):
                if (i // sq + j // sq) % 2 == 0:
                    img[i:i+sq, j:j+sq] = np.clip(np.array(off) * 0.7, 0, 255).astype(np.uint8)
        return img
    if kind == "grad":
        base = rng.integers(30, 120)
        grad = np.linspace(base, np.clip(base + rng.integers(-60, 60), 5, 200), W)
        return np.stack([grad]*H, axis=0)[:, :, None].repeat(3, axis=-1).astype(np.uint8)
    raise ValueError(kind)


def gen_episode(T, W, H, bg_kind, mode, rng, start=None, targ=None):
    """Scripted episode. Returns dict(frames, joints, action, cube, target, mode)."""
    L1, L2 = 16.0, 13.0
    base = np.array([W * 0.5, H - 6])

    def fk(t1, t2):
        j1 = base + L1 * np.array([np.sin(t1), -np.cos(t1)])
        ee = j1 + L2 * np.array([np.sin(t1 + t2), -np.cos(t1 + t2)])
        return j1, ee

    if start is None:
        start = np.array([rng.uniform(W*0.22, W*0.40), H - 16])
    if targ is None:
        targ = np.array([rng.uniform(W*0.60, W*0.78), H - 16])

    def ik(px, py):
        d = np.clip(np.linalg.norm(np.array([px, py]) - base) / (L1 + L2), 0.05, 0.95)
        r = d * (L1 + L2)
        cos_t2 = np.clip((r**2 - L1**2 - L2**2) / (2*L1*L2), -1, 1)
        t2 = np.arccos(cos_t2)
        ang = np.arctan2(px - base[0], base[1] - py)
        t1 = ang - np.arctan2(L2 * np.sin(t2), L1 + L2 * np.cos(t2))
        return t1, t2

    # waypoints in (x, y) cube-center space
    if mode == 0:  # direct
        lift_h = 4
        wps = [(start[0], start[1]), (start[0], start[1] - lift_h),
               (targ[0], targ[1] - lift_h), (targ[0], targ[1] - 3)]
        wp_times = [3, 4, 12, 14]          # frame indices when each wp is reached
    elif mode == 1:  # high-arc detour
        mid = (start[0] + targ[0]) / 2.0
        lift_h = 10
        wps = [(start[0], start[1]), (start[0], start[1] - lift_h),
               (mid, H - 30), (targ[0], targ[1] - lift_h), (targ[0], targ[1] - 3)]
        wp_times = [3, 5, 8, 12, 14]
    else:
        raise ValueError(mode)

    def wp_at(t):
        if t <= wp_times[0]:
            return np.array(wps[0], dtype=float)
        for i in range(len(wp_times) - 1):
            t0, t1 = wp_times[i], wp_times[i + 1]
            if t0 <= t <= t1:
                u = (t - t0) / max(t1 - t0, 1)
                return np.array(wps[i]) + (np.array(wps[i + 1]) - np.array(wps[i])) * u
        return np.array(wps[-1], dtype=float)

    T_reach, T_grasp = 4, 1
    rest = (0.0, -0.5)
    t1s, t2s, grips, cxs, cys = [], [], [], [], []
    grip = 0.0
    for t in range(T):
        if t < T_reach:
            u = (t + 1) / T_reach
            ik0 = ik(start[0], start[1])
            t1 = rest[0] + (ik0[0] - rest[0]) * u
            t2 = rest[1] + (ik0[1] - rest[1]) * u
        elif t < T_reach + T_grasp:
            t1, t2 = ik(start[0], start[1])
            grip = 1.0
        else:
            c = wp_at(t)
            t1, t2 = ik(c[0], c[1])
        if t >= T_reach + T_grasp + 11:  # release + back after place phase
            grip = 0.0
        if t < T_reach + T_grasp:
            grip = 0.0
        # grip: hold during transport, release at end
        if T_reach + T_grasp <= t < T_reach + T_grasp + 11:
            grip = 1.0
        cube = wp_at(t) if t >= T_reach + T_grasp else np.array([start[0], start[1]])
        t1s.append(t1); t2s.append(t2); grips.append(grip)
        cxs.append(cube[0]); cys.append(cube[1])

    bg = background(bg_kind, H, W, rng) if isinstance(bg_kind, str) else bg_kind
    frames = []
    for t in range(T):
        scene = draw_scene(bg, (cxs[t], cys[t]))
        frames.append(draw_arm_into(scene, t1s[t], t2s[t], grips[t],
                                    color=(200, 200, 200), width=3))
    frames = np.stack(frames).astype(np.uint8)
    joints = np.stack([np.array(t1s), np.array(t2s), np.array(grips)], axis=-1).astype(np.float32)
    # action chunk: 2 control steps per frame, linear interpolation of joints,
    # padded to 32 steps with a final hold
    n_steps = (T - 1) * 2 + 2
    action = np.zeros((n_steps, 3), dtype=np.float32)
    for s in range((T - 1) * 2):
        f = s / 2.0
        f0, u = int(np.floor(f)), f - int(np.floor(f))
        f1 = min(f0 + 1, T - 1)
        action[s] = joints[f0] * (1 - u) + joints[f1] * u
    action[(T - 1) * 2:] = joints[-1]
    cube_pos = np.stack([np.array(cxs), np.array(cys)], axis=-1).astype(np.float32)
    return dict(frames=frames, joints=joints, action=action, cube=cube_pos,
                target=targ.astype(np.float32), mode=mode, bg=bg,
                start=np.array([start[0], start[1]], dtype=np.float32))


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "joint_data"
    rng = np.random.default_rng(0)
    T, H, W = 16, 64, 64
    bgs = ["plain", "checker", "grad"]

    os.makedirs(out_dir, exist_ok=True)
    for split, n in [("train", 2000), ("val", 200)]:
        d = os.path.join(out_dir, split)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            ep = gen_episode(T, W, H, bgs[rng.integers(0, len(bgs))],
                             int(rng.integers(0, 2)), rng)
            np.savez_compressed(os.path.join(d, f"{i:05d}.npz"), **ep)
        print(f"{split}: {n} clips done")

    # planprobe: pairs sharing first frame + target, different modes
    d = os.path.join(out_dir, "planprobe")
    os.makedirs(d, exist_ok=True)
    for i in range(120):
        bg = background(bgs[rng.integers(0, len(bgs))], H, W, rng)
        start = np.array([rng.uniform(W*0.22, W*0.40), H - 16])
        targ = np.array([rng.uniform(W*0.60, W*0.78), H - 16])
        epA = gen_episode(T, W, H, bg, 0, rng, start=start, targ=targ)
        epB = gen_episode(T, W, H, bg, 1, rng, start=start, targ=targ)
        assert np.array_equal(epA["frames"][0], epB["frames"][0]), "first frames must match"
        np.savez_compressed(os.path.join(d, f"{i:05d}_A.npz"), **epA)
        np.savez_compressed(os.path.join(d, f"{i:05d}_B.npz"), **epB)
    print("planprobe: 120 pairs done")
    print("total done")


if __name__ == "__main__":
    main()
