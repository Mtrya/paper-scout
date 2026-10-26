#!/usr/bin/env python
"""Build the input variants used by the Ropedia W0 action-interface audit.

Variants written under <out>/<name>/ with the standard clip layout
(video.mp4, data.npz, skeleton.mp4, prompt.txt):

  perm_RICE_JCAM    first frame + text from agibot_rice_cooker, camera + skeleton from
                    xp10m_jigsaw_puzzle  (does a static gripper scene start moving like a human
                    head-mounted camera?)
  perm_PLAY_PUSH    first frame + text from xp_teleop_playdoh_tub, camera + skeleton from
                    omega_home_push_chair (same embodiment, different trajectory and hand path)
  synthcam_RICE     first frame + text + skeleton from agibot_rice_cooker, camera replaced by a
                    synthetic strafe-and-yaw trajectory (the clip's own camera is exactly static)
  skelshift_RICE    agibot_rice_cooker with its skeleton translated 40 px right
  skelshuffle_RICE  the same clip with frame 0 fixed and frames 1..80 permuted (seed 0)

usage: python make_variants.py --examples examples --out variants
"""
import argparse, json, os, shutil

import numpy as np
from PIL import Image
import imageio


def copy_clip(src, dst, camera_from=None, skeleton_from=None, prompt_from=None):
    os.makedirs(dst, exist_ok=True)
    shutil.copyfile(os.path.join(src, "video.mp4"), os.path.join(dst, "video.mp4"))
    shutil.copyfile(os.path.join(camera_from or src, "data.npz"), os.path.join(dst, "data.npz"))
    shutil.copyfile(os.path.join(skeleton_from or src, "skeleton.mp4"), os.path.join(dst, "skeleton.mp4"))
    with open(os.path.join(prompt_from or src, "prompt.txt"), encoding="utf-8") as f:
        prompt = f.read()
    with open(os.path.join(dst, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)


def synthetic_camera(src, dst, yaw_deg=-15.0, strafe_m=0.5, forward_m=0.1):
    """Overwrite data.npz with a smooth yaw+strafe trajectory, keeping the clip's intrinsics."""
    z = np.load(os.path.join(src, "data.npz"), allow_pickle=True)
    K = z["intrinsics"]
    T = K.shape[0]
    c2w = np.zeros((T, 4, 4), dtype=np.float64)
    for t in range(T):
        a = np.radians(yaw_deg) * t / (T - 1)
        s = t / (T - 1)
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        c2w[t, :3, :3] = Ry
        c2w[t, :3, 3] = [strafe_m * s, 0.0, forward_m * s]
    c2w[:, 3, 3] = 1.0
    np.savez(os.path.join(dst, "data.npz"), intrinsics=K, extrinsics_c2w=c2w,
             fps=z["fps"] if "fps" in z else 30, image_hw=z["image_hw"] if "image_hw" in z else np.array([480, 640]))


def shift_skeleton(src, dst, dx=40):
    r = imageio.get_reader(os.path.join(src, "skeleton.mp4"))
    frames = [Image.fromarray(f).convert("RGB") for f in r]
    r.close()
    w = imageio.get_writer(os.path.join(dst, "skeleton.mp4"), fps=30, codec="libx264", quality=8,
                           pixelformat="yuv420p")
    for f in frames:
        im = Image.new("RGB", f.size, (0, 0, 0))
        im.paste(f, (dx, 0))
        w.append_data(np.asarray(im))
    w.close()


def shuffle_skeleton(src, dst, seed=0):
    r = imageio.get_reader(os.path.join(src, "skeleton.mp4"))
    frames = [np.asarray(f) for f in r]
    r.close()
    order = np.concatenate(([0], np.random.default_rng(seed).permutation(np.arange(1, len(frames)))))
    with open(os.path.join(dst, "skeleton_order.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "source_indices": order.tolist()}, f)
    w = imageio.get_writer(os.path.join(dst, "skeleton.mp4"), fps=30, codec="libx264", quality=8,
                           pixelformat="yuv420p")
    for index in order:
        w.append_data(frames[index])
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default="examples")
    ap.add_argument("--out", default="variants")
    args = ap.parse_args()
    ex = args.examples
    meta = {}

    dst = os.path.join(args.out, "perm_RICE_JCAM")
    copy_clip(os.path.join(ex, "agibot_rice_cooker"), dst,
              camera_from=os.path.join(ex, "xp10m_jigsaw_puzzle"),
              skeleton_from=os.path.join(ex, "xp10m_jigsaw_puzzle"))
    meta["perm_RICE_JCAM"] = dict(video="agibot_rice_cooker", camera="xp10m_jigsaw_puzzle",
                                  skeleton="xp10m_jigsaw_puzzle",
                                  note="human head-mounted camera trajectory + human hand skeleton on a static gripper scene")

    dst = os.path.join(args.out, "perm_PLAY_PUSH")
    copy_clip(os.path.join(ex, "xp_teleop_playdoh_tub"), dst,
              camera_from=os.path.join(ex, "omega_home_push_chair"),
              skeleton_from=os.path.join(ex, "omega_home_push_chair"))
    meta["perm_PLAY_PUSH"] = dict(video="xp_teleop_playdoh_tub", camera="omega_home_push_chair",
                                  skeleton="omega_home_push_chair",
                                  note="same embodiment, trajectory and hand path from a different task")

    src = os.path.join(ex, "agibot_rice_cooker")
    dst = os.path.join(args.out, "synthcam_RICE")
    copy_clip(src, dst)
    synthetic_camera(src, dst)
    meta["synthcam_RICE"] = dict(video="agibot_rice_cooker", camera="synthetic (yaw -15 deg, strafe +0.5 m, forward +0.1 m)",
                                 skeleton="agibot_rice_cooker",
                                 note="camera motion the scene has never seen in this clip; tests whether the ray map steers the viewpoint")

    dst = os.path.join(args.out, "skelshift_RICE")
    copy_clip(src, dst)
    shift_skeleton(src, dst, dx=40)
    meta["skelshift_RICE"] = dict(video="agibot_rice_cooker", camera="agibot_rice_cooker",
                                  skeleton="agibot_rice_cooker + 40 px x-shift",
                                  note="tests whether hand placement in the skeleton is followed at all")

    dst = os.path.join(args.out, "skelshuffle_RICE")
    copy_clip(src, dst)
    shuffle_skeleton(src, dst, seed=0)
    meta["skelshuffle_RICE"] = dict(video="agibot_rice_cooker", camera="agibot_rice_cooker",
                                    skeleton="frame 0 unchanged; frames 1..80 permuted with seed 0",
                                    note="temporal-order sensitivity; discontinuous input is not a ground-truth future")

    with open(os.path.join(args.out, "variants.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
