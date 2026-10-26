#!/usr/bin/env python
"""Run a batch of Ropedia W0 inference cases in one process (model loaded once).

Each case is a dict:
  name         : output stem
  clip         : clip directory holding video.mp4 / data.npz / skeleton.mp4 / prompt.txt
  prompt       : optional prompt override
  first_frame  : optional image path used as the conditioning frame instead of the clip's frame 0
  camera       : "on" (clip data.npz) | "off" | path to an alternative data.npz / .npy
  skeleton     : "on" (clip skeleton.mp4) | "off" | path to an alternative skeleton.mp4
  scene_scale  : float, multiplier on camera translation (default 1.0)
  seed         : int (default 0)
  cfg          : float (default 3.0)
  steps        : int (default 50)

usage:
  python audit_run.py --cases cases.json --out outputs/audit \
      --ckpt weights/w0-5b-midtrain.safetensors --base_model_dir models/Wan2.2-TI2V-5B
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image

from ropedia_w0.cli import (
    attach_condition_branches, load_checkpoints, load_pipeline, probe_video_length,
    read_video_window, read_video_window_for_gt,
)
from ropedia_w0.utils.data import save_video
from ropedia_w0.utils.data.camera_plucker import (
    _C2W_ALIASES, _K_ALIASES, _pick, compute_plucker_from_camera_params, load_camera_params,
)


def build_case(pipe, case, args):
    clip = case["clip"]
    seed = int(case.get("seed", 0))
    cfg = float(case.get("cfg", args.cfg_scale))
    steps = int(case.get("steps", args.num_inference_steps))
    n = int(case.get("num_frames", args.num_frames))
    H, W = int(case.get("height", args.height)), int(case.get("width", args.width))

    prompt = case.get("prompt")
    if prompt is None:
        with open(os.path.join(clip, "prompt.txt"), encoding="utf-8") as f:
            prompt = f.read().strip()

    t_src = probe_video_length(os.path.join(clip, "video.mp4"))
    start = int(case.get("start_frame", 0))
    idxs = list(range(start, start + n))

    if case.get("first_frame"):
        first = Image.open(case["first_frame"]).convert("RGB").resize((W, H))
    else:
        first = read_video_window(os.path.join(clip, "video.mp4"), [start], H, W)[0][0]

    cam = case.get("camera", "on")
    plucker = None
    if cam != "off":
        path = os.path.join(clip, "data.npz") if cam == "on" else cam
        params = load_camera_params(path)
        K_all = np.asarray(_pick(params, _K_ALIASES, "intrinsics"))[idxs]
        T_c2w = np.asarray(_pick(params, _C2W_ALIASES, "c2w"))[idxs]
        plucker = compute_plucker_from_camera_params(
            camera_params={"K_all": K_all, "T_c2w": T_c2w},
            height=H, width=W, relative_first_frame=True,
            scene_scale=float(case.get("scene_scale", 1.0)),
        )

    skel = case.get("skeleton", "on")
    skel_frames = None
    if skel != "off":
        path = os.path.join(clip, "skeleton.mp4") if skel == "on" else skel
        skel_frames = read_video_window(path, idxs, H, W)[0]

    return dict(clip=clip, prompt=prompt, first_frame=first, plucker=plucker,
                skel_frames=skel_frames, seed=seed, cfg=cfg, steps=steps,
                H=H, W=W, n=n, idxs=idxs, t_src=t_src, start=start)


def install_low_vram(pipeline):
    """Stage whole models unless they already have finer-grained CPU offload."""
    def load_models_to_device(model_names):
        model_names = set(model_names)
        for name, module in pipeline.named_children():
            if name in model_names or not hasattr(module, "to"):
                continue
            if getattr(module, "vram_management_enabled", False):
                for layer in module.modules():
                    if hasattr(layer, "offload"):
                        layer.offload()
            else:
                module.to("cpu")
        torch.cuda.empty_cache()
        for name, module in pipeline.named_children():
            if name not in model_names:
                continue
            if getattr(module, "vram_management_enabled", False):
                for layer in module.modules():
                    if hasattr(layer, "onload"):
                        layer.onload()
            else:
                module.to(pipeline.device)
        print(f"[low-vram] on gpu: {sorted(model_names)}  "
              f"allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    pipeline.load_models_to_device = load_models_to_device
    return pipeline


def load_pipeline_on_cpu(base_model_dir, layer_offload=False, text_layer_offload=False):
    """Build the pipeline with every model on CPU so they can be staged onto a small GPU later."""
    import ropedia_w0.pipelines.wan_video as wv
    from ropedia_w0.cli import load_pipeline
    orig = wv.WanVideoPipeline.from_pretrained
    wv.WanVideoPipeline.from_pretrained = staticmethod(
        lambda *a, **kw: orig(*a, **{**kw, "device": "cpu"}))
    try:
        pipe = load_pipeline("5b", base_model_dir, layer_offload=layer_offload, text_layer_offload=text_layer_offload)
    finally:
        wv.WanVideoPipeline.from_pretrained = orig
    pipe.device = "cuda"
    pipe.device_type = "cuda"
    if layer_offload or text_layer_offload:
        pipe.vram_management_enabled = pipe.check_vram_management_state()
    return pipe



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="JSON list of cases")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base_model_dir", default="models/Wan2.2-TI2V-5B")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--cfg_scale", type=float, default=3.0)
    ap.add_argument("--low_vram", action="store_true", help="offload models that are not in use to CPU")
    ap.add_argument("--layer_offload", action="store_true", help="keep model weights on CPU and stage individual layers on the GPU")
    ap.add_argument("--text_layer_offload", action="store_true", help="offload text-encoder layers, keeping the DiT whole during its phase")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="build and print the conditions without loading the model")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.cases, encoding="utf-8") as f:
        cases = json.load(f)

    if args.dry_run:
        for case in cases:
            c = build_case(None, case, args)
            print(f"{case['name']:22s} frames={c['n']} plucker={None if c['plucker'] is None else tuple(c['plucker'].shape)} "
                  f"skeleton={None if c['skel_frames'] is None else len(c['skel_frames'])} "
                  f"fps={int(case.get('fps', 16))} prompt={c['prompt'][:60]!r}", flush=True)
        return


    if args.low_vram or args.layer_offload or args.text_layer_offload:
        pipe = load_pipeline_on_cpu(args.base_model_dir, layer_offload=args.layer_offload, text_layer_offload=args.text_layer_offload)
    else:
        pipe = load_pipeline("5b", args.base_model_dir)
    attach_condition_branches(pipe, camera_in_dim=96, skeleton_in_dim=48)
    load_checkpoints(pipe, "5b", argparse.Namespace(ckpt=args.ckpt, ckpt_type="full", camera_in_dim=96, skeleton_in_dim=48))
    if (args.low_vram or args.text_layer_offload) and not args.layer_offload:
        install_low_vram(pipe)

    log_path = os.path.join(args.out, "run_log.jsonl")
    for case in cases:
        name = case["name"]
        pred_path = os.path.join(args.out, f"{name}_pred.mp4")
        if args.skip_existing and os.path.exists(pred_path):
            print(f"[skip] {name}", flush=True)
            continue
        c = build_case(pipe, case, args)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        video = pipe(
            prompt=c["prompt"], negative_prompt="",
            input_image=c["first_frame"],
            camera_plucker=c["plucker"],
            hand_skeleton_video=c["skel_frames"],
            hand_skeleton_mask=torch.ones(c["n"]) if c["skel_frames"] is not None else None,
            seed=c["seed"], height=c["H"], width=c["W"], num_frames=c["n"],
            cfg_scale=c["cfg"], num_inference_steps=c["steps"], tiled=True,
        )
        dt = time.time() - t0
        peak_gb = torch.cuda.max_memory_reserved() / 1e9
        save_video(video, pred_path, fps=int(case.get("fps", 16)), quality=5)
        if case.get("save_gt", True):
            read_video_window_for_gt(os.path.join(c["clip"], "video.mp4"), c["idxs"],
                                     os.path.join(args.out, f"{name}_gt.mp4"),
                                     c["H"], c["W"], int(case.get("fps", 16)))
        rec = dict(name=name, clip=c["clip"], prompt=c["prompt"], seed=c["seed"], cfg=c["cfg"],
                   steps=c["steps"], num_frames=c["n"], height=c["H"], width=c["W"],
                   fps=int(case.get("fps", 16)),
                   camera=bool(c["plucker"] is not None), skeleton=bool(c["skel_frames"] is not None),
                   scene_scale=float(case.get("scene_scale", 1.0)), seconds=round(dt, 1),
                   gpu_peak_reserved_gb=round(peak_gb, 3),
                   camera_source=case.get("camera", "on"), skeleton_source=case.get("skeleton", "on"),
                   first_frame=case.get("first_frame"))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[done] {name}  {dt:.1f}s  gpu_peak={peak_gb:.2f}GB", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
