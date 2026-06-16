#!/usr/bin/env python3
"""
Trace the inference data-flow for the two DreamX-World inference modes without
executing the model.
"""
from pathlib import Path
import re


def extract_steps(source, func_name):
    """Heuristic: pull numbered comments and key statements from a function."""
    text = Path(source).read_text(encoding="utf-8", errors="ignore")
    # find function start
    pat = re.compile(rf"def {func_name}\(.*?\n(?:(?:\s+.*\n)*)?", re.DOTALL)
    m = pat.search(text)
    if not m:
        return []
    func_body = m.group(0)
    # collect comments that look like steps and key calls
    steps = []
    for line in func_body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and (re.search(r"\d+\." , stripped) or any(k in stripped.lower() for k in ["prepare", "encode", "denois", "decode", "camera", "latent", "kv cache"])):
            steps.append(re.sub(r"^#+\s*", "", stripped))
        elif any(k in stripped for k in ["pipeline(", "pipeline.inference(", "self.generator(", "self.vae.decode"]):
            steps.append(stripped[:120])
    return steps


def trace_bidirectional():
    src = Path(__file__).resolve().parent / "src" / "pipeline" / "pipeline_dreamxworld.py"
    print("=" * 70)
    print("Bidirectional T2V/I2V flow (pipeline_dreamxworld.py::__call__)")
    print("=" * 70)
    steps = [
        "1. Check inputs (prompt, height/width divisibility)",
        "2. Encode prompt with UMT5-XXL -> prompt_embeds / negative_prompt_embeds",
        "3. Prepare latents from noise (VAE temporal/spatial compression)",
        "4. Prepare camera conditioning:",
        "   - action_seq + speed -> camera trajectory (utils/inference_utils.py)",
        "   - trajectory -> PRoPE dict {viewmats [T,4,4], K [T,3,3]}",
        "5. Optionally encode start_image for I2V and build latent mask",
        "6. Denoising loop (num_inference_steps, e.g. 30-50):",
        "   a. Apply classifier-free guidance: concat [uncond, cond]",
        "   b. Transformer forward with y_camera -> noise prediction",
        "   c. Scheduler step -> update latents",
        "   d. Re-anchor first frame for I2V",
        "7. VAE decode latents -> pixel video",
        "8. Save video with save_videos_grid",
    ]
    print("\n".join(steps))
    print()
    raw = extract_steps(src, "__call__")
    if raw:
        print("Key statements extracted from source:")
        for s in raw[:20]:
            print("  •", s)
    print()


def trace_ar():
    src = Path(__file__).resolve().parent / "src" / "pipeline" / "pipeline_causal_camera.py"
    print("=" * 70)
    print("Autoregressive long-horizon flow (pipeline_causal_camera.py::inference)")
    print("=" * 70)
    steps = [
        "1. Encode input image with Wan2.2 VAE -> initial_latent",
        "2. Build noise tensor [B, num_latent_frames, 48, H/8, W/8]; frame 0 = initial_latent",
        "3. Build camera trajectory -> PRoPE dict (chunk_relative=True for long rollouts)",
        "4. Initialise / reset KV cache (kv_cache1) and cross-attention cache",
        "5. Cache initial latent frames through generator at t=0",
        "6. For each temporal block (num_frame_per_block=3 latent frames):",
        "   a. Spatial denoising loop over denoising_step_list [1000,750,500,250]",
        "   b. Predict x0, add noise for next step, re-anchor first frame",
        "   c. Final step stores clean latents in output buffer",
        "   d. Rerun generator with context_noise to update KV cache",
        "7. VAE decode_to_pixel(output) -> video",
        "8. Optional Lab color correction and write_video",
    ]
    print("\n".join(steps))
    print()
    raw = extract_steps(src, "inference")
    if raw:
        print("Key statements extracted from source:")
        for s in raw[:20]:
            print("  •", s)
    print()


def main():
    trace_bidirectional()
    trace_ar()


if __name__ == "__main__":
    main()
