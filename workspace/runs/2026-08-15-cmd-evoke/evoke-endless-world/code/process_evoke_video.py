"""回拉的 EVOKE 推理 mp4 抽帧拼条(2x3, 标时间戳)。"""
import subprocess, sys, os
from PIL import Image, ImageDraw

def extract_frames(video, n=6, tmpdir="/tmp/evoke_frames"):
    os.makedirs(tmpdir, exist_ok=True)
    # 时长
    out = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                          "-of", "csv=p=0", video], capture_output=True, text=True)
    dur = float(out.stdout.strip())
    frames = []
    for i in range(n):
        t = dur * (i + 0.5) / n
        fn = f"{tmpdir}/f{i}.png"
        subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-ss", f"{t:.2f}", "-i", video,
                        "-frames:v", "1", fn], check=True)
        frames.append((t, fn))
    return frames, dur

def make_strip(video, out_png, label="", cols=3, rows=2, thumb_w=480):
    frames, dur = extract_frames(video, n=cols * rows)
    ims = []
    for t, fn in frames:
        im = Image.open(fn)
        w = thumb_w
        h = int(im.height * w / im.width)
        im = im.resize((w, h))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, w, 30], fill=(0, 0, 0))
        d.text((8, 8), f"t={t:.1f}s", fill=(255, 255, 255))
        ims.append(im)
    cell_h = ims[0].height
    grid = Image.new("RGB", (cols * thumb_w, rows * cell_h), (255, 255, 255))
    for i, im in enumerate(ims):
        grid.paste(im, ((i % cols) * thumb_w, (i // cols) * cell_h))
    grid.save(out_png)
    print(f"{label or video}: {dur:.1f}s -> {out_png}")

if __name__ == "__main__":
    for video, out, label in [
        ("output_evoke/smoke/i2v_coral_reef/geo_pred.mp4", "runs/2026-08-15-cmd-evoke/assets/evoke_smoke_strip.png", "smoke"),
        ("output_evoke/i2v22/i2v_coral_reef/geo_pred.mp4", "runs/2026-08-15-cmd-evoke/assets/evoke_i2v22_strip.png", "i2v22"),
        ("output_evoke/seg6/aurora/geo_pred.mp4", "runs/2026-08-15-cmd-evoke/assets/evoke_seg_strip.png", "segment"),
        ("output_evoke/i2v_nowarp/i2v_coral_reef/geo_pred.mp4", "runs/2026-08-15-cmd-evoke/assets/evoke_nowarp_strip.png", "warp-off"),
    ]:
        if os.path.exists(video):
            try:
                make_strip(video, out, label)
            except Exception as e:
                print(f"FAIL {label}: {e}")
        else:
            print(f"missing {video}")
