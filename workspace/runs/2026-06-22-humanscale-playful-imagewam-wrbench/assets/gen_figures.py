from PIL import Image, ImageDraw, ImageFont
import math, os

FONT = "/usr/share/fonts/TTF/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"

def get_font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT, size)
    except Exception:
        return ImageFont.load_default()

def humanscale_figure(path):
    W, H = 1000, 520
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    margin = dict(left=90, right=220, top=70, bottom=70)
    plot_w = W - margin["left"] - margin["right"]
    plot_h = H - margin["top"] - margin["bottom"]

    # data
    hours = [100, 1000, 5000]
    seen = [0.0080, 0.0072, 0.0067]
    unseen = [0.0234, 0.0216, 0.0204]
    robot_seen = 0.0071
    robot_unseen = 0.0254
    baseline_seen = 0.0103
    baseline_unseen = 0.0268

    x_min, x_max = math.log(50), math.log(12000)
    y_min, y_max = 0.005, 0.029

    def tx(h):
        return margin["left"] + (math.log(h) - x_min) / (x_max - x_min) * plot_w
    def ty(v):
        return margin["top"] + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    # axes
    d.line([(margin["left"], margin["top"]), (margin["left"], H - margin["bottom"])], fill="#333", width=2)
    d.line([(margin["left"], H - margin["bottom"]), (W - margin["right"], H - margin["bottom"])], fill="#333", width=2)

    # grid / ticks
    for h in [100, 500, 1000, 5000]:
        x = tx(h)
        d.line([(x, H - margin["bottom"]), (x, margin["top"])], fill="#eee", width=1)
        d.text((x, H - margin["bottom"] + 8), f"{h}h", fill="#555", font=get_font(16), anchor="mm")
    for v in [0.006, 0.010, 0.014, 0.018, 0.022, 0.026]:
        y = ty(v)
        d.line([(margin["left"], y), (W - margin["right"], y)], fill="#eee", width=1)
        d.text((margin["left"] - 10, y), f"{v:.3f}", fill="#555", font=get_font(16), anchor="rm")

    def dashed_line(y, color, on=8, off=4):
        x0, x1 = margin["left"], W - margin["right"]
        x = x0
        while x < x1:
            seg = min(on, x1 - x)
            d.line([(x, y), (x + seg, y)], fill=color, width=2)
            x += on + off

    # baselines
    dashed_line(ty(baseline_seen), "#ff7f0e")
    d.text((W - margin["right"] + 5, ty(baseline_seen)), "Wan2.2 no pretrain (Seen)", fill="#ff7f0e", font=get_font(14), anchor="lm")
    dashed_line(ty(baseline_unseen), "#ff7f0e", on=4, off=6)
    d.text((W - margin["right"] + 5, ty(baseline_unseen)), "Wan2.2 no pretrain (Unseen)", fill="#ff7f0e", font=get_font(14), anchor="lm")

    # robot baselines (horizontal bars at 5k)
    d.line([(tx(5000) - 30, ty(robot_seen)), (tx(5000) + 30, ty(robot_seen))], fill="#2ca02c", width=3)
    d.text((tx(5000), ty(robot_seen) - 10), "robot Seen", fill="#2ca02c", font=get_font(13), anchor="mb")
    d.line([(tx(5000) - 30, ty(robot_unseen)), (tx(5000) + 30, ty(robot_unseen))], fill="#2ca02c", width=3)
    d.text((tx(5000), ty(robot_unseen) + 10), "robot Unseen", fill="#2ca02c", font=get_font(13), anchor="mt")

    # fitted curves
    def fit(seen_flag):
        a, b = (0.009530, 0.000332) if seen_flag else (0.026932, 0.000767)
        pts = []
        for h in range(60, 12000, 30):
            v = a - b * math.log(h)
            pts.append((tx(h), ty(v)))
        return pts
    d.line(fit(True), fill="#1f77b4", width=3)
    d.line(fit(False), fill="#d62728", width=3)

    # points
    for h, v in zip(hours, seen):
        d.ellipse([(tx(h)-6, ty(v)-6), (tx(h)+6, ty(v)+6)], fill="#1f77b4", outline="white", width=2)
    for h, v in zip(hours, unseen):
        d.ellipse([(tx(h)-6, ty(v)-6), (tx(h)+6, ty(v)+6)], fill="#d62728", outline="white", width=2)

    # labels
    d.text((tx(5000)+10, ty(seen[-1])), "Ego Seen", fill="#1f77b4", font=get_font(15), anchor="lm")
    d.text((tx(5000)+10, ty(unseen[-1])), "Ego Unseen", fill="#d62728", font=get_font(15), anchor="lm")

    d.text((W//2, 22), "HumanScale: egocentric pretraining scales log-linearly", fill="#111", font=get_font(18, bold=True), anchor="mm")
    d.text((W//2, 44), "and beats real-robot data on out-of-distribution generalization", fill="#111", font=get_font(18, bold=True), anchor="mm")
    d.text((W//2, H - 25), "Action loss after identical post-training on AgiBot World · Reconstructed from reported numbers", fill="#777", font=get_font(14), anchor="mm")
    d.text((margin["left"], H - margin["bottom"] + 35), "Pretraining hours (log scale)", fill="#333", font=get_font(16), anchor="lm")
    d.text((20, H//2), "Action loss", fill="#333", font=get_font(16), anchor="mm", direction="ttb")

    img.save(path, "PNG")

def imagewam_figure(path):
    W, H = 900, 480
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    categories = ["ImageWAM\n(prefix only)", "Video-WAM\n8 frames", "Video-WAM\n16 frames"]
    tokens = [469, 3136, 6272]
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    max_t = max(tokens)
    chart_left, chart_top = 120, 100
    chart_w, chart_h = 600, 280
    bar_w = 90
    d.text((W//2, 35), "ImageWAM visual-token budget vs. video-WAM baselines", fill="#111", font=get_font(20, bold=True), anchor="mm")
    d.text((W//2, 65), "Resolution 224×448 (LIBERO 2-camera) · Source: ImageWAM paper Table 5 / configs", fill="#555", font=get_font(15), anchor="mm")
    # axes
    d.line([(chart_left, chart_top), (chart_left, chart_top+chart_h)], fill="#333", width=2)
    d.line([(chart_left, chart_top+chart_h), (chart_left+chart_w, chart_top+chart_h)], fill="#333", width=2)
    # y ticks
    for t in [0, 2000, 4000, 6000]:
        y = chart_top + chart_h - (t/max_t)*chart_h
        d.line([(chart_left, y), (chart_left+chart_w, y)], fill="#eee", width=1)
        d.text((chart_left-10, y), f"{t}", fill="#555", font=get_font(16), anchor="rm")
    # bars
    n = len(categories)
    gap = chart_w // (n+1)
    for i, (cat, tok, col) in enumerate(zip(categories, tokens, colors)):
        x = chart_left + gap*(i+1) - bar_w//2
        h = tok/max_t*chart_h
        y = chart_top + chart_h - h
        d.rectangle([(x, y), (x+bar_w, chart_top+chart_h)], fill=col, outline="white", width=2)
        # value label
        d.text((x+bar_w//2, y-10), f"{tok}", fill=col, font=get_font(17, bold=True), anchor="mb")
        # category label (multiline)
        lines = cat.split("\n")
        ly = chart_top + chart_h + 12
        for line in lines:
            d.text((x+bar_w//2, ly), line, fill="#333", font=get_font(14), anchor="mt")
            ly += 18
    d.text((chart_left, chart_top+chart_h+55), "Visual tokens materialized at inference", fill="#333", font=get_font(16), anchor="lm")
    d.text((20, H//2), "Tokens", fill="#333", font=get_font(16), anchor="mm", direction="ttb")
    # annotation
    d.text((chart_left+chart_w+20, chart_top+80), "ImageWAM uses\n~7.5% of a 16-frame\nvideo-WAM's tokens", fill="#1f77b4", font=get_font(15), anchor="lm")
    img.save(path, "PNG")

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    humanscale_figure(os.path.join(base, "humanscale_scaling.png"))
    imagewam_figure(os.path.join(base, "imagewam_token_budget.png"))
    print("figures saved")
