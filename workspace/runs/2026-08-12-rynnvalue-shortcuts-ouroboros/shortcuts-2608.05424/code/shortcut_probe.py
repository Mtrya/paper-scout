"""Invisible-shortcut probe on VLA-relevant encoders.

Reimplements the paper's two diagnostic metrics (Sec.2) on a small controlled
set: imagenette (10-class ImageNet subset, full-res photos), recompressed at
random JPEG quality labels.

  MP  (metadata prediction): logistic regression predicts the JPEG quality
      label from frozen features, 5-class, chance = 20%.
  SPD (semantic prediction distraction): kNN semantic classification when
      the query shares its JPEG label with positives (pos-same) vs with
      negatives (neg-same); Delta = |A_pos-same - A_neg-same|.
  DISP: cosine displacement same-image q95->q30 relative to inter-image
      distance (normalized).

Encoders: SigLIP-SO400M (pi0's tower), CLIP ViT-B/16, DINOv2-B (paper's
least-sensitive reference).
"""
import argparse
import io
import json
import os

import numpy as np
import torch
from PIL import Image

QUALITIES = [30, 50, 70, 85, 95]


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB").copy()


def load_imagenette(subset_dir, per_class):
    classes = sorted(os.listdir(subset_dir))
    images, labels = [], []
    for ci, c in enumerate(classes):
        files = sorted(os.listdir(os.path.join(subset_dir, c)))[:per_class]
        for f in files:
            images.append(Image.open(os.path.join(subset_dir, c, f)).convert("RGB"))
            labels.append(ci)
    return images, np.array(labels), classes


CLASS_NAMES = ["tench", "English springer", "cassette player", "chain saw",
               "church", "French horn", "garbage truck", "gas pump",
               "golf ball", "parachute"]


def load_imagenette_parquet(parquet_path, per_class):
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    images, labels = [], []
    for ci in sorted(df["label"].unique()):
        sub = df[df["label"] == ci].head(per_class)
        for _, row in sub.iterrows():
            img = row["image"]
            if isinstance(img, dict):
                img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            images.append(img)
            labels.append(int(ci))
    return images, np.array(labels), CLASS_NAMES


@torch.no_grad()
def extract_feats(name, model, proc, images, device, bs=32):
    feats = []
    for i in range(0, len(images), bs):
        chunk = images[i : i + bs]
        if name == "siglip":
            inp = proc(images=chunk, return_tensors="pt").to(device)
            f = model.get_image_features(**inp)
        elif name == "clip":
            inp = proc(images=chunk, return_tensors="pt").to(device)
            f = model.get_image_features(**inp)
        elif name == "dinov2":
            inp = proc(images=chunk, return_tensors="pt").to(device)
            f = model(**inp).last_hidden_state[:, 0]
        f = torch.nn.functional.normalize(f.float(), dim=-1)
        feats.append(f.cpu())
    return torch.cat(feats).numpy()


def load_encoder(name, device):
    root = os.environ.get("ENCODER_ROOT", "")
    def resolve(repo_id):
        local = os.path.join(root, repo_id.replace("/", "--")) if root else ""
        return local if local and os.path.isdir(local) else repo_id
    if name == "siglip":
        from transformers import SiglipModel, SiglipProcessor
        rid = resolve("google/siglip-so400m-patch14-384")
        m = SiglipModel.from_pretrained(rid, torch_dtype=torch.float32).to(device).eval()
        p = SiglipProcessor.from_pretrained(rid)
        return m, p, m
    if name == "clip":
        from transformers import CLIPModel, CLIPProcessor
        rid = resolve("openai/clip-vit-base-patch16")
        m = CLIPModel.from_pretrained(rid).to(device).eval()
        p = CLIPProcessor.from_pretrained(rid)
        return m, p, m
    if name == "dinov2":
        from transformers import AutoModel, AutoImageProcessor
        rid = resolve("facebook/dinov2-base")
        m = AutoModel.from_pretrained(rid).to(device).eval()
        p = AutoImageProcessor.from_pretrained(rid)
        return m, p, m
    raise ValueError(name)


def mp_score(feats_by_q, labels_meta, seed=0):
    """Logistic regression: predict JPEG label from features (train/test split)."""
    from sklearn.linear_model import LogisticRegression
    X = np.concatenate(list(feats_by_q.values()), 0)
    y = np.concatenate([np.full(len(v), qi) for qi, v in feats_by_q.items()])
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    tr, te = idx[: int(0.7 * len(X))], idx[int(0.7 * len(X)) :]
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X[tr], y[tr])
    return float(clf.score(X[te], y[te]))


def spd_score(F_query, F_pos, F_neg, sem_labels, k=5):
    """kNN semantic accuracy. Query from F_query; gallery = same-class
    exemplars from F_pos + different-class exemplars from F_neg.
    Paper setups (query fixed q95):
      pos-same: positives q95, negatives q30 (query shares metadata w/ positives)
      neg-same: positives q30, negatives q95 (query shares metadata w/ negatives)
    """
    n = len(sem_labels)
    acc = []
    for i in range(n):
        pos_mask = sem_labels == sem_labels[i]
        pos_mask[i] = False
        neg_mask = ~pos_mask
        neg_mask[i] = False
        gallery = np.concatenate([F_pos[pos_mask], F_neg[neg_mask]], 0)
        glabels = np.concatenate([sem_labels[pos_mask], sem_labels[neg_mask]])
        d = gallery @ F_query[i]
        nn = np.argsort(-d)[:k]
        pred = np.bincount(glabels[nn], minlength=sem_labels.max() + 1).argmax()
        acc.append(pred == sem_labels[i])
    return float(np.mean(acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenette_dir", default=None, help="dir with class subdirs of JPEGs")
    ap.add_argument("--parquet", default=None, help="imagenette parquet (image/label columns)")
    ap.add_argument("--per_class", type=int, default=50)
    ap.add_argument("--encoders", default="siglip,clip,dinov2")
    ap.add_argument("--output", default="shortcut_probe.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.parquet:
        images, sem_labels, classes = load_imagenette_parquet(args.parquet, args.per_class)
    else:
        images, sem_labels, classes = load_imagenette(args.imagenette_dir, args.per_class)
    n = len(images)
    print(f"{n} images, {len(classes)} classes", flush=True)

    # assign random JPEG labels, recompress
    rng = np.random.RandomState(42)
    meta = rng.randint(0, len(QUALITIES), n)
    by_q_imgs = {}
    for qi, q in enumerate(QUALITIES):
        sel = np.where(meta == qi)[0]
        by_q_imgs[q] = [jpeg_roundtrip(images[i], q) for i in sel]
        print(f"q{q}: {len(sel)} images recompressed", flush=True)

    results = {"qualities": QUALITIES, "per_class": args.per_class, "encoders": {}}
    for enc in args.encoders.split(","):
        print(f"== {enc}", flush=True)
        model, proc, _ = load_encoder(enc, device)
        feats_by_q = {}
        for q, imgs in by_q_imgs.items():
            sel_labels = np.concatenate([np.where(meta == QUALITIES.index(q))[0]])
            feats_by_q[q] = extract_feats(enc, model, proc, imgs, device)
        # MP: predict quality label
        acc_mp = mp_score({q: f for q, f in feats_by_q.items()}, None)
        # SPD: pos-same (query+positives q95, negatives q30) vs neg-same
        # build per-image features at q95 and q30 regardless of assignment
        all_q95 = extract_feats(enc, model, proc, [jpeg_roundtrip(im, 95) for im in images], device)
        all_q30 = extract_feats(enc, model, proc, [jpeg_roundtrip(im, 30) for im in images], device)
        a_pos_same = spd_score(all_q95, all_q95, all_q30, sem_labels)
        a_neg_same = spd_score(all_q95, all_q30, all_q95, sem_labels)
        # baseline: everything q95 (no metadata contrast)
        a_base = spd_score(all_q95, all_q95, all_q95, sem_labels)
        # displacement: same image q95->q30 vs inter-image
        same_d = 1 - (all_q95 * all_q30).sum(1)
        rng2 = np.random.RandomState(1)
        perm = rng2.permutation(n)
        inter_d = 1 - (all_q95 * all_q95[perm]).sum(1)
        results["encoders"][enc] = {
            "MP_jpeg_acc": acc_mp,
            "MP_chance": 1.0 / len(QUALITIES),
            "SP_baseline_q95": a_base,
            "SP_pos_same": a_pos_same,
            "SP_neg_same": a_neg_same,
            "SPD_delta": abs(a_pos_same - a_neg_same),
            "disp_same_q95_q30_mean": float(same_d.mean()),
            "disp_inter_mean": float(inter_d.mean()),
            "disp_ratio": float(same_d.mean() / inter_d.mean()),
        }
        print(json.dumps(results["encoders"][enc], indent=1), flush=True)
        del model
        torch.cuda.empty_cache()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=1)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
