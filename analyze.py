import argparse, json, os, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import timm
from PIL import Image
from sklearn.manifold import TSNE

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DEVICE = torch.device("cpu")

def load_net(arch, ckpt_path, num_classes=2, img_size=224):
    kw = dict(pretrained=False, num_classes=0)
    if "swin" in arch or "vit" in arch:
        kw["img_size"] = img_size
    backbone = timm.create_model(arch, **kw)
    dim = backbone(torch.zeros(1, 3, img_size, img_size)).shape[1]
    fc = nn.Linear(dim, num_classes)
    net = nn.Sequential(backbone, fc)
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    backbone_sd = {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}
    backbone.load_state_dict(backbone_sd, strict=False)
    return backbone, dim

def load_pairs(csv_path, split="test", limit=200):
    import csv as _csv
    root = Path(csv_path).parent
    rows = []
    with open(csv_path, newline="") as f:
        for r in _csv.DictReader(f):
            if r["split"] == split:
                rows.append(r)
    rows = rows[:limit]
    return rows, root

def preprocess(path, size=224):
    img = Image.open(path)
    if img.mode == "L":
        img = img.convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    t = TF.to_tensor(img)
    return TF.normalize(t, MEAN, STD).unsqueeze(0)

@torch.no_grad()
def extract_features(backbone, paths):
    backbone.eval()
    feats = []
    for p in paths:
        x = preprocess(p)
        feats.append(backbone(x).squeeze(0).numpy())
    return np.array(feats)

def plot_tsne(runs_dir, csv_path, out_dir):
    runs = Path(runs_dir)
    rows, root = load_pairs(csv_path, split="test")
    labels = np.array([int(r["label"]) for r in rows])
    conds  = np.array([int(r["condition"]) for r in rows])
    rgb_paths = [str(root / r["rgb"])     for r in rows]
    th_paths  = [str(root / r["thermal"]) for r in rows]

    method_ckpts = {
        "teacher (RGB)":   (runs / "teacher"    / "best.pt", "swin_tiny_patch4_window7_224", "rgb"),
        "student_ce (Th)": (runs / "student_ce" / "best.pt", "mobilenetv3_large_100",         "th"),
        "align (Th)":      (runs / "align"      / "best.pt", "mobilenetv3_large_100",         "th"),
        "full (Th)":       (runs / "full"        / "best.pt", "mobilenetv3_large_100",         "th"),
    }

    method_ids = {}
    all_embs, all_labels_list, all_conds_list = {}, {}, {}

    for mid, (name, (ckpt, arch, modal)) in enumerate(method_ckpts.items()):
        if not ckpt.exists():
            print(f"  Skipping {name}: {ckpt} not found")
            continue
        backbone, _ = load_net(arch, ckpt)
        paths = rgb_paths if modal == "rgb" else th_paths
        feats = extract_features(backbone, paths)
        method_ids[mid] = name
        print(f"  Extracted {len(feats)}x{feats.shape[1]}d features from {name}")
        perp = min(30, max(5, len(feats) // 3))
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
        emb = tsne.fit_transform(feats)
        all_embs[mid] = emb
        all_labels_list[mid] = labels[:len(feats)]
        all_conds_list[mid]  = conds[:len(feats)]

    if not all_embs:
        print("No checkpoints found for t-SNE. Skipping.")
        return

    fig, axes = plt.subplots(1, len(method_ids), figsize=(5 * len(method_ids), 4.5))
    if len(method_ids) == 1: axes = [axes]
    class_colors = {0: "#4477AA", 1: "#EE6677"}
    for ax, (mid, name) in zip(axes, method_ids.items()):
        emb = all_embs[mid]
        L   = all_labels_list[mid]
        for lbl, color in class_colors.items():
            sel = L == lbl
            ax.scatter(emb[sel, 0], emb[sel, 1], c=color, s=14, alpha=0.7, label="person" if lbl else "background")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(fontsize=8, framealpha=0.7)
    fig.suptitle("t-SNE: features coloured by CLASS", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "tsne.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(method_ids), figsize=(5 * len(method_ids), 4.5))
    if len(method_ids) == 1: axes = [axes]
    cond_colors = {1: "#DDAA33", 0: "#222255"}
    for ax, (mid, name) in zip(axes, method_ids.items()):
        emb = all_embs[mid]
        C   = all_conds_list[mid]
        for cond, color in cond_colors.items():
            sel = C == cond
            ax.scatter(emb[sel, 0], emb[sel, 1], c=color, s=14, alpha=0.7, label="day" if cond else "night")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(fontsize=8, framealpha=0.7)
    fig.suptitle("t-SNE: features coloured by DAY / NIGHT", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "tsne_modal.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

def plot_gate(runs_dir, csv_path, out_dir):
    from model import Net, KDHeads
    from train import PairedDataset

    runs = Path(runs_dir)
    ckpt = runs / "full" / "best.pt"
    t_ckpt = runs / "teacher" / "best.pt"
    if not (ckpt.exists() and t_ckpt.exists()):
        print("  Gate plot skipped: full/ or teacher/ checkpoint missing.")
        return

    num_classes = 2
    dim = 256
    student = Net("mobilenetv3_large_100", num_classes, False, 224).to(DEVICE)
    teacher = Net("swin_tiny_patch4_window7_224", num_classes, False, 224).to(DEVICE)
    heads   = KDHeads("full", student.dim, teacher.dim, num_classes, dim).to(DEVICE)

    student.load_state_dict(torch.load(ckpt,   map_location=DEVICE, weights_only=True))
    teacher.load_state_dict(torch.load(t_ckpt, map_location=DEVICE, weights_only=True))

    from torch.utils.data import DataLoader
    ds = PairedDataset(csv_path, "test", train=False)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    student.eval(); teacher.eval(); heads.eval()
    day_gates, night_gates = [], []

    with torch.no_grad():
        for rgb, th, y, cond in dl:
            fs, _ = student(th)
            ft, _ = teacher(rgb)
            zs = heads.ps(fs.float())
            zt = heads.pt(ft.float())
            zt_n = torch.nn.functional.normalize(zt, dim=1)
            g = heads.gate(zt_n).cpu().numpy()
            c = cond.numpy()
            day_gates.append(g[c == 1])
            night_gates.append(g[c == 0])

    day_g   = np.concatenate([x for x in day_gates   if len(x)], axis=0)   if any(len(x) for x in day_gates)   else np.zeros((1, dim))
    night_g = np.concatenate([x for x in night_gates if len(x)], axis=0)   if any(len(x) for x in night_gates) else np.zeros((1, dim))

    mean_day   = day_g.mean(axis=0)
    mean_night = night_g.mean(axis=0)
    order = mean_day.argsort()[::-1]

    x = np.arange(dim)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.bar(x, mean_day[order],   color="#DDAA33", alpha=0.8)
    ax1.set_ylabel("Gate value (day)",   fontsize=10)
    ax1.axhline(0.5, color="k", lw=0.8, ls="--")
    ax1.set_ylim(0, 1)
    ax2.bar(x, mean_night[order], color="#222255", alpha=0.8)
    ax2.set_ylabel("Gate value (night)", fontsize=10)
    ax2.axhline(0.5, color="k", lw=0.8, ls="--")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Feature dimension (sorted by day gate ↓)", fontsize=10)
    fig.suptitle("Per-dimension confidence gate: Day vs Night", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "gate.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

def plot_daynight(runs_dir, out_dir):
    runs = Path(runs_dir)
    methods = ["student_ce", "logit_kd", "feat_kd", "align", "align_conf", "full"]
    day_accs, night_accs = {}, {}

    for m in methods:
        p = runs / m / "history.json"
        if not p.exists():
            continue
        h = json.loads(p.read_text())
        best_ep = max(h, key=lambda r: r.get("val_f1", 0))
        day_accs[m]   = 100 * best_ep.get("val_acc_day",   0)
        night_accs[m] = 100 * best_ep.get("val_acc_night", 0)

    if not day_accs:
        return

    labels = list(day_accs.keys())
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, [day_accs[m]   for m in labels], w, label="Day",   color="#DDAA33", alpha=0.9)
    b2 = ax.bar(x + w/2, [night_accs[m] for m in labels], w, label="Night", color="#222255", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Accuracy (%) at best val epoch", fontsize=11)
    ax.set_ylim(50, 105)
    ax.axhline(100, color="grey", lw=0.8, ls=":")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    ax.legend(fontsize=10)
    ax.set_title("Day vs Night accuracy", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "daynight.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

def plot_confidence(runs_dir, csv_path, out_dir):
    from model import Net
    from train import PairedDataset
    from torch.utils.data import DataLoader
    import torch.nn.functional as F_

    runs = Path(runs_dir)
    t_ckpt = runs / "teacher" / "best.pt"
    if not t_ckpt.exists():
        return

    teacher = Net("swin_tiny_patch4_window7_224", 2, False, 224).to(DEVICE)
    teacher.load_state_dict(torch.load(t_ckpt, map_location=DEVICE, weights_only=True))
    teacher.eval()

    ds = PairedDataset(csv_path, "test", train=False)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    day_w, night_w = [], []
    with torch.no_grad():
        for rgb, th, y, cond in dl:
            _, lt = teacher(rgb.to(DEVICE))
            probs = F_.softmax(lt.float(), dim=1)
            w = probs.gather(1, y.to(DEVICE).unsqueeze(1)).squeeze(1).cpu().numpy()
            c = cond.numpy()
            day_w.extend(w[c == 1].tolist())
            night_w.extend(w[c == 0].tolist())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 25)
    if day_w:
        ax.hist(day_w,   bins=bins, alpha=0.7, color="#DDAA33", label=f"Day   (n={len(day_w)}, μ={np.mean(day_w):.2f})")
    if night_w:
        ax.hist(night_w, bins=bins, alpha=0.7, color="#222255", label=f"Night (n={len(night_w)}, μ={np.mean(night_w):.2f})")
    ax.set_xlabel("Teacher confidence  w = p(true class)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Sample-level teacher confidence weights", fontsize=12)
    ax.legend(fontsize=10)
    ax.axvline(0.5, color="red", lw=1.2, ls="--", alpha=0.7, label="w=0.5")
    fig.tight_layout()
    fig.savefig(out_dir / "confidence.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--csv", default="data/kaist_patches/pairs.csv")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--skip_tsne",  action="store_true")
    ap.add_argument("--skip_gate",  action="store_true")
    ap.add_argument("--skip_conf",  action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if not a.skip_tsne: plot_tsne(a.runs, a.csv, out)
    plot_daynight(a.runs, out)
    if not a.skip_conf: plot_confidence(a.runs, a.csv, out)
    if not a.skip_gate: plot_gate(a.runs, a.csv, out)

if __name__ == "__main__":
    main()
