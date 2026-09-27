import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

# Import architectures from model.py
from model import Net, KDHeads

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
KD_METHODS = ("logit_kd", "feat_kd", "align", "align_conf", "full")

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
class PairedDataset(Dataset):
    """Returns (rgb, thermal, label, condition). Geometric augmentation is applied
    IDENTICALLY to both images so the pair stays spatially matched."""

    def __init__(self, csv_path, split, train, size=224, limit=None):
        self.root = Path(csv_path).parent
        with open(csv_path, newline="") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        if limit:
            self.rows = self.rows[:limit]
        self.train, self.size = train, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        rgb = Image.open(self.root / r["rgb"]).convert("RGB")
        th = Image.open(self.root / r["thermal"]).convert("L").convert("RGB")  # 1ch -> 3ch
        if self.train:
            if random.random() < 0.5:
                rgb, th = TF.hflip(rgb), TF.hflip(th)
            i0, j0, h0, w0 = T.RandomResizedCrop.get_params(rgb, scale=(0.7, 1.0), ratio=(0.9, 1.1))
            rgb = TF.resized_crop(rgb, i0, j0, h0, w0, (self.size, self.size))
            th = TF.resized_crop(th, i0, j0, h0, w0, (self.size, self.size))
            rgb = TF.adjust_brightness(rgb, random.uniform(0.7, 1.3))  # photometric aug: RGB only
        elif rgb.size != (self.size, self.size):
            rgb, th = TF.resize(rgb, (self.size, self.size)), TF.resize(th, (self.size, self.size))
        rgb = TF.normalize(TF.to_tensor(rgb), MEAN, STD)
        th = TF.normalize(TF.to_tensor(th), MEAN, STD)
        return rgb, th, int(r["label"]), int(r["condition"])

# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(net, loader, key, device):
    net.eval()
    ys, ps, cs = [], [], []
    for rgb, th, y, cond in loader:
        x = (rgb if key == "rgb" else th).to(device)
        _, logits = net(x)
        ps.append(logits.argmax(1).cpu())
        ys.append(y)
        cs.append(cond)
    if not ys:
        return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    y, p, c = torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(cs).numpy()
    avg = "binary" if len(set(y)) <= 2 else "macro"
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average=avg, zero_division=0)
    res = dict(acc=accuracy_score(y, p), precision=pr, recall=rc, f1=f1)
    for name, val in (("day", 1), ("night", 0)):  # day/night breakdown: key for your analysis
        m = c == val
        if m.any():
            res[f"acc_{name}"] = accuracy_score(y[m], p[m])
    return {k: float(v) for k, v in res.items()}

@torch.no_grad()
def latency_ms(net, size, device, n=50):
    net.eval()
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(10):
        net(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(n):
        net(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    return (time.time() - t0) / n * 1000

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(s)

def get_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["teacher", "student_ce", *KD_METHODS])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher_arch", default="swin_tiny_patch4_window7_224")
    ap.add_argument("--student_arch", default="mobilenetv3_large_100")
    ap.add_argument("--teacher_ckpt", default=None)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--T", type=float, default=4.0, help="KD temperature")
    ap.add_argument("--alpha", type=float, default=1.0, help="weight of logit KD")
    ap.add_argument("--beta", type=float, default=1.0, help="weight of feature loss")
    ap.add_argument("--gamma", type=float, default=0.5, help="weight of shared-head semantic CE")
    ap.add_argument("--lam_gate", type=float, default=2.0, help="weight of gate keep-ratio regulariser")
    ap.add_argument("--rho", type=float, default=0.7, help="target mean gate value")
    ap.add_argument("--conf_thresh", type=float, default=0.0, help="hard threshold on sample confidence")
    ap.add_argument("--dim", type=int, default=256, help="common semantic space size")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (default 0 for macOS/sandbox safety)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="debug: use only N samples per split")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    a = ap.parse_args()

    set_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(a.device)
    print(f"Using device: {device}")
    use_amp = device.type == "cuda" and not a.no_amp
    pretrained = not a.no_pretrained

    def mk(split, train):
        ds = PairedDataset(a.csv, split, train, a.img_size, a.limit)
        drop_last = train and (len(ds) >= a.bs)
        return DataLoader(
            ds,
            batch_size=min(a.bs, max(len(ds), 1)),
            shuffle=train,
            num_workers=a.workers,
            drop_last=drop_last,
            pin_memory=(device.type == "cuda"),
        )

    train_dl, val_dl, test_dl = mk("train", True), mk("val", False), mk("test", False)
    num_classes = 2

    is_teacher = a.method == "teacher"
    arch = a.teacher_arch if is_teacher else a.student_arch
    key = "rgb" if is_teacher else "thermal"
    net = Net(arch, num_classes, pretrained, a.img_size).to(device)

    teacher, heads = None, None
    if a.method in KD_METHODS:
        assert a.teacher_ckpt, "--teacher_ckpt is required for KD methods"
        teacher = Net(a.teacher_arch, num_classes, False, a.img_size).to(device)
        teacher.load_state_dict(torch.load(a.teacher_ckpt, map_location=device, weights_only=True))
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        heads = KDHeads(a.method, net.dim, teacher.dim, num_classes, a.dim).to(device)

    params = list(net.parameters()) + (list(heads.parameters()) if heads else [])
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, history = -1.0, []
    torch.save(net.state_dict(), out / "best.pt")

    for ep in range(1, a.epochs + 1):
        net.train()
        if heads:
            heads.train()
        agg, n = {}, 0
        for rgb, th, y, _ in train_dl:
            rgb, th, y = rgb.to(device), th.to(device), y.to(device)
            x = rgb if key == "rgb" else th
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                f, logits = net(x)
                if teacher is None:
                    loss, logs = F.cross_entropy(logits.float(), y), {}
                else:
                    with torch.no_grad():
                        ft, lt = teacher(rgb)
                    loss, logs = heads(f, ft, logits, lt, y, a)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            logs["loss"] = loss.item()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0) + v
            n += 1
        sched.step()

        val = evaluate(net, val_dl, key, device)
        row = dict(epoch=ep, **{k: v / max(n, 1) for k, v in agg.items()}, **{f"val_{k}": v for k, v in val.items()})
        history.append(row)
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if val["f1"] >= best_f1:
            best_f1 = val["f1"]
            torch.save(net.state_dict(), out / "best.pt")

    net.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
    test = evaluate(net, test_dl, key, device)
    result = dict(method=a.method, seed=a.seed, best_val_f1=best_f1, test=test,
                  params_M=sum(p.numel() for p in net.parameters()) / 1e6,
                  latency_ms_bs1=latency_ms(net, a.img_size, device), args=vars(a))
    (out / "results.json").write_text(json.dumps(result, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print("\nTEST:", json.dumps(test, indent=2))
    print(f"params={result['params_M']:.2f}M  latency={result['latency_ms_bs1']:.1f} ms/img (bs=1)")

if __name__ == "__main__":
    main()
