import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from model import Net

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
CLASSES = {0: "background", 1: "person"}

def preprocess(path, size=224):
    img = Image.open(path)
    if img.mode == "L":
        img = img.convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    t = TF.to_tensor(img)
    return TF.normalize(t, MEAN, STD).unsqueeze(0)   # (1, 3, H, W)

def load_student(ckpt_path, arch="mobilenetv3_large_100", num_classes=2, img_size=224):
    net = Net(arch, num_classes, False, img_size)
    
    # Load full state_dict (backbone + fc)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",  default="runs/full/best.pt",       help="Path to student checkpoint")
    ap.add_argument("--arch",  default="mobilenetv3_large_100")
    ap.add_argument("--image", default=None,                       help="Thermal image path. If None, picks randomly from pairs.csv")
    ap.add_argument("--csv",   default="data/kaist_patches/pairs.csv")
    ap.add_argument("--size",  type=int, default=224)
    a = ap.parse_args()

    # pick an image
    if a.image is None:
        root = Path(a.csv).parent
        rows = [r for r in csv.DictReader(open(a.csv)) if r["split"] == "test"]
        row  = random.choice(rows)
        image_path = str(root / row["thermal"])
        gt_label   = int(row["label"])
        print(f"Auto-selected test sample: {image_path}  (ground truth: {CLASSES[gt_label]})")
    else:
        image_path = a.image
        gt_label   = None

    # load model + run inference
    print(f"Loading checkpoint: {a.ckpt}")
    net = load_student(a.ckpt, a.arch, img_size=a.size)

    x = preprocess(image_path, a.size)
    with torch.no_grad():
        _, logits = net(x)          # (1, num_classes)
        probs  = F.softmax(logits, dim=1).squeeze(0)

    pred  = int(probs.argmax())
    conf  = float(probs[pred])

    print(f"\n{'─'*45}")
    print(f"  Image      : {image_path}")
    print(f"  Prediction : {CLASSES[pred]}  ({conf*100:.1f}% confidence)")
    if gt_label is not None:
        correct = "✓ correct" if pred == gt_label else "✗ wrong"
        print(f"  Ground truth: {CLASSES[gt_label]}  [{correct}]")
    print(f"  Scores     : background={probs[0]*100:.1f}%  person={probs[1]*100:.1f}%")
    print(f"{'─'*45}\n")

if __name__ == "__main__":
    main()
