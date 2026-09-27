from flask import Flask, render_template, jsonify, send_from_directory
import os
import random
import csv
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from model import Net

app = Flask(__name__)

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
CLASSES = {0: "background", 1: "person"}
CKPT_PATH = "runs/full/best.pt"
CSV_PATH = "data/kaist_patches/pairs.csv"

# Global states
model = None
history = []  # Maintain in-memory history of inferences
MAX_HISTORY = 10

def preprocess(path, size=224):
    img = Image.open(path)
    if img.mode == "L":
        img = img.convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    t = TF.to_tensor(img)
    return TF.normalize(t, MEAN, STD).unsqueeze(0)

def load_student():
    global model
    if model is None:
        model = Net("mobilenetv3_large_100", 2, False, 224)
        sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
        model.eval()
    return model

@app.route("/")
def index():
    # Pass structural metadata to frontend context
    metadata = {
        "dataset": "KAIST Multispectral",
        "teacher_arch": "Swin-T",
        "teacher_params": "27.5M",
        "student_arch": "MobileNetV3",
        "student_params": "4.2M",
        "metrics": {
            "test_acc": "70.0%",
            "test_f1": "0.0%", 
            "acc_day": "66.7%",
            "acc_night": "71.4%"
        },
        "kd_params": {
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": 0.5,
            "rho": 0.7,
            "conf_thresh": 0.5
        }
    }
    return render_template("index.html", meta=metadata)

@app.route("/data/<path:filename>")
def serve_image(filename):
    return send_from_directory("data", filename)

@app.route("/api/infer", methods=["GET"])
def infer():
    root = Path(CSV_PATH).parent
    rows = [r for r in csv.DictReader(open(CSV_PATH)) if r["split"] == "test"]
    row = random.choice(rows)
    
    thermal_rel = row["thermal"]
    rgb_rel = row["rgb"]
    image_path = str(root / thermal_rel)
    gt_label = int(row["label"])
    
    net = load_student()
    x = preprocess(image_path)
    with torch.no_grad():
        _, logits = net(x)
        probs = F.softmax(logits, dim=1).squeeze(0)
    
    pred = int(probs.argmax())
    conf = float(probs[pred])
    is_correct = bool(pred == gt_label)
    
    # Structure result
    result = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "image_url": f"/data/kaist_patches/{thermal_rel}",
        "rgb_url": f"/data/kaist_patches/{rgb_rel}",
        "prediction": CLASSES[pred],
        "confidence": round(conf, 4),
        "ground_truth": CLASSES[gt_label],
        "is_correct": is_correct,
        "scores": {
            "background": round(float(probs[0]), 4),
            "person": round(float(probs[1]), 4)
        }
    }
    
    # Prepend to history
    history.insert(0, result)
    if len(history) > MAX_HISTORY:
        history.pop()
        
    return jsonify({"current": result, "history": history})

if __name__ == "__main__":
    load_student()
    app.run(host="0.0.0.0", port=5001, debug=True)
