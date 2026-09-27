import csv
import os
import random
from PIL import Image

def make_dummy(path):
    img = Image.new('RGB', (224, 224), color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    img.save(path)

csv_path = "data/kaist_patches/pairs.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["rgb", "thermal", "label", "condition", "split"])
    for split, count in [("train", 10), ("val", 4), ("test", 30)]:
        for i in range(count):
            rgb_path = f"rgb/{split}_{i}.jpg"
            th_path = f"thermal/{split}_{i}.jpg"
            make_dummy(f"data/kaist_patches/{rgb_path}")
            make_dummy(f"data/kaist_patches/{th_path}")
            label = random.randint(0, 1)
            condition = random.randint(0, 1) # day/night
            writer.writerow([rgb_path, th_path, label, condition, split])
print("Dummy data generated.")
