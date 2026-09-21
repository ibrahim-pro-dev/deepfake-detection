import sys, os
sys.path.insert(0, r"C:\Users\yashm\OneDrive\Desktop\Deepfake-detection")

import numpy as np
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

face = Image.fromarray(
    np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
)

for name in (
    "Smogy/SmOGy-Ai-images-detector",
    "dima806/deepfake_vs_real_image_detection",
):
    print("=" * 60)
    print("MODEL:", name)
    proc = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForImageClassification.from_pretrained(name)
    model.eval()

    print("  id2label:", model.config.id2label)
    print("  label2id:", model.config.label2id)

    inputs = proc(images=face, return_tensors="pt")
    import torch
    with torch.no_grad():
        out = model(**inputs)
    probs = torch.nn.functional.softmax(out.logits, dim=1)
    conf, pred = torch.max(probs, dim=1)
    label = model.config.id2label[pred.item()]
    is_ai = any(t in label.lower() for t in ("ai", "artificial", "fake", "generated"))
    risk = conf.item() if is_ai else 1.0 - conf.item()
    print("  synthetic-face -> label=%-12s conf=%.3f  risk=%.3f" % (label, conf.item(), risk))

print("=" * 60)
print("DONE")
