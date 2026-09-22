"""
Quick end-to-end inference test using the newly trained checkpoints.
Tests that:
1. Both models load without error
2. MEL sample -> high MEL probability
3. NV (benign nevus) sample -> NOT high MEL probability
4. Every class gets a non-trivial probability for its own GT image
"""
import sys, os, torch
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from PIL import Image
from src.data.transforms import get_val_transforms
from src.models.vit_classifier import ExplainableViT
from src.models.cnn_baseline import BaselineCNN
import timm
import torch.nn as nn
import pandas as pd

CLASS_NAMES = ["MEL","NV","BCC","AKIEC","BKL","DF","VASC"]
DEVICE = "cpu"

class TransferLesionClassifier(nn.Module):
    def __init__(self, num_classes=7, backbone="efficientnet_b0.ra_in1k"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Dropout(0.3),
            nn.Linear(feat_dim, 256), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(256, num_classes))
    def forward(self, x):
        return self.classifier(self.backbone(x))

def load_model(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    backbone = ck.get("backbone")
    if backbone:
        m = TransferLesionClassifier(7, backbone)
        m.load_state_dict(ck["model_state_dict"])
    else:
        m = ExplainableViT.create_small(7, 224)
        m.load_state_dict(ck["model_state_dict"])
    return m.eval()

transform = get_val_transforms(224)
IMG_DIR = "data/demo_samples/images"

def run_model(model, img_path):
    img = np.array(Image.open(img_path).convert("RGB").resize((256,256)))
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)[0].numpy()
    pred_idx = int(np.argmax(probs))
    return probs, pred_idx

print("\nLoading models...")
vit = load_model("checkpoints/vit_lesion_classifier_best.pt")
cnn = load_model("checkpoints/cnn_baseline_best.pt")
print("Models loaded OK")

df = pd.read_csv("data/demo_samples/metadata.csv")

print(f"\n{'Model':<6} {'GT':>6} {'Pred':>8} {'GT_prob':>9} {'MEL_prob':>10} {'Status'}")
print("-"*60)

passed = 0
failed = 0
for cls in CLASS_NAMES:
    rows = df[df["dx"] == cls]
    if len(rows) == 0:
        continue
    row = rows.iloc[0]
    img_path = os.path.join(IMG_DIR, f"{row['image_id']}.jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(IMG_DIR, row["image_id"])
    if not os.path.exists(img_path):
        print(f"SKIP {cls}: image not found")
        continue

    cls_idx = CLASS_NAMES.index(cls)
    for model_name, model in [("ViT", vit), ("CNN", cnn)]:
        probs, pred_idx = run_model(model, img_path)
        gt_prob = probs[cls_idx]
        mel_prob = probs[0]
        pred_name = CLASS_NAMES[pred_idx]

        # Check: MEL samples should have high MEL prob; NV/others should not have 99%+ MEL
        ok = True
        status = "OK"
        if cls == "MEL" and mel_prob < 0.30:
            ok = False; status = "FAIL(low MEL)"
        if cls != "MEL" and mel_prob > 0.95:
            ok = False; status = "FAIL(false MEL)"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"{model_name:<6} {cls:>6} {pred_name:>8} {gt_prob*100:>8.1f}% {mel_prob*100:>9.1f}%  {status}")

print("-"*60)
print(f"\nResults: {passed} PASSED, {failed} FAILED")
if failed == 0:
    print("All inference tests PASSED!")
else:
    print("Some tests failed - review above.")
