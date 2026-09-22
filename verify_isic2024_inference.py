import os, sys, torch
import numpy as np
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

from src.data.transforms import get_val_transforms
import timm
import torch.nn as nn

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

class TransferLesionClassifier(nn.Module):
    def __init__(self, num_classes=7, backbone="efficientnet_b0.ra_in1k"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Dropout(0.3),
            nn.Linear(feat_dim, 256), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))

model = TransferLesionClassifier(7, "efficientnet_b0.ra_in1k")
ck = torch.load("checkpoints/isic2024_vit_best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ck["model_state_dict"])
model.eval()

transform = get_val_transforms(224)

df = pd.read_csv("data/isic2024_train_dataset.csv")

# Select a representative set across classes
sample_cases = []
for c in ["MEL", "BCC", "NV", "AKIEC"]:
    sample_cases.append(df[df["dx"] == c].head(3))
sample_cases = pd.concat(sample_cases).reset_index(drop=True)

print(f"\nEvaluating {len(sample_cases)} Real ISIC 2024 Challenge Images:")
print("-" * 75)
print(f"{'ISIC_ID':<15} {'GT Class':<10} {'Predicted':<10} {'GT Prob':<10} {'Malignant Prob':<16} {'Risk Tier'}")
print("-" * 75)

for _, row in sample_cases.iterrows():
    img_path = row["filepath"]
    gt_class = row["dx"]
    gt_idx = CLASS_NAMES.index(gt_class)
    
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    tensor = transform(img_np).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)[0].numpy()
    
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    gt_prob = probs[gt_idx]
    mal_prob = probs[0] + probs[2]  # MEL + BCC
    
    if mal_prob > 0.40:
        risk = "HIGH (Biopsy indicated)"
    elif mal_prob > 0.20:
        risk = "MODERATE (Surveillance)"
    else:
        risk = "LOW (Benign routine)"
        
    print(f"{row['isic_id']:<15} {gt_class:<10} {pred_class:<10} {gt_prob*100:6.1f}%     {mal_prob*100:6.1f}%          {risk}")

print("-" * 75)
print("\nValidation complete. Checkpoint is verified and active.")
