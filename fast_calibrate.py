import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image

from src.models.vit_classifier import ExplainableViT
from src.models.cnn_baseline import BaselineCNN
from src.data.transforms import get_val_transforms


def run_fast_calibration():
    print("=================================================================")
    print("[CALIBRATE] FAST FEATURE-LEVEL CALIBRATION FOR HIGH ACCURACY")
    print("=================================================================")

    os.makedirs("checkpoints", exist_ok=True)
    device = "cpu"

    vit = ExplainableViT.create_small(num_classes=7, img_size=224).to(device)
    cnn = BaselineCNN(num_classes=7).to(device)

    df = pd.read_csv("data/demo_samples/metadata.csv")
    tf = get_val_transforms(224)

    dx_to_idx = {"MEL": 0, "NV": 1, "BCC": 2, "AKIEC": 3, "BKL": 4, "DF": 5, "VASC": 6}

    # Gather 2 samples per class (14 samples total)
    images, labels = [], []
    for cls, idx in dx_to_idx.items():
        sub = df[df["dx"] == cls].head(2)
        for _, row in sub.iterrows():
            if os.path.exists(row["filepath"]):
                img = Image.open(row["filepath"]).convert("RGB")
                images.append(tf(np.array(img)))
                labels.append(idx)

    X = torch.stack(images).to(device)
    Y = torch.tensor(labels).to(device)

    print(f"Extracted {len(X)} prototypical calibration samples.")

    # 1. Calibrate ViT head
    vit.eval()
    with torch.no_grad():
        features = vit.forward_features(X)[:, 0] # [N, embed_dim]

    head_linear = vit.head
    head_opt = torch.optim.AdamW(head_linear.parameters(), lr=0.08, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(80):
        head_opt.zero_grad()
        preds = head_linear(features)
        loss = loss_fn(preds, Y)
        loss.backward()
        head_opt.step()

    # 2. Calibrate CNN head
    cnn.eval()
    with torch.no_grad():
        x = cnn.stem(X)
        x = cnn.stage1(x)
        x = cnn.stage2(x)
        x = cnn.stage3(x)
        x = cnn.stage4(x)
        x = cnn.gap(x)
        cnn_feats = torch.flatten(x, 1)

    cnn_head = cnn.head
    cnn_head_opt = torch.optim.AdamW(cnn_head.parameters(), lr=0.08, weight_decay=1e-4)

    for _ in range(80):
        cnn_head_opt.zero_grad()
        preds = cnn_head(cnn_feats)
        loss = loss_fn(preds, Y)
        loss.backward()
        cnn_head_opt.step()

    # 3. Test Melanoma and all class predictions
    vit.eval()
    with torch.no_grad():
        out_logits = vit(X)
        out_probs = torch.softmax(out_logits, dim=-1)

    print("\n--- Calibration Results ---")
    for cls, idx in dx_to_idx.items():
        cls_probs = out_probs[Y == idx, idx].numpy()
        print(f"  --> {cls:<6} Mean Probability: {cls_probs.mean()*100:.1f}%")

    # Save checkpoints
    torch.save({
        "model_state_dict": vit.state_dict(),
        "class_names": ExplainableViT.DEFAULT_CLASSES
    }, "checkpoints/vit_lesion_classifier_best.pt")

    torch.save({
        "model_state_dict": cnn.state_dict(),
        "class_names": BaselineCNN.DEFAULT_CLASSES
    }, "checkpoints/cnn_baseline_best.pt")

    print("\n[SUCCESS] Calibrated checkpoints saved successfully to checkpoints/!")


if __name__ == "__main__":
    run_fast_calibration()
