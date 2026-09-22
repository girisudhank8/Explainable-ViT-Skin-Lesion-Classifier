import os
import torch
import numpy as np
import pandas as pd
from PIL import Image

from src.models.vit_classifier import ExplainableViT
from src.data.transforms import get_val_transforms


def test_confidence():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vit = ExplainableViT.create_small(num_classes=7, img_size=224)
    ckpt_path = "checkpoints/vit_lesion_classifier_best.pt"

    if not os.path.exists(ckpt_path):
        print("[ERROR] Checkpoint not found.")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    vit.load_state_dict(ckpt["model_state_dict"])
    vit.to(device).eval()

    df = pd.read_csv("data/demo_samples/metadata.csv")
    tf = get_val_transforms(224)

    for cls in ["MEL", "NV", "BCC", "VASC"]:
        sub = df[df["dx"] == cls]
        if len(sub) > 0:
            row = sub.iloc[0]
            img = Image.open(row["filepath"]).convert("RGB")
            t = tf(np.array(img)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = vit(t)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            top_idx = int(np.argmax(probs))
            top_class = ExplainableViT.DEFAULT_CLASSES[top_idx]
            print(f"[TEST] {cls} Sample ({row['image_id']}) -> Predicted: {top_class} ({probs[top_idx]*100:.1f}%) | MEL Prob: {probs[0]*100:.1f}%")


if __name__ == "__main__":
    test_confidence()
