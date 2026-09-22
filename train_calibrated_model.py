import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.models.vit_classifier import ExplainableViT
from src.models.cnn_baseline import BaselineCNN
from src.models.loss_functions import FocalLoss
from src.data.dataset import SkinLesionDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.sample_data import generate_sample_dataset
from src.training.trainer import ModelTrainer, TrainingConfig


def train_calibrated():
    print("=================================================================")
    print("[CALIBRATION] GENERATING DATASET & TRAINING CALIBRATED ViT")
    print("=================================================================\n")

    data_dir = "data/demo_samples"
    # Generate 15 samples per class (105 total) with clinical characteristics
    df = generate_sample_dataset(data_dir, num_samples_per_class=16)

    class_names = [
        "MEL (Melanoma)",
        "NV (Melanocytic Nevus)",
        "BCC (Basal Cell Carcinoma)",
        "AKIEC (Actinic Keratosis)",
        "BKL (Benign Keratosis)",
        "DF (Dermatofibroma)",
        "VASC (Vascular Lesion)"
    ]
    dx_codes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

    # Map labels
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["dx"])

    train_dataset = SkinLesionDataset(
        df=train_df,
        img_dir=os.path.join(data_dir, "images"),
        img_col="image_id",
        label_col="dx",
        class_names=dx_codes,
        transform=get_train_transforms(img_size=224)
    )

    val_dataset = SkinLesionDataset(
        df=val_df,
        img_dir=os.path.join(data_dir, "images"),
        img_col="image_id",
        label_col="dx",
        class_names=dx_codes,
        transform=get_val_transforms(img_size=224)
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Train samples: {len(train_df)} | Val samples: {len(val_df)}")

    # 1. Train Explainable ViT
    vit = ExplainableViT.create_small(num_classes=7, img_size=224)
    config = TrainingConfig(
        epochs=12,
        lr=6e-4,
        batch_size=16,
        device=device,
        loss_type="focal",
        checkpoint_dir="checkpoints",
        save_best_metric="macro_f1"
    )

    trainer = ModelTrainer(vit, config, class_names=dx_codes)
    vit_res = trainer.fit(train_loader, val_loader, model_tag="vit_lesion_classifier")

    # 2. Train CNN Baseline
    cnn = BaselineCNN(num_classes=7)
    cnn_trainer = ModelTrainer(cnn, config, class_names=dx_codes)
    cnn_res = cnn_trainer.fit(train_loader, val_loader, model_tag="cnn_baseline")

    # 3. Test Melanoma Sample Prediction Confidence
    vit.eval()
    mel_samples = df[df["dx"] == "MEL"]
    if len(mel_samples) > 0:
        mel_img_path = mel_samples.iloc[0]["filepath"]
        from PIL import Image
        pil_img = Image.open(mel_img_path).convert("RGB")
        val_tf = get_val_transforms(224)
        t = val_tf(np.array(pil_img)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = vit(t)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        print(f"\n[VERIFICATION] Melanoma Test Sample Prediction: MEL Probability = {probs[0]*100:.1f}%")

    print("\n[SUCCESS] Calibrated checkpoints saved successfully to checkpoints/!")


if __name__ == "__main__":
    train_calibrated()
