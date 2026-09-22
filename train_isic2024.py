import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.vit_classifier import ExplainableViT, PretrainedTimmViT
from src.models.hybrid_vit import HybridViT
from src.models.loss_functions import FocalLoss
from src.data.isic2024_loader import ISIC2024Dataset, create_isic2024_dataloaders
from src.data.transforms import get_train_transforms, get_val_transforms
from src.eval.isic_metrics import calculate_isic2024_metrics
from src.data.sample_data import generate_sample_dataset


def create_demo_isic2024_dataset(output_dir: str = "data/isic2024_demo") -> str:
    """Creates a sample dataset structured exactly like ISIC 2024 Challenge (train-metadata.csv + crops)."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "train-metadata.csv")
    if not os.path.exists(csv_path):
        sample_df = generate_sample_dataset(output_dir, num_samples_per_class=8)
        # Add ISIC 2024 specific metadata fields
        sample_df["isic_id"] = sample_df["image_id"]
        sample_df["target"] = sample_df["dx"].apply(lambda x: 1 if x in ["MEL", "BCC"] else 0)
        sample_df["iddx_1"] = sample_df["dx"]
        sample_df["patient_id"] = [f"PAT_{i%10:04d}" for i in range(len(sample_df))]
        sample_df.to_csv(csv_path, index=False)
        print(f"[ISIC 2024] Generated demo dataset at {output_dir} with {len(sample_df)} cases.")
    return csv_path


def train_isic2024():
    parser = argparse.ArgumentParser(description="Train Explainable Vision Transformer on ISIC 2024 Challenge 3D-TBP Dataset")
    parser.add_argument("--data_dir", type=str, default="data/isic2024_demo", help="Path to image directory or HDF5 folder")
    parser.add_argument("--csv_file", type=str, default=None, help="Path to train-metadata.csv")
    parser.add_argument("--hdf5_file", type=str, default=None, help="Path to train-image.hdf5 (optional)")
    parser.add_argument("--model_type", type=str, default="hybrid_vit", choices=["hybrid_vit", "pretrained_vit", "custom_vit"],
                        help="Model architecture (Paper 02 HybridViT, Pretrained ViT, or Custom ViT)")
    parser.add_argument("--target_mode", type=str, default="multiclass", choices=["binary", "multiclass"],
                        help="Classification target mode")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--loss_type", type=str, default="focal", choices=["focal", "ce"], help="Loss function")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma parameter")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    args = parser.parse_args()

    # If no CSV path provided, check default or create demo
    if args.csv_file is None:
        default_csv = os.path.join(args.data_dir, "train-metadata.csv")
        if not os.path.exists(default_csv):
            args.csv_file = create_demo_isic2024_dataset(args.data_dir)
        else:
            args.csv_file = default_csv

    print(f"\n=================================================================")
    print(f"[ISIC 2024] TRAINING ON 3D-TBP SKIN LESION DATASET")
    print(f"=================================================================")
    print(f"Metadata: {args.csv_file}")
    print(f"Image Source: {args.hdf5_file if args.hdf5_file else args.data_dir}")
    print(f"Model Architecture: {args.model_type.upper()}")
    print(f"Target Mode: {args.target_mode.upper()} | Loss: {args.loss_type.upper()}")
    print(f"=================================================================\n")

    df = pd.read_csv(args.csv_file)
    image_src = args.hdf5_file if args.hdf5_file else os.path.join(args.data_dir, "images") if os.path.exists(os.path.join(args.data_dir, "images")) else args.data_dir

    num_classes = 7 if args.target_mode == "multiclass" else 2

    # Create DataLoaders with Balanced Sampling
    train_loader, val_loader, data_info = create_isic2024_dataloaders(
        metadata_df=df,
        image_source=image_src,
        train_transform=get_train_transforms(img_size=224),
        val_transform=get_val_transforms(img_size=224),
        batch_size=args.batch_size,
        val_split=0.25,
        use_balanced_sampler=True,
        target_mode=args.target_mode
    )

    print(f"Dataset Split: {data_info['train_samples']} Train samples | {data_info['val_samples']} Validation samples")

    # Instantiate selected model architecture
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model_type == "hybrid_vit":
        model = HybridViT(img_size=224, num_classes=num_classes)
    elif args.model_type == "pretrained_vit":
        model = PretrainedTimmViT(model_name="vit_small_patch16_224", num_classes=num_classes, pretrained=True)
    else:
        model = ExplainableViT.create_small(num_classes=num_classes, img_size=224)

    model.to(device)

    # Loss Function & Optimizer
    if args.loss_type == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_pauc = -1.0

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / max(len(train_loader), 1)

        # Validation & ISIC 2024 Metric Calculation
        model.eval()
        val_loss = 0.0
        val_trues = []
        val_probs = []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                val_trues.extend(labels.cpu().numpy())
                val_probs.append(probs)

        avg_val_loss = val_loss / max(len(val_loader), 1)
        y_true_arr = np.array(val_trues)
        y_prob_arr = np.concatenate(val_probs, axis=0) if val_probs else np.zeros((0, num_classes))

        # Binary malignant probability for pAUC evaluation
        if args.target_mode == "binary":
            malignant_probs = y_prob_arr[:, 1]
            binary_trues = y_true_arr
        else:
            # Multi-class: sum of MEL (idx 0) and BCC (idx 2) probabilities
            malignant_probs = y_prob_arr[:, 0] + (y_prob_arr[:, 2] if num_classes > 2 else 0)
            binary_trues = (y_true_arr == 0) | (y_true_arr == 2)

        metrics = calculate_isic2024_metrics(binary_trues, malignant_probs)
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"pAUC(>80%): {metrics['pauc_80']:.4f} | Total AUC: {metrics['total_roc_auc']:.4f} | "
            f"BalAcc: {metrics['balanced_accuracy']*100:.1f}% | Sens@95%Spec: {metrics['sens_at_95_spec']*100:.1f}%"
        )

        # Checkpoint Best Model
        if metrics["pauc_80"] > best_pauc or epoch == 1:
            best_pauc = metrics["pauc_80"]
            ckpt_path = os.path.join(args.checkpoint_dir, "vit_lesion_classifier_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
                "model_type": args.model_type,
                "target_mode": args.target_mode,
                "class_names": ExplainableViT.DEFAULT_CLASSES[:num_classes]
            }, ckpt_path)
            print(f"  --> [CHECKPOINT] Saved new best model to {ckpt_path} (pAUC: {best_pauc:.4f})")

    print(f"\n=================================================================")
    print(f"[SUCCESS] ISIC 2024 TRAINING COMPLETED! Best pAUC: {best_pauc:.4f}")
    print(f"Checkpoints saved to {args.checkpoint_dir}/vit_lesion_classifier_best.pt")
    print(f"=================================================================\n")


if __name__ == "__main__":
    train_isic2024()
