import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.models.vit_classifier import ExplainableViT
from src.models.cnn_baseline import BaselineCNN
from src.data.dataset import SkinLesionDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.sample_data import generate_sample_dataset
from src.training.trainer import ModelTrainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Train Explainable ViT or Baseline CNN on Skin Lesion Dataset")
    parser.add_argument("--data_dir", type=str, default="data/demo_samples", help="Path to data directory")
    parser.add_argument("--csv_file", type=str, default="data/demo_samples/metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--model_type", type=str, default="vit", choices=["vit", "cnn"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--loss_type", type=str, default="focal", choices=["focal", "ce"], help="Loss function")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    # Generate sample dataset if data does not exist
    if not os.path.exists(args.csv_file):
        print(f"Dataset not found at {args.csv_file}. Generating demo dermoscopy dataset...")
        generate_sample_dataset(args.data_dir, num_samples_per_class=12)

    df = pd.read_csv(args.csv_file)
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df["dx"] if "dx" in df else None)

    class_names = [
        "MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"
    ]

    # Datasets and Loaders
    train_dataset = SkinLesionDataset(
        df=train_df,
        img_dir=os.path.join(args.data_dir, "images") if os.path.exists(os.path.join(args.data_dir, "images")) else args.data_dir,
        img_col="image_id" if "image_id" in train_df else train_df.columns[0],
        label_col="dx" if "dx" in train_df else train_df.columns[1],
        class_names=class_names,
        transform=get_train_transforms(img_size=224)
    )

    val_dataset = SkinLesionDataset(
        df=val_df,
        img_dir=os.path.join(args.data_dir, "images") if os.path.exists(os.path.join(args.data_dir, "images")) else args.data_dir,
        img_col="image_id" if "image_id" in val_df else val_df.columns[0],
        label_col="dx" if "dx" in val_df else val_df.columns[1],
        class_names=class_names,
        transform=get_val_transforms(img_size=224)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    class_weights = train_dataset.get_class_weights()

    # Model instantiation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model_type == "vit":
        model = ExplainableViT.create_small(num_classes=len(class_names), img_size=224)
        tag = "vit_lesion_classifier"
    else:
        model = BaselineCNN(num_classes=len(class_names))
        tag = "cnn_baseline"

    config = TrainingConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        loss_type=args.loss_type,
        checkpoint_dir=args.checkpoint_dir
    )

    trainer = ModelTrainer(model, config, class_names=class_names, class_weights=class_weights)
    result = trainer.fit(train_loader, val_loader, model_tag=tag)
    print(f"\nTraining completed! Best {config.save_best_metric}: {result['best_metric']:.4f}")


if __name__ == "__main__":
    main()
