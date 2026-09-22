import os
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..eval.metrics import calculate_clinical_metrics
from ..models.loss_functions import FocalLoss


@dataclass
class TrainingConfig:
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    loss_type: str = "focal"  # 'focal' or 'ce'
    focal_gamma: float = 2.0
    checkpoint_dir: str = "checkpoints"
    save_best_metric: str = "macro_f1"


class ModelTrainer:
    """
    Modular Trainer for Explainable ViT and Baseline CNN models.
    Supports Class-Balanced Focal Loss, Cosine LR Annealing, and Clinical Checkpointing.
    """
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        class_names: List[str],
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model
        self.config = config
        self.class_names = class_names
        self.device = config.device
        self.model.to(self.device)

        # Loss function selection
        if config.loss_type == "focal":
            self.criterion = FocalLoss(alpha=class_weights, gamma=config.focal_gamma)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device) if class_weights is not None else None)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-6
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_balanced_acc": [],
            "val_macro_f1": [],
            "val_melanoma_sens": []
        }
        self.best_metric_val = -1.0

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for images, labels, _ in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(num_batches, 1)

    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, Any]]:
        self.model.eval()
        total_loss = 0.0
        y_trues = []
        y_probs = []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                y_trues.extend(labels.cpu().numpy())
                y_probs.append(probs)

        avg_loss = total_loss / max(len(val_loader), 1)
        y_trues_arr = np.array(y_trues)
        y_probs_arr = np.concatenate(y_probs, axis=0) if y_probs else np.zeros((0, len(self.class_names)))

        metrics = calculate_clinical_metrics(y_trues_arr, y_probs_arr, self.class_names)
        return avg_loss, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_tag: str = "vit_lesion_classifier"
    ) -> Dict[str, Any]:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        print(f"\n--- Starting Training: {model_tag} ({self.config.epochs} epochs on {self.device}) ---")

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate(val_loader)
            self.scheduler.step()
            elapsed = time.time() - t0

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["val_balanced_acc"].append(val_metrics["balanced_accuracy"])
            self.history["val_macro_f1"].append(val_metrics["macro_f1"])
            self.history["val_melanoma_sens"].append(val_metrics["melanoma_sensitivity"])

            print(
                f"Epoch [{epoch:02d}/{self.config.epochs:02d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Acc: {val_metrics['accuracy']*100:.1f}% | BalAcc: {val_metrics['balanced_accuracy']*100:.1f}% | "
                f"Macro-F1: {val_metrics['macro_f1']:.3f} | Mel-Sens: {val_metrics['melanoma_sensitivity']*100:.1f}%"
            )

            # Checkpoint on best target metric
            target_val = val_metrics.get(self.config.save_best_metric, val_metrics["macro_f1"])
            if target_val > self.best_metric_val:
                self.best_metric_val = target_val
                best_path = os.path.join(self.config.checkpoint_dir, f"{model_tag}_best.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "metrics": val_metrics,
                    "class_names": self.class_names
                }, best_path)
                print(f"  --> Saved new best checkpoint: {best_path} ({self.config.save_best_metric}: {target_val:.4f})")

        return {
            "history": self.history,
            "best_metric": self.best_metric_val
        }
