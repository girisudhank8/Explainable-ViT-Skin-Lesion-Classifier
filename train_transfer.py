"""
Transfer Learning script for skin lesion classification.
Strategy: Use ImageNet-pretrained EfficientNet-B0 from timm.
Phase 1 (5 epochs): Freeze backbone, train only classifier head
Phase 2 (20 epochs): Unfreeze all, fine-tune with low LR

This is the CORRECT approach for small datasets (84 train samples).
Training from scratch on 84 images will NEVER converge for a 6M param model.
"""
import sys, os, argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

sys.stdout.reconfigure(line_buffering=True)

from src.data.dataset import SkinLesionDataset
from src.data.transforms import get_train_transforms, get_val_transforms


# ── Focal Loss ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Pretrained Transfer Learning Model ────────────────────────────────────
class TransferLesionClassifier(nn.Module):
    """
    EfficientNet-B0 pretrained on ImageNet, with custom 7-class head.
    Optionally wraps a ViT backbone (vit_small_patch16_224) if available.
    """
    def __init__(self, num_classes=7, backbone="efficientnet_b0.ra_in1k", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        print(f"  Backbone: {backbone} | feat_dim={feat_dim} | pretrained={pretrained}", flush=True)

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("  Backbone FROZEN (head-only phase)", flush=True)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("  Backbone UNFROZEN (full fine-tune phase)", flush=True)


# ── Train / Validate ────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, criterion, device):
    model.train()
    total = 0.0
    for imgs, lbls, _ in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        opt.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total, yt, yp = 0.0, [], []
    for imgs, lbls, _ in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = model(imgs)
        total += criterion(logits, lbls).item()
        yt.extend(lbls.cpu().numpy())
        yp.extend(logits.argmax(1).cpu().numpy())
    yt, yp = np.array(yt), np.array(yp)
    acc  = accuracy_score(yt, yp)
    bal  = balanced_accuracy_score(yt, yp)
    f1   = f1_score(yt, yp, average="macro", zero_division=0)
    mel  = (yp[yt==0]==0).mean() if (yt==0).sum()>0 else 0.0
    per_class_acc = {}
    CLASS_NAMES = ["MEL","NV","BCC","AKIEC","BKL","DF","VASC"]
    for i, cn in enumerate(CLASS_NAMES):
        mask = (yt == i)
        per_class_acc[cn] = float((yp[mask]==i).mean()) if mask.sum()>0 else 0.0
    return total/max(len(loader),1), acc, bal, f1, mel, per_class_acc


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="efficientnet_b0.ra_in1k",
                    help="timm model name. e.g. efficientnet_b0.ra_in1k or vit_small_patch16_224.augreg_in21k_ft_in1k")
    ap.add_argument("--phase1_epochs", type=int, default=8,
                    help="Epochs with frozen backbone (head only)")
    ap.add_argument("--phase2_epochs", type=int, default=22,
                    help="Epochs with full fine-tuning")
    ap.add_argument("--lr_head", type=float, default=1e-3,
                    help="LR for head-only phase")
    ap.add_argument("--lr_finetune", type=float, default=5e-5,
                    help="LR for full fine-tune phase")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--data_dir", default="data/demo_samples")
    ap.add_argument("--tag", default="vit_lesion_classifier",
                    help="Checkpoint tag (vit_lesion_classifier or cnn_baseline)")
    args = ap.parse_args()

    CLASS_NAMES = ["MEL","NV","BCC","AKIEC","BKL","DF","VASC"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}", flush=True)
    print(f"Transfer Learning | Backbone: {args.backbone}", flush=True)
    print(f"Device: {DEVICE} | Phase1: {args.phase1_epochs}ep | Phase2: {args.phase2_epochs}ep", flush=True)
    print(f"{'='*70}\n", flush=True)

    # Dataset
    df = pd.read_csv(os.path.join(args.data_dir, "metadata.csv"))
    img_dir = os.path.join(args.data_dir, "images")
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df["dx"])
    print(f"Train: {len(train_df)}  Val: {len(val_df)}", flush=True)

    mkds = lambda d, t: SkinLesionDataset(
        df=d, img_dir=img_dir, img_col="image_id", label_col="dx",
        class_names=CLASS_NAMES, transform=t)
    train_ds = mkds(train_df, get_train_transforms(224))
    val_ds   = mkds(val_df,   get_val_transforms(224))
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    vl = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    cw = train_ds.get_class_weights().to(DEVICE)
    print(f"Class weights: {cw.cpu().numpy().round(2)}", flush=True)

    # Model
    model = TransferLesionClassifier(
        num_classes=7, backbone=args.backbone, pretrained=True
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}", flush=True)

    criterion = FocalLoss(gamma=2.0, weight=cw)
    os.makedirs("checkpoints", exist_ok=True)
    best_f1 = -1.0
    best_path = f"checkpoints/{args.tag}_best.pt"

    def run_phase(phase_name, epochs, lr, start_ep=1):
        nonlocal best_f1
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-4
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)

        print(f"\n--- {phase_name} (LR={lr}) ---", flush=True)
        print(f"{'Ep':>3} {'TrLoss':>8} {'VaLoss':>8} {'Acc':>7} {'BalAcc':>7} {'F1':>7} {'MelSns':>7}", flush=True)
        print("-"*65, flush=True)

        for ep in range(start_ep, start_ep + epochs):
            t0 = time.time()
            tl_loss = train_epoch(model, tl, opt, criterion, DEVICE)
            vl_loss, acc, bal, f1, mel, pca = validate(model, vl, criterion, DEVICE)
            sched.step()
            saved = ""
            if f1 > best_f1:
                best_f1 = f1
                torch.save({
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "backbone": args.backbone,
                    "metrics": {"accuracy": acc, "balanced_accuracy": bal,
                                "macro_f1": f1, "melanoma_sensitivity": mel,
                                "per_class_accuracy": pca}
                }, best_path)
                saved = " *"
            print(f"{ep:>3} {tl_loss:>8.4f} {vl_loss:>8.4f} "
                  f"{acc*100:>6.1f}% {bal*100:>6.1f}% {f1:>7.3f} {mel*100:>6.1f}%"
                  f"  ({time.time()-t0:.1f}s){saved}", flush=True)

        # Print per-class accuracies at end of phase
        _, _, _, _, _, pca = validate(model, vl, criterion, DEVICE)
        print(f"\nPer-class acc: { {k: f'{v*100:.0f}%' for k,v in pca.items()} }", flush=True)
        return start_ep + epochs

    # Phase 1: Head only
    model.freeze_backbone()
    next_ep = run_phase("PHASE 1: Head-Only Training", args.phase1_epochs, args.lr_head, start_ep=1)

    # Phase 2: Full fine-tune
    model.unfreeze_backbone()
    run_phase("PHASE 2: Full Fine-Tune", args.phase2_epochs, args.lr_finetune, start_ep=next_ep)

    print(f"\n{'='*70}", flush=True)
    print(f"Training complete. Best Macro-F1: {best_f1:.4f}", flush=True)
    print(f"Checkpoint: {best_path}", flush=True)

if __name__ == "__main__":
    main()
