"""
Clean training script for ViT and CNN models on the 112-sample demo dataset.
Uses full-model training with focal loss, cosine LR schedule.
Run:
    python train_clean.py --model vit
    python train_clean.py --model cnn
"""
import sys, os, argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(line_buffering=True)

from src.models.vit_classifier import ExplainableViT
from src.models.cnn_baseline import BaselineCNN
from src.data.dataset import SkinLesionDataset
from src.data.transforms import get_train_transforms, get_val_transforms

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")
    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for imgs, lbls, _ in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
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
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    acc  = accuracy_score(yt, yp)
    bal  = balanced_accuracy_score(yt, yp)
    f1   = f1_score(yt, yp, average="macro", zero_division=0)
    mel  = (yp[yt==0]==0).mean() if (yt==0).sum()>0 else 0.0
    return total/max(len(loader),1), acc, bal, f1, mel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["vit","cnn"], default="vit")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--data_dir", default="data/demo_samples")
    args = ap.parse_args()

    CLASS_NAMES = ["MEL","NV","BCC","AKIEC","BKL","DF","VASC"]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:{DEVICE}  Model:{args.model.upper()}  Epochs:{args.epochs}", flush=True)

    df = pd.read_csv(os.path.join(args.data_dir, "metadata.csv"))
    img_dir = os.path.join(args.data_dir, "images")
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=42, stratify=df["dx"])
    print(f"Train:{len(train_df)}  Val:{len(val_df)}", flush=True)

    mkds = lambda d, t: SkinLesionDataset(df=d, img_dir=img_dir,
        img_col="image_id", label_col="dx", class_names=CLASS_NAMES, transform=t)
    train_ds = mkds(train_df, get_train_transforms(224))
    val_ds   = mkds(val_df,   get_val_transforms(224))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    cw = train_ds.get_class_weights().to(DEVICE)
    print(f"ClassWeights:{cw.cpu().numpy().round(2)}", flush=True)

    if args.model == "vit":
        model = ExplainableViT.create_small(num_classes=7, img_size=224).to(DEVICE)
        tag = "vit_lesion_classifier"
    else:
        model = BaselineCNN(num_classes=7).to(DEVICE)
        tag = "cnn_baseline"
    print(f"Params:{sum(p.numel() for p in model.parameters()):,}", flush=True)

    criterion = FocalLoss(gamma=2.0, weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    os.makedirs("checkpoints", exist_ok=True)
    best_f1, best_path = -1.0, f"checkpoints/{tag}_best.pt"

    print("\n"+"-"*72, flush=True)
    print(f"{'Ep':>3} {'TrLoss':>8} {'VaLoss':>8} {'Acc':>7} {'BalAcc':>7} {'F1':>7} {'MelSns':>7}", flush=True)
    print("-"*72, flush=True)

    for ep in range(1, args.epochs+1):
        t0 = time.time()
        tl = train_epoch(model, train_loader, opt, criterion, DEVICE)
        vl, acc, bal, f1, mel = validate(model, val_loader, criterion, DEVICE)
        sched.step()
        tag_save = ""
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"epoch":ep,"model_state_dict":model.state_dict(),
                        "class_names":CLASS_NAMES,
                        "metrics":{"accuracy":acc,"balanced_accuracy":bal,
                                   "macro_f1":f1,"melanoma_sensitivity":mel}}, best_path)
            tag_save = " *"
        print(f"{ep:>3} {tl:>8.4f} {vl:>8.4f} {acc*100:>6.1f}% {bal*100:>6.1f}% {f1:>7.3f} {mel*100:>6.1f}%  ({time.time()-t0:.1f}s){tag_save}", flush=True)

    print("-"*72, flush=True)
    print(f"Done. Best Macro-F1={best_f1:.4f} -> {best_path}", flush=True)

if __name__ == "__main__":
    main()
