import os
import numpy as np
import torch
import pandas as pd

from src.eval.isic_metrics import compute_isic2024_pauc, calculate_isic2024_metrics
from src.models.hybrid_vit import HybridViT
from src.xai.attention_rollout import AttentionRollout
from src.data.isic2024_loader import ISIC2024Dataset, create_isic2024_dataloaders
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.sample_data import generate_sample_dataset


def test_isic2024_pipeline():
    print("=================================================================")
    print("[TEST] RUNNING ISIC 2024 3D-TBP PIPELINE & HYBRID ViT VERIFICATION")
    print("=================================================================\n")

    # 1. Test pAUC calculation
    print("[1/4] Testing Official ISIC 2024 pAUC Metric (pAUC > 80% TPR)...")
    y_true = np.array([0]*80 + [1]*20)
    y_prob_perfect = np.array([0.1]*80 + [0.9]*20)
    pauc_perf = compute_isic2024_pauc(y_true, y_prob_perfect, min_tpr=0.80)
    assert pauc_perf > 0.90, f"Expected near 1.0 for perfect predictions, got {pauc_perf}"
    print(f"  --> pAUC computation PASSED (Perfect score: {pauc_perf:.4f}).")

    # 2. Test HybridViT Architecture (Paper 02)
    print("\n[2/4] Testing Hybrid CNN-Transformer (HybridViT)...")
    hybrid = HybridViT(img_size=224, num_classes=7)
    dummy_x = torch.randn(2, 3, 224, 224)
    logits = hybrid(dummy_x)
    assert logits.shape == (2, 7), f"Expected logits shape (2, 7), got {logits.shape}"
    
    attn_mats = hybrid.get_all_attention_matrices()
    assert len(attn_mats) == 8, f"Expected 8 layers of attention, got {len(attn_mats)}"
    print("  --> HybridViT Forward Pass and Multi-Head Attention hook extraction PASSED.")

    # 3. Test Attention Rollout on HybridViT
    print("\n[3/4] Testing Attention Rollout on HybridViT...")
    rollout = AttentionRollout(hybrid, discard_ratio=0.85, add_residual=True)
    heatmaps, rollout_logits = rollout(dummy_x)
    assert heatmaps.shape == (2, 224, 224), f"Expected shape (2, 224, 224), got {heatmaps.shape}"
    print("  --> Attention Rollout on HybridViT PASSED.")

    # 4. Test ISIC 2024 Dataset & Loader
    print("\n[4/4] Testing ISIC 2024 Loader & Balanced Sampling...")
    demo_dir = "data/isic2024_demo"
    df = generate_sample_dataset(demo_dir, num_samples_per_class=6)
    df["isic_id"] = df["image_id"]
    df["target"] = df["dx"].apply(lambda x: 1 if x in ["MEL", "BCC"] else 0)
    
    train_loader, val_loader, info = create_isic2024_dataloaders(
        metadata_df=df,
        image_source=os.path.join(demo_dir, "images"),
        train_transform=get_train_transforms(224),
        val_transform=get_val_transforms(224),
        batch_size=8,
        use_balanced_sampler=True
    )
    assert len(train_loader) > 0 and len(val_loader) > 0
    print(f"  --> ISIC 2024 DataLoaders created successfully ({info['train_samples']} train, {info['val_samples']} val).")

    print("\n=================================================================")
    print("[SUCCESS] ALL ISIC 2024 PIPELINE & HYBRID ViT TESTS PASSED!")
    print("=================================================================")


if __name__ == "__main__":
    test_isic2024_pipeline()
