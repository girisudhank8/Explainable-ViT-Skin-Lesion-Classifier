import os
import sys
import numpy as np
import torch
import pandas as pd

from src.models.vit_classifier import ExplainableViT
from src.models.cnn_baseline import BaselineCNN
from src.xai.attention_rollout import AttentionRollout
from src.xai.gradcam import GradCAM
from src.xai.visualizer import overlay_heatmap, detect_artifact_vs_pathology
from src.data.transforms import get_val_transforms, get_train_transforms
from src.data.dataset import SkinLesionDataset
from src.data.sample_data import generate_sample_dataset
from src.eval.metrics import calculate_clinical_metrics
from src.training.trainer import ModelTrainer, TrainingConfig
from torch.utils.data import DataLoader


def run_all_tests():
    print("=================================================================")
    print("[TEST] RUNNING SYSTEM VALIDATION FOR EXPLAINABLE ViT PROTOTYPE")
    print("=================================================================\n")

    # 1. Generate demo dataset
    print("[1/6] Testing Data Generator...")
    demo_dir = "data/demo_samples"
    df = generate_sample_dataset(demo_dir, num_samples_per_class=4)
    assert len(df) == 28, f"Expected 28 samples, got {len(df)}"
    print("  --> Dataset generation PASSED. Total samples:", len(df))

    # 2. Test ViT Forward Pass & Attention Hooks
    print("\n[2/6] Testing ExplainableViT Architecture...")
    dummy_input = torch.randn(2, 3, 224, 224)
    vit = ExplainableViT.create_small(num_classes=7, img_size=224)
    logits = vit(dummy_input)
    assert logits.shape == (2, 7), f"Expected logits shape (2, 7), got {logits.shape}"
    
    attn_mats = vit.get_all_attention_matrices()
    assert len(attn_mats) == 8, f"Expected 8 layers of attention for ViT-small, got {len(attn_mats)}"
    assert attn_mats[0].shape == (2, 4, 197, 197), f"Expected (2, 4, 197, 197), got {attn_mats[0].shape}"
    print("  --> ViT Forward Pass & Multi-Head Attention hook extraction PASSED.")

    # 3. Test Attention Rollout (XAI)
    print("\n[3/6] Testing Attention Rollout Engine...")
    rollout = AttentionRollout(vit, discard_ratio=0.85, add_residual=True)
    heatmaps, rollout_logits = rollout(dummy_input)
    assert heatmaps.shape == (2, 224, 224), f"Expected heatmap shape (2, 224, 224), got {heatmaps.shape}"
    assert np.all(heatmaps >= 0.0) and np.all(heatmaps <= 1.0), "Heatmaps must be normalized in [0, 1]"
    print("  --> Attention Rollout computation & spatial upsampling PASSED.")

    # 4. Test CNN Baseline & Grad-CAM
    print("\n[4/6] Testing Baseline CNN & Grad-CAM...")
    cnn = BaselineCNN(num_classes=7)
    gradcam = GradCAM(cnn)
    cnn_heatmaps, cnn_logits = gradcam(dummy_input)
    assert cnn_heatmaps.shape == (2, 224, 224), f"Expected CNN heatmap shape (2, 224, 224), got {cnn_heatmaps.shape}"
    print("  --> Baseline CNN & Grad-CAM PASSED.")

    # 5. Test Artifact vs Pathology Diagnostics
    print("\n[5/6] Testing Artifact Diagnostics & Overlay...")
    dummy_rgb = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    blended = overlay_heatmap(dummy_rgb, heatmaps[0], colormap="turbo", alpha=0.5)
    assert blended.shape == (224, 224, 3)
    diag = detect_artifact_vs_pathology(dummy_rgb, heatmaps[0])
    assert "central_attention_ratio" in diag and "is_artifact_suspect" in diag
    print("  --> Heatmap overlay and artifact analysis PASSED.")

    # 6. Test Training & Checkpoint Generation (2 quick epochs)
    print("\n[6/6] Testing Training Engine & Checkpointing...")
    class_names = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    dataset = SkinLesionDataset(
        df=df,
        img_dir=os.path.join(demo_dir, "images"),
        class_names=class_names,
        transform=get_train_transforms(img_size=224)
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    config = TrainingConfig(epochs=2, lr=1e-3, batch_size=8, device="cpu", checkpoint_dir="checkpoints")
    trainer = ModelTrainer(vit, config, class_names=class_names)
    fit_res = trainer.fit(loader, loader, model_tag="vit_lesion_classifier")
    
    assert os.path.exists("checkpoints/vit_lesion_classifier_best.pt"), "Checkpoint file was not created!"
    print("  --> ViT Trainer and Checkpoint saving PASSED.")

    # Also save CNN checkpoint
    cnn_trainer = ModelTrainer(cnn, config, class_names=class_names)
    cnn_trainer.fit(loader, loader, model_tag="cnn_baseline")
    assert os.path.exists("checkpoints/cnn_baseline_best.pt"), "CNN Checkpoint file was not created!"
    print("  --> CNN Baseline Trainer and Checkpoint saving PASSED.")

    print("\n=================================================================")
    print("[SUCCESS] ALL TESTS PASSED SUCCESSFULLY! PROTOTYPE READY.")
    print("=================================================================")


if __name__ == "__main__":
    run_all_tests()
