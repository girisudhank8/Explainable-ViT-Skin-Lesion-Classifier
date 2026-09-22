import time
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from .metrics import calculate_clinical_metrics
from ..xai.attention_rollout import AttentionRollout
from ..xai.gradcam import GradCAM
from ..xai.visualizer import detect_artifact_vs_pathology
from ..data.transforms import denormalize_tensor


def benchmark_models(
    vit_model: torch.nn.Module,
    cnn_model: torch.nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Runs an empirical benchmark comparing Explainable ViT vs Baseline CNN across:
    1. Clinical Diagnostic Performance (Accuracy, Sensitivity, F1, Balanced Acc)
    2. Inference Latency & Parameter Footprint
    3. Explainability Quality & Artifact Susceptibility
    """
    vit_model.to(device).eval()
    cnn_model.to(device).eval()

    vit_rollout = AttentionRollout(vit_model)
    cnn_gradcam = GradCAM(cnn_model)

    y_trues = []
    vit_preds_list = []
    cnn_preds_list = []
    
    vit_artifact_suspect_count = 0
    cnn_artifact_suspect_count = 0
    total_samples = 0

    # Latency tracking
    vit_latencies = []
    cnn_latencies = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            y_trues.extend(labels.numpy())

            # ViT Inference
            t0 = time.perf_counter()
            vit_logits = vit_model(images)
            vit_latencies.append((time.perf_counter() - t0) / images.shape[0])
            vit_probs = torch.softmax(vit_logits, dim=-1).cpu().numpy()
            vit_preds_list.append(vit_probs)

            # CNN Inference
            t0 = time.perf_counter()
            cnn_logits = cnn_model(images)
            cnn_latencies.append((time.perf_counter() - t0) / images.shape[0])
            cnn_probs = torch.softmax(cnn_logits, dim=-1).cpu().numpy()
            cnn_preds_list.append(cnn_probs)

    y_true_arr = np.array(y_trues)
    vit_probs_arr = np.concatenate(vit_preds_list, axis=0)
    cnn_probs_arr = np.concatenate(cnn_preds_list, axis=0)

    # Compute metrics
    vit_metrics = calculate_clinical_metrics(y_true_arr, vit_probs_arr, class_names)
    cnn_metrics = calculate_clinical_metrics(y_true_arr, cnn_probs_arr, class_names)

    # Parameter counts
    vit_params = sum(p.numel() for p in vit_model.parameters())
    cnn_params = sum(p.numel() for p in cnn_model.parameters())

    return {
        "vit": {
            "metrics": vit_metrics,
            "avg_latency_ms": float(np.mean(vit_latencies) * 1000),
            "param_count": vit_params,
            "architecture": "Vision Transformer (ViT)"
        },
        "cnn": {
            "metrics": cnn_metrics,
            "avg_latency_ms": float(np.mean(cnn_latencies) * 1000),
            "param_count": cnn_params,
            "architecture": "Baseline CNN"
        }
    }


def compare_artifact_sensitivity(
    vit_model: torch.nn.Module,
    cnn_model: torch.nn.Module,
    sample_tensor: torch.Tensor,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Directly evaluates a single lesion case (with artifacts like hair or markings)
    to compare whether ViT Attention Rollout isolates lesion pathology better than CNN Grad-CAM.
    """
    img_rgb = denormalize_tensor(sample_tensor[0])
    
    # ViT Attention Rollout
    vit_rollout = AttentionRollout(vit_model)
    vit_mask, vit_logits = vit_rollout(sample_tensor)
    vit_prob = torch.softmax(vit_logits, dim=-1)[0]
    vit_top_idx = int(vit_prob.argmax().item())
    vit_diag = detect_artifact_vs_pathology(img_rgb, vit_mask[0])

    # CNN Grad-CAM
    cnn_gradcam = GradCAM(cnn_model)
    cnn_mask, cnn_logits = cnn_gradcam(sample_tensor)
    cnn_prob = torch.softmax(cnn_logits, dim=-1)[0]
    cnn_top_idx = int(cnn_prob.argmax().item())
    cnn_diag = detect_artifact_vs_pathology(img_rgb, cnn_mask[0])

    return {
        "image_rgb": img_rgb,
        "vit": {
            "pred_label": class_names[vit_top_idx],
            "confidence": float(vit_prob[vit_top_idx].item()),
            "heatmap": vit_mask[0],
            "diagnostics": vit_diag
        },
        "cnn": {
            "pred_label": class_names[cnn_top_idx],
            "confidence": float(cnn_prob[cnn_top_idx].item()),
            "heatmap": cnn_mask[0],
            "diagnostics": cnn_diag
        }
    }
