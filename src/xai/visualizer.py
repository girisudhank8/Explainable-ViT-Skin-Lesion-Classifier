import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Tuple, Optional, Dict, Any
import cv2


def overlay_heatmap(
    img_rgb: np.ndarray,
    heatmap: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.55
) -> np.ndarray:
    """
    Overlays a 2D [0, 1] normalized heatmap onto an RGB image [0, 255] or [0, 1].

    Args:
        img_rgb: uint8 array (H, W, 3) or float array (H, W, 3) in [0, 1]
        heatmap: float array (H, W) in [0, 1]
        colormap: matplotlib colormap name ('jet', 'turbo', 'viridis', 'plasma')
        alpha: heatmap transparency (0 = only image, 1 = only heatmap)

    Returns:
        blended_rgb: uint8 array (H, W, 3) in [0, 255]
    """
    if img_rgb.dtype != np.uint8:
        if img_rgb.max() <= 1.0:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        else:
            img_rgb = img_rgb.astype(np.uint8)

    h, w = img_rgb.shape[:2]
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    heatmap = np.clip(heatmap, 0.0, 1.0)
    cmap_fn = cm.get_cmap(colormap)
    colored_heatmap = cmap_fn(heatmap)[:, :, :3]  # [H, W, 3] in [0, 1]
    colored_heatmap_uint8 = (colored_heatmap * 255).astype(np.uint8)

    # Alpha blend
    blended = (1.0 - alpha) * img_rgb.astype(np.float32) + alpha * colored_heatmap_uint8.astype(np.float32)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended


def generate_explanation_figure(
    img_rgb: np.ndarray,
    vit_heatmap: np.ndarray,
    vit_pred_label: str,
    vit_pred_conf: float,
    cnn_heatmap: Optional[np.ndarray] = None,
    cnn_pred_label: Optional[str] = None,
    cnn_pred_conf: Optional[float] = None
) -> plt.Figure:
    """
    Generates a structured clinical visual explanation figure comparing original dermoscopy,
    ViT Attention Rollout, and optional CNN Grad-CAM.
    """
    ncols = 3 if cnn_heatmap is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), dpi=150)

    # 1. Original Dermoscopy
    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Dermoscopy Image\n(High-Resolution)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. ViT Attention Rollout
    vit_overlay = overlay_heatmap(img_rgb, vit_heatmap, colormap="turbo", alpha=0.55)
    axes[1].imshow(vit_overlay)
    axes[1].set_title(f"ViT Attention Rollout\n{vit_pred_label} ({vit_pred_conf*100:.1f}%)", fontsize=11, fontweight="bold", color="darkgreen")
    axes[1].axis("off")

    # 3. Optional CNN Grad-CAM
    if cnn_heatmap is not None and ncols == 3:
        cnn_overlay = overlay_heatmap(img_rgb, cnn_heatmap, colormap="jet", alpha=0.55)
        axes[2].imshow(cnn_overlay)
        axes[2].set_title(f"Baseline CNN Grad-CAM\n{cnn_pred_label} ({cnn_pred_conf*100:.1f}%)", fontsize=11, fontweight="bold", color="darkblue")
        axes[2].axis("off")

    plt.tight_layout()
    return fig


def detect_artifact_vs_pathology(
    img_rgb: np.ndarray,
    heatmap: np.ndarray,
    focus_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Analyzes spatial focus distribution to detect whether attention is centered on
    lesion pathology vs peripheral image artifacts (hairs, dark corners, ruler marks).

    Returns diagnostics dictionary with center-of-mass focus, peripheral ratio, and artifact warning flag.
    """
    h, w = img_rgb.shape[:2]
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    if heatmap.max() > heatmap.min():
        norm_map = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        norm_map = heatmap

    # Binary mask of high attention
    high_attn_mask = norm_map >= focus_threshold
    
    # Calculate margin vs central core (central 60% of image is typical lesion location)
    margin_y = int(h * 0.2)
    margin_x = int(w * 0.2)
    center_mask = np.zeros_like(norm_map, dtype=bool)
    center_mask[margin_y:h-margin_y, margin_x:w-margin_x] = True
    
    central_attention = norm_map[center_mask].sum()
    peripheral_attention = norm_map[~center_mask].sum()
    total_attention = central_attention + peripheral_attention + 1e-8
    
    central_ratio = float(central_attention / total_attention)
    peripheral_ratio = float(peripheral_attention / total_attention)
    
    # Check for hair/artifact edge focus
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if len(img_rgb.shape) == 3 else img_rgb
    edges = cv2.Canny(gray, 50, 150) > 0
    edge_overlap = (high_attn_mask & edges).sum() / (high_attn_mask.sum() + 1e-8)

    is_artifact_suspect = (peripheral_ratio > 0.45) or (edge_overlap > 0.40 and central_ratio < 0.5)

    return {
        "central_attention_ratio": central_ratio,
        "peripheral_attention_ratio": peripheral_ratio,
        "edge_overlap_ratio": float(edge_overlap),
        "is_artifact_suspect": is_artifact_suspect,
        "interpretation": "High pathology focus on central lesion tissue." if not is_artifact_suspect else "Warning: Potential artifact interference (peripheral/hair bias)."
    }
