from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score, f1_score


def compute_isic2024_pauc(y_true: np.ndarray, y_pred_prob: np.ndarray, min_tpr: float = 0.80) -> float:
    """
    Computes the official ISIC 2024 Challenge metric:
    Partial Area Under the ROC Curve (pAUC) above a minimum True Positive Rate (min_tpr = 0.80).

    Formula as per ISIC 2024 Challenge:
    pAUC = (AUC - (1 - min_tpr)^2 / 2) / (min_tpr * (1 - min_tpr / 2))
    Normalized so that perfect prediction = 1.0, random guess = 0.5.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_prob = np.asarray(y_pred_prob).astype(float)

    # Check if both classes are present
    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)

    # Find the region where TPR >= min_tpr
    # We invert the axis to integrate FPR from TPR=min_tpr to TPR=1.0
    idx = np.where(tpr >= min_tpr)[0]
    if len(idx) == 0:
        return 0.0

    # Include the interpolated point at TPR = min_tpr
    first_idx = idx[0]
    if first_idx > 0 and tpr[first_idx] > min_tpr:
        # Linear interpolation for FPR at min_tpr
        tpr_prev, tpr_curr = tpr[first_idx - 1], tpr[first_idx]
        fpr_prev, fpr_curr = fpr[first_idx - 1], fpr[first_idx]
        if tpr_curr != tpr_prev:
            interp_fpr = fpr_prev + (min_tpr - tpr_prev) * (fpr_curr - fpr_prev) / (tpr_curr - tpr_prev)
        else:
            interp_fpr = fpr_prev
        tpr_sub = np.concatenate(([min_tpr], tpr[idx]))
        fpr_sub = np.concatenate(([interp_fpr], fpr[idx]))
    else:
        tpr_sub = tpr[idx]
        fpr_sub = fpr[idx]

    # Partial AUC calculation via trapezoidal rule (NumPy 2.0+ uses trapezoid)
    trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    raw_pauc = trap_fn(tpr_sub, fpr_sub)

    # ISIC 2024 metric normalization
    max_area = 1.0 - min_tpr
    min_area = (1.0 - min_tpr) * (1.0 - min_tpr) / 2.0  # random guessing area
    normalized_pauc = (raw_pauc - min_area) / (max_area - min_area) if (max_area - min_area) > 0 else raw_pauc

    return float(np.clip(normalized_pauc, 0.0, 1.0))


def compute_sensitivity_at_specificity(y_true: np.ndarray, y_pred_prob: np.ndarray, target_spec: float = 0.95) -> float:
    """Computes Sensitivity (TPR) at a fixed high Clinical Specificity (e.g., 95%)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred_prob = np.asarray(y_pred_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return 0.0

    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    target_fpr = 1.0 - target_spec

    # Find the highest TPR where FPR <= target_fpr
    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) == 0:
        return 0.0
    return float(np.max(tpr[valid_idx]))


def calculate_isic2024_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, float]:
    """Computes full evaluation suite for ISIC 2024 3D-TBP dataset."""
    y_true = np.asarray(y_true).astype(int)
    y_pred_prob = np.asarray(y_pred_prob).astype(float)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        total_auc = float(auc(fpr, tpr))
    except Exception:
        total_auc = 0.5

    pauc = compute_isic2024_pauc(y_true, y_pred_prob, min_tpr=0.80)
    sens_at_95spec = compute_sensitivity_at_specificity(y_true, y_pred_prob, target_spec=0.95)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "pauc_80": pauc,
        "total_roc_auc": total_auc,
        "sens_at_95_spec": sens_at_95spec,
        "balanced_accuracy": bal_acc,
        "f1_score": f1
    }
