from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)


def calculate_clinical_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Calculates comprehensive clinical and statistical performance metrics for multi-class skin lesion diagnosis.
    
    Args:
        y_true: 1D array of true class indices [N]
        y_pred_probs: 2D array of predicted class probabilities [N, num_classes]
        class_names: List of class name strings
    """
    y_pred = np.argmax(y_pred_probs, axis=1)
    num_classes = len(class_names)

    # Standard & Balanced Accuracies
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # Multi-class Precision, Recall (Sensitivity), and F1
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
    )

    per_class_dict = {}
    for i, cname in enumerate(class_names):
        per_class_dict[cname] = {
            "precision": float(p_per_class[i]),
            "sensitivity_recall": float(r_per_class[i]),
            "f1_score": float(f1_per_class[i]),
            "support": int(support[i])
        }

    # One-vs-Rest ROC-AUC
    try:
        if num_classes == 2:
            roc_auc_macro = roc_auc_score(y_true, y_pred_probs[:, 1])
            roc_auc_weighted = roc_auc_macro
        else:
            roc_auc_macro = roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average="macro")
            roc_auc_weighted = roc_auc_score(y_true, y_pred_probs, multi_class="ovr", average="weighted")
    except Exception:
        roc_auc_macro = 0.0
        roc_auc_weighted = 0.0

    # Specific clinical metrics (Melanoma sensitivity is highest clinical priority)
    mel_idx = 0
    for idx, name in enumerate(class_names):
        if "MEL" in name.upper() or "MELANOMA" in name.upper():
            mel_idx = idx
            break

    melanoma_sensitivity = float(r_per_class[mel_idx]) if mel_idx < len(r_per_class) else 0.0

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "roc_auc_macro": float(roc_auc_macro),
        "roc_auc_weighted": float(roc_auc_weighted),
        "melanoma_sensitivity": melanoma_sensitivity,
        "per_class": per_class_dict,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()
    }


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix - Skin Lesion Classification",
    normalize: bool = True
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    if normalize:
        cm_disp = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        fmt = ".2f"
    else:
        cm_disp = cm
        fmt = "d"

    sns.heatmap(
        cm_disp,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax
    )
    ax.set_ylabel("True Diagnosis (Ground Truth)", fontweight="bold")
    ax.set_xlabel("Predicted Diagnosis (AI Decision)", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: List[str]
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    num_classes = len(class_names)

    for i in range(num_classes):
        # Binary target for class i
        y_binary = (y_true == i).astype(int)
        if len(np.unique(y_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_binary, y_pred_probs[:, i])
            roc_score = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_score:.3f})", lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Random Chance (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
    ax.set_title("Multi-Class One-vs-Rest (OvR) ROC Curves", fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig
