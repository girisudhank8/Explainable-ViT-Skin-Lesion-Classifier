# Lazy imports — heavy deps (matplotlib, seaborn, sklearn) may be
# blocked by WDAC policy on some machines, so we guard them here.
try:
    from .metrics import calculate_clinical_metrics, compute_confusion_matrix, plot_confusion_matrix, plot_multiclass_roc
    from .benchmark import benchmark_models, compare_artifact_sensitivity
    __all__ = [
        "calculate_clinical_metrics",
        "compute_confusion_matrix",
        "plot_confusion_matrix",
        "plot_multiclass_roc",
        "benchmark_models",
        "compare_artifact_sensitivity",
    ]
except Exception:
    # Visualization/benchmark modules unavailable — training still works fine
    __all__ = []
