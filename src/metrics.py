"""Evaluation metrics and reporting utilities for multi-class classification."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Compute primary and detailed classification metrics.
    
    Returns:
        Dict with overall accuracy, macro-F1, weighted-F1, confusion matrix,
        and per-class DataFrame.
    """
    if labels is None and class_names is not None:
        labels = list(range(len(class_names)))

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    num_entries = len(labels) if labels is not None else len(precision)
    names = class_names if class_names is not None else [f"Class_{i}" for i in range(num_entries)]

    per_class_df = pd.DataFrame({
        "class_id": labels if labels is not None else list(range(num_entries)),
        "class_name": names[:num_entries],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "per_class": per_class_df,
    }
