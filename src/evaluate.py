"""Metric helpers for both tasks.

Kept deliberately small and dependency-light: each function takes arrays and
returns a plain ``dict`` of JSON-serialisable numbers, so results can be written
straight to ``models/metrics.json`` and read back by the README and tests.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


def outcome_metrics(y_true, y_pred, y_proba=None, labels=None) -> dict:
    """Classification metrics for Task A.

    ``y_proba`` (optional) enables log-loss; ``labels`` fixes the class order for
    the confusion matrix and probability columns.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    if labels is not None:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["labels"] = list(labels)
    if y_proba is not None:
        metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=labels))
    return metrics


def rating_metrics(y_true, y_pred) -> dict:
    """Regression metrics for Task B."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def higher_rated_wins(rating_diff) -> np.ndarray:
    """Heuristic Task A baseline: the higher-rated side wins (ties -> white)."""
    rating_diff = np.asarray(rating_diff)
    return np.where(rating_diff >= 0, "white", "black")
