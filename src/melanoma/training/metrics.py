"""Binary classification metrics with explicit handling of degenerate cases."""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _validate_binary(y_true):
    y_true = np.asarray(y_true).ravel()
    if y_true.size == 0:
        raise ValueError("y_true is empty")
    unique = np.unique(y_true)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(f"y_true must be binary (0/1), got values {unique[:5]}")
    return y_true


def binary_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Compute clinical binary metrics. NaN is returned when a metric is undefined.

    sensitivity is NaN with no positive samples, specificity is NaN with no
    negative samples, and AUC is NaN without probabilities or with one class.
    """
    y_true = _validate_binary(y_true)
    y_pred = np.asarray(y_pred).ravel()
    if y_pred.size != y_true.size:
        raise ValueError(f"Length mismatch: y_true {y_true.size} vs y_pred {y_pred.size}")
    if not np.all(np.isin(np.unique(y_pred), [0, 1])):
        raise ValueError("y_pred must be binary (0/1)")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    n = y_true.size

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(sensitivity) or (precision + sensitivity) == 0:
        f1 = float("nan") if (np.isnan(precision) or np.isnan(sensitivity)) else 0.0
    else:
        f1 = 2 * precision * sensitivity / (precision + sensitivity)

    defined = [value for value in (sensitivity, specificity) if not np.isnan(value)]
    balanced = float(np.mean(defined)) if defined else float("nan")

    auc = float("nan")
    if y_prob is not None:
        y_prob = np.asarray(y_prob).ravel()
        if y_prob.size != y_true.size:
            raise ValueError(f"Length mismatch: y_true {y_true.size} vs y_prob {y_prob.size}")
        if np.isfinite(y_prob).all() and len(np.unique(y_true)) == 2:
            auc = float(roc_auc_score(y_true, y_prob))

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "accuracy": (tp + tn) / n,
        "balanced_accuracy": balanced,
        "auc": auc,
        "n": n,
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def sensitivity_at_specificity(y_true, y_prob, target_specificity: float = 0.95) -> dict:
    """Sensitivity at the highest threshold whose specificity meets the target."""
    y_true = _validate_binary(y_true)
    y_prob = np.asarray(y_prob).ravel()
    if y_prob.size != y_true.size:
        raise ValueError(f"Length mismatch: y_true {y_true.size} vs y_prob {y_prob.size}")
    if not 0.0 < target_specificity < 1.0:
        raise ValueError(f"target_specificity must be in (0, 1), got {target_specificity}")
    if len(np.unique(y_true)) < 2:
        return {"threshold": float("nan"), "sensitivity": float("nan"), "specificity": float("nan")}

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificities = 1.0 - fpr
    feasible = specificities >= target_specificity
    if not np.any(feasible):
        return {"threshold": float("nan"), "sensitivity": float("nan"), "specificity": float("nan")}
    idx = int(np.argmax(np.where(feasible, tpr, -np.inf)))
    return {
        "threshold": float(thresholds[idx]),
        "sensitivity": float(tpr[idx]),
        "specificity": float(specificities[idx]),
    }
