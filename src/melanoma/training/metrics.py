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
    """Sensitivity at the operating point that meets the target specificity.

    Among thresholds whose specificity is at least the target, this returns
    the one with the highest sensitivity, together with its threshold and
    achieved specificity.
    """
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


BOOTSTRAP_METRICS = ("sensitivity", "specificity", "balanced_accuracy", "accuracy", "auc")


def bootstrap_ci(
    y_true,
    y_prob,
    metric: str = "sensitivity",
    threshold: float = 0.5,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap confidence interval for a binary metric.

    Resamples rows with replacement and skips degenerate resamples (for
    example one with no positives for sensitivity, or one with a single class
    for AUC). Returns the point estimate on the full sample plus the lower and
    upper percentile bounds and the number of valid bootstrap samples.
    """
    if metric not in BOOTSTRAP_METRICS:
        raise ValueError(f"metric must be one of {BOOTSTRAP_METRICS}, got {metric!r}")
    if not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    y_true = _validate_binary(y_true)
    y_prob = np.asarray(y_prob).ravel()
    if y_prob.size != y_true.size:
        raise ValueError(f"Length mismatch: y_true {y_true.size} vs y_prob {y_prob.size}")

    if metric == "auc":
        if len(np.unique(y_true)) < 2:
            estimate = float("nan")
        else:
            estimate = float(roc_auc_score(y_true, y_prob))
    else:
        predictions = (y_prob >= threshold).astype(int)
        estimate = float(binary_metrics(y_true, predictions, y_prob)[metric])

    rng = np.random.default_rng(seed)
    values = []
    n = y_true.size
    for _ in range(n_boot):
        indices = rng.integers(0, n, n)
        sample_true = y_true[indices]
        sample_prob = y_prob[indices]
        if metric == "auc":
            if len(np.unique(sample_true)) < 2:
                continue
            value = float(roc_auc_score(sample_true, sample_prob))
        else:
            sample_pred = (sample_prob >= threshold).astype(int)
            value = float(binary_metrics(sample_true, sample_pred, sample_prob)[metric])
        if np.isfinite(value):
            values.append(value)

    if not values:
        return {
            "estimate": estimate,
            "low": float("nan"),
            "high": float("nan"),
            "n_boot": 0,
        }

    alpha = (1.0 - confidence) / 2.0
    low, high = np.percentile(values, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return {
        "estimate": estimate,
        "low": float(low),
        "high": float(high),
        "n_boot": len(values),
    }
