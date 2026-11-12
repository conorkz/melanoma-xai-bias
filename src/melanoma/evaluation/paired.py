"""Paired bootstrap comparisons for thresholded sensitivity."""

import numpy as np


def _align(frame_a, frame_b):
    merged = frame_a[["image_id", "y_true", "y_pred"]].merge(
        frame_b[["image_id", "y_true", "y_pred"]],
        on="image_id",
        suffixes=("_a", "_b"),
    )
    if not (merged["y_true_a"] == merged["y_true_b"]).all():
        raise ValueError("The two frames disagree on y_true for at least one image")
    return merged


def _sensitivity(y_true, y_pred):
    positives = y_true == 1
    if positives.sum() == 0:
        return float("nan")
    return float(y_pred[positives].mean())


def _bootstrap_delta(y_true, pred_a, pred_b, n_boot, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_true))
    values = []
    for _ in range(n_boot):
        sample = rng.choice(indices, size=len(indices), replace=True)
        if (y_true[sample] == 1).sum() == 0:
            continue
        values.append(
            _sensitivity(y_true[sample], pred_a[sample])
            - _sensitivity(y_true[sample], pred_b[sample])
        )
    if not values:
        return None
    return np.asarray(values)


def _summarize(point, values):
    if values is None:
        return {
            "delta": float(point),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
            "n_boot_valid": 0,
        }
    p_value = 2.0 * min(float((values <= 0).mean()), float((values >= 0).mean()))
    return {
        "delta": float(point),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "p_value": float(min(1.0, p_value)),
        "n_boot_valid": int(len(values)),
    }


def paired_sensitivity_change(frame_a, frame_b, n_boot=1000, seed=0):
    """Sensitivity change from condition A to condition B, paired by image."""
    merged = _align(frame_a, frame_b)
    y_true = merged["y_true_a"].to_numpy()
    pred_a = merged["y_pred_a"].to_numpy()
    pred_b = merged["y_pred_b"].to_numpy()
    point = _sensitivity(y_true, pred_b) - _sensitivity(y_true, pred_a)
    values = _bootstrap_delta(y_true, pred_b, pred_a, n_boot, seed)
    result = _summarize(point, values)
    result["n_pairs"] = int(len(merged))
    result["n_positive"] = int(y_true.sum())
    return result


def difference_in_differences(frame_a1, frame_a2, frame_b1, frame_b2, n_boot=1000, seed=0):
    """Difference in the A-to-B sensitivity change between two architectures."""
    merged = frame_a1[["image_id", "y_true", "y_pred"]].merge(
        frame_a2[["image_id", "y_pred"]], on="image_id", suffixes=("_a1", "_a2")
    )
    merged = merged.merge(
        frame_b1[["image_id", "y_pred"]], on="image_id"
    ).rename(columns={"y_pred": "y_pred_b1"})
    merged = merged.merge(
        frame_b2[["image_id", "y_pred"]], on="image_id"
    ).rename(columns={"y_pred": "y_pred_b2"})

    y_true = merged["y_true"].to_numpy()
    a1 = merged["y_pred_a1"].to_numpy()
    a2 = merged["y_pred_a2"].to_numpy()
    b1 = merged["y_pred_b1"].to_numpy()
    b2 = merged["y_pred_b2"].to_numpy()
    point = (_sensitivity(y_true, a2) - _sensitivity(y_true, a1)) - (
        _sensitivity(y_true, b2) - _sensitivity(y_true, b1)
    )

    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_true))
    values = []
    for _ in range(n_boot):
        sample = rng.choice(indices, size=len(indices), replace=True)
        if (y_true[sample] == 1).sum() == 0:
            continue
        delta_a = _sensitivity(y_true[sample], a2[sample]) - _sensitivity(
            y_true[sample], a1[sample]
        )
        delta_b = _sensitivity(y_true[sample], b2[sample]) - _sensitivity(
            y_true[sample], b1[sample]
        )
        values.append(delta_a - delta_b)
    result = _summarize(point, np.asarray(values) if values else None)
    result["n_pairs"] = int(len(merged))
    result["n_positive"] = int(y_true.sum())
    return result
