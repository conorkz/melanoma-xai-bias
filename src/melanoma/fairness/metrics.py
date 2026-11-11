"""Group-wise fairness metrics."""

import numpy as np
import pandas as pd

from melanoma.training.metrics import binary_metrics, bootstrap_ci


def _validate(frame, group_col, y_true_col, y_prob_col):
    missing = [column for column in (group_col, y_true_col, y_prob_col) if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def group_metrics(
    frame: pd.DataFrame,
    group_col: str = "tone_group",
    y_true_col: str = "y_true",
    y_prob_col: str = "y_prob",
    threshold: float = 0.5,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute classification metrics and bootstrap intervals per group."""
    _validate(frame, group_col, y_true_col, y_prob_col)
    rows = []
    for group, subset in frame.groupby(group_col, sort=True):
        y_true = subset[y_true_col].to_numpy()
        y_prob = subset[y_prob_col].to_numpy()
        predictions = (y_prob >= threshold).astype(int)
        metrics = binary_metrics(y_true, predictions, y_prob)
        metrics[group_col] = group
        metrics["fpr"] = (
            1.0 - metrics["specificity"] if np.isfinite(metrics["specificity"]) else float("nan")
        )
        metrics["fnr"] = (
            1.0 - metrics["sensitivity"] if np.isfinite(metrics["sensitivity"]) else float("nan")
        )
        for metric in ("sensitivity", "specificity", "auc"):
            interval = bootstrap_ci(
                y_true,
                y_prob,
                metric=metric,
                threshold=threshold,
                n_boot=n_boot,
                seed=seed,
            )
            metrics[f"{metric}_low"] = interval["low"]
            metrics[f"{metric}_high"] = interval["high"]
        rows.append(metrics)
    return pd.DataFrame(rows)


def disparity_summary(group_frame: pd.DataFrame, group_col: str = "group", metric: str = "sensitivity") -> dict:
    """Summarize the maximum group gap and the light/dark ratio."""
    if group_col not in group_frame.columns:
        raise ValueError(f"Group column '{group_col}' not found")
    if metric not in group_frame.columns:
        raise ValueError(f"Metric column '{metric}' not found")
    values = group_frame.set_index(group_col)[metric].astype(float)
    finite = values[np.isfinite(values)]
    if finite.empty:
        return {"max_min_gap": float("nan"), "light_dark_ratio": float("nan")}
    light = finite.get("light", float("nan"))
    dark = finite.get("dark", float("nan"))
    ratio = (
        float(light / dark)
        if np.isfinite(light) and np.isfinite(dark) and dark != 0
        else float("nan")
    )
    return {
        "max_min_gap": float(finite.max() - finite.min()),
        "light_dark_ratio": ratio,
    }
