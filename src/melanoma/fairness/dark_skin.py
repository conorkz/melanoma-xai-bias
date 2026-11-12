"""Dark-skin robustness analysis on the HAM10000 test split."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from melanoma.training.metrics import binary_metrics


def assign_tone_groups(frame, threshold):
    """Assign light, medium, and dark groups for a given L* cutoff."""
    if "L" not in frame.columns:
        raise ValueError("frame must contain an L column")
    if threshold > 70:
        raise ValueError(f"threshold must be <= 70, got {threshold}")
    values = frame["L"].to_numpy()
    groups = np.full(len(frame), "medium", dtype=object)
    groups[values > 70] = "light"
    groups[values <= threshold] = "dark"
    result = frame.copy()
    result["tone_group"] = groups
    return result


def threshold_sensitivity(predictions, tone, thresholds=(55, 58, 60)):
    """Compute group metrics at several dark-stratum L* cutoffs."""
    required = {"image_id", "run_id", "y_true", "y_prob", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    if "L" not in tone.columns:
        raise ValueError("tone must contain an L column")

    base = predictions.merge(tone[["image_id", "L"]], on="image_id", how="left")
    base = base.dropna(subset=["L"])
    rows = []
    for threshold in thresholds:
        assigned = assign_tone_groups(base, threshold)
        for run_id, run in assigned.groupby("run_id", sort=False):
            for group in ("light", "medium", "dark"):
                subset = run[run["tone_group"] == group]
                if subset.empty:
                    continue
                metrics = binary_metrics(
                    subset["y_true"].to_numpy(),
                    subset["y_pred"].to_numpy(),
                    subset["y_prob"].to_numpy(),
                )
                rows.append(
                    {
                        "threshold": threshold,
                        "run_id": run_id,
                        "config_id": subset["config_id"].iloc[0]
                        if "config_id" in subset
                        else "",
                        "backbone": subset["backbone"].iloc[0]
                        if "backbone" in subset
                        else "",
                        "seed": int(subset["seed"].iloc[0])
                        if "seed" in subset
                        else -1,
                        "tone_group": group,
                        "n": metrics["n"],
                        "n_positive": metrics["n_positive"],
                        "sensitivity": metrics["sensitivity"],
                        "specificity": metrics["specificity"],
                        "auc": metrics["auc"],
                        "fpr": 1.0 - metrics["specificity"]
                        if np.isfinite(metrics["specificity"])
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def paired_auc_difference(frame_a, frame_b, n_boot=1000, seed=0):
    """AUC difference between two disjoint groups with a bootstrap interval."""
    point = _auc(frame_a["y_true"], frame_a["y_prob"]) - _auc(
        frame_b["y_true"], frame_b["y_prob"]
    )
    if not np.isfinite(point):
        return {
            "delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
            "n_boot_valid": 0,
        }

    rng = np.random.default_rng(seed)
    indices_a = np.arange(len(frame_a))
    indices_b = np.arange(len(frame_b))
    y_a = frame_a["y_true"].to_numpy()
    p_a = frame_a["y_prob"].to_numpy()
    y_b = frame_b["y_true"].to_numpy()
    p_b = frame_b["y_prob"].to_numpy()
    deltas = []
    for _ in range(n_boot):
        sample_a = rng.choice(indices_a, size=len(indices_a), replace=True)
        sample_b = rng.choice(indices_b, size=len(indices_b), replace=True)
        auc_a = _auc(y_a[sample_a], p_a[sample_a])
        auc_b = _auc(y_b[sample_b], p_b[sample_b])
        if np.isfinite(auc_a) and np.isfinite(auc_b):
            deltas.append(auc_a - auc_b)
    if not deltas:
        return {
            "delta": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
            "n_boot_valid": 0,
        }
    deltas = np.asarray(deltas)
    p_value = 2.0 * min(float((deltas <= 0).mean()), float((deltas >= 0).mean()))
    return {
        "delta": float(point),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_value": float(min(1.0, p_value)),
        "n_boot_valid": int(len(deltas)),
    }


def _design_matrix(frame):
    """Build a standardized design matrix and return it with feature names."""
    frame = frame.copy()
    frame["age"] = pd.to_numeric(frame.get("age"), errors="coerce")
    frame["age"] = frame["age"].fillna(frame["age"].median())
    frame["sex_male"] = (frame.get("sex", "unknown") == "male").astype(float)
    locations = frame.get("localization", pd.Series("unknown", index=frame.index))
    top = locations.value_counts().head(5).index.tolist()
    locations = locations.where(locations.isin(top), "other")
    dummies = pd.get_dummies(locations, prefix="site", drop_first=True, dtype=float)

    numeric = pd.DataFrame(
        {
            "L": frame["L"].astype(float),
            "delta_L": frame["delta_L"].astype(float),
            "log_area": np.log1p(frame["lesion_area"].clip(lower=0).astype(float)),
            "age": frame["age"].astype(float),
        },
        index=frame.index,
    )
    matrix = pd.concat([numeric, frame[["sex_male"]], dummies], axis=1)
    means = matrix.mean()
    stds = matrix.std(ddof=0).replace(0, 1.0)
    matrix = (matrix - means) / stds
    return matrix.to_numpy(dtype=float), matrix.columns.tolist(), frame


def continuous_error_model(
    predictions, tone, contrast, n_boot=500, seed=0, outcomes=None
):
    """Adjusted logistic models of error, false positive, and false negative.

    The model is fit separately for each architecture and outcome. Standard
    errors come from a cluster bootstrap over lesions, so repeated images and
    seeds of the same lesion are not treated as independent.
    """
    outcomes = outcomes or ("error", "false_positive", "false_negative")
    merged = predictions.merge(tone, on="image_id", how="left")
    merged = merged.merge(
        contrast[["image_id", "delta_L", "lesion_area"]], on="image_id", how="left"
    )
    merged = merged.dropna(subset=["L", "delta_L", "lesion_area"])
    if "lesion_id" not in merged.columns:
        merged["lesion_id"] = merged["image_id"]

    rows = []
    for outcome in outcomes:
        if outcome == "error":
            base = merged.copy()
            base["outcome"] = (base["y_pred"] != base["y_true"]).astype(int)
        elif outcome == "false_positive":
            base = merged[merged["y_true"] == 0].copy()
            base["outcome"] = (base["y_pred"] == 1).astype(int)
        elif outcome == "false_negative":
            base = merged[merged["y_true"] == 1].copy()
            base["outcome"] = (base["y_pred"] == 0).astype(int)
        else:
            raise ValueError(f"Unknown outcome: {outcome}")

        for backbone, subset in base.groupby("backbone", sort=False):
            matrix, names, subset = _design_matrix(subset)
            y = subset["outcome"].to_numpy()
            if len(np.unique(y)) < 2:
                continue
            model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            model.fit(matrix, y)
            coefficients = [model.coef_[0]]

            lesion_ids = subset["lesion_id"].to_numpy()
            unique = pd.unique(lesion_ids)
            index_by_lesion = {
                lesion: np.flatnonzero(lesion_ids == lesion) for lesion in unique
            }
            rng = np.random.default_rng(seed)
            for _ in range(n_boot):
                sampled = rng.choice(unique, size=len(unique), replace=True)
                indices = np.concatenate([index_by_lesion[lesion] for lesion in sampled])
                if len(np.unique(y[indices])) < 2:
                    continue
                boot = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
                try:
                    boot.fit(matrix[indices], y[indices])
                except ValueError:
                    continue
                coefficients.append(boot.coef_[0])
            coefficients = np.asarray(coefficients)
            if len(coefficients) < 2:
                continue
            for index, term in enumerate(names):
                values = coefficients[:, index]
                p_value = 2.0 * min(
                    float((values <= 0).mean()), float((values >= 0).mean())
                )
                rows.append(
                    {
                        "outcome": outcome,
                        "backbone": backbone,
                        "term": term,
                        "odds_ratio": float(np.exp(np.median(values))),
                        "ci_low": float(np.exp(np.percentile(values, 2.5))),
                        "ci_high": float(np.exp(np.percentile(values, 97.5))),
                        "p_value": float(min(1.0, p_value)),
                        "n": int(len(subset)),
                        "n_events": int(y.sum()),
                        "n_boot": int(len(coefficients)),
                    }
                )
    return pd.DataFrame(rows)


def binned_error_curve(predictions, tone, contrast, n_bins=10):
    """Mean error rate by L* quantile, averaged over runs per image."""
    merged = predictions.merge(tone[["image_id", "L"]], on="image_id", how="left")
    merged = merged.merge(
        contrast[["image_id", "delta_L"]], on="image_id", how="left"
    )
    merged = merged.dropna(subset=["L", "delta_L"])
    per_image = (
        merged.assign(error=(merged["y_pred"] != merged["y_true"]).astype(float))
        .groupby("image_id", as_index=False)
        .agg(error_rate=("error", "mean"), L=("L", "first"), delta_L=("delta_L", "first"))
    )
    per_image["L_bin"] = pd.qcut(per_image["L"], q=n_bins, duplicates="drop")
    summary = (
        per_image.groupby("L_bin", observed=True)
        .agg(
            n=("image_id", "size"),
            mean_L=("L", "mean"),
            mean_delta_L=("delta_L", "mean"),
            error_rate=("error_rate", "mean"),
            error_sem=("error_rate", lambda values: values.std(ddof=1) / np.sqrt(len(values))),
        )
        .reset_index()
    )
    return summary


def binned_error_by_contrast(predictions, tone, contrast, n_bins=6):
    """Mean error rate by L* bin within lesion-contrast tertiles."""
    merged = predictions.merge(tone[["image_id", "L"]], on="image_id", how="left")
    merged = merged.merge(
        contrast[["image_id", "delta_L"]], on="image_id", how="left"
    )
    merged = merged.dropna(subset=["L", "delta_L"])
    per_image = (
        merged.assign(error=(merged["y_pred"] != merged["y_true"]).astype(float))
        .groupby("image_id", as_index=False)
        .agg(error_rate=("error", "mean"), L=("L", "first"), delta_L=("delta_L", "first"))
    )
    per_image["contrast_tertile"] = pd.qcut(
        per_image["delta_L"], q=3, labels=["low", "medium", "high"]
    )
    rows = []
    for tertile, subset in per_image.groupby("contrast_tertile", observed=True):
        subset = subset.copy()
        subset["L_bin"] = pd.qcut(subset["L"], q=n_bins, duplicates="drop")
        summary = (
            subset.groupby("L_bin", observed=True)
            .agg(
                n=("image_id", "size"),
                mean_L=("L", "mean"),
                error_rate=("error_rate", "mean"),
            )
            .reset_index()
        )
        summary["contrast_tertile"] = str(tertile)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)
