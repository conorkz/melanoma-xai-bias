"""Calibration metrics for binary melanoma predictions."""

import numpy as np
import pandas as pd


def _validate(y_true, y_prob):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.size == 0:
        raise ValueError("y_true is empty")
    if y_true.size != y_prob.size:
        raise ValueError(
            f"Length mismatch: y_true {y_true.size} vs y_prob {y_prob.size}"
        )
    if not np.all(np.isin(np.unique(y_true), [0, 1])):
        raise ValueError("y_true must be binary (0/1)")
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob contains non-finite values")
    if (y_prob < 0).any() or (y_prob > 1).any():
        raise ValueError("y_prob must be in [0, 1]")
    return y_true, y_prob


def _bin_indices(y_prob, n_bins):
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins}")
    indices = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    return indices


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Expected calibration error with equal-width probability bins."""
    y_true, y_prob = _validate(y_true, y_prob)
    indices = _bin_indices(y_prob, n_bins)
    total = len(y_true)
    error = 0.0
    for bin_index in range(n_bins):
        mask = indices == bin_index
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(y_prob[mask].mean())
        accuracy = float(y_true[mask].mean())
        error += (count / total) * abs(accuracy - confidence)
    return float(error)


def brier_score(y_true, y_prob):
    """Mean squared error between predicted probability and outcome."""
    y_true, y_prob = _validate(y_true, y_prob)
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_curve(y_true, y_prob, n_bins=10):
    """Per-bin sample count, mean confidence, and observed positive rate."""
    y_true, y_prob = _validate(y_true, y_prob)
    indices = _bin_indices(y_prob, n_bins)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for bin_index in range(n_bins):
        mask = indices == bin_index
        rows.append(
            {
                "bin": bin_index,
                "bin_low": float(edges[bin_index]),
                "bin_high": float(edges[bin_index + 1]),
                "n": int(mask.sum()),
                "mean_prob": float(y_prob[mask].mean()) if mask.any() else np.nan,
                "observed_rate": float(y_true[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calibration_summary(predictions, tone=None):
    """Compute overall and per-group calibration for each run."""
    required = {"run_id", "y_true", "y_prob"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")

    frame = predictions
    if tone is not None:
        frame = frame.merge(tone[["image_id", "tone_group"]], on="image_id", how="left")
        frame["tone_group"] = frame["tone_group"].fillna("unknown")
    else:
        frame = frame.assign(tone_group="all")

    rows = []
    groups = ["all"] + sorted(
        group for group in frame["tone_group"].unique() if group != "all"
    )
    for run_id, run in frame.groupby("run_id", sort=False):
        for group in groups:
            subset = run if group == "all" else run[run["tone_group"] == group]
            if subset.empty:
                continue
            y_true = subset["y_true"].to_numpy()
            y_prob = subset["y_prob"].to_numpy()
            if len(np.unique(y_true)) < 2:
                ece = float("nan")
            else:
                ece = expected_calibration_error(y_true, y_prob)
            rows.append(
                {
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
                    "n": len(subset),
                    "ece": ece,
                    "brier": brier_score(y_true, y_prob),
                }
            )
    return pd.DataFrame(rows)


def _logit(y_prob, eps=1e-6):
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def prior_shift_correction(y_prob, train_prior, deployed_prior):
    """Move scores from the training prior to the deployed prior.

    A class-balanced sampler trains at an effective prior of 0.5. Under the
    label-shift assumption the correct posterior at the true prevalence is
    obtained by adding the log prior odds ratio to the logit.
    """
    for name, value in (("train_prior", train_prior), ("deployed_prior", deployed_prior)):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {value}")
    shift = np.log(deployed_prior / (1.0 - deployed_prior)) - np.log(
        train_prior / (1.0 - train_prior)
    )
    return _sigmoid(_logit(y_prob) + shift)


def fit_temperature(y_true, y_prob, bounds=(0.05, 20.0), n_grid=400):
    """Fit a single temperature on held-out scores by minimizing the NLL.

    A log-spaced grid search is used: it is deterministic, needs no optimizer
    dependency, and is accurate enough for a scalar parameter.
    """
    y_true = np.asarray(y_true, dtype=float)
    logits = _logit(y_prob)
    temperatures = np.exp(np.linspace(np.log(bounds[0]), np.log(bounds[1]), n_grid))
    best_t, best_nll = 1.0, np.inf
    for t in temperatures:
        p = np.clip(_sigmoid(logits / t), 1e-12, 1 - 1e-12)
        nll = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def apply_temperature(y_prob, temperature):
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return _sigmoid(_logit(y_prob) / temperature)
