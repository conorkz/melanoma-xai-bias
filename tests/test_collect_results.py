"""Tests for scripts/collect_results.py."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(stem):
    path = PROJECT_ROOT / "scripts" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run(
    root,
    run_id,
    config_id,
    seed,
    sensitivity,
    specificity=0.9,
    balanced_accuracy=0.85,
    auc=0.8,
):
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    payload = {
        "run_id": run_id,
        "config_id": config_id,
        "run_name": run_id,
        "seed": seed,
        "backbone": "efficientnet_b0",
        "loss": "focal",
        "sampler": "class_balanced",
        "augmentation": "standard",
        "img_size": 160,
        "best_epoch": 5,
        "test": {
            "at_target": {
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": balanced_accuracy,
                "auc": auc,
                "n": 100,
                "n_positive": 10,
            },
            "bootstrap": {
                "sensitivity": {
                    "low": sensitivity - 0.05,
                    "high": sensitivity + 0.05,
                }
            },
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_collect_results_aggregates_seeds(tmp_path):
    runs_dir = tmp_path / "runs"
    write_run(runs_dir, "a1", "config_a", 1, 0.8)
    write_run(runs_dir, "a2", "config_a", 2, 0.9)
    write_run(runs_dir, "b1", "config_b", 1, 0.7)

    collect = load_script("collect_results")
    out_dir = tmp_path / "reports"
    rc = collect.main(["--runs-dir", str(runs_dir), "--out-dir", str(out_dir)])
    assert rc == 0

    aggregate = pd.read_csv(out_dir / "rq1_benchmark.csv")
    assert len(aggregate) == 2
    row_a = aggregate[aggregate["config_id"] == "config_a"].iloc[0]
    assert row_a["n_seeds"] == 2
    assert row_a["n_runs"] == 2
    assert row_a["sensitivity_mean"] == pytest.approx(0.85)
    assert row_a["sensitivity_std"] == pytest.approx(0.0707106781)

    row_b = aggregate[aggregate["config_id"] == "config_b"].iloc[0]
    assert row_b["n_seeds"] == 1
    assert pd.isna(row_b["sensitivity_std"])

    assert (out_dir / "rq1_benchmark_runs.csv").exists()
    assert (out_dir / "rq1_benchmark.md").exists()
    text = (out_dir / "rq1_benchmark.md").read_text(encoding="utf-8")
    assert "config_a" in text
    assert "config_b" in text


def test_collect_results_skips_runs_without_test(tmp_path):
    runs_dir = tmp_path / "runs"
    write_run(runs_dir, "a1", "config_a", 1, 0.8)
    run_dir = runs_dir / "no_test"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps({"run_id": "no_test", "test": None}), encoding="utf-8"
    )

    collect = load_script("collect_results")
    rc = collect.main(["--runs-dir", str(runs_dir), "--out-dir", str(tmp_path / "reports")])
    assert rc == 0
    aggregate = pd.read_csv(tmp_path / "reports" / "rq1_benchmark.csv")
    assert len(aggregate) == 1


def test_collect_results_filters_by_run_prefix(tmp_path):
    runs_dir = tmp_path / "runs"
    write_run(runs_dir, "stage1_a1", "stage1_a", 1, 0.8)
    write_run(runs_dir, "stage2_b1", "stage2_b", 1, 0.7)

    collect = load_script("collect_results")
    out_dir = tmp_path / "reports"
    rc = collect.main(
        [
            "--runs-dir",
            str(runs_dir),
            "--out-dir",
            str(out_dir),
            "--run-prefix",
            "stage2",
        ]
    )
    assert rc == 0
    aggregate = pd.read_csv(out_dir / "rq1_benchmark.csv")
    assert len(aggregate) == 1
    assert set(aggregate["config_id"]) == {"stage2_b"}


def test_collect_results_empty_returns_error(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    collect = load_script("collect_results")
    rc = collect.main(["--runs-dir", str(runs_dir), "--out-dir", str(tmp_path / "reports")])
    assert rc == 1
