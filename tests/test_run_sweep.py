"""Tests for scripts/run_sweep.py."""

import importlib.util
from pathlib import Path

import yaml

from conftest import make_synthetic_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(stem):
    path = PROJECT_ROOT / "scripts" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_base_config(tmp_path):
    base = {
        "run_name": "pilot",
        "seed": 42,
        "data": {
            "metadata_csv": "data/x.csv",
            "images_dir": "data/images",
            "split": {
                "val_fraction": 0.15,
                "test_fraction": 0.15,
                "seed": 42,
                "file": "outputs/splits/x.csv",
            },
            "task": "binary",
        },
        "model": {
            "backbone": "efficientnet_b0",
            "pretrained": False,
            "num_classes": 2,
            "drop_rate": 0.0,
        },
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "lr": 0.001,
            "weight_decay": 0.0,
            "img_size": 160,
            "loss": "focal",
            "focal_gamma": 2.0,
            "focal_alpha": 0.75,
            "sampler": "class_balanced",
            "augmentation": "standard",
            "num_workers": 0,
            "early_stopping_patience": 2,
            "max_batches": 1,
        },
        "evaluation": {
            "target_specificity": 0.95,
            "bootstrap": {"n_boot": 10, "confidence": 0.95, "seed": 0},
        },
    }
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return path


def write_sweep_config(tmp_path, base_path):
    sweep = {
        "name": "test_sweep",
        "base": str(base_path),
        "seeds": [1, 2],
        "variants": [
            {
                "name": "ce_random",
                "overrides": {"training.loss": "ce", "training.sampler": "random"},
            },
            {
                "name": "focal_cb",
                "overrides": {
                    "training.loss": "focal",
                    "training.sampler": "class_balanced",
                },
            },
        ],
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(sweep), encoding="utf-8")
    return path


def test_set_nested_creates_and_updates():
    sweep = load_script("run_sweep")
    config = {"training": {"loss": "ce"}}
    sweep.set_nested(config, "training.loss", "focal")
    sweep.set_nested(config, "training.focal_gamma", 2.0)
    sweep.set_nested(config, "model.backbone", "resnet50")
    assert config == {
        "training": {"loss": "focal", "focal_gamma": 2.0},
        "model": {"backbone": "resnet50"},
    }


def test_expand_sweep_variants_and_seeds():
    sweep = load_script("run_sweep")
    config = {
        "base": "base.yaml",
        "name": "s",
        "seeds": [1, 2],
        "variants": [
            {"name": "a", "overrides": {"training.loss": "ce"}},
            {"name": "b", "overrides": {"training.loss": "focal"}},
        ],
    }
    runs = sweep.expand_sweep(config)
    assert len(runs) == 4
    assert {run["name"] for run in runs} == {
        "s_a_seed1",
        "s_a_seed2",
        "s_b_seed1",
        "s_b_seed2",
    }
    assert [run["overrides"]["seed"] for run in runs] == [1, 2, 1, 2]
    assert {run["config_id"] for run in runs} == {"s_a", "s_b"}


def write_synthetic_base_config(tmp_path):
    dataset = make_synthetic_dataset(tmp_path / "data", n_lesions_per_class=4, img_size=32)
    base = {
        "run_name": "sweep_e2e",
        "seed": 42,
        "data": {
            "metadata_csv": str(dataset["metadata_csv"]),
            "images_dir": str(dataset["images_dir"]),
            "split": {
                "val_fraction": 0.2,
                "test_fraction": 0.2,
                "seed": 42,
                "file": str(tmp_path / "split.csv"),
            },
            "task": "binary",
        },
        "model": {
            "backbone": "resnet18",
            "pretrained": False,
            "num_classes": 2,
            "drop_rate": 0.0,
        },
        "training": {
            "epochs": 1,
            "batch_size": 4,
            "lr": 0.001,
            "weight_decay": 0.0,
            "img_size": 32,
            "loss": "focal",
            "focal_gamma": 2.0,
            "focal_alpha": 0.75,
            "sampler": "random",
            "augmentation": "standard",
            "num_workers": 0,
            "early_stopping_patience": 2,
            "max_batches": 1,
        },
        "evaluation": {
            "target_specificity": 0.95,
            "bootstrap": {"n_boot": 10, "confidence": 0.95, "seed": 0},
        },
    }
    path = tmp_path / "base_e2e.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return path


def test_dry_run_generates_configs(tmp_path):
    base = write_base_config(tmp_path)
    sweep_path = write_sweep_config(tmp_path, base)
    out_dir = tmp_path / "sweeps"

    sweep = load_script("run_sweep")
    rc = sweep.main(["--config", str(sweep_path), "--out-dir", str(out_dir), "--dry-run"])
    assert rc == 0

    config_paths = sorted((out_dir / "configs").glob("*.yaml"))
    assert len(config_paths) == 4
    generated = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in config_paths]
    assert {item["run_name"] for item in generated} == {
        "test_sweep_ce_random_seed1",
        "test_sweep_ce_random_seed2",
        "test_sweep_focal_cb_seed1",
        "test_sweep_focal_cb_seed2",
    }
    assert {item["seed"] for item in generated} == {1, 2}
    assert all(item["config_id"].startswith("test_sweep_") for item in generated)
    assert all(item["model"]["backbone"] == "efficientnet_b0" for item in generated)

    ce_runs = [item for item in generated if item["training"]["loss"] == "ce"]
    assert len(ce_runs) == 2
    assert all(item["training"]["sampler"] == "random" for item in ce_runs)


def test_sweep_executes_training_subprocess(tmp_path):
    base = write_synthetic_base_config(tmp_path)
    sweep_path = tmp_path / "sweep_e2e.yaml"
    sweep = {
        "name": "e2e",
        "base": str(base),
        "seeds": [1],
        "variants": [{"name": "ce", "overrides": {"training.loss": "ce"}}],
    }
    sweep_path.write_text(yaml.safe_dump(sweep), encoding="utf-8")

    runner = load_script("run_sweep")
    runs_dir = tmp_path / "runs"
    checkpoints_dir = tmp_path / "checkpoints"
    rc = runner.main(
        [
            "--config",
            str(sweep_path),
            "--out-dir",
            str(tmp_path / "sweep_out"),
            "--runs-dir",
            str(runs_dir),
            "--checkpoints-dir",
            str(checkpoints_dir),
        ]
    )
    assert rc == 0
    assert (runs_dir / "e2e_ce_seed1" / "metrics.json").exists()
    assert (runs_dir / "e2e_ce_seed1" / "predictions_test.csv").exists()
    assert (checkpoints_dir / "e2e_ce_seed1.pt").exists()
