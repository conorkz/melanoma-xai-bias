"""End-to-end tests that run the actual scripts on synthetic data."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import make_synthetic_dataset
from melanoma.models.factory import build_model
from melanoma.training.trainer import save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(stem):
    path = PROJECT_ROOT / "scripts" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_config(tmp_path, dataset, run_name="e2e_test", metadata_override=None):
    config = {
        "run_name": run_name,
        "seed": 42,
        "data": {
            "root": str(dataset["root"]),
            "metadata_csv": str(metadata_override or dataset["metadata_csv"]),
            "images_dir": str(dataset["images_dir"]),
            "split": {
                "val_fraction": 0.2,
                "test_fraction": 0.2,
                "seed": 42,
                "file": str(tmp_path / "split.csv"),
            },
            "subset_size": 1000,
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
            "sampler": "class_balanced",
            "augmentation": "standard",
            "num_workers": 0,
            "early_stopping_patience": 2,
            "grad_clip": 1.0,
            "max_batches": 1,
        },
        "evaluation": {
            "target_specificity": 0.95,
            "bootstrap": {"n_boot": 50, "confidence": 0.95, "seed": 0},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_smoke_test_end_to_end(tmp_path):
    dataset = make_synthetic_dataset(tmp_path / "data", n_lesions_per_class=4, img_size=32)
    config = write_config(tmp_path, dataset)
    checkpoint = tmp_path / "ckpt.pt"
    output_dir = tmp_path / "out"

    smoke = load_script("smoke_test")
    rc = smoke.main(
        [
            "--config",
            str(config),
            "--device",
            "cpu",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 0
    assert checkpoint.exists()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run_name"] == "e2e_test"
    assert len(metrics["history"]["train_loss"]) == 1
    assert metrics["history"]["best_epoch"] is not None
    assert "final_val_metrics" in metrics
    assert (output_dir / "training_curves.png").exists()


def test_smoke_test_missing_metadata_returns_error(tmp_path):
    dataset = make_synthetic_dataset(tmp_path / "data", n_lesions_per_class=2, img_size=32)
    config = write_config(tmp_path, dataset, metadata_override=tmp_path / "missing.csv")

    smoke = load_script("smoke_test")
    rc = smoke.main(
        [
            "--config",
            str(config),
            "--device",
            "cpu",
            "--checkpoint",
            str(tmp_path / "ckpt.pt"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 1


def test_demo_gradcam_end_to_end(tmp_path):
    dataset = make_synthetic_dataset(tmp_path / "data", n_lesions_per_class=2, img_size=32)
    config = write_config(tmp_path, dataset, run_name="demo_test")
    checkpoint = tmp_path / "ckpt.pt"
    save_checkpoint(checkpoint, build_model("resnet18", num_classes=2, pretrained=False))
    out_dir = tmp_path / "gradcam"

    demo = load_script("demo_gradcam")
    rc = demo.main(
        [
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(out_dir),
            "--device",
            "cpu",
            "--samples-per-class",
            "2",
            "--classes",
            "mel",
            "nv",
        ]
    )
    assert rc == 0
    assert (out_dir / "gradcam_grid.png").exists()
    assert len(list(out_dir.glob("*_gradcam.png"))) == 4


def test_train_end_to_end(tmp_path):
    dataset = make_synthetic_dataset(tmp_path / "data", n_lesions_per_class=4, img_size=32)
    config = write_config(tmp_path, dataset, run_name="train_e2e")
    checkpoint = tmp_path / "train.pt"
    output_dir = tmp_path / "train_out"

    train = load_script("train")
    rc = train.main(
        [
            "--config",
            str(config),
            "--device",
            "cpu",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
            "--max-batches",
            "1",
        ]
    )
    assert rc == 0
    assert checkpoint.exists()

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    json.dumps(metrics, allow_nan=False)
    assert metrics["run_id"] == "train_e2e"
    assert metrics["test"] is not None
    assert "sensitivity" in metrics["test"]["at_target"]
    assert metrics["threshold"] == metrics["test"]["at_target"]["threshold"]
    assert len(metrics["history"]["train_loss"]) == 1
    assert metrics["split_summary"]["train"]["total"] > 0

    assert (output_dir / "training_curves.png").exists()
    assert (output_dir / "predictions_val.csv").exists()
    assert (output_dir / "predictions_test.csv").exists()
    assert (output_dir / "run.json").exists()

    predictions = pd.read_csv(output_dir / "predictions_test.csv")
    expected_columns = {
        "image_id",
        "split",
        "y_true",
        "y_prob",
        "y_pred",
        "threshold",
        "run_id",
        "config_id",
        "seed",
    }
    assert expected_columns <= set(predictions.columns)
    assert (predictions["split"] == "test").all()
    expected_predictions = (predictions["y_prob"].to_numpy() >= metrics["threshold"]).astype(int)
    assert np.array_equal(predictions["y_pred"].to_numpy(), expected_predictions)

    manifest = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 42
    assert manifest["config"]["model"]["backbone"] == "resnet18"

    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    split_frame = pd.read_csv(config_data["data"]["split"]["file"])
    assert set(split_frame["split"]) == {"train", "val", "test"}
    assert split_frame["image_id"].nunique() == len(dataset["frame"])


def test_train_missing_metadata_returns_error(tmp_path):
    dataset = make_synthetic_dataset(tmp_path / "data", n_lesions_per_class=2, img_size=32)
    config = write_config(tmp_path, dataset, metadata_override=tmp_path / "missing.csv")

    train = load_script("train")
    rc = train.main(
        [
            "--config",
            str(config),
            "--device",
            "cpu",
            "--checkpoint",
            str(tmp_path / "train.pt"),
            "--output-dir",
            str(tmp_path / "train_out"),
        ]
    )
    assert rc == 1


def test_json_safe_converts_non_finite_and_numpy_values():
    train = load_script("train")
    value = {
        "nan": float("nan"),
        "inf": float("inf"),
        "float32": np.float32(0.5),
        "int64": np.int64(3),
        "bool": np.bool_(True),
        "nested": [float("nan"), {"ok": 1.0}],
        "none": None,
    }
    safe = train.json_safe(value)
    assert safe["nan"] is None
    assert safe["inf"] is None
    assert safe["float32"] == 0.5
    assert safe["int64"] == 3
    assert safe["bool"] is True
    assert safe["nested"] == [None, {"ok": 1.0}]
    assert safe["none"] is None
    json.dumps(safe, allow_nan=False)


def test_train_no_test_skips_test_artifacts(tmp_path):
    dataset = make_synthetic_dataset(tmp_path / "data", n_lesions_per_class=4, img_size=32)
    config = write_config(tmp_path, dataset, run_name="train_no_test")
    output_dir = tmp_path / "train_out"

    train = load_script("train")
    rc = train.main(
        [
            "--config",
            str(config),
            "--device",
            "cpu",
            "--checkpoint",
            str(tmp_path / "train.pt"),
            "--output-dir",
            str(output_dir),
            "--max-batches",
            "1",
            "--no-test",
        ]
    )
    assert rc == 0
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test"] is None
    assert not (output_dir / "predictions_test.csv").exists()


def test_resolve_splits_is_cached_and_stable(tmp_path, synthetic_dataset):
    train = load_script("train")
    split_file = tmp_path / "split.csv"
    data_config = {
        "split": {
            "val_fraction": 0.2,
            "test_fraction": 0.2,
            "seed": 7,
            "file": str(split_file),
        }
    }
    first = train.resolve_splits(synthetic_dataset["frame"], data_config)
    assert split_file.exists()
    second = train.resolve_splits(synthetic_dataset["frame"], data_config)
    assert first["split"].tolist() == second["split"].tolist()


def test_resolve_splits_rejects_mismatched_file(tmp_path, synthetic_dataset):
    train = load_script("train")
    split_file = tmp_path / "split.csv"
    pd.DataFrame({"image_id": ["not_an_image"], "split": ["train"]}).to_csv(split_file, index=False)
    data_config = {
        "split": {
            "val_fraction": 0.2,
            "test_fraction": 0.2,
            "seed": 7,
            "file": str(split_file),
        }
    }
    with pytest.raises(ValueError, match="Split file"):
        train.resolve_splits(synthetic_dataset["frame"], data_config)


def test_demo_gradcam_missing_checkpoint_returns_error(tmp_path):
    dataset = make_synthetic_dataset(tmp_path / "data", n_lesions_per_class=2, img_size=32)
    config = write_config(tmp_path, dataset)

    demo = load_script("demo_gradcam")
    rc = demo.main(
        [
            "--config",
            str(config),
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--out-dir",
            str(tmp_path / "gradcam"),
            "--device",
            "cpu",
        ]
    )
    assert rc == 1
