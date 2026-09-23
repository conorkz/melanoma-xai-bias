"""End-to-end tests that run the actual scripts on synthetic data."""

import importlib.util
import json
from pathlib import Path

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
            "split": {"val_fraction": 0.2, "test_fraction": 0.2},
            "subset_size": 1000,
            "task": "binary",
        },
        "model": {"backbone": "resnet18", "pretrained": False, "num_classes": 2},
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
            "num_workers": 0,
            "early_stopping_patience": 2,
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
