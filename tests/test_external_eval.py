"""Tests for the external dataset evaluation script."""

import importlib.util
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(stem):
    path = PROJECT_ROOT / "scripts" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_metadata(path):
    pd.DataFrame(
        {
            "isic_id": ["a", "b", "c"],
            "image_type": ["dermoscopic", "dermoscopic", "clinical: overview"],
            "diagnosis": ["melanoma", "nevus", "melanoma"],
            "fitzpatrick_skin_type": ["II", None, "III"],
        }
    ).to_csv(path, index=False)


def test_load_metadata_filters_and_maps_labels(tmp_path):
    evaluator = load_script("evaluate_external")
    metadata_path = tmp_path / "metadata.csv"
    write_metadata(metadata_path)

    class Args:
        metadata = str(metadata_path)
        image_type = "dermoscopic"
        id_column = "isic_id"
        label_column = "diagnosis"
        positive_label = "melanoma"
        tone_column = "fitzpatrick_skin_type"
        max_images = None

    frame = evaluator.load_metadata(Args)
    assert len(frame) == 2
    assert frame["image_id"].tolist() == ["a", "b"]
    assert frame["y_true"].tolist() == [1, 0]
    assert frame["tone_group"].tolist() == ["II", "unknown"]


def test_summarize_computes_seed_mean_and_std():
    evaluator = load_script("evaluate_external")
    frame = pd.DataFrame(
        {
            "backbone": ["resnet50"] * 3,
            "tone_group": ["dark"] * 3,
            "sensitivity": [0.5, 0.6, 0.7],
            "specificity": [0.9, 0.9, 0.9],
            "balanced_accuracy": [0.7, 0.75, 0.8],
            "auc": [0.8, 0.82, 0.84],
            "fpr": [0.1, 0.1, 0.1],
            "n": [43, 43, 43],
            "n_positive": [6, 6, 6],
        }
    )
    summary = evaluator.summarize(frame)
    row = summary.iloc[0]
    assert row["sensitivity_mean"] == 0.6
    assert row["sensitivity_std"] > 0
    assert row["n_mean"] == 43
