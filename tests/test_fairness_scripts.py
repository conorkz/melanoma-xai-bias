"""End-to-end tests for the fairness scripts."""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(stem):
    path = PROJECT_ROOT / "scripts" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_image(path, rgb, size=64):
    image = np.full((size, size, 3), rgb, dtype=np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def write_mask(path, size=64):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[24:40, 24:40] = 255
    cv2.imwrite(str(path), mask)


def test_build_tone_table_end_to_end(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    write_image(images_dir / "img0.jpg", (230, 190, 170))
    write_image(images_dir / "img1.jpg", (90, 60, 45))
    write_mask(masks_dir / "img0_segmentation.png")
    write_mask(masks_dir / "img1_segmentation.png")

    metadata = pd.DataFrame({"image_id": ["img0", "img1"]})
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    output = tmp_path / "tone.csv"

    script = load_script("build_tone_table")
    rc = script.main(
        [
            "--metadata",
            str(metadata_path),
            "--images-dir",
            str(images_dir),
            "--masks-dir",
            str(masks_dir),
            "--out",
            str(output),
        ]
    )
    assert rc == 0
    tone = pd.read_csv(output)
    assert len(tone) == 2
    assert {"image_id", "L", "b", "ita", "n_pixels", "ita_group", "tone_group"} <= set(
        tone.columns
    )
    light = tone.loc[tone["image_id"] == "img0", "ita"].iloc[0]
    dark = tone.loc[tone["image_id"] == "img1", "ita"].iloc[0]
    assert light > dark
    assert tone.loc[tone["image_id"] == "img0", "tone_group"].iloc[0] == "light"
    assert tone.loc[tone["image_id"] == "img1", "tone_group"].iloc[0] == "dark"


def write_predictions(tmp_path):
    run_dir = tmp_path / "predictions" / "stage2_demo_seed1"
    run_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "image_id": [f"light{i}" for i in range(4)] + [f"dark{i}" for i in range(4)],
            "split": ["test"] * 8,
            "y_true": [1, 1, 0, 0, 1, 1, 0, 0],
            "y_prob": [0.9, 0.8, 0.2, 0.1, 0.9, 0.1, 0.2, 0.1],
            "y_pred": [1, 1, 0, 0, 1, 0, 0, 0],
            "threshold": [0.5] * 8,
            "run_id": ["stage2_demo_seed1"] * 8,
            "config_id": ["stage2_demo"] * 8,
            "seed": [1] * 8,
        }
    )
    frame.to_csv(run_dir / "predictions_test.csv", index=False)

    tone = pd.DataFrame(
        {
            "image_id": [f"light{i}" for i in range(4)] + [f"dark{i}" for i in range(4)],
            "tone_group": ["light"] * 4 + ["dark"] * 4,
        }
    )
    tone_path = tmp_path / "tone.csv"
    tone.to_csv(tone_path, index=False)
    return tmp_path / "predictions", tone_path


def test_analyze_fairness_end_to_end(tmp_path):
    predictions_root, tone_path = write_predictions(tmp_path)
    out_dir = tmp_path / "fairness"

    script = load_script("analyze_fairness")
    rc = script.main(
        [
            "--predictions-root",
            str(predictions_root),
            "--tone-table",
            str(tone_path),
            "--out-dir",
            str(out_dir),
            "--run-prefix",
            "stage2",
            "--n-boot",
            "20",
        ]
    )
    assert rc == 0
    assert (out_dir / "fairness_runs.csv").exists()
    assert (out_dir / "fairness_summary.csv").exists()
    assert (out_dir / "fairness_disparity.csv").exists()
    assert (out_dir / "fairness.md").exists()

    summary = pd.read_csv(out_dir / "fairness_summary.csv")
    assert len(summary) == 2
    light = summary[summary["tone_group"] == "light"].iloc[0]
    dark = summary[summary["tone_group"] == "dark"].iloc[0]
    assert light["sensitivity_mean"] == 1.0
    assert dark["sensitivity_mean"] == 0.5

    disparity = pd.read_csv(out_dir / "fairness_disparity.csv")
    assert disparity["max_min_gap"].iloc[0] == 0.5
    assert disparity["light_dark_ratio"].iloc[0] == 2.0
