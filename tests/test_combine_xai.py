"""Tests for combining per-architecture XAI results."""

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


def write_metrics(path, backbone):
    rows = []
    for tone, mass in (("light", 0.6), ("dark", 0.3)):
        for seed in (1, 2):
            for image in range(4):
                rows.append(
                    {
                        "backbone": backbone,
                        "seed": seed,
                        "run_id": f"{backbone}_{seed}",
                        "image_id": f"{backbone}_{tone}_{seed}_{image}",
                        "tone_group": tone,
                        "y_true": image % 2,
                        "y_prob": 0.4 + image * 0.1,
                        "y_pred": image % 2,
                        "correct": 1,
                        "attribution_mass": mass,
                        "pointing_game": 1.0 if tone == "light" else 0.0,
                        "attribution_iou": 0.4 if tone == "light" else 0.1,
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_combine_writes_summary_and_correlations(tmp_path):
    combiner = load_script("combine_xai_results")
    root = tmp_path / "xai"
    write_metrics(root / "resnet50" / "xai_metrics.csv", "resnet50")
    write_metrics(root / "vit_s" / "xai_metrics.csv", "vit_s")

    status = combiner.main(
        ["--root", str(root), "--architectures", "resnet50,vit_s,missing"]
    )
    assert status == 0
    assert (root / "xai_all_metrics.csv").exists()
    summary = pd.read_csv(root / "xai_all_summary.csv")
    assert set(summary["backbone"]) == {"resnet50", "vit_s"}
    assert len(summary) == 4
    correlations = pd.read_csv(root / "xai_all_correlations.csv")
    assert len(correlations) == 2
    assert (root / "xai.md").exists()
