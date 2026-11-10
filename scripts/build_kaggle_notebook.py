#!/usr/bin/env python
"""Generate the Kaggle benchmark notebook.

Keeping the notebook in a build script makes the setup steps reviewable and
regenerable instead of being trapped in a large JSON file.
"""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "notebooks" / "kaggle_benchmark.ipynb"
DEFAULT_STAGE1_OUTPUT = PROJECT_ROOT / "notebooks" / "kaggle_stage1.ipynb"

INTRO = """# RQ1 benchmark on Kaggle

This notebook runs the Stage 1 imbalance screen and the Stage 2 architecture comparison.

## Before you run

1. Open **Settings** in the right sidebar.
2. Set **Accelerator** to **GPU P100** or **GPU T4 x2**.
3. Set **Internet** to **On**.
4. Kaggle requires phone verification before GPU and internet access are enabled.
5. Optional: attach the dataset `kmader/skin-cancer-mnist-ham10000` through **Add Input**. If it is not attached, the notebook downloads HAM10000 from Harvard Dataverse.

## Limits

- 12 hours per CPU or GPU session.
- 30 GPU hours per week.
- 20 GB of auto-saved disk in `/kaggle/working`.

Stage 1 is 24 runs (about 3 to 4 GPU hours). Stage 2 is 18 runs (about 3 to 5 GPU hours). Run both in one session if possible, then download `teacher_update.zip`.
"""

ENVIRONMENT = """import os
import sys

import torch

print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Working directory:", os.getcwd())
"""

CLONE = """import os
import subprocess
import sys

REPO_URL = "https://github.com/conorkz/melanoma-xai-bias.git"
REPO_DIR = "/kaggle/working/melanoma-xai-bias"

if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

os.chdir(REPO_DIR)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "timm", "albumentations", "captum", "pyyaml"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"],
    check=True,
)
print("Repo ready:", os.getcwd())
"""

DATA_MARKDOWN = """## Data setup

The next cell links the attached Kaggle HAM10000 dataset if it is present. Otherwise it downloads the original archive from Harvard Dataverse and verifies the MD5 checksums. The download is about 2.8 GB.
"""

DATA = """import os
import shutil
import subprocess
import sys
from pathlib import Path

os.chdir(REPO_DIR)
images_dir = Path("data/HAM10000/images")
metadata_csv = Path("data/raw/HAM10000_metadata.csv")
images_dir.mkdir(parents=True, exist_ok=True)
metadata_csv.parent.mkdir(parents=True, exist_ok=True)

kaggle_root = Path("/kaggle/input/skin-cancer-mnist-ham10000")
linked = False
if (kaggle_root / "HAM10000_metadata.csv").exists():
    shutil.copy2(kaggle_root / "HAM10000_metadata.csv", metadata_csv)
    for part in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
        part_dir = kaggle_root / part
        if part_dir.exists():
            for source in part_dir.glob("*.jpg"):
                target = images_dir / source.name
                if not target.exists():
                    target.symlink_to(source)
    linked = True
    print("Linked HAM10000 from the attached Kaggle dataset")

n_images = len(list(images_dir.glob("*.jpg")))
n_rows = sum(1 for _ in metadata_csv.open()) - 1 if metadata_csv.exists() else 0
if n_images != 10015 or n_rows != 10015:
    if linked:
        for path in images_dir.glob("*.jpg"):
            path.unlink()
    print("Attached dataset incomplete or absent; downloading from Harvard Dataverse")
    subprocess.run([sys.executable, "scripts/download_data.py"], check=True)
    n_images = len(list(images_dir.glob("*.jpg")))
    n_rows = sum(1 for _ in metadata_csv.open()) - 1

print("Metadata rows:", n_rows)
print("Images:", n_images)
assert n_images == 10015 and n_rows == 10015, "HAM10000 is incomplete"
print("Data ready.")
"""

STAGE1_MARKDOWN = """## Stage 1: loss, sampling, and augmentation screen

24 runs: 8 configurations x 3 seeds, EfficientNet-B0 at 160 px, 15 epochs.

If the session resets, rerun this cell. `--skip-existing` skips completed runs.
"""

STAGE1 = """import os
import subprocess
import sys

os.chdir(REPO_DIR)
subprocess.run(
    [
        sys.executable,
        "scripts/run_sweep.py",
        "--config",
        "configs/sweep_stage1.yaml",
        "--skip-existing",
    ],
    check=True,
)
"""

COLLECT1 = """import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

os.chdir(REPO_DIR)
subprocess.run(
    [
        sys.executable,
        "scripts/collect_results.py",
        "--runs-dir",
        "outputs/runs",
        "--run-prefix",
        "stage1",
        "--out-dir",
        "results",
        "--prefix",
        "rq1_stage1",
    ],
    check=True,
)

display(Markdown(Path("results/rq1_stage1.md").read_text()))
table = pd.read_csv("results/rq1_stage1.csv")
selection_columns = [
    "config_id",
    "n_seeds",
    "val_sensitivity_mean",
    "val_sensitivity_std",
    "val_balanced_accuracy_mean",
    "sensitivity_mean",
    "sensitivity_std",
    "sensitivity_ci_low",
    "sensitivity_ci_high",
]
display(table.sort_values("val_sensitivity_mean", ascending=False)[selection_columns])
"""

WINNER_MARKDOWN = """## Choose the Stage 1 winner

Look at `val_sensitivity_mean` first. This is validation sensitivity at 95% specificity. Use `val_balanced_accuracy_mean` as the tie-breaker and check `val_sensitivity_std`.

Do **not** choose by the test columns.

Set `WINNER` in the next cell to match the winning `config_id`:

| config_id | loss | gamma | sampler | augmentation |
|---|---|---|---|---|
| stage1_ce_random | ce | - | random | standard |
| stage1_ce_cb | ce | - | class_balanced | standard |
| stage1_wce_random | weighted_ce | - | random | standard |
| stage1_focal2_random | focal | 2.0 | random | standard |
| stage1_focal2_cb | focal | 2.0 | class_balanced | standard |
| stage1_focal3_cb | focal | 3.0 | class_balanced | standard |
| stage1_ce_random_strong | ce | - | random | strong |
| stage1_focal2_cb_strong | focal | 2.0 | class_balanced | strong |
"""

WINNER = """import os
import subprocess
from pathlib import Path

import yaml

os.chdir(REPO_DIR)

# Edit this to match the Stage 1 winner.
# Example for stage1_focal3_cb_strong:
# WINNER = {"loss": "focal", "focal_gamma": 3.0, "sampler": "class_balanced", "augmentation": "strong"}
WINNER = {
    "loss": "focal",
    "focal_gamma": 2.0,
    "sampler": "class_balanced",
    "augmentation": "standard",
}

path = Path("configs/sweep_stage2.yaml")
sweep = yaml.safe_load(path.read_text())
for variant in sweep["variants"]:
    if variant["name"].endswith("_best_imbalance"):
        overrides = variant.setdefault("overrides", {})
        overrides["training.loss"] = WINNER["loss"]
        overrides["training.sampler"] = WINNER["sampler"]
        overrides["training.augmentation"] = WINNER["augmentation"]
        if WINNER["loss"] == "focal":
            overrides["training.focal_gamma"] = WINNER["focal_gamma"]
        else:
            overrides.pop("training.focal_gamma", None)
path.write_text(yaml.safe_dump(sweep, sort_keys=False))
print(path.read_text())

subprocess.run(["git", "-C", REPO_DIR, "add", "configs/sweep_stage2.yaml"], check=True)
staged = subprocess.run(["git", "-C", REPO_DIR, "diff", "--cached", "--quiet"])
if staged.returncode != 0:
    subprocess.run(
        ["git", "-C", REPO_DIR, "config", "user.email", "kaggle@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", REPO_DIR, "config", "user.name", "Kaggle Run"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", REPO_DIR, "commit", "-m", "Select Stage 1 winner for Stage 2"],
        check=True,
    )
    print("Committed the Stage 2 configuration")
else:
    print("Stage 2 configuration unchanged; nothing to commit")
"""

STAGE2_MARKDOWN = """## Stage 2: architecture comparison

18 runs: 3 architectures x 2 configurations x 3 seeds, all at 224 px.

Verify the printed Stage 2 config above before running this cell.
"""

STAGE2 = """import os
import subprocess
import sys

os.chdir(REPO_DIR)
subprocess.run(
    [
        sys.executable,
        "scripts/run_sweep.py",
        "--config",
        "configs/sweep_stage2.yaml",
        "--skip-existing",
    ],
    check=True,
)
"""

COLLECT2 = """import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

os.chdir(REPO_DIR)
subprocess.run(
    [
        sys.executable,
        "scripts/collect_results.py",
        "--runs-dir",
        "outputs/runs",
        "--run-prefix",
        "stage2",
        "--out-dir",
        "results",
        "--prefix",
        "rq1_stage2",
    ],
    check=True,
)

display(Markdown(Path("results/rq1_stage2.md").read_text()))
table = pd.read_csv("results/rq1_stage2.csv")
columns = [
    "config_id",
    "n_seeds",
    "sensitivity_mean",
    "sensitivity_std",
    "sensitivity_ci_low",
    "sensitivity_ci_high",
    "specificity_mean",
    "balanced_accuracy_mean",
    "auc_mean",
]
display(table.sort_values("sensitivity_mean", ascending=False)[columns])
"""

FIGURES_MARKDOWN = """## Best run and figures

The next cell picks the Stage 2 configuration with the highest mean test sensitivity at 95% specificity. This is descriptive, not a new model selection step. It then picks the run whose seed is the median of the three seeds and displays that run's training curves and ROC curve. Do not pick the luckiest seed.
"""

FIGURES = """import json
import os
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

os.chdir(REPO_DIR)
table = pd.read_csv("results/rq1_stage2.csv")
best = table.sort_values("sensitivity_mean", ascending=False).iloc[0]
print("Best configuration:", best["config_id"])
print("Test sensitivity:", round(float(best["sensitivity_mean"]), 3))

runs = []
for metrics_path in Path("outputs/runs").glob("*/metrics.json"):
    data = json.loads(metrics_path.read_text())
    if data.get("config_id") == best["config_id"]:
        runs.append((int(data["seed"]), metrics_path.parent))
runs.sort()
if not runs:
    raise RuntimeError(f"No runs found for {best['config_id']}")
print("Seeds:", [seed for seed, _ in runs])
median_seed, median_run = runs[len(runs) // 2]
print("Median-seed run:", median_run)

for name in ("training_curves.png", "roc_test.png"):
    path = median_run / name
    if path.exists():
        print(path)
        display(Image(filename=str(path)))
"""

PACKAGE = """import os
import subprocess
from pathlib import Path

os.chdir(REPO_DIR)
subprocess.run(
    ["zip", "-r", "-q", "teacher_update.zip", "results", "outputs/runs"],
    check=True,
)
print("Download this file from the Kaggle Output pane:")
print(Path("teacher_update.zip").resolve())
print("")
print("Best checkpoint to keep for Phase 3:")
print(Path(f"checkpoints/{median_run.name}.pt").resolve())
"""

NEXT_STEPS = """## Next steps

1. Download `teacher_update.zip` from the Output pane.
2. On your Mac, unzip it into the repo.
3. Run `cp RESULTS_MEMO_TEMPLATE.md RESULTS_MEMO.md`.
4. Paste `results/rq1_stage1.md` and `results/rq1_stage2.md` into the memo.
5. Use the validation columns to explain the Stage 1 selection and the test columns for the headline Stage 2 result.
6. Prepare the slides using the outline in `RESULTS_MEMO_TEMPLATE.md`.

Do not show the 500-image debug run. The full 15-epoch results are the teacher update.
"""

STAGE1_INTRO = """# RQ1 Stage 1 on Kaggle

This notebook runs only Stage 1: the 24-run loss, sampling, and augmentation screen.

## Before you run

1. Open **Settings**.
2. Set **Accelerator** to **GPU P100** or **GPU T4 x2**.
3. Set **Internet** to **On**.
4. Attach the dataset `kmader/skin-cancer-mnist-ham10000` if it is not already attached.

Stage 1 takes about 3 to 4 GPU hours. The notebook writes `stage1_results.zip` at the end.
"""

PACKAGE_STAGE1 = """import os
import subprocess
from pathlib import Path

os.chdir(REPO_DIR)
subprocess.run(
    ["zip", "-r", "-q", "stage1_results.zip", "results", "outputs/runs"],
    check=True,
)
print("Download this file from the Kaggle Output pane:")
print(Path("stage1_results.zip").resolve())
"""

NEXT_STEPS_STAGE1 = """## Next steps

1. Download `stage1_results.zip`.
2. Open `results/rq1_stage1.csv` and choose the winner by `val_sensitivity_mean`. Use `val_balanced_accuracy_mean` as the tie-breaker and check `val_sensitivity_std`.
3. Run the Stage 2 notebook with the winning settings.

Do not choose the winner by the test columns.
"""

STAGE1_CELLS = [
    new_markdown_cell(STAGE1_INTRO),
    new_code_cell(ENVIRONMENT),
    new_code_cell(CLONE),
    new_markdown_cell(DATA_MARKDOWN),
    new_code_cell(DATA),
    new_markdown_cell(STAGE1_MARKDOWN),
    new_code_cell(STAGE1),
    new_code_cell(COLLECT1),
    new_code_cell(PACKAGE_STAGE1),
    new_markdown_cell(NEXT_STEPS_STAGE1),
]

CELLS = [
    new_markdown_cell(INTRO),
    new_code_cell(ENVIRONMENT),
    new_code_cell(CLONE),
    new_markdown_cell(DATA_MARKDOWN),
    new_code_cell(DATA),
    new_markdown_cell(STAGE1_MARKDOWN),
    new_code_cell(STAGE1),
    new_code_cell(COLLECT1),
    new_markdown_cell(WINNER_MARKDOWN),
    new_code_cell(WINNER),
    new_markdown_cell(STAGE2_MARKDOWN),
    new_code_cell(STAGE2),
    new_code_cell(COLLECT2),
    new_markdown_cell(FIGURES_MARKDOWN),
    new_code_cell(FIGURES),
    new_code_cell(PACKAGE),
    new_markdown_cell(NEXT_STEPS),
]


NOTEBOOK_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12"},
}


def _write(cells, path):
    notebook = new_notebook(cells=cells)
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"cell-{index:02d}"
    notebook["metadata"] = NOTEBOOK_METADATA
    nbformat.validate(notebook)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)
    return path


def build(path=DEFAULT_OUTPUT):
    """Write the full Stage 1 plus Stage 2 Kaggle notebook and return its path."""
    return _write(CELLS, path)


def build_stage1(path=DEFAULT_STAGE1_OUTPUT):
    """Write the Stage 1-only Kaggle notebook used for the CLI batch run."""
    return _write(STAGE1_CELLS, path)


if __name__ == "__main__":
    print(build())
    print(build_stage1())
