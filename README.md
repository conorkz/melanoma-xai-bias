# melanoma-xai-bias

Research prototype for the Master's thesis **"Interpretable Computer Vision for Fair Melanoma Detection"**.

This repository is the Phase 0 and Phase 1 deliverable: a reproducible project skeleton, an exploratory data analysis of HAM10000, a working training pipeline with class imbalance handling (focal loss, class-balanced sampling), gradient-based explanations (Grad-CAM via Captum), and lesion-aware data splitting. It runs end to end on a MacBook Air M1; heavier training is designed for free cloud GPUs (Kaggle, Colab, Lightning AI).

The full research plan lives in `RESEARCH_PLAN.md` in the parent research folder.

## Project structure

```
melanoma-xai-bias/
  configs/smoke_test.yaml        single reproducible training config
  notebooks/01_eda_ham10000.ipynb  executed EDA with figures
  scripts/
    download_data.py             fetch HAM10000 from Harvard Dataverse
    smoke_test.py                tiny training run (subset, few epochs)
    demo_gradcam.py              Grad-CAM heatmaps for sample lesions
    build_eda_notebook.py        regenerate the EDA notebook
    run_tests.sh                 full test suite
  src/melanoma/
    config.py utils.py
    data/       download, metadata, splits, dataset, transforms
    models/     backbone factory (timm)
    training/   losses (focal), sampler, metrics, trainer
    xai/        Grad-CAM (Captum), heatmap overlays
    viz/        plotting helpers
  tests/                        pytest suite covering edge cases
  data/                         (gitignored) raw downloads and images
  checkpoints/                  (gitignored) trained weights
  outputs/                      (gitignored) metrics, figures, heatmaps
```

## Setup

Requires Python 3.10 to 3.12 (3.12 recommended) and [uv](https://github.com/astral-sh/uv).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

Alternative without uv:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

## Data

Downloads about 2.8 GB from Harvard Dataverse (metadata plus two image archives) and verifies MD5 checksums:

```bash
.venv/bin/python scripts/download_data.py
```

The script is resumable, skips already-complete files, normalizes the metadata to `data/raw/HAM10000_metadata.csv`, and reports any missing images. Set `--metadata-only` to fetch just the CSV.

## Tests

```bash
scripts/run_tests.sh          # or: .venv/bin/python -m pytest tests -v
```

The suite covers config loading, metadata validation, grouped splits, transforms, dataset edge cases (missing, corrupt, grayscale images), model factory, focal loss numerics, class-balanced sampling, metrics with degenerate splits, the trainer (checkpoints, early stopping), Grad-CAM (CNN and ViT), the download logic with mocked network sessions (resume, checksum mismatch, zip path traversal), and end-to-end runs of both scripts on synthetic data.

## Smoke test (runs on the M1)

Trains ResNet-18 on a stratified subset with focal loss and class-balanced sampling:

```bash
.venv/bin/python scripts/smoke_test.py            # uses configs/smoke_test.yaml
.venv/bin/python scripts/smoke_test.py --epochs 1 --subset-size 800   # quick pass
```

Outputs: `checkpoints/smoke_test.pt`, `outputs/smoke_test/metrics.json`, `outputs/smoke_test/training_curves.png`. The script auto-selects the MPS device on Apple Silicon and prints sensitivity, specificity, and AUC.

## Grad-CAM demo

```bash
.venv/bin/python scripts/demo_gradcam.py
```

Renders heatmap overlays for melanoma and nevus samples into `outputs/gradcam/` (individual PNGs plus a grid). Runs on CPU by default for attribution stability; a checkpoint from the smoke test is required.

## EDA notebook

The notebook is committed already executed (figures included). To rebuild and re-run it:

```bash
.venv/bin/python scripts/build_eda_notebook.py
.venv/bin/python -m ipykernel install --user --name melanoma-xai-bias --display-name "Python (melanoma-xai-bias)"
.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/01_eda_ham10000.ipynb
```

## Hardware notes

- MacBook Air M1: all data preparation, tests, XAI analysis, evaluation, and the deployment chapter run locally. Smoke training uses the MPS backend and finishes in minutes.
- Real training (architecture grid, seeds, mitigation runs) is planned for free cloud GPUs, about 30 GPU-hours per week on Kaggle, with Colab and Lightning AI as overflow. See the compute section of the research plan.

## Troubleshooting

- **Dataverse HTTP 403**: the API blocks default Python user agents; the download module sends a browser user agent and uses `format=original` so checksums match the originals.
- **First training run downloads pretrained weights**: timm fetches ImageNet weights from Hugging Face once (about 45 MB for ResNet-18); after that it works offline. Use `pretrained: false` in the config for fully offline runs.
- **Download interrupted**: rerun the same command; partial files resume via HTTP Range, stale partials are restarted automatically.
- **First matplotlib import is slow**: one-time font cache build.
- **MPS attribution**: Grad-CAM runs on CPU by default; `--device mps` is available but CPU is more stable for gradients.
