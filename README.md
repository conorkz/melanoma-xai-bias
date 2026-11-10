# melanoma-xai-bias

Research prototype for the Master's thesis **"Interpretable Computer Vision for Fair Melanoma Detection"**.

This repository covers Phase 0, Phase 1, and the Phase 2 benchmark infrastructure: a reproducible project skeleton, an exploratory data analysis of HAM10000, a config-driven training pipeline with class imbalance handling (focal loss, class-balanced sampling), gradient-based explanations (Grad-CAM via Captum), lesion-aware data splitting, and a sweep runner plus results aggregation for the RQ1 benchmark. It runs end to end on a MacBook Air M1; heavier training is designed for free cloud GPUs (Kaggle, Colab, Lightning AI).

The full research plan lives in `RESEARCH_PLAN.md` in the parent research folder.

## Project structure

```
melanoma-xai-bias/
  configs/
    smoke_test.yaml              single reproducible smoke config
    pilot.yaml                   base config for the RQ1 benchmark
    sweep_stage1.yaml            24-run imbalance and augmentation screen
    sweep_stage2.yaml            18-run architecture comparison
  notebooks/01_eda_ham10000.ipynb  executed EDA with figures
  scripts/
    download_data.py             fetch HAM10000 from Harvard Dataverse
    train.py                     full training, test evaluation, prediction export
    run_sweep.py                 expand and execute a config sweep
    collect_results.py           aggregate seeds into the benchmark table
    smoke_test.py                tiny training run (subset, few epochs)
    demo_gradcam.py              Grad-CAM heatmaps for sample lesions
    build_eda_notebook.py        regenerate the EDA notebook
    run_tests.sh                 full test suite
  src/melanoma/
    config.py utils.py tracking.py
    data/       download, metadata, splits, subsets, dataset, transforms
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

The suite covers config loading, metadata validation, grouped splits, lesion-safe subsetting, transforms, dataset edge cases (missing, corrupt, grayscale images), model factory, focal loss numerics, class-balanced sampling, metrics with degenerate splits and bootstrap intervals, the trainer (checkpoints, early stopping, best-checkpoint restore), experiment tracking, Grad-CAM (CNN and ViT), the download logic with mocked network sessions (resume, checksum mismatch, zip path traversal), and end-to-end runs of the training, sweep, and aggregation scripts on synthetic data.

## Smoke test (runs on the M1)

Trains ResNet-18 on a stratified subset with focal loss and class-balanced sampling:

```bash
.venv/bin/python scripts/smoke_test.py            # uses configs/smoke_test.yaml
.venv/bin/python scripts/smoke_test.py --epochs 1 --subset-size 800   # quick pass
```

Outputs: `checkpoints/smoke_test.pt`, `outputs/smoke_test/metrics.json`, `outputs/smoke_test/training_curves.png`. The script auto-selects the MPS device on Apple Silicon and prints sensitivity, specificity, and AUC.

## RQ1 pilot benchmark

`configs/pilot.yaml` defines one run: EfficientNet-B0 at 160 px, focal loss, class-balanced sampling, 15 epochs, and a validation-selected threshold at 95% specificity. `scripts/train.py` trains one run, restores the best validation checkpoint, evaluates the untouched test split once, and exports per-sample predictions.

Stage 1 is a 24-run pilot (8 configurations x 3 seeds) covering loss, sampling, and augmentation. The two `strong` variants test the aggressive augmentation regime from the research plan:

```bash
# Generate the run configs without training
.venv/bin/python scripts/run_sweep.py --config configs/sweep_stage1.yaml --dry-run

# Run the grid on a GPU machine; --skip-existing resumes safely
.venv/bin/python scripts/run_sweep.py --config configs/sweep_stage1.yaml --skip-existing

# Build the benchmark table
.venv/bin/python scripts/collect_results.py \
  --runs-dir outputs/runs --run-prefix stage1 \
  --out-dir results --prefix rq1_stage1
```

Choose the winning Stage 1 configuration from the `val_*` columns in `results/rq1_stage1.csv`, not the test columns. Test metrics are reported once for the selected configuration and must not drive selection.

Outputs per run:

- `outputs/runs/<run_name>/metrics.json`: validation and test metrics, bootstrap CIs, threshold, split summary, history.
- `outputs/runs/<run_name>/run.json`: config snapshot, git commit, seed, device, package versions.
- `outputs/runs/<run_name>/predictions_{val,test}.csv`: one row per image, the bridge to the fairness and XAI phases.
- `results/rq1_stage1.csv` and `results/rq1_stage1.md`: mean and standard deviation across seeds.

The test split is evaluated once per run. The operating threshold is chosen on validation at 95% specificity and then applied unchanged to test. Every run reuses the same lesion-aware split cached at `outputs/splits/ham10000_lesion_seed42.csv`.

For a fast local pipeline check on a subset, use a separate split file so the cached full-data split is not overwritten:

```bash
.venv/bin/python scripts/train.py \
  --config configs/pilot.yaml --device mps \
  --subset-size 500 --epochs 1 --max-batches 5 \
  --split-file outputs/splits/debug_subset500.csv \
  --output-dir outputs/debug/train_smoke \
  --checkpoint checkpoints/debug_train_smoke.pt
```

After Stage 1, update the `best_imbalance` variants in `configs/sweep_stage2.yaml` if a different loss, sampler, or augmentation regime wins on the validation columns. All Stage 2 runs use 224 px so the architecture comparison is not confounded by resolution. Then run the architecture comparison:

```bash
.venv/bin/python scripts/run_sweep.py --config configs/sweep_stage2.yaml --skip-existing
.venv/bin/python scripts/collect_results.py \
  --runs-dir outputs/runs --run-prefix stage2 \
  --out-dir results --prefix rq1_stage2
```

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
