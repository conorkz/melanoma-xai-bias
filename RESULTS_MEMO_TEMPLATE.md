# RQ1 pilot: imbalance correction and architecture comparison

**Student:** Nursultan Zhanat
**Supervisor:** Prof. Sun Baoshan

## 1. Question and hypothesis

**RQ1.** How do CNN-based and transformer-based architectures compare on imbalanced melanoma classification when combined with modern imbalance correction, measured by sensitivity, balanced accuracy, and AUC rather than raw accuracy?

**H1.** Focal loss plus class-balanced sampling improves minority class sensitivity over plain cross-entropy across the tested architectures.

This update reports the first controlled benchmark for RQ1. It is the in-domain control condition for the fairness analysis that follows.

## 2. Protocol

- **Data:** HAM10000, 10,015 dermoscopic images, binary endpoint (melanoma versus all other diagnoses).
- **Split:** lesion-aware train/validation/test split, shared by every run and cached at `outputs/splits/ham10000_lesion_seed42.csv`.
- **Stage 1:** EfficientNet-B0 at 160 px, 15 epochs, eight loss, sampling, and augmentation configurations, three seeds each (24 runs).
- **Stage 2:** ResNet-50, EfficientNet-B0, and ViT-S/16 at 224 px, three conditions (CE baseline, focal2 + class-balanced, CE + strong augmentation), three seeds each (27 runs).
- **Metrics:** sensitivity, specificity, balanced accuracy, AUC, and sensitivity at 95% specificity.
- **Threshold:** selected on validation at 95% specificity, frozen, then applied once to the untouched test split.
- **Selection:** Stage 1 is a descriptive screen. The Stage 2 conditions are pre-declared. Test metrics are reported for the three conditions and are never used for selection.
- **Uncertainty:** mean and standard deviation across seeds, plus 1,000-sample bootstrap 95% confidence intervals on test.
- **Reproducibility:** every run writes its config snapshot, git commit, seed, device, package versions, and per-sample predictions.

## 3. Results

### Stage 1: loss, sampling, and augmentation screen

[Paste the table from `results/rq1_stage1.md`. The `val_*` columns describe the screen; the test columns are descriptive only.]

### Stage 2: architecture comparison

[Paste the table from `results/rq1_stage2.md`. The table has three conditions per architecture. The test columns are the headline results for the pre-declared conditions.]

Headline numbers:

- Best configuration: [fill in]
- Test sensitivity at 95% specificity: [fill in] (95% CI [fill in] to [fill in])
- Change versus CE baseline: [fill in] sensitivity points, [fill in] balanced accuracy points
- Best architecture: [fill in]
- Cost in specificity: [fill in]

## 4. Figures

- `outputs/runs/<best_run>/training_curves.png`: training loss and validation balanced accuracy.
- `outputs/runs/<best_run>/roc_test.png`: test ROC curve.
- `results/rq1_stage1.md`: mean and standard deviation table across seeds.

## 5. Interpretation

1. Does H1 hold? Where does the gain appear, and where does it disappear?
2. Is there a sensitivity-specificity trade-off, and is it acceptable for a screening tool?
3. Does aggressive augmentation help, hurt, or interact with the loss and sampler?
4. Does the architecture ranking change between the three conditions?

## 6. Limitations

- HAM10000 only; no cross-dataset generalization yet.
- Binary melanoma versus rest; the seven-class task is a secondary experiment.
- 160 px pilots, 15 epochs, three seeds.
- No skin tone labels in this update, so no fairness claims are made here.
- The test split has roughly 170 melanoma images, so confidence intervals are wide; group-level analyses will be wider.

## 7. Next step

Phase 3: evaluate the frozen best model on tone-annotated data, report per-Fitzpatrick-group sensitivity and false negative rates, and test whether attribution leakage predicts the gap before it appears in group metrics.

## 8. Reproduce

```bash
# Stage 1: 24 runs
.venv/bin/python scripts/run_sweep.py --config configs/sweep_stage1.yaml --skip-existing
.venv/bin/python scripts/collect_results.py \
  --runs-dir outputs/runs --run-prefix stage1 \
  --out-dir results --prefix rq1_stage1

# Stage 2: 27 runs
.venv/bin/python scripts/run_sweep.py --config configs/sweep_stage2.yaml --skip-existing
.venv/bin/python scripts/collect_results.py \
  --runs-dir outputs/runs --run-prefix stage2 \
  --out-dir results --prefix rq1_stage2
```

On Kaggle, replace `.venv/bin/python` with `python`.
