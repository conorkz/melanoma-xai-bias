# Melanoma benchmark: imbalance correction, fairness, XAI, and external validation

**Student:** Nursultan Zhanat
**Supervisor:** Prof. Sun Baoshan
**Date:** 11 November 2026
**Code:** Stage 1 at commit `026f907`, Stage 2 at commit `6b8615e`. Both runs report a clean working tree.

## 1. Question and hypothesis

**RQ1.** How do CNN-based and transformer-based architectures compare on imbalanced melanoma classification when combined with modern imbalance correction, measured by sensitivity, balanced accuracy, and AUC rather than raw accuracy?

**H1.** Focal loss plus class-balanced sampling improves minority class sensitivity over plain cross-entropy across the tested architectures.

This update reports the controlled benchmark for RQ1, the lightness-stratified fairness analysis, the quantitative XAI analysis, and the HIBA external validation.

## 2. Protocol

- **Data:** HAM10000, 10,015 dermoscopic images, binary endpoint (melanoma versus all other diagnoses).
- **Split:** lesion-aware train/validation/test split, shared by every run and cached. Test n = 1,527, melanoma n = 186.
- **Stage 1:** EfficientNet-B0 at 160 px, 15 epochs, eight loss, sampling, and augmentation configurations, three seeds each (24 runs).
- **Stage 2:** ResNet-50, EfficientNet-B0, and ViT-S/16 at 224 px, 15 epochs, three pre-declared conditions, three seeds each (27 runs).
  - CE baseline: CE, random sampler, standard augmentation.
  - Planned imbalance correction: focal gamma 2, class-balanced sampling, standard augmentation.
  - Stage 1 winner: CE, random sampler, strong augmentation.
- **Metrics:** sensitivity, specificity, balanced accuracy, AUC, and sensitivity at 95% specificity.
- **Threshold:** selected on validation at 95% specificity, frozen, then applied once to the untouched test split.
- **Selection:** Stage 1 is a descriptive screen. The Stage 2 conditions are pre-declared. Test metrics are reported for the three conditions and are never used for selection.
- **Uncertainty:** mean and standard deviation across three seeds, plus 1,000-sample bootstrap 95% confidence intervals on test.
- **Reproducibility:** every run writes its config snapshot, git commit, seed, device, package versions, and per-sample predictions.

## 3. Results

### Stage 1: loss, sampling, and augmentation screen

| Configuration | Val sens | Test sens (95% CI) | Specificity | Balanced acc | AUC |
|---|---|---|---|---|---|
| stage1_ce_random_strong | 0.556 ± 0.027 | 0.513 (0.438 to 0.582) | 0.943 | 0.728 | 0.901 |
| stage1_ce_random | 0.536 ± 0.016 | 0.530 (0.456 to 0.600) | 0.948 | 0.739 | 0.897 |
| stage1_ce_cb | 0.521 ± 0.086 | 0.484 (0.411 to 0.554) | 0.938 | 0.711 | 0.890 |
| stage1_focal2_random | 0.509 ± 0.041 | 0.487 (0.412 to 0.560) | 0.946 | 0.717 | 0.894 |
| stage1_focal2_cb | 0.502 ± 0.039 | 0.480 (0.408 to 0.551) | 0.943 | 0.712 | 0.897 |
| stage1_wce_random | 0.502 ± 0.043 | 0.493 (0.421 to 0.564) | 0.937 | 0.715 | 0.896 |
| stage1_focal2_cb_strong | 0.483 ± 0.013 | 0.491 (0.421 to 0.558) | 0.943 | 0.717 | 0.896 |
| stage1_focal3_cb | 0.449 ± 0.051 | 0.478 (0.405 to 0.553) | 0.946 | 0.712 | 0.899 |

### Stage 2: architecture comparison

| Configuration | Val sens | Test sens (95% CI) | Specificity | Balanced acc | AUC |
|---|---|---|---|---|---|
| stage2_efficientnet_b0_ce_random_strong | 0.521 ± 0.021 | 0.563 (0.489 to 0.636) | 0.952 | 0.757 | 0.912 |
| stage2_efficientnet_b0_ce_random | 0.553 ± 0.035 | 0.556 (0.480 to 0.624) | 0.953 | 0.754 | 0.914 |
| stage2_efficientnet_b0_focal2_cb | 0.481 ± 0.042 | 0.477 (0.405 to 0.548) | 0.950 | 0.713 | 0.891 |
| stage2_resnet50_ce_random | 0.526 ± 0.044 | 0.570 (0.497 to 0.639) | 0.951 | 0.761 | 0.897 |
| stage2_resnet50_ce_random_strong | 0.549 ± 0.015 | 0.568 (0.498 to 0.637) | 0.942 | 0.755 | 0.899 |
| stage2_resnet50_focal2_cb | 0.489 ± 0.027 | 0.453 (0.379 to 0.523) | 0.947 | 0.700 | 0.884 |
| stage2_vit_s_focal2_cb | 0.474 ± 0.034 | 0.482 (0.408 to 0.556) | 0.940 | 0.711 | 0.872 |
| stage2_vit_s_ce_random_strong | 0.466 ± 0.030 | 0.435 (0.366 to 0.503) | 0.941 | 0.688 | 0.866 |
| stage2_vit_s_ce_random | 0.423 ± 0.042 | 0.400 (0.329 to 0.471) | 0.941 | 0.670 | 0.854 |

### Headline numbers

- **Stage 1 validation winner:** CE with strong augmentation, 0.556 ± 0.027.
- **Planned imbalance correction:** focal2 + class-balanced, 0.502 ± 0.039 on validation. It did not beat the CE baseline (0.536 ± 0.016).
- **Stage 2, focal2 + class-balanced versus CE baseline:**
  - ResNet-50: 0.453 versus 0.570, **-11.7 sensitivity points**.
  - EfficientNet-B0: 0.477 versus 0.556, **-7.9 points**.
  - ViT-S/16: 0.482 versus 0.400, **+8.2 points**.
- **Architecture ranking at CE baseline:** ResNet-50 (0.570) > EfficientNet-B0 (0.556) > ViT-S/16 (0.400).
- **Architecture ranking under focal2 + class-balanced:** ViT-S/16 (0.482) > EfficientNet-B0 (0.477) > ResNet-50 (0.453).
- **Best overall test sensitivity:** ResNet-50 CE baseline 0.570 (95% CI 0.497 to 0.639); ResNet-50 CE + strong 0.568 (0.498 to 0.637); EfficientNet-B0 CE + strong 0.563 (0.489 to 0.636). The intervals overlap.
- **Best AUC:** EfficientNet-B0 CE baseline 0.914.

### Fairness across lightness strata

- Test set: 644 light images with 81 melanomas, 840 medium with 99, and 43 dark with 6.
- The dark stratum has the lowest AUC in all nine Stage 2 configurations: 0.71 to 0.87, against 0.83 to 0.90 for light and 0.88 to 0.93 for medium.
- The dark stratum has the highest false positive rate in all nine configurations: 0.08 to 0.19, against 0.04 to 0.07 for light and medium.
- Dark-stratum sensitivity ranges from 0.22 to 0.72, and the bootstrap intervals span 0 to 1. With six melanoma cases, the estimate is underpowered.
- The medium stratum usually has the highest sensitivity. The light stratum is often lower, which is consistent with a lesion-contrast effect.
- Full table: `results/fairness/lightness/fairness.md`. Figure: `results/figures/fairness_by_tone.png`.

### Quantitative XAI

Grad-CAM on the CE-baseline models, three seeds, 343 tone-stratified test images per seed (all 43 dark images plus 150 light and 150 medium).

| Architecture | Light mass | Dark mass | Light pointing | Dark pointing | Dark error rate | rho (mass, tone) |
|---|---|---|---|---|---|---|
| ResNet-50 | 0.593 | 0.504 | 0.882 | 0.674 | 0.178 | -0.167 |
| EfficientNet-B0 | 0.456 | 0.342 | 0.718 | 0.449 | 0.140 | -0.170 |
| ViT-S/16 | 0.341 | 0.265 | 0.604 | 0.248 | 0.217 | -0.184 |

- Attribution mass inside the lesion is lower in the dark stratum for all three architectures. The mass-tone correlation is negative in every case with p < 1e-7.
- Pointing-game accuracy is lowest in the dark stratum in every architecture.
- Attribution mass is a weak predictor of individual errors: rho = -0.023 for EfficientNet-B0, 0.047 for ResNet-50, and 0.089 for ViT-S/16. Only the ViT correlation is significant, and the effect is small.
- The XAI signal is group-level, not per-image. Full results: `results/xai/xai.md`. Figure: `results/figures/xai_by_tone.png`.

### HIBA external validation

- 1,280 dermoscopic images and 196 melanomas. Frozen HAM10000 threshold at 95% specificity, no refitting.
- Overall sensitivity and specificity: ResNet-50 0.391 / 0.944, EfficientNet-B0 0.396 / 0.922, ViT-S/16 0.221 / 0.967.
- AUC: 0.807, 0.782, and 0.791. The CNN-transformer gap transfers; absolute sensitivity drops by 16 to 18 points relative to HAM10000.
- Fitzpatrick I, II, III: CNN sensitivity 0.35 to 0.43; ViT 0.20 to 0.33. No monotonic skin-tone gradient.
- Fitzpatrick IV: 25 images and zero melanomas, so sensitivity is undefined. No Fitzpatrick V-VI images.
- Full table: `results/external/hiba/external_summary.csv`.

## 4. Figures

- `results/figures/stage2_best_sensitivity_training_curves.png`
- `results/figures/stage2_best_sensitivity_roc_test.png`
- `results/figures/stage2_best_auc_training_curves.png`
- `results/figures/stage2_best_auc_roc_test.png`

The best-sensitivity figure uses the median seed of ResNet-50 CE baseline. The best-AUC figure uses the median seed of EfficientNet-B0 CE baseline.

## 5. Interpretation

1. **Does H1 hold? It is architecture-dependent.** Focal2 + class-balanced improves ViT-S/16 by 8.2 sensitivity points but reduces ResNet-50 by 11.7 and EfficientNet-B0 by 7.9. H1 is supported for the transformer and rejected for the CNNs in this pilot.
2. **The architecture ranking flips.** The CNNs lead at the CE baseline, but the ViT leads under focal2 + class-balanced. The AUC ranking is more stable (EfficientNet-B0 > ResNet-50 > ViT in both conditions), which suggests the flip is specific to thresholded sensitivity and may involve calibration or threshold placement.
3. **Strong augmentation is not a universal fix.** It is neutral for ResNet-50 (0.568 versus 0.570), small positive for EfficientNet-B0 (0.563 versus 0.556), and positive for ViT (0.435 versus 0.400). The intervals overlap, so this is suggestive, not conclusive.
4. **Specificity is stable.** All conditions operate near 95% specificity on validation; test specificity is 0.94 to 0.955. The sensitivity differences are not explained by a large specificity shift.

## 6. Limitations

- Training uses HAM10000 only. HIBA adds one external dermoscopic source, but cross-dataset generalization is still narrow.
- Binary endpoint. Seven-class classification is a secondary experiment.
- Three seeds. The confidence intervals are wide.
- 160 px Stage 1, 224 px Stage 2. Do not compare numbers across stages.
- The test set has 186 melanoma images. The dark lightness stratum has 43 images and 6 melanomas.
- HAM10000 has no Fitzpatrick labels. Fairness uses an image-based L* proxy, not clinical labels.
- HIBA Fitzpatrick IV has 25 images and no melanomas, and HIBA has no Fitzpatrick V-VI. The dark-skin claim remains untested on clinical labels.
- The top Stage 2 rows are within the confidence intervals. Treat the ranking as pilot evidence, not a final claim.

## 7. Next step

The remaining experiment is a clinical dark-skin evaluation with DDI or PASSION. DDI is the smaller, biopsy-proven option and is the faster path. The imaging modality difference from HAM10000 must be reported as a limitation. A second extension is PROVe-AI, which has the same Fitzpatrick I-IV range as HIBA and is therefore not a solution to the V-VI gap.

## 8. Reproduce

```bash
# Stage 1: 24 runs
python scripts/run_sweep.py --config configs/sweep_stage1.yaml --skip-existing
python scripts/collect_results.py \
  --runs-dir outputs/runs --run-prefix stage1 \
  --out-dir results --prefix rq1_stage1

# Stage 2: 27 runs
python scripts/run_sweep.py --config configs/sweep_stage2.yaml --skip-existing
python scripts/collect_results.py \
  --runs-dir outputs/runs --run-prefix stage2 \
  --out-dir results --prefix rq1_stage2

# Fairness, XAI, and external validation
python scripts/build_tone_table.py
python scripts/analyze_fairness.py
python scripts/plot_fairness.py
python scripts/analyze_xai.py --backbones efficientnet_b0 --seeds 1,2,3 --max-images 150 --out-dir results/xai/efficientnet_b0
python scripts/analyze_xai.py --backbones resnet50 --seeds 1,2,3 --max-images 150 --out-dir results/xai/resnet50
python scripts/analyze_xai.py --backbones vit_s --seeds 1,2,3 --max-images 150 --out-dir results/xai/vit_s
python scripts/combine_xai_results.py
python scripts/plot_xai.py
python scripts/evaluate_external.py --device mps --out-dir results/external/hiba
```

On Kaggle, replace `.venv/bin/python` with `python`.
