# Paper plan

**Working title:** Architecture-dependent effects of imbalance correction in melanoma classification: fairness, calibration, and explanation analysis

**Status:** the RQ1 benchmark, the fairness analysis, the calibration analysis, the paired ranking tests, the three-architecture XAI analysis, and the nine-run HIBA external validation are complete. The 27-run HIBA validation, the clinical dark-skin evaluation, and the PROVe-AI extension are pending. The LaTeX draft in `paper/` contains the current results.

## Contributions

1. A controlled benchmark of ResNet-50, EfficientNet-B0, and ViT-S/16 on HAM10000 under three pre-declared conditions: CE baseline, focal loss with class-balanced sampling, and CE with strong augmentation. Each condition uses three seeds, a lesion-aware split, and a validation-selected threshold at 95% specificity.
2. Evidence that focal loss with class-balanced sampling helps ViT-S/16 but hurts the two CNNs. The paired within-architecture intervals include zero, but the difference-in-differences between each CNN and the ViT excludes zero. The AUC ranking is stable.
3. A calibration analysis showing that focal loss with class-balanced sampling roughly triples expected calibration error, which identifies calibration and threshold placement as the mechanism behind the sensitivity reversal.
4. A lightness-stratified fairness analysis at three cutoffs. The dark stratum has a higher false positive rate at every cutoff, while the dark AUC deficit is present only at the strictest cutoff. Adjusted models show that darker skin is associated with more false positives and lighter skin with more false negatives, with lesion contrast moving both error types in opposite directions.
5. A quantitative XAI analysis across all three architectures showing that attribution mass inside the lesion is lower in the dark stratum, and that false negatives have lower mass than false positives.
6. An external validation on HIBA that reproduces the CNN-transformer gap but shows a 16 to 18 point sensitivity drop, with no Fitzpatrick IV melanomas and no Fitzpatrick V-VI images.

## Section outline

1. Introduction. The clinical problem, class imbalance, skin tone bias, the black-box problem, the gap, the contributions.
2. Related work. Melanoma classification and public datasets, imbalance correction, fairness in dermatology, XAI in dermatology, quantitative XAI metrics.
3. Methods. Data, splits, preprocessing, architectures, training, conditions, metrics and threshold, statistical analysis, fairness analysis, XAI metrics, reproducibility.
4. Results. Stage 1 screen, Stage 2 architecture comparison, ranking reversal, fairness pilot, XAI results.
5. Discussion. The architecture-dependent effect, the ranking flip and AUC stability, the fairness signal, the light and medium contrast effect, the representation gap, implications for model selection.
6. Limitations. HAM10000 only, binary endpoint, three seeds, 160 and 224 px, L* proxy, no FST V-VI dermoscopic data, clinical modality confound, no reader study, no deployment.
7. Ethics. Public de-identified data, no clinical claims, Fitzpatrick limitations, representation gap.
8. Conclusion.
9. Reproducibility appendix. Configs, seeds, splits, per-sample predictions, code.

## Tables and figures

- Table 1: Stage 1 screen, generated from `results/rq1_stage1.csv`.
- Table 2: Stage 2 architecture comparison, generated from `results/rq1_stage2.csv`.
- Table 3: Paired sensitivity tests, generated from `results/ranking/ranking_summary.csv`.
- Table 4: Test-set calibration, generated from `results/calibration/calibration_overall.csv`.
- Table 5: Lightness-stratified fairness summary, generated from `results/fairness/lightness/fairness_summary.csv`.
- Table 6: Dark-skin cutoff sensitivity, generated from `results/dark_skin/threshold_sensitivity.csv`.
- Table 7: Adjusted false-positive and false-negative models, generated from `results/dark_skin/continuous_model.csv`.
- Table 8: XAI metrics by architecture and lightness group, generated from `results/xai/xai_all_summary.csv`.
- Table 9: XAI Spearman correlations, generated from `results/xai/xai_all_correlations.csv`.
- Table 10: HIBA external validation by Fitzpatrick type, generated from `results/external/hiba/external_summary.csv`.
- Table 11: HIBA external validation by training condition, generated from `results/external/hiba/external_summary.csv`.
- Figure 1: Stage 2 ROC curve for the best run.
- Figure 2: fairness AUC and false positive rate by tone group.
- Figure 3: dark-skin robustness, cutoff sensitivity and adjusted models.
- Figure 4: attribution mass and pointing game by tone group.
- Figure 5: calibration reliability and ECE by tone group.
- Figure 6: training curves for the best run.

## Experiment status

| Experiment | Status | Output |
|---|---|---|
| Stage 1 imbalance and augmentation screen | Done, 24 runs | `results/rq1_stage1.*` |
| Stage 2 architecture comparison | Done, 27 runs | `results/rq1_stage2.*` |
| Paired ranking tests | Done | `results/ranking/` |
| Test-set calibration | Done, 27 runs | `results/calibration/` |
| HAM10000 L* fairness pilot | Done | `results/fairness/lightness/` |
| Dark-skin cutoff and adjusted models | Done | `results/dark_skin/` |
| HIBA external dermoscopic validation | Done, 18 runs (CE + focal) | `results/external/hiba/` |
| Clinical dark-skin validation | Dataset not selected | DDI or PASSION |
| Quantitative XAI on HAM10000 | Done, 3 architectures x 3 seeds | `results/xai/` |

## Target venues

Do not choose a final venue until the XAI and external results are in. Candidate journals:

- Medical Image Analysis
- IEEE Journal of Biomedical and Health Informatics
- Computers in Biology and Medicine
- Computer Methods and Programs in Biomedicine
- Artificial Intelligence in Medicine
- Scientific Reports or PLOS Digital Health, broader open access

Candidate workshops for an early version:

- ISIC Skin Image Analysis Workshop at MICCAI
- MICCAI workshops on fairness, uncertainty, or explainability
- ISBI workshops

Check scope, review time, open access fee, and indexing. Ask the supervisor which venue fits the department's expectations.

## Timeline

- Weeks 1 to 2: finish the ResNet-50 and ViT-S/16 XAI analyses and the HIBA external evaluation. Done.
- Weeks 3 to 4: clinical dark-skin evaluation with DDI or PASSION.
- Weeks 5 to 6: figures, statistical tests, and the full results section.
- Weeks 7 to 8: supervisor feedback, polish, and submission.

A workshop version is possible in 3 to 4 weeks with the RQ1 benchmark and the L* fairness pilot. A journal submission needs the XAI and external results.

## Open decisions

1. Clinical dark-skin dataset: DDI (small, biopsy-proven, FST I-VI) or PASSION (larger, FST III-VI). DDI is the faster option.
2. Whether the paper is a single combined paper or two papers: a benchmark paper and a fairness and XAI paper. The combined paper is stronger for a master's thesis.
3. Whether the supervisor is a co-author. This should be settled before submission.
4. Whether to add PROVe-AI, which has the same Fitzpatrick I-IV limitation as HIBA.

## Writing rules

- No em dashes or en dashes. Use hyphens, commas, or separate sentences.
- Every table is generated by `scripts/build_paper_tables.py`. Do not hand-type numbers.
- Report means and standard deviations across three seeds, plus bootstrap confidence intervals on test.
- Do not use the test set for model selection.
- Call the L* strata image-based lightness groups, not Fitzpatrick labels.
