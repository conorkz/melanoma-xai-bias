# Paper plan

**Working title:** Imbalance correction reverses CNN-transformer rankings in melanoma classification: a controlled benchmark and fairness analysis

**Status:** the RQ1 benchmark, the first fairness analysis, and the EfficientNet-B0 XAI pilot are complete. The ResNet-50 and ViT-S/16 XAI analyses and the external tone-labeled validation are in progress. The LaTeX skeleton in `paper/` contains the current results and placeholders for the pending experiments.

## Contributions

1. A controlled benchmark of ResNet-50, EfficientNet-B0, and ViT-S/16 on HAM10000 under three pre-declared conditions: CE baseline, focal loss with class-balanced sampling, and CE with strong augmentation. Each condition uses three seeds, a lesion-aware split, and a validation-selected threshold at 95% specificity.
2. Evidence that focal loss with class-balanced sampling helps ViT-S/16 but hurts the two CNNs, which reverses the architecture ranking on sensitivity at 95% specificity. The AUC ranking is stable.
3. A lightness-stratified fairness analysis that quantifies the representation gap and a consistent dark-stratum AUC and false positive rate signal, while showing that the dark-stratum sensitivity estimate is underpowered.
4. A quantitative XAI protocol for attribution leakage and faithfulness, used to test whether attribution mass outside the lesion predicts errors and tone group. Pending.

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
- Table 3: Lightness-stratified fairness summary, generated from `results/fairness/lightness/fairness_summary.csv`.
- Table 4: EfficientNet-B0 XAI pilot, generated from `results/xai/efficientnet_b0/xai_summary.csv`.
- Figure 1: protocol diagram (to be drawn).
- Figure 2: Stage 2 sensitivity at 95% specificity by architecture and condition (to be drawn).
- Figure 3: ROC curve for the best Stage 2 run.
- Figure 4: fairness sensitivity and AUC by tone group (to be drawn).
- Figure 5: attribution mass inside the lesion by tone group and error status (EfficientNet-B0 pilot done; other architectures pending).

## Experiment status

| Experiment | Status | Output |
|---|---|---|
| Stage 1 imbalance and augmentation screen | Done, 24 runs | `results/rq1_stage1.*` |
| Stage 2 architecture comparison | Done, 27 runs | `results/rq1_stage2.*` |
| HAM10000 L* fairness pilot | Done | `results/fairness/lightness/` |
| HIBA external dermoscopic validation | Data downloaded, evaluation pending | `data/external/hiba/` |
| Clinical dark-skin validation | Dataset not selected | DDI or PASSION |
| Quantitative XAI on HAM10000 | EfficientNet-B0 pilot done; ResNet-50 and ViT pending | `results/xai/efficientnet_b0/` |

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

- Weeks 1 to 2: finish the ResNet-50 and ViT-S/16 XAI analyses and the HIBA external evaluation.
- Weeks 3 to 4: clinical dark-skin evaluation with DDI or PASSION.
- Weeks 5 to 6: figures, statistical tests, and the full results section.
- Weeks 7 to 8: supervisor feedback, polish, and submission.

A workshop version is possible in 3 to 4 weeks with the RQ1 benchmark and the L* fairness pilot. A journal submission needs the XAI and external results.

## Open decisions

1. Clinical dark-skin dataset: DDI (small, biopsy-proven, FST I-VI) or PASSION (larger, FST III-VI). DDI is the faster option.
2. Whether the paper is a single combined paper or two papers: a benchmark paper and a fairness and XAI paper. The combined paper is stronger for a master's thesis.
3. Whether the supervisor is a co-author. This should be settled before submission.
4. Expand the XAI pilot to ResNet-50 and ViT-S/16 before submission.

## Writing rules

- No em dashes or en dashes. Use hyphens, commas, or separate sentences.
- Every table is generated by `scripts/build_paper_tables.py`. Do not hand-type numbers.
- Report means and standard deviations across three seeds, plus bootstrap confidence intervals on test.
- Do not use the test set for model selection.
- Call the L* strata image-based lightness groups, not Fitzpatrick labels.
