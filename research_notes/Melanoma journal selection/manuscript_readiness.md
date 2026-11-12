# Manuscript readiness audit: melanoma XAI bias paper

**Scope and method.** Read-only audit of the current manuscript, tables, figures, results CSVs, and code on 12 November 2026. I read `paper/main.tex`, all eight section files, all eleven table files, `paper/README.md`, `paper/submission_checklist.md`, `PAPER_PLAN.md`, `RESULTS_MEMO.md`, `paper/cover_letter.md`, `references.bib`, the results CSVs, and the relevant analysis scripts. I also recomputed several checks from the saved per-sample predictions and per-seed result files. No files were modified.

**Bottom line.** The paper has a genuinely useful controlled design (3 architectures x 3 conditions x 3 seeds, lesion-aware split, frozen threshold, public code and per-sample predictions), and the writing is clear. It is not submission-ready today. The main problems are scientific, not cosmetic: the reported 95% intervals are not valid confidence intervals, the "AUC ranking is stable" claim is contradicted by the AUC changes, the external validation does not reproduce the interaction once seed variability is shown, and the XAI false-negative/false-positive comparison is confounded by using the predicted class as the Grad-CAM target. Several abstract and conclusion claims are stronger than the statistics support. The fixes are feasible with the existing data, but the paper needs a claim-level rewrite and a statistical reanalysis before any journal will accept it. A realistic timeline is 2 focused days for the core fixes plus 1 to 2 weeks for the optional experiments and journal formatting.

**Strengths to preserve.**
- The 3 x 3 x 3 design is controlled, the split is lesion-aware, and the threshold is validation-selected. This is better than most melanoma benchmark papers.
- The per-sample test predictions, validation predictions, and 27 checkpoints are present locally, so the analyses can be redone without retraining.
- The code repository is public and resolves at commit `e444228` (`git ls-remote https://github.com/conorkz/melanoma-xai-bias.git HEAD` returned `e444228b331b0b5b8959e9ffd04e3feadbaae610`).
- The paper is honest about the lack of Fitzpatrick V-VI data in the Limitations and Discussion, and it does not claim clinical utility.
- The error-type-specific fairness framing is more interesting than a single sensitivity gap, and the false-negative L* association is the most robust fairness result in the paper.
- The figures use a colorblind-safe palette and the main figures are 300 dpi.

---

## 1. Are the claims supported by the reported statistics?

### Takeaway

Partly. The within-architecture null result is supported, and the EfficientNet-B0 versus ViT-S/16 interaction is fairly robust. The ResNet-50 versus ViT-S/16 interaction is marginal and does not survive a seed-level analysis. The calibration claim is only partly supported: focal loss clearly worsens calibration for the CNNs, but the ViT's ECE rises by only 36%, and the ViT's AUC improves, so calibration does not explain the ViT gain. The external validation is not supportable as written because the ViT external sensitivity is highly variable across seeds. The fairness AUC deficit is not statistically significant at any cutoff. The XAI error-type claim is significant only for EfficientNet-B0. The paper would receive a major revision or rejection as written.

### Cited findings

**1. The Stage 2 "95% CI" is the mean of the per-run bootstrap bounds, not a confidence interval for the mean.** `scripts/collect_results.py:100-101` aggregates `sensitivity_ci_low` and `sensitivity_ci_high` with `mean` across seeds. The table then prints that average as "Test sens (95% CI)" (`paper/tables/stage2.tex:14`; `paper/sections/04_results.tex:15`). This is not a valid 95% CI. The per-run intervals describe test-set uncertainty for a single run; the mean of their bounds is neither a per-run interval nor a seed-level interval. A seed-level t interval is much wider:

| Configuration | Reported "95% CI" | Seed-level 95% CI from the 3 seed sensitivities | Seed values |
|---|---|---|---|
| ResNet-50 CE | 0.497 to 0.639 | 0.469 to 0.671 | 0.608, 0.527, 0.575 |
| EfficientNet-B0 CE | 0.480 to 0.624 | 0.465 to 0.646 | 0.543, 0.527, 0.597 |
| ViT-S/16 CE | 0.329 to 0.471 | 0.291 to 0.508 | 0.349, 0.430, 0.419 |
| ViT-S/16 CE+strong | 0.366 to 0.503 | 0.228 to 0.643 | 0.349, 0.516, 0.441 |
| ViT-S/16 Focal+CB | 0.408 to 0.556 | 0.428 to 0.536 | 0.505, 0.462, 0.478 |

The seed-level intervals were computed from `results/rq1_stage2_runs.csv` (per-seed `sensitivity`) with a two-sided t interval on 2 degrees of freedom. The reported intervals are systematically narrower. The same issue applies to `paper/tables/stage1.tex:11-18`.

**2. The paired-test table uses the envelope of three per-seed intervals and the maximum per-seed p-value.** `scripts/analyze_ranking.py:142-146` aggregates `ci_low` with `min`, `ci_high` with `max`, and `p_value` with `max` across seeds. The table then labels the result "95% CI" and "p" (`paper/tables/ranking.tex:4,11-20`). The per-seed deltas in `results/ranking/ranking_tests.csv` show why this matters:

| Comparison | Per-seed delta | Mean | Seed-level 95% CI | Reported interval | Reported p |
|---|---|---|---|---|---|
| EfficientNet-B0 focal vs CE | -0.022, -0.134, -0.081 | -0.079 | -0.219 to +0.061 | -0.210 to +0.055 | 0.590 |
| ResNet-50 focal vs CE | -0.172, -0.129, -0.048 | -0.116 | -0.272 to +0.039 | -0.244 to +0.031 | 0.248 |
| ViT-S/16 focal vs CE | +0.156, +0.032, +0.059 | +0.082 | -0.079 to +0.244 | -0.038 to +0.237 | 0.440 |
| DiD EfficientNet-B0 minus ViT-S/16 | -0.177, -0.167, -0.140 | -0.161 | -0.209 to -0.113 | -0.289 to -0.049 | 0.008 |
| DiD ResNet-50 minus ViT-S/16 | -0.328, -0.161, -0.108 | -0.199 | -0.484 to +0.087 | -0.437 to -0.011 | 0.032 |

The EfficientNet-B0 versus ViT-S/16 interaction is robust: all three seeds are negative and the seed-level interval excludes zero. The ResNet-50 versus ViT-S/16 interaction is not robust: the seed-level interval includes zero, and the reported p=0.032 is the maximum of the three per-seed p-values, not a test of the mean. The abstract's "the difference-in-differences between each CNN and the ViT excludes zero" (`paper/main.tex:34`) is therefore true only under the test-image bootstrap, not under seed-level uncertainty. A reviewer will ask for a hierarchical bootstrap or a mixed-effects model; with three seeds, the honest conclusion is "the direction is consistent for ResNet-50 across all three seeds, but the seed-level interval includes zero."

**3. The AUC ranking is not statistically stable, and the ViT closes the gap.** The paper says "The AUC ranking is stable" (`paper/main.tex:34`; `paper/sections/04_results.tex:30`; `paper/sections/05_discussion.tex:13`; `paper/sections/08_conclusion.tex:3`). The point estimates are 0.914, 0.897, 0.854 at CE and 0.891, 0.884, 0.872 under focal for EfficientNet-B0, ResNet-50, and ViT-S/16. The order is preserved, but the ViT gains +0.018 AUC while the CNNs lose 0.013 to 0.023. The EfficientNet-B0 versus ViT-S/16 gap narrows from 0.060 to 0.019. My paired bootstrap over test images (2,000 resamples, paired by image, from `results/predictions/`) gives:

| Contrast | Mean AUC difference | 95% bootstrap interval |
|---|---|---|
| ViT-S/16 minus EfficientNet-B0, CE | -0.060 | -0.088 to -0.031 |
| ViT-S/16 minus EfficientNet-B0, focal | -0.018 | -0.047 to +0.016 |
| ViT-S/16 minus ResNet-50, CE | -0.043 | -0.085 to -0.003 |
| ViT-S/16 minus ResNet-50, focal | -0.012 | -0.047 to +0.018 |

At CE the ViT is significantly worse than both CNNs in AUC. Under focal the difference is no longer distinguishable from zero. The sensitivity reversal is therefore not purely a threshold effect: the ViT's ranking quality improves, and the AUC gap closes. The paper's "AUC stable" is a point-estimate ordering, not a statistical stability claim. There is no AUC difference test anywhere in the paper or the results directory. `results/dark_skin/auc_differences.csv` tests dark versus light/medium AUC, not architecture AUC. This is the single most important missing analysis in the paper.

**4. "Focal loss roughly triples expected calibration error" is false for the ViT.** `paper/main.tex:34`, `paper/sections/01_introduction.tex:20`, and `paper/sections/08_conclusion.tex:3` say "roughly triples." Table 4 (`paper/tables/calibration.tex:10-18`) shows:

| Architecture | CE ECE | Focal ECE | Ratio |
|---|---|---|---|
| ResNet-50 | 0.046 | 0.203 | 4.4x |
| EfficientNet-B0 | 0.060 | 0.178 | 3.0x |
| ViT-S/16 | 0.146 | 0.199 | 1.36x |

The ViT CE ECE also has a standard deviation of 0.109, so the "worst-calibrated CE baseline" ranking is not stable. The paper's own Discussion (`paper/sections/05_discussion.tex:15`) lists the ViT numbers while still saying "roughly triples." This is an internal contradiction, not just a wording issue.

**5. Calibration is a correlate, not an identified mechanism.** The abstract says calibration "identifies calibration as a mechanism behind the thresholded reversal" (`paper/main.tex:34`). But the ViT's AUC improves at the same time as its ECE rises modestly, so its sensitivity gain is not explained by calibration. The CNNs lose AUC and gain ECE, so their sensitivity loss is a mixture of ranking and calibration. The paper has no temperature-scaling or recalibration experiment. The Discussion is more cautious ("Why the same correction helps the ViT and hurts the CNNs remains unexplained," `paper/sections/05_discussion.tex:15`), but the abstract, introduction, and conclusion are not. The claim needs to be softened to "calibration degradation tracks the CNN sensitivity loss, but it does not explain the ViT gain."

**6. The external validation does not reproduce the interaction once seed variability is shown.** Table 11 (`paper/tables/external_conditions.tex:11-16`) reports only means across three seeds. `results/external/hiba/external_metrics.csv` has the per-seed values:

| Architecture | Condition | Sensitivity seeds | Mean | SD |
|---|---|---|---|---|
| EfficientNet-B0 | CE | 0.362, 0.393, 0.434 | 0.396 | 0.036 |
| EfficientNet-B0 | Focal+CB | 0.184, 0.327, 0.250 | 0.253 | 0.071 |
| ResNet-50 | CE | 0.398, 0.444, 0.332 | 0.391 | 0.056 |
| ResNet-50 | Focal+CB | 0.235, 0.362, 0.347 | 0.315 | 0.070 |
| ViT-S/16 | CE | 0.082, 0.245, 0.337 | 0.221 | 0.129 |
| ViT-S/16 | Focal+CB | 0.301, 0.372, 0.270 | 0.315 | 0.052 |

The paired seed-level changes are: EfficientNet-B0 -0.178, -0.066, -0.184 (consistent); ResNet-50 -0.163, -0.082, +0.015 (mixed); ViT-S/16 +0.219, +0.127, -0.067 (mixed, one seed opposite). The ViT mean +9.4 points is an average of two positive seeds and one negative seed, with a CE SD of 0.129. The statement "reproduces the interaction" (`paper/sections/04_results.tex:113`; `paper/sections/08_conclusion.tex:7`; abstract `paper/main.tex:34`) is too strong. At best, the external evidence is directionally consistent for EfficientNet-B0 and suggestive for the ViT; it is not confirmatory. The table needs SDs or CIs, and the text needs the per-seed values.

**7. The fairness AUC deficit is not statistically significant at any cutoff.** `results/dark_skin/auc_differences.csv` at cutoff 55 contains dark-versus-light and dark-versus-medium AUC differences for all nine CE-baseline runs. Every 95% CI includes zero (p-values range 0.08 to 0.96). At cutoffs 58 and 60 the point estimates are near zero or reversed. The paper's "the dark stratum has the lowest AUC in all nine configurations" (`paper/sections/04_results.tex:55`) is a point-estimate description, not a significant effect. The abstract's "the dark AUC deficit is present only at the strictest cutoff" (`paper/main.tex:34`) implies a real deficit at cutoff 55; the data do not support "deficit" at any cutoff. This is selective reporting: the null AUC differences are in the results directory but not in the paper.

**8. The dark false-positive elevation is consistent but imprecise, and the paper does not test it.** At cutoff 55 the dark FPR is higher than both light and medium in 9/9 runs. At cutoff 58 it is higher in 8/9, and at cutoff 60 in 8/9 versus light and 9/9 versus medium. But the dark group has only 37 negatives at cutoff 55, and Fisher exact tests of dark versus light give p=0.053, 0.012, 0.429, 0.062, 0.020, 0.438, 0.040, 0.521, 0.004 (4 of 9 runs below 0.05). Table 6 (`paper/tables/dark_skin.tex:10-20`) reports only means across the nine CE-baseline runs, with no SDs or CIs. The statement "the false positive elevation does survive" (`paper/sections/05_discussion.tex:21`) is directionally true but statistically overstated. The paper should report the per-run counts, a cluster bootstrap CI, and the number of runs with a significant difference.

**9. The fairness abstract is symmetric, but the evidence is asymmetric.** The abstract says "darker skin is associated with more false positives and lighter skin with more false negatives" (`paper/main.tex:34`). Table 7 (`paper/tables/adjusted_model.tex:11-17`) shows the false-positive L* odds ratio is significant only for ResNet-50 (0.78, 0.61 to 0.94); EfficientNet-B0 is 0.88 (0.67 to 1.11) and ViT-S/16 is 0.77 (0.57 to 1.03), both including 1. The false-negative L* odds ratio is significant for all three (1.74, 1.66, 1.75, all CIs exclude 1). The robust fairness finding is the false-negative association, not the symmetric error-type shift. The paper should lead with the false-negative result and qualify the false-positive result as "significant for ResNet-50, directionally consistent for the other two."

**10. The contrast claim is also asymmetric.** The abstract says "lesion-skin contrast moves both error types in opposite directions" (`paper/main.tex:34`). The false-positive contrast ORs are 2.05, 1.69, 2.42, all significant. The false-negative contrast ORs are 0.58 (0.35 to 1.02), 0.74 (0.51 to 1.13), 0.48 (0.29 to 0.75); only the ViT interval excludes 1. The claim should be "higher contrast is consistently associated with more false positives; the negative association with false negatives is significant only for ViT-S/16."

**11. The XAI false-negative versus false-positive comparison is confounded by the target class.** `scripts/analyze_xai.py:244` sets `target_class = int(row["y_pred"])`, the predicted class, not the true class. For a false negative, the heatmap explains why the model predicted benign, not why it missed the melanoma. For a false positive, the heatmap explains why the model predicted melanoma. Comparing "mass inside the lesion" between these two heatmaps is therefore not a clean comparison of error types. The text's interpretation "false negatives look like attention outside the melanoma" (`paper/sections/05_discussion.tex:39`) may be an artifact of the target class. This is a major XAI validity issue. The fix is either to rerun Grad-CAM with the true class as the target for all images, or to reframe the analysis as "predicted-class attribution" and remove the error-type causal interpretation.

**12. The XAI false-negative versus false-positive difference is significant only for EfficientNet-B0.** The text reports EffNet 0.285 vs 0.522, ViT 0.298 vs 0.402, ResNet 0.578 vs 0.565 (`paper/sections/04_results.tex:90`). Those are pooled image-seed means. `results/xai/xai_all_metrics.csv` has 3 seeds per image. A cluster bootstrap over unique images (5,000 resamples) gives: EfficientNet-B0 -0.229 (95% CI -0.363 to -0.086), ResNet-50 -0.006 (-0.182 to +0.177), ViT-S/16 -0.080 (-0.187 to +0.032). Only EfficientNet-B0 is significant. The abstract's "false negatives have lower mass than false positives" (`paper/main.tex:34`) and the conclusion's same claim (`paper/sections/08_conclusion.tex:7`) are overgeneralized. The unique counts are also small: 7 FP and 15 FN for EfficientNet-B0, 11 and 16 for ResNet-50, 16 and 27 for ViT-S/16.

**13. The XAI "all three architectures" claim is not significant for ResNet-50.** A cluster bootstrap over unique images for dark versus light attribution mass gives: EfficientNet-B0 -0.115 (-0.192 to -0.035), ResNet-50 -0.088 (-0.176 to +0.001), ViT-S/16 -0.075 (-0.141 to -0.010). The ResNet interval just includes zero. The abstract's "Across all three architectures, Grad-CAM attribution mass inside the lesion is lower on darker images" (`paper/main.tex:34`) should be "significant for EfficientNet-B0 and ViT-S/16; directionally consistent but not significant for ResNet-50."

**14. The XAI correlation p-values treat repeated image-seed rows as independent.** `results/xai/xai_all_correlations.csv` has n=1,027 to 1,029 rows, but these are 43 unique dark images repeated 3 times (plus 150 light and 150 medium per seed). The p<1e-7 values in the Results (`paper/sections/04_results.tex:86`) are not valid for the unique-image population. The Methods acknowledges this (`paper/sections/03_methods.tex:71`), but the Results and abstract use "significant" language. The ViT mass-error correlation is 0.089 with p=0.004, which the paper calls "near zero" (`paper/sections/04_results.tex:90`); it is small but statistically detectable under the pooled analysis. The paper should replace the p-values with cluster-bootstrap CIs or per-seed correlations.

**15. The "seven-class classification is a secondary experiment" sentence is false.** `paper/sections/06_limitations.tex:6` and `RESULTS_MEMO.md:145` say seven-class classification is a secondary experiment. There is no seven-class analysis in `results/`, `paper/tables/`, or `paper/sections/`. The paper only reports the binary endpoint. Remove the sentence or run the experiment.

**16. The split file is not released, despite the ethics statement.** `paper/sections/07_ethics.tex:11` says "The code, configs, splits, per-sample predictions, and table generation scripts are released with the paper." The split file referenced in `configs/pilot.yaml:15` is `outputs/splits/ham10000_lesion_seed42.csv`; it is absent, and `.gitignore:17` ignores `outputs/`. The split is regenerable from the code and seed 42, but the statement is false as written. The paper also never states the split seed. Fix by committing the split file or changing the statement to "the split construction code and seed are released."

**17. The "pre-declared conditions" wording hides one data-dependent choice.** The three Stage 2 conditions are CE baseline, focal loss with class-balanced sampling, and CE with strong augmentation. The first two were planned; the third is the Stage 1 validation winner (`paper/sections/03_methods.tex:30`). The abstract, introduction, and conclusion call all three "pre-declared" (`paper/main.tex:34`; `paper/sections/01_introduction.tex:18`; `paper/sections/03_methods.tex:25`). The conditions were fixed before Stage 2 test results, which is good, but one was selected on Stage 1 validation. The precise wording is "two pre-declared conditions plus the Stage 1 validation winner, all fixed before Stage 2 test evaluation."

**18. The "test set is touched once per run" wording is misleading.** `paper/sections/01_introduction.tex:13` says this. The test set is not used for model selection, but it is used for the primary metrics, calibration, fairness, and XAI. The paper should say "the test set was not used for model selection; it was used for the primary evaluation and for secondary descriptive analyses."

**19. The L* cutoff rationale is missing.** `paper/sections/03_methods.tex:47,55` says the cutoff 55 is "pre-declared" and "specific to the HAM10000 lightness distribution," but it does not say when or how it was fixed, or whether the test set influenced it. The `lightness_group` function uses 70/55 (`src/melanoma/fairness/tone.py:92-98`). A reviewer will ask for the rationale and the timing. Add one sentence: "The 70/55 thresholds were fixed on the training set before test evaluation" or the actual procedure.

**20. There is no formal test of the "AUC ranking" or the "calibration tracks the reversal" claims.** The paper's central contrast is "AUC stable, sensitivity flips." There is no DeLong test, no paired AUC bootstrap for architecture differences, and no AUC interaction test. The calibration analysis is descriptive; there is no recalibration experiment. These are the two missing analyses that would make the mechanism claim credible.

### Inferences

- The controlled design is sound, but the inference layer is not. The reported intervals and p-values are the weakest part of the paper. A statistically literate reviewer will find them quickly because the code and CSVs are public.
- The paper's strongest defensible claims are: (a) focal loss with class-balanced sampling has an architecture-dependent effect, with a robust EfficientNet-B0 versus ViT-S/16 interaction and a directionally consistent but uncertain ResNet-50 versus ViT-S/16 interaction; (b) focal loss worsens calibration for the CNNs and changes both ranking quality and calibration; (c) the fairness signal is an error-type shift, with a robust false-negative L* association and a consistent but imprecise dark-stratum false-positive elevation; (d) the XAI analysis is exploratory and shows a dark-versus-light attribution mass difference for EfficientNet-B0 and ViT-S/16.
- The paper's weakest claims are: (a) "AUC ranking is stable"; (b) "calibration identifies the mechanism"; (c) "external validation reproduces the interaction"; (d) "false negatives have lower mass than false positives" across all architectures; (e) "darker skin is associated with more false positives" across all architectures; (f) "seven-class classification is a secondary experiment."
- The fixes are feasible with the saved predictions. A proper hierarchical bootstrap, AUC difference tests, external per-seed CIs, and XAI cluster bootstraps can all be computed from `results/predictions/`, `results/external/hiba/external_metrics.csv`, and `results/xai/xai_all_metrics.csv` without retraining.

### Gaps

- No temperature-scaling or isotonic recalibration experiment. This is the most direct test of the calibration mechanism and can be done with `outputs/.../predictions_val.csv` (all 27 runs have validation predictions).
- No AUC difference or AUC interaction tests. The paper's central "AUC stable" claim is untested.
- No group-level FPR/AUC confidence intervals in the paper. The CSVs have the data; the tables do not show it.
- No true-class Grad-CAM analysis. The current heatmaps explain the predicted class.
- No per-seed external CIs in the paper. The CSV has the per-seed values.
- No formal test of the XAI group differences. The correlations are pooled and pseudo-replicated.

---

## 2. Are there inconsistencies between the abstract, results, discussion, tables, and figures?

### Takeaway

Yes, and they are numerous enough that the paper needs a line-by-line consistency pass. The most serious are the "AUC stable," "roughly triples," and "external reproduces" claims, the Figure 3C caption, the "seven-class" limitation, and the table captions that describe intervals and sample sizes inaccurately. The Results and Discussion are often more cautious than the abstract and conclusion; the paper should align everything to the cautious version.

### Cited findings

| # | Claim or caption | Where | What the paper or data show | Fix |
|---|---|---|---|---|
| 1 | "The AUC ranking is stable" | `paper/main.tex:34`; `paper/sections/04_results.tex:30`; `paper/sections/05_discussion.tex:13`; `paper/sections/08_conclusion.tex:3` | ViT AUC 0.854 to 0.872 (+0.018); CNNs lose 0.013 to 0.023; EfficientNet-B0 versus ViT-S/16 gap 0.060 to 0.019; the focal gap is not significant (audit bootstrap -0.018, -0.047 to +0.016) | Say "the point-estimate ordering is preserved, but the ViT closes the AUC gap; the sensitivity reversal is not purely a threshold effect" |
| 2 | "Focal loss roughly triples ECE" | `paper/main.tex:34`; `paper/sections/01_introduction.tex:20`; `paper/sections/08_conclusion.tex:3` | ResNet 4.4x, EfficientNet 3.0x, ViT 1.36x (Table 4) | Say "roughly triples for the two CNNs, but rises by about one third for ViT-S/16" |
| 3 | "Strong augmentation keeps calibration close to the CE baseline, ECE 0.046 to 0.058" | `paper/sections/04_results.tex:36` | The next sentence says ViT CE 0.146 to strong 0.058, a 60% reduction (`paper/sections/04_results.tex:38`) | Split the sentence: "close for the CNNs; it substantially improves the ViT" |
| 4 | "Difference-in-differences between each CNN and the ViT excludes zero" | `paper/main.tex:34`; `paper/sections/01_introduction.tex:19`; `paper/sections/05_discussion.tex:7` | ResNet-ViT per-seed deltas -0.328, -0.161, -0.108; seed-level CI -0.484 to +0.087; reported p=0.032 is the max per-seed p (`scripts/analyze_ranking.py:146`) | Say "robust for EfficientNet-B0; directionally consistent but not robust to seed-level uncertainty for ResNet-50" |
| 5 | "Darker skin is associated with more false positives" | `paper/main.tex:34`; `paper/sections/08_conclusion.tex:5` | FP L* OR significant only for ResNet-50 (0.78, 0.61 to 0.94); EfficientNet 0.88 (0.67 to 1.11); ViT 0.77 (0.57 to 1.03) | Qualify: "significant for ResNet-50, directionally consistent for the other two" |
| 6 | "Lesion-skin contrast moves both error types in opposite directions" | `paper/main.tex:34`; `paper/sections/04_results.tex:59` | FN contrast OR 0.58 (0.35 to 1.02), 0.74 (0.51 to 1.13), 0.48 (0.29 to 0.75); only ViT excludes 1 | Qualify: "FP association robust; FN association significant only for ViT" |
| 7 | "False negatives have lower attribution mass than false positives" | `paper/main.tex:34`; `paper/sections/08_conclusion.tex:7` | ResNet 0.578 vs 0.565; cluster bootstrap -0.006 (-0.182 to +0.177); ViT -0.080 (-0.187 to +0.032); only EfficientNet significant | Say "significant for EfficientNet-B0; not for ResNet-50 or ViT-S/16" |
| 8 | "Weak pooled mass-error correlation" | `paper/main.tex:34`; `paper/sections/04_results.tex:90` | ViT rho=0.089, p=0.004 (`results/xai/xai_all_correlations.csv`) | Say "near zero for EfficientNet-B0 and ResNet-50; small but detectable for ViT-S/16 under the pooled analysis" |
| 9 | "External validation reproduces the interaction" | `paper/main.tex:34`; `paper/sections/04_results.tex:113`; `paper/sections/08_conclusion.tex:7` | ViT CE 0.082, 0.245, 0.337; focal 0.301, 0.372, 0.270; paired diffs +0.219, +0.127, -0.067 | Say "directionally consistent for EfficientNet-B0; mixed for ResNet-50 and ViT-S/16; external evidence is suggestive, not confirmatory" |
| 10 | "Fitzpatrick IV ... almost no melanoma cases" | `paper/sections/05_discussion.tex:31` | FST IV 25 images, 0 melanomas (Table 10) | Say "no Fitzpatrick IV melanomas" |
| 11 | "16 to 18 point sensitivity drop" | `paper/sections/05_discussion.tex:31`; abstract | Per-seed drops range roughly 8 to 27 points across architectures and seeds | Report the mean and SD or the per-seed range; say "16 to 18 points on average" |
| 12 | "All three architectures place less attribution inside the lesion on darker images" | `paper/main.tex:34`; `paper/sections/05_discussion.tex:37` | ResNet dark-light -0.088 (-0.176 to +0.001); ViT -0.075 (-0.141 to -0.010); EffNet -0.115 (-0.192 to -0.035) | Say "significant for EfficientNet-B0 and ViT-S/16; borderline for ResNet-50" |
| 13 | "Seven-class classification is a secondary experiment" | `paper/sections/06_limitations.tex:6`; `RESULTS_MEMO.md:145` | No seven-class results anywhere | Remove or run the experiment |
| 14 | Figure 3C caption: "Adjusted odds ratios for the pooled error model" | `paper/sections/04_results.tex:77` | The plotted values match the false-positive model in `paper/tables/adjusted_model.tex:11-13` (ViT 0.77/2.42, ResNet 0.78/2.05, EffNet 0.88/1.69), not the pooled `outcome=error` model in `results/dark_skin/continuous_model.csv` | Change caption to "false-positive model" or replot the pooled model |
| 15 | Table 4 caption: "Test-set calibration at the validation-selected threshold" | `paper/tables/calibration.tex:4` | ECE and Brier are computed from probabilities; they do not depend on the threshold | Say "test-set calibration (ECE and Brier) for the Stage 2 models" |
| 16 | Table 3 caption: "intervals are 95% bootstrap intervals over images" | `paper/tables/ranking.tex:4` | The intervals are the envelope of three per-seed intervals; the p is the max per-seed p | Say "per-seed intervals and their envelope; p is the maximum per-seed p" or, better, replace with a hierarchical analysis |
| 17 | Table 1/2 captions: "Values are mean and standard deviation across three seeds" | `paper/tables/stage1.tex:4`; `paper/tables/stage2.tex:4` | Only validation sensitivity has an SD; test specificity, balanced accuracy, and AUC are point estimates | Add SDs or change the caption |
| 18 | Table 8 caption: "343 test images per seed" | `paper/tables/xai.tex:4` | The n column is pooled across seeds: 450 light, 450 medium, 129 dark | Say "n is pooled across three seeds; the sample is 150 light + 150 medium + 43 dark per seed" |
| 19 | Table 9 caption: no pseudo-replication caveat | `paper/tables/xai_correlations.tex:4` | n=1,027 to 1,029 image-seed rows for 43 unique dark images; p-values treat repeated images as independent | Add the caveat and replace p-values with cluster-bootstrap CIs |
| 20 | Table 10/11 captions: "Values are means across three seeds" | `paper/tables/external.tex:4`; `paper/tables/external_conditions.tex:4` | The CSV has SDs; the ViT CE sensitivity SD is 0.129 | Add SDs or CIs |
| 21 | "The code, configs, splits, ... are released" | `paper/sections/07_ethics.tex:11` | The split file `outputs/splits/ham10000_lesion_seed42.csv` is missing; `outputs/` is gitignored | Commit the split file or change the statement |
| 22 | "The test set is touched once per run" | `paper/sections/01_introduction.tex:13` | The test set is reused for calibration, fairness, and XAI (not for model selection) | Say "not used for model selection; reused for descriptive secondary analyses" |
| 23 | "Three pre-declared conditions" | `paper/main.tex:34`; `paper/sections/01_introduction.tex:18`; `paper/sections/03_methods.tex:25` | The strong condition is the Stage 1 validation winner (`paper/sections/03_methods.tex:30`) | Say "two pre-declared conditions plus the Stage 1 validation winner, all fixed before Stage 2 test evaluation" |
| 24 | "Absolute CE-baseline sensitivity remains 16 to 18 points below HAM10000" | `paper/main.tex:34`; `paper/sections/08_conclusion.tex:7` | The per-seed drops are 21.0, 8.3, 24.3 for ResNet; 21.3, 13.4, 13.0 for EfficientNet; 27.5, 18.5, 8.2 for ViT | Say "16 to 18 points on average; the per-seed range is wide" |

### Inferences

- The abstract and conclusion are the most overclaimed parts. The Results and Discussion already contain many of the necessary caveats; the abstract and conclusion did not inherit them.
- The Figures and Tables are not the source of truth for the claims. Several captions are inaccurate, and the text sometimes reports numbers that are not in the tables (e.g., the FN/FP mass values, the pooled error model ORs).
- The paper appears to have been assembled from `RESULTS_MEMO.md` without a final consistency pass. The memo itself contains the same overclaims (`RESULTS_MEMO.md:114,135,140,145`).
- A single "claims table" with the exact number, the source CSV, and the caveat would prevent these inconsistencies. The table generation script already exists; the text is the weak link.

### Gaps

- No systematic number traceability check. The checklist says "Every number in the paper traces to a CSV" (`paper/submission_checklist.md:25`), but the checklist is unchecked and the "seven-class" sentence has no CSV.
- No line-by-line internal review by a second author. The corresponding author's revision is not visible in the current text.

---

## 3. Which required journal statements are present or missing?

### Takeaway

The paper has the core author, affiliation, funding, competing-interest, ethics, data, and code statements. It is missing several statements that the likely targets require: first-author ORCID, CRediT roles, acknowledgments, a generative-AI declaration, a reporting-guideline checklist (CLEAR, TRIPOD+AI, or STARD), permanent data/code identifiers, the split file, a supplement, word/figure/table counts, a running title, and line numbers. The ethics and data statements are too vague for a clinical journal, and the cover letter still has placeholders.

### Cited findings

**Present.**

| Statement | Location | Status |
|---|---|---|
| Title, authors, affiliations | `paper/main.tex:16-25` | Present |
| Corresponding author, email, ORCID (S.B.) | `paper/main.tex:21-23` | Present |
| Keywords | `paper/main.tex:37` | Present |
| Author contributions (narrative) | `paper/sections/07_ethics.tex:13-15` | Present, but not CRediT |
| IRB statement | `paper/sections/07_ethics.tex:17-19` | Present ("not applicable") |
| Informed consent | `paper/sections/07_ethics.tex:21-23` | Present ("not applicable") |
| Data availability | `paper/sections/07_ethics.tex:25-27` | Present, but vague |
| Code availability | `paper/sections/07_ethics.tex:29-31` | Present, but no commit or DOI |
| Funding | `paper/sections/07_ethics.tex:33-35` | Present |
| Competing interests | `paper/sections/07_ethics.tex:37-39` | Present |

**Missing or incomplete.**

| Statement | Why it matters | Where to add | Effort |
|---|---|---|---|
| ORCID for N. Zhanat | Many journals require an ORCID for the submitting author, and some require it for all authors; the checklist flags it (`paper/submission_checklist.md:14`) | Title page | 5 minutes |
| CRediT author contributions | Many journals in this space (Elsevier, Springer, PLOS) use CRediT; IEEE journals vary; the current statement is a narrative | `07_ethics.tex` | 30 minutes |
| Acknowledgments | The checklist flags it (`paper/submission_checklist.md:21`) | `07_ethics.tex` | 10 minutes |
| Generative-AI declaration | Elsevier and Springer require disclosure of generative AI use in writing; the paper has none | `07_ethics.tex` | 10 minutes |
| Reporting-guideline checklist | The paper cites CLEAR (`02_related_work.tex:5`) but does not use it; a dermatology AI paper should include CLEAR, and a diagnostic accuracy paper should include STARD or TRIPOD+AI | Supplement | 2 to 3 hours |
| Permanent data identifiers | The statement says "ISIC archive" and "CC-BY" but gives no DOI or URL; add the HAM10000 and HIBA DOIs/URLs | `07_ethics.tex` | 30 minutes |
| Permanent code identifier | The GitHub URL is public, but there is no commit hash or Zenodo DOI; add `e444228` and a DOI | `07_ethics.tex` | 1 hour plus Zenodo deposit |
| Split file | The ethics statement says splits are released, but `outputs/splits/ham10000_lesion_seed42.csv` is missing; the split seed 42 is not stated | Repo plus `03_methods.tex` | 30 minutes |
| Supplement | The checklist says "Supplementary material" (`paper/submission_checklist.md:54`); none exists | New file | 1 day |
| Word/figure/table counts, running title | Many journals require these on the title page; none are present | Title page | 15 minutes |
| Line numbers | Many journals request line numbers for review | `main.tex` (`\usepackage{lineno}`) | 5 minutes |
| Prior presentation | If any part was presented at a workshop, state it; if not, state "not presented" | Title page or cover letter | 5 minutes |
| Original dataset ethics approval | HIBA and HAM10000 were collected under their own ethics approvals; the paper says "not applicable" without citing them | `07_ethics.tex` | 30 minutes |
| Data and code licenses | State the license for the code (e.g., MIT) and the data (HAM10000 and HIBA terms) | `07_ethics.tex` or repo | 30 minutes |
| Clinical trial registration | Not applicable, but some clinical journals require an explicit statement | `07_ethics.tex` | 5 minutes |
| Patient/public involvement | Not applicable for public data, but some clinical journals require it | `07_ethics.tex` | 5 minutes |

**The checklist itself is incomplete.** `paper/submission_checklist.md` has 42 unchecked items and 7 checked items. The checked items are the author list, S.B. ORCID, corresponding author, author contributions, funding, competing interests, and ethics. Every scientific, reference, figure/table, file, open-science, and final check is unchecked. The checklist is a useful template, but it has not been used as a gate.

**The cover letter is not final.** `paper/cover_letter.md:3-5,7,9,11,19` has placeholders for the journal name, editor name, editorial office address, and date. It also repeats the overclaims: "The AUC ranking is stable, and focal loss roughly triples expected calibration error, which identifies calibration as the mechanism" (line 13), "false negatives have lower attribution mass than false positives" (line 15), and "The CNN-transformer gap transfers" (line 17). It must be rewritten after the scientific fixes.

### Inferences

- For Medical Image Analysis or Computers in Biology and Medicine, the minimum additions are CRediT, a generative-AI declaration, acknowledgments, a data/code DOI, the split file, and a supplement. The abstract must be cut to the journal limit.
- For a clinical dermatology journal, the paper would also need a structured abstract, key points, a reporting-guideline checklist, and a clinical data-sharing statement. The current paper is not formatted for that.
- The first-author ORCID and CRediT are trivial and should be done immediately. The reporting guideline and supplement are the largest administrative tasks.
- The cover letter should not be sent until the scientific claims are rewritten; otherwise it will advertise the weakest claims.

### Gaps

- The target journal is not chosen, so the exact statement requirements are unknown. The journal metrics note (`research_notes/Melanoma journal selection/journal_metrics_cost.md:37-51,180-196`) gives the limits for 15 candidates, but the final choice is still open.
- The authors' institutional APC coverage and the supervisor's venue expectations are unknown; these affect the journal choice.

---

## 4. Which figures or tables are missing, redundant, or misplaced?

### Takeaway

The paper has 11 tables and 6 figures. That is too many for some targets and too many for the paper's focus. Table 1 (Stage 1 screen), Figures 5 and 6 (training/ROC for one seed), Table 9 (XAI correlations), and Table 10 (HIBA per-type) can move to the supplement. Figure 3C has the wrong caption. The training/ROC figures are 150 dpi, below the 300 dpi checklist. The paper is missing a dataset/split table, a per-seed results table, an external uncertainty table, and a table of the XAI false-negative/false-positive mass values.

### Cited findings

**Current display items.**

| Item | Content | Location | Recommendation |
|---|---|---|---|
| Table 1 | Stage 1 screen, 8 configurations | `paper/tables/stage1.tex`; cited `paper/sections/04_results.tex:9` | Move to supplement; keep a 2 to 3 sentence summary in Methods/Results |
| Table 2 | Stage 2 comparison, 9 configurations | `paper/tables/stage2.tex`; cited `paper/sections/04_results.tex:25` | Keep, but fix the CI and add SDs |
| Table 3 | Paired sensitivity tests | `paper/tables/ranking.tex`; cited `paper/sections/04_results.tex:26` | Keep, but replace the envelope/max p with a proper analysis; add per-seed values |
| Table 4 | Calibration | `paper/tables/calibration.tex`; cited `paper/sections/04_results.tex:42` | Keep, but fix the caption and add CIs |
| Table 5 | Lightness-stratified fairness | `paper/tables/fairness.tex`; cited `paper/sections/04_results.tex:63` | Keep, but add group CIs, not just seed SD |
| Table 6 | Dark-skin cutoff sensitivity | `paper/tables/dark_skin.tex`; cited `paper/sections/04_results.tex:64` | Keep, but add SDs/CIs and the number of runs with a significant difference |
| Table 7 | Adjusted FP/FN models | `paper/tables/adjusted_model.tex`; cited `paper/sections/04_results.tex:65` | Keep, but add the pooled model or remove the pooled-model claims; full models to supplement |
| Table 8 | XAI metrics by tone group | `paper/tables/xai.tex`; cited `paper/sections/04_results.tex:94` | Keep, but fix the n caption, add SDs, add FN/FP mass rows |
| Table 9 | XAI Spearman correlations | `paper/tables/xai_correlations.tex`; cited `paper/sections/04_results.tex:95` | Move to supplement; replace p-values with cluster-bootstrap CIs |
| Table 10 | HIBA by Fitzpatrick type | `paper/tables/external.tex`; cited `paper/sections/04_results.tex:119` | Move to supplement; the per-type estimates are underpowered |
| Table 11 | HIBA by condition | `paper/tables/external_conditions.tex`; cited `paper/sections/04_results.tex:120` | Keep, add SDs/CIs and the strong-augmentation condition if run |
| Figure 1 | Calibration reliability and group ECE | `paper/sections/04_results.tex:44-49` | Keep; add sample size and CIs |
| Figure 2 | Fairness AUC and FPR by tone | `paper/sections/04_results.tex:67-72` | Keep; the y-axis starts at 0.65, which exaggerates differences; consider a full 0 to 1 axis or a note |
| Figure 3 | Dark-skin robustness | `paper/sections/04_results.tex:74-79` | Keep, but fix panel C caption and the truncated y-axis |
| Figure 4 | XAI mass and pointing game | `paper/sections/04_results.tex:97-102` | Keep; add unique-image n, not pooled n |
| Figure 5 | Training curves, one seed | `paper/sections/04_results.tex:126-131` | Move to supplement; 150 dpi |
| Figure 6 | ROC, one seed | `paper/sections/04_results.tex:133-138` | Move to supplement; 150 dpi |

**Specific problems.**

- `paper/sections/04_results.tex:77`: Figure 3C caption says "pooled error model," but the plotted values are the false-positive model. The pooled model (`outcome=error`) is in `results/dark_skin/continuous_model.csv` and is discussed in `paper/sections/04_results.tex:59`. This is a clear caption/figure mismatch.
- `paper/figures/stage2_best_sensitivity_training_curves.png` and `paper/figures/stage2_best_sensitivity_roc_test.png` are 150 dpi. The checklist requires 300 dpi or vector (`paper/submission_checklist.md:44`). Re-render at 300 dpi or move to supplement.
- `paper/tables/xai.tex:4` says "343 test images per seed," but the n column is pooled across seeds (450 light, 450 medium, 129 dark). A reader may think there are 450 unique light images. Fix the caption.
- `paper/tables/xai_correlations.tex:4` does not mention that n=1,027 to 1,029 are image-seed rows, not unique images. The p-values are pseudo-replicated.
- `paper/tables/calibration.tex:4` says "at the validation-selected threshold." ECE and Brier are threshold-independent. Fix.
- `paper/tables/ranking.tex:4` says "95% bootstrap intervals over images." The interval is the envelope of per-seed intervals; the p is the max per-seed p. Fix.
- `paper/tables/stage1.tex:4` and `paper/tables/stage2.tex:4` say "mean and standard deviation across three seeds," but only validation sensitivity has an SD. Add SDs for the other metrics or change the caption.
- `paper/tables/external.tex:4` and `paper/tables/external_conditions.tex:4` say "means across three seeds" but show no SDs. The CSV has them; the ViT CE SD is 0.129.
- `paper/sections/04_results.tex:122-138`: the "Figures" subsection is not a scientific subsection. It only describes Figures 5 and 6. Move them to supplement or integrate them into the relevant results, and remove the subsection.

**Missing display items.**

- A dataset/split table with images, unique lesions, melanomas, and L* group counts for train, validation, and test. The paper gives only the test counts in the text (`paper/sections/03_methods.tex:7`; `paper/sections/04_results.tex:53`).
- A per-seed results table for the main comparisons. The paper reports means and intervals; the per-seed values are in `results/ranking/ranking_tests.csv` and `results/rq1_stage2_runs.csv` but not in the paper. Given the seed-level uncertainty, this is essential.
- An external uncertainty table with per-seed sensitivity, specificity, AUC, and SD/CIs. The current table 11 has only means.
- A table of the XAI FN/FP mass values with unique FP/FN counts and CIs. The values are only in the text (`paper/sections/04_results.tex:90`).
- A full adjusted-model table with all covariates (age, sex, site, log area) or a supplement. The current table 7 shows only L* and contrast.
- A table or figure of the L* and contrast distributions by tone group. The paper discusses contrast as a key confounder but does not show its distribution.
- A calibration score-distribution figure. The ECE values are high and the threshold is near zero; a score histogram by tone group would explain the threshold instability.
- A CLEAR or TRIPOD+AI checklist as a supplement.
- The split file and per-sample predictions as supplementary data.

### Inferences

- Moving Table 1, Figures 5-6, Table 9, and Table 10 to supplement leaves 8 tables and 4 figures in the main text. That fits most journals, including Frontiers' 15 display-item limit, and focuses the paper on the main RQ1, calibration, fairness, and external validation.
- The main text currently has 11 tables for a 6,600-word paper; that is table-heavy. A reviewer may find the paper hard to read. The supplement can carry the Stage 1 screen and the per-type external estimates without losing content.
- The missing per-seed table is the most important addition. It would make the seed-level uncertainty visible and prevent the overclaims.

### Gaps

- No supplement exists. The authors need to create one before submitting to any journal that requires supplementary material.
- No dataset/split table. The split counts are only in the text, and the split file is missing.
- No score-distribution figure. The calibration story would be clearer with one.

---

## 5. What remains for the clinical Fitzpatrick V-VI evidence, the strong-augmentation external validation, the second attribution method, and the extra seeds?

### Takeaway

The strong-augmentation external validation is a one-hour win: the checkpoints, HIBA images, and evaluation script all exist. The second attribution method and extra seeds are optional 1 to 2 day experiments; the second method is more valuable for the XAI claim than more seeds. The clinical Fitzpatrick V-VI evidence is the main strategic gap and cannot be filled with the available public dermoscopic data: Pipsqueak has only 16 core images, and DDI/PASSION are clinical and cross-modality. A strong FST V-VI claim is not feasible today; the paper must either reframe or target a methods journal where the gap is a limitation, not a fatal flaw.

### Cited findings

**Clinical Fitzpatrick V-VI evidence.**

- The paper has no clinical dark-skin data. `paper/sections/03_methods.tex:59` says a clinical dataset "is left for future work, with the imaging modality difference stated as a limitation."
- HIBA has Fitzpatrick I-IV only: 97 type I, 804 type II, 246 type III, 25 type IV, 108 unlabeled in the dermoscopic subset. Type IV has 0 melanomas. There are no type V-VI images (`results/external/hiba/external_summary.csv`; `paper/tables/external.tex:11-27`; `paper/sections/04_results.tex:111`).
- PROVe-AI has the same Fitzpatrick I-IV limitation (`paper/sections/02_related_work.tex:19`). It is not a solution to the V-VI gap.
- Pipsqueak is a public dermoscopic dataset of melanocytic lesions in darker skin. The ScienceDirect article says it comprises "16 core images expanded to 80 through controlled" augmentation, is open access, and the raw data are available at `https://github.com/TommasoRuga/Pipsqueak` ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1877750326000761)). The paper cites Pipsqueak (`paper/sections/02_related_work.tex:17`; `references.bib` key `ruga2026pipsqueak`) but does not discuss its size or why it was not used. Sixteen core images are too few for a stable sensitivity estimate, especially with few melanomas. The paper's "new dermoscopic data collection" (`paper/sections/05_discussion.tex:33`) is accurate, but it should cite Pipsqueak's 16 images to make the data bottleneck concrete.
- DDI is biopsy-proven clinical images with Fitzpatrick labels; PASSION is clinical images from Sub-Saharan Africa with Fitzpatrick III-VI. Both differ in imaging modality from HAM10000 dermoscopy. A cross-modality evaluation would confound skin tone with domain shift; the paper already says this (`paper/sections/05_discussion.tex:33`). Adding DDI/PASSION would not produce a clean dermoscopic FST V-VI estimate.
- A strong FST V-VI claim requires new dermoscopic data collection or a carefully controlled multi-dataset cross-modality study. Neither is a one-day task. The paper's current framing as an image-based lightness study is the honest choice for a methods journal. For a fairness-focused journal, the lack of FST V-VI data is a likely rejection reason.

**Strong-augmentation external validation.**

- The 9 strong-augmentation checkpoints exist: `outputs/kaggle/rq1-stage-2/checkpoints/melanoma-xai-bias/checkpoints/stage2_*_ce_random_strong_seed{1,2,3}.pt` (27 checkpoints total, including CE and focal).
- The HIBA images and metadata exist: `data/external/hiba/images/`, `data/external/hiba/hiba-skin-lesions.csv`. The dermoscopic subset is defined by `image_type == "dermoscopic"` and contains 1,280 images.
- `scripts/evaluate_external.py` defaults to `--conditions ce_random,focal2_cb,ce_random_strong`, so the strong condition is supported. The paper says it is not evaluated (`paper/README.md:69`; `paper/sections/06_limitations.tex:5`).
- Running the strong condition would add a third row per architecture to Table 11 and complete the 3 x 3 external design. It is a one-hour inference job on the existing checkpoints, plus a table update.
- The strong regime is described as "designed to simulate domain shift" (`paper/sections/03_methods.tex:11`). If that rationale is correct, strong augmentation should improve HIBA sensitivity. The external strong evaluation would test the rationale directly. This is a strong scientific reason to run it, not just a completeness check.

**Second attribution method.**

- The paper uses Grad-CAM only. Integrated Gradients is cited (`paper/sections/01_introduction.tex:9`; `paper/sections/02_related_work.tex:31`) but not used. The limitations say "A second attribution method is not included" (`paper/sections/06_limitations.tex:13`).
- The checkpoints and images are available; the repo already uses Captum for Grad-CAM (`src/melanoma/xai/gradcam.py`). Integrated Gradients can be added with Captum's `IntegratedGradients` and the same 343-image sample.
- Cost: 3,087 heatmaps (343 x 3 seeds x 3 architectures), plus the baseline choice and the metric computation. One to two days. The result would test whether the mass-tone and mass-error patterns are method-specific.
- A cheaper alternative is a model-parameter randomization sanity check (Adebayo et al., cited at `paper/sections/02_related_work.tex:33` but not used). Randomize the model weights, recompute heatmaps on a subset, and show that the similarity to the original heatmap drops. This is a few hours and addresses the standard "heatmaps are insensitive to the model" criticism. It is not a second method, but it is a useful robustness check.

**Extra seeds.**

- The paper uses three seeds per condition. The conclusion says the next steps include "two additional seeds for the six focal and CE cells" (`paper/sections/08_conclusion.tex:9`). That would be 12 runs (3 architectures x 2 conditions x 2 seeds), or 18 runs if all three conditions are balanced.
- The Stage 1 config says 24 runs take about 3 to 4 Kaggle GPU-hours (`configs/sweep_stage1.yaml:2`), so 18 Stage 2 runs are roughly 3 hours of training. The downstream analyses (predictions, fairness, calibration, ranking, XAI) would need to be rerun for the new seeds. The XAI rerun for 2 extra seeds adds 2,058 heatmaps.
- The benefit: tighter seed-level intervals for the ResNet-50 versus ViT-S/16 interaction and the external ViT estimate. The cost: redoing all tables and the paper. For a near-term submission, fixing the statistical aggregation is more valuable than adding seeds. For a revision, extra seeds would strengthen the paper.
- If the authors add seeds, they should add them to all three conditions, not only focal and CE, to keep the design balanced. The current plan would leave the strong condition at 3 seeds.

### Inferences

- Run the strong-augmentation external validation now. It is the cheapest, most visible gap, and it directly tests the "strong augmentation simulates domain shift" rationale.
- Add Integrated Gradients only if the target journal is XAI-focused or if the authors want the XAI section to be a co-equal contribution. Otherwise, add the model-parameter randomization sanity check and keep Grad-CAM-only with a clear limitation.
- Do not add seeds until the statistical aggregation is fixed. More seeds with the current aggregation would produce narrower-looking intervals that are still not valid. Fix the inference first, then add seeds for a revision if needed.
- The clinical FST V-VI gap is a strategic decision, not a quick experiment. The paper should either (a) reframe as a benchmark/calibration study with a secondary image-based lightness analysis, or (b) add a cross-modality DDI/PASSION analysis with explicit caveats, or (c) accept that a fairness-focused reviewer will reject the paper for lack of FST V-VI data. The target journal choice should reflect this.

### Gaps

- No public dermoscopic dataset with enough FST V-VI melanoma cases to support a stable estimate. Pipsqueak's 16 core images are too few. This is a genuine data gap, not an oversight.
- No strong-augmentation external results, despite the checkpoints and data being available. This is an avoidable gap.
- No second attribution method or sanity check. This is a defensible limitation for a benchmark paper, but a weakness for an XAI paper.
- No extra seeds. This is a precision limitation, not a validity limitation, if the seed-level uncertainty is reported properly.

---

## 6. What is blocking submission today, what can be done in one day, and what is optional?

### Takeaway

The paper is not submission-ready today. The blocking issues are scientific (invalid intervals, overclaimed AUC/calibration/external/XAI/fairness) and administrative (target journal, abstract length, cover letter, supplement, statements). Most scientific issues are fixable in 1 to 2 days with the existing predictions; the clinical FST V-VI gap is a longer-term strategic decision. The paper should not be submitted until the statistical claims are corrected, because a reviewer will find the interval and p-value problems in the public code.

### Must fix before submission (blocking)

1. **Choose the target journal and reformat.** The abstract is 323 words. It exceeds the 250-word limit at Computers in Biology and Medicine, Computer Methods and Programs in Biomedicine, and Artificial Intelligence in Medicine, and the 200-word limit at MDPI Diagnostics/Cancers/Bioengineering. It fits Frontiers' 350-word limit. IEEE JBHI's 8-page limit is a poor fit for 11 tables and 6 figures. Choose Medical Image Analysis or Computers in Biology and Medicine first, then cut the abstract to the journal limit. `research_notes/Melanoma journal selection/journal_metrics_cost.md:37-51,180-196`.
2. **Replace the invalid 95% CIs.** `scripts/collect_results.py:100-101` averages per-seed CI bounds; `scripts/analyze_ranking.py:142-146` uses the envelope and max p. Use a hierarchical bootstrap (resample lesions/images and seeds), a mixed-effects model, or report per-seed values with seed-level t intervals. Update Tables 1, 2, 3, 5, 6, 8, 10, and 11, and every text sentence that cites a "95% CI."
3. **Test the AUC claims.** Add paired AUC difference and AUC interaction tests for the architecture and condition comparisons. The current "AUC ranking is stable" claim is contradicted by the ViT's +0.018 AUC and the narrowing gap. Rewrite the Results subsection, the abstract, the Discussion, and the Conclusion.
4. **Rewrite the calibration claim.** "Roughly triples" is false for the ViT. "Calibration identifies the mechanism" is not supported because the ViT's AUC improves. Say "calibration degradation tracks the CNN sensitivity loss; the ViT gain is not explained by calibration." Add a temperature-scaling or isotonic recalibration experiment if possible.
5. **Rewrite the external validation claim.** Add per-seed SDs/CIs to Tables 10 and 11. The ViT CE sensitivity is 0.082, 0.245, 0.337 and the focal is 0.301, 0.372, 0.270; the +9.4 point mean is not consistent. Say "directionally consistent for EfficientNet-B0; mixed for ResNet-50 and ViT-S/16; the external evidence is suggestive, not confirmatory." Remove "reproduces the interaction" from the abstract and conclusion.
6. **Fix the XAI target class or reframe the error-type comparison.** `scripts/analyze_xai.py:244` uses the predicted class. Rerun with the true class, or relabel the analysis as predicted-class attribution and remove the causal FN/FP interpretation. The FN/FP mass difference is significant only for EfficientNet-B0; the ViT and ResNet cluster-bootstrap CIs include zero. The dark-light mass difference is significant for EfficientNet-B0 and ViT-S/16, not ResNet-50.
7. **Fix the fairness overclaims.** The dark AUC deficit is not significant at any cutoff (`results/dark_skin/auc_differences.csv`). The FP L* association is significant only for ResNet-50; the FN L* association is significant for all three. The FN contrast association is significant only for ViT. Rewrite the abstract's symmetric "darker skin -> more FP, lighter skin -> more FN" and "contrast moves both error types" claims.
8. **Fix the false "seven-class" limitation.** `paper/sections/06_limitations.tex:6` and `RESULTS_MEMO.md:145` claim a secondary seven-class experiment that does not exist. Remove it or run it.
9. **Fix the "pre-declared" and "test set touched once" wording.** The strong-augmentation condition is the Stage 1 validation winner; the test set is reused for descriptive analyses. Say this explicitly. Add the L* cutoff rationale and timing.
10. **Fix the split-release claim.** Commit `outputs/splits/ham10000_lesion_seed42.csv` or change `paper/sections/07_ethics.tex:11` to "split construction code and seed 42 are released." State the split seed in the Methods.
11. **Fix the figure and table captions.** Figure 3C is the false-positive model, not the pooled model. Table 4's "at the validation-selected threshold" is wrong for ECE. Table 3's "95% bootstrap intervals" is wrong. Table 8's "343 per seed" is wrong. Tables 1/2's "mean and SD" is inaccurate. Tables 10/11 lack SDs. Table 9 lacks the pseudo-replication caveat.
12. **Add the missing statements.** First-author ORCID, CRediT, acknowledgments, generative-AI declaration, reporting-guideline checklist (CLEAR/TRIPOD+AI/STARD), data/code DOIs, commit hash, data/code licenses, word/figure/table counts, running title, and line numbers. `paper/submission_checklist.md:14,21,25-33,37-40,44-48,52-57,61-65,69-74`.
13. **Finalize the cover letter.** Remove all placeholders (`paper/cover_letter.md:3-11,19`) and rewrite the claims to match the corrected abstract.
14. **Re-render or move the 150 dpi figures.** `paper/figures/stage2_best_sensitivity_training_curves.png` and `paper/figures/stage2_best_sensitivity_roc_test.png` are 150 dpi; the checklist requires 300 dpi or vector (`paper/submission_checklist.md:44`). Move them to supplement or re-render at 300 dpi.

### One day (scientific quick wins)

1. Recompute the paired tests with a hierarchical bootstrap or seed-level t intervals; update Table 3 and the claims. The per-seed data is in `results/ranking/ranking_tests.csv`.
2. Add AUC difference and AUC interaction tests; update the "AUC stable" claims. The per-sample predictions are in `results/predictions/`.
3. Add per-seed SDs/CIs to Tables 10 and 11 from `results/external/hiba/external_metrics.csv`.
4. Run the strong-augmentation external validation with `scripts/evaluate_external.py --conditions ce_random_strong`; add the third condition to Table 11.
5. Add a matched-specificity in-domain analysis: compute sensitivity at exactly 95% test specificity for all 9 configurations, as a secondary analysis. This removes the achieved-specificity differences and tests whether the sensitivity reversal persists at a common operating point.
6. Add group-level FPR/AUC confidence intervals and the number of runs with a significant dark-stratum difference. The data is in `results/dark_skin/threshold_sensitivity.csv` and `results/dark_skin/auc_differences.csv`.
7. Fix the figure/table captions and the "seven-class," "pre-declared," "test set touched once," and "split released" sentences.
8. Cut the abstract to 250 words (or 200 for MDPI) and rewrite the cover letter.
9. Add the split seed, HIBA subset definition, data/code DOIs, and the missing statements.
10. Build the supplement skeleton: Stage 1 table, training/ROC figures, per-seed metrics, full adjusted models, CLEAR checklist, split file, per-sample predictions.

### One to two days (optional, high value)

1. Temperature-scaling recalibration on the validation predictions, then re-select the threshold at 95% specificity and recompute test sensitivity. This is the direct test of the calibration mechanism. All 27 runs have `predictions_val.csv`; the checkpoints are available. If the focal sensitivity at 95% specificity moves toward the CE baseline after recalibration, the mechanism is supported. If not, the calibration claim should be dropped.
2. XAI true-class rerun: compute Grad-CAM with the true class as the target for all images, compare the FN/FP mass, and add a model-parameter randomization sanity check. This is the most important XAI validity fix.
3. Integrated Gradients as a second attribution method. One to two days, depending on the implementation.
4. Extra seeds for all three conditions (18 runs) if the authors can rerun the downstream analyses. The training is about 3 hours; the reanalysis is the larger cost.
5. A mixed-effects or one-seed sensitivity analysis for the adjusted FP/FN models. The current models are fit on 558 repeated FN rows for 186 melanomas; the cluster bootstrap handles the SEs but not the point-estimate bias from repeated rows.

### Optional (not blocking for a methods journal)

1. DDI/PASSION cross-modality analysis. It would not produce a clean FST V-VI dermoscopic estimate, and the domain shift may dominate. It is a strategic addition, not a quick fix.
2. A clinician reader study. It would strengthen the clinical relevance of the XAI claims, but it is a major new study.
3. PROVe-AI external validation. It has the same Fitzpatrick I-IV limitation and may not add much beyond HIBA.
4. A seven-class experiment, if the authors want to keep the "secondary experiment" claim.
5. A Zenodo archive with the 27 checkpoints and the split file. It would improve reproducibility, especially for the XAI analysis.
6. A graphical abstract, highlights, or a web demo.
7. A Bayesian analysis or equivalence testing for the null within-architecture effects.

### Inferences

- A realistic 2-day sprint can fix the core scientific claims and the main administrative gaps. The temperature-scaling and true-class XAI rerun are the two analyses most likely to change the paper's conclusions; they may take a second day.
- The paper should not be submitted before the interval and p-value issues are fixed. The public code makes the current aggregation easy to audit; a reviewer who checks `collect_results.py` or `analyze_ranking.py` will find the problem.
- The target journal decision should come first. It determines the abstract length, the format, the supplement, and whether the clinical FST V-VI gap is fatal.

### Gaps

- The authors' compute budget, data access, and supervisor's venue expectations are unknown. The timeline assumes the existing checkpoints and data are available, which they are locally.
- The clinical FST V-VI gap has no quick fix. The paper's target journal choice is the main lever.

---

## 7. Is the paper too long or too short for a typical journal, and which sections could be moved to supplementary material?

### Takeaway

The paper is not too short. It is comprehensive, at roughly 6,600 to 7,000 words of prose, 25 pages in the current single-column preprint format, 11 tables, 6 figures, 40 references, and a 323-word abstract. It fits Medical Image Analysis, Computers in Biology and Medicine, Computer Methods and Programs in Biomedicine, Artificial Intelligence in Medicine, Scientific Reports, and PLOS Digital Health in length. It is too long for IEEE JBHI's 8-page limit, and its 17 display items exceed Frontiers' 15-item limit. The abstract is too long for the 250-word limit at CIBM, CMPB, and AIM and the 200-word limit at MDPI; it fits Frontiers' 350. The paper is a better fit for a methods journal than a clinical journal.

### Cited findings

**Length.**

- Section word counts (my regex excludes itemized lists, so the Limitations count is understated): Introduction 306, Related Work 915, Methods 1,170, Results 2,164, Discussion 1,410, Ethics 323, Conclusion 354. The Limitations itemize is about 350 words. Total prose is roughly 6,600 to 7,000 words.
- PDF: 25 pages (`paper/main.pdf`); 11 tables, 6 figures, 40 references, 323-word abstract.
- The main text is table-heavy: 11 tables for 6,600 words. A typical melanoma XAI paper has 6 to 10 figures and 50 to 80 references; this paper has 6 figures and 40 references. The reference count is on the low side for a methods journal, but not a blocker.
- The paper's prose is repetitive: the abstract, Introduction contributions (`paper/sections/01_introduction.tex:17-24`), Results, Discussion, and Conclusion repeat the same six claims and many of the same numbers. Cutting the repetition by 15 to 20% would bring the paper to about 5,500 words without losing content.

**Journal limits (from `research_notes/Melanoma journal selection/journal_metrics_cost.md`).**

| Journal | Limit | Fit for this paper |
|---|---|---|
| Medical Image Analysis | No page/word limit found; abstract limit not verified | Good fit; 25 pages is acceptable; high novelty bar |
| IEEE JBHI | 8 pages regular; USD 250/page for pages 9-10, USD 350/page from 11 | Poor fit; 11 tables + 6 figures will exceed 8 pages and incur charges |
| Computers in Biology and Medicine | Abstract 250 words; no word limit found | Good fit; cut abstract by 73 words |
| Computer Methods and Programs in Biomedicine | Abstract 250 words; no word limit found | Good fit; cut abstract by 73 words |
| Artificial Intelligence in Medicine | Abstract 250 words; no word limit found | Good fit, but 303-day median to acceptance is slow |
| Scientific Reports | No strict limit found; abstract limit not verified (likely 200) | Acceptable fit; check abstract limit |
| PLOS Digital Health | No limit found | Acceptable, but the paper's clinical validation is weak for a digital health journal |
| Frontiers in Medicine/Oncology | 12,000 words, 350-word abstract, 15 figures/tables | Length and abstract fit, but 17 display items exceed 15; move 2+ to supplement |
| Diagnostics/Cancers/Bioengineering (MDPI) | Abstract 200 words; no length limit | Length fits, but cut abstract by 123 words |
| npj Digital Medicine | No limit found | Stretch; clinical validation is weak for this venue |
| BMC Medical Imaging | No strict limit found; abstract limit not verified | Acceptable fallback |

**What to move to supplementary material.**

- Table 1 (Stage 1 screen): the Stage 1 screen is a pre-experiment to choose the strong-augmentation condition. It is not central to the main claims. Move the table and the 8-configuration detail to supplement; keep a 2 to 3 sentence summary in Methods/Results.
- Figures 5 and 6 (training curves and ROC for one seed): illustrative, not evidential. Move to supplement.
- Table 9 (XAI correlations): the key numbers can be in the text; the full correlation table can be supplementary, especially after replacing p-values with cluster-bootstrap CIs.
- Table 10 (HIBA by Fitzpatrick type): the per-type estimates are underpowered and not central to the main external interaction claim. Move to supplement; keep the All row and the condition comparison in the main text.
- The full adjusted models (all covariates) can be supplementary; keep the L* and contrast ORs in the main text.
- The per-seed metrics for all tables can be supplementary; keep the means and CIs in the main text.
- The CLEAR/TRIPOD+AI checklist, split file, and per-sample predictions can be supplementary.

**What to keep in the main text.**

- Table 2 (Stage 2 comparison), Table 3 (paired tests), Table 4 (calibration), Table 5 (fairness), Table 6 (cutoff sensitivity), Table 7 (adjusted models), Table 8 (XAI), Table 11 (external conditions).
- Figures 1 to 4 (calibration, fairness, dark-skin robustness, XAI).
- The main RQ1, calibration, fairness, and external narrative.

### Inferences

- Moving Table 1, Figures 5-6, Table 9, and Table 10 to supplement leaves 8 tables and 4 figures in the main text. That fits Frontiers' 15-item limit and most other journals, and it focuses the paper on the claims that matter.
- The paper is a better fit for Medical Image Analysis or Computers in Biology and Medicine than for a clinical dermatology journal. The clinical validation is too thin for JAMA Dermatology or the British Journal of Dermatology, and the paper is not formatted for a clinical journal.
- The abstract must be cut to 250 words for Elsevier and 200 for MDPI. A 250-word version should keep: the RQ1 architecture-dependent effect, the calibration nuance (CNN loss vs ViT gain), the fairness error-type finding (FN robust, FP qualified), and the external caveat (suggestive, not confirmatory). It should drop the detailed XAI FN/FP numbers and the cutoff-specific sensitivity numbers.
- The paper could be split into two papers: (1) benchmark, calibration, and external validation; (2) fairness and XAI. The combined paper is stronger for a master's thesis, but a journal reviewer may find it too broad. The PAPER_PLAN open decision #2 (`PAPER_PLAN.md:92-94`) is still unresolved. If the target journal's scope is narrow, splitting is a serious option.
- The 40-reference list is on the low side for Medical Image Analysis or Computers in Biology and Medicine. Adding 10 to 15 recent melanoma AI, foundation-model, and fairness references would improve the related work and the reviewer's impression. This is a nice-to-have, not a blocker.

### Gaps

- Medical Image Analysis's abstract limit was not verified; check the guide for authors.
- Scientific Reports' abstract limit was not verified; likely 200 words.
- The two-column page count after reformatting is unknown; the 25-page preprint count is not the journal count.
- The authors' APC budget and institutional agreements are unknown; the journal metrics note gives the list prices but not the authors' coverage.

---

## 8. Prioritized action list

### Must fix before submission

1. Choose the target journal; reformat; cut the abstract to the journal limit.
2. Replace the invalid 95% CIs and p-values with a hierarchical or seed-level analysis; update Tables 1, 2, 3, 5, 6, 8, 10, 11 and all text claims.
3. Add AUC difference and AUC interaction tests; rewrite the "AUC ranking is stable" claim.
4. Rewrite the calibration claim: "roughly triples" is false for the ViT; calibration does not explain the ViT gain. Add a temperature-scaling test if possible.
5. Rewrite the external validation claim: add per-seed SDs/CIs; the ViT change is not consistent across seeds; remove "reproduces the interaction."
6. Fix the XAI target class or reframe the FN/FP comparison; the FN/FP difference is significant only for EfficientNet-B0, and the dark-light mass difference is not significant for ResNet-50.
7. Fix the fairness overclaims: the AUC deficit is not significant at any cutoff; the FP L* association is significant only for ResNet-50; the FN contrast association is significant only for ViT.
8. Fix the "seven-class," "pre-declared," "test set touched once," and "split released" sentences.
9. Fix Figure 3C and the table captions; re-render or move the 150 dpi figures.
10. Add the missing statements: first-author ORCID, CRediT, acknowledgments, generative-AI declaration, reporting-guideline checklist, data/code DOIs, commit hash, split file, supplement, word/figure/table counts, running title, line numbers.
11. Finalize the cover letter; remove all placeholders and align the claims with the corrected abstract.

### Should fix (one day, high value)

1. Strong-augmentation external validation (existing checkpoints; one hour).
2. Matched-specificity in-domain sensitivity analysis (existing predictions; one hour).
3. Group-level FPR/AUC confidence intervals and per-run consistency counts.
4. Per-seed values in the main tables.
5. Temperature-scaling recalibration on validation (half day).
6. XAI true-class rerun or a clear predicted-class caveat (half day).
7. Supplement with full models, per-seed metrics, CLEAR checklist, split file, and per-sample predictions.

### Nice to have

1. Integrated Gradients as a second attribution method.
2. Extra seeds for all three conditions.
3. DDI/PASSION cross-modality analysis.
4. Clinician reader study.
5. PROVe-AI external validation.
6. Seven-class experiment, if the authors want to keep the claim.
7. Zenodo archive with the 27 checkpoints.
8. Graphical abstract or highlights.
9. Bayesian analysis or equivalence testing.
10. 10 to 15 additional references for a methods journal.

---

## 9. Honest reviewer-risk summary

A reviewer at Medical Image Analysis or Computers in Biology and Medicine will likely raise the following, in rough order of severity:

1. **The intervals are not confidence intervals.** The code averages per-seed CI bounds and takes the max per-seed p-value. This is a fundamental statistical error, and the public code makes it easy to find. It affects every headline claim. This alone could trigger a major revision or rejection.
2. **"AUC stable" is contradicted by the AUC changes.** The ViT gains AUC, the CNNs lose AUC, and the ViT-CNN gap closes to non-significance. The paper's central contrast between "stable AUC" and "unstable sensitivity" is not supported. The authors need to test the AUC differences and rewrite the mechanism.
3. **The external validation does not reproduce the interaction.** The ViT external sensitivity is 0.082, 0.245, 0.337 across seeds; the +9.4 point mean is not consistent. The table has no SDs. A reviewer will ask for the per-seed values and will not accept "reproduces."
4. **The XAI error-type comparison is confounded.** Using the predicted class as the Grad-CAM target means the false-negative heatmaps explain the benign prediction, not the missed melanoma. The FN/FP mass difference is significant only for EfficientNet-B0. A reviewer with XAI expertise will catch this.
5. **The fairness claims are too strong.** The dark AUC deficit is not significant at any cutoff; the FP L* association is significant only for ResNet-50; the FN contrast association is significant only for ViT. The abstract's symmetric story is not supported. The FN L* association is the robust finding and should be the lead.
6. **The paper is broad and somewhat disconnected.** The XAI analysis is on CE-baseline models only and is not connected to the focal intervention. The fairness and XAI sections read as separate studies. A reviewer may say the paper is two papers stapled together. The authors should either connect the XAI to the focal condition or narrow the claims.
7. **The clinical FST V-VI gap is a fundamental limitation.** The paper's title includes "fairness," but the darkest group is 43 images and 6 melanomas, and there are no FST V-VI images. A fairness-focused reviewer will say the paper cannot support a clinical fairness claim. The paper is honest about this, but the title and abstract may overpromise.
8. **The XAI analysis is one method, one condition, and unadjusted for lesion size/contrast.** A reviewer at an XAI venue will ask for a second method, a true-class analysis, and a confound-adjusted comparison. The current limitations mention the second method but not the target-class confound or the unadjusted comparison.
9. **The "seven-class secondary experiment" is false.** A reviewer who checks the results will find no seven-class analysis. This is a credibility issue, even if small.
10. **The "pre-declared" conditions and the "test set touched once" wording are misleading.** The strong condition is the Stage 1 winner, and the test set is reused for descriptive analyses. The authors should disclose this precisely.
11. **The split file and some data/code statements are not reproducible as written.** The split file is missing, the commit hash is not in the paper, and the data/code DOIs are absent. A reproducibility-focused reviewer will check.
12. **The abstract is too long for the likely targets.** At 323 words, it exceeds the 250-word limit at CIBM, CMPB, and AIM and the 200-word limit at MDPI. The authors need to choose the journal and cut accordingly.
13. **The reference list is on the low side for a methods journal.** Forty references is acceptable but thin for Medical Image Analysis or Computers in Biology and Medicine. Add recent melanoma AI/foundation-model/fairness work.
14. **The training/ROC figures are 150 dpi.** This is a production issue, not a scientific one, but the checklist requires 300 dpi or vector. Move them to supplement or re-render.

**What a reviewer will like.** The controlled design, the lesion-aware split, the frozen threshold, the per-sample predictions, the public code, the honest limitations, the error-type-specific fairness framing, and the quantitative XAI metrics. The paper is a solid empirical study with a fixable inference layer. The authors should not lose the strengths while fixing the claims.

---

## 10. Suggested immediate sequence

**Day 1 (scientific core).**
1. Recompute the paired tests with a hierarchical bootstrap or seed-level t intervals; update Table 3 and the abstract's "each CNN" claim.
2. Add AUC difference/interaction tests; update the "AUC stable" claims.
3. Add per-seed SDs/CIs to Tables 10 and 11; rewrite the external claim.
4. Run the strong-augmentation external validation; add the third condition to Table 11.
5. Add the matched-specificity in-domain analysis.
6. Fix the Figure 3C caption, Table 3/4/8/9/10/11 captions, and the "seven-class," "pre-declared," "test set touched once," and "split released" sentences.

**Day 2 (claims and administration).**
1. Cut the abstract to 250 words (or 200 for MDPI); rewrite the cover letter.
2. Rewrite the Introduction contributions, Discussion, and Conclusion to match the corrected claims.
3. Add the missing statements: ORCID, CRediT, acknowledgments, generative-AI declaration, reporting guideline, data/code DOIs, commit hash, split seed, word/figure/table counts, line numbers.
4. Build the supplement skeleton: Stage 1 table, training/ROC figures, per-seed metrics, full adjusted models, CLEAR checklist, split file, per-sample predictions.
5. Re-render or move the 150 dpi figures.
6. Compile and check for unresolved references and overfull boxes.

**Day 3 to 5 (optional, high value).**
1. Temperature-scaling recalibration on the validation predictions; recompute sensitivity at 95% specificity; test whether the reversal attenuates.
2. XAI true-class rerun; compare FN/FP mass; add a model-parameter randomization sanity check.
3. If the target is XAI-focused, add Integrated Gradients.
4. If the authors can rerun all analyses, add two seeds to all three conditions.

**Before submission.**
1. Choose the target journal and download its class/style.
2. Reformat the manuscript and the references.
3. Check the abstract, word, figure, and table limits.
4. Verify every number in the text against the CSVs.
5. Verify every reference resolves with a DOI or arXiv ID.
6. Check the cover letter for placeholders.
7. Check the supplement and data/code availability.
8. Save the submission confirmation.

---

## Appendix: audit checks the authors can rerun

These are the independent checks I used; they are read-only and can be reproduced from the saved predictions.

**Seed-level CIs for the main tables.** Use `results/rq1_stage2_runs.csv` (per-seed `sensitivity`) and a two-sided t interval on 2 degrees of freedom. The reported intervals in `paper/tables/stage2.tex` are the mean of per-seed CI bounds from `scripts/collect_results.py:100-101`.

**Paired-test seed-level CIs.** Use `results/ranking/ranking_tests.csv`. The per-seed deltas and the seed-level t intervals are in the table in Section 1. The reported intervals are the `ci_low_min`/`ci_high_max` envelope from `scripts/analyze_ranking.py:142-146`.

**AUC difference bootstrap.** Use the per-sample predictions in `results/predictions/`. Merge the two runs on `image_id`, resample test images with replacement, compute `roc_auc_score` for each condition, and take the difference. The key results are in Section 1, item 3. A 2,000-resample bootstrap is sufficient for the audit; the paper should use 10,000 for the final analysis.

**Fairness AUC differences.** `results/dark_skin/auc_differences.csv` already has the per-run dark-versus-light and dark-versus-medium AUC differences with 95% CIs. At cutoff 55, every CI includes zero.

**Fairness FPR tests.** `results/dark_skin/threshold_sensitivity.csv` has the per-run `n`, `n_positive`, and `fpr`. The number of false positives is `fpr * (n - n_positive)`. A Fisher exact test of dark versus light at cutoff 55 gives the p-values in Section 1, item 8. The paper should use a cluster bootstrap over images for the final analysis, not a Fisher exact test, because the same images are reused across seeds.

**XAI cluster bootstrap.** `results/xai/xai_all_metrics.csv` has per-image, per-seed `attribution_mass`. Average the mass per unique `image_id` across seeds, then bootstrap over unique images within the FP and FN groups, or within the dark and light groups. The key results are in Section 1, items 12 and 13.

**Strong external validation.** Run:
`python scripts/evaluate_external.py --conditions ce_random_strong --device mps --merge --out-dir results/external/hiba`
with the existing checkpoints under `outputs/kaggle/rq1-stage-2/checkpoints/melanoma-xai-bias/checkpoints/` and the HIBA images under `data/external/hiba/images/`. Add the third condition to `paper/tables/external_conditions.tex`.

**Temperature scaling.** Use `predictions_val.csv` from each run to fit a temperature `T` by minimizing negative log-likelihood on the validation probabilities, apply `T` to the test probabilities, re-select the threshold at 95% specificity on the recalibrated validation probabilities, and recompute test sensitivity/ECE. This is the direct test of the calibration mechanism. The validation predictions exist for all 27 runs under `outputs/kaggle/rq1-stage-2/melanoma-xai-bias/extracted/outputs/runs/*/predictions_val.csv`.

**Split file.** The split seed is 42 (`configs/pilot.yaml:13`), and the split file path is `outputs/splits/ham10000_lesion_seed42.csv` (`configs/pilot.yaml:15`). The file is missing; regenerate it with the split code in `src/melanoma/data/splits.py` and commit it, or change the ethics statement to say the split construction code and seed are released.
