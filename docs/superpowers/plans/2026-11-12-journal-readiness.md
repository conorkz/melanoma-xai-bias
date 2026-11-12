# Journal Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current workshop-level draft into a journal-ready manuscript with stronger dark-skin evidence, calibration analysis, formal ranking tests, and external validation of all three conditions.

**Architecture:** Add four analysis modules under `src/melanoma/` with focused scripts and tests. Reuse the 27 frozen Stage 2 runs and their per-image test predictions, the 10,015 HAM10000 lesion masks, and the 9 downloaded checkpoints. Update the paper from generated CSVs only, never hand-typed numbers.

**Tech Stack:** Python 3.12, pandas, numpy, scipy, scikit-learn, OpenCV, matplotlib, pytest, LaTeX with tectonic.

**Spec:** The journal-readiness assessment from the 12 November 2026 review, plus the existing `PAPER_PLAN.md` and `results/` artifacts.

## Global Constraints

- No em dashes or en dashes in any writing.
- Every number in the paper comes from a result CSV.
- The test split is never used for model selection.
- The 27 Stage 2 runs and their checkpoints are frozen. Do not retrain.
- Dark-skin groups are image-based lightness strata, not clinical Fitzpatrick labels.
- Report mean and standard deviation across three seeds.
- Keep all new analysis code tested and reproducible from a single command.

## Review Focus

- A dark group defined at L* <= 55 has only 6 melanomas. Every new dark-skin claim must show what happens at L* <= 58 and L* <= 60, and with continuous L*.
- Lesion contrast is a plausible confounder. The adjusted model must include it, and the paper must say whether the tone association survives adjustment.
- Paired tests must resample images, not seeds, and must keep the pairing between conditions.
- Calibration must be computed on the test split with stored probabilities, not re-inferred.
- External sensitivity at 95% specificity must be labeled as an oracle threshold, not as a deployable result.

---

### Task 1: Lesion contrast features

**Files:**
- Create: `src/melanoma/fairness/contrast.py`
- Create: `scripts/build_contrast_table.py`
- Create: `tests/test_contrast.py`
- Output: `results/contrast_ham10000.csv`

**Interfaces:**
- `lesion_contrast(image_rgb, mask) -> dict` with keys `lesion_L`, `skin_L`, `delta_L`, `delta_E`, `lesion_area`, `skin_pixels`.
- `build_contrast_table(image_dir, mask_dir, tone_table, out_path) -> pandas.DataFrame` with one row per image.

- [ ] Write tests for a synthetic image where the lesion is darker than the surround and the mask is a centered square. Assert the sign of `delta_L`, that `delta_E` is positive, and that `lesion_area` equals the square area.
- [ ] Run the tests and confirm they fail.
- [ ] Implement `lesion_contrast` with CIELAB conversion, an 11-pixel perilesional ring, exclusion of very dark and very bright skin pixels, and a `delta_E` 76 formula.
- [ ] Implement the table builder and CLI with `--images-dir`, `--masks-dir`, `--tone-table`, `--out`.
- [ ] Run on all 10,015 HAM10000 images and write `results/contrast_ham10000.csv`.
- [ ] Assert no more than 1 percent missing rows and no negative `lesion_area`.
- [ ] Commit.

### Task 2: Dark-skin robustness analysis

**Files:**
- Create: `src/melanoma/fairness/dark_skin.py`
- Create: `scripts/analyze_dark_skin.py`
- Create: `tests/test_dark_skin.py`
- Output: `results/dark_skin/`

**Interfaces:**
- `threshold_sensitivity(predictions, tone, thresholds=(55, 58, 60)) -> DataFrame` with group-wise sensitivity, specificity, AUC, FPR, n, n_positive per threshold and run.
- `continuous_error_model(predictions, tone, contrast, covariates) -> dict` with cluster-bootstrap odds ratios for L*, delta_L, log area, age, sex, and site.
- `paired_auc_difference(predictions, tone, threshold, group_a, group_b, n_boot=1000) -> dict` with delta AUC, CI, and p-value.

- [ ] Write tests with a synthetic frame where darker images have higher error and lower AUC, and assert that the threshold table recovers the direction at all three thresholds.
- [ ] Write a test that the paired AUC difference is zero when both groups have identical scores.
- [ ] Run the tests and confirm they fail.
- [ ] Implement the three functions. Use lesion-level cluster bootstrap for the error model and image-level paired bootstrap for AUC differences.
- [ ] Run on the 9 CE-baseline runs and write `threshold_sensitivity.csv`, `continuous_model.csv`, `auc_differences.csv`, and `dark_skin.md`.
- [ ] Generate `results/figures/dark_skin_robustness.png` with three panels: AUC by threshold, error rate by L* decile adjusted for contrast, and forest plot of the adjusted odds ratios.
- [ ] Commit.

### Task 3: Calibration analysis

**Files:**
- Create: `src/melanoma/evaluation/calibration.py`
- Create: `scripts/analyze_calibration.py`
- Create: `tests/test_calibration.py`
- Output: `results/calibration/`

**Interfaces:**
- `expected_calibration_error(y_true, y_prob, n_bins=10) -> float`
- `brier_score(y_true, y_prob) -> float`
- `reliability_curve(y_true, y_prob, n_bins=10) -> DataFrame`
- `calibration_summary(runs_root) -> DataFrame` with one row per run and per tone group.

- [ ] Write tests for a perfectly calibrated constant predictor and for a deliberately overconfident predictor. Assert ECE near zero for the first and large for the second.
- [ ] Run the tests and confirm they fail.
- [ ] Implement the metrics and the summary script.
- [ ] Run on all 27 runs and write `calibration_runs.csv`, `calibration_summary.csv`, and `calibration.md`.
- [ ] Generate `results/figures/calibration_reliability.png` with reliability curves for the three CE-baseline architectures and ECE by tone group.
- [ ] Commit.

### Task 4: Paired ranking tests

**Files:**
- Create: `src/melanoma/evaluation/paired.py`
- Create: `scripts/analyze_ranking.py`
- Create: `tests/test_paired.py`
- Output: `results/ranking/`

**Interfaces:**
- `paired_sensitivity_change(predictions_a, predictions_b, n_boot=1000) -> dict` with delta, CI, and p-value.
- `difference_in_differences(predictions_a1, predictions_a2, predictions_b1, predictions_b2, n_boot=1000) -> dict`.

- [ ] Write tests where condition B is strictly better on every image and assert a positive delta with a CI that excludes zero.
- [ ] Write a test where the two architectures change by the same amount and assert a zero difference-in-differences.
- [ ] Run the tests and confirm they fail.
- [ ] Implement the paired bootstrap, matching images by `image_id` within each seed.
- [ ] Run for ResNet-50, EfficientNet-B0, and ViT-S/16, and for each architecture pair.
- [ ] Write `ranking_tests.csv` and `ranking.md`.
- [ ] Commit.

### Task 5: External validation of all conditions

**Files:**
- Modify: `scripts/evaluate_external.py`
- Modify: `scripts/build_paper_tables.py`
- Test: `tests/test_external_eval.py`
- Output: `results/external/hiba/`

**Interfaces:**
- Add `--conditions ce_random,focal2_cb,ce_random_strong`.
- Add `sensitivity_at_95_specificity` as a secondary per-run metric.

- [ ] Add a test that the run-id builder returns 27 ids for three conditions and three seeds.
- [ ] Add a test that the 95 percent specificity helper returns a threshold and a sensitivity on a synthetic ROC.
- [ ] Run the tests and confirm they fail.
- [ ] Implement the condition filter and the oracle-threshold helper.
- [ ] Run all 27 checkpoints on HIBA with `--merge`.
- [ ] Write `external_by_condition.csv` and update the paper table generator.
- [ ] Commit.

### Task 6: Paper integration

**Files:**
- Modify: `paper/sections/04_results.tex`
- Modify: `paper/sections/05_discussion.tex`
- Modify: `paper/sections/06_limitations.tex`
- Modify: `paper/sections/03_methods.tex`
- Modify: `paper/main.tex`
- Modify: `paper/references.bib`
- Modify: `scripts/build_paper_tables.py`
- Create: `paper/tables/dark_skin.tex`
- Create: `paper/tables/calibration.tex`
- Create: `paper/tables/ranking.tex`
- Create: `paper/figures/dark_skin_robustness.png`

- [ ] Add the new tables and figures to the generator and compile.
- [ ] Rewrite the fairness results around threshold sensitivity and the adjusted model.
- [ ] Rewrite the ranking result around the paired tests and soften any claim the tests do not support.
- [ ] Add calibration results to the mechanism discussion.
- [ ] Add external all-condition results.
- [ ] Expand related work to at least 35 verified references.
- [ ] Add data availability, code availability, conflict of interest, funding, and acknowledgments.
- [ ] Run the humanizer pass on every changed section.
- [ ] Compile with tectonic and confirm no overfull or underfull warnings, no undefined citations, and a clean 18 to 22 page PDF.
- [ ] Commit.

### Task 7: Submission package

**Files:**
- Create: `paper/cover_letter.md`
- Create: `paper/submission_checklist.md`
- Modify: `paper/README.md`

- [ ] Draft the cover letter with the contribution, the main result, and the fit to the target journal.
- [ ] Add the checklist: affiliations, ORCID, corresponding author, data and code links, ethics statement, figure permissions, reference check.
- [ ] Record the target journal decision and its formatting requirements.
- [ ] Commit.

## Verification

- [ ] `pytest tests -o addopts="" -q` passes.
- [ ] `python scripts/build_paper_tables.py` regenerates every table.
- [ ] `tectonic paper/main.tex` writes a PDF with no warnings.
- [ ] Every new number in the paper traces to a CSV under `results/`.
- [ ] Git working tree is clean and the branch is pushed.
