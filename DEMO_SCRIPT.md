> This script covers the Phase 0 and Phase 1 meeting. For the next supervisor update, use `RESULTS_MEMO_TEMPLATE.md` with the RQ1 benchmark output.

Goal: present the beginner phase of the thesis (Phase 0 and Phase 1) as working, reproducible software with real data evidence, and align on next steps. Target length 12 to 15 minutes plus questions.

## Pre-meeting checklist (10 minutes before)

```bash
cd "path/to/melanoma-xai-bias"
source .venv/bin/activate
scripts/run_tests.sh                      # confirm suite is green
.venv/bin/python scripts/smoke_test.py    # only if checkpoint is missing or stale
.venv/bin/python scripts/demo_gradcam.py  # pre-generate heatmaps as backup
```

- [ ] `data/raw/HAM10000_metadata.csv` and `data/HAM10000/images/` exist (10,015 images)
- [ ] `checkpoints/smoke_test.pt` exists
- [ ] `outputs/smoke_test/metrics.json` and `training_curves.png` exist
- [ ] `outputs/gradcam/gradcam_grid.png` exists
- [ ] `notebooks/01_eda_ham10000.ipynb` is open in a viewer (already executed with figures)
- [ ] Terminal open at the project root (keep this script's commands in shell history)

## Demo flow

### 1. Framing (1 min)
Open the README. One sentence: "This is the working foundation of the thesis: data pipeline, imbalance handling, explainability, and evaluation, all tested and reproducible on my laptop, with cloud training planned for the full experiments." Point at the project tree: code lives in `src/melanoma`, experiments are driven by configs, everything is covered by tests.

### 2. EDA notebook highlights (3 min)
Open `notebooks/01_eda_ham10000.ipynb`.

- Class imbalance chart: nevus dominates, melanoma is about 11 percent, the ratio is roughly 58 to 1. Say why this matters: a model that predicts "nevus" always looks 67 percent accurate while catching zero cancers. This is why the design uses focal loss, class-balanced sampling, and sensitivity-first metrics.
- Sample grid: note the visual similarity between melanoma and nevus, the core difficulty.
- Split leakage demo: about 20 percent of lesions have multiple images; the notebook prints how many lesions a naive random split leaks into both sides. This is why all splits are lesion-aware.
- Tone proxy histogram: show it, then immediately state the caveat: this is a luminance proxy, not Fitzpatrick labels; the real fairness analysis uses Fitzpatrick17k and the Fitzpatrick scale critique will be discussed honestly.

### 3. The training pipeline (2 min)
Show `configs/smoke_test.yaml`: one file controls backbone, loss (focal, gamma 2, alpha 0.75), class-balanced sampler, image size, epochs. Mention: same config system scales to the full grid later; every run writes metrics JSON, training curves, and a checkpoint.

### 4. Live demo: Grad-CAM (3 min)

```bash
.venv/bin/python scripts/demo_gradcam.py
open outputs/gradcam/gradcam_grid.png
```

Narrate: each tile shows P(melanoma) from the trained model plus the heatmap on the pixels that drove the decision. This is the interface the clinical story depends on, and in Phase 4 these heatmaps become the bias diagnostic: attribution mass outside the lesion is the early warning signal.

### 5. Results and honesty about scale (2 to 3 min)
```bash
cat outputs/smoke_test/metrics.json
open outputs/smoke_test/training_curves.png
```
State clearly: this is a smoke run on a 2,000 image subset for pipeline validation, not a result to interpret scientifically. The numbers exist to prove the pipeline works end to end; real numbers come from the full training grid on Kaggle GPUs (about 30 GPU-hours per week free).

### 6. Next steps to agree on (2 min)
- Pilot factorial (loss and sampling interactions) on EfficientNet-B0 at 160 px, on Kaggle
- Confirm evaluation protocol: sensitivity at 95 percent specificity as the headline metric
- Confirm fairness framing (XAI-based bias detection) versus the pure generalization framing; both share this infrastructure
- Ask for: tone-annotated dataset access confirmation (Fitzpatrick17k), any department GPU allocation, and feedback on the planned mitigation experiments

## Likely questions and suggested answers

- **Why not just report accuracy?** Because accuracy is dominated by the majority class and hides false negatives; sensitivity, balanced accuracy, and AUC are the clinically meaningful metrics.
- **Why HAM10000 before ISIC 2020?** It is 3x smaller, well annotated, and fast to iterate on; ISIC 2020 is the scale-up step once the configuration is frozen.
- **What is actually novel?** Not "CNN versus ViT" and not "just Grad-CAM". The novel angle is using quantitative XAI metrics (attribution localization, faithfulness) to detect skin tone bias, and measuring whether bias-aware training closes both the metric gap and the attribution gap.
- **Is this too big for one student?** The scope is deliberately staged: three architectures, a frozen grid, pilots on subsets, and free cloud GPUs. The plan document has a risk register with cut points (deployment chapter and generative augmentation are the designated cuts).
- **Why should anyone trust the heatmaps?** They are validated quantitatively, not shown as decoration: fraction of attribution mass inside the lesion mask, pointing game, deletion and insertion curves.

## Fallback plan (if anything fails live)

Do not debug during the meeting. Switch to pre-generated artifacts:

1. Notebook figures: `outputs/eda/` (class distribution, sample grid, tone proxy, augmentation preview)
2. Smoke results: `outputs/smoke_test/metrics.json` and `training_curves.png`
3. Heatmaps: `outputs/gradcam/gradcam_grid.png`
4. Tests: show a screenshot or run `scripts/run_tests.sh` once; if it is slow, skip it, the suite is already verified.

If the terminal misbehaves entirely, the notebook plus figures folder is enough to run the whole conversation.
