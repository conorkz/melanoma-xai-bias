# Coordinator verification notes

Checked against the raw result CSVs on 12 November 2026.

- External ViT per-seed sensitivity. CE: 0.0816, 0.2449, 0.3367. Focal: 0.3010, 0.3724, 0.2704. Changes: +0.219, +0.127, -0.067. Mean +0.093. The mean is positive but one seed is negative, so "reproduces the interaction" needs qualification.
- External CNN focal changes are negative in every seed. ResNet-50: -0.172, -0.129, -0.048. EfficientNet-B0: -0.022, -0.134, -0.081.
- Ranking difference-in-differences per seed. ResNet-50 minus ViT-S/16: -0.328, -0.161, -0.108; every per-seed 95% interval excludes zero. EfficientNet-B0 minus ViT-S/16: -0.177, -0.167, -0.140; every per-seed interval excludes zero. The summary table's min/max envelope is not a proper pooled 95% CI, but the direction is consistent in all three seeds.
- Dark-versus-light and dark-versus-medium AUC differences. 0 of 54 run-level comparisons are significant; all bootstrap intervals include zero.
- ECE ratio focal/CE. ResNet-50 4.4x, EfficientNet-B0 3.0x, ViT-S/16 1.36x. "Roughly triples" is false for the ViT.
- XAI target class. `scripts/analyze_xai.py` uses the predicted class as the Grad-CAM target. False-negative heatmaps therefore explain the benign prediction, not the missed melanoma.
- Figure resolution. `stage2_best_sensitivity_training_curves.png` and `stage2_best_sensitivity_roc_test.png` are 150 dpi. Journals usually require 300 dpi or vector.
- Split file. No split CSV is present under `results/`, although the paper says splits are released.
- Figure 3C. The current plotting code filters `outcome == "error"`, so the figure shows the pooled error model, not the false-positive model. The audit claim that it plots the false-positive model is incorrect.
- Seven-class classification. It was never run. The limitation sentence calling it a secondary experiment is false.
- Abstract length. 310 to 323 words. MDPI limit is about 200 words; CIBM, CMPB, and AIM are about 250; Scientific Reports recommends 200.
