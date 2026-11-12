# Per-sample test predictions

One CSV per Stage 2 run. The filename is `<run_id>_test.csv`.

Columns:

- `image_id`: HAM10000 image identifier.
- `split`: always `test` in these files.
- `y_true`: binary melanoma label.
- `y_prob`: melanoma probability.
- `y_pred`: prediction at the validation-selected threshold.
- `threshold`: threshold selected on validation at 95% specificity.
- `run_id`: run identifier.
- `config_id`: architecture and condition.
- `seed`: training seed.

The 27 files cover three architectures (ResNet-50, EfficientNet-B0, ViT-S/16), three conditions (CE baseline, focal loss with class-balanced sampling, CE with strong augmentation), and three seeds. These files regenerate the Stage 2, fairness, calibration, ranking, dark-skin, and XAI summaries without re-running inference.
