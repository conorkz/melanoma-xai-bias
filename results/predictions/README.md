# Per-sample test predictions

One CSV per Stage 2 and Stage 3 run. The filename is `<run_id>_test.csv`.

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

The 27 `stage2_` files cover the original protocol: three architectures (ResNet-50, EfficientNet-B0, ViT-S/16), three conditions (CE baseline, focal loss with class-balanced sampling, CE with strong augmentation), and three seeds.

The 75 `stage3_` files cover the revised protocol (learning rate selected per architecture, checkpoint selected by validation AUC): the same three architectures, five conditions (the three above plus focal loss with a random sampler and CE with class-balanced sampling), and five seeds. The Stage 3 files regenerate the benchmark, fairness, calibration, dark-skin, and seed-level summaries under `results/stage3/` without re-running inference.
