# RQ1 benchmark

The operating threshold was selected on validation at the target specificity.
Choose configurations using the `val_*` columns. The test columns are reported once
for the selected configuration and must not be used for model selection.

| config_id | backbone | loss | sampler | augmentation | img_size | n_seeds | n_runs | n_test | n_positive | val_sensitivity_mean | val_sensitivity_std | val_specificity_mean | val_specificity_std | val_balanced_accuracy_mean | val_balanced_accuracy_std | val_auc_mean | val_auc_std | sensitivity_mean | sensitivity_std | specificity_mean | specificity_std | balanced_accuracy_mean | balanced_accuracy_std | auc_mean | auc_std | sensitivity_ci_low | sensitivity_ci_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stage2_efficientnet_b0_ce_random | efficientnet_b0 | ce | random | standard | 224 | 3 | 3 | 1527 | 186 | 0.553 | 0.035 | 0.951 | 0.000 | 0.752 | 0.018 | 0.895 | 0.015 | 0.556 | 0.037 | 0.953 | 0.004 | 0.754 | 0.019 | 0.914 | 0.008 | 0.480 | 0.624 |
| stage2_efficientnet_b0_ce_random_strong | efficientnet_b0 | ce | random | strong | 224 | 3 | 3 | 1527 | 186 | 0.521 | 0.021 | 0.952 | 0.002 | 0.737 | 0.011 | 0.883 | 0.007 | 0.563 | 0.065 | 0.952 | 0.005 | 0.757 | 0.030 | 0.912 | 0.009 | 0.489 | 0.636 |
| stage2_efficientnet_b0_focal2_cb | efficientnet_b0 | focal | class_balanced | standard | 224 | 3 | 3 | 1527 | 186 | 0.481 | 0.042 | 0.953 | 0.002 | 0.717 | 0.021 | 0.888 | 0.009 | 0.477 | 0.073 | 0.950 | 0.005 | 0.713 | 0.038 | 0.891 | 0.014 | 0.405 | 0.548 |
| stage2_resnet50_ce_random | resnet50 | ce | random | standard | 224 | 3 | 3 | 1527 | 186 | 0.526 | 0.044 | 0.952 | 0.002 | 0.739 | 0.022 | 0.887 | 0.022 | 0.570 | 0.041 | 0.951 | 0.009 | 0.761 | 0.016 | 0.897 | 0.003 | 0.497 | 0.639 |
| stage2_resnet50_ce_random_strong | resnet50 | ce | random | strong | 224 | 3 | 3 | 1527 | 186 | 0.549 | 0.015 | 0.955 | 0.004 | 0.752 | 0.007 | 0.895 | 0.007 | 0.568 | 0.030 | 0.942 | 0.006 | 0.755 | 0.012 | 0.899 | 0.010 | 0.498 | 0.637 |
| stage2_resnet50_focal2_cb | resnet50 | focal | class_balanced | standard | 224 | 3 | 3 | 1527 | 186 | 0.489 | 0.027 | 0.952 | 0.002 | 0.721 | 0.013 | 0.884 | 0.010 | 0.453 | 0.066 | 0.947 | 0.007 | 0.700 | 0.030 | 0.884 | 0.003 | 0.379 | 0.523 |
| stage2_vit_s_ce_random | vit_small_patch16_224 | ce | random | standard | 224 | 3 | 3 | 1527 | 186 | 0.423 | 0.042 | 0.954 | 0.006 | 0.689 | 0.019 | 0.867 | 0.009 | 0.400 | 0.044 | 0.941 | 0.007 | 0.670 | 0.019 | 0.854 | 0.014 | 0.329 | 0.471 |
| stage2_vit_s_ce_random_strong | vit_small_patch16_224 | ce | random | strong | 224 | 3 | 3 | 1527 | 186 | 0.466 | 0.030 | 0.951 | 0.000 | 0.708 | 0.015 | 0.868 | 0.007 | 0.435 | 0.083 | 0.941 | 0.002 | 0.688 | 0.042 | 0.866 | 0.012 | 0.366 | 0.503 |
| stage2_vit_s_focal2_cb | vit_small_patch16_224 | focal | class_balanced | standard | 224 | 3 | 3 | 1527 | 186 | 0.474 | 0.034 | 0.955 | 0.003 | 0.714 | 0.016 | 0.879 | 0.007 | 0.482 | 0.022 | 0.940 | 0.011 | 0.711 | 0.015 | 0.872 | 0.011 | 0.408 | 0.556 |
