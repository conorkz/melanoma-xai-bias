# RQ1 benchmark

The operating threshold was selected on validation at the target specificity.
Choose configurations using the `val_*` columns. The test columns are reported once
for the selected configuration and must not be used for model selection.

| config_id | backbone | loss | sampler | augmentation | img_size | n_seeds | n_runs | n_test | n_positive | val_sensitivity_mean | val_sensitivity_std | val_specificity_mean | val_specificity_std | val_balanced_accuracy_mean | val_balanced_accuracy_std | val_auc_mean | val_auc_std | sensitivity_mean | sensitivity_std | specificity_mean | specificity_std | balanced_accuracy_mean | balanced_accuracy_std | auc_mean | auc_std | sensitivity_ci_low | sensitivity_ci_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stage1_ce_cb | efficientnet_b0 | ce | class_balanced | standard | 160 | 3 | 3 | 1527 | 186 | 0.521 | 0.086 | 0.951 | 0.001 | 0.736 | 0.042 | 0.891 | 0.009 | 0.484 | 0.033 | 0.938 | 0.003 | 0.711 | 0.018 | 0.890 | 0.013 | 0.411 | 0.554 |
| stage1_ce_random | efficientnet_b0 | ce | random | standard | 160 | 3 | 3 | 1527 | 186 | 0.536 | 0.016 | 0.952 | 0.002 | 0.744 | 0.008 | 0.894 | 0.016 | 0.530 | 0.024 | 0.948 | 0.002 | 0.739 | 0.013 | 0.897 | 0.010 | 0.456 | 0.600 |
| stage1_ce_random_strong | efficientnet_b0 | ce | random | strong | 160 | 3 | 3 | 1527 | 186 | 0.556 | 0.027 | 0.951 | 0.001 | 0.754 | 0.013 | 0.890 | 0.007 | 0.513 | 0.008 | 0.943 | 0.007 | 0.728 | 0.006 | 0.901 | 0.007 | 0.438 | 0.582 |
| stage1_focal2_cb | efficientnet_b0 | focal | class_balanced | standard | 160 | 3 | 3 | 1527 | 186 | 0.502 | 0.039 | 0.952 | 0.002 | 0.727 | 0.020 | 0.896 | 0.013 | 0.480 | 0.048 | 0.943 | 0.003 | 0.712 | 0.023 | 0.897 | 0.006 | 0.408 | 0.551 |
| stage1_focal2_cb_strong | efficientnet_b0 | focal | class_balanced | strong | 160 | 3 | 3 | 1527 | 186 | 0.483 | 0.013 | 0.952 | 0.001 | 0.717 | 0.007 | 0.891 | 0.006 | 0.491 | 0.087 | 0.943 | 0.012 | 0.717 | 0.038 | 0.896 | 0.006 | 0.421 | 0.558 |
| stage1_focal2_random | efficientnet_b0 | focal | random | standard | 160 | 3 | 3 | 1527 | 186 | 0.509 | 0.041 | 0.952 | 0.002 | 0.730 | 0.020 | 0.895 | 0.020 | 0.487 | 0.022 | 0.946 | 0.006 | 0.717 | 0.013 | 0.894 | 0.009 | 0.412 | 0.560 |
| stage1_focal3_cb | efficientnet_b0 | focal | class_balanced | standard | 160 | 3 | 3 | 1527 | 186 | 0.449 | 0.051 | 0.952 | 0.002 | 0.700 | 0.025 | 0.888 | 0.012 | 0.478 | 0.019 | 0.946 | 0.001 | 0.712 | 0.009 | 0.899 | 0.004 | 0.405 | 0.553 |
| stage1_wce_random | efficientnet_b0 | weighted_ce | random | standard | 160 | 3 | 3 | 1527 | 186 | 0.502 | 0.043 | 0.954 | 0.002 | 0.728 | 0.020 | 0.890 | 0.006 | 0.493 | 0.025 | 0.937 | 0.006 | 0.715 | 0.016 | 0.896 | 0.005 | 0.421 | 0.564 |
