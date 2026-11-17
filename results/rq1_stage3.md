# RQ1 benchmark

The operating threshold was selected on validation at the target specificity.
Choose configurations using the `val_*` columns. The test columns are reported once
for the selected configuration and must not be used for model selection.

| config_id | backbone | loss | sampler | augmentation | img_size | n_seeds | n_runs | n_test | n_positive | val_sensitivity_mean | val_sensitivity_std | val_specificity_mean | val_specificity_std | val_balanced_accuracy_mean | val_balanced_accuracy_std | val_auc_mean | val_auc_std | sensitivity_mean | sensitivity_std | specificity_mean | specificity_std | balanced_accuracy_mean | balanced_accuracy_std | auc_mean | auc_std | sensitivity_ci_low | sensitivity_ci_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stage3_efficientnet_b0_ce_cb | efficientnet_b0 | ce | class_balanced | standard | 224 | 5 | 5 | 1527 | 186 | 0.521 | 0.042 | 0.953 | 0.003 | 0.737 | 0.022 | 0.896 | 0.004 | 0.503 | 0.006 | 0.953 | 0.006 | 0.728 | 0.003 | 0.902 | 0.006 | 0.430 | 0.574 |
| stage3_efficientnet_b0_ce_random | efficientnet_b0 | ce | random | standard | 224 | 5 | 5 | 1527 | 186 | 0.568 | 0.037 | 0.953 | 0.002 | 0.760 | 0.018 | 0.901 | 0.003 | 0.557 | 0.065 | 0.945 | 0.008 | 0.751 | 0.033 | 0.907 | 0.015 | 0.485 | 0.625 |
| stage3_efficientnet_b0_ce_random_strong | efficientnet_b0 | ce | random | strong | 224 | 5 | 5 | 1527 | 186 | 0.538 | 0.021 | 0.953 | 0.003 | 0.746 | 0.009 | 0.891 | 0.006 | 0.543 | 0.032 | 0.949 | 0.008 | 0.746 | 0.017 | 0.905 | 0.005 | 0.470 | 0.613 |
| stage3_efficientnet_b0_focal2_cb | efficientnet_b0 | focal | class_balanced | standard | 224 | 5 | 5 | 1527 | 186 | 0.536 | 0.022 | 0.953 | 0.001 | 0.744 | 0.011 | 0.897 | 0.006 | 0.540 | 0.041 | 0.949 | 0.006 | 0.744 | 0.021 | 0.911 | 0.012 | 0.468 | 0.610 |
| stage3_efficientnet_b0_focal2_random | efficientnet_b0 | focal | random | standard | 224 | 5 | 5 | 1527 | 186 | 0.568 | 0.018 | 0.952 | 0.002 | 0.760 | 0.009 | 0.902 | 0.006 | 0.532 | 0.036 | 0.948 | 0.005 | 0.740 | 0.020 | 0.906 | 0.013 | 0.457 | 0.602 |
| stage3_resnet50_ce_cb | resnet50 | ce | class_balanced | standard | 224 | 5 | 5 | 1527 | 186 | 0.521 | 0.022 | 0.951 | 0.001 | 0.736 | 0.011 | 0.886 | 0.006 | 0.527 | 0.027 | 0.947 | 0.006 | 0.737 | 0.013 | 0.892 | 0.006 | 0.455 | 0.597 |
| stage3_resnet50_ce_random | resnet50 | ce | random | standard | 224 | 5 | 5 | 1527 | 186 | 0.505 | 0.015 | 0.953 | 0.002 | 0.729 | 0.008 | 0.891 | 0.006 | 0.580 | 0.031 | 0.943 | 0.006 | 0.762 | 0.015 | 0.906 | 0.008 | 0.508 | 0.645 |
| stage3_resnet50_ce_random_strong | resnet50 | ce | random | strong | 224 | 5 | 5 | 1527 | 186 | 0.528 | 0.029 | 0.951 | 0.001 | 0.740 | 0.015 | 0.890 | 0.013 | 0.563 | 0.026 | 0.945 | 0.010 | 0.754 | 0.018 | 0.893 | 0.011 | 0.492 | 0.631 |
| stage3_resnet50_focal2_cb | resnet50 | focal | class_balanced | standard | 224 | 5 | 5 | 1527 | 186 | 0.519 | 0.033 | 0.952 | 0.001 | 0.736 | 0.016 | 0.901 | 0.003 | 0.540 | 0.025 | 0.945 | 0.008 | 0.742 | 0.010 | 0.901 | 0.008 | 0.466 | 0.609 |
| stage3_resnet50_focal2_random | resnet50 | focal | random | standard | 224 | 5 | 5 | 1527 | 186 | 0.494 | 0.029 | 0.952 | 0.001 | 0.723 | 0.014 | 0.894 | 0.005 | 0.534 | 0.033 | 0.950 | 0.009 | 0.742 | 0.015 | 0.896 | 0.007 | 0.461 | 0.604 |
| stage3_vit_s_ce_cb | vit_small_patch16_224 | ce | class_balanced | standard | 224 | 5 | 5 | 1527 | 186 | 0.551 | 0.040 | 0.953 | 0.003 | 0.752 | 0.019 | 0.906 | 0.002 | 0.594 | 0.016 | 0.948 | 0.007 | 0.771 | 0.007 | 0.920 | 0.001 | 0.522 | 0.663 |
| stage3_vit_s_ce_random | vit_small_patch16_224 | ce | random | standard | 224 | 5 | 5 | 1527 | 186 | 0.581 | 0.034 | 0.954 | 0.002 | 0.767 | 0.017 | 0.910 | 0.004 | 0.620 | 0.036 | 0.947 | 0.009 | 0.784 | 0.018 | 0.923 | 0.008 | 0.549 | 0.688 |
| stage3_vit_s_ce_random_strong | vit_small_patch16_224 | ce | random | strong | 224 | 5 | 5 | 1527 | 186 | 0.572 | 0.025 | 0.952 | 0.002 | 0.762 | 0.012 | 0.905 | 0.003 | 0.605 | 0.041 | 0.950 | 0.009 | 0.778 | 0.018 | 0.921 | 0.007 | 0.534 | 0.673 |
| stage3_vit_s_focal2_cb | vit_small_patch16_224 | focal | class_balanced | standard | 224 | 5 | 5 | 1527 | 186 | 0.588 | 0.037 | 0.952 | 0.001 | 0.770 | 0.018 | 0.911 | 0.007 | 0.597 | 0.044 | 0.950 | 0.007 | 0.773 | 0.021 | 0.920 | 0.007 | 0.525 | 0.663 |
| stage3_vit_s_focal2_random | vit_small_patch16_224 | focal | random | standard | 224 | 5 | 5 | 1527 | 186 | 0.582 | 0.008 | 0.951 | 0.001 | 0.767 | 0.004 | 0.913 | 0.003 | 0.644 | 0.025 | 0.951 | 0.004 | 0.797 | 0.011 | 0.928 | 0.004 | 0.576 | 0.710 |
