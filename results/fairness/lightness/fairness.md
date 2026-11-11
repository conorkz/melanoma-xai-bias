# Tone-based fairness analysis

Groups are image-based tone strata, not clinical Fitzpatrick labels. Group column: `tone_group`.

| config_id | tone_group | n_seeds | n | n_positive | sensitivity_mean | sensitivity_std | fpr_mean | auc_mean |
|---|---|---|---|---|---|---|---|---|
| stage2_efficientnet_b0_ce_random | dark | 3 | 43 | 6 | 0.722 | 0.096 | 0.117 | 0.871 |
| stage2_efficientnet_b0_ce_random | light | 3 | 644 | 81 | 0.477 | 0.031 | 0.044 | 0.899 |
| stage2_efficientnet_b0_ce_random | medium | 3 | 840 | 99 | 0.609 | 0.050 | 0.045 | 0.931 |
| stage2_efficientnet_b0_ce_random_strong | dark | 3 | 43 | 6 | 0.556 | 0.096 | 0.090 | 0.833 |
| stage2_efficientnet_b0_ce_random_strong | light | 3 | 644 | 81 | 0.519 | 0.054 | 0.046 | 0.902 |
| stage2_efficientnet_b0_ce_random_strong | medium | 3 | 840 | 99 | 0.599 | 0.086 | 0.049 | 0.926 |
| stage2_efficientnet_b0_focal2_cb | dark | 3 | 43 | 6 | 0.500 | 0.167 | 0.081 | 0.806 |
| stage2_efficientnet_b0_focal2_cb | light | 3 | 644 | 81 | 0.420 | 0.064 | 0.046 | 0.880 |
| stage2_efficientnet_b0_focal2_cb | medium | 3 | 840 | 99 | 0.522 | 0.076 | 0.052 | 0.904 |
| stage2_resnet50_ce_random | dark | 3 | 43 | 6 | 0.444 | 0.096 | 0.117 | 0.803 |
| stage2_resnet50_ce_random | light | 3 | 644 | 81 | 0.502 | 0.036 | 0.049 | 0.872 |
| stage2_resnet50_ce_random | medium | 3 | 840 | 99 | 0.633 | 0.050 | 0.045 | 0.923 |
| stage2_resnet50_ce_random_strong | dark | 3 | 43 | 6 | 0.556 | 0.096 | 0.135 | 0.854 |
| stage2_resnet50_ce_random_strong | light | 3 | 644 | 81 | 0.527 | 0.026 | 0.049 | 0.884 |
| stage2_resnet50_ce_random_strong | medium | 3 | 840 | 99 | 0.603 | 0.038 | 0.061 | 0.915 |
| stage2_resnet50_focal2_cb | dark | 3 | 43 | 6 | 0.722 | 0.192 | 0.162 | 0.848 |
| stage2_resnet50_focal2_cb | light | 3 | 644 | 81 | 0.399 | 0.093 | 0.046 | 0.870 |
| stage2_resnet50_focal2_cb | medium | 3 | 840 | 99 | 0.481 | 0.066 | 0.054 | 0.901 |
| stage2_vit_s_ce_random | dark | 3 | 43 | 6 | 0.333 | 0.167 | 0.144 | 0.740 |
| stage2_vit_s_ce_random | light | 3 | 644 | 81 | 0.342 | 0.091 | 0.058 | 0.834 |
| stage2_vit_s_ce_random | medium | 3 | 840 | 99 | 0.451 | 0.065 | 0.055 | 0.878 |
| stage2_vit_s_ce_random_strong | dark | 3 | 43 | 6 | 0.222 | 0.096 | 0.126 | 0.763 |
| stage2_vit_s_ce_random_strong | light | 3 | 644 | 81 | 0.407 | 0.054 | 0.067 | 0.853 |
| stage2_vit_s_ce_random_strong | medium | 3 | 840 | 99 | 0.471 | 0.132 | 0.050 | 0.882 |
| stage2_vit_s_focal2_cb | dark | 3 | 43 | 6 | 0.389 | 0.096 | 0.189 | 0.712 |
| stage2_vit_s_focal2_cb | light | 3 | 644 | 81 | 0.362 | 0.062 | 0.046 | 0.845 |
| stage2_vit_s_focal2_cb | medium | 3 | 840 | 99 | 0.586 | 0.010 | 0.065 | 0.903 |

## Disparity

| config_id | max_min_gap | light_dark_ratio |
|---|---|---|
| stage2_efficientnet_b0_ce_random | 0.245 | 0.661 |
| stage2_efficientnet_b0_ce_random_strong | 0.081 | 0.933 |
| stage2_efficientnet_b0_focal2_cb | 0.102 | 0.840 |
| stage2_resnet50_ce_random | 0.189 | 1.130 |
| stage2_resnet50_ce_random_strong | 0.076 | 0.948 |
| stage2_resnet50_focal2_cb | 0.323 | 0.553 |
| stage2_vit_s_ce_random | 0.118 | 1.025 |
| stage2_vit_s_ce_random_strong | 0.249 | 1.833 |
| stage2_vit_s_focal2_cb | 0.224 | 0.931 |
