# Tone-based fairness analysis

Groups are image-based tone strata, not clinical Fitzpatrick labels. Group column: `tone_group`.

| config_id | tone_group | n_seeds | n | n_positive | sensitivity_mean | sensitivity_std | fpr_mean | auc_mean |
|---|---|---|---|---|---|---|---|---|
| stage3_efficientnet_b0_ce_cb | dark | 5 | 43 | 6 | 0.433 | 0.190 | 0.070 | 0.868 |
| stage3_efficientnet_b0_ce_cb | light | 5 | 644 | 81 | 0.432 | 0.025 | 0.043 | 0.889 |
| stage3_efficientnet_b0_ce_cb | medium | 5 | 840 | 99 | 0.566 | 0.027 | 0.049 | 0.916 |
| stage3_efficientnet_b0_ce_random | dark | 5 | 43 | 6 | 0.600 | 0.149 | 0.119 | 0.815 |
| stage3_efficientnet_b0_ce_random | light | 5 | 644 | 81 | 0.477 | 0.106 | 0.053 | 0.895 |
| stage3_efficientnet_b0_ce_random | medium | 5 | 840 | 99 | 0.620 | 0.067 | 0.053 | 0.923 |
| stage3_efficientnet_b0_ce_random_strong | dark | 5 | 43 | 6 | 0.600 | 0.091 | 0.076 | 0.832 |
| stage3_efficientnet_b0_ce_random_strong | light | 5 | 644 | 81 | 0.481 | 0.047 | 0.046 | 0.899 |
| stage3_efficientnet_b0_ce_random_strong | medium | 5 | 840 | 99 | 0.590 | 0.018 | 0.053 | 0.916 |
| stage3_efficientnet_b0_focal2_cb | dark | 5 | 43 | 6 | 0.533 | 0.274 | 0.076 | 0.863 |
| stage3_efficientnet_b0_focal2_cb | light | 5 | 644 | 81 | 0.494 | 0.069 | 0.042 | 0.902 |
| stage3_efficientnet_b0_focal2_cb | medium | 5 | 840 | 99 | 0.578 | 0.068 | 0.056 | 0.922 |
| stage3_efficientnet_b0_focal2_random | dark | 5 | 43 | 6 | 0.500 | 0.204 | 0.119 | 0.842 |
| stage3_efficientnet_b0_focal2_random | light | 5 | 644 | 81 | 0.452 | 0.073 | 0.044 | 0.890 |
| stage3_efficientnet_b0_focal2_random | medium | 5 | 840 | 99 | 0.600 | 0.026 | 0.055 | 0.923 |
| stage3_resnet50_ce_cb | dark | 5 | 43 | 6 | 0.667 | 0.118 | 0.130 | 0.850 |
| stage3_resnet50_ce_cb | light | 5 | 644 | 81 | 0.467 | 0.042 | 0.047 | 0.870 |
| stage3_resnet50_ce_cb | medium | 5 | 840 | 99 | 0.568 | 0.025 | 0.055 | 0.913 |
| stage3_resnet50_ce_random | dark | 5 | 43 | 6 | 0.567 | 0.091 | 0.141 | 0.844 |
| stage3_resnet50_ce_random | light | 5 | 644 | 81 | 0.494 | 0.045 | 0.050 | 0.881 |
| stage3_resnet50_ce_random | medium | 5 | 840 | 99 | 0.651 | 0.034 | 0.057 | 0.929 |
| stage3_resnet50_ce_random_strong | dark | 5 | 43 | 6 | 0.367 | 0.139 | 0.076 | 0.804 |
| stage3_resnet50_ce_random_strong | light | 5 | 644 | 81 | 0.536 | 0.021 | 0.052 | 0.876 |
| stage3_resnet50_ce_random_strong | medium | 5 | 840 | 99 | 0.598 | 0.036 | 0.057 | 0.912 |
| stage3_resnet50_focal2_cb | dark | 5 | 43 | 6 | 0.500 | 0.118 | 0.146 | 0.840 |
| stage3_resnet50_focal2_cb | light | 5 | 644 | 81 | 0.504 | 0.033 | 0.043 | 0.889 |
| stage3_resnet50_focal2_cb | medium | 5 | 840 | 99 | 0.572 | 0.049 | 0.060 | 0.915 |
| stage3_resnet50_focal2_random | dark | 5 | 43 | 6 | 0.600 | 0.149 | 0.205 | 0.821 |
| stage3_resnet50_focal2_random | light | 5 | 644 | 81 | 0.454 | 0.035 | 0.038 | 0.882 |
| stage3_resnet50_focal2_random | medium | 5 | 840 | 99 | 0.596 | 0.073 | 0.051 | 0.915 |
| stage3_vit_s_ce_cb | dark | 5 | 43 | 6 | 0.433 | 0.091 | 0.086 | 0.833 |
| stage3_vit_s_ce_cb | light | 5 | 644 | 81 | 0.568 | 0.029 | 0.050 | 0.907 |
| stage3_vit_s_ce_cb | medium | 5 | 840 | 99 | 0.624 | 0.039 | 0.051 | 0.936 |
| stage3_vit_s_ce_random | dark | 5 | 43 | 6 | 0.300 | 0.139 | 0.086 | 0.817 |
| stage3_vit_s_ce_random | light | 5 | 644 | 81 | 0.575 | 0.033 | 0.053 | 0.906 |
| stage3_vit_s_ce_random | medium | 5 | 840 | 99 | 0.677 | 0.038 | 0.052 | 0.942 |
| stage3_vit_s_ce_random_strong | dark | 5 | 43 | 6 | 0.400 | 0.149 | 0.124 | 0.781 |
| stage3_vit_s_ce_random_strong | light | 5 | 644 | 81 | 0.573 | 0.058 | 0.048 | 0.913 |
| stage3_vit_s_ce_random_strong | medium | 5 | 840 | 99 | 0.644 | 0.036 | 0.049 | 0.935 |
| stage3_vit_s_focal2_cb | dark | 5 | 43 | 6 | 0.467 | 0.183 | 0.086 | 0.833 |
| stage3_vit_s_focal2_cb | light | 5 | 644 | 81 | 0.548 | 0.039 | 0.049 | 0.902 |
| stage3_vit_s_focal2_cb | medium | 5 | 840 | 99 | 0.644 | 0.060 | 0.049 | 0.939 |
| stage3_vit_s_focal2_random | dark | 5 | 43 | 6 | 0.433 | 0.091 | 0.070 | 0.838 |
| stage3_vit_s_focal2_random | light | 5 | 644 | 81 | 0.588 | 0.036 | 0.056 | 0.912 |
| stage3_vit_s_focal2_random | medium | 5 | 840 | 99 | 0.703 | 0.034 | 0.043 | 0.947 |

## Disparity

| config_id | max_min_gap | light_dark_ratio |
|---|---|---|
| stage3_efficientnet_b0_ce_cb | 0.134 | 0.997 |
| stage3_efficientnet_b0_ce_random | 0.144 | 0.794 |
| stage3_efficientnet_b0_ce_random_strong | 0.119 | 0.802 |
| stage3_efficientnet_b0_focal2_cb | 0.084 | 0.926 |
| stage3_efficientnet_b0_focal2_random | 0.148 | 0.904 |
| stage3_resnet50_ce_cb | 0.200 | 0.700 |
| stage3_resnet50_ce_random | 0.157 | 0.871 |
| stage3_resnet50_ce_random_strong | 0.231 | 1.461 |
| stage3_resnet50_focal2_cb | 0.072 | 1.007 |
| stage3_resnet50_focal2_random | 0.146 | 0.757 |
| stage3_vit_s_ce_cb | 0.191 | 1.311 |
| stage3_vit_s_ce_random | 0.377 | 1.918 |
| stage3_vit_s_ce_random_strong | 0.244 | 1.432 |
| stage3_vit_s_focal2_cb | 0.178 | 1.175 |
| stage3_vit_s_focal2_random | 0.270 | 1.356 |
