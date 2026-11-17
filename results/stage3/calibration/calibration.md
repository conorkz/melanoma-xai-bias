# Calibration

Expected calibration error (10 bins) and Brier score on the test split.

| Configuration | ECE | Brier |
|---|---|---|
| stage3_efficientnet_b0_ce_cb | 0.084 | 0.092 |
| stage3_efficientnet_b0_ce_random | 0.051 | 0.076 |
| stage3_efficientnet_b0_ce_random_strong | 0.052 | 0.076 |
| stage3_efficientnet_b0_focal2_cb | 0.092 | 0.091 |
| stage3_efficientnet_b0_focal2_random | 0.075 | 0.082 |
| stage3_resnet50_ce_cb | 0.092 | 0.103 |
| stage3_resnet50_ce_random | 0.040 | 0.072 |
| stage3_resnet50_ce_random_strong | 0.044 | 0.075 |
| stage3_resnet50_focal2_cb | 0.119 | 0.097 |
| stage3_resnet50_focal2_random | 0.124 | 0.096 |
| stage3_vit_s_ce_cb | 0.091 | 0.095 |
| stage3_vit_s_ce_random | 0.060 | 0.074 |
| stage3_vit_s_ce_random_strong | 0.046 | 0.071 |
| stage3_vit_s_focal2_cb | 0.050 | 0.073 |
| stage3_vit_s_focal2_random | 0.053 | 0.067 |

## After prior-shift and temperature correction

| Configuration | ECE raw | ECE prior | ECE prior+TS | T |
|---|---|---|---|---|
| stage3_efficientnet_b0_ce_cb | 0.084 | 0.075 | 0.022 | 3.27 |
| stage3_efficientnet_b0_ce_random | 0.051 | 0.051 | 0.026 | 2.25 |
| stage3_efficientnet_b0_ce_random_strong | 0.052 | 0.052 | 0.028 | 2.11 |
| stage3_efficientnet_b0_focal2_cb | 0.092 | 0.091 | 0.048 | 2.05 |
| stage3_efficientnet_b0_focal2_random | 0.075 | 0.032 | 0.028 | 1.01 |
| stage3_resnet50_ce_cb | 0.092 | 0.058 | 0.025 | 1.99 |
| stage3_resnet50_ce_random | 0.040 | 0.040 | 0.024 | 1.45 |
| stage3_resnet50_ce_random_strong | 0.044 | 0.044 | 0.042 | 1.28 |
| stage3_resnet50_focal2_cb | 0.119 | 0.091 | 0.053 | 1.72 |
| stage3_resnet50_focal2_random | 0.124 | 0.041 | 0.031 | 0.81 |
| stage3_vit_s_ce_cb | 0.091 | 0.053 | 0.027 | 2.11 |
| stage3_vit_s_ce_random | 0.060 | 0.060 | 0.042 | 1.92 |
| stage3_vit_s_ce_random_strong | 0.046 | 0.046 | 0.043 | 1.69 |
| stage3_vit_s_focal2_cb | 0.050 | 0.087 | 0.046 | 2.10 |
| stage3_vit_s_focal2_random | 0.053 | 0.026 | 0.019 | 1.15 |
