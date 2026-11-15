# Calibration

Expected calibration error (10 bins) and Brier score on the test split.

| Configuration | ECE | Brier |
|---|---|---|
| stage2_efficientnet_b0_ce_random | 0.060 | 0.076 |
| stage2_efficientnet_b0_ce_random_strong | 0.052 | 0.076 |
| stage2_efficientnet_b0_focal2_cb | 0.178 | 0.134 |
| stage2_resnet50_ce_random | 0.046 | 0.074 |
| stage2_resnet50_ce_random_strong | 0.046 | 0.076 |
| stage2_resnet50_focal2_cb | 0.203 | 0.133 |
| stage2_vit_s_ce_random | 0.146 | 0.126 |
| stage2_vit_s_ce_random_strong | 0.058 | 0.089 |
| stage2_vit_s_focal2_cb | 0.199 | 0.141 |

## After prior-shift and temperature correction

| Configuration | ECE raw | ECE prior | ECE prior+TS | T |
|---|---|---|---|---|
| stage2_efficientnet_b0_ce_random | 0.060 | 0.060 | 0.026 | 2.71 |
| stage2_efficientnet_b0_ce_random_strong | 0.052 | 0.052 | 0.047 | 2.11 |
| stage2_efficientnet_b0_focal2_cb | 0.178 | 0.082 | 0.045 | 1.63 |
| stage2_resnet50_ce_random | 0.046 | 0.046 | 0.035 | 1.62 |
| stage2_resnet50_ce_random_strong | 0.046 | 0.046 | 0.040 | 1.40 |
| stage2_resnet50_focal2_cb | 0.203 | 0.087 | 0.061 | 1.53 |
| stage2_vit_s_ce_random | 0.146 | 0.146 | 0.119 | 0.67 |
| stage2_vit_s_ce_random_strong | 0.058 | 0.058 | 0.043 | 0.78 |
| stage2_vit_s_focal2_cb | 0.199 | 0.084 | 0.045 | 1.58 |
