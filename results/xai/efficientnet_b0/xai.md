# Quantitative XAI

Attribution mass, pointing game, and attribution IoU for the HAM10000 test split.

| backbone | tone_group | n | n_positive | attribution_mass_mean | pointing_game_mean | attribution_iou_mean | error_rate |
|---|---|---|---|---|---|---|---|
| efficientnet_b0 | dark | 129 | 18 | 0.342 | 0.449 | 0.208 | 0.140 |
| efficientnet_b0 | light | 450 | 36 | 0.456 | 0.718 | 0.333 | 0.076 |
| efficientnet_b0 | medium | 450 | 54 | 0.374 | 0.613 | 0.284 | 0.091 |

## Correlations

| backbone | n | rho (error) | p (error) | rho (tone) | p (tone) |
|---|---|---|---|---|---|
| efficientnet_b0 | 1027 | -0.023 | 0.458 | -0.170 | 0.000 |
