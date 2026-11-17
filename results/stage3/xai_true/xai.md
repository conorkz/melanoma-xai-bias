# Quantitative XAI

Attribution mass, pointing game, and attribution IoU for the HAM10000 test split.

| backbone | tone_group | n | n_positive | attribution_mass_mean | pointing_game_mean | attribution_iou_mean | error_rate |
|---|---|---|---|---|---|---|---|
| efficientnet_b0 | dark | 215 | 30 | 0.327 | 0.481 | 0.209 | 0.158 |
| efficientnet_b0 | light | 3220 | 405 | 0.430 | 0.671 | 0.311 | 0.112 |
| efficientnet_b0 | medium | 4200 | 495 | 0.350 | 0.574 | 0.260 | 0.092 |
| resnet50 | dark | 215 | 30 | 0.478 | 0.660 | 0.309 | 0.181 |
| resnet50 | light | 3220 | 405 | 0.511 | 0.795 | 0.352 | 0.107 |
| resnet50 | medium | 4200 | 495 | 0.453 | 0.730 | 0.331 | 0.092 |
| vit_s | dark | 215 | 30 | 0.501 | 0.688 | 0.303 | 0.172 |
| vit_s | light | 3220 | 405 | 0.547 | 0.749 | 0.370 | 0.100 |
| vit_s | medium | 4200 | 495 | 0.488 | 0.712 | 0.329 | 0.084 |

## Correlations

| backbone | n | rho (error) | p (error) | rho (tone) | p (tone) |
|---|---|---|---|---|---|
| efficientnet_b0 | 7509 | 0.008 | 0.489 | -0.143 | 0.000 |
| resnet50 | 7626 | -0.186 | 0.000 | -0.107 | 0.000 |
| vit_s | 7635 | 0.090 | 0.000 | -0.148 | 0.000 |
