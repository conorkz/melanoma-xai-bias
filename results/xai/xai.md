# Quantitative XAI

Attribution mass, pointing game, and attribution IoU for the HAM10000 test split.

| backbone | tone_group | n | n_positive | attribution_mass_mean | pointing_game_mean | attribution_iou_mean | error_rate |
|---|---|---|---|---|---|---|---|
| efficientnet_b0 | dark | 129 | 18 | 0.342 | 0.449 | 0.208 | 0.140 |
| efficientnet_b0 | light | 450 | 36 | 0.456 | 0.718 | 0.333 | 0.076 |
| efficientnet_b0 | medium | 450 | 54 | 0.374 | 0.613 | 0.284 | 0.091 |
| resnet50 | dark | 129 | 18 | 0.504 | 0.674 | 0.330 | 0.178 |
| resnet50 | light | 450 | 36 | 0.593 | 0.882 | 0.404 | 0.076 |
| resnet50 | medium | 450 | 54 | 0.501 | 0.809 | 0.362 | 0.078 |
| vit_s | dark | 129 | 18 | 0.265 | 0.248 | 0.154 | 0.217 |
| vit_s | light | 450 | 36 | 0.341 | 0.604 | 0.277 | 0.109 |
| vit_s | medium | 450 | 54 | 0.251 | 0.367 | 0.173 | 0.116 |

## Correlations

| backbone | n | rho (error) | p (error) | rho (tone) | p (tone) |
|---|---|---|---|---|---|
| efficientnet_b0 | 1027 | -0.023 | 0.458 | -0.170 | 0.000 |
| resnet50 | 1029 | 0.047 | 0.131 | -0.167 | 0.000 |
| vit_s | 1029 | 0.089 | 0.004 | -0.184 | 0.000 |
