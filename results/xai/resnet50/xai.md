# Quantitative XAI

Attribution mass, pointing game, and attribution IoU for the HAM10000 test split.

| backbone | tone_group | n | n_positive | attribution_mass_mean | pointing_game_mean | attribution_iou_mean | error_rate |
|---|---|---|---|---|---|---|---|
| resnet50 | dark | 129 | 18 | 0.504 | 0.674 | 0.330 | 0.178 |
| resnet50 | light | 450 | 36 | 0.593 | 0.882 | 0.404 | 0.076 |
| resnet50 | medium | 450 | 54 | 0.501 | 0.809 | 0.362 | 0.078 |

## Correlations

| backbone | n | rho (error) | p (error) | rho (tone) | p (tone) |
|---|---|---|---|---|---|
| resnet50 | 1029 | 0.047 | 0.131 | -0.167 | 0.000 |
