# Quantitative XAI

Attribution mass, pointing game, and attribution IoU for the HAM10000 test split.

| backbone | tone_group | n | n_positive | attribution_mass_mean | pointing_game_mean | attribution_iou_mean | error_rate |
|---|---|---|---|---|---|---|---|
| vit_s | dark | 129 | 18 | 0.265 | 0.248 | 0.154 | 0.217 |
| vit_s | light | 450 | 36 | 0.341 | 0.604 | 0.277 | 0.109 |
| vit_s | medium | 450 | 54 | 0.251 | 0.367 | 0.173 | 0.116 |

## Correlations

| backbone | n | rho (error) | p (error) | rho (tone) | p (tone) |
|---|---|---|---|---|---|
| vit_s | 1029 | 0.089 | 0.004 | -0.184 | 0.000 |
