# Quantitative XAI

Attribution mass, pointing game, and attribution IoU for the HAM10000 test split.

| backbone | tone_group | n | n_positive | attribution_mass_mean | pointing_game_mean | attribution_iou_mean | error_rate |
|---|---|---|---|---|---|---|---|
| efficientnet_b0 | dark | 129 | 18 | 0.331 | 0.448 | 0.195 | 0.140 |
| efficientnet_b0 | light | 450 | 36 | 0.450 | 0.702 | 0.324 | 0.076 |
| efficientnet_b0 | medium | 450 | 54 | 0.368 | 0.598 | 0.280 | 0.091 |
| resnet50 | dark | 129 | 18 | 0.465 | 0.636 | 0.301 | 0.178 |
| resnet50 | light | 450 | 36 | 0.571 | 0.847 | 0.387 | 0.076 |
| resnet50 | medium | 450 | 54 | 0.480 | 0.777 | 0.346 | 0.078 |
| vit_s | dark | 129 | 18 | 0.261 | 0.248 | 0.144 | 0.217 |
| vit_s | light | 450 | 36 | 0.341 | 0.607 | 0.273 | 0.109 |
| vit_s | medium | 450 | 54 | 0.260 | 0.391 | 0.185 | 0.116 |

## Correlations

| backbone | n | rho (error) | p (error) | rho (tone) | p (tone) |
|---|---|---|---|---|---|
| efficientnet_b0 | 1011 | -0.129 | 0.000 | -0.176 | 0.000 |
| resnet50 | 1028 | -0.230 | 0.000 | -0.174 | 0.000 |
| vit_s | 1029 | 0.136 | 0.000 | -0.181 | 0.000 |
