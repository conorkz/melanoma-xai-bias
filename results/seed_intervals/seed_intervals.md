# Seed-level inference

The seed is the experimental unit. Intervals are 95% t intervals across three seeds.

## Stage 2 sensitivity

| Configuration | Mean | SD | 95% t interval |
|---|---|---|---|
| efficientnet_b0 / ce_random | 0.556 | 0.037 | 0.465 to 0.646 |
| efficientnet_b0 / ce_random_strong | 0.563 | 0.065 | 0.402 to 0.724 |
| efficientnet_b0 / focal2_cb | 0.477 | 0.073 | 0.295 to 0.658 |
| resnet50 / ce_random | 0.570 | 0.041 | 0.469 to 0.671 |
| resnet50 / ce_random_strong | 0.568 | 0.030 | 0.495 to 0.642 |
| resnet50 / focal2_cb | 0.453 | 0.066 | 0.289 to 0.618 |
| vit_s / ce_random | 0.400 | 0.044 | 0.291 to 0.508 |
| vit_s / ce_random_strong | 0.435 | 0.083 | 0.228 to 0.643 |
| vit_s / focal2_cb | 0.482 | 0.022 | 0.428 to 0.536 |

## Paired sensitivity changes

| Architecture | Condition | Mean | SD | 95% t interval |
|---|---|---|---|---|
| efficientnet_b0 | ce_random_strong | +0.007 | 0.101 | -0.244 to +0.258 |
| efficientnet_b0 | focal2_cb | -0.079 | 0.056 | -0.219 to +0.061 |
| resnet50 | ce_random_strong | -0.002 | 0.027 | -0.069 to +0.065 |
| resnet50 | focal2_cb | -0.116 | 0.063 | -0.272 to +0.039 |
| vit_s | ce_random_strong | +0.036 | 0.045 | -0.075 to +0.147 |
| vit_s | focal2_cb | +0.082 | 0.065 | -0.079 to +0.244 |

## Difference in differences

| Comparison | Mean | SD | 95% t interval |
|---|---|---|---|
| efficientnet_b0 minus vit_s | -0.161 | 0.019 | -0.209 to -0.113 |
| resnet50 minus efficientnet_b0 | -0.038 | 0.099 | -0.283 to +0.208 |
| resnet50 minus vit_s | -0.199 | 0.115 | -0.484 to +0.087 |
