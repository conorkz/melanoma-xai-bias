# Seed-level inference

The seed is the experimental unit. Intervals are 95% t intervals across 5 seeds.

## stage3 sensitivity

| Configuration | Mean | SD | 95% t interval |
|---|---|---|---|
| efficientnet_b0 / ce_cb | 0.503 | 0.006 | 0.496 to 0.511 |
| efficientnet_b0 / ce_random | 0.557 | 0.065 | 0.476 to 0.638 |
| efficientnet_b0 / ce_random_strong | 0.543 | 0.032 | 0.503 to 0.583 |
| efficientnet_b0 / focal2_cb | 0.540 | 0.041 | 0.489 to 0.590 |
| efficientnet_b0 / focal2_random | 0.532 | 0.036 | 0.488 to 0.577 |
| resnet50 / ce_cb | 0.527 | 0.027 | 0.494 to 0.560 |
| resnet50 / ce_random | 0.580 | 0.031 | 0.541 to 0.618 |
| resnet50 / ce_random_strong | 0.563 | 0.026 | 0.532 to 0.595 |
| resnet50 / focal2_cb | 0.540 | 0.025 | 0.509 to 0.571 |
| resnet50 / focal2_random | 0.534 | 0.033 | 0.494 to 0.575 |
| vit_s / ce_cb | 0.594 | 0.016 | 0.574 to 0.613 |
| vit_s / ce_random | 0.620 | 0.036 | 0.575 to 0.666 |
| vit_s / ce_random_strong | 0.605 | 0.041 | 0.554 to 0.657 |
| vit_s / focal2_cb | 0.597 | 0.044 | 0.542 to 0.652 |
| vit_s / focal2_random | 0.644 | 0.025 | 0.613 to 0.675 |

## Paired sensitivity changes

| Architecture | Condition | Mean | SD | 95% t interval |
|---|---|---|---|---|
| efficientnet_b0 | ce_cb | -0.054 | 0.063 | -0.132 to +0.025 |
| efficientnet_b0 | ce_random_strong | -0.014 | 0.046 | -0.072 to +0.044 |
| efficientnet_b0 | focal2_cb | -0.017 | 0.084 | -0.122 to +0.088 |
| efficientnet_b0 | focal2_random | -0.025 | 0.039 | -0.073 to +0.024 |
| resnet50 | ce_cb | -0.053 | 0.054 | -0.120 to +0.014 |
| resnet50 | ce_random_strong | -0.016 | 0.034 | -0.058 to +0.026 |
| resnet50 | focal2_cb | -0.040 | 0.052 | -0.105 to +0.025 |
| resnet50 | focal2_random | -0.045 | 0.053 | -0.111 to +0.021 |
| vit_s | ce_cb | -0.027 | 0.036 | -0.071 to +0.018 |
| vit_s | ce_random_strong | -0.015 | 0.020 | -0.040 to +0.010 |
| vit_s | focal2_cb | -0.024 | 0.067 | -0.106 to +0.059 |
| vit_s | focal2_random | +0.024 | 0.052 | -0.041 to +0.088 |

## Difference in differences

| Condition | Comparison | Mean | SD | 95% t interval |
|---|---|---|---|---|
| ce_cb | efficientnet_b0 minus vit_s | -0.027 | 0.088 | -0.136 to +0.082 |
| ce_cb | resnet50 minus efficientnet_b0 | +0.001 | 0.099 | -0.122 to +0.124 |
| ce_cb | resnet50 minus vit_s | -0.026 | 0.067 | -0.109 to +0.057 |
| ce_random_strong | efficientnet_b0 minus vit_s | +0.001 | 0.044 | -0.054 to +0.056 |
| ce_random_strong | resnet50 minus efficientnet_b0 | -0.002 | 0.057 | -0.072 to +0.068 |
| ce_random_strong | resnet50 minus vit_s | -0.001 | 0.038 | -0.048 to +0.046 |
| focal2_cb | efficientnet_b0 minus vit_s | +0.006 | 0.098 | -0.115 to +0.128 |
| focal2_cb | resnet50 minus efficientnet_b0 | -0.023 | 0.097 | -0.143 to +0.097 |
| focal2_cb | resnet50 minus vit_s | -0.016 | 0.079 | -0.115 to +0.082 |
| focal2_random | efficientnet_b0 minus vit_s | -0.048 | 0.072 | -0.137 to +0.041 |
| focal2_random | resnet50 minus efficientnet_b0 | -0.020 | 0.091 | -0.134 to +0.093 |
| focal2_random | resnet50 minus vit_s | -0.069 | 0.054 | -0.135 to -0.002 |
