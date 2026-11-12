# Paired ranking tests

Sensitivity changes paired by test image. Intervals are 95% bootstrap intervals.

| Test | Comparison | Architecture | Delta | 95% CI | p (max across seeds) | Seeds |
|---|---|---|---|---|---|---|
| did | efficientnet_b0_minus_vit_s | efficientnet_b0_vs_vit_s | -0.161 | -0.289 to -0.049 | 0.008 | 3 |
| did | resnet50_minus_efficientnet_b0 | resnet50_vs_efficientnet_b0 | -0.038 | -0.273 to 0.137 | 0.936 | 3 |
| did | resnet50_minus_vit_s | resnet50_vs_vit_s | -0.199 | -0.437 to -0.011 | 0.032 | 3 |
| paired | ce_random_strong_vs_ce | efficientnet_b0 | 0.007 | -0.173 to 0.177 | 0.526 | 3 |
| paired | ce_random_strong_vs_ce | resnet50 | -0.002 | -0.095 to 0.091 | 0.966 | 3 |
| paired | ce_random_strong_vs_ce | vit_s | 0.036 | -0.067 to 0.157 | 1.000 | 3 |
| paired | focal2_cb_vs_ce | efficientnet_b0 | -0.079 | -0.210 to 0.055 | 0.590 | 3 |
| paired | focal2_cb_vs_ce | resnet50 | -0.116 | -0.244 to 0.031 | 0.248 | 3 |
| paired | focal2_cb_vs_ce | vit_s | 0.082 | -0.038 to 0.237 | 0.440 | 3 |
