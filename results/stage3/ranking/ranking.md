# Paired ranking tests

Sensitivity changes paired by test image. Intervals are 95% bootstrap intervals.

| Test | Comparison | Architecture | Delta | 95% CI | p (max across seeds) | Seeds |
|---|---|---|---|---|---|---|
| did | efficientnet_b0_minus_vit_s | efficientnet_b0_vs_vit_s | 0.006 | -0.246 to 0.232 | 0.942 | 5 |
| did | resnet50_minus_efficientnet_b0 | resnet50_vs_efficientnet_b0 | -0.023 | -0.266 to 0.185 | 0.500 | 5 |
| did | resnet50_minus_vit_s | resnet50_vs_vit_s | -0.016 | -0.197 to 0.201 | 1.000 | 5 |
| paired | ce_random_strong_vs_ce | efficientnet_b0 | -0.014 | -0.131 to 0.136 | 0.988 | 5 |
| paired | ce_random_strong_vs_ce | resnet50 | -0.016 | -0.111 to 0.109 | 0.618 | 5 |
| paired | ce_random_strong_vs_ce | vit_s | -0.015 | -0.100 to 0.087 | 0.964 | 5 |
| paired | focal2_cb_vs_ce | efficientnet_b0 | -0.017 | -0.179 to 0.180 | 0.278 | 5 |
| paired | focal2_cb_vs_ce | resnet50 | -0.040 | -0.168 to 0.099 | 0.990 | 5 |
| paired | focal2_cb_vs_ce | vit_s | -0.024 | -0.164 to 0.108 | 0.612 | 5 |
