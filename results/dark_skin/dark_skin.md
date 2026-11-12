# Dark-skin robustness

Group metrics at three L* cutoffs, adjusted error models, and continuous trends.

## AUC by cutoff

| Cutoff | Group | n | Melanomas | AUC | FPR |
|---|---|---|---|---|---|
| 55 | dark | 43 | 6 | 0.805 | 0.126 |
| 55 | light | 644 | 81 | 0.868 | 0.050 |
| 55 | medium | 840 | 99 | 0.911 | 0.049 |
| 58 | dark | 106 | 12 | 0.891 | 0.086 |
| 58 | light | 644 | 81 | 0.868 | 0.050 |
| 58 | medium | 777 | 93 | 0.908 | 0.048 |
| 60 | dark | 179 | 23 | 0.881 | 0.073 |
| 60 | light | 644 | 81 | 0.868 | 0.050 |
| 60 | medium | 704 | 82 | 0.912 | 0.047 |

## Adjusted odds ratios

| Outcome | Architecture | Term | OR | 95% CI | p |
|---|---|---|---|---|---|
| error | EfficientNet-B0 | L | 1.017 | 0.840 to 1.182 | 0.777 |
| error | EfficientNet-B0 | delta_L | 1.550 | 1.316 to 1.828 | 0.000 |
| error | EfficientNet-B0 | log_area | 1.321 | 1.064 to 1.671 | 0.007 |
| error | EfficientNet-B0 | age | 1.388 | 1.183 to 1.728 | 0.000 |
| error | EfficientNet-B0 | sex_male | 1.088 | 0.918 to 1.297 | 0.319 |
| error | EfficientNet-B0 | site_back | 1.389 | 1.089 to 1.987 | 0.020 |
| error | EfficientNet-B0 | site_lower extremity | 1.378 | 1.068 to 2.013 | 0.013 |
| error | EfficientNet-B0 | site_other | 1.463 | 1.091 to 2.364 | 0.020 |
| error | EfficientNet-B0 | site_trunk | 0.773 | 0.482 to 1.112 | 0.206 |
| error | EfficientNet-B0 | site_upper extremity | 1.455 | 1.139 to 1.883 | 0.000 |
| error | ResNet-50 | L | 0.955 | 0.780 to 1.130 | 0.671 |
| error | ResNet-50 | delta_L | 1.622 | 1.341 to 2.020 | 0.000 |
| error | ResNet-50 | log_area | 1.316 | 1.071 to 1.646 | 0.020 |
| error | ResNet-50 | age | 1.351 | 1.103 to 1.688 | 0.000 |
| error | ResNet-50 | sex_male | 1.027 | 0.855 to 1.253 | 0.738 |
| error | ResNet-50 | site_back | 1.537 | 1.169 to 2.431 | 0.000 |
| error | ResNet-50 | site_lower extremity | 1.685 | 1.249 to 2.672 | 0.000 |
| error | ResNet-50 | site_other | 1.677 | 1.198 to 2.710 | 0.013 |
| error | ResNet-50 | site_trunk | 0.851 | 0.584 to 1.277 | 0.472 |
| error | ResNet-50 | site_upper extremity | 1.501 | 1.138 to 2.115 | 0.013 |
| error | ViT-S/16 | L | 0.920 | 0.761 to 1.065 | 0.319 |
| error | ViT-S/16 | delta_L | 1.692 | 1.437 to 2.028 | 0.000 |
| error | ViT-S/16 | log_area | 1.541 | 1.284 to 1.950 | 0.000 |
| error | ViT-S/16 | age | 1.497 | 1.237 to 1.846 | 0.000 |
| error | ViT-S/16 | sex_male | 1.010 | 0.832 to 1.225 | 0.924 |
| error | ViT-S/16 | site_back | 1.268 | 1.031 to 1.789 | 0.027 |
| error | ViT-S/16 | site_lower extremity | 1.212 | 0.956 to 1.681 | 0.159 |
| error | ViT-S/16 | site_other | 1.326 | 0.987 to 1.866 | 0.073 |
| error | ViT-S/16 | site_trunk | 0.569 | 0.367 to 0.822 | 0.013 |
| error | ViT-S/16 | site_upper extremity | 1.299 | 1.062 to 1.696 | 0.013 |

## False positive and false negative models

| Outcome | Architecture | Term | OR | 95% CI | p |
|---|---|---|---|---|---|
| false_positive | EfficientNet-B0 | L | 0.876 | 0.667 to 1.108 | 0.233 |
| false_positive | EfficientNet-B0 | delta_L | 1.688 | 1.391 to 2.139 | 0.000 |
| false_positive | ResNet-50 | L | 0.775 | 0.613 to 0.943 | 0.013 |
| false_positive | ResNet-50 | delta_L | 2.048 | 1.661 to 2.678 | 0.000 |
| false_positive | ViT-S/16 | L | 0.765 | 0.568 to 1.028 | 0.066 |
| false_positive | ViT-S/16 | delta_L | 2.422 | 1.938 to 3.142 | 0.000 |
| false_negative | EfficientNet-B0 | L | 1.662 | 1.164 to 2.342 | 0.000 |
| false_negative | EfficientNet-B0 | delta_L | 0.739 | 0.510 to 1.125 | 0.159 |
| false_negative | ResNet-50 | L | 1.739 | 1.110 to 2.579 | 0.013 |
| false_negative | ResNet-50 | delta_L | 0.579 | 0.352 to 1.017 | 0.060 |
| false_negative | ViT-S/16 | L | 1.754 | 1.140 to 2.602 | 0.013 |
| false_negative | ViT-S/16 | delta_L | 0.480 | 0.294 to 0.750 | 0.007 |
