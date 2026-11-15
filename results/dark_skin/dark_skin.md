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
| error | EfficientNet-B0 | L | 1.005 | 0.805 to 1.176 | 0.977 |
| error | EfficientNet-B0 | delta_L | 1.282 | 1.087 to 1.529 | 0.000 |
| error | EfficientNet-B0 | log_area | 1.408 | 1.170 to 1.757 | 0.000 |
| error | EfficientNet-B0 | age | 1.223 | 1.034 to 1.492 | 0.013 |
| error | EfficientNet-B0 | sex_male | 1.091 | 0.925 to 1.336 | 0.312 |
| error | EfficientNet-B0 | site_back | 1.150 | 0.875 to 1.662 | 0.365 |
| error | EfficientNet-B0 | site_lower extremity | 1.243 | 0.928 to 1.792 | 0.140 |
| error | EfficientNet-B0 | site_other | 1.049 | 0.741 to 1.673 | 0.764 |
| error | EfficientNet-B0 | site_trunk | 1.013 | 0.664 to 1.462 | 0.957 |
| error | EfficientNet-B0 | site_upper extremity | 1.249 | 0.964 to 1.602 | 0.113 |
| error | EfficientNet-B0 | source_vidir_modern | 1.181 | 0.939 to 1.526 | 0.120 |
| error | EfficientNet-B0 | source_vidir_molemax | 0.212 | 0.101 to 0.327 | 0.000 |
| error | EfficientNet-B0 | source_vienna_dias | 1.029 | 0.831 to 1.238 | 0.837 |
| error | ResNet-50 | L | 0.974 | 0.786 to 1.181 | 0.791 |
| error | ResNet-50 | delta_L | 1.295 | 1.050 to 1.625 | 0.007 |
| error | ResNet-50 | log_area | 1.435 | 1.179 to 1.793 | 0.000 |
| error | ResNet-50 | age | 1.203 | 1.006 to 1.499 | 0.047 |
| error | ResNet-50 | sex_male | 1.045 | 0.869 to 1.275 | 0.645 |
| error | ResNet-50 | site_back | 1.309 | 0.965 to 2.078 | 0.086 |
| error | ResNet-50 | site_lower extremity | 1.606 | 1.135 to 2.590 | 0.000 |
| error | ResNet-50 | site_other | 1.233 | 0.856 to 1.985 | 0.292 |
| error | ResNet-50 | site_trunk | 1.176 | 0.790 to 1.781 | 0.472 |
| error | ResNet-50 | site_upper extremity | 1.312 | 0.934 to 1.858 | 0.080 |
| error | ResNet-50 | source_vidir_modern | 1.398 | 1.092 to 1.826 | 0.000 |
| error | ResNet-50 | source_vidir_molemax | 0.193 | 0.075 to 0.321 | 0.000 |
| error | ResNet-50 | source_vienna_dias | 1.024 | 0.833 to 1.238 | 0.757 |
| error | ViT-S/16 | L | 0.923 | 0.745 to 1.093 | 0.319 |
| error | ViT-S/16 | delta_L | 1.362 | 1.142 to 1.633 | 0.000 |
| error | ViT-S/16 | log_area | 1.734 | 1.406 to 2.131 | 0.000 |
| error | ViT-S/16 | age | 1.297 | 1.094 to 1.573 | 0.000 |
| error | ViT-S/16 | sex_male | 1.031 | 0.852 to 1.258 | 0.811 |
| error | ViT-S/16 | site_back | 1.047 | 0.815 to 1.479 | 0.804 |
| error | ViT-S/16 | site_lower extremity | 1.102 | 0.818 to 1.530 | 0.605 |
| error | ViT-S/16 | site_other | 0.918 | 0.666 to 1.285 | 0.651 |
| error | ViT-S/16 | site_trunk | 0.726 | 0.514 to 0.991 | 0.040 |
| error | ViT-S/16 | site_upper extremity | 1.103 | 0.864 to 1.468 | 0.399 |
| error | ViT-S/16 | source_vidir_modern | 1.400 | 1.111 to 1.731 | 0.000 |
| error | ViT-S/16 | source_vidir_molemax | 0.199 | 0.072 to 0.314 | 0.000 |
| error | ViT-S/16 | source_vienna_dias | 1.035 | 0.866 to 1.216 | 0.711 |

## False positive and false negative models

| Outcome | Architecture | Term | OR | 95% CI | p |
|---|---|---|---|---|---|
| false_positive | EfficientNet-B0 | L | 0.814 | 0.619 to 0.997 | 0.053 |
| false_positive | EfficientNet-B0 | delta_L | 1.437 | 1.207 to 1.826 | 0.000 |
| false_positive | ResNet-50 | L | 0.754 | 0.600 to 0.934 | 0.000 |
| false_positive | ResNet-50 | delta_L | 1.590 | 1.241 to 2.004 | 0.000 |
| false_positive | ViT-S/16 | L | 0.722 | 0.535 to 0.949 | 0.013 |
| false_positive | ViT-S/16 | delta_L | 2.006 | 1.557 to 2.590 | 0.000 |
| false_negative | EfficientNet-B0 | L | 1.644 | 1.141 to 2.515 | 0.013 |
| false_negative | EfficientNet-B0 | delta_L | 0.719 | 0.460 to 1.141 | 0.159 |
| false_negative | ResNet-50 | L | 1.594 | 1.009 to 2.367 | 0.047 |
| false_negative | ResNet-50 | delta_L | 0.605 | 0.343 to 1.088 | 0.100 |
| false_negative | ViT-S/16 | L | 1.569 | 1.045 to 2.410 | 0.047 |
| false_negative | ViT-S/16 | delta_L | 0.471 | 0.289 to 0.746 | 0.007 |
