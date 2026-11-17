# Dark-skin robustness

Group metrics at three L* cutoffs, adjusted error models, and continuous trends.

## AUC by cutoff

| Cutoff | Group | n | Melanomas | AUC | FPR |
|---|---|---|---|---|---|
| 55 | dark | 43 | 6 | 0.826 | 0.115 |
| 55 | light | 644 | 81 | 0.894 | 0.052 |
| 55 | medium | 840 | 99 | 0.932 | 0.054 |
| 58 | dark | 106 | 12 | 0.897 | 0.092 |
| 58 | light | 644 | 81 | 0.894 | 0.052 |
| 58 | medium | 777 | 93 | 0.931 | 0.052 |
| 60 | dark | 179 | 23 | 0.896 | 0.080 |
| 60 | light | 644 | 81 | 0.894 | 0.052 |
| 60 | medium | 704 | 82 | 0.934 | 0.052 |

## Adjusted odds ratios

| Outcome | Architecture | Term | OR | 95% CI | p |
|---|---|---|---|---|---|
| error | EfficientNet-B0 | L | 1.012 | 0.824 to 1.179 | 0.897 |
| error | EfficientNet-B0 | delta_L | 1.250 | 1.042 to 1.509 | 0.007 |
| error | EfficientNet-B0 | log_area | 1.453 | 1.168 to 1.758 | 0.000 |
| error | EfficientNet-B0 | age | 1.226 | 1.025 to 1.485 | 0.000 |
| error | EfficientNet-B0 | sex_male | 1.107 | 0.915 to 1.323 | 0.226 |
| error | EfficientNet-B0 | site_back | 1.047 | 0.803 to 1.504 | 0.784 |
| error | EfficientNet-B0 | site_lower extremity | 1.150 | 0.847 to 1.740 | 0.372 |
| error | EfficientNet-B0 | site_other | 0.948 | 0.674 to 1.444 | 0.824 |
| error | EfficientNet-B0 | site_trunk | 0.875 | 0.618 to 1.247 | 0.545 |
| error | EfficientNet-B0 | site_upper extremity | 1.105 | 0.833 to 1.504 | 0.445 |
| error | EfficientNet-B0 | source_vidir_modern | 1.269 | 1.047 to 1.596 | 0.013 |
| error | EfficientNet-B0 | source_vidir_molemax | 0.239 | 0.110 to 0.366 | 0.000 |
| error | EfficientNet-B0 | source_vienna_dias | 1.068 | 0.904 to 1.240 | 0.445 |
| error | ResNet-50 | L | 0.935 | 0.757 to 1.128 | 0.452 |
| error | ResNet-50 | delta_L | 1.234 | 1.024 to 1.483 | 0.033 |
| error | ResNet-50 | log_area | 1.466 | 1.219 to 1.825 | 0.000 |
| error | ResNet-50 | age | 1.237 | 1.040 to 1.532 | 0.013 |
| error | ResNet-50 | sex_male | 1.011 | 0.844 to 1.210 | 0.910 |
| error | ResNet-50 | site_back | 1.204 | 0.901 to 1.693 | 0.259 |
| error | ResNet-50 | site_lower extremity | 1.344 | 0.976 to 2.014 | 0.106 |
| error | ResNet-50 | site_other | 1.049 | 0.734 to 1.573 | 0.791 |
| error | ResNet-50 | site_trunk | 1.045 | 0.596 to 1.601 | 0.817 |
| error | ResNet-50 | site_upper extremity | 1.194 | 0.892 to 1.631 | 0.213 |
| error | ResNet-50 | source_vidir_modern | 1.418 | 1.170 to 1.795 | 0.000 |
| error | ResNet-50 | source_vidir_molemax | 0.185 | 0.062 to 0.307 | 0.000 |
| error | ResNet-50 | source_vienna_dias | 1.062 | 0.891 to 1.251 | 0.412 |
| error | ViT-S/16 | L | 0.946 | 0.763 to 1.148 | 0.571 |
| error | ViT-S/16 | delta_L | 1.191 | 0.976 to 1.468 | 0.126 |
| error | ViT-S/16 | log_area | 1.338 | 1.089 to 1.638 | 0.000 |
| error | ViT-S/16 | age | 1.222 | 1.035 to 1.484 | 0.007 |
| error | ViT-S/16 | sex_male | 1.076 | 0.893 to 1.294 | 0.525 |
| error | ViT-S/16 | site_back | 1.091 | 0.807 to 1.517 | 0.585 |
| error | ViT-S/16 | site_lower extremity | 1.173 | 0.790 to 1.709 | 0.432 |
| error | ViT-S/16 | site_other | 0.895 | 0.614 to 1.290 | 0.578 |
| error | ViT-S/16 | site_trunk | 1.253 | 0.819 to 1.813 | 0.259 |
| error | ViT-S/16 | site_upper extremity | 1.012 | 0.755 to 1.341 | 0.930 |
| error | ViT-S/16 | source_vidir_modern | 1.090 | 0.882 to 1.346 | 0.419 |
| error | ViT-S/16 | source_vidir_molemax | 0.156 | 0.052 to 0.258 | 0.000 |
| error | ViT-S/16 | source_vienna_dias | 1.022 | 0.841 to 1.213 | 0.824 |

## False positive and false negative models

| Outcome | Architecture | Term | OR | 95% CI | p |
|---|---|---|---|---|---|
| false_positive | EfficientNet-B0 | L | 0.795 | 0.625 to 0.950 | 0.027 |
| false_positive | EfficientNet-B0 | delta_L | 1.394 | 1.138 to 1.751 | 0.000 |
| false_positive | ResNet-50 | L | 0.654 | 0.497 to 0.792 | 0.000 |
| false_positive | ResNet-50 | delta_L | 1.519 | 1.229 to 1.938 | 0.000 |
| false_positive | ViT-S/16 | L | 0.785 | 0.595 to 0.958 | 0.020 |
| false_positive | ViT-S/16 | delta_L | 1.325 | 1.071 to 1.670 | 0.013 |
| false_negative | EfficientNet-B0 | L | 2.013 | 1.476 to 2.855 | 0.000 |
| false_negative | EfficientNet-B0 | delta_L | 0.596 | 0.370 to 0.945 | 0.027 |
| false_negative | ResNet-50 | L | 2.045 | 1.391 to 3.166 | 0.000 |
| false_negative | ResNet-50 | delta_L | 0.492 | 0.307 to 0.779 | 0.007 |
| false_negative | ViT-S/16 | L | 1.423 | 0.979 to 2.271 | 0.066 |
| false_negative | ViT-S/16 | delta_L | 0.526 | 0.292 to 0.953 | 0.033 |
