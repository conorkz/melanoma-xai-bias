# Cover letter

[Journal name]
[Editor name]
[Editorial office address]

[Date]

Dear [Editor name],

Please consider our manuscript, "Architecture-dependent effects of imbalance correction in melanoma classification: fairness, calibration, and explanation analysis," for publication in [Journal name].

The manuscript asks whether a standard imbalance correction changes the ranking of CNN and transformer models, whether calibration explains the change, and how the resulting models behave across skin tone and lesion contrast. We benchmark ResNet-50, EfficientNet-B0, and ViT-S/16 on HAM10000 under three pre-declared conditions, with three seeds, a lesion-aware split, and a validation-selected threshold at 95% specificity. Focal loss with class-balanced sampling improves the transformer and reduces both CNNs in the point estimates. The paired within-architecture intervals include zero, but the difference-in-differences between each CNN and the ViT excludes zero. The AUC ranking is stable, and focal loss roughly triples expected calibration error, which identifies calibration as the mechanism behind the thresholded reversal.

The fairness analysis is error-type-specific. The dark stratum has a higher false positive rate at L* cutoffs of 55, 58, and 60, while the dark AUC deficit is present only at the strictest cutoff. Adjusted models show that darker skin is associated with more false positives and lighter skin with more false negatives, and that lesion-skin contrast moves the two error types in opposite directions. A quantitative XAI analysis shows that all three architectures place less attribution inside the lesion on darker images, and that false negatives have lower attribution mass than false positives.

The manuscript also reports an external validation on 1,280 HIBA dermoscopic images. The CNN-transformer gap transfers, but sensitivity drops by 16 to 18 points, and HIBA contains no Fitzpatrick IV melanomas and no Fitzpatrick V-VI images. We state this data limitation explicitly rather than overstate the fairness claim.

We believe the work fits [Journal name] because it combines a controlled benchmark, a calibration analysis, a fairness analysis, and a quantitative explanation analysis in one reproducible study. The code, split, per-sample predictions, and table generators are publicly available.

This manuscript is original, has not been published elsewhere, and is not under consideration by another journal. All authors have approved the submission and declare no competing interests.

Sincerely,

Nursultan Zhanat
School of Computer Science and Technology, Tiangong University, Tianjin 300387, China
Tianjin Key Laboratory of Autonomous Intelligence Technology and Systems, Tiangong University, Tianjin 300387, China

On behalf of the authors, including the corresponding author:
Sun Baoshan
sunbaoshan@tiangong.edu.cn
Tiangong University, Tianjin 300387, China
