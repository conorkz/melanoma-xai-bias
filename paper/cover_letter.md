# Cover letter

Journal of Imaging Informatics in Medicine
Editor-in-Chief
Editorial Office

12 November 2026

Dear Editor-in-Chief,

Please consider our manuscript, "Architecture-dependent effects of imbalance correction in melanoma classification: a controlled audit with fairness, calibration, and explanation analysis," for publication in the Journal of Imaging Informatics in Medicine.

The manuscript asks whether a standard imbalance correction changes CNN and transformer performance, whether calibration explains the change, and how the models behave across skin tone and lesion contrast. We benchmark ResNet-50, EfficientNet-B0, and ViT-S/16 on HAM10000 under three pre-declared conditions, with three seeds, a lesion-aware split, and a validation-selected threshold at 95% specificity. Focal loss with class-balanced sampling changes the sensitivity of all three architectures in a consistent direction across seeds, but every within-architecture 95% t interval includes zero. The EfficientNet-B0 versus ViT interaction is significant at the seed level; the ResNet-50 versus ViT interaction is not. The AUC ordering is preserved, but the ViT closes the gap under focal loss. Calibration error rises 3.0x to 4.4x for the CNNs and 1.36x for the ViT, so calibration tracks the CNN loss rather than the ViT gain.

The fairness analysis is error-type-specific and deliberately cautious. The dark stratum has a higher false positive rate at L* cutoffs of 55, 58, and 60, but none of the dark-versus-light or dark-versus-medium AUC differences is significant. The false-negative L* association is significant for all three architectures; the false-positive association is significant only for ResNet-50. The XAI analysis uses Grad-CAM with the true class as target. All three architectures place less attribution inside the lesion on darker images, but the mass-error relationship differs by architecture and the false-negative versus false-positive mass values do not separate consistently.

The manuscript also reports external validation on 1,280 HIBA dermoscopic images. The CNN sensitivity loss under focal loss is consistent in direction, while the ViT gain is seed-dependent. Strong augmentation, which was near-neutral in-domain, is the best external condition for all three architectures. HIBA contains no Fitzpatrick IV melanomas and no Fitzpatrick V-VI images, and we state that limitation explicitly rather than overstate the fairness claim.

We believe the work fits JIIM because it combines a controlled benchmark, a calibration analysis, a fairness audit, and a quantitative explanation analysis in one reproducible study. The code, split, per-sample predictions, and table generators are publicly available.

This manuscript is original, has not been published elsewhere, and is not under consideration by another journal. All authors have approved the submission and declare no competing interests.

A blinded version of the manuscript and a separate title page with the author information and statements are included for the journal's double-blind review process.

Sincerely,

Nursultan Zhanat
School of Computer Science and Technology, Tiangong University, Tianjin 300387, China
Tianjin Key Laboratory of Autonomous Intelligence Technology and Systems, Tiangong University, Tianjin 300387, China
zhanat.nursultan1@gmail.com

On behalf of the authors, including the corresponding author:
Sun Baoshan
sunbaoshan@tiangong.edu.cn
Tiangong University, Tianjin 300387, China
