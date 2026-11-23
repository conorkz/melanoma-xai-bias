# Cover letter

Health Information Science and Systems
Prof. Yanchun Zhang, Editor-in-Chief

22 November 2026

Dear Prof. Zhang,

Please consider our manuscript, "When imbalance correction appears architecture-dependent: a protocol audit of melanoma classifiers with fairness, calibration, and explanation analysis," for publication as a research article in Health Information Science and Systems.

Published comparisons of medical image classifiers often train every architecture with one shared protocol and then report that a training intervention helps some architectures and hurts others. We tested whether this kind of finding holds up. Our first benchmark on HAM10000, which used one learning rate for all architectures and selected checkpoints by validation balanced accuracy at a 0.5 threshold, found a clear interaction: focal loss with class-balanced sampling raised ViT-S/16 sensitivity by 8.2 points and lowered ResNet-50 and EfficientNet-B0 by 11.7 and 7.9 points, every seed agreed, and the EfficientNet-B0 versus ViT interaction was significant. We then re-ran the benchmark with a learning rate selected per architecture, checkpoint selection by validation AUC, five seeds, and ablations that separate focal loss from class-balanced sampling. The interaction disappeared. The correction lowered sensitivity by 1.7 to 4.0 points in all three architectures, every interval included zero, and the transformer, which had been undertrained, had the highest point-estimate AUC.

We report both protocols because the first result passed the checks that usually support such a claim: seed agreement, a significant interaction, a calibration analysis, and an external dataset. A 2026 benchmark of imbalance methods across six CNNs reached an architecture-specific conclusion with one fixed learning rate, so the issue is current. The paper ends with three concrete recommendations for researchers who build or review model comparisons for clinical decision support.

The fairness findings did not depend on the protocol. In error models adjusted for acquisition source and lesion contrast, lighter perilesional skin was associated with more false negatives in both CNNs and darker skin with more false positives in all three architectures. The darkest stratum had the highest false positive rate, the worst calibration, and the lowest Grad-CAM pointing-game accuracy, although the stratum is small and we say so. On 1,280 external HIBA dermoscopic images, imbalance correction had no detectable effect, while strong augmentation improved AUC in every seed and sensitivity in 14 of 15 runs.

We think the work fits the journal's interest in data analysis for health decision making and in trustworthy, generalizable AI. The code, split, per-sample predictions for all 102 benchmark runs, and table generators are publicly available.

This manuscript is original, has not been published elsewhere, and is not under consideration by another journal. All authors have approved the submission and declare no competing interests.

Sincerely,

Sun Baoshan
School of Computer Science and Technology, Tiangong University, Tianjin 300387, China
Tianjin Key Laboratory of Autonomous Intelligence Technology and Systems, Tiangong University, Tianjin 300387, China
sunbaoshan@tiangong.edu.cn
ORCID: 0009-0008-2967-9201

On behalf of the co-author:
Nursultan Zhanat
zhanat.nursultan1@gmail.com
ORCID: 0009-0006-7839-4785
