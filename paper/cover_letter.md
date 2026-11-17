# Cover letter

Journal of Imaging Informatics in Medicine
Editor-in-Chief
Editorial Office

17 November 2026

Dear Editor-in-Chief,

Please consider our manuscript, "When imbalance correction appears architecture-dependent: a protocol audit of melanoma classifiers with fairness, calibration, and explanation analysis," for publication in the Journal of Imaging Informatics in Medicine.

We set out to test whether focal loss with class-balanced sampling affects CNNs and a vision transformer differently in melanoma classification on HAM10000. Our first benchmark, which used one learning rate for all architectures and selected checkpoints by validation balanced accuracy at a 0.5 threshold, found a clear interaction: the correction raised ViT-S/16 sensitivity by 8.2 points and lowered ResNet-50 and EfficientNet-B0 by 11.7 and 7.9 points, every seed agreed, and the EfficientNet-B0 versus ViT interaction was significant. We then re-ran the benchmark with a learning rate selected per architecture, checkpoint selection by validation AUC, five seeds, and ablations that separate focal loss from class-balanced sampling. The interaction disappeared. The correction lowered sensitivity by 1.7 to 4.0 points in all three architectures, every interval included zero, and the transformer, which had been undertrained, became the strongest model.

We report both protocols because the first result passed the checks that usually support such a claim: seed agreement, a significant interaction, a calibration analysis, and an external dataset. We think this makes the manuscript useful to readers of JIIM who build or review architecture comparisons in medical imaging. The paper ends with three concrete recommendations for such benchmarks.

The fairness findings did not depend on the protocol. In error models adjusted for acquisition source and lesion contrast, lighter perilesional skin was associated with more false negatives and darker skin with more false positives in all three architectures. The darkest stratum had the highest false positive rate, the worst calibration, and the lowest Grad-CAM pointing-game accuracy, although the stratum is small and the per-run AUC differences are not significant. On 1,280 external HIBA dermoscopic images, imbalance correction had no effect, while strong augmentation improved sensitivity in every seed of every architecture. HIBA contains no Fitzpatrick IV melanomas and no Fitzpatrick V-VI images, and we state that limitation explicitly.

The code, split, per-sample predictions for all 102 benchmark runs, and table generators are publicly available.

This manuscript is original, has not been published elsewhere, and is not under consideration by another journal. All authors have approved the submission and declare no competing interests.

A blinded version of the manuscript and a separate title page with the author information and statements are included for the journal's double-blind review process.

Sincerely,

Sun Baoshan
School of Computer Science and Technology, Tiangong University, Tianjin 300387, China
Tianjin Key Laboratory of Autonomous Intelligence Technology and Systems, Tiangong University, Tianjin 300387, China
sunbaoshan@tiangong.edu.cn
ORCID: 0000-0003-2193-3232

On behalf of the co-author:
Nursultan Zhanat
zhanat.nursultan1@gmail.com
ORCID: 0009-0006-7839-4785
