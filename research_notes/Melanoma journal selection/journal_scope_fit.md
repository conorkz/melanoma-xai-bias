# Melanoma AI journal selection: scope fit and evidence bar

Paper profile used for this assessment: ResNet-50, EfficientNet-B0 and ViT-S/16 on HAM10000; three imbalance-correction conditions; calibration; paired ranking tests; error-type-specific skin-tone fairness; Grad-CAM; external validation on HIBA; three seeds. This is a strong mid-tier benchmarking and audit paper, not a prospective clinical validation study.

The 16-journal set below is intentionally mixed. Four venues are high-bar clinical or digital medicine journals that are useful as aspirational targets but are unlikely to accept this paper as it stands. Six are realistic mid-tier dermatology, imaging or AI journals. Six are broad or applied open-access venues with a technical-soundness bar. The best realistic targets are the Journal of Imaging Informatics in Medicine, Computer Methods and Programs in Biomedicine, and BMC Medical Imaging.

Method note: I combined publisher aims-and-scope pages, Europe PMC searches limited to 2024-2026, Crossref checks for retractions, and legitimacy sources (Beall's List, Think Check Submit, DOAJ, COPE). I did not use impact factor as a selection criterion. Where a publisher page stated an impact factor, I report it with the year given on the page.

## 1. Which dermatology, medical imaging, health informatics, and general AI/medicine journals explicitly publish melanoma or skin-lesion AI work?

### Takeaway

Sixteen journals have a credible record of publishing melanoma or skin-lesion AI work, or an explicit scope that covers it. The strongest scope matches are the Journal of Imaging Informatics in Medicine, Computer Methods and Programs in Biomedicine, BMC Medical Imaging, and Skin Research and Technology, but Skin Research and Technology carries a separate AI retraction risk discussed in section 5.

### The 16-journal set

Tier definitions used here:
- T1: high-bar clinical or digital medicine venue. Prospective validation, multicenter data, clinical impact or broad generalizability expected.
- T2: realistic mid-tier specialty venue. Retrospective comparative studies, benchmark audits, external validation and methodological rigor are publishable.
- T3: broad or applied open-access venue. Technical soundness and reproducibility matter more than perceived impact or novelty.

| # | Journal (publisher) | Tier | Scope fit for this paper | Evidence bar | Realism | 2024-2026 melanoma or skin AI signal |
|---|---|---|---|---|---|---|
| 1 | JAMA Dermatology (AMA) | T1 | High for clinical melanoma AI; low for a pure benchmark | Prospective or clinical-impact evidence; multicenter studies; evidence synthesis | Low | Federated learning for melanoma diagnostics, 2024; 2026 systematic review of prospective AI melanoma diagnostics |
| 2 | British Journal of Dermatology (Wiley/BAD) | T1 | High for clinical dermatology; moderate for AI methods | Prospective clinical trials, translational or clinical relevance, originality | Low | Prospective AI melanoma decision-support trial in primary care, 2024 |
| 3 | npj Digital Medicine (Nature Portfolio) | T1 | High for digital medicine and fairness; moderate for single-dataset imaging | Generalizability, implementation, broad significance | Low-medium | Bias recognition and mitigation review, 2025; dermatology multimodal AI, 2026 |
| 4 | JEADV (Wiley/EADV) | T1 | High for clinical dermatology; moderate for AI benchmarks | Multicenter or clinically relevant retrospective work; prospective work for clinical claims | Low-medium | Multicentric retrospective AI BCC study with LC-OCT, 2026; melanoma AI meta-analysis, 2025 |
| 5 | Dermatology (Karger) | T2 | High for clinical dermatology; moderate for AI | Retrospective comparative studies with clinical framing | Medium-high | GPT-4 Omni versus specialised neural networks on smartphone images, 2026 |
| 6 | JAAD International (Elsevier/AAD) | T2 | High for clinical dermatology; moderate for AI | Clinical relevance; brief reports and comparative studies | Medium | Multimodal LLMs for melanoma versus nevi, 2025 |
| 7 | Skin Research and Technology (Wiley) | T2 | Very high on scope: dermoscopy, imaging, image analysis, AI | Technical and clinical soundness; current AI retraction cluster raises scrutiny | Medium, risk-adjusted lower | Scope explicitly includes AI and dermoscopy; retracted melanoma deep learning paper, 2024/2025 |
| 8 | Computers in Biology and Medicine (Elsevier) | T2 | High for AI and imaging methods; moderate for clinical dermatology | Methodological novelty, strong baselines, reproducibility | Medium | Skin-CAD explainable skin cancer classifier, 2024; melanoma deep learning systematic review, 2025 |
| 9 | Computer Methods and Programs in Biomedicine (Elsevier) | T2 | High for methods, validation and fairness | Rigorous validation, reproducibility, clinical relevance | Medium-high | Skin colour bias in deep learning skin lesion segmentation, 2024; self-supervised multimodal skin lesion classification, 2025 |
| 10 | Journal of Imaging Informatics in Medicine (Springer/SIIM) | T2 | Very high: AI, computer-aided diagnosis, fairness and bias are explicit scope items | Imaging informatics relevance, technical validation | High | Swin transformer skin cancer classifier, 2024; open Dermatology collection; JIF 3.1 (2025) |
| 11 | Scientific Reports (Nature Portfolio) | T3 | High for broad science; moderate for clinical dermatology | Technical soundness, not perceived importance | High | Skin tone bias reduction with sketch-guided fusion, 2026; many HAM10000 benchmarks |
| 12 | PLOS ONE (PLOS) | T3 | High for broad science; moderate for clinical dermatology | Technical rigor; derivative work needs explicit justification | High | SCGAN-generated HAM10000 fairness extension, 2026; explainable fusion models, 2026 |
| 13 | BMC Medical Imaging (BMC/Springer Nature) | T3 | High for imaging; moderate for clinical dermatology | Scientific validity, not perceived impact | High | Optimised CNN for skin cancer classification, 2024; IF 3.2 (2024) |
| 14 | Diagnostics (MDPI) | T3 | High for diagnostics, imaging and AI | Technical soundness; fast review; APC | High | Reliability-audited explainable AI for skin lesion triage under domain shift, 2026 |
| 15 | Frontiers in Medicine (Frontiers) | T3 | High for broad medicine and dermatology | Computational studies of public data must include appropriate validation | High | Explainable EfficientNet-B0 framework for HAM10000 segmentation and classification, 2025 |
| 16 | JMIR Dermatology (JMIR) | T3 | High for digital dermatology and AI; moderate for imaging methods | Digital health relevance, transparency, APC | High | Multimodal LLMs for melanoma across different skin tones, 2024 |

### Journal-by-journal scope and evidence bar

#### T1: high-bar venues

**JAMA Dermatology.** The journal publishes clinical dermatology, melanoma and digital health work, including AI. Its recent AI melanoma content is either multicenter or evidence synthesis, not single-dataset benchmarking. A 2026 systematic review examined prospective evidence on AI-assisted melanoma diagnostics, which is a useful signal of the current evidence bar. A HAM10000 benchmark with one external dermoscopic set is unlikely to clear desk review unless it is reframed as a clinically important audit with a strong patient-level question. Realism: low.

**British Journal of Dermatology.** The scope statement emphasises the highest-quality dermatological research, clinical trials, translational research, systematic reviews and interest to a multidisciplinary audience; the overriding criteria are scientific merit, originality and interest. A 2024 prospective real-life clinical trial of AI-based decision support for melanoma detection in primary care is the kind of AI paper the journal publishes. A retrospective benchmark is a poor fit unless it is a research letter with a clear clinical message. Realism: low.

**npj Digital Medicine.** This is a digital medicine journal with a strong fairness and validation agenda. It published a 2025 review on bias recognition and mitigation in AI healthcare and a 2026 paper on multimodal dermatology AI trained from unimodal datasets. The bar is generalizability, implementation and broad significance, not accuracy on a single public dataset. Realism: low-medium.

**JEADV.** The journal publishes clinical and translational dermatology and has an explicit AI in dermatology collection. It published a 2026 multicentric retrospective AI study for basal cell carcinoma diagnosis with LC-OCT and a 2025 meta-analysis of AI melanoma diagnosis. Multicenter retrospective work can pass, but a single-dataset benchmark is unlikely to survive desk review. Realism: low-medium.

#### T2: realistic mid-tier venues

**Dermatology (Karger).** A clinical and investigative dermatology journal with a broad scope, including clinical and laboratory findings. Its stated impact factor is 2.7 (Q2 Dermatology, SCIE) on the journal page. It published a 2026 comparison of GPT-4 Omni against specialised neural networks for malignant lesion identification from smartphone images and structured clinical data. Retrospective comparative AI work is in scope if the clinical question is clear. Realism: medium-high.

**JAAD International.** The open-access companion title to the Journal of the American Academy of Dermatology. It publishes brief reports and comparative AI studies, including a 2025 evaluation of multimodal LLMs for distinguishing melanoma from nevi in clinical and dermoscopic images. Clinical relevance matters more than methods novelty. Realism: medium.

**Skin Research and Technology.** The scope explicitly includes dermoscopy, digital imaging, image analysis and artificial intelligence, which makes it one of the closest topical matches in the set. The journal is legitimate and published by Wiley. However, a Crossref check found multiple retractions of AI papers in 2024-2026, including a melanoma deep learning paper. That does not prove the journal is predatory, but it is a due-diligence signal for an AI submission. Realism: medium, lower after adjusting for reputational risk.

**Computers in Biology and Medicine.** The scope covers computer applications in bioscience and medicine, with sections for foundations of AI in medicine and medical imaging. It published Skin-CAD, an explainable deep learning classifier for skin cancer from dermoscopic images, in 2024. The journal explicitly rejects minor architecture changes with small performance gains and no comparison to real state-of-the-art methods, so the paper must be framed as a benchmark or audit, not as a new model. Realism: medium.

**Computer Methods and Programs in Biomedicine.** The journal focuses on computer methods and programs in biomedicine with a strong validation culture. It published a 2024 study on skin colour bias in deep learning-based skin lesion segmentation, which is the closest methodological precedent in the set, and a 2025 self-supervised multimodal skin lesion classification paper. Fairness, calibration and validation work is welcome if the protocol is rigorous. Realism: medium-high.

**Journal of Imaging Informatics in Medicine.** The Springer/SIIM journal, formerly the Journal of Digital Imaging. Its scope explicitly lists AI, machine learning, deep learning, computer-aided diagnosis, and AI safety, fairness and bias. It published a 2024 Swin transformer skin cancer classifier, currently has an open Dermatology collection, and states a JIF of 3.1 (2025). This is the best scope-to-bar match in the set: the paper's fairness audit, calibration analysis and external validation fit the journal's stated interests, and the evidence bar is realistic for a strong master's project. Realism: high.

#### T3: broad and applied open-access venues

**Scientific Reports.** A Nature Portfolio broad open-access journal. Its stated criteria are scientific validity and technical soundness; manuscripts are not assessed on perceived importance, significance or impact. It published a 2026 paper on reducing skin tone bias in dermatology AI via sketch-guided multimodal fusion and many HAM10000 benchmark papers. Realism: high.

**PLOS ONE.** A broad open-access journal. Its criteria are scientific validity, strong methodology and high ethical standards, not perceived significance. It published a 2026 SCGAN-generated skin-tone-balanced extension of HAM10000 and a 2026 explainable fusion paper for multi-class skin lesion classification. Derivative work needs a clear scientific rationale; a pure "another HAM10000 benchmark" submission may be rejected on that ground. Realism: high, with a framing risk.

**BMC Medical Imaging.** A BMC series imaging journal. It explicitly does not make editorial decisions on perceived interest or potential impact; manuscripts are considered if they are scientifically valid, with a sound research question, appropriate methods and adherence to community standards. It published a 2024 optimised CNN for automated dermatological lesion classification and states an impact factor of 3.2 (2024). Realism: high.

**Diagnostics.** An MDPI diagnostics journal whose scope explicitly includes medical imaging and machine learning or AI in diagnostics. It published a 2026 reliability-audited explainable AI study for multi-class skin lesion triage under cross-device domain shift. Fast review and APC apply; perceived prestige is lower than the T1 and T2 venues. Realism: high.

**Frontiers in Medicine.** A broad-scope medicine journal. Its policy states that computational studies of public data must include appropriate validation. The HIBA external validation helps satisfy that policy. It published a 2025 explainable hybrid deep learning framework for skin lesion segmentation and multi-class classification on HAM10000. Realism: high.

**JMIR Dermatology.** A digital dermatology journal. It published a 2024 study assessing multimodal LLMs for melanoma identification across different skin tones. Strong fit for a fairness and digital health framing; less fit for a pure imaging methods contribution. Realism: high.

#### Health informatics venues

General health informatics journals are a weak fit for this paper. They publish fairness methods and reviews, but their unit of analysis is usually the health system, clinician workflow or patient outcome, not a dermoscopic classifier. JAMIA Open published a 2025 scoping review on algorithmic individual fairness in healthcare, which shows the topic is welcome, but I found no 2024-2026 melanoma image benchmark there. International Journal of Medical Informatics and BMC Medical Informatics and Decision Making are similarly oriented toward informatics questions rather than imaging methods. JMIR Dermatology is the exception in the set because it combines dermatology and digital health and has published melanoma AI fairness work. If the authors want a health informatics framing, JMIR Dermatology is the realistic choice; JAMIA Open is a stretch and would require a workflow or implementation question. [JAMIA Open fairness review](https://doi.org/10.1093/jamiaopen/ooae149).

### Cited Findings

- JAMA Dermatology published a 2024 federated learning study for melanoma diagnostics and a 2026 systematic review of prospective evidence on AI-assisted melanoma diagnostics: [JAMA Dermatology federated learning](https://doi.org/10.1001/jamadermatol.2023.5550); [JAMA Dermatology systematic review](https://jamanetwork.com/journals/jamadermatology/fullarticle/2846489).
- BJD published a 2024 prospective real-life clinical trial of AI-based decision support for melanoma detection in primary care: [BJD prospective trial](https://doi.org/10.1093/bjd/ljae021).
- npj Digital Medicine published a 2025 review on bias recognition and mitigation in AI healthcare and a 2026 dermatology multimodal AI paper: [npj Digital Medicine bias review](https://doi.org/10.1038/s41746-025-01503-7); [npj Digital Medicine dermatology paper](https://doi.org/10.1038/s41746-026-03040-3).
- JEADV published a 2026 multicentric retrospective AI BCC study and a 2025 melanoma AI meta-analysis: [JEADV BCC study](https://doi.org/10.1111/jdv.70099); [JEADV meta-analysis](https://doi.org/10.1111/jdv.20781).
- Dermatology (Karger) states an impact factor of 2.7 (Q2 Dermatology, SCIE) and published a 2026 GPT-4 Omni versus specialised neural network comparison: [Dermatology journal page](https://karger.com/drm); [Dermatology AI comparison](https://doi.org/10.1159/000550153).
- JAAD International published a 2025 multimodal LLM melanoma versus nevi study: [JAAD International](https://doi.org/10.1016/j.jdin.2025.08.008).
- Skin Research and Technology's scope explicitly includes dermoscopy, digital imaging, image analysis and artificial intelligence: [SRT scope](https://onlinelibrary.wiley.com/page/journal/16000846/homepage/productinformation.html).
- Computers in Biology and Medicine published Skin-CAD, an explainable HAM10000 skin cancer classifier, in 2024: [Skin-CAD](https://doi.org/10.1016/j.compbiomed.2024.108798).
- Computer Methods and Programs in Biomedicine published a 2024 study of skin colour bias in skin lesion segmentation: [CMPB fairness study](https://doi.org/10.1016/j.cmpb.2024.108044).
- The Journal of Imaging Informatics in Medicine scope explicitly includes AI, computer-aided diagnosis, and AI safety, fairness and bias: [JIIM scope](https://link.springer.com/journal/10278/aims-and-scope); it published a 2024 Swin transformer skin cancer classifier: [JIIM Swin transformer](https://doi.org/10.1007/s10278-024-01140-8); it states a JIF of 3.1 (2025) and has an open Dermatology collection: [JIIM journal page](https://link.springer.com/journal/10278).
- Scientific Reports states that papers must be scientifically valid and technically sound, not assessed on perceived importance: [Scientific Reports referee guide](https://www.nature.com/srep/guide-to-referees); it published a 2026 skin tone bias paper: [Scientific Reports fairness paper](https://doi.org/10.1038/s41598-026-59316-x).
- PLOS ONE evaluates scientific validity and methodology, not perceived significance: [PLOS ONE criteria](https://journals.plos.org/plosone/s/criteria-for-publication); it published a 2026 SCGAN HAM10000 fairness extension: [PLOS ONE SCGAN](https://doi.org/10.1371/journal.pone.0331081).
- BMC Medical Imaging does not make decisions on perceived interest or impact: [BMC Medical Imaging scope](https://bmcmedimaging.biomedcentral.com/submission-guidelines/aims-and-scope); it states an impact factor of 3.2 (2024): [BMC Medical Imaging metrics](https://bmcmedimaging.biomedcentral.com/submission-guidelines/peer-review-policy); it published a 2024 optimised CNN skin cancer classifier: [BMC Medical Imaging CNN](https://doi.org/10.1186/s12880-024-01356-8).
- Diagnostics scope includes medical imaging and AI in diagnostics: [Diagnostics scope](https://www.mdpi.com/journal/diagnostics/about); it published a 2026 reliability-audited XAI skin lesion triage study: [Diagnostics XAI](https://doi.org/10.3390/diagnostics16203368).
- Frontiers in Medicine requires appropriate validation for computational studies of public data: [Frontiers in Medicine scope](https://www.frontiersin.org/journals/medicine/about); it published a 2025 explainable EfficientNet-B0 HAM10000 framework: [Frontiers in Medicine paper](https://doi.org/10.3389/fmed.2025.1681542).
- JMIR Dermatology published a 2024 multimodal LLM melanoma study across skin tones: [JMIR Dermatology](https://doi.org/10.2196/55508).

### Inferences

- The 16 journals split into three practical groups. T1 venues are useful for a "stretch" submission only if the paper is reframed around clinical impact. T2 venues reward a clear methodological or fairness question. T3 venues reward technical soundness and reproducibility.
- The phrase "explicitly publishes melanoma AI" is not enough. The relevant question is whether the journal has published a paper with a similar evidence design in the last two years. The strongest signals are CMPB's skin colour bias paper, JIIM's Swin transformer paper, and the broad-OA fairness papers in Scientific Reports and PLOS ONE.
- A journal can be legitimate and still be a poor fit. Skin Research and Technology is the clearest example: the scope is almost perfect, but the AI retraction cluster changes the risk calculation.

### Gaps

- I did not systematically verify every journal's 2025 Journal Citation Reports impact factor, acceptance rate or APC. Acceptance rates are generally not public. Impact factors are reported only where a publisher page stated a value, and those values may not reflect the latest JCR release.
- I did not retrieve the full aims and scope pages for JAMA Dermatology, npj Digital Medicine, JAAD International or JMIR Dermatology. Their inclusion is based on recent articles and publisher-level knowledge, not a verbatim scope quote.
- I did not search every possible dermatology or informatics title. The set is a curated shortlist, not an exhaustive census.

## 2. Which journals accept benchmark, fairness, calibration, or XAI studies without prospective clinical validation?

### Takeaway

Eleven of the 16 have direct 2024-2026 evidence of publishing retrospective benchmark, fairness or XAI work without prospective clinical validation. Four T1 clinical venues expect a higher level of clinical evidence, although npj Digital Medicine is more methods-tolerant. Calibration is the weakest-supported contribution across the set: it is usually a secondary metric, and the closest calibration-focused melanoma paper was published in MELBA, not in any of the 16.

### Acceptance matrix

The ratings below are assessments based on published examples and scope statements, not on editorial promises or acceptance-rate data. "Yes" means a directly relevant paper or an explicit scope commitment was found. "Likely" means the scope supports it but no directly comparable paper was found. "No" means the journal's recent evidence bar makes a single-dataset benchmark unlikely to pass.

| Journal | Benchmark | Fairness | Calibration | XAI | Prospective validation required? |
|---|---|---|---|---|---|
| JAMA Dermatology | No for single-dataset | Likely as part of a clinical study | Likely as secondary | Likely as secondary | Yes for clinical claims |
| British Journal of Dermatology | No | Likely | Unclear | Unclear | Yes for clinical claims |
| npj Digital Medicine | Yes if generalisable | Yes | Yes | Yes | Not always, but implementation and generalisability expected |
| JEADV | No for single-dataset | Likely | Unlikely | Likely | No for multicentric retrospective work |
| Dermatology (Karger) | Yes | Likely | Likely | Likely | No |
| JAAD International | Yes, as a brief report | Likely | Unclear | Unclear | No |
| Skin Research and Technology | Yes by scope | Likely | Likely | Likely | No, but AI retraction risk |
| Computers in Biology and Medicine | Yes if methodological | Yes | Likely | Yes | No |
| Computer Methods and Programs in Biomedicine | Yes | Yes | Likely | Yes | No |
| Journal of Imaging Informatics in Medicine | Yes | Yes, explicit scope | Yes | Yes | No |
| Scientific Reports | Yes | Yes | Yes | Yes | No |
| PLOS ONE | Yes, with justification | Yes | Yes | Yes | No |
| BMC Medical Imaging | Yes | Likely | Likely | Likely | No |
| Diagnostics | Yes | Likely | Yes, reliability audit | Yes | No |
| Frontiers in Medicine | Yes, with validation | Likely | Likely | Yes | No, but external validation required for public-data studies |
| JMIR Dermatology | Yes | Yes | Likely | Likely | No |

### What the evidence shows

**Benchmark studies without prospective validation.** Direct precedents exist in Computers in Biology and Medicine, Computer Methods and Programs in Biomedicine, the Journal of Imaging Informatics in Medicine, Scientific Reports, PLOS ONE, BMC Medical Imaging, Diagnostics, Frontiers in Medicine, JMIR Dermatology, Dermatology and JAAD International. These journals have published retrospective, public-dataset or single-centre AI studies in the last two years.

**Fairness studies.** The strongest precedents are CMPB's 2024 skin colour bias paper, Scientific Reports' 2026 skin tone bias paper, PLOS ONE's 2026 SCGAN HAM10000 fairness extension, and JMIR Dermatology's 2024 melanoma LLM study across skin tones. Fairness is no longer a niche topic at these venues. The main reviewer question will be whether skin tone is measured or inferred, and whether the fairness metric is clinically meaningful.

**Calibration studies.** Calibration is usually treated as a secondary metric, not a standalone contribution. I found no 2024-2026 paper in the 16 journals whose primary contribution was calibration benchmarking in melanoma. The closest match is the MELBA 2025 paper "On the Role of Calibration in Benchmarking Algorithmic Fairness for Skin Cancer Detection", which combines calibration and group fairness on ISIC 2020 and PROVE-AI. That paper is a useful precedent but MELBA is a specialist machine-learning venue, not a clinical dermatology journal. This is both a gap and an opportunity: a paper that makes calibration a first-class endpoint will stand out, but it may need a methods-oriented venue.

**XAI studies.** Grad-CAM is widely accepted as a secondary analysis in T2 and T3 journals. The risk is that reviewers treat it as illustrative rather than evaluative. A paper that only shows heatmaps and says "the model looks at the lesion" will be seen as thin. Adding deletion/insertion tests, pointing-game accuracy, or inter-architecture heatmap agreement would strengthen the XAI contribution.

**Prospective validation.** JAMA Dermatology and BJD are the clearest "prospective or clinical impact" venues. JEADV accepts multicentric retrospective work but not a single-dataset benchmark. npj Digital Medicine is more flexible on study design but expects generalisability and implementation relevance. The remaining journals do not require prospective validation for a retrospective benchmark, although Frontiers in Medicine explicitly requires appropriate validation for computational studies of public data, which the HIBA set helps satisfy.

### Cited Findings

- CMPB published a 2024 study on skin colour bias in deep learning-based skin lesion segmentation: [CMPB fairness study](https://doi.org/10.1016/j.cmpb.2024.108044).
- Scientific Reports published a 2026 paper on reducing skin tone bias in dermatology AI: [Scientific Reports fairness paper](https://doi.org/10.1038/s41598-026-59316-x).
- PLOS ONE published a 2026 SCGAN-generated skin-tone-balanced extension of HAM10000: [PLOS ONE SCGAN](https://doi.org/10.1371/journal.pone.0331081).
- JMIR Dermatology published a 2024 multimodal LLM melanoma study across skin tones: [JMIR Dermatology](https://doi.org/10.2196/55508).
- Diagnostics published a 2026 reliability-audited XAI study for skin lesion triage under cross-device domain shift: [Diagnostics XAI](https://doi.org/10.3390/diagnostics16203368).
- Frontiers in Medicine requires appropriate validation for computational studies of public data: [Frontiers in Medicine scope](https://www.frontiersin.org/journals/medicine/about).
- MELBA published "On the Role of Calibration in Benchmarking Algorithmic Fairness for Skin Cancer Detection" in 2025: [MELBA paper](https://doi.org/10.59275/j.melba.2025-ae66).
- JIIM's scope explicitly lists AI safety, fairness and bias: [JIIM scope](https://link.springer.com/journal/10278/aims-and-scope).
- BJD published a prospective AI melanoma trial, which is evidence of a higher clinical evidence bar: [BJD prospective trial](https://doi.org/10.1093/bjd/ljae021).
- JAMA Dermatology published a 2026 systematic review focused on prospective AI melanoma evidence: [JAMA Dermatology systematic review](https://jamanetwork.com/journals/jamadermatology/fullarticle/2846489).

### Inferences

- The paper's best positioning is "benchmarking and auditing" rather than "new model". The fairness and calibration angles are the differentiators; the three architectures are the test bed, not the contribution.
- Calibration should be presented as a co-primary endpoint with fairness, not as an afterthought. That is the least crowded part of the literature and the part most likely to interest methods-oriented reviewers.
- A clinical dermatology journal will ask "what does this change for patients?" A methods or informatics journal will ask "what does this change for how models are evaluated?" The same paper needs different abstracts for T1/T2 versus T2/T3 venues.

### Gaps

- I did not find a 2024-2026 calibration-focused melanoma paper in any of the 16 journals. The absence is based on targeted searches, not a systematic review, so a negative claim should be treated cautiously.
- I did not verify whether any of the 16 journals has an explicit editorial policy requiring prospective validation for AI clinical claims. The ratings are inferred from published article types and scope statements.
- Acceptance rates are not public for most of these journals. No estimate is given.

## 3. Which venues are realistic for a master's-level study with three seeds, one public training dataset, and one external dermoscopic dataset?

### Takeaway

Eight venues are realistic first choices: JIIM, CMPB, BMC Medical Imaging, Scientific Reports, PLOS ONE, Diagnostics, Frontiers in Medicine and JMIR Dermatology. Dermatology (Karger) and CBM are realistic stretch options. JAMA Dermatology, BJD, npj Digital Medicine and JEADV are not realistic for this paper as it stands. The best three are JIIM, CMPB and BMC Medical Imaging. A sensible submission order is JIIM, then CMPB, then BMC Medical Imaging, with Scientific Reports and Diagnostics as fallbacks.

### Realism ranking

| Band | Journals | Why |
|---|---|---|
| High | JIIM, CMPB, BMC Medical Imaging, Scientific Reports, PLOS ONE, Diagnostics, Frontiers in Medicine, JMIR Dermatology | These venues publish retrospective public-dataset AI work; the evidence bar is technical soundness, validation and reproducibility, not prospective clinical impact |
| Medium-high | Dermatology (Karger), CBM | Clinical or methods relevance is required, but retrospective comparative work is publishable; CBM will expect a stronger methodological contribution |
| Medium | JAAD International, Skin Research and Technology | Clinical relevance or scope fit is good, but JAAD International favours clinical studies and SRT has an AI retraction risk |
| Low-medium | JEADV, npj Digital Medicine | Multicenter or generalisability expectations exceed a single-dataset benchmark |
| Low | JAMA Dermatology, BJD | Prospective or clinical-impact evidence expected; a HAM10000 benchmark is not competitive |

### Best three venues

**1. Journal of Imaging Informatics in Medicine.**
Why it is the best fit: the scope explicitly includes AI, computer-aided diagnosis and AI safety, fairness and bias. The journal has published skin cancer transformer work, has an open Dermatology collection, and states a JIF of 3.1 (2025). The audience understands imaging pipelines, external validation and calibration. The evidence bar is realistic for a strong master's paper with three seeds and an external set.
How to frame the paper: "Auditing skin-lesion classifiers for calibration and skin-tone fairness: a multi-architecture benchmark with external validation." Emphasise the imaging informatics implications: how to report fairness and calibration in CAD pipelines, and why aggregate accuracy hides subgroup harm.

**2. Computer Methods and Programs in Biomedicine.**
Why it is the best fit: it published the closest methodological precedent, a 2024 skin colour bias study in skin lesion segmentation. The journal values rigorous validation, reproducibility and clinically relevant methods. A fairness and calibration audit with paired tests and external validation is a natural fit.
How to frame the paper: "A reproducible benchmark of imbalance mitigation, calibration and skin-tone fairness for melanoma classifiers." Emphasise protocol, statistical testing, open code and the effect of imbalance correction on subgroup calibration.

**3. BMC Medical Imaging.**
Why it is the best fit: the journal's stated criterion is scientific validity, not perceived impact. It publishes HAM10000-style CNN benchmarks and has an impact factor of 3.2 (2024). The risk of desk rejection for "not novel enough" is lower than at CBM or CMPB.
How to frame the paper: "External validation and fairness audit of three deep learning architectures for melanoma classification." Emphasise scientific validity, limitations and the absence of clinical deployment claims.

Fallbacks: Scientific Reports for visibility, Diagnostics for speed, Frontiers in Medicine if the HIBA validation is strong and the paper is framed around clinical translation.

### Worst-fit venues

**JAMA Dermatology.** The journal's recent AI melanoma work is multicenter, federated or evidence-synthesis based, and its 2026 systematic review focuses on prospective evidence. A single-dataset benchmark will not be competitive. This is not a criticism of the paper; it is a mismatch of evidence stage.

**British Journal of Dermatology.** The journal publishes prospective clinical trials of AI in real clinical pathways. A retrospective HAM10000 benchmark does not answer a clinical-effectiveness question. A research letter might be possible, but the main paper would be better elsewhere.

**npj Digital Medicine.** The journal expects generalisability, implementation or broad significance. One public training set plus one external dermoscopic set is a reasonable start, but it is not enough for a digital medicine audience that routinely sees multi-site and deployment studies.

**JEADV.** A multicentric retrospective AI study can pass, but a single-dataset benchmark is unlikely to clear desk review. The journal's AI collection is clinically oriented.

**Near-misses that are not in the 16.** Medical Image Analysis is a top methods journal with a strong architecture and segmentation focus; a benchmark audit without a new method is a poor fit. Melanoma Research is a clinical and translational melanoma journal; its AI papers are prognostic or biomarker studies, not image-classifier benchmarks. International Journal of Medical Informatics is a health informatics journal, not an imaging methods journal; it has very few skin-lesion AI papers.

**Skin Research and Technology.** This is not a worst-fit venue on scope. It is a risk-adjusted "wait" venue. The scope is excellent, but the AI retraction cluster means the paper could be judged against a troubled recent record. If the authors submit, they should check the latest retractions, avoid special issues with weak guest editors, and be ready for extra scrutiny.

### Three-seed and single-dataset problems reviewers will raise

These are methodological risks, not journal-selection facts. They are included because they affect which venues will accept the paper.

**The paired test may be underpowered.** If the paired ranking test uses only the three seed-level scores, a two-sided Wilcoxon signed-rank test has a minimum attainable p-value of 0.25. That cannot reach conventional significance. The paper must pair at the fold, image or lesion level, or use a repeated cross-validation design such as 5x2 CV, or use a test with more paired units. If the paper already has 5-fold cross-validation times three seeds, then 15 paired units are available and the concern is reduced. This should be clarified in the methods before submission.

**Three seeds is acceptable but not generous.** Broad OA and applied journals will accept three seeds if variance, confidence intervals and paired tests are reported. Methods-heavy journals may ask for more seeds or for a nested cross-validation design. The paper should report seed-to-seed variance, not just mean and standard deviation.

**One training dataset is the main generalisability limit.** HIBA is a useful external set, but it is still dermoscopic and public. Reviewers at T1 and T2 venues may ask for ISIC 2019/2020, PAD-UFES-20 or DDI. At T3 venues, a clear limitation section and a domain-shift discussion may be enough. Adding even one more external set would materially improve the paper's chances at CMPB and JIIM.

**Skin tone labels are a likely weak point.** HAM10000 does not provide Fitzpatrick skin type labels. If the paper infers skin tone, it must describe and validate the inference, report uncertainty, and avoid treating inferred labels as ground truth. Clinical dermatology reviewers will push hard on this. Methods and informatics reviewers will accept an inferred label if the limitations are explicit.

**Calibration needs more than ECE.** ECE is sensitive to binning. Report adaptive ECE, Brier score, reliability diagrams by subgroup, and confidence intervals. A single ECE number is not enough to support a claim that one model is better calibrated than another.

**Grad-CAM alone is not an XAI contribution.** Add a faithfulness test such as deletion/insertion, a pointing-game analysis, or agreement across architectures. Otherwise Grad-CAM is an illustration, not evidence.

### Cited Findings

- JIIM's scope explicitly includes AI safety, fairness and bias: [JIIM scope](https://link.springer.com/journal/10278/aims-and-scope).
- CMPB published a 2024 skin colour bias study in skin lesion segmentation: [CMPB fairness study](https://doi.org/10.1016/j.cmpb.2024.108044).
- BMC Medical Imaging states an impact factor of 3.2 (2024) and evaluates scientific validity rather than impact: [BMC Medical Imaging metrics](https://bmcmedimaging.biomedcentral.com/submission-guidelines/peer-review-policy); [BMC Medical Imaging scope](https://bmcmedimaging.biomedcentral.com/submission-guidelines/aims-and-scope).
- JAMA Dermatology published a 2026 systematic review focused on prospective AI melanoma evidence: [JAMA Dermatology systematic review](https://jamanetwork.com/journals/jamadermatology/fullarticle/2846489).
- BJD published a 2024 prospective AI melanoma trial in primary care: [BJD prospective trial](https://doi.org/10.1093/bjd/ljae021).
- JEADV published a 2026 multicentric retrospective AI BCC study: [JEADV BCC study](https://doi.org/10.1111/jdv.70099).
- Skin Research and Technology's scope includes dermoscopy, digital imaging and AI: [SRT scope](https://onlinelibrary.wiley.com/page/journal/16000846/homepage/productinformation.html).
- A Crossref search found retractions of AI papers in Skin Research and Technology, including a melanoma deep learning paper: [retraction notice](https://doi.org/10.1111/srt.70296); [retracted paper](https://doi.org/10.1111/srt.13770).

### Inferences

- The realistic path is T2 methods/imaging first, T3 broad OA as fallback, and T1 only after a substantially stronger clinical validation study.
- The paper's strongest selling point is not "three architectures". It is the combination of calibration and skin-tone fairness under imbalance correction, with an external set. That combination is still uncommon in dermatology AI.
- A pre-submission statistical fix is more valuable than another architecture. If the paired test is underpowered, the paper's central ranking claim will fail at any serious methods journal.

### Gaps

- I did not inspect the paper's actual cross-validation design, so the three-seed power concern is conditional. It applies if the tests use only seed-level aggregates.
- I did not verify whether HIBA contains Fitzpatrick skin type labels or whether the authors infer them for HIBA as well as HAM10000. This affects the fairness claims and the choice of venue.
- I did not estimate acceptance probabilities. The realism ratings are qualitative.

## 4. What recent similar papers have appeared in each journal?

### Takeaway

Every journal in the 16 has at least one 2024-2026 paper that is close enough to cite as a precedent, except Skin Research and Technology, where the closest melanoma AI paper I found was retracted. The closest overall precedents are CMPB's skin colour bias study, JIIM's Swin transformer paper, Scientific Reports' 2026 fairness paper, PLOS ONE's SCGAN HAM10000 paper, and Diagnostics' 2026 reliability-audited XAI paper.

### Comparable papers by journal

| Journal | Recent paper | Year | Why it is comparable | Link |
|---|---|---|---|---|
| JAMA Dermatology | Federated learning for decentralised AI in melanoma diagnostics | 2024 | Melanoma AI with a multicenter/federated design | [DOI](https://doi.org/10.1001/jamadermatol.2023.5550) |
| JAMA Dermatology | Prospective evidence on AI-assisted melanoma diagnostics: systematic review and meta-analysis | 2026 | Shows the journal's current evidence bar | [URL](https://jamanetwork.com/journals/jamadermatology/fullarticle/2846489) |
| British Journal of Dermatology | Evaluation of AI-based decision support for cutaneous melanoma in primary care: prospective real-life trial | 2024 | Prospective clinical validation of melanoma AI | [DOI](https://doi.org/10.1093/bjd/ljae021) |
| npj Digital Medicine | Bias recognition and mitigation strategies in AI healthcare applications | 2025 | Fairness framing and mitigation review | [DOI](https://doi.org/10.1038/s41746-025-01503-7) |
| npj Digital Medicine | Synthesised clinical notes enable robust multimodal AI from unimodal dermatology datasets | 2026 | Dermatology AI with a generalisability angle | [DOI](https://doi.org/10.1038/s41746-026-03040-3) |
| JEADV | AI-assisted basal cell carcinoma diagnosis with LC-OCT: multicentric retrospective study | 2026 | Multicentric retrospective dermatology AI | [DOI](https://doi.org/10.1111/jdv.70099) |
| JEADV | Diagnosis melanoma with AI systems: meta-analysis and systematic review | 2025 | Melanoma AI evidence synthesis | [DOI](https://doi.org/10.1111/jdv.20781) |
| Dermatology (Karger) | GPT-4 Omni versus specialised neural networks for malignant lesions from smartphone images | 2026 | Retrospective comparative AI on skin lesion images | [DOI](https://doi.org/10.1159/000550153) |
| JAAD International | Multimodal LLMs for distinguishing melanoma from nevi in clinical and dermoscopic images | 2025 | Melanoma image classification with a clinical comparator | [DOI](https://doi.org/10.1016/j.jdin.2025.08.008) |
| Skin Research and Technology | Retracted melanoma deep learning paper | 2024 | Evidence of AI retraction risk, not a precedent | [Retraction](https://doi.org/10.1111/srt.70296); [Paper](https://doi.org/10.1111/srt.13770) |
| Computers in Biology and Medicine | Skin-CAD: explainable deep learning classification of skin cancer from dermoscopic images | 2024 | Explainable HAM10000-style skin cancer classification | [DOI](https://doi.org/10.1016/j.compbiomed.2024.108798) |
| Computer Methods and Programs in Biomedicine | Understanding skin colour bias in deep learning-based skin lesion segmentation | 2024 | Skin tone fairness in skin lesion analysis | [DOI](https://doi.org/10.1016/j.cmpb.2024.108044) |
| Journal of Imaging Informatics in Medicine | Enhancing skin cancer diagnosis with a Swin transformer | 2024 | Transformer-based skin cancer classification in an imaging informatics journal | [DOI](https://doi.org/10.1007/s10278-024-01140-8) |
| Scientific Reports | Reducing skin tone bias in dermatology AI via sketch-guided multimodal fusion | 2026 | Skin tone fairness in dermatology AI | [DOI](https://doi.org/10.1038/s41598-026-59316-x) |
| PLOS ONE | SCGAN-generated skin-tone-balanced extension of HAM10000 | 2026 | HAM10000 fairness and calibration-adjacent benchmark | [DOI](https://doi.org/10.1371/journal.pone.0331081) |
| PLOS ONE | Explainable depth-wise and channel-wise fusion models for multi-class skin lesion classification | 2026 | Explainable HAM10000 benchmark | [DOI](https://doi.org/10.1371/journal.pone.0340901) |
| BMC Medical Imaging | Enhanced skin cancer diagnosis using an optimised CNN architecture | 2024 | HAM10000-style CNN benchmark | [DOI](https://doi.org/10.1186/s12880-024-01356-8) |
| Diagnostics | Reliability-audited explainable AI for multi-class skin lesion triage under cross-device domain shift | 2026 | Calibration, reliability, XAI and domain shift in skin lesions | [DOI](https://doi.org/10.3390/diagnostics16203368) |
| Frontiers in Medicine | Explainable hybrid deep learning for skin lesion segmentation and multi-class classification | 2025 | EfficientNet-B0, HAM10000, Grad-CAM | [DOI](https://doi.org/10.3389/fmed.2025.1681542) |
| JMIR Dermatology | Multimodal LLMs for melanoma identification across different skin tones | 2024 | Skin-tone fairness in melanoma AI | [DOI](https://doi.org/10.2196/55508) |

### Closest topical matches outside the 16

**MELBA 2025, "On the Role of Calibration in Benchmarking Algorithmic Fairness for Skin Cancer Detection."** This is the closest paper to the proposed contribution. It evaluates the ISIC 2020 challenge models and PROVE-AI, stratifies by sex, race (Fitzpatrick skin tone) and age, and argues that calibration should complement AUROC-based fairness metrics. It is published in Machine Learning for Biomedical Imaging (MELBA), a specialist machine-learning venue. If the paper's calibration contribution is the main novelty, MELBA is worth considering as a 17th option, even though it is not a clinical dermatology journal. [MELBA paper](https://doi.org/10.59275/j.melba.2025-ae66).

**Journal of Computational Science 2026, "From underrepresentation to evidence: Pipsqueak dataset and explainable analysis of skin tone bias in melanoma detection."** This paper combines a curated darker-skin dataset with fairness metrics and explainable AI. It shows that a fairness-focused melanoma paper can land in a computational science venue, not only a dermatology or imaging journal. [Journal of Computational Science paper](https://www.sciencedirect.com/science/article/abs/pii/S1877750326000761).

**Frontiers in Oncology 2025, "EnsembleSkinNet."** A transfer-learning ensemble with Grad-CAM for skin cancer detection. This is a useful comparator for the "XAI as a secondary contribution" framing, and it shows the kind of paper that broad oncology venues accept. [Frontiers in Oncology paper](https://doi.org/10.3389/fonc.2025.1699960).

### Cited Findings

- The MELBA 2025 paper combines calibration and fairness benchmarking for skin cancer detection: [MELBA paper](https://doi.org/10.59275/j.melba.2025-ae66).
- The Journal of Computational Science published a 2026 skin tone bias paper with a curated darker-skin dataset and explainable AI: [Pipsqueak paper](https://www.sciencedirect.com/science/article/abs/pii/S1877750326000761).
- Frontiers in Oncology published a 2025 ensemble skin cancer paper with Grad-CAM: [EnsembleSkinNet](https://doi.org/10.3389/fonc.2025.1699960).
- Diagnostics published a 2026 reliability-audited XAI skin lesion triage paper: [Diagnostics XAI](https://doi.org/10.3390/diagnostics16203368).
- PLOS ONE published a 2026 explainable fusion paper for multi-class skin lesion classification: [PLOS ONE fusion](https://doi.org/10.1371/journal.pone.0340901).

### Inferences

- The paper's closest intellectual competitor is the MELBA calibration-and-fairness paper, not the many HAM10000 architecture papers. The authors should cite it and explain what their paper adds: three architectures, three imbalance conditions, paired ranking tests, error-type-specific fairness and HIBA external validation.
- The Journal of Computational Science paper shows that a fairness-and-XAI melanoma paper can be published outside dermatology. That is a useful fallback if clinical journals reject the paper for lack of clinical validation.
- Grad-CAM is common in broad OA venues but increasingly seen as insufficient in methods venues. The XAI contribution needs a quantitative faithfulness component to be competitive at CMPB or JIIM.

### Gaps

- I did not find a clean 2024-2026 melanoma AI paper in Skin Research and Technology. The only close match was retracted. This is a gap in the evidence for SRT as a target, and it reinforces the risk assessment.
- I did not search IEEE, SPIE or conference proceedings. The notes focus on journals, as requested.
- I did not verify the current indexing status of MELBA in PubMed or Scopus. That matters if the authors need a journal that counts toward a thesis requirement.

## 5. Which journals are predatory or high-risk, and should be avoided?

### Takeaway

None of the 16 journals is predatory by the usual documentary criteria. All are published by established publishers. The predatory risk sits with lookalike titles from publishers such as OMICS International and Longdom Publishing, and with weak special issues in otherwise legitimate journals. The one legitimate title in the set with a visible AI-specific risk is Skin Research and Technology, because of a cluster of retractions in AI papers.

### Predatory publishers and titles

**OMICS International.** Beall's List, an archived list of potential predatory publishers, includes OMICS International. The US Federal Trade Commission charged OMICS in 2016 with deceiving researchers about its publishing services. This is a documented regulatory action, not a blog allegation. Avoid OMICS dermatology or clinical dermatology titles, and avoid any journal whose publisher cannot be verified independently. [Beall's List publishers](https://beallslist.net/publishers); [FTC action](https://www.ftc.gov/news-events/press-releases/2016/08/ftc-charges-academic-journal-publisher-omics-group-deceived-researchers).

**Longdom Publishing.** Beall's List also includes Longdom Publishing, connected to International Scholars Journals. Treat Longdom dermatology titles as high-risk unless the specific journal can be verified through DOAJ, COPE, PubMed or Scopus. [Beall's List publishers](https://beallslist.net/publishers).

**Dermatology-specific scale of the problem.** A 2025 bibliometric review of predatory journals in dermatology reports 61 potential predatory dermatology publishers, 57 journals and 4,164 articles. The exact counts depend on the list used, but the order of magnitude is a warning: predatory titles in dermatology are not rare. [Predatory journals in dermatology review](https://vlex.co.uk/vid/predatory-journals-in-dermatology-1072314310).

**Contamination of evidence syntheses.** A 2022 JMIR Dermatology study found that systematic reviews in top dermatology journals sometimes include primary studies published in suspected predatory journals. The risk is not only personal career damage; it is contamination of the evidence base. [JMIR Dermatology study](https://doi.org/10.2196/39365).

### Legitimate but high-risk or poor-fit venues

**Skin Research and Technology.** Legitimate Wiley journal, excellent scope, but multiple AI retractions in 2024-2026, including a melanoma deep learning paper. The retraction notices do not prove the journal is predatory. They do mean an AI submission may face heightened scrutiny, and the authors should check the latest retraction record before submitting. [Retraction notice](https://doi.org/10.1111/srt.70296); [Retracted paper](https://doi.org/10.1111/srt.13770).

**MDPI journals (Diagnostics, Cancers, Bioengineering).** Legitimate open-access publisher, COPE member, with transparent APCs and peer review. The risks are high volume, fast review, special-issue proliferation and variable perceived prestige. None of these makes the publisher predatory. Check the guest editors and the special issue, and do not submit to a special issue solely because it promises a fast decision. [MDPI COPE statement](https://www.mdpi.com/journal/diagnostics/about).

**Frontiers.** Legitimate open-access publisher with a broad-scope, APC-based model. The risk is perceived prestige and the possibility of a weak review experience, not fraud. Frontiers in Medicine's validation policy for public-data computational studies is actually helpful for this paper. [Frontiers in Medicine scope](https://www.frontiersin.org/journals/medicine/about).

**PLOS ONE and Scientific Reports.** Legitimate broad OA journals with sound-science criteria. The risk is different: PLOS ONE may reject derivative work without a clear justification, and Scientific Reports has a very high volume with variable perceived prestige. Neither is predatory.

**JAMA Dermatology, BJD, npj Digital Medicine, JEADV.** Not predatory and not high-risk in the integrity sense. They are simply high-bar venues that are unlikely to accept this paper at its current evidence stage. Do not confuse "high rejection risk" with "predatory".

### Legitimacy checklist

Use this before submitting anywhere, including the 16:

1. Check the publisher and the exact journal title on Beall's List, DOAJ, COPE and the publisher's own site. Lookalike titles are common.
2. Check PubMed/MEDLINE, Scopus or Web of Science indexing. A journal that claims indexing but cannot show a record is a red flag.
3. Check the editorial board. Real names, real affiliations, real publications. A board of unverifiable names is a red flag.
4. Check the last two years of papers. Are there papers close to your topic? Are they scientifically plausible? Are there retractions?
5. Check the APC and the refund policy. A legitimate journal states fees before submission. Pressure to pay quickly is a red flag.
6. Check the review process. A guaranteed acceptance, a one-week review, or an email invitation that flatters you is a red flag.
7. Check Think Check Submit and ask your university librarian. This is faster than learning the hard way. [Think Check Submit](https://thinkchecksubmit.org/); [DOAJ](https://doaj.org/).

### Cited Findings

- Beall's List includes OMICS International and Longdom Publishing as potential predatory publishers: [Beall's List publishers](https://beallslist.net/publishers).
- The FTC charged OMICS with deceiving researchers: [FTC action](https://www.ftc.gov/news-events/press-releases/2016/08/ftc-charges-academic-journal-publisher-omics-group-deceived-researchers).
- A 2025 bibliometric review reports 61 potential predatory dermatology publishers, 57 journals and 4,164 articles: [Predatory journals in dermatology review](https://vlex.co.uk/vid/predatory-journals-in-dermatology-1072314310).
- A 2022 JMIR Dermatology study found suspected predatory publications in systematic reviews in top dermatology journals: [JMIR Dermatology study](https://doi.org/10.2196/39365).
- Skin Research and Technology has retracted AI papers, including a melanoma deep learning paper: [Retraction notice](https://doi.org/10.1111/srt.70296); [Retracted paper](https://doi.org/10.1111/srt.13770).
- MDPI states COPE membership and a peer-review ethics policy: [MDPI COPE statement](https://www.mdpi.com/journal/diagnostics/about).
- Think Check Submit provides a journal trust checklist: [Think Check Submit](https://thinkchecksubmit.org/).
- DOAJ provides an independent index of open-access journals: [DOAJ](https://doaj.org/).

### Inferences

- The biggest predatory risk is not the 16 journals. It is a lookalike title with a similar name, a flattering email, and a fast acceptance promise. Verify the publisher, not just the journal name.
- A legitimate journal with a retraction cluster is a different kind of risk from a predatory journal. The paper may still be publishable there, but the authors should know the record and avoid special issues with weak oversight.
- High APC and high volume are not evidence of predation. MDPI, Frontiers, PLOS and BMC are legitimate. The appropriate response is to weigh prestige, cost and review quality, not to accuse them of fraud.

### Gaps

- I did not independently verify the COPE or DOAJ status of every journal in the 16. The statement that none is predatory is based on publisher identity, Beall's List checks and general indexing knowledge, not a journal-by-journal audit.
- I did not audit retractions for all 16 journals. Skin Research and Technology stood out in a targeted Crossref check; other journals may have retractions that I did not find.
- The 2025 dermatology predatory review's counts depend on the list used. The number should be cited as an indicator, not a precise census.


