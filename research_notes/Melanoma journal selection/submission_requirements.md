# Melanoma AI Journal Submission Requirements

Compiled 12 November 2026 from the official author guidelines and template downloads listed below. The paper's current affiliation block already follows MDPI style, so MDPI is treated as the baseline format.

**Primary source key (all retrieved 12 November 2026):**

- S1 = [Diagnostics Instructions for Authors](https://www.mdpi.com/journal/diagnostics/instructions)
- S2 = [Cancers Instructions for Authors](https://www.mdpi.com/journal/cancers/instructions)
- S3 = [Bioengineering Instructions for Authors](https://www.mdpi.com/journal/bioengineering/instructions)
- S4 = [Computers in Biology and Medicine Guide for Authors](https://www.sciencedirect.com/journal/computers-in-biology-and-medicine/publish/guide-for-authors)
- S5 = [Computer Methods and Programs in Biomedicine Guide for Authors](https://www.sciencedirect.com/journal/computer-methods-and-programs-in-biomedicine/publish/guide-for-authors)
- S6 = [Artificial Intelligence in Medicine Guide for Authors](https://www.sciencedirect.com/journal/artificial-intelligence-in-medicine/publish/guide-for-authors)
- S7 = [IEEE JBHI Prepare and Submit Your Manuscript](https://www.embs.org/jbhi/prepare-and-submit-your-manuscript)
- S8 = [Scientific Reports Submission Guidelines](https://www.nature.com/srep/author-instructions/submission-guidelines)
- S9 = [PLOS Digital Health Submission Guidelines](https://journals.plos.org/digitalhealth/s/submission-guidelines)
- S10 = [Medical Image Analysis Guide for Authors](https://www.sciencedirect.com/journal/medical-image-analysis/publish/guide-for-authors)
- S11 = [npj Digital Medicine Submission Guidelines](https://www.nature.com/npjdigitalmed/for-authors-and-referees/submission-guidelines)

**Template and policy source key:**

- T1 = [MDPI Preparing Manuscripts in LaTeX](https://www.mdpi.com/authors/latex)
- T2 = [MDPI ACS LaTeX template ZIP](https://res.mdpi.com/data/MDPI_template_ACS.zip?v=20260911)
- T3 = [Diagnostics Word template](https://mdpi-res.com/files/word-templates/diagnostics-template.dot)
- T4 = [Cancers Word template](https://mdpi-res.com/files/word-templates/cancers-template.dot)
- T5 = [Bioengineering Word template](https://mdpi-res.com/files/word-templates/bioengineering-template.dot)
- T6 = [Elsevier LaTeX instructions](https://www.elsevier.com/latex)
- T7 = [Elsevier els-cas LaTeX template ZIP](https://assets.ctfassets.net/o78em1y1w4i4/5uFmLZJTPDMAUjFnHRpjj8/6f19a979146eb93263763d87a894ab0d/els-cas-templates.zip)
- T8 = [IEEE template selector](https://template-selector.ieee.org/secure/templateSelector/publicationType)
- T9 = [IEEE Author Center article templates](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/)
- T10 = [Springer Nature LaTeX author support](https://www.springernature.com/gp/authors/campaigns/latex-author-support)
- T11 = [PLOS Digital Health LaTeX guidelines](https://journals.plos.org/digitalhealth/s/latex)
- T12 = [PLOS LaTeX template ZIP](https://journals.plos.org/digitalhealth/s/file?id=1457/PLOS_latex_template.zip)
- T13 = [PLOS Digital Health data availability policy](https://journals.plos.org/digitalhealth/s/data-availability)
- T14 = [PLOS Digital Health materials, software and code sharing](https://journals.plos.org/digitalhealth/s/materials-software-and-code-sharing)
- T15 = [PLOS Digital Health best practices in research reporting](https://journals.plos.org/digitalhealth/s/best-practices-in-research-reporting)
- T16 = [npj Digital Medicine content types](https://www.nature.com/npjdigitalmed/for-authors-and-referees/about/content-types)
- T17 = [PLOS Digital Health journal information](https://journals.plos.org/digitalhealth/s/journal-information)

## 1. Is there an official LaTeX template, and where is it downloaded?

### Takeaway

All 11 journals accept LaTeX. MDPI and PLOS provide LaTeX packages with bibliography style files; Elsevier, IEEE, and Springer Nature provide publisher-wide templates rather than journal-specific classes. The three MDPI journals are the only targets that combine a journal-branded Word template with a free-format first-submission option.

### Cited Findings

| Journal | Official guidelines | LaTeX template and download | Initial-submission note |
|---|---|---|---|
| Diagnostics (MDPI) | S1 | MDPI ACS class, T1 and T2 | Free-format first submission; journal-specific Word template T3; 120 MB total upload cap [S1] |
| Cancers (MDPI) | S2 | MDPI ACS class, T1 and T2 | Free-format first submission; Word template T4; 120 MB cap [S2] |
| Bioengineering (MDPI) | S3 | MDPI ACS class, T1 and T2 | Free-format first submission; Word template T5; 120 MB cap [S3] |
| Computers in Biology and Medicine (Elsevier) | S4 | Elsevier els-cas, T6 and T7 | No need to specifically format a first submission; manuscripts that reproduce final journal layout are sent back [S4] |
| Computer Methods and Programs in Biomedicine (Elsevier) | S5 | Elsevier els-cas, T6 and T7 | Editable .tex or .doc/.docx; PDF is not an acceptable source file [S5] |
| Artificial Intelligence in Medicine (Elsevier) | S6 | Elsevier els-cas, T6 and T7 | Editable .tex or .doc/.docx; single-column Word, double-column only for LaTeX [S6] |
| IEEE Journal of Biomedical and Health Informatics | S7 | IEEE template selector, T8 and T9 | IEEE single-spaced double-column, 11 pt or larger, embedded figures and tables, PDF upload [S7] |
| Scientific Reports (Nature Portfolio) | S8 | Springer Nature LaTeX template, T10 | Word, LaTeX, or PDF accepted at first submission; combined text and figures up to 3 MB [S8] |
| PLOS Digital Health | S9 | PLOS LaTeX template, T11 and T12 | LaTeX manuscripts submitted as PDF initially; .tex source at acceptance [T11] |
| Medical Image Analysis (Elsevier) | S10 | Elsevier els-cas, T6 and T7 | Editable .tex or .doc/.docx; single-column Word, double-column only for LaTeX [S10] |
| npj Digital Medicine (Nature Portfolio) | S11 | Springer Nature LaTeX template, T10 | Initial submission as PDF or Word; LaTeX accepted at acceptance stage [S11] |

Additional template facts:

- The MDPI ACS ZIP contains `template.tex`, `Definitions/mdpi.cls`, `Definitions/mdpi.bst`, and APA/Chicago bibliography styles; the page states the package was last updated 21 October 2026 [T1, T2].
- The Elsevier els-cas ZIP contains `cas-sc.cls`, `cas-dc.cls`, `cas-refs.bib`, and `cas-model2-names.bst` [T7].
- The PLOS template ZIP contains a BibTeX style sheet and the page states all PLOS journals use Vancouver style [T11, T12].
- MDPI journal pages state that Word manuscripts must be a single file and LaTeX manuscripts must be a single ZIP with all source files and images so the editorial office can recompile the PDF [S1, S2, S3].
- JBHI requires the IEEE double-column format for review, not just at acceptance, and the manuscript including figures must be converted to PDF for upload [S7].
- Scientific Reports allows text and figures in one file up to 3 MB for first submission, while revised manuscripts must be Word or LaTeX, not PDF [S8].
- npj Digital Medicine accepts LaTeX at acceptance but asks for compiled PDFs before then [S11].

### Inferences

- MDPI is the lowest-conversion target because the current affiliation block already matches its front matter, the journal accepts free-format first submission, and the LaTeX class is available with a BibTeX style.
- CBM is also a low-effort first submission because it explicitly says not to format for the journal and warns that final-layout manuscripts will be returned.
- JBHI, MIA, and npj are the highest-conversion targets: JBHI requires a 14-page IEEE double-column build, MIA requires graphical abstract plus highlights plus data deposit, and npj requires a 150-word abstract plus a Nature Portfolio Reporting Summary.
- For a manuscript currently in MDPI LaTeX, the practical route is to keep the MDPI build as the master, then create separate els-cas, IEEEtran, and Springer Nature builds only for the shortlisted targets.
- For a melanoma imaging paper, Diagnostics and Cancers are the clinical venues; Bioengineering and Medical Image Analysis are the engineering and imaging venues; CBM, CMPB, AIM, and JBHI are the methods and informatics venues; Scientific Reports, PLOS Digital Health, and npj Digital Medicine are broad multidisciplinary venues.
- PLOS Digital Health explicitly lists Medical Imaging as a section, which is a direct fit for a melanoma imaging study [T17].
- AIM explicitly warns that mere application of well-known algorithms is not original, so the XAI contribution must be methodologically substantive for that target [S6].

### Gaps

- None of the Elsevier targets provides a journal-specific LaTeX class; final formatting is applied at proof, so page-count estimates from els-cas may differ from the published layout.
- The JBHI page limit is a typeset-page limit; the exact page count can only be checked after compiling in the IEEE template.
- Several ScienceDirect pages were bot-protected during retrieval; the content above comes from the official pages as rendered through a reader service, so exact wording should be rechecked immediately before submission.

## 2. What are the abstract, manuscript, figure, table, and reference limits?

### Takeaway

Limits vary widely. Only CMPB, JBHI, Scientific Reports, npj Digital Medicine, and PLOS Digital Health state hard or recommended length numbers. MDPI, CBM, AIM, and MIA do not state a manuscript length cap. Abstract limits range from 150 words at npj Digital Medicine to 350 words in the CMPB journal-specific section.

### Cited Findings

| Journal | Abstract | Main text or page limit | Figures and tables | References |
|---|---|---|---|---|
| Diagnostics (MDPI) | Structured abstract of around 250 words with headings Background/Objectives, Methods, Results, Conclusions [S1] | No maximum length stated [S1] | No numeric limit stated; 120 MB total upload; figures preferably at least 600 dpi in PNG, JPEG, or TIFF; table font no smaller than 8 pt [S1] | No maximum stated [S1] |
| Cancers (MDPI) | Mandatory Simple Summary of no more than 150 words, then a structured abstract of around 250 words with the same four headings [S2] | No restriction on maximum length for research manuscripts [S2] | No numeric limit stated; 120 MB total upload; 600 dpi preferred [S2] | Research articles should have more than 30 references; self-citation should be controlled within 10%, with justification above that [S2] |
| Bioengineering (MDPI) | About 200 words maximum, single paragraph, structured style but without headings [S3] | No maximum length stated [S3] | No numeric limit stated; 120 MB total upload; 600 dpi preferred [S3] | No maximum stated [S3] |
| Computers in Biology and Medicine (Elsevier) | No more than 250 words [S4] | No maximum length stated; no need to format for first submission [S4] | No numeric limit stated; tables must be editable text, figures supplied as separate files [S4] | No maximum stated; numbered in order of appearance [S4] |
| Computer Methods and Programs in Biomedicine (Elsevier) | Journal-specific section says abstract of 350 words, structured as Background and Objectives, Methods, Results, Conclusions [S5] | Should not normally exceed 3,500 words excluding the abstract [S5] | No specific limits for tables and figures [S5] | Up to 50 references [S5] |
| Artificial Intelligence in Medicine (Elsevier) | No more than 250 words [S6] | No maximum length stated [S6] | No numeric limit stated [S6] | No maximum stated [S6] |
| IEEE Journal of Biomedical and Health Informatics | No more than 250 words; no abbreviations, footnotes, references, or equations [S7] | 14 pages for regular papers, 16 pages for review papers, including supplementary material [S7] | Figures and tables embedded in text at submission; separate files and one figure per page after acceptance [S7] | No maximum stated; IEEE numbered style [S7] |
| Scientific Reports | No more than 200 words, unstructured, no references or figures [S8] | Main text no more than 4,500 words, not including Abstract, Methods, References, and figure legends; articles ideally no more than 11 typeset pages; no strict limit [S8] | Figure legends limited to 350 words per figure; tables maximum size of one page [S8] | Limited to 60 references, though not strictly enforced [S8] |
| PLOS Digital Health | No more than 300 words, with Background, Methodology/Principal Findings, and Conclusions/Significance conceptually but no headings; plus a 150 to 200 word non-technical Author Summary [S9] | Manuscripts can be any length; no restrictions on word count, number of figures, or amount of supporting information [S9] | No restriction on number of figures; tables placed in text after first citation [S9] | No maximum stated; Vancouver style [S9] |
| Medical Image Analysis (Elsevier) | No more than 250 words [S10] | No maximum length stated [S10] | Highlights required; graphical abstract required; no numeric figure or table limit stated [S10] | No maximum stated; Harvard author-date style [S10] |
| npj Digital Medicine | Article: no more than 150 words, no subheadings; Review: no more than 70 words [S11, T16] | No strict word or page limit; Review main text typically 3,000 to 4,000 words [S11, T16] | Figure legends limited to 350 words per figure [S11, T16] | Limited to 60 references, though not strictly enforced [S11, T16] |

Additional limit facts:

- Cancers states that submissions without a Simple Summary will be returned directly [S2].
- CMPB contains an internal conflict: the generic Elsevier abstract section says the abstract must not exceed 250 words, while the journal-specific section says 350 words [S5]. The journal-specific 350-word instruction is more specific, but the conflict is unresolved.
- CMPB systematic reviews are capped at 4,000 words excluding a 350-word abstract, with up to 5 tables and figures in total and unlimited references [S5].
- Scientific Reports also recommends a title of no more than 20 words and allows up to 6 keywords [S8].
- Scientific Reports does not support graphical abstracts [S8].
- npj Digital Medicine requires a title of no more than 15 words for Articles and Reviews [T16].
- MDPI journals require 3 to 10 keywords [S1, S2, S3].
- CBM, AIM, and MIA require 1 to 7 keywords; CMPB requires 3 to 6 [S4, S5, S6, S10].
- MDPI figures must be inserted into the main text close to their first citation and numbered in order of appearance [S1, S2, S3].
- JBHI regular papers over 8 pages incur mandatory overlength charges: $250 per page for pages 9 and 10, and $350 per page for page 11 and beyond, within the 14-page cap [S7].

### Article-type caveat

| Journal | Review or systematic review rule |
|---|---|
| Diagnostics, Cancers, Bioengineering | Review format supported at all three; Cancers reviews must include at least two figures or tables [S1, S2, S3] |
| Computers in Biology and Medicine | Review is a listed article type; no separate length rule stated [S4] |
| Computer Methods and Programs in Biomedicine | Systematic reviews capped at 4,000 words, 5 tables/figures total, unlimited references [S5] |
| Artificial Intelligence in Medicine | Review is a listed article type; no separate length rule stated [S6] |
| IEEE JBHI | Review papers require prior contact with the Editor-in-Chief and a review proposal; 16-page limit and five independent reviewers [S7] |
| Scientific Reports | Publishes original research only, as Article or Registered Report; reviews are not an article type [S8] |
| PLOS Digital Health | Publishes primary research articles plus commissioned opinions and reviews; unsolicited reviews are not the main route [T17] |
| npj Digital Medicine | Review is a content type, but systematic reviews, scoping reviews, and meta-analyses must be submitted as Articles, not Reviews [T16] |
| Medical Image Analysis | Review route not verified in the retrieved guide [S10] |

### Inferences

- If the manuscript is a review, Scientific Reports is not eligible and PLOS Digital Health is unlikely unless the review is commissioned. JBHI would require pre-approval and a 16-page build.
- If the manuscript is original research, the main length constraints are CMPB at 3,500 words, JBHI at 14 pages, Scientific Reports at 4,500 words recommended, and npj at a 150-word abstract.
- CMPB is the most aggressive cut: a typical melanoma AI paper with extensive methods and results would need to fit 3,500 words and 50 references, with the abstract at 350 words or 250 words to satisfy the conflicting instructions.
- MDPI and CBM are the most flexible for a long manuscript, but MDPI's 120 MB upload cap and CBM's no-final-layout rule still shape the build.

### Gaps

- CBM, AIM, and MIA do not state a manuscript length or reference cap in their guides; any cap may exist only in the submission system or editorial practice.
- MIA may have a page or word limit in the submission system that is not in the public guide.
- The CMPB 250 versus 350 word abstract conflict should be resolved with the editorial office before submission.

## 3. What statements are required: author contributions, funding, conflicts, data availability, code availability, ethics, informed consent, acknowledgments, ORCID?

### Takeaway

Every journal requires an ethics statement for human data and a conflict-of-interest declaration; most also require an explicit informed-consent statement. Data availability is mandatory at MDPI, Scientific Reports, PLOS Digital Health, npj Digital Medicine, and effectively at MIA through its Option C deposit rule; it is only encouraged at CBM, CMPB, and AIM. Code availability is an explicit separate statement only at PLOS and npj. ORCID is mandatory for all authors at JBHI and for corresponding authors at PLOS and npj.

### Cited Findings

**Authorship, funding, conflicts, acknowledgments, ORCID**

| Journal | Author contributions | Funding | Conflicts of interest | Acknowledgments | ORCID |
|---|---|---|---|---|---|
| Diagnostics, Cancers, Bioengineering | Author Contributions section required [S1, S2, S3] | Funding section required, with funder and grant number or "no external funding" [S1, S2, S3] | Required; "The authors declare no conflict of interest" if none [S1, S2, S3] | Optional; also used for GenAI disclosure [S1, S2, S3] | Optional; ORCID icon added if provided [S1] |
| CBM | CRediT author statement required [S4] | Funding sources section required [S4] | Declaration of competing interests required through Elsevier declarations tool [S4] | Separate section directly before references [S4] | Not stated in the guide [S4] |
| CMPB | CRediT author statement required [S5] | Funding sources required [S5] | Declaration of competing interests required [S5] | Separate section before references [S5] | Not stated [S5] |
| AIM | CRediT author statement required [S6] | Funding sources required [S6] | Declaration of competing interests required [S6] | Separate section before references [S6] | Not stated [S6] |
| MIA | CRediT author statement required [S10] | Funding sources required [S10] | Declaration of competing interests required [S10] | Separate section before references [S10] | Not stated [S10] |
| IEEE JBHI | Not specified on the author page [S7] | Not specified [S7] | Not specified [S7] | Not specified [S7] | Required for all authors [S7] |
| Scientific Reports | Author Contribution Statement mandatory [S8] | Include in Acknowledgements or a separate Funding section [S8] | Competing interests statement mandatory for each author [S8] | Optional [S8] | Not stated [S8] |
| PLOS Digital Health | Author contributions required [S9] | Financial Disclosure entered in the submission system, not in the manuscript [S9] | Competing interests required [S9] | Allowed; do not include funding sources [S9] | Corresponding author must provide ORCID at submission [S9] |
| npj Digital Medicine | Mandatory, all authors individually by initials [S11, T16] | Declare in Acknowledgements; a separate Funding statement is not permitted [T16] | Mandatory for each author [S11, T16] | Brief; includes grant numbers [S11] | Corresponding author must provide ORCID before final submission [S11] |

**Data, code, ethics, consent**

| Journal | Data availability | Code availability | Ethics / IRB | Informed consent |
|---|---|---|---|---|
| Diagnostics, Cancers, Bioengineering | Mandatory Data Availability Statement for all articles except Book Review, Editorial, Obituary, Correction, Retraction, and Expression of Concern [S1, S2, S3] | No separate statement; release code in a recognized repository or as supplementary material, and name software versions [S1] | Institutional Review Board statement with project code and approval date required for human studies; "Not applicable" allowed if no human or animal work [S1] | Informed Consent Statement required for human studies; "Not applicable" allowed if no human work [S1] |
| CBM | Elsevier Option B: data deposit encouraged, and a statement explaining why data cannot be shared if applicable [S4] | No separate statement; research data includes software and code, release encouraged [S4] | Statement with ethical approval date and reference number required for human studies [S4] | Informed consent required for human participants [S4] |
| CMPB | Option B encouraged, plus a Data statement encouraged at submission [S5] | No separate statement; code release encouraged [S5] | Ethics statement with approval number required [S5] | Informed consent required [S5] |
| AIM | Option B encouraged, plus a Data statement encouraged [S6] | No separate statement; code release encouraged [S6] | Ethics statement with approval number required [S6] | Informed consent required [S6] |
| MIA | Option C: authors are required to deposit data in a repository and link it, or provide a statement explaining why sharing is not possible [S10] | Research data includes software and code; deposit or explain [S10] | Ethics statement with approval number required [S10] | Informed consent required [S10] |
| IEEE JBHI | Not specified as a separate statement; public dataset source and citation required [S7] | Not specified [S7] | IRB approval and explicit reference number required for human studies [S7] | Not explicitly stated beyond IRB approval [S7] |
| Scientific Reports | Data Availability Statement mandatory at end of main text before References [S8] | Not explicit in the submission guidelines; Nature Portfolio policy may apply [S8] | Ethics statement in Methods for human or animal work [S8] | Informed consent required for identifiable participants [S8] |
| PLOS Digital Health | Mandatory; all data necessary to replicate findings must be publicly available without restriction at publication, with limited exceptions [S9, T13] | Required where author-generated code underpins the findings; state access in the Data Availability Statement [T14] | Ethics statement required [S9] | Informed consent required, including whether written or oral [S9] |
| npj Digital Medicine | Mandatory Data Availability Statement after Methods and before References [S11] | Required where custom code is central to the conclusions, under a Methods heading "Code availability" [S11] | Ethical approval details and consent in Methods [S11, T16] | Required for human participants [S11] |

Additional statement facts:

- MDPI's model Institutional Review Board statement is: "The study was conducted in accordance with the Declaration of Helsinki, and approved by the Institutional Review Board (or Ethics Committee) of NAME OF INSTITUTE (protocol code XXX and date of approval)" [S1].
- MDPI's model informed consent statement is: "Informed consent was obtained from all subjects involved in the study," or "Patient consent was waived due to REASON," or "Not applicable" [S1].
- MDPI's Data Availability Statement is mandatory even when no new data were created or data are restricted [S1].
- MDPI requires a statement on generative AI use in the Materials and Methods section, and an Acknowledgments statement if GenAI was used to generate text, data, graphics, study design, or analysis [S1, S2, S3].
- Elsevier requires a "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process" as a new section before the reference list [S4, S5, S6].
- Elsevier's declarations tool must always be completed, and the resulting Word document is uploaded at the attach/upload files step [S4, S5, S6].
- Medical Image Analysis requires a short biography of each author, maximum 100 words, in an editable format [S10].
- JBHI also requires a signed Author Consent Form for multi-author manuscripts, uploaded as a supporting document, plus the IEEE copyright form on acceptance [S7].
- PLOS requires a Financial Disclosure statement in the submission system; if unfunded, use "The author(s) received no specific funding for this work" [S9].
- npj Digital Medicine requires a Nature Portfolio Reporting Summary and an Editorial Policy Checklist with the revised manuscript [S11].

### Inferences

- For a retrospective study using public melanoma image datasets, MDPI, PLOS, and npj are the clearest: each has a defined Data Availability Statement route and explicit ethics/consent options.
- CBM, CMPB, and AIM are more flexible on data sharing, but they still require a statement explaining why data cannot be shared if that is the case.
- MIA is the most demanding: data deposit is required, not merely encouraged, so a public melanoma dataset and a reproducible split must be ready before submission.
- JBHI's ORCID requirement for all authors and signed consent form are easy to miss and can delay submission.

### Gaps

- JBHI does not state a data availability, code availability, funding, or conflict-of-interest format on its author page; IEEE-wide policies or the submission portal may impose additional requirements.
- Scientific Reports does not explicitly state a code availability requirement in its submission guidelines, although Nature Portfolio editorial policy may require one for custom code.
- Elsevier guides do not state an ORCID requirement, but the submission system may request it.
- No journal states whether a separate "data statement" is required for a review that analyzes previously published data; this may need editorial clarification.

## 4. What is the reference style, and does the journal accept BibTeX?

### Takeaway

MDPI uses ACS numbered style after a free-format initial submission. Elsevier uses numbered square-bracket references at CBM, CMPB, and AIM, but Harvard author-date at MIA. IEEE uses IEEE numbered style, Nature Portfolio uses Nature numbered style, and PLOS uses Vancouver. All major LaTeX routes support BibTeX.

### Cited Findings

| Journal | Style at first submission | Final style | BibTeX support |
|---|---|---|---|
| Diagnostics, Cancers, Bioengineering | Any consistent style is accepted under free-format submission [S1, S2, S3] | ACS numbered style, square brackets in text, full title in reference list [S1, S2, S3] | Yes: MDPI ACS template includes `mdpi.bst` [T1, T2] |
| CBM | No strict formatting at submission; any consistent style [S4] | Numbered square brackets in order of appearance; journal style applied at proof [S4] | Yes: els-cas template includes `cas-refs.bib` and `cas-model2-names.bst` [T7] |
| CMPB | No strict formatting at submission; any consistent style [S5] | Numbered square brackets in order of appearance [S5] | Yes: els-cas template [T7] |
| AIM | No strict formatting at submission; any consistent style [S6] | Numbered square brackets in order of appearance [S6] | Yes: els-cas template [T7] |
| MIA | No strict formatting at submission; any consistent style [S10] | Harvard author-date, alphabetical, not numbered [S10] | Yes: els-cas template with author-date bibliography style [T7] |
| IEEE JBHI | IEEE numbered style required [S7] | IEEE numbered style, numerals in square brackets [S7] | Yes: IEEEtran template and BibTeX route through T8/T9 [T8, T9] |
| Scientific Reports | Nature numbered style required [S8] | Nature numbered style; references are not copy-edited [S8] | Yes: Springer Nature LaTeX template [T10] |
| PLOS Digital Health | Vancouver numbered style required [S9] | Vancouver numbered style, first six authors then et al. [S9] | Yes: PLOS template includes BibTeX style sheet [T11, T12] |
| npj Digital Medicine | Nature numbered style required [S11] | Nature numbered style [S11] | Yes: Springer Nature LaTeX template [T10] |

Additional reference facts:

- MDPI reference lists must include the full title and use abbreviated journal names, year, volume, and page range [S1].
- MDPI Cancers requires self-citation to be controlled within 10%, with justification if above 10% [S2].
- Elsevier does not set strict reference formatting at submission but requires every reference to correspond to a real source, with DOIs where available [S4, S5, S6].
- MIA uses author-date citations such as "Allan, 2020a, 2020b" and arranges the list alphabetically, then chronologically [S10].
- JBHI requires complete IEEE references: author initials and last name, title, periodical, volume, inclusive pages, month, year [S7].
- Scientific Reports uses Nature style and does not copy-edit references, so incorrect formatting stays in the published article [S8].
- PLOS uses ICMJE/Vancouver sample references and numbers references in order of appearance [S9].
- npj Digital Medicine limits references to 60, though not strictly enforced [S11, T16].

### Inferences

- If the manuscript is currently in MDPI ACS style, the least risky conversion is to keep numeric bracketed citations for CBM, CMPB, AIM, JBHI, Scientific Reports, PLOS, and npj, and to switch only MIA to author-date.
- MIA is the one target where a numeric reference manager style will produce the wrong output; budget time for a Harvard conversion.
- Because MDPI accepts any consistent style at first submission, the current numeric style can be submitted as-is, with ACS formatting deferred to revision.
- PLOS warns that references are linked electronically and must be formatted correctly; Scientific Reports states that it does not copy-edit references, so the submitted bibliography must be correct at acceptance [S8, S9].

### Gaps

- The exact BibTeX style file used by IEEE JBHI is not named on the JBHI author page; IEEEtran is the standard route but was not verified from a JBHI-specific file.
- MDPI free-format submission means the final ACS style may be applied by copyeditors, so the author's exact reference punctuation may change.

## 5. What supplementary material is allowed, and how should per-sample predictions or code be linked?

### Takeaway

All target journals allow supplementary files. MDPI is the most permissive on format and size (any format, 120 MB total). JBHI counts supplementary material toward its 14-page limit. Scientific Reports and npj Digital Medicine prefer a single merged PDF supplement. For per-sample predictions, the safest cross-journal route is a CSV or XLSX supplementary table plus a code repository DOI cited in the Data Availability Statement.

### Cited Findings

| Journal | Supplementary material | Size and format notes | Code and per-sample prediction linking |
|---|---|---|---|
| Diagnostics, Cancers, Bioengineering | Allowed as "Supplementary Files" in any format, common non-proprietary formats recommended [S1, S2, S3] | Total upload for all files must not exceed 120 MB [S1, S2, S3] | Release code in a recognized public repository or upload as supplementary; name and version of all software used [S1] |
| CBM | Allowed; cite all supplementary files in the manuscript text and include a caption [S4] | Files appear online as received and are not typeset [S4] | Data and code deposit encouraged under Option B; no separate code statement [S4] |
| CMPB | Allowed; cite in text and caption each file [S5] | Files appear online as received, not typeset; only add or replace at revision [S5] | Data statement encouraged; code deposit encouraged [S5] |
| AIM | Allowed; cite in text and caption each file [S6] | Files appear online as received, not typeset [S6] | Data statement encouraged; code deposit encouraged [S6] |
| MIA | Allowed; cite in text and caption each file [S10] | Files appear online as received, not typeset [S10] | Option C requires data deposit and link, or a statement explaining why not [S10] |
| IEEE JBHI | Permitted, but the 14-page regular-paper limit and 16-page review limit include supplementary material [S7] | No separate size or format rule stated [S7] | No code or data statement specified [S7] |
| Scientific Reports | Allowed; combine and supply as a single separate file, preferably PDF [S8] | No appendices in the main body; all supplementary information in dedicated files [S8] | Data Availability Statement mandatory; code repository link can be included there [S8] |
| PLOS Digital Health | Supporting Information allowed; label files S1 Appendix, S1 Table, S2 Table, etc. [S9] | Any format; supporting information guidelines apply [S9] | Code must be available without restriction if it underpins findings; repository with DOI, open license, documentation, and Data Availability Statement link [T14] |
| npj Digital Medicine | Allowed; all cited supplementary information in a separate, single merged PDF [S11] | Supplementary Methods are not permitted; all Methods in the main manuscript [S11] | Code availability statement in Methods for custom code central to conclusions [S11] |

Additional supplementary facts:

- MDPI states: "For work where novel computer code was developed, authors should release the code either by depositing in a recognized, public repository or uploading as supplementary information to the publication" [S1].
- MDPI requires the name and version of all software used to be clearly indicated [S1].
- PLOS states that all author-generated code underpinning the findings must be available without restrictions upon publication, and may require code as a condition of publication when central [T14].
- PLOS recommends sharing code through a repository that issues persistent identifiers, with an open-source license and documentation, and stating access in the Data Availability Statement [T14].
- npj Digital Medicine requires a "Code availability" statement in the Methods for custom code deemed central to the conclusions [S11].
- Scientific Reports requires a Data Availability Statement in all manuscripts [S8].
- MIA's Option C requires deposit in a relevant repository, a link during submission, and a statement if sharing is not possible [S10].
- JBHI's page limit explicitly includes supplementary material [S7].

### Inferences

- For this paper, deposit code and per-sample predictions in Zenodo or Figshare with a DOI, cite the DOI in the Data Availability Statement, and include a CSV of predictions as a supplementary table.
- A prediction CSV should include at least image or sample ID, true label, predicted probability, threshold, and split. That schema is an inference, not a journal rule.
- Avoid "data available on request" for PLOS Digital Health; it is not compliant unless a specific legal or ethical restriction is justified [T13, T14].
- For MDPI, a single ZIP supplement containing the prediction CSV, code snapshot, and README is acceptable within the 120 MB cap.
- For JBHI, supplementary material competes with the 14-page limit, so a large prediction CSV may need to be a repository link rather than an appendix.

### Gaps

- No journal specifies a standard schema for per-sample predictions or a required file naming convention beyond PLOS's S1/S2 labels.
- JBHI does not state a supplementary format or a code availability requirement.
- Scientific Reports does not explicitly require a code availability statement in its submission guidelines, even though its Data Availability Statement is mandatory.

## 6. What is the cover letter requirement, and are suggested reviewers required?

### Takeaway

A cover letter is mandatory at all three MDPI journals, IEEE JBHI, Scientific Reports, PLOS Digital Health, and npj Digital Medicine. The Elsevier guides for CBM, CMPB, AIM, and MIA do not mention a cover letter. Reviewer suggestions are mandatory at MDPI (three) and PLOS (at least four), expected in the Scientific Reports cover letter, encouraged at JBHI (four), and optional at npj.

### Cited Findings

| Journal | Cover letter | Suggested reviewers |
|---|---|---|
| Diagnostics, Cancers, Bioengineering | Required with every submission; must explain significance, context, and fit to the journal [S1, S2, S3] | Three potential reviewers with detailed contact information; not current collaborators and no co-publication with any author in the last three years; different institutions [S1] |
| CBM | Not mentioned in the guide for authors [S4] | Not mentioned [S4] |
| CMPB | Not mentioned [S5] | Not mentioned [S5] |
| AIM | Not mentioned [S6] | Not mentioned [S6] |
| MIA | Not mentioned [S10] | Not mentioned [S10] |
| IEEE JBHI | Required in the text box in IEEE Author Portal; must state innovation and significance, originality, prior manuscript ID for revisions, and conference-paper extension details [S7] | Four potential reviewers encouraged; must be independent and from a different institution to all authors [S7] |
| Scientific Reports | Required; must include corresponding author affiliation and contact, why the work fits the journal, suitable reviewers, excluded reviewers, and any prior discussion with an editor [S8] | Names and contact information of suitable reviewers should be included in the cover letter [S8] |
| PLOS Digital Health | Required as a separate file; must explain why the manuscript is suitable for the journal, why the work is important, and what it adds [S9] | At least four potential reviewers; avoid recent collaborators and researchers at the same institution [S9] |
| Medical Image Analysis | Not mentioned [S10] | Not mentioned [S10] |
| npj Digital Medicine | Required; must include corresponding author affiliation and contact, context and importance, confidential conflicts or related work, and may include suggested or excluded referees [S11] | Optional; suggested or excluded referees can be included in the cover letter [S11] |

Additional cover-letter facts:

- MDPI requires two statements in every cover letter: "We confirm that neither the manuscript nor any parts of its content are currently under consideration for publication with or published in another journal," and "All authors have approved the manuscript and agree with its submission to [Journal]" [S1, S2, S3].
- MDPI says proposed and excluded reviewers should be entered in the submission system, not in the cover letter [S1].
- JBHI requires the cover letter to confirm that the manuscript is entirely original and has not been copyrighted, published, submitted, or accepted elsewhere [S7].
- Scientific Reports also asks whether the authors have had prior discussions with a Scientific Reports editor about the work [S8].
- PLOS Digital Health allows authors to recommend an Academic Editor, but the editors may choose another [S9].

### Inferences

- A modular cover letter should contain a journal-specific fit paragraph, the mandatory originality and author-approval statements, a data and code availability sentence, and a reviewer section that can be removed for MDPI.
- For MDPI, keep reviewer names out of the cover letter and enter them in SuSy as instructed [S1].
- For PLOS, the cover letter should explicitly address open data and code, because those policies are central to the journal's scope [S9, T13, T14].
- For JBHI, the cover letter should state the regular-paper category and confirm that the 14-page limit is met.

### Gaps

- Elsevier's submission systems may offer optional cover letter and reviewer fields that are not documented in the guide for authors.
- JBHI's current Author Portal may request statements beyond those listed on the public author page.
- No journal states a word limit for the cover letter.

## 7. Are there special requirements for AI/medical imaging papers, such as model cards, data statements, or reproducibility checklists?

### Takeaway

No journal in this set explicitly requires a model card. The reproducibility burden is enforced through data and code availability statements, reporting guidelines (STARD for diagnostic accuracy and TRIPOD for prediction models), and the Nature Portfolio Reporting Summary at npj Digital Medicine. MIA is the strictest on data deposit, requiring Option C rather than encouraging sharing.

### Cited Findings

| Journal | AI or medical imaging specific requirement | Reporting checklist | Data and code |
|---|---|---|---|
| Diagnostics, Cancers, Bioengineering | Generative AI tools cannot be authors; GenAI use must be disclosed in Materials and Methods and, for text, data, graphics, study design, or analysis, in Acknowledgments [S1, S2, S3] | No AI-specific checklist or model card requirement stated [S1, S2, S3] | Data Availability Statement mandatory; code release encouraged [S1] |
| CBM | Generative AI declaration section required before references; AI must not create or alter primary observed images [S4] | Endorses reporting guidelines; names STARD for diagnostic accuracy, TRIPOD for prediction models, REMARK for biomarkers, PRISMA and MOOSE for reviews, CARE and SRQR for descriptive data, TIDIER and CHEERS for interventions [S4] | Option B data deposit encouraged; code release encouraged [S4] |
| CMPB | Generative AI declaration section required; no AI-specific checklist stated [S5] | Data statement encouraged; no named AI checklist [S5] | Option B data deposit encouraged [S5] |
| AIM | Generative AI declaration required; clinical assessment of usefulness and impact strongly recommended; mere application of well-known algorithms is not considered original [S6] | No AI-specific checklist stated [S6] | Option B data deposit encouraged [S6] |
| MIA | Generative AI declaration required; highlights and graphical abstract required; Discussion section required [S10] | No AI-specific checklist stated [S10] | Option C data deposit required, with link or explanation [S10] |
| IEEE JBHI | IRB approval number required for human studies; public dataset source and citation required [S7] | No AI-specific checklist stated [S7] | No separate data or code statement specified [S7] |
| Scientific Reports | Data Availability Statement mandatory; no AI-specific checklist or model card requirement stated [S8] | No Nature Portfolio Reporting Summary requirement is mentioned in the submission guidelines [S8] | Data availability mandatory; code availability not explicit [S8] |
| PLOS Digital Health | Data and code sharing policies apply to AI and imaging studies [T13, T14] | Diagnostic accuracy studies should conform to STARD [T15] | Data available without restriction; code available where it underpins findings [T13, T14] |
| npj Digital Medicine | Nature Portfolio AI policy applies; LLMs cannot be authors and use must be documented in Methods [S11] | Nature Portfolio Reporting Summary and Editorial Policy Checklist required [S11] | Data Availability mandatory; Code availability where applicable [S11] |

Additional AI and imaging facts:

- Elsevier's GenAI declaration must be a new section before the reference list, titled "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process" [S4, S5, S6].
- Elsevier states that AI tools must not be used to create or alter images that represent primary observed or experimental data, including western blots, microscopy, histology, or patient images [S4, S5, S6].
- MDPI's Diagnostics journal has detailed original-image requirements for experimental images, including uncropped originals and image-processing disclosure; this is not an AI-specific rule but applies to imaging figures [S1].
- CBM explicitly names TRIPOD for studies that develop, validate, or update a prognostic or diagnostic prediction model [S4].
- PLOS explicitly requires STARD for reports of diagnostic accuracy [T15].
- npj Digital Medicine requires the Nature Portfolio Reporting Summary and Editorial Policy Checklist, and the Reporting Summary is published with accepted primary research articles [S11].
- No author guideline in S1 to S11 mentions a "model card" requirement [S1 to S11].

### Inferences

- For a melanoma AI paper, prepare a STARD or TRIPOD+AI checklist, a Data Availability Statement, a Code Availability statement, calibration and decision-curve results, and per-sample predictions. A short model card is useful internal documentation even if no journal requires it.
- If targeting npj Digital Medicine, plan for the Nature Portfolio Reporting Summary and Editorial Policy Checklist before the revision deadline.
- If targeting PLOS Digital Health, plan for open data and open code, and use STARD if the paper reports diagnostic accuracy.
- If targeting MIA, prepare a repository deposit and a graphical abstract before submission; the data deposit is not optional.
- If the paper is a review, Scientific Reports is not eligible and PLOS Digital Health is unlikely unless commissioned; JBHI would require pre-approval and a 16-page review build.

### Gaps

- No journal explicitly requires a model card or a separate data statement beyond the Data Availability Statement; absence of mention is not proof that a submission portal will not ask for one.
- IEEE and PLOS AI-specific editorial policies were not fully reviewed beyond the pages cited.
- MIA's review article requirements were not verified.
- The CMPB abstract limit conflict (250 versus 350 words) remains unresolved and should be checked with the editorial office.
- APCs and waiver policies were outside the scope of this note and were not collected.


