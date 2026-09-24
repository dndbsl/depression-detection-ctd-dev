# Reference verification and changes

Audit date: 24 September 2026. Scope: the 17 unique works cited by the current main.tex. All 17 cited works are covered by 16 source-supplied BibTeX entries and one explicitly authorized manual webpage entry. An additional supplied entry describes a different, currently uncited conference paper and is retained separately. The 18 uncited legacy database records are archived and are not claimed as verified. The README webpage entry is manually authored with explicit approval; it is not an official export. No RIS, NBIB or CSL citation-format conversion was used.

## `valstar2016avec` — acquired

**AVEC 2016: Depression, mood, and emotion recognition workshop and challenge**

Acquisition: University BibTeX display.

Source: [https://ibug.doc.ic.ac.uk/publications/by_year/2016/](https://ibug.doc.ic.ac.uk/publications/by_year/2016/)

Verification: [https://ibug.doc.ic.ac.uk/media/uploads/documents/1605.01600.pdf](https://ibug.doc.ic.ac.uk/media/uploads/documents/1605.01600.pdf)

Original: [originals/valstar2016avec.bib](originals/valstar2016avec.bib); SHA-256 `94925652575bf389684c08175eab377376184389ba732ccb13ef24152074fb00`.

Bibliographic fields checked against official publication/institutional source; exceptions documented below.

| Field | Verified working value |
| --- | --- |
| author | Valstar, Michel and Gratch, Jonathan and Schuller, Bj{\"o}rn and Ringeval, Fabien and Lalanne, Denis and Torres Torres, Mercedes and Scherer, Stefan and Stratou, Giota and Cowie, Roddy and Pantic, Maja |
| pages | 3--10 |
| booktitle | Proceedings of the 6th International Workshop on Audio/Visual Emotion Challenge |
| title | {AVEC 2016: Depression, mood, and emotion recognition workshop and challenge} |
| year | 2016 |
| doi | 10.1145/2988257.2988258 |

Changes:

- Corrected AVEC capitalization and author byline against the university-hosted author manuscript with ACM publication imprint; encoded Torres Torres as the complete family name, restored Schuller’s accent, and used the published Michel Valstar byline rather than the repository’s M. F. Valstar. Added DOI from its publication imprint.
- Protected the complete title with exactly two whole-title brace layers; retained necessary internal TeX groups.

Notes:

- Selected the 10-author AVEC workshop paper, pp. 3–10. A Twente repository record conflates its DOI with the different six-author “Summary for AVEC 2016”, pp. 1483–1484; that record was rejected. University PDF carries an arXiv stamp, but the selected BibTeX is explicitly the published conference record, not an arXiv replacement.

## `cummins2015review` — acquired

**A Review of Depression and Suicide Risk Assessment Using Speech Analysis**

Acquisition: University BibTeX display.

Source: [https://vgl.ict.usc.edu/bibtexbrowser.php?bib=ICT.bib&key=cummins_review_2015](https://vgl.ict.usc.edu/bibtexbrowser.php?bib=ICT.bib&key=cummins_review_2015)

Verification: [https://www.sciencedirect.com/science/article/pii/S0167639315000369](https://www.sciencedirect.com/science/article/pii/S0167639315000369)

Original: [originals/cummins2015review.bib](originals/cummins2015review.bib); SHA-256 `dd0f453ad3604bcdd7128541bac94a58a7ce468eaeffedf4eebceb831a78b1bf`.

Bibliographic fields checked against official publication/institutional source; exceptions documented below.

| Field | Verified working value |
| --- | --- |
| title | {A Review of Depression and Suicide Risk Assessment Using Speech Analysis} |
| volume | 71 |
| doi | 10.1016/j.specom.2015.03.004 |
| journal | Speech Communication |
| author | Cummins, Nicholas and Scherer, Stefan and Krajewski, Jarek and Schnieder, Sebastian and Epps, Julien and Quatieri, Thomas F. |
| year | 2015 |
| pages | 10 -- 49 |

Changes:

- Export key cummins_review_2015 mapped to existing manuscript key cummins2015review; no manuscript citation key changed.
- Removed the resolver URL prefix from the DOI field (official export held http://dx.doi.org/10.1016/j.specom.2015.03.004); DOI identity unchanged.
- Protected the complete title with exactly two whole-title brace layers; retained necessary internal TeX groups.

Notes:

- Publisher page confirms Speech Communication 71, July 2015, pp. 10–49. Original spacing in pages={10 -- 49} retained. No issue number supplied or inferred.

## `yamamoto2020timing` — acquired

**Using speech recognition technology to investigate the association between timing-related speech features and depression severity**

Acquisition: Publisher BibTeX download.

Source: [https://journals.plos.org/plosone/article/citation/bibtex?id=10.1371/journal.pone.0238726](https://journals.plos.org/plosone/article/citation/bibtex?id=10.1371/journal.pone.0238726)

Verification: [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0238726](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0238726)

Original: [originals/yamamoto2020timing.bib](originals/yamamoto2020timing.bib); SHA-256 `aef2a04cbd2044f9bea4f97c0c430b58df98e7abd036effec31b18e9471b8ab3`.

Bibliographic fields checked against official publication/institutional source; exceptions documented below.

| Field | Verified working value |
| --- | --- |
| doi | 10.1371/journal.pone.0238726 |
| author | Yamamoto, Mao AND Takamiya, Akihiro AND Sawada, Kyosuke AND Yoshimura, Michitaka AND Kitazawa, Momoko AND Liang, Kuo-ching AND Fujita, Takanori AND Mimura, Masaru AND Kishimoto, Taishiro |
| journal | PLOS ONE |
| title | {Using speech recognition technology to investigate the association between timing-related speech features and depression severity} |
| year | 2020 |
| volume | 15 |
| pages | e0238726 |
| number | 9 |

Changes:

- Export key 10.1371/journal.pone.0238726 mapped to existing manuscript key yamamoto2020timing; no manuscript citation key changed.
- Corrected exported pages={1-10} (PDF-internal page range) to the publisher’s bibliographic article identifier e0238726. This is a documented metadata correction, not visual normalization.
- Protected the complete title with exactly two whole-title brace layers; retained necessary internal TeX groups.
- Normalized existing month {09} to unquoted macro sep.

Notes:

- Publisher article page: 15(9):e0238726, 11 September 2020; all nine authors and DOI match.

## `aldeneh2019mood` — acquired

**Identifying Mood Episodes Using Dialogue Features from Clinical Interviews**

Acquisition: Conference BibTeX copy tool.

Source: [https://www.isca-archive.org/interspeech_2019/aldeneh19_interspeech.html](https://www.isca-archive.org/interspeech_2019/aldeneh19_interspeech.html)

Verification: [https://www.isca-archive.org/interspeech_2019/aldeneh19_interspeech.html](https://www.isca-archive.org/interspeech_2019/aldeneh19_interspeech.html)

Original: [originals/aldeneh2019mood.bib](originals/aldeneh2019mood.bib); SHA-256 `6f5947d467f4dd208257649afa5d565fea246d3540064096a281d04616dddce6`.

Bibliographic fields checked against official publication/institutional source; exceptions documented below.

| Field | Verified working value |
| --- | --- |
| title | {Identifying Mood Episodes Using Dialogue Features from Clinical Interviews} |
| author | Zakaria Aldeneh and Mimansa Jaiswal and Michael Picheny and Melvin G. McInnis and Mower Provost, Emily |
| year | 2019 |
| booktitle | {Interspeech 2019} |
| pages | 1926--1930 |
| doi | 10.21437/Interspeech.2019-1878 |

Changes:

- Export key aldeneh19_interspeech mapped to existing manuscript key aldeneh2019mood; no manuscript citation key changed.
- Encoded Emily Mower Provost as Mower Provost, Emily to preserve the compound family name; ISCA’s ungrouped natural-order name otherwise renders E. M. Provost. Confirmed by the University of Michigan faculty directory.
- Protected the complete title with exactly two whole-title brace layers; retained necessary internal TeX groups.

Notes:

- Family-name evidence: https://eecs.engin.umich.edu/people/mower-provost-emily/. Other metadata agrees with the ISCA page; no month inferred.

## `chou2019joint` — acquired

**JOINT LEARNING OF CONVERSATIONAL TEMPORAL DYNAMICS AND ACOUSTIC FEATURES FOR SPEECH DECEPTION DETECTION IN DIALOG GAMES**

Acquisition: User-supplied publisher BibTeX export, imported 24 September 2026.

Uploaded filename: `IEEE Xplore Citation BibTeX Download 2026.9.24.2.18.48.bib`.

Source: [https://ieeexplore.ieee.org/document/9023050](https://ieeexplore.ieee.org/document/9023050)

Verification: [https://biic.ee.nthu.edu.tw/research.php?id=27](https://biic.ee.nthu.edu.tw/research.php?id=27)

Verification: [https://www.apsipa.org/proceedings/2019/pdfs/63.pdf](https://www.apsipa.org/proceedings/2019/pdfs/63.pdf)

Original: [originals/chou2019joint.bib](originals/chou2019joint.bib); SHA-256 `815061b80400b1ab1009893d338e2b4c704b0b87f27d01569f2995e654cbe236`.

User-supplied export matched to the previously verified work; bibliographic fields checked against official sources listed below. Export transport itself was supplied by the user, not independently reproduced.

| Field | Verified working value |
| --- | --- |
| author | Chou, Huang-Cheng and Liu, Yi-Wen and Lee, Chi-Chun |
| booktitle | 2019 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC) |
| title | {JOINT LEARNING OF CONVERSATIONAL TEMPORAL DYNAMICS AND ACOUSTIC FEATURES FOR SPEECH DECEPTION DETECTION IN DIALOG GAMES} |
| year | 2019 |
| volume |  |
| number |  |
| pages | 1044-1050 |
| doi | 10.1109/APSIPAASC47483.2019.9023050 |

Changes:

- Changed export key 9023050 to chou2019joint; existing manuscript citation keys were preserved.
- Protected the complete title with double braces; retained exported capitalization and internal TeX. Field names normalized to lowercase during serialization (BibTeX is case-insensitive).

Notes:

- Original uploaded bytes preserved in originals/; uploaded filename retained in provenance and the pre-import archive. No manual citation-format conversion. No absent month was added.

Previous acquisition limitation, now resolved by the uploaded file: IEEE citation-export request returned HTTP 418. Conference and university sources verify the publication but no BibTeX was acquired from them.

## `chou2021automatic` — acquired

**Automatic Deception Detection using Multiple Speech and Language Communicative Descriptors in Dialogs**

Acquisition: Publisher citation tool, active BibTeX style download.

Source: [https://www.cambridge.org/core/services/aop-citation-tool/download/?downloadType=bibtex&productIds=11F85303C702CD6C7EAFCB4B5BE8E683&citationStyle=bibtex](https://www.cambridge.org/core/services/aop-citation-tool/download/?downloadType=bibtex&productIds=11F85303C702CD6C7EAFCB4B5BE8E683&citationStyle=bibtex)

Verification: [https://www.cambridge.org/core/journals/apsipa-transactions-on-signal-and-information-processing/article/automatic-deception-detection-using-multiple-speech-and-language-communicative-descriptors-in-dialogs/11F85303C702CD6C7EAFCB4B5BE8E683](https://www.cambridge.org/core/journals/apsipa-transactions-on-signal-and-information-processing/article/automatic-deception-detection-using-multiple-speech-and-language-communicative-descriptors-in-dialogs/11F85303C702CD6C7EAFCB4B5BE8E683)

Verification: [https://biic.ee.nthu.edu.tw/research.php?id=23](https://biic.ee.nthu.edu.tw/research.php?id=23)

Verification: [https://www.nowpublishers.com/article/OpenAccessDownload/SIP-169](https://www.nowpublishers.com/article/OpenAccessDownload/SIP-169)

Original: [originals/chou2021automatic.bib](originals/chou2021automatic.bib); SHA-256 `ede1f285076c7adb6e18bf0ebad3fa8eebea29626265a18c18d2a65d4b55a0e6`.

Authors, title, journal, volume, year, article identifier e5 and DOI agree with the original Cambridge publication.

| Field | Verified working value |
| --- | --- |
| title | {Automatic Deception Detection using Multiple Speech and Language Communicative Descriptors in Dialogs} |
| volume | 10 |
| doi | 10.1017/ATSIP.2021.6 |
| journal | APSIPA Transactions on Signal and Information Processing |
| author | Chou, Huang-Cheng and Liu, Yi-Wen and Lee, Chi-Chun |
| year | 2021 |
| pages | e5 |

Changes:

- Changed export key Chou_Liu_Lee_2021 to existing manuscript key chou2021automatic.
- Protected the complete title with double braces.
- Removed the publisher export’s stray trailing <div></div> HTML after the BibTeX record; preserved the untouched response in originals/. No month or issue number inferred.

Notes:

- The old easybib endpoint returned an empty body; recovered the export through the active publisher citationTool interface. The downloaded attachment has a .txt filename but its contents are publisher-generated BibTeX, not locally converted text.
- Huang-Cheng Chou; Yi-Wen Liu; Chi-Chun Lee. APSIPA Transactions on Signal and Information Processing, vol. 10, e5, 2021. Published online 16 April 2021. DOI 10.1017/ATSIP.2021.6. Migrated Emerald record shows 10(1):1–9, while the original Cambridge paper identifies e5; original article identity retained in the audit.

## `fushimi2026beyond` — acquired

**Beyond Short-Frame Acoustic Features: Capturing Long-Term Speech Patterns for Depression Detection**

Acquisition: User-supplied publisher BibTeX export, imported 24 September 2026.

Uploaded filename: `technologies-v14-i04_20260924.bib`.

Source: [https://www.mdpi.com/2227-7080/14/4/198](https://www.mdpi.com/2227-7080/14/4/198)

Verification: [https://www.mdpi.com/2227-7080/14/4/198](https://www.mdpi.com/2227-7080/14/4/198)

Original: [originals/fushimi2026beyond.bib](originals/fushimi2026beyond.bib); SHA-256 `e133d15232e2dee852bec049759b9c7a30ff4d08e2d6edc541137be6a2745d85`.

User-supplied export matched to the previously verified work; bibliographic fields checked against official sources listed below. Export transport itself was supplied by the user, not independently reproduced.

| Field | Verified working value |
| --- | --- |
| author | Fushimi, Shizuku and Azani, Mohammad Aiman and Chiba, Mizuto and Okada, Yoshifumi |
| title | {Beyond Short-Frame Acoustic Features: Capturing Long-Term Speech Patterns for Depression Detection} |
| journal | Technologies |
| volume | 14 |
| year | 2026 |
| number | 4 |
| article-number | 198 |
| doi | 10.3390/technologies14040198 |

Changes:

- Changed export key technologies14040198 to fushimi2026beyond; existing manuscript citation keys were preserved.
- Protected the complete title with double braces; retained exported capitalization and internal TeX. Field names normalized to lowercase during serialization (BibTeX is case-insensitive).
- Retained article-number={198}; added note={Article 198} because IEEEtran ignores the publisher’s ARTICLE-NUMBER field. No page field was invented or changed.

Notes:

- Original uploaded bytes preserved in originals/; uploaded filename retained in provenance and the pre-import archive. No manual citation-format conversion. No absent month was added.
- Indexed official MDPI page confirms all four authors, title, Technologies 14(4), article 198, year 2026 and DOI; direct page access remains rate-limited.

Previous acquisition limitation, now resolved by the uploaded file: MDPI’s BibTeX route returned HTTP 403. The indexed official article page verifies existence and publication details; no export was fabricated.

## `burdisso2024prompts` — acquired

**DAIC-WOZ: On the Validity of Using the Therapist's prompts in Automatic Depression Detection from Clinical Interviews**

Acquisition: Conference BibTeX download.

Source: [https://aclanthology.org/2024.clinicalnlp-1.8.bib](https://aclanthology.org/2024.clinicalnlp-1.8.bib)

Verification: [https://aclanthology.org/2024.clinicalnlp-1.8/](https://aclanthology.org/2024.clinicalnlp-1.8/)

Original: [originals/burdisso2024prompts.bib](originals/burdisso2024prompts.bib); SHA-256 `6e82393b6ab56ab7be370b27503f73298b8e1c1d0591468b63387421a34021f8`.

Bibliographic fields checked against official publication/institutional source; exceptions documented below.

| Field | Verified working value |
| --- | --- |
| title | {DAIC-WOZ: On the Validity of Using the Therapist{'}s prompts in Automatic Depression Detection from Clinical Interviews} |
| author | Burdisso, Sergio  and       Reyes-Ram{\'i}rez, Ernesto  and       Villatoro-tello, Esa{\'u}  and       S{\'a}nchez-Vega, Fernando  and       Lopez Monroy, Adrian  and       Motlicek, Petr |
| booktitle | Proceedings of the 6th Clinical Natural Language Processing Workshop |
| year | 2024 |
| doi | 10.18653/v1/2024.clinicalnlp-1.8 |
| pages | 82--90 |

Changes:

- Export key burdisso-etal-2024-daic mapped to existing manuscript key burdisso2024prompts; no manuscript citation key changed.
- Protected the complete title with exactly two whole-title brace layers; retained necessary internal TeX groups.

## `agarwal2024discourse` — acquired

**Analysing relevance of Discourse Structure for Improved Mental Health Estimation**

Acquisition: Conference BibTeX download.

Source: [https://aclanthology.org/2024.clpsych-1.9.bib](https://aclanthology.org/2024.clpsych-1.9.bib)

Verification: [https://aclanthology.org/2024.clpsych-1.9/](https://aclanthology.org/2024.clpsych-1.9/)

Original: [originals/agarwal2024discourse.bib](originals/agarwal2024discourse.bib); SHA-256 `5bf2bbe861dbb8af36b7af5578d5ba3d14b044c6f9c2a7f958a200346d544a2b`.

Bibliographic fields checked against official publication/institutional source; exceptions documented below.

| Field | Verified working value |
| --- | --- |
| title | {Analysing relevance of Discourse Structure for Improved Mental Health Estimation} |
| author | Agarwal, Navneet  and       Dias, Ga{\"e}l  and       Dollfus, Sonia |
| booktitle | Proceedings of the 9th Workshop on Computational Linguistics and Clinical Psychology (CLPsych 2024) |
| year | 2024 |
| doi | 10.18653/v1/2024.clpsych-1.9 |
| pages | 127--132 |

Changes:

- Export key agarwal-etal-2024-analysing mapped to existing manuscript key agarwal2024discourse; no manuscript citation key changed.
- Protected the complete title with exactly two whole-title brace layers; retained necessary internal TeX groups.

## `gratch2014daic` — acquired

**The Distress Analysis Interview Corpus of human and computer interviews**

Acquisition: Conference BibTeX download.

Source: [https://aclanthology.org/L14-1421.bib](https://aclanthology.org/L14-1421.bib)

Verification: [https://aclanthology.org/L14-1421/](https://aclanthology.org/L14-1421/)

Original: [originals/gratch2014daic.bib](originals/gratch2014daic.bib); SHA-256 `32a5a37f5046285451aeb0d76b983392d25b91c5cc4b4059a46e535f7a3244e2`.

Bibliographic fields checked against official publication/institutional source; exceptions documented below.

| Field | Verified working value |
| --- | --- |
| title | {The Distress Analysis Interview Corpus of human and computer interviews} |
| author | Gratch, Jonathan  and       Artstein, Ron  and       Lucas, Gale  and       Stratou, Giota  and       Scherer, Stefan  and       Nazarian, Angela  and       Wood, Rachel  and       Boberg, Jill  and       DeVault, David  and       Marsella, Stacy  and       Traum, David  and       Rizzo, Skip  and       Morency, Louis-Philippe |
| booktitle | Proceedings of the Ninth International Conference on Language Resources and Evaluation ({LREC}'14) |
| year | 2014 |
| pages | 3123--3128 |

Changes:

- Export key gratch-etal-2014-distress mapped to existing manuscript key gratch2014daic; no manuscript citation key changed.
- Protected the complete title with exactly two whole-title brace layers; retained necessary internal TeX groups.

## `kroenke2009phq8` — acquired

**The PHQ-8 as a measure of current depression in the general population**

Acquisition: User-supplied publisher BibTeX export, imported 24 September 2026.

Uploaded filename: `S0165032708002826.bib`.

Source: [https://www.sciencedirect.com/science/article/pii/S0165032708002826](https://www.sciencedirect.com/science/article/pii/S0165032708002826)

Verification: [https://www.sciencedirect.com/science/article/pii/S0165032708002826](https://www.sciencedirect.com/science/article/pii/S0165032708002826)

Verification: [https://pubmed.ncbi.nlm.nih.gov/18752852/](https://pubmed.ncbi.nlm.nih.gov/18752852/)

Original: [originals/kroenke2009phq8.bib](originals/kroenke2009phq8.bib); SHA-256 `a7877728dc806d8e2b79b9af6b92d7941c7e388f231bc6d9cd28fc68a0b57aa7`.

User-supplied export matched to the previously verified work; bibliographic fields checked against official sources listed below. Export transport itself was supplied by the user, not independently reproduced.

| Field | Verified working value |
| --- | --- |
| title | {The PHQ-8 as a measure of current depression in the general population} |
| journal | Journal of Affective Disorders |
| volume | 114 |
| number | 1--3 |
| pages | 163-173 |
| year | 2009 |
| doi | 10.1016/j.jad.2008.06.026 |
| author | Kurt Kroenke and Tara W. Strine and Robert L. Spitzer and Janet B. W. Williams and Joyce T. Berry and Ali H. Mokdad |

Changes:

- Changed export key KROENKE2009163 to kroenke2009phq8; existing manuscript citation keys were preserved.
- Protected the complete title with double braces; retained exported capitalization and internal TeX. Field names normalized to lowercase during serialization (BibTeX is case-insensitive).
- Corrected exported issue 1 to combined issue 1–3, confirmed by the NLM/PubMed record; pages 163-173 retained.
- Removed https://doi.org/ from the DOI field; the DOI identity and URL metadata are unchanged.
- Separated Janet B.W. Williams’s initials as Janet B. W. Williams so BibTeX retains both verified middle initials.

Notes:

- Original uploaded bytes preserved in originals/; uploaded filename retained in provenance and the pre-import archive. No manual citation-format conversion. No absent month was added.
- Issue year 2009 retained; online publication 27 August 2008 and print issue April 2009 are distinct. ScienceDirect direct access remains blocked; combined issue verified with NLM/PubMed.

Previous acquisition limitation, now resolved by the uploaded file: ScienceDirect citation-export request returned HTTP 403. PubMed supplies NBIB, which was not converted into an alleged official BibTeX export.

## `baileyDaicProcessing` — manual entry, explicitly authorized

**DAIC-WOZ Processing: List of Known Errors**

Manually authored webpage BibTeX from the verified official repository README, with explicit user authorization.

Authorization: User explicitly instructed “cite it then” after being told the entry would be manually created, on 2026-09-24.

Source: [https://github.com/adbailey1/daic_woz_process/blob/380372c5e8ef342bd3811342b838cbb610afda10/README.md](https://github.com/adbailey1/daic_woz_process/blob/380372c5e8ef342bd3811342b838cbb610afda10/README.md)

Repository owner Andrew Bailey confirmed by public GitHub profile; pinned README bytes match the archived source; version date confirmed by README-specific commit metadata.

The repository supplies no official BibTeX for this webpage. No purported original export was created. The title is a descriptive label for the repository and its relevant README section; year 2021 is the cited version year.

| Field | Verified working value |
| --- | --- |
| author | Bailey, Andrew |
| title | {DAIC-WOZ Processing: List of Known Errors} |
| year | 2021 |
| url | https://github.com/adbailey1/daic_woz_process/blob/380372c5e8ef342bd3811342b838cbb610afda10/README.md |
| note | {README}, version of September 6, 2021. Accessed: September 24, 2026. [Online]. Available: \url{https://github.com/adbailey1/daic_woz_process/blob/380372c5e8ef342bd3811342b838cbb610afda10/README.md} |

Source snapshots:

- [originals/baileyDaicProcessing.versioned-README.md](originals/baileyDaicProcessing.versioned-README.md); SHA-256 `0071f77fffc8098431932dd7b9c4e93086a4ede841255c53d8c4b4ea3210d1be`.
- [originals/bailey.readme-commits.json](originals/bailey.readme-commits.json); SHA-256 `c5a51711c67c8610386a52b39497875c16c6d6fd1624be21e29aa243314e2aa7`.
- [evidence/bailey-owner.profile.json](evidence/bailey-owner.profile.json); SHA-256 `4071657dcc9d527d3681ee7bc07485e3ea7f6c84ae53740e2a8906a1c70febdf`.

Manual construction and corrections:

- Created a manual @misc webpage entry under the existing manuscript citation key; no manuscript citation was renamed.
- Used a descriptive title identifying the repository and the List of known errors section, protected with double braces.
- Used 2021 from the actual README version date (6 September 2021); did not infer a date from another paper’s bibliography. No month field added.
- Preserved the public commit-specific GitHub URL as metadata and duplicated it in the note so the webpage link remains printed with CTLuse_url={no}.
- Added the verified version date and actual access date to the note; no DOI, journal, volume or pages invented.
- Protected the README acronym inside the note with an internal brace group after IEEEtran initially rendered it as rEADME. Explicitly applied the same correction to the new individual snapshot; this is not automatic synchronization.

Notes:

- This entry is explicitly manual, not a publisher/author BibTeX export. The source README is preserved in originals/; no artificial export file was placed there.
- The related Bailey/Plumbley gender-bias paper remains separately retained as bailey2021gender.bib and is not substituted for the webpage.
- The README directly lists interruptions, absent interviewer transcripts, timing misalignment and the session-409 label error.

## `cao2025pdch` — acquired

**A Multimodal Depression Consultation Dataset of Speech and Text with HAMD-17 Assessments**

Acquisition: Publisher BibTeX download.

Source: [https://citation-needed.springer.com/v2/references/10.1038/s41597-025-05817-9?format=bibtex&flavour=citation](https://citation-needed.springer.com/v2/references/10.1038/s41597-025-05817-9?format=bibtex&flavour=citation)

Verification: [https://www.nature.com/articles/s41597-025-05817-9](https://www.nature.com/articles/s41597-025-05817-9)

Original: [originals/cao2025pdch.bib](originals/cao2025pdch.bib); SHA-256 `262ba7fb411ac23cd470e6ed1ab73c56362c59c7a6c0fb865af946e1d24fecdc`.

Bibliographic fields checked against official publication/institutional source; exceptions documented below.

| Field | Verified working value |
| --- | --- |
| author | Cao, Pengfei and Zhang, Yuanzhe and Zhang, Chenxiang and Chen, Wei and Liu, Yan and Xu, Shuang and Xu, Miao and Jin, Wenqing and Xu, Jinjie and Wang, Dan and Wang, Wei and Wang, Xue and Wang, Wen and Ren, Yanping and Zhao, Jun and Li, Rena and Liu, Kang |
| title | {A Multimodal Depression Consultation Dataset of Speech and Text with HAMD-17 Assessments} |
| journal | Scientific Data |
| year | 2025 |
| volume | 12 |
| number | 1 |
| pages | 1577 |
| doi | 10.1038/s41597-025-05817-9 |

Changes:

- Export key Cao2025 mapped to existing manuscript key cao2025pdch; no manuscript citation key changed.
- Protected the complete title with exactly two whole-title brace layers; retained necessary internal TeX groups.
- Normalized existing month "Sep" to unquoted macro sep.

Notes:

- Publisher confirms all 17 authors, Scientific Data 12, article 1577, 29 September 2025 and DOI. Export number=1 and pages=1577 retained; no Art. no. normalization.

## `hamilton1960rating` — acquired

**A RATING SCALE FOR DEPRESSION**

Acquisition: User-supplied publisher BibTeX export, imported 24 September 2026.

Uploaded filename: `a-rating-scale-for-depression.bib`.

Source: [https://jnnp.bmj.com/content/23/1/56](https://jnnp.bmj.com/content/23/1/56)

Verification: [https://pmc.ncbi.nlm.nih.gov/articles/PMC495331/](https://pmc.ncbi.nlm.nih.gov/articles/PMC495331/)

Original: [originals/hamilton1960rating.bib](originals/hamilton1960rating.bib); SHA-256 `d616053cbc120ce6114479b6c49da7837034732053900ac7a1205f3531af9817`.

User-supplied export matched to the previously verified work; bibliographic fields checked against official sources listed below. Export transport itself was supplied by the user, not independently reproduced.

| Field | Verified working value |
| --- | --- |
| author | Hamilton, Max |
| title | {A RATING SCALE FOR DEPRESSION} |
| volume | 23 |
| number | 1 |
| pages | 56--62 |
| year | 1960 |
| doi | 10.1136/jnnp.23.1.56 |
| journal | Journal of Neurology, Neurosurgery \& Psychiatry |

Changes:

- Changed export key Hamilton56 to hamilton1960rating; existing manuscript citation keys were preserved.
- Protected the complete title with double braces; retained exported capitalization and internal TeX. Field names normalized to lowercase during serialization (BibTeX is case-insensitive).
- Retained the already escaped journal ampersand and eprint PDF URL; this is a published journal article, not an arXiv reference. No missing month was inferred.

Notes:

- Original uploaded bytes preserved in originals/; uploaded filename retained in provenance and the pre-import archive. No manual citation-format conversion. No absent month was added.
- BMJ-provided PMC scan confirms Max Hamilton, title, volume 23, issue 1, pages 56–62 and year 1960. Export’s all-capital title matches the paper and was retained.

Previous acquisition limitation, now resolved by the uploaded file: BMJ article page returned HTTP 403, preventing discovery of a usable publisher BibTeX export. The publisher-supplied PMC scan verifies the paper, but PMC offers NBIB, not BibTeX. No conversion made.

## `ferrer2024confidence` — acquired

**Confidence Intervals for evaluation in machine learning**

Acquisition: Author-supplied BibTeX block in official software README.

Source: [https://raw.githubusercontent.com/luferrer/ConfidenceIntervals/master/README.md](https://raw.githubusercontent.com/luferrer/ConfidenceIntervals/master/README.md)

Verification: [https://github.com/luferrer/ConfidenceIntervals](https://github.com/luferrer/ConfidenceIntervals)

Original: [originals/ferrer2024confidence.bib](originals/ferrer2024confidence.bib); SHA-256 `d842d5310fc2966ce0321481cc85a8a36746cb67fd24643c28719de86a2e3c41`.

Bibliographic fields checked against official publication/institutional source; exceptions documented below.

| Field | Verified working value |
| --- | --- |
| author | Ferrer, Luciana and Riera, Pablo |
| title | {Confidence Intervals for evaluation in machine learning} |

Changes:

- Export key Confidence_Intervals mapped to existing manuscript key ferrer2024confidence; no manuscript citation key changed.
- Changed unsupported @software to IEEEtran-compatible @misc, preserving the supplied fields; added a software/online-resource note so its primary resource link remains visible.
- Protected the complete title with exactly two whole-title brace layers; retained necessary internal TeX groups.

Notes:

- Author-supplied BibTeX has no publication year or month. The legacy 2024 is unsupported and was not copied; the existing citation key is retained. This entry was supplied as BibTeX, not converted from the accompanying plain-text recommendation.

## `chen2022wavlm` — acquired

**WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing**

Acquisition: Author organisation BibTeX download (Microsoft Research).

Source: [https://www.microsoft.com/en-us/research/publication/wavlm-large-scale-self-supervised-pre-training-for-full-stack-speech-processing/bibtex/](https://www.microsoft.com/en-us/research/publication/wavlm-large-scale-self-supervised-pre-training-for-full-stack-speech-processing/bibtex/)

Verification: [https://doi.org/10.1109/JSTSP.2022.3188113](https://doi.org/10.1109/JSTSP.2022.3188113)

Original: [originals/chen2022wavlm.bib](originals/chen2022wavlm.bib); SHA-256 `f9d04f5e0790be4223d14c16acbd8abad8b318fd7512df35cb190e9151d527b0`.

Bibliographic fields checked against official publication/institutional source; exceptions documented below.

| Field | Verified working value |
| --- | --- |
| author | Chen, Sanyuan and Wang, Chengyi and Chen, Zhengyang and Wu, Yu and Liu, Shujie and Chen, Zhuo and Li, Jinyu and Kanda, Naoyuki and Yoshioka, Takuya and Xiao, Xiong and Wu, Jian and Zhou, Long and Ren, Shuo and Qian, Yanmin and Qian, Yao and Wu, Jian and Zeng, Michael and Yu, Xiangzhan and Wei, Furu |
| title | {WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing} |
| year | 2022 |
| pages | 1505-1518 |
| journal | IEEE Journal of Selected Topics in Signal Processing |
| volume | 16 |
| number | 6 |
| doi | 10.1109/JSTSP.2022.3188113 |

Changes:

- Added the DOI verified against IEEE’s publication record; Microsoft’s official export omitted it.
- Protected the complete title with exactly two whole-title brace layers; retained necessary internal TeX groups.
- Normalized existing month {July} to unquoted macro jul.

Notes:

- Microsoft export month July corresponds to online publication on 4 July 2022. IEEE issue 16(6), pp. 1505–1518 is October 2022; retained exported month and year, without silently replacing the online date. Both occurrences of Jian Wu retained in the 19-author byline, as supplied by the publication.

## `liu2019roberta` — acquired

**RoBERTa: A Robustly Optimized BERT Pretraining Approach**

Acquisition: arXiv BibTeX download.

Source: [https://arxiv.org/bibtex/1907.11692](https://arxiv.org/bibtex/1907.11692)

Verification: [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692)

Original: [originals/liu2019roberta.bib](originals/liu2019roberta.bib); SHA-256 `367893fe7053b1eeb99e50cf68b65f8455bf4ec9c9b562329d250f342927952e`.

Bibliographic fields checked against official publication/institutional source; exceptions documented below.

| Field | Verified working value |
| --- | --- |
| title | {RoBERTa: A Robustly Optimized BERT Pretraining Approach} |
| author | Yinhan Liu and Myle Ott and Naman Goyal and Jingfei Du and Mandar Joshi and Danqi Chen and Omer Levy and Mike Lewis and Luke Zettlemoyer and Veselin Stoyanov |
| year | 2019 |

Changes:

- Export key liu2019robertarobustlyoptimizedbert mapped to existing manuscript key liu2019roberta; no manuscript citation key changed.
- Added note={arXiv:1907.11692}, because IEEEtran.bst v1.14 has no eprint field and ordinary URLs are suppressed.
- Protected the complete title with exactly two whole-title brace layers; retained necessary internal TeX groups.

Notes:

- arXiv confirms all ten authors, exact title, identifier and 2019 initial publication year. No journal venue, issue or pages invented.

## `bailey2021gender` — retained separately: different cited work

The uploaded IEEE file is a valid entry for **Gender Bias in Depression Detection Using Audio Features**, by Andrew Bailey and Mark D. Plumbley, EUSIPCO 2021, pp. 596–600, DOI 10.23919/EUSIPCO54536.2021.9615933. It is not the “List of Known Errors” README cited by the manuscript. It was renamed to [bailey2021gender.bib](bailey2021gender.bib), with a matching new key and a double-braced title, and is **excluded from references_official.bib**. The existing `baileyDaicProcessing` key now resolves to the separately authorized manual README webpage entry.

Source: [https://ieeexplore.ieee.org/document/9615933](https://ieeexplore.ieee.org/document/9615933)

Verification: [https://openresearch.surrey.ac.uk/esploro/outputs/conferenceProceeding/Gender-Bias-in-Depression-Detection-Using/99571523302346](https://openresearch.surrey.ac.uk/esploro/outputs/conferenceProceeding/Gender-Bias-in-Depression-Detection-Using/99571523302346)

Original: [originals/bailey2021gender.bib](originals/bailey2021gender.bib); SHA-256 `79b04ad96194133a37324340fde74bcf8e9fc4a86166dad22fa04d72425239b1`.

IEEE exports year 2021. Surrey records first online publication on 8 December 2021 and publication date March 2022. The exported year was retained; the repository’s later date was recorded rather than silently substituted. Original uploaded filename and acquisition details are recorded in provenance.json.

## LaTeX integration and archive

The manuscript uses IEEEtran and references_official exactly; the local .latexmkrc supplies the canonical search paths. Because article/spconf does not define the IEEE control command, a small bstctlcite definition writes the control record to the auxiliary file before the first citation. An initial nocite alias produced a spurious warning for the deliberately unprinted control record; that was corrected and is absent from the final logs. Existing url/hyperref packages support the visible software link.

Pre-change main.tex, main.pdf and auxiliary files were copied to archive/20260924T090359Z/. The legacy references.bib was retired to archive/legacy-references.bib. Manuscript prose and citation keys are unchanged. The canonical bibliography was backed up again before adding the recovered Cambridge export.

Individual .bib files are initial snapshots, not automatically synchronized with the combined file. Editors, publishers, abstracts and keywords were not forced into a common schema. No journal/conference-name abbreviation pass was applied. Complete titles are protected and only existing month fields were normalized. See validation_report.json and README.md for compilation outcomes.

Before this import, the five uploaded files, existing bibliography, reports, validation artifacts, and manuscript files were backed up in archive/user-import-20260924T092500Z/. All five uploads also have byte-identical originals in originals/. The checker now accepts hyphenated field names such as ARTICLE-NUMBER and only requires an arXiv note when archiveprefix is arXiv; BMJ’s eprint field is a PDF URL.

Full standalone compilation passes for 17 references. Full manuscript compilation produces a five-page PDF with all 17 citations resolved and no BibTeX errors/warnings or LaTeX citation warnings. Underfull-line layout diagnostics remain; no overfull boxes were reported. Reference pages 4–5, the article identifier, author output, PHQ issue/initials, webpage URL and dates, and visible software/arXiv information were inspected.

The manual webpage addition was authorized on 24 September 2026 after the user reviewed the distinction between the README and the related conference paper. Pre-addition bibliography, reports, manuscript and validation artifacts are backed up in archive/manual-webpage-20260924T093156Z/. A further backup preserves the first manual entry before protecting README capitalization. No manuscript prose or citation keys were changed.

On 24 September 2026, the user requested disabling repeated-author dashes. Added `CTLdash_repeated_names = {no}` to the combined bibliography’s control entry. References 5 and 6 now both print H.-C. Chou, Y.-W. Liu, and C.-C. Lee. Previous files are backed up in archive/repeated-authors-20260924T093558Z/. No reference metadata or citation keys changed.
