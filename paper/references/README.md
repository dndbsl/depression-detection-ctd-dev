# Canonical references

Updated 24 September 2026. This is the sole canonical references folder for `../main.tex`.

**All 17 cited works now have bibliography entries: 16 source-supplied BibTeX entries and one explicitly authorized manual webpage entry.** The combined file contains **18 entries: 17 references and one IEEEtran control record**. Fourteen reference entries are publisher/conference/university/arXiv exports (four supplied by the user); two are supplied by the authors’ official sources: Microsoft Research for WavLM and the Ferrer/Riera software README. A separate supplied Bailey conference-paper entry is retained outside the combined file because it is a different cited object. The Bailey README entry was manually created with the user’s explicit approval on 24 September 2026 and is labeled as such in the BibTeX comments and audit records. It is not an official export. No RIS, NBIB or CSL conversion was used.

## Files and editing

- [references_official.bib](references_official.bib) is the **editing source of truth** after consolidation.
- Individual key-named `.bib` files are working snapshots from their acquisition/import. The separate `bailey2021gender.bib` is not included in the combined bibliography. **They do not automatically synchronize with the combined file.**
- [originals/](originals/) preserves downloaded exports and source-page snapshots. Embedded BibTeX was extracted from its supplied HTML/Markdown block, not constructed from metadata. Original HTML/README sources are retained alongside these extracted blocks.
- [provenance.json](provenance.json) and [verification_report.md](verification_report.md) record each cited work’s source, acquisition method, verification, original-export hash and corrections.
- [missing_entries.md](missing_entries.md) records that no manuscript entries are missing; the README still lacks an official export and is covered by the authorized manual entry.
- [archive/](archive/) holds pre-change backups, the legacy 35-entry database and historical assembly material. The legacy database’s 18 uncited entries were preserved but are outside this audit.
- [validation_report.json](validation_report.json) records the checks. [Bibliography preview](validation/bibliography-check.pdf) contains all 17 references; [manuscript build](validation/main.pdf) resolves every citation. The manuscript PDF is also saved at `../main.pdf`.

## User upload import and webpage resolution

All five uploaded files were renamed, and their entry keys were matched to the filenames:

| Uploaded file | Renamed file | Included in combined file |
| --- | --- | --- |
| IEEE Xplore … 2.18.48.bib | `chou2019joint.bib` | Yes |
| technologies-v14-i04_20260924.bib | `fushimi2026beyond.bib` | Yes |
| S0165032708002826.bib | `kroenke2009phq8.bib` | Yes |
| a-rating-scale-for-depression.bib | `hamilton1960rating.bib` | Yes |
| IEEE Xplore … 2.21.25.bib | `bailey2021gender.bib` | No: related paper, not cited webpage |

`baileyDaicProcessing` now cites the repository’s “List of Known Errors” README using a manually created entry authorized by the user. The separate Bailey upload describes “Gender Bias in Depression Detection Using Audio Features” and remains outside the combined bibliography. Full original filenames and source details are in provenance.json. No manuscript citation key or prose was changed in this import.

Before modification, the uploaded files and existing files were backed up in [archive/user-import-20260924T092500Z/](archive/user-import-20260924T092500Z/). Untouched uploads are also preserved in originals/ under the corresponding working key names. The four recovered publications have been removed from the unavailable list.

## Authorized manual webpage entry

[baileyDaicProcessing.bib](baileyDaicProcessing.bib) is a snapshot of the manually created entry in the canonical combined bibliography. Author Andrew Bailey is confirmed by the repository owner’s public GitHub profile. The descriptive title identifies the repository and the “List of known errors” section.

The entry links to the exact README commit `380372c5e8ef342bd3811342b838cbb610afda10`, dated **6 September 2021**, with access date **24 September 2026**. The public URL remains visible through the note despite `CTLuse_url = {no}`. The year comes from that README version. No month field, DOI, journal, volume or pages was invented. Internal braces preserve the `README` acronym against IEEEtran’s note capitalization.

The original README and verification metadata are preserved, but no manual BibTeX file is represented as an official export in originals/. Before adding the entry, existing files were backed up in [archive/manual-webpage-20260924T093156Z/](archive/manual-webpage-20260924T093156Z/).

## Corrections and date distinctions

- PLOS exported PDF-internal pages `1-10`; the working entry uses the publisher’s bibliographic identifier `e0238726`. Other page/article fields were preserved; no systematic “Art. no.” changes were made.
- The AVEC university export’s byline, compound family name and `AVEC` capitalization were corrected against the paper; its DOI was added from the publication imprint. WavLM’s omitted DOI was also added. Cummins’s DOI resolver prefix was removed from the DOI field.
- `Mower Provost, Emily` is encoded with the complete family name, confirmed by the university directory.
- WavLM’s exported July 2022 month represents online publication; the journal issue is October 2022. The exported month is retained as `jul` and the distinction is documented.
- PHQ-8 appeared online on 27 August 2008 and in the April **2009** issue. Both dates are recorded in its verification details. The new export’s issue `1` was corrected to `1--3` against NLM/PubMed; the DOI resolver prefix was removed and Janet B.W. Williams’s initials separated as `B. W.` for correct BibTeX output.
- Ferrer/Riera supply no year: the legacy `2024` was not copied, while `ferrer2024confidence` remains the citation key. Their supplied `@software` was changed to IEEEtran-compatible `@misc`; a note preserves the primary software URL in print.
- RoBERTa has `note = {arXiv:1907.11692}` because IEEEtran ignores `eprint`. Ordinary URLs remain metadata and are suppressed by the control entry.
- Cambridge’s export contains trailing `<div></div>` HTML. The working record removes it, maps the citation key and protects the title; the original response remains unchanged.

- MDPI’s `article-number = {198}` is retained; `note = {Article 198}` was added because IEEEtran ignores the publisher’s article-number field. This preserves the identifier in print without altering page/article metadata.
- Chou 2019 and Hamilton 1960 retain the capitalization supplied by their official exports, including all-capital titles. No missing months were added.
- The separate Bailey export uses 2021; Surrey records first online publication in December 2021 and a March 2022 publication date. The exported year is retained and this difference documented.

## Validation

- Actual BibTeX syntax check passed. There are no duplicate keys or structural formatting errors.
- The standalone bibliography completed a **full LaTeX–BibTeX–LaTeX compilation**, with no BibTeX errors/warnings or unresolved citations and 17 printed references. Its log contains underfull-line layout diagnostics.
- All manuscript citation keys were preserved. All 17 resolve, including `baileyDaicProcessing`.
- Inspected `.bbl` output, extracted PDF text and the manuscript reference pages (4–5), including MDPI’s visible article identifier and the corrected PHQ issue and initials. Lists longer than six authors print the first author and *et al.*; six-author lists retain all six. The database retains every author. The software URL, commit-specific GitHub webpage URL, version/access dates and arXiv identifier remain visible; ordinary URLs are suppressed.
- The **full manuscript successfully compiled to a five-page PDF with all 17 citations resolved**, with no BibTeX errors/warnings or LaTeX citation warnings. Underfull-line layout diagnostics remain; no overfull boxes were reported.

## Git storage

The canonical bibliography, individual snapshots, original BibTeX exports, source-page snapshots, verification metadata, reports and validation `.tex` source are versioned. Backup archives, generated files in `validation/`, downloaded paper PDFs and citation-download debug files remain local under `.gitignore`. The small public GitHub owner record is retained for webpage authorship verification. Links to local backups and validation artifacts refer to files in the working workspace; a fresh clone can regenerate the validation output. The reviewed manuscript PDF is versioned at `../main.pdf`.

## Building

From `paper/`:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`paper/.latexmkrc` adds this folder to `BIBINPUTS` and `BSTINPUTS`, preserving default search paths. The unmodified CTAN `IEEEtran.bst` v1.14 is included with its license text. The manuscript contains exactly:

```latex
\bibliographystyle{IEEEtran}
\bibliography{references_official}
```

The `article`/`spconf` document defines `\bstctlcite` to write the control citation directly to the auxiliary file. `\bstctlcite{IEEEexample:BSTcontrol}` appears immediately after `\begin{document}`, before the first citation. Existing `url` and `hyperref` packages support `\url`. The requested author-list and URL control values are `yes`, `6`, `1`, and `no`. The user also requested `CTLdash_repeated_names = {no}`, so consecutive references with identical authors print their names rather than a dash.

Read-only structural checks and standalone bibliography compilation:

```sh
python3 references/bibtex_audit.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=references/validation references/validation/bibliography-check.tex
```

When another entry is acquired, preserve its raw export, back up and edit the **combined file** using the existing key, update provenance/README/missing list, and rerun validation. Do not reconstruct the combined file from the individual snapshots. The initial assembly script is archived as historical material.
