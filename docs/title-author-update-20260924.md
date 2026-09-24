# Title and author update — 2026-09-24

The user’s subsequent instruction replaces the previously locked title with:

```tex
\title{Can Conversational Temporal Dynamics Improve Depression Detection in Dyads? A Preliminary Investigation in Multi-Modality Perspectives}
```

Anisha Pattanayak is the second author, ORCID `0009-0005-2556-4472`, with affiliation 1: Ming Hsieh Department of Electrical and Computer Engineering, University of Southern California, USA. Her affiliation is italicized and wraps across two lines within the template margins. Hanie Kang’s existing Computer Science affiliation is preserved as 2; the other authors’ SAIL affiliation is preserved as 3. The author names use two centered lines.

Base commit: `27289c305e3ead075ce64a734393a1c1e75636df`. The pre-edit manuscript is preserved as `paper/revision-history/main-before-title-author-20260924.tex`.

The scientific manuscript from the abstract onward is byte-identical to the pre-edit copy. All 170 ledger values and their sources are unchanged; only TeX line references were regenerated. The existing title assertion in `scripts/verify_paper_revision.py` now follows the user’s exact requested title, with this pass’s output directory as its default.

Validation:

- Rebuilt with `pdflatex -interaction=nonstopmode -halt-on-error main.tex`, `bibtex main`, and two further pdfLaTeX passes, from `paper/`.
- Five PDF pages, with technical content within the first four and references only on the fifth. No overfull boxes or undefined references/citations. First-page author order, affiliation layout and ORCID were checked.
- `python scripts/build_paper_numbers.py` refreshed line references; `python scripts/verify_paper_revision.py` passed all 152 published-number checks and verified unchanged scientific source/result hashes.
- No new experiments or scientific test reruns were needed for this title/byline-only change. The prior 31-test pass remains the last scientific test run.
- DAIC gate artifact remains byte-identical: MD5 `ca0b9cb2dd4c799b3a7c12532ceafd7e`.

Logs and machine-readable validation: `output/title_author_20260924/`. Current local PDF SHA-256: `4f2ad0dfdf683f689562f96e8e73b1be7a1a287e6d1502b55b239e83b6bd72ac`.

The earlier presentation audit and manifest describe the prior title/byline snapshot. This note records the user-requested superseding metadata change. Changes are committed locally on `icassp2027`; no push or external author communication is performed.
