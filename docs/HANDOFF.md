# Handoff — start here

You are picking up work on this repo with no prior chat history. This file is
the entry point. Read it fully before running anything.

## What this project is

Code for an SLT 2026 submission on **conversational temporal dynamics (CTD)** —
dyadic Ask/Res turn-pair timing as a standalone modality for depression
detection on DAIC-WOZ, fused with frozen WavLM-large and RoBERTa-large.

**The paper was rejected.** We are revising for **ICASSP 2027**. The Area Chair
gave three reasons: modest novelty, limited generalization, and an inconclusive
test-set improvement.

## Read these, in this order

1. `reviews/slt2026/reviews.md` — the verbatim SLT reviews, meta-review, and our
   rebuttal. **Confidential** (see constraints below).
2. `reviews/slt2026/action-items.md` — every reviewer concern as a tracked item
   with a stable ID (`SLT-01`…`SLT-22`, `MC-01`…`MC-13`). Cite these IDs in
   commit messages.
3. `docs/plan-preflight.md` — **your current assignment.** Detailed plan for
   `SLT-04`, `SLT-05`, `SLT-10`, `SLT-13`.
4. `docs/multi-corpus-plan.md` — the multi-corpus extension. **Context only, do
   not start this work yet.**

## Your assignment

Execute `docs/plan-preflight.md`: Step 0 (environment + reproduction gate),
then `SLT-13`, then `feature_groups.py`, then `SLT-04` and `SLT-05`, then
`SLT-10`. Follow the step order in that document — it encodes real
dependencies.

**Do not start any new-corpus work** (`MC-*` items, E-DAIC / PDCH / CMDC). Those
four items must land first because `SLT-04`/`SLT-05` define the feature groups
that become the multi-corpus feature tiers. Running a corpus first makes the
tier boundaries look post-hoc to a reviewer.

All four items are **CPU-only** — transcripts and labels only, no GPU, no WavLM,
no RoBERTa.

## Environment

Nothing is installed (no numpy, pandas, sklearn, scipy, or conda):

```bash
python3 -m venv ~/.venvs/ctd && source ~/.venvs/ctd/bin/activate
pip install -r src/ctd/requirements.txt confidence_intervals
export DAIC_WOZ_ROOT=~/data/DAIC
```

`torch` is listed in `requirements.txt` but no CTD-path module imports it;
skip it if it's slow to install.

## THE GATE — do this before any analysis

```bash
python src/ctd/ml_splits.py
```

Must reproduce **dev macro-F1 0.746 / test 0.631**, selected **`C=0.3`**, and
cohort counts **102 / 33 / 45**.

Every number in your assignment is a delta against this baseline. **If it does
not reproduce, stop and report it** — that is its own finding, not something to
work around.

## Hard constraints

- **Never mutate `CTD_FEATURE_NAMES`** in `src/ctd/constants.py`. It asserts
  `== 24`, and `feature_extraction.py` asserts its output keys match exactly.
  All feature subsetting happens by *column selection* downstream.
- **Never re-tune anything on the test split.** Dev is for selection, test is
  reported once. This protocol is the paper's main strength — reviewer rwao
  called it "above the norm." Preserve it.
- **Do not change the deployed model** (24-D session-mean, L2 LogReg `C=0.3`,
  `class_weight='balanced'`, threshold 0.5). Document quirks instead of fixing
  them; changing it invalidates published numbers.
- **Never `git push origin`.** `origin` is a **public** GitHub repo. Work goes
  to the `private` remote only. See git setup below.
- **`reviews/` is confidential** — those reviews were shared only with chairs
  and authors. A `pre-push` hook blocks them from reaching `origin`; do not
  bypass it with `--no-verify`.

## Git setup

```
branch    icassp2027   (tracks private/icassp2027 — plain `git push` is safe)
origin    https://github.com/dndbsl/depression-detection-ctd.git        PUBLIC
private   https://github.com/dndbsl/depression-detection-ctd-dev.git    private
tag       slt2026-submission   marks the exact state submitted to SLT
```

Commit identity is already configured. A `.git/hooks/pre-push` guard refuses any
push of `reviews/` to `origin`. Commit messages should cite item IDs, e.g.
`SLT-04: add bootstrapped coefficient table`.

Publishing to the public repo happens later via squash-merge to `main`, and is
**not** part of this assignment.

## Traps already identified — read before coding

These were found by inspection and are documented with rationale in
`docs/plan-preflight.md`. Do not rediscover them the hard way.

1. **The CTD detector is implemented twice** — `ml_splits.py::run()` (via
   `functionals.py` `__amean` columns) and `fusion/fusion_late.py::ctd_probs()`
   (via `np.nanmean` over `bags.py`). Both are NaN-aware means and *should*
   agree, but nothing asserts it. Add that assertion before ablating, or your
   ablation will silently disagree with fusion.
2. **The 24 features contain 8 reciprocal pairs by construction**
   (`ask_ud` = u/d vs `ask_du` = d/u, and 7 more). L2 splits weight arbitrarily
   between collinear partners, so a naive "largest coefficient" table is
   fragile and would invite a *new* criticism. Report three triangulating views
   as specified in §3 of the plan.
3. **Do not claim the ask-side ablation improves performance.** The deltas
   (+.006 dev, +.030 test at n=33/45) are almost certainly not distinguishable
   from zero, and this paper was already criticized for resting on a +.038 test
   delta whose CI touches zero. Let a paired bootstrap decide the wording; the
   supportable claim is "preserved."
4. **A seed sweep for `SLT-10` is the wrong instrument.** The CTD pipeline is a
   deterministic convex fit, so seed variance is exactly 0.000. Use data
   resampling instead, and write the splitter group-aware (PDCH will reuse it).
5. **`run()` selects on dev balanced accuracy while the paper headlines
   macro-F1.** Defensible but undocumented. Document it and report both — do
   not change it.
6. **`SLT-13` is also wrong in the repo**, not just the paper:
   `src/acoustic-depr-wavlm/README.md` line 21 lists 409 as excluded, but
   `daic_cleaning.py` *relabels* 409 and excludes 9 others. 189 − 9 = 180 =
   102/33/45.

## Data on disk (context only — do not use yet)

```
~/data/DAIC     DAIC-WOZ, 189 sessions          (this is DAIC_WOZ_ROOT)
~/data/E-DAIC   275 sessions — contains ALL 189 DAIC-WOZ sessions; 86 are new
~/data/PDCH     Mandarin clinical, 72 subjects / 99 sessions
~/data/CMDC     Mandarin, 78 subjects — deprioritized
```

## When you finish

Update statuses in `reviews/slt2026/action-items.md`, add a `RESULTS.md`
section for the new tables, and report: whether the gate reproduced, the
ablation paired-bootstrap outcome, and whether the selected `C` was stable
across resamples.
