# Pre-declared test family — shared-signal audit (DAIC-WOZ × PDCH)

**Written 2026-09-24, before any of the tests below were computed.** The git
commit timestamp is the evidence. This file is an *appendix* to
[`preregistration-pdch.md`](preregistration-pdch.md); it does not amend,
reinterpret or extend predictions P1–P6, which remain closed.

## Why this file exists, and what it can and cannot claim

`MC-01`/`MC-05`–`MC-08` established that the deployed 24-D CTD detector is at
chance on PDCH across all five tiers, and that severity regression is null
(`output/pdch_ctd_results.md`). The question now is **not** "can CTD be made to
work on PDCH" — fishing for a lucky feature subset after seeing a null is
exactly the dev/test-fitting `SLT-02` criticised. The question is:

> What, if anything, is *commonly* informative about depression-related labels
> across DAIC-WOZ and PDCH, and what fails for an identifiable reason?

### Disclosure: what had already been seen when this was written

This is a re-analysis, not a blind pre-registration, and the distinction is
per-quantity:

| Quantity | Already visible before this file? | Consequence |
|---|---|---|
| PDCH univariate \|r\| / AUC for the 7 tier-B features | **Yes** — `pdch_ctd_results.md` §Pre-registered predictions ranks all live tier-B features by \|r\| and gives `res_h` r = +0.158, AUC = 0.646, `ask_over_res` \|r\| = 0.255 | F1 is **re-analysis** for tier B: it re-tests known numbers under a declared multiplicity policy. It cannot claim discovery for `res_h` or `ask_over_res`. |
| PDCH top-5 fitted coefficients | **Yes** — same file | Not used as a selection criterion anywhere below. |
| PDCH within-subject Δ correlations (9 features) | **Yes** — same file, already labelled exploratory | Excluded from every family below; not re-tested. |
| PDCH univariate r for the remaining 15 (non-tier-B) features | **No** | F1 is genuinely pre-declared for these. |
| DAIC-WOZ `res_h` train/dev/test r and AUC | **Yes** — `PF-05`, `ctd_interpretability.md` | The DAIC **test** split is therefore withheld from every estimate in F1–F4 (see §Estimation sets). |
| Any Spearman/continuous association on either corpus (F2) | **No** | Pre-declared. |
| Any item-level (PHQ8_Moving / HAMD item 8, 9) association (F3) | **No** | Pre-declared. |
| Any residualized or robust-functional variant (F4) | **No** | Pre-declared. |

Everything marked "Yes" is reported below as **re-analysis**; everything marked
"No" is reported as **pre-declared**. No result from either category may be
described as demonstrating transfer.

## Estimation sets, fixed in advance

- **DAIC-WOZ:** `train + dev` pooled, n = 135 (102 + 33), label `PHQ8_Binary`.
  **The official test split (n = 45) is withheld from every association
  estimate in F1–F4** and is touched only once, in E3, to report a
  pre-declared candidate detector under the deployed protocol. This is what
  keeps `res_h`'s known test AUC (0.829) from contaminating candidate
  selection.
- **PDCH:** all 62 labelled sessions / 46 subjects, label `HAMD17_ge17`. There
  is no held-out PDCH split at this n, per `MC-08`; every PDCH model number is
  subject-grouped cross-validated.
- **Features:** the 22 live CTD features — `CTD_FEATURE_NAMES` minus `ask_bt`
  and `res_bt`, which are exactly constant on PDCH under mono webrtcvad and so
  are structurally unmeasurable there, not merely weak. Session-level
  aggregation is the deployed `__amean` column unless a family says otherwise.
- **Silence threshold:** pinned at the DAIC-deployed 0.2 s on both corpora.
  Justification fixed in advance: PDCH threshold sensitivity is already known
  to be flat (macro-F1 0.509–0.521 across 0.05–1.0 s,
  `output/pdch_ctd_results.md`), so re-selecting it would add a selection
  surface without adding information, and pinning keeps the two corpora
  directly comparable.
- **No pooling.** `PHQ8_Binary` and `HAMD17_ge17` never share a column. Every
  supervised fit is within one corpus. Cross-corpus work is comparison of
  effects, ranks and signs only.

## Test families

Multiplicity is controlled **within each family within each corpus** by
Benjamini–Hochberg FDR at **q = 0.10**. Families are corrected separately and
are reported separately; no result is promoted across families.

### F1 — Primary: binary-label univariate concordance (22 features × 2 corpora)

Per feature per corpus: point-biserial *r*, single-feature ROC AUC (scored in
the feature's native direction, so AUC > 0.5 ⇔ r > 0), and a *p*-value.

**PDCH *p*-values are permutation-based, not parametric.** 16 of 46 subjects
contribute two sessions, so session-level parametric *p* is anti-conservative.
The null is built by permuting labels **at the subject level** (a subject's
label vector moves as a unit), 5,000 permutations, seed 20260924.

Pre-declared decision rule:

- **Confirmed shared signal** ⇔ BH-q ≤ 0.10 in **both** corpora **and**
  sign(r_DAIC) = sign(r_PDCH).
- **Opposite sign** with BH-q ≤ 0.10 in both ⇒ recorded as *interpretation
  failure* (the feature means different things in the two settings), not as
  shared signal.
- **One corpus only** (BH-q ≤ 0.10 in exactly one) ⇒ *corpus-specific*.
- **Neither** ⇒ *null*, and the reason is assigned in E5 to one of: confound,
  measurement limit, or insufficient power.
- **Weak-concordance tier (EXPLORATORY, pre-declared):** same sign, \|r\| ≥
  0.15 in both corpora, and nominal p < 0.05 in at least one. This tier exists
  so that E3 is not dead-ended by a strict rule at n = 62; **any E3 result
  arising from it is exploratory and may not be called transfer.**

Global calibration, reported alongside and descriptive only (the 22 features
are rank-deficient by construction — rank 13 on PDCH, 15 on DAIC — so they are
not independent tests):

1. Sign-agreement count across the 22 features vs. Binomial(22, 0.5).
2. The subject-grouped permutation null for PDCH's **max \|r\| across the 22
   features**, which calibrates "the best-looking feature" against chance
   directly.

### F2 — Secondary: continuous-severity concordance

Spearman ρ of each of the 22 features against DAIC `PHQ8_Score` (train+dev)
and PDCH `HAMD17_total` (62 sessions). Same FDR policy, same subject-grouped
permutation null on PDCH. Separate family; a feature null in F1 but surviving
here is reported as *continuous-only* and is explicitly not a classifier claim.

### F3 — Secondary: psychomotor item-level

The narrowest available test of the psychomotor-retardation reading.

- **DAIC:** `PHQ8_Moving` ("moving or speaking so slowly that other people
  could have noticed, or the opposite — being fidgety or restless"), Spearman
  ρ, train+dev. This column ships in the official AVEC2017 label CSV; it is
  read as-is and **no DAIC label is invented or re-derived**. It is an addition
  to what the repo currently uses (`PHQ8_Binary`/`PHQ8_Score`) and is declared
  as such here.
- **PDCH:** HAMD-17 **item 8** (retardation) and **item 9** (agitation), read
  directly from the shipped item columns. **Totals are never re-derived from
  items** (`MC-07`: 7 rows code item 14 with the not-assessed sentinel 9, which
  the shipped total already excludes). Items 8 and 9 carry no sentinel and are
  used unmodified. Noted in advance: both items are empirically capped at 2 in
  this cohort despite a 0–4 range, so range restriction will attenuate ρ.

All 22 features, FDR within each (corpus, target) cell.

### F4 — Exploratory: robust and length-residualized variants

Genuinely unseen when this file was written. Every result from F4 is labelled
EXPLORATORY in the output and may not be reported as confirmation.

Column selection from the existing functional bank (**not new features**):

1. `res_h__pctl50` — session **median** response latency (robust to the long
   right tail that `__amean` is sensitive to).
2. `res_h__cv` — coefficient of variation of response latency across turns.
3. `res_h__pctlrange20_80` — within-session spread of response latency.

Newly derived and **named as exploratory**:

4. `res_h_resid`, `ask_d_resid`, `res_d_resid`, `duration_sum_resid` — the
   residual of the corresponding `__amean` after OLS regression on
   `log(dialogue span in seconds)` and `log(1 + n turn pairs)`, fitted within
   corpus. When used inside a detector the regression is refitted **within the
   training fold only**.

Rationale fixed in advance: PDCH interviews are ~2× longer than DAIC-WOZ's and
clinician-led, so absolute-duration features may be measuring interview format
rather than patient state; residualizing is the direct test of that. 7 tests,
FDR within family within corpus, binary label.

## Model-level checks (E1, E3) — no *p*-value family

These are reported as effect estimates with distributions, not as hypothesis
tests, and their success criteria are fixed here.

- **E1a — confound-matched PDCH re-test.** Positives and negatives matched on
  dialogue span and turn count (1:1 nearest neighbour on the standardized
  pair, within subject-grouping). Re-run `all24` and `no_ask` under the
  existing subject-grouped nested CV. Pre-declared meaning: the null
  **softens** only if mean macro-F1 ≥ 0.60 **and** mean ROC AUC ≥ 0.60.
  Anything less is reported as "null survives matching."
- **E1b — within-fold residualization.** The six absolute-time features
  (`ask_d`, `res_d`, `res_minus_ask`, `ask_minus_res`, `duration_sum`,
  `res_h`) residualized on log span and log turn count, fitted on the training
  fold only. Run on **both** corpora; the DAIC run is the control — if DAIC's
  signal also collapses, the deployed detector is partly a session-length
  detector, which is a finding in its own right.
- **E1c — power calibration (the null's own control).** DAIC-WOZ subsampled to
  PDCH's cohort shape — n = 62 sessions at 43.5% prevalence — 100 subsamples,
  run through the identical subject-grouped nested CV. This asks what a *known
  positive* corpus looks like at PDCH's sample size. Declared in advance: if
  the DAIC-at-n=62 macro-F1 distribution substantially overlaps PDCH's
  observed 0.520 ± 0.055, then the PDCH null is **underpowered**, and E5 must
  say so rather than attributing the null to the corpus.
- **E3 — candidate detectors.** Run **only** on features promoted by F1
  (confirmed tier, or weak-concordance tier flagged exploratory). 1-feature and
  k-feature L2 LogReg in the deployed pipeline; DAIC reported under the
  deployed train/dev/test protocol (test touched once) and PDCH under
  subject-grouped CV. **If F1 promotes nothing, E3 is skipped and reported as
  skipped.** The search is not broadened to recover a positive.

## What would count as failure of this audit

Stated so it can fail: if F1 promotes no feature, F2 and F3 promote no
feature, matching does not soften the null, and E1c shows DAIC is *well*
separated from chance at n = 62, then the honest conclusion is that CTD carries
no shared cross-corpus signal detectable here — and the ICASSP claim must be
the negative result plus its mechanism, not a transfer claim.
