# SLT 2026 → ICASSP 2027 — Action Items

Derived from [`reviews.md`](reviews.md) (verbatim reviews). Every item cites the
reviewer(s) who raised it so we can trace each change back to a real request.

**How to use this file**

- Reference an item ID in commit messages, e.g. `SLT-04: add per-feature
  coefficient table`. That keeps the review trail attached to the code history.
- Update the `Status` column as work lands. Statuses: `todo`, `wip`, `done`,
  `wontfix` (add a one-line reason under the item if `wontfix`).
- `P0` items are the three reasons the Area Chair gave for rejection. If we only
  fix `P1`/`P2` items, we will very likely be rejected again for the same
  reasons.

**The rejection, in one sentence.** The protocol and the CTD framing were
*liked*; we were rejected for **modest novelty, limited generalization, and an
inconclusive test-set improvement**. Note the split verdict: d9s3 said accept
(4), rwao borderline (3), HoqK reject (2).

---

## P0 — Reasons for rejection (must fix)

| ID | Item | Raised by | Status |
|---|---|---|---|
| SLT-01 | **Generalization beyond one split.** No cross-corpus, cross-lingual, or subgroup validation. Large dev→test drop (CTD .746→.631; full .804→.669). Single fixed partition. | HoqK, rwao, d9s3 | done for bounded scope — nested DAIC controls lead Results with mean±SD figure; PDCH boundary evidence consolidated (2026-09-24) |
| SLT-02 | **Test gain is not statistically established.** T+CTD .631→.669 (+.038) with paired CIs touching zero; test CIs wide and overlapping. Central claim rests on dev. | HoqK, rwao | done for bounded scope — fusion secondary and explicitly inconclusive; neural rerun unavailable |
| SLT-03 | **Novelty is thin.** 24-feature set is borrowed from Chou et al. [13,14]; timing-helps was already shown by Aldeneh et al. [9] and Fushimi et al. [10]. Contribution reads as "known features, new task." | HoqK (Novelty 2), rwao (3) | done — established features; nested controls and protocol sensitivity lead the bounded empirical contribution |

### SLT-01 — Generalization

The single highest-value fix. Three reviewers asked, and d9s3 (our most
confident reviewer, Confidence 5, who voted accept) named cross-corpus
explicitly as the main gap.

**Now planned in detail in [`docs/multi-corpus-plan.md`](../../docs/multi-corpus-plan.md)**,
against the actual data in `~/data`: DAIC-WOZ + E-DAIC-Extended (86 disjoint
sessions) + **PDCH** (Mandarin clinical, 62 labeled sessions). CMDC is demoted
to an optional robustness check because it cannot express dyadic timing at all
(8 of 24 features). Findings from that audit that change the approach:

- **E-DAIC fully contains DAIC-WOZ** (all 189 sessions; only 86 are new). The
  corpora must be defined as disjoint sets and the overlap disclosed, or this
  becomes an integrity problem rather than a generalization win.
- **PDCH supports all 24 features**, but only after VAD refinement — its
  turn-level 1 s timestamps leave 10 of 24 features constant or NaN as-is.
- **The three corpora span three interviewer regimes** (wizard-driven,
  autonomous, human clinician), which converts the ask-side confound from a
  defensive ablation into a designed experiment.

Still to do independently of the corpora: per-subgroup breakdowns (gender,
severity band) — cheap, and rwao's Ethics score of 4 hints at it.

Code: `src/common/daic_cleaning.py`, `src/ctd/ml_splits.py`,
`src/fusion/run_fusion_wavlm_seeds.py`.

### SLT-02 — Statistical strength

Do not re-submit with a +.038 test delta whose CI touches zero as the headline.
Either assess the effect's stability through nested resampling (which does not
increase the number of independent subjects), collect independent participants,
or reframe the paper's claim to something the evidence supports — e.g.
"CTD is competitive with 1024-d encoders at 24 dimensions" is defensible on
current numbers, whereas "fusion improves over the best single modality" is not.

Keep the parameter-free `mean_prob[T+CTD]` result (dev .738 / test .650) as
evidence the trend is not purely dev-tuning — it was the rebuttal's strongest
counterpoint and neither reviewer rebutted it.

Code: `src/fusion/paired_delta_analysis.py`.

### SLT-03 — Novelty

The rebuttal's framing ("transferring dyadic CTD to depression as a standalone
modality") was explicitly judged insufficient by the Area Chair, so restating it
will not work. Needs an actual new contribution. Candidates:

- Make the interpretability claim the *contribution* rather than an aside: a
  validated, clinically-readable timing account of psychomotor retardation
  (builds on SLT-04/SLT-05, where we already have results in hand).
- Cross-corpus transfer of CTD as the headline (pairs with SLT-01).
- Model the temporal *structure* (within-session trajectory, turn-pair
  sequence) rather than session-mean pooling — session-mean over 24 features
  discards all ordering, which is a genuine methodological gap a reviewer could
  see as new.

---

## P1 — Major concerns

| ID | Item | Raised by | Status |
|---|---|---|---|
| SLT-04 | **Interpretability claimed but never used.** No coefficients, no feature-group ablation, no per-feature correlation — despite a plain L2 LogReg that yields coefficients for free. | HoqK (major) | done |
| SLT-05 | **Ask-side (wizard) confound not ruled out.** Response-side-only ablation was left as future work. | rwao (major), HoqK | done for bounded scope — nested comparisons prioritized; original split is sensitivity, CIs including zero are not equivalence; confounding unresolved |
| SLT-06 | **No prior work re-run on our split.** Tables II/III contain only our own models. Agarwal & Dias (CLPsych 2024) report .80 test; Milintsevich et al. (2023) .74 — a ~.13 gap never mentioned. | HoqK (major), d9s3 | wontfix in this pass — Agarwal context sentence, explicitly unmatched and not rerun; no local matched-baseline artifacts |
| SLT-07 | **Also report on the official 107/35/47 split.** Reporting only the cleaned 102/33/45 split forfeits comparability with all prior work "for no stated gain." | HoqK | done (CTD) — 107/35/47 sensitivity; T/T+CTD unavailable without neural artifacts |
| SLT-08 | **Weak acoustic baseline may drive two headline results.** WavLM test .545 is below common numbers; frozen probe, utterance-level pooling (turn-level collapses to .333). Both "CTD beats the encoders" and "zero acoustic weight" may be probe-specific. | HoqK, rwao | done for bounded scope — frozen-probe/test-seed disclosure + seed mean; stronger acoustic rerun deferred |
| SLT-09 | **Convex-weight grid too coarse.** Step 0.1 means "exactly zero" acoustic weight could be a rounding artifact. | rwao | analysis done, needs writing up |
| SLT-10 | **No seed/variability sweep for CTD or fusion.** Reported for the neural baselines but not for the CTD detector or the fusion system itself. | rwao | done (CTD only; fusion sweep is separate, out of scope for this pass) |
| SLT-11 | **Code repository was empty at review time.** Directly cost us Reproducibility 3 from HoqK and was named in his summary line. | HoqK | done |

### SLT-04 / SLT-05 — Already answered in rebuttal, must land in the paper

We have these results; they were never in the submitted paper. From the
rebuttal:

- Largest coefficient is ask-side (`ask_d`).
- Dropping all nine ask-only features **preserves** performance:
  dev .752 / test .661 vs. full .746 / .631. The signal is dyadic, not
  wizard-driven.
- Largest response-side effect is response latency `res_h`, direction-consistent
  on train/dev/test — consistent with psychomotor retardation.

This is a strong result that rebuts rwao's major concern (b) *and* feeds
SLT-03's novelty angle. It needs to become a table and a figure, with the
generating code committed and reproducible — not a rebuttal footnote.

Note the ask-only ablation *beats* the full model on both dev and test. Worth
investigating rather than burying: it suggests the nine ask-side features are
adding noise, which would motivate a leaner response-side CTD set.

Code: `src/ctd/ml_splits.py`, `src/ctd/constants.py`,
`src/ctd/feature_extraction.py`.

**Done — SLT-04.** `src/ctd/interpret.py`. Confirms both rebuttal claims:
largest |coefficient| is `ask_d`, and `res_h` is sign-consistent on
train/dev/test. Reports three triangulating views (bootstrap-CI coefficients,
dev permutation importance, per-split univariate direction) plus a
collinearity diagnostic and pair-level aggregation, per Step 3a. The
collinearity check found a **new, more severe issue than documented**: several
feature pairs are *exactly* linearly dependent (|r| = 1.0) at the
session-mean level — not just the 8 documented reciprocal pairs, e.g.
`ask_ud`/`ask_sd` also correlate at exactly −1.0. This makes the design matrix
rank-deficient, so plain VIF is numerically meaningless (blows up uniformly to
~1e15 for all 24 features); the script detects and reports the exact
dependencies directly instead. Outputs: `output/ctd_interpretability.{json,md}`,
`output/ctd_coef_forest.png`.

**Done — SLT-05.** `src/ctd/ablation_groups.py`. Runs all 5 configs
(`all24`/`no_ask`/`res_only`/`ask_only`/`no_cross`) through the identical
deployed protocol (LogReg L2, select C on **dev balanced accuracy** — not
macro-F1, matching the documented-but-undocumented selection metric — refit
train+dev, test once). Paired bootstrap (B=2000) on the all24−no_ask macro-F1
delta: dev −0.068 [−0.216, 0.083], test −0.010 [−0.089, 0.054] — **both CIs
include zero**, so the verdict is "preserved," not "improves," per Step 4a.

**Important: these numbers differ from the rebuttal's reported no_ask figures**
(dev .752 / test .661). The rebuttal reused the full model's `C=0.3` for the
ablation configs rather than reselecting C on dev per config, as the deployed
protocol requires. Reselecting gives `no_ask` `C=1.0`, dev macro-F1 0.814,
test macro-F1 0.641 — the test delta vs. `all24` shrinks from the rebuttal's
claimed +.030 to +.010, which only strengthens "preserved." This should be
corrected in the manuscript, not just the rebuttal-era number reused. Output:
`output/ctd_ablation_groups.{json,md}`.

### SLT-08 — Acoustic baseline

Rebuttal position was to scope the zero-weight finding as probe-specific and
report the six-seed mean (dev .667±.053 / test .506±.034) instead of the single
seed-44 number. That is honest but defensive. To actually close this, run at
least one *fine-tuned* WavLM configuration or an alternative pooling window. If
acoustics still earn zero weight with a stronger probe, the finding becomes
genuinely interesting instead of a possible artifact.

Code: `src/acoustic-depr-wavlm/` (`train.py`, `model/attention_pool.py`,
`features/wavlm_extractor.py`).

### SLT-09 — Grid granularity

Finer grid (step .02) already run: dev optimum stays at (0, .3, .7). Add to the
paper. Code: `src/fusion/run_fusion_wavlm_seeds.py`.

### SLT-10 — Variability via data resampling

A seed sweep is the wrong instrument for the CTD detector (`SimpleImputer ->
StandardScaler -> LogisticRegression(lbfgs)` is a deterministic convex fit;
seed variance is exactly 0.000). Reframed as 100x repeated stratified-group
resampling of train+dev (group-aware from the start via `session_id`, so the
splitter is reusable for PDCH's 2-sessions/subject structure — `MC-08`):

- Held-out macro-F1 across repeats: **0.630 ± 0.069** (deployed dev number:
  0.746 — a single split can land well above or below this typical range).
- **C is not stable**: modal selection is a **tie** between C=0.1 and C=1.0
  (26% of repeats each); the deployed C=0.3 wins only 19% of repeats. Report
  C=0.3 as one plausible choice among several, not uniquely optimal.
- **Head-to-head vs. RoBERTa (test macro-F1 0.631): CTD wins 0% of 100
  repeats** (mean 0.611 ± 0.031). The published "CTD ties RoBERTa on test"
  result sits at or near the ceiling of what this resampling procedure ever
  reproduces — a materially more honest (and more cautious) answer to "is CTD
  really the best single modality" than resting on the one official split, and
  it should replace that framing in the paper, not merely supplement it.

Also added `tests/test_ctd_consistency.py` (plan-preflight.md Step 0a,
previously skipped): confirms `ml_splits.py`'s `__amean` path and
`fusion/fusion_late.py`'s `np.nanmean`-over-`bags.py` path produce identical
session-mean matrices and identical fitted probabilities, so the SLT-04/05/10
findings above transfer to the deployed fusion system rather than being an
artifact of one code path.

Code: `src/ctd/resampling.py`. Output: `output/ctd_resampling.{json,md}`.
Fusion-weight variability (as opposed to the CTD detector alone) is separate,
GPU-dependent work and out of scope for this CPU-only pass.

### SLT-11 — Code release

Resolved: the repository is public at
`https://github.com/dndbsl/depression-detection-ctd` with the full pipeline.
**Action for resubmission:** cite this stable URL directly, not a
`shorturl.at` redirect (the rebuttal used one, and the reviewer had already been
burned by a dead link). Consider archiving a release to Zenodo for a DOI.

---

## P2 — Presentation, framing, and corrections

| ID | Item | Raised by | Status |
|---|---|---|---|
| SLT-12 | **Abstract oversells.** Gives 0.804/0.669 with no dev-tuning caveat. Clarify dev is primary and test is reported for reference. | HoqK, rwao | done — bounded abstract; no reliable fusion advantage |
| SLT-13 | **Session count arithmetic.** "Excluding 10 sessions leaves 180" implies 190 to start, but DAIC-WOZ ships 189. | HoqK | done |
| SLT-14 | **Title reads awkwardly** — "in Multi-Modality Perspectives". | HoqK | done — shortened empirical-study title |
| SLT-15 | **Table II: T+CTD and A+T+CTD rows are identical** (zero acoustic weight); explained but confusing at a glance. | HoqK, d9s3 | done — duplicate three-way row collapsed with zero-weight disclosure |
| SLT-16 | **Table II should restate deployed decision thresholds inline** (currently text-only). | rwao | done — operating thresholds inline in Table I |
| SLT-17 | **eGeMAPS 240-D variant has no numbers.** Mentioned as "did not beat" the compact model; give at least one comparison score. | rwao | wontfix in paper — richer-functional comparison omitted for length; canonical rerun retained in audit |
| SLT-18 | **Consolidate the results narrative into a delta table.** Sec. VII-B/C leans on many closely-spaced numbers. | rwao | done — compact result tables and three result sections |
| SLT-19 | **Add the in-text prior-work comparison into Table 2.** | d9s3 | done — small, explicitly non-matched published-context table |
| SLT-20 | **Literature is dated** — nearly half is >5 years old; lacking recent work. | d9s3 | done for scope — recent timing and prompt-shortcut references retained |
| SLT-21 | **Reference formatting** — inconsistent arXiv preprint vs. venue style. | rwao | done for cited references — official IEEEbib style, resolved citations |
| SLT-22 | **Ethics discussion is thin** (rwao scored Ethics 4, "minor issues or missing discussion"). | rwao | wip — deployment limits included; author ethics/funding confirmation pending |

---

## Follow-ups from the preflight run (`SLT-04/05/10/13` executed)

Gate reproduced (`all24` dev macro-F1 .746 / test .631, `C=0.3`); all 6 tests
pass. New items arising from the results.

| ID | Item | Why | Status |
|---|---|---|---|
| PF-01 | **Fix the SLT-10 head-to-head comparison.** It reports "CTD beats RoBERTa in 0% of repeats" by comparing CTD's *resampled distribution* against RoBERTa's *deployed seed-43* number (.631). RoBERTa's seed distribution is **.604 ± .025** (README:97, RESULTS.md:70). Against that, CTD's .611 ± .031 is marginally *ahead*. Compare distribution to distribution. | The current framing is both unfair to CTD and methodologically wrong (distribution vs. best-of-3). Would be needlessly self-damaging in the paper. | todo |
| PF-02 | **Report effective dimensionality: the 24-D matrix has rank 15.** The collinearity check was pairwise (\|r\|>0.999) and so missed multi-feature dependencies. Verified: `duration_sum` = `ask_d` + `res_d`, `res_minus_ask` = `res_d` − `ask_d`, `ask_minus_res` = −that, all exact to ~1e-14. `np.linalg.matrix_rank` on the 180×24 centered session-mean matrix returns **15**. | The paper calls this a "compact 24-D descriptor"; it is really 15-D. Better to state it ourselves than have a reviewer find it. Extend the diagnostic to SVD/rank, not just pairwise r. | superseded by `SA-04` (algebraic reason; rank 13 of the 22 live features on both corpora) |
| PF-03 | **Unify the selection metric.** `ablation_groups.py` selects `C` on dev **macro-F1**; `resampling.py` selects on dev **balanced accuracy**; `ml_splits.py` selects on **balanced accuracy**. Pick one, document it. | Two new scripts disagree with each other and with the deployed pipeline. | done — balanced-accuracy selection verified; stale ablation JSON description corrected |
| PF-04 | **`res_only` collapses — update the corpus plan.** Tier C (8 response-only features) gives dev .607 / test .517, near chance, while `no_ask` (cross+res, 15) gives .814/.641. The signal lives in the **cross-speaker** group. | Makes CMDC effectively unusable for CTD (tier C only) and makes PDCH VAD refinement mandatory rather than optional, since `res_h` is a cross feature. | done in [`multi-corpus-plan.md`](../../docs/multi-corpus-plan.md) |
| PF-05 | **`res_h` is the strongest univariate feature on test** (r = .528, AUC = **.829**, sign-consistent across train/dev/test) — higher test AUC than the full 24-D model achieves. Suggests a parsimonious `res_h`-centric detector may generalize better. | Strong lead for ICASSP and confirms the psychomotor-retardation story. **But it is a post-hoc test-set observation** — selecting on it would be test fitting. Pre-commit it and validate on PDCH. | **retired 2026-09-24** — not reproducible under resampling; see `SA-01` |

### What the ablation actually showed (for the write-up)

| Config | n | Dev macro-F1 | Test macro-F1 |
|---|---:|---:|---:|
| `all24` | 24 | 0.746 | 0.631 |
| `no_ask` (cross + res) | 15 | **0.814** | **0.641** |
| `no_cross` (ask + res) | 17 | 0.700 | 0.527 |
| `ask_only` | 9 | 0.594 | 0.593 |
| `res_only` | 8 | 0.607 | 0.517 |

The paired bootstrap on `all24` − `no_ask` includes zero on both splits
(dev −0.068 [−0.216, +0.083]; test −0.010 [−0.089, +0.054]), so the claim is
**"preserved," not "improved"** — correctly handled by the run.

Two readings worth carrying into the paper. First, dropping the nine ask-side
features does not hurt, which answers rwao's and HoqK's confound concern
(`SLT-05`). Second, and more interesting for `SLT-03`: removing the
*cross-speaker* group is what breaks the detector, so the signal is genuinely
**relational** rather than either interviewer- or participant-side. That is a
positive, defensible novelty claim.

Note `ask_d` still dominates the fitted model (largest coefficient, −0.867,
sign-consistency 1.00; permutation importance 0.243, ~5× the next feature) even
though removing all ask features costs nothing — consistent with `ask_d`
absorbing weight it does not need.

---

## Multi-corpus work items (ICASSP 2027)

New items arising from the `~/data` audit. Full rationale in
[`docs/multi-corpus-plan.md`](../../docs/multi-corpus-plan.md).

| ID | Item | Serves | Status |
|---|---|---|---|
| MC-01 | **Corpus adapter layer.** `turn_pairing.py` hardcodes `"Ellie"`/`"Participant"`; `constants.py` hardcodes one dataset root. Needs a uniform `Utterance` interface with DAIC / E-DAIC / CMDC backends + a 24/15/8 tier selector. | prerequisite | **done (PDCH backend)** — `src/ctd/pdch_adapter.py::load_pdch_session()` emits `list[Utterance]`; `turn_pairing.build_turn_pairs()` now takes an `AskResLabels` parameter (DAIC-WOZ default unchanged) so PDCH keeps its native `医生`/`患者` tags. `feature_extraction.py`/`functionals.py` run unmodified. E-DAIC/CMDC backends still todo. |
| MC-02 | **Feature tiers 24 → 15 → 8.** Pre-commit the nested sets so cross-corpus numbers are comparable. Depends on SLT-04/05. | SLT-01, SLT-03 | tiers exercised on both corpora by the 2026-09-24 audit; see `SA-04` on effective rank |
| MC-03 | **E-DAIC diarization.** No speaker column in any of 275 transcripts. Autonomous TTS voice vs. human should make this tractable; use the ASR `Confidence` column to filter. | SLT-01 | todo |
| MC-04 | **E-DAIC-Extended experiment** (86 sessions, 600–718). Autonomous agent ⇒ wizard confound structurally absent. | SLT-01, SLT-03, SLT-05 | **recommended next P0** (2026-09-24 audit §4); blocked on `MC-03` |
| MC-05 | **PDCH chunk stitching.** Timestamps restart at `00:00` in each 1500 s chunk; must add chunk offsets before turn pairing. Drop the 2 wavs lacking timestamped transcripts. | SLT-01 | **done** — `pdch_adapter.stitch_session_turns()`; offsets from wav headers, monotonicity asserted. ⚠ inventory differs from the plan: all **167** wavs have a timestamped transcript, so 0 were dropped (not 2); 51 turns (0.16%) dropped instead (3 unparseable speaker, 48 non-monotonic starts). |
| MC-06 | **PDCH VAD refinement.** Turn-level 1 s timestamps leave **10 of 24 features constant or NaN** and quantize `res_h`. VAD inside each turn span restores intra-turn features and sub-second `res_h`. | SLT-01 | **de-risked — spike passed** (97.3% alignment; >1 utt in 78.8% of turns vs 8.2%; `res_h` median 0.60 s). Use max-overlap assignment, not clipping. → **done** — full corpus (100 sessions, 49.8 h): 5.10 utts/turn, 92.8% turn coverage, `res_h` median 0.60 s. Restores the 10 intra-turn features (54–67 → 100 distinct session values; 34–45% NaN → 0%). ⚠ **but** `ask_bt`/`res_bt` become exactly constant: mono-channel VAD cannot represent overlap, so interruption is unmeasurable, not merely rare. 22/24 live. |
| MC-07 | **PDCH label decision.** HAMD ≥ 8 gives 85% prevalence (unusable). Use HAMD-17 ≥ 17 (44%) as primary + severity regression as secondary; never pool with PHQ-8. Only 62/99 sessions have a total. | SLT-01 | **done** — `src/ctd/pdch_labels.py`. 62 labelled sessions, 27 positive (43.5%) at HAMD-17 ≥ 17; severity regression is null (MAE 6.65 vs 6.55 predict-the-mean). Note: 7 rows code item 14 as a `9` not-assessed sentinel — the shipped `total` already excludes it, so never re-derive the total from the item columns. |
| MC-08 | **PDCH subject-grouped splits.** 27 of 72 subjects have two sessions with different labels; session-level splitting leaks. Use `GroupKFold` on subject. | integrity | **done** — `pdch_experiment.py`: `StratifiedGroupKFold` on `subject_id`, 5-fold × 20 repeats, nested selection of `C` **and** the silence threshold; no fixed test split at n=62. Silence threshold re-selected (0.2 s is not favoured; sensitivity is flat, macro-F1 0.509–0.521 across 0.05–1.0 s). |
| MC-09 | **Chinese semantic encoder** or documented omission of the semantic branch for PDCH. RoBERTa-large is English-only. | SLT-01 | deprioritized behind `MC-04` (2026-09-24 audit §4) |
| MC-10 | **Environment setup.** No Python packages installed at all (no numpy/pandas/sklearn/conda). | prerequisite | todo |
| MC-11 | **Disclose the E-DAIC ⊃ DAIC-WOZ overlap in the paper**, and report only disjoint subject sets. | integrity | todo |
| MC-12 | **Licensing check** (PDCH terms; CMDC EULA if used) before any derived artifact enters the public repo. | integrity | todo |
| MC-13 | *(optional)* **CMDC as tier-C robustness check.** Second Mandarin cohort, 8 features only. Not load-bearing. | SLT-01 | deferred |

### PDCH outcome (2026-09-23) — read before planning the ICASSP submission

`MC-01`/`MC-05`/`MC-06`/`MC-07`/`MC-08` are built and run
([`output/pdch_ctd_results.md`](../../output/pdch_ctd_results.md)). **CTD does
not transfer to PDCH.** All five pre-registered tiers sit at chance (macro-F1
0.468–0.522, balanced accuracy ≈ 0.52, AUC 0.445–0.497 under subject-grouped
CV); severity regression is null too (MAE 6.65 vs a 6.55 predict-the-mean
baseline). Pre-registered outcomes: P1 fail, P2 pass, P3 fail, P4 weak pass,
P5 pass, P6 pass — but P5/P6 are satisfied trivially by a null, so this is one
weak correctly-signed `res_h` association (AUC 0.646, p = .22), not four
confirmations. Full reasoning in
[`docs/preregistration-pdch.md`](../../docs/preregistration-pdch.md) §Appendix.

This is a real finding, not a bug: the identical CV code scores macro-F1
0.652 ± 0.015 on DAIC-WOZ, `MC-06` hit every spike target, and the label join
was verified independently.

Consequences for the P0 items:

- **`SLT-01` cannot be answered the way the plan assumed.** The honest claim is
  no longer "dyadic CTD transfers cross-lingually"; it is that CTD is
  corpus-specific, and the PDCH null is itself the generalization evidence
  reviewers asked for. That is a publishable but *different* paper — it needs
  a decision before writing, not during.
- **`SLT-03` (novelty).** The three-interviewer-regime design in
  `multi-corpus-plan.md` §3 still stands, but its payoff is now a negative
  transfer result. `MC-04` (E-DAIC-Extended, same language, different
  interviewer) becomes the load-bearing test of whether the failure is
  cross-lingual or cross-corpus — worth more now than before.
- **Open confound.** PDCH interviews are ~2× longer, inpatient, and
  clinician-led; `multi-corpus-plan.md` §5 predicted session-length effects
  and they were not controlled for here beyond per-corpus standardization.
- **`MC-09` is moot for this pass.** PDCH results are **CTD-only**; no Chinese
  semantic or acoustic branch was built (out of scope per the task handoff),
  so no fusion number exists for PDCH. If the paper reports PDCH at all, state
  the missing modality explicitly rather than implying parity with DAIC-WOZ.
- **New:** `ask_bt`/`res_bt` are unmeasurable under mono-channel VAD. Any
  future corpus needing tier-A/B parity requires either per-speaker channels
  or an overlap-aware diarizer.

---

### Shared-signal audit outcome (2026-09-24) — read with the 2026-09-23 entry above

Pre-declared in [`docs/exploratory-shared-signal-pdch-daic.md`](../../docs/exploratory-shared-signal-pdch-daic.md)
before any number existed; evidence in
[`output/shared_signal_audit.md`](../../output/shared_signal_audit.md); decision
memo in [`docs/shared-signal-audit.md`](../../docs/shared-signal-audit.md).
Code: `src/ctd/shared_signal_audit.py`. DAIC-WOZ gate byte-stable throughout.

**The 2026-09-23 conclusion needs narrowing, not reversing.** That entry said
"CTD is corpus-specific, and the PDCH null is itself the generalization
evidence." The audit's power calibration shows that claim overstates what 62
sessions can support:

- DAIC-WOZ subsampled to PDCH's exact cohort shape (n = 62 at 43.5% positive,
  100 subsamples, identical subject-grouped nested CV) scores macro-F1
  **0.605 ± 0.063**. PDCH's observed 0.520 sits at the **10th percentile** of
  that distribution (z = −1.35) — low, but *inside* what a corpus with known
  signal produces at this n. On **AUC**, though, PDCH's 0.476 falls below
  **all 100** subsamples (z = −2.14). So part of the null is power and part is
  a real corpus difference; only the AUC statement is safe.
- **The null is not a length confound.** Matching positives to negatives on
  session span and turn count (optimal 1:1, span SMD −0.231 → 0.003) leaves
  `all24` at macro-F1 0.506 ± 0.049 / AUC 0.498. Within-fold residualization of
  the six absolute-time features is equally flat (0.519 → 0.492).
- **Zero features are concordant.** None of the 22 live features clears BH-FDR
  q ≤ 0.10 in both corpora; sign agreement is 12/22 (chance). PDCH's best
  feature (|r| = 0.255) has permutation p = **0.34** against the
  subject-grouped max-|r| null.

**What did hold.** `res_h` against *continuous* severity replicates in sign and
size across instruments: Spearman ρ = +0.245 on DAIC-WOZ train+dev (BH-q
0.051) and ρ = +0.267 on PDCH (p = 0.086, BH-q 0.431). It still fails as a
detector (`E3`, pre-declared criterion): PDCH CV macro-F1 0.575 ± 0.012 /
AUC 0.591.

| ID | Item | Why | Status |
|---|---|---|---|
| SA-01 | **Retire `PF-05`.** `res_h` alone scores DAIC **test** macro-F1 0.779 / AUC 0.829 but **dev 0.581** and **pooled subject-grouped CV 0.595 ± 0.006**. The 0.829 is one 45-session split, not a property of the feature. | `PF-05` is currently a live lead for ICASSP; building on it would repeat exactly the `SLT-02` error. Replacing it with the continuous-severity result above keeps the psychomotor story and loses the fragility. | done — retired favorable single-feature test headline; no stable-property claim |
| SA-02 | **Disclose that ~a fifth of the deployed detector's AUC lift is session length, and that it lives on the ask side.** Residualizing the six absolute-time features within folds moves DAIC `all24` from macro-F1 0.649 → 0.609 and AUC 0.700 → 0.658, while `no_ask` is untouched (0.601 → 0.603, AUC 0.660 → 0.663). | A reviewer who runs this check finds it; better stated by us. Also independently corroborates `SLT-05`'s conclusion that the ask tier is the confounded one. | done for scope — structure controls and ask-side sensitivity; no causal AUC decomposition |
| SA-03 | **`ask_d` reverses sign between interviewer regimes.** Only feature clearing FDR on DAIC-WOZ (r = −0.250, q = 0.066); PDCH r = **+0.231**. Wizard asks longer questions of *less*-depressed participants, human clinicians the opposite. | This is `multi-corpus-plan.md` §3's predicted confound showing up as data, and it is the concrete mechanism the negative result needs. Load-bearing for `SLT-03` novelty. | wontfix as mechanism claim — corpus changes confounded; sign reversal not promoted |
| SA-04 | **State the effective dimensionality algebraically.** Turn duration is exactly speech + silence, so `*_ud`, `*_sd`, `*_du`, `*_su`, `*_ds`, `*_us` are six monotone transforms of one number per speaker (verified to 4.4e-16 on both corpora). The centred 22-live-feature matrix has **rank 13** on both. | Supersedes `PF-02`'s pairwise finding with the reason. "Compact 24-D descriptor" is really 13-D; say so before a reviewer does. | done — observed centered rank of live features disclosed |
| SA-05 | **Lead, do not report: PDCH response timing tracks clinician-rated psychomotor items** where it does not track the HAMD-17 total or the binary label — response silence fraction vs. item 8 (retardation) ρ = −0.331, p = 0.008, BH-q = 0.062; `res_h` vs. item 9 ρ = +0.296, q = 0.066. DAIC's self-reported `PHQ8_Moving` shows nothing (max \|ρ\| = 0.195, p = 0.19). | Pre-declared (family F3), so it is not fishing — but the search-aware statistic is borderline (max-\|ρ\| permutation p = 0.072 / 0.102), item 8 is non-zero in 18 of 62 sessions, and items 8 and 9 correlate +0.338 rather than opposing. Pre-register for a future cohort. | done for scope — item exploration not promoted; PHQ8_Moving null retained |

**Next P0 recommended: `MC-04`, not `MC-09`.** Reasoning in the decision memo
§4. In one line: `MC-04` holds language, instrument, label rule and pipeline
constant and varies **only** the interviewer regime, so it is the sole
available direct test of `SA-03`; it runs at n = 86 where `E1c` shows the design
begins to resolve anything; and `MC-09` would add a modality to a corpus whose
timing channel is both at chance and under-powered.

### SLT-13 — Resolved in code, wording fix needed in paper

Verified against `src/common/daic_cleaning.py`: `KNOWN_ERRORS` holds **10
data-integrity cases**, but only **9 are excluded** (318, 321, 341, 362, 373,
444, 451, 458, 480). The 10th, **409**, is `action: "relabel"` — PHQ-8 score 10
but binary label 0 in the official file — so it is *corrected and kept*, not
dropped.

So the arithmetic is right and the prose is wrong: 189 − 9 = 180 = 102/33/45. ✓

Fix: say "ten sessions with documented integrity issues: nine excluded and one
relabeled," and state 189 as the starting count explicitly. No data or results
change.

**Done.** Fixed `src/acoustic-depr-wavlm/README.md:21` to state nine excluded
vs. one relabeled explicitly. Added `tests/test_daic_cleaning.py`, which
asserts `len(excluded_ids()) == 9`, the exact excluded set, `409 in
corrected_labels()`, and (when `DAIC_WOZ_ROOT` is set) per-split kept counts of
102/33/45 from 189 — makes the count un-driftable rather than merely fixed
once. Manuscript wording above is ready to paste into the paper.

## 2026-09-24 — Manuscript revision and nested-control follow-up

This dated update supersedes interpretive claims in earlier notes that
nonsignificance establishes preserved performance, ask-only ablation rules out
wizard confounding, or PDCH within-corpus CV measures detector transfer.
Historical numerical reports and verbatim reviews remain unchanged.

- SLT-03/04: paper reframed around robustness, with coefficient evidence,
  feature-group table and new simple-control comparison.
- SLT-05: unresolved. Nested train+dev macro-F1 is .635 full CTD versus .566
  no-ask; cross-speaker features also retain interviewer information.
- SLT-02/10: new timing-only nested evaluation completed; full text/fusion
  nested evaluation still needs missing frozen encoder caches. Resampling
  does not increase the independent subject count.
- SLT-06: published comparison added with explicit protocol differences;
  a matched baseline rerun is still outstanding.
- SLT-08: RESULTS.md documents test-based acoustic seed selection. The paper
  discloses it and retains the six-seed mean; original scores are descriptive.
- SLT-13: active manuscript now correctly states nine exclusions and one
  retained label correction. Baseline reproduced and six real-data tests pass.

See [revision audit](../../docs/revision-audit-20260924.md) for evidence,
commands, artifact paths, and outstanding submission requirements.

## 2026-09-24 — Bounded revision status (supersedes earlier interpretations)

SLT-01/02/03/05 and SA-01/04/05 are addressed within the locked empirical
scope. SLT-06 is context-only; SLT-07 includes the official-cohort CTD
sensitivity and scopes missing neural predictions; SLT-08 retains the
probe/seed disclosures. E6 is scoped out because raw encoder artifacts are
missing. MC-03/04/09 are not started. See the
[revision audit](../../docs/revision-audit-20260924.md) for reruns, exact number
provenance and submission blockers. Historical outcome prose is retained.
