# Preflight plan — `SLT-04`, `SLT-05`, `SLT-10`, `SLT-13`

Everything that must land on DAIC-WOZ **before** any new corpus. Tracked in
[`reviews/slt2026/action-items.md`](../reviews/slt2026/action-items.md);
corpus work in [`multi-corpus-plan.md`](multi-corpus-plan.md).

**Why these four, in this order.** `SLT-04`/`SLT-05` define the feature groups
that become the multi-corpus tiers (`MC-02`), so running a new corpus first
would make the tier boundaries look post-hoc. `SLT-10` builds the resampling
harness every later experiment reuses. `SLT-13` is nearly free and touches the
same file.

**Scope note.** All four are CPU-only and need only transcripts + labels —
no GPU, no WavLM, no RoBERTa. They can run start to finish on this machine.

---

## Step 0 — Environment and a reproduction gate (`MC-10`)

No Python packages are installed (no numpy, pandas, sklearn, scipy, or conda).

```bash
python3 -m venv ~/.venvs/ctd && source ~/.venvs/ctd/bin/activate
pip install -r src/ctd/requirements.txt confidence_intervals
export DAIC_WOZ_ROOT=~/data/DAIC
```

`torch` is in `requirements.txt` but no CTD-path module imports it; install it
only if it comes for free.

**Gate — do not start the analysis until this passes.** Run
`python src/ctd/ml_splits.py` and confirm it reproduces the published CTD
numbers (dev macro-F1 **0.746** / test **0.631**, selected `C=0.3`) and the
cohort counts **102 / 33 / 45**. Every number below is a delta against this
baseline; if the baseline doesn't reproduce, fix that first and treat it as its
own finding.

### ⚠ 0a. The CTD detector is implemented twice — reconcile before ablating

The deployed 24-D detector exists in two places that must agree:

- `src/ctd/ml_splits.py::run()` — features via `functionals.py` `__amean`
  columns, pipeline `SimpleImputer(median) → StandardScaler → LogReg`.
- `src/fusion/fusion_late.py::ctd_probs()` (lines 117–133) — features via
  `np.nanmean` over `bags.py`, with train-median imputation applied in
  `fusion/extract_ctd_roberta.py`.

Both are NaN-aware means so they *should* be identical, but nothing asserts it.
Add that assertion first (`tests/test_ctd_consistency.py`): identical
session-mean matrices and identical probabilities per split. Otherwise an
ablation defined in one path silently disagrees with fusion.

### ⚠ 0b. Selection metric differs from the reported metric

`run()` selects hyperparameters on **dev balanced accuracy**, while the paper
headlines **macro-F1**. That is defensible but currently undocumented, and it is
exactly the kind of protocol detail HoqK audits. Do **not** change it — that
would change the deployed model. Document it, and report both metrics in the
new tables.

---

## Step 1 — `SLT-13`: session-count wording (do this first, it's ~30 min)

The code is correct; two documents are not.

**Confirmed root cause.** `common/daic_cleaning.py::KNOWN_ERRORS` holds 10
data-integrity cases, of which **9** have `action: "exclude"` (318, 321, 341,
362, 373, 444, 451, 458, 480). The 10th, **409**, is `action: "relabel"`
(PHQ-8 = 10 but binary label 0 in the official file) and is *corrected and
kept*. So 189 − 9 = 180 = 102/33/45. ✓

**Repo bug found while checking:** `src/acoustic-depr-wavlm/README.md` line 21
reads `Excluded sessions: {318, 321, 341, 362, 451, 458, 480, 373, 444, 409}` —
it lists 409 as excluded. All pipelines route through
`daic_cleaning.apply_cleaning`, so the behaviour is right and only the prose is
wrong, but this is the same error the reviewer caught in the paper. Fix it here
too.

Deliverables:

1. Correct `src/acoustic-depr-wavlm/README.md` to separate "9 excluded" from
   "1 relabeled (409)".
2. Add `tests/test_daic_cleaning.py` asserting `len(excluded_ids()) == 9`, the
   exact excluded set, `409 in corrected_labels()`, and (skipped when
   `DAIC_WOZ_ROOT` is unset) per-split kept counts of 102/33/45 from 189. This
   makes the count un-driftable rather than merely fixed once.
3. Record the manuscript wording: *"ten sessions with documented integrity
   issues: nine excluded and one relabeled,"* stating 189 as the starting count.

---

## Step 2 — Shared foundation: `src/ctd/feature_groups.py`

Both `SLT-04` and `SLT-05` need one authoritative grouping, and it later becomes
the multi-corpus tier selector (`MC-02`). Build it once.

| Group | n | Features |
|---|---|---|
| `ASK_ONLY` | 9 | `ask_d`, `ask_ud`, `ask_du`, `ask_sd`, `ask_ds`, `ask_su`, `ask_us`, `ask_st`, `ask_bt` |
| `CROSS` | 7 | `res_minus_ask`, `ask_minus_res`, `duration_sum`, `res_over_ask`, `ask_over_res`, `res_h`, `res_bt` |
| `RES_ONLY` | 8 | `res_d`, `res_ud`, `res_du`, `res_sd`, `res_ds`, `res_su`, `res_us`, `res_st` |

Constraints:

- **Never mutate `CTD_FEATURE_NAMES`.** `constants.py` asserts `== 24`, and
  `feature_extraction.py` asserts its output keys match exactly. Subsetting must
  happen by *column selection* downstream.
- Assert at import that the three groups partition the 24 names exactly.
- Named configs: `all24` (24), `no_ask` (= CROSS + RES_ONLY, 15), `res_only`
  (8), `ask_only` (9).
- Docstring the one nuance: `ask_bt` is grouped ask-side because it measures
  *interviewer* behaviour (interrupting the participant), but it is computed
  using the response end time, so it is not derivable from ask timestamps
  alone. This matters for the corpus tiers later.

---

## Step 3 — `SLT-04`: interpretability (`src/ctd/interpret.py`)

HoqK's point is that a plain L2 LogReg yields coefficients for free and we never
reported them. The rebuttal claims the largest coefficient is `ask_d` and that
`res_h` is direction-consistent across train/dev/test. Make that reproducible.

### ⚠ 3a. The blocker nobody has flagged: the feature set is built from reciprocal pairs

`ask_ud` = u/d and `ask_du` = d/u are reciprocals of each other — and so are
`ask_sd`/`ask_ds`, `ask_su`/`ask_us`, `res_ud`/`res_du`, `res_sd`/`res_ds`,
`res_su`/`res_us`, plus `res_over_ask`/`ask_over_res` and the sign-flipped pair
`res_minus_ask`/`ask_minus_res`. That is **8 near-collinear pairs by
construction**.

L2 regularization splits weight arbitrarily between collinear partners, so a
single "largest coefficient is `ask_d`" claim is fragile — a reviewer can
reasonably argue the ranking would reshuffle under a different seed or
regularization strength. Publishing a naive coefficient table here would invite
a *new* criticism while answering the old one.

**Therefore report three triangulating views, not one table:**

1. **Standardized coefficients with bootstrap stability.** `StandardScaler`
   precedes the classifier, so coefficients are already on a comparable scale.
   Resample train (B = 2000), refit, and report each coefficient's 95% CI and
   **sign-consistency rate**. A feature only gets called "driving the
   prediction" if its sign is stable. Also report odds ratio per 1 SD for
   clinical readability.
2. **Permutation importance on dev.** Robust to collinearity *within the fitted
   model* in a way single coefficients are not, and it measures what the
   deployed detector actually uses.
3. **Univariate direction consistency.** Per-feature point-biserial correlation
   and single-feature AUC, computed **separately on train / dev / test**. This
   is what substantiates the `res_h` claim, and it is immune to the collinearity
   problem entirely.

Also emit a collinearity diagnostic (correlation matrix and/or VIF) and
report pair-level aggregate importance (summed |coef| within each reciprocal
pair) so the reader sees the grouping rather than being misled by a split.

Outputs: `output/ctd_interpretability.json`, a paper-ready markdown table, and
a coefficient forest plot for the figure.

---

## Step 4 — `SLT-05`: ask-side ablation (`src/ctd/ablation_groups.py`)

rwao called the missing response-side-only ablation "a fairly important omission
for the paper's central claim." Run it properly.

Configurations, each through the *identical* deployed protocol (fit train →
select on dev → refit train+dev → test once, with bootstrap CIs):

| Config | n features | Purpose |
|---|---|---|
| `all24` | 24 | published baseline |
| `no_ask` | 15 | **the headline ablation** — is the signal wizard-driven? |
| `res_only` | 8 | participant-only, fully confound-free |
| `ask_only` | 9 | how much does interviewer timing alone carry? |
| `no_cross` | 17 | leave-one-group-out completeness |

### ⚠ 4a. Do not claim the ablation "improves" performance

The rebuttal reports `no_ask` at dev .752 / test .661 against `all24` at
.746 / .631 — I previously read that as the ask features adding noise. With
n = 33 dev and n = 45 test those deltas (+.006, +.030) are very unlikely to be
distinguishable from zero, and the paper was already criticized for resting on a
+.038 test delta whose CI touches zero.

Compute a **paired bootstrap** on the `all24` − `no_ask` difference and let it
decide the wording. The claim that actually supports the no-confound conclusion
is **"performance is preserved"** — which is both weaker and sufficient. Reserve
"the ask features add noise" for the case where the paired CI excludes zero.

### ⚠ 4b. Multiplicity on test

Five configurations × one test evaluation each is five looks at the test set.
Keep **dev as primary** for all ablations, report test for reference only, and
state the multiplicity explicitly. Nothing may be re-tuned on test. This is the
discipline HoqK was asking for, so make it visible rather than implicit.

Outputs: `output/ctd_ablation_groups.json` + markdown table.

---

## Step 5 — `SLT-10`: variability (`src/ctd/resampling.py`)

### ⚠ 5a. A seed sweep is the wrong instrument here

rwao asked for "a similar sweep over CTD or the fusion weights" by analogy with
the neural baselines' seed variance. But the CTD detector is
`SimpleImputer → StandardScaler → LogisticRegression(lbfgs)` — a deterministic
fit on a convex objective. **Seed variance is exactly zero**, and reporting
`± 0.000` would look evasive.

The honest analogue is **data** resampling, so answer the question that was
meant rather than the one that was asked, and say so in one sentence.

Protocol:

- **Repeated stratified resampling over train+dev only** (test untouched),
  R = 100 repeats: refit, re-select `C` on the held-out portion, record
  metrics. Report mean ± sd.
- **Hyperparameter stability:** the distribution of the selected `C` across
  repeats. If `C` flips around, the deployed "C = 0.3" is arbitrary and should
  be reported as such.
- **Head-to-head rate:** fraction of repeats in which the 24-D CTD detector
  beats RoBERTa's 0.631 — a direct, honest answer to "is CTD really the best
  single modality," which currently rests on one dev split.
- Write the splitter **group-aware** from the start (`GroupKFold` on subject).
  DAIC-WOZ is one session per subject so it is a no-op here, but PDCH has 27
  subjects with two sessions each (`MC-08`) and will reuse this code.

Output: `output/ctd_resampling.json`. Feeds `SLT-02` directly.

---

## Step 6 — Write-up and wiring

- New `RESULTS.md` section for interpretability, ablation, and variability
  tables.
- Update `src/ctd/README.md` with the three new entry points.
- Update statuses in `action-items.md`; cite IDs in commit messages
  (`SLT-04: ...`).

---

## Ordering and dependencies

```
Step 0  env + reproduction gate + consistency assertion
   ├─ Step 1  SLT-13            (independent, do first — quick win)
   └─ Step 2  feature_groups.py (blocks 3 and 4)
         ├─ Step 3  SLT-04  interpret.py
         └─ Step 4  SLT-05  ablation_groups.py
      Step 5  SLT-10  resampling.py   (independent of 3/4; shares protocol helper)
      Step 6  write-up
```

Steps 3 and 4 can proceed in parallel once Step 2 lands. Step 5 only needs
Step 0.

## Risk register

| Risk | Mitigation |
|---|---|
| `ml_splits.py` doesn't reproduce .746/.631 | Step 0 gate — stop and fix before any analysis; treat as its own finding |
| Two CTD implementations diverge | Assertion test in Step 0a before ablating |
| Collinear reciprocal pairs make coefficients unstable | Three triangulating views + pair-level aggregation (Step 3a) |
| Overclaiming the ablation delta | Paired bootstrap decides the wording (Step 4a) |
| Test-set multiplicity across 5 ablations | Dev primary, test for reference, multiplicity stated (Step 4b) |
| "Seed sweep" reported as ±0.000 | Reframe as data resampling and explain (Step 5a) |
