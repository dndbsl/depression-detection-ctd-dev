# Active experiment plan: three days, existing data, CPU

Updated 2026-09-24 after the user's explicit choice of a few days of CPU-friendly work. This replaces the earlier two-week proposal as the active plan. It specifies future experiments; they have not been executed. Time estimates are planning estimates, to be checked with a runtime pilot.

**Research question:** Does CTD add predictive information beyond interviewer prompt selection, interview structure, and a simple participant-language baseline?

The intended contribution is a controlled evaluation of conversational timing. This scope can strengthen the existing evidence, but it cannot provide independent confirmation, establish clinical validity, or reproduce the unavailable neural fusion runs.

## 1. Deliverables and priorities

| Priority | Experiment | Main comparison | Deliverable |
|---|---|---|---|
| Required | A: stronger timing controls | Controls+CTD versus controls | One main comparison table |
| Required | B: CPU participant-text baseline | Text fused with controls+CTD versus text fused with controls | One nested fusion table |
| Next | C: targeted prompt removal | CTD improvement before/after targeted and control deletions | One sensitivity figure/table |
| Required | Audit and reporting | Identical cohorts, saved predictions, honest uncertainty | Reproducible artifacts and revised claim list |

Prioritize finishing A and B with valid comparisons. C is the first item to defer if preparing a reliable segment-removal implementation exceeds the time budget. Do not replace it with an unvalidated deletion script or add new feature searches in its place.

## 2. Shared evaluation protocol

- Use the existing 135 cleaned DAIC train+dev subjects; retain the existing labels and exclusions. The official test cohort is not used for new selection or scoring in this short plan.
- Store and reuse identical 5-outer/4-inner subject-grouped folds over 10 repeats (outer seeds 0–9). Start with one repeat to measure runtime and check the implementation. Do not select repeat seeds based on scores; if a smaller repeat count is necessary, decide from runtime before inspecting comparative results and use it for every configuration.
- Use median imputation, standardization, and class-balanced L2 logistic regression for dense timing/control features. Fit every learned transform on the current fitting fold only.
- Retain the existing timing C grid `{0.01, 0.03, 0.1, 0.3, 1.0}`, selected by mean inner balanced accuracy. Use decision threshold 0.5 for the primary timing, text, and fusion results. This preserves a direct connection to the current timing protocol and avoids giving fusion exclusive threshold tuning.
- Primary metric: macro-F1. Secondary: ROC-AUC, balanced accuracy, sensitivity, specificity, positive-class F1. Pool held-out predictions within each repeat, then report the mean and SD across repeats and paired model differences.
- If time permits, report a uniformly applied secondary threshold-tuning analysis over `{0.05, 0.10, ..., 0.95}`, using pooled inner-held-out predictions and maximizing macro-F1. Report fixed and tuned results separately; do not choose the more favorable protocol as the headline after seeing outcomes.
- Save subject/session IDs, labels, split membership, held-out probabilities and decisions, selected C/fusion weights/thresholds, source hashes, and package versions. Keep restricted artifacts local.
- Prespecify two primary contrasts: A's `B+D minus B`, and B's `F(T,B+D) minus F(T,B)`. Everything else is a secondary descriptive comparison. These are exploratory follow-ups after earlier results were inspected, not retrospective preregistration.

## 3. First half-day: make the existing evaluation usable

1. Extend `src/ctd/pdch_experiment.py::nested_cv_classification` or factor out a shared runner so it saves fold assignments and held-out predictions. It currently computes predictions internally but returns aggregates and selection counts.
2. Check that no subject crosses a split, every subject has one outer prediction per repeat, labels match across feature sources, and all preprocessing is fitted within the relevant training fold.
3. Reproduce the existing full-CTD and latency+structure configurations on the stored folds. Investigate mismatches before introducing new representations.
4. Snapshot existing artifacts and direct new results to a new local run directory. Keep the old neural metrics distinct from newly computed CPU results.

No model downloading or new audio extraction is needed.

## 4. Experiment A: stronger timing controls

Reuse `src/ctd/timing_controls.py` and the existing feature caches. Define:

- `S`: mean response latency, log session span, log(1+turn-pair count).
- `Q`: interviewer prompt-template frequency features. Normalize text with a fixed label-blind rule (lowercase, whitespace and punctuation normalization); learn the template vocabulary within each fitting fold, combine rare templates into an other category, and report coverage. A concrete starting rule is to retain templates present in at least three fitting subjects. Use relative frequencies plus an unknown/other fraction; do not select templates by depression association.
- `B = S+Q`: the stronger control baseline.
- `D`: the existing 24 CTD features.
- `R`: the existing eight response-internal features.
- `I`: the eight strictly interviewer-internal features obtained by removing `ask_bt` from the existing ask-side group. That overlap feature also uses participant timing.

Run nine configurations: `S`, `Q`, `B`, `D`, `R`, `I`, `B+R`, `B+no_ask`, and `B+D`. Avoid duplicate columns when concatenating, especially latency, which already appears in D and no_ask.

**Primary question:** Does `B+D` improve over `B`? Secondary comparisons locate the information within the feature bank. The no-ask representation still depends on interviewer timestamps and must not be described as interviewer-free.

Interpret improved prediction as incremental information conditional on these particular controls. A null contrast may reflect control sufficiency, redundancy, or insufficient precision; it does not prove that CTD has no information. Report interval width and effect magnitude.

## 5. Experiment B: CPU text and fusion

Reuse the existing cleaned transcripts and participant speaker filtering. This is a new lightweight language baseline, not a recreation of RoBERTa or WavLM.

**Text representation:** concatenate participant language by session after the existing marker cleaning. Exclude interviewer text and transcript metadata. Use TF-IDF word unigrams and bigrams, `min_df=3`, `max_features=3000`, sublinear term frequency, and L2 document normalization. Fit the vocabulary and IDF inside every fitting fold, including inner folds. Preserve sparse matrices and do not apply dense mean-centering to TF-IDF.

**Text estimator:** class-balanced logistic regression; select C from `{0.01, 0.1, 1, 10, 100}` by mean inner balanced accuracy. Fix feature settings before scoring. The different C range is prespecified for the differently scaled text representation; both modality grids have five candidates.

**Fusion:** `F(T,X) = w*p_T + (1-w)*p_X`, with `w` selected from `{0, 0.1, ..., 1}` by pooled inner-held-out balanced accuracy at threshold 0.5. Break ties by the weight closest to 0.5 and then the smaller weight. Use inner-held-out probabilities from the selected component configurations, never their in-sample fitting probabilities. Refit components on outer training data and apply the selected rule only to outer-held-out subjects. All selection remains inside the outer loop.

Run `T`, `F(T,B)`, `F(T,B+D)`, and `F(T,B+R)`. Reuse the matching timing/control scores from A where the fold identities and fitting recipes agree; retain inner predictions as well as outer predictions to support correct fusion selection.

**Primary contrast:** `F(T,B+D) minus F(T,B)`. Compare both with T alone as a secondary check. An advantage supports complementarity to a lexical baseline; it does not establish superiority over strong pretrained language models. A weak TF-IDF score is not a reason to search many representations until timing appears beneficial.

## 6. Experiment C: prompt dependence, if time permits

Freeze a label-blind list/rule for mental-health-history prompts using interviewer text and the existing prompt-bias literature. Define the removed unit as the targeted question and its corresponding participant response. Inspect matching quality without looking at labels or model-score changes.

Compare three conditions using B and B+D:

1. Original interviews.
2. Targeted segments removed.
3. Other eligible question/response segments removed with approximately matched duration per session, using three fixed seeds. Report all three deletion results; do not select the most favorable seed.

Recompute features on surviving original interview blocks. Preserve original timestamps, discard pair contributions whose required context was removed, and never join formerly nonadjacent turns to invent a latency or overlap. Recompute retained prompt counts and turn counts. Record original session span and total retained block duration explicitly; use the same duration-control definitions for all three conditions.

Require at least five usable original turn pairs in every compared condition. Determine the common eligible subject set without using outcomes, report exclusions and retained speech/turn coverage, and rerun the original condition on that same cohort. Apply removal consistently to fitting and held-out data.

Compare the CTD improvement, not just each system's absolute score. A larger reduction after targeted deletion than control deletion suggests prompt dependence, but deletion also changes content and cannot establish a causal mechanism.

## 7. Uncertainty within the CPU budget

Report paired mean differences and split variability for all planned comparisons. For the two primary contrasts, add 2,000 paired subject-bootstrap resamples of the saved predictions: sample subjects, retain all repeat-specific predictions for each sampled subject, compute metrics within each repeat, and then average the repeat-wise differences. Do not pool the ten predictions per subject into 1,350 independent observations or average probabilities into an unplanned ensemble.

Label these as **conditional descriptive intervals**. They do not include refitting/selection uncertainty or fully account for cross-subject dependence induced by shared training sets. Do not convert them into definitive significance claims or use ordinary t-tests over repeats. Full-pipeline bootstrap and independent-cohort inference are deferred under this time budget.

## 8. PDCH and manuscript work

Keep existing PDCH results as a limited within-corpus finding. Clarify the inpatient severity target, the 62-session/46-subject mapping, missing labels, and the absence of validated timing boundaries. These points can be improved without collecting new annotations.

If PDCH regression is re-reported, first correct its reference baseline: `baseline_mae_predict_train_mean` currently uses the full-cohort mean. Use outer-training-fold mean/median predictions. Do not describe this reference-baseline issue as evidence that the fitted nested CTD classifier is invalid.

Update the README's stale selection/provenance claims to match the current paper. Present newly computed CPU text results in their own clearly labeled table; unavailable neural runs cannot acquire fresh uncertainty estimates from aggregate metrics alone.

## 9. Three-day schedule and stopping rules

| Day | Work | Required outcome |
|---|---|---|
| 1 | Prediction export, reproduction check, Experiment A | Timing/control comparison and saved predictions |
| 2 | Experiment B with fold-local TF-IDF and nested fusion | Reproducible lexical-baseline comparison |
| 3 | Experiment C if ready; paired summaries and paper updates | Sensitivity analysis or a clearly stated omission; finalized tables and claim list |

Deferred: GPU/neural reruns, manual boundary annotation, new corpora, architecture sweeps, speculative feature expansion, and large refitting bootstraps. No new data, downloads, paid services, or recruitment are required by the active plan.

If A/B show robust descriptive gains, emphasize incremental utility under the tested controls and lightweight text representation. If controls absorb the advantage, emphasize what the study learned about interviewer dependence, with uncertainty. If differences remain imprecise, report that explicitly. Do not seek a positive narrative by changing folds, cutoffs, or feature banks after seeing the results.

## References and deferred roadmap

Prompt-control motivation: [Burdisso et al., ClinicalNLP 2024](https://aclanthology.org/2024.clinicalnlp-1.8/). PDCH cohort interpretation: [Cao et al., Scientific Data 2025](https://www.nature.com/articles/s41597-025-05817-9). CV-dependence background: [Bengio and Grandvalet, JMLR 2004](https://jmlr.csail.mit.edu/papers/v5/grandvalet04a.html).

The earlier [expanded proposal](experiment-plan-icassp2027.md) is retained only as a future roadmap; its ten-day schedule and GPU/annotation requirements are not part of this active plan.
