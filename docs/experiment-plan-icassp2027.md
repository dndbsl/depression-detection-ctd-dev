# Experiment plan for the ICASSP 2027 revision

**Superseded scope:** The user selected a few days of work using existing data and CPU-friendly experiments. Follow [the active three-day CPU plan](experiment-plan-cpu-icassp2027.md). The expanded proposal below is retained as a deferred roadmap, not the current execution plan.

Prepared 2026-09-24. Proposed follow-up analyses; no experiments in this document have been run. Planning assumption: roughly ten working days, with access to a timing annotator and a GPU for one frozen text-embedding extraction. Runtime estimates require a pilot. Earlier DAIC and PDCH results have already been inspected, so this is an exploratory follow-up specification, not retrospective preregistration or independent confirmation.

The central question is: **Does conversational timing add predictive information after accounting for interview structure and prompt content, and how much does its apparent utility depend on timing measurement?**

## 1. What is already available

| Resource | Observed state | Consequence |
|---|---|---|
| DAIC timing features and nested controls | Existing code and aggregate results | Extend the evaluator; do not rebuild the entire pipeline |
| Nested evaluator | Computes out-of-fold predictions but returns aggregate metrics and selection counts | Save fold assignments and per-subject predictions before new comparisons |
| DAIC/PDCH data roots | Local data directories are accessible | CPU analyses and annotation preparation are feasible |
| RoBERTa artifacts | Prior audit reports missing embeddings/checkpoints/predictions; preprocessing and training code exist | Rebuild frozen embeddings; do not reconstruct predictions from aggregate scores |
| GPU | `nvidia-smi` cannot communicate with the driver in this environment | Working GPU access is unverified; allow a separate extraction environment or a measured CPU fallback |
| PDCH | Label loader uses 62 scored sessions and groups them into 46 subjects | Verify mapping provenance; preserve grouping in every split and resample |
| Repository documentation | README still describes test-independent selection and parameter-free fusion, conflicting with the current paper's disclosures | Update provenance documentation as part of the final revision |

Relevant implementation files: `src/ctd/timing_controls.py`, `src/ctd/pdch_experiment.py`, `src/ctd/feature_groups.py`, `src/ctd/pdch_adapter.py`, `src/ctd/pdch_labels.py`, `src/semantic-depr-roberta/`, and `src/fusion/`.

## 2. Freeze the evaluation before adding experiments

- DAIC development cohort: the existing cleaned 135 train+dev subjects. Preserve existing exclusions and the documented label correction. Keep the official test cohort out of new selection and prompt-dictionary construction. Its historical results remain exploratory because it has already been inspected.
- Use identical stored 5-outer/4-inner subject-grouped folds, with outer repeat seeds 0–9, across compared configurations. Each subject receives one outer-held-out prediction per repeat. PDCH uses the same grouping discipline; both sessions of one subject stay together.
- CTD/control estimator: training-fold median imputation, standardization, class-balanced L2 logistic regression. Retain the existing C grid `{0.01, 0.03, 0.1, 0.3, 1.0}`. Fit prompt vocabularies and all learned transforms inside training folds.
- Primary metric: macro-F1. Secondary: ROC-AUC, balanced accuracy, sensitivity, specificity, and positive-class F1. Summarize pooled out-of-fold performance within each repeat; show paired configuration differences and repeat variability.
- For new primary comparisons, select C by mean inner balanced accuracy at probability threshold 0.5, then select the decision threshold on the selected model's pooled inner-held-out probabilities, using `{0.05, 0.10, ..., 0.95}` to maximize macro-F1. Apply this opportunity to every compared system. Break C ties by smaller C, threshold ties by closeness to 0.5, then smaller threshold. Include a fixed-0.5 sensitivity analysis to connect with the existing paper.
- For fusion, select component hyperparameters and fusion parameters only inside the outer training set. Use inner-held-out component probabilities to select weights and threshold; never use component predictions on their own fitting rows. The inner scores are selection scores, not reported performance estimates.
- Save predictions, subject/session IDs, labels, outer/inner memberships, selected hyperparameters, preprocessing state, source hashes, versions, and elapsed time. Keep restricted derived artifacts local; publish code and permitted aggregates.
- Register two primary predictive contrasts: E1's `B+D minus B`, and E3's `F(T,B+D) minus F(T,B)`. Treat remaining ablations as secondary. Do not expand the feature search after inspecting their scores.

## 3. E0 — Provenance and evaluation audit (first half-day)

**Purpose:** prevent an apparently stronger result from depending on inconsistent cohorts, mislabeled units, or unsaved predictions.

1. Save a cohort manifest and exclusions for both corpora, including subjects, sessions, positive counts, and reasons for missing labels. Confirm PDCH's A/B identifier interpretation from the release documentation or other evidence; conservative grouping alone does not establish a longitudinal design.
2. Retain the provided HAMD total as the primary label source. Audit missing totals and item sentinel codes; do not silently sum items, impute missing targets, or change the cohort to obtain a better result.
3. Extend the nested evaluator to emit the artifacts listed above. Add narrow checks for subject overlap, fold coverage, label alignment across modalities, and fold-local preprocessing.
4. Run one small reproduction of existing timing results before introducing new configurations.
5. Correct the PDCH regression reference when it is reused: the current `baseline_mae_predict_train_mean` is computed from the full-cohort mean. Replace it with outer-training-fold predictions; include a training-fold median baseline for MAE. This affects the reference baseline, not automatically the fitted CTD regressor.

**Deliverables:** manifests, stored split files, prediction export, and a short audit report.

## 4. E1 — Incremental timing information beyond stronger controls (CPU, days 1–3)

**Hypothesis:** CTD improves held-out prediction after conditioning on interview length, latency, and prompt selection.

Define:

- `S`: existing simple controls — mean response latency, log session span, log(1+turn-pair count).
- `Q`: normalized interviewer prompt-template counts, with a frozen label-blind text-normalization rule. Learn the template vocabulary from each fitting fold; map unseen templates to an unknown category. Do not select templates by association with depression. Report vocabulary size and coverage.
- `B = S+Q`: the stronger control baseline.
- `D`: the original 24 CTD coordinates.
- `R`: the existing eight response-internal coordinates.
- `I`: a strictly interviewer-internal subset. Exclude `ask_bt`, which also uses participant timing; the existing nine-feature ask-side group is not strictly interviewer-only.

Run `S`, `Q`, `B`, `R`, `I`, `D`, `B+R`, `B+no_ask`, and `B+D`. Remove exact duplicate columns when concatenating, particularly response latency already contained in D. Apply the same training and threshold recipe to all rows.

**Primary contrast:** `B+D minus B`. **Secondary contrasts:** `B+D minus B+no_ask`, and `B+R minus B`. These distinguish incremental utility from the existing comparison of two separate predictors. No-ask remains an interviewer-dependent representation because it retains cross-speaker features.

Add one prespecified compact timing baseline or a faithful reimplementation of a prior timing feature set if it fits the time budget. Keep its model and folds matched. A fixed nonredundant subset of D is a useful sensitivity check: derive it from feature identities before scoring, not label-driven feature selection. Do not assume the reported rank of 13 automatically specifies the right subset.

**Deliverable:** the main controlled-comparison table, paired differences, selected-C/threshold distributions, and exact feature definitions.

**Interpretation:** a persistent advantage supports incremental utility conditional on these controls. An advantage that disappears supports a narrower interview-structure explanation. Neither result establishes a causal depression mechanism.

## 5. E2 — Prompt-content sensitivity (CPU, days 3–4)

**Hypothesis:** the CTD advantage is not confined to targeted mental-health-history portions of the interview.

Freeze a label-blind rule identifying these prompt segments, motivated by the published DAIC prompt-bias analysis. Review the rule without looking at prediction changes. Remove the identified question/response segments and recompute all timing and structure features. Preserve original timestamps and segment discontinuities: never create a new response latency across a deleted block.

Repeat the E1 primary comparison on the retained material. Include a duration-matched deletion of other eligible segments as a negative control, with fixed random seeds and no selection of the most favorable deletion. Apply the same removal policy in fitting and held-out data. Specify a minimum usable-pair rule in advance, report coverage, and compare all conditions on their common eligible subjects.

**Deliverable:** original vs targeted-removal vs control-removal performance differences, with retained session/turn coverage. Removal changes content and sample composition, so treat this as a sensitivity experiment, not proof of causation.

## 6. E3 — Rebuild the text baseline and test incremental fusion (days 4–7)

**Hypothesis:** timing contributes beyond participant language and the stronger interview controls.

1. Rebuild raw participant-only frozen RoBERTa-large turn embeddings using the existing preprocessing. Cache the unstandardized embeddings once, with encoder revision and transcript hashes. Do not fit normalization on the full cohort. Start with a few sessions to measure extraction cost and validate speaker filtering.
2. Use fixed mean pooling and a deterministic regularized logistic head for the controlled experiment. This is a new reproducible text probe; do not label it a reproduction of unavailable neural checkpoints. Prespecify its five-value C grid `{0.0001, 0.001, 0.01, 0.1, 1.0}` and use the shared inner selection recipe.
3. Let `F(T,X)` be convex fusion of the text probability and an X-based logistic model. Select the text weight from `{0,0.1,...,1}` and the shared threshold grid on inner-held-out probabilities only. Refit components on the outer training fold before applying the chosen fusion rule to outer-held-out subjects. Report all weight choices; do not interpret them as modality importance.
4. Compare `T`, `F(T,B)`, `F(T,B+D)`, `F(T,B+no_ask)`, and `F(T,B+R)`. Add `F(T,D)` to connect to the original text–CTD result. Give text alone the same threshold tuning. Report equal-weight fusion with the same threshold policy as a secondary baseline.

**Primary contrast:** `F(T,B+D) minus F(T,B)`. Also show the difference from T alone.

**Deliverable:** fully nested fusion table with per-subject predictions. A three-point macro-F1 gain may be a useful prespecified practical target, but it is not a clinical threshold or an acceptance guarantee; report estimates and uncertainty even when gains are smaller or negative.

**Fallback:** if extraction cannot be scheduled, use fold-fitted participant TF-IDF plus logistic regression for a CPU text control, identify its limited scope, and reduce the paper's multimodal claims. Do not treat that fallback as evidence that timing beats strong language representations.

## 7. E4 — Validate timing measurement and make PDCH interpretable (days 2–8)

**Question:** can the extraction procedure measure the quantities whose cross-corpus behavior is being interpreted?

**Manual audit.** Select approximately 12 distinct subjects per corpus, stratifying on duration and automated timing-quality indicators. Sample about 20 Ask/Res transitions per selected session, with enough surrounding audio to annotate within-turn speech/silence and overlap. Keep annotators blind to depression scores and model predictions. Double-annotate at least 25% and adjudicate discrepancies. Separate a development portion from an untouched annotation-evaluation portion if pipeline adjustments are made.

Report speaker-attribution agreement; onset, offset, and response-latency median absolute error and 90th-percentile error; missed speech; overlap detection; and agreement on derived timing descriptors. Treat the session/subject as the uncertainty unit. Approximately 240 transitions per corpus is an engineering audit, not 240 independent patients or a new prediction-validation cohort. Determine measurement tolerances before scoring, relative to the 0.2-second gap rule and observed annotation agreement.

**Matched extraction experiment.** On DAIC, compare native timing, DAIC processed through the same mono-VAD/speaker-assignment rules as PDCH, and a version using one-second quantized turn labels plus that same refinement. Use the same available 22-coordinate bank in matched corpus comparisons. Retain an explicit 24-vs-22 DAIC ablation. Quantization alone does not reproduce all PDCH annotation errors, so name these manipulations accurately. If manual errors support it, add one prespecified perturbation distribution estimated from the annotation development portion.

**PDCH re-evaluation.** After fixing extraction based on annotation quality, rerun within-PDCH grouped prediction with the existing binary endpoint and continuous HAMD severity regression. Compare regression against outer-training mean/median references. Use only the existing available labels; analyze missing-label coverage separately. This is severity stratification in an inpatient cohort, not screening against healthy controls or direct DAIC-to-PDCH transfer.

**Deliverables:** a measurement-validation table, DAIC extraction-sensitivity plot, and a compact PDCH table. If measurement remains poor, narrow the negative conclusion to this extraction pipeline. If measurement is adequate and prediction remains weak, report the uncertainty around that limitation.

## 8. Uncertainty and independent evidence

Do not use a t-test over the ten repeats or convert repeat SD into a standard error using sqrt(10). Training sets overlap and repeated predictions do not create additional subjects.

For the two primary contrasts, budget a paired subject-bootstrap that reruns the complete nested fitting/selection procedure, using identical sampled subjects and folds for the two compared methods. All copies/sessions of a sampled subject must remain grouped. Start with a timing-only runtime pilot of 20 resamples; freeze a feasible resample count (target 1,000) before inspecting interval endpoints. Record redraws needed for infeasible class/group splits. Treat these as approximate exploratory intervals; resampling cannot remove previous researcher adaptation or guarantee small-sample coverage.

If complete refitting is unaffordable, report paired subject-resampling intervals conditional on the saved predictions, keeping all repeats for a subject together and recomputing the mean of repeat-wise differences. Explicitly state that these omit training/selection uncertainty and dependence induced by shared fitted training sets. Do not use them to claim definitive significance. Apply multiplicity control if formal tests are reported for both primary contrasts; label other comparisons exploratory.

**Highest-value optional extension:** independent participants with compatible timing and screening labels. Existing repository notes identify a potentially disjoint E-DAIC subset, but those notes are not a current audit: recheck subject overlap, label availability, prior exposure, speaker attribution, and annotation quality. Freeze models and analysis before inspecting the target labels. A freshly opened old split does not become independent if its participants or results were already used. Report DAIC-to-new-cohort testing separately from within-new-cohort retraining.

## 9. Schedule, stopping rules, and paper deliverables

| Working days | Work | Dependency |
|---|---|---|
| 1 | E0, save folds/predictions, freeze comparisons | Existing timing data/code |
| 2–3 | E1; begin manual timing annotation | CPU; annotator availability |
| 4 | E2; text-extraction pilot | Prompt rule; working encoder environment |
| 5–7 | E3 and primary paired uncertainty analyses | Frozen text cache; measured compute budget |
| 7–8 | E4 extraction comparisons and PDCH refresh | Completed annotation audit |
| 9–10 | Check artifacts, rewrite claims, produce tables and figures | All prespecified results, including negative ones |

If only a few days are available, complete E0/E1/E2 and a CPU text control, and write a narrower timing-evaluation paper. If a month and suitable data are available, prioritize independent validation after E1/E3 and annotation validation. Repairing WavLM or adding one compact established acoustic baseline is secondary to answering the central questions; large architecture sweeps are outside this plan.

Retain all planned results. Do not keep changing representations until a comparison becomes significant. Use outcomes to determine the paper's emphasis:

- E1 and E3 gains survive controls: incremental timing utility, with uncertainty and corpus limits stated.
- E1's gain disappears after prompt controls/removal: interviewer-dependent benchmark behavior, supported by controlled evidence.
- E4 shows strong sensitivity to extraction: measurement validity becomes a principal result; broad PDCH boundary claims are reduced.
- Results remain imprecise: publish a limited exploratory conclusion; additional CV repeats do not replace new participants.

Target main-paper artifacts: one diagram showing extraction and nested selection; one controlled timing table; one nested text-fusion table if completed; one timing-quality/sensitivity figure; and a short PDCH severity table. Put exhaustive feature definitions, manifests, secondary ablations, and implementation provenance in the permitted reproducibility materials. Keep essential evidence in the main paper.

## Sources motivating the plan

- Interviewer prompt controls: [Burdisso et al., ClinicalNLP 2024](https://aclanthology.org/2024.clinicalnlp-1.8/).
- PDCH population and recording structure: [Cao et al., Scientific Data 2025](https://www.nature.com/articles/s41597-025-05817-9).
- Timing-feature comparison: [Fushimi et al., Technologies 2026](https://www.mdpi.com/2227-7080/14/4/198), and [SpeechT-RAG](https://arxiv.org/abs/2502.10950).
- Cross-validation uncertainty: [Bengio and Grandvalet, JMLR 2004](https://jmlr.csail.mit.edu/papers/v5/grandvalet04a.html). Dependence makes naive fold/repeat-based variance estimates unsuitable; the practical bootstrap above is an approximate analysis choice, not an exact remedy supplied by that theorem.
