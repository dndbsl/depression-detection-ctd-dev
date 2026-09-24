# SLT 2026 — Reviews (verbatim)

**Submission 435** · *"Can Conversational Temporal Dynamics Improve Depression
Detection in Dyads? A Preliminary Investigation in Multi-Modality Perspectives"*

**Decision: Reject** (Senior Area Chairs, 31 Aug 2026)

Authors: Sudarsana Reddy Kadiri, Hanie Kang, Huang-Cheng Chou, Shrikanth Narayanan

This file is the unedited record of what reviewers wrote. Do not rewrite or
"clean up" the text here — it is the source of truth we cite from
[`action-items.md`](action-items.md). Actionable items are tracked there.

---

## Meta Review — Area Chair W8mY (19 Aug 2026)

**Feedback To Authors:**

The reviewers appreciated the careful protocol and the treatment of
conversational timing as a standalone modality. The rebuttal addresses concerns
about interviewer-driven timing cues and the acoustic baseline, but the main
limitations remain the modest novelty, limited generalization, and inconclusive
test-set improvement.

---

## Rebuttal by Authors (16 Aug 2026)

We thank the reviewers. [d9s3] found the work technically sound and publishable
after minor revision. [rwao] called the protocol "above the norm" for DAIC-WOZ
and recognized the first-class CTD framing. [HoqK] found the framing reasonable
and the paper clear.

1. **Novelty [HoqK, rwao].** CTD definitions are not new; our contribution is
   transferring dyadic CTD to depression as a standalone modality and testing
   its complementarity with SSL encoders under one protocol.
2. **Evidence/generalization.** We acknowledge the dev–test drop, single split,
   and non-definitive test gain: CTD ties RoBERTa at .631 and paired CIs touch
   zero. Parameter-free T+CTD (.650) shows the trend is not solely dev tuning.
3. **Interpretability/confound [HoqK, rwao].** Both analyses are done and will
   be added. The largest coefficient is ask-side (`ask_d`), so we test it
   directly: dropping all nine ask-only features preserves performance
   (dev .752 / test .661 vs. full .746 / .631) — the signal is dyadic, not
   wizard-driven. The largest response-side effect is response latency `res_h`,
   direction-consistent on train/dev/test.
4. **Prior work [HoqK, d9s3].** We keep the official partition and only drop
   integrity cases; no subject is reassigned. Agarwal & Dias (.80) is not
   directly comparable: they encode therapist+participant text; we exclude
   interviewer text given prompt shortcuts. Their code is unreleased, so we
   cannot re-run it on our split; ours is open.
5. **Acoustic baseline.** We will scope the zero acoustic weight as
   probe-specific. Across our six WavLM seeds the acoustic detector spans
   dev .667±.053 / test .506±.034; we will report this seed mean rather than one
   seed, under which CTD is still the strongest single modality on dev. The
   CTD–semantic result has no acoustic term and is identical across all six
   seeds; a finer convex grid (.02) leaves the dev optimum at (0, .3, .7). The
   ~.80 figures in Sec. II-A are dev, not test; we will say "competitive with
   frozen encoders."
6. **Code.** Ready: https://shorturl.at/qBj92

---

## Official Review — Reviewer HoqK (29 Jul 2026)

> **Scores:** Novelty 2 (Limited) · Technical Quality 3 (Adequate) ·
> Reproducibility 3 (Moderate) · Ethics 5 (None) · **Rating 2 (Reject)** ·
> Confidence 3 (Fairly confident)

**One-line summary:** Honest but preliminary study of turn-timing for depression
screening, the evidence is thin, generalization is untested, the "interpretable"
features are never inspected, and the code repository is empty.

### 1. Summary and contributions

The paper asks whether the timing of the exchange between an interviewer and a
participant helps detect depression on the DAIC-WOZ dataset. The authors use a
set of 24 handcrafted temporal features called Conversational Temporal Dynamics
(CTD). They treat CTD as its own modality next to a frozen WavLM-large acoustic
model and a frozen RoBERTa-large text model, and join the three with score-level
convex-weighted late fusion. Main results show CTD is the best single modality
on dev (macro-F1 0.746), and fusion reaches 0.804 dev and 0.669 test macro-F1.

### 2. Novelty and significance

The framing is reasonable but the new part is small. The 24-feature set is
reused from Chou et al. [13, 14], so the feature design is not new. Prior work
already shows timing helps: Aldeneh et al. [9] use dyadic turn-taking features
for mood episodes, and Fushimi et al. [10] use session-level pause/response
intervals on DAIC-WOZ. The paper itself says this lines already suggest timing
is useful. So the contribution is mainly moving a known feature set to a new
task and studying how much weight it earns in fusion. That is a useful check,
but it is close to existing work. The interpretability point (24 readable
features vs 1024-d encoders) is a nice practical angle, but as noted below it is
claimed and never used.

### 3. Technical quality, experiments, and comparison with related work

Main problems:

- The evidence for the headline claim is thin. On test, CTD does not beat
  RoBERTa (both tie at 0.631, Section VII-A), so the "best single modality"
  claim hold only on dev. The fusion test gain is small (T+CTD: 0.631 -> 0.669,
  +0.038, Section VII-B), the paired CIs touch zero, and the test CIs are wide
  and overlapping (Section VII-D). So the test result is not a clear win, which
  the authors admit.
- Weak evidence of generalization. The dev-to-test drop is large for CTD alone
  (0.746 -> 0.631) and the full system (0.804 -> 0.669) (Table II). The fusion
  weights and threshold are grid-searched on a small dev set (n=33), so the dev
  number reflects fitting to that specific set, and the learned weights
  (0.0, 0.3, 0.7) are themselves a dev-fitted choice. There is no cross-corpus,
  cross-lingual, or subgroup validation, so nothing shows the features work
  beyond this one dev split, or that they would transfer to a real deployment
  rather then partly encoding dev-specific prompt behavior.
- The "interpretable" features are never inspected. The paper highlights CTD as
  interpretable ("fully interpretable timing module"), and the detector is a
  plain L2-regularized logistic regression on the 24-d vector (Section IV-C),
  which gives per-feature coefficients for free. Yet it never reports which
  features drive the prediction: no coefficients, no feature-group ablation, no
  per-feature correlation. This matters for the main claim: for instance, (a) if
  the useful features are response-side ones (`res_h`, `res_sd`, `res_d`), that
  supports a real psychomotor-retardation signal (Section II-B, [6], [7]);
  (b) if ask-side features (`ask_d`, `ask_ud`, `ask_st`) carry the weight, that
  points to the wizard-of-Oz prompt confound and a dev artifact. The
  interpretable aspect could settle this and was not used.
- No comparison with prior work, and the test set numbers look low. Tables II
  and III contain only the authors' own models, no prior method is re-run on
  this split. On dev, 0.804 is below the mentioned 0.81 of Li et al.
  (Section II-A). On test the gap is larger: the best system is 0.669
  (Table II), while non-recent published text-only systems report higher, e.g.
  Agarwal & Dias (CLPsych 2024,
  https://aclanthology.org/2024.clpsych-1.9.pdf) at 0.80 test, who also report
  Milintsevich et al. (2023) at 0.74. These use the official 107/35/47 split,
  not this paper's 102/33/45 split, so it is not exactly the same test set,
  however, a gap of about 0.13 below a text-only model that uses no timing is
  large and never mentioned. At least one such baseline should be re-run on this
  split. Better still, the authors should also report their own system on the
  official 107/35/47 split: the cleaning is justified, but reporting only the
  cleaned split gives up comparability with all prior work for no stated gain.
  Showing both would restore comparability and, if the cleaning matters, the gap
  between the two would itself support the leakage argument.
- The "CTD beats the encoders" and "zero acoustic weight" results may depend on
  a weak acoustic baseline. The authors call the baselines "not exhaustively
  optimized", and their WavLM test macro-F1 is 0.545 (Table II), below common
  WavLM numbers noted in Section II-A. So CTD's relative standing may be
  inflated and the zero acoustic weight may be specific to this probe.

### 4. Clarity, organization, presentation

The paper is clear and well organized. Fig. 1 gives a good overview, Fig. 2
explains one Ask/Res pair, and Table I lists the 24 features. The writing is
honest about limits (Section VIII). One small confusion: in Table II the T+CTD
and A+T+CTD rows show the same numbers (0.804/0.669) because the acoustic weight
is zero, this is explained but may confuse a fast reader. Reporting dev as the
headline while the test gain is not significant oversells the result, the
abstract gives 0.804/0.669 without the dev-tuning caveat.

### 5. Major concerns

The main problems above reduce to:

- No feature-level analysis despite the interpretability claim (a missed
  contribution and a gap in the evidence for the main claim).
- Weak evidence of generalization: large dev-to-test drop, dev-tuned weights and
  threshold on n=33, and no external validation.
- The test improvement is not clearly significant, so the claim rests mostly on
  dev.
- Lack of comparison with prior work.
- The CTD-vs-encoders and zero-acoustic-weight results may be driven by a weak
  acoustic baseline.
- Limited novelty: the feature set is borrowed and timing was already shown
  useful by prior work.

### 6. Minor comments and corrections

- Code link (footnote 1): the repository is empty, so unless the code is actually
  released the reproducibility claim should be revisited.
- Verify the session count: excluding 10 sessions leaves 180 (as mentioned in
  Section III), which implies 190 to start, but the standard DAIC-WOZ release has
  189 (107/35/47). This this mismatch needs clarification.
- The title phrase "in Multi-Modality Perspectives" reads awkwardly, consider
  better rewording.

---

## Official Review — Reviewer rwao (24 Jul 2026)

> **Scores:** Novelty 3 (Moderate) · Technical Quality 4 (Good) ·
> Reproducibility 4 (Good) · Ethics 4 (Minor issues / missing discussion) ·
> **Rating 3 (Borderline)** · Confidence 1 (Low confidence)

**One-line summary:** A compact turn-timing descriptor, borrowed from deception
detection, outperforms frozen WavLM/RoBERTa individually and drives acoustics to
zero weight in fusion. Protocol is careful, but the test-set gain is small and
not statistically decisive.

### Summary and contributions

The paper asks whether conversational temporal dynamics (CTD) — turn-level
timing between an interviewer and participant, operationalized as a
24-dimensional feature set over Ask/Res turn pairs (durations, differences,
ratios, voiced/silence ratios, response latency, backchannel/silence counts) -
can serve as a standalone modality for depression detection on DAIC-WOZ,
alongside frozen WavLM-large (acoustic) and RoBERTa-large (semantic) baselines.

### Novelty and significance

The core feature set itself is not new, and the individual components (fusion of
acoustics/text, timing as a depression cue) each have precedent in the cited
literature. The novelty is the specific framing: treating dyadic turn-pair
timing as a first-class modality on equal footing with 1024-d neural encoders
via score-level fusion, on DAIC-WOZ specifically, and showing that a dev-tuned
convex combination drives the acoustic weight to zero.

### Technical quality, correctness, completeness, experiments, comparison with related work

The experimental protocol is a genuine strength: subject-independent splits, a
documented and justified exclusion of ten sessions with cataloged data-integrity
issues, leakage-safe median imputation fit on train only, dev-only threshold and
weight tuning, train+dev refit for test-time deployment, 2,000-resample
bootstrap CIs, and paired bootstrap CIs for fusion-vs-single-modality deltas.
This level of rigor is above the norm for DAIC-WOZ work, and the paper is
commendably explicit about the resulting statistical caveats rather than
overselling them (Sec. VIII). The related-work coverage of acoustic/semantic
DAIC-WOZ baselines, speech-timing literature, and multimodal fusion is
reasonably thorough and situates the contribution against three specific prior
lines.

That said, several technical-quality concerns should be addressed:

- **Sample size and statistical power.** Dev n=33, test n=45, with ~30% positive
  prevalence. Bootstrap CIs are wide and overlapping (e.g., test macro-F1 CI
  [0.509, 0.806] for the proposed system vs. [0.472, 0.771] for CTD alone), and
  the paired CI for the fusion improvement over CTD/RoBERTa "touches zero." The
  paper is honest about this, but it means the central claim ("fusion improves
  over the best single modality") is not statistically established at
  conventional confidence — it is an observed effect on one small split, not a
  demonstrated effect.
- **Single fixed split.** All results come from one train/dev/test partition.
  Even though this follows DAIC-WOZ convention, the seed-variability numbers
  reported for the neural baselines (RoBERTa test F1 0.604±0.025, WavLM
  0.506±0.034) suggest a similar sweep over CTD or the fusion weights themselves
  would strengthen the claim; it is not clear whether such a sweep was performed
  for the CTD detector or the fusion system.
- **Acoustic baseline fairness.** The acoustic detector's weak performance is
  central to the "zero weight" finding, but the acoustic branch is a frozen,
  non-fine-tuned probe with a granularity choice (utterance-level pooling) that
  the authors themselves show is sensitive to design decisions (turn-level
  pooling collapses performance to test F1 0.333). Since the acoustic weight
  going to zero is one of the paper's headline results, it would help to know how
  sensitive this conclusion is to reasonable alternative acoustic configurations
  (e.g., fine-tuning, different pooling windows) rather than a single fixed probe
  design.
- **Grid granularity.** Convex weights are searched with step 0.1, so "exactly
  zero" acoustic weight is a coarse-grid outcome; a finer grid or a smoothness
  argument would make the "zero weight" framing more convincing (vs. e.g., a
  small nonzero optimum being rounded away).
- **Ask-side timing confound.** The paper appropriately flags that Ellie's
  ask-side timing is wizard-controlled and could reflect prompt selection rather
  than participant behavior, but does not report the response-side-only ablation
  that would isolate this - it is left as future work. Given that this confound
  could inflate CTD's apparent signal, it is a fairly important omission for the
  paper's central claim rather than a minor one.

### Clarity, organization, and presentation

The paper is clearly written and logically organized (Introduction → Related
Work → Resource → Baselines → Methodology → Setup → Results → Discussion →
Conclusion), with a helpful pipeline figure (Fig. 1) and turn-pair diagram
(Fig. 2). Table I's feature taxonomy and Tables II–III's fusion comparisons are
easy to follow. The abstract and introduction accurately represent what is shown
in the results (no overclaiming), and the discussion section is unusually candid
about limitations. Minor readability issue: the results narrative in Sec. VII-B/C
leans on many closely-spaced numbers (0.746→0.804, +0.058, +0.114, etc.) that
could be consolidated into a single delta table for faster reading.

### Major concerns

(a) The strength of the central claim is limited by the small, single-split
evaluation and overlapping/zero-touching confidence intervals — this is
explicitly acknowledged, which is good practice, but reviewers should weigh the
claims accordingly rather than treat 0.804/0.669 as robust point estimates.
(b) The ask-side timing confound (wizard-controlled prompts) is a plausible
alternative explanation for part of CTD's signal and is not empirically ruled
out. (c) Generalization is untested beyond DAIC-WOZ; no cross-corpus or
cross-lingual validation is presented (the authors correctly list this as future
work).

### Minor comments and corrections

- Table II should probably restate the deployed decision thresholds inline
  (currently only in text) for at-a-glance readability.
- Consider clarifying in the abstract that the reported metrics are "macro-F1 on
  the development set primarily, with test reported for reference," since a first
  read of the abstract could suggest both are equally primary results.
- The eGeMAPS-functional CTD variant (240-d) is mentioned as a robustness check
  that "did not beat" the compact model, but no numbers are given; even one line
  with the comparison score would strengthen the claim that the simpler model was
  not just a convenient choice.
- Ensure reference formatting consistency for arXiv preprints vs. venue papers.

---

## Official Review — Reviewer d9s3 (17 Jul 2026)

> **Scores:** Novelty 4 (Good) · Technical Quality 4 (Good) ·
> Reproducibility 5 (Excellent) · Ethics 5 (None) · **Rating 4 (Accept)** ·
> Confidence 5 (Very confident)

This paper presents a multimodal depression detection approach with the use of
conversational temporal dynamics. The topic of depression detection is highly
relevant and applicable in real life. The justification of the used methods
seems convincing enough. The pipeline is fair and explained well, the
experimental procedures also seem technically correct. Authors mentioned the
known pitfalls of the DAIC-WOZ and dealt with them accordingly, that shows a
good understanding of the data and is a thorough approach to the data
preparation. The analysis of the achieved results is detailed. However, the
paper has some issues:

- It's interesting how the proposed A+T+CTD system coincides with T+CTD,
  excluding audio modality (Tables 2 and 3) by giving it zero weight. Generally,
  the achieved results reflect a pattern observed in other multimodal works at
  least on the DAIC datasets: the text modality performs better compared to the
  acoustic one.
- There's a comparison with other works in the text, that can be added into
  Table 2, for example. Almost a half of the literature is older than 5 years,
  which is fine for some fundamental works, but it's lacking recent works.
- It would be nice to see results in cross-corpus settings or using any other
  relevant corpora, not only DAIC-WOZ.

Overall, the paper is rather good and interesting. It could be published after a
minor revision.
