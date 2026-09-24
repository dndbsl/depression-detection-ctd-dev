# Multi-corpus extension plan (ICASSP 2027)

Addresses [`SLT-01`](../reviews/slt2026/action-items.md) (generalization),
`SLT-02` (statistical strength), and `SLT-03` (novelty) by extending from one
corpus to three.

Everything below was verified against the copies in `~/data` on 2026-09-16, not
taken from the papers. Findings that contradict the obvious assumption are
marked **⚠**.

**Corpus choice: PDCH is the cross-lingual corpus; CMDC is demoted to an
optional robustness check.** Rationale in §0.

---

## 0. Why PDCH over CMDC

CTD is a *dyadic* measure — the construct is timing **between** two speakers.
That makes the corpora unequal for our purpose:

| | CMDC | PDCH |
|---|---|---|
| Interviewer recorded | **no** | **yes** (医生 / 患者 labels) |
| Continuous timeline | **no** (per-answer files) | **yes** |
| Ask-side features (9) | ❌ impossible | ✅ |
| Cross-speaker features (7), incl. `res_h` | ❌ impossible | ✅ (see §2c) |
| Response intra-turn (8) | ✅ via VAD | ✅ |
| Of 24 CTD features | **8** | **24** |

On CMDC the cross-lingual claim would be "a reduced 8-feature proxy transfers."
On PDCH it is "**dyadic CTD transfers**" — the same construct the paper is
about. That is a materially stronger answer to `SLT-01`, and it keeps the
cross-corpus columns comparable instead of forcing a footnote per dataset.

PDCH also brings things CMDC cannot: real clinician interviews (not scripted
Q&A), item-level HAMD-17 enabling a severity-regression secondary task, and 27
subjects with two sessions each for within-subject analysis.

Keep CMDC as an **optional tier-C-only robustness check** if a second
Mandarin cohort is wanted. Do not make it load-bearing.

---

## 1. What is actually on disk

| Corpus | Path | Subjects | Audio | Transcripts | Speaker labels |
|---|---|---|---|---|---|
| DAIC-WOZ | `~/data/DAIC` | 189 | 16 kHz mono, full session | human-aligned, `start_time/stop_time/speaker/value` | **yes** (`Ellie`/`Participant`) |
| E-DAIC | `~/data/E-DAIC` | 275 | 16 kHz mono, full session | ASR, `Start_Time,End_Time,Text,Confidence` | **⚠ no** |
| **PDCH** | `~/data/PDCH` | **72** (99 sessions) | 22.05 kHz mono, **chunked** | turn-level + `MM:SS` timestamps | **yes** (`医生`/`患者`) |
| CMDC | `~/data/CMDC` | 78 (26 MDD / 52 HC) | 48 kHz stereo, per-answer files | one untimed string per answer | **⚠ n/a** |

### ⚠ 1a. E-DAIC fully contains DAIC-WOZ

```
DAIC-WOZ sessions : 189   range 300-492
E-DAIC sessions   : 275   range 300-718
OVERLAP           : 189   (all of DAIC-WOZ)
E-DAIC only       :  86   range 600-718
```

Only **86** sessions (IDs 600–718) are new recordings. Mapped onto E-DAIC's
official splits:

| E-DAIC split | n | in DAIC-WOZ | new (600+) |
|---|---|---|---|
| train | 163 | 143 | 20 |
| dev | 56 | 46 | 10 |
| **test** | 56 | **0** | **56** |

E-DAIC's official *test* split is entirely new sessions, but its train/dev are
mostly DAIC-WOZ subjects. Reporting both corpora with official splits would
train on the subjects we evaluate in the DAIC-WOZ experiment. Use the
**disjoint** definition (§3) and disclose the overlap; after `SLT-13` we know
this reviewer pool audits split arithmetic.

### ⚠ 1b. E-DAIC has no speaker labels

All 275 transcripts share the identical 4-column ASR header with no speaker
column, and both speakers appear in one unlabeled stream (session 718 mixes
`"how long should this person last"` with `"I'm doing okay it's Friday"`).
Ask/Res pairing requires **diarization**. Mitigating: in sessions 600–718 the
agent is fully autonomous with a synthetic TTS voice, so synthetic-vs-human
separation is easy; the `Confidence` column allows filtering weak ASR.

---

## 2. PDCH specifics — verified, with four gotchas

Structure: `wav_data/<subject><session>/` where session is `A` or `B`, e.g.
`001A`. Each directory holds chunked audio `N.wav` plus `N.txt`,
`N_correction.txt`, and `N_correction_timestamp_emotion.txt`.

```
session directories : 100      timestamped transcripts : 165
labeled sessions    :  99      unique subjects         :  72
subjects with A+B   :  27      speaker tags: 医生 16109 / 患者 16451 / ? 3
```

Format — turn-level, one `MM:SS-MM:SS` line then one `speaker：text` line:

```
00:00-00:02
患者：小学没毕业，
00:02-00:05
医生：小学那会儿那个时候都家里没条件。
```

### ⚠ 2a. 1500 s is chunking, not truncation

67 of 167 audio files are *exactly* 1500.0 s, which looks like a 25-minute
truncation. It is not:

```
dirs with 2 chunks: 63  → chunk0 is exactly 1500.0s in 63/63
dirs with 1 chunk : 35  → chunk0 is exactly 1500.0s in  0/35
dirs with 3 chunks:  2
001A: chunk0=1500.0s  chunk1=  95.6s  total=1595.6s
001B: chunk0=1500.0s  chunk1=1391.7s  total=2891.7s
```

Long recordings are split at 25-minute boundaries, so **no data is lost** — but
**timestamps restart at `00:00` in every chunk**. Concatenating without adding
the chunk offset would corrupt turn pairing at each boundary and produce
nonsense `res_h` there. Reconstruct the session timeline first; 2 wav files have
no timestamped transcript and must be dropped.

### ⚠ 2b. Only 62 of 99 sessions have a usable label

`HAMD_annotation_en.xlsx` has 99 rows of item-level HAMD-17 (items 1–17 plus
total), keyed by **session** (`007A`), but 37 rows have a blank total:

```
HAMD-17 total: n=62  min=1  max=32  mean=15.7
  >= 7 : 55/62 (89%)
  >= 8 : 53/62 (85%)
  >= 17: 27/62 (44%)
```

**The standard clinical cutoff is unusable here.** At HAMD ≥ 8, 85% of the
cohort is positive — this is an inpatient sample where nearly everyone is
depressed. A majority-class baseline scores 85% accuracy and macro-F1 would
hinge on 9 negative cases.

Label strategy:

- **Primary: HAMD-17 ≥ 17** (moderate-or-worse), giving 27/62 = 44% prevalence —
  balanced and close to DAIC's ~30%. The construct shifts to *severity
  stratification* rather than screening; say so explicitly.
- **Secondary: regression on HAMD-17 total.** Richer use of the data, and
  severity prediction from timing is a cleaner novelty angle than re-running
  binary screening.
- **Do not** use HAMD ≥ 8. Do not pool PDCH labels with PHQ-8 labels: HAMD-17
  is clinician-rated, PHQ-8 is self-report.

### ⚠ 2c. Timestamps are 1-second, turn-level — 10 features degenerate as-is

Across 32,398 adjacent turn pairs:

```
gap == 0 (contiguous): 21255 (65.6%)
gap  < 0 (overlap)   :   110 ( 0.3%)
gap  > 0             : 11033 (34.1%)
median gap 0   mean 0.69   min -60   max 694
consecutive same-speaker turns: 2650 (8.2%)
```

Two problems follow.

**`res_h` is coarsely quantized.** Response latency is a sub-second
phenomenon; at 1 s resolution with 66% of gaps recorded as 0, `res_h` collapses
to mostly 0 and 1. The implausible extremes (−60 s, 694 s) are annotation noise
needing cleaning.

**Intra-turn silence features degenerate.** The transcript is *turn*-level, not
utterance-level, so a speaker run is a single utterance 92% of the time. With
`len(utts) < 2`, `turn_s()` returns 0, which forces
`ask_ud = ask_du = 1.0`, `ask_sd = ask_su = 0.0`, `ask_st = 0`, and
`ask_ds = ask_us = NaN` — and the same for `res_*`. That is **10 of 24
features constant or undefined**.

**Fix — VAD refinement, and it solves both.** We have the audio and the turn
boundaries, so run VAD *inside* each turn span to recover utterance-level
segments. This restores all intra-turn silence features at proper resolution,
and refining each turn's true speech onset/offset also recovers **sub-second
`res_h`** instead of the 1 s label. This is the single most important
engineering step for PDCH; without it PDCH looks superficially complete while
silently feeding constants into the detector.

#### ✅ Feasibility spike result — VAD refinement works (`scripts/spike_pdch_vad.py`)

Run on 10 sessions / 3,103 turns, webrtcvad aggressiveness 2, 30 ms frames,
22.05 kHz → 16 kHz:

| Question | Result | Verdict |
|---|---|---|
| Q1 alignment holds per turn | **97.3%** of labelled turns contain VAD speech | ✅ |
| Q2 intra-turn structure recovered | **4.71** utterances/turn; >1 utterance in **78.8%** of turns vs. **8.2%** from labels alone | ✅ the 10 degenerate features come back |
| Q3 sub-second `res_h` | median **0.60 s**, p10 0.06 / p25 0.21 / p75 1.29 / p90 2.28; 0.2% negative (overlap); **155 distinct values vs. 22** | ✅ plausible for a clinical interview |

**Go.** All three gates pass, so `MC-06` is de-risked and PDCH is viable.

⚠ **Use max-overlap assignment, not clipping.** Turn labels supply *speaker
attribution*; VAD supplies the *boundaries*. Clipping VAD segments to the
labelled span pins any boundary-straddling segment to the turn edge and
manufactures an exact-zero gap — measured at **26% spurious zeros**, which
would have looked like genuine near-instant responses. Assigning each unclipped
segment to the turn it overlaps most drops that to **0.0%**. The spike script
implements the correct method; `segs_within()` is retained for intra-turn
counting only and carries a warning.

Caveat: `res_bt` (overlap/interruption) is still unlikely to survive — only
0.2% of refined gaps are negative, consistent with contiguous annotation rather
than a genuine absence of interruptions in clinical dialogue. Treat `res_bt` as
not transferable to PDCH until shown otherwise.

### ⚠ 2d. Subject-level splitting is mandatory

27 of 72 subjects have two sessions, with different labels (`007`: A=26, B=24;
`013`: A=10, B=32; `011`: A=3, B=19). Splitting by session would put the same
person in train and test. Use `GroupKFold` on subject ID. The paired sessions
are also an opportunity: within-subject change in CTD against change in HAMD is
a strong, confound-resistant analysis that no reviewer asked for.

### 2e. Minor

Audio is 22.05 kHz mono → resample to 16 kHz for WavLM. Despite the filename,
`_correction_timestamp_emotion.txt` carries **no emotion annotation** (a single
stage-direction parenthetical corpus-wide); `N.txt` and `N_correction.txt` were
identical in the files checked, so prefer the `_correction_timestamp_*` variant
for timing and treat the name as a misnomer.

---

## 3. Corpus definitions to use (zero subject overlap)

| Name in paper | Definition | n | Interviewer | Language | Label |
|---|---|---|---|---|---|
| **DAIC-WOZ** | official cleaned splits, as in the SLT paper | 180 | wizard-controlled agent | English | PHQ-8 ≥ 10 |
| **E-DAIC-Extended** | sessions 600–718 only (disjoint from DAIC-WOZ) | 86 | **fully autonomous** agent | English | PHQ-8 ≥ 10 |
| **PDCH** | labeled sessions, subject-grouped CV | 62 sessions / ~50 subjects | **human clinician** | Mandarin | HAMD-17 ≥ 17 |

Three datasets, no shared subjects, and — the part worth building the paper
around — **three distinct interviewer regimes**.

**The interviewer axis is the scientific prize.** The confound rwao and HoqK
both raised is that ask-side timing reflects the interviewer, not the patient.
These three corpora span the full range: wizard-driven, fully autonomous, and
free human clinician. That turns a defensive ablation into a designed
experiment, and it is the strongest available answer to `SLT-03` (novelty).

Note the direction of the confound in PDCH is *worse* than in DAIC-WOZ: a real
clinician adapts to the patient, plausibly spending longer with sicker ones, so
ask-side features there may encode severity-driven clinician behavior. Predict
this in advance and test it with the tier design (§4) rather than discovering
it in review.

---

## 4. Feature tiers — what transfers where

Derived from `src/ctd/feature_extraction.py`. The 24 features partition by the
signal they require:

| Tier | n | Features | DAIC-WOZ | E-DAIC-Ext | PDCH | CMDC |
|---|---|---|---|---|---|---|
| **A** ask-side | 9 | `ask_d`, `ask_ud`, `ask_du`, `ask_sd`, `ask_ds`, `ask_su`, `ask_us`, `ask_st`, `ask_bt` | ✅ | ✅ diarization | ✅ +VAD | ❌ |
| **B** cross-speaker | 7 | `res_minus_ask`, `ask_minus_res`, `duration_sum`, `res_over_ask`, `ask_over_res`, `res_h`, `res_bt` | ✅ | ✅ diarization | ✅ +VAD | ❌ |
| **C** response intra-turn | 8 | `res_d`, `res_ud`, `res_du`, `res_sd`, `res_ds`, `res_su`, `res_us`, `res_st` | ✅ | ✅ | ✅ +VAD | ✅ VAD |

### ⚠ 4a. Measured tier performance changes the risk picture

The `SLT-05` ablation has now been run on DAIC-WOZ
(`output/ctd_ablation_groups.md`):

| Tier config | n | Dev macro-F1 | Test macro-F1 |
|---|---:|---:|---:|
| A+B+C (`all24`) | 24 | 0.746 | 0.631 |
| B+C (`no_ask`) | 15 | **0.814** | **0.641** |
| A+C (`no_cross`) | 17 | 0.700 | 0.527 |
| A only (`ask_only`) | 9 | 0.594 | 0.593 |
| **C only (`res_only`)** | 8 | **0.607** | **0.517** |

Two consequences:

- **Tier C alone is near chance (test .517).** Since tier C is the *only* set
  computable on CMDC, CMDC is effectively unusable for CTD — it would show the
  method failing for a reason that has nothing to do with language. This
  retires `MC-13` rather than merely deprioritizing it.
- **The signal lives in tier B (cross-speaker).** Removing it (`no_cross`)
  costs more than removing the interviewer side. `res_h` is a tier-B feature,
  and on PDCH `res_h` is the one degraded by 1-second quantization — so
  **`MC-06` (VAD refinement) is now mandatory, not an optimization.** Without
  it PDCH would test the method on its weakest tier.

With PDCH the full 24 are available on all three primary corpora, so the tiers
stop being a per-corpus compromise and become the **confound experiment**:
A+B+C = 24 (everything), B+C = 15 (no interviewer-only cues — the configuration
the rebuttal already reports at dev .752 / test .661), C = 8 (participant only,
fully confound-free). Running all three tiers on all three corpora yields a
3 × 3 table in which the confound question is answered by construction.

CMDC, if included, contributes tier C only.

---

## 5. Controls required for PDCH

Beyond the corpus-specific gotchas in §2:

**Language.** Mandarin is syllable-timed and tonal; English is stress-timed.
Absolute durations are not comparable across languages. Standardize **within
corpus** (z-score on that corpus's training fold only — never fit a scaler
across corpora), and prefer dimensionless ratios (`res_ud`, `res_sd`,
`res_su`) over absolute durations for cross-lingual claims. Predict in advance
that ratios transfer and absolute durations do not, so it reads as a hypothesis
rather than a rescue.

**Re-derive the silence threshold.** `_silence_gap_count` hardcodes
`threshold: float = 0.2` s, tuned on DAIC-WOZ. Re-select it on PDCH training
folds after VAD and report sensitivity.

**Setting.** PDCH is a clinical inpatient cohort; DAIC-WOZ is community
screening. Severity distribution, interview length (up to ~48 min vs ~15 min),
and rapport all differ. Cap or normalize session length so `duration_sum` isn't
a corpus detector.

**Demographics.** Check `HAMD_annotation.xlsx` sheets 2–3 for age/gender; speech
timing is strongly age-dependent, and subgroup reporting also closes rwao's
Ethics-4 gap.

**Semantic branch.** RoBERTa-large is English-only. Use a Chinese encoder
(`chinese-roberta-wwm-ext`) or multilingual XLM-R — the latter keeps one
architecture across all corpora, which is cleaner for a fusion comparison. If it
cannot be made comparable, report PDCH as **CTD + acoustic only** and say why; a
missing modality, explained, is fine, an incomparable one is not.

**Licensing.** Confirm PDCH redistribution terms before any derived artifact
enters the public repo. Current policy (code and aggregate metrics only) should
satisfy it, but verify — same for CMDC's `CMDC_EULA.pdf` if used.

---

## 6. Engineering prerequisite

`src/ctd/turn_pairing.py` hardcodes the string literals `"Ellie"` and
`"Participant"` (lines 63, 71, 75), and `constants.py` hardcodes a single
`DAIC_WOZ_ROOT`. No other corpus provides those labels.

Needed before any new-corpus experiment: a corpus adapter layer yielding a
uniform `list[Utterance]` with `(start, end, speaker)` per session, with
backends for DAIC-WOZ (read labels), E-DAIC (diarize), and PDCH (parse
`MM:SS` turns, stitch chunk offsets, VAD-refine within turns) — plus a tier
selector so one detector runs at 24/15/8 features. Build this before the
experiments or the CTD code forks three ways.

---

## 7. Ordering

Do these first, on DAIC-WOZ, before touching a new corpus:

1. **`SLT-04` + `SLT-05`** — coefficients and the ask-side ablation. These
   *define* the tier structure in §4, so running a new corpus first would make
   the tier boundaries look post-hoc. Results already exist in the rebuttal.
2. **`SLT-10`** — resampling/seed machinery for the CTD detector. Build once,
   reuse for all corpora; also feeds `SLT-02`.
3. **`SLT-13`** — one-line wording fix, free while in `daic_cleaning.py`.
4. **Corpus adapter refactor** (§6).

Then **PDCH** (adds cross-lingual *and* the human-clinician regime; the biggest
single win, and the heaviest lift because of VAD refinement), then
**E-DAIC-Extended** (cheap once diarization works, completes the interviewer
axis). CMDC only if a second Mandarin cohort is wanted.

Deferred — independent of this work: `SLT-06`, `SLT-07` (prior-work baselines
and official-split reporting), `SLT-08` (acoustic probe), all P2 items.
