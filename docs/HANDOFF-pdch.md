# Handoff — PDCH transfer (MC-01, MC-05, MC-06, MC-07, MC-08)

You are picking up work on this repo with no prior chat history. This file is
the entry point for **this specific task**: implementing CTD transfer to the
PDCH corpus. Read it fully before running anything.

## What this project is

Code for an SLT 2026 submission on **conversational temporal dynamics (CTD)** —
dyadic Ask/Res turn-pair timing as a standalone modality for depression
detection on DAIC-WOZ, fused with frozen WavLM-large and RoBERTa-large. **The
paper was rejected**; we are revising for **ICASSP 2027**. One of the three
rejection reasons was limited generalization (`SLT-01`), and PDCH (Mandarin
clinical interviews) is the corpus chosen to answer it — see
`docs/multi-corpus-plan.md` for why PDCH was chosen over CMDC.

## Read these, in this order

1. `reviews/slt2026/action-items.md` — every reviewer concern as a tracked
   item (`SLT-*`, `MC-*`, `PF-*`). Cite item IDs in commit messages.
2. `docs/multi-corpus-plan.md` §2 ("PDCH specifics") and §4 ("Feature
   tiers") — the full spec for what you're building. Everything in it was
   verified against the actual files in `~/data/PDCH`, not assumed.
3. `docs/preregistration-pdch.md` — **read this before writing any code that
   touches a PDCH label.** Six falsifiable predictions were committed with a
   git timestamp before any PDCH feature was ever joined to any label. You
   MUST NOT edit the predictions. If you have a correction, add a dated entry
   to its "Appendix — dated amendments" section instead.
4. `src/ctd/feature_groups.py` — the ASK/CROSS/RES_ONLY tier definitions your
   PDCH experiment must reuse unchanged.
5. `scripts/spike_pdch_vad.py` — a feasibility spike that already ran and
   passed (see "What's already proven" below). Read it before writing the
   real VAD pipeline; it contains a fixed bug you must not reintroduce.

## Your assignment

Build the PDCH backend end to end and run the pre-registered analysis:

1. **`MC-01`** — a corpus adapter that produces `list[Utterance]`
   (`start, end, speaker`) per PDCH session, matching the shape
   `src/ctd/turn_pairing.py` already consumes for DAIC-WOZ, so
   `feature_extraction.py` and `functionals.py` can run **unmodified**.
2. **`MC-05`** — chunk-offset stitching (PDCH sessions are split into
   ~1500 s audio chunks with timestamps that restart at `00:00` in each).
3. **`MC-06`** — VAD refinement inside each labelled turn span, using the
   *vetted* method from the spike (max-overlap assignment of **unclipped**
   segments — not clipping; see "Traps" below).
4. **`MC-07`** — label construction: HAMD-17 ≥ 17, plus a severity-regression
   secondary target on the HAMD-17 total.
5. **`MC-08`** — subject-grouped cross-validation (27/72 subjects have two
   sessions).
6. Run all five feature-tier configs from `feature_groups.FEATURE_CONFIGS`
   (`all24`, `no_ask`, `res_only`, `ask_only`, `no_cross`) through the CV
   protocol, and evaluate the six pre-registered predictions (P1–P6) in
   `docs/preregistration-pdch.md`.

### Non-goals — do not do these

- **Do not start E-DAIC or CMDC work** (`MC-03`, `MC-04`, `MC-13`). Out of
  scope for this task.
- **Do not touch the DAIC-WOZ pipeline's behavior.** You may add code (e.g. a
  new adapter module), but `ml_splits.py`'s DAIC-WOZ numbers must be
  byte-for-byte reproducible after your changes — rerun
  `python src/ctd/ml_splits.py` at the end and confirm dev .746 / test .631
  is unchanged.
- **Do not build the semantic (RoBERTa) or acoustic (WavLM) branches for
  PDCH** (`MC-09`). CTD-only for this task. If you reach the point of writing
  up results and the semantic branch is in scope for the paper, document its
  omission — do not build a rushed Chinese encoder integration here.
- **Do not edit `docs/preregistration-pdch.md`'s predictions.** Report
  results against them; append corrections in the dated appendix only.
- **Do not mutate `CTD_FEATURE_NAMES`** in `src/ctd/constants.py` or the
  `ASK_ONLY`/`CROSS`/`RES_ONLY` groups in `feature_groups.py`. All PDCH
  feature selection is by column selection against the existing 24 names.
- **Never `git push origin`.** `origin` is the **public** GitHub repo. Work
  goes to the `private` remote only.

## What's already proven — do not redo this work

A feasibility spike (`scripts/spike_pdch_vad.py`, already committed and run)
answered the make-or-break question before any pipeline was built: does VAD
recover usable sub-second timing from PDCH's coarse 1-second turn labels?
**Yes.** On 10 sessions / 3,103 turns:

| Question | Result |
|---|---|
| Q1 alignment | 97.3% of labelled turns contain VAD speech |
| Q2 intra-turn structure | 4.71 utterances/turn; >1 utterance in 78.8% of turns (vs. 8.2% from labels alone) |
| Q3 sub-second `res_h` | median 0.60 s, p10 0.06 / p25 0.21 / p75 1.29 / p90 2.28; 155 distinct values vs. 22 |

Your job is to turn this spike into the real pipeline — same VAD approach
(`webrtcvad`, aggressiveness 2, 30 ms frames, resampled to 16 kHz), same
assignment method, but wired through the actual adapter/feature/CV code
instead of a standalone script, and run on the full corpus (not 10 sessions).

## Environment

The venv already exists with the base CTD requirements installed:

```bash
source ~/.venvs/ctd/bin/activate
export DAIC_WOZ_ROOT=~/data/DAIC   # still needed for the DAIC-WOZ regression check
```

You additionally need `webrtcvad`, which hit a real dependency conflict last
time — `webrtcvad` imports `pkg_resources`, which modern `setuptools`
(≥ 81) no longer ships:

```bash
pip install "setuptools<81"   # must come first, or webrtcvad import fails
pip install webrtcvad
```

Also install `openpyxl` (for `HAMD_annotation_en.xlsx`) if not already
present.

## THE GATE — run this before touching PDCH

```bash
python src/ctd/ml_splits.py
```

Must still print dev macro-F1 **0.746** / test **0.631**, `C=0.3`, cohort
102/33/45 — confirming your changes haven't disturbed the DAIC-WOZ path.
**If it doesn't reproduce, stop and report it before writing any PDCH code.**

## Engineering spec

### Data locations

```
~/data/PDCH/wav_data/<subject><session>/     e.g. 001A, 001B, 013A
    N.wav                                    audio, chunk N (N = 0, 1, 2...)
    N_correction_timestamp_emotion.txt       turn-level transcript for chunk N
~/data/PDCH/HAMD_annotation_en.xlsx          HAMD-17, keyed by session (e.g. "007A")
```

- 100 session directories, 72 unique subjects, 165 timestamped transcript
  files, 99 have some label row, only **62** have a non-blank HAMD-17 total.
- Speaker tags in transcript lines are `医生` (doctor/ask) and `患者`
  (patient/response) — analogous to DAIC-WOZ's `Ellie`/`Participant`.
- Transcript line format, two lines per turn:
  ```
  00:00-00:02
  患者：小学没毕业，
  ```
  Parse with the same regex as the spike:
  `^(\d+):(\d+)-(\d+):(\d+)\s*$` for the timestamp line, then split the next
  line on `：` for `(speaker, text)`.

### Step A — chunk stitching (`MC-05`)

Multi-chunk sessions (63 of 100 have ≥2 `N.wav` files) have timestamps that
**restart at `00:00` in every chunk**. Before any turn pairing:

1. For each session, sort chunks by index `N`.
2. Compute each chunk's audio duration from its `.wav` header.
3. Add the cumulative duration of prior chunks as an offset to every
   timestamp parsed from that chunk's transcript.
4. Concatenate all chunks' turns into one session-level timeline.
5. Two `.wav` files have no matching timestamped transcript — drop them, and
   log which sessions this affects (should not exceed the 2 you'd expect from
   `docs/multi-corpus-plan.md` §2a).

Verify by construction: turn timestamps within a session must be
monotonically non-decreasing after stitching. Assert this.

### Step B — VAD refinement (`MC-06`)

Reuse the vetted approach from `scripts/spike_pdch_vad.py`
(`load_16k`, `vad_segments`, `assign_segments`) rather than reimplementing:

1. Resample each session's audio to 16 kHz mono (`scipy.signal.resample_poly`,
   already in the spike).
2. Run `webrtcvad.Vad(2)` on 30 ms frames to get voiced/unvoiced, merge into
   `(start_s, end_s)` segments.
3. **Assign each unclipped VAD segment to the turn it overlaps most**
   (`assign_segments` in the spike) to get per-turn utterance lists.

**⚠ Trap already found and fixed once — do not reintroduce it.** An earlier
version of this spike *clipped* VAD segments to the labelled turn boundary
before computing gaps. That pins any segment straddling a turn boundary to
the edge and manufactures an exact-zero gap: measured cost was **26%
spurious zero-latency turns**. The fix (already in the spike) is to assign
each *unclipped* segment to its best-overlap turn and compute gaps between
the assigned segments' true (unclipped) times. `segs_within()` in the spike
is deliberately kept only for intra-turn counting and is commented with this
warning — do not use it for any inter-turn gap feature (`res_h` or anything
that touches a turn boundary).

4. Within each turn, the assigned VAD segments become the `Utterance` list
   your adapter emits (this is what restores the 10 intra-turn features that
   were degenerate under raw 1-second turn labels).

### Step C — the adapter itself (`MC-01`, PDCH backend)

Produce a function with this shape (mirroring what `turn_pairing.py` already
assumes as input):

```python
def load_pdch_session(session_dir: Path) -> list[Utterance]:
    """Stitched, VAD-refined utterances for one PDCH session, speaker-labelled
    'Ellie'-equivalent ('医生') and 'Participant'-equivalent ('患者')."""
```

Two integration choices, pick whichever is less invasive when you're in the
code — document which you chose:

- **(a)** Emit utterances with speaker values `"Ellie"` / `"Participant"`
  directly (reusing the DAIC-WOZ string constants so `turn_pairing.py`'s
  `build_turn_pairs()` works completely unmodified), or
- **(b)** Generalize `turn_pairing.py`'s two hardcoded literals
  (`"Ellie"`, `"Participant"`, currently at lines ~63/71/75) into a small
  `AskResLabels` parameter with a DAIC-WOZ default, and pass PDCH's `医生`/
  `患者` explicitly. This is the more correct fix long-term (E-DAIC will need
  the same generalization later) but touches shared code — if you do this,
  the DAIC-WOZ gate check above must still pass unchanged afterward.

Either way, once you have `list[Utterance]`, **`feature_extraction.py`
(`extract_turn_features`) runs as-is** — that's the entire point of building
an adapter instead of reimplementing CTD math for PDCH.

### Step D — labels (`MC-07`)

From `HAMD_annotation_en.xlsx`: 99 rows keyed by session ID (e.g. `007A`), 62
with a non-blank total.

- **Primary label:** `HAMD17_total >= 17` → binary. (**Not** `>= 8` — that
  gives 85% prevalence on this cohort and is useless.)
- **Secondary target:** regression on `HAMD17_total` directly.
- Drop the 37 sessions with a blank total from the classification task (they
  can still be used for the VAD/feature pipeline validation, just not for
  fitting/evaluating the detector).
- Do **not** merge or rescale against PHQ-8 — different instrument, different
  rater (clinician vs. self-report). Keep DAIC-WOZ and PDCH labels in
  separate columns, never in one shared "depression" column.

### Step E — subject-grouped CV (`MC-08`)

27 of 72 subjects contribute two sessions (`A`/`B`) with *different* labels
(e.g. subject `013`: A=10, B=32) — session-level splitting leaks subject
identity into both train and eval.

- Subject ID = first 3 characters of the session directory name
  (`"013A"` → subject `"013"`).
- Reuse `sklearn.model_selection.StratifiedGroupKFold` exactly as
  `src/ctd/resampling.py` already does for its DAIC-WOZ resampling
  (`groups=session_id`, or here `groups=subject_id`) — that code was
  deliberately written group-aware in anticipation of this reuse; look at it
  before writing new CV code.
- With only 62 labelled sessions / ~50 distinct subjects, do **not** carve out
  a fixed held-out test split the way DAIC-WOZ does. Use repeated
  `StratifiedGroupKFold` (e.g. 5-fold × repeats) and report mean ± sd, the
  same "distribution, not single point estimate" approach `SLT-10` already
  established. A single fixed PDCH test split at this n would be exactly the
  small-sample fragility this whole multi-corpus effort is trying to move
  away from.
- Re-select the silence-gap threshold (`_silence_gap_count`'s
  `threshold: float = 0.2` in `feature_extraction.py`, currently DAIC-tuned)
  within PDCH training folds, and report its sensitivity — don't reuse 0.2 s
  unexamined.

### Step F — run and evaluate against the pre-registration

Run all five configs in `feature_groups.FEATURE_CONFIGS` through the
identical deployed protocol (impute → scale → L2 LogReg,
`class_weight='balanced'`, `C` selected within folds) under the CV from Step
E, then check each of `docs/preregistration-pdch.md`'s P1–P6 against the
result — do not just report metrics, explicitly state pass/fail for each
prediction.

## Traps carried over from prior work — read before coding

1. **The 24 features contain exact linear dependencies, not just high
   correlation.** `duration_sum = ask_d + res_d` and
   `res_minus_ask = res_d - ask_d` are exact algebraic identities (verified to
   ~1e-14 on DAIC-WOZ; `matrix_rank` of the 24-D session-mean matrix is 15,
   not 24). Expect the same on PDCH — don't be surprised by a rank-deficient
   design matrix, and don't try to "fix" it by dropping features arbitrarily;
   this is inherent to the feature definitions, not a PDCH bug.
2. **On DAIC-WOZ, `res_only` (8 features) is near chance** (test macro-F1
   .517) while `no_ask` (cross+res, 15 features) is the best config (.641).
   The signal lives in the cross-speaker group, and `res_h` is in that group.
   This is *why* the VAD spike mattered so much — if it had failed, PDCH
   would only support the weakest tier.
3. **`res_bt` (overlap/interruption) is unlikely to transfer.** In the spike,
   only 0.2% of refined gaps were negative (i.e., overlapping speech), which
   is more consistent with contiguous, non-overlap-aware annotation than a
   genuine absence of interruptions in real clinical dialogue. Don't be
   alarmed if `res_bt` looks degenerate on PDCH; report it as a known
   limitation rather than debugging it further.
4. **`ask_d` dominates the fitted DAIC-WOZ model** (largest coefficient,
   sign-consistency 1.00) even though the ablation shows removing all
   ask-side features costs nothing. `docs/preregistration-pdch.md` P6
   predicts `ask_d` should **not** dominate on PDCH (human clinicians adapt to
   the patient, unlike the DAIC-WOZ wizard-driven agent) — if it does
   dominate on PDCH too, report that as evidence against the
   psychomotor-retardation interpretation, not as a bug to fix.
5. **`C` was unstable under resampling on DAIC-WOZ** (modal values tied
   between 0.1 and 1.0 at 26% each; deployed `C=0.3` won only 19% of the
   time). Expect similar instability on PDCH and report `C` as a distribution
   from your CV folds, not a single winning value.

## Git setup

```
branch    icassp2027   (tracks private/icassp2027 — plain `git push` is safe)
origin    https://github.com/dndbsl/depression-detection-ctd.git        PUBLIC
private   https://github.com/dndbsl/depression-detection-ctd-dev.git    private
```

A `.git/hooks/pre-push` guard refuses any push of `reviews/` to `origin`; do
not bypass it. Cite item IDs in commit messages, e.g. `MC-06: VAD refinement
pipeline for PDCH`.

## When you finish

- Confirm the DAIC-WOZ gate still reproduces exactly.
- `output/pdch_ctd_results.json` + a markdown table, following the format of
  `output/ctd_ablation_groups.md` / `output/ctd_resampling.md`.
- Append a dated entry to `docs/preregistration-pdch.md`'s appendix stating
  the pass/fail outcome of P1–P6, with numbers.
- Update statuses for `MC-01`, `MC-05`, `MC-06`, `MC-07`, `MC-08` in
  `reviews/slt2026/action-items.md`.
- Report explicitly: did the DAIC-WOZ gate hold, what fraction of PDCH
  sessions survived the full pipeline (stitching → VAD → labelled), and
  which of P1–P6 passed.
