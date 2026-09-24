#!/usr/bin/env python3
"""PDCH corpus adapter: stitching, VAD refinement, and `list[Utterance]` output.

Covers three action items:

* `MC-05` chunk stitching -- PDCH sessions are split into ~1500 s audio chunks
  whose transcript timestamps restart at ``00:00``; the cumulative duration of
  prior chunks is added before anything else happens.
* `MC-06` VAD refinement -- the shipped timestamps are turn-level and
  1-second-quantized, which leaves 10 of the 24 CTD features constant or NaN
  and collapses `res_h` onto the integers. Running VAD *inside* the labelled
  spans restores intra-turn structure and sub-second latency.
* `MC-01` the adapter itself -- `load_pdch_session()` returns the same
  `list[Utterance]` shape `turn_pairing.build_turn_pairs()` already consumes,
  so `feature_extraction.py` and `functionals.py` run unmodified.

The VAD method (`load_16k`, `vad_segments`, and the assignment rule) is the one
vetted by `scripts/spike_pdch_vad.py`: webrtcvad aggressiveness 2, 30 ms frames,
resampled to 16 kHz, with each **unclipped** segment assigned to the turn it
overlaps most.

.. warning::
   Do not clip VAD segments to the labelled turn boundary before computing
   gaps. Clipping pins any boundary-straddling segment to the turn edge and
   manufactures an exact-zero gap; the spike measured 26% spurious
   zero-latency turns from that bug. Turn labels supply *speaker attribution*
   only -- VAD supplies the *boundaries*.
"""
from __future__ import annotations

import re
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import webrtcvad
from scipy.signal import resample_poly

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from turn_pairing import AskResLabels, Utterance  # noqa: E402

PDCH_ROOT = Path.home() / "data" / "PDCH"
PDCH_WAV_ROOT = PDCH_ROOT / "wav_data"

# Speaker tags, analogous to DAIC-WOZ's Ellie / Participant.
ASK = "医生"   # doctor / clinician -> Ask side
RES = "患者"   # patient            -> Res side
PDCH_LABELS = AskResLabels(ask=ASK, res=RES)

# VAD configuration -- identical to the vetted spike.
VAD_SR = 16000
FRAME_MS = 30
VAD_AGGRESSIVENESS = 2

TS_RE = re.compile(r"^(\d+):(\d+)-(\d+):(\d+)\s*$")
CHUNK_TRANSCRIPT = "{n}_correction_timestamp_emotion.txt"


@dataclass(frozen=True)
class Turn:
    """One labelled turn span, in session-level seconds after stitching."""

    start: float
    end: float
    speaker: str
    chunk: int


@dataclass
class SessionReport:
    """Per-session provenance, so pipeline attrition is auditable."""

    session_id: str
    n_chunks: int = 0
    chunk_durations: list[float] = field(default_factory=list)
    n_turns_raw: int = 0
    n_turns_dropped_unknown_speaker: int = 0
    n_turns_dropped_non_monotonic: int = 0
    n_turns_kept: int = 0
    n_vad_segments: int = 0
    n_turns_with_vad: int = 0
    n_utterances: int = 0
    audio_seconds: float = 0.0

    @property
    def vad_coverage(self) -> float:
        return self.n_turns_with_vad / self.n_turns_kept if self.n_turns_kept else 0.0

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["vad_coverage"] = self.vad_coverage
        return d


# --------------------------------------------------------------------------
# Transcript parsing
# --------------------------------------------------------------------------
def parse_turns(path: Path) -> list[tuple[float, float, str]]:
    """Parse one chunk transcript into (start_s, end_s, speaker) triples.

    Same two-line format and regex as the spike: an ``MM:SS-MM:SS`` line
    followed by a ``speaker：text`` line.
    """
    lines = path.read_text(encoding="utf8", errors="ignore").splitlines()
    out: list[tuple[float, float, str]] = []
    i = 0
    while i < len(lines) - 1:
        m = TS_RE.match(lines[i].strip())
        if m:
            a = int(m.group(1)) * 60 + int(m.group(2))
            b = int(m.group(3)) * 60 + int(m.group(4))
            text = lines[i + 1]
            spk = text.split("：")[0] if "：" in text else "?"
            out.append((float(a), float(b), spk))
            i += 2
        else:
            i += 1
    return out


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / float(w.getframerate())


def session_chunks(session_dir: Path) -> list[int]:
    """Chunk indices that have *both* a wav and a timestamped transcript.

    Per `MC-05` step 5, a wav with no timestamped transcript is dropped; the
    caller logs it. Returned sorted by chunk index, which is the order the
    chunks were recorded in and therefore the order offsets accumulate in.
    """
    wavs = {int(p.stem) for p in session_dir.glob("*.wav") if p.stem.isdigit()}
    kept = [n for n in sorted(wavs) if (session_dir / CHUNK_TRANSCRIPT.format(n=n)).exists()]
    return kept


def dropped_chunks(session_dir: Path) -> list[int]:
    """Chunk indices with a wav but no timestamped transcript."""
    wavs = {int(p.stem) for p in session_dir.glob("*.wav") if p.stem.isdigit()}
    return [n for n in sorted(wavs)
            if not (session_dir / CHUNK_TRANSCRIPT.format(n=n)).exists()]


# --------------------------------------------------------------------------
# MC-05 -- chunk stitching
# --------------------------------------------------------------------------
def stitch_session_turns(session_dir: Path) -> tuple[list[Turn], SessionReport]:
    """Concatenate a session's chunks onto one monotonic timeline.

    Timestamps restart at ``00:00`` in every chunk, so the cumulative duration
    of all prior chunks (read from the wav headers, not assumed to be 1500 s)
    is added to every timestamp parsed from that chunk.

    Two classes of turn are dropped, both logged in the report:

    * ``?`` speaker -- 3 turns corpus-wide whose line carries no ``：``
      separator, so no speaker can be attributed.
    * non-monotonic starts -- 33 turns corpus-wide (0.10%) whose start jumps
      backwards while the end continues forwards, e.g. ``23:54-23:55`` followed
      by ``23:00-24:06``. The seconds field of the start is corrupt and the
      true span is unrecoverable, so the turn is dropped rather than guessed
      at. A running maximum makes the rule single-pass and order-independent:
      keep a turn only if its start is at least the largest start seen so far.
    """
    report = SessionReport(session_id=session_dir.name)
    chunks = session_chunks(session_dir)
    report.n_chunks = len(chunks)

    offset = 0.0
    raw: list[Turn] = []
    for n in chunks:
        dur = wav_duration_seconds(session_dir / f"{n}.wav")
        report.chunk_durations.append(dur)
        for a, b, spk in parse_turns(session_dir / CHUNK_TRANSCRIPT.format(n=n)):
            raw.append(Turn(start=a + offset, end=b + offset, speaker=spk, chunk=n))
        offset += dur
    report.audio_seconds = offset
    report.n_turns_raw = len(raw)

    kept: list[Turn] = []
    max_start = -np.inf
    for t in raw:
        if t.speaker not in (ASK, RES):
            report.n_turns_dropped_unknown_speaker += 1
            continue
        if t.start < max_start:
            report.n_turns_dropped_non_monotonic += 1
            continue
        max_start = t.start
        kept.append(t)
    report.n_turns_kept = len(kept)

    # Verify by construction (MC-05): the stitched timeline is non-decreasing.
    starts = [t.start for t in kept]
    assert all(starts[i] <= starts[i + 1] for i in range(len(starts) - 1)), (
        f"{session_dir.name}: turn starts not monotonically non-decreasing after stitching"
    )
    assert all(t.end >= t.start for t in kept), f"{session_dir.name}: turn with end < start"
    return kept, report


# --------------------------------------------------------------------------
# MC-06 -- VAD refinement (method vetted in scripts/spike_pdch_vad.py)
# --------------------------------------------------------------------------
def load_16k(path: Path) -> np.ndarray:
    """Read a wav as mono int16 at 16 kHz (webrtcvad accepts nothing else)."""
    with wave.open(str(path)) as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        x = np.frombuffer(w.readframes(n), dtype=np.int16)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    x = x.astype(np.float32)
    if sr != VAD_SR:
        g = np.gcd(sr, VAD_SR)
        x = resample_poly(x, VAD_SR // g, sr // g)
    return np.clip(x, -32768, 32767).astype(np.int16)


def vad_segments(x16: np.ndarray) -> list[tuple[float, float]]:
    """Merge contiguous voiced frames into (start_s, end_s) segments."""
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    fl = int(VAD_SR * FRAME_MS / 1000)
    nf = len(x16) // fl
    voiced = np.zeros(nf, dtype=bool)
    raw = x16[: nf * fl].tobytes()
    step = fl * 2  # bytes per frame (int16)
    for i in range(nf):
        voiced[i] = vad.is_speech(raw[i * step : (i + 1) * step], VAD_SR)
    segs, start = [], None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start * FRAME_MS / 1000, i * FRAME_MS / 1000))
            start = None
    if start is not None:
        segs.append((start * FRAME_MS / 1000, nf * FRAME_MS / 1000))
    return segs


def assign_segments(
    segs: list[tuple[float, float]], turns: list[Turn],
) -> dict[int, list[tuple[float, float]]]:
    """turn index -> **unclipped** VAD segments, assigned by maximum overlap.

    Identical semantics to `spike_pdch_vad.assign_segments` (strict ``>``, so
    ties go to the lowest turn index; zero-overlap segments are dropped), but
    the O(segments x turns) scan is narrowed with two binary searches. On a
    2,000-turn session the brute-force version is ~10M overlap computations;
    `tests/test_pdch_adapter.py` asserts the two agree exactly.

    Keeping segments unclipped is what recovers real sub-second latencies --
    see the module docstring's warning.
    """
    if not segs or not turns:
        return {}
    starts = np.fromiter((t.start for t in turns), dtype=float, count=len(turns))
    ends = np.fromiter((t.end for t in turns), dtype=float, count=len(turns))
    # `starts` is non-decreasing after stitching; `ends` need not be, so use a
    # running maximum (which is non-decreasing) as a conservative lower bound.
    cummax_end = np.maximum.accumulate(ends)

    owner: dict[int, list[tuple[float, float]]] = {}
    for s, e in segs:
        lo = int(np.searchsorted(cummax_end, s, side="right"))
        hi = int(np.searchsorted(starts, e, side="left"))
        best, best_j = 0.0, None
        for j in range(lo, hi):
            ov = min(e, ends[j]) - max(s, starts[j])
            if ov > best:
                best, best_j = ov, j
        if best_j is not None:
            owner.setdefault(best_j, []).append((s, e))
    return owner


# --------------------------------------------------------------------------
# MC-01 -- the adapter
# --------------------------------------------------------------------------
def load_pdch_session(session_dir: Path) -> tuple[list[Utterance], SessionReport]:
    """Stitched, VAD-refined utterances for one PDCH session.

    Speakers keep their native PDCH tags (``医生`` = Ask, ``患者`` = Res);
    `turn_pairing.build_turn_pairs()` is told which is which via
    `PDCH_LABELS`, so no corpus's data has to masquerade as another's.

    Returns utterances sorted by start time. VAD segments are globally disjoint
    and each is assigned to exactly one turn, so sorting by start yields a
    well-defined speaker sequence for `_speaker_runs()`.
    """
    turns, report = stitch_session_turns(session_dir)

    segs: list[tuple[float, float]] = []
    offset = 0.0
    for n, dur in zip(session_chunks(session_dir), report.chunk_durations):
        chunk_segs = vad_segments(load_16k(session_dir / f"{n}.wav"))
        segs.extend((s + offset, e + offset) for s, e in chunk_segs)
        offset += dur
    report.n_vad_segments = len(segs)

    owner = assign_segments(segs, turns)
    report.n_turns_with_vad = len(owner)

    utts: list[Utterance] = []
    for j, seg_list in owner.items():
        spk = turns[j].speaker
        utts.extend(Utterance(start=s, end=e, speaker=spk) for s, e in seg_list)
    utts.sort(key=lambda u: (u.start, u.end))
    report.n_utterances = len(utts)
    return utts, report


def subject_of(session_id: str) -> str:
    """Subject ID is the first 3 characters of the session name (`013A` -> `013`)."""
    return session_id[:3]


def session_dirs(root: Path = PDCH_WAV_ROOT) -> list[Path]:
    return sorted(d for d in root.iterdir() if d.is_dir())
