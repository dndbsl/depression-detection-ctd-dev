"""`MC-01`/`MC-05`/`MC-06`: the PDCH adapter's load-bearing invariants.

Four things here are easy to break silently and expensive to notice later:

1. `assign_segments()` is a binary-search-narrowed rewrite of the brute-force
   scan in `scripts/spike_pdch_vad.py`. The spike's version is the *vetted*
   one -- the feasibility numbers in `docs/multi-corpus-plan.md` §2c came from
   it -- so the fast version has to agree with it exactly, not approximately.
2. Segments must stay **unclipped**. Clipping to the labelled turn boundary
   manufactured 26% spurious zero-latency turns once already.
3. Stitching must put chunk `N` after chunk `N-1` on one monotonic timeline.
4. The generalized `build_turn_pairs(..., labels=)` must leave DAIC-WOZ's
   default behaviour bit-identical.

Tests 1-3 are skipped when PDCH is not on disk; test 4 needs no data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CTD_ROOT = PROJECT_ROOT / "src" / "ctd"
for p in (str(CTD_ROOT), str(PROJECT_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from turn_pairing import AskResLabels, Utterance, build_turn_pairs  # noqa: E402

pdch = pytest.importorskip("pdch_adapter")
spike = pytest.importorskip("spike_pdch_vad")

needs_pdch = pytest.mark.skipif(
    not pdch.PDCH_WAV_ROOT.exists(), reason="requires the PDCH corpus on disk"
)

SAMPLE_SESSIONS = ["001A", "013A", "005B"]


def _session_segments(session_dir: Path):
    turns, report = pdch.stitch_session_turns(session_dir)
    segs, offset = [], 0.0
    for n, dur in zip(pdch.session_chunks(session_dir), report.chunk_durations):
        segs.extend((s + offset, e + offset)
                    for s, e in pdch.vad_segments(pdch.load_16k(session_dir / f"{n}.wav")))
        offset += dur
    return turns, segs


@needs_pdch
@pytest.mark.parametrize("name", SAMPLE_SESSIONS)
def test_assign_segments_matches_vetted_spike(name):
    """The fast assignment must reproduce the spike's brute-force result exactly."""
    turns, segs = _session_segments(pdch.PDCH_WAV_ROOT / name)
    fast = pdch.assign_segments(segs, turns)
    brute = spike.assign_segments(segs, [(t.start, t.end, t.speaker) for t in turns])
    assert fast == brute


@needs_pdch
@pytest.mark.parametrize("name", SAMPLE_SESSIONS)
def test_assigned_segments_are_unclipped_and_disjoint(name):
    """Every emitted utterance is a whole VAD segment, never clipped to a turn edge.

    Clipping is the bug that manufactured 26% exact-zero response latencies
    (`docs/multi-corpus-plan.md` §2c). If a segment ever gets truncated to a
    labelled boundary this test fails.
    """
    turns, segs = _session_segments(pdch.PDCH_WAV_ROOT / name)
    seg_set = {(round(s, 6), round(e, 6)) for s, e in segs}
    utts, _ = pdch.load_pdch_session(pdch.PDCH_WAV_ROOT / name)

    for u in utts:
        assert (round(u.start, 6), round(u.end, 6)) in seg_set

    # Disjoint and sorted -- what makes turn_d == turn_u + turn_s hold exactly.
    for a, b in zip(utts, utts[1:]):
        assert a.start <= b.start
        assert a.end <= b.start + 1e-9


@needs_pdch
@pytest.mark.parametrize("name", SAMPLE_SESSIONS)
def test_stitching_is_monotonic_and_offsets_accumulate(name):
    session_dir = pdch.PDCH_WAV_ROOT / name
    turns, report = pdch.stitch_session_turns(session_dir)

    starts = np.array([t.start for t in turns])
    assert np.all(np.diff(starts) >= 0)
    assert all(t.end >= t.start for t in turns)

    # A later chunk's turns must land after every earlier chunk's audio.
    cum = np.cumsum([0.0, *report.chunk_durations])
    for t in turns:
        k = report.chunk_durations and pdch.session_chunks(session_dir).index(t.chunk)
        assert t.start >= cum[k] - 1e-6
    assert report.n_turns_kept <= report.n_turns_raw


@needs_pdch
def test_speakers_are_native_pdch_tags():
    """PDCH data is never relabelled to masquerade as DAIC-WOZ (integration (b))."""
    utts, _ = pdch.load_pdch_session(pdch.PDCH_WAV_ROOT / SAMPLE_SESSIONS[0])
    assert {u.speaker for u in utts} <= {pdch.ASK, pdch.RES}


def test_build_turn_pairs_default_is_daic_woz():
    """The `labels` parameter must not change behaviour for existing callers."""
    utts = [
        Utterance(0.0, 1.0, "Ellie"),
        Utterance(1.5, 3.0, "Participant"),
        Utterance(3.5, 4.0, "Ellie"),
        Utterance(4.5, 6.0, "Participant"),
    ]
    default = build_turn_pairs(1, utts)
    explicit = build_turn_pairs(1, utts, AskResLabels("Ellie", "Participant"))
    assert len(default) == len(explicit) == 2
    assert [(p.ask_utts, p.res_utts) for p in default] == \
           [(p.ask_utts, p.res_utts) for p in explicit]


def test_build_turn_pairs_honours_custom_labels():
    utts = [
        Utterance(0.0, 1.0, pdch.ASK),
        Utterance(1.5, 3.0, pdch.RES),
        Utterance(3.5, 4.0, pdch.ASK),
        Utterance(4.5, 6.0, pdch.RES),
    ]
    pairs = build_turn_pairs("013A", utts, pdch.PDCH_LABELS)
    assert len(pairs) == 2
    assert pairs[0].ask_utts[0].speaker == pdch.ASK
    assert pairs[0].res_utts[0].speaker == pdch.RES
    # The DAIC-WOZ default must find nothing in PDCH-tagged data.
    assert build_turn_pairs("013A", utts) == []
