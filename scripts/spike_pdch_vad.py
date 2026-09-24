#!/usr/bin/env python3
"""Feasibility spike (MC-06): can VAD recover sub-second turn onsets in PDCH?

Throwaway diagnostic, not part of the pipeline. Answers three go/no-go
questions before any PDCH engineering is committed:

  Q1  Do VAD segments fall inside the labelled turn spans (alignment holds
      at turn level, not just session level)?
  Q2  Does VAD recover >1 utterance per turn? If not, the 10 intra-turn
      silence features stay degenerate (turn_s == 0) and PDCH can only ever
      support a reduced feature set.
  Q3  Is the VAD-refined res_h distribution sub-second and plausible?
      Label-derived res_h is ~0 for 66% of turn pairs because the annotation
      is contiguous; a credible refined median is roughly 0.2-0.9 s.

Usage: python scripts/spike_pdch_vad.py [n_sessions]
"""
from __future__ import annotations

import re
import sys
import wave
from pathlib import Path

import numpy as np
import webrtcvad
from scipy.signal import resample_poly

PDCH = Path.home() / "data" / "PDCH" / "wav_data"
VAD_SR = 16000
FRAME_MS = 30
VAD_AGGRESSIVENESS = 2
ASK, RES = "医生", "患者"

TS = re.compile(r"^(\d+):(\d+)-(\d+):(\d+)\s*$")


def parse_turns(path: Path) -> list[tuple[float, float, str]]:
    lines = path.read_text(encoding="utf8", errors="ignore").splitlines()
    out, i = [], 0
    while i < len(lines) - 1:
        m = TS.match(lines[i].strip())
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


def load_16k(path: Path) -> np.ndarray:
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


def segs_within(segs, a: float, b: float, min_dur: float = 0.09):
    """VAD segments clipped to [a, b], dropping slivers.

    Used for intra-turn counting only. Do NOT use for res_h: clipping pins a
    boundary-straddling segment to the turn edge, manufacturing an exact-zero
    gap. Measured cost: 26% spurious zeros. Use assign_segments() instead.
    """
    out = []
    for s, e in segs:
        lo, hi = max(s, a), min(e, b)
        if hi - lo >= min_dur:
            out.append((lo, hi))
    return out


def assign_segments(segs, turns):
    """turn index -> unclipped VAD segments, assigned by maximum overlap.

    Turn labels give speaker attribution; VAD gives the true boundaries. Keeping
    segments unclipped is what recovers real sub-second latencies.
    """
    owner: dict[int, list[tuple[float, float]]] = {}
    for s, e in segs:
        best, best_j = 0.0, None
        for j, (a, b, _) in enumerate(turns):
            ov = min(e, b) - max(s, a)
            if ov > best:
                best, best_j = ov, j
        if best_j is not None:
            owner.setdefault(best_j, []).append((s, e))
    return owner


def main() -> None:
    n_sessions = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    dirs = sorted(d for d in PDCH.iterdir() if d.is_dir())[:n_sessions]

    utt_counts, refined_h, label_h, covered, n_turns_tot = [], [], [], [], 0
    per_session = []

    for d in dirs:
        tpath, wpath = d / "0_correction_timestamp_emotion.txt", d / "0.wav"
        if not (tpath.exists() and wpath.exists()):
            continue
        turns = parse_turns(tpath)
        if not turns:
            continue
        segs = vad_segments(load_16k(wpath))

        s_utts, s_in = [], 0
        for a, b, _ in turns:
            u = segs_within(segs, a, b)
            s_utts.append(len(u))
            s_in += len(u) > 0
        utt_counts.extend(s_utts)
        covered.append(s_in / len(turns))
        n_turns_tot += len(turns)

        # res_h on consecutive ASK -> RES pairs, via unclipped max-overlap assignment
        owner = assign_segments(segs, turns)
        s_h = []
        for j in range(len(turns) - 1):
            (a0, b0, s0), (a1, b1, s1) = turns[j], turns[j + 1]
            if s0 != ASK or s1 != RES:
                continue
            label_h.append(a1 - b0)
            ask_u, res_u = owner.get(j), owner.get(j + 1)
            if ask_u and res_u:
                s_h.append(min(u[0] for u in res_u) - max(u[1] for u in ask_u))
        refined_h.extend(s_h)
        per_session.append((d.name, len(turns), np.mean(s_utts), s_in / len(turns), len(s_h)))

    uc = np.array(utt_counts)
    rh = np.array(refined_h)
    lh = np.array(label_h)

    print(f"\n=== PDCH VAD feasibility spike — {len(per_session)} sessions, "
          f"{n_turns_tot} turns ===\n")
    print(f"{'session':<8}{'turns':>7}{'utt/turn':>10}{'turns w/ VAD':>14}{'ask→res pairs':>15}")
    for name, nt, mu, cov, nh in per_session:
        print(f"{name:<8}{nt:>7}{mu:>10.2f}{cov:>13.0%}{nh:>15}")

    print(f"\nQ1 alignment : {np.mean(covered):.1%} of labelled turns contain VAD speech")
    print(f"Q2 intra-turn: mean {uc.mean():.2f} utterances/turn | "
          f">1 utterance in {(uc > 1).mean():.1%} of turns "
          f"(label-only baseline: 8.2%)")
    if rh.size:
        q = np.percentile(rh, [10, 25, 50, 75, 90])
        print(f"Q3 res_h     : n={rh.size}  median={np.median(rh):.2f}s  "
              f"mean={rh.mean():.2f}s")
        print(f"             deciles p10={q[0]:.2f} p25={q[1]:.2f} p50={q[2]:.2f} "
              f"p75={q[3]:.2f} p90={q[4]:.2f}")
        print(f"             sub-second: {(np.abs(rh) < 1).mean():.1%} | "
              f"negative (overlap): {(rh < 0).mean():.1%}")
        print(f"   label-derived res_h : median={np.median(lh):.2f}s  "
              f"zero-valued={np.mean(lh == 0):.1%}  (n={lh.size})")
        uniq = len(np.unique(np.round(rh, 2)))
        print(f"   resolution gained   : {uniq} distinct refined values vs "
              f"{len(np.unique(lh))} distinct label values")


if __name__ == "__main__":
    main()
