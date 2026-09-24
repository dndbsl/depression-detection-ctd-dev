#!/usr/bin/env python3
"""Run the expensive PDCH stage once (MC-05 + MC-06) and cache the result.

VAD over ~35 hours of audio is the only slow step in the PDCH pipeline, and
nothing downstream depends on it beyond the utterance list. Caching it here
means the silence-threshold sweep and the CV protocol can be re-run in
seconds.

Outputs (under `output/pdch/`):
  pdch_utterances.csv  session_id, start, end, speaker   (VAD-refined)
  pdch_session_report.csv  per-session provenance / attrition

Usage: python scripts/build_pdch_cache.py [n_workers]
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "ctd"))

import pdch_adapter as A  # noqa: E402

OUT_DIR = REPO_ROOT / "output" / "pdch"


def _one(session_dir: Path) -> tuple[str, list[tuple[float, float, str]], dict, list[int]]:
    utts, rep = A.load_pdch_session(session_dir)
    rows = [(u.start, u.end, u.speaker) for u in utts]
    return session_dir.name, rows, rep.to_dict(), A.dropped_chunks(session_dir)


def main() -> None:
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dirs = A.session_dirs()
    print(f"PDCH: {len(dirs)} session directories, {n_workers} workers")

    utt_rows, reports, dropped = [], [], []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, (name, rows, rep, drop) in enumerate(ex.map(_one, dirs), 1):
            utt_rows.extend((name, s, e, spk) for s, e, spk in rows)
            reports.append(rep)
            if drop:
                dropped.append((name, drop))
            print(f"  [{i:>3}/{len(dirs)}] {name}: {rep['n_turns_kept']} turns, "
                  f"{rep['n_utterances']} utts, VAD coverage {rep['vad_coverage']:.1%}",
                  flush=True)

    utts = pd.DataFrame(utt_rows, columns=["session_id", "start", "end", "speaker"])
    utts.to_csv(OUT_DIR / "pdch_utterances.csv", index=False)
    rep_df = pd.DataFrame(reports)
    rep_df["chunk_durations"] = rep_df["chunk_durations"].apply(
        lambda v: ";".join(f"{x:.1f}" for x in v)
    )
    rep_df.to_csv(OUT_DIR / "pdch_session_report.csv", index=False)

    print(f"\nSessions: {len(rep_df)}")
    print(f"Audio: {rep_df['audio_seconds'].sum()/3600:.1f} h")
    print(f"Turns raw {rep_df['n_turns_raw'].sum()} -> kept {rep_df['n_turns_kept'].sum()} "
          f"(dropped: {rep_df['n_turns_dropped_unknown_speaker'].sum()} unknown-speaker, "
          f"{rep_df['n_turns_dropped_non_monotonic'].sum()} non-monotonic)")
    print(f"Utterances: {len(utts)}  ({len(utts)/rep_df['n_turns_kept'].sum():.2f} per turn)")
    print(f"VAD coverage (turns containing speech): "
          f"{rep_df['n_turns_with_vad'].sum()/rep_df['n_turns_kept'].sum():.1%}")
    print(f"Wavs dropped for lack of a timestamped transcript: "
          f"{sum(len(d) for _, d in dropped)} {dropped if dropped else ''}")
    print(f"\nSaved -> {OUT_DIR / 'pdch_utterances.csv'}")
    print(f"Saved -> {OUT_DIR / 'pdch_session_report.csv'}")


if __name__ == "__main__":
    main()
