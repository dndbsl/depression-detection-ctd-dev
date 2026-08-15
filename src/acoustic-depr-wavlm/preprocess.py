"""Transcript tokenisation + cleaning -> per-split manifest CSVs.

This is the first runnable step. It does NOT touch the audio waveform (no denoising,
VAD, or normalisation) -- it only segments and filters the transcripts and writes the
manifests that every later step consumes.

Usage:
    python preprocess.py \
        --data-root "$DAIC_WOZ_ROOT/data" \
        --labels-root "$DAIC_WOZ_ROOT/labels" \
        --out-dir manifests/
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from config import DATA_ROOT, LABELS_ROOT, MANIFEST_DIR
from data.qa_pairing import build_response_turns
from data.splits import load_split_labels
from data.transcript import parse_transcript

MANIFEST_COLUMNS = [
    "participant_id",
    "utterance_index",
    "start_time",
    "stop_time",
    "duration",
    "split",
    "PHQ8_Binary",
    "PHQ8_Score",
]


def build_split_manifest(split: str, data_root: str, labels_root: str,
                         segmentation: str = "utterance"):
    """Return (manifest_df, dropped_sessions, per_session_stats) for one split.

    segmentation:
      * "utterance" -> one row per participant transcript turn (default),
      * "qa"        -> one row per CTD-style Ask/Response participant response turn.
    """
    labels = load_split_labels(split, labels_root)
    rows: list[dict] = []
    dropped_sessions: list[int] = []
    per_session_stats: list[dict] = []

    for _, lab in labels.iterrows():
        pid = int(lab["Participant_ID"])
        if segmentation == "qa":
            utts, stats = build_response_turns(pid, data_root)
        else:
            utts, stats = parse_transcript(pid, data_root)
        stats["split"] = split
        per_session_stats.append(stats)

        if stats.get("error") == "transcript_missing":
            print(f"[preprocess] {split}: session {pid} transcript missing -> excluded")
            dropped_sessions.append(pid)
            continue
        if len(utts) == 0:
            print(f"[preprocess] {split}: session {pid} has 0 valid utterances -> excluded")
            dropped_sessions.append(pid)
            continue

        for u in utts:
            rows.append({
                "participant_id": u.participant_id,
                "utterance_index": u.utterance_index,
                "start_time": u.start_time,
                "stop_time": u.stop_time,
                "duration": u.duration,
                "split": split,
                "PHQ8_Binary": int(lab["PHQ8_Binary"]),
                "PHQ8_Score": int(lab["PHQ8_Score"]),
            })

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    return manifest, dropped_sessions, per_session_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--labels-root", default=LABELS_ROOT)
    ap.add_argument("--out-dir", default=MANIFEST_DIR)
    ap.add_argument("--segmentation", default="utterance", choices=["utterance", "qa"],
                    help="'qa' = CTD-style Ask/Response participant-response turns.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_stats: list[dict] = []
    summary_rows: list[dict] = []

    print(f"[preprocess] segmentation mode: {args.segmentation}")
    for split in ("train", "dev", "test"):
        manifest, dropped, stats = build_split_manifest(
            split, args.data_root, args.labels_root, segmentation=args.segmentation)
        out_path = os.path.join(args.out_dir, f"manifest_{split}.csv")
        manifest.to_csv(out_path, index=False)
        all_stats.extend(stats)

        n_sessions = manifest["participant_id"].nunique()
        n_utts = len(manifest)
        n_pos = manifest.drop_duplicates("participant_id")["PHQ8_Binary"].sum() if n_utts else 0
        print(
            f"[preprocess] {split}: wrote {out_path} | "
            f"{n_sessions} sessions, {n_utts} utterances "
            f"(depressed sessions={int(n_pos)}), dropped sessions={dropped}")
        if n_utts:
            print(
                f"             duration sec: min={manifest['duration'].min():.3f} "
                f"mean={manifest['duration'].mean():.3f} max={manifest['duration'].max():.3f}")
        summary_rows.append({
            "split": split,
            "sessions": n_sessions,
            "utterances": n_utts,
            "depressed_sessions": int(n_pos),
            "dropped_sessions": ";".join(map(str, dropped)) if dropped else "",
        })

    # Per-session drop log for transparency.
    stats_df = pd.DataFrame(all_stats)
    stats_path = os.path.join(args.out_dir, "preprocess_session_stats.csv")
    stats_df.to_csv(stats_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, "preprocess_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[preprocess] wrote per-session stats -> {stats_path}")
    print(f"[preprocess] wrote summary -> {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
