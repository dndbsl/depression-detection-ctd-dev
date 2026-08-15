"""Extract and cache per-utterance WavLM-large hidden states.

For every session in the manifests, this:
  1. loads the session audio once,
  2. slices each participant utterance,
  3. runs the frozen WavLM-large with ``output_hidden_states=True``,
  4. mean-pools each of the 25 hidden-state layers over time,
  5. stacks utterances into a tensor ``[num_utterances, 25, 1024]``,
  6. saves it to ``cache/{participant_id}.pt`` (resumable: existing files skipped).

The cache stores the *per-layer* pooled vectors (not a single weighted vector) so the
learnable softmax layer-weighting can be trained without ever re-running WavLM.

Usage:
    python -m features.wavlm_extractor \
        --manifest-dir manifests/ \
        --audio-root "$DAIC_WOZ_ROOT/data" \
        --cache-dir cache/
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WavLMModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    CACHE_DIR,
    DATA_ROOT,
    FEATURE_DIM,
    MANIFEST_DIR,
    NUM_LAYERS,
    SAMPLE_RATE,
    WAVLM_MODEL_NAME,
)
from data.audio_utils import load_session_audio, slice_segment  # noqa: E402


def load_manifests(manifest_dir: str) -> pd.DataFrame:
    frames = []
    for split in ("train", "dev", "test"):
        path = os.path.join(manifest_dir, f"manifest_{split}.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No manifests found in {manifest_dir}; run preprocess.py first.")
    return pd.concat(frames, ignore_index=True)


@torch.no_grad()
def extract_session(
    pid: int,
    utt_rows: pd.DataFrame,
    model: WavLMModel,
    feature_extractor,
    audio_root: str,
    device: str,
    batch_size: int,
    max_seconds: float,
) -> tuple[torch.Tensor, dict]:
    """Return ([num_utt, 25, 1024] float16 tensor, stats) for one session."""
    samples, _ = load_session_audio(pid, audio_root)
    utt_rows = utt_rows.sort_values("utterance_index")
    max_samples = int(max_seconds * SAMPLE_RATE)

    segments: list[np.ndarray] = []
    n_truncated = 0
    for _, r in utt_rows.iterrows():
        seg = slice_segment(samples, float(r["start_time"]), float(r["stop_time"]))
        if len(seg) == 0:
            # Should not happen given preprocessing, but guard against empty slices.
            seg = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)
        if len(seg) > max_samples:
            seg = seg[:max_samples]
            n_truncated += 1
        segments.append(seg)

    pooled_per_utt: list[torch.Tensor] = []
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        inputs = feature_extractor(
            batch,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_values = inputs["input_values"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            out = model(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        # tuple of NUM_LAYERS tensors, each [B, T', D]
        hidden = torch.stack(out.hidden_states, dim=1)  # [B, 25, T', D]

        # Valid output-frame counts per utterance for masked mean-pool over time.
        out_lens = model._get_feat_extract_output_lengths(
            attention_mask.sum(-1)).to(torch.long)  # [B]
        T = hidden.shape[2]
        frame_idx = torch.arange(T, device=device).unsqueeze(0)        # [1, T']
        time_mask = (frame_idx < out_lens.unsqueeze(1)).float()        # [B, T']
        time_mask = time_mask.unsqueeze(1).unsqueeze(-1)               # [B, 1, T', 1]

        summed = (hidden * time_mask).sum(dim=2)                       # [B, 25, D]
        denom = time_mask.sum(dim=2).clamp(min=1.0)                    # [B, 1, 1]
        pooled = summed / denom                                        # [B, 25, D]
        pooled_per_utt.append(pooled.float().cpu())

    feats = torch.cat(pooled_per_utt, dim=0)                           # [num_utt, 25, D]
    assert feats.shape[1] == NUM_LAYERS and feats.shape[2] == FEATURE_DIM, feats.shape
    stats = {"participant_id": pid, "num_utt": feats.shape[0], "n_truncated": n_truncated}
    return feats.to(torch.float16), stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-dir", default=MANIFEST_DIR)
    ap.add_argument("--audio-root", default=DATA_ROOT)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seconds", type=float, default=60.0,
                    help="Truncate any single utterance longer than this (OOM guard).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit-sessions", type=int, default=0,
                    help="If >0, only process the first N sessions (smoke test).")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    manifest = load_manifests(args.manifest_dir)
    pids = sorted(manifest["participant_id"].unique())
    if args.limit_sessions > 0:
        pids = pids[:args.limit_sessions]

    print(f"[extract] loading {WAVLM_MODEL_NAME} on {args.device} ...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(WAVLM_MODEL_NAME)
    model = WavLMModel.from_pretrained(WAVLM_MODEL_NAME)
    model.eval().to(args.device)
    for p in model.parameters():
        p.requires_grad_(False)

    all_stats = []
    for pid in tqdm(pids, desc="sessions"):
        out_path = os.path.join(args.cache_dir, f"{pid}.pt")
        if os.path.exists(out_path):
            continue
        utt_rows = manifest[manifest["participant_id"] == pid]
        try:
            feats, stats = extract_session(
                pid, utt_rows, model, feature_extractor,
                args.audio_root, args.device, args.batch_size, args.max_seconds)
        except Exception as e:  # noqa: BLE001
            print(f"[extract] session {pid} FAILED: {e}")
            all_stats.append({"participant_id": pid, "num_utt": -1, "error": str(e)})
            continue
        torch.save({"features": feats, "num_utt": int(stats["num_utt"]),
                    "participant_id": int(pid)}, out_path)
        all_stats.append(stats)
        tqdm.write(
            f"[extract] {pid}: {stats['num_utt']} utt -> {tuple(feats.shape)} "
            f"(truncated={stats['n_truncated']})")

    stats_df = pd.DataFrame(all_stats)
    stats_path = os.path.join(args.cache_dir, "extract_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"[extract] done. stats -> {stats_path}")


if __name__ == "__main__":
    main()
