"""Audio loading and time-span slicing for DAIC-WOZ sessions.

Audio is already 16 kHz mono PCM16, so no resampling or channel mixing is needed.
We read slices directly with ``soundfile`` (frame-accurate, avoids loading the full
file repeatedly when possible).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_ROOT, SAMPLE_RATE  # noqa: E402


def audio_path(participant_id: int, data_root: str = DATA_ROOT) -> str:
    return os.path.join(data_root, f"{participant_id}_P", f"{participant_id}_AUDIO.wav")


def slice_segment(
    samples: np.ndarray,
    start_time: float,
    stop_time: float,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Return the [start_time, stop_time] slice of an already-loaded waveform."""
    start_idx = max(0, int(round(start_time * sample_rate)))
    stop_idx = min(len(samples), int(round(stop_time * sample_rate)))
    if stop_idx <= start_idx:
        return np.zeros(0, dtype=np.float32)
    return samples[start_idx:stop_idx]


def load_session_audio(participant_id: int, data_root: str = DATA_ROOT) -> tuple[np.ndarray, int]:
    """Load the full session waveform as float32 in [-1, 1]."""
    path = audio_path(participant_id, data_root)
    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Session {participant_id} sample rate {sr} != expected {SAMPLE_RATE}")
    return samples, sr
