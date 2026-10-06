"""Split one 16 kHz mic into two waveforms with MossFormer2_SS_16K.

ClearVoice always returns two tracks and rescales each to the mixture
loudness, so a second track can be a copy of a single speaker. Callers
compare speaker embeddings before opening a second person.
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_separator():
    os.chdir(PROJECT_ROOT)
    from clearvoice import ClearVoice

    logger.info("Loading MossFormer2_SS_16K…")
    model = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])
    logger.info("MossFormer2 separator ready")
    return model


def separate_two(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two waveforms the same length as `audio`."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = int(audio.size)
    if n < 1600:
        return audio, np.zeros(n, dtype=np.float32)
    with _LOCK:
        raw = get_separator()(audio[np.newaxis, :])
    tracks = _as_tracks(raw, n)
    return tracks[0], tracks[1]


def _as_tracks(raw, n: int) -> list[np.ndarray]:
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim >= 2 and arr.shape[0] == 2:
            items = [arr[0], arr[1]]
        elif arr.ndim >= 2 and arr.shape[1] == 2:
            items = [arr[:, 0], arr[:, 1]]
        else:
            items = [arr]
    tracks: list[np.ndarray] = []
    for item in items[:2]:
        wave = np.asarray(item, dtype=np.float32).reshape(-1)
        if wave.size < n:
            wave = np.pad(wave, (0, n - wave.size))
        tracks.append(wave[:n])
    while len(tracks) < 2:
        tracks.append(np.zeros(n, dtype=np.float32))
    return tracks
