"""Shared AV-TSE + PhoWhisper inference."""

from __future__ import annotations

import logging

import numpy as np

from app.asr import clean_transcript, looks_like_speech, pcm16_to_float32, transcribe_pcm16
from app.av_tse import SAMPLE_RATE, get_av_tse

logger = logging.getLogger(__name__)


def run_av_asr(pcm_bytes: bytes, face_crops: list, use_tse: bool = True) -> tuple[str, bool]:
    """Extract target voice (optional) then transcribe. Returns (text, used_tse)."""
    audio = pcm16_to_float32(pcm_bytes)
    if not looks_like_speech(audio, SAMPLE_RATE):
        return "", False

    used_tse = False
    if use_tse and face_crops:
        try:
            extracted = get_av_tse().extract(audio, face_crops)
            audio = extracted
            used_tse = True
        except Exception:
            logger.exception("AV-TSE failed; using raw mic")

    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    if not looks_like_speech(audio, SAMPLE_RATE):
        return "", used_tse

    pcm = (audio * 32767.0).astype(np.int16).tobytes()
    text = clean_transcript(transcribe_pcm16(pcm, sample_rate=SAMPLE_RATE))
    return text, used_tse
