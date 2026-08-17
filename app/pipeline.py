"""Shared AV-TSE + PhoWhisper inference."""

from __future__ import annotations

import logging

import numpy as np

from app.asr import looks_like_speech, pcm16_to_float32, speech_stats, transcribe_pcm16
from app.av_tse import SAMPLE_RATE, get_av_tse

logger = logging.getLogger(__name__)


def _to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def run_av_asr(
    pcm_bytes: bytes, face_crops: list, use_tse: bool = True, final: bool = False
) -> tuple[str, bool]:
    """Extract target voice (optional) then transcribe. Returns (text, used_tse)."""
    raw = pcm16_to_float32(pcm_bytes)
    if not looks_like_speech(raw, SAMPLE_RATE):
        logger.info("skip ASR: raw not speech %s", speech_stats(raw, SAMPLE_RATE))
        return "", False

    audio = raw
    used_tse = False
    if use_tse and face_crops:
        try:
            extracted = get_av_tse().extract(raw, face_crops)
            if looks_like_speech(extracted, SAMPLE_RATE):
                audio = extracted
                used_tse = True
            else:
                logger.info(
                    "AV-TSE output failed VAD %s; using raw mic",
                    speech_stats(extracted, SAMPLE_RATE),
                )
        except Exception:
            logger.exception("AV-TSE failed; using raw mic")

    text = transcribe_pcm16(_to_pcm16(audio), sample_rate=SAMPLE_RATE, final=final)
    if not text and used_tse:
        logger.info("ASR empty after TSE; retry raw mic")
        text = transcribe_pcm16(_to_pcm16(raw), sample_rate=SAMPLE_RATE, final=final)
        used_tse = False
    if not text:
        logger.info("ASR empty after speech-like audio stats=%s", speech_stats(raw, SAMPLE_RATE))
    return text, used_tse
