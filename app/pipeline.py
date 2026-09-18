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
    pcm_bytes: bytes,
    face_crops: list,
    use_tse: bool = True,
    final: bool = False,
    allow_greetings: bool | None = None,
    fallback_raw: bool = False,
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
                    "AV-TSE output failed VAD %s%s",
                    speech_stats(extracted, SAMPLE_RATE),
                    "; using raw mic" if fallback_raw else "; skip (not this speaker)",
                )
                if not fallback_raw:
                    return "", False
        except Exception:
            logger.exception("AV-TSE failed%s", "; using raw mic" if fallback_raw else "")
            if not fallback_raw:
                return "", False
    elif use_tse and not fallback_raw:
        logger.info("skip ASR: no face crops for target speaker")
        return "", False

    text = transcribe_pcm16(
        _to_pcm16(audio),
        sample_rate=SAMPLE_RATE,
        final=final,
        allow_greetings=allow_greetings,
    )
    if not text and used_tse and fallback_raw:
        logger.info("ASR empty after TSE; retry raw mic")
        text = transcribe_pcm16(
            _to_pcm16(raw),
            sample_rate=SAMPLE_RATE,
            final=final,
            allow_greetings=allow_greetings,
        )
        used_tse = False
    if not text:
        logger.info("ASR empty after speech-like audio stats=%s", speech_stats(raw, SAMPLE_RATE))
    return text, used_tse


def run_file_asr(audio_f32: np.ndarray, face_crops: list) -> tuple[str, bool, np.ndarray]:
    """MP4 / one-shot clip: AV-TSE then PhoWhisper. Still skip obvious silence."""
    audio = np.clip(np.asarray(audio_f32, dtype=np.float32).reshape(-1), -1.0, 1.0)
    if audio.size < SAMPLE_RATE // 4:
        return "", False, audio
    used_tse = False
    if face_crops:
        try:
            extracted = get_av_tse().extract(audio, face_crops)
            extracted = np.clip(np.asarray(extracted, dtype=np.float32).reshape(-1), -1.0, 1.0)
            if looks_like_speech(extracted, SAMPLE_RATE):
                audio = extracted
                used_tse = True
            elif not looks_like_speech(audio, SAMPLE_RATE):
                logger.info(
                    "skip file ASR: not speech raw=%s tse=%s",
                    speech_stats(audio, SAMPLE_RATE),
                    speech_stats(extracted, SAMPLE_RATE),
                )
                return "", False, audio
            else:
                logger.info(
                    "AV-TSE output failed VAD %s; using raw audio",
                    speech_stats(extracted, SAMPLE_RATE),
                )
        except Exception:
            logger.exception("AV-TSE failed; using raw audio")
    elif not looks_like_speech(audio, SAMPLE_RATE):
        logger.info("skip file ASR: raw not speech %s", speech_stats(audio, SAMPLE_RATE))
        return "", False, audio
    pcm = (audio * 32767.0).astype(np.int16).tobytes()
    text = transcribe_pcm16(pcm, sample_rate=SAMPLE_RATE, final=True, allow_greetings=True)
    return text, used_tse, audio


def run_file_asr_pcm(pcm_bytes: bytes, face_crops: list) -> tuple[str, bool]:
    text, used_tse, _ = run_file_asr(pcm16_to_float32(pcm_bytes), face_crops)
    return text, used_tse
