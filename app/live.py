"""Sliding-window live ASR: partials while speaking, finalize on pause."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.asr import looks_like_speech, pcm16_to_float32
from app.av_tse import SAMPLE_RATE
from app.pipeline import run_av_asr

MIN_PARTIAL_SEC = 1.0
HOP_SEC = 0.55
MAX_UTTER_SEC = 6.0
MAX_KEEP_SEC = 8.0
SILENCE_FINAL_SEC = 0.65


@dataclass
class LiveStream:
    sr: int = SAMPLE_RATE
    buf: bytearray = field(default_factory=bytearray)
    crops: list[Any] = field(default_factory=list)
    last_text: str = ""
    last_decode_at: float = 0.0
    last_audio_at: float = 0.0
    busy: bool = False
    had_speech: bool = False

    def buf_sec(self) -> float:
        return len(self.buf) / (self.sr * 2)

    def push_audio(self, pcm: bytes, now: float) -> None:
        audio = pcm16_to_float32(pcm)
        if audio.size == 0 or float(np.max(np.abs(audio))) < 0.025:
            return
        self.buf.extend(pcm)
        self.last_audio_at = now
        max_b = int(self.sr * MAX_KEEP_SEC) * 2
        if len(self.buf) > max_b:
            del self.buf[: len(self.buf) - max_b]

    def push_crop(self, crop: Any) -> None:
        if crop is None:
            return
        self.crops.append(crop)
        if len(self.crops) > 80:
            self.crops = self.crops[-80:]

    def snapshot(self) -> tuple[bytes, list[Any]]:
        return bytes(self.buf), list(self.crops)

    def commit(self) -> None:
        self.buf.clear()
        self.crops.clear()
        self.had_speech = False
        self.last_text = ""
        self.last_audio_at = 0.0

    def discard(self) -> None:
        """Drop buffered noise without emitting text."""
        self.commit()

    def want_partial(self, now: float) -> bool:
        return (
            not self.busy
            and self.buf_sec() >= MIN_PARTIAL_SEC
            and (now - self.last_decode_at) >= HOP_SEC
        )

    def want_silence_final(self, now: float) -> bool:
        return (
            not self.busy
            and self.buf_sec() >= 0.7
            and self.last_audio_at > 0
            and (now - self.last_audio_at) >= SILENCE_FINAL_SEC
        )

    def want_len_final(self) -> bool:
        return not self.busy and self.buf_sec() >= MAX_UTTER_SEC


def infer_window(pcm: bytes, crops: list, use_tse: bool, final: bool = False) -> tuple[str, bool]:
    audio = pcm16_to_float32(pcm)
    if not looks_like_speech(audio, SAMPLE_RATE):
        return "", False
    return run_av_asr(pcm, crops, use_tse=use_tse, final=final)
