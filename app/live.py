"""Realtime capture: TalkNet/VAD gate, then AV-TSE + PhoWhisper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.av_tse import SAMPLE_RATE
from app.pipeline import run_av_asr

# File upload works on a full clip. Live does the same on ~2.5s slices
# only after there is actual voice, not continuous mic silence.
MIN_FILE_SEC = 2.2
HOP_SEC = 1.6
MAX_UTTER_SEC = 8.0
MAX_KEEP_SEC = 12.0
SILENCE_FINAL_SEC = 0.7
PRE_ROLL_SEC = 0.45
VOICE_PEAK = 0.02


@dataclass
class LiveStream:
    sr: int = SAMPLE_RATE
    buf: bytearray = field(default_factory=bytearray)
    crops: list[Any] = field(default_factory=list)
    last_crop: Any = None
    last_text: str = ""
    last_decode_at: float = 0.0
    last_audio_at: float = 0.0
    last_voice_at: float = 0.0
    last_silence_at: float = 0.0
    busy: bool = False
    had_speech: bool = False

    def buf_sec(self) -> float:
        return len(self.buf) / (self.sr * 2)

    def push_audio(self, pcm: bytes, now: float, *, voiced: bool = True) -> None:
        if not pcm:
            return
        self.last_audio_at = now
        if voiced:
            self.last_voice_at = now
            self.buf.extend(pcm)
        elif self.last_voice_at > 0:
            self.buf.extend(pcm)
        else:
            self.buf.extend(pcm)
            self.trim_to(PRE_ROLL_SEC)
        max_b = int(self.sr * MAX_KEEP_SEC) * 2
        if len(self.buf) > max_b:
            del self.buf[: len(self.buf) - max_b]

    def push_crop(self, crop: Any) -> None:
        if crop is not None:
            self.last_crop = crop
            self.crops.append(crop)
        elif self.last_crop is not None:
            self.crops.append(self.last_crop)
        if len(self.crops) > 96:
            self.crops = self.crops[-96:]

    def snapshot(self) -> tuple[bytes, list[Any]]:
        return bytes(self.buf), list(self.crops)

    def commit(self) -> None:
        self.buf.clear()
        self.crops.clear()
        self.had_speech = False
        self.last_text = ""
        self.last_audio_at = 0.0
        self.last_voice_at = 0.0

    def discard(self) -> None:
        self.commit()

    def trim_to(self, seconds: float) -> None:
        keep = int(self.sr * seconds) * 2
        if len(self.buf) > keep:
            self.buf[:] = self.buf[-keep:]
        if len(self.crops) > 16:
            self.crops = self.crops[-16:]

    def idle_silence(self) -> bool:
        return self.last_voice_at <= 0 and not self.busy

    def should_emit_silence(self, now: float, interval: float = 2.0) -> bool:
        if now - self.last_silence_at < interval:
            return False
        self.last_silence_at = now
        return True

    def want_partial(self, now: float) -> bool:
        return (
            not self.busy
            and self.last_voice_at > 0
            and self.buf_sec() >= MIN_FILE_SEC
            and (now - self.last_decode_at) >= HOP_SEC
        )

    def want_silence_final(self, now: float) -> bool:
        return (
            not self.busy
            and self.last_voice_at > 0
            and self.buf_sec() >= 1.2
            and (now - self.last_voice_at) >= SILENCE_FINAL_SEC
        )

    def want_len_final(self) -> bool:
        return not self.busy and self.last_voice_at > 0 and self.buf_sec() >= MAX_UTTER_SEC


def chunk_is_voiced(pcm: bytes, peak_min: float = VOICE_PEAK) -> bool:
    if len(pcm) < 64:
        return False
    peak = float(np.max(np.abs(np.frombuffer(pcm, dtype=np.int16)))) / 32768.0
    return peak >= peak_min


def speaking_blocked(require_speaking: bool, face: dict) -> bool:
    """Face visible and not talking → do not send this audio to ASR."""
    return bool(require_speaking) and bool(face.get("found")) and not bool(face.get("speaking"))


def audio_is_voiced(
    pcm: bytes,
    *,
    speaking: bool,
    require_speaking: bool,
    face_found: bool,
) -> bool:
    if require_speaking:
        return bool(face_found and speaking and chunk_is_voiced(pcm))
    return chunk_is_voiced(pcm)


def infer_window(
    pcm: bytes,
    crops: list,
    use_tse: bool = True,
    final: bool = False,
    allow_greetings: bool = False,
) -> tuple[str, bool]:
    return run_av_asr(
        pcm,
        crops,
        use_tse=use_tse,
        final=final,
        allow_greetings=allow_greetings,
    )
