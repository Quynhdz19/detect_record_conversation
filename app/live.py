"""Realtime capture that runs the same AV-TSE + PhoWhisper path as MP4 upload."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.av_tse import SAMPLE_RATE
from app.pipeline import run_file_asr_pcm

# File upload works on a full clip. Live does the same on ~2.5s slices.
MIN_FILE_SEC = 2.2
HOP_SEC = 1.6
MAX_UTTER_SEC = 8.0
MAX_KEEP_SEC = 12.0
SILENCE_FINAL_SEC = 0.7


@dataclass
class LiveStream:
    sr: int = SAMPLE_RATE
    buf: bytearray = field(default_factory=bytearray)
    crops: list[Any] = field(default_factory=list)
    last_crop: Any = None
    last_text: str = ""
    last_decode_at: float = 0.0
    last_audio_at: float = 0.0
    busy: bool = False
    had_speech: bool = False

    def buf_sec(self) -> float:
        return len(self.buf) / (self.sr * 2)

    def push_audio(self, pcm: bytes, now: float) -> None:
        if not pcm:
            return
        self.buf.extend(pcm)
        self.last_audio_at = now
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

    def discard(self) -> None:
        self.commit()

    def trim_to(self, seconds: float) -> None:
        keep = int(self.sr * seconds) * 2
        if len(self.buf) > keep:
            self.buf[:] = self.buf[-keep:]
        if len(self.crops) > 16:
            self.crops = self.crops[-16:]

    def want_partial(self, now: float) -> bool:
        return (
            not self.busy
            and self.buf_sec() >= MIN_FILE_SEC
            and (now - self.last_decode_at) >= HOP_SEC
        )

    def want_silence_final(self, now: float) -> bool:
        return (
            not self.busy
            and self.buf_sec() >= 1.2
            and self.last_audio_at > 0
            and (now - self.last_audio_at) >= SILENCE_FINAL_SEC
        )

    def want_len_final(self) -> bool:
        return not self.busy and self.buf_sec() >= MAX_UTTER_SEC


def infer_window(pcm: bytes, crops: list, use_tse: bool = True, final: bool = False) -> tuple[str, bool]:
    """Always the file pipeline (TSE + PhoWhisper). Flags kept for call-site compat."""
    del use_tse, final
    return run_file_asr_pcm(pcm, crops)
