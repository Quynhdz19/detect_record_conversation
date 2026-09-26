"""Realtime capture: TalkNet/VAD gate, then AV-TSE + PhoWhisper."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.av_tse import SAMPLE_RATE
from app.pipeline import run_av_asr

# whisper_streaming keeps one continuous utterance (min chunk ~1s) and
# endpoints on silence. faster-whisper pads pauses so Whisper is not fed
# audio with the gaps between words cut out.
MIN_FILE_SEC = 1.1
HOP_SEC = 0.9
MAX_UTTER_SEC = 12.0
MAX_KEEP_SEC = 14.0
SILENCE_FINAL_SEC = 0.7
PRE_ROLL_SEC = 0.6
VOICE_PEAK = 0.035
TURN_ENERGY = 0.018
MIN_VOICED_SEC = 0.12
COMMIT_OVERLAP_SEC = 0.6


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
    face_miss: int = 0
    voiced_sec: float = 0.0
    voiced_run_sec: float = 0.0
    saw_target: bool = False
    in_turn: bool = False
    speaker: str = ""
    pending_speaker: str = ""
    voiceprints: dict = field(default_factory=dict)
    enroll_buf: dict = field(default_factory=dict)
    enroll_new: dict = field(default_factory=dict)
    enroll_busy: bool = False

    def push_enroll(self, label: str, pcm: bytes) -> bytes | None:
        """Collect a clean clip of one person. Returns audio once there is enough to embed."""
        if not label or not pcm:
            return None
        buf = self.enroll_buf.setdefault(label, bytearray())
        buf.extend(pcm)
        max_b = self.sr * 4 * 2
        if len(buf) > max_b:
            del buf[: len(buf) - max_b]
        self.enroll_new[label] = self.enroll_new.get(label, 0.0) + len(pcm) / (self.sr * 2)
        if self.enroll_busy or self.enroll_new[label] < 1.2 or len(buf) < self.sr * 2:
            return None
        self.enroll_new[label] = 0.0
        self.enroll_busy = True
        return bytes(buf[-(self.sr * 2 * 2) :])

    def take_turn_for(self, label: str) -> tuple[str, bytes, list[Any]] | None:
        """Switch A/B. Returns the previous speaker's audio if a turn must be closed."""
        if not label:
            return None
        if self.busy:
            self.pending_speaker = label
            return None
        if not self.speaker:
            self.speaker = label
            return None
        if label == self.speaker:
            return None
        prev = self.speaker
        if self.in_turn and self.last_voice_at > 0 and self.buf_sec() >= 0.8:
            pcm, crops = self.snapshot()
            self.commit(keep_sec=0.0)
            self.speaker = label
            return prev, pcm, crops
        self.speaker = label
        return None

    def buf_sec(self) -> float:
        return len(self.buf) / (self.sr * 2)

    def push_audio(
        self,
        pcm: bytes,
        now: float,
        *,
        energy: bool = False,
        target: bool = False,
    ) -> None:
        """Once the opposite face starts a turn, keep the waveform continuous.

        Cutting each inter-word gap (old lip/TalkNet gate) is what made
        normal-speed speech miss words. Endpointing waits for real silence.
        """
        if not pcm:
            return
        self.last_audio_at = now
        chunk_sec = len(pcm) / (self.sr * 2)
        if target:
            self.saw_target = True
            self.in_turn = True
        if not self.in_turn:
            self.buf.extend(pcm)
            self.trim_to(PRE_ROLL_SEC)
            return
        self.buf.extend(pcm)
        if energy:
            self.last_voice_at = now
            self.voiced_sec += chunk_sec
            self.voiced_run_sec = min(2.0, self.voiced_run_sec + chunk_sec)
        else:
            self.voiced_run_sec = max(0.0, self.voiced_run_sec - chunk_sec)
        max_b = int(self.sr * MAX_KEEP_SEC) * 2
        if len(self.buf) > max_b:
            del self.buf[: len(self.buf) - max_b]

    def note_face(self, crop: Any, *, found: bool) -> None:
        """Keep TSE crops only while a face is visible. Lost face → drop leftover audio."""
        if found:
            self.face_miss = 0
            self.push_crop(crop)
            return
        self.face_miss += 1
        self.last_crop = None
        if self.face_miss >= 15 and not self.busy:
            self.discard()

    def push_crop(self, crop: Any) -> None:
        if crop is not None:
            self.last_crop = crop
            self.crops.append(crop)
        if len(self.crops) > 90:
            self.crops = self.crops[-90:]

    def snapshot(self) -> tuple[bytes, list[Any]]:
        return bytes(self.buf), list(self.crops)

    def commit(self, keep_sec: float = COMMIT_OVERLAP_SEC) -> None:
        """End an utterance but keep a tail so words said during ASR are not wiped."""
        keep_b = int(self.sr * max(0.0, keep_sec)) * 2
        tail = bytes(self.buf[-keep_b:]) if keep_b and self.buf else b""
        tail_crops = list(self.crops[-25:]) if keep_b and self.crops else []
        last_v = self.last_voice_at
        last_crop = self.last_crop
        self.buf.clear()
        self.crops.clear()
        self.had_speech = False
        self.last_text = ""
        self.last_audio_at = 0.0
        self.face_miss = 0
        self.voiced_run_sec = 0.0
        self.saw_target = False
        self.in_turn = False
        if tail:
            self.buf.extend(tail)
            self.crops = tail_crops
            self.last_crop = last_crop
            self.voiced_sec = min(self.voiced_sec, keep_sec)
            if last_v > 0 and (time.time() - last_v) < SILENCE_FINAL_SEC:
                self.last_voice_at = last_v
                self.in_turn = True
                self.saw_target = True
            else:
                self.last_voice_at = 0.0
        else:
            self.last_voice_at = 0.0
            self.last_crop = None
            self.voiced_sec = 0.0

    def discard(self) -> None:
        self.commit(keep_sec=0.0)

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
            and self.voiced_sec >= MIN_VOICED_SEC
            and self.buf_sec() >= MIN_FILE_SEC
            and (now - self.last_decode_at) >= HOP_SEC
        )

    def want_silence_final(self, now: float) -> bool:
        return (
            not self.busy
            and self.last_voice_at > 0
            and self.voiced_sec >= MIN_VOICED_SEC
            and self.buf_sec() >= 0.8
            and (now - self.last_voice_at) >= SILENCE_FINAL_SEC
            and (now - self.last_decode_at) >= 0.45
        )

    def want_len_final(self) -> bool:
        return (
            not self.busy
            and self.last_voice_at > 0
            and self.voiced_sec >= MIN_VOICED_SEC
            and self.buf_sec() >= MAX_UTTER_SEC
        )


def chunk_is_voiced(pcm: bytes, peak_min: float = VOICE_PEAK) -> bool:
    if len(pcm) < 64:
        return False
    peak = float(np.max(np.abs(np.frombuffer(pcm, dtype=np.int16)))) / 32768.0
    return peak >= peak_min


def speaking_blocked(require_speaking: bool, face: dict) -> bool:
    """No face, or face not talking → do not send nearby audio to ASR."""
    if not face.get("found"):
        return True
    if require_speaking and not face.get("speaking"):
        return True
    return False


def audio_is_voiced(
    pcm: bytes,
    *,
    speaking: bool,
    require_speaking: bool,
    face_found: bool,
    now: float = 0.0,
    last_voice_at: float = 0.0,
) -> bool:
    if not face_found:
        return False
    energy = chunk_is_voiced(pcm)
    if not energy:
        return False
    if not require_speaking:
        return True
    if speaking:
        return True
    # Do not treat leftover room energy as this face. Hangover only
    # appends a short tail in push_audio without refreshing last_voice_at.
    return False


def infer_window(
    pcm: bytes,
    crops: list,
    use_tse: bool = True,
    final: bool = False,
    allow_greetings: bool = False,
    fallback_raw: bool = False,
    separate_voices: bool = False,
) -> tuple[str, bool]:
    return run_av_asr(
        pcm,
        crops,
        use_tse=use_tse,
        final=final,
        allow_greetings=allow_greetings,
        fallback_raw=fallback_raw,
        separate_voices=separate_voices,
    )
