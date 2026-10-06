"""One mic, Vietnamese script.

GTCRN strips non-speech noise. Silero finds the utterance. Zipformer reads
that denoised audio directly. MossFormer2 is not on this path: it is trained
on English mixtures and splits one Vietnamese voice into two damaged tracks,
so the transcript no longer matches what was said.

A second person is labeled only when a later utterance has a clearly
different voice. One utterance stays one line.
"""

from __future__ import annotations

import logging

import numpy as np

from app.asr import _zipformer_text
from app.denoise import StreamDenoiser
from app.voice import blend, embed_f32

logger = logging.getLogger(__name__)

SR = 16000
PARTIAL_SEC = 0.4
MIN_UTT_SEC = 0.35
NEW_VOICE = 0.38


class DuoMic:
    def __init__(self) -> None:
        import sherpa_onnx

        from app.asr import _ensure_silero_vad

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(_ensure_silero_vad())
        config.silero_vad.threshold = 0.5
        config.silero_vad.min_silence_duration = 0.5
        config.silero_vad.min_speech_duration = 0.25
        config.silero_vad.max_speech_duration = 15
        config.sample_rate = SR
        config.provider = "cpu"
        self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
        self._window = 512
        self._denoiser = StreamDenoiser()
        self._pending = np.zeros(0, dtype=np.float32)
        self._utt = np.zeros(0, dtype=np.float32)
        self._since = 0
        self.prints: dict[str, np.ndarray] = {}

    def accept(self, audio: np.ndarray) -> list[tuple[str, bool, str]]:
        clean = self._denoiser.accept(audio)
        if clean.size == 0:
            return []
        self._pending = np.concatenate([self._pending, clean])
        events: list[tuple[str, bool, str]] = []
        while self._pending.size >= self._window:
            frame = np.ascontiguousarray(self._pending[: self._window])
            self._pending = self._pending[self._window :]
            self._vad.accept_waveform(frame)
            if self._vad.is_speech_detected():
                self._utt = np.concatenate([self._utt, frame])
                self._since += frame.size
                if self._since >= int(SR * PARTIAL_SEC) and self._utt.size >= int(SR * MIN_UTT_SEC):
                    events.extend(self._emit(self._utt, final=False))
                    self._since = 0
            while not self._vad.empty():
                segment = np.array(self._vad.front.samples, dtype=np.float32)
                self._vad.pop()
                if segment.size >= int(SR * MIN_UTT_SEC):
                    events.extend(self._emit(segment, final=True))
                self._reset_utt()
        return events

    def flush(self) -> list[tuple[str, bool, str]]:
        if self._utt.size < int(SR * MIN_UTT_SEC):
            self._reset_utt()
            return []
        events = self._emit(self._utt, final=True)
        self._reset_utt()
        return events

    def _reset_utt(self) -> None:
        self._utt = np.zeros(0, dtype=np.float32)
        self._since = 0

    def _emit(self, audio: np.ndarray, final: bool) -> list[tuple[str, bool, str]]:
        text = _zipformer_text(audio, SR)
        if not text:
            return []
        vec = embed_f32(audio) if audio.size >= int(SR * 0.8) else None
        if final:
            label = _assign(vec, self.prints)
            if vec is not None:
                self.prints[label] = blend(self.prints.get(label), vec)
        else:
            label = _match(vec, self.prints)
        logger.info("script %s final=%s %r", label, final, text)
        return [(text, final, label)]


def _scores(vec: np.ndarray, prints: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(np.dot(vec, emb)) for name, emb in prints.items()}


def _match(vec: np.ndarray | None, prints: dict[str, np.ndarray]) -> str:
    if not prints or vec is None:
        return "A"
    scores = _scores(vec, prints)
    return max(scores, key=scores.get)


def _assign(vec: np.ndarray | None, prints: dict[str, np.ndarray]) -> str:
    """Keep the same person on A. Open B only when this utterance is another voice."""
    if not prints or vec is None:
        return "A"
    scores = _scores(vec, prints)
    best = max(scores, key=scores.get)
    if scores[best] >= NEW_VOICE or len(prints) >= 2:
        return best
    return "B" if "B" not in prints else best
