"""One mic, Vietnamese script for two people taking turns.

Silero cuts each utterance. Zipformer reads the raw mic audio so tones and
consonants stay intact. A voiceprint is taken once the utterance is long
enough, then that label is locked until the person stops. A close call keeps
the previous person instead of flipping A and B.
"""

from __future__ import annotations

import logging

import numpy as np

from app.asr import _zipformer_text
from app.voice import blend, embed_f32

logger = logging.getLogger(__name__)

SR = 16000
PARTIAL_SEC = 0.4
MIN_UTT_SEC = 0.35
# Below this, a long clip is a new person. Above it, stay with the known voice.
NEW_VOICE = 0.30
# Two enrolled voices must be this far apart before the label is allowed to change.
SWITCH_MARGIN = 0.06
# Only fold a clip into a voiceprint when it clearly is that person.
ENROLL_SCORE = 0.50


class DuoMic:
    def __init__(self) -> None:
        import sherpa_onnx

        from app.asr import _ensure_silero_vad

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(_ensure_silero_vad())
        config.silero_vad.threshold = 0.5
        config.silero_vad.min_silence_duration = 0.7
        config.silero_vad.min_speech_duration = 0.25
        config.silero_vad.max_speech_duration = 15
        config.sample_rate = SR
        config.provider = "cpu"
        self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
        self._window = 512
        self._pending = np.zeros(0, dtype=np.float32)
        self._utt = np.zeros(0, dtype=np.float32)
        self._since = 0
        self._utt_label = ""
        self._shown = ""
        self._last = "A"
        self.prints: dict[str, np.ndarray] = {}

    def accept(self, audio: np.ndarray) -> list[tuple[str, bool, str]]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return []
        self._pending = np.concatenate([self._pending, audio])
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
        self._utt_label = ""
        self._shown = ""

    def _emit(self, audio: np.ndarray, final: bool) -> list[tuple[str, bool, str]]:
        text = _zipformer_text(audio, SR)
        if not text and not final:
            return []
        vec = embed_f32(audio) if audio.size >= int(SR * 1.0) else None
        if self._utt_label:
            label = self._utt_label
        else:
            label = _decide(vec, self.prints, self._last, audio.size)
            if vec is not None:
                self._utt_label = label
        if final and vec is not None:
            _enroll(label, vec, self.prints)
            self._last = label
        events: list[tuple[str, bool, str]] = []
        if self._shown and self._shown != label:
            events.append(("", True, self._shown))
        self._shown = label
        if text:
            events.append((text, final, label))
            logger.info("script %s final=%s %r", label, final, text)
        return events


def _scores(vec: np.ndarray, prints: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(np.dot(vec, emb)) for name, emb in prints.items()}


def _decide(
    vec: np.ndarray | None,
    prints: dict[str, np.ndarray],
    last: str,
    n_samples: int,
) -> str:
    if not prints or vec is None:
        return last if last in ("A", "B") else "A"
    scores = _scores(vec, prints)
    ranked = sorted(scores, key=scores.get, reverse=True)
    best = ranked[0]
    margin = scores[best] - (scores[ranked[1]] if len(ranked) > 1 else -1.0)
    long = n_samples >= int(SR * 1.2)
    if len(prints) < 2 and long and scores[best] < NEW_VOICE:
        label = "B" if "B" not in prints else best
        logger.info("new voice %s (best %s=%.3f)", label, best, scores[best])
        return label
    if len(prints) >= 2 and margin < SWITCH_MARGIN:
        logger.info(
            "keep %s (best %s=%.3f margin=%.3f)",
            last,
            best,
            scores[best],
            margin,
        )
        return last if last in prints else best
    logger.info("voice %s scores=%s", best, {k: round(v, 3) for k, v in scores.items()})
    return best


def _enroll(label: str, vec: np.ndarray, prints: dict[str, np.ndarray]) -> None:
    previous = prints.get(label)
    if previous is None:
        prints[label] = vec
        return
    score = float(np.dot(vec, previous))
    if score < ENROLL_SCORE:
        logger.info("skip enroll %s score=%.3f", label, score)
        return
    prints[label] = blend(previous, vec, keep=0.88)
