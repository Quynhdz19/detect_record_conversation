"""Tell person A's voice from person B's voice (3D-Speaker embedding)."""

from __future__ import annotations

import logging
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
_MODEL = _ROOT / "checkpoints" / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)


def _ensure_model() -> Path:
    if _MODEL.exists() and _MODEL.stat().st_size > 1_000_000:
        return _MODEL
    logger.info("Downloading speaker embedding model…")
    _MODEL.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_URL, _MODEL)
    return _MODEL


@lru_cache(maxsize=1)
def get_extractor():
    import sherpa_onnx

    path = _ensure_model()
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(path),
        num_threads=2,
        debug=False,
        provider="cpu",
    )
    if not config.validate():
        raise RuntimeError(f"Bad speaker model: {path}")
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
    logger.info("Speaker embedding ready (dim=%s)", extractor.dim)
    return extractor


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    return vec / (float(np.linalg.norm(vec)) + 1e-8)


def embed_f32(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray | None:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size < int(sample_rate * 0.8):
        return None
    extractor = get_extractor()
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate, np.ascontiguousarray(audio))
    stream.input_finished()
    if not extractor.is_ready(stream):
        return None
    return _unit(extractor.compute(stream))


def embed_pcm16(pcm: bytes, sample_rate: int = 16000) -> np.ndarray | None:
    if len(pcm) < sample_rate:
        return None
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return embed_f32(audio, sample_rate)


def blend(previous: np.ndarray | None, new: np.ndarray, keep: float = 0.75) -> np.ndarray:
    if previous is None:
        return _unit(new)
    mixed = keep * _unit(previous) + (1.0 - keep) * _unit(new)
    return _unit(mixed)


def match_voice(pcm: bytes, prints: dict[str, np.ndarray], sample_rate: int = 16000) -> str:
    """Name whose enrolled voice this clip matches. Empty if only one voice or a tie."""
    if len(prints) < 2 or len(pcm) < sample_rate:
        return ""
    vec = embed_pcm16(pcm, sample_rate)
    if vec is None:
        return ""
    scores = {name: float(np.dot(vec, _unit(emb))) for name, emb in prints.items()}
    ranked = sorted(scores, key=scores.get, reverse=True)
    best, other = ranked[0], ranked[1]
    if scores[best] < 0.35 or scores[best] - scores[other] < 0.05:
        return ""
    logger.info("voice match %s scores=%s", best, {k: round(v, 3) for k, v in scores.items()})
    return best
