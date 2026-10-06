"""16 kHz speech enhancement.

Live mic uses sherpa-onnx GTCRN (streaming): it strips steady and
non-speech noise and keeps the voice. FRCRN stays for offline chunks.
"""

from __future__ import annotations

import logging
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_GTCRN_PATH = _ROOT / "checkpoints" / "gtcrn_simple.onnx"
_GTCRN_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speech-enhancement-models/gtcrn_simple.onnx"
)


@lru_cache(maxsize=1)
def get_denoiser():
    from clearvoice import ClearVoice

    logger.info("Loading FRCRN_SE_16K…")
    model = ClearVoice(task="speech_enhancement", model_names=["FRCRN_SE_16K"])
    logger.info("FRCRN denoiser ready")
    return model


def denoise_pcm16(pcm: bytes, sample_rate: int = 16000) -> bytes:
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    clean = denoise_f32(audio, sample_rate)
    return (np.clip(clean, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def _ensure_gtcrn() -> Path:
    if _GTCRN_PATH.exists() and _GTCRN_PATH.stat().st_size > 100_000:
        return _GTCRN_PATH
    logger.info("Downloading GTCRN denoiser…")
    _GTCRN_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_GTCRN_URL, _GTCRN_PATH)
    return _GTCRN_PATH


class StreamDenoiser:
    """Streaming GTCRN. One instance per mic session; state is the noise estimate."""

    def __init__(self) -> None:
        import sherpa_onnx

        config = sherpa_onnx.OnlineSpeechDenoiserConfig()
        config.model.gtcrn.model = str(_ensure_gtcrn())
        config.model.num_threads = 1
        config.model.provider = "cpu"
        self._denoiser = sherpa_onnx.OnlineSpeechDenoiser(config)

    def accept(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return audio
        out = self._denoiser.run(np.ascontiguousarray(audio), 16000)
        return np.asarray(out.samples, dtype=np.float32).reshape(-1)


def denoise_f32(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size < sample_rate // 4:
        return audio
    model = get_denoiser()
    batch = audio[np.newaxis, :]
    out = model(batch)
    out = np.asarray(out, dtype=np.float32).reshape(-1)
    if out.size < audio.size:
        out = np.pad(out, (0, audio.size - out.size))
    return out[: audio.size]
