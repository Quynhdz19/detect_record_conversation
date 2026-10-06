"""Vietnamese ASR. Live path uses Zipformer-30M (sherpa-onnx); PhoWhisper is fallback."""

from __future__ import annotations

import logging
import tarfile
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logger = logging.getLogger(__name__)

# Streaming-style RNN-T trained on ~6000h Vietnamese. Much smaller and
# faster than PhoWhisper-small, and it does not wait on a 30s Whisper window.
MODEL_ID = "hynt/Zipformer-30M-RNNT-6000h"
WHISPER_ID = "vinai/PhoWhisper-small"
_ROOT = Path(__file__).resolve().parents[1]
_ZIP_DIR = _ROOT / "checkpoints" / "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"
_ZIP_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2"
)


def pick_device() -> tuple[str, str]:
    if torch.backends.mps.is_available():
        return "mps", "float16"
    if torch.cuda.is_available():
        return "cuda:0", "float16"
    return "cpu", "float32"


@lru_cache(maxsize=1)
def get_transcriber():
    device, dtype_name = pick_device()
    torch_dtype = torch.float16 if dtype_name == "float16" else torch.float32
    logger.info("Loading %s on %s (%s)...", MODEL_ID, device, dtype_name)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if device == "cpu":
        device_index: int | str = -1
    elif device.startswith("cuda"):
        device_index = int(device.split(":")[-1])
    else:
        device_index = "mps"

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch_dtype,
        device=device_index,
    )
    logger.info("Model ready.")
    return asr


def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return audio / 32768.0


def peak_normalize(audio: np.ndarray, target: float = 0.85, max_gain: float = 8.0) -> np.ndarray:
    """Bring conversational speech up without exploding hush."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak < 0.02:
        return audio
    return np.clip(audio * min(target / peak, max_gain), -1.0, 1.0)


def _ensure_zipformer() -> Path:
    if (_ZIP_DIR / "encoder.int8.onnx").exists() and (_ZIP_DIR / "tokens.txt").exists():
        return _ZIP_DIR
    logger.info("Downloading %s …", MODEL_ID)
    _ZIP_DIR.parent.mkdir(parents=True, exist_ok=True)
    tmp = _ZIP_DIR.parent / "zipformer-vi.tar.bz2"
    urllib.request.urlretrieve(_ZIP_URL, tmp)
    with tarfile.open(tmp) as tar:
        tar.extractall(_ZIP_DIR.parent)
    tmp.unlink(missing_ok=True)
    if not (_ZIP_DIR / "encoder.int8.onnx").exists():
        raise FileNotFoundError(f"Zipformer extract missing encoder in {_ZIP_DIR}")
    return _ZIP_DIR


@lru_cache(maxsize=1)
def get_zipformer():
    import sherpa_onnx

    root = _ensure_zipformer()
    logger.info("Loading %s (int8 CPU)…", MODEL_ID)
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(root / "encoder.int8.onnx"),
        decoder=str(root / "decoder.onnx"),
        joiner=str(root / "joiner.int8.onnx"),
        tokens=str(root / "tokens.txt"),
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="modified_beam_search",
        max_active_paths=4,
        provider="cpu",
    )
    logger.info("Zipformer ready.")
    return rec


_VAD_PATH = _ROOT / "checkpoints" / "silero_vad.onnx"
_VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"


def _ensure_silero_vad() -> Path:
    if _VAD_PATH.exists() and _VAD_PATH.stat().st_size > 100_000:
        return _VAD_PATH
    logger.info("Downloading Silero VAD…")
    _VAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_VAD_URL, _VAD_PATH)
    return _VAD_PATH


def _zipformer_text(audio: np.ndarray, sample_rate: int) -> str:
    rec = get_zipformer()
    stream = rec.create_stream()
    stream.accept_waveform(sample_rate, np.ascontiguousarray(audio, dtype=np.float32))
    rec.decode_stream(stream)
    return (stream.result.text or "").strip().lower()


class MicZipformer:
    """sherpa-onnx mic path: Silero VAD + offline Zipformer-30M.

    Same pattern as sherpa-onnx-vad-microphone-simulated-streaming-asr:
    partial text while speech continues, final text when VAD endpoints.
    """

    def __init__(self) -> None:
        import sherpa_onnx

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(_ensure_silero_vad())
        config.silero_vad.threshold = 0.5
        config.silero_vad.min_silence_duration = 0.5
        config.silero_vad.min_speech_duration = 0.25
        config.silero_vad.max_speech_duration = 15
        config.sample_rate = 16000
        config.provider = "cpu"
        self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
        self._window = 512
        self._pending = np.zeros(0, dtype=np.float32)
        self._utt = np.zeros(0, dtype=np.float32)
        self._since_partial = 0

    def accept(self, audio: np.ndarray) -> list[tuple[str, bool]]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return []
        self._pending = np.concatenate([self._pending, audio])
        events: list[tuple[str, bool]] = []
        while self._pending.size >= self._window:
            frame = np.ascontiguousarray(self._pending[: self._window])
            self._pending = self._pending[self._window :]
            self._vad.accept_waveform(frame)
            if self._vad.is_speech_detected():
                self._utt = np.concatenate([self._utt, frame])
                self._since_partial += frame.size
                if (
                    self._since_partial >= 16000 * 0.4
                    and self._utt.size >= int(16000 * 0.35)
                ):
                    text = _zipformer_text(self._utt, 16000)
                    if text:
                        events.append((text, False))
                    self._since_partial = 0
            while not self._vad.empty():
                segment = np.array(self._vad.front.samples, dtype=np.float32)
                self._vad.pop()
                if segment.size >= int(16000 * 0.25):
                    text = _zipformer_text(segment, 16000)
                    if text:
                        events.append((text, True))
                self._utt = np.zeros(0, dtype=np.float32)
                self._since_partial = 0
        return events

    def flush(self) -> list[tuple[str, bool]]:
        if self._utt.size < int(16000 * 0.25):
            self._utt = np.zeros(0, dtype=np.float32)
            return []
        text = _zipformer_text(self._utt, 16000)
        self._utt = np.zeros(0, dtype=np.float32)
        self._since_partial = 0
        return [(text, True)] if text else []


# Whisper invents these on silence / room noise
_HALLUCINATION_EXACT = {
    "cảm ơn",
    "cảm ơn bạn",
    "cảm ơn các bạn",
    "cảm ơn các bạn đã theo dõi",
    "xin chào",
    "xin chào các bạn",
    "hẹn gặp lại",
    "tạm biệt",
    "bạn",
    "ừ",
    "ừm",
    "à",
    "ờ",
    "uh",
    "um",
    "you",
    "the",
    "thank you",
    "thanks for watching",
    "subscribe",
    ".",
    "...",
    "…",
}

_GREETING_OK_ON_FINAL = {
    "xin chào",
    "cảm ơn",
    "cảm ơn bạn",
    "hẹn gặp lại",
    "tạm biệt",
}

_HALLUCINATION_SUBSTR = (
    "hãy subscribe",
    "đăng ký kênh",
    "hãy đăng ký",
    "phụ đề được thực hiện",
    "phụ đề được thực hiện bởi",
    "cảm ơn bạn đã",
    "cảm ơn các bạn đã",
    "nhiều doanh nghiệp",
    "thanks for watching",
    "please subscribe",
    "vietsub",
    "subtitle",
)


def speech_stats(audio: np.ndarray, sample_rate: int = 16000) -> dict:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return {"peak": 0.0, "rms": 0.0, "voiced_ratio": 0.0}
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio * audio)))
    frame = max(1, int(sample_rate * 0.02))
    n = audio.size // frame
    if n < 1:
        return {"peak": peak, "rms": rms, "voiced_ratio": 1.0 if rms > 0.02 else 0.0}
    frames = audio[: n * frame].reshape(n, frame)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    voiced_ratio = float(np.mean(frame_rms > 0.018))
    return {"peak": peak, "rms": rms, "voiced_ratio": voiced_ratio}


def looks_like_speech(audio: np.ndarray, sample_rate: int = 16000) -> bool:
    """Reject hush / noise floor; keep conversational speech (AGC off).

    Measured quiet room: rms ~0.005–0.007, voiced_ratio ~0.00–0.03.
    """
    s = speech_stats(audio, sample_rate)
    return s["peak"] >= 0.048 and s["rms"] >= 0.008 and s["voiced_ratio"] >= 0.045


def _text_too_long_for_noise(text: str, stats: dict) -> bool:
    """Drop news-like Whisper inventions on near-silence only."""
    n_words = len((text or "").split())
    if n_words < 6:
        return False
    return stats.get("rms", 1.0) < 0.009 and stats.get("voiced_ratio", 1.0) < 0.08


def _no_speech_prob(asr, audio: np.ndarray, sample_rate: int) -> float:
    """First-decoder-step P(<|nospeech|>). High ⇒ Whisper thinks the clip is silence."""
    try:
        model = asr.model
        fe = asr.feature_extractor
        tok = asr.tokenizer
        gen = model.generation_config
        no_speech_id = getattr(gen, "no_speech_token_id", None)
        if no_speech_id is None:
            no_speech_id = tok.convert_tokens_to_ids("<|nospeech|>")
        if no_speech_id is None or int(no_speech_id) < 0:
            return 0.0
        param = next(model.parameters())
        feats = fe(audio, sampling_rate=sample_rate, return_tensors="pt")
        input_features = feats.input_features.to(device=param.device, dtype=param.dtype)
        start_id = gen.decoder_start_token_id or model.config.decoder_start_token_id
        decoder_input_ids = torch.tensor([[start_id]], device=param.device)
        with torch.inference_mode():
            out = model(input_features=input_features, decoder_input_ids=decoder_input_ids)
            prob = torch.softmax(out.logits[0, -1].float(), dim=-1)[int(no_speech_id)]
        return float(prob)
    except Exception:
        logger.debug("no_speech_prob failed", exc_info=True)
        return 0.0


def clean_transcript(
    text: str, *, final: bool = False, allow_greetings: bool | None = None
) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    collapsed = " ".join(raw.split())
    lower = collapsed.lower().strip(" .,-!?…\"'")
    if not lower or len(lower) < 2:
        return ""
    if any(p in lower for p in _HALLUCINATION_SUBSTR):
        return ""
    parts = lower.split()
    if len(parts) >= 3 and len(set(parts)) == 1:
        return ""
    if lower in _HALLUCINATION_EXACT:
        # "xin chào" / "cảm ơn" are also Whisper's default silence hallucinations.
        # Only keep them when the caller saw real speech in this window.
        keep_greeting = allow_greetings if allow_greetings is not None else final
        if keep_greeting and lower in _GREETING_OK_ON_FINAL:
            return collapsed
        return ""
    return collapsed


def transcribe_pcm16(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
    language: Optional[str] = "vi",
    final: bool = False,
    allow_greetings: bool | None = None,
) -> str:
    if len(pcm_bytes) < int(sample_rate * 0.7):  # < ~0.35s of int16 mono
        return ""

    # Mild gain only. Zipformer was trained on normal speech, not 8x boosted noise.
    audio = peak_normalize(pcm16_to_float32(pcm_bytes), target=0.55, max_gain=3.0)
    stats = speech_stats(audio, sample_rate)
    if not looks_like_speech(audio, sample_rate):
        logger.info("skip ASR: not speech %s", stats)
        return ""

    try:
        raw_text = _zipformer_text(audio, sample_rate)
        text = clean_transcript(raw_text, final=final, allow_greetings=allow_greetings)
        if text:
            logger.info("Zipformer ok %r stats=%s", text, stats)
        else:
            logger.info("Zipformer cleaned empty from %r stats=%s", raw_text, stats)
        return text
    except Exception:
        logger.exception("Zipformer failed; falling back to PhoWhisper")

    asr = get_transcriber()
    nsp = _no_speech_prob(asr, audio, sample_rate)
    if nsp >= 0.7 and stats["rms"] < 0.01 and stats["voiced_ratio"] < 0.08:
        logger.info("skip ASR: no_speech_prob=%.2f stats=%s", nsp, stats)
        return ""

    generate_kwargs = {
        "task": "transcribe",
        "temperature": 0.0,
        "do_sample": False,
        "num_beams": 3 if final else 1,
    }
    if language:
        generate_kwargs["language"] = language

    result = asr(
        {"array": audio, "sampling_rate": sample_rate},
        generate_kwargs=generate_kwargs,
        return_timestamps=False,
    )
    raw_text = result.get("text") or ""
    if _text_too_long_for_noise(raw_text, stats):
        logger.info("drop hallucination %r stats=%s nsp=%.2f", raw_text, stats, nsp)
        return ""
    text = clean_transcript(raw_text, final=final, allow_greetings=allow_greetings)
    if text:
        logger.info("ASR ok %r stats=%s nsp=%.2f", text, stats, nsp)
    else:
        logger.info("ASR cleaned empty from %r stats=%s", raw_text, stats)
    return text
