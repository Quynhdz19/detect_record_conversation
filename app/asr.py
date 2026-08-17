"""Vietnamese ASR with PhoWhisper-small."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logger = logging.getLogger(__name__)

MODEL_ID = "vinai/PhoWhisper-small"


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
    "phụ đề được thực hiện",
    "phụ đề được thực hiện bởi",
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
    """Reject silence / faint room noise before calling ASR."""
    s = speech_stats(audio, sample_rate)
    return s["peak"] >= 0.048 and s["rms"] >= 0.009 and s["voiced_ratio"] >= 0.14


def clean_transcript(text: str, *, final: bool = False) -> str:
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
        if final and lower in _GREETING_OK_ON_FINAL:
            return collapsed
        return ""
    return collapsed


def transcribe_pcm16(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
    language: Optional[str] = "vi",
    final: bool = False,
) -> str:
    if len(pcm_bytes) < sample_rate:  # < ~0.5s of int16 mono
        return ""

    audio = pcm16_to_float32(pcm_bytes)
    if not looks_like_speech(audio, sample_rate):
        return ""

    asr = get_transcriber()
    generate_kwargs = {
        "task": "transcribe",
        "temperature": 0.0,
        "no_repeat_ngram_size": 3,
    }
    if language:
        generate_kwargs["language"] = language

    result = asr(
        {"array": audio, "sampling_rate": sample_rate},
        generate_kwargs=generate_kwargs,
        return_timestamps=False,
    )
    return clean_transcript(result.get("text") or "", final=final)
