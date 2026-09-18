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

    audio = pcm16_to_float32(pcm_bytes)
    stats = speech_stats(audio, sample_rate)
    if not looks_like_speech(audio, sample_rate):
        logger.info("skip ASR: not speech %s", stats)
        return ""

    asr = get_transcriber()
    nsp = _no_speech_prob(asr, audio, sample_rate)
    if nsp >= 0.7 and stats["rms"] < 0.01 and stats["voiced_ratio"] < 0.08:
        logger.info("skip ASR: no_speech_prob=%.2f stats=%s", nsp, stats)
        return ""

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
