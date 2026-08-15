"""Process an MP4 through face-cue AV-TSE + PhoWhisper."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

from app.asr import transcribe_pcm16
from app.av_tse import SAMPLE_RATE, get_av_tse
from app.vision import get_tracker

logger = logging.getLogger(__name__)

TMP_ROOT = Path(__file__).resolve().parents[1] / "tmp"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

# Demo limits
MAX_SECONDS = 30.0
FACE_SAMPLE_FPS = 8.0


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-800:]}"
        )


def extract_audio_wav(video_path: Path, wav_path: Path, max_seconds: float = MAX_SECONDS) -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-t",
            str(max_seconds),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "wav",
            str(wav_path),
        ]
    )


def collect_face_crops(video_path: Path, max_seconds: float = MAX_SECONDS) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Không mở được video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0, fps * max_seconds))
    step = max(1, int(round(fps / FACE_SAMPLE_FPS)))
    tracker = get_tracker()
    crops: list[np.ndarray] = []
    last_crop = None
    idx = 0

    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        cue = tracker.analyze_bgr(frame)
        if cue.found and cue.face_bgr is not None:
            crops.append(cue.face_bgr)
            last_crop = cue.face_bgr
        elif last_crop is not None:
            crops.append(last_crop)
        idx += step

    cap.release()
    return crops


def process_mp4(video_bytes: bytes, filename: str = "input.mp4") -> dict:
    job_id = uuid.uuid4().hex[:10]
    work = TMP_ROOT / job_id
    work.mkdir(parents=True, exist_ok=True)

    suffix = Path(filename).suffix.lower() or ".mp4"
    video_path = work / f"input{suffix}"
    wav_path = work / "mix.wav"
    out_wav = work / "tse.wav"
    video_path.write_bytes(video_bytes)

    logger.info("Processing %s (%d bytes)", video_path.name, len(video_bytes))
    extract_audio_wav(video_path, wav_path)
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"Unexpected sample rate {sr}")

    crops = collect_face_crops(video_path)
    used_tse = False
    if crops:
        try:
            extracted = get_av_tse().extract(audio, crops)
            audio = extracted
            used_tse = True
        except Exception:
            logger.exception("AV-TSE on video failed; using original audio")
    else:
        logger.warning("No face crops found; ASR on raw audio")

    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    sf.write(str(out_wav), audio, SAMPLE_RATE)
    pcm = (audio * 32767.0).astype(np.int16).tobytes()
    text = transcribe_pcm16(pcm, sample_rate=SAMPLE_RATE)

    return {
        "job_id": job_id,
        "text": text or "(không nhận được lời nói)",
        "used_tse": used_tse,
        "face_frames": len(crops),
        "duration_sec": round(len(audio) / SAMPLE_RATE, 2),
        "audio_url": f"/tmp_audio/{job_id}/tse.wav",
        "video_url": f"/tmp_audio/{job_id}/{video_path.name}",
    }
