"""Web demo: camera + mic → AV-TSE → PhoWhisper Vietnamese ASR."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.asr import (
    MODEL_ID,
    clean_transcript,
    get_transcriber,
    looks_like_speech,
    pcm16_to_float32,
    pick_device,
    transcribe_pcm16,
)
from app.av_tse import SAMPLE_RATE, get_av_tse
from app.video_pipeline import TMP_ROOT, process_mp4
from app.vision import get_tracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Shared boot status so UI can poll without blocking server start
BOOT: dict[str, Any] = {
    "ready": False,
    "stage": "starting",
    "error": None,
    "asr_ready": False,
    "vision_ready": False,
    "av_tse_ready": False,
}


def _warm_models() -> None:
    try:
        device, dtype = pick_device()
        BOOT["stage"] = f"loading PhoWhisper ({device})"
        logger.info("Warming PhoWhisper %s on %s/%s", MODEL_ID, device, dtype)
        get_transcriber()
        BOOT["asr_ready"] = True

        BOOT["stage"] = "loading Face Landmarker"
        get_tracker()
        BOOT["vision_ready"] = True

        BOOT["stage"] = "loading AV-TSE AV_MossFormer2_TSE_16K"
        logger.info("Warming AV-TSE AV_MossFormer2_TSE_16K...")
        get_av_tse()
        BOOT["av_tse_ready"] = True

        BOOT["stage"] = "ready"
        BOOT["ready"] = True
        logger.info("All models ready")
    except Exception as exc:
        logger.exception("Model warm-up failed")
        BOOT["stage"] = "error"
        BOOT["error"] = str(exc)
        BOOT["ready"] = False


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Serve HTTP immediately; load heavy models in background
    threading.Thread(target=_warm_models, name="warm-models", daemon=True).start()
    yield


app = FastAPI(title="Detect Giọng Nói Demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/tmp_audio", StaticFiles(directory=TMP_ROOT), name="tmp_audio")


@app.get("/")
async def index():
    resp = FileResponse(STATIC_DIR / "index.html")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/health")
async def health():
    device, dtype = pick_device()
    return {
        "ok": True,
        "ready": BOOT["ready"],
        "stage": BOOT["stage"],
        "error": BOOT["error"],
        "asr_ready": BOOT["asr_ready"],
        "vision_ready": BOOT["vision_ready"],
        "av_tse_ready": BOOT["av_tse_ready"],
        "asr_model": MODEL_ID,
        "av_tse_model": "AV_MossFormer2_TSE_16K",
        "device": device,
        "dtype": dtype,
    }


@app.post("/api/process-video")
async def api_process_video(file: UploadFile = File(...)):
    if not BOOT["ready"]:
        raise HTTPException(status_code=503, detail=f"Models not ready: {BOOT['stage']}")
    name = file.filename or "input.mp4"
    lower = name.lower()
    if not lower.endswith((".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v")):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ video: mp4/mov/webm/mkv/avi")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File rỗng")
    if len(data) > 120 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File quá lớn (>120MB)")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, process_mp4, data, name)
    except Exception as exc:
        logger.exception("process-video failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(result)


@app.middleware("http")
async def no_cache_static(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, max-age=0"
    return response



def _run_av_asr(pcm_bytes: bytes, face_crops: list) -> tuple[str, bool]:
    """AV-TSE extract then PhoWhisper. Returns (text, used_tse)."""
    audio = pcm16_to_float32(pcm_bytes)
    if not looks_like_speech(audio, SAMPLE_RATE):
        return "", False

    used_tse = False
    if face_crops and BOOT.get("av_tse_ready"):
        try:
            extracted = get_av_tse().extract(audio, face_crops)
            audio = extracted
            used_tse = True
        except Exception:
            logger.exception("AV-TSE failed; using raw mic")

    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    if not looks_like_speech(audio, SAMPLE_RATE):
        return "", used_tse

    pcm = (audio * 32767.0).astype(np.int16).tobytes()
    text = clean_transcript(transcribe_pcm16(pcm, sample_rate=SAMPLE_RATE))
    return text, used_tse


@app.websocket("/ws")
async def ws_session(websocket: WebSocket):
    await websocket.accept()
    sample_rate = SAMPLE_RATE
    audio_buf = bytearray()
    face_crops: list[Any] = []
    last_face: dict[str, Any] = {"found": False}
    last_transcript_at = 0.0
    last_text = ""
    chunk_sec = 3.0
    require_speaking = True
    busy = False
    loop = asyncio.get_event_loop()

    # Wait briefly if models still loading
    waited = 0
    while not BOOT["ready"] and BOOT["stage"] != "error" and waited < 120:
        await websocket.send_json(
            {"type": "status", "text": f"Đang load model: {BOOT['stage']}"}
        )
        await asyncio.sleep(0.5)
        waited += 0.5

    if BOOT["stage"] == "error" or not BOOT["asr_ready"]:
        await websocket.send_json(
            {
                "type": "error",
                "text": BOOT.get("error") or "Model chưa sẵn sàng",
            }
        )
        await websocket.close()
        return

    await websocket.send_json(
        {
            "type": "ready",
            "asr_model": MODEL_ID,
            "av_tse_model": "AV_MossFormer2_TSE_16K",
            "device": pick_device()[0],
            "sample_rate": sample_rate,
            "av_tse_ready": BOOT["av_tse_ready"],
        }
    )

    async def process_chunk(final: bool = False) -> None:
        nonlocal audio_buf, face_crops, last_transcript_at, last_text, busy
        if busy:
            return
        min_bytes = int(sample_rate * 1.2) * 2
        if len(audio_buf) < min_bytes:
            return
        if not BOOT["asr_ready"]:
            await websocket.send_json(
                {"type": "status", "text": "ASR chưa sẵn sàng…"}
            )
            return
        chunk = bytes(audio_buf)
        if not looks_like_speech(pcm16_to_float32(chunk), sample_rate):
            audio_buf.clear()
            face_crops.clear()
            await websocket.send_json({"type": "status", "text": "Im lặng — bỏ qua"})
            return
        busy = True
        crops = list(face_crops)
        overlap = int(sample_rate * 0.35) * 2
        audio_buf[:] = audio_buf[-overlap:]
        face_crops.clear()
        try:
            await websocket.send_json({"type": "status", "text": "AV-TSE đang tách giọng…"})
            text, used_tse = await loop.run_in_executor(None, _run_av_asr, chunk, crops)
            text = clean_transcript(text)
            if text and text.lower() != last_text.lower():
                last_text = text
                last_transcript_at = time.time()
                await websocket.send_json(
                    {
                        "type": "transcript",
                        "text": text,
                        "final": final,
                        "used_tse": used_tse,
                        "face": last_face,
                    }
                )
            await websocket.send_json({"type": "status", "text": "Đang nghe…"})
        finally:
            busy = False

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if message.get("text") is not None:
                data = json.loads(message["text"])
                msg_type = data.get("type")
                if msg_type == "config":
                    require_speaking = bool(data.get("require_speaking", False))
                    await websocket.send_json(
                        {"type": "config_ok", "require_speaking": require_speaking}
                    )
                elif msg_type == "flush":
                    await process_chunk(final=True)
                continue

            raw = message.get("bytes")
            if raw is None or len(raw) < 2:
                continue
            tag, payload = raw[0], raw[1:]

            if tag == 1:  # jpeg
                if not BOOT["vision_ready"]:
                    continue
                cue = await loop.run_in_executor(None, get_tracker().analyze_jpeg, payload)
                last_face = {
                    "found": cue.found,
                    "x": cue.x,
                    "y": cue.y,
                    "w": cue.w,
                    "h": cue.h,
                    "mouth_open": cue.mouth_open,
                    "speaking": cue.speaking,
                    "cx": cue.cx,
                    "cy": cue.cy,
                }
                if cue.found and cue.face_bgr is not None:
                    face_crops.append(cue.face_bgr)
                    if len(face_crops) > 40:
                        face_crops = face_crops[-40:]
                await websocket.send_json({"type": "face", **last_face})
                continue

            if tag != 2:
                continue

            if require_speaking and last_face.get("found") and not last_face.get("speaking"):
                continue

            audio_buf.extend(payload)
            buf_sec = len(audio_buf) / (sample_rate * 2)
            now = time.time()
            if buf_sec >= chunk_sec and (now - last_transcript_at) > 0.8:
                await process_chunk(final=False)

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception:
        logger.exception("WebSocket session error")
        try:
            await websocket.close()
        except Exception:
            pass
