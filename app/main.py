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
)
from app.av_tse import SAMPLE_RATE, get_av_tse
from app.live import LiveStream, infer_window
from app.pi_api import router as pi_router
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
    "asd_ready": False,
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

        try:
            BOOT["stage"] = "loading TalkNet-ASD (TalkSet)"
            logger.info("Warming TalkNet-ASD…")
            from app.asd import get_talknet

            get_talknet()
            BOOT["asd_ready"] = True
        except Exception:
            logger.exception("TalkNet warm-up failed; using lip VAD fallback")
            BOOT["asd_ready"] = False

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
app.include_router(pi_router)
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
        "asd_ready": BOOT.get("asd_ready", False),
        "asr_model": MODEL_ID,
        "av_tse_model": "AV_MossFormer2_TSE_16K",
        "asd_model": "TalkNet-ASD TalkSet",
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



@app.websocket("/ws")
async def ws_session(websocket: WebSocket):
    await websocket.accept()
    sample_rate = SAMPLE_RATE
    live = LiveStream(sr=sample_rate)
    last_face: dict[str, Any] = {"found": False}
    require_speaking = True
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
            "mode": "live",
        }
    )

    async def process_chunk(final: bool = False) -> None:
        if live.busy or live.buf_sec() < 0.7:
            return
        if not BOOT["asr_ready"]:
            await websocket.send_json({"type": "status", "text": "ASR chưa sẵn sàng…"})
            return
        chunk, crops = live.snapshot()
        live.busy = True
        live.last_decode_at = time.time()
        try:
            use_tse = final and bool(BOOT.get("av_tse_ready"))
            if not looks_like_speech(pcm16_to_float32(chunk), sample_rate):
                if final:
                    live.discard()
                    await websocket.send_json({"type": "status", "text": "Im lặng — bỏ qua"})
                elif live.buf_sec() > 2.0:
                    keep = int(sample_rate * 0.4) * 2
                    live.buf[:] = live.buf[-keep:]
                return
            live.had_speech = True
            await websocket.send_json(
                {"type": "status", "text": "Đang nhận chữ…" if not final else "AV-TSE đang tách giọng…"}
            )
            text, used_tse = await loop.run_in_executor(
                None, infer_window, chunk, crops, use_tse, final
            )
            text = clean_transcript(text, final=final)
            if text and (final or text.lower() != live.last_text.lower()):
                live.last_text = text
                await websocket.send_json(
                    {
                        "type": "transcript",
                        "text": text,
                        "final": final,
                        "used_tse": used_tse,
                        "face": last_face,
                    }
                )
            if final:
                live.commit()
            await websocket.send_json({"type": "status", "text": "Đang nghe…"})
        finally:
            live.busy = False

    async def maybe_decode() -> None:
        now = time.time()
        if live.want_len_final() or live.want_silence_final(now):
            await process_chunk(final=True)
        elif live.want_partial(now):
            await process_chunk(final=False)

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
                    "lip_active": cue.lip_active,
                    "asd_score": getattr(cue, "asd_score", 0.0),
                    "cx": cue.cx,
                    "cy": cue.cy,
                }
                if cue.found and cue.face_bgr is not None:
                    live.push_crop(cue.face_bgr)
                await websocket.send_json({"type": "face", **last_face})
                if require_speaking and last_face.get("found") and not last_face.get("speaking"):
                    if live.had_speech:
                        await maybe_decode()
                    else:
                        live.discard()
                else:
                    await maybe_decode()
                continue

            if tag != 2:
                continue

            get_tracker().note_pcm16(payload, time.time())
            if require_speaking and last_face.get("found") and not last_face.get("speaking"):
                if live.had_speech:
                    await maybe_decode()
                else:
                    live.discard()
                continue

            live.push_audio(payload, time.time())
            await maybe_decode()

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception:
        logger.exception("WebSocket session error")
        try:
            await websocket.close()
        except Exception:
            pass
