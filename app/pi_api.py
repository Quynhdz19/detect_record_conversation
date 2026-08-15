"""API for Raspberry Pi (or any client) to stream camera + mic and get conversation text."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import (
    APIRouter,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)

from app.asr import MODEL_ID, clean_transcript, looks_like_speech, pcm16_to_float32, pick_device
from app.av_tse import SAMPLE_RATE
from app.conversation import append_turn, create_session, get_session
from app.pipeline import run_av_asr
from app.vision import get_tracker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pi", tags=["pi"])

PI_API_TOKEN = os.environ.get("PI_API_TOKEN", "").strip()


def _boot() -> dict[str, Any]:
    from app.main import BOOT

    return BOOT


def _check_token(token: str | None) -> None:
    if not PI_API_TOKEN:
        return
    if (token or "").strip() != PI_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")


def _decode_audio_bytes(data: bytes, filename: str = "audio.wav") -> bytes:
    """Return PCM16 mono 16 kHz bytes."""
    name = (filename or "").lower()
    if name.endswith(".pcm") or name.endswith(".raw"):
        return data
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    except Exception as exc:
        # assume already pcm16
        if len(data) >= 2 and len(data) % 2 == 0:
            return data
        raise HTTPException(status_code=400, detail=f"Không đọc được audio: {exc}") from exc
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16).tobytes()


def _crops_from_jpegs(frames: list[bytes]) -> list:
    tracker = get_tracker()
    crops = []
    for jpeg in frames:
        if not jpeg:
            continue
        cue = tracker.analyze_jpeg(jpeg)
        if cue.found and cue.face_bgr is not None:
            crops.append(cue.face_bgr)
    return crops


@router.get("/info")
async def pi_info():
    boot = _boot()
    device, dtype = pick_device()
    return {
        "ok": True,
        "ready": boot["ready"],
        "stage": boot["stage"],
        "sample_rate": SAMPLE_RATE,
        "protocol": "websocket",
        "audio_format": "pcm_s16le mono 16kHz",
        "video_format": "jpeg frames",
        "asr_model": MODEL_ID,
        "av_tse_model": "AV_MossFormer2_TSE_16K",
        "device": device,
        "dtype": dtype,
        "stream": "WS /api/pi/ws?device_id=pi&session_id=optional&token=optional",
        "auth": "optional ?token= (env PI_API_TOKEN)",
    }


@router.post("/session")
async def pi_create_session(
    device_id: str = Form(default=""),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
    token: str | None = Query(default=None),
):
    _check_token(x_api_token or token)
    return create_session(device_id=device_id)


@router.get("/session/{session_id}")
async def pi_get_session(
    session_id: str,
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
    token: str | None = Query(default=None),
):
    _check_token(x_api_token or token)
    sess = get_session(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found")
    return sess


@router.post("/utterance")
async def pi_utterance(
    session_id: str = Form(...),
    audio: UploadFile = File(..., description="wav or raw pcm16le mono 16kHz"),
    frames: list[UploadFile] = File(default_factory=list, description="optional JPEG face frames"),
    x_api_token: str | None = Header(default=None, alias="X-API-Token"),
    token: str | None = Query(default=None),
):
    """One-shot: Pi gửi 1 đoạn mic + vài frame camera, nhận text + hội thoại."""
    _check_token(x_api_token or token)
    boot = _boot()
    if not boot.get("asr_ready"):
        raise HTTPException(status_code=503, detail=f"Models not ready: {boot.get('stage')}")
    if get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="session not found — POST /api/pi/session trước")

    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="audio rỗng")
    pcm = _decode_audio_bytes(raw, audio.filename or "audio.wav")

    jpegs: list[bytes] = []
    for f in frames:
        data = await f.read()
        if data:
            jpegs.append(data)

    loop = asyncio.get_event_loop()
    crops = []
    if jpegs and boot.get("vision_ready"):
        crops = await loop.run_in_executor(None, _crops_from_jpegs, jpegs)

    text, used_tse = await loop.run_in_executor(
        None, run_av_asr, pcm, crops, bool(boot.get("av_tse_ready"))
    )
    text = clean_transcript(text)
    sess = append_turn(session_id, text, used_tse=used_tse) if text else get_session(session_id)
    return {
        "ok": True,
        "text": text,
        "used_tse": used_tse,
        "face_frames": len(crops),
        "session": sess,
        "conversation": (sess or {}).get("conversation", ""),
    }


@router.websocket("/ws")
async def pi_ws(
    websocket: WebSocket,
    session_id: str | None = Query(default=None),
    device_id: str = Query(default=""),
    token: str | None = Query(default=None),
):
    """
    Stream realtime từ Pi.

    Binary:
      0x01 + JPEG
      0x02 + PCM16LE mono 16kHz
    JSON:
      {"type":"flush"}  — chốt câu
      {"type":"config","require_speaking":true}
    Server gửi:
      {"type":"transcript","text":"...","conversation":"...","turns":[...]}
    """
    await websocket.accept()
    try:
        _check_token(token)
    except HTTPException as exc:
        await websocket.send_json({"type": "error", "text": exc.detail})
        await websocket.close(code=4401)
        return

    boot = _boot()
    waited = 0.0
    while not boot.get("ready") and boot.get("stage") != "error" and waited < 120:
        await websocket.send_json({"type": "status", "text": f"loading: {boot.get('stage')}"})
        await asyncio.sleep(0.5)
        waited += 0.5
        boot = _boot()

    if boot.get("stage") == "error" or not boot.get("asr_ready"):
        await websocket.send_json({"type": "error", "text": boot.get("error") or "not ready"})
        await websocket.close()
        return

    sess = get_session(session_id) if session_id else None
    if sess is None:
        sess = create_session(device_id=device_id)
    session_id = sess["session_id"]

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

    await websocket.send_json(
        {
            "type": "ready",
            "session_id": session_id,
            "sample_rate": sample_rate,
            "asr_model": MODEL_ID,
            "device": pick_device()[0],
            "protocol": "websocket",
        }
    )
    last_ping = time.time()

    async def process_chunk(final: bool = False) -> None:
        nonlocal audio_buf, face_crops, last_transcript_at, last_text, busy, sess
        if busy:
            return
        min_bytes = int(sample_rate * 1.2) * 2
        if len(audio_buf) < min_bytes:
            return
        chunk = bytes(audio_buf)
        if not looks_like_speech(pcm16_to_float32(chunk), sample_rate):
            audio_buf.clear()
            face_crops.clear()
            await websocket.send_json({"type": "status", "text": "silence"})
            return
        busy = True
        crops = list(face_crops)
        overlap = int(sample_rate * 0.35) * 2
        audio_buf[:] = audio_buf[-overlap:]
        face_crops.clear()
        try:
            text, used_tse = await loop.run_in_executor(
                None, run_av_asr, chunk, crops, bool(_boot().get("av_tse_ready"))
            )
            text = clean_transcript(text)
            if text and text.lower() != last_text.lower():
                last_text = text
                last_transcript_at = time.time()
                sess = append_turn(session_id, text, used_tse=used_tse) or sess
                await websocket.send_json(
                    {
                        "type": "transcript",
                        "text": text,
                        "final": final,
                        "used_tse": used_tse,
                        "session_id": session_id,
                        "conversation": sess.get("conversation", ""),
                        "turns": sess.get("turns", []),
                    }
                )
        finally:
            busy = False

    try:
        while True:
            if time.time() - last_ping > 15:
                await websocket.send_json({"type": "ping"})
                last_ping = time.time()
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text") is not None:
                data = json.loads(message["text"])
                kind = data.get("type")
                if kind == "config":
                    require_speaking = bool(data.get("require_speaking", True))
                    await websocket.send_json(
                        {"type": "config_ok", "require_speaking": require_speaking}
                    )
                elif kind == "flush":
                    await process_chunk(final=True)
                elif kind == "get_conversation":
                    await websocket.send_json(
                        {"type": "conversation", **(get_session(session_id) or {})}
                    )
                elif kind == "pong":
                    pass
                continue

            raw = message.get("bytes")
            if raw is None or len(raw) < 2:
                continue
            tag, payload = raw[0], raw[1:]
            if tag == 1:
                if not _boot().get("vision_ready"):
                    continue
                cue = await loop.run_in_executor(None, get_tracker().analyze_jpeg, payload)
                last_face = {
                    "found": cue.found,
                    "speaking": cue.speaking,
                    "x": cue.x,
                    "y": cue.y,
                    "w": cue.w,
                    "h": cue.h,
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
            if buf_sec >= chunk_sec and (time.time() - last_transcript_at) > 0.8:
                await process_chunk(final=False)
    except WebSocketDisconnect:
        logger.info("Pi WS disconnected session=%s", session_id)
    except Exception:
        logger.exception("Pi WS error")
        try:
            await websocket.close()
        except Exception:
            pass
