"""Web demo: camera + mic → AV-TSE → PhoWhisper Vietnamese ASR."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time

import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.asr import (
    MODEL_ID,
    clean_transcript,
    get_zipformer,
    looks_like_speech,
    pcm16_to_float32,
    pick_device,
)
from app.av_tse import SAMPLE_RATE, get_av_tse
from app.live import (
    TURN_ENERGY,
    VOICE_PEAK,
    LiveStream,
    MIN_FILE_SEC,
    chunk_is_voiced,
    infer_window,
    speaking_blocked,
)
from app.pi_api import router as pi_router
from app.video_pipeline import TMP_ROOT, process_mp4
from app.vision import get_tracker
from app.denoise import denoise_pcm16
from app.duo import OWNER_ENROLL_SEC, DuoMic
from app.voice import assign_speaker, blend, embed_pcm16, get_extractor, match_voice

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
_face_lock = threading.Lock()

STATIC_DIR = Path(__file__).resolve().parent / "static"
# TalkNet needs ~1 s of lips before it confirms speech; keep that audio so the first word survives.
PREROLL_SEC = 1.0


def _talknet_says_speaking() -> bool:
    """Lips must match the sound. A still face next to a TV scores below zero."""
    from app.asd import get_talknet, talknet_ready

    if not talknet_ready():
        return False
    net = get_talknet()
    return bool(net.has_score and net.speaking)

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
        BOOT["stage"] = "loading Zipformer-30M (CPU)"
        logger.info("Warming %s", MODEL_ID)
        get_zipformer()
        BOOT["asr_ready"] = True
        try:
            BOOT["stage"] = "loading speaker embedding"
            get_extractor()
        except Exception:
            logger.exception("Speaker embedding unavailable; A/B voice filter off")
        try:
            BOOT["stage"] = "loading GTCRN denoiser"
            from app.denoise import StreamDenoiser

            StreamDenoiser()
        except Exception:
            logger.exception("Denoiser unavailable; stream stays noisy")

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
    duo = DuoMic()
    owner_shown: tuple[str, bool, int] | None = None
    from app.faceid import FaceBook

    faces = FaceBook()
    face_person = ""
    last_face_emb = None
    last_face: dict[str, Any] = {"found": False, "lip_active": False}
    last_lip_at = 0.0
    mouth_open = False
    preroll: list[np.ndarray] = []
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

    async def emit_silence() -> None:
        now = time.time()
        if live.should_emit_silence(now):
            await websocket.send_json({"type": "status", "text": "silence"})

    def kick_lane(label: str, final: bool) -> None:
        row = live.lanes.get(label)
        if row is None or live.busy or row.busy:
            return
        if row.buf_sec() < (0.8 if final else MIN_FILE_SEC) or row.last_voice_at <= 0:
            return
        live.busy = True
        row.busy = True
        row.last_decode_at = time.time()
        asyncio.create_task(process_lane(label, final))

    async def process_lane(label: str, final: bool) -> None:
        row = live.lanes.get(label)
        try:
            if row is None or not BOOT["asr_ready"]:
                return
            await websocket.send_json({"type": "status", "text": f"Tách giọng {label}…"})
            chunk, _crops = row.snapshot()
            # Voice lanes have no face crop. Transcribe the slice already assigned to this person.
            text, used_tse = await loop.run_in_executor(
                None,
                infer_window,
                chunk,
                [],
                False,
                final,
                final,
                True,
                False,
            )
            text = clean_transcript(text, final=final, allow_greetings=final)
            if text and len(live.voiceprints) >= 2:
                owner = await loop.run_in_executor(
                    None, match_voice, chunk, dict(live.voiceprints)
                )
                if owner and owner != label:
                    logger.info("drop %s subtitle; voice is %s", label, owner)
                    text = ""
            if text:
                logger.info("subtitle %s final=%s %r", label, final, text)
                await websocket.send_json(
                    {
                        "type": "transcript",
                        "text": text,
                        "final": final,
                        "used_tse": used_tse,
                        "speaker": label,
                        "face": last_face,
                    }
                )
            if final:
                row.commit()
        except Exception:
            logger.exception("lane decode failed %s", label)
        finally:
            if row is not None:
                row.busy = False
            live.busy = False

    def kick_decode(final: bool) -> None:
        # Never await ASR on the receive path — mic chunks must keep landing in the buffer.
        if live.busy or live.buf_sec() < MIN_FILE_SEC or live.last_voice_at <= 0:
            return
        if require_speaking and not live.saw_target:
            return
        live.busy = True
        live.last_decode_at = time.time()
        asyncio.create_task(process_chunk(final, live.speaker))

    async def process_chunk(
        final: bool = False,
        speaker: str = "",
        pcm: bytes | None = None,
        crops_in: list | None = None,
    ) -> None:
        try:
            if not BOOT["asr_ready"]:
                await websocket.send_json({"type": "status", "text": "ASR chưa sẵn sàng…"})
                return
            who = speaker or live.speaker or ""
            await websocket.send_json(
                {"type": "status", "text": f"Đang nghe người {who}…" if who else "Đang nghe…"}
            )
            owns_buffer = pcm is None
            if owns_buffer:
                chunk, crops = live.snapshot()
            else:
                chunk, crops = pcm, list(crops_in or [])
            allow_greetings = final and live.last_voice_at > 0
            # Buffer is already the confirmed turn, so raw is safe if TSE
            # smears a single talker (common at normal speed, little overlap).
            fallback_raw = (not require_speaking) or live.saw_target or bool(who)
            separate = len(live.voiceprints) >= 2
            text, used_tse = await loop.run_in_executor(
                None,
                infer_window,
                chunk,
                crops,
                True,
                final,
                allow_greetings,
                fallback_raw,
                separate,
            )
            if text and separate:
                owner = await loop.run_in_executor(
                    None, match_voice, chunk, dict(live.voiceprints)
                )
                if owner and who and owner != who:
                    logger.info("drop text for %s; voice matches %s", who, owner)
                    text = ""
            text = clean_transcript(text, final=final, allow_greetings=allow_greetings)
            if text:
                live.had_speech = True
                live.last_text = text
                logger.info(
                    "send transcript final=%s buf=%.2fs %r",
                    final,
                    live.buf_sec(),
                    text,
                )
                await websocket.send_json(
                    {
                        "type": "transcript",
                        "text": text,
                        "final": final,
                        "used_tse": used_tse,
                        "speaker": who,
                        "face": last_face,
                    }
                )
            else:
                logger.info("live ASR empty final=%s buf=%.2fs", final, live.buf_sec())
            if final and owns_buffer:
                live.commit()
            if text:
                await websocket.send_json({"type": "status", "text": "Đang nghe…"})
            else:
                await emit_silence()
        except Exception:
            logger.exception("live decode failed")
        finally:
            live.busy = False
            if live.pending_speaker and live.pending_speaker != live.speaker:
                live.speaker = live.pending_speaker
            live.pending_speaker = ""

    async def _route_voice(sample: bytes, now: float) -> None:
        try:
            try:
                clean = await loop.run_in_executor(None, denoise_pcm16, sample)
            except Exception:
                logger.exception("denoise failed; using noisy audio")
                clean = sample
            audio = pcm16_to_float32(clean)
            if not looks_like_speech(audio):
                return
            vec = await loop.run_in_executor(None, embed_pcm16, clean)
            if vec is None:
                return
            label = assign_speaker(vec, live.voiceprints)
            live.voiceprints[label] = blend(live.voiceprints.get(label), vec)
            live.lane(label).push(clean, now, lips=True)
            logger.info("routed audio to %s (%d voices)", label, len(live.voiceprints))
            await websocket.send_json(
                {
                    "type": "voices",
                    "labels": sorted(live.voiceprints),
                    "active": label,
                }
            )
            for who, is_final in live.due_lanes(time.time()):
                kick_lane(who, is_final)
        except Exception:
            logger.exception("voice route failed")
        finally:
            live.enroll_busy = False

    async def _remember_voice(label: str, sample: bytes) -> None:
        try:
            vec = await loop.run_in_executor(None, embed_pcm16, sample)
            if vec is None:
                return
            live.voiceprints[label] = blend(live.voiceprints.get(label), vec)
            logger.info("remembered voice %s (%d enrolled)", label, len(live.voiceprints))
        except Exception:
            logger.exception("voice enroll failed")
        finally:
            live.enroll_busy = False

    async def maybe_decode() -> None:
        now = time.time()
        if not last_face.get("found"):
            await emit_silence()
            return
        if live.idle_silence():
            live.trim_to(0.45)
            await emit_silence()
            return
        blocked = speaking_blocked(require_speaking, last_face)
        if live.want_len_final() or live.want_silence_final(now):
            kick_decode(True)
        elif blocked and live.last_voice_at <= 0:
            return
        elif live.want_partial(now):
            kick_decode(False)

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if message.get("text") is not None:
                data = json.loads(message["text"])
                msg_type = data.get("type")
                if msg_type == "config":
                    require_speaking = bool(data.get("require_speaking", True))
                    await websocket.send_json(
                        {"type": "config_ok", "require_speaking": require_speaking}
                    )
                elif msg_type == "flush":
                    for text, is_final, speaker in duo.flush():
                        text = clean_transcript(text, final=is_final, allow_greetings=True)
                        if text:
                            await websocket.send_json(
                                {
                                    "type": "transcript",
                                    "text": text,
                                    "final": True,
                                    "used_tse": False,
                                    "speaker": speaker,
                                }
                            )
                elif msg_type == "reset_owner":
                    duo.reset_owner()
                    owner_shown = None
                continue

            raw = message.get("bytes")
            if raw is None or len(raw) < 2:
                continue
            tag, payload = raw[0], raw[1:]

            if tag == 1:
                def _see(jpeg: bytes = payload):
                    with _face_lock:
                        return get_tracker().analyze_jpeg(jpeg)

                try:
                    cue = await loop.run_in_executor(None, _see)
                except Exception:
                    logger.exception("face frame failed")
                    continue
                if cue.found and cue.face_emb is not None and cue.face_emb is not last_face_emb:
                    last_face_emb = cue.face_emb
                    face_person = faces.identify(cue.face_emb)
                synced = _talknet_says_speaking()
                talking = bool(cue.found and cue.lip_active and synced)
                if talking:
                    last_lip_at = time.time()
                last_face = {
                    "found": bool(cue.found),
                    "lip_active": talking,
                    "speaking": talking,
                    "asd_score": round(float(cue.asd_score), 2),
                    "x": cue.x,
                    "y": cue.y,
                    "w": cue.w,
                    "h": cue.h,
                    "speaker": cue.speaker or "",
                    "person": face_person if cue.found else "",
                }
                people = []
                for person in get_tracker().people:
                    people.append(
                        {
                            "speaker": person.get("speaker") or "",
                            "x": person.get("x", 0),
                            "y": person.get("y", 0),
                            "w": person.get("w", 0),
                            "h": person.get("h", 0),
                            "speaking": bool(person.get("lip_active")),
                            "lip_active": bool(person.get("lip_active")),
                            "frontal": bool(person.get("frontal", True)),
                        }
                    )
                await websocket.send_json(
                    {"type": "face", "people": people, **last_face}
                )
                continue
            if tag != 2:
                continue

            audio = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
            try:
                get_tracker().note_pcm16(payload)
            except Exception:
                logger.exception("TalkNet audio feed failed")
            # Only the mouth in frame may open the mic. A closed mouth drops the sound.
            lips_now = last_lip_at > 0 and (time.time() - last_lip_at) < 0.45
            if lips_now:
                if not mouth_open:
                    logger.info("mouth gate open (asd=%.2f)", last_face.get("asd_score", 0.0))
                    head = preroll + [audio]
                    audio = np.concatenate(head)
                mouth_open = True
                preroll = []

                def _hear(chunk: np.ndarray = audio, who: str = face_person) -> list:
                    duo.set_person(who)
                    return duo.accept(chunk)

                events = await loop.run_in_executor(None, _hear)
            elif mouth_open:
                mouth_open = False
                logger.info("mouth gate closed")
                events = await loop.run_in_executor(None, duo.flush)
            else:
                preroll.append(audio)
                while sum(p.size for p in preroll) > int(SAMPLE_RATE * PREROLL_SEC):
                    preroll.pop(0)
                continue
            owner_now = (duo.person, duo.owner_ready, round(duo.owner_sec))
            if owner_now != owner_shown:
                owner_shown = owner_now
                who = duo.person or "người này"
                label = (
                    f"{who}: giọng đã đăng ký"
                    if duo.owner_ready
                    else f"{who}: đang đăng ký giọng {duo.owner_sec:.0f}/{OWNER_ENROLL_SEC:.0f}s"
                )
                await websocket.send_json({"type": "status", "text": label})
            for text, is_final, speaker in events:
                text = clean_transcript(text, final=is_final, allow_greetings=is_final)
                if not text:
                    if is_final:
                        await websocket.send_json(
                            {
                                "type": "transcript",
                                "text": "",
                                "final": True,
                                "used_tse": False,
                                "speaker": speaker,
                            }
                        )
                    continue
                await websocket.send_json(
                    {
                        "type": "transcript",
                        "text": text,
                        "final": is_final,
                        "used_tse": False,
                        "speaker": speaker,
                    }
                )

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception:
        logger.exception("WebSocket session error")
        try:
            await websocket.close()
        except Exception:
            pass
