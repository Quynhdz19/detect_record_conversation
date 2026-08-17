"""TalkNet-ASD (TalkSet) — who in the camera is actually talking."""

from __future__ import annotations

import logging
import sys
import threading
import time
import urllib.request
from collections import deque
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
import python_speech_features
import torch
import torch.nn as nn

from third_party.talknet.loss import lossAV
from third_party.talknet.model.talkNetModel import talkNetModel

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHT_PATH = PROJECT_ROOT / "checkpoints" / "TalkNet_ASD" / "pretrain_TalkSet.model"
WEIGHT_URL = (
    "https://huggingface.co/AgnisOS/2ndbrain-models/resolve/main/talknet/pretrain_TalkSet.model"
)

VIS_FPS = 25
MFCC_FPS = 100
SAMPLE_RATE = 16000
WINDOW_SEC = 1.2
MIN_SEC = 1.0
SCORE_EVERY = 0.28
SPEAK_THRESH = 0.0  # official Columbia eval: score > 0


def pick_asd_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # Conv3d frontend is unreliable on MPS; CPU is safer on Mac.
    return "cpu"


class _TalkNetBundle(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = talkNetModel()
        self.lossAV = lossAV()


def _ensure_weights() -> Path:
    WEIGHT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if WEIGHT_PATH.exists() and WEIGHT_PATH.stat().st_size > 1_000_000:
        return WEIGHT_PATH
    logger.info("Downloading TalkNet TalkSet weights…")
    tmp = WEIGHT_PATH.with_suffix(".part")
    urllib.request.urlretrieve(WEIGHT_URL, tmp)
    tmp.replace(WEIGHT_PATH)
    return WEIGHT_PATH


def _load_bundle(device: str) -> _TalkNetBundle:
    path = _ensure_weights()
    bundle = _TalkNetBundle()
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    own = bundle.state_dict()
    loaded = 0
    for name, param in raw.items():
        key = name.replace("module.", "")
        if key in own and own[key].shape == param.shape:
            own[key].copy_(param)
            loaded += 1
    bundle.load_state_dict(own)
    bundle.to(device)
    bundle.eval()
    logger.info("TalkNet loaded (%s keys) on %s", loaded, device)
    return bundle


def _face_to_112(face_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
    return gray[56:168, 56:168].copy()


class TalkNetASD:
    """Sliding-window TalkNet: 25 fps face + 16 kHz MFCC → speaking score."""

    def __init__(self):
        self.device = pick_asd_device()
        self.bundle = _load_bundle(self.device)
        self._lock = threading.Lock()
        self._faces: deque[np.ndarray] = deque(maxlen=int(VIS_FPS * 3))
        self._pcm = bytearray()
        self._last_infer = 0.0
        self._infer_busy = False
        self.last_score = 0.0
        self.speaking = False
        self._hold = 0
        self.ready = True

    def push_face(self, face_bgr: np.ndarray) -> None:
        if face_bgr is None or face_bgr.size == 0:
            return
        with self._lock:
            self._faces.append(_face_to_112(face_bgr))

    def push_pcm16(self, pcm: bytes) -> None:
        if not pcm:
            return
        with self._lock:
            self._pcm.extend(pcm)
            max_b = SAMPLE_RATE * 3 * 2
            if len(self._pcm) > max_b:
                del self._pcm[: len(self._pcm) - max_b]

    def maybe_update(self, now: float | None = None) -> float:
        t = time.time() if now is None else now
        with self._lock:
            if self._infer_busy or t - self._last_infer < SCORE_EVERY:
                return self.last_score
            n_face = len(self._faces)
            pcm_sec = len(self._pcm) / (SAMPLE_RATE * 2)
            if n_face < 10 or pcm_sec < MIN_SEC:
                return self.last_score
            faces = list(self._faces)
            pcm = bytes(self._pcm)
            self._last_infer = t
            self._infer_busy = True
        threading.Thread(target=self._bg_infer, args=(faces, pcm), daemon=True).start()
        return self.last_score

    def _bg_infer(self, faces: list[np.ndarray], pcm: bytes) -> None:
        try:
            score = float(self._infer(faces, pcm))
            with self._lock:
                self.last_score = score
                if score > SPEAK_THRESH:
                    self._hold = 4
                    self.speaking = True
                elif self._hold > 0:
                    self._hold -= 1
                    self.speaking = True
                else:
                    self.speaking = False
        except Exception:
            logger.exception("TalkNet infer failed")
        finally:
            with self._lock:
                self._infer_busy = False

    def _infer(self, faces: list[np.ndarray], pcm: bytes) -> float:
        audio = np.frombuffer(pcm, dtype=np.int16)
        take = int(WINDOW_SEC * SAMPLE_RATE)
        if audio.size > take:
            audio = audio[-take:]
        mfcc = python_speech_features.mfcc(
            audio, samplerate=SAMPLE_RATE, numcep=13, winlen=0.025, winstep=0.010
        )
        t_audio = mfcc.shape[0] // 4
        need = min(int(WINDOW_SEC * VIS_FPS), t_audio, max(int(MIN_SEC * VIS_FPS), 25))
        if need < 25 or mfcc.shape[0] < need * 4:
            return self.last_score

        # Repeat/interpolate our ~12 fps crops up to 25 fps
        idx = np.linspace(0, len(faces) - 1, need)
        vis = np.stack([faces[int(round(i))] for i in idx], axis=0).astype(np.float32)
        mfcc = mfcc[: need * 4].astype(np.float32)

        audio_t = torch.from_numpy(mfcc).unsqueeze(0).to(self.device)
        vis_t = torch.from_numpy(vis).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            embed_a = self.bundle.model.forward_audio_frontend(audio_t)
            embed_v = self.bundle.model.forward_visual_frontend(vis_t)
            embed_a, embed_v = self.bundle.model.forward_cross_attention(embed_a, embed_v)
            out = self.bundle.model.forward_audio_visual_backend(embed_a, embed_v)
            scores = self.bundle.lossAV(out)
        # Use the last ~0.3s of scores (more responsive)
        arr = np.asarray(scores).reshape(-1)
        tail = arr[-8:] if arr.size else arr
        return float(np.mean(tail)) if tail.size else 0.0


_asd: TalkNetASD | None = None
_asd_lock = threading.Lock()


def talknet_ready() -> bool:
    return _asd is not None and bool(getattr(_asd, "ready", False))


@lru_cache(maxsize=1)
def get_talknet() -> TalkNetASD:
    global _asd
    with _asd_lock:
        if _asd is None:
            _asd = TalkNetASD()
        return _asd
