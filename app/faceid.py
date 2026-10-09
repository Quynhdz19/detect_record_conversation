"""Who is in front of the camera: ArcFace (InsightFace buffalo_sc w600k_mbf) on MediaPipe landmarks."""

from __future__ import annotations

import io
import logging
import os
import threading
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL = Path(__file__).resolve().parents[1] / "checkpoints" / "arcface" / "w600k_mbf.onnx"
MODEL_ZIP = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"
# ArcFace 112×112 template: left eye, right eye, nose, left mouth, right mouth (image coordinates).
_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)
# MediaPipe indices for those five points; eyes use the mean of both corners.
_EYE_L, _EYE_R, _NOSE, _MOUTH_L, _MOUTH_R = (33, 133), (362, 263), 1, 61, 291
# MobileFaceNet: same person ~0.5–0.8 across frames, strangers mostly below 0.3.
SAME_FACE = float(os.environ.get("SAME_FACE", "0.40"))


def five_points(face, w: int, h: int) -> np.ndarray:
    def at(i: int) -> tuple[float, float]:
        return face[i].x * w, face[i].y * h

    eye_l = np.mean([at(i) for i in _EYE_L], axis=0)
    eye_r = np.mean([at(i) for i in _EYE_R], axis=0)
    return np.array([eye_l, eye_r, at(_NOSE), at(_MOUTH_L), at(_MOUTH_R)], dtype=np.float32)


def _ensure_model() -> Path:
    if MODEL.exists():
        return MODEL
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading ArcFace buffalo_sc…")
    with urllib.request.urlopen(MODEL_ZIP) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extract("w600k_mbf.onnx", MODEL.parent)
    return MODEL


class FaceEmbedder:
    def __init__(self) -> None:
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(_ensure_model()), providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self._lock = threading.Lock()
        logger.info("ArcFace ready")

    def embed(self, bgr: np.ndarray, pts5: np.ndarray) -> np.ndarray | None:
        mat = cv2.estimateAffinePartial2D(pts5, _TEMPLATE, method=cv2.LMEDS)[0]
        if mat is None:
            return None
        crop = cv2.warpAffine(bgr, mat, (112, 112), borderValue=0)
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        with self._lock:
            vec = self.sess.run(None, {self.input: blob})[0][0]
        return vec / (np.linalg.norm(vec) + 1e-8)


@lru_cache(maxsize=1)
def get_face_embedder() -> FaceEmbedder:
    return FaceEmbedder()


class FaceBook:
    """Per-session list of people seen, so A coming back after B is still A."""

    def __init__(self) -> None:
        self.people: list[tuple[str, np.ndarray]] = []

    def identify(self, vec: np.ndarray) -> str:
        if self.people:
            scores = [float(np.dot(vec, emb)) for _, emb in self.people]
            best = int(np.argmax(scores))
            if scores[best] >= SAME_FACE:
                name, emb = self.people[best]
                mixed = 0.9 * emb + 0.1 * vec
                self.people[best] = (name, mixed / (np.linalg.norm(mixed) + 1e-8))
                return name
        name = f"Người {len(self.people) + 1}"
        self.people.append((name, vec))
        logger.info("new face %s", name)
        return name
