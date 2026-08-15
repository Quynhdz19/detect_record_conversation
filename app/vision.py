"""Face + mouth activity + face crop from camera frames (AV-TSE cue)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

UPPER_LIP = 13
LOWER_LIP = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"


@dataclass
class FaceCue:
    found: bool
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    mouth_open: float = 0.0
    speaking: bool = False
    cx: float = 0.5
    cy: float = 0.5
    face_bgr: Optional[np.ndarray] = field(default=None, repr=False)


class FaceMouthTracker:
    def __init__(self, speaking_threshold: float = 0.035):
        self.speaking_threshold = speaking_threshold
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing face model at {MODEL_PATH}. Run download step first."
            )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=3,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._mouth_history: list[float] = []

    def close(self) -> None:
        self._landmarker.close()

    def analyze_jpeg(self, jpeg_bytes: bytes) -> FaceCue:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return FaceCue(found=False)
        return self.analyze_bgr(bgr)

    def analyze_bgr(self, bgr: np.ndarray) -> FaceCue:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            self._mouth_history.clear()
            return FaceCue(found=False)

        best = None
        best_dist = 1e9
        for face in result.face_landmarks:
            xs = [lm.x for lm in face]
            ys = [lm.y for lm in face]
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            dist = (cx - 0.5) ** 2 + (cy - 0.5) ** 2
            if dist < best_dist:
                best_dist = dist
                best = face

        assert best is not None
        xs = [lm.x for lm in best]
        ys = [lm.y for lm in best]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))

        # Expand box a bit for AV-TSE face crop (model expects face track ~224)
        pad = 0.25
        bw = x1 - x0
        bh = y1 - y0
        x0e = max(0.0, x0 - pad * bw)
        x1e = min(1.0, x1 + pad * bw)
        y0e = max(0.0, y0 - pad * bh)
        y1e = min(1.0, y1 + pad * bh)
        px0, px1 = int(x0e * w), int(x1e * w)
        py0, py1 = int(y0e * h), int(y1e * h)
        if px1 <= px0 + 4 or py1 <= py0 + 4:
            face_bgr = None
        else:
            face_bgr = bgr[py0:py1, px0:px1].copy()
            face_bgr = cv2.resize(face_bgr, (224, 224), interpolation=cv2.INTER_AREA)

        up = best[UPPER_LIP]
        lo = best[LOWER_LIP]
        left = best[MOUTH_LEFT]
        right = best[MOUTH_RIGHT]
        mouth_w = max(abs(right.x - left.x), 1e-6)
        mouth_open = abs(lo.y - up.y) / mouth_w

        self._mouth_history.append(mouth_open)
        if len(self._mouth_history) > 12:
            self._mouth_history.pop(0)
        variance = float(np.var(self._mouth_history)) if len(self._mouth_history) >= 4 else 0.0
        speaking = mouth_open > self.speaking_threshold or variance > 0.00015

        return FaceCue(
            found=True,
            x=x0,
            y=y0,
            w=x1 - x0,
            h=y1 - y0,
            mouth_open=mouth_open,
            speaking=speaking,
            cx=cx,
            cy=cy,
            face_bgr=face_bgr,
        )


_tracker: Optional[FaceMouthTracker] = None


def get_tracker() -> FaceMouthTracker:
    global _tracker
    if _tracker is None:
        _tracker = FaceMouthTracker()
    return _tracker
