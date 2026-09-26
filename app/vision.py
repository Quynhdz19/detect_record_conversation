"""Face crop + visual/audio talking detection (V-VAD, not raw lip jitter)."""

from __future__ import annotations

import time
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
NOSE_TIP = 1
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
FRONTAL_MIN = 0.42
# ClearVoice AV-TSE / TalkNet demo: square face crop, pad 0.40, then 224×224.
CROP_SCALE = 0.40

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"

# Speech lips oscillate; a still/smiling mouth does not.
_TALK_BLENDS = (
    "mouthFunnel",
    "mouthPucker",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
)


@dataclass
class FaceCue:
    found: bool
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    mouth_open: float = 0.0
    speaking: bool = False
    lip_active: bool = False
    asd_score: float = 0.0
    cx: float = 0.5
    cy: float = 0.5
    frontal: bool = False
    frontal_score: float = 0.0
    face_bgr: Optional[np.ndarray] = field(default=None, repr=False)


def _blend_map(categories) -> dict[str, float]:
    out: dict[str, float] = {}
    for cat in categories or []:
        name = getattr(cat, "category_name", None) or getattr(cat, "display_name", "")
        out[str(name)] = float(getattr(cat, "score", 0.0))
    return out


def _face_box(face) -> tuple[float, float, float, float]:
    xs = [lm.x for lm in face]
    ys = [lm.y for lm in face]
    return min(xs), min(ys), max(xs), max(ys)


def _frontal_score(face) -> float:
    """1 = nhìn thẳng camera, 0 = nghiêng / nghiêng vai."""
    left, right, nose = face[LEFT_EYE_OUTER], face[RIGHT_EYE_OUTER], face[NOSE_TIP]
    eye_w = abs(right.x - left.x) + 1e-6
    yaw = abs(nose.x - 0.5 * (left.x + right.x)) / eye_w
    x0, y0, x1, y1 = _face_box(face)
    aspect = (x1 - x0) / max(y1 - y0, 1e-6)
    yaw_score = max(0.0, 1.0 - yaw / 0.45)
    aspect_score = min(1.0, aspect / 0.72)
    return float(0.72 * yaw_score + 0.28 * aspect_score)


def _pick_opposite_face(faces) -> int:
    """Người đối diện: mặt to, nhìn cam, gần giữa khung."""
    best_i, best = 0, -1.0
    for i, face in enumerate(faces):
        x0, y0, x1, y1 = _face_box(face)
        area = max(1e-6, (x1 - x0) * (y1 - y0))
        front = _frontal_score(face)
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        center = 1.0 - min(1.0, ((cx - 0.5) ** 2 + (cy - 0.48) ** 2) ** 0.5)
        score = area * (0.3 + 0.7 * front) * (0.75 + 0.25 * center)
        if score > best:
            best, best_i = score, i
    return best_i


def crop_face_clearvoice(
    bgr: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    crop_scale: float = CROP_SCALE,
) -> Optional[np.ndarray]:
    """Square 224 crop used by ClearVoice video_process.crop_video / TalkNet demo."""
    h, w = bgr.shape[:2]
    cx = 0.5 * (x0 + x1) * w
    cy = 0.5 * (y0 + y1) * h
    s = max((x1 - x0) * w, (y1 - y0) * h) / 2.0
    if s < 4:
        return None
    bsi = max(1, int(s * (1 + 2 * crop_scale)))
    padded = np.pad(bgr, ((bsi, bsi), (bsi, bsi), (0, 0)), mode="constant", constant_values=110)
    mx, my = cx + bsi, cy + bsi
    face = padded[
        int(my - s) : int(my + s * (1 + 2 * crop_scale)),
        int(mx - s * (1 + crop_scale)) : int(mx + s * (1 + crop_scale)),
    ]
    if face.size == 0 or min(face.shape[:2]) < 8:
        return None
    return cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)


def _zero_crossings(arr: np.ndarray) -> int:
    if arr.size < 3:
        return 0
    signs = np.sign(arr)
    signs[signs == 0] = 1
    return int(np.sum(np.abs(np.diff(signs)) > 0))


class FaceMouthTracker:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing face model at {MODEL_PATH}. Run download step first."
            )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=3,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._ts_ms = 0
        self._act_ema = 0.0
        self._hp_hist: list[float] = []
        self._hold = 0
        self._prev_mouth: Optional[np.ndarray] = None
        self._prev_cheek: Optional[np.ndarray] = None
        self._last_loud_at = 0.0
        self._last_audio_at = 0.0
        self._speak_hold = 0
        self._box_ema: Optional[np.ndarray] = None

    def close(self) -> None:
        self._landmarker.close()

    def note_audio(self, peak: float, now: Optional[float] = None) -> None:
        """Call on every mic chunk so talking = lips + voice, not just a still grin."""
        t = time.time() if now is None else now
        self._last_audio_at = t
        if peak >= 0.035:
            self._last_loud_at = t

    def note_pcm16(self, pcm: bytes, now: Optional[float] = None) -> None:
        if len(pcm) < 2:
            self.note_audio(0.0, now)
            return
        peak = float(np.max(np.abs(np.frombuffer(pcm, dtype=np.int16)))) / 32768.0
        self.note_audio(peak, now)
        try:
            from app.asd import get_talknet, talknet_ready

            if talknet_ready():
                get_talknet().push_pcm16(pcm)
        except Exception:
            pass

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
        self._ts_ms = max(self._ts_ms + 1, int(time.time() * 1000))
        result = self._landmarker.detect_for_video(mp_image, self._ts_ms)
        if not result.face_landmarks:
            self._reset_motion()
            return FaceCue(found=False)

        best_i = _pick_opposite_face(result.face_landmarks)
        best = result.face_landmarks[best_i]
        frontal_score = _frontal_score(best)
        frontal = frontal_score >= FRONTAL_MIN
        xs = [lm.x for lm in best]
        ys = [lm.y for lm in best]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))

        # Smooth bbox like ClearVoice medfilt(13) so TSE/TalkNet see a stable track.
        cx_px = 0.5 * (x0 + x1) * w
        cy_px = 0.5 * (y0 + y1) * h
        half = max((x1 - x0) * w, (y1 - y0) * h) / 2.0
        vec = np.array([cx_px, cy_px, half], dtype=np.float32)
        if self._box_ema is None:
            self._box_ema = vec
        else:
            self._box_ema = 0.65 * self._box_ema + 0.35 * vec
        cx_s, cy_s, s_s = (float(v) for v in self._box_ema)
        x0s = (cx_s - s_s) / max(w, 1)
        x1s = (cx_s + s_s) / max(w, 1)
        y0s = (cy_s - s_s) / max(h, 1)
        y1s = (cy_s + s_s) / max(h, 1)
        face_bgr = crop_face_clearvoice(bgr, x0s, y0s, x1s, y1s)

        up, lo = best[UPPER_LIP], best[LOWER_LIP]
        left, right = best[MOUTH_LEFT], best[MOUTH_RIGHT]
        mouth_w = max(abs(right.x - left.x), 1e-6)
        mouth_open = abs(lo.y - up.y) / mouth_w

        blends = {}
        if result.face_blendshapes and len(result.face_blendshapes) > best_i:
            blends = _blend_map(result.face_blendshapes[best_i])
        activity = self._talk_activity(blends, mouth_open)
        flow = self._mouth_flow(bgr, best, w, h)
        lip_active = self._visual_talking(activity, flow)

        now = time.time()
        audio_voice = (now - self._last_loud_at) < 0.55
        asd_score = 0.0
        has_asd = False
        talknet_speaking = False
        try:
            from app.asd import get_talknet, talknet_ready

            if talknet_ready() and face_bgr is not None:
                net = get_talknet()
                net.push_face(face_bgr)
                asd_score = net.maybe_update(now)
                has_asd = bool(getattr(net, "has_score", False))
                talknet_speaking = bool(getattr(net, "speaking", False))
        except Exception:
            pass
        # TalkNet on a webcam often stays <=0 during real speech (sync lag).
        # A strong negative score still means the loud voice is someone else.
        # Otherwise lips are enough to open the turn; audio stays continuous after that.
        other_voice = has_asd and asd_score < -0.45 and not lip_active
        if not frontal or other_voice:
            self._speak_hold = 0
        elif (lip_active or talknet_speaking) and audio_voice:
            self._speak_hold = 12
        elif self._speak_hold > 0:
            self._speak_hold -= 1
        speaking = frontal and audio_voice and self._speak_hold > 0 and not other_voice

        return FaceCue(
            found=True,
            x=x0,
            y=y0,
            w=x1 - x0,
            h=y1 - y0,
            mouth_open=mouth_open,
            speaking=speaking,
            lip_active=lip_active,
            asd_score=asd_score,
            cx=cx,
            cy=cy,
            frontal=frontal,
            frontal_score=frontal_score,
            face_bgr=face_bgr,
        )

    def _reset_motion(self) -> None:
        self._hp_hist.clear()
        self._hold = 0
        self._act_ema = 0.0
        self._speak_hold = 0
        self._prev_mouth = None
        self._prev_cheek = None
        self._box_ema = None

    @staticmethod
    def _talk_activity(blends: dict[str, float], mouth_open: float) -> float:
        jaw = blends.get("jawOpen", 0.0)
        artic = sum(blends.get(k, 0.0) for k in _TALK_BLENDS)
        smile = 0.5 * (blends.get("mouthSmileLeft", 0.0) + blends.get("mouthSmileRight", 0.0))
        base = jaw + 0.55 * artic if blends else float(mouth_open)
        return max(0.0, base - 0.55 * smile)

    def _mouth_flow(self, bgr: np.ndarray, face, w: int, h: int) -> float:
        """Residual optical flow: mouth minus cheek (cancels head motion)."""
        xs = [face[i].x for i in (UPPER_LIP, LOWER_LIP, MOUTH_LEFT, MOUTH_RIGHT)]
        ys = [face[i].y for i in (UPPER_LIP, LOWER_LIP, MOUTH_LEFT, MOUTH_RIGHT)]
        mx0, mx1 = min(xs), max(xs)
        my0, my1 = min(ys), max(ys)
        pad_x, pad_y = 0.35 * (mx1 - mx0), 0.55 * (my1 - my0)
        def crop(x0, y0, x1, y1) -> Optional[np.ndarray]:
            a, b = int(max(0, x0) * w), int(min(1, x1) * w)
            c, d = int(max(0, y0) * h), int(min(1, y1) * h)
            if b <= a + 4 or d <= c + 4:
                return None
            gray = cv2.cvtColor(bgr[c:d, a:b], cv2.COLOR_BGR2GRAY)
            return cv2.resize(gray, (48, 32), interpolation=cv2.INTER_AREA)

        mouth = crop(mx0 - pad_x, my0 - pad_y, mx1 + pad_x, my1 + pad_y)
        cheek = crop(face[234].x - 0.04, face[234].y - 0.04, face[234].x + 0.08, face[234].y + 0.08)
        if mouth is None:
            return 0.0
        flow_m = self._flow_mag(self._prev_mouth, mouth)
        flow_c = self._flow_mag(self._prev_cheek, cheek) if cheek is not None else 0.0
        self._prev_mouth = mouth
        self._prev_cheek = cheek
        return max(0.0, flow_m - 0.85 * flow_c)

    @staticmethod
    def _flow_mag(prev: Optional[np.ndarray], curr: np.ndarray) -> float:
        if prev is None or prev.shape != curr.shape:
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 2, 15, 2, 5, 1.2, 0)
        return float(np.mean(np.linalg.norm(flow, axis=2)))

    def _visual_talking(self, activity: float, flow: float) -> bool:
        self._act_ema = 0.88 * self._act_ema + 0.12 * activity
        hp = activity - self._act_ema
        self._hp_hist.append(hp)
        if len(self._hp_hist) > 18:
            self._hp_hist.pop(0)
        hist = np.asarray(self._hp_hist, dtype=np.float32)
        if hist.size < 4:
            return False
        zc = _zero_crossings(hist[-14:])
        std = float(np.std(hist[-14:]))
        # Talking: periodic open/close. Still face / webcam jitter stays below this.
        raw = (std >= 0.012 and zc >= 2) or (flow >= 0.28 and std >= 0.01)
        if raw:
            self._hold = 8
            return True
        if self._hold > 0:
            self._hold -= 1
            return True
        return False


_tracker: Optional[FaceMouthTracker] = None


def get_tracker() -> FaceMouthTracker:
    global _tracker
    if _tracker is None or not hasattr(_tracker, "_box_ema"):
        if _tracker is not None:
            try:
                _tracker.close()
            except Exception:
                pass
        _tracker = FaceMouthTracker()
    return _tracker
