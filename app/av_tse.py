"""Audio-Visual Target Speaker Extraction via ClearVoice AV_MossFormer2_TSE_16K."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "AV_MossFormer2_TSE_16K"
VIDEO_FPS = 25
SAMPLE_RATE = 16000
VISUAL_MEAN = 0.4161
VISUAL_STD = 0.1688


def _patch_clearvoice_mps() -> None:
    """ClearVoice hardcodes .cuda() in overlap_and_add; fix for Mac MPS/CPU."""
    try:
        from clearvoice.models.av_mossformer2_tse import av_mossformer2 as mod
    except Exception:
        return

    def overlap_and_add(signal, frame_step):
        import math

        outer_dimensions = signal.size()[:-2]
        frames, frame_length = signal.size()[-2:]
        subframe_length = math.gcd(frame_length, frame_step)
        subframe_step = frame_step // subframe_length
        subframes_per_frame = frame_length // subframe_length
        output_size = frame_step * (frames - 1) + frame_length
        output_subframes = output_size // subframe_length
        subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)
        frame = torch.arange(0, output_subframes, device=signal.device).unfold(
            0, subframes_per_frame, subframe_step
        )
        frame = frame.contiguous().view(-1)
        result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
        result.index_add_(-2, frame, subframe_signal)
        result = result.view(*outer_dimensions, -1)
        return result

    mod.overlap_and_add = overlap_and_add


class AVTargetSpeakerExtractor:
    """Lip/face-conditioned TSE using ClearVoice AV_MossFormer2_TSE_16K."""

    def __init__(self):
        # ClearVoice resolves checkpoints relative to CWD
        os.chdir(PROJECT_ROOT)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        _patch_clearvoice_mps()

        from clearvoice import ClearVoice

        logger.info("Loading AV_MossFormer2_TSE_16K (may download ~hundreds MB)...")
        clear = ClearVoice(
            task="target_speaker_extraction",
            model_names=["AV_MossFormer2_TSE_16K"],
        )
        self.wrapper = clear.models[0]
        self.model = self.wrapper.model
        self.args = self.wrapper.args
        self.device = self.wrapper.device
        self.model.eval()
        logger.info("AV-TSE ready on %s", self.device)

    @staticmethod
    def face_bgr_to_visual_frame(face_bgr: np.ndarray) -> np.ndarray:
        """Match ClearVoice preprocessing: gray 224 → center 112 → normalize."""
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_AREA)
        face = gray[56:168, 56:168]  # center 112x112
        visual = face.astype(np.float32) / 255.0
        visual = (visual - VISUAL_MEAN) / VISUAL_STD
        return visual

    def prepare_visual(
        self, face_crops_bgr: list[np.ndarray], n_audio_samples: int
    ) -> np.ndarray:
        target_frames = max(1, int(round(n_audio_samples / SAMPLE_RATE * VIDEO_FPS)))
        if not face_crops_bgr:
            # neutral face placeholder (zeros after normalize still ok-ish; prefer last)
            frames = [np.zeros((112, 112), dtype=np.float32)]
        else:
            frames = [self.face_bgr_to_visual_frame(f) for f in face_crops_bgr]

        # Resample frame list to exact target length
        idx = np.linspace(0, len(frames) - 1, target_frames)
        visual = np.stack([frames[int(round(i))] for i in idx], axis=0)
        return visual[np.newaxis, ...].astype(np.float32)  # [1, T, 112, 112]

    @torch.inference_mode()
    def extract(
        self,
        audio_f32: np.ndarray,
        face_crops_bgr: list[np.ndarray],
    ) -> np.ndarray:
        audio = np.asarray(audio_f32, dtype=np.float32).reshape(-1)
        if audio.size < SAMPLE_RATE:  # <1s
            return audio

        peak = float(np.max(np.abs(audio))) + 1e-8
        audio = audio / peak

        visual = self.prepare_visual(face_crops_bgr, audio.shape[0])
        audio_b = audio[np.newaxis, :]

        # Prefer direct model call for short chunks (demo uses ~3s)
        audio_t = torch.from_numpy(audio_b).to(self.device)
        visual_t = torch.from_numpy(visual).to(self.device)
        try:
            out = self.model(audio_t, visual_t).detach().float().cpu().numpy()
            out = np.squeeze(out)
        except Exception:
            logger.exception("AV-TSE model forward failed; falling back to mixture")
            return (audio * peak).astype(np.float32)

        if out.ndim > 1:
            out = out.reshape(-1)
        # restore rough level
        out = out.astype(np.float32)
        out_peak = float(np.max(np.abs(out))) + 1e-8
        out = out / out_peak * min(peak, 0.95)
        return out


_extractor: Optional[AVTargetSpeakerExtractor] = None


@lru_cache(maxsize=1)
def get_av_tse() -> AVTargetSpeakerExtractor:
    global _extractor
    if _extractor is None:
        _extractor = AVTargetSpeakerExtractor()
    return _extractor
