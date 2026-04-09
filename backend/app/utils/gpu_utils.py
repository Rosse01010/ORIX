"""
gpu_utils.py
────────────
Factory for face detector + embedder.

Default stack:
  Detector : InsightFace SCRFD-10G  — fastest crowd detector, handles 100+ faces
  Embedder : InsightFace ArcFace R100 — most accurate embedding (512-dim)
  Alignment: 5-point landmark warp before embedding (built into InsightFace)

Fallback:
  MediaPipe BlazeFace + pixel-mean embedder (CPU, no GPU required)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol

import numpy as np

from app.config import settings
from app.utils.logging_utils import get_logger

log = get_logger(__name__)


# ── Shared data types ──────────────────────────────────────────────────────────

@dataclass
class FaceDetection:
    bbox: List[int]           # [x, y, w, h] absolute pixels
    crop: np.ndarray          # aligned face chip, RGB (112×112 for ArcFace)
    det_score: float          # detector confidence 0–1
    kps: Optional[np.ndarray] = None  # 5 keypoints [[x,y], ...]


class FaceDetector(Protocol):
    def detect(self, frame: np.ndarray) -> List[FaceDetection]: ...


class FaceEmbedder(Protocol):
    def embed(self, aligned_crop: np.ndarray) -> List[float]: ...


# ── InsightFace ────────────────────────────────────────────────────────────────

class InsightFaceDetector:
    """
    Uses FaceAnalysis from InsightFace which bundles:
      - SCRFD-10G-KPS  → detection + 5-point landmarks in one pass
      - ArcFace R100   → embedding (built into buffalo_l pack)

    det_size=(1280, 1280) maximises crowd coverage; lower for speed.
    """

    def __init__(self) -> None:
        from insightface.app import FaceAnalysis

        providers = (
            settings.onnx_provider_list
            if settings.use_gpu
            else ["CPUExecutionProvider"]
        )
        self._app = FaceAnalysis(
            name=settings.insightface_model,   # buffalo_l = SCRFD + ArcFaceR100
            root=settings.model_dir,
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )
        ctx_id = settings.gpu_device_id if settings.use_gpu else -1
        # Larger det_size = more faces detected in crowd (at cost of speed)
        self._app.prepare(ctx_id=ctx_id, det_size=(1280, 1280))
        log.info(
            "insightface_ready",
            model=settings.insightface_model,
            ctx_id=ctx_id,
            det_size="1280x1280",
        )

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self._app.get(rgb)

        results: List[FaceDetection] = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            w, h = x2 - x1, y2 - y1
            # Aligned 112×112 face chip (InsightFace warps it automatically)
            crop = getattr(f, "normed_embedding", None)
            aligned = _get_aligned_chip(rgb, f, x1, y1, x2, y2)
            kps = f.kps if hasattr(f, "kps") and f.kps is not None else None

            results.append(FaceDetection(
                bbox=[x1, y1, w, h],
                crop=aligned,
                det_score=float(f.det_score),
                kps=kps,
            ))
        return results


def _get_aligned_chip(
    rgb: np.ndarray, face, x1: int, y1: int, x2: int, y2: int
) -> np.ndarray:
    """
    Return the 5-point aligned chip if available, else plain crop.
    InsightFace stores the aligned image internally but doesn't expose
    it directly — we crop the RGB frame as fallback.
    """
    import cv2
    if hasattr(face, "kps") and face.kps is not None:
        try:
            from insightface.utils import face_align
            chip = face_align.norm_crop(rgb, landmark=face.kps, image_size=112)
            return chip  # already RGB 112×112
        except Exception:
            pass
    # Fallback: plain bounding box crop resized to 112×112
    crop = rgb[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return np.zeros((112, 112, 3), dtype=np.uint8)
    return cv2.resize(crop, (112, 112))


class InsightFaceEmbedder:
    """
    ArcFace R100 embedding extracted from the FaceAnalysis pipeline.
    Produces 512-dim L2-normalised vectors.
    """

    def __init__(self, detector: InsightFaceDetector) -> None:
        self._app = detector._app

    def embed(self, aligned_crop: np.ndarray) -> List[float]:
        """
        aligned_crop: RGB 112×112 numpy array (output of _get_aligned_chip).
        Runs ArcFace R100 on the chip and returns a 512-dim unit vector.
        """
        import cv2
        faces = self._app.get(aligned_crop)
        if faces:
            emb = faces[0].normed_embedding
            if emb is not None:
                return emb.tolist()
        # Direct model call as fallback
        try:
            rec = self._app.models.get("recognition")
            if rec:
                bgr = cv2.cvtColor(aligned_crop, cv2.COLOR_RGB2BGR)
                emb = rec.get_feat(bgr)
                norm = np.linalg.norm(emb)
                return (emb / (norm + 1e-10)).flatten().tolist()
        except Exception:
            pass
        return [0.0] * 512


# ── Public factories ───────────────────────────────────────────────────────────
# MediaPipe fallback removed: it produced fake 512-dim pixel-mean embeddings
# that are incompatible with ArcFace. InsightFace is the only supported backend.

def build_detector() -> FaceDetector:
    """Build InsightFace SCRFD detector. Raises on failure (no silent fallback)."""
    return InsightFaceDetector()


def build_embedder(detector: Optional[object] = None) -> FaceEmbedder:
    """Build ArcFace R100 embedder. Requires an InsightFaceDetector instance."""
    det = detector if isinstance(detector, InsightFaceDetector) else InsightFaceDetector()
    return InsightFaceEmbedder(det)
