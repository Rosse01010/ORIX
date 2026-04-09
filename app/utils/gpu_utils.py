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


# ── MediaPipe fallback ─────────────────────────────────────────────────────────

class MediaPipeDetector:
    def __init__(self) -> None:
        import mediapipe as mp
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=settings.detection_confidence,
        )
        log.info("mediapipe_detector_ready")

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        results = self._detector.process(rgb)
        faces: List[FaceDetection] = []
        if not results.detections:
            return faces
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x = max(0, int(bb.xmin * w))
            y = max(0, int(bb.ymin * h))
            fw = int(bb.width * w)
            fh = int(bb.height * h)
            crop = cv2.resize(rgb[y:y+fh, x:x+fw], (112, 112)) if fw > 0 and fh > 0 else np.zeros((112, 112, 3), dtype=np.uint8)
            score = det.score[0] if det.score else 0.0
            faces.append(FaceDetection(bbox=[x, y, fw, fh], crop=crop, det_score=float(score)))
        return faces


class MediaPipeEmbedder:
    def embed(self, aligned_crop: np.ndarray) -> List[float]:
        import cv2
        resized = cv2.resize(aligned_crop, (112, 112)).astype(np.float32) / 255.0
        flat = resized.flatten()[:512]
        norm = np.linalg.norm(flat)
        return (flat / (norm + 1e-10)).tolist()


# ── Public factories ───────────────────────────────────────────────────────────

def build_detector() -> FaceDetector:
    if settings.detector_backend.lower() == "insightface":
        try:
            return InsightFaceDetector()
        except Exception as e:
            log.warning("insightface_fallback", error=str(e))
    log.info("using_mediapipe_detector")
    return MediaPipeDetector()


def build_embedder(detector: Optional[object] = None) -> FaceEmbedder:
    if settings.embedder_backend.lower() == "arcface":
        try:
            det = detector if isinstance(detector, InsightFaceDetector) else InsightFaceDetector()
            return InsightFaceEmbedder(det)
        except Exception as e:
            log.warning("arcface_fallback", error=str(e))
    log.info("using_mediapipe_embedder")
    return MediaPipeEmbedder()
