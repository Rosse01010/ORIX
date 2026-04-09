"""
insightface_service.py
──────────────────────
Thread-safe singleton for InsightFace face detection + ArcFace embedding.

Reuses the existing gpu_utils.py factory functions so the API container
loads the SAME models (CPU mode) that workers use (GPU mode).

Why a singleton:
  - InsightFace model loading is expensive (~2-4 s, 300+ MB VRAM/RAM).
  - Multiple API requests must share one loaded model instance.
  - Thread safety is required because FastAPI runs request handlers
    concurrently via asyncio + thread pool.
"""
from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np

from app.config import settings
from app.utils.logging_utils import get_logger

log = get_logger(__name__)

# Embedding dimension produced by ArcFace R100
EMBEDDING_DIM = 512
# Embedding version tag for this model
EMBEDDING_VERSION = "arcface_r100_v1"


class InsightFaceService:
    """
    Singleton service wrapping InsightFace detection + ArcFace embedding.

    Thread-safe: uses a lock around lazy initialization.
    The detector and embedder instances from gpu_utils are themselves
    stateless after init (ONNX Runtime handles thread safety internally).
    """

    _instance: Optional["InsightFaceService"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "InsightFaceService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._detector = None
        self._embedder = None
        self._init_lock = threading.Lock()
        self._initialized = True

    # ── Lazy loading ──────────────────────────────────────────────────

    def _ensure_models(self) -> None:
        """Load models on first use (lazy). Thread-safe."""
        if self._detector is not None:
            return
        with self._init_lock:
            if self._detector is not None:
                return
            from app.utils.gpu_utils import build_detector, build_embedder
            log.info(
                "insightface_service_loading",
                model=settings.insightface_model,
                use_gpu=settings.use_gpu,
            )
            self._detector = build_detector()
            self._embedder = build_embedder(self._detector)
            log.info("insightface_service_ready")

    # ── Public API ────────────────────────────────────────────────────

    def get_model(self):
        """Return the underlying FaceAnalysis app (for advanced use)."""
        self._ensure_models()
        return self._detector

    def detect_faces(self, frame: np.ndarray, max_faces: int = 10) -> list:
        """
        Detect faces in a BGR frame.

        Args:
            frame: BGR numpy array (OpenCV format).
            max_faces: Maximum faces to return (sorted by area, descending).

        Returns:
            List of FaceDetection from gpu_utils.
        """
        self._ensure_models()
        faces = self._detector.detect(frame)
        if len(faces) > max_faces:
            faces.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
            faces = faces[:max_faces]
        return faces

    def extract_embedding(self, aligned_crop: np.ndarray) -> List[float]:
        """
        Extract a 512-dim L2-normalised ArcFace embedding from an aligned face chip.

        Args:
            aligned_crop: RGB 112x112 numpy array.

        Returns:
            512-dim list of floats (unit vector).
        """
        self._ensure_models()
        embedding = self._embedder.embed(aligned_crop)
        # Ensure L2 normalization
        arr = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 1e-10:
            arr = arr / norm
        return arr.tolist()

    @property
    def is_ready(self) -> bool:
        return self._detector is not None

    @staticmethod
    def embedding_version() -> str:
        return EMBEDDING_VERSION

    @staticmethod
    def embedding_dim() -> int:
        return EMBEDDING_DIM


# Module-level accessor — all consumers import this
def get_insightface_service() -> InsightFaceService:
    """Return the global InsightFace singleton."""
    return InsightFaceService()
