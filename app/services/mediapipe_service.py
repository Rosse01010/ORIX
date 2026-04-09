"""
Detección facial con MediaPipe BlazeFace (CPU).
Rápido y liviano: solo detecta bounding boxes y recorta caras
antes de enviarlas al worker GPU para generar embeddings.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Tamaño de entrada esperado por FaceNet
_CROP_SIZE = (160, 160)


@dataclass
class FaceDetection:
    """Resultado de una detección facial en un frame."""
    bbox: tuple[int, int, int, int]   # (x, y, w, h) en píxeles absolutos
    confidence: float
    crop: np.ndarray                  # (160, 160, 3) float32 normalizado [0, 1]


class MediaPipeService:
    """
    Wrapper de MediaPipe Face Detection (BlazeFace).
    Una instancia por proceso — no es thread-safe.
    """

    def __init__(self, min_confidence: float = 0.6, model_selection: int = 0):
        """
        Args:
            min_confidence:  Score mínimo de detección (0–1).
            model_selection: 0 = corto alcance (<2 m), 1 = largo alcance (<5 m).
        """
        self.min_confidence = min_confidence
        self.model_selection = model_selection
        self._detector = None
        self._init()

    # ── Inicialización ────────────────────────────────────────────────

    def _init(self) -> None:
        try:
            import mediapipe as mp

            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=self.model_selection,
                min_detection_confidence=self.min_confidence,
            )
            logger.info(
                "MediaPipe BlazeFace inicializado "
                f"(model={self.model_selection}, min_conf={self.min_confidence})"
            )
        except Exception as exc:
            logger.error(f"Error inicializando MediaPipe: {exc}")
            raise

    # ── API pública ───────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        """
        Detecta rostros en un frame BGR de OpenCV.

        Returns:
            Lista de FaceDetection con recortes listos para FaceNet.
            Lista vacía si no se detectó ningún rostro.
        """
        if self._detector is None:
            return []

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        if not results.detections:
            return []

        h, w = frame_bgr.shape[:2]
        detections: List[FaceDetection] = []

        for det in results.detections:
            bbox = det.location_data.relative_bounding_box

            # Convertir coordenadas relativas a absolutas con clipping
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            bw = min(int(bbox.width * w), w - x)
            bh = min(int(bbox.height * h), h - y)

            # Ignorar detecciones demasiado pequeñas
            if bw < 20 or bh < 20:
                continue

            # Recortar y normalizar para FaceNet
            crop_bgr = frame_bgr[y : y + bh, x : x + bw]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, _CROP_SIZE, interpolation=cv2.INTER_LINEAR)
            crop_norm = crop_resized.astype(np.float32) / 255.0

            detections.append(
                FaceDetection(
                    bbox=(x, y, bw, bh),
                    confidence=float(det.score[0]),
                    crop=crop_norm,
                )
            )

        return detections

    def close(self) -> None:
        """Libera recursos del detector."""
        if self._detector is not None:
            self._detector.close()
            self._detector = None
            logger.info("MediaPipe BlazeFace cerrado")
