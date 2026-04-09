"""
Generación de embeddings faciales reales con FaceNet (TensorFlow/Keras).
Carga el modelo .h5 una vez y opera en batches para maximizar throughput GPU.
Si el modelo no existe, opera en modo desarrollo con embeddings aleatorios.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class FaceNetService:
    """
    Wrapper de FaceNet para generación de embeddings de 512 dimensiones.

    Uso típico:
        svc = FaceNetService("/app/models/facenet_model.h5")
        embeddings = svc.generate_embeddings(crops)   # (N, 512) normalizado L2
    """

    INPUT_SHAPE = (160, 160, 3)

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
        self._dev_mode = False
        self._load_model()

    # ── Carga del modelo ─────────────────────────────────────────────

    def _load_model(self) -> None:
        try:
            import tensorflow as tf

            # Suprimir logs de TF durante la carga
            tf.get_logger().setLevel("ERROR")

            self._model = tf.keras.models.load_model(
                self.model_path, compile=False
            )
            # Warmup para inicializar pesos en GPU antes de la primera inferencia real
            dummy = np.zeros((1, *self.INPUT_SHAPE), dtype=np.float32)
            self._model.predict(dummy, verbose=0)

            logger.info(
                f"FaceNet cargado: {self.model_path} | "
                f"input={self._model.input_shape} | "
                f"output={self._model.output_shape}"
            )

        except FileNotFoundError:
            self._dev_mode = True
            logger.warning(
                f"[DEV MODE] Modelo FaceNet no encontrado en '{self.model_path}'. "
                "Se usarán embeddings aleatorios L2-normalizados."
            )
        except Exception as exc:
            logger.error(f"Error cargando FaceNet: {exc}")
            raise

    # ── API pública ──────────────────────────────────────────────────

    def generate_embeddings(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Genera embeddings para un batch de recortes faciales.

        Args:
            crops: Lista de arrays float32 con forma (160, 160, 3), valores en [0, 1].

        Returns:
            Array float32 con forma (N, 512), normalizado L2.
            Array vacío (0, 512) si la lista de entrada está vacía.
        """
        if not crops:
            return np.empty((0, 512), dtype=np.float32)

        batch = np.stack(crops, axis=0)  # (N, 160, 160, 3)

        if self._dev_mode or self._model is None:
            raw = np.random.randn(len(crops), 512).astype(np.float32)
        else:
            raw = self._model.predict(batch, verbose=0)

        return self._l2_normalize(raw)

    # ── Utilidades ───────────────────────────────────────────────────

    @staticmethod
    def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
        """Normalización L2 para usar distancia coseno como similitud."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        return (embeddings / norms).astype(np.float32)

    @staticmethod
    def prewhiten(img: np.ndarray) -> np.ndarray:
        """
        Pre-whitening estándar de FaceNet (opcional, alternativa a /255).
        Usar si el modelo fue entrenado con esta normalización.
        """
        mean = np.mean(img)
        std = np.std(img)
        std_adj = max(std, 1.0 / np.sqrt(img.size))
        return (img - mean) / std_adj

    @property
    def is_ready(self) -> bool:
        """True si el modelo real está cargado (False en modo desarrollo)."""
        return self._model is not None
