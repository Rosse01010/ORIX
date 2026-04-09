"""
Clase RTSPCamera: mantiene una conexión RTSP activa con reconexión automática.
Lee frames en un hilo de fondo para no bloquear el loop de procesamiento.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class RTSPCamera:
    """
    Buffer de un frame del stream RTSP con reconexión automática.

    Uso:
        cam = RTSPCamera(url="rtsp://...", camera_id="cam-00")
        cam.start()
        frame = cam.get_frame()   # None si aún no hay frame
        cam.stop()
    """

    # Tiempo entre reintentos de conexión (segundos)
    RECONNECT_DELAY: float = 5.0
    # Fallos consecutivos antes de considerar el stream perdido
    MAX_CONSECUTIVE_FAILURES: int = 5

    def __init__(self, url: str, camera_id: str):
        self.url = url
        self.camera_id = camera_id

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Ciclo de vida ────────────────────────────────────────────────

    def start(self) -> None:
        """Arranca el hilo de captura en segundo plano."""
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"rtsp-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"[{self.camera_id}] Hilo de captura iniciado → {self.url}")

    def stop(self) -> None:
        """Detiene el hilo y libera recursos de OpenCV."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=8)
        self._release_cap()
        logger.info(f"[{self.camera_id}] Hilo de captura detenido")

    # ── Lectura de frames ────────────────────────────────────────────

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Devuelve el frame más reciente (copia, thread-safe).
        Retorna None si todavía no hay frame disponible.
        """
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    # ── Internos ─────────────────────────────────────────────────────

    def _connect(self) -> bool:
        """Intenta abrir la fuente RTSP. Retorna True si tuvo éxito."""
        self._release_cap()
        self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        # Reducir buffer interno a 1 frame para minimizar latencia
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok = self._cap.isOpened()
        if ok:
            logger.info(f"[{self.camera_id}] Conectado a {self.url}")
        else:
            logger.warning(f"[{self.camera_id}] No se pudo conectar a {self.url}")
        return ok

    def _release_cap(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _capture_loop(self) -> None:
        """
        Loop principal del hilo de captura.
        Conecta, lee frames y reconecta automáticamente si el stream falla.
        """
        retries = 0

        while self._running:
            if not self._connect():
                retries += 1
                logger.warning(
                    f"[{self.camera_id}] Reintento de conexión #{retries} "
                    f"en {self.RECONNECT_DELAY}s..."
                )
                time.sleep(self.RECONNECT_DELAY)
                continue

            retries = 0
            failures = 0

            while self._running:
                ok, raw_frame = self._cap.read()

                if not ok:
                    failures += 1
                    logger.warning(
                        f"[{self.camera_id}] Fallo de lectura #{failures}"
                    )
                    if failures >= self.MAX_CONSECUTIVE_FAILURES:
                        logger.error(
                            f"[{self.camera_id}] Stream perdido tras "
                            f"{failures} fallos. Reconectando..."
                        )
                        break
                    time.sleep(0.1)
                    continue

                failures = 0
                resized = cv2.resize(
                    raw_frame,
                    (settings.FRAME_WIDTH, settings.FRAME_HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                )
                with self._lock:
                    self._frame = resized

        logger.info(f"[{self.camera_id}] Hilo de captura finalizado")

    # ── Propiedades de diagnóstico ───────────────────────────────────

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
