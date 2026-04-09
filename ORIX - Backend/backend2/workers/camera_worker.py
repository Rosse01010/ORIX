"""
Worker de cámara: proceso independiente por cada fuente RTSP.
  1. Conecta a la cámara usando RTSPCamera (con reconexión automática).
  2. Salta N frames (FRAME_SKIP) para reducir carga de CPU.
  3. Detecta caras con MediaPipe BlazeFace.
  4. Envía los recortes faciales a la cola GPU.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import time
from typing import Optional


def run_camera_worker(
    camera_id: str,
    rtsp_url: str,
    face_queue: mp.Queue,
    frame_skip: int = 5,
) -> None:
    """
    Función de proceso de cámara.

    Args:
        camera_id:  Identificador único de la cámara (ej. "cam-00").
        rtsp_url:   URL completa del stream RTSP.
        face_queue: Cola compartida hacia el GPU worker.
        frame_skip: Procesar 1 de cada N frames leídos.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [CAMERA:{camera_id}] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    from app.services.rtsp_service import RTSPCamera
    from app.services.mediapipe_service import MediaPipeService

    camera = RTSPCamera(url=rtsp_url, camera_id=camera_id)
    detector = MediaPipeService(min_confidence=0.6, model_selection=0)

    camera.start()
    logger.info(f"Worker iniciado: {camera_id} → {rtsp_url} (skip={frame_skip})")

    frame_count = 0
    faces_sent = 0

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            frame_count += 1

            # Saltar frames para reducir carga
            if frame_count % frame_skip != 0:
                continue

            # Detectar rostros con MediaPipe (CPU, rápido)
            detections = detector.detect(frame)
            if not detections:
                continue

            for det in detections:
                item = {
                    "camera_id": camera_id,
                    "crop": det.crop,             # ndarray (160, 160, 3) float32
                    "bbox": det.bbox,             # (x, y, w, h)
                    "confidence": det.confidence,
                    "timestamp": time.time(),
                }
                try:
                    face_queue.put_nowait(item)
                    faces_sent += 1
                except Exception:
                    # Cola llena: descartar silenciosamente para no bloquear
                    pass

            if frame_count % 100 == 0:
                logger.debug(
                    f"frames={frame_count} | caras_enviadas={faces_sent} "
                    f"| queue_size={face_queue.qsize()}"
                )

    except KeyboardInterrupt:
        logger.info("Interrupción recibida")
    except Exception as exc:
        logger.error(f"Error fatal: {exc}", exc_info=True)
    finally:
        camera.stop()
        detector.close()
        logger.info(f"Worker {camera_id} finalizado (frames={frame_count})")
