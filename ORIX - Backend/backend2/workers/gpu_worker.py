"""
Worker GPU: consume recortes de la cola en memoria, arma batches
y genera embeddings reales con FaceNet. Publica resultados a Redis Streams.

Estrategia de batching:
  - Espera hasta GPU_BATCH_SIZE items O hasta que expire el timeout.
  - Esto maximiza el throughput GPU sin sacrificar latencia inaceptable.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import time
from typing import List


def run_gpu_worker(
    face_queue: mp.Queue,
    batch_size: int = 16,
    batch_timeout_ms: int = 50,
) -> None:
    """
    Proceso único GPU.

    Args:
        face_queue:       Cola compartida (recibe dicts de camera_worker).
        batch_size:       Máximo de recortes por inferencia.
        batch_timeout_ms: Ms máximo esperando completar el batch.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [GPU-WORKER] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    from app.services.facenet_service import FaceNetService
    from app.config import settings
    import redis

    # Inicializar FaceNet (carga el modelo y hace warmup)
    facenet = FaceNetService(model_path=settings.FACENET_MODEL_PATH)
    redis_client = redis.from_url(settings.REDIS_URL)

    timeout_s = batch_timeout_ms / 1000.0
    logger.info(
        f"GPU Worker listo | batch_size={batch_size} | timeout={batch_timeout_ms}ms"
        f" | facenet_ready={facenet.is_ready}"
    )

    batches_processed = 0

    while True:
        batch: List[dict] = _collect_batch(face_queue, batch_size, timeout_s)

        if not batch:
            continue

        crops = [item["crop"] for item in batch]

        try:
            embeddings = facenet.generate_embeddings(crops)
        except Exception as exc:
            logger.error(f"Error en inferencia FaceNet: {exc}", exc_info=True)
            continue

        # Publicar cada embedding al Redis Stream
        pipe = redis_client.pipeline(transaction=False)
        for item, emb in zip(batch, embeddings):
            payload = {
                "camera_id":  item["camera_id"],
                "embedding":  json.dumps(emb.tolist()),
                "confidence": str(round(item["confidence"], 4)),
                "timestamp":  str(item["timestamp"]),
            }
            pipe.xadd(
                settings.REDIS_STREAM_KEY,
                payload,
                maxlen=10_000,     # retener últimos 10k mensajes
                approximate=True,  # recorte aproximado (más rápido)
            )
        pipe.execute()

        batches_processed += 1
        logger.debug(
            f"Batch #{batches_processed}: {len(batch)} embeddings → Redis Streams"
        )


def _collect_batch(queue: mp.Queue, max_size: int, timeout_s: float) -> List[dict]:
    """
    Acumula hasta `max_size` items de la cola o hasta que expire el timeout.
    Retorna lista (puede estar vacía si la cola estaba vacía al inicio).
    """
    batch: List[dict] = []
    deadline = time.monotonic() + timeout_s

    while len(batch) < max_size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            item = queue.get(timeout=min(remaining, 0.02))
            batch.append(item)
        except Exception:
            break  # timeout o cola vacía

    return batch
