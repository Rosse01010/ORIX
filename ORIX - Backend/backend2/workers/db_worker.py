"""
Worker de base de datos.
  1. Consume embeddings desde Redis Streams (consumer group para escalado).
  2. Busca coincidencias con pgvector usando operador <=>.
  3. Persiste el evento en PostgreSQL.
  4. Publica alerta en Redis Pub/Sub → API → WebSocket → frontend.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid


async def run_db_worker() -> None:
    """Coroutine principal. Ejecutar con asyncio.run()."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [DB-WORKER] %(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    import numpy as np
    import redis.asyncio as aioredis

    from app.config import settings
    from app.database import AsyncSessionLocal, init_db
    from app.services.recognition_service import RecognitionService

    # Inicializar DB (crea tablas y extensión vector si no existen)
    await init_db()

    redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    recognition_svc = RecognitionService()

    # Crear consumer group si no existe (mkstream=True crea el stream si no existe)
    try:
        await redis_client.xgroup_create(
            settings.REDIS_STREAM_KEY,
            settings.REDIS_CONSUMER_GROUP,
            id="$",           # solo mensajes nuevos desde ahora
            mkstream=True,
        )
        logger.info(f"Consumer group '{settings.REDIS_CONSUMER_GROUP}' creado")
    except Exception:
        logger.info(f"Consumer group '{settings.REDIS_CONSUMER_GROUP}' ya existe")

    consumer_name = f"db-worker-{int(time.time())}"
    logger.info(f"DB Worker iniciado como: {consumer_name}")

    while True:
        try:
            # Leer hasta 50 mensajes, bloqueando hasta 2 segundos si no hay
            response = await redis_client.xreadgroup(
                groupname=settings.REDIS_CONSUMER_GROUP,
                consumername=consumer_name,
                streams={settings.REDIS_STREAM_KEY: ">"},
                count=50,
                block=2000,
            )

            if not response:
                continue

            for _stream, entries in response:
                for entry_id, data in entries:
                    try:
                        await _process_entry(
                            entry_id=entry_id,
                            data=data,
                            redis_client=redis_client,
                            recognition_svc=recognition_svc,
                        )
                    except Exception as exc:
                        logger.error(f"Error procesando {entry_id}: {exc}", exc_info=True)
                    finally:
                        # Confirmar procesamiento incluso si falló (evitar requeue infinito)
                        await redis_client.xack(
                            settings.REDIS_STREAM_KEY,
                            settings.REDIS_CONSUMER_GROUP,
                            entry_id,
                        )

        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error(f"Error en ciclo de consumo: {exc}", exc_info=True)
            await asyncio.sleep(2)

    await redis_client.aclose()
    logger.info("DB Worker finalizado")


async def _process_entry(
    entry_id: str,
    data: dict,
    redis_client,
    recognition_svc,
) -> None:
    """
    Procesa un mensaje del stream:
      1. Decodifica el embedding.
      2. Busca coincidencia en pgvector.
      3. Persiste el evento.
      4. Publica alerta.
    """
    import numpy as np
    from sqlalchemy import text

    from app.config import settings
    from app.database import AsyncSessionLocal

    logger = logging.getLogger(__name__)

    # ── Decodificar ──
    try:
        camera_id = data["camera_id"]
        embedding = np.array(json.loads(data["embedding"]), dtype=np.float32)
        confidence = float(data.get("confidence", 0.0))
        timestamp = data.get("timestamp", str(time.time()))
    except (KeyError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(f"Mensaje malformado [{entry_id}]: {exc}")
        return

    # ── Buscar en pgvector ──
    async with AsyncSessionLocal() as session:
        match = await recognition_svc.find_match(session, embedding)

        if match:
            person_id, person_name, distance = match
            status = "matched"
        else:
            person_id, person_name, distance = None, None, None
            status = "unknown"

        # ── Persistir evento ──
        event_id = str(uuid.uuid4())
        await session.execute(
            text("""
                INSERT INTO detection_events
                    (id, person_id, camera_id, similarity, status, detected_at)
                VALUES (:id, :pid, :cam, :sim, :status, NOW())
            """),
            {
                "id":     event_id,
                "pid":    person_id,
                "cam":    camera_id,
                "sim":    distance,
                "status": status,
            },
        )
        await session.commit()

    # ── Publicar alerta a Redis Pub/Sub → API WebSocket ──
    alert = {
        "event":       "face_detected",
        "event_id":    event_id,
        "camera_id":   camera_id,
        "status":      status,
        "person_id":   person_id,
        "person_name": person_name,
        "similarity":  round(distance, 4) if distance is not None else None,
        "confidence":  round(confidence, 4),
        "timestamp":   timestamp,
    }
    await redis_client.publish(settings.REDIS_ALERT_CHANNEL, json.dumps(alert))

    if status == "matched":
        logger.info(
            f"[{camera_id}] ✓ MATCH → {person_name} "
            f"(dist={distance:.4f}, conf={confidence:.2f})"
        )
    else:
        logger.debug(
            f"[{camera_id}] Desconocido (conf={confidence:.2f})"
        )


def main() -> None:
    asyncio.run(run_db_worker())


if __name__ == "__main__":
    main()
