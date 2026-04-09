"""
Punto de entrada de la API FastAPI.
  1. Inicializa la base de datos (pgvector).
  2. Arranca un listener async de alertas Redis → WebSocket broadcast.
  3. Registra las rutas REST y WebSocket.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.routes.recognition import router as recognition_router
from app.websocket.manager import ws_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Listener Redis Pub/Sub → WebSocket ───────────────────────────────────────

async def _redis_alert_listener() -> None:
    """
    Se suscribe al canal Redis de alertas.
    Cada mensaje publicado por el db_worker se retransmite a todos
    los clientes WebSocket conectados.
    """
    import redis.asyncio as aioredis

    client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    pubsub = client.pubsub()
    await pubsub.subscribe(settings.REDIS_ALERT_CHANNEL)
    logger.info(f"Suscrito a Redis Pub/Sub canal: '{settings.REDIS_ALERT_CHANNEL}'")

    try:
        async for message in pubsub.listen():
            if message.get("type") != "message":
                continue
            try:
                event = json.loads(message["data"])
                await ws_manager.broadcast(event)
            except json.JSONDecodeError as exc:
                logger.warning(f"Mensaje Redis malformado: {exc}")
            except Exception as exc:
                logger.error(f"Error en listener Redis: {exc}")
    except asyncio.CancelledError:
        pass
    finally:
        await pubsub.unsubscribe(settings.REDIS_ALERT_CHANNEL)
        await client.aclose()
        logger.info("Listener Redis cerrado")


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    logger.info("Inicializando PostgreSQL + pgvector...")
    await init_db()
    logger.info("Base de datos lista")

    listener_task = asyncio.create_task(
        _redis_alert_listener(), name="redis-alert-listener"
    )

    yield  # aplicación corriendo

    # ── Shutdown ──
    listener_task.cancel()
    try:
        await listener_task
    except asyncio.CancelledError:
        pass
    logger.info("Aplicación FastAPI detenida correctamente")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    application = FastAPI(
        title="Surveillance Facial Recognition API",
        description="API de reconocimiento facial con pgvector, MediaPipe y FaceNet.",
        version="1.0.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],     # ⚠ restringir en producción
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(recognition_router)

    @application.get("/", include_in_schema=False)
    async def root():
        return {"message": "Surveillance API running", "docs": "/docs"}

    return application


app = create_app()
