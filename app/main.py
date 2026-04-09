"""
FastAPI application entry-point.

Socket.IO is wrapped around the FastAPI app so both share the same port.
Run with:  uvicorn app.main:socket_app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

import socketio
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from structlog import get_logger

from app.config import settings
from app.database import AsyncSessionLocal, init_db
from app.routes import health as health_router
from app.routes import recognition as recognition_router
from app.routes import auth as auth_router
from app.routes import cameras as cameras_router
from app.routes import users as users_router
from app.routes import candidates as candidates_router
from app.seed import seed_all
from app.websocket.manager import manager
from app.websocket.notifications import relay_events_task
from app.websocket.socketio_manager import sio
from app.utils.logging_utils import configure_logging

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging(settings.worker_log_level)
    log.info("startup", env=settings.app_env)

    # Init DB tables + pgvector extension
    await init_db()
    log.info("db_ready")

    # Seed default users and cameras
    async with AsyncSessionLocal() as session:
        await seed_all(session)
    log.info("db_seeded")

    # Start Redis → Socket.IO relay
    relay_task = asyncio.create_task(relay_events_task())
    log.info("relay_started")

    yield

    relay_task.cancel()
    try:
        await relay_task
    except asyncio.CancelledError:
        pass
    log.info("shutdown")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ORIX Face Recognition API",
    description="Real-time facial recognition — FastAPI + Socket.IO + pgvector",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────
app.include_router(health_router.router)
app.include_router(recognition_router.router)
app.include_router(auth_router.router)
app.include_router(cameras_router.router)
app.include_router(users_router.router)
app.include_router(candidates_router.router)


# ── Legacy native WebSocket (kept for backward compat) ─────────────────────────
@app.websocket("/ws/detections")
async def ws_detections(
    websocket: WebSocket,
    cameras: Optional[str] = Query(None),
) -> None:
    cam_list: Optional[List[str]] = (
        [c.strip() for c in cameras.split(",") if c.strip()] if cameras else None
    )
    await manager.connect(websocket, cameras=cam_list)
    try:
        while True:
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket)


# ── Socket.IO + FastAPI combined ASGI app ─────────────────────────────────────
# This is the object uvicorn must target: app.main:socket_app
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)
