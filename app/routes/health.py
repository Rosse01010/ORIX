"""
Health-check endpoints.
"""
from __future__ import annotations

import time
from typing import Any, Dict

import redis.asyncio as aioredis
from fastapi import APIRouter
from sqlalchemy import text

from app.config import settings
from app.database import AsyncSessionLocal
from app.websocket.manager import manager

router = APIRouter(tags=["health"])

_start_time = time.time()


@router.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/health/detailed")
async def health_detailed() -> Dict[str, Any]:
    uptime = round(time.time() - _start_time, 1)
    checks: Dict[str, Any] = {
        "uptime_seconds": uptime,
        "websocket_clients": manager.active_connections,
    }

    # PostgreSQL
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"

    # Redis
    try:
        rc = aioredis.from_url(settings.redis_url, decode_responses=True)
        await rc.ping()
        await rc.aclose()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    overall = "ok" if all(
        v in ("ok", manager.active_connections) or not isinstance(v, str) or not v.startswith("error")
        for v in checks.values()
    ) else "degraded"

    return {"status": overall, **checks}
