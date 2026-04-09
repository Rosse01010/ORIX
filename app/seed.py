"""
seed.py
───────
Seeds the database with default users and cameras on first startup.
Safe to call on every boot — uses upsert-style checks before inserting.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Camera, OrixUser
from app.routes.auth import hash_password
from app.utils.logging_utils import get_logger

log = get_logger(__name__)

# Default dashboard users
_DEFAULT_USERS = [
    {"username": "admin",    "password": "admin123",    "role": "admin"},
    {"username": "operator", "password": "operator123", "role": "operator"},
    {"username": "user",     "password": "user123",     "role": "user"},
]


async def seed_users(session: AsyncSession) -> None:
    for u in _DEFAULT_USERS:
        result = await session.execute(
            select(OrixUser).where(OrixUser.username == u["username"])
        )
        if result.scalar_one_or_none():
            continue
        session.add(
            OrixUser(
                username=u["username"],
                hashed_password=hash_password(u["password"]),
                role=u["role"],
            )
        )
        log.info("seed_user_created", username=u["username"], role=u["role"])
    await session.commit()


async def seed_cameras(session: AsyncSession) -> None:
    for idx, source in enumerate(settings.camera_source_list):
        cam_id = f"cam_{idx:02d}"
        result = await session.execute(
            select(Camera).where(Camera.id == cam_id)
        )
        if result.scalar_one_or_none():
            continue

        # For USB/device-index sources, stream_url is left empty (no HTTP stream)
        stream_url = source if source.startswith("rtsp://") or source.startswith("http") else ""

        session.add(
            Camera(
                id=cam_id,
                name=f"Camera {idx + 1}",
                location="",
                source=source,
                stream_url=stream_url,
                status="online",
            )
        )
        log.info("seed_camera_created", camera_id=cam_id, source=source)
    await session.commit()


async def seed_all(session: AsyncSession) -> None:
    await seed_users(session)
    await seed_cameras(session)
