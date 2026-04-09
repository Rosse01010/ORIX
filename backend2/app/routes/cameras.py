"""
Camera management routes.

GET    /cameras         → list all cameras
GET    /cameras/{id}    → single camera
PATCH  /cameras/{id}    → update name / location / status
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db_dep
from app.models import Camera
from app.websocket.socketio_manager import emit_camera_status

router = APIRouter(prefix="/cameras", tags=["cameras"])


# ── Schemas ────────────────────────────────────────────────────────────────────

class CameraOut(BaseModel):
    id: str
    name: str
    location: str
    streamUrl: str
    status: str


class CameraPatch(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    status: Optional[str] = None


def _to_out(cam: Camera) -> CameraOut:
    return CameraOut(
        id=cam.id,
        name=cam.name,
        location=cam.location,
        streamUrl=cam.stream_url,
        status=cam.status,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("", response_model=List[CameraOut])
async def list_cameras(
    db: AsyncSession = Depends(get_db_dep),
) -> List[CameraOut]:
    result = await db.execute(select(Camera).order_by(Camera.id))
    return [_to_out(c) for c in result.scalars().all()]


@router.get("/{camera_id}", response_model=CameraOut)
async def get_camera(
    camera_id: str,
    db: AsyncSession = Depends(get_db_dep),
) -> CameraOut:
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = result.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return _to_out(cam)


@router.patch("/{camera_id}", response_model=CameraOut)
async def patch_camera(
    camera_id: str,
    body: CameraPatch,
    db: AsyncSession = Depends(get_db_dep),
) -> CameraOut:
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    cam = result.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    if body.name is not None:
        cam.name = body.name
    if body.location is not None:
        cam.location = body.location
    if body.status is not None:
        old_status = cam.status
        cam.status = body.status
        if old_status != body.status:
            await emit_camera_status(camera_id, body.status)

    await db.flush()
    return _to_out(cam)
