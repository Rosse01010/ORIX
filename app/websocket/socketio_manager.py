"""
socketio_manager.py
───────────────────
Socket.IO AsyncServer (singleton) that the frontend connects to.

Handles:
  subscribe-camera   → puts socket in room "camera:<id>"
  unsubscribe-camera → leaves that room
  face-detected      → received from frontend (browser-side detection)
  ack-alert          → client acknowledges an alert

Emits:
  detection-result   → bounding boxes from GPU pipeline
  alert              → face detection alert (unknown person, etc.)
  camera-status      → camera online/offline changes
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import socketio
from structlog import get_logger

log = get_logger(__name__)

# ── Singleton Socket.IO server ─────────────────────────────────────────────────
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25,
)


def _room(camera_id: str) -> str:
    return f"camera:{camera_id}"


# ── Connection lifecycle ───────────────────────────────────────────────────────

@sio.event
async def connect(sid: str, environ: Dict, auth: Optional[Dict] = None) -> bool:
    token = (auth or {}).get("token")
    log.info("socketio_connect", sid=sid, has_token=bool(token))
    # Token validation is optional here; auth routes protect the REST API.
    # For production, validate JWT here and return False to reject.
    return True


@sio.event
async def disconnect(sid: str) -> None:
    log.info("socketio_disconnect", sid=sid)


# ── Camera room subscriptions ─────────────────────────────────────────────────

@sio.on("subscribe-camera")
async def on_subscribe_camera(sid: str, data: Dict[str, Any]) -> None:
    camera_id = data.get("cameraId", "")
    if camera_id:
        await sio.enter_room(sid, _room(camera_id))
        log.debug("socketio_subscribe", sid=sid, camera_id=camera_id)


@sio.on("unsubscribe-camera")
async def on_unsubscribe_camera(sid: str, data: Dict[str, Any]) -> None:
    camera_id = data.get("cameraId", "")
    if camera_id:
        await sio.leave_room(sid, _room(camera_id))
        log.debug("socketio_unsubscribe", sid=sid, camera_id=camera_id)


# ── Client-emitted events ─────────────────────────────────────────────────────

@sio.on("face-detected")
async def on_face_detected(sid: str, data: Dict[str, Any]) -> None:
    """Received when the browser's face-api.js detects a face locally."""
    camera_id = data.get("cameraId", "")
    count = data.get("count", 0)
    log.debug("browser_face_detected", sid=sid, camera_id=camera_id, count=count)


@sio.on("ack-alert")
async def on_ack_alert(sid: str, data: Dict[str, Any]) -> None:
    alert_id = data.get("alertId", "")
    log.debug("alert_acknowledged", sid=sid, alert_id=alert_id)


# ── Emit helpers (called by notifications relay) ───────────────────────────────

async def emit_detection(
    camera_id: str,
    boxes: list,
    candidates: list | None = None,
) -> None:
    """Broadcast detection bounding boxes to all subscribers of this camera."""
    await sio.emit(
        "detection-result",
        {
            "cameraId": camera_id,
            "boxes": boxes,
            "candidates": candidates or [],
        },
        room=_room(camera_id),
    )


async def emit_alert(
    camera_id: str,
    alert_type: str,
    level: str,
    message: str,
    meta: Optional[Dict] = None,
) -> None:
    """Broadcast an alert to all subscribers of this camera."""
    alert = {
        "id": str(uuid.uuid4()),
        "cameraId": camera_id,
        "type": alert_type,
        "level": level,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "meta": meta or {},
    }
    await sio.emit("alert", alert, room=_room(camera_id))


async def emit_camera_status(camera_id: str, status: str) -> None:
    """Notify all clients about a camera status change."""
    await sio.emit(
        "camera-status",
        {"cameraId": camera_id, "status": status},
    )
