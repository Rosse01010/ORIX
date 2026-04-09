"""
notifications.py
────────────────
Redis Streams consumer that relays detection events to Socket.IO clients.

Reads from stream:events, transforms the payload to the frontend format,
and emits via Socket.IO:
  - detection-result  → bounding boxes with names/confidence
  - alert             → when an unknown face is detected
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import redis.asyncio as aioredis
from structlog import get_logger

from app.config import settings
from app.websocket.socketio_manager import emit_alert, emit_detection

log = get_logger(__name__)

CONSUMER_GROUP = "ws_notifier"
CONSUMER_NAME = "api_0"


async def _ensure_group(client: aioredis.Redis, stream: str) -> None:
    try:
        await client.xgroup_create(stream, CONSUMER_GROUP, id="$", mkstream=True)
    except aioredis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def _transform_bboxes(bboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert backend bbox format → frontend BoundingBox format.

    Backend:  {x, y, width, height, name, confidence}
    Frontend: {x, y, width, height, label, confidence}
    """
    return [
        {
            "x": b.get("x", 0),
            "y": b.get("y", 0),
            "width": b.get("width", 0),
            "height": b.get("height", 0),
            "label": b.get("name", "Unknown"),
            "confidence": b.get("confidence", 0.0),
        }
        for b in bboxes
    ]


async def relay_events_task() -> None:
    """
    Continuously reads from stream:events and emits to Socket.IO clients.
    Started as an asyncio background task on API startup.
    """
    client = aioredis.from_url(settings.redis_url, decode_responses=True)
    stream = settings.stream_events

    await _ensure_group(client, stream)
    log.info("notifications_relay_started", stream=stream)

    while True:
        try:
            results = await client.xreadgroup(
                CONSUMER_GROUP,
                CONSUMER_NAME,
                {stream: ">"},
                count=20,
                block=500,
            )
            if not results:
                await asyncio.sleep(0.01)
                continue

            for _stream, messages in results:
                for msg_id, fields in messages:
                    await _handle_message(fields)
                    await client.xack(stream, CONSUMER_GROUP, msg_id)

        except aioredis.ConnectionError:
            log.warning("notifications_redis_reconnect")
            await asyncio.sleep(2)
        except Exception as exc:
            log.exception("notifications_relay_error", error=str(exc))
            await asyncio.sleep(1)


async def _handle_message(fields: Dict[str, str]) -> None:
    try:
        payload: Dict[str, Any] = json.loads(fields.get("payload", "{}"))
        camera_id: str = payload.get("camera", "")
        bboxes: List[Dict] = payload.get("bboxes", [])

        if not camera_id:
            return

        boxes = _transform_bboxes(bboxes)

        # Emit detection boxes + candidates to subscribed clients
        candidates = payload.get("candidates", [])
        if boxes:
            await emit_detection(camera_id, boxes, candidates)

        # Emit alert for every unknown face detected
        unknown = [b for b in bboxes if b.get("name") == "Unknown"]
        if unknown:
            await emit_alert(
                camera_id=camera_id,
                alert_type="face-detected",
                level="warning",
                message=f"{len(unknown)} unknown face(s) detected",
                meta={"count": len(unknown)},
            )

    except Exception as exc:
        log.warning("notifications_bad_message", error=str(exc))
