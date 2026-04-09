"""
WebSocket connection manager.
Broadcasts detection events to all connected frontend clients.
Supports per-camera channel subscriptions.
"""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket
from structlog import get_logger

log = get_logger(__name__)


class ConnectionManager:
    """
    Thread-safe (asyncio) manager for WebSocket connections.

    Clients can subscribe to specific cameras or receive all events.
    """

    def __init__(self) -> None:
        # all active connections
        self._connections: Set[WebSocket] = set()
        # camera_id -> set of subscribed connections
        self._subscriptions: Dict[str, Set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def connect(
        self, websocket: WebSocket, cameras: Optional[List[str]] = None
    ) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
            if cameras:
                for cam in cameras:
                    self._subscriptions[cam].add(websocket)
        log.info(
            "ws_connected",
            total=len(self._connections),
            cameras=cameras,
        )

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)
            for subs in self._subscriptions.values():
                subs.discard(websocket)
        log.info("ws_disconnected", total=len(self._connections))

    # ── Broadcasting ───────────────────────────────────────────────────────────

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Send an event to all connected clients."""
        payload = json.dumps(message)
        await self._send_to_all(self._connections, payload)

    async def broadcast_to_camera(
        self, camera_id: str, message: Dict[str, Any]
    ) -> None:
        """Send an event only to clients subscribed to a specific camera.

        Falls back to broadcasting to everyone if no subscriptions exist for
        this camera (so the frontend always receives events by default).
        """
        async with self._lock:
            targets = self._subscriptions.get(camera_id)
            if targets:
                snapshot = set(targets)
            else:
                snapshot = set(self._connections)

        payload = json.dumps(message)
        await self._send_to_all(snapshot, payload)

    async def _send_to_all(
        self, connections: Set[WebSocket], payload: str
    ) -> None:
        dead: List[WebSocket] = []
        results = await asyncio.gather(
            *[ws.send_text(payload) for ws in connections],
            return_exceptions=True,
        )
        for ws, result in zip(connections, results):
            if isinstance(result, Exception):
                log.warning("ws_send_failed", error=str(result))
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.discard(ws)
                    for subs in self._subscriptions.values():
                        subs.discard(ws)

    # ── Stats ──────────────────────────────────────────────────────────────────

    @property
    def active_connections(self) -> int:
        return len(self._connections)


# Singleton used across the API process
manager = ConnectionManager()
