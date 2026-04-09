"""
Gestor de conexiones WebSocket.
Mantiene el registro de clientes activos y provee broadcast asíncrono.
"""
from __future__ import annotations

import json
import logging
from typing import Dict

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Registra conexiones WebSocket de clientes frontend y permite
    enviar eventos a uno o todos de forma asíncrona.
    """

    def __init__(self) -> None:
        # client_id → WebSocket activo
        self._connections: Dict[str, WebSocket] = {}

    # ── Gestión de conexiones ────────────────────────────────────────

    async def connect(self, client_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[client_id] = websocket
        logger.info(
            f"WS conectado: {client_id} | total={len(self._connections)}"
        )

    async def disconnect(self, client_id: str) -> None:
        self._connections.pop(client_id, None)
        logger.info(
            f"WS desconectado: {client_id} | total={len(self._connections)}"
        )

    # ── Envío de mensajes ────────────────────────────────────────────

    async def send_to(self, client_id: str, data: dict) -> None:
        """Envía un mensaje JSON a un cliente específico."""
        ws = self._connections.get(client_id)
        if ws is None:
            return
        try:
            await ws.send_json(data)
        except Exception as exc:
            logger.warning(f"Error enviando a {client_id}: {exc}")
            await self.disconnect(client_id)

    async def broadcast(self, data: dict) -> None:
        """
        Envía un mensaje JSON a todos los clientes conectados.
        Las conexiones muertas se eliminan automáticamente.
        """
        if not self._connections:
            return

        payload = json.dumps(data, default=str)
        dead: list[str] = []

        for client_id, ws in list(self._connections.items()):
            try:
                await ws.send_text(payload)
            except Exception as exc:
                logger.warning(f"Broadcast falló para {client_id}: {exc}")
                dead.append(client_id)

        for client_id in dead:
            await self.disconnect(client_id)

    # ── Propiedades ──────────────────────────────────────────────────

    @property
    def active_connections(self) -> int:
        return len(self._connections)


# Instancia global compartida por toda la aplicación FastAPI
ws_manager = WebSocketManager()
