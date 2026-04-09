"""
Rutas de la API de reconocimiento facial.
  REST  → registro de personas, historial de eventos, health check.
  WS    → stream de alertas en tiempo real al frontend.
"""
from __future__ import annotations

import uuid
import logging
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.websocket.manager import ws_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/recognition", tags=["recognition"])


# ── Schemas Pydantic ─────────────────────────────────────────────────────────

class PersonCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    embedding: List[float] = Field(..., min_items=128, max_items=2048)


class PersonResponse(BaseModel):
    person_id: str
    name: str


class DetectionEventOut(BaseModel):
    id: str
    camera_id: str
    status: str
    similarity: Optional[float]
    detected_at: str
    person_name: Optional[str]


# ── REST endpoints ───────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    """Verifica que la API esté en línea."""
    return {
        "status": "ok",
        "websocket_clients": ws_manager.active_connections,
    }


@router.post("/persons", response_model=PersonResponse, status_code=201)
async def register_person(
    payload: PersonCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Registra una nueva persona con su embedding facial.
    El embedding debe ser un vector L2-normalizado de 512 dims (FaceNet).
    """
    from app.services.recognition_service import RecognitionService

    embedding = np.array(payload.embedding, dtype=np.float32)

    # Re-normalizar por seguridad
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm

    svc = RecognitionService()
    person_id = await svc.register_person(db, payload.name, embedding)

    return PersonResponse(person_id=person_id, name=payload.name)


@router.get("/persons/{person_id}")
async def get_person(person_id: str, db: AsyncSession = Depends(get_db)):
    """Obtiene los datos básicos de una persona registrada."""
    result = await db.execute(
        text("SELECT id, name, created_at FROM persons WHERE id = :id"),
        {"id": person_id},
    )
    row = result.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Persona no encontrada")
    return dict(row._mapping)


@router.get("/events", response_model=List[DetectionEventOut])
async def list_events(
    limit: int = 50,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Últimos N eventos de detección.
    Filtrar por status='matched' o status='unknown'.
    """
    base_query = """
        SELECT
            e.id, e.camera_id, e.status,
            e.similarity, e.detected_at::text,
            p.name AS person_name
        FROM detection_events e
        LEFT JOIN persons p ON e.person_id = p.id
        {where}
        ORDER BY e.detected_at DESC
        LIMIT :limit
    """
    where = "WHERE e.status = :status" if status else ""
    params: dict = {"limit": limit}
    if status:
        params["status"] = status

    result = await db.execute(text(base_query.format(where=where)), params)
    return [dict(row._mapping) for row in result.fetchall()]


@router.post("/index/rebuild")
async def rebuild_vector_index(db: AsyncSession = Depends(get_db)):
    """
    (Re)construye el índice IVFFLAT sobre embeddings.
    Llamar después de insertar muchas personas nuevas.
    """
    from app.services.recognition_service import RecognitionService
    svc = RecognitionService()
    await svc.ensure_index(db)
    return {"message": "Índice IVFFLAT reconstruido correctamente"}


# ── WebSocket ────────────────────────────────────────────────────────────────

@router.websocket("/ws")
async def websocket_events(websocket: WebSocket):
    """
    Endpoint WebSocket para recibir alertas de reconocimiento en tiempo real.

    El servidor hace push de eventos; el cliente sólo necesita mantener
    la conexión abierta. Protocolo de evento:
        {
          "event":       "face_detected",
          "event_id":    "<uuid>",
          "camera_id":   "cam-00",
          "status":      "matched" | "unknown",
          "person_id":   "<uuid>" | null,
          "person_name": "Juan Pérez" | null,
          "similarity":  0.12 | null,
          "confidence":  0.97,
          "timestamp":   "1718000000.0"
        }
    """
    client_id = str(uuid.uuid4())
    await ws_manager.connect(client_id, websocket)
    try:
        while True:
            # Mantiene viva la conexión; el push lo hace el listener de Redis
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning(f"WS error [{client_id}]: {exc}")
    finally:
        await ws_manager.disconnect(client_id)
