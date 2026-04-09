"""
candidates.py
─────────────
Endpoint for manual identity confirmation from the similarity panel.

POST /api/candidates/confirm
  Body: { face_embedding: float[], person_id: string, camera_id: string }
  → Adds the face embedding to the confirmed person's embeddings
    (improves future recognition of that angle automatically)
"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db_dep
from app.models import Person, PersonEmbedding
from app.utils.logging_utils import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/candidates", tags=["candidates"])


class ConfirmRequest(BaseModel):
    person_id: str
    embedding: List[float]        # 512-dim vector from the unrecognised face
    angle_hint: str = "unknown"
    quality_score: float = 0.5
    camera_id: str = ""


class ConfirmResponse(BaseModel):
    person_id: str
    name: str
    embeddings_total: int
    message: str


@router.post("/confirm", response_model=ConfirmResponse)
async def confirm_identity(
    body: ConfirmRequest,
    db: AsyncSession = Depends(get_db_dep),
) -> ConfirmResponse:
    """
    Called when an operator clicks a candidate in the similarity panel.
    Adds the unrecognised face's embedding to the selected person's profile
    so that same angle is recognised automatically in the future.
    """
    import uuid

    result = await db.execute(
        select(Person).where(
            Person.id == uuid.UUID(body.person_id),
            Person.active == True,
        )
    )
    person = result.scalar_one_or_none()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    if len(body.embedding) != 512:
        raise HTTPException(status_code=422, detail="Embedding must be 512-dimensional")

    import json as _json
    db.add(PersonEmbedding(
        person_id=person.id,
        embedding_vec=_json.dumps(body.embedding),
        angle_hint=body.angle_hint,
        quality_score=body.quality_score,
    ))
    await db.flush()

    # Count total embeddings for this person
    from sqlalchemy import text
    count_r = await db.execute(
        text("SELECT COUNT(*) FROM person_embeddings WHERE person_id = :pid"),
        {"pid": str(person.id)},
    )
    total = count_r.scalar() or 0

    log.info(
        "identity_confirmed",
        person=person.name,
        angle=body.angle_hint,
        camera=body.camera_id,
        total_embeddings=total,
    )

    return ConfirmResponse(
        person_id=str(person.id),
        name=person.name,
        embeddings_total=total,
        message=f"Embedding added. {person.name} will now be recognised from this angle.",
    )
