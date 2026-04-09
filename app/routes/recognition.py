"""
recognition.py
──────────────
REST endpoints for facial recognition and person management.

POST /api/recognize           – recognize faces in an uploaded image
GET  /api/persons             – list all known persons
POST /api/persons             – register person with one or more photos
POST /api/persons/{id}/photos – add more photos (angles) to existing person
DELETE /api/persons/{id}      – soft-delete person
"""
from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from PIL import Image
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db_dep
from app.models import Person, PersonEmbedding

router = APIRouter(prefix="/api/recognition", tags=["recognition"])


@router.get("/health")
async def recognition_health():
    return {"status": "ok", "websocket_clients": 0}


# ── Schemas ────────────────────────────────────────────────────────────────────

class BBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    name: str
    confidence: float
    quality: float = 0.0
    angle: str = "frontal"


class RecognitionResponse(BaseModel):
    camera: str
    timestamp: str
    bboxes: List[BBox]


class PersonOut(BaseModel):
    id: str
    name: str
    active: bool
    embedding_count: int
    created_at: str


# ── Lazy model loader ──────────────────────────────────────────────────────────

_detector = None
_embedder = None


def _get_models():
    global _detector, _embedder
    if _detector is None:
        from app.utils.gpu_utils import build_detector, build_embedder
        _detector = build_detector()
        _embedder = build_embedder(_detector)
    return _detector, _embedder


# ── Multi-embedding search ─────────────────────────────────────────────────────

async def _search_person(
    db: AsyncSession, embedding: List[float]
) -> tuple[Optional[str], str, float]:
    """
    Search across ALL embeddings for ALL persons.
    Returns the person with the highest cosine similarity match.
    Uses numpy cosine similarity (pgvector fallback mode).
    """
    from app.utils.vector_search import search_best_async
    return await search_best_async(db, embedding, settings.similarity_threshold)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_image(
    camera: str = Form("upload"),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_dep),
) -> RecognitionResponse:
    """Synchronous recognition on a single uploaded image."""
    from app.utils.face_quality import composite_quality, angle_hint_from_yaw
    from app.utils.preprocessing import preprocess_frame

    contents = await file.read()
    img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    # Convert RGB → BGR for OpenCV pipeline
    import cv2
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_bgr = preprocess_frame(
        img_bgr,
        settings.camera_resize_width,
        settings.camera_resize_height,
    )

    detector, embedder = _get_models()
    faces = detector.detect(img_bgr)
    bboxes: List[BBox] = []

    for face in faces:
        w, h = face.bbox[2], face.bbox[3]
        quality, yaw, pitch, roll, _ = composite_quality(
            face.crop, face.kps, w, h, face.det_score
        )
        if quality < 0.1:
            continue
        embedding = embedder.embed(face.crop)
        _, name, confidence = await _search_person(db, embedding)
        angle = angle_hint_from_yaw(yaw)
        x, y = face.bbox[0], face.bbox[1]
        bboxes.append(BBox(
            x=x, y=y, width=w, height=h,
            name=name,
            confidence=round(confidence, 4),
            quality=round(quality, 3),
            angle=angle,
        ))

    return RecognitionResponse(
        camera=camera,
        timestamp=datetime.now(timezone.utc).isoformat(),
        bboxes=bboxes,
    )


@router.get("/persons", response_model=List[PersonOut])
async def list_persons(
    db: AsyncSession = Depends(get_db_dep),
) -> List[PersonOut]:
    result = await db.execute(
        select(Person).where(Person.active == True)
    )
    persons = result.scalars().all()
    out = []
    for p in persons:
        count_r = await db.execute(
            text("SELECT COUNT(*) FROM person_embeddings WHERE person_id = :pid"),
            {"pid": str(p.id)},
        )
        count = count_r.scalar() or 0
        out.append(PersonOut(
            id=str(p.id),
            name=p.name,
            active=p.active,
            embedding_count=count,
            created_at=p.created_at.isoformat(),
        ))
    return out


@router.post("/persons", response_model=PersonOut, status_code=status.HTTP_201_CREATED)
async def register_person(
    name: str = Form(...),
    files: List[UploadFile] = File(...),   # accept multiple photos at once
    db: AsyncSession = Depends(get_db_dep),
) -> PersonOut:
    """
    Register a person with one or more photos.
    Send multiple photos at different angles for best recognition.
    """
    from app.utils.face_quality import composite_quality, angle_hint_from_yaw
    import cv2

    detector, embedder = _get_models()

    # Create person record first
    person = Person(name=name)
    db.add(person)
    await db.flush()

    embeddings_added = 0

    for upload in files:
        contents = await upload.read()
        img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        faces = detector.detect(img_bgr)
        if not faces:
            continue

        # Use the largest face in each photo
        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        w, h = face.bbox[2], face.bbox[3]
        quality, yaw, pitch, roll, _ = composite_quality(
            face.crop, face.kps, w, h, face.det_score
        )
        angle_hint = angle_hint_from_yaw(yaw)
        embedding = embedder.embed(face.crop)

        import json as _json
        db.add(PersonEmbedding(
            person_id=person.id,
            embedding_vec=_json.dumps(embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)),
            angle_hint=angle_hint,
            quality_score=quality,
        ))
        embeddings_added += 1

    if embeddings_added == 0:
        raise HTTPException(
            status_code=422,
            detail="No face detected in any of the uploaded images.",
        )

    await db.flush()

    return PersonOut(
        id=str(person.id),
        name=person.name,
        active=person.active,
        embedding_count=embeddings_added,
        created_at=person.created_at.isoformat(),
    )


@router.post("/persons/{person_id}/photos", response_model=Dict[str, Any])
async def add_photos(
    person_id: str,
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db_dep),
) -> Dict[str, Any]:
    """Add more photos (angles) to an existing person to improve recognition."""
    from app.utils.face_quality import composite_quality, angle_hint_from_yaw
    import cv2

    result = await db.execute(
        select(Person).where(Person.id == uuid.UUID(person_id), Person.active == True)
    )
    person = result.scalar_one_or_none()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found.")

    detector, embedder = _get_models()
    added = 0

    for upload in files:
        contents = await upload.read()
        img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        faces = detector.detect(img_bgr)
        if not faces:
            continue

        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        w, h = face.bbox[2], face.bbox[3]
        quality, yaw, pitch, roll, _ = composite_quality(
            face.crop, face.kps, w, h, face.det_score
        )
        angle_hint = angle_hint_from_yaw(yaw)
        embedding = embedder.embed(face.crop)

        import json as _json
        db.add(PersonEmbedding(
            person_id=person.id,
            embedding_vec=_json.dumps(embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)),
            angle_hint=angle_hint,
            quality_score=quality,
        ))
        added += 1

    await db.flush()
    return {"person_id": person_id, "photos_added": added}


@router.delete("/persons/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_person(
    person_id: str,
    db: AsyncSession = Depends(get_db_dep),
) -> None:
    result = await db.execute(
        select(Person).where(Person.id == uuid.UUID(person_id))
    )
    person = result.scalar_one_or_none()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found.")
    person.active = False


# ── Browser-based enrollment (face-api.js embedding) ─────────────────────────

class BrowserEnrollPayload(BaseModel):
    name: str
    embedding: List[float]
    linkedin_url: Optional[str] = None
    instagram_handle: Optional[str] = None
    twitter_handle: Optional[str] = None
    notes: Optional[str] = None


class BrowserEnrollResponse(BaseModel):
    person_id: str
    name: str
    linkedin_url: Optional[str] = None
    instagram_handle: Optional[str] = None
    twitter_handle: Optional[str] = None
    notes: Optional[str] = None


@router.post("/persons/enroll", response_model=BrowserEnrollResponse, status_code=status.HTTP_201_CREATED)
async def enroll_person_browser(
    payload: BrowserEnrollPayload,
    db: AsyncSession = Depends(get_db_dep),
) -> BrowserEnrollResponse:
    """
    Register a person using a pre-computed face embedding from the browser
    (face-api.js). Accepts optional social-profile links provided by the
    person at enrollment time.
    """
    import json as _json

    person = Person(
        name=payload.name,
        linkedin_url=payload.linkedin_url,
        instagram_handle=payload.instagram_handle,
        twitter_handle=payload.twitter_handle,
        notes=payload.notes,
    )
    db.add(person)
    await db.flush()

    db.add(PersonEmbedding(
        person_id=person.id,
        embedding_vec=_json.dumps(payload.embedding),
        angle_hint="frontal",
        quality_score=1.0,
    ))
    await db.flush()

    return BrowserEnrollResponse(
        person_id=str(person.id),
        name=person.name,
        linkedin_url=person.linkedin_url,
        instagram_handle=person.instagram_handle,
        twitter_handle=person.twitter_handle,
        notes=person.notes,
    )


class PersonDetailOut(BaseModel):
    id: str
    name: str
    linkedin_url: Optional[str] = None
    instagram_handle: Optional[str] = None
    twitter_handle: Optional[str] = None
    notes: Optional[str] = None
    created_at: str


@router.get("/persons/{person_id}", response_model=PersonDetailOut)
async def get_person(
    person_id: str,
    db: AsyncSession = Depends(get_db_dep),
) -> PersonDetailOut:
    result = await db.execute(
        select(Person).where(Person.id == uuid.UUID(person_id), Person.active == True)
    )
    person = result.scalar_one_or_none()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found.")
    return PersonDetailOut(
        id=str(person.id),
        name=person.name,
        linkedin_url=person.linkedin_url,
        instagram_handle=person.instagram_handle,
        twitter_handle=person.twitter_handle,
        notes=person.notes,
        created_at=person.created_at.isoformat(),
    )
