"""
recognition.py
──────────────
REST endpoints for facial recognition and person management.

Changes from original:
  - Replaced _get_models() with insightface_service singleton.
  - All embeddings stored with embedding_type + embedding_version.
  - Browser enrollment REJECTS 128-dim face-api.js embeddings (dimension mismatch).
  - Consistent cosine similarity metric everywhere.
"""
from __future__ import annotations

import io
import time
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
from app.services.insightface_service import get_insightface_service, EMBEDDING_DIM, EMBEDDING_VERSION

router = APIRouter(prefix="/api/recognition", tags=["recognition"])


@router.get("/health")
async def recognition_health():
    svc = get_insightface_service()
    return {
        "status": "ok",
        "model_loaded": svc.is_ready,
        "embedding_dim": EMBEDDING_DIM,
        "embedding_version": EMBEDDING_VERSION,
        "similarity_threshold": settings.similarity_threshold,
        "candidate_min_sim": settings.candidate_min_sim,
    }


# ── Schemas ────────────────────────────────────────────────────────────────────

class BBox(BaseModel):
    x: int
    y: int
    width: int
    height: int
    name: str
    confidence: float
    confidence_tier: str = "low"
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


# ── Embedding helpers ──────────────────────────────────────────────────────────

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-10)


def _classify_tier(similarity: float, name: str) -> str:
    """Map cosine similarity to a confidence tier (ArcFace ranges)."""
    if name == "Unknown":
        return "low"
    if similarity >= 0.55:
        return "high"
    if similarity >= settings.similarity_threshold:
        return "moderate"
    return "low"


# ── Search ─────────────────────────────────────────────────────────────────────

async def _search_person(
    db: AsyncSession, embedding: List[float]
) -> tuple[Optional[str], str, float]:
    from app.utils.vector_search import search_best_async
    return await search_best_async(
        db, embedding, settings.similarity_threshold,
        embedding_version=EMBEDDING_VERSION,
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/recognize", response_model=RecognitionResponse)
async def recognize_image(
    camera: str = Form("upload"),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_dep),
) -> RecognitionResponse:
    """Synchronous recognition on a single uploaded image."""
    import cv2
    from app.utils.face_quality import composite_quality, angle_hint_from_yaw
    from app.utils.preprocessing import preprocess_frame

    contents = await file.read()
    img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_bgr = preprocess_frame(
        img_bgr, settings.camera_resize_width, settings.camera_resize_height
    )

    svc = get_insightface_service()
    faces = svc.detect_faces(img_bgr, max_faces=settings.max_faces_per_frame)
    bboxes: List[BBox] = []

    for face in faces:
        w, h = face.bbox[2], face.bbox[3]
        quality, yaw, pitch, roll, _ = composite_quality(
            face.crop, face.kps, w, h, face.det_score
        )
        if quality < 0.1:
            continue

        embedding = svc.extract_embedding(face.crop)

        _, name, confidence = await _search_person(db, embedding)
        tier = _classify_tier(confidence, name)
        angle = angle_hint_from_yaw(yaw)
        x, y = face.bbox[0], face.bbox[1]

        bboxes.append(BBox(
            x=x, y=y, width=w, height=h,
            name=name,
            confidence=round(confidence, 4),
            confidence_tier=tier,
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
    result = await db.execute(select(Person).where(Person.active == True))
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
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db_dep),
) -> PersonOut:
    """
    Register a person with one or more photos.
    Uses the InsightFace singleton for detection + embedding.
    Stores embedding_type and embedding_version for each embedding.
    """
    import cv2
    import json as _json
    from app.utils.face_quality import composite_quality, angle_hint_from_yaw

    svc = get_insightface_service()
    person = Person(name=name)
    db.add(person)
    await db.flush()

    all_embeddings: List[List[float]] = []

    for upload in files:
        contents = await upload.read()
        img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        faces = svc.detect_faces(img_bgr, max_faces=1)
        if not faces:
            continue

        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        w, h = face.bbox[2], face.bbox[3]
        quality, yaw, pitch, roll, _ = composite_quality(
            face.crop, face.kps, w, h, face.det_score
        )
        angle_hint = angle_hint_from_yaw(yaw)
        embedding = svc.extract_embedding(face.crop)

        db.add(PersonEmbedding(
            person_id=person.id,
            embedding_vec=_json.dumps(embedding),
            embedding_type="arcface",
            embedding_version=EMBEDDING_VERSION,
            angle_hint=angle_hint,
            quality_score=quality,
        ))
        all_embeddings.append(embedding)

    if not all_embeddings:
        raise HTTPException(
            status_code=422,
            detail="No face detected in any of the uploaded images.",
        )

    # Store template embedding (VGGFace2-style aggregation) when > 1 photo
    if len(all_embeddings) > 1:
        from app.utils.vector_search import compute_template_embedding
        template_emb = compute_template_embedding(all_embeddings)
        db.add(PersonEmbedding(
            person_id=person.id,
            embedding_vec=_json.dumps(template_emb),
            embedding_type="arcface",
            embedding_version=EMBEDDING_VERSION,
            angle_hint="template",
            quality_score=1.0,
        ))

    await db.flush()

    return PersonOut(
        id=str(person.id),
        name=person.name,
        active=person.active,
        embedding_count=len(all_embeddings),
        created_at=person.created_at.isoformat(),
    )


@router.post("/persons/{person_id}/photos", response_model=Dict[str, Any])
async def add_photos(
    person_id: str,
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db_dep),
) -> Dict[str, Any]:
    """Add more photos (angles) to an existing person. Updates the template embedding."""
    import cv2
    import json as _json
    from app.utils.face_quality import composite_quality, angle_hint_from_yaw

    result = await db.execute(
        select(Person).where(Person.id == uuid.UUID(person_id), Person.active == True)
    )
    person = result.scalar_one_or_none()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found.")

    svc = get_insightface_service()
    new_embeddings: List[List[float]] = []

    for upload in files:
        contents = await upload.read()
        img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        faces = svc.detect_faces(img_bgr, max_faces=1)
        if not faces:
            continue

        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        w, h = face.bbox[2], face.bbox[3]
        quality, yaw, pitch, roll, _ = composite_quality(
            face.crop, face.kps, w, h, face.det_score
        )
        angle_hint = angle_hint_from_yaw(yaw)
        embedding = svc.extract_embedding(face.crop)

        db.add(PersonEmbedding(
            person_id=person.id,
            embedding_vec=_json.dumps(embedding),
            embedding_type="arcface",
            embedding_version=EMBEDDING_VERSION,
            angle_hint=angle_hint,
            quality_score=quality,
        ))
        new_embeddings.append(embedding)

    # Rebuild template embedding from ALL existing embeddings
    if new_embeddings:
        existing_r = await db.execute(
            text(
                "SELECT embedding_vec FROM person_embeddings "
                "WHERE person_id = :pid AND angle_hint != 'template' "
                "AND embedding_version = :ver"
            ),
            {"pid": person_id, "ver": EMBEDDING_VERSION},
        )
        all_vecs = [
            _json.loads(row[0]) for row in existing_r.fetchall()
        ]
        if len(all_vecs) > 1:
            from app.utils.vector_search import compute_template_embedding
            template_emb = compute_template_embedding(all_vecs)
            await db.execute(
                text(
                    "DELETE FROM person_embeddings "
                    "WHERE person_id = :pid AND angle_hint = 'template'"
                ),
                {"pid": person_id},
            )
            db.add(PersonEmbedding(
                person_id=person.id,
                embedding_vec=_json.dumps(template_emb),
                embedding_type="arcface",
                embedding_version=EMBEDDING_VERSION,
                angle_hint="template",
                quality_score=1.0,
            ))

    await db.flush()
    return {"person_id": person_id, "photos_added": len(new_embeddings)}


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


# ── Browser enrollment (face-api.js embedding) ────────────────────────────────

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
    embedding_dim: int
    embedding_type: str
    embedding_version: str
    linkedin_url: Optional[str] = None
    instagram_handle: Optional[str] = None
    twitter_handle: Optional[str] = None
    notes: Optional[str] = None
    warning: Optional[str] = None


@router.post("/persons/enroll", response_model=BrowserEnrollResponse, status_code=status.HTTP_201_CREATED)
async def enroll_person_browser(
    payload: BrowserEnrollPayload,
    db: AsyncSession = Depends(get_db_dep),
) -> BrowserEnrollResponse:
    """
    Register a person using a pre-computed face embedding from the browser.

    CRITICAL: Rejects embeddings that are not 512-dim.
    face-api.js produces 128-dim embeddings which are INCOMPATIBLE with
    ArcFace 512-dim embeddings. Mixing dimensions corrupts the search index.
    """
    import json as _json

    dim = len(payload.embedding)

    # HARD REJECT non-512 embeddings to prevent dimension mismatch corruption
    if dim != EMBEDDING_DIM:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Embedding dimension is {dim}, expected {EMBEDDING_DIM}. "
                f"face-api.js (128-dim) embeddings are incompatible with "
                f"ArcFace (512-dim). Use the server-side /persons endpoint "
                f"with photo uploads instead."
            ),
        )

    emb = _l2_normalize(np.array(payload.embedding, dtype=np.float32)).tolist()

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
        embedding_vec=_json.dumps(emb),
        embedding_type="arcface",
        embedding_version=EMBEDDING_VERSION,
        angle_hint="frontal",
        quality_score=1.0,
    ))
    await db.flush()

    return BrowserEnrollResponse(
        person_id=str(person.id),
        name=person.name,
        embedding_dim=dim,
        embedding_type="arcface",
        embedding_version=EMBEDDING_VERSION,
        linkedin_url=person.linkedin_url,
        instagram_handle=person.instagram_handle,
        twitter_handle=person.twitter_handle,
        notes=person.notes,
    )


# ── Test endpoint ──────────────────────────────────────────────────────────────

class TestResult(BaseModel):
    faces: List[Dict[str, Any]]
    matches: List[Dict[str, Any]]
    latency_ms: float
    model_version: str


@router.post("/test", response_model=TestResult)
async def recognition_test(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_dep),
) -> TestResult:
    """
    Diagnostic endpoint: detect faces, extract embeddings, search matches,
    and return detailed results with latency breakdown.
    """
    import cv2
    from app.utils.face_quality import composite_quality, angle_hint_from_yaw

    t_start = time.monotonic()

    contents = await file.read()
    img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    svc = get_insightface_service()

    t_det = time.monotonic()
    faces = svc.detect_faces(img_bgr, max_faces=settings.max_faces_per_frame)
    detection_ms = (time.monotonic() - t_det) * 1000

    face_results: List[Dict[str, Any]] = []
    match_results: List[Dict[str, Any]] = []

    for face in faces:
        w, h = face.bbox[2], face.bbox[3]
        quality, yaw, pitch, roll, _ = composite_quality(
            face.crop, face.kps, w, h, face.det_score
        )

        t_emb = time.monotonic()
        embedding = svc.extract_embedding(face.crop)
        embedding_ms = (time.monotonic() - t_emb) * 1000

        face_results.append({
            "bbox": {"x": face.bbox[0], "y": face.bbox[1],
                     "w": face.bbox[2], "h": face.bbox[3]},
            "det_score": round(face.det_score, 4),
            "quality": round(quality, 4),
            "yaw": round(yaw, 1),
            "pitch": round(pitch, 1),
            "roll": round(roll, 1),
            "angle": angle_hint_from_yaw(yaw),
            "embedding_dim": len(embedding),
            "embedding_ms": round(embedding_ms, 1),
        })

        pid, name, sim = await _search_person(db, embedding)
        match_results.append({
            "person_id": pid,
            "name": name,
            "similarity": round(sim, 4),
            "tier": _classify_tier(sim, name),
        })

    total_ms = (time.monotonic() - t_start) * 1000

    return TestResult(
        faces=face_results,
        matches=match_results,
        latency_ms=round(total_ms, 1),
        model_version=EMBEDDING_VERSION,
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
