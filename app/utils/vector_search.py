"""
vector_search.py
────────────────
Pure-Python / numpy cosine similarity search.

Used when pgvector is not available (PostgreSQL 18 / Windows without the
extension compiled). Embeddings are stored as JSON text in the
`person_embeddings.embedding_vec` column; this module loads them all into
memory and does a brute-force cosine search.

For production scale (>10 k embeddings) consider FAISS instead.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Sync version (used by db_worker via psycopg2/SQLAlchemy core) ──────────────

def search_best_sync(
    conn,
    embedding: List[float],
    similarity_threshold: float,
    min_candidate_sim: float = 0.20,
    top_k: int = 5,
) -> Tuple[Optional[str], str, float]:
    """
    Synchronous cosine search across all person_embeddings rows.
    Returns (person_id | None, name, best_similarity).
    """
    from sqlalchemy import text
    rows = conn.execute(
        text(
            "SELECT pe.person_id::text AS pid, p.name, pe.embedding_vec "
            "FROM person_embeddings pe "
            "JOIN persons p ON p.id = pe.person_id "
            "WHERE p.active = true"
        )
    ).fetchall()

    query = np.array(embedding, dtype=np.float32)
    best_pid: Optional[str] = None
    best_name = "Unknown"
    best_sim = 0.0
    best_by_person: Dict[str, Tuple[str, float]] = {}

    for row in rows:
        pid, name, vec_json = row.pid, row.name, row.embedding_vec
        try:
            vec = np.array(json.loads(vec_json), dtype=np.float32)
        except Exception:
            continue
        sim = _cosine_sim(query, vec)
        prev = best_by_person.get(pid, (name, -1.0))
        if sim > prev[1]:
            best_by_person[pid] = (name, sim)

    for pid, (name, sim) in best_by_person.items():
        if sim > best_sim:
            best_sim = sim
            best_pid = pid
            best_name = name

    if best_sim >= similarity_threshold:
        return best_pid, best_name, best_sim
    return None, "Unknown", 0.0


def search_candidates_sync(
    conn,
    embedding: List[float],
    min_candidate_sim: float = 0.20,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Return top-K persons by cosine similarity (for the similarity panel).
    """
    from sqlalchemy import text
    rows = conn.execute(
        text(
            "SELECT pe.person_id::text AS pid, p.name, pe.embedding_vec "
            "FROM person_embeddings pe "
            "JOIN persons p ON p.id = pe.person_id "
            "WHERE p.active = true"
        )
    ).fetchall()

    query = np.array(embedding, dtype=np.float32)
    best_by_person: Dict[str, Tuple[str, float]] = {}

    for row in rows:
        pid, name, vec_json = row.pid, row.name, row.embedding_vec
        try:
            vec = np.array(json.loads(vec_json), dtype=np.float32)
        except Exception:
            continue
        sim = _cosine_sim(query, vec)
        prev = best_by_person.get(pid, (name, -1.0))
        if sim > prev[1]:
            best_by_person[pid] = (name, sim)

    candidates = [
        {"person_id": pid, "name": name, "similarity": round(sim, 4)}
        for pid, (name, sim) in best_by_person.items()
        if sim >= min_candidate_sim
    ]
    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    return candidates[:top_k]


# ── Async version (used by FastAPI routes via asyncpg/SQLAlchemy async) ────────

async def search_best_async(
    db,
    embedding: List[float],
    similarity_threshold: float,
) -> Tuple[Optional[str], str, float]:
    """
    Async cosine search. `db` is an AsyncSession.
    """
    from sqlalchemy import text

    result = await db.execute(
        text(
            "SELECT pe.person_id::text AS pid, p.name, pe.embedding_vec "
            "FROM person_embeddings pe "
            "JOIN persons p ON p.id = pe.person_id "
            "WHERE p.active = true"
        )
    )
    rows = result.fetchall()

    query = np.array(embedding, dtype=np.float32)
    best_by_person: Dict[str, Tuple[str, float]] = {}

    for row in rows:
        pid, name, vec_json = row.pid, row.name, row.embedding_vec
        try:
            vec = np.array(json.loads(vec_json), dtype=np.float32)
        except Exception:
            continue
        sim = _cosine_sim(query, vec)
        prev = best_by_person.get(pid, (name, -1.0))
        if sim > prev[1]:
            best_by_person[pid] = (name, sim)

    best_pid: Optional[str] = None
    best_name = "Unknown"
    best_sim = 0.0
    for pid, (name, sim) in best_by_person.items():
        if sim > best_sim:
            best_sim = sim
            best_pid = pid
            best_name = name

    if best_sim >= similarity_threshold:
        return best_pid, best_name, best_sim
    return None, "Unknown", 0.0
