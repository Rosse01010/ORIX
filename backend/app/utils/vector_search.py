"""
vector_search.py
────────────────
Pure-Python / numpy cosine similarity search for ArcFace embeddings.

IMPORTANT — ArcFace metric notes (Deng et al., CVPR 2019):
  ArcFace produces 512-dim L2-normalised embeddings on the unit hypersphere.
  Similarity is measured as cosine similarity ∈ [-1, 1]:
    ≥ 0.55  → strong match (same identity, frontal)
    0.40–0.54 → probable match (off-angle, age gap, partial occlusion)
    0.30–0.39 → weak / uncertain — show candidate panel, do NOT auto-identify
    < 0.30   → different identity

  These ranges are derived from ArcFace's angle-distribution analysis
  (Figure 7 in the paper) and VGGFace2 cross-pose similarity matrices
  (Table IV: front-to-profile averages ~0.49–0.69 for VGGFace2 models).

  The default SIMILARITY_THRESHOLD=0.40 in .env is appropriate for
  multi-angle crowd scenarios.  Use 0.55 for strict frontal-only.

Used when pgvector is not available. Embeddings stored as JSON text in
`person_embeddings.embedding_vec`. Brute-force cosine search (O(N)).
For > 10 k embeddings consider FAISS with an IVF index.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Cosine similarity (embeddings are already L2-normalised by ArcFace) ────────

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    ArcFace guarantees ||a|| = ||b|| = 1, so this reduces to a dot product,
    but we keep the full formula for safety with non-normalised inputs.
    """
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """Ensure unit-norm (needed for embeddings coming from the browser)."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


# ── Per-person best-embedding aggregation ─────────────────────────────────────
#
# VGGFace2 (Cao et al., 2018) shows that averaging face descriptors across
# angles before computing similarity ("template aggregation") significantly
# improves cross-pose recognition.  We implement a lightweight version:
# for each person we keep the BEST single-embedding score rather than a
# mean, because the stored embeddings may span wildly different angles and
# a naive mean can degrade the unit-sphere geometry.  A proper template
# average would require re-normalising after averaging — see enroll endpoint.


def _aggregate_per_person(
    rows, query: np.ndarray
) -> Dict[str, Tuple[str, float]]:
    """
    Given DB rows (pid, name, embedding_vec), return a dict
    {pid: (name, best_cosine_sim)} — one entry per person.
    """
    best: Dict[str, Tuple[str, float]] = {}
    for row in rows:
        pid, name, vec_json = row.pid, row.name, row.embedding_vec
        try:
            vec = _l2_normalize(np.array(json.loads(vec_json), dtype=np.float32))
        except Exception:
            continue
        sim = _cosine_sim(query, vec)
        prev_sim = best.get(pid, (name, -1.0))[1]
        if sim > prev_sim:
            best[pid] = (name, sim)
    return best


# ── Sync version ───────────────────────────────────────────────────────────────

def search_best_sync(
    conn,
    embedding: List[float],
    similarity_threshold: float,
    min_candidate_sim: float = 0.30,
    top_k: int = 5,
    embedding_version: str = "arcface_r100_v1",
) -> Tuple[Optional[str], str, float]:
    """
    Synchronous cosine search across all person_embeddings rows.
    Returns (person_id | None, name, best_cosine_similarity).

    A match is returned only when best_sim >= similarity_threshold.
    Callers should treat 0.40–0.54 as "probable" and surface the
    candidate panel for operator confirmation.

    Only compares against embeddings of the same version to prevent
    cross-model comparisons (e.g. 128-dim face-api.js vs 512-dim ArcFace).
    """
    from sqlalchemy import text

    rows = conn.execute(
        text(
            "SELECT pe.person_id::text AS pid, p.name, pe.embedding_vec "
            "FROM person_embeddings pe "
            "JOIN persons p ON p.id = pe.person_id "
            "WHERE p.active = true "
            "AND pe.embedding_version = :emb_version"
        ),
        {"emb_version": embedding_version},
    ).fetchall()

    query = _l2_normalize(np.array(embedding, dtype=np.float32))
    best_by_person = _aggregate_per_person(rows, query)

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
    return None, "Unknown", best_sim


def search_candidates_sync(
    conn,
    embedding: List[float],
    min_candidate_sim: float = 0.30,
    top_k: int = 5,
    embedding_version: str = "arcface_r100_v1",
) -> List[Dict[str, Any]]:
    """
    Return top-K persons by cosine similarity for the candidate panel.

    min_candidate_sim=0.30 avoids flooding the panel with noise.
    Based on ArcFace angle distributions, genuine pairs rarely fall
    below 0.35 even for extreme pose, so 0.30 provides a safe buffer.

    Only compares against embeddings of the same version.
    """
    from sqlalchemy import text

    rows = conn.execute(
        text(
            "SELECT pe.person_id::text AS pid, p.name, pe.embedding_vec "
            "FROM person_embeddings pe "
            "JOIN persons p ON p.id = pe.person_id "
            "WHERE p.active = true "
            "AND pe.embedding_version = :emb_version"
        ),
        {"emb_version": embedding_version},
    ).fetchall()

    query = _l2_normalize(np.array(embedding, dtype=np.float32))
    best_by_person = _aggregate_per_person(rows, query)

    candidates = [
        {"person_id": pid, "name": name, "similarity": round(sim, 4)}
        for pid, (name, sim) in best_by_person.items()
        if sim >= min_candidate_sim
    ]
    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    return candidates[:top_k]


# ── Async version ──────────────────────────────────────────────────────────────

async def search_best_async(
    db,
    embedding: List[float],
    similarity_threshold: float,
    embedding_version: str = "arcface_r100_v1",
) -> Tuple[Optional[str], str, float]:
    """
    Async cosine search. `db` is an AsyncSession.
    Returns (person_id | None, name, best_cosine_similarity).

    Only compares against embeddings of the same version.
    """
    from sqlalchemy import text

    result = await db.execute(
        text(
            "SELECT pe.person_id::text AS pid, p.name, pe.embedding_vec "
            "FROM person_embeddings pe "
            "JOIN persons p ON p.id = pe.person_id "
            "WHERE p.active = true "
            "AND pe.embedding_version = :emb_version"
        ),
        {"emb_version": embedding_version},
    )
    rows = result.fetchall()

    query = _l2_normalize(np.array(embedding, dtype=np.float32))
    best_by_person = _aggregate_per_person(rows, query)

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
    return None, "Unknown", best_sim


# ── Template-averaged embedding (for enroll) ──────────────────────────────────

def compute_template_embedding(embeddings: List[List[float]]) -> List[float]:
    """
    Compute a L2-normalised mean embedding from multiple face crops of the
    same person (the "template aggregation" strategy from VGGFace2).

    This average embedding can be stored as an additional PersonEmbedding
    with angle_hint="template" for fast single-shot retrieval in addition
    to the per-angle embeddings used for diversity matching.
    """
    if not embeddings:
        return [0.0] * 512
    stack = np.array(embeddings, dtype=np.float32)   # (N, 512)
    mean = stack.mean(axis=0)                         # (512,)
    return _l2_normalize(mean).tolist()
