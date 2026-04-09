"""
recognition_service.py
──────────────────────
pgvector-based facial recognition service.

SIMILARITY STANDARDIZATION:
  pgvector <=> operator returns cosine DISTANCE in [0, 2].
  We convert to cosine SIMILARITY = 1 - distance, range [-1, 1].
  All thresholds in config are expressed as cosine similarity.

  This ensures the same metric is used everywhere:
    - vector_search.py (numpy brute-force fallback)
    - this service (pgvector)
    - db_worker.py
    - recognition routes
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional, Tuple

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings

logger = logging.getLogger(__name__)

MatchResult = Tuple[str, str, float]  # (person_id, name, cosine_similarity)


class RecognitionService:
    """
    Facial recognition over pgvector.
    All operations are async and require an AsyncSession.
    """

    async def find_match(
        self,
        session: AsyncSession,
        embedding: np.ndarray,
        threshold: Optional[float] = None,
        embedding_version: str = "arcface_r100_v1",
    ) -> Optional[MatchResult]:
        """
        Find the most similar person using pgvector cosine distance.

        pgvector <=> returns cosine distance in [0, 2]:
          0 = identical, 1 = orthogonal, 2 = opposite

        We convert: similarity = 1 - distance
        And compare against the similarity threshold.

        Args:
            session:           Async SQLAlchemy session.
            embedding:         L2-normalised 512-dim vector.
            threshold:         Minimum cosine similarity for a match.
            embedding_version: Only compare against embeddings of same version.

        Returns:
            (person_id, name, cosine_similarity) or None.
        """
        min_sim = threshold if threshold is not None else settings.similarity_threshold

        vec_str = "[" + ",".join(f"{v:.8f}" for v in embedding.tolist()) + "]"

        # Filter by embedding_version to avoid cross-model comparisons.
        # Convert pgvector cosine distance to similarity inline.
        query = text("""
            SELECT
                pe.person_id,
                p.name,
                (1.0 - (pe.embedding_vec <=> CAST(:vec AS vector))) AS similarity
            FROM person_embeddings pe
            JOIN persons p ON p.id = pe.person_id
            WHERE p.active = true
              AND pe.embedding_version = :emb_version
            ORDER BY pe.embedding_vec <=> CAST(:vec AS vector) ASC
            LIMIT 1
        """)

        try:
            result = await session.execute(query, {
                "vec": vec_str,
                "emb_version": embedding_version,
            })
            row = result.fetchone()
        except Exception as exc:
            logger.error(f"pgvector search error: {exc}")
            return None

        if row is None:
            return None

        person_id, name, similarity = str(row.person_id), row.name, float(row.similarity)

        if similarity < min_sim:
            logger.debug(f"No match: similarity {similarity:.4f} < threshold {min_sim}")
            return None

        return person_id, name, similarity

    async def register_person(
        self,
        session: AsyncSession,
        name: str,
        embedding: np.ndarray,
        embedding_version: str = "arcface_r100_v1",
    ) -> str:
        """Register a new person with their embedding."""
        person_id = str(uuid.uuid4())
        vec_str = "[" + ",".join(f"{v:.8f}" for v in embedding.tolist()) + "]"

        await session.execute(
            text("""
                INSERT INTO persons (id, name, created_at)
                VALUES (:id, :name, NOW())
            """),
            {"id": person_id, "name": name},
        )
        await session.execute(
            text("""
                INSERT INTO person_embeddings
                    (id, person_id, embedding_vec, embedding_type, embedding_version, created_at)
                VALUES
                    (gen_random_uuid(), :pid, CAST(:vec AS vector), 'arcface', :ver, NOW())
            """),
            {"pid": person_id, "vec": vec_str, "ver": embedding_version},
        )
        await session.commit()

        logger.info(f"Person registered: '{name}' -> {person_id}")
        return person_id

    async def ensure_index(self, session: AsyncSession) -> None:
        """Create IVFFLAT cosine index on person_embeddings.embedding_vec."""
        await session.execute(text("""
            CREATE INDEX IF NOT EXISTS person_embeddings_vec_idx
            ON person_embeddings
            USING ivfflat (embedding_vec vector_cosine_ops)
            WITH (lists = 100)
        """))
        await session.commit()
        logger.info("IVFFLAT index created/verified on person_embeddings.embedding_vec")
