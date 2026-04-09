"""
Búsqueda de similitud vectorial en PostgreSQL con pgvector.
Usa el operador <=> (distancia coseno) que aprovecha el índice IVFFLAT/HNSW.
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

# Tipo de retorno de find_match: (person_id, name, cosine_distance)
MatchResult = Tuple[str, str, float]


class RecognitionService:
    """
    Servicio de reconocimiento facial sobre pgvector.
    Todas las operaciones son async y requieren una AsyncSession de SQLAlchemy.
    """

    # ── Búsqueda ─────────────────────────────────────────────────────

    async def find_match(
        self,
        session: AsyncSession,
        embedding: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Optional[MatchResult]:
        """
        Busca la persona más similar al embedding dado usando distancia coseno.

        El operador <=> de pgvector retorna distancia coseno ∈ [0, 2]:
          0   → idéntico
          1   → ortogonal
          2   → opuesto

        Args:
            session:   Sesión async de SQLAlchemy.
            embedding: Vector L2-normalizado de 512 dims.
            threshold: Distancia máxima aceptable (default: settings.SIMILARITY_THRESHOLD).

        Returns:
            (person_id, name, distance) o None si no hay match dentro del umbral.
        """
        max_dist = threshold if threshold is not None else settings.SIMILARITY_THRESHOLD

        # pgvector espera el vector como string '[x1, x2, ...]'
        vec_str = "[" + ",".join(f"{v:.8f}" for v in embedding.tolist()) + "]"

        query = text("""
            SELECT id, name, (embedding <=> CAST(:vec AS vector)) AS distance
            FROM persons
            ORDER BY distance ASC
            LIMIT 1
        """)

        try:
            result = await session.execute(query, {"vec": vec_str})
            row = result.fetchone()
        except Exception as exc:
            logger.error(f"Error en búsqueda pgvector: {exc}")
            return None

        if row is None:
            return None

        person_id, name, distance = row.id, row.name, float(row.distance)

        if distance > max_dist:
            logger.debug(f"Sin match: distancia {distance:.4f} > umbral {max_dist}")
            return None

        return person_id, name, distance

    # ── Registro ─────────────────────────────────────────────────────

    async def register_person(
        self,
        session: AsyncSession,
        name: str,
        embedding: np.ndarray,
    ) -> str:
        """
        Registra una nueva persona con su embedding en la base de datos.

        Returns:
            person_id (UUID) generado.
        """
        person_id = str(uuid.uuid4())
        vec_str = "[" + ",".join(f"{v:.8f}" for v in embedding.tolist()) + "]"

        await session.execute(
            text("""
                INSERT INTO persons (id, name, embedding, created_at)
                VALUES (:id, :name, CAST(:vec AS vector), NOW())
            """),
            {"id": person_id, "name": name, "vec": vec_str},
        )
        await session.commit()

        logger.info(f"Persona registrada: '{name}' → {person_id}")
        return person_id

    # ── Índice vectorial ─────────────────────────────────────────────

    async def ensure_index(self, session: AsyncSession) -> None:
        """
        Crea un índice IVFFLAT sobre el campo embedding si no existe.
        Llamar tras insertar un volumen significativo de personas.
        Ajustar `lists` según el número de filas (sqrt(N) es un buen default).
        """
        await session.execute(text("""
            CREATE INDEX IF NOT EXISTS persons_embedding_idx
            ON persons
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """))
        await session.commit()
        logger.info("Índice IVFFLAT creado/verificado en persons.embedding")
