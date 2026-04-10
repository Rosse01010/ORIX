"""
Async SQLAlchemy engine + session factory.
Detects pgvector availability at startup and switches to fallback mode.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

log = logging.getLogger(__name__)

# Global flag — set to False when pgvector is not installed
PGVECTOR_AVAILABLE = True

engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    pass


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db_dep() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    global PGVECTOR_AVAILABLE
    from sqlalchemy import text

    async with engine.begin() as conn:
        # Try enabling pgvector
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            PGVECTOR_AVAILABLE = True
            log.info("pgvector extension enabled")
        except Exception:
            PGVECTOR_AVAILABLE = False
            log.warning(
                "pgvector not available — running in fallback mode "
                "(embeddings stored as JSON, similarity computed in Python)"
            )

        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
        except Exception:
            pass

    # Import models AFTER deciding PGVECTOR_AVAILABLE
    from app import models  # noqa: F401
    from app.osint_graph.models import orm as graph_models  # noqa: F401

    # Conditionally import OSINT models so the audit table gets created
    from app.config import settings as _cfg
    if _cfg.osint_enabled:
        from app.osint.core import models as _osint_models  # noqa: F401
        log.info("OSINT audit log table registered")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create HNSW indexes for graph embedding columns if pgvector available
    if PGVECTOR_AVAILABLE:
        async with engine.begin() as conn:
            for idx_sql in [
                "CREATE INDEX IF NOT EXISTS ix_graph_identity_hnsw "
                "ON graph_identity_nodes "
                "USING hnsw ((cluster_center_embedding::vector(512)) vector_cosine_ops)",
                "CREATE INDEX IF NOT EXISTS ix_graph_truth_hnsw "
                "ON graph_master_truth_nodes "
                "USING hnsw ((reference_embedding::vector(512)) vector_cosine_ops)",
            ]:
                try:
                    await conn.execute(text(idx_sql))
                except Exception as e:
                    log.debug("hnsw_index_skip", reason=str(e))
            log.info("pgvector HNSW indexes ready")

    log.info("Database tables ready")
