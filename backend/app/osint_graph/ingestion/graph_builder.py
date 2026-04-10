"""
Graph Builder — Ingestion pipeline for batch processing (Level 2).

Processes face data from multiple sources and builds the identity graph:
    dataset → face extraction → embedding → identity resolution → graph update

Supports:
    - Batch embedding ingestion (pre-computed)
    - Existing ORIX person import (bridge to core system)
    - HuggingFace dataset ingestion
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.osint_graph.core.graph_engine import GraphEngine
from app.osint_graph.ingestion.dataset_linker import DatasetLinker
from app.osint_graph.storage.unified_db import UnifiedGraphDB
from app.osint_graph.utils.normalization import l2_normalize

log = logging.getLogger(__name__)


class GraphBuilder:
    """Builds the identity graph from multiple data sources."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.db = UnifiedGraphDB(session)
        self.engine = GraphEngine(session)
        self.dataset_linker = DatasetLinker(session)

    async def import_existing_persons(self) -> Dict[str, Any]:
        """Import all existing ORIX persons into the OSINT identity graph."""
        source = await self.db.create_source_node(
            source_type="api", name="ORIX Core System",
            reliability_score=0.9,
            metadata={"type": "internal_import"},
        )

        result = await self.session.execute(
            text(
                "SELECT p.id, p.name, pe.id AS emb_id, pe.embedding_vec, "
                "pe.angle_hint, pe.quality_score "
                "FROM persons p "
                "JOIN person_embeddings pe ON pe.person_id = p.id "
                "WHERE p.active = true ORDER BY p.id"
            )
        )
        rows = result.fetchall()
        if not rows:
            return {"imported": 0, "identities": 0, "faces": 0}

        persons: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            pid = str(r[0])
            if pid not in persons:
                persons[pid] = {"name": r[1], "embeddings": []}
            try:
                emb = json.loads(r[3]) if r[3] else []
                if len(emb) == 512:
                    persons[pid]["embeddings"].append({
                        "embedding": emb,
                        "angle_hint": r[4] or "frontal",
                        "quality_score": r[5] or 1.0,
                    })
            except (json.JSONDecodeError, TypeError):
                continue

        identity_count = 0
        face_count = 0
        for pid, data in persons.items():
            if not data["embeddings"]:
                continue

            embs = [np.array(e["embedding"], dtype=np.float32) for e in data["embeddings"]]
            centroid = l2_normalize(np.mean(embs, axis=0))

            identity = await self.db.create_identity_node(
                name=data["name"],
                cluster_center_embedding=centroid.tolist(),
                identity_score=50.0,
                face_count=len(embs),
                person_id=uuid.UUID(pid),
            )
            identity_count += 1

            for emb_data in data["embeddings"]:
                face = await self.db.create_face_node(
                    embedding=emb_data["embedding"],
                    confidence=1.0,
                    quality_score=emb_data["quality_score"],
                    angle_hint=emb_data["angle_hint"],
                    identity_id=identity.id,
                    source_id=source.id,
                    person_id=uuid.UUID(pid),
                )
                await self.db.create_edge(
                    edge_type="face_to_identity",
                    source_node_id=face.id, source_node_type="face",
                    target_node_id=identity.id, target_node_type="identity",
                    weight=1.0,
                )
                await self.db.create_edge(
                    edge_type="face_to_source",
                    source_node_id=face.id, source_node_type="face",
                    target_node_id=source.id, target_node_type="source",
                    weight=0.9,
                )
                face_count += 1

        return {
            "imported": len(persons), "identities": identity_count,
            "faces": face_count, "source_id": str(source.id),
        }

    async def ingest_embedding_batch(
        self,
        embeddings: List[Dict[str, Any]],
        source_type: str = "dataset",
        source_name: str = "batch_import",
        dataset_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest a batch of pre-computed embeddings."""
        source = await self.db.create_source_node(
            source_type=source_type, name=source_name,
            reliability_score=0.7,
        )

        stats = {
            "processed": 0, "identities_created": 0,
            "identities_assigned": 0, "candidate_merges": 0, "errors": 0,
        }

        for item in embeddings:
            try:
                emb = item.get("embedding", [])
                if len(emb) != 512:
                    stats["errors"] += 1
                    continue

                result = await self.engine.process_face(
                    embedding=emb, image_url=item.get("image_url"),
                    camera_id=item.get("camera_id"),
                    source_id=source.id, name_hint=item.get("name"),
                    enrich_entities=False,
                )

                action = result.get("action", "")
                if action == "created":
                    stats["identities_created"] += 1
                elif action == "assigned":
                    stats["identities_assigned"] += 1
                elif action == "candidate_merge":
                    stats["candidate_merges"] += 1

                labels = item.get("labels", [])
                if labels and dataset_key:
                    await self.dataset_linker.link_dataset_labels(
                        identity_id=uuid.UUID(result["identity_id"]),
                        dataset_key=dataset_key, labels=labels,
                    )
                stats["processed"] += 1

            except Exception as e:
                log.warning("batch_ingest_error", error=str(e), index=stats["processed"])
                stats["errors"] += 1

        return stats
