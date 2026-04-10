"""
Graph Engine — Top-level Orchestrator (Level 2).

Combines all subsystems:
    - Identity Resolution (evidence-based clustering)
    - Entity Linking (Wikipedia/Wikidata cross-modal)
    - Truth Anchor (P18 verification)
    - Stability Engine (volatility tracking)
    - Unified DB (single-transaction graph+vector)

Usage:
    engine = GraphEngine(session)
    result = await engine.process_face(embedding, name_hint="Tim Cook")
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.osint_graph.core.identity_resolver import IdentityResolver
from app.osint_graph.core.similarity_engine import SimilarityEngine
from app.osint_graph.core.stability_engine import StabilityEngine
from app.osint_graph.intelligence.entity_linker import EntityLinker
from app.osint_graph.intelligence.truth_anchor import TruthAnchor
from app.osint_graph.storage.unified_db import UnifiedGraphDB

log = logging.getLogger(__name__)


class GraphEngine:
    """
    Top-level orchestrator for the OSINT Identity Graph (Level 2).
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.db = UnifiedGraphDB(session)
        self.resolver = IdentityResolver(session)
        self.entity_linker = EntityLinker(session)
        self.truth_anchor = TruthAnchor(session)
        self.stability = StabilityEngine(session)
        self.similarity = SimilarityEngine()

    async def process_face(
        self,
        embedding: List[float],
        image_url: Optional[str] = None,
        quality_score: float = 1.0,
        angle_hint: str = "frontal",
        camera_id: Optional[str] = None,
        source_id: Optional[uuid.UUID] = None,
        name_hint: Optional[str] = None,
        enrich_entities: bool = False,
        verify_truth: bool = False,
    ) -> Dict[str, Any]:
        """
        Full pipeline: embedding → identity resolution → entity linking → verification.
        """
        # Step 1: Resolve face to identity
        resolution = await self.resolver.resolve(
            new_embedding=embedding,
            image_url=image_url,
            quality_score=quality_score,
            angle_hint=angle_hint,
            camera_id=camera_id,
            source_id=source_id,
            name_hint=name_hint,
        )

        identity_id = uuid.UUID(resolution["identity_id"])
        entity_name = name_hint or resolution.get("name")

        # Step 2: Entity linking
        linked_entities = []
        if enrich_entities and entity_name:
            link_result = await self.entity_linker.link_identity(
                identity_id=identity_id, name=entity_name,
            )
            linked_entities = link_result.get("linked_entities", [])

        # Step 3: Truth anchor verification
        verification = None
        if verify_truth and entity_name:
            verification = await self.truth_anchor.verify_identity(
                identity_id=identity_id, name=entity_name,
            )

        # Step 4: Get graph neighbors
        neighbors = await self.db.get_identity_graph(identity_id, depth=1)

        return {
            "identity_id": str(identity_id),
            "face_id": resolution["face_id"],
            "action": resolution["action"],
            "similarity": resolution["similarity"],
            "cosine_distance": resolution.get("cosine_distance", 0.0),
            "identity_score": resolution["identity_score"],
            "name": resolution.get("name"),
            "linked_entities": linked_entities,
            "graph_neighbors": neighbors.get("related_identities", []),
            "candidates": resolution.get("candidates"),
            "verification": verification,
        }

    async def resolve_embedding(
        self, embedding: List[float]
    ) -> Dict[str, Any]:
        """Lightweight query — find matching identity without creating nodes."""
        nearest = await self.db.search_nearest_identities(
            embedding, top_k=5, min_similarity=0.3,
        )
        if not nearest:
            return {
                "identity_id": None, "confidence": 0.0,
                "linked_entities": [], "graph_neighbors": [],
            }

        best = nearest[0]
        identity_id = uuid.UUID(best["identity_id"])
        graph = await self.db.get_identity_graph(identity_id, depth=2)

        return {
            "identity_id": best["identity_id"],
            "confidence": best["similarity"],
            "name": best.get("name"),
            "verified": best.get("verified", False),
            "stability_score": best.get("stability_score", 0.0),
            "linked_entities": graph.get("linked_entities", []),
            "graph_neighbors": [
                r["identity_id"] for r in graph.get("related_identities", [])
            ],
            "truth_anchors": graph.get("truth_anchors", []),
            "candidates": nearest[:5],
        }

    async def get_identity_detail(
        self, identity_id: uuid.UUID
    ) -> Optional[Dict[str, Any]]:
        """Full identity with 2-hop graph, entities, truth anchors."""
        graph = await self.db.get_identity_graph(identity_id, depth=2)
        if "error" in graph:
            return None

        # Add truth anchor details
        graph["truth_anchor_details"] = (
            await self.truth_anchor.get_truth_anchors_for_identity(identity_id)
        )
        return graph

    async def merge_identities(
        self, source_id: uuid.UUID, target_id: uuid.UUID,
        reason: str = "manual_merge",
    ) -> Dict[str, Any]:
        return await self.resolver.merge_identities(
            source_id, target_id, reason,
        )

    async def enrich_identity(
        self, identity_id: uuid.UUID, name: str,
    ) -> Dict[str, Any]:
        return await self.entity_linker.link_identity(
            identity_id=identity_id, name=name,
        )

    async def verify_identity(
        self, identity_id: uuid.UUID, name: str,
        reference_embedding: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        return await self.truth_anchor.verify_identity(
            identity_id=identity_id, name=name,
            reference_embedding=reference_embedding,
        )

    async def get_volatile_identities(
        self, min_volatility: float = 0.5,
    ) -> List[Dict[str, Any]]:
        return await self.db.get_volatile_identities(min_volatility)

    async def search_identities(
        self, embedding: List[float], top_k: int = 10,
        min_similarity: float = 0.3,
    ) -> List[Dict[str, Any]]:
        return await self.db.search_nearest_identities(
            embedding, top_k=top_k, min_similarity=min_similarity,
        )

    async def get_graph_stats(self) -> Dict[str, Any]:
        return await self.db.get_graph_stats()

    async def create_source(
        self, source_type: str, name: str,
        url: Optional[str] = None, reliability_score: float = 0.5,
    ) -> str:
        node = await self.db.create_source_node(
            source_type=source_type, name=name,
            url=url, reliability_score=reliability_score,
        )
        return str(node.id)
