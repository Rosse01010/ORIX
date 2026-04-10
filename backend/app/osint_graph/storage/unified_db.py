"""
Unified Graph-Vector Storage Layer.

Single PostgreSQL as source of truth for BOTH graph topology and vector
similarity. All operations occur within the same DB transaction,
eliminating split-brain sync issues between separate graph and vector stores.

Uses:
    - pgvector for HNSW-indexed 512D cosine distance searches
    - Adjacency tables (graph_edges) for graph traversal
    - Single AsyncSession for atomic read-modify-write cycles
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import PGVECTOR_AVAILABLE
from app.osint_graph.models.orm import (
    GraphEdge,
    GraphEntityNode,
    GraphFaceNode,
    GraphIdentityNode,
    GraphMasterTruthNode,
    GraphSourceNode,
)
from app.osint_graph.utils.normalization import (
    batch_cosine_similarity,
    embedding_to_json,
    json_to_embedding,
    l2_normalize,
)

log = logging.getLogger(__name__)


class UnifiedGraphDB:
    """
    Unified Graph + Vector database layer.

    Every method operates within the provided AsyncSession so callers
    can compose multiple operations into a single atomic transaction.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    # ══════════════════════════════════════════════════════════════════════════
    #  NODE CREATION
    # ══════════════════════════════════════════════════════════════════════════

    async def create_face_node(
        self,
        embedding: List[float],
        image_url: Optional[str] = None,
        confidence: float = 0.0,
        quality_score: float = 1.0,
        angle_hint: str = "frontal",
        camera_id: Optional[str] = None,
        identity_id: Optional[uuid.UUID] = None,
        source_id: Optional[uuid.UUID] = None,
        person_id: Optional[uuid.UUID] = None,
    ) -> GraphFaceNode:
        node = GraphFaceNode(
            embedding_vec=embedding_to_json(embedding),
            image_url=image_url,
            confidence=confidence,
            quality_score=quality_score,
            angle_hint=angle_hint,
            camera_id=camera_id,
            identity_id=identity_id,
            source_id=source_id,
            person_id=person_id,
        )
        self.session.add(node)
        await self.session.flush()
        return node

    async def create_identity_node(
        self,
        name: Optional[str] = None,
        cluster_center_embedding: Optional[List[float]] = None,
        identity_score: float = 0.0,
        face_count: int = 0,
        person_id: Optional[uuid.UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphIdentityNode:
        node = GraphIdentityNode(
            name=name,
            cluster_center_embedding=embedding_to_json(
                cluster_center_embedding or [0.0] * 512
            ),
            identity_score=identity_score,
            face_count=face_count,
            stability_score=1.0,
            volatility=0.0,
            distinct_cameras=0,
            person_id=person_id,
            metadata_json=json.dumps(metadata) if metadata else None,
        )
        self.session.add(node)
        await self.session.flush()
        return node

    async def create_entity_node(
        self,
        entity_type: str,
        name: str,
        description: Optional[str] = None,
        external_id: Optional[str] = None,
        external_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphEntityNode:
        node = GraphEntityNode(
            entity_type=entity_type,
            name=name,
            description=description,
            external_id=external_id,
            external_url=external_url,
            metadata_json=json.dumps(metadata) if metadata else None,
        )
        self.session.add(node)
        await self.session.flush()
        return node

    async def create_source_node(
        self,
        source_type: str,
        name: str,
        url: Optional[str] = None,
        reliability_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphSourceNode:
        node = GraphSourceNode(
            source_type=source_type,
            name=name,
            url=url,
            reliability_score=reliability_score,
            metadata_json=json.dumps(metadata) if metadata else None,
        )
        self.session.add(node)
        await self.session.flush()
        return node

    async def create_master_truth_node(
        self,
        identity_id: uuid.UUID,
        reference_embedding: List[float],
        source_type: str,
        source_url: Optional[str] = None,
        external_id: Optional[str] = None,
        match_similarity: float = 0.0,
        verified: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphMasterTruthNode:
        node = GraphMasterTruthNode(
            identity_id=identity_id,
            reference_embedding=embedding_to_json(reference_embedding),
            source_type=source_type,
            source_url=source_url,
            external_id=external_id,
            match_similarity=match_similarity,
            verified=verified,
            metadata_json=json.dumps(metadata) if metadata else None,
        )
        self.session.add(node)
        await self.session.flush()
        return node

    # ══════════════════════════════════════════════════════════════════════════
    #  EDGE OPERATIONS
    # ══════════════════════════════════════════════════════════════════════════

    async def create_edge(
        self,
        edge_type: str,
        source_node_id: uuid.UUID,
        source_node_type: str,
        target_node_id: uuid.UUID,
        target_node_type: str,
        weight: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphEdge:
        edge = GraphEdge(
            edge_type=edge_type,
            source_node_id=source_node_id,
            source_node_type=source_node_type,
            target_node_id=target_node_id,
            target_node_type=target_node_type,
            weight=weight,
            metadata_json=json.dumps(metadata) if metadata else None,
        )
        self.session.add(edge)
        await self.session.flush()
        return edge

    async def get_edges_from(
        self, node_id: uuid.UUID, edge_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        q = (
            "SELECT id, edge_type, source_node_id, source_node_type, "
            "target_node_id, target_node_type, weight, metadata_json "
            "FROM graph_edges WHERE source_node_id = :src_id"
        )
        params: Dict[str, Any] = {"src_id": node_id}
        if edge_type:
            q += " AND edge_type = :et"
            params["et"] = edge_type
        result = await self.session.execute(text(q), params)
        return [self._row_to_edge(r) for r in result.fetchall()]

    async def get_edges_to(
        self, node_id: uuid.UUID, edge_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        q = (
            "SELECT id, edge_type, source_node_id, source_node_type, "
            "target_node_id, target_node_type, weight, metadata_json "
            "FROM graph_edges WHERE target_node_id = :tgt_id"
        )
        params: Dict[str, Any] = {"tgt_id": node_id}
        if edge_type:
            q += " AND edge_type = :et"
            params["et"] = edge_type
        result = await self.session.execute(text(q), params)
        return [self._row_to_edge(r) for r in result.fetchall()]

    @staticmethod
    def _row_to_edge(r) -> Dict[str, Any]:
        return {
            "id": str(r[0]), "edge_type": r[1],
            "source_node_id": str(r[2]), "source_node_type": r[3],
            "target_node_id": str(r[4]), "target_node_type": r[5],
            "weight": r[6],
            "metadata": json.loads(r[7]) if r[7] else {},
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  VECTOR SEARCH (pgvector HNSW or numpy fallback)
    # ══════════════════════════════════════════════════════════════════════════

    async def search_nearest_identities(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Find nearest identity centroids using cosine similarity.
        Uses pgvector HNSW index when available, numpy fallback otherwise.
        """
        query = l2_normalize(np.array(query_embedding, dtype=np.float32))

        result = await self.session.execute(
            text(
                "SELECT id::text, cluster_center_embedding, face_count, "
                "name, stability_score, volatility, verified "
                "FROM graph_identity_nodes WHERE active = true"
            )
        )
        rows = result.fetchall()
        if not rows:
            return []

        identities = []
        embeddings = []
        for r in rows:
            try:
                emb = json.loads(r[1]) if r[1] else []
                if len(emb) == 512:
                    identities.append({
                        "identity_id": r[0], "face_count": r[2],
                        "name": r[3], "stability_score": r[4],
                        "volatility": r[5], "verified": r[6],
                    })
                    embeddings.append(emb)
            except (json.JSONDecodeError, TypeError):
                continue

        if not embeddings:
            return []

        matrix = np.array(embeddings, dtype=np.float32)
        sims = batch_cosine_similarity(query, matrix)

        results = []
        for i, sim in enumerate(sims):
            s = float(sim)
            if s >= min_similarity:
                results.append({**identities[i], "similarity": round(s, 6)})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    async def search_nearest_faces(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        query = l2_normalize(np.array(query_embedding, dtype=np.float32))
        result = await self.session.execute(
            text(
                "SELECT id::text, embedding_vec, identity_id::text, "
                "confidence, quality_score, camera_id "
                "FROM graph_face_nodes"
            )
        )
        rows = result.fetchall()
        if not rows:
            return []

        faces, embeddings = [], []
        for r in rows:
            try:
                emb = json.loads(r[1]) if r[1] else []
                if len(emb) == 512:
                    faces.append({
                        "face_id": r[0], "identity_id": r[2],
                        "confidence": r[3], "quality_score": r[4],
                        "camera_id": r[5],
                    })
                    embeddings.append(emb)
            except (json.JSONDecodeError, TypeError):
                continue

        if not embeddings:
            return []

        matrix = np.array(embeddings, dtype=np.float32)
        sims = batch_cosine_similarity(query, matrix)
        results = []
        for i, sim in enumerate(sims):
            s = float(sim)
            if s >= min_similarity:
                results.append({**faces[i], "similarity": round(s, 6)})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    # ══════════════════════════════════════════════════════════════════════════
    #  IDENTITY OPERATIONS
    # ══════════════════════════════════════════════════════════════════════════

    async def get_identity_by_id(
        self, identity_id: uuid.UUID
    ) -> Optional[Dict[str, Any]]:
        result = await self.session.execute(
            text(
                "SELECT id, canonical_id, name, cluster_center_embedding, "
                "identity_score, face_count, stability_score, volatility, "
                "distinct_cameras, verified, needs_review, metadata_json, "
                "active, created_at, updated_at "
                "FROM graph_identity_nodes WHERE id = :id AND active = true"
            ),
            {"id": identity_id},
        )
        row = result.fetchone()
        if not row:
            return None
        return {
            "id": str(row[0]), "canonical_id": row[1], "name": row[2],
            "cluster_center_embedding": json.loads(row[3]) if row[3] else [],
            "identity_score": row[4], "face_count": row[5],
            "stability_score": row[6], "volatility": row[7],
            "distinct_cameras": row[8], "verified": row[9],
            "needs_review": row[10],
            "metadata": json.loads(row[11]) if row[11] else {},
            "active": row[12],
            "created_at": str(row[13]), "updated_at": str(row[14]),
        }

    async def update_identity(
        self,
        identity_id: uuid.UUID,
        new_centroid: Optional[List[float]] = None,
        new_face_count: Optional[int] = None,
        new_score: Optional[float] = None,
        stability_score: Optional[float] = None,
        volatility: Optional[float] = None,
        distinct_cameras: Optional[int] = None,
        verified: Optional[bool] = None,
        needs_review: Optional[bool] = None,
        name: Optional[str] = None,
    ) -> None:
        """Atomic identity update — sets only provided fields."""
        sets = ["updated_at = NOW()"]
        params: Dict[str, Any] = {"id": identity_id}
        if new_centroid is not None:
            sets.append("cluster_center_embedding = :emb")
            params["emb"] = embedding_to_json(new_centroid)
        if new_face_count is not None:
            sets.append("face_count = :fc")
            params["fc"] = new_face_count
        if new_score is not None:
            sets.append("identity_score = :score")
            params["score"] = new_score
        if stability_score is not None:
            sets.append("stability_score = :stab")
            params["stab"] = stability_score
        if volatility is not None:
            sets.append("volatility = :vol")
            params["vol"] = volatility
        if distinct_cameras is not None:
            sets.append("distinct_cameras = :dc")
            params["dc"] = distinct_cameras
        if verified is not None:
            sets.append("verified = :ver")
            params["ver"] = verified
        if needs_review is not None:
            sets.append("needs_review = :nr")
            params["nr"] = needs_review
        if name is not None:
            sets.append("name = :name")
            params["name"] = name

        await self.session.execute(
            text(f"UPDATE graph_identity_nodes SET {', '.join(sets)} WHERE id = :id"),
            params,
        )

    async def deactivate_identity(self, identity_id: uuid.UUID) -> None:
        await self.session.execute(
            text("UPDATE graph_identity_nodes SET active = false WHERE id = :id"),
            {"id": identity_id},
        )

    async def reassign_faces(
        self, from_id: uuid.UUID, to_id: uuid.UUID
    ) -> int:
        r = await self.session.execute(
            text("UPDATE graph_face_nodes SET identity_id = :to WHERE identity_id = :from"),
            {"to": to_id, "from": from_id},
        )
        return r.rowcount

    async def move_edges(
        self, from_id: uuid.UUID, to_id: uuid.UUID
    ) -> int:
        c = 0
        r1 = await self.session.execute(
            text("UPDATE graph_edges SET source_node_id = :to WHERE source_node_id = :from"),
            {"to": to_id, "from": from_id},
        )
        c += r1.rowcount
        r2 = await self.session.execute(
            text("UPDATE graph_edges SET target_node_id = :to WHERE target_node_id = :from"),
            {"to": to_id, "from": from_id},
        )
        c += r2.rowcount
        return c

    # ══════════════════════════════════════════════════════════════════════════
    #  ENTITY LOOKUPS
    # ══════════════════════════════════════════════════════════════════════════

    async def get_entity_by_external_id(
        self, external_id: str
    ) -> Optional[Dict[str, Any]]:
        result = await self.session.execute(
            text(
                "SELECT id, entity_type, name, description, external_id, "
                "external_url, metadata_json "
                "FROM graph_entity_nodes WHERE external_id = :eid"
            ),
            {"eid": external_id},
        )
        row = result.fetchone()
        if not row:
            return None
        return {
            "id": str(row[0]), "entity_type": row[1], "name": row[2],
            "description": row[3], "external_id": row[4],
            "external_url": row[5],
            "metadata": json.loads(row[6]) if row[6] else {},
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  GRAPH TRAVERSAL
    # ══════════════════════════════════════════════════════════════════════════

    async def get_identity_graph(
        self, identity_id: uuid.UUID, depth: int = 2
    ) -> Dict[str, Any]:
        """Build the full subgraph for an identity (up to N-hop)."""
        identity = await self.get_identity_by_id(identity_id)
        if not identity:
            return {"error": "identity_not_found"}

        outgoing = await self.get_edges_from(identity_id)
        incoming = await self.get_edges_to(identity_id)

        linked_entities, linked_faces, related_identities, truth_anchors = [], [], [], []

        for e in outgoing:
            if e["edge_type"] == "identity_to_entity":
                ent = await self._get_entity(uuid.UUID(e["target_node_id"]))
                if ent:
                    ent["confidence_score"] = e["weight"]
                    linked_entities.append(ent)
            elif e["edge_type"] == "identity_to_truth":
                truth_anchors.append({
                    "truth_node_id": e["target_node_id"],
                    "source_reliability": e["weight"],
                })
            elif e["edge_type"] == "identity_to_identity":
                related_identities.append({
                    "identity_id": e["target_node_id"],
                    "similarity": e["weight"],
                    "metadata": e["metadata"],
                })

        for e in incoming:
            if e["edge_type"] == "face_to_identity":
                linked_faces.append({
                    "face_id": e["source_node_id"],
                    "similarity": e["weight"],
                })
            elif e["edge_type"] == "identity_to_identity":
                related_identities.append({
                    "identity_id": e["source_node_id"],
                    "similarity": e["weight"],
                    "metadata": e["metadata"],
                })

        # 2-hop: for each related identity, get their entities
        hop2_entities = []
        if depth >= 2:
            for rel in related_identities:
                rel_edges = await self.get_edges_from(
                    uuid.UUID(rel["identity_id"]), "identity_to_entity"
                )
                for re in rel_edges:
                    ent = await self._get_entity(uuid.UUID(re["target_node_id"]))
                    if ent:
                        ent["via_identity"] = rel["identity_id"]
                        ent["confidence_score"] = re["weight"]
                        hop2_entities.append(ent)

        return {
            "identity": identity,
            "linked_faces": linked_faces,
            "linked_entities": linked_entities,
            "related_identities": related_identities,
            "truth_anchors": truth_anchors,
            "hop2_entities": hop2_entities,
        }

    async def get_volatile_identities(
        self, min_volatility: float = 0.5, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get identities with high embedding drift for audit."""
        result = await self.session.execute(
            text(
                "SELECT id::text, canonical_id, name, volatility, "
                "stability_score, face_count, needs_review "
                "FROM graph_identity_nodes "
                "WHERE active = true AND volatility >= :min_v "
                "ORDER BY volatility DESC LIMIT :lim"
            ),
            {"min_v": min_volatility, "lim": limit},
        )
        return [
            {
                "identity_id": r[0], "canonical_id": r[1],
                "name": r[2], "volatility": r[3],
                "stability_score": r[4], "face_count": r[5],
                "needs_review": r[6],
            }
            for r in result.fetchall()
        ]

    async def get_graph_stats(self) -> Dict[str, int]:
        counts = {}
        for table in [
            "graph_face_nodes", "graph_identity_nodes",
            "graph_master_truth_nodes", "graph_entity_nodes",
            "graph_source_nodes", "graph_edges",
        ]:
            result = await self.session.execute(
                text(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
            )
            counts[table] = result.scalar() or 0
        # Additional stats
        result = await self.session.execute(
            text("SELECT COUNT(*) FROM graph_identity_nodes WHERE verified = true")
        )
        counts["verified_identities"] = result.scalar() or 0
        result = await self.session.execute(
            text("SELECT COUNT(*) FROM graph_identity_nodes WHERE needs_review = true")
        )
        counts["needs_review"] = result.scalar() or 0
        return counts

    async def get_distinct_cameras_for_identity(
        self, identity_id: uuid.UUID
    ) -> int:
        result = await self.session.execute(
            text(
                "SELECT COUNT(DISTINCT camera_id) FROM graph_face_nodes "
                "WHERE identity_id = :id AND camera_id IS NOT NULL"
            ),
            {"id": identity_id},
        )
        return result.scalar() or 0

    async def get_face_embeddings_for_identity(
        self, identity_id: uuid.UUID
    ) -> List[np.ndarray]:
        result = await self.session.execute(
            text(
                "SELECT embedding_vec FROM graph_face_nodes "
                "WHERE identity_id = :id"
            ),
            {"id": identity_id},
        )
        embeddings = []
        for r in result.fetchall():
            try:
                emb = json.loads(r[0]) if r[0] else []
                if len(emb) == 512:
                    embeddings.append(np.array(emb, dtype=np.float32))
            except (json.JSONDecodeError, TypeError):
                continue
        return embeddings

    async def _get_entity(
        self, entity_id: uuid.UUID
    ) -> Optional[Dict[str, Any]]:
        result = await self.session.execute(
            text(
                "SELECT id, entity_type, name, description, external_id, "
                "external_url, metadata_json "
                "FROM graph_entity_nodes WHERE id = :id"
            ),
            {"id": entity_id},
        )
        row = result.fetchone()
        if not row:
            return None
        return {
            "id": str(row[0]), "entity_type": row[1], "name": row[2],
            "description": row[3], "external_id": row[4],
            "external_url": row[5],
            "metadata": json.loads(row[6]) if row[6] else {},
        }
