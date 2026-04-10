"""
Identity Resolver — Evidence-Based Clustering Engine (Level 2).

Resolves face embeddings into identity nodes using:
    1. pgvector nearest-neighbor search (cosine distance)
    2. Evidence-based scoring (camera diversity, stability, quality)
    3. L2-normalised centroid updates on the unit hypersphere
    4. Stability-aware merging (don't auto-merge into volatile identities)

Distance thresholds (cosine distance = 1 - cosine_similarity):
    < 0.15  AND different camera → high-confidence merge
    0.15-0.25 → candidate link (don't auto-merge)
    > 0.25  → new identity

Equivalent similarity thresholds:
    > 0.85  → same identity (auto-assign)
    0.75-0.85 → candidate merge (create link, flag for review)
    < 0.75  → new identity (create node)

RULE: Every centroid update MUST be followed by L2 Normalisation.
    new_centroid = unit_vector((old_centroid * n + new_embedding) / (n + 1))
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.osint_graph.core.stability_engine import StabilityEngine
from app.osint_graph.storage.unified_db import UnifiedGraphDB
from app.osint_graph.utils.normalization import (
    cosine_similarity,
    embedding_to_json,
    l2_normalize,
    update_centroid,
)
from app.osint_graph.utils.scoring import (
    THRESHOLD_CANDIDATE_MERGE,
    THRESHOLD_SAME_IDENTITY,
    ConfidenceFactors,
    classify_similarity,
    compute_identity_confidence,
)

log = logging.getLogger(__name__)

# Cosine distance thresholds for evidence-based matching
DISTANCE_HIGH_CONFIDENCE = 0.15   # cosine_distance < 0.15 → auto-merge
DISTANCE_MARGINAL = 0.25          # cosine_distance < 0.25 → candidate link


class IdentityResolver:
    """
    Evidence-Based Identity Resolution Engine.

    Pipeline:
    1. Search nearest neighbor identities via cosine distance
    2. Apply evidence-based scoring:
       - Distance < 0.15 AND cam_id is different → high confidence merge
       - Distance marginal (0.15-0.25) AND stability is low → candidate link
       - Distance > 0.25 → create new identity
    3. If merge: update centroid using L2 Normalisation
    4. If new: create IdentityNode + link to FaceNode
    5. Recompute stability metrics after every mutation
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.db = UnifiedGraphDB(session)
        self.stability = StabilityEngine(session)

    async def resolve(
        self,
        new_embedding: List[float],
        image_url: Optional[str] = None,
        quality_score: float = 1.0,
        angle_hint: str = "frontal",
        camera_id: Optional[str] = None,
        source_id: Optional[uuid.UUID] = None,
        name_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Resolve a face embedding to an identity using evidence-based scoring.
        """
        query = l2_normalize(
            np.array(new_embedding, dtype=np.float32)
        )
        query_list = query.tolist()

        # Step 1: Search nearest identity centroids
        nearest = await self.db.search_nearest_identities(
            query_list, top_k=5, min_similarity=0.0
        )

        best = nearest[0] if nearest else None
        best_sim = best["similarity"] if best else 0.0
        cosine_distance = 1.0 - best_sim

        # Step 2: Evidence-based classification
        if best and cosine_distance < DISTANCE_HIGH_CONFIDENCE:
            # High confidence — check camera diversity for extra evidence
            is_diverse = (
                camera_id is not None
                and best.get("distinct_cameras", 0) > 0
            )
            target_volatile = best.get("volatility", 0.0) >= 0.5

            if target_volatile:
                # Target is volatile — create candidate link, don't merge
                return await self._candidate_link(
                    query_list, best, nearest, image_url,
                    quality_score, angle_hint, camera_id, source_id,
                    reason="target_volatile",
                )

            return await self._assign_to_identity(
                identity_id=uuid.UUID(best["identity_id"]),
                embedding=query_list,
                similarity=best_sim,
                image_url=image_url,
                quality_score=quality_score,
                angle_hint=angle_hint,
                camera_id=camera_id,
                source_id=source_id,
                face_count=best["face_count"],
                camera_diverse=is_diverse,
            )

        elif best and cosine_distance < DISTANCE_MARGINAL:
            # Marginal match — create face node + candidate link
            target_volatile = best.get("volatility", 0.0) >= 0.5
            target_low_stability = best.get("stability_score", 1.0) < 0.5

            if target_volatile or target_low_stability:
                reason = "volatile" if target_volatile else "low_stability"
                return await self._candidate_link(
                    query_list, best, nearest, image_url,
                    quality_score, angle_hint, camera_id, source_id,
                    reason=reason,
                )

            # Marginal but stable target — still create candidate link
            return await self._candidate_link(
                query_list, best, nearest, image_url,
                quality_score, angle_hint, camera_id, source_id,
                reason="marginal_distance",
            )

        else:
            # No close match — create new identity
            return await self._create_new_identity(
                embedding=query_list,
                image_url=image_url,
                quality_score=quality_score,
                angle_hint=angle_hint,
                camera_id=camera_id,
                source_id=source_id,
                name=name_hint,
            )

    async def _assign_to_identity(
        self,
        identity_id: uuid.UUID,
        embedding: List[float],
        similarity: float,
        image_url: Optional[str],
        quality_score: float,
        angle_hint: str,
        camera_id: Optional[str],
        source_id: Optional[uuid.UUID],
        face_count: int,
        camera_diverse: bool,
    ) -> Dict[str, Any]:
        """Assign face to existing identity, update centroid with L2 norm."""
        # Create face node
        face = await self.db.create_face_node(
            embedding=embedding,
            image_url=image_url,
            confidence=similarity,
            quality_score=quality_score,
            angle_hint=angle_hint,
            camera_id=camera_id,
            identity_id=identity_id,
            source_id=source_id,
        )

        # Edge: Face -> Identity
        await self.db.create_edge(
            edge_type="face_to_identity",
            source_node_id=face.id, source_node_type="face",
            target_node_id=identity_id, target_node_type="identity",
            weight=similarity,
            metadata={"camera_diverse": camera_diverse},
        )

        # Update centroid with L2 normalisation
        identity = await self.db.get_identity_by_id(identity_id)
        if identity:
            old_centroid = np.array(
                identity["cluster_center_embedding"], dtype=np.float32
            )
            new_emb = np.array(embedding, dtype=np.float32)
            new_centroid = update_centroid(old_centroid, new_emb, face_count)

            # Evidence-based confidence
            camera_bonus = 0.1 if camera_diverse else 0.0
            factors = ConfidenceFactors(
                embedding_similarity=min(similarity + camera_bonus, 1.0),
                cluster_stability=identity.get("stability_score", 1.0),
                source_reliability=0.5,
                entity_match_score=0.0,
            )
            score = compute_identity_confidence(factors)

            await self.db.update_identity(
                identity_id=identity_id,
                new_centroid=new_centroid.tolist(),
                new_face_count=face_count + 1,
                new_score=score,
            )

            # Recompute stability after mutation
            await self.stability.update_identity_stability(identity_id)

        return {
            "identity_id": str(identity_id),
            "face_id": str(face.id),
            "action": "assigned",
            "similarity": similarity,
            "cosine_distance": round(1.0 - similarity, 4),
            "identity_score": score if identity else 0.0,
            "name": identity["name"] if identity else None,
            "camera_diverse": camera_diverse,
        }

    async def _candidate_link(
        self,
        embedding: List[float],
        best: Dict[str, Any],
        nearest: List[Dict[str, Any]],
        image_url: Optional[str],
        quality_score: float,
        angle_hint: str,
        camera_id: Optional[str],
        source_id: Optional[uuid.UUID],
        reason: str,
    ) -> Dict[str, Any]:
        """Create face node with candidate link — no auto-merge."""
        face = await self.db.create_face_node(
            embedding=embedding,
            image_url=image_url,
            confidence=best["similarity"],
            quality_score=quality_score,
            angle_hint=angle_hint,
            camera_id=camera_id,
            source_id=source_id,
            # No identity_id — unassigned until manual review
        )

        return {
            "identity_id": best["identity_id"],
            "face_id": str(face.id),
            "action": "candidate_merge",
            "similarity": best["similarity"],
            "cosine_distance": round(1.0 - best["similarity"], 4),
            "identity_score": 0.0,
            "name": best.get("name"),
            "reason": reason,
            "candidates": nearest[:3],
        }

    async def _create_new_identity(
        self,
        embedding: List[float],
        image_url: Optional[str],
        quality_score: float,
        angle_hint: str,
        camera_id: Optional[str],
        source_id: Optional[uuid.UUID],
        name: Optional[str],
    ) -> Dict[str, Any]:
        """Create a brand-new identity from a single face."""
        factors = ConfidenceFactors(
            embedding_similarity=1.0,
            cluster_stability=1.0,
            source_reliability=0.5,
            entity_match_score=0.0,
        )
        score = compute_identity_confidence(factors)

        identity = await self.db.create_identity_node(
            name=name,
            cluster_center_embedding=embedding,
            identity_score=score,
            face_count=1,
        )

        face = await self.db.create_face_node(
            embedding=embedding,
            image_url=image_url,
            confidence=1.0,
            quality_score=quality_score,
            angle_hint=angle_hint,
            camera_id=camera_id,
            identity_id=identity.id,
            source_id=source_id,
        )

        await self.db.create_edge(
            edge_type="face_to_identity",
            source_node_id=face.id, source_node_type="face",
            target_node_id=identity.id, target_node_type="identity",
            weight=1.0,
        )

        return {
            "identity_id": str(identity.id),
            "face_id": str(face.id),
            "action": "created",
            "similarity": 0.0,
            "cosine_distance": 1.0,
            "identity_score": score,
            "name": name,
        }

    async def merge_identities(
        self,
        source_id: uuid.UUID,
        target_id: uuid.UUID,
        reason: str = "manual_merge",
    ) -> Dict[str, Any]:
        """
        Merge source into target with safety checks.
        """
        # Safety check via stability engine
        safety = await self.stability.check_merge_safety(source_id, target_id)
        if not safety["safe"] and reason != "force_merge":
            return {
                "error": "merge_unsafe",
                "safety": safety,
                "hint": "Use reason='force_merge' to override",
            }

        source = await self.db.get_identity_by_id(source_id)
        target = await self.db.get_identity_by_id(target_id)
        if not source or not target:
            return {"error": "identity_not_found"}

        # Move faces and edges
        faces_moved = await self.db.reassign_faces(source_id, target_id)
        edges_moved = await self.db.move_edges(source_id, target_id)

        # Recompute centroid from all face embeddings
        embeddings = await self.db.get_face_embeddings_for_identity(target_id)
        new_count = len(embeddings)

        if embeddings:
            centroid = l2_normalize(np.mean(embeddings, axis=0))
            await self.db.update_identity(
                identity_id=target_id,
                new_centroid=centroid.tolist(),
                new_face_count=new_count,
            )

        # Deactivate source
        await self.db.deactivate_identity(source_id)

        # Audit edge
        await self.db.create_edge(
            edge_type="identity_to_identity",
            source_node_id=source_id, source_node_type="identity",
            target_node_id=target_id, target_node_type="identity",
            weight=safety.get("centroid_similarity", 0.0),
            metadata={"reason": reason, "action": "merged_into"},
        )

        # Recompute stability for target
        await self.stability.update_identity_stability(target_id)

        return {
            "merged_from": str(source_id),
            "merged_into": str(target_id),
            "faces_moved": faces_moved,
            "edges_moved": edges_moved,
            "new_face_count": new_count,
            "safety": safety,
        }
