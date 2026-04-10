"""
Stability Engine — Identity Volatility Metrics.

Tracks how tightly clustered an identity's embeddings are and whether
they are drifting over time. High volatility indicates a potential
"Identity Collision" (two distinct people being merged by mistake).

Metrics:
    stability_score (0-1): How tightly clustered the embeddings are.
        1.0 = all embeddings nearly identical (reliable identity)
        0.0 = embeddings scattered across hypersphere (unreliable)

    volatility (0-1): Rate of centroid drift.
        0.0 = stable (centroid not moving)
        1.0 = highly volatile (centroid shifting significantly)

Actions:
    volatility >= 0.5 → needs_review = True (manual audit)
    volatility >= 0.7 → emit alert via /api/osint-graph/alerts/volatility
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.osint_graph.storage.unified_db import UnifiedGraphDB
from app.osint_graph.utils.normalization import cosine_similarity, l2_normalize

log = logging.getLogger(__name__)

# Thresholds
VOLATILITY_REVIEW_THRESHOLD = 0.5
VOLATILITY_ALERT_THRESHOLD = 0.7


class StabilityEngine:
    """
    Computes and updates identity stability and volatility metrics.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.db = UnifiedGraphDB(session)

    async def compute_stability(
        self, identity_id: uuid.UUID
    ) -> Dict[str, float]:
        """
        Compute stability metrics for an identity from its face embeddings.

        Returns:
            {
                "stability_score": float (0-1),
                "volatility": float (0-1),
                "mean_intra_sim": float,
                "min_intra_sim": float,
                "distinct_cameras": int,
            }
        """
        embeddings = await self.db.get_face_embeddings_for_identity(identity_id)

        if len(embeddings) < 2:
            return {
                "stability_score": 1.0,
                "volatility": 0.0,
                "mean_intra_sim": 1.0,
                "min_intra_sim": 1.0,
                "distinct_cameras": await self.db.get_distinct_cameras_for_identity(identity_id),
            }

        # Compute centroid
        centroid = l2_normalize(np.mean(embeddings, axis=0))

        # Similarity of each embedding to centroid
        sims_to_centroid = [
            cosine_similarity(centroid, emb) for emb in embeddings
        ]

        mean_sim = float(np.mean(sims_to_centroid))
        min_sim = float(np.min(sims_to_centroid))
        std_sim = float(np.std(sims_to_centroid))

        # Stability: map mean similarity from [0.5, 1.0] → [0.0, 1.0]
        stability = min(max((mean_sim - 0.5) * 2.0, 0.0), 1.0)

        # Volatility: high std + low min_sim = volatile
        # A well-clustered identity has std < 0.05 and min_sim > 0.7
        volatility = min(std_sim * 5.0 + max(0.0, 0.7 - min_sim), 1.0)

        distinct_cameras = await self.db.get_distinct_cameras_for_identity(identity_id)

        return {
            "stability_score": round(stability, 4),
            "volatility": round(volatility, 4),
            "mean_intra_sim": round(mean_sim, 4),
            "min_intra_sim": round(min_sim, 4),
            "distinct_cameras": distinct_cameras,
        }

    async def update_identity_stability(
        self, identity_id: uuid.UUID
    ) -> Dict[str, Any]:
        """
        Recompute and persist stability metrics for an identity.
        Flags for review if volatility exceeds threshold.
        """
        metrics = await self.compute_stability(identity_id)

        needs_review = metrics["volatility"] >= VOLATILITY_REVIEW_THRESHOLD

        await self.db.update_identity(
            identity_id=identity_id,
            stability_score=metrics["stability_score"],
            volatility=metrics["volatility"],
            distinct_cameras=metrics["distinct_cameras"],
            needs_review=needs_review,
        )

        if needs_review:
            log.warning(
                "identity_high_volatility",
                identity_id=str(identity_id),
                volatility=metrics["volatility"],
                stability=metrics["stability_score"],
            )

        return {
            "identity_id": str(identity_id),
            **metrics,
            "needs_review": needs_review,
        }

    async def check_merge_safety(
        self,
        source_id: uuid.UUID,
        target_id: uuid.UUID,
    ) -> Dict[str, Any]:
        """
        Evaluate whether merging two identities is safe.

        Returns safety assessment with recommendation.
        """
        source = await self.db.get_identity_by_id(source_id)
        target = await self.db.get_identity_by_id(target_id)

        if not source or not target:
            return {"safe": False, "reason": "identity_not_found"}

        # Get centroids
        src_centroid = np.array(
            source["cluster_center_embedding"], dtype=np.float32
        )
        tgt_centroid = np.array(
            target["cluster_center_embedding"], dtype=np.float32
        )

        centroid_sim = cosine_similarity(src_centroid, tgt_centroid)

        # Safety checks
        reasons = []

        if centroid_sim < 0.70:
            reasons.append(
                f"low_centroid_similarity ({centroid_sim:.3f} < 0.70)"
            )
        if target["volatility"] >= VOLATILITY_REVIEW_THRESHOLD:
            reasons.append(
                f"target_high_volatility ({target['volatility']:.3f})"
            )
        if source["volatility"] >= VOLATILITY_REVIEW_THRESHOLD:
            reasons.append(
                f"source_high_volatility ({source['volatility']:.3f})"
            )

        safe = len(reasons) == 0
        return {
            "safe": safe,
            "centroid_similarity": round(centroid_sim, 4),
            "source_volatility": source["volatility"],
            "target_volatility": target["volatility"],
            "recommendation": "proceed" if safe else "manual_review",
            "reasons": reasons,
        }
