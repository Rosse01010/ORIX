"""
Truth Anchor — Master Truth Node Management.

Manages verified OSINT reference embeddings for cross-modal verification.

Pipeline:
    1. Identity has a name (e.g., "Tim Cook")
    2. EntityLinker finds Wikidata QID (e.g., Q215079)
    3. TruthAnchor retrieves P18 (official image URL)
    4. Downloads image → generates reference embedding (via InsightFace)
    5. Compares reference embedding to identity centroid
    6. If similarity >= 0.85 → create MasterTruthNode, promote to VERIFIED

All images come from Wikimedia Commons (public domain / free license).
"""
from __future__ import annotations

import io
import logging
import uuid
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.osint_graph.intelligence.entity_linker import EntityLinker
from app.osint_graph.storage.unified_db import UnifiedGraphDB
from app.osint_graph.utils.normalization import cosine_similarity, l2_normalize

log = logging.getLogger(__name__)

VERIFICATION_THRESHOLD = 0.85


class TruthAnchor:
    """
    Cross-modal verification engine.

    Downloads official reference images from Wikidata P18,
    generates embeddings, and verifies identity nodes.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.db = UnifiedGraphDB(session)
        self.entity_linker = EntityLinker(session)

    async def verify_identity(
        self,
        identity_id: uuid.UUID,
        name: str,
        reference_embedding: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Attempt to verify an identity against OSINT reference data.

        If reference_embedding is provided, uses it directly.
        Otherwise, searches Wikidata for the name, retrieves P18
        image, and attempts to generate a reference embedding.

        Returns verification result.
        """
        identity = await self.db.get_identity_by_id(identity_id)
        if not identity:
            return {"verified": False, "reason": "identity_not_found"}

        centroid = np.array(
            identity["cluster_center_embedding"], dtype=np.float32
        )
        if np.linalg.norm(centroid) < 1e-10:
            return {"verified": False, "reason": "empty_centroid"}

        # If reference embedding provided directly, use it
        if reference_embedding and len(reference_embedding) == 512:
            return await self._verify_with_embedding(
                identity_id=identity_id,
                centroid=centroid,
                ref_embedding=np.array(reference_embedding, dtype=np.float32),
                source_type="user_provided",
                source_url=None,
                external_id=None,
            )

        # Otherwise, search Wikidata for P18 official image
        return await self._verify_via_wikidata(
            identity_id=identity_id,
            name=name,
            centroid=centroid,
        )

    async def _verify_via_wikidata(
        self,
        identity_id: uuid.UUID,
        name: str,
        centroid: np.ndarray,
    ) -> Dict[str, Any]:
        """Search Wikidata, download P18 image, generate embedding, verify."""
        # Search for matching Wikidata entity
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://www.wikidata.org/w/api.php",
                    params={
                        "action": "wbsearchentities",
                        "search": name,
                        "language": "en",
                        "limit": 3,
                        "format": "json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            return {"verified": False, "reason": f"wikidata_search_failed: {e}"}

        candidates = data.get("search", [])
        if not candidates:
            return {"verified": False, "reason": "no_wikidata_match"}

        # Try each candidate for P18 image
        for candidate in candidates:
            qid = candidate.get("id", "")
            label = candidate.get("label", "")

            p18_url = await self.entity_linker.get_wikidata_p18_url(qid)
            if not p18_url:
                continue

            # Download the reference image
            ref_embedding = await self._download_and_embed(p18_url)
            if ref_embedding is None:
                continue

            result = await self._verify_with_embedding(
                identity_id=identity_id,
                centroid=centroid,
                ref_embedding=ref_embedding,
                source_type="wikidata_p18",
                source_url=p18_url,
                external_id=qid,
            )

            if result["verified"]:
                # Update identity name if not set
                identity = await self.db.get_identity_by_id(identity_id)
                if identity and not identity.get("name"):
                    await self.db.update_identity(
                        identity_id=identity_id, name=label,
                    )
                result["qid"] = qid
                result["label"] = label
                result["p18_url"] = p18_url
                return result

        return {
            "verified": False,
            "reason": "no_p18_match_above_threshold",
            "candidates_checked": len(candidates),
        }

    async def _verify_with_embedding(
        self,
        identity_id: uuid.UUID,
        centroid: np.ndarray,
        ref_embedding: np.ndarray,
        source_type: str,
        source_url: Optional[str],
        external_id: Optional[str],
    ) -> Dict[str, Any]:
        """Compare reference embedding to centroid and create truth node if matched."""
        ref_norm = l2_normalize(ref_embedding)
        similarity = cosine_similarity(centroid, ref_norm)

        verified = similarity >= VERIFICATION_THRESHOLD

        if verified:
            # Create MasterTruthNode
            truth_node = await self.db.create_master_truth_node(
                identity_id=identity_id,
                reference_embedding=ref_norm.tolist(),
                source_type=source_type,
                source_url=source_url,
                external_id=external_id,
                match_similarity=similarity,
                verified=True,
            )

            # Create edge: Identity -> TruthNode
            await self.db.create_edge(
                edge_type="identity_to_truth",
                source_node_id=identity_id,
                source_node_type="identity",
                target_node_id=truth_node.id,
                target_node_type="truth",
                weight=similarity,
                metadata={
                    "source_type": source_type,
                    "verification": "automated",
                },
            )

            # Promote identity to VERIFIED
            await self.db.update_identity(
                identity_id=identity_id, verified=True,
            )

            log.info(
                "identity_verified",
                identity_id=str(identity_id),
                similarity=round(similarity, 4),
                source_type=source_type,
            )

        return {
            "verified": verified,
            "similarity": round(similarity, 4),
            "threshold": VERIFICATION_THRESHOLD,
            "source_type": source_type,
            "source_url": source_url,
            "truth_node_id": str(truth_node.id) if verified else None,
        }

    async def _download_and_embed(
        self, image_url: str
    ) -> Optional[np.ndarray]:
        """
        Download an image and generate a 512D ArcFace embedding.

        Uses InsightFace if available, otherwise returns None
        (the caller should provide pre-computed embeddings).
        """
        try:
            async with httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=True,
                headers={"User-Agent": "ORIX-OSINT/2.0 (research; contact@example.com)"},
            ) as client:
                resp = await client.get(image_url)
                resp.raise_for_status()
                image_bytes = resp.content

            if len(image_bytes) < 1000:
                return None

            # Try to use InsightFace for embedding
            try:
                import cv2
                from app.utils.gpu_utils import build_detector, build_embedder

                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    return None

                detector = build_detector()
                faces = detector.detect(img)
                if not faces:
                    return None

                embedder = build_embedder()
                embedding = embedder.embed(faces[0].crop)
                return l2_normalize(
                    np.array(embedding, dtype=np.float32)
                )

            except ImportError:
                log.debug(
                    "insightface_unavailable_for_truth_anchor",
                    url=image_url,
                )
                return None

        except Exception as e:
            log.warning(
                "truth_anchor_download_failed",
                url=image_url, error=str(e),
            )
            return None

    async def get_truth_anchors_for_identity(
        self, identity_id: uuid.UUID
    ) -> List[Dict[str, Any]]:
        """Get all truth anchors linked to an identity."""
        from sqlalchemy import text as sa_text

        result = await self.session.execute(
            sa_text(
                "SELECT id, source_type, source_url, external_id, "
                "match_similarity, verified, created_at "
                "FROM graph_master_truth_nodes "
                "WHERE identity_id = :id ORDER BY created_at DESC"
            ),
            {"id": identity_id},
        )
        return [
            {
                "id": str(r[0]), "source_type": r[1],
                "source_url": r[2], "external_id": r[3],
                "match_similarity": r[4], "verified": r[5],
                "created_at": str(r[6]),
            }
            for r in result.fetchall()
        ]
