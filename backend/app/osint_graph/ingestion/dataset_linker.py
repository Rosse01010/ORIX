"""
Dataset Linker — connects faces to public dataset labels.

Supports linking identities to labels from:
    - LFW (Labeled Faces in the Wild) person names
    - CelebA attribute tags
    - VGGFace2, WikiFace
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.osint_graph.storage.unified_db import UnifiedGraphDB

log = logging.getLogger(__name__)

KNOWN_DATASETS = {
    "lfw": {
        "name": "Labeled Faces in the Wild",
        "url": "http://vis-www.cs.umass.edu/lfw/",
        "reliability": 0.9,
    },
    "celeba": {
        "name": "CelebFaces Attributes Dataset",
        "url": "https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html",
        "reliability": 0.85,
    },
    "vggface2": {
        "name": "VGGFace2",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/",
        "reliability": 0.9,
    },
    "wikiface": {
        "name": "WikiFace",
        "url": "https://www.wikidata.org/",
        "reliability": 0.75,
    },
}


class DatasetLinker:

    def __init__(self, session: AsyncSession):
        self.session = session
        self.db = UnifiedGraphDB(session)

    async def create_dataset_source(self, dataset_key: str) -> Optional[str]:
        info = KNOWN_DATASETS.get(dataset_key)
        if not info:
            return None
        node = await self.db.create_source_node(
            source_type="dataset", name=info["name"],
            url=info["url"], reliability_score=info["reliability"],
            metadata={"dataset_key": dataset_key},
        )
        return str(node.id)

    async def link_dataset_labels(
        self, identity_id: uuid.UUID, dataset_key: str, labels: List[str],
    ) -> Dict[str, Any]:
        linked = []
        for label in labels:
            entity = await self.db.create_entity_node(
                entity_type="dataset", name=label,
                description=f"Label from {dataset_key}: {label}",
                external_id=f"{dataset_key}:{label}",
                metadata={"dataset": dataset_key, "label": label},
            )
            await self.db.create_edge(
                edge_type="identity_to_entity",
                source_node_id=identity_id, source_node_type="identity",
                target_node_id=entity.id, target_node_type="entity",
                weight=0.85,
                metadata={"source": f"{dataset_key}_label"},
            )
            linked.append({"entity_id": str(entity.id), "label": label, "dataset": dataset_key})
        return {"identity_id": str(identity_id), "labels_linked": len(linked), "linked": linked}
