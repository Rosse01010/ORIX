"""
Entity Linker — Proactive Cross-Modal OSINT Intelligence.

Enhanced entity linking with:
    - Wikipedia entity resolution (public API)
    - Wikidata structured data (P18 official image, P106 occupation,
      P108 employer, P27 citizenship)
    - Dataset label linking (LFW, CelebA, VGGFace2)
    - User-provided metadata enrichment

All sources are public and legal. No scraping or authentication bypass.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.osint_graph.storage.unified_db import UnifiedGraphDB

log = logging.getLogger(__name__)

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# Wikidata properties we extract
WIKIDATA_PROPERTIES = {
    "P18": "image",
    "P106": "occupation",
    "P108": "employer",
    "P27": "country_of_citizenship",
    "P569": "date_of_birth",
    "P19": "place_of_birth",
}


class EntityLinker:
    """
    Links identity nodes to external knowledge via public APIs.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.db = UnifiedGraphDB(session)

    async def link_identity(
        self,
        identity_id: uuid.UUID,
        name: str,
        search_wikipedia: bool = True,
        search_wikidata: bool = True,
        dataset_labels: Optional[List[str]] = None,
        user_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Search public sources and create entity links.
        Returns summary of all linked entities.
        """
        linked = []

        if search_wikipedia:
            for entity in (await self._search_wikipedia(name))[:3]:
                node = await self._create_or_get_entity(
                    entity_type="person",
                    name=entity["title"],
                    description=entity.get("snippet"),
                    external_id=f"wikipedia:{entity.get('pageid', '')}",
                    external_url=entity.get("url"),
                    metadata={"source": "wikipedia", "pageid": entity.get("pageid")},
                )
                await self.db.create_edge(
                    edge_type="identity_to_entity",
                    source_node_id=identity_id,
                    source_node_type="identity",
                    target_node_id=node.id,
                    target_node_type="entity",
                    weight=entity.get("relevance_score", 0.5),
                    metadata={"source": "wikipedia_search"},
                )
                linked.append({
                    "entity_id": str(node.id), "name": entity["title"],
                    "type": "wikipedia",
                    "confidence": entity.get("relevance_score", 0.5),
                })

        if search_wikidata:
            wikidata_results = await self._search_wikidata_entities(name)
            for entity in wikidata_results[:3]:
                node = await self._create_or_get_entity(
                    entity_type=entity.get("entity_type", "person"),
                    name=entity["label"],
                    description=entity.get("description"),
                    external_id=entity.get("qid"),
                    external_url=entity.get("url"),
                    metadata=entity.get("properties", {}),
                )
                await self.db.create_edge(
                    edge_type="identity_to_entity",
                    source_node_id=identity_id,
                    source_node_type="identity",
                    target_node_id=node.id,
                    target_node_type="entity",
                    weight=entity.get("relevance_score", 0.5),
                    metadata={
                        "source": "wikidata",
                        "qid": entity.get("qid"),
                        "has_p18": entity.get("has_p18", False),
                    },
                )
                linked.append({
                    "entity_id": str(node.id), "name": entity["label"],
                    "type": "wikidata", "qid": entity.get("qid"),
                    "confidence": entity.get("relevance_score", 0.5),
                    "has_official_image": entity.get("has_p18", False),
                    "properties": entity.get("properties", {}),
                })

        if dataset_labels:
            for label in dataset_labels:
                node = await self._create_or_get_entity(
                    entity_type="dataset", name=label,
                    description=f"Dataset label: {label}",
                    external_id=f"dataset:{label}",
                )
                await self.db.create_edge(
                    edge_type="identity_to_entity",
                    source_node_id=identity_id,
                    source_node_type="identity",
                    target_node_id=node.id,
                    target_node_type="entity",
                    weight=0.85,
                    metadata={"source": "dataset_label"},
                )
                linked.append({
                    "entity_id": str(node.id), "name": label,
                    "type": "dataset", "confidence": 0.85,
                })

        if user_metadata:
            node = await self._create_or_get_entity(
                entity_type="person",
                name=user_metadata.get("name", name),
                description=user_metadata.get("description"),
                external_id=f"user:{identity_id}",
                metadata=user_metadata,
            )
            await self.db.create_edge(
                edge_type="identity_to_entity",
                source_node_id=identity_id,
                source_node_type="identity",
                target_node_id=node.id,
                target_node_type="entity",
                weight=0.9,
                metadata={"source": "user_provided"},
            )
            linked.append({
                "entity_id": str(node.id),
                "name": user_metadata.get("name", name),
                "type": "user_metadata", "confidence": 0.9,
            })

        return {
            "identity_id": str(identity_id),
            "entities_linked": len(linked),
            "linked_entities": linked,
        }

    async def get_wikidata_p18_url(self, qid: str) -> Optional[str]:
        """
        Retrieve the official image URL (Property P18) for a Wikidata entity.
        Used by TruthAnchor for cross-modal verification.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    WIKIDATA_API,
                    params={
                        "action": "wbgetclaims",
                        "entity": qid,
                        "property": "P18",
                        "format": "json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            claims = data.get("claims", {}).get("P18", [])
            if not claims:
                return None

            filename = (
                claims[0]
                .get("mainsnak", {})
                .get("datavalue", {})
                .get("value", "")
            )
            if not filename:
                return None

            # Construct Wikimedia Commons URL
            filename_encoded = quote_plus(filename.replace(" ", "_"))
            md5 = __import__("hashlib").md5(
                filename.replace(" ", "_").encode()
            ).hexdigest()
            url = (
                f"https://upload.wikimedia.org/wikipedia/commons/"
                f"{md5[0]}/{md5[0:2]}/{filename_encoded}"
            )
            return url

        except Exception as e:
            log.warning("wikidata_p18_failed", qid=qid, error=str(e))
            return None

    async def get_wikidata_properties(
        self, qid: str
    ) -> Dict[str, Any]:
        """Fetch structured properties for a Wikidata entity."""
        properties = {}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                props = ",".join(WIKIDATA_PROPERTIES.keys())
                resp = await client.get(
                    WIKIDATA_API,
                    params={
                        "action": "wbgetclaims",
                        "entity": qid,
                        "property": props,
                        "format": "json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            claims = data.get("claims", {})
            for prop_id, prop_name in WIKIDATA_PROPERTIES.items():
                prop_claims = claims.get(prop_id, [])
                if prop_claims:
                    value = (
                        prop_claims[0]
                        .get("mainsnak", {})
                        .get("datavalue", {})
                        .get("value", "")
                    )
                    if isinstance(value, dict):
                        value = value.get("id", str(value))
                    properties[prop_name] = value

        except Exception as e:
            log.warning("wikidata_props_failed", qid=qid, error=str(e))

        return properties

    async def _search_wikipedia(self, query: str) -> List[Dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    WIKIPEDIA_API,
                    params={
                        "action": "query", "list": "search",
                        "srsearch": query, "srlimit": 5, "format": "json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            results = []
            for item in data.get("query", {}).get("search", []):
                title = item.get("title", "")
                relevance = 0.8 if query.lower() in title.lower() else 0.4
                results.append({
                    "title": title,
                    "pageid": item.get("pageid", 0),
                    "snippet": item.get("snippet", ""),
                    "url": f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}",
                    "relevance_score": relevance,
                })
            return results
        except Exception as e:
            log.warning("wikipedia_search_failed", query=query, error=str(e))
            return []

    async def _search_wikidata_entities(
        self, query: str
    ) -> List[Dict[str, Any]]:
        """Search Wikidata with property extraction."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    WIKIDATA_API,
                    params={
                        "action": "wbsearchentities",
                        "search": query, "language": "en",
                        "limit": 5, "format": "json",
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            results = []
            for item in data.get("search", []):
                qid = item.get("id", "")
                label = item.get("label", "")
                description = item.get("description", "")
                relevance = 0.8 if query.lower() == label.lower() else 0.4

                entity_type = "person"
                desc_lower = description.lower()
                if any(w in desc_lower for w in [
                    "company", "corporation", "organization", "organisation"
                ]):
                    entity_type = "organization"

                # Check for P18 availability
                props = await self.get_wikidata_properties(qid)
                has_p18 = "image" in props

                results.append({
                    "qid": qid, "label": label,
                    "description": description,
                    "url": f"https://www.wikidata.org/wiki/{qid}",
                    "entity_type": entity_type,
                    "relevance_score": relevance,
                    "has_p18": has_p18,
                    "properties": props,
                })
            return results

        except Exception as e:
            log.warning("wikidata_search_failed", query=query, error=str(e))
            return []

    async def _create_or_get_entity(
        self, entity_type: str, name: str,
        description: Optional[str] = None,
        external_id: Optional[str] = None,
        external_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if external_id:
            existing = await self.db.get_entity_by_external_id(external_id)
            if existing:
                class _Ref:
                    def __init__(self, eid): self.id = uuid.UUID(eid)
                return _Ref(existing["id"])

        return await self.db.create_entity_node(
            entity_type=entity_type, name=name,
            description=description, external_id=external_id,
            external_url=external_url, metadata=metadata,
        )
