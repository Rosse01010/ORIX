"""
Risk & Confidence Scoring Engine for the OSINT Identity Graph (Level 2).

Computes identity_confidence_score as a weighted sum of:
    - embedding_similarity   (how close faces are to cluster center)
    - cluster_stability      (variance within the cluster)
    - source_reliability     (quality of data sources)
    - entity_matches         (number of corroborating entities)

Normalised to a 0-100 scale.

Also provides volatility classification and threshold constants.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from app.osint_graph.utils.normalization import cosine_similarity, l2_normalize


# ── Scoring weights ──────────────────────────────────────────────────────────

WEIGHT_EMBEDDING_SIM = 0.35
WEIGHT_CLUSTER_STABILITY = 0.25
WEIGHT_SOURCE_RELIABILITY = 0.20
WEIGHT_ENTITY_MATCHES = 0.20


# ── Clustering thresholds (cosine similarity) ────────────────────────────────

THRESHOLD_SAME_IDENTITY = 0.85     # cos_dist < 0.15
THRESHOLD_CANDIDATE_MERGE = 0.75   # cos_dist < 0.25
THRESHOLD_NEW_IDENTITY = 0.75      # cos_dist >= 0.25


# ── Volatility thresholds ────────────────────────────────────────────────────

VOLATILITY_REVIEW = 0.5
VOLATILITY_ALERT = 0.7


@dataclass
class ConfidenceFactors:
    """Individual factors that compose the identity confidence score."""
    embedding_similarity: float = 0.0
    cluster_stability: float = 0.0
    source_reliability: float = 0.0
    entity_match_score: float = 0.0


def compute_identity_confidence(factors: ConfidenceFactors) -> float:
    """
    Compute identity confidence score on 0-100 scale.
    Each factor is expected to be in [0.0, 1.0] range.
    """
    raw = (
        WEIGHT_EMBEDDING_SIM * factors.embedding_similarity
        + WEIGHT_CLUSTER_STABILITY * factors.cluster_stability
        + WEIGHT_SOURCE_RELIABILITY * factors.source_reliability
        + WEIGHT_ENTITY_MATCHES * factors.entity_match_score
    )
    return round(min(max(raw * 100.0, 0.0), 100.0), 2)


def compute_cluster_stability(embeddings: List[np.ndarray]) -> float:
    """
    Measure how tightly clustered the face embeddings are.
    Returns 1.0 for perfectly clustered, 0.0 for scattered.
    """
    if len(embeddings) < 2:
        return 1.0

    centroid = l2_normalize(np.mean(embeddings, axis=0))
    similarities = [cosine_similarity(centroid, emb) for emb in embeddings]
    mean_sim = float(np.mean(similarities))
    return min(max((mean_sim - 0.5) * 2.0, 0.0), 1.0)


def compute_source_reliability(source_scores: List[float]) -> float:
    """Aggregate reliability from multiple sources."""
    if not source_scores:
        return 0.0
    return float(np.mean(source_scores))


def compute_entity_match_score(
    entity_count: int, max_expected: int = 5
) -> float:
    """Diminishing returns: 1 entity = 0.2, 3 = 0.6, 5+ = 1.0."""
    if entity_count <= 0:
        return 0.0
    return min(entity_count / max_expected, 1.0)


def classify_similarity(similarity: float) -> str:
    """
    Classify cosine similarity into identity resolution tiers.

    > 0.85  → same_identity    (auto-assign, cosine_distance < 0.15)
    0.75-0.85 → candidate_merge  (flag for review, cosine_distance 0.15-0.25)
    < 0.75  → new_identity     (create fresh cluster)
    """
    if similarity >= THRESHOLD_SAME_IDENTITY:
        return "same_identity"
    elif similarity >= THRESHOLD_CANDIDATE_MERGE:
        return "candidate_merge"
    else:
        return "new_identity"


def classify_volatility(volatility: float) -> str:
    """Classify identity volatility level."""
    if volatility >= VOLATILITY_ALERT:
        return "critical"
    elif volatility >= VOLATILITY_REVIEW:
        return "high"
    elif volatility >= 0.3:
        return "moderate"
    else:
        return "stable"
