"""
Belief ranking with configurable weighted scoring.
Combines confidence, relevance, recency, and tension.
"""

from typing import List
from uuid import UUID

from ...storage import Belief
from ..models.belief import utcnow

# default ranking weights
DEFAULT_WEIGHTS = {
    "confidence": 0.4,
    "relevance": 0.4,
    "recency": 0.1,
    "tension": 0.1,
}


def rank_beliefs(
    beliefs: List[Belief],
    relevance_scores: dict[UUID, float],
    weights: dict[str, float] | None = None,
) -> List[Belief]:
    """
    Rank beliefs by weighted score. Returns sorted list (highest first).

    Default weights: confidence 0.4, relevance 0.4, recency 0.1, tension 0.1
    """
    if not beliefs:
        return []

    # default weights if not provided
    if weights is None:
        weights = DEFAULT_WEIGHTS

    now = utcnow()
    scored_beliefs: list[tuple[Belief, float]] = []

    for b in beliefs:
        # get relevance score (default 0 if not found)
        relevance = relevance_scores.get(b.id, 0.0)

        # compute recency score (0-1, newer = higher)
        age_seconds = (now - b.updated_at).total_seconds()
        # simple 1 / (1 + age_hours) decay; fresh beliefs score higher
        recency = max(0.0, 1.0 / (1.0 + age_seconds / 3600.0))

        # weighted combination
        score = (
            weights.get("confidence", 0.0) * b.confidence
            + weights.get("relevance", 0.0) * relevance
            + weights.get("recency", 0.0) * recency
            + weights.get("tension", 0.0) * b.tension
        )

        scored_beliefs.append((b, score))

    # sort by score descending
    scored_beliefs.sort(key=lambda x: x[1], reverse=True)

    return [b for b, _ in scored_beliefs]


__all__ = ["rank_beliefs"]
