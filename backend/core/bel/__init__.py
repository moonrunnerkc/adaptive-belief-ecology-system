"""
Belief Ecology Loop (BEL) implementation.
"""

from .contradiction import compute_tensions
from .decay import apply_decay
from .loop import BeliefEcologyLoop
from .ranking import rank_beliefs
from .relevance import compute_relevance_scores

__all__ = [
    "BeliefEcologyLoop",
    "apply_decay",
    "compute_tensions",
    "compute_relevance_scores",
    "rank_beliefs",
]
