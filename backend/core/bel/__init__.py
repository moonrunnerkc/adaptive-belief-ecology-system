"""
Belief Ecology Loop (BEL) implementation.
"""

from .clustering import BeliefClusterManager, Cluster, ClusteringConfig
from .contradiction import compute_tensions
from .decay import apply_decay
from .loop import BeliefEcologyLoop
from .ranking import rank_beliefs
from .relevance import compute_relevance_scores
from .semantic_contradiction import (
    check_contradiction,
    ContradictionResult,
    Proposition,
    RuleBasedContradictionDetector,
)
from .snapshot_compression import compress_snapshot, decompress_snapshot
from .snapshot_logger import log_snapshot

__all__ = [
    "BeliefEcologyLoop",
    "BeliefClusterManager",
    "Cluster",
    "ClusteringConfig",
    "apply_decay",
    "compute_tensions",
    "compute_relevance_scores",
    "rank_beliefs",
    "log_snapshot",
    "compress_snapshot",
    "decompress_snapshot",
    "check_contradiction",
    "ContradictionResult",
    "Proposition",
    "RuleBasedContradictionDetector",
]
