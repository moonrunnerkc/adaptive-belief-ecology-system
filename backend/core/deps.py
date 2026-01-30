"""
Dependency injection for FastAPI and other components.
Provides singletons for stores and config.
"""

from typing import Optional

from ..storage import InMemoryBeliefStore, InMemorySnapshotStore
from ..storage.base import BeliefStoreABC, SnapshotStoreABC
from .config import ABESSettings, settings

# singleton stores for in-memory mode
_belief_store: Optional[InMemoryBeliefStore] = None
_snapshot_store: Optional[InMemorySnapshotStore] = None
_bel: Optional["BeliefEcologyLoop"] = None
_cluster_manager: Optional["BeliefClusterManager"] = None
_scheduler: Optional["AgentScheduler"] = None


def get_belief_store() -> BeliefStoreABC:
    """Get the belief store singleton."""
    global _belief_store
    if _belief_store is None:
        _belief_store = InMemoryBeliefStore()
    return _belief_store


def get_snapshot_store() -> SnapshotStoreABC:
    """Get the snapshot store singleton."""
    global _snapshot_store
    if _snapshot_store is None:
        _snapshot_store = InMemorySnapshotStore()
    return _snapshot_store


def get_bel() -> "BeliefEcologyLoop":
    """Get the BEL singleton."""
    global _bel
    if _bel is None:
        from .bel.loop import BeliefEcologyLoop
        _bel = BeliefEcologyLoop(
            belief_store=get_belief_store(),
            snapshot_store=get_snapshot_store(),
        )
    return _bel


def get_cluster_manager() -> "BeliefClusterManager":
    """Get the cluster manager singleton."""
    global _cluster_manager
    if _cluster_manager is None:
        from .bel.clustering import BeliefClusterManager
        _cluster_manager = BeliefClusterManager()
    return _cluster_manager


def get_scheduler() -> "AgentScheduler":
    """Get the agent scheduler singleton."""
    global _scheduler
    if _scheduler is None:
        from ..agents.scheduler import AgentScheduler
        _scheduler = AgentScheduler()
    return _scheduler


def get_settings() -> ABESSettings:
    """Get global settings."""
    return settings


def reset_singletons() -> None:
    """Reset all singletons. For testing."""
    global _belief_store, _snapshot_store, _bel, _cluster_manager, _scheduler
    _belief_store = None
    _snapshot_store = None
    _bel = None
    _cluster_manager = None
    _scheduler = None


__all__ = [
    "get_belief_store",
    "get_snapshot_store",
    "get_bel",
    "get_cluster_manager",
    "get_scheduler",
    "get_settings",
    "reset_singletons",
]
