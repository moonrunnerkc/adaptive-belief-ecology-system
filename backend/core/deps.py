"""
Dependency injection for FastAPI and other components.
Provides singletons for stores and config.
"""

from typing import AsyncGenerator

from ..storage import InMemoryBeliefStore, InMemorySnapshotStore
from ..storage.base import BeliefStoreABC, SnapshotStoreABC
from .config import ABESSettings, settings

# singleton stores for in-memory mode
_belief_store: InMemoryBeliefStore | None = None
_snapshot_store: InMemorySnapshotStore | None = None


async def get_belief_store() -> AsyncGenerator[BeliefStoreABC, None]:
    """Dependency that yields the belief store. In-memory for now."""
    global _belief_store
    if _belief_store is None:
        _belief_store = InMemoryBeliefStore()
    yield _belief_store


async def get_snapshot_store() -> AsyncGenerator[SnapshotStoreABC, None]:
    """Dependency that yields the snapshot store. In-memory for now."""
    global _snapshot_store
    if _snapshot_store is None:
        _snapshot_store = InMemorySnapshotStore()
    yield _snapshot_store


def get_settings() -> ABESSettings:
    """Get global settings. Not async since it's just a singleton."""
    return settings


__all__ = [
    "get_belief_store",
    "get_snapshot_store",
    "get_settings",
]
