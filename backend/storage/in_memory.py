"""
In-memory storage for dev/testing. Not for production use.
"""

import asyncio
from typing import List, Optional
from uuid import UUID

from ..core.models.belief import Belief, BeliefStatus
from ..core.models.snapshot import BeliefSnapshot, Snapshot
from .base import BeliefStoreABC, SnapshotDiff, SnapshotStoreABC


class InMemoryBeliefStore(BeliefStoreABC):
    """Dict-based belief storage. Good for tests, not for real workloads."""

    def __init__(self):
        self._beliefs: dict[UUID, Belief] = {}
        self._lock = asyncio.Lock()

    async def create(self, belief: Belief) -> Belief:
        async with self._lock:
            if belief.id in self._beliefs:
                raise ValueError(f"Belief {belief.id} already exists")
            self._beliefs[belief.id] = belief
            return belief

    async def get(self, belief_id: UUID) -> Optional[Belief]:
        return self._beliefs.get(belief_id)

    async def list(
        self,
        status: Optional[BeliefStatus] = None,
        cluster_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Belief]:
        results = []
        for b in self._beliefs.values():
            # filter by status
            if status and b.status != status:
                continue
            # filter by cluster
            if cluster_id and b.cluster_id != cluster_id:
                continue
            # filter by tags (at least one match)
            if tags and not any(t in b.tags for t in tags):
                continue
            # filter by confidence range
            if min_confidence is not None and b.confidence < min_confidence:
                continue
            if max_confidence is not None and b.confidence > max_confidence:
                continue
            results.append(b)

        # sort by updated_at desc (convention, not enforced by interface)
        results.sort(key=lambda b: b.updated_at, reverse=True)
        return results[offset : offset + limit]

    async def update(self, belief: Belief) -> Belief:
        async with self._lock:
            if belief.id not in self._beliefs:
                raise ValueError(f"Belief {belief.id} does not exist")
            self._beliefs[belief.id] = belief
            return belief

    async def delete(self, belief_id: UUID) -> bool:
        async with self._lock:
            if belief_id in self._beliefs:
                del self._beliefs[belief_id]
                return True
            return False

    async def search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        status: Optional[BeliefStatus] = None,
    ) -> List[Belief]:
        """
        Stub for now - embedding search needs a vector index.
        Returns empty list until we wire up actual similarity computation.
        """
        # TODO: implement cosine similarity once we have embeddings on beliefs
        return []

    async def bulk_update(self, beliefs: List[Belief]) -> int:
        async with self._lock:
            count = 0
            for b in beliefs:
                if b.id in self._beliefs:
                    self._beliefs[b.id] = b
                    count += 1
            return count


class InMemorySnapshotStore(SnapshotStoreABC):
    """Dict-based snapshot storage. Time travel for dev mode."""

    def __init__(self):
        self._snapshots: dict[UUID, Snapshot] = {}
        self._lock = asyncio.Lock()

    async def save_snapshot(self, snapshot: Snapshot) -> Snapshot:
        async with self._lock:
            if snapshot.id in self._snapshots:
                raise ValueError(f"Snapshot {snapshot.id} already exists")
            self._snapshots[snapshot.id] = snapshot
            return snapshot

    async def get_snapshot(self, snapshot_id: UUID) -> Optional[Snapshot]:
        return self._snapshots.get(snapshot_id)

    async def list_snapshots(
        self,
        min_iteration: Optional[int] = None,
        max_iteration: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Snapshot]:
        results = []
        for s in self._snapshots.values():
            # filter by iteration range
            if min_iteration is not None and s.metadata.iteration < min_iteration:
                continue
            if max_iteration is not None and s.metadata.iteration > max_iteration:
                continue
            results.append(s)

        # sort by iteration asc
        results.sort(key=lambda s: s.metadata.iteration)
        return results[offset : offset + limit]

    async def compare_snapshots(
        self, snapshot_id_1: UUID, snapshot_id_2: UUID
    ) -> SnapshotDiff:
        s1 = self._snapshots.get(snapshot_id_1)
        s2 = self._snapshots.get(snapshot_id_2)

        if not s1:
            raise ValueError(f"Snapshot {snapshot_id_1} not found")
        if not s2:
            raise ValueError(f"Snapshot {snapshot_id_2} not found")

        # build sets of belief IDs
        ids_1 = {b.id for b in s1.beliefs}
        ids_2 = {b.id for b in s2.beliefs}

        added = list(ids_2 - ids_1)
        removed = list(ids_1 - ids_2)

        # check for modifications (same ID but different content/confidence)
        modified = []
        for b1 in s1.beliefs:
            if b1.id in ids_2:
                b2 = next(b for b in s2.beliefs if b.id == b1.id)
                if b1.content != b2.content or b1.confidence != b2.confidence:
                    modified.append(b1.id)

        return SnapshotDiff(
            beliefs_added=added,
            beliefs_removed=removed,
            beliefs_modified=modified,
            tension_delta=s2.global_tension - s1.global_tension,
            iteration_delta=s2.metadata.iteration - s1.metadata.iteration,
        )


__all__ = ["InMemoryBeliefStore", "InMemorySnapshotStore"]
