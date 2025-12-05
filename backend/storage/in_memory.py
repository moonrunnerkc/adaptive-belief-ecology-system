"""Simple in-memory store for dev tests"""

import asyncio
from typing import List, Optional
from uuid import UUID

from ..core.models.belief import Belief, BeliefStatus
from ..core.models.snapshot import BeliefSnapshot, Snapshot, SnapshotDiff
from .base import BeliefStoreABC, SnapshotStoreABC


class InMemoryBeliefStore(BeliefStoreABC):
    """Dict-based belief storage. Good for tests, not for real workloads."""

    def __init__(self):
        self._beliefs: dict[UUID, Belief] = {}
        self._lock = asyncio.Lock()

    async def create(self, belief: Belief) -> Belief:
        async with self._lock:
            if belief.id in self._beliefs:
                raise ValueError(f"belief {belief.id} exists already")
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
            if status and b.status != status:
                continue
            if cluster_id and b.cluster_id != cluster_id:
                continue
            if tags and not any(t in b.tags for t in tags):
                continue
            if min_confidence is not None and b.confidence < min_confidence:
                continue
            if max_confidence is not None and b.confidence > max_confidence:
                continue
            results.append(b)

        results.sort(key=lambda b: b.updated_at, reverse=True)
        return results[offset : offset + limit]

    async def update(self, belief: Belief) -> Belief:
        async with self._lock:
            if belief.id not in self._beliefs:
                raise ValueError(f"no belief {belief.id}")
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
        # not implemented
        return []

    async def bulk_update(self, beliefs: List[Belief]) -> int:
        async with self._lock:
            for b in beliefs:
                if b.id in self._beliefs:
                    self._beliefs[b.id] = b
            return len(beliefs)


class InMemorySnapshotStore(SnapshotStoreABC):
    """Dict-based snapshot storage. Time travel for dev mode."""

    def __init__(self, compress: bool = True):
        self._snapshots: dict[UUID, Snapshot | bytes] = {}
        self._compressed: dict[UUID, bool] = {}
        self._lock = asyncio.Lock()
        self.compress = compress

    async def save_snapshot(self, snapshot: Snapshot) -> Snapshot:
        # local import avoids circular reference
        from ..core.bel.snapshot_compression import compress_snapshot

        async with self._lock:
            if snapshot.id in self._snapshots:
                raise ValueError(f"snapshot {snapshot.id} exists")

            if self.compress:
                compressed = compress_snapshot(snapshot)
                self._snapshots[snapshot.id] = compressed
                self._compressed[snapshot.id] = True
            else:
                self._snapshots[snapshot.id] = snapshot
                self._compressed[snapshot.id] = False

            return snapshot

    async def get_snapshot(self, snapshot_id: UUID) -> Optional[Snapshot]:
        # local import avoids circular reference
        from ..core.bel.snapshot_compression import decompress_snapshot

        data = self._snapshots.get(snapshot_id)
        if data is None:
            return None

        if self._compressed.get(snapshot_id, False):
            return decompress_snapshot(data)

        return data

    async def get_compressed_size(self, snapshot_id: UUID) -> int:
        """Get size of compressed snapshot in bytes. Returns 0 if not found or not compressed."""
        data = self._snapshots.get(snapshot_id)
        if data is None:
            return 0

        if self._compressed.get(snapshot_id, False):
            return len(data)

        return 0

    async def list_snapshots(
        self,
        min_iteration: Optional[int] = None,
        max_iteration: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Snapshot]:
        results = []
        for snapshot_id in self._snapshots.keys():
            s = await self.get_snapshot(snapshot_id)
            if s is None:
                continue

            # skip if outside iteration range
            if min_iteration is not None and s.metadata.iteration < min_iteration:
                continue
            if max_iteration is not None and s.metadata.iteration > max_iteration:
                continue
            results.append(s)

        results.sort(key=lambda s: s.metadata.iteration, reverse=True)
        return results[offset : offset + limit]

    async def compare_snapshots(
        self, snapshot_id_1: UUID, snapshot_id_2: UUID
    ) -> SnapshotDiff:
        s1 = await self.get_snapshot(snapshot_id_1)
        s2 = await self.get_snapshot(snapshot_id_2)

        if not s1:
            raise ValueError(f"missing snapshot {snapshot_id_1}")
        if not s2:
            raise ValueError(f"missing snapshot {snapshot_id_2}")

        return Snapshot.diff(s1, s2)


__all__ = ["InMemoryBeliefStore", "InMemorySnapshotStore"]
