"""
Belief Ecology Loop (BEL) - the heart of the system.
Runs the 7-step ecology process: load, decay, detect contradictions,
trigger actions, compute relevance, rank, and snapshot.
"""

from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from ...storage import Belief, BeliefStatus, Snapshot, SnapshotMetadata
from ...storage.base import BeliefStoreABC, SnapshotStoreABC
from ..config import ABESSettings
from ..models.belief import utcnow
from ..models.snapshot import BeliefSnapshot


class BeliefEcologyLoop:
    """
    Main ecology loop. Orchestrates belief dynamics: decay, contradiction
    detection, relevance scoring, and snapshotting.
    """

    def __init__(
        self,
        belief_store: BeliefStoreABC,
        snapshot_store: SnapshotStoreABC,
        settings: ABESSettings,
        embedding_model: Optional[SentenceTransformer] = None,
    ):
        self.belief_store = belief_store
        self.snapshot_store = snapshot_store
        self.settings = settings
        self.iteration_count = 0

        # lazy-load embedding model
        self._embedding_model = embedding_model

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy init for embedding model - only loads when needed."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.settings.embedding_model)
        return self._embedding_model

    async def run_iteration(self, context: str) -> tuple[List[Belief], Snapshot]:
        """
        Run one full BEL iteration. Returns ranked beliefs and the snapshot.

        Steps:
        1. Load active beliefs
        2. Apply decay
        3. Compute contradictions/tensions
        4. Trigger ecological actions
        5. Compute relevance to context
        6. Rank beliefs
        7. Log snapshot
        """
        self.iteration_count += 1

        # step 1: load active beliefs
        beliefs = await self._load_active_beliefs()

        # step 2: apply decay
        await self._apply_decay(beliefs)

        # step 3: compute contradictions and tensions
        await self._compute_tensions(beliefs)

        # step 4: trigger ecological actions (mutation/resolution scheduling)
        actions = await self._trigger_ecological_actions(beliefs)

        # step 5: compute relevance to current context
        await self._compute_relevance(beliefs, context)

        # step 6: rank beliefs
        ranked = self._rank_beliefs(beliefs)

        # step 7: log snapshot
        snapshot = await self._log_snapshot(beliefs, context, actions)

        return ranked, snapshot

    async def _load_active_beliefs(self) -> List[Belief]:
        """Step 1: Load all active beliefs from storage."""
        return await self.belief_store.list(
            status=BeliefStatus.Active,
            limit=1000,  # TODO: make configurable
        )

    async def _apply_decay(self, beliefs: List[Belief]) -> None:
        """
        Step 2: Apply time-based confidence decay.
        Decay formula: confidence *= decay_factor^(hours_since_reinforcement)
        """
        now = utcnow()
        updates = []

        for b in beliefs:
            # compute hours since last reinforcement
            delta = now - b.origin.last_reinforced
            hours = delta.total_seconds() / 3600.0

            # exponential decay with configurable half-life
            # default decay_rate of 0.95 means ~5% loss per hour
            decay_factor = 0.95  # TODO: pull from settings
            b.confidence *= decay_factor**hours

            # auto-transition to decaying if confidence drops
            if b.confidence < 0.3 and b.status == BeliefStatus.Active:
                b.status = BeliefStatus.Decaying

            b.updated_at = now
            updates.append(b)

        if updates:
            await self.belief_store.bulk_update(updates)

    async def _compute_tensions(self, beliefs: List[Belief]) -> None:
        """
        Step 3: Compute contradiction scores between belief pairs.
        Aggregate into per-belief tension values.

        TODO: Full contradiction detection
        - embed belief contents and find semantically similar pairs
        - use LLM to score actual contradictions (not just similarity)
        - aggregate scores into per-belief tension
        - for large N, consider belief ID map or spatial index
        """
        # stub - zeros out tension until we wire up contradiction detection
        for b in beliefs:
            b.tension = 0.0

    async def _trigger_ecological_actions(self, beliefs: List[Belief]) -> List[dict]:
        """
        Step 4: Schedule mutations or resolutions based on tension/confidence.

        High tension + low confidence → mutation candidate
        High tension + high confidence → resolution candidate
        """
        actions = []

        for b in beliefs:
            if b.tension > 0.7:  # TODO: pull threshold from settings
                if b.confidence < 0.5:
                    # schedule mutation
                    actions.append(
                        {
                            "type": "mutation_candidate",
                            "belief_id": str(b.id),
                            "tension": b.tension,
                            "confidence": b.confidence,
                        }
                    )
                elif b.confidence > 0.7:
                    # schedule resolution
                    actions.append(
                        {
                            "type": "resolution_candidate",
                            "belief_id": str(b.id),
                            "tension": b.tension,
                            "confidence": b.confidence,
                        }
                    )

        return actions

    async def _compute_relevance(self, beliefs: List[Belief], context: str) -> None:
        """
        Step 5: Compute relevance of each belief to current context via embedding similarity.
        """
        if not context.strip():
            # no context, set all relevance to 0
            for b in beliefs:
                b.relevance = 0.0  # type: ignore
            return

        # embed context
        context_embedding = self.embedding_model.encode(context, convert_to_numpy=True)

        # embed all belief contents
        contents = [b.content for b in beliefs]
        if not contents:
            return

        belief_embeddings = self.embedding_model.encode(contents, convert_to_numpy=True)

        # compute cosine similarity
        for i, b in enumerate(beliefs):
            similarity = np.dot(context_embedding, belief_embeddings[i]) / (
                np.linalg.norm(context_embedding) * np.linalg.norm(belief_embeddings[i])
            )
            # normalize to 0-1 range, cast from numpy to float
            b.relevance = float(max(0.0, min(1.0, similarity)))  # type: ignore

    def _rank_beliefs(self, beliefs: List[Belief]) -> List[Belief]:
        """
        Step 6: Rank beliefs by weighted formula.
        Score = w_rel * relevance + w_conf * confidence + w_rec * recency + w_ten * tension
        """
        # configurable weights (TODO: pull from settings or RL policy)
        w_relevance = 0.4
        w_confidence = 0.3
        w_recency = 0.2
        w_tension = 0.1

        now = utcnow()

        for b in beliefs:
            # compute recency score (0-1, with 1 being most recent)
            hours_old = (now - b.updated_at).total_seconds() / 3600.0
            recency = 1.0 / (1.0 + hours_old)  # decay function

            relevance = getattr(b, "relevance", 0.0)

            # weighted score
            score = (
                w_relevance * relevance
                + w_confidence * b.confidence
                + w_recency * recency
                + w_tension * b.tension
            )

            b.score = score  # type: ignore

        # sort by score desc
        ranked = sorted(beliefs, key=lambda b: getattr(b, "score", 0.0), reverse=True)
        return ranked

    async def _log_snapshot(
        self,
        beliefs: List[Belief],
        context: str,
        actions: List[dict],
    ) -> Snapshot:
        """
        Step 7: Persist current ecology state as a snapshot.
        """
        # convert beliefs to frozen snapshots
        belief_snapshots = [
            BeliefSnapshot(
                id=b.id,
                content=b.content,
                confidence=b.confidence,
                origin=b.origin,
                tags=b.tags,
                tension=b.tension,
                cluster_id=b.cluster_id,
                status=b.status,
                parent_id=b.parent_id,
                use_count=b.use_count,
                created_at=b.created_at,
                updated_at=b.updated_at,
            )
            for b in beliefs
        ]

        # compute global tension
        global_tension = sum(b.tension for b in beliefs) / max(len(beliefs), 1)

        snapshot = Snapshot(
            metadata=SnapshotMetadata(
                iteration=self.iteration_count,
                context_summary=context[:200] if context else None,
            ),
            beliefs=belief_snapshots,
            global_tension=global_tension,
            cluster_metrics={},  # TODO: compute cluster stats
            agent_actions=actions,
            rl_state_action=None,  # TODO: RL integration
        )

        await self.snapshot_store.save_snapshot(snapshot)
        return snapshot


__all__ = ["BeliefEcologyLoop"]
