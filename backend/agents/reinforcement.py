# Author: Bradley R. Kinnard
"""
ReinforcementAgent - boosts confidence of beliefs relevant to incoming context.
"""

import re
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

from ..core.models.belief import Belief
from ..storage.base import BeliefStoreABC


# if similarity > this, belief gets reinforced
RELEVANCE_THRESHOLD = 0.6  # lowered for chat paraphrasing

# confidence boost per reinforcement (additive, capped at 1.0)
CONFIDENCE_BOOST = 0.05  # smaller boost, more reinforcements needed

# min seconds between reinforcements for same belief
COOLDOWN_SECONDS = 10  # lowered for interactive chat

# max confidence a belief can reach via reinforcement alone
MAX_REINFORCED_CONFIDENCE = 0.95


def _extract_numbers(text: str) -> list[tuple[float, str]]:
    """Extract numbers with their context (unit hints)."""
    pattern = r'(\d+(?:\.\d+)?)\s*(%|degrees?|°|F|C|mph|dollars?|\$|minutes?|hours?|days?|years?|feet|ft|miles?|lbs?|kg)?'
    matches = re.findall(pattern, text, re.IGNORECASE)
    results = []
    for num_str, unit in matches:
        try:
            num = float(num_str)
            results.append((num, unit.lower() if unit else ""))
        except ValueError:
            pass
    return results


def _has_numeric_conflict(text1: str, text2: str) -> bool:
    """Check if two texts have conflicting numeric values."""
    nums1 = _extract_numbers(text1)
    nums2 = _extract_numbers(text2)

    if not nums1 or not nums2:
        return False

    for n1, u1 in nums1:
        for n2, u2 in nums2:
            # Units should match (or both be empty)
            if u1 != u2 and u1 and u2:
                continue
            # Check if numbers differ significantly (>20% difference)
            if n1 == 0 and n2 == 0:
                continue
            max_val = max(abs(n1), abs(n2))
            if max_val > 0:
                diff_pct = abs(n1 - n2) / max_val
                if diff_pct > 0.2:
                    return True
    return False


def _cosine_batch(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Dot product for normalized vecs. Same as cosine sim."""
    return candidates @ query


class ReinforcementAgent:
    """Finds beliefs relevant to incoming text and reinforces them."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError("pip install sentence-transformers") from e
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _is_on_cooldown(self, belief: Belief) -> bool:
        """True if belief was reinforced too recently."""
        now = datetime.now(timezone.utc)
        elapsed = (now - belief.origin.last_reinforced).total_seconds()
        return elapsed < COOLDOWN_SECONDS

    async def _get_embeddings(
        self, beliefs: List[Belief], store: BeliefStoreABC
    ) -> np.ndarray:
        """Load stored embeddings where available, embed the rest."""
        model = self._get_model()
        embs = []
        need_encode_idx = []
        need_encode_texts = []

        for idx, b in enumerate(beliefs):
            stored = None
            if hasattr(store, "get_embedding"):
                try:
                    stored = await store.get_embedding(b.id)
                except Exception:
                    pass

            if stored is not None:
                embs.append((idx, np.array(stored)))
            else:
                need_encode_idx.append(idx)
                need_encode_texts.append(b.content)

        # batch encode missing
        if need_encode_texts:
            new_embs = model.encode(
                need_encode_texts, convert_to_numpy=True, normalize_embeddings=True
            )
            for i, orig_idx in enumerate(need_encode_idx):
                embs.append((orig_idx, new_embs[i]))

        # sort back to original order
        embs.sort(key=lambda x: x[0])
        return np.array([e[1] for e in embs])

    async def reinforce(
        self, incoming: str, beliefs: List[Belief], store: BeliefStoreABC
    ) -> List[Belief]:
        """
        Find beliefs similar to incoming text. Boost their confidence,
        update last_reinforced, increment use_count, persist.

        Returns list of reinforced beliefs.
        """
        if not incoming or not incoming.strip():
            return []
        if not beliefs:
            return []

        model = self._get_model()

        # embed incoming
        incoming_emb = model.encode(
            [incoming], convert_to_numpy=True, normalize_embeddings=True
        )[0]

        # get belief embeddings (cached or fresh)
        belief_embs = await self._get_embeddings(beliefs, store)

        # compute similarities
        sims = _cosine_batch(incoming_emb, belief_embs)

        reinforced = []
        for idx, belief in enumerate(beliefs):
            if sims[idx] < RELEVANCE_THRESHOLD:
                continue

            # skip if on cooldown
            if self._is_on_cooldown(belief):
                continue

            # skip if already at ceiling
            if belief.confidence >= MAX_REINFORCED_CONFIDENCE:
                continue

            # CRITICAL: skip if incoming has conflicting numeric value
            # e.g., don't reinforce "40 degrees" when incoming says "70 degrees"
            if _has_numeric_conflict(incoming, belief.content):
                continue

            # boost confidence, respecting ceiling
            new_conf = min(MAX_REINFORCED_CONFIDENCE, belief.confidence + CONFIDENCE_BOOST)
            belief.confidence = new_conf
            belief.increment_use()
            belief.reinforce()

            reinforced.append(belief)

        # persist all at once
        if reinforced:
            await store.bulk_update(reinforced)

        return reinforced


__all__ = ["ReinforcementAgent"]
