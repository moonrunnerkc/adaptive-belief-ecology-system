# Author: Bradley R. Kinnard
"""
ContradictionAuditorAgent - detects high-tension belief pairs.
"""

import hashlib
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Set, Tuple
from uuid import UUID

import numpy as np

from ..core.config import settings
from ..core.models.belief import Belief

logger = logging.getLogger(__name__)


# tags that lower the threshold - more aggressive detection for high-stakes beliefs
# tradeoff: hardcoded values, but easy to tune if we find better thresholds empirically
SENSITIVE_TAGS: Dict[str, float] = {
    "critical": 0.4,
    "safety": 0.5,
    "core_value": 0.5,
}

# LRU cap - 2000 is plenty for typical workloads
EMBEDDING_CACHE_MAX = 2000

# keeps the O(n^2) pairwise check from killing us
MAX_BELIEFS_PER_AUDIT = 500

# cap tension so downstream doesn't see crazy values
MAX_TENSION_VALUE = 10.0

# safety limit on pairwise comparisons - derived from MAX_BELIEFS_PER_AUDIT
# at 500 beliefs we get 500*499/2 = 124,750 pairs, so 5M is generous headroom
MAX_PAIRWISE_COMPARISONS = 5_000_000


def _stable_hash(content: str) -> str:
    """SHA256 hash for cache invalidation. Deterministic across runs unlike hash()."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _contains_word(text: str, word: str) -> bool:
    # simple word boundary check - escape in case word has regex chars
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _is_likely_negation(text1: str, text2: str) -> bool:
    """Cheap negation heuristic. Catches "X" vs "not X" style contradictions.

    Tradeoff: simple and fast, but will miss subtle stuff like "I prefer coffee"
    vs "tea is my favorite". Good enough for a first pass - we can layer LLM
    scoring later if needed.

    Caller is responsible for gating on semantic similarity before invoking.
    """
    t1 = text1.lower()
    t2 = text2.lower()

    negations = ["not", "no", "never", "don't", "doesn't", "didn't",
                 "isn't", "aren't", "wasn't", "weren't", "won't"]

    neg1 = sum(1 for neg in negations if f" {neg} " in f" {t1} ")
    neg2 = sum(1 for neg in negations if f" {neg} " in f" {t2} ")

    # one has negation, other doesn't = likely contradiction
    if (neg1 > 0) != (neg2 > 0):
        return True

    # TODO: consider pulling opposites from a config file if this list grows
    opposites = [("true", "false"), ("yes", "no"), ("always", "never"),
                 ("good", "bad"), ("like", "dislike"), ("love", "hate")]

    for w1, w2 in opposites:
        if (_contains_word(t1, w1) and _contains_word(t2, w2)) or \
           (_contains_word(t1, w2) and _contains_word(t2, w1)):
            return True

    return False


@dataclass
class ContradictionDetectedEvent:
    """Fired when belief tension crosses threshold."""

    belief_id: UUID
    tension: float
    threshold: float
    timestamp: datetime


class ContradictionAuditorAgent:
    """Scans beliefs for high tension and emits events."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        # debounce: track which beliefs were already flagged
        self._above_threshold: Set[UUID] = set()
        # LRU cache: (content_hash, embedding) - hash is SHA256 for stability
        self._embedding_cache: OrderedDict[UUID, Tuple[str, list]] = OrderedDict()

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError("pip install sentence-transformers") from e
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _get_threshold_for_belief(self, belief: Belief) -> float:
        """Threshold can be lowered by sensitive tags. Takes min of all matches."""
        base = settings.tension_threshold_high
        thresholds = [base]

        for tag in belief.tags:
            tag_lower = tag.lower()
            for sensitive_key, sensitive_threshold in SENSITIVE_TAGS.items():
                # match "safety" or "safety.high" etc
                if tag_lower == sensitive_key or tag_lower.startswith(sensitive_key + "."):
                    thresholds.append(sensitive_threshold)
                    break

        return min(thresholds)

    def _cache_embeddings(self, beliefs: List[Belief]) -> None:
        """Encode beliefs we haven't seen (or whose content changed)."""
        model = self._get_model()
        to_encode = []
        to_encode_ids = []

        for b in beliefs:
            content_hash = _stable_hash(b.content)

            if b.id in self._embedding_cache:
                cached_hash, _ = self._embedding_cache[b.id]
                if cached_hash == content_hash:
                    self._embedding_cache.move_to_end(b.id)  # bump LRU
                    continue
                # stale - content changed

            to_encode.append(b.content)
            to_encode_ids.append((b.id, content_hash))

        if to_encode:
            embs = model.encode(to_encode, convert_to_numpy=True)
            for i, (bid, chash) in enumerate(to_encode_ids):
                self._embedding_cache[bid] = (chash, embs[i].tolist())

            # evict oldest
            while len(self._embedding_cache) > EMBEDDING_CACHE_MAX:
                self._embedding_cache.popitem(last=False)

    def _compute_tensions_from_cache(
        self, beliefs: List[Belief], similarity_threshold: float = 0.7
    ) -> dict[UUID, float]:
        """Pairwise tension via cached embeddings."""
        if not beliefs:
            return {}

        # sort by UUID hex for deterministic pairwise ordering
        beliefs = sorted(beliefs, key=lambda b: str(b.id))

        # re-encode any beliefs that got evicted from cache
        missing = [b for b in beliefs if b.id not in self._embedding_cache]
        if missing:
            self._cache_embeddings(missing)

        embeddings = []
        for b in beliefs:
            _, emb = self._embedding_cache[b.id]
            embeddings.append(np.array(emb))

        tension_scores: dict[UUID, float] = {b.id: 0.0 for b in beliefs}

        n = len(beliefs)
        total_pairs = n * (n - 1) // 2
        if total_pairs > MAX_PAIRWISE_COMPARISONS:
            logger.warning(
                f"pairwise comparisons ({total_pairs}) exceeds limit, truncating"
            )

        comparison_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                comparison_count += 1
                if comparison_count > MAX_PAIRWISE_COMPARISONS:
                    break

                b1, b2 = beliefs[i], beliefs[j]

                denom = np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                if denom == 0:
                    continue
                similarity = float(np.dot(embeddings[i], embeddings[j]) / denom)

                if similarity > similarity_threshold:
                    if _is_likely_negation(b1.content, b2.content):
                        tension_scores[b1.id] += similarity
                        tension_scores[b2.id] += similarity

            if comparison_count > MAX_PAIRWISE_COMPARISONS:
                break

        # clamp
        for bid in tension_scores:
            if tension_scores[bid] > MAX_TENSION_VALUE:
                tension_scores[bid] = MAX_TENSION_VALUE

        return tension_scores

    async def _load_persisted_state(self, store) -> None:
        """Try loading debounce state from store."""
        if not hasattr(store, "load_contradiction_state"):
            return
        try:
            loaded = await store.load_contradiction_state()
            if loaded:
                self._above_threshold = set(loaded)
        except Exception:
            pass  # memory fallback is fine

    async def _save_persisted_state(self, store) -> None:
        if not hasattr(store, "save_contradiction_state"):
            return
        try:
            await store.save_contradiction_state(list(self._above_threshold))
        except Exception:
            pass  # not the end of the world

    def _prune_stale_debounce_state(self, belief_ids: Set[UUID]) -> None:
        self._above_threshold &= belief_ids

    async def audit(
        self, beliefs: List[Belief], store=None
    ) -> List[ContradictionDetectedEvent]:
        """
        Compute tensions and emit events for beliefs that newly crossed threshold.
        Optionally persists state via store if it has save/load_contradiction_state.
        """
        if not beliefs:
            return []

        if len(beliefs) > MAX_BELIEFS_PER_AUDIT:
            logger.warning(
                f"audit got {len(beliefs)} beliefs, capping at {MAX_BELIEFS_PER_AUDIT}"
            )
            beliefs = beliefs[:MAX_BELIEFS_PER_AUDIT]

        if store is not None:
            await self._load_persisted_state(store)

        # clean up stale debounce entries
        current_ids = {b.id for b in beliefs}
        self._prune_stale_debounce_state(current_ids)

        self._cache_embeddings(beliefs)
        tension_scores = self._compute_tensions_from_cache(beliefs)

        for b in beliefs:
            if b.id not in tension_scores:
                logger.warning(f"tension missing for belief {b.id}")

        now = datetime.now(timezone.utc)
        events = []
        currently_above: Set[UUID] = set()

        for belief in beliefs:
            t = tension_scores.get(belief.id, 0.0)
            threshold = self._get_threshold_for_belief(belief)

            if t > threshold:
                currently_above.add(belief.id)
                # debounce: only fire if newly above
                if belief.id not in self._above_threshold:
                    events.append(
                        ContradictionDetectedEvent(
                            belief_id=belief.id,
                            tension=t,
                            threshold=threshold,
                            timestamp=now,
                        )
                    )

        self._above_threshold = currently_above

        if store is not None:
            await self._save_persisted_state(store)

        return events


__all__ = ["ContradictionAuditorAgent", "ContradictionDetectedEvent"]
