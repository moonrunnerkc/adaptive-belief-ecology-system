# Author: Bradley R. Kinnard
"""
BeliefCreatorAgent - embeds candidates, dedupes via cosine similarity, persists novel beliefs.
"""

import re
from typing import List, Optional
from uuid import UUID

import numpy as np

from ..core.models.belief import Belief, BeliefStatus, OriginMetadata
from ..storage.base import BeliefStoreABC


# word boundary patterns for tagging - avoids partial matches like "cached" -> "cache"
# TODO: tokenization might improve accuracy vs regex word boundaries
_TAG_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bcache\b", re.I), "infra.cache"),
    (re.compile(r"\bweights?\b", re.I), "model.weights"),
    (re.compile(r"\btensor\b", re.I), "model.tensor"),
    (re.compile(r"\boom\b", re.I), "error.oom"),
    (re.compile(r"\bcrash\b", re.I), "error.crash"),
    (re.compile(r"\bcorrupt\b", re.I), "error.corrupt"),
    (re.compile(r"\btimeout\b", re.I), "perf.timeout"),
    (re.compile(r"\bconverge\b", re.I), "training.converge"),
]

DEDUPE_THRESHOLD = 0.95


def _cosine_batch(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Cosine sim between one query vec and batch of candidates. All must be normalized."""
    return candidates @ query


class BeliefCreatorAgent:
    """Dedupes candidates against store, creates Beliefs for novel ones."""

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

    def _assign_tags(self, text: str) -> List[str]:
        tags = []
        for pat, tag in _TAG_PATTERNS:
            if pat.search(text):
                tags.append(tag)
        return tags

    async def create_beliefs(
        self,
        candidates: List[str],
        origin: OriginMetadata,
        store: BeliefStoreABC,
        user_id: Optional[UUID] = None,
    ) -> List[Belief]:
        """Embed, dedupe, persist. Returns list of created beliefs."""
        # filter junk early
        candidates = [c for c in candidates if c and c.strip()]
        if not candidates:
            return []

        model = self._get_model()

        # normalized for cosine
        cand_embs = model.encode(candidates, convert_to_numpy=True, normalize_embeddings=True)

        # track embeddings we've seen - grows as we create beliefs
        seen_embs = None

        created: List[Belief] = []

        for idx, cand in enumerate(candidates):
            emb = cand_embs[idx]

            # fetch neighbors for this specific candidate
            neighbors = []
            try:
                neighbors = await store.search_by_embedding(emb.tolist(), top_k=5)
            except Exception:
                pass
            if not neighbors:
                try:
                    neighbors = await store.list(limit=50)
                except Exception:
                    pass

            # encode neighbors fresh each time (different per candidate)
            nb_embs = None
            if neighbors:
                texts = [n.content for n in neighbors]
                nb_embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

            # combine store neighbors with beliefs created earlier in this batch
            check_embs = nb_embs
            if seen_embs is not None:
                if check_embs is not None:
                    check_embs = np.vstack([check_embs, seen_embs])
                else:
                    check_embs = seen_embs

            # dedupe check
            is_dup = False
            if check_embs is not None and len(check_embs) > 0:
                sims = _cosine_batch(emb, check_embs)
                if np.max(sims) > DEDUPE_THRESHOLD:
                    is_dup = True

            if is_dup:
                continue

            tags = self._assign_tags(cand)
            belief = Belief(
                content=cand,
                confidence=0.8,
                origin=origin,
                tags=tags,
                status=BeliefStatus.Active,
                user_id=user_id,
            )

            created_belief = await store.create(belief)

            # save embedding if store supports it
            if hasattr(store, "save_embedding"):
                await store.save_embedding(created_belief.id, emb.tolist())

            created.append(created_belief)

            # add to seen so later candidates dedupe against this one too
            if seen_embs is None:
                seen_embs = emb.reshape(1, -1)
            else:
                seen_embs = np.vstack([seen_embs, emb])

        return created


__all__ = ["BeliefCreatorAgent"]
