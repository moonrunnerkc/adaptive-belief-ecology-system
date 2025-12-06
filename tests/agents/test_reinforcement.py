# Author: Bradley R. Kinnard
"""Tests for ReinforcementAgent."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4
from datetime import datetime, timezone, timedelta


class MockOriginMetadata:
    def __init__(self, source: str = "test"):
        self.source = source
        self.turn_index = None
        self.episode_id = None
        self.timestamp = datetime.now(timezone.utc)
        # default: old enough to not be on cooldown
        self.last_reinforced = datetime.now(timezone.utc) - timedelta(seconds=120)


class MockBelief:
    def __init__(self, content: str, confidence: float = 0.5):
        self.id = uuid4()
        self.content = content
        self.confidence = confidence
        self.origin = MockOriginMetadata()
        self.tags = []
        self.tension = 0.0
        self.cluster_id = None
        self.status = "active"
        self.parent_id = None
        self.use_count = 0
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def increment_use(self):
        self.use_count += 1
        self.updated_at = datetime.now(timezone.utc)

    def reinforce(self):
        self.origin.last_reinforced = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.bulk_update = AsyncMock(return_value=0)
    # no get_embedding by default
    del store.get_embedding
    return store


@pytest.fixture
def mock_model():
    model = MagicMock()
    return model


class TestReinforce:
    """Test reinforcement logic."""

    @pytest.mark.asyncio
    async def test_empty_incoming(self, mock_store):
        from backend.agents.reinforcement import ReinforcementAgent

        agent = ReinforcementAgent()
        beliefs = [MockBelief("some belief")]

        result = await agent.reinforce("", beliefs, mock_store)
        assert result == []
        mock_store.bulk_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_beliefs(self, mock_store):
        from backend.agents.reinforcement import ReinforcementAgent

        agent = ReinforcementAgent()

        result = await agent.reinforce("hello world", [], mock_store)
        assert result == []
        mock_store.bulk_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_reinforces_similar_belief(self, mock_store, mock_model):
        from backend.agents.reinforcement import ReinforcementAgent, CONFIDENCE_BOOST

        agent = ReinforcementAgent()
        agent._model = mock_model

        # make model return high similarity
        def encode_same(texts, **kwargs):
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

        mock_model.encode = MagicMock(side_effect=encode_same)

        belief = MockBelief("the cache is full", confidence=0.5)
        original_confidence = belief.confidence

        result = await agent.reinforce("cache problem", [belief], mock_store)

        assert len(result) == 1
        assert result[0].confidence == original_confidence + CONFIDENCE_BOOST
        assert result[0].use_count == 1
        mock_store.bulk_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_reinforce_low_similarity(self, mock_store, mock_model):
        from backend.agents.reinforcement import ReinforcementAgent

        agent = ReinforcementAgent()
        agent._model = mock_model

        call_count = [0]

        def encode_orthogonal(texts, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return np.array([[1.0, 0.0, 0.0]])
            else:
                return np.array([[0.0, 1.0, 0.0] for _ in texts])

        mock_model.encode = MagicMock(side_effect=encode_orthogonal)

        belief = MockBelief("unrelated content", confidence=0.5)
        original_confidence = belief.confidence

        result = await agent.reinforce("something else", [belief], mock_store)

        assert result == []
        assert belief.confidence == original_confidence
        mock_store.bulk_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_confidence_caps_at_ceiling(self, mock_store, mock_model):
        from backend.agents.reinforcement import ReinforcementAgent, MAX_REINFORCED_CONFIDENCE

        agent = ReinforcementAgent()
        agent._model = mock_model

        def encode_same(texts, **kwargs):
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

        mock_model.encode = MagicMock(side_effect=encode_same)

        # at ceiling - should not reinforce
        belief = MockBelief("high confidence belief", confidence=MAX_REINFORCED_CONFIDENCE)

        result = await agent.reinforce("related text", [belief], mock_store)

        assert result == []

    @pytest.mark.asyncio
    async def test_confidence_respects_ceiling(self, mock_store, mock_model):
        from backend.agents.reinforcement import ReinforcementAgent, MAX_REINFORCED_CONFIDENCE

        agent = ReinforcementAgent()
        agent._model = mock_model

        def encode_same(texts, **kwargs):
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

        mock_model.encode = MagicMock(side_effect=encode_same)

        # near ceiling - should cap
        belief = MockBelief("near ceiling", confidence=0.9)

        result = await agent.reinforce("related text", [belief], mock_store)

        assert len(result) == 1
        assert result[0].confidence == MAX_REINFORCED_CONFIDENCE

    @pytest.mark.asyncio
    async def test_cooldown_blocks_reinforcement(self, mock_store, mock_model):
        from backend.agents.reinforcement import ReinforcementAgent

        agent = ReinforcementAgent()
        agent._model = mock_model

        def encode_same(texts, **kwargs):
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

        mock_model.encode = MagicMock(side_effect=encode_same)

        belief = MockBelief("recent belief", confidence=0.5)
        # just reinforced - on cooldown
        belief.origin.last_reinforced = datetime.now(timezone.utc)

        result = await agent.reinforce("related", [belief], mock_store)

        assert result == []
        mock_store.bulk_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_reinforces_multiple_beliefs(self, mock_store, mock_model):
        from backend.agents.reinforcement import ReinforcementAgent

        agent = ReinforcementAgent()
        agent._model = mock_model

        def encode_same(texts, **kwargs):
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

        mock_model.encode = MagicMock(side_effect=encode_same)

        beliefs = [
            MockBelief("belief one", confidence=0.5),
            MockBelief("belief two", confidence=0.6),
        ]

        result = await agent.reinforce("related", beliefs, mock_store)

        assert len(result) == 2
        mock_store.bulk_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_stored_embeddings(self, mock_model):
        from backend.agents.reinforcement import ReinforcementAgent

        agent = ReinforcementAgent()
        agent._model = mock_model

        # store with get_embedding
        store = MagicMock()
        store.bulk_update = AsyncMock(return_value=1)
        stored_emb = [1.0, 0.0, 0.0]
        store.get_embedding = AsyncMock(return_value=stored_emb)

        # model should only encode incoming, not beliefs
        def encode_incoming(texts, **kwargs):
            return np.array([[1.0, 0.0, 0.0] for _ in texts])

        mock_model.encode = MagicMock(side_effect=encode_incoming)

        belief = MockBelief("stored belief", confidence=0.5)

        result = await agent.reinforce("related", [belief], store)

        assert len(result) == 1
        store.get_embedding.assert_called_once_with(belief.id)


class TestLazyLoading:
    def test_model_not_loaded_on_init(self):
        from backend.agents.reinforcement import ReinforcementAgent

        agent = ReinforcementAgent()
        assert agent._model is None


__all__ = []
