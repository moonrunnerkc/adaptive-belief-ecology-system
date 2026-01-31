# Author: Bradley R. Kinnard
"""
Hybrid LLM Provider - Routes queries between local (Ollama) and online (OpenAI).

Uses local LLM for belief-grounded responses and online LLM only when:
- Query requires real-time/live information
- Query needs current events, weather, traffic, news
- Query explicitly asks to "look up" or "search for" something
"""

import logging
import re
from typing import AsyncIterator, List, Optional

from ..core.config import settings
from ..core.models.belief import Belief
from .openai_provider import OpenAIProvider
from .provider import ChatMessage, ChatResponse, OllamaProvider, StreamChunk

logger = logging.getLogger(__name__)


# Patterns that indicate a query needs real-time/live information
_LIVE_INFO_PATTERNS = [
    # Time-sensitive queries
    r"\b(right now|currently|today|tonight|this morning|this evening)\b.*\b(weather|traffic|news|stock|price)\b",
    r"\b(weather|traffic|news|stock|price).*\b(right now|currently|today|tonight)\b",
    r"\bwhat('s| is) the (weather|traffic|news|time)\b",
    r"\bhow('s| is) the (weather|traffic)\b",
    r"\b(current|live|real-time|latest)\s+(weather|traffic|news|stock|price|score)\b",

    # Location + current state
    r"\b(in|at|near)\s+\w+.*\b(right now|currently|today)\b",

    # Explicit search/lookup requests
    r"\b(look up|search for|find out|check|google)\b",
    r"\bcan you (find|search|look|check)\b",

    # News and events
    r"\b(latest|recent|breaking|current)\s+(news|events|updates|headlines)\b",
    r"\bwhat('s| is) happening\b",
    r"\bwhat happened (today|yesterday|this week)\b",

    # Sports scores
    r"\b(score|result)s?\s+(of|for|in)\b.*\b(game|match)\b",
    r"\bwho (won|is winning|scored)\b",

    # Stock/crypto prices
    r"\b(stock|share|crypto|bitcoin|eth)\s+price\b",
    r"\bhow much is\s+\w+\s+(worth|trading)\b",

    # Time queries for other locations
    r"\bwhat time is it in\b",
]

# Compile patterns for efficiency
_LIVE_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in _LIVE_INFO_PATTERNS]


def _needs_live_info(message: str) -> bool:
    """Check if a message needs real-time/live information."""
    return any(pat.search(message) for pat in _LIVE_PATTERNS_COMPILED)


class HybridProvider:
    """
    Routes between Ollama (local) and OpenAI (online) based on query type.

    - Local Ollama: For belief-grounded responses, general conversation
    - Online OpenAI: For real-time info (weather, traffic, news, etc.)
    """

    # Enhanced system prompt for live queries
    LIVE_QUERY_SYSTEM_PROMPT = """You are ABES with access to current information.

The user is asking about real-time or current information. Answer based on your knowledge,
but be clear about what you know vs what might need verification.

What you know about the user:
{belief_context}

IMPORTANT:
- For time-sensitive queries, give your best answer but note the date/time limitations
- Be helpful but honest about information freshness
- If you don't have current data, say so clearly"""

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.1:8b-instruct-q4_0",
        openai_api_key: str = "",
        openai_model: str = "gpt-4o-mini",
        timeout: float = 120.0,
    ):
        self._ollama = OllamaProvider(
            base_url=ollama_base_url,
            model=ollama_model,
            timeout=timeout,
        )
        self._openai = OpenAIProvider(
            api_key=openai_api_key or settings.openai_api_key,
            model=openai_model or settings.openai_model,
            timeout=timeout,
        )
        self._openai_available = bool(openai_api_key or settings.openai_api_key)

    def _get_last_user_message(self, messages: List[ChatMessage]) -> str:
        """Extract the last user message for routing."""
        for msg in reversed(messages):
            if msg.role == "user":
                return msg.content
        return ""

    async def chat(
        self,
        messages: List[ChatMessage],
        beliefs: Optional[List[Belief]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """Route chat to appropriate provider based on query type."""
        last_message = self._get_last_user_message(messages)

        # Check if query needs live info
        if self._openai_available and _needs_live_info(last_message):
            logger.info(f"Routing to OpenAI for live info query: {last_message[:50]}...")
            return await self._openai.chat(messages, beliefs, temperature, max_tokens)

        # Default to local Ollama
        logger.debug(f"Routing to Ollama for standard query")
        return await self._ollama.chat(messages, beliefs, temperature, max_tokens)

    async def chat_stream(
        self,
        messages: List[ChatMessage],
        beliefs: Optional[List[Belief]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[StreamChunk]:
        """Route streaming chat to appropriate provider."""
        last_message = self._get_last_user_message(messages)

        if self._openai_available and _needs_live_info(last_message):
            logger.info(f"Routing stream to OpenAI for live info query")
            async for chunk in self._openai.chat_stream(messages, beliefs, temperature, max_tokens):
                yield chunk
        else:
            async for chunk in self._ollama.chat_stream(messages, beliefs, temperature, max_tokens):
                yield chunk

    async def health_check(self) -> bool:
        """Check if primary (Ollama) provider is healthy."""
        return await self._ollama.health_check()

    async def close(self) -> None:
        """Close both providers."""
        await self._ollama.close()
        await self._openai.close()


__all__ = ["HybridProvider", "_needs_live_info"]
