# Author: Bradley R. Kinnard
"""
LLM Provider - Ollama integration for ABES applications.
Supports chat, embeddings, and streaming responses.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator, Optional
from uuid import UUID

import httpx

from ..core.config import settings
from ..core.models.belief import Belief

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """A message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    belief_ids: list[UUID] = field(default_factory=list)  # beliefs referenced in this message


@dataclass
class ChatResponse:
    """Response from LLM with metadata."""

    content: str
    model: str
    tokens_prompt: int
    tokens_completion: int
    duration_ms: float
    beliefs_used: list[UUID] = field(default_factory=list)


@dataclass
class StreamChunk:
    """A chunk of streaming response."""

    content: str
    done: bool
    model: str = ""
    tokens_prompt: int = 0
    tokens_completion: int = 0


class OllamaProvider:
    """
    Ollama LLM provider for ABES.
    Handles chat completion with belief context injection.
    """

    SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant powered by ABES (Adaptive Belief Ecology System).

Your responses are informed by a living memory of beliefs that evolve over time. These beliefs:
- Have confidence levels (how certain they are)
- Have tension levels (contradiction pressure from conflicting beliefs)
- Can mutate, reinforce, or deprecate based on new information

Current belief context:
{belief_context}

Guidelines:
1. Use the provided beliefs to inform your responses
2. If beliefs conflict, acknowledge the tension naturally
3. When you learn new information, it will be extracted as new beliefs
4. Be conversational and helpful
5. If asked about your beliefs or memory, explain what you know and your confidence level"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b-instruct-q4_0",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _format_belief_context(self, beliefs: list[Belief], max_beliefs: int = 15) -> str:
        """Format beliefs into context string for system prompt."""
        if not beliefs:
            return "No beliefs currently stored."

        # Sort by relevance score if available, else by confidence
        sorted_beliefs = sorted(
            beliefs,
            key=lambda b: (getattr(b, "score", 0.0), b.confidence),
            reverse=True,
        )[:max_beliefs]

        lines = []
        for i, b in enumerate(sorted_beliefs, 1):
            conf_pct = int(b.confidence * 100)
            tension_indicator = "⚡" if b.tension > 0.5 else ""
            lines.append(f"{i}. [{conf_pct}% confidence{tension_indicator}] {b.content}")

        return "\n".join(lines)

    def _build_system_prompt(self, beliefs: list[Belief]) -> str:
        """Build system prompt with belief context."""
        context = self._format_belief_context(beliefs)
        return self.SYSTEM_PROMPT_TEMPLATE.format(belief_context=context)

    async def chat(
        self,
        messages: list[ChatMessage],
        beliefs: list[Belief] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """
        Generate a chat response using Ollama.
        Injects belief context into system prompt.
        """
        beliefs = beliefs or []
        client = await self._get_client()

        # Build message list with system prompt
        ollama_messages = [
            {"role": "system", "content": self._build_system_prompt(beliefs)}
        ]

        for msg in messages:
            ollama_messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        start = datetime.now(timezone.utc)

        response = await client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": ollama_messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        )
        response.raise_for_status()

        data = response.json()
        duration = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        return ChatResponse(
            content=data.get("message", {}).get("content", ""),
            model=data.get("model", self.model),
            tokens_prompt=data.get("prompt_eval_count", 0),
            tokens_completion=data.get("eval_count", 0),
            duration_ms=duration,
            beliefs_used=[b.id for b in beliefs[:15]],
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        beliefs: list[Belief] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a chat response using Ollama.
        Yields chunks as they arrive.
        """
        beliefs = beliefs or []
        client = await self._get_client()

        ollama_messages = [
            {"role": "system", "content": self._build_system_prompt(beliefs)}
        ]

        for msg in messages:
            ollama_messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        async with client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": ollama_messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = data.get("message", {}).get("content", "")
                done = data.get("done", False)

                yield StreamChunk(
                    content=content,
                    done=done,
                    model=data.get("model", self.model),
                    tokens_prompt=data.get("prompt_eval_count", 0) if done else 0,
                    tokens_completion=data.get("eval_count", 0) if done else 0,
                )

    async def health_check(self) -> bool:
        """Check if Ollama is available."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False


# Singleton instance
_provider: Optional[OllamaProvider] = None


def get_llm_provider() -> OllamaProvider:
    """Get or create the LLM provider singleton."""
    global _provider
    if _provider is None:
        _provider = OllamaProvider(
            base_url=getattr(settings, "ollama_base_url", "http://localhost:11434"),
            model=getattr(settings, "ollama_model", "llama3.1:8b-instruct-q4_0"),
        )
    return _provider


__all__ = [
    "ChatMessage",
    "ChatResponse",
    "StreamChunk",
    "OllamaProvider",
    "get_llm_provider",
]
