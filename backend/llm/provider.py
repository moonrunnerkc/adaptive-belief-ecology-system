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

    SYSTEM_PROMPT_TEMPLATE = """You are ABES (Adaptive Belief Ecology System), an AI assistant with living memory.

YOUR IDENTITY:
- Your name is ABES (pronounced "ay-bees")
- You were created by Bradley R. Kinnard as a research project in belief ecology
- You have persistent memory that evolves over time through conversations
- When asked "who are you?" or "what is your name?", respond that you are ABES

You have a living memory of facts learned about the USER from your conversations. These are things the USER told you about themselves, their preferences, their life, etc.

CRITICAL RULES:
1. These are facts ABOUT THE USER, not about you
2. When asked "what do you know about me?" you MUST list ALL facts below, not just some
3. Never claim the conversation "just started" if there are facts stored below
4. Never deny having information that is listed below
5. If facts exist below, you have memory - don't claim otherwise
6. When asked about yourself, explain you are ABES with evolving belief-based memory

What you know about the user:
{belief_context}

HANDLING CONFLICTING INFORMATION:
- Each fact has a confidence percentage based on how often it was mentioned
- If two facts conflict (e.g., "it's warm" vs "it's cold"), prefer the one with HIGHER confidence
- Higher confidence = mentioned more times = more likely to be current/accurate
- When reporting conflicting facts, say something like: "You've mentioned X several times (90% confident), though you also said Y once (75% confident). Based on the stronger evidence, I believe X."
- Items marked with ⚠️ may conflict with other information

Guidelines:
1. Use ALL these facts to give personalized, contextual responses
2. Refer to the user's information correctly (e.g., "You mentioned you have a dog..." not "My dog...")
3. When asked about the user, provide a COMPLETE summary of everything listed above
4. For conflicting facts, favor the higher-confidence one and explain why
5. When the user shares new information, acknowledge it naturally"""

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
            return "No information learned about the user yet."

        # Sort by relevance score if available, else by confidence
        sorted_beliefs = sorted(
            beliefs,
            key=lambda b: (getattr(b, "score", 0.0), b.confidence),
            reverse=True,
        )[:max_beliefs]

        lines = []
        for i, b in enumerate(sorted_beliefs, 1):
            conf_pct = int(b.confidence * 100)
            tension_indicator = " ⚠️ may conflict with other info" if b.tension > 0.3 else ""
            # Transform first-person to second-person for clarity
            content = self._transform_to_user_perspective(b.content)
            lines.append(f"- {content} ({conf_pct}% confident{tension_indicator})")

        return "\n".join(lines)

    def _transform_to_user_perspective(self, content: str) -> str:
        """Transform first-person statements to user perspective.

        'My name is Brad' -> 'User's name is Brad'
        'I have two dogs' -> 'User has two dogs'
        'I love coffee' -> 'User loves coffee'
        """
        import re

        # First-person to user perspective transformations
        transformations = [
            (r"^My\s+", "User's "),
            (r"^I am\s+", "User is "),
            (r"^I'm\s+", "User is "),
            (r"^I have\s+", "User has "),
            (r"^I've\s+", "User has "),
            (r"^I was\s+", "User was "),
            (r"^I love\s+", "User loves "),
            (r"^I like\s+", "User likes "),
            (r"^I prefer\s+", "User prefers "),
            (r"^I think\s+", "User thinks "),
            (r"^I believe\s+", "User believes "),
            (r"^I want\s+", "User wants "),
            (r"^I need\s+", "User needs "),
            (r"^I enjoy\s+", "User enjoys "),
            (r"^I hate\s+", "User hates "),
            (r"^I dislike\s+", "User dislikes "),
            (r"^I work\s+", "User works "),
            (r"^I live\s+", "User lives "),
            # Generic "I verb" pattern - convert to "User verbs"
            (r"^I\s+(\w+)\s+", self._verb_transform),
        ]

        result = content
        for pattern, replacement in transformations:
            if callable(replacement):
                match = re.match(pattern, result, re.IGNORECASE)
                if match:
                    result = replacement(match, result)
                    break
            else:
                new_result = re.sub(pattern, replacement, result, count=1, flags=re.IGNORECASE)
                if new_result != result:
                    result = new_result
                    break

        return result

    def _verb_transform(self, match, full_text: str) -> str:
        """Transform 'I verb' to 'User verbs'."""
        import re
        verb = match.group(1).lower()

        # Add 's' for third person (simple heuristic)
        if verb.endswith(('s', 'x', 'z', 'ch', 'sh')):
            verb_3p = verb + 'es'
        elif verb.endswith('y') and len(verb) > 1 and verb[-2] not in 'aeiou':
            verb_3p = verb[:-1] + 'ies'
        else:
            verb_3p = verb + 's'

        rest = full_text[match.end():]
        return f"User {verb_3p} {rest}"

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
