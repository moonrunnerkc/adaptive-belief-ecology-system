# Author: Bradley R. Kinnard
"""
Chat Service - orchestrates the full ABES chat pipeline.
Combines agent processing, belief management, and LLM response generation.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator, Callable, Optional
from uuid import UUID, uuid4

from ..agents import (
    PerceptionAgent,
    BeliefCreatorAgent,
    ReinforcementAgent,
    ContradictionAuditorAgent,
    RelevanceCuratorAgent,
    DecayControllerAgent,
    MutationEngineerAgent,
)
from ..core.config import settings
from ..core.models.belief import Belief, BeliefStatus, OriginMetadata
from ..llm import ChatMessage, get_llm_provider
from ..storage.base import BeliefStoreABC

logger = logging.getLogger(__name__)


@dataclass
class BeliefEvent:
    """Event emitted when beliefs change."""

    event_type: str  # "created", "reinforced", "mutated", "deprecated", "tension_changed"
    belief_id: UUID
    content: str
    confidence: float
    tension: float
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ChatTurn:
    """A single turn in a conversation."""

    id: UUID = field(default_factory=uuid4)
    user_message: str = ""
    assistant_message: str = ""
    beliefs_created: list[UUID] = field(default_factory=list)
    beliefs_reinforced: list[UUID] = field(default_factory=list)
    beliefs_mutated: list[UUID] = field(default_factory=list)
    beliefs_deprecated: list[UUID] = field(default_factory=list)
    beliefs_used: list[UUID] = field(default_factory=list)
    events: list[BeliefEvent] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0


@dataclass
class ChatSession:
    """A chat session with conversation history."""

    id: UUID = field(default_factory=uuid4)
    turns: list[ChatTurn] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_messages(self, max_turns: int = 10) -> list[ChatMessage]:
        """Convert recent turns to ChatMessage list for LLM."""
        messages = []
        for turn in self.turns[-max_turns:]:
            if turn.user_message:
                messages.append(ChatMessage(role="user", content=turn.user_message))
            if turn.assistant_message:
                messages.append(ChatMessage(role="assistant", content=turn.assistant_message))
        return messages


class ChatService:
    """
    Full ABES chat service.
    Processes user messages through the belief ecology and generates responses.
    """

    def __init__(
        self,
        belief_store: BeliefStoreABC,
        event_callback: Optional[Callable[[BeliefEvent], None]] = None,
    ):
        self.belief_store = belief_store
        self.event_callback = event_callback

        # Initialize agents (none need store in constructor)
        self._perception = PerceptionAgent()
        self._creator = BeliefCreatorAgent()
        self._reinforcement = ReinforcementAgent()
        self._auditor = ContradictionAuditorAgent()
        self._relevance = RelevanceCuratorAgent()
        self._decay = DecayControllerAgent()
        self._mutation = MutationEngineerAgent()

        # Session storage
        self._sessions: dict[UUID, ChatSession] = {}

    def _emit_event(self, event: BeliefEvent) -> None:
        """Emit a belief event to the callback if registered."""
        if self.event_callback:
            try:
                self.event_callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    def get_or_create_session(self, session_id: Optional[UUID] = None) -> ChatSession:
        """Get existing session or create new one."""
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        session = ChatSession(id=session_id or uuid4())
        self._sessions[session.id] = session
        return session

    async def _validate_and_correct_response(
        self,
        response: str,
        beliefs: list,
        llm,
        messages: list,
        max_retries: int = 1,
    ) -> str:
        """
        Validate LLM response against beliefs and correct if needed.

        Returns original response if valid, or corrected response if contradictions found.
        """
        from .response_validator import validate_response, get_correction_prompt

        validation = validate_response(response, beliefs)

        if validation.is_valid:
            return response

        logger.warning(
            f"Response validation failed: {len(validation.contradictions)} contradictions found"
        )

        # try to correct
        for attempt in range(max_retries):
            correction_prompt = get_correction_prompt(
                response, validation.contradictions, beliefs
            )

            # append correction request
            corrected_messages = messages + [
                ChatMessage(role="assistant", content=response),
                ChatMessage(role="user", content=correction_prompt),
            ]

            corrected = await llm.chat(
                messages=corrected_messages,
                beliefs=beliefs,
                temperature=0.3,  # lower temp for correction
                max_tokens=settings.llm_max_tokens,
            )

            # validate corrected response
            revalidation = validate_response(corrected.content, beliefs)

            if revalidation.is_valid:
                logger.info("Response corrected successfully")
                return corrected.content

            logger.warning(f"Correction attempt {attempt + 1} still has contradictions")
            response = corrected.content

        # give up - return last attempt with warning prefix
        return f"[Note: Response may contain inaccuracies]\n\n{response}"

    async def process_message(
        self,
        message: str,
        session_id: Optional[UUID] = None,
        context: Optional[str] = None,
        user_id: Optional[UUID] = None,
    ) -> ChatTurn:
        """
        Process a user message through the full ABES pipeline.

        Steps:
        1. Extract candidate beliefs from message (Perception)
        2. Create or deduplicate beliefs (Creator)
        3. Reinforce existing similar beliefs (Reinforcement)
        4. Apply decay to all beliefs (Decay)
        5. Compute tensions between beliefs (Auditor)
        6. Rank beliefs by relevance to context (Relevance)
        7. Generate LLM response with belief context
        8. Return turn with all events
        """
        start = datetime.now(timezone.utc)
        session = self.get_or_create_session(session_id)
        turn = ChatTurn(user_message=message)

        # Step 1: Perception - extract claims from message
        candidates = await self._perception.ingest(message, {"source_type": "chat"})
        logger.info(f"Extracted {len(candidates)} candidate beliefs from message")

        # Step 2: Create beliefs (with deduplication)
        if candidates:
            origin = OriginMetadata(source="chat")
            created_beliefs = await self._creator.create_beliefs(
                candidates=candidates,
                origin=origin,
                store=self.belief_store,
                user_id=user_id,  # Associate with user
                session_id=str(session.id),  # Track which session created this
            )

            for belief in created_beliefs:
                turn.beliefs_created.append(belief.id)
                turn.events.append(BeliefEvent(
                    event_type="created",
                    belief_id=belief.id,
                    content=belief.content,
                    confidence=belief.confidence,
                    tension=belief.tension,
                    details={"source": "user_message"},
                ))
                self._emit_event(turn.events[-1])

        # Step 3: Reinforce existing beliefs (user-scoped only)
        all_beliefs = await self.belief_store.list(
            status=BeliefStatus.Active, limit=1000, user_id=user_id
        )
        reinforced_beliefs = await self._reinforcement.reinforce(
            incoming=message,
            beliefs=all_beliefs,
            store=self.belief_store,
        )

        for belief in reinforced_beliefs:
            turn.beliefs_reinforced.append(belief.id)
            turn.events.append(BeliefEvent(
                event_type="reinforced",
                belief_id=belief.id,
                content=belief.content,
                confidence=belief.confidence,
                tension=belief.tension,
            ))
            self._emit_event(turn.events[-1])

        # Step 4: Apply decay (user-scoped)
        all_beliefs = await self.belief_store.list(
            status=BeliefStatus.Active, limit=1000, user_id=user_id
        )
        decay_events, modified_beliefs = await self._decay.process_beliefs(all_beliefs)

        for belief in modified_beliefs:
            await self.belief_store.update(belief)
            if belief.status == BeliefStatus.Deprecated:
                turn.beliefs_deprecated.append(belief.id)
                turn.events.append(BeliefEvent(
                    event_type="deprecated",
                    belief_id=belief.id,
                    content=belief.content,
                    confidence=belief.confidence,
                    tension=belief.tension,
                    details={"reason": "decay"},
                ))
                self._emit_event(turn.events[-1])

        # Step 5: Compute tensions (user-scoped)
        all_beliefs = await self.belief_store.list(
            status=BeliefStatus.Active, limit=1000, user_id=user_id
        )
        contradiction_events = await self._auditor.audit(all_beliefs, store=self.belief_store)

        # Build tension map from events
        tension_map: dict[UUID, float] = {}
        for event in contradiction_events:
            # ContradictionDetectedEvent has belief_id and tension
            tension_map[event.belief_id] = max(
                tension_map.get(event.belief_id, 0.0),
                event.tension,
            )
            # Emit tension event for UI
            belief = next((b for b in all_beliefs if b.id == event.belief_id), None)
            if belief:
                turn.events.append(BeliefEvent(
                    event_type="tension_changed",
                    belief_id=event.belief_id,
                    content=belief.content,
                    confidence=belief.confidence,
                    tension=event.tension,
                    details={"threshold": event.threshold},
                ))
                self._emit_event(turn.events[-1])

        # Update beliefs with new tensions
        for belief in all_beliefs:
            if belief.id in tension_map:
                old_tension = belief.tension
                new_tension = tension_map[belief.id]
                if abs(new_tension - old_tension) > 0.1:
                    belief.tension = new_tension
                    await self.belief_store.update(belief)

        # Step 6: Mutation - only evolve when there's genuine ambiguity
        # If one belief is clearly more confident, don't mutate - let it "win"
        all_beliefs = await self.belief_store.list(
            status=BeliefStatus.Active, limit=1000, user_id=user_id
        )

        # Group beliefs by high tension (potential contradictions)
        high_tension_beliefs = [b for b in all_beliefs if b.tension >= 0.5]

        for belief in high_tension_beliefs:
            # Find the contradicting belief
            contradicting = None
            for other in all_beliefs:
                if other.id != belief.id and other.id in tension_map and other.tension >= 0.5:
                    contradicting = other
                    break

            if not contradicting:
                continue

            # Only mutate if confidences are similar (within 10%)
            # If one is more confident from more evidence, it "wins"
            confidence_diff = abs(belief.confidence - contradicting.confidence)

            if confidence_diff > 0.10:
                # Clear winner - deprecate the loser, don't mutate
                loser = belief if belief.confidence < contradicting.confidence else contradicting
                winner = contradicting if belief.confidence < contradicting.confidence else belief
                if loser.id == belief.id:  # Only process each pair once
                    # Deprecate loser instead of just reducing confidence
                    loser.status = BeliefStatus.Deprecated
                    loser.confidence *= 0.5
                    await self.belief_store.update(loser)

                    # Boost winner slightly
                    winner.confidence = min(0.95, winner.confidence + 0.05)
                    winner.tension = 0.0  # Clear tension since contradiction resolved
                    await self.belief_store.update(winner)

                    turn.beliefs_deprecated.append(loser.id)
                    turn.events.append(BeliefEvent(
                        event_type="deprecated",
                        belief_id=loser.id,
                        content=loser.content,
                        confidence=loser.confidence,
                        tension=loser.tension,
                        details={"reason": "contradiction_resolved", "winner_id": str(winner.id)},
                    ))
                    self._emit_event(turn.events[-1])
                    logger.info(f"Contradiction resolved: {winner.content[:30]}... wins over {loser.content[:30]}...")
                continue

            # Similar confidence - need to hedge/mutate
            if belief.confidence < 0.9:  # Don't mutate very high confidence beliefs
                proposal = self._mutation.propose_mutation(
                    belief=belief,
                    contradicting=contradicting,
                    all_beliefs=all_beliefs,
                )

                if proposal:
                    # The proposal already contains the mutated belief
                    mutated = proposal.mutated_belief
                    await self.belief_store.create(mutated)

                    # Mark original as mutated
                    belief.status = BeliefStatus.Mutated
                    await self.belief_store.update(belief)

                    turn.beliefs_mutated.append(mutated.id)
                    turn.events.append(BeliefEvent(
                        event_type="mutated",
                        belief_id=mutated.id,
                        content=mutated.content,
                        confidence=mutated.confidence,
                        tension=0.0,
                        details={
                            "original_id": str(belief.id),
                            "original_content": belief.content,
                            "strategy": proposal.strategy,
                        },
                    ))
                    self._emit_event(turn.events[-1])
                    logger.info(f"Mutated belief: {belief.content[:30]}... -> {mutated.content[:30]}...")

        # Step 7: Get beliefs for LLM context (hierarchical: session first, then user)
        # IMPORTANT: user_id is the ceiling - never cross-user

        # First, get session-specific beliefs (this conversation)
        session_beliefs = await self.belief_store.list(
            status=BeliefStatus.Active, limit=500, user_id=user_id,
            session_id=str(session.id) if session else None
        )

        # Then get all user beliefs (including other sessions)
        all_user_beliefs = await self.belief_store.list(
            status=BeliefStatus.Active, limit=1000, user_id=user_id
        )

        # Mark session beliefs for the LLM to distinguish
        for b in session_beliefs:
            b.tags = list(set(b.tags) | {"this_session"})

        # For generic questions about user memory, include ALL beliefs
        # This handles "what do you know about me?" type questions
        lower_msg = message.lower()
        is_memory_query = any(phrase in lower_msg for phrase in [
            "what do you know",
            "what you know",
            "tell me what you know",
            "tell me about me",
            "what have you learned",
            "summarize what you know",
            "everything you know",
            "do you remember",
            "do you know about",
        ])

        if is_memory_query and all_user_beliefs:
            # Include all beliefs for memory queries - don't filter by relevance
            top_beliefs = sorted(all_user_beliefs, key=lambda b: b.confidence, reverse=True)[:settings.llm_context_beliefs]
            logger.info(f"Memory query detected - using all {len(top_beliefs)} beliefs")
        else:
            # Normal relevance-based ranking
            context_str = context or message
            top_beliefs = await self._relevance.get_top_beliefs(
                beliefs=all_user_beliefs,
                context=context_str,
                top_k=settings.llm_context_beliefs,
                tension_map=tension_map,
            )

        # Step 7: Generate LLM response
        llm = get_llm_provider()

        # Build conversation history
        messages = session.to_messages(max_turns=5)
        messages.append(ChatMessage(role="user", content=message))

        # Get response with top beliefs as context
        response = await llm.chat(
            messages=messages,
            beliefs=top_beliefs,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

        # Step 8: Validate response against beliefs (catch hallucinations)
        validated_response = await self._validate_and_correct_response(
            response.content,
            top_beliefs,
            llm,
            messages,
        )

        turn.assistant_message = validated_response
        turn.beliefs_used = [b.id for b in top_beliefs]
        turn.duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        # Add turn to session
        session.turns.append(turn)

        logger.info(
            f"Chat turn completed: {len(turn.beliefs_created)} created, "
            f"{len(turn.beliefs_reinforced)} reinforced, "
            f"duration={turn.duration_ms:.0f}ms"
        )

        return turn

    async def process_message_stream(
        self,
        message: str,
        session_id: Optional[UUID] = None,
        context: Optional[str] = None,
    ) -> AsyncIterator[tuple[str, Optional[ChatTurn]]]:
        """
        Process message with streaming LLM response.
        Yields (content_chunk, None) for each chunk, then ("", ChatTurn) at the end.
        """
        start = datetime.now(timezone.utc)
        session = self.get_or_create_session(session_id)
        turn = ChatTurn(user_message=message)

        # Steps 1-5: Same belief processing as non-streaming
        candidates = await self._perception.ingest(message, {"source_type": "chat"})

        if candidates:
            origin = OriginMetadata(source="chat")
            created_beliefs = await self._creator.create_beliefs(
                candidates=candidates,
                origin=origin,
                store=self.belief_store,
            )
            for belief in created_beliefs:
                turn.beliefs_created.append(belief.id)
                turn.events.append(BeliefEvent(
                    event_type="created",
                    belief_id=belief.id,
                    content=belief.content,
                    confidence=belief.confidence,
                    tension=belief.tension,
                    details={"source": "user_message"},
                ))
                self._emit_event(turn.events[-1])

        # Reinforce
        all_beliefs = await self.belief_store.list(status=BeliefStatus.Active, limit=1000)
        reinforced = await self._reinforcement.reinforce(message, all_beliefs, self.belief_store)
        for belief in reinforced:
            turn.beliefs_reinforced.append(belief.id)
            turn.events.append(BeliefEvent(
                event_type="reinforced",
                belief_id=belief.id,
                content=belief.content,
                confidence=belief.confidence,
                tension=belief.tension,
            ))
            self._emit_event(turn.events[-1])

        # Rank beliefs
        all_beliefs = await self.belief_store.list(status=BeliefStatus.Active, limit=1000)
        top_beliefs = await self._relevance.get_top_beliefs(
            beliefs=all_beliefs,
            context=message,
            top_k=settings.llm_context_beliefs,
        )

        # Stream LLM response
        llm = get_llm_provider()
        messages = session.to_messages(max_turns=5)
        messages.append(ChatMessage(role="user", content=message))

        full_response = ""
        async for chunk in llm.chat_stream(
            messages=messages,
            beliefs=top_beliefs,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        ):
            full_response += chunk.content
            yield (chunk.content, None)

        turn.assistant_message = full_response
        turn.beliefs_used = [b.id for b in top_beliefs]
        turn.duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        session.turns.append(turn)
        yield ("", turn)

    def get_session(self, session_id: UUID) -> Optional[ChatSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[ChatSession]:
        """List all sessions."""
        return list(self._sessions.values())

    def clear_session(self, session_id: UUID) -> bool:
        """Clear a session's history."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False


# Singleton
_chat_service: Optional[ChatService] = None


def get_chat_service(belief_store: BeliefStoreABC) -> ChatService:
    """Get or create the chat service singleton."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService(belief_store)
    return _chat_service


__all__ = [
    "BeliefEvent",
    "ChatTurn",
    "ChatSession",
    "ChatService",
    "get_chat_service",
]
