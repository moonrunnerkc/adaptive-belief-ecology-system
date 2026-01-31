"""
Pydantic models for belief storage in the Belief Ecology engine.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# Python 3.10 compat - StrEnum added in 3.11
class StrEnum(str, Enum):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BeliefBaseModel(BaseModel):
    model_config = {
        "from_attributes": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        },
    }


class OriginMetadata(BeliefBaseModel):
    """Tracks where a belief came from and when it was last reinforced."""

    source: str
    turn_index: Optional[int] = None
    episode_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=utcnow)
    last_reinforced: datetime = Field(default_factory=utcnow)


class BeliefStatus(StrEnum):
    Active = "active"
    Decaying = "decaying"
    Mutated = "mutated"
    Deprecated = "deprecated"


class Belief(BeliefBaseModel):
    """Belief record used by the ecology layer. Tracks confidence, tension, and lineage."""

    id: UUID = Field(default_factory=uuid4)
    content: str
    confidence: float  # 0.0 to 1.0
    origin: OriginMetadata
    tags: List[str] = Field(default_factory=list)
    tension: float = 0.0  # contradiction pressure
    cluster_id: Optional[UUID] = None
    status: BeliefStatus = BeliefStatus.Active
    parent_id: Optional[UUID] = None  # tracks mutations
    use_count: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    session_id: Optional[str] = None  # for chat session grouping
    user_id: Optional[UUID] = None  # owner of this belief

    # computed fields for ranking (set by BEL loop, not persisted)
    relevance: Optional[float] = None  # similarity to current context
    score: Optional[float] = None  # composite ranking score

    # NOTE: updated_at only refreshed via helper methods (increment_use, reinforce, apply_decay)
    # or dict reconstruction. Direct field assignment (belief.confidence = X) will NOT update it.
    # This is intentional - use helpers for tracked changes, or manage timestamps at repo layer.

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("tension")
    @classmethod
    def validate_tension(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"tension cannot be negative, got {v}")
        return v

    @field_validator("use_count")
    @classmethod
    def validate_use_count(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"use_count cannot be negative, got {v}")
        return v

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        """Strip whitespace; reject empty content."""
        if not v or not v.strip():
            raise ValueError("content cannot be empty or whitespace-only")
        return v.strip()

    @model_validator(mode="before")
    @classmethod
    def auto_update_timestamp(cls, values):
        # used when reconstructing an existing belief from dict data; sets updated_at if missing
        if isinstance(values, dict) and "id" in values and "updated_at" not in values:
            values["updated_at"] = utcnow()
        return values

    def increment_use(self) -> None:
        self.use_count += 1
        self.updated_at = utcnow()

    def reinforce(self) -> None:
        self.origin.last_reinforced = utcnow()
        self.updated_at = utcnow()

    def apply_decay(self, decay_factor: float) -> None:
        """Apply multiplicative confidence decay. Auto-transitions to Decaying if confidence drops below 0.3."""
        if not 0.0 < decay_factor <= 1.0:
            raise ValueError(f"decay_factor must be in (0.0, 1.0], got {decay_factor}")

        self.confidence *= decay_factor
        self.updated_at = utcnow()

        # auto status transition
        if self.confidence < 0.3 and self.status == BeliefStatus.Active:
            self.status = BeliefStatus.Decaying


__all__ = ["utcnow", "OriginMetadata", "BeliefStatus", "Belief"]
