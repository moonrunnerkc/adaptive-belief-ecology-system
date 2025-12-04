"""
Snapshot models for capturing belief ecology state at specific iterations.
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from .belief import BeliefStatus, OriginMetadata, utcnow


class SnapshotBaseModel(BaseModel):
    model_config = {
        "from_attributes": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        },
    }


class SnapshotMetadata(SnapshotBaseModel):
    """Metadata about when and why a snapshot was captured."""

    iteration: int
    timestamp: datetime = Field(default_factory=utcnow)
    context_summary: Optional[str] = None

    @field_validator("iteration")
    @classmethod
    def validate_iteration(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"iteration cannot be negative, got {v}")
        return v


class BeliefSnapshot(SnapshotBaseModel):
    """
    Read-only copy of a belief at snapshot time.
    frozen=True prevents field reassignment. Nested objects (origin, tags) are still mutable.
    """

    id: UUID
    content: str
    confidence: float
    origin: OriginMetadata
    tags: List[str]
    tension: float
    cluster_id: Optional[UUID]
    status: BeliefStatus
    parent_id: Optional[UUID]
    use_count: int
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True,
        "frozen": True,  # prevents field assignment
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        },
    }


class Snapshot(SnapshotBaseModel):
    """
    Complete ecology state at one iteration. Used for time travel, replay,
    and analysis of how beliefs evolved.
    """

    id: UUID = Field(default_factory=uuid4)
    metadata: SnapshotMetadata
    beliefs: List[BeliefSnapshot]
    global_tension: float = 0.0
    cluster_metrics: Dict[str, dict] = Field(
        default_factory=dict
    )  # TODO: tighten to TypedDict
    agent_actions: List[dict] = Field(
        default_factory=list
    )  # TODO: define AgentAction model
    rl_state_action: Optional[dict] = None  # TODO: define RLStateAction model

    @field_validator("global_tension")
    @classmethod
    def validate_global_tension(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"global_tension cannot be negative, got {v}")
        return v


__all__ = ["SnapshotMetadata", "BeliefSnapshot", "Snapshot"]
