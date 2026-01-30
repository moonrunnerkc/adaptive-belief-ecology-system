# Author: Bradley R. Kinnard
"""
Belief API routes.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from ..schemas import (
    BeliefCreate,
    BeliefUpdate,
    BeliefResponse,
    BeliefListResponse,
)
from ...core.deps import get_belief_store
from ...core.models.belief import Belief, BeliefStatus, OriginMetadata

router = APIRouter(prefix="/beliefs", tags=["beliefs"])


def _belief_to_response(b: Belief) -> BeliefResponse:
    """Convert internal Belief to API response."""
    return BeliefResponse(
        id=b.id,
        content=b.content,
        confidence=b.confidence,
        status=b.status.value,
        tension=b.tension,
        cluster_id=b.cluster_id,
        parent_id=b.parent_id,
        use_count=b.use_count,
        tags=b.tags,
        source=b.origin.source,
        created_at=b.created_at,
        updated_at=b.updated_at,
    )


@router.get("", response_model=BeliefListResponse)
async def list_beliefs(
    status: Optional[str] = Query(None, description="Filter by status"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """List beliefs with optional filtering and pagination."""
    store = get_belief_store()

    # parse status filter
    status_filter = None
    if status:
        try:
            status_filter = BeliefStatus(status)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status}")

    # get beliefs
    beliefs = await store.list(
        status=status_filter,
        limit=page_size,
        offset=(page - 1) * page_size,
    )

    # filter by tag if specified
    if tag:
        beliefs = [b for b in beliefs if tag in b.tags]

    # get total count
    all_beliefs = await store.list(status=status_filter, limit=10000)
    total = len(all_beliefs)

    return BeliefListResponse(
        beliefs=[_belief_to_response(b) for b in beliefs],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{belief_id}", response_model=BeliefResponse)
async def get_belief(belief_id: UUID):
    """Get a single belief by ID."""
    store = get_belief_store()
    belief = await store.get(belief_id)

    if not belief:
        raise HTTPException(404, f"Belief {belief_id} not found")

    return _belief_to_response(belief)


@router.post("", response_model=BeliefResponse, status_code=201)
async def create_belief(req: BeliefCreate):
    """Create a new belief."""
    store = get_belief_store()

    belief = Belief(
        content=req.content,
        confidence=req.confidence,
        origin=OriginMetadata(
            source=req.source,
        ),
        tags=req.tags,
    )

    await store.create(belief)
    return _belief_to_response(belief)


@router.patch("/{belief_id}", response_model=BeliefResponse)
async def update_belief(belief_id: UUID, req: BeliefUpdate):
    """Update an existing belief."""
    store = get_belief_store()
    belief = await store.get(belief_id)

    if not belief:
        raise HTTPException(404, f"Belief {belief_id} not found")

    if req.confidence is not None:
        belief.confidence = req.confidence

    if req.tags is not None:
        belief.tags = req.tags

    if req.status is not None:
        try:
            belief.status = BeliefStatus(req.status)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {req.status}")

    belief.updated_at = datetime.now(timezone.utc)
    await store.update(belief)

    return _belief_to_response(belief)


@router.delete("/{belief_id}", status_code=204)
async def delete_belief(belief_id: UUID):
    """Delete a belief (marks as deprecated)."""
    store = get_belief_store()
    belief = await store.get(belief_id)

    if not belief:
        raise HTTPException(404, f"Belief {belief_id} not found")

    belief.status = BeliefStatus.Deprecated
    belief.updated_at = datetime.now(timezone.utc)
    await store.update(belief)


@router.post("/{belief_id}/reinforce", response_model=BeliefResponse)
async def reinforce_belief(
    belief_id: UUID,
    boost: float = Query(0.1, ge=0.0, le=0.5),
):
    """Reinforce a belief, boosting its confidence."""
    store = get_belief_store()
    belief = await store.get(belief_id)

    if not belief:
        raise HTTPException(404, f"Belief {belief_id} not found")

    belief.confidence = min(1.0, belief.confidence + boost)
    belief.origin.last_reinforced = datetime.now(timezone.utc)
    belief.use_count += 1
    belief.updated_at = datetime.now(timezone.utc)

    await store.update(belief)
    return _belief_to_response(belief)
