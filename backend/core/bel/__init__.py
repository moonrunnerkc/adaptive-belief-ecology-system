"""
Belief Ecology Loop (BEL) implementation.
"""

from .contradiction import compute_tensions
from .decay import apply_decay
from .loop import BeliefEcologyLoop

__all__ = ["BeliefEcologyLoop", "apply_decay", "compute_tensions"]
