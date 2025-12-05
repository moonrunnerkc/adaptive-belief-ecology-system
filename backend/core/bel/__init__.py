"""
Belief Ecology Loop (BEL) implementation.
"""

from .decay import apply_decay
from .loop import BeliefEcologyLoop

__all__ = ["BeliefEcologyLoop", "apply_decay"]
