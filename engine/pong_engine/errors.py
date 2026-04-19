from __future__ import annotations


class PongEngineError(Exception):
    """Base class for engine errors."""


class InvalidGameStateError(PongEngineError):
    """Raised when a GameState violates domain invariants."""


class InvalidActionError(PongEngineError):
    """Raised when an invalid paddle action reaches the engine."""


class MissingRngError(PongEngineError):
    """Raised when a function that needs RNG did not receive one."""
