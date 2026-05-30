"""
KEY ROTATION UTILITY
====================

Thread-safe round-robin index for rotating across multiple Groq API keys.
Each call to next_index() advances the counter by one (wrapping around).

Usage:
    from app.utils.key_rotation import KeyRotation
    rotator = KeyRotation(num_keys=3)
    idx = rotator.next_index()   # 0, 1, 2, 0, 1, 2, …
"""

import threading


class KeyRotation:
    """
    Thread-safe round-robin counter.

    Args:
        num_keys: Total number of API keys available.
    """

    def __init__(self, num_keys: int):
        if num_keys < 1:
            raise ValueError("num_keys must be >= 1")
        self._num_keys = num_keys
        self._index = 0
        self._lock = threading.Lock()

    def next_index(self) -> int:
        """Return the next key index and advance the counter."""
        with self._lock:
            idx = self._index
            self._index = (self._index + 1) % self._num_keys
            return idx

    def current_index(self) -> int:
        """Return the current index without advancing."""
        with self._lock:
            return self._index

    @property
    def num_keys(self) -> int:
        return self._num_keys


# ── Module-level singleton ────────────────────────────────────────────────────
# Lazily initialised in main.py after GROQ_API_KEYS are loaded.
_rotator: "KeyRotation | None" = None


def init_rotator(num_keys: int) -> KeyRotation:
    """Create (or replace) the module-level rotator. Call once at startup."""
    global _rotator
    _rotator = KeyRotation(num_keys)
    return _rotator


def get_next_key_index() -> int:
    """Return the next key index from the module-level rotator."""
    if _rotator is None:
        return 0
    return _rotator.next_index()


def get_current_key_index() -> int:
    """Return the current key index without advancing."""
    if _rotator is None:
        return 0
    return _rotator.current_index()
