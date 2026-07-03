"""Populate the strategy registry.

Call :func:`load_strategies` before using the registry. Kept separate from
``observations/__init__`` to avoid an import cycle (serializers import from
``observations.base`` / ``observations.registry``).
"""

from __future__ import annotations

_LOADED = False
_STRATEGY_MODULES = ["salience", "tools_mode", "screenshot"]


def load_strategies() -> None:
    global _LOADED
    if _LOADED:
        return
    import modalitybench.observations.serializers  # noqa: F401  (registers 7 serializers)

    for mod in _STRATEGY_MODULES:
        try:  # screenshot needs Pillow at call time, not import time — safe to import here
            __import__(f"modalitybench.observations.{mod}")
        except ImportError:
            pass
    _LOADED = True
