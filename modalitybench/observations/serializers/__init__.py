"""Serializer implementations.

Importing this package registers every serializer with the strategy registry (Phase 2+).
It is intentionally import-safe when individual serializer modules are absent so that
``mb list-strategies`` works at every stage of development.
"""

from __future__ import annotations

# Serializer modules register themselves on import. Guarded so a partially-built tree still
# imports cleanly.
_SERIALIZER_MODULES = [
    "raw_html",
    "body_html",
    "pruned_html",
    "axtree",
    "flat_elements",
    "afus",
    "fct",
]

for _mod in _SERIALIZER_MODULES:
    try:  # pragma: no cover - import guard
        __import__(f"{__name__}.{_mod}")
    except ImportError:
        pass
