"""Registry of observation strategies, keyed by name.

Serializers and strategies register themselves here (via ``@register_strategy`` or a
factory) so configs can refer to them by string and ``mb list-strategies`` can enumerate
them. Strategies are stored as zero-arg *factories* so per-run parameters (e.g. screenshot
scale, salience top-k) can be bound at construction time by more specific factories.
"""

from __future__ import annotations

from collections.abc import Callable

from modalitybench.observations.base import ObservationStrategy

StrategyFactory = Callable[[], ObservationStrategy]

STRATEGIES: dict[str, StrategyFactory] = {}


def register_strategy(name: str) -> Callable[[StrategyFactory], StrategyFactory]:
    """Decorator registering a zero-arg factory under ``name``."""

    def deco(factory: StrategyFactory) -> StrategyFactory:
        if name in STRATEGIES:
            raise ValueError(f"strategy {name!r} already registered")
        STRATEGIES[name] = factory
        return factory

    return deco


def get_strategy(name: str) -> ObservationStrategy:
    if name not in STRATEGIES:
        raise KeyError(
            f"unknown strategy {name!r}; known: {', '.join(sorted(STRATEGIES)) or '(none)'}"
        )
    return STRATEGIES[name]()


def list_strategies() -> list[str]:
    return sorted(STRATEGIES)
