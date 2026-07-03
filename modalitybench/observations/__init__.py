"""Observation strategies: capture + serialize + ref registry.

An *observation strategy* is the load-bearing abstraction of ModalityBench. It turns a
page (live Playwright page or an offline HTML/graph snapshot) into an ``Observation`` the
model can consume, and carries a ``ref_registry`` mapping the element refs it emits back to
actionable locators.
"""

from modalitybench.observations.base import (
    ContentBlock,
    ImageBlock,
    Observation,
    ObservationStrategy,
    PageGraph,
    PageNode,
    RefRegistry,
    TextBlock,
    ToolSpec,
)
from modalitybench.observations.registry import (
    STRATEGIES,
    get_strategy,
    list_strategies,
    register_strategy,
)

__all__ = [
    "ContentBlock",
    "ImageBlock",
    "Observation",
    "ObservationStrategy",
    "PageGraph",
    "PageNode",
    "RefRegistry",
    "TextBlock",
    "ToolSpec",
    "STRATEGIES",
    "get_strategy",
    "list_strategies",
    "register_strategy",
]
