"""flat_elements — a flat list of interactive elements, no hierarchy.

One line per actionable element: ``[e3] textbox "Card number" (required) = value``. The
Agent-E / Set-of-Marks style: cheapest text format that still names every affordance, but
discards structural context.
"""

from __future__ import annotations

from modalitybench.observations.base import PageGraph, PageNode
from modalitybench.observations.registry import register_strategy
from modalitybench.observations.serializers._common import make_observation


def _line(node: PageNode) -> str:
    parts = [f"[{node.ref}] {node.role or node.tag}"]
    if node.name:
        parts.append(f' "{node.name}"')
    if node.value:
        parts.append(f" = {node.value}")
    if node.flags:
        parts.append(" (" + ", ".join(sorted(node.flags)) + ")")
    if node.scope:
        parts.append(f"  @{node.scope}")
    return "".join(parts)


class FlatElementsSerializer:
    name = "flat_elements"

    def observe(self, graph: PageGraph, *, task_text=None):
        lines = [
            _line(n)
            for n in graph.nodes()
            if n.ref is not None and n.visible and n.is_interactive
        ]
        return make_observation(graph, self.name, "\n".join(lines))


@register_strategy("flat_elements")
def _make() -> FlatElementsSerializer:
    return FlatElementsSerializer()
