"""axtree — an accessibility-tree text rendering.

Indented ``role "name" [value] {flags} ref`` lines, generic wrapper nodes collapsed. This is
the closest analogue to what tools like Playwright MCP emit, included as a strong baseline.
Indentation is intentional here (it is the axtree convention), unlike the flat/AFUS/FCT
formats which drop it to save tokens.
"""

from __future__ import annotations

from modalitybench.observations.base import PageGraph, PageNode
from modalitybench.observations.registry import register_strategy
from modalitybench.observations.serializers._common import make_observation

_INDENT = "  "


def _keep(node: PageNode) -> bool:
    return bool(node.role) or node.ref is not None or bool(node.text)


def _render(node: PageNode, depth: int, lines: list[str]) -> None:
    if not node.visible:
        return
    if _keep(node):
        parts = [f"{_INDENT * depth}{node.role or node.tag}"]
        name = node.name or node.text
        if name:
            parts.append(f' "{name}"')
        if node.value:
            parts.append(f" [{node.value}]")
        if node.flags:
            parts.append(" {" + ",".join(sorted(node.flags)) + "}")
        if node.ref:
            parts.append(f" {node.ref}")
        lines.append("".join(parts))
        child_depth = depth + 1
    else:
        child_depth = depth  # collapse generic container
    for child in node.children:
        _render(child, child_depth, lines)


class AxtreeSerializer:
    name = "axtree"

    def observe(self, graph: PageGraph, *, task_text=None):
        lines: list[str] = []
        for child in graph.root.children:
            _render(child, 0, lines)
        return make_observation(graph, self.name, "\n".join(lines))


@register_strategy("axtree")
def _make() -> AxtreeSerializer:
    return AxtreeSerializer()
