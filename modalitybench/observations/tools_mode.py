"""tools_mode — "no upfront DOM": expose query tools over the page instead of dumping it.

The initial observation is just a tiny landmark outline plus a set of query tools
(``outline``, ``find``, ``read``). The agent explores by calling them; the loop answers each
from the captured ``PageGraph`` and appends the result. This trades a large fixed observation
for a few small targeted reads — the extreme end of the reduction spectrum.

The strategy stashes a bound :class:`GraphQueryHandler` in ``observation.meta['tool_handler']``
so the agent loop can answer queries without recapturing the page.
"""

from __future__ import annotations

from modalitybench.observations.base import (
    Observation,
    PageGraph,
    PageNode,
    TextBlock,
    ToolSpec,
)
from modalitybench.observations.registry import register_strategy
from modalitybench.observations.serializers._common import build_registry, short_type

_TOOLS = [
    ToolSpec("outline", "List the page's landmark sections.", {"type": "object", "properties": {}}),
    ToolSpec(
        "find",
        "Search the page for elements whose name/role matches a query. Returns matching refs.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    ),
    ToolSpec(
        "read",
        "Read the full details of one element (or a section's elements) by ref.",
        {"type": "object", "properties": {"ref": {"type": "string"}}, "required": ["ref"]},
    ),
]


class GraphQueryHandler:
    """Answers tools-mode queries against a captured graph (text results)."""

    def __init__(self, graph: PageGraph) -> None:
        self.graph = graph
        self._by_ref = graph.by_ref()

    def outline(self) -> str:
        scopes: list[str] = []
        seen = set()
        for n in self.graph.nodes():
            if n.scope and n.scope not in seen:
                seen.add(n.scope)
                scopes.append(n.scope)
        body = "\n".join(f"- {s}" for s in scopes) or "(no named sections)"
        return f"Sections:\n{body}\nUse find(query) to locate elements."

    def find(self, query: str) -> str:
        q = (query or "").lower().split()
        hits: list[tuple[int, PageNode]] = []
        for n in self.graph.nodes():
            if n.ref is None or not n.visible:
                continue
            hay = " ".join(
                filter(None, [n.name, n.text, n.role, n.scope, *(n.attrs.values())])
            ).lower()
            score = sum(1 for w in q if w in hay)
            if score:
                hits.append((score, n))
        hits.sort(key=lambda t: (-t[0], int(t[1].ref[1:])))
        if not hits:
            return f"No elements match {query!r}."
        lines = [f"Matches for {query!r}:"]
        for _, n in hits[:12]:
            lines.append(self._describe(n))
        return "\n".join(lines)

    def read(self, ref: str) -> str:
        node = self._by_ref.get(ref)
        if node is None:
            return f"No element with ref {ref!r}."
        # If it's a section label, read its members; else the element itself.
        members = [n for n in self.graph.nodes() if n.ref and n.scope == node.name]
        if members and not node.is_interactive:
            return "\n".join([f"Section {node.name!r}:"] + [self._describe(m) for m in members])
        return self._describe(node)

    def _describe(self, n: PageNode) -> str:
        parts = [f"[{n.ref}] {short_type(n)} {n.role or n.tag}"]
        if n.name:
            parts.append(f'"{n.name}"')
        if n.value:
            parts.append(f"= {n.value}")
        if n.flags:
            parts.append("(" + ",".join(sorted(n.flags)) + ")")
        if n.scope:
            parts.append(f"@{n.scope}")
        return " ".join(parts)


class ToolsModeStrategy:
    name = "tools_mode"

    def observe(self, graph: PageGraph, *, task_text=None) -> Observation:
        handler = GraphQueryHandler(graph)
        outline = handler.outline()
        text = (
            "You are in tools mode: the full page is NOT shown. Explore it with the "
            "query actions outline(), find(query), read(ref), then act.\n\n" + outline
        )
        obs = Observation(
            content_blocks=[TextBlock(text=text)],
            tools=list(_TOOLS),
            ref_registry=build_registry(graph),
            meta={
                "serializer": self.name,
                "chars": len(text),
                "bytes": len(text.encode("utf-8")),
                "element_count": len(graph.by_ref()),
                "tools_mode": True,
            },
        )
        obs.meta["tool_handler"] = handler
        return obs


@register_strategy("tools_mode")
def _make() -> ToolsModeStrategy:
    return ToolsModeStrategy()
