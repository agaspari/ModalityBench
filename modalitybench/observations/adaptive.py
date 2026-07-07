"""adaptive — the ``request_detail`` hybrid: lean affordances + on-demand text escalation.

The default observation is the lean interactive-affordance view (the ``flat_elements`` line
format), plus a **stub** advertising how much static text the affordance list dropped. The
model self-routes: navigation legs act on the lean view in one round; extraction legs call
``request_text()`` / ``request_detail(ref)`` to pull the text that flat filters out, then act.

"tools_mode redeemed": unlike naive tools_mode (empty default → an exploration sub-loop on
every step), the default here is already actionable and the round cap is low (3), so most
steps commit with zero meta overhead. Because the live loop evicts observations (history is
action strings only), an escalated text pull lives in context for exactly the step that
needed it and then vanishes — fixed-``axtree`` pays for that text on every step.

The handler subclasses :class:`GraphQueryHandler`; the text it returns already exists in the
graph (``dom_capture`` keeps text-leaf nodes) — flat just filters them out with
``and n.is_interactive``.
"""

from __future__ import annotations

from modalitybench.observations.base import Observation, PageGraph, PageNode, TextBlock
from modalitybench.observations.registry import register_strategy
from modalitybench.observations.serializers._common import build_registry
from modalitybench.observations.serializers.flat_elements import _line
from modalitybench.observations.tools_mode import GraphQueryHandler


def _text_leaves(graph: PageGraph) -> list[PageNode]:
    """Visible, non-interactive nodes carrying their own text — what flat drops."""
    return [
        n
        for n in graph.nodes()
        if n.text and n.text.strip() and not n.is_interactive and n.visible
    ]


def _text_line(n: PageNode) -> str:
    t = n.text.strip()
    return f"{t}  @{n.scope}" if n.scope else t


class AdaptiveHandler(GraphQueryHandler):
    """GraphQueryHandler + the two text-escalation meta-actions."""

    def request_text(self) -> str:
        leaves = _text_leaves(self.graph)
        if not leaves:
            return "(no additional static text on this page)"
        return "\n".join(_text_line(n) for n in leaves)

    def request_detail(self, ref: str) -> str:
        return self.read(ref or "")


class AdaptiveStrategy:
    name = "adaptive"

    def observe(self, graph: PageGraph, *, task_text=None) -> Observation:
        lines = [
            _line(n)
            for n in graph.nodes()
            if n.ref is not None and n.visible and n.is_interactive
        ]
        body = "\n".join(lines) if lines else "(no interactive elements visible on this page)"

        n_hidden = len(_text_leaves(graph))
        if n_hidden:
            body += (
                f"\n\n[{n_hidden} text region(s) on this page are hidden from this view. "
                "Call request_text() to read them, or request_detail(ref) for one element "
                "or section. Most steps don't need this — act on the affordances above when "
                "you can.]"
            )

        handler = AdaptiveHandler(graph)
        reg = build_registry(graph)
        obs = Observation(
            content_blocks=[TextBlock(text=body)],
            ref_registry=reg,
            meta={
                "serializer": self.name,
                "chars": len(body),
                "bytes": len(body.encode("utf-8")),
                "element_count": len(reg),
                "tools_mode": True,
                "max_meta_rounds": 3,
                "hidden_text_regions": n_hidden,
            },
        )
        obs.meta["tool_handler"] = handler
        return obs


@register_strategy("adaptive")
def _make() -> AdaptiveStrategy:
    return AdaptiveStrategy()
