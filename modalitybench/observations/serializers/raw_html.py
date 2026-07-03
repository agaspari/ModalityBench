"""raw_html — the full page HTML, untouched. The token-cost baseline."""

from __future__ import annotations

from modalitybench.observations.base import PageGraph
from modalitybench.observations.registry import register_strategy
from modalitybench.observations.serializers._common import make_observation, render_html


class RawHtmlSerializer:
    name = "raw_html"

    def observe(self, graph: PageGraph, *, task_text=None):
        # Prefer the captured raw HTML (honest baseline); fall back to re-rendering.
        html = graph.meta.get("html")
        if not html:
            html = render_html(graph.root, inject_ref=False)
        return make_observation(graph, self.name, html)


@register_strategy("raw_html")
def _make() -> RawHtmlSerializer:
    return RawHtmlSerializer()
