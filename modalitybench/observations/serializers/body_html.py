"""body_html — body only, scripts/styles/comments stripped, refs injected.

Re-serialized from the PageGraph (which already excludes script/style/head), keeping the
element hierarchy and a broad attribute set. A middle-ground baseline between raw HTML and
the aggressively pruned variant.
"""

from __future__ import annotations

from modalitybench.observations.base import PageGraph
from modalitybench.observations.registry import register_strategy
from modalitybench.observations.serializers._common import make_observation, render_html


class BodyHtmlSerializer:
    name = "body_html"

    def observe(self, graph: PageGraph, *, task_text=None):
        html = render_html(graph.root, prune_hidden=False, inject_ref=True)
        return make_observation(graph, self.name, html)


@register_strategy("body_html")
def _make() -> BodyHtmlSerializer:
    return BodyHtmlSerializer()
