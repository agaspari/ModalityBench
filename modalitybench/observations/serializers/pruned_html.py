"""pruned_html — hidden elements removed, attributes allowlisted, long text truncated.

The most aggressive HTML variant: keeps the tag hierarchy (so structural cues survive) but
drops invisible nodes, restricts attributes to a small interaction-relevant set, and caps
text length. Refs injected for actionability.
"""

from __future__ import annotations

from modalitybench.observations.base import PageGraph
from modalitybench.observations.registry import register_strategy
from modalitybench.observations.serializers._common import make_observation, render_html

_ALLOW = {"type", "name", "href", "placeholder", "role", "aria-label", "value", "alt"}
_MAX_TEXT = 120


class PrunedHtmlSerializer:
    name = "pruned_html"

    def observe(self, graph: PageGraph, *, task_text=None):
        html = render_html(
            graph.root,
            prune_hidden=True,
            allow_attrs=_ALLOW,
            max_text=_MAX_TEXT,
            inject_ref=True,
        )
        return make_observation(graph, self.name, html)


@register_strategy("pruned_html")
def _make() -> PrunedHtmlSerializer:
    return PrunedHtmlSerializer()
