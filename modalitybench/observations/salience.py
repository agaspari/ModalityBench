"""salience — task-aware pruning wrapper around any DOM serializer.

Scores each salient node against the task text (heuristic token overlap + an interactive
bonus), keeps the top-k, prunes everything else from a *copy* of the graph, then delegates
to a base serializer. This is a decorator strategy: ``salience_afus`` is AFUS over the pruned
graph, ``salience_flat`` is flat_elements, etc. The heuristic scorer is intentionally simple;
an embedding scorer is on the roadmap.
"""

from __future__ import annotations

import copy

from modalitybench.observations.base import Observation, PageGraph, PageNode
from modalitybench.observations.registry import get_strategy, register_strategy

_LANDMARK_TAGS = {"nav", "main", "header", "footer", "aside", "section", "form"}
_LANDMARK_ROLES = {
    "navigation", "main", "banner", "contentinfo", "complementary", "form", "region",
}


def _score(node: PageNode, words: set[str]) -> float:
    hay = " ".join(
        filter(None, [node.name, node.text, node.role, node.scope, *node.attrs.values()])
    ).lower()
    overlap = sum(1 for w in words if len(w) > 2 and w in hay)
    score = overlap * 2.0
    if node.is_interactive:
        score += 0.5
    return score


def _clone_keep(node: PageNode, keep: set[str]) -> PageNode | None:
    kept_children: list[PageNode] = []
    for child in node.children:
        c = _clone_keep(child, keep)
        if c is not None:
            kept_children.append(c)
    node_kept = node.ref is not None and node.ref in keep
    is_landmark = node.tag in _LANDMARK_TAGS or (node.role in _LANDMARK_ROLES)
    if not (node_kept or kept_children or (is_landmark and kept_children)):
        return None
    clone = copy.copy(node)
    clone.children = kept_children
    clone.flags = set(node.flags)
    clone.attrs = dict(node.attrs)
    if node.ref is not None and not node_kept:
        clone.ref = None  # keep as structural wrapper, drop its ref
        clone.locator = None
    return clone


def _prune_graph(graph: PageGraph, keep: set[str]) -> PageGraph:
    root = _clone_keep(graph.root, keep) or PageNode(tag="body")
    return PageGraph(root=root, url=graph.url, title=graph.title, source=graph.source)


class SalienceStrategy:
    def __init__(self, base: str, top_k: int = 15) -> None:
        self.base = base
        self.top_k = top_k
        self.name = f"salience_{base}"

    def observe(self, graph: PageGraph, *, task_text=None) -> Observation:
        words = {w.strip(".,!?\"'()").lower() for w in (task_text or "").split()}
        salient = [n for n in graph.nodes() if n.ref is not None and n.visible]
        ranked = sorted(
            salient,
            key=lambda n: (-_score(n, words), 0 if n.is_interactive else 1, int(n.ref[1:])),
        )
        keep = {n.ref for n in ranked[: self.top_k]}
        pruned = _prune_graph(graph, keep)
        obs = get_strategy(self.base).observe(pruned, task_text=task_text)
        obs.meta["serializer"] = self.name
        obs.meta["salience_base"] = self.base
        obs.meta["salience_kept"] = len(pruned.by_ref())
        obs.meta["salience_top_k"] = self.top_k
        return obs


@register_strategy("salience_flat")
def _make_flat() -> SalienceStrategy:
    return SalienceStrategy("flat_elements", top_k=15)


@register_strategy("salience_afus")
def _make_afus() -> SalienceStrategy:
    return SalienceStrategy("afus", top_k=15)


@register_strategy("salience_fct")
def _make_fct() -> SalienceStrategy:
    return SalienceStrategy("fct", top_k=15)
