"""Paginated-extraction — a browser-free live task source for the context-lifecycle study.

The agent lands on page 1 of an N-page catalog, each page listing a handful of products as
**static text** (name + price), plus a ``Next`` link. The goal is to walk every page and
return all products as JSON. Scoring is extraction fidelity against the known catalog.

Two properties make this the testbed the headline needs (see ``RESEARCH-NOTES.md``):

* **It's genuinely extraction-shaped.** Items are non-interactive text, so an interactive-only
  serializer (``flat_elements``) can't see them at all — this is where ``adaptive`` must
  escalate (``request_text``) and where a text serializer (``pruned_html``) pays on every page.
* **It's multi-step over a growing trajectory.** Under ``history_mode: accumulate`` the model
  re-reads every page it has visited each step, so cumulative billed tokens grow ~quadratically
  in page count; under ``evict`` they stay flat. Varying ``page_counts`` sweeps that curve.

Browser-free: pages are generated HTML captured via ``graph_from_html`` (no Playwright), so it
runs in CI and sidesteps the browser/ARM install pain that blocked WebShop.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

from modalitybench.agents.actions import Action
from modalitybench.observations.base import PageGraph, RefRegistry
from modalitybench.observations.dom_capture import graph_from_html
from modalitybench.tasks.base import Episode, LiveOutcome, Task, TaskResult

_ADJECTIVES = [
    "Alpine", "Coastal", "Vivid", "Rustic", "Nimble", "Lunar", "Copper", "Verdant",
    "Silent", "Golden", "Crimson", "Arctic", "Hazel", "Cobalt", "Amber", "Onyx",
]
_NOUNS = [
    "Kettle", "Satchel", "Lantern", "Notebook", "Compass", "Blanket", "Tumbler",
    "Planter", "Headlamp", "Umbrella", "Backpack", "Speaker", "Wallet", "Mug",
    "Scarf", "Charger",
]


@dataclass
class _Handle:
    goal: str
    items: list[dict[str, Any]]  # full ground-truth catalog
    pages: list[list[dict[str, Any]]]  # items chunked per page
    page_idx: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


def _catalog(seed: int, n_items: int) -> list[dict[str, Any]]:
    """Deterministic, unique-named catalog for a given seed/size."""
    rng = random.Random(seed)
    names: set[str] = set()
    items: list[dict[str, Any]] = []
    i = 0
    while len(items) < n_items:
        name = f"{rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)} {1000 + i}"
        i += 1
        if name in names:
            continue
        names.add(name)
        price = round(rng.uniform(4.0, 99.0), 2)
        items.append({"name": name, "price": price})
    return items


def _page_html(handle: _Handle) -> str:
    idx = handle.page_idx
    n = len(handle.pages)
    rows = "".join(
        f'<li class="item"><span class="name">{it["name"]}</span>'
        f' — <span class="price">${it["price"]:.2f}</span></li>'
        for it in handle.pages[idx]
    )
    nav = f'<a id="next" href="#p{idx + 1}">Next</a>' if idx < n - 1 else "<span>End</span>"
    return (
        f"<html><head><title>Catalog</title></head><body>"
        f"<h1>Product catalog — page {idx + 1} of {n}</h1>"
        f'<ul class="products">{rows}</ul>'
        f'<nav>{nav}</nav>'
        f"</body></html>"
    )


class PaginatedExtractionSource:
    name = "paginated_extraction"
    is_live = True

    def __init__(
        self,
        *,
        page_counts: list[int] | None = None,
        items_per_page: int = 10,
        seed: int = 0,
    ) -> None:
        self.page_counts = page_counts or [2, 4, 6]
        self.items_per_page = items_per_page
        self.seed = seed

    # -- TaskSource API ------------------------------------------------------

    def tasks(self) -> list[Task]:
        return [
            Task(
                task_id=f"extract-{npages}p",
                goal=self._goal(npages),
                source=self.name,
                seed=self.seed,
                meta={"n_pages": npages, "items_per_page": self.items_per_page},
            )
            for npages in self.page_counts
        ]

    def _goal(self, npages: int) -> str:
        total = npages * self.items_per_page
        return (
            f"This catalog has {npages} pages with {total} products total. Visit every page "
            f"(click the Next link to advance), collect every product's name and price, then "
            f'call done with answer set to a JSON array of {{"name": ..., "price": ...}} '
            f"objects — one per product across all pages."
        )

    def reset(self, task: Task) -> _Handle:
        npages = int(task.meta["n_pages"])
        ipp = int(task.meta["items_per_page"])
        items = _catalog(task.seed or 0, npages * ipp)
        pages = [items[i : i + ipp] for i in range(0, len(items), ipp)]
        return _Handle(goal=task.goal, items=items, pages=pages, page_idx=0)

    def capture(self, handle: _Handle, *, screenshot: bool = False) -> PageGraph:
        return graph_from_html(
            _page_html(handle), url=f"catalog://p{handle.page_idx + 1}", source="html"
        )

    def apply(
        self, handle: _Handle, action: Action, registry: RefRegistry, graph: PageGraph
    ) -> LiveOutcome:
        node = graph.by_ref().get(action.ref) if action.ref else None
        label = (node.name or node.text or "").strip().lower() if node else ""
        is_next = label == "next"
        if action.kind == "click" and is_next:
            if handle.page_idx < len(handle.pages) - 1:
                handle.page_idx += 1
                return LiveOutcome(reward=0.0, terminated=False, info=f"page {handle.page_idx + 1}")
            return LiveOutcome(reward=0.0, terminated=False, info="already on last page")
        return LiveOutcome(reward=0.0, terminated=False, info="no-op")

    def close(self, handle: _Handle) -> None:
        pass

    def score(self, task: Task, episode: Episode) -> TaskResult:
        gt = _catalog(task.seed or 0, int(task.meta["n_pages"]) * int(task.meta["items_per_page"]))
        gt_by_name = {_norm(i["name"]): float(i["price"]) for i in gt}
        extracted = _parse_answer(episode.meta.get("answer"))

        seen_names: set[str] = set()
        name_hits = 0
        exact_hits = 0
        for name, price in extracted:
            k = _norm(name)
            if k in gt_by_name and k not in seen_names:
                seen_names.add(k)
                name_hits += 1
                if price is not None and abs(price - gt_by_name[k]) < 0.01:
                    exact_hits += 1

        n_gt = len(gt_by_name)
        n_pred = len(extracted)
        name_recall = name_hits / n_gt if n_gt else 0.0
        exact_recall = exact_hits / n_gt if n_gt else 0.0
        precision = name_hits / n_pred if n_pred else 0.0
        f1 = (2 * precision * name_recall / (precision + name_recall)) if (precision + name_recall) else 0.0
        return TaskResult(
            success=exact_recall >= 0.99,
            reward=round(exact_recall, 4),
            metrics={
                "extraction_f1": round(f1, 4),
                "name_recall": round(name_recall, 4),
                "exact_recall": round(exact_recall, 4),
                "precision": round(precision, 4),
                "n_items_gt": float(n_gt),
                "n_items_extracted": float(n_pred),
                "n_pages": float(task.meta["n_pages"]),
                "n_steps": float(len(episode.steps)),
            },
        )


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _parse_answer(answer: Any) -> list[tuple[str, float | None]]:
    """Tolerantly parse the model's done(answer=...) into (name, price) pairs."""
    if answer is None:
        return []
    data: Any = answer
    if isinstance(answer, str):
        text = answer.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("["), text.rfind("]")
            if start != -1 and end > start:
                try:
                    data = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return []
            else:
                return []
    if not isinstance(data, list):
        return []
    out: list[tuple[str, float | None]] = []
    for row in data:
        if isinstance(row, dict):
            name = row.get("name") or row.get("product") or row.get("title")
            if name is None:
                continue
            out.append((str(name), _price(row.get("price"))))
        elif isinstance(row, (list, tuple)) and row:
            out.append((str(row[0]), _price(row[1]) if len(row) > 1 else None))
    return out


def _price(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d+(?:\.\d+)?", str(v))
    return float(m.group()) if m else None
