"""Paginated-extraction source + the evict/accumulate history modes (browser-free)."""

from __future__ import annotations

import json
import re

from modalitybench.agents.loop import evaluate_live
from modalitybench.agents.model_client import MockClient
from modalitybench.metrics.tokens import TokenCounter
from modalitybench.observations import get_strategy
from modalitybench.observations.loader import load_strategies
from modalitybench.tasks.paginated_extraction import (
    PaginatedExtractionSource,
    _catalog,
    _parse_answer,
)

load_strategies()


def _src(pages=2, ipp=3, seed=0):
    return PaginatedExtractionSource(page_counts=[pages], items_per_page=ipp, seed=seed)


# -- generation & scoring ---------------------------------------------------


def test_catalog_is_deterministic_and_unique():
    a = _catalog(0, 20)
    b = _catalog(0, 20)
    assert a == b
    assert len({i["name"] for i in a}) == 20  # unique names


def test_capture_hides_items_from_flat_but_not_text_serializers():
    src = _src()
    g = src.capture(src.reset(src.tasks()[0]))
    first = _catalog(0, 6)[0]["name"]
    flat = get_strategy("flat_elements").observe(g).text()
    pruned = get_strategy("pruned_html").observe(g).text()
    assert "Next" in flat and first not in flat  # interactive-only floors on extraction
    assert first in pruned  # text serializer sees the items


def test_apply_advances_pages_on_next():
    from modalitybench.agents.actions import Action

    src = _src(pages=3)
    h = src.reset(src.tasks()[0])
    g = src.capture(h)
    next_ref = next(n.ref for n in g.nodes() if (n.name or "").strip().lower() == "next")

    assert h.page_idx == 0
    out = src.apply(h, Action(kind="click", ref=next_ref), None, g)
    assert h.page_idx == 1 and not out.terminated


def test_score_full_vs_partial_answer():
    src = _src(pages=2, ipp=3)
    task = src.tasks()[0]
    items = _catalog(0, 6)
    from modalitybench.tasks.base import Episode

    full = Episode(task_id=task.task_id, strategy="x", model="m")
    full.meta["answer"] = json.dumps(items)
    r_full = src.score(task, full)
    assert r_full.success and r_full.reward == 1.0

    half = Episode(task_id=task.task_id, strategy="x", model="m")
    half.meta["answer"] = json.dumps(items[:3])
    r_half = src.score(task, half)
    assert not r_half.success
    assert abs(r_half.metrics["exact_recall"] - 0.5) < 1e-9


def test_parse_answer_tolerates_formats():
    assert _parse_answer('[{"name": "A", "price": "$5.00"}]') == [("A", 5.0)]
    assert _parse_answer('junk [{"name":"A","price":9}] tail') == [("A", 9.0)]
    assert _parse_answer(None) == []
    assert _parse_answer("not json") == []


# -- end-to-end loop: navigation + extraction + history modes ----------------


def _find_next_ref(text: str) -> str | None:
    """Locate the 'Next' link's ref in either bracket (flat/axtree) or HTML (pruned) form."""
    m = re.search(r"\[(\w+)\][^\n]*\bNext\b", text)  # flat: [e8] link "Next"
    if m:
        return m.group(1)
    for mm in re.finditer(r'ref="(\w+)"[^>]*>\s*([^<]*)', text):  # pruned: <a ref="e8">Next
        if "Next" in mm.group(2):
            return mm.group(1)
    return None


def _nav_responder(answer_json: str):
    """Click the current page's Next link; when there's no Next (last page), submit answer."""

    def responder(*, system, blocks, tools):
        text = "\n".join(getattr(b, "text", "") for b in blocks)
        current = text.split("CURRENT PAGE OBSERVATION:")[-1]  # ignore accumulated prior pages
        ref = _find_next_ref(current)
        if ref:
            return json.dumps({"action": "click", "ref": ref})
        return json.dumps({"action": "done", "answer": answer_json})

    return responder


def test_evaluate_live_walks_all_pages_and_scores(monkeypatch):
    src = _src(pages=3, ipp=3)
    task = src.tasks()[0]
    answer = json.dumps(_catalog(0, 9))
    client = MockClient(responder=_nav_responder(answer))
    ep = evaluate_live(src, task, "pruned_html", client, TokenCounter(), max_steps=10)
    # 3 pages => 2 Next clicks + 1 done
    assert [s.action_kind for s in ep.steps] == ["click", "click", "done"]
    assert ep.meta["answer"] == answer
    result = src.score(task, ep)
    assert result.success and result.reward == 1.0


def test_accumulate_bills_more_than_evict(monkeypatch):
    task = _src(pages=3, ipp=4).tasks()[0]
    answer = json.dumps(_catalog(0, 12))

    def run(mode: str) -> int:
        src = _src(pages=3, ipp=4)
        client = MockClient(responder=_nav_responder(answer))
        ep = evaluate_live(
            src, task, "pruned_html", client, TokenCounter(), max_steps=10, history_mode=mode
        )
        return sum(s.input_tokens for s in ep.steps), client

    (evict_tok, evict_client) = run("evict")
    (acc_tok, acc_client) = run("accumulate")

    # Accumulate re-sends prior pages every step => strictly more billed input tokens.
    assert acc_tok > evict_tok
    # ...and the earlier-pages block only appears in accumulate mode.
    acc_text = "\n".join(
        getattr(b, "text", "") for c in acc_client.calls for b in c["blocks"]
    )
    evict_text = "\n".join(
        getattr(b, "text", "") for c in evict_client.calls for b in c["blocks"]
    )
    assert "EARLIER PAGES" in acc_text
    assert "EARLIER PAGES" not in evict_text
