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


# Item names look like "Hazel Mug 1000" (Adjective Noun 4-digit-id) in any serializer's output.
_ITEM_RE = re.compile(r"[A-Z][a-z]+ [A-Z][a-z]+ \d{4}")


def _sees_only_agent(*, escalate: bool = True):
    """A **memoryless** agent that can only act on what its current prompt actually shows it.

    This is the key test tool: unlike a scripted mock, it is never handed the answer — it
    reports *exactly* the item names present in its context, so each serializer's observation
    becomes the tested contract:

    * hide the items (``flat_elements``) -> the agent sees none -> it floors (reward 0);
    * show them (``pruned_html``) -> it extracts what's on screen;
    * put them behind a stub (``adaptive``) -> it must ``request_text`` first to see them.

    Because it has no memory across steps, under ``evict`` it can only submit the *last* page it
    saw, while under ``accumulate`` the re-sent earlier pages are in-context so it submits them
    all — which is precisely the eviction claim, now deterministic and API-free.

    Policy each turn (using only the current prompt): escalate if items are hidden behind a
    stub; else click Next while one is on the current page; else submit every item name visible
    anywhere in the context.
    """

    def responder(*, system, blocks, tools):
        full = "\n".join(getattr(b, "text", "") for b in blocks)
        current = full.split("CURRENT PAGE OBSERVATION:")[-1]
        if escalate and "request_text()" in current and not _ITEM_RE.search(current):
            return json.dumps({"action": "request_text"})  # adaptive: reveal hidden items
        ref = _find_next_ref(current)
        if ref:
            return json.dumps({"action": "click", "ref": ref})  # walk to the next page
        names = list(dict.fromkeys(_ITEM_RE.findall(full)))  # only what it has actually seen
        return json.dumps({"action": "done", "answer": json.dumps([{"name": n} for n in names])})

    return responder


def _run(strategy, pages, ipp, *, mode="evict", escalate=True, max_steps=15, seed=0):
    src = PaginatedExtractionSource(page_counts=[pages], items_per_page=ipp, seed=seed)
    task = src.tasks()[0]
    client = MockClient(responder=_sees_only_agent(escalate=escalate))
    ep = evaluate_live(src, task, strategy, client, TokenCounter(), max_steps, history_mode=mode)
    return src.score(task, ep), ep, client


def test_pruned_agent_walks_all_pages_and_extracts():
    # A sees-only agent on a text serializer, accumulate mode: navigates every page and
    # legitimately reconstructs the whole catalog (not handed to it).
    result, ep, _ = _run("pruned_html", pages=3, ipp=3, mode="accumulate")
    assert [s.action_kind for s in ep.steps] == ["click", "click", "done"]
    assert result.success and result.reward == 1.0


# -- the thesis, as deterministic tests --------------------------------------


def test_modality_floor_flat_cannot_extract_pruned_can():
    # Single page (no navigation): the ONLY difference is what the serializer reveals.
    flat_r, _, _ = _run("flat_elements", pages=1, ipp=5)
    pruned_r, _, _ = _run("pruned_html", pages=1, ipp=5)
    assert flat_r.reward == 0.0  # items are non-interactive text -> flat hides them -> floor
    assert pruned_r.reward == 1.0  # same page, text serializer -> full extraction


def test_eviction_lifecycle_accumulate_beats_evict_on_extraction():
    # pruned_html over 3 pages: evict loses earlier pages (memoryless agent sees only the last),
    # accumulate re-sends them so the agent can report all. The context lifecycle IS the result.
    evict_r, _, _ = _run("pruned_html", pages=3, ipp=4, mode="evict")
    acc_r, _, _ = _run("pruned_html", pages=3, ipp=4, mode="accumulate")
    assert acc_r.reward == 1.0
    assert evict_r.reward < acc_r.reward  # only ~last page survives eviction
    assert evict_r.metrics["name_recall"] < 0.5  # ~1 of 3 pages


def test_adaptive_escalation_is_load_bearing():
    # adaptive hides items behind a stub. Escalating agent reads them; the identical agent that
    # refuses to escalate floors like flat -> proves the request_text escape hatch carries the win.
    esc_r, _, _ = _run("adaptive", pages=1, ipp=5, escalate=True)
    noesc_r, _, _ = _run("adaptive", pages=1, ipp=5, escalate=False)
    assert esc_r.reward == 1.0
    assert noesc_r.reward == 0.0
    assert esc_r.reward > noesc_r.reward


def test_accumulate_context_contains_prior_pages_and_bills_more():
    _, evict_ep, evict_client = _run("pruned_html", pages=3, ipp=4, mode="evict")
    _, acc_ep, acc_client = _run("pruned_html", pages=3, ipp=4, mode="accumulate")

    def alltext(c):
        return "\n".join(getattr(b, "text", "") for call in c.calls for b in call["blocks"])

    # The accumulated context literally carries the earlier pages; evict never does.
    assert "EARLIER PAGES" in alltext(acc_client) and "EARLIER PAGES" not in alltext(evict_client)
    # And re-sending them costs strictly more real input tokens.
    assert sum(s.input_tokens for s in acc_ep.steps) > sum(s.input_tokens for s in evict_ep.steps)


# -- edge / resilience -------------------------------------------------------


def test_stuck_agent_hits_max_steps_and_scores_zero():
    # An agent that never advances and never submits must terminate cleanly at max_steps.
    src = _src(pages=3)
    task = src.tasks()[0]
    client = MockClient(responder=lambda **_: json.dumps({"action": "click", "ref": "zzz"}))
    ep = evaluate_live(src, task, "flat_elements", client, TokenCounter(), max_steps=5)
    assert len(ep.steps) == 5  # bounded, no infinite loop
    assert src.score(task, ep).reward == 0.0  # never submitted an answer


def test_score_dedups_and_normalizes_names():
    from modalitybench.tasks.base import Episode

    src = _src(pages=2, ipp=3)
    task = src.tasks()[0]
    items = _catalog(0, 6)
    ep = Episode(task_id=task.task_id, strategy="x", model="m")
    # Same item twice (should count once) + one with messy case/whitespace (should still match).
    messy = {"name": f"  {items[1]['name'].upper()}  ", "price": items[1]["price"]}
    ep.meta["answer"] = json.dumps([items[0], items[0], messy])
    r = src.score(task, ep)
    assert r.metrics["exact_recall"] == round(2 / 6, 4)  # two distinct items matched


def test_catalog_varies_by_seed():
    assert _catalog(1, 10) == _catalog(1, 10)
    assert _catalog(1, 10) != _catalog(2, 10)
