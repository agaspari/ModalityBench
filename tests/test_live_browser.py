"""Real Playwright browser tests against a local page.

These validate the live stack that MiniWoB would exercise — JS DOM capture (refs, bboxes,
screenshot), ref->selector resolution, the ActionExecutor, and evaluate_live — without
needing the MiniWoB dataset (MINIWOB_URL). Skipped automatically if Chromium isn't installed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from modalitybench.agents.actions import Action
from modalitybench.agents.executor import ActionExecutor
from modalitybench.agents.loop import evaluate_live
from modalitybench.agents.model_client import MockClient
from modalitybench.metrics.tokens import TokenCounter
from modalitybench.observations.dom_capture import graph_from_page
from modalitybench.observations.serializers._common import build_registry
from modalitybench.tasks.base import LiveOutcome, Task, TaskResult

HTML = """<!doctype html><html><body><main aria-label='demo'>
<h1>Order</h1>
<button id='go' onclick="document.getElementById('ok').style.display='block'">Submit order</button>
<div id='ok' style='display:none'>SUCCESS</div>
</main></body></html>"""


@pytest.fixture
def page():
    sync_api = pytest.importorskip("playwright.sync_api")
    try:
        pw = sync_api.sync_playwright().start()
        browser = pw.chromium.launch(headless=True)
    except Exception as exc:  # browser binary missing / can't launch
        pytest.skip(f"Chromium unavailable: {exc}")
    pg = browser.new_page()
    yield pg
    browser.close()
    pw.stop()


def test_live_capture_has_refs_bboxes_screenshot(page):
    page.set_content(HTML)
    graph = graph_from_page(page)
    refs = graph.by_ref()
    assert refs, "expected at least one captured ref"
    btn = next(n for n in refs.values() if n.tag == "button")
    assert btn.bbox and btn.bbox[2] > 0 and btn.bbox[3] > 0
    assert btn.locator and btn.locator["type"] == "selector"
    assert graph.meta.get("screenshot"), "expected a screenshot on live capture"


def test_live_executor_clicks_by_ref(page):
    page.set_content(HTML)
    graph = graph_from_page(page, screenshot=False)
    btn = next(n for n in graph.by_ref().values() if n.tag == "button")
    ActionExecutor(page).execute(Action(kind="click", ref=btn.ref), build_registry(graph))
    assert page.locator("#ok").is_visible()


class _FakeLiveSource:
    name = "fake"
    is_live = True

    def __init__(self, page, goal):
        self._page = page
        self._goal = goal

    def reset(self, task):
        return SimpleNamespace(page=self._page, goal=self._goal)

    def apply(self, handle, action, registry, graph):
        try:
            ActionExecutor(handle.page).execute(action, registry)
        except Exception as exc:
            return LiveOutcome(reward=0.0, terminated=False, info=str(exc))
        ok = handle.page.locator("#ok").is_visible()
        return LiveOutcome(reward=1.0 if ok else 0.0, terminated=ok, info="")

    def close(self, handle):
        pass

    def score(self, task, episode):
        r = float(episode.meta.get("reward", 0.0))
        return TaskResult(success=r > 0.5, reward=r)


def test_evaluate_live_end_to_end(page):
    page.set_content(HTML)
    graph = graph_from_page(page, screenshot=False)
    btn = next(n for n in graph.by_ref().values() if n.tag == "button")
    mock = MockClient(
        responses=[f'{{"action": "click", "ref": "{btn.ref}"}}', '{"action": "done"}']
    )
    source = _FakeLiveSource(page, "submit the order")
    task = Task(task_id="demo", goal="submit the order", source="fake")
    episode = evaluate_live(source, task, "flat_elements", mock, TokenCounter(None), max_steps=5)
    result = source.score(task, episode)
    assert result.success
    assert episode.meta["reward"] == 1.0
    assert episode.steps[0].action_kind == "click"
