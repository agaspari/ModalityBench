"""WebShop task-source tests: pure parsing + apply/score against a stand-in page.

No real browser and no WEBSHOP_URL / WebShop data needed — the Playwright page is faked so
the ref->selector executor path and terminal-reward detection are exercised deterministically.
"""

from __future__ import annotations

import pytest

from modalitybench.agents.actions import Action
from modalitybench.observations.base import PageGraph, PageNode, RefRegistry
from modalitybench.tasks.base import Episode, Task
from modalitybench.tasks.webshop import (
    WebShopSource,
    _Handle,
    parse_instruction,
    parse_reward,
)


# -- pure parsing -----------------------------------------------------------


def test_parse_instruction_from_labelled_line():
    assert parse_instruction("Instruction: Find a red mug under $20\nSearch") == (
        "Find a red mug under $20"
    )
    # Falls back to the first non-empty line when the "Instruction:" label is absent.
    assert parse_instruction("\n\nGoal text here\nmore") == "Goal text here"
    assert parse_instruction("") == ""


def test_parse_reward_score_line():
    text = "Thank you for shopping with us!\nYour score (min 0.0, max 1.0): 0.75"
    done, reward = parse_reward("http://x/done/s/asin", text)
    assert done and reward == 0.75


def test_parse_reward_detects_done_by_url_and_clamps():
    done, reward = parse_reward("http://x/done/s/a", "reward: 1.5")
    assert done and reward == 1.0  # clamped
    done, reward = parse_reward("http://x/done/s/a", "no number here")
    assert done and reward == 0.0  # done but unparsable


def test_parse_reward_not_done_on_normal_page():
    done, reward = parse_reward("http://x/search_results/s/rug/1", "Results for rug")
    assert not done and reward == 0.0


# -- tasks / score ----------------------------------------------------------


def test_tasks_from_n_and_prefix():
    tasks = WebShopSource(n=3, session_prefix="fixed_").tasks()
    assert [t.meta["session_id"] for t in tasks] == ["fixed_0", "fixed_1", "fixed_2"]
    assert all(t.source == "webshop" and t.goal == "" for t in tasks)


def test_tasks_explicit_ids_and_limit():
    tasks = WebShopSource(session_ids=["a", "b", "c"], limit=2).tasks()
    assert [t.meta["session_id"] for t in tasks] == ["a", "b"]


def test_score_success_requires_full_reward():
    src = WebShopSource()
    task = Task(task_id="webshop#fixed_0", goal="g", source="webshop")
    ep_full = Episode(task_id=task.task_id, strategy="s", model="m", meta={"reward": 1.0})
    ep_partial = Episode(task_id=task.task_id, strategy="s", model="m", meta={"reward": 0.6})
    assert src.score(task, ep_full).success
    assert not src.score(task, ep_partial).success
    assert src.score(task, ep_partial).reward == 0.6


# -- apply against a stand-in page ------------------------------------------


class _Locator:
    def __init__(self, log):
        self.log = log

    def click(self):
        self.log.append("click")

    def fill(self, text):
        self.log.append(f"fill:{text}")


class _Mouse:
    def __init__(self, log):
        self.log = log

    def wheel(self, dx, dy):
        self.log.append(f"wheel:{dy}")


class _Keyboard:
    def __init__(self, log):
        self.log = log

    def press(self, key):
        self.log.append(f"press:{key}")


class _FakePage:
    def __init__(self, url, body):
        self.url = url
        self._body = body
        self.log = []
        self.mouse = _Mouse(self.log)
        self.keyboard = _Keyboard(self.log)

    def locator(self, selector):
        self.log.append(f"locator:{selector}")
        return _Locator(self.log)

    def inner_text(self, selector):
        return self._body

    def wait_for_load_state(self, *a, **k):
        pass


def _registry():
    reg = RefRegistry()
    reg.add("e1", {"type": "selector", "value": '[data-mb-ref="e1"]'})
    return reg


def _handle(url, body):
    page = _FakePage(url, body)
    return _Handle(pw=None, browser=None, page=page, goal="g", session_id="s"), page


def test_apply_click_then_terminal_reward():
    src = WebShopSource()
    handle, page = _handle(
        "http://x/done/s/asin",
        "Thank you for shopping with us! Your score (min 0.0, max 1.0): 0.75",
    )
    out = src.apply(
        handle, Action(kind="click", ref="e1"), _registry(), PageGraph(root=PageNode(tag="body"))
    )
    assert "click" in page.log
    assert out.terminated and out.reward == 0.75


def test_apply_non_terminal_before_buy():
    src = WebShopSource()
    handle, _ = _handle("http://x/search_results/s/rug/1", "Results for rug")
    out = src.apply(
        handle, Action(kind="click", ref="e1"), _registry(), PageGraph(root=PageNode(tag="body"))
    )
    assert not out.terminated and out.reward == 0.0


def test_apply_unresolved_ref_is_non_terminal_noop():
    src = WebShopSource()
    handle, page = _handle("http://x/s", "page")
    out = src.apply(
        handle, Action(kind="click", ref="missing"), _registry(),
        PageGraph(root=PageNode(tag="body")),
    )
    assert not out.terminated and out.reward == 0.0
    assert "click" not in page.log  # never touched the page


def test_reset_requires_webshop_url(monkeypatch):
    monkeypatch.delenv("WEBSHOP_URL", raising=False)
    with pytest.raises(RuntimeError, match="WEBSHOP_URL"):
        WebShopSource().reset(Task(task_id="t", goal="", meta={"session_id": "fixed_0"}))
