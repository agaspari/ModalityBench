"""Live-harness logic tests that don't need a browser: action translation + executor."""

from __future__ import annotations

import pytest

from modalitybench.agents.actions import Action
from modalitybench.agents.executor import ActionExecutor, ExecutionError
from modalitybench.observations.base import PageGraph, PageNode, RefRegistry
from modalitybench.tasks.miniwob import action_to_code


def _graph_with_button(bbox=(10.0, 20.0, 40.0, 10.0)) -> PageGraph:
    node = PageNode(tag="button", ref="e1", role="button", name="Go", bbox=bbox)
    return PageGraph(root=PageNode(tag="body", children=[node]))


# -- action_to_code (BrowserGym translation) --------------------------------


def test_action_to_code_click_uses_bbox_center():
    g = _graph_with_button()
    assert action_to_code(Action(kind="click", ref="e1"), g) == "mouse_click(30, 25)"


def test_action_to_code_type():
    g = _graph_with_button()
    code = action_to_code(Action(kind="type", ref="e1", text="hi"), g)
    assert code == "mouse_click(30, 25)\nkeyboard_type('hi')"


def test_action_to_code_nav_actions():
    g = _graph_with_button()
    assert action_to_code(Action(kind="scroll", value="down"), g) == "scroll(0, 200)"
    assert action_to_code(Action(kind="scroll", value="up"), g) == "scroll(0, -200)"
    assert action_to_code(Action(kind="press", key="Enter"), g) == "keyboard_press('Enter')"
    assert action_to_code(Action(kind="goto", url="/x"), g) == "goto('/x')"


def test_action_to_code_none_without_bbox():
    g = PageGraph(root=PageNode(tag="body", children=[PageNode(tag="button", ref="e1")]))
    assert action_to_code(Action(kind="click", ref="e1"), g) is None
    assert action_to_code(Action(kind="click", ref="e999"), g) is None


# -- executor (mock page) ---------------------------------------------------


class _Locator:
    def __init__(self, log):
        self.log = log

    def click(self):
        self.log.append("click")

    def fill(self, text):
        self.log.append(f"fill:{text}")

    def select_option(self, label=None):
        self.log.append(f"select:{label}")


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


class _Page:
    def __init__(self):
        self.log = []
        self.last_selector = None
        self.mouse = _Mouse(self.log)
        self.keyboard = _Keyboard(self.log)

    def locator(self, selector):
        self.last_selector = selector
        return _Locator(self.log)

    def goto(self, url):
        self.log.append(f"goto:{url}")


def _registry():
    reg = RefRegistry()
    reg.add("e1", {"type": "selector", "value": '[data-mb-ref="e1"]'})
    reg.add("e2", {"type": "backend_id", "value": "42"})
    return reg


def test_executor_click():
    page = _Page()
    ex = ActionExecutor(page)
    ex.execute(Action(kind="click", ref="e1"), _registry())
    assert page.last_selector == '[data-mb-ref="e1"]'
    assert "click" in page.log


def test_executor_type_and_nav():
    page = _Page()
    ex = ActionExecutor(page)
    ex.execute(Action(kind="type", ref="e1", text="hello"), _registry())
    ex.execute(Action(kind="scroll", value="down"), _registry())
    ex.execute(Action(kind="press", key="Tab"), _registry())
    assert "fill:hello" in page.log
    assert "wheel:500" in page.log
    assert "press:Tab" in page.log


def test_executor_done_is_terminal():
    ex = ActionExecutor(_Page())
    res = ex.execute(Action(kind="done", answer="42"), _registry())
    assert res.done and res.answer == "42"


def test_executor_rejects_unresolved_and_offline_refs():
    ex = ActionExecutor(_Page())
    with pytest.raises(ExecutionError):
        ex.execute(Action(kind="click", ref="nope"), _registry())
    # backend_id locators are not executable on a live page.
    with pytest.raises(ExecutionError):
        ex.execute(Action(kind="click", ref="e2"), _registry())
