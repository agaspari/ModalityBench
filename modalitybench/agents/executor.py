"""Action executor: resolve a ref via the registry, then act on a Playwright page.

Kept free of Playwright imports so it can be unit-tested with a stand-in page object that
records calls. Only ``selector``-type locators are actionable (live pages); ``backend_id``
locators (offline snapshots) are not executable and raise :class:`ExecutionError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modalitybench.agents.actions import Action
from modalitybench.observations.base import RefRegistry


class ExecutionError(RuntimeError):
    pass


@dataclass
class ExecResult:
    done: bool = False
    answer: str | None = None
    info: str = ""


class ActionExecutor:
    def __init__(self, page: Any) -> None:
        self.page = page

    def execute(self, action: Action, registry: RefRegistry) -> ExecResult:
        kind = action.kind
        if kind == "done":
            return ExecResult(done=True, answer=action.answer)
        if kind == "scroll":
            dy = -500 if (action.value or "down").lower() == "up" else 500
            self.page.mouse.wheel(0, dy)
            return ExecResult(info=f"scrolled {action.value or 'down'}")
        if kind == "press":
            self.page.keyboard.press(action.key or "Enter")
            return ExecResult(info=f"pressed {action.key}")
        if kind == "goto":
            self.page.goto(action.url)
            return ExecResult(info=f"navigated {action.url}")

        # Element-targeted actions need a resolvable selector.
        selector = self._selector(action, registry)
        locator = self.page.locator(selector)
        if kind == "click":
            locator.click()
            return ExecResult(info=f"clicked {action.ref}")
        if kind == "type":
            locator.fill(action.text or "")
            return ExecResult(info=f"typed into {action.ref}")
        if kind == "select":
            locator.select_option(label=action.value)
            return ExecResult(info=f"selected {action.value} in {action.ref}")
        raise ExecutionError(f"unhandled action kind {kind!r}")

    def _selector(self, action: Action, registry: RefRegistry) -> str:
        if not action.ref:
            raise ExecutionError(f"{action.kind} requires a ref")
        loc = registry.resolve(action.ref)
        if loc is None:
            raise ExecutionError(f"unknown ref {action.ref!r}")
        if loc.get("type") != "selector":
            raise ExecutionError(
                f"ref {action.ref!r} is not executable on a live page "
                f"(locator type {loc.get('type')!r})"
            )
        return loc["value"]
