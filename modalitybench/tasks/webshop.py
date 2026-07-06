"""WebShop live task source (Yao et al., NeurIPS 2022).

WebShop is a simulated e-commerce benchmark: the agent gets an instruction
("I want a machine-washable blue rug under $50"), then searches, browses results, opens a
product, selects options, and clicks *Buy Now*. Reward in ``[0, 1]`` measures attribute /
option / price match to the goal; success is a full ``1.0``.

Unlike MiniWoB (which flows actions through BrowserGym's ``env.step``), WebShop is a plain
Flask/HTML app. We run that server locally and point Playwright straight at it, so
ModalityBench's own capture + serializers + :class:`~modalitybench.agents.executor.ActionExecutor`
drive it unchanged — no BrowserGym, no WebShop ``WebAgentSiteEnv``.

**Setup:** WebShop's data + index build is heavier than MiniWoB (see ``configs/webshop-smoke.yaml``).
Run its Flask app, then set ``WEBSHOP_URL`` at its base URL, e.g.::

    export WEBSHOP_URL='http://127.0.0.1:3000'

and install the browser extra: ``pip install 'modalitybench[browser]' && playwright install chromium``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from modalitybench.agents.actions import Action
from modalitybench.observations.base import PageGraph, RefRegistry
from modalitybench.tasks.base import Episode, LiveOutcome, Task, TaskResult

# Success in WebShop is an exact reward of 1.0; allow for float noise.
_SUCCESS_EPS = 1e-3


@dataclass
class _Handle:
    pw: Any
    browser: Any
    page: Any
    goal: str
    session_id: str
    reward: float = 0.0
    terminated: bool = False


class WebShopSource:
    name = "webshop"
    is_live = True

    def __init__(
        self,
        *,
        session_ids: list[str] | None = None,
        n: int = 5,
        session_prefix: str = "fixed_",
        headless: bool = True,
        limit: int | None = None,
    ) -> None:
        # Either an explicit list of session ids, or ``n`` derived as ``{prefix}{i}``.
        # With the reference WebShop server, ``fixed_<i>`` selects goal ``i`` deterministically.
        self.session_ids = session_ids
        self.n = n
        self.session_prefix = session_prefix
        self.headless = headless
        self.limit = limit

    # -- TaskSource API ------------------------------------------------------

    def tasks(self) -> list[Task]:
        ids = self.session_ids or [f"{self.session_prefix}{i}" for i in range(self.n)]
        out = [
            Task(
                task_id=f"webshop#{sid}",
                goal="",  # filled at reset from the landing page instruction
                source=self.name,
                meta={"session_id": sid},
            )
            for sid in ids
        ]
        return out[: self.limit] if self.limit else out

    def reset(self, task: Task) -> _Handle:
        base = os.environ.get("WEBSHOP_URL")
        if not base:
            raise RuntimeError(
                "WEBSHOP_URL is not set. Point it at a running WebShop Flask server, e.g. "
                "WEBSHOP_URL='http://127.0.0.1:3000'. See configs/webshop-smoke.yaml for setup."
            )
        from playwright.sync_api import sync_playwright

        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=self.headless)
        page = browser.new_page()
        session_id = str(task.meta["session_id"])
        page.goto(f"{base.rstrip('/')}/{session_id}")
        _settle(page)
        goal = task.goal or parse_instruction(_body_text(page))
        return _Handle(
            pw=pw, browser=browser, page=page, goal=goal, session_id=session_id
        )

    def apply(
        self, handle: _Handle, action: Action, registry: RefRegistry, graph: PageGraph
    ) -> LiveOutcome:
        from modalitybench.agents.executor import ActionExecutor, ExecutionError

        try:
            ActionExecutor(handle.page).execute(action, registry)
        except ExecutionError as exc:
            # Unresolved / non-executable ref: no-op, keep going (non-terminal).
            return LiveOutcome(reward=handle.reward, terminated=False, info=str(exc))
        _settle(handle.page)

        done, reward = parse_reward(_page_url(handle.page), _body_text(handle.page))
        if done:
            handle.reward = reward
            handle.terminated = True
        return LiveOutcome(reward=handle.reward, terminated=handle.terminated, info="")

    def close(self, handle: _Handle) -> None:
        for obj, meth in ((handle.browser, "close"), (handle.pw, "stop")):
            try:
                if obj is not None:
                    getattr(obj, meth)()
            except Exception:
                pass

    def score(self, task: Task, episode: Episode) -> TaskResult:
        reward = float(episode.meta.get("reward", 0.0))
        return TaskResult(
            success=reward >= 1.0 - _SUCCESS_EPS,
            reward=reward,
            metrics={"reward": reward, "n_steps": float(len(episode.steps))},
        )


# ---------------------------------------------------------------------------
# Pure parsing helpers (unit-tested without a browser)
# ---------------------------------------------------------------------------

_INSTRUCTION_RE = re.compile(r"instruction:\s*(.+)", re.IGNORECASE)
# WebShop's done page: "Your score (min 0.0, max 1.0): 0.75".
_SCORE_RE = re.compile(r"score\s*\(min[^)]*\)\s*[:=]?\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_REWARD_RE = re.compile(r"reward\s*[:=]?\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def parse_instruction(text: str) -> str:
    """Pull the goal instruction out of a WebShop landing/search page's visible text.

    Prefers the ``Instruction: ...`` line WebShop renders; falls back to the first
    non-empty line so a differently-themed server still yields *something* usable.
    """
    m = _INSTRUCTION_RE.search(text)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def parse_reward(url: str, text: str) -> tuple[bool, float]:
    """Detect the terminal "done" page and extract its reward, clamped to ``[0, 1]``.

    Returns ``(is_done, reward)``. Done is signalled by the ``/done`` route or the
    "Thank you for shopping" banner; reward is read from the score line (or a bare
    ``reward: N``), defaulting to ``0.0`` when the page is done but unparsable.
    """
    low = text.lower()
    done = "/done" in url.lower() or "thank you for shopping" in low
    if not done:
        return False, 0.0
    m = _SCORE_RE.search(text) or _REWARD_RE.search(text)
    reward = float(m.group(1)) if m else 0.0
    return True, max(0.0, min(1.0, reward))


# ---------------------------------------------------------------------------
# Small Playwright shims (guarded so they never crash the loop)
# ---------------------------------------------------------------------------


def _settle(page: Any) -> None:
    try:
        page.wait_for_load_state("networkidle", timeout=5000)
    except Exception:
        pass


def _body_text(page: Any) -> str:
    try:
        return page.inner_text("body")
    except Exception:
        return ""


def _page_url(page: Any) -> str:
    try:
        return str(page.url)
    except Exception:
        return ""
