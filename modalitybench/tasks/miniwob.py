"""MiniWoB++ live task source via BrowserGym.

BrowserGym provides the task setup, the Playwright page, and the reward; ModalityBench takes
the observations (its own capture + serializers) and executes actions by translating
ref-based actions into BrowserGym coordinate actions using the bounding boxes from our
capture — so reward flows through ``env.step`` while the observation pipeline stays ours.

**Setup:** BrowserGym does not bundle the MiniWoB HTML. Set ``MINIWOB_URL`` to a served copy
of ``miniwob-plusplus/miniwob/html/miniwob/`` (see the README) before running, and install
the browser extra: ``pip install 'modalitybench[browser]' && playwright install chromium``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from modalitybench.agents.actions import Action
from modalitybench.observations.base import PageGraph, RefRegistry
from modalitybench.tasks.base import Episode, LiveOutcome, Task, TaskResult

_DEFAULT_TASKS = [
    "click-test",
    "click-button",
    "click-link",
    "focus-text",
    "enter-text",
]


@dataclass
class _Handle:
    env: Any
    page: Any
    goal: str


class MiniWobSource:
    name = "miniwob"
    is_live = True

    def __init__(
        self,
        *,
        tasks: list[str] | None = None,
        seeds: list[int] | None = None,
        headless: bool = True,
        limit: int | None = None,
    ) -> None:
        self.task_names = tasks or _DEFAULT_TASKS
        self.seeds = seeds or [0]
        self.headless = headless
        self.limit = limit

    # -- TaskSource API ------------------------------------------------------

    def tasks(self) -> list[Task]:
        out: list[Task] = []
        for name in self.task_names:
            for seed in self.seeds:
                out.append(
                    Task(
                        task_id=f"{name}#{seed}",
                        goal="",  # filled at reset from the env
                        source=self.name,
                        seed=seed,
                        meta={"miniwob": name},
                    )
                )
        return out[: self.limit] if self.limit else out

    def reset(self, task: Task) -> _Handle:
        if not os.environ.get("MINIWOB_URL"):
            raise RuntimeError(
                "MINIWOB_URL is not set. Point it at a served copy of the MiniWoB HTML, e.g. "
                "MINIWOB_URL='file:///path/to/miniwob-plusplus/miniwob/html/miniwob/'. "
                "See the README (MiniWoB setup)."
            )
        import browsergym.miniwob  # noqa: F401  (registers the envs)
        import gymnasium as gym
        from browsergym.core.action.highlevel import HighLevelActionSet

        mapping = HighLevelActionSet(
            subsets=["coord", "nav"], strict=False, multiaction=True
        ).to_python_code
        env = gym.make(
            f"browsergym/miniwob.{task.meta['miniwob']}",
            action_mapping=mapping,
            headless=self.headless,
            wait_for_user_message=False,
        )
        obs, _info = env.reset(seed=task.seed)
        page = env.unwrapped.page
        return _Handle(env=env, page=page, goal=_goal_text(obs))

    def capture(self, handle: _Handle, *, screenshot: bool = False) -> PageGraph:
        from modalitybench.observations.dom_capture import graph_from_page

        return graph_from_page(handle.page, screenshot=screenshot)

    def apply(
        self, handle: _Handle, action: Action, registry: RefRegistry, graph: PageGraph
    ) -> LiveOutcome:
        # BrowserGym's coord actions treat inputs as *screenshot* coordinates and divide by
        # ``_bgym_scale_factor`` (map_coordinates); our bboxes are page coordinates, so we
        # pre-multiply to cancel it out. Factor is read live (defaults to 1.0 if absent).
        scale = float(getattr(handle.page, "_bgym_scale_factor", 1.0) or 1.0)
        code = action_to_code(action, graph, scale=scale)
        if code is None:
            return LiveOutcome(reward=0.0, terminated=False, info="no-op (unresolved ref)")
        _obs, reward, terminated, truncated, info = handle.env.step(code)
        return LiveOutcome(
            reward=float(reward),
            terminated=bool(terminated or truncated),
            info=str(info.get("last_action_error", "")),
        )

    def close(self, handle: _Handle) -> None:
        try:
            handle.env.close()
        except Exception:
            pass

    def score(self, task: Task, episode: Episode) -> TaskResult:
        reward = float(episode.meta.get("reward", 0.0))
        return TaskResult(
            success=reward > 0.5,
            reward=reward,
            metrics={"reward": reward, "n_steps": float(len(episode.steps))},
        )


# ---------------------------------------------------------------------------
# Action translation (pure — unit-tested without a browser)
# ---------------------------------------------------------------------------


def action_to_code(action: Action, graph: PageGraph, *, scale: float = 1.0) -> str | None:
    """Translate a ref-based action into a BrowserGym coordinate/nav action string.

    ``scale`` pre-multiplies click coordinates to cancel BrowserGym's ``map_coordinates``
    down-scaling (its ``_bgym_scale_factor``); pass 1.0 for raw page coordinates. Returns
    ``None`` when the action can't be grounded (unresolved ref / missing bbox).
    """
    kind = action.kind
    if kind == "scroll":
        dy = -200 if (action.value or "down").lower() == "up" else 200
        return f"scroll(0, {dy})"
    if kind == "press":
        return f"keyboard_press({(action.key or 'Enter')!r})"
    if kind == "goto":
        return f"goto({(action.url or '')!r})"

    center = _center(action.ref, graph)
    if center is None:
        return None
    cx, cy = _fmt(center[0] * scale), _fmt(center[1] * scale)
    if kind == "click" or kind == "select":
        return f"mouse_click({cx}, {cy})"
    if kind == "type":
        text = action.text or ""
        return f"mouse_click({cx}, {cy})\nkeyboard_type({text!r})"
    return None


def _fmt(v: float) -> str:
    """Compact coordinate: drop the trailing ``.0`` for whole numbers (e.g. scale=1.0)."""
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"


def _center(ref: str | None, graph: PageGraph) -> tuple[int, int] | None:
    if not ref:
        return None
    node = graph.by_ref().get(ref)
    if node is None or not node.bbox:
        return None
    x, y, w, h = node.bbox
    return int(x + w / 2), int(y + h / 2)


def _goal_text(obs: dict[str, Any]) -> str:
    goal_obj = obs.get("goal_object") or []
    parts = [p.get("text", "") for p in goal_obj if p.get("type") == "text"]
    text = " ".join(t for t in parts if t).strip()
    return text or str(obs.get("goal", "")).strip()
