"""Task/episode data types and the TaskSource protocol.

A ``TaskSource`` yields ``Task`` objects and knows how to score an ``Episode``. Live sources
(MiniWoB++) also expose the Playwright page so the harness can capture observations and
execute actions; offline sources (Mind2Web) provide cached ``PageGraph`` snapshots and a
per-step ground truth instead of a live loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from modalitybench.observations.base import PageGraph


@dataclass
class Task:
    """A single benchmark task."""

    task_id: str
    goal: str  # natural-language instruction shown to the agent
    source: str = ""  # task-source name, e.g. "miniwob" | "mind2web_offline"
    seed: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepRecord:
    """One agent step: observation size, the action taken, model usage, and outcome."""

    step_index: int
    action_kind: str | None = None
    action: dict[str, Any] | None = None
    action_raw: str = ""
    # Observation size stats (the whole point of the benchmark).
    obs_tokens: int = 0
    obs_bytes: int = 0
    obs_element_count: int = 0
    obs_has_image: bool = False
    # Model usage for this step.
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    latency_s: float = 0.0
    # Per-step correctness (offline element-selection eval populates this).
    correct: bool | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """The full trajectory for one (strategy, model, task) cell."""

    task_id: str
    strategy: str
    model: str
    steps: list[StepRecord] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveOutcome:
    """Result of applying one action to a live task environment."""

    reward: float = 0.0
    terminated: bool = False
    info: str = ""


@dataclass
class TaskResult:
    """Scored outcome of an episode, produced by ``TaskSource.score``."""

    success: bool
    reward: float = 0.0
    # Extra per-source metrics (e.g. element_accuracy, action_f1 for Mind2Web).
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@runtime_checkable
class TaskSource(Protocol):
    """A source of tasks plus its scoring logic.

    Two shapes are supported and distinguished by ``is_live``:

    * **Live** (``is_live=True``): ``reset(task)`` provisions a page and returns a handle
      the harness uses to capture graphs and execute actions; ``score`` reads terminal
      reward. Implemented in :mod:`modalitybench.tasks.miniwob`.
    * **Offline** (``is_live=False``): ``snapshots(task)`` yields ``(PageGraph, ground_truth)``
      per step for element-selection scoring, no browser. Implemented in
      :mod:`modalitybench.tasks.mind2web_offline`.
    """

    name: str
    is_live: bool

    def tasks(self) -> list[Task]:
        ...

    def score(self, task: Task, episode: Episode) -> TaskResult:
        ...


@runtime_checkable
class OfflineTaskSource(TaskSource, Protocol):
    def snapshots(self, task: Task) -> list[tuple[PageGraph, dict[str, Any]]]:
        """Yield ``(graph, ground_truth)`` for each step of the task."""
        ...
