"""Results recorder: append-only JSONL under ``results/<run_id>/``.

Files written:

* ``run.json``      — run config + metadata (written once at start).
* ``steps.jsonl``   — one row per agent step (observation size, action, usage).
* ``episodes.jsonl``— one row per (strategy, model, task) cell (success, totals, cost).
* ``completed.jsonl`` — cell ids that finished, enabling resumable runs.

The dashboard reads ``episodes.jsonl`` (and optionally ``steps.jsonl``) directly.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from modalitybench.agents.model_client import Usage
from modalitybench.metrics.cost import cost_for_usage
from modalitybench.tasks.base import Episode, StepRecord, TaskResult


def cell_id(strategy: str, model: str, task_id: str, history_mode: str = "evict") -> str:
    return f"{strategy}::{model}::{history_mode}::{task_id}"


class Recorder:
    def __init__(self, run_id: str, base_dir: str | Path = "results") -> None:
        self.run_id = run_id
        self.dir = Path(base_dir) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._steps_path = self.dir / "steps.jsonl"
        self._episodes_path = self.dir / "episodes.jsonl"
        self._completed_path = self.dir / "completed.jsonl"
        self._completed = self._load_completed()

    # -- lifecycle -----------------------------------------------------------

    def mark_started(self, config: dict[str, Any]) -> None:
        (self.dir / "run.json").write_text(
            json.dumps({"run_id": self.run_id, "config": config}, indent=2),
            encoding="utf-8",
        )

    # -- resumability --------------------------------------------------------

    def _load_completed(self) -> set[str]:
        if not self._completed_path.exists():
            return set()
        done: set[str] = set()
        for line in self._completed_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                done.add(json.loads(line)["cell_id"])
        return done

    def is_completed(
        self, strategy: str, model: str, task_id: str, history_mode: str = "evict"
    ) -> bool:
        return cell_id(strategy, model, task_id, history_mode) in self._completed

    def completed_cells(self) -> set[str]:
        return set(self._completed)

    # -- writing -------------------------------------------------------------

    def record_step(
        self, strategy: str, model: str, task_id: str, step: StepRecord,
        history_mode: str = "evict",
    ) -> None:
        row = {
            "run_id": self.run_id,
            "cell_id": cell_id(strategy, model, task_id, history_mode),
            "strategy": strategy,
            "model": model,
            "history_mode": history_mode,
            "task_id": task_id,
            **dataclasses.asdict(step),
        }
        self._append(self._steps_path, row)

    def record_episode(
        self,
        strategy: str,
        model: str,
        task_id: str,
        episode: Episode,
        result: TaskResult,
        history_mode: str = "evict",
    ) -> None:
        total = _sum_usage(episode)
        cost = cost_for_usage(model, total)
        obs_tokens = sum(s.obs_tokens for s in episode.steps)
        row = {
            "run_id": self.run_id,
            "cell_id": cell_id(strategy, model, task_id, history_mode),
            "strategy": strategy,
            "model": model,
            "history_mode": history_mode,
            "task_id": task_id,
            "success": result.success,
            "reward": result.reward,
            "metrics": result.metrics,
            "error": result.error,
            "n_steps": len(episode.steps),
            "obs_tokens_total": obs_tokens,
            "obs_tokens_mean": round(obs_tokens / len(episode.steps), 1)
            if episode.steps
            else 0,
            "usage": total.as_dict(),
            "latency_s_total": round(sum(s.latency_s for s in episode.steps), 3),
            "cost": cost,
        }
        self._append(self._episodes_path, row)
        cid = cell_id(strategy, model, task_id, history_mode)
        self._append(self._completed_path, {"cell_id": cid})
        self._completed.add(cid)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _append(path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=_json_default) + "\n")


def _sum_usage(episode: Episode) -> Usage:
    total = Usage()
    for s in episode.steps:
        total = total.add(
            Usage(
                input_tokens=s.input_tokens,
                output_tokens=s.output_tokens,
                cache_read_input_tokens=s.cache_read_input_tokens,
                cache_creation_input_tokens=s.cache_creation_input_tokens,
            )
        )
    return total


def _json_default(o: Any) -> Any:
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)
