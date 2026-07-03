"""Offline Mind2Web element-selection task source.

Each task is a multi-step web task; each step provides a cleaned-HTML snapshot and a
ground-truth target element (by ``backend_node_id``) plus the operation. Scoring is
teacher-forced per-step element accuracy and operation F1 — no browser, fully reproducible.

Two backends:
* ``local`` — a bundled JSON sample (default), so the loader and scoring run with no network.
* ``hf`` — the ``osunlp/Mind2Web`` dataset via ``datasets`` (requires network + the ``data``
  extra). Field mapping matches the released dataset.
"""

from __future__ import annotations

import json
import re
from importlib.resources import files
from typing import Any

from modalitybench.observations.base import PageGraph
from modalitybench.observations.dom_capture import graph_from_html
from modalitybench.tasks.base import Episode, Task, TaskResult

_ID_ATTR = "backend_node_id"
_OP_TO_KIND = {"CLICK": "click", "TYPE": "type", "SELECT": "select"}


class Mind2WebOffline:
    name = "mind2web_offline"
    is_live = False

    def __init__(
        self,
        *,
        backend: str = "local",
        split: str = "test_domain",
        limit: int | None = None,
        task_ids: list[str] | None = None,
        hf_name: str = "osunlp/Mind2Web",
    ) -> None:
        self.backend = backend
        self.split = split
        self.limit = limit
        self.task_ids = set(task_ids) if task_ids else None
        self.hf_name = hf_name
        self._records: list[dict[str, Any]] = self._load()

    # -- loading -------------------------------------------------------------

    def _load(self) -> list[dict[str, Any]]:
        if self.backend == "local":
            raw = json.loads(
                files("modalitybench.data").joinpath("mind2web_sample.json").read_text(
                    encoding="utf-8"
                )
            )
        elif self.backend == "hf":
            raw = self._load_hf()
        else:
            raise ValueError(f"unknown mind2web backend {self.backend!r}")

        records = []
        for r in raw:
            if self.task_ids and r["annotation_id"] not in self.task_ids:
                continue
            records.append(r)
            if self.limit and len(records) >= self.limit:
                break
        return records

    def _load_hf(self) -> list[dict[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "the 'hf' backend needs the data extra: pip install 'modalitybench[data]'"
            ) from exc
        ds = load_dataset(self.hf_name, split=self.split)
        out = []
        for row in ds:
            out.append(
                {
                    "annotation_id": row["annotation_id"],
                    "confirmed_task": row["confirmed_task"],
                    "website": row.get("website", ""),
                    "domain": row.get("domain", ""),
                    "actions": [_normalise_hf_action(a) for a in row["actions"]],
                }
            )
        return out

    # -- TaskSource API ------------------------------------------------------

    def tasks(self) -> list[Task]:
        return [
            Task(
                task_id=r["annotation_id"],
                goal=r["confirmed_task"],
                source=self.name,
                meta={"website": r.get("website", ""), "domain": r.get("domain", "")},
            )
            for r in self._records
        ]

    def snapshots(self, task: Task) -> list[tuple[PageGraph, dict[str, Any]]]:
        record = next(r for r in self._records if r["annotation_id"] == task.task_id)
        out: list[tuple[PageGraph, dict[str, Any]]] = []
        for step in record["actions"]:
            graph = graph_from_html(
                step["cleaned_html"], url=task.meta.get("website", ""), id_attr=_ID_ATTR,
                source="mind2web",
            )
            op = step["operation"]
            gt = {
                "backend_ids": {str(c["backend_node_id"]) for c in step["pos_candidates"]},
                "op": op["op"],
                "kind": _OP_TO_KIND.get(op["op"], "click"),
                "value": op.get("value", "") or "",
                "action_repr": _repr_action(op, step["pos_candidates"]),
            }
            out.append((graph, gt))
        return out

    def score(self, task: Task, episode: Episode) -> TaskResult:
        """Aggregate per-step records into element accuracy, op F1, and task success."""
        if not episode.steps:
            return TaskResult(success=False, error="no steps recorded")
        ele = [1.0 if s.correct else 0.0 for s in episode.steps]
        op_hits = [1.0 if s.meta.get("op_correct") else 0.0 for s in episode.steps]
        f1s = [float(s.meta.get("action_f1", 0.0)) for s in episode.steps]
        step_success = [
            1.0 if (s.correct and s.meta.get("op_correct")) else 0.0 for s in episode.steps
        ]
        task_success = all(s == 1.0 for s in step_success)
        n = len(episode.steps)
        return TaskResult(
            success=task_success,
            reward=sum(step_success) / n,
            metrics={
                "element_accuracy": round(sum(ele) / n, 4),
                "operation_hit": round(sum(op_hits) / n, 4),
                "action_f1": round(sum(f1s) / n, 4),
                "step_success_rate": round(sum(step_success) / n, 4),
                "n_steps": float(n),
            },
        )


# ---------------------------------------------------------------------------
# Scoring helpers (used by the offline evaluator)
# ---------------------------------------------------------------------------


def action_f1(pred_kind: str, pred_value: str, gt_kind: str, gt_value: str) -> float:
    """Token-level F1 over the operation label + value, à la Mind2Web operation F1."""
    pred_tokens = _op_tokens(pred_kind, pred_value)
    gt_tokens = _op_tokens(gt_kind, gt_value)
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = _multiset_overlap(pred_tokens, gt_tokens)
    if common == 0:
        return 0.0
    prec = common / len(pred_tokens)
    rec = common / len(gt_tokens)
    return 2 * prec * rec / (prec + rec)


def _op_tokens(kind: str, value: str) -> list[str]:
    toks = [kind.lower()]
    toks += re.findall(r"\w+", value.lower())
    return toks


def _multiset_overlap(a: list[str], b: list[str]) -> int:
    from collections import Counter

    ca, cb = Counter(a), Counter(b)
    return sum((ca & cb).values())


def _repr_action(op: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    tag = candidates[0].get("tag", "element") if candidates else "element"
    v = op.get("value", "")
    return f"{op['op']} <{tag}>" + (f' "{v}"' if v else "")


def _normalise_hf_action(a: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_uid": a.get("action_uid", ""),
        "operation": {
            "op": a["operation"]["op"],
            "value": a["operation"].get("value", ""),
        },
        "pos_candidates": [
            {"backend_node_id": str(c["backend_node_id"]), "tag": c.get("tag", "")}
            for c in a["pos_candidates"]
        ],
        "cleaned_html": a["cleaned_html"],
    }
