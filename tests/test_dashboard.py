"""Phase 5 tests: dashboard HTML builder and CSV/JSON exporters."""

from __future__ import annotations

import csv
import json

import pytest

from modalitybench.dashboard.build import (
    _oracle_points,
    build_dashboard,
    export_results,
)


def _step(strategy, task, idx, *, correct, in_tok, model="mock"):
    return {
        "strategy": strategy, "model": model, "task_id": task, "step_index": idx,
        "correct": correct, "input_tokens": in_tok,
    }


def test_oracle_picks_cheapest_correct_representation():
    # 1 task, 2 steps, 2 strategies. step0: both correct -> pick cheap (10).
    # step1: only rich correct -> must pay rich (100). quality=2/2, tokens=110.
    steps = [
        _step("cheap", "t1", 0, correct=True, in_tok=10),
        _step("rich", "t1", 0, correct=True, in_tok=100),
        _step("cheap", "t1", 1, correct=False, in_tok=10),
        _step("rich", "t1", 1, correct=True, in_tok=100),
    ]
    pts = _oracle_points(steps)
    assert len(pts) == 1
    assert pts[0].quality == 1.0
    assert pts[0].input_tokens == 110.0


def test_oracle_counts_unsolvable_step_and_pays_cheapest():
    steps = [
        _step("cheap", "t1", 0, correct=True, in_tok=10),
        _step("rich", "t1", 0, correct=True, in_tok=100),
        _step("cheap", "t1", 1, correct=False, in_tok=5),   # neither works this step
        _step("rich", "t1", 1, correct=False, in_tok=50),
    ]
    pts = _oracle_points(steps)
    assert pts[0].quality == 0.5             # 1 of 2 steps solvable
    assert pts[0].input_tokens == 10 + 5     # step1 unsolvable -> pay the cheapest


def test_oracle_empty_for_live_runs_without_correctness():
    steps = [_step("s", "t1", 0, correct=None, in_tok=10)]
    assert _oracle_points(steps) == []


def _episode(strategy: str, task: str, *, acc: float, tokens: int, cost: float) -> dict:
    """A minimal episodes.jsonl row shaped like the recorder emits."""
    return {
        "run_id": "r",
        "strategy": strategy,
        "model": "mock",
        "task_id": task,
        "success": acc >= 0.99,
        "reward": acc,
        "metrics": {"element_accuracy": acc, "operation_hit": acc},
        "n_steps": 2,
        "obs_tokens_total": tokens,
        "usage": {"input_tokens": tokens * 3, "output_tokens": 20},
        "latency_s_total": 1.5,
        "cost": {"total_cost": cost},
    }


def _write_run(base, run_id: str, rows: list[dict]):
    run_dir = base / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "episodes.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return run_dir


@pytest.fixture
def run_dir(tmp_path):
    rows = [
        _episode("afus", "t1", acc=1.0, tokens=100, cost=0.01),
        _episode("afus", "t2", acc=0.5, tokens=120, cost=0.02),
        _episode("fct", "t1", acc=0.5, tokens=300, cost=0.03),
        _episode("fct", "t2", acc=0.5, tokens=320, cost=0.04),
    ]
    _write_run(tmp_path, "run-x", rows)
    return tmp_path


def test_build_dashboard_self_contained(run_dir):
    out = build_dashboard("run-x", results_dir=run_dir)
    html = out.read_text(encoding="utf-8")

    # Two strategy cells (afus, fct) aggregated from four episodes.
    assert "2 strategy cells, 4 episodes" in html
    # Plotly is inlined, not pulled from a CDN via an external <script src>
    # (strict-CSP / offline safe). The bundle itself contains a cdn.plot.ly
    # string literal, so we check specifically for an external script load.
    assert 'src="https://cdn.plot.ly' not in html
    assert "plotly" in html.lower()
    # A domain metric present in the rows becomes the quality axis.
    assert "element_accuracy" in html
    # Aggregated table shows both strategies.
    assert "<table>" in html and "afus" in html and "fct" in html


def test_build_dashboard_custom_out(run_dir, tmp_path):
    dest = tmp_path / "custom" / "dash.html"
    out = build_dashboard("run-x", results_dir=run_dir, out=dest)
    assert out == dest and dest.exists()


def test_build_dashboard_missing_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_dashboard("nope", results_dir=tmp_path)


def test_export_json(run_dir):
    out = export_results("run-x", fmt="json", results_dir=run_dir)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == 4
    row = data[0]
    # Nested usage/cost/metrics are flattened with prefixes.
    assert "usage_input_tokens" in row
    assert "cost_total_cost" in row
    assert "metric_element_accuracy" in row
    assert row["strategy"] == "afus"


def test_export_csv(run_dir):
    out = export_results("run-x", fmt="csv", results_dir=run_dir)
    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert "usage_input_tokens" in rows[0]
    assert "metric_element_accuracy" in rows[0]


def test_export_rejects_unknown_format(run_dir):
    with pytest.raises(ValueError):
        export_results("run-x", fmt="xml", results_dir=run_dir)
