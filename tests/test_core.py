"""Phase 1 core-primitive tests: actions, tokens, cost, and the recorder round-trip."""

from __future__ import annotations

import json

import pytest

from modalitybench.agents.actions import ActionParseError, parse_action
from modalitybench.agents.model_client import MockClient, Usage
from modalitybench.metrics.cost import cost_for_usage
from modalitybench.metrics.recorder import Recorder, cell_id
from modalitybench.metrics.tokens import TokenCounter, estimate_tokens_local
from modalitybench.observations.base import TextBlock
from modalitybench.tasks.base import Episode, StepRecord, TaskResult


def test_parse_action_flat_form():
    a = parse_action('{"action": "type", "ref": "e4", "text": "hi"}')
    assert a.kind == "type"
    assert a.ref == "e4"
    assert a.text == "hi"


def test_parse_action_single_key_form():
    a = parse_action('{"click": {"ref": "e2"}}')
    assert a.kind == "click"
    assert a.ref == "e2"


def test_parse_action_embedded_in_prose():
    a = parse_action('I will click it. {"action": "click", "ref": "e9"} done.')
    assert a.kind == "click" and a.ref == "e9"


def test_parse_action_rejects_unknown():
    with pytest.raises(ActionParseError):
        parse_action('{"action": "teleport"}')


def test_token_estimate_monotonic():
    short = estimate_tokens_local("hello")
    long = estimate_tokens_local("hello " * 100)
    assert 0 < short < long


def test_token_counter_falls_back_local_without_client():
    tc = TokenCounter(client=None)
    assert tc.mode == "approx"
    n = tc.count_blocks([TextBlock(text="one two three")])
    assert n > 0


def test_cost_for_usage_opus():
    u = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    c = cost_for_usage("claude-opus-4-8", u)
    assert c["input_cost"] == pytest.approx(5.0)
    assert c["output_cost"] == pytest.approx(25.0)
    assert c["total_cost"] == pytest.approx(30.0)
    assert c["priced_as_fallback"] is False


def test_cost_unknown_model_flagged_fallback():
    c = cost_for_usage("some-future-model", Usage(input_tokens=1_000_000))
    assert c["priced_as_fallback"] is True


def test_recorder_roundtrip(tmp_path):
    rec = Recorder("run-x", base_dir=tmp_path)
    rec.mark_started({"hello": "world"})
    episode = Episode(task_id="t1", strategy="afus", model="mock")
    episode.steps.append(
        StepRecord(
            step_index=0,
            action_kind="click",
            action={"ref": "e1"},
            obs_tokens=120,
            input_tokens=200,
            output_tokens=30,
        )
    )
    rec.record_step("afus", "mock", "t1", episode.steps[0])
    rec.record_episode("afus", "mock", "t1", episode, TaskResult(success=True, reward=1.0))

    # Resumability: the cell is now marked complete.
    assert rec.is_completed("afus", "mock", "t1")
    reloaded = Recorder("run-x", base_dir=tmp_path)
    assert cell_id("afus", "mock", "t1") in reloaded.completed_cells()

    ep_rows = [
        json.loads(line)
        for line in (tmp_path / "run-x" / "episodes.jsonl").read_text().splitlines()
    ]
    assert len(ep_rows) == 1
    row = ep_rows[0]
    assert row["success"] is True
    assert row["obs_tokens_total"] == 120
    assert row["usage"]["input_tokens"] == 200
    assert "total_cost" in row["cost"]


def test_mock_client_scripted():
    mc = MockClient(responses=['{"action": "done"}'])
    r = mc.complete(system="s", blocks=[TextBlock(text="page text here")], tools=None)
    assert r.text == '{"action": "done"}'
    assert len(mc.calls) == 1
