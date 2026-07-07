"""Offline Mind2Web element-selection eval (Phase 3), driven by a scripted MockClient."""

from __future__ import annotations

import json

import pytest

import modalitybench.observations.serializers  # noqa: F401
from modalitybench.agents.model_client import MockClient
from modalitybench.metrics.tokens import TokenCounter
from modalitybench.runner.matrix import evaluate_offline
from modalitybench.tasks.mind2web_offline import (
    Mind2WebOffline,
    _normalise_hf_action,
    action_f1,
)


def _correct_ref(graph, gt) -> str:
    for node in graph.nodes():
        if node.ref and node.locator and str(node.locator["value"]) in gt["backend_ids"]:
            return node.ref
    raise AssertionError("target element not found in graph")


def _scripted_responses(source, task, *, correct: bool):
    responses = []
    for graph, gt in source.snapshots(task):
        if correct:
            ref = _correct_ref(graph, gt)
            payload = {"action": gt["kind"], "ref": ref}
            if gt["value"]:
                payload["text"] = gt["value"]
        else:
            payload = {"action": "click", "ref": "e999"}  # never valid
        responses.append(json.dumps(payload))
    return responses


def test_sample_loads():
    src = Mind2WebOffline(backend="local")
    tasks = src.tasks()
    assert len(tasks) == 2
    assert all(t.goal for t in tasks)
    snaps = src.snapshots(tasks[0])
    assert len(snaps) == 3
    graph, gt = snaps[0]
    assert gt["backend_ids"] == {"101"}
    assert gt["kind"] == "type"


def test_perfect_agent_scores_full():
    src = Mind2WebOffline(backend="local")
    task = src.tasks()[0]
    mock = MockClient(responses=_scripted_responses(src, task, correct=True))
    ep = evaluate_offline(src, task, "afus", mock, TokenCounter(None), max_steps=15)
    result = src.score(task, ep)
    assert result.metrics["element_accuracy"] == 1.0
    assert result.metrics["step_success_rate"] == 1.0
    assert result.success is True
    # Observation size was measured for every step.
    assert all(s.obs_tokens > 0 for s in ep.steps)


def test_wrong_agent_scores_zero():
    src = Mind2WebOffline(backend="local")
    task = src.tasks()[0]
    mock = MockClient(responses=_scripted_responses(src, task, correct=False))
    ep = evaluate_offline(src, task, "fct", mock, TokenCounter(None), max_steps=15)
    result = src.score(task, ep)
    assert result.metrics["element_accuracy"] == 0.0
    assert result.success is False


@pytest.mark.parametrize("strategy", ["afus", "fct", "flat_elements", "pruned_html"])
def test_perfect_agent_across_serializers(strategy):
    src = Mind2WebOffline(backend="local")
    task = src.tasks()[1]  # the newsletter form task
    mock = MockClient(responses=_scripted_responses(src, task, correct=True))
    ep = evaluate_offline(src, task, strategy, mock, TokenCounter(None), max_steps=15)
    assert src.score(task, ep).metrics["element_accuracy"] == 1.0


def test_action_f1():
    assert action_f1("click", "", "click", "") == 1.0
    assert action_f1("type", "hello world", "type", "hello world") == 1.0
    assert action_f1("click", "", "type", "hello") < 1.0
    assert action_f1("type", "wrong", "click", "") == 0.0


def test_normalise_hf_action_accepts_dict_and_json_string_forms():
    # `datasets` yields dict fields for train; the raw test.zip encodes them as JSON strings.
    dict_form = {
        "action_uid": "a1",
        "operation": {"op": "CLICK", "value": ""},
        "pos_candidates": [{"backend_node_id": 42, "tag": "a"}],
        "cleaned_html": "<html></html>",
    }
    str_form = {
        "action_uid": "a1",
        "operation": json.dumps({"op": "CLICK", "value": ""}),
        "pos_candidates": [json.dumps({"backend_node_id": 42, "tag": "a"})],
        "cleaned_html": "<html></html>",
    }
    a, b = _normalise_hf_action(dict_form), _normalise_hf_action(str_form)
    assert a == b
    assert a["operation"]["op"] == "CLICK"
    assert a["pos_candidates"] == [{"backend_node_id": "42", "tag": "a"}]
