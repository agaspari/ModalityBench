from __future__ import annotations

import pytest

from modalitybench.agents.actions import Action
from modalitybench.agents.model_client import MockClient, Usage
from modalitybench.agents.loop import decide_with_guardrail
from modalitybench.observations.base import Observation, PageGraph, PageNode, TextBlock, RefRegistry


def test_decide_with_guardrail_fresh_action():
    # Setup simple page graph
    root = PageNode(tag="body")
    n1 = PageNode(tag="button", ref="e1", role="button", name="Button 1", visible=True)
    n1.locator = {"type": "selector", "value": "btn-1"}
    root.children = [n1]
    graph = PageGraph(root=root)

    registry = RefRegistry()
    registry.add("e1", n1.locator)

    obs = Observation(
        content_blocks=[TextBlock(text='[e1] button "Button 1"')],
        ref_registry=registry
    )

    client = MockClient(responder=lambda **_: '{"action": "click", "ref": "e1"}', model="mock")
    tried_actions = set()

    # First call: fresh action, should succeed and record the action
    action, usage, latency, raw, meta_calls = decide_with_guardrail(
        client, "click button 1", obs, [],
        tried_actions=tried_actions,
        graph=graph,
    )

    assert action.kind == "click"
    assert action.ref == "e1"
    assert len(tried_actions) == 1
    assert ("click", "btn-1", None) in tried_actions
    assert "guardrail_escalation" not in meta_calls


def test_decide_with_guardrail_redundant_action_trigger():
    # Setup page graph with static text and two buttons
    root = PageNode(tag="body")
    n1 = PageNode(tag="button", ref="e1", role="button", name="Button 1", visible=True)
    n1.locator = {"type": "selector", "value": "btn-1"}
    n2 = PageNode(tag="button", ref="e2", role="button", name="Button 2", visible=True)
    n2.locator = {"type": "selector", "value": "btn-2"}
    n3 = PageNode(tag="div", text="Important Static Label", visible=True)
    root.children = [n1, n2, n3]
    graph = PageGraph(root=root)

    registry = RefRegistry()
    registry.add("e1", n1.locator)
    registry.add("e2", n2.locator)

    obs = Observation(
        content_blocks=[TextBlock(text='[e1] button "Button 1"\n[e2] button "Button 2"')],
        ref_registry=registry
    )

    # Initialize tried_actions as if we already clicked Button 1 on this page
    tried_actions = {("click", "btn-1", None)}

    # Client returns button 1 first (redundant), then button 2 (valid fallback)
    idx = 0
    responses = [
        '{"action": "click", "ref": "e1"}',
        '{"action": "click", "ref": "e2"}'
    ]
    def responder(**_):
        nonlocal idx
        res = responses[idx]
        idx = min(idx + 1, len(responses) - 1)
        return res

    client = MockClient(responder=responder, model="mock")

    action, usage, latency, raw, meta_calls = decide_with_guardrail(
        client, "click buttons", obs, [],
        tried_actions=tried_actions,
        graph=graph,
    )

    # The guardrail should have intercepted the click on e1, masked it, escalated the static text,
    # and returned click on e2.
    assert action.kind == "click"
    assert action.ref == "e2"
    assert "guardrail_escalation" in meta_calls

    # e1 should have been deleted from registry
    assert "e1" not in obs.ref_registry

    # e1 line should have been masked from original text
    text_content = "\n".join(b.text for b in obs.content_blocks if hasattr(b, 'text'))
    assert "[e1]" not in text_content
    assert "[e2]" in text_content

    # Warning message and static text should be appended
    assert "Your previous action click(e1) was idempotent" in text_content
    assert "Important Static Label" in text_content

    # Tried actions should now also contain button 2 click
    assert ("click", "btn-2", None) in tried_actions
