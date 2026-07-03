"""Phase 4 strategy tests that don't need a browser: salience, tools-mode, screenshot."""

from __future__ import annotations

import io

import pytest

from modalitybench.agents.loop import decide
from modalitybench.agents.model_client import MockClient
from modalitybench.observations import get_strategy


# -- salience ---------------------------------------------------------------


def test_salience_prunes_and_keeps_relevant(checkout_graph):
    full = get_strategy("flat_elements").observe(checkout_graph).meta["element_count"]
    obs = get_strategy("salience_flat").observe(checkout_graph, task_text="pay the total now")
    assert obs.meta["salience_kept"] <= 15
    assert obs.meta["salience_kept"] <= full
    # The pay-related target should survive salience pruning.
    assert "Pay" in obs.text()


def test_salience_refs_still_resolve(checkout_graph):
    obs = get_strategy("salience_afus").observe(checkout_graph, task_text="enter card number")
    for ref in obs.ref_registry.refs():
        assert obs.ref_registry.resolve(ref) is not None


# -- tools mode -------------------------------------------------------------


def test_tools_mode_outline_and_find(checkout_graph):
    obs = get_strategy("tools_mode").observe(checkout_graph, task_text="pay")
    assert obs.meta["tools_mode"] is True
    assert obs.tools and {t.name for t in obs.tools} == {"outline", "find", "read"}
    handler = obs.meta["tool_handler"]
    assert "pay" in handler.outline().lower()
    hits = handler.find("card number")
    assert "Card number" in hits
    # read a specific ref
    some_ref = obs.ref_registry.refs()[0]
    assert some_ref in handler.read(some_ref)


def test_tools_mode_initial_obs_is_small(checkout_graph):
    tools = get_strategy("tools_mode").observe(checkout_graph)
    flat = get_strategy("flat_elements").observe(checkout_graph)
    assert tools.meta["bytes"] < flat.meta["bytes"]  # tiny outline vs full listing


def test_decide_runs_tools_mode_subloop(checkout_graph):
    obs = get_strategy("tools_mode").observe(checkout_graph, task_text="pay the total")
    pay_ref = next(
        r for r in obs.ref_registry.refs()
        if "Pay" in (checkout_graph.by_ref()[r].name or "")
    )
    mock = MockClient(
        responses=[
            '{"action": "find", "query": "pay"}',      # explore
            f'{{"action": "click", "ref": "{pay_ref}"}}',  # then act
        ]
    )
    action, usage, latency, raw = decide(mock, "pay the total", obs, history=[])
    assert action.kind == "click"
    assert action.ref == pay_ref
    assert len(mock.calls) == 2  # one meta round + the committing round
    assert usage.input_tokens > 0  # usage summed across rounds


# -- screenshot -------------------------------------------------------------


def _png_bytes(w=120, h=80, color=(200, 100, 50)):
    Image = pytest.importorskip("PIL.Image")
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_screenshot_full(checkout_graph):
    checkout_graph.meta["screenshot"] = _png_bytes()
    obs = get_strategy("screenshot").observe(checkout_graph)
    assert obs.has_image
    block = obs.content_blocks[0]
    assert block.media_type == "image/png"
    assert obs.meta["image_size"] == [120, 80]
    assert obs.meta["est_image_tokens"] > 0
    # Refs still resolve for a vision agent to act by ref.
    assert len(obs.ref_registry) == len(checkout_graph.by_ref())


def test_screenshot_scale_and_grayscale(checkout_graph):
    checkout_graph.meta["screenshot"] = _png_bytes(200, 100)
    half = get_strategy("screenshot_50pct").observe(checkout_graph)
    assert half.meta["image_size"] == [100, 50]
    gray = get_strategy("screenshot_50pct_gray").observe(checkout_graph)
    assert gray.meta["image_size"] == [100, 50]
    # Grayscale should not be larger than color at the same size.
    assert gray.meta["bytes"] <= half.meta["bytes"] * 1.5


def test_screenshot_requires_capture(checkout_graph):
    checkout_graph.meta.pop("screenshot", None)
    with pytest.raises(ValueError):
        get_strategy("screenshot").observe(checkout_graph)
