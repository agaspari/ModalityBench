"""Serializer format + invariant tests (Phase 2)."""

from __future__ import annotations

import pytest

from modalitybench.observations import get_strategy, list_strategies

ALL = ["raw_html", "body_html", "pruned_html", "axtree", "flat_elements", "afus", "fct"]
COMPACT = ["axtree", "flat_elements", "afus", "fct"]


def test_all_serializers_registered():
    assert set(ALL).issubset(set(list_strategies()))


@pytest.mark.parametrize("name", ALL)
def test_serializer_nonempty(name, any_graph):
    obs = get_strategy(name).observe(any_graph)
    assert obs.text().strip(), f"{name} produced empty output"
    assert obs.meta["serializer"] == name


def test_element_count_identical_across_serializers(checkout_graph):
    # All serializers build their registry from the same graph → identical ref set size.
    counts = {n: get_strategy(n).observe(checkout_graph).meta["element_count"] for n in ALL}
    assert len(set(counts.values())) == 1, counts


def test_reduction_ordering(any_graph):
    b = {n: get_strategy(n).observe(any_graph).meta["bytes"] for n in ALL}
    assert b["raw_html"] >= b["body_html"] >= b["pruned_html"]
    for c in COMPACT:
        assert b[c] < b["pruned_html"], f"{c} not smaller than pruned_html"


def test_hidden_content_excluded_from_compact(checkout_graph):
    # csrf hidden input value and a display:none banner must never reach compact serializers.
    for name in COMPACT + ["pruned_html"]:
        text = get_strategy(name).observe(checkout_graph).text()
        assert "should-not-appear" not in text, name
        assert "marketing banner" not in text, name


def test_scripts_excluded_except_raw(checkout_graph):
    # raw_html is the untouched baseline and legitimately contains scripts; every
    # re-rendered / compact serializer must strip them.
    for name in [n for n in ALL if n != "raw_html"]:
        text = get_strategy(name).observe(checkout_graph).text()
        assert "tracking noise" not in text, name


def test_afus_no_indentation(any_graph):
    text = get_strategy("afus").observe(any_graph).text()
    for line in text.splitlines():
        assert line == line.lstrip(), f"AFUS line is indented: {line!r}"
    assert text.splitlines()[0].startswith("@page")


def test_afus_checkout_lines(checkout_graph):
    text = get_strategy("afus").observe(checkout_graph).text()
    assert 'in:e4 "Card number" req empty' in text
    assert 'sel:e7 "Country" ="United States" opts=4' in text
    assert 'chk:e8 "Save card" off' in text
    assert 'btn:e11 "Pay $142.50" disabled' in text
    assert "@sec pay" in text


def test_fct_header_and_shape(any_graph):
    text = get_strategy("fct").observe(any_graph).text()
    lines = text.splitlines()
    assert lines[0] == "ref|t|name|val|flags|scope"
    for row in lines[1:]:
        assert row.count("|") == 5, f"FCT row has wrong field count: {row!r}"


def test_fct_ditto_compression(checkout_graph):
    text = get_strategy("fct").observe(checkout_graph).text()
    assert "e4|in|Card number||req,empty|pay" in text
    # Subsequent rows in the same 'pay' scope use the ditto marker.
    assert '|"' in text
    assert 'e7|se|Country|United States|opts=4|"' in text
