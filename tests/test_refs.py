"""Ref-registry invariants (Phase 2).

The ref registry is the contract between serializer and executor: every ref a serializer can
emit must resolve, and the ref set must be identical across serializers so an action is valid
regardless of which observation produced it.
"""

from __future__ import annotations

import re

from modalitybench.observations import get_strategy

ALL = ["raw_html", "body_html", "pruned_html", "axtree", "flat_elements", "afus", "fct"]


def test_every_ref_resolves(any_graph):
    for name in ALL:
        obs = get_strategy(name).observe(any_graph)
        for ref in obs.ref_registry.refs():
            loc = obs.ref_registry.resolve(ref)
            assert loc is not None and "type" in loc and "value" in loc


def test_ref_set_identical_across_serializers(checkout_graph):
    ref_sets = {
        name: set(get_strategy(name).observe(checkout_graph).ref_registry.refs())
        for name in ALL
    }
    first = next(iter(ref_sets.values()))
    for name, s in ref_sets.items():
        assert s == first, f"{name} has a divergent ref set"


def test_refs_contiguous(checkout_graph):
    refs = get_strategy("fct").observe(checkout_graph).ref_registry.refs()
    nums = sorted(int(r[1:]) for r in refs)
    assert nums == list(range(1, len(nums) + 1)), nums


def test_refs_are_e_prefixed(any_graph):
    obs = get_strategy("afus").observe(any_graph)
    assert all(re.fullmatch(r"e\d+", r) for r in obs.ref_registry.refs())


def test_interactive_refs_resolve_for_actions(checkout_graph):
    # The 'Pay' button and 'Card number' field must be actionable.
    obs = get_strategy("flat_elements").observe(checkout_graph)
    text = obs.text()
    assert "Pay $142.50" in text and "Card number" in text
    # Pull a ref out of the flat listing and confirm it resolves.
    m = re.search(r"\[(e\d+)\] .*Card number", text)
    assert m, text
    assert obs.ref_registry.resolve(m.group(1)) is not None
