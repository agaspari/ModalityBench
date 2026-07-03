"""Shared test fixtures: load the bundled sample pages into PageGraphs."""

from __future__ import annotations

from importlib.resources import files

import pytest

from modalitybench.observations.dom_capture import graph_from_html
from modalitybench.observations.loader import load_strategies

load_strategies()  # register serializers + wrapper strategies


def _sample_html(name: str) -> str:
    return files("modalitybench.data").joinpath("samples", f"{name}.html").read_text(
        encoding="utf-8"
    )


@pytest.fixture
def checkout_graph():
    return graph_from_html(
        _sample_html("checkout"),
        url="https://shop.example.com/checkout/payment",
        title="Checkout · Payment",
    )


@pytest.fixture
def search_graph():
    return graph_from_html(_sample_html("search"), url="https://shop.example.com/search")


@pytest.fixture
def form_graph():
    return graph_from_html(_sample_html("form"), url="https://shop.example.com/contact")


@pytest.fixture(params=["checkout", "search", "form"])
def any_graph(request):
    return graph_from_html(_sample_html(request.param), url="https://example.com/x")
