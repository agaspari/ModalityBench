"""Offline serializer bench: token / byte / element comparison over sample pages.

Runs every registered serializer over a set of page snapshots and reports how compact each
is — the first shareable result of the project, and the cheap path (no agent loop; token
counts are the only optional API cost, and even those fall back to a local estimate).
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from modalitybench.metrics.tokens import TokenCounter


def load_samples() -> list[tuple[str, str]]:
    """Return ``(name, html)`` for every bundled sample page."""
    base = files("modalitybench.data").joinpath("samples")
    out: list[tuple[str, str]] = []
    for entry in sorted(base.iterdir(), key=lambda p: p.name):
        if entry.name.endswith(".html"):
            out.append((entry.name[:-5], entry.read_text(encoding="utf-8")))
    return out


def run_serialize_bench(
    *,
    fixtures: bool = True,
    out: Path = Path("results/serialize_bench.json"),
    approx: bool = False,
    console: Console | None = None,
) -> dict[str, Any]:
    console = console or Console()
    import modalitybench.observations.serializers  # noqa: F401  (registers serializers)
    from modalitybench.observations import get_strategy, list_strategies
    from modalitybench.observations.dom_capture import graph_from_html

    client = None
    if not approx:
        try:
            from modalitybench.agents.model_client import AnthropicClient

            client = AnthropicClient()
        except Exception as exc:  # no key / SDK issue → local estimate
            console.print(f"[yellow]Exact token counting unavailable ({exc}); using approx.[/]")
    tc = TokenCounter(client=client)

    samples = load_samples()
    strategies = list_strategies()
    if not samples:
        console.print("[red]No sample pages found.[/]")
        return {}

    # rows[strategy][sample] = {tokens, bytes, chars, elements}
    rows: dict[str, dict[str, dict[str, int]]] = {s: {} for s in strategies}
    for sample_name, html in samples:
        graph = graph_from_html(html, url=f"https://example.com/{sample_name}")
        for strat in strategies:
            obs = get_strategy(strat).observe(graph)
            rows[strat][sample_name] = {
                "tokens": tc.count_blocks(obs.content_blocks),
                "bytes": obs.meta["bytes"],
                "chars": obs.meta["chars"],
                "elements": obs.meta["element_count"],
            }

    sample_names = [s for s, _ in samples]
    _print_table(console, rows, sample_names, strategies, tc.mode)

    result = {
        "token_mode": tc.mode,
        "samples": sample_names,
        "serializers": rows,
        "summary": _summary(rows, sample_names, strategies),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    console.print(f"\n[green]Wrote[/] {out}")
    return result


def _summary(rows, sample_names, strategies) -> dict[str, dict[str, float]]:
    """Mean tokens per serializer and reduction factor vs raw_html."""
    raw = "raw_html"
    summary: dict[str, dict[str, float]] = {}
    for strat in strategies:
        toks = [rows[strat][s]["tokens"] for s in sample_names]
        mean = sum(toks) / len(toks)
        # per-sample reduction vs raw, then averaged.
        reductions = []
        for s in sample_names:
            r = rows.get(raw, {}).get(s, {}).get("tokens")
            if r:
                reductions.append(rows[strat][s]["tokens"] / r)
        summary[strat] = {
            "mean_tokens": round(mean, 1),
            "frac_of_raw": round(sum(reductions) / len(reductions), 3) if reductions else 1.0,
        }
    return summary


def _print_table(console, rows, sample_names, strategies, mode) -> None:
    table = Table(title=f"Serializer token comparison ({mode} tokens)")
    table.add_column("serializer", style="cyan")
    for s in sample_names:
        table.add_column(s, justify="right")
    table.add_column("mean", justify="right", style="bold")
    table.add_column("×raw", justify="right", style="green")

    # Order rows by mean tokens ascending (most compact first), raw_html last.
    ordered = sorted(strategies, key=lambda st: sum(rows[st][s]["tokens"] for s in sample_names))
    raw_means = {s: rows.get("raw_html", {}).get(s, {}).get("tokens", 0) for s in sample_names}
    for strat in ordered:
        toks = [rows[strat][s]["tokens"] for s in sample_names]
        mean = sum(toks) / len(toks)
        fracs = [
            rows[strat][s]["tokens"] / raw_means[s]
            for s in sample_names
            if raw_means[s]
        ]
        frac = sum(fracs) / len(fracs) if fracs else 1.0
        cells = [str(t) for t in toks]
        table.add_row(strat, *cells, f"{mean:.0f}", f"{frac:.2f}")
    console.print(table)
