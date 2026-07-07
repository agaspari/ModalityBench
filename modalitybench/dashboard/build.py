"""Build a self-contained dashboard.html from a run's results, and export CSV/JSON.

The dashboard inlines Plotly (no CDN — a strict-CSP-safe single file you can share). Charts:
the token/quality Pareto frontier, per-strategy token bars, cost and latency bars, and a
per-cell drill-down table. Every chart's modebar exports PNG/SVG. ``mb export`` dumps the
episode rows as CSV or JSON for posting elsewhere.
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console


def _load_episodes(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "episodes.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no episodes.jsonl in {run_dir}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_steps(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "steps.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@dataclass
class _OraclePoint:
    model: str
    quality: float  # fraction of steps solved by *some* representation (per-step upper bound)
    input_tokens: float  # mean per-episode cost picking the cheapest sufficient rep each step


def _oracle_points(steps: list[dict[str, Any]]) -> list[_OraclePoint]:
    """The 'magic genie' upper bound: per step, the cheapest representation that gets it right.

    A free post-hoc join over a multi-strategy run's per-step records — no new model calls.
    Only meaningful where per-step correctness exists (offline element-selection), so returns
    empty for live runs. Quality = fraction of steps solvable by *any* strategy (no single
    serializer dominates, so the union beats the best fixed one); cost = sum of the cheapest
    correct strategy's real input tokens each step (or the cheapest overall when no rep works).
    """
    if not any(s.get("correct") is not None for s in steps):
        return []
    by_model: dict[str, list[dict[str, Any]]] = {}
    for s in steps:
        by_model.setdefault(s.get("model", ""), []).append(s)

    out: list[_OraclePoint] = []
    for model, ms in by_model.items():
        # (task -> step_index -> {strategy: (correct, input_tokens)})
        tasks: dict[str, dict[int, dict[str, tuple[bool, int]]]] = {}
        for s in ms:
            step_map = tasks.setdefault(s["task_id"], {}).setdefault(s["step_index"], {})
            step_map[s["strategy"]] = (bool(s.get("correct")), int(s.get("input_tokens", 0)))
        # need >1 strategy for a routing choice to exist
        n_strats = len({s["strategy"] for s in ms})
        if n_strats < 2:
            continue
        task_acc: list[float] = []
        task_tok: list[float] = []
        for step_map in tasks.values():
            flags: list[float] = []
            toks: list[int] = []
            for choices in step_map.values():
                correct = [tok for _s, (c, tok) in choices.items() if c]
                flags.append(1.0 if correct else 0.0)
                toks.append(min(correct) if correct else min(t for _c, t in choices.values()))
            if flags:
                task_acc.append(sum(flags) / len(flags))
                task_tok.append(float(sum(toks)))
        if task_acc:
            out.append(_OraclePoint(model, _mean(task_acc), _mean(task_tok)))
    return out


@dataclass
class _CellAgg:
    strategy: str
    model: str
    n: int = 0
    success: float = 0.0
    reward: float = 0.0
    obs_tokens: float = 0.0
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    cost: float = 0.0
    latency: float = 0.0
    quality_metric: str = "reward"
    metrics: dict[str, float] = field(default_factory=dict)


def _aggregate(rows: list[dict[str, Any]]) -> list[_CellAgg]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        buckets.setdefault((r["strategy"], r["model"]), []).append(r)

    out: list[_CellAgg] = []
    for (strategy, model), items in buckets.items():
        n = len(items)
        agg = _CellAgg(strategy=strategy, model=model, n=n)
        agg.success = _mean(1.0 if i["success"] else 0.0 for i in items)
        agg.reward = _mean(i.get("reward", 0.0) for i in items)
        agg.obs_tokens = _mean(i.get("obs_tokens_total", 0) for i in items)
        agg.input_tokens = _mean(i["usage"]["input_tokens"] for i in items)
        agg.output_tokens = _mean(i["usage"]["output_tokens"] for i in items)
        agg.cost = _mean(i["cost"]["total_cost"] for i in items)
        agg.latency = _mean(i.get("latency_s_total", 0.0) for i in items)
        # Prefer a domain metric (Mind2Web element_accuracy) as "quality" when present.
        metric_keys: set[str] = set()
        for i in items:
            metric_keys |= set(i.get("metrics", {}).keys())
        for k in metric_keys:
            agg.metrics[k] = _mean(i.get("metrics", {}).get(k, 0.0) for i in items)
        if "element_accuracy" in agg.metrics:
            agg.quality_metric = "element_accuracy"
        out.append(agg)
    out.sort(key=lambda a: a.obs_tokens)
    return out


def _mean(xs) -> float:
    xs = list(xs)
    return round(statistics.fmean(xs), 4) if xs else 0.0


def _quality(agg: _CellAgg) -> float:
    if agg.quality_metric == "element_accuracy":
        return agg.metrics.get("element_accuracy", agg.reward)
    return agg.reward


# Palette (validated via the dataviz skill: categorical #2a78d6/#1baf7a/#eb6834 pass CVD;
# blue sequential ramp for magnitude). Kept as module constants so the whole page reads
# as one system.
_BLUE = "#2a78d6"
_AQUA = "#1baf7a"
_ORANGE = "#eb6834"
_INK = "#0b0b0b"
_BLUE_SEQ = [
    [0.0, "#eef5fe"],
    [0.2, "#b7d3f6"],
    [0.4, "#6da7ec"],
    [0.6, "#3987e5"],
    [0.8, "#256abf"],
    [1.0, "#104281"],
]


def _episode_quality(row: dict[str, Any], quality_metric: str) -> float:
    """The per-episode value the heatmap plots — matches the frontier's quality axis.

    Offline sources (Mind2Web) expose ``element_accuracy`` (per-step partial credit); binary
    ``success`` there is all-or-nothing across a 5-15 step task and floors to 0, so the grid
    would be dead. Live sources fall back to ``reward``.
    """
    if quality_metric == "element_accuracy":
        return float(row.get("metrics", {}).get("element_accuracy", row.get("reward", 0.0)))
    return float(row.get("reward", 0.0))


def _task_matrix(
    rows: list[dict[str, Any]],
    quality_metric: str = "reward",
) -> tuple[list[str], list[str], list[list[float | None]], list[list[str]], list[list[int]]]:
    """Mean quality per (strategy, base-task), for the per-task heatmap.

    ``task_id`` is ``"<task>#<seed>"``; we aggregate over seeds. Tasks are ordered by
    overall quality (best first) so floored tasks cluster at the bottom.
    """
    cells: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        strat = r["strategy"]
        task = str(r["task_id"]).split("#")[0]
        cells.setdefault((strat, task), []).append(_episode_quality(r, quality_metric))

    strategies = sorted({s for s, _ in cells})
    tasks = sorted({t for _, t in cells})

    def overall(t: str) -> float:
        vals = [v for (_s, tt), lst in cells.items() if tt == t for v in lst]
        return sum(vals) / len(vals) if vals else 0.0

    tasks.sort(key=lambda t: (-overall(t), t))

    z: list[list[float | None]] = []
    text: list[list[str]] = []
    counts: list[list[int]] = []
    for s in strategies:
        zr: list[float | None] = []
        tr: list[str] = []
        nr: list[int] = []
        for t in tasks:
            lst = cells.get((s, t))
            if lst:
                m = sum(lst) / len(lst)
                zr.append(round(m, 4))
                tr.append(f"{m:.2f}")
                nr.append(len(lst))
            else:
                zr.append(None)
                tr.append("")
                nr.append(0)
        z.append(zr)
        text.append(tr)
        counts.append(nr)
    return strategies, tasks, z, text, counts


def build_dashboard(
    run_id: str,
    *,
    results_dir: Path = Path("results"),
    out: Path | None = None,
    console: Console | None = None,
) -> Path:
    console = console or Console()
    import plotly.graph_objects as go

    run_dir = Path(results_dir) / run_id
    rows = _load_episodes(run_dir)
    aggs = _aggregate(rows)
    out = out or (run_dir / "dashboard.html")

    models = {a.model for a in aggs}
    labels = [a.strategy if len(models) == 1 else f"{a.strategy}·{a.model}" for a in aggs]
    quality_name = aggs[0].quality_metric if aggs else "reward"
    # The 'magic genie' upper bound: cheapest sufficient representation per step (free post-hoc
    # join, offline runs only). Its gap above the best fixed strategy is the routing headroom.
    oracle_pts = _oracle_points(_load_steps(run_dir))

    # 1) Frontier: REAL billed input tokens (x, lower better) vs quality (y, higher better).
    #    Real usage.input_tokens is the honest cost axis — observation size lies (see #3).
    pareto = go.Figure()
    pareto.add_trace(
        go.Scatter(
            x=[a.input_tokens for a in aggs],
            y=[_quality(a) for a in aggs],
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=13, color=_BLUE, line=dict(width=1.5, color="#ffffff")),
            hovertemplate="<b>%{text}</b><br>real input tok/ep=%{x:.0f}<br>"
            + quality_name + "=%{y:.3f}<extra></extra>",
        )
    )
    if oracle_pts:
        pareto.add_trace(
            go.Scatter(
                x=[o.input_tokens for o in oracle_pts],
                y=[o.quality for o in oracle_pts],
                mode="markers+text",
                text=["oracle" for _ in oracle_pts],
                textposition="bottom center",
                marker=dict(size=17, color=_ORANGE, symbol="star",
                            line=dict(width=1.5, color="#ffffff")),
                name="oracle (per-step upper bound)",
                hovertemplate="<b>oracle — per-step upper bound</b><br>real input tok/ep="
                "%{x:.0f}<br>" + quality_name + "=%{y:.3f}<extra></extra>",
            )
        )
    pareto.update_layout(
        title="Accuracy vs REAL billed input tokens per episode (upper-left is better)",
        xaxis_title="mean real billed input tokens / episode",
        yaxis_title=quality_name,
        template="plotly_white",
        height=460,
        showlegend=bool(oracle_pts),
    )

    # 2) Per-task success matrix: which tasks carry signal (passable) for each strategy.
    #    Missing cells (a strategy that hasn't run a task yet, e.g. mid-run) are coalesced to
    #    0.0 for a fully numeric z — Plotly's categorical heatmap fails axis scaling on nulls —
    #    and left unannotated (cell_n == 0) so they read as blank "not run" rather than a real 0.
    strategies, tasks_x, z, cell_text, cell_n = _task_matrix(rows, quality_name)
    # Transpose to tasks-as-rows: with many tasks a tall matrix (a few strategy columns, one
    # readable horizontal task label per row) scans far better than a wide 2-row strip.
    ns, nt = len(strategies), len(tasks_x)
    zT = [[(z[s][t] if z[s][t] is not None else 0.0) for s in range(ns)] for t in range(nt)]
    textT = [[cell_text[s][t] for s in range(ns)] for t in range(nt)]
    nT = [[cell_n[s][t] for s in range(ns)] for t in range(nt)]
    heat = go.Figure(
        go.Heatmap(
            z=zT, x=strategies, y=tasks_x, zmin=0, zmax=1, colorscale=_BLUE_SEQ,
            colorbar=dict(title=quality_name), customdata=nT,
            hovertemplate="<b>%{y}</b><br>%{x}<br>" + quality_name + "=%{z:.2f}"
            "<br>n=%{customdata}<extra></extra>",
        )
    )
    anns = [
        dict(
            x=strategies[xi], y=tasks_x[yi], text=textT[yi][xi], showarrow=False,
            font=dict(size=11, color="#ffffff" if zT[yi][xi] > 0.5 else _INK),
        )
        for yi in range(nt)
        for xi in range(ns)
        if nT[yi][xi] > 0
    ]
    # Height grows ~30px per task (best-first from the top). A hard floor of 300px keeps the
    # plot area well clear of the 120px top+bottom margins even for few-task runs — otherwise a
    # too-short figure collapses the axis and Plotly throws "Something went wrong with axis
    # scaling" in setScale (seen on 1-2 task smoke runs).
    heat.update_layout(
        title=f"Per-task {quality_name} (task × strategy) — darker = better",
        annotations=anns, template="plotly_white",
        height=max(300, 130 + 30 * max(nt, 1)),
        margin=dict(t=70, l=180, b=50, r=20),
        yaxis=dict(autorange="reversed"),
    )

    # 3) The honesty gap: apparent observation size vs real billed input tokens.
    ratios = [(a.input_tokens / a.obs_tokens if a.obs_tokens else 0.0) for a in aggs]
    gap = go.Figure()
    gap.add_trace(
        go.Bar(name="apparent (observation tokens)", x=labels,
               y=[a.obs_tokens for a in aggs], marker_color=_ORANGE)
    )
    gap.add_trace(
        go.Bar(name="real billed (input tokens)", x=labels,
               y=[a.input_tokens for a in aggs], marker_color=_BLUE,
               text=[f"{r:.0f}×" for r in ratios], textposition="outside")
    )
    gap.update_layout(
        title="The honesty gap: observation size vs real billed input tokens (× = real ÷ apparent)",
        barmode="group", yaxis_title="tokens / episode",
        template="plotly_white", height=430,
    )

    # 4) Token bars: input vs output per strategy.
    tokens = go.Figure()
    tokens.add_trace(
        go.Bar(name="input tokens", x=labels, y=[a.input_tokens for a in aggs],
               marker_color=_BLUE)
    )
    tokens.add_trace(
        go.Bar(name="output tokens", x=labels, y=[a.output_tokens for a in aggs],
               marker_color=_AQUA)
    )
    tokens.update_layout(
        title="Model tokens per episode", barmode="group", template="plotly_white", height=420
    )

    # 5) Cost bar.
    cost = go.Figure(go.Bar(x=labels, y=[a.cost for a in aggs], marker_color=_ORANGE))
    cost.update_layout(
        title="Mean cost per episode (USD)", yaxis_title="$", template="plotly_white", height=380
    )

    # 6) Latency bar.
    latency = go.Figure(go.Bar(x=labels, y=[a.latency for a in aggs], marker_color=_BLUE))
    latency.update_layout(
        title="Mean latency per episode (s)", yaxis_title="seconds",
        template="plotly_white", height=380
    )

    config = {"displaylogo": False, "toImageButtonOptions": {"format": "svg"}}
    figs = [pareto, heat, gap, tokens, cost, latency]
    chart_html = []
    for i, fig in enumerate(figs):
        inner = fig.to_html(full_html=False, include_plotlyjs=(i == 0), config=config)
        chart_html.append(f'<div class="chart">{inner}</div>')

    table_html = _table(aggs, quality_name, oracle_pts)
    out.parent.mkdir(parents=True, exist_ok=True)
    page = _PAGE_TEMPLATE.format(
        run_id=run_id,
        n_cells=len(aggs),
        n_episodes=len(rows),
        charts="\n".join(chart_html),
        table=table_html,
        quality_name=quality_name,
    )
    out.write_text(page, encoding="utf-8")
    console.print(f"[green]Wrote[/] {out}  ({len(aggs)} strategy cells, {len(rows)} episodes)")
    return out


def _table(
    aggs: list[_CellAgg], quality_name: str, oracle_pts: list["_OraclePoint"] | None = None
) -> str:
    head = (
        "<tr><th>strategy</th><th>model</th><th>tasks</th>"
        f"<th>{quality_name}</th><th>success</th><th>obs tokens</th>"
        "<th>input tok</th><th>output tok</th><th>cost $</th><th>latency s</th></tr>"
    )
    body = []
    for a in aggs:
        body.append(
            "<tr>"
            f"<td>{a.strategy}</td><td>{a.model}</td><td>{a.n}</td>"
            f"<td>{_quality(a):.3f}</td><td>{a.success:.2f}</td>"
            f"<td>{a.obs_tokens:.0f}</td><td>{a.input_tokens:.0f}</td>"
            f"<td>{a.output_tokens:.0f}</td><td>{a.cost:.4f}</td><td>{a.latency:.2f}</td>"
            "</tr>"
        )
    for o in oracle_pts or []:
        # Synthetic upper bound — only quality + input tokens are defined; dashes elsewhere.
        body.append(
            "<tr style='font-style:italic;color:#b1560f'>"
            f"<td>oracle (upper bound)</td><td>{o.model}</td><td>—</td>"
            f"<td>{o.quality:.3f}</td><td>—</td><td>—</td>"
            f"<td>{o.input_tokens:.0f}</td><td>—</td><td>—</td><td>—</td>"
            "</tr>"
        )
    return f"<table>{head}{''.join(body)}</table>"


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def export_results(
    run_id: str,
    *,
    fmt: str = "csv",
    results_dir: Path = Path("results"),
    console: Console | None = None,
) -> Path:
    console = console or Console()
    run_dir = Path(results_dir) / run_id
    rows = _load_episodes(run_dir)
    flat = [_flatten(r) for r in rows]
    if fmt == "json":
        out = run_dir / "episodes_export.json"
        out.write_text(json.dumps(flat, indent=2), encoding="utf-8")
    elif fmt == "csv":
        out = run_dir / "episodes_export.csv"
        fields = sorted({k for row in flat for k in row})
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat)
    else:
        raise ValueError(f"unknown export format {fmt!r} (use csv|json)")
    console.print(f"[green]Exported[/] {len(flat)} rows -> {out}")
    return out


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    out = {
        "strategy": row["strategy"],
        "model": row["model"],
        "task_id": row["task_id"],
        "success": row["success"],
        "reward": row.get("reward", 0.0),
        "n_steps": row.get("n_steps", 0),
        "obs_tokens_total": row.get("obs_tokens_total", 0),
    }
    for k, v in row.get("usage", {}).items():
        out[f"usage_{k}"] = v
    for k, v in row.get("cost", {}).items():
        out[f"cost_{k}"] = v
    for k, v in row.get("metrics", {}).items():
        out[f"metric_{k}"] = v
    return out


_PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>ModalityBench — {run_id}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 24px;
          background: #fafafa; color: #1a1a1a; }}
  h1 {{ margin: 0 0 4px; }}
  .sub {{ color: #666; margin-bottom: 20px; }}
  .chart {{ background: #fff; border: 1px solid #e5e5e5; border-radius: 8px; margin: 16px 0;
            padding: 8px; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 14px; }}
  th, td {{ border: 1px solid #e5e5e5; padding: 6px 10px; text-align: right; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  th {{ background: #f0f0f0; }}
  tr:nth-child(even) td {{ background: #fbfbfb; }}
</style></head>
<body>
  <h1>ModalityBench</h1>
  <div class="sub">run <b>{run_id}</b> — {n_cells} strategy cells, {n_episodes} episodes ·
       quality metric: <b>{quality_name}</b></div>
  <div class="chart">{charts}</div>
  <h2>Per-strategy summary</h2>
  {table}
</body></html>
"""
