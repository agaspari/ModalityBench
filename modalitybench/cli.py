"""ModalityBench CLI (``mb``).

Commands:
  mb list-strategies            list registered observation strategies
  mb serialize-bench            token/size comparison of serializers over snapshots
  mb run <config.yaml>          run a strategy x model x task matrix
  mb dashboard <run_id>         build a self-contained dashboard.html
  mb export <run_id>            export results as csv / json

Commands beyond Phase 1 print a clear "not yet available" notice until their phase lands.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, help="ModalityBench — observation-reduction benchmark.")
console = Console()


def _load_registry() -> None:
    """Import serializer/strategy packages so the registry is populated."""
    try:
        import modalitybench.observations.serializers  # noqa: F401
    except ImportError:
        pass
    try:
        import modalitybench.observations.strategies  # noqa: F401
    except ImportError:
        pass


@app.command("list-strategies")
def list_strategies_cmd() -> None:
    """List all registered observation strategies."""
    _load_registry()
    from modalitybench.observations import list_strategies

    names = list_strategies()
    if not names:
        console.print("[yellow]No strategies registered yet (Phase 2 adds serializers).[/]")
        raise typer.Exit()
    table = Table(title="Registered observation strategies")
    table.add_column("name", style="cyan")
    for n in names:
        table.add_row(n)
    console.print(table)


@app.command("serialize-bench")
def serialize_bench_cmd(
    fixtures: bool = typer.Option(True, help="Use bundled HTML fixtures."),
    out: Path = typer.Option(Path("results/serialize_bench.json"), help="Output JSON path."),
    approx: bool = typer.Option(
        False, "--approx", help="Use the local token heuristic instead of the exact API count."
    ),
) -> None:
    """Compare serializers by token/byte/element count over page snapshots (no agent loop)."""
    try:
        from modalitybench.runner.serialize_bench import run_serialize_bench
    except ImportError:
        console.print("[yellow]serialize-bench arrives in Phase 2.[/]")
        raise typer.Exit(code=1)
    run_serialize_bench(fixtures=fixtures, out=out, approx=approx, console=console)


@app.command("run")
def run_cmd(config: Path = typer.Argument(..., help="Path to a run config YAML.")) -> None:
    """Run a strategy x model x task matrix defined by a config file."""
    try:
        from modalitybench.runner.matrix import run_matrix
    except ImportError:
        console.print("[yellow]The matrix runner arrives in Phase 3.[/]")
        raise typer.Exit(code=1)
    from modalitybench.runner.config import load_config

    run_matrix(load_config(config), console=console)


@app.command("dashboard")
def dashboard_cmd(
    run_id: str = typer.Argument(..., help="Run id under the results dir."),
    results_dir: Path = typer.Option(Path("results"), help="Results base dir."),
    out: Path = typer.Option(None, help="Output HTML path (default: <run>/dashboard.html)."),
) -> None:
    """Build a self-contained dashboard.html from a run's results."""
    try:
        from modalitybench.dashboard.build import build_dashboard
    except ImportError:
        console.print("[yellow]The dashboard arrives in Phase 5.[/]")
        raise typer.Exit(code=1)
    build_dashboard(run_id, results_dir=results_dir, out=out, console=console)


@app.command("export")
def export_cmd(
    run_id: str = typer.Argument(...),
    fmt: str = typer.Option("csv", "--format", help="csv | json"),
    results_dir: Path = typer.Option(Path("results")),
) -> None:
    """Export a run's episode results as csv or json for sharing."""
    try:
        from modalitybench.dashboard.build import export_results
    except ImportError:
        console.print("[yellow]Exports arrive in Phase 5.[/]")
        raise typer.Exit(code=1)
    export_results(run_id, fmt=fmt, results_dir=results_dir, console=console)


if __name__ == "__main__":
    app()
