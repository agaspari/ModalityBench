"""Matrix runner: expand strategy × model × task, run each cell, record, resume.

Dispatches per task source: offline sources (Mind2Web) get the teacher-forced
element-selection evaluator here; live sources (MiniWoB++) get the agent loop from
:mod:`modalitybench.agents.loop` (Phase 4).
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

from modalitybench.agents.loop import decide
from modalitybench.agents.model_client import MockClient, ModelClient
from modalitybench.metrics.recorder import Recorder
from modalitybench.metrics.tokens import TokenCounter
from modalitybench.runner.config import ModelConfig, RunConfig
from modalitybench.tasks.base import Episode, StepRecord, Task, TaskSource


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def build_task_source(cfg) -> TaskSource:
    src = cfg.source
    opts = dict(cfg.options)
    if src == "mind2web_offline":
        from modalitybench.tasks.mind2web_offline import Mind2WebOffline

        return Mind2WebOffline(**opts)
    if src == "miniwob":
        from modalitybench.tasks.miniwob import MiniWobSource

        return MiniWobSource(**opts)
    if src == "webshop":
        from modalitybench.tasks.webshop import WebShopSource

        return WebShopSource(**opts)
    raise ValueError(f"unknown task source {src!r}")


# Default API base URL + key env var per OpenAI-compatible provider family. A model's own
# `base_url` / `api_key_env` in the config override these.
_PROVIDER_DEFAULTS = {
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "zhipu": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPUAI_API_KEY"),
}


def build_model_client(mcfg: ModelConfig) -> ModelClient:
    from modalitybench.metrics.tokens import model_family

    if mcfg.mock:
        # Deterministic stand-in for pipeline dry runs: always clicks e1.
        return MockClient(responder=lambda **_: '{"action": "click", "ref": "e1"}',
                          model="mock")

    family = model_family(mcfg.name)
    if family in _PROVIDER_DEFAULTS or mcfg.base_url:
        from modalitybench.agents.model_client import OpenAICompatibleClient

        default_url, default_env = _PROVIDER_DEFAULTS.get(family, (None, None))
        return OpenAICompatibleClient(
            model=mcfg.name,
            base_url=mcfg.base_url or default_url,
            api_key_env=mcfg.api_key_env or default_env,
            max_tokens=mcfg.max_tokens,
        )

    from modalitybench.agents.model_client import AnthropicClient

    return AnthropicClient(
        model=mcfg.name,
        max_tokens=mcfg.max_tokens,
        thinking=mcfg.thinking,
        effort=mcfg.effort,
    )


# ---------------------------------------------------------------------------
# Offline element-selection evaluator
# ---------------------------------------------------------------------------


def evaluate_offline(
    source: Any,
    task: Task,
    strategy_name: str,
    model_client: ModelClient,
    token_counter: TokenCounter,
    max_steps: int,
) -> Episode:
    from modalitybench.observations import get_strategy
    from modalitybench.tasks.mind2web_offline import action_f1

    strat = get_strategy(strategy_name)
    episode = Episode(task_id=task.task_id, strategy=strategy_name, model=model_client.model)
    history: list[str] = []

    for i, (graph, gt) in enumerate(source.snapshots(task)):
        if i >= max_steps:
            break
        obs = strat.observe(graph, task_text=task.goal)
        obs_tokens = token_counter.count_blocks(obs.content_blocks)
        # decide() runs the tools-mode meta-loop (adaptive/tools_mode escalate here) and
        # sums every meta-round's usage into `usage`; plain serializers commit in one round.
        action, usage, latency, raw, meta_calls = decide(
            model_client, task.goal, obs, history
        )
        step = StepRecord(
            step_index=i,
            action_kind=action.kind,
            action={"ref": action.ref, "text": action.text, "value": action.value},
            action_raw=raw,
            obs_tokens=obs_tokens,
            obs_bytes=obs.meta["bytes"],
            obs_element_count=obs.meta["element_count"],
            obs_has_image=obs.has_image,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            latency_s=latency,
        )
        loc = obs.ref_registry.resolve(action.ref) if action.ref else None
        pred_backend = str(loc["value"]) if loc else None
        ele_correct = bool(pred_backend and pred_backend in gt["backend_ids"])
        op_correct = action.kind == gt["kind"]
        f1 = action_f1(
            action.kind, action.text or action.value or "", gt["kind"], gt["value"]
        )
        step.correct = ele_correct
        step.meta = {
            "op_correct": op_correct,
            "action_f1": round(f1, 4),
            "pred_ref": action.ref,
            "pred_backend": pred_backend,
            "gt_backend_ids": sorted(gt["backend_ids"]),
            "gt_op": gt["kind"],
            "meta_calls": meta_calls,
        }

        episode.steps.append(step)
        history.append(gt["action_repr"])  # teacher forcing

    return episode


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_matrix(config: RunConfig, console: Console | None = None) -> Recorder:
    console = console or Console()
    from modalitybench.observations.loader import load_strategies

    load_strategies()
    source = build_task_source(config.tasks)
    recorder = Recorder(config.run_id, base_dir=config.results_dir)
    recorder.mark_started(config.model_dump())
    tasks = source.tasks()

    # Token counter is per-model (model-aware): exact only where we have an exact counter for
    # that model (Anthropic), else approx — so mixed-provider runs stay honestly labelled.
    from modalitybench.metrics.tokens import build_token_counter

    counters: dict[str, TokenCounter] = {}
    modes: dict[str, str] = {}

    def counter_for(model_name: str) -> TokenCounter:
        if model_name not in counters:
            tc = build_token_counter(model_name, config.token_count, console=console)
            counters[model_name] = tc
            modes[model_name] = tc.mode
        return counters[model_name]

    agg: dict[tuple[str, str], list[Any]] = {}
    for strategy, mcfg in config.cells():
        model_client = build_model_client(mcfg)
        model_name = model_client.model
        token_counter = counter_for(model_name)
        for task in tasks:
            if config.resume and recorder.is_completed(strategy, model_name, task.task_id):
                console.print(f"[dim]skip {strategy}/{model_name}/{task.task_id} (done)[/]")
                continue
            try:
                if source.is_live:
                    from modalitybench.agents.loop import evaluate_live

                    episode = evaluate_live(
                        source, task, strategy, model_client, token_counter, config.max_steps
                    )
                else:
                    episode = evaluate_offline(
                        source, task, strategy, model_client, token_counter, config.max_steps
                    )
                result = source.score(task, episode)
            except Exception as exc:  # noqa: BLE001 — one bad cell must not kill the run
                # Not recorded → resume retries it on the next run. Keeps a long live run
                # alive through a provider timeout / transient browser error.
                console.print(
                    f"[red]FAIL[/] {strategy}/{model_name}/{task.task_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            for step in episode.steps:
                recorder.record_step(strategy, model_name, task.task_id, step)
            recorder.record_episode(strategy, model_name, task.task_id, episode, result)
            agg.setdefault((strategy, model_name), []).append((episode, result))
            console.print(
                f"[green]OK[/] {strategy}/{model_name}/{task.task_id} "
                f"success={result.success} reward={result.reward:.2f} "
                f"obs_tok~{sum(s.obs_tokens for s in episode.steps)}"
            )

    _print_summary(console, agg, modes)
    console.print(f"\n[green]Results in[/] {recorder.dir}")
    return recorder


def _print_summary(console, agg, modes: dict[str, str]) -> None:
    if not agg:
        console.print("[yellow]Nothing ran (all cells already complete?).[/]")
        return
    table = Table(title="Run summary")
    table.add_column("strategy", style="cyan")
    table.add_column("model")
    table.add_column("tok mode")
    table.add_column("tasks", justify="right")
    table.add_column("success", justify="right")
    table.add_column("reward", justify="right")
    table.add_column("mean obs tok", justify="right")
    for (strategy, model), items in sorted(agg.items()):
        n = len(items)
        succ = sum(1 for _, r in items if r.success) / n
        reward = sum(r.reward for _, r in items) / n
        obs = [sum(s.obs_tokens for s in ep.steps) for ep, _ in items]
        mean_obs = sum(obs) / n if obs else 0
        table.add_row(
            strategy, model, modes.get(model, "?"), str(n),
            f"{succ:.2f}", f"{reward:.2f}", f"{mean_obs:.0f}"
        )
    console.print(table)
