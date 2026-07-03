"""Live agent loop: observe -> decide -> act -> repeat, against a live TaskSource.

Handles the tools-mode exploration sub-loop (the model may call ``outline``/``find``/``read``
meta-actions, answered from the captured graph, before committing to a real action). All
model usage — including meta rounds — is summed into the step record so token accounting
stays honest across strategies.
"""

from __future__ import annotations

from modalitybench.agents.actions import (
    META_KINDS,
    META_SCHEMA,
    Action,
    ActionParseError,
    parse_action,
)
from modalitybench.agents.model_client import ModelClient, Usage
from modalitybench.agents.prompts import SYSTEM_PROMPT, build_user_blocks
from modalitybench.observations.base import Observation, PageGraph, TextBlock
from modalitybench.tasks.base import Episode, StepRecord, Task

_META_HELP = "\nWhen the page is in tools mode you may also use:\n" + "\n".join(
    f"  - {d}" for d in META_SCHEMA.values()
)


def _needs_screenshot(strategy_name: str) -> bool:
    return strategy_name.startswith("screenshot")


def _describe_action(action: Action) -> str:
    bits = [action.kind]
    if action.ref:
        bits.append(action.ref)
    if action.text:
        bits.append(f'"{action.text}"')
    if action.value:
        bits.append(f"={action.value}")
    return " ".join(bits)


def _run_meta(handler, action: Action) -> str:
    if action.kind == "outline":
        return handler.outline()
    if action.kind == "find":
        return handler.find(action.query or action.text or "")
    if action.kind == "read":
        return handler.read(action.ref or "")
    return "(unknown query)"


def decide(
    client: ModelClient,
    goal: str,
    obs: Observation,
    history: list[str],
    *,
    max_meta_rounds: int = 6,
) -> tuple[Action, Usage, float, str]:
    """Get the next real action, running the tools-mode sub-loop if the obs exposes tools."""
    tools_mode = bool(obs.meta.get("tools_mode"))
    handler = obs.meta.get("tool_handler")
    system = SYSTEM_PROMPT + (_META_HELP if tools_mode else "")
    transcript: list[str] = []
    total = Usage()
    latency = 0.0
    last_text = ""

    rounds = max_meta_rounds if tools_mode else 1
    for _ in range(rounds):
        blocks = build_user_blocks(goal, obs, history=history)
        if transcript:
            blocks.append(TextBlock(text="QUERY RESULTS:\n" + "\n".join(transcript)))
        resp = client.complete(system=system, blocks=blocks, tools=None)
        total = total.add(resp.usage)
        latency += resp.latency_s
        last_text = resp.text
        try:
            action = parse_action(resp.text)
        except ActionParseError:
            return Action(kind="done"), total, latency, last_text
        if tools_mode and handler is not None and action.kind in META_KINDS:
            result = _run_meta(handler, action)
            label = action.query or action.ref or ""
            transcript.append(f"> {action.kind}({label})\n{result}")
            continue
        return action, total, latency, last_text

    # Ran out of exploration rounds without committing — stop cleanly.
    return Action(kind="done"), total, latency, last_text


def evaluate_live(
    source,
    task: Task,
    strategy_name: str,
    model_client: ModelClient,
    token_counter,
    max_steps: int,
) -> Episode:
    from modalitybench.observations import get_strategy
    from modalitybench.observations.dom_capture import graph_from_page
    from modalitybench.observations.loader import load_strategies

    load_strategies()
    strat = get_strategy(strategy_name)
    handle = source.reset(task)
    episode = Episode(task_id=task.task_id, strategy=strategy_name, model=model_client.model)
    episode.meta["reward"] = 0.0
    history: list[str] = []

    try:
        for i in range(max_steps):
            graph: PageGraph = graph_from_page(
                handle.page, screenshot=_needs_screenshot(strategy_name)
            )
            obs = strat.observe(graph, task_text=handle.goal)
            obs_tokens = token_counter.count_blocks(obs.content_blocks, system=SYSTEM_PROMPT)
            action, usage, latency, raw = decide(model_client, handle.goal, obs, history)

            step = StepRecord(
                step_index=i,
                action_kind=action.kind,
                action={"ref": action.ref, "text": action.text, "value": action.value},
                action_raw=raw,
                obs_tokens=obs_tokens,
                obs_bytes=obs.meta.get("bytes", 0),
                obs_element_count=obs.meta.get("element_count", 0),
                obs_has_image=obs.has_image,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens,
                cache_creation_input_tokens=usage.cache_creation_input_tokens,
                latency_s=latency,
            )

            if action.kind == "done":
                episode.steps.append(step)
                break

            outcome = source.apply(handle, action, obs.ref_registry, graph)
            episode.meta["reward"] = outcome.reward
            step.meta = {"info": outcome.info, "reward": outcome.reward}
            episode.steps.append(step)
            history.append(_describe_action(action))
            if outcome.terminated:
                break
    finally:
        source.close(handle)

    return episode
