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
    if action.kind == "request_text":
        return handler.request_text()
    if action.kind == "request_detail":
        return handler.request_detail(action.ref or "")
    return "(unknown query)"


def decide(
    client: ModelClient,
    goal: str,
    obs: Observation,
    history: list[str],
    *,
    max_meta_rounds: int = 6,
    prior_observations: list[str] | None = None,
) -> tuple[Action, Usage, float, str, list[str]]:
    """Get the next real action, running the tools-mode sub-loop if the obs exposes tools.

    Returns ``(action, usage, latency, raw, meta_calls)`` where ``meta_calls`` is the ordered
    list of query/escalation meta-actions invoked before committing (the router trace) — empty
    when the strategy committed on the first look.
    """
    tools_mode = bool(obs.meta.get("tools_mode"))
    handler = obs.meta.get("tool_handler")
    system = SYSTEM_PROMPT + (_META_HELP if tools_mode else "")
    transcript: list[str] = []
    meta_calls: list[str] = []
    total = Usage()
    latency = 0.0
    last_text = ""

    rounds = obs.meta.get("max_meta_rounds", max_meta_rounds) if tools_mode else 1
    for _ in range(rounds):
        blocks = build_user_blocks(
            goal, obs, history=history, prior_observations=prior_observations
        )
        if transcript:
            blocks.append(TextBlock(text="QUERY RESULTS:\n" + "\n".join(transcript)))
        resp = client.complete(system=system, blocks=blocks, tools=None)
        total = total.add(resp.usage)
        latency += resp.latency_s
        last_text = resp.text
        try:
            action = parse_action(resp.text)
        except ActionParseError:
            return Action(kind="done"), total, latency, last_text, meta_calls
        if tools_mode and handler is not None and action.kind in META_KINDS:
            result = _run_meta(handler, action)
            label = action.query or action.ref or ""
            transcript.append(f"> {action.kind}({label})\n{result}")
            meta_calls.append(action.kind)
            continue
        return action, total, latency, last_text, meta_calls

    # Ran out of exploration rounds without committing — stop cleanly.
    return Action(kind="done"), total, latency, last_text, meta_calls


def decide_with_guardrail(
    client: ModelClient,
    goal: str,
    obs: Observation,
    history: list[str],
    *,
    tried_actions: set[tuple[str, str | None, str | None]],
    graph: PageGraph | None = None,
    max_meta_rounds: int = 6,
    prior_observations: list[str] | None = None,
) -> tuple[Action, Usage, float, str, list[str]]:
    """Decide next action, intercepting redundant actions and auto-escalating to rich text.

    Live-loop only. The interception fires when the model re-proposes an action already in
    ``tried_actions`` — i.e. it is stuck re-clicking a dead element because the page did not
    change in response. That failure mode only exists when the agent's action drives the page
    (evaluate_live). Offline element selection is teacher-forced and force-advances every step,
    so a redundant-action loop never forms there; evaluate_offline calls plain ``decide``.
    """
    action, usage, latency, raw, meta_calls = decide(
        client, goal, obs, history,
        max_meta_rounds=max_meta_rounds,
        prior_observations=prior_observations,
    )

    if action.kind in {"click", "type", "select"} and action.ref:
        loc = obs.ref_registry.resolve(action.ref)
        loc_val = str(loc["value"]) if loc else action.ref
        action_key = (action.kind, loc_val, action.text or action.value)

        if action_key in tried_actions:
            # We hit a redundant action loop! Intercept and auto-escalate.
            from modalitybench.observations.adaptive import _text_leaves, _text_line
            from modalitybench.observations.base import TextBlock

            # 1. Mask out the redundant element ref
            new_blocks = []
            for block in obs.content_blocks:
                if isinstance(block, TextBlock):
                    # Filter out the line containing the redundant ref
                    lines = [line for line in block.text.splitlines() if f"[{action.ref}]" not in line]
                    new_blocks.append(TextBlock(text="\n".join(lines)))
                else:
                    new_blocks.append(block)
            obs.content_blocks = new_blocks

            # 2. Extract visible static text from graph (auto-escalation)
            escalation_text = ""
            if graph:
                leaves = _text_leaves(graph)
                if leaves:
                    escalation_text = "\n".join(_text_line(n) for n in leaves)

            # 3. Add warning and escalation content to observation
            warning_msg = (
                f"\n\n[SYSTEM NOTICE: Your previous action {action.kind}({action.ref}) was idempotent "
                f"or redundant. To assist you, the system has masked that element and automatically "
                f"retrieved the page's static text below. Select a DIFFERENT action.]\n"
            )
            if escalation_text:
                warning_msg += f"\nPAGE STATIC TEXT:\n{escalation_text}"
            else:
                warning_msg += "\n(No static text found on page)"

            obs.content_blocks.append(TextBlock(text=warning_msg))

            # 4. Remove ref from registry to prevent repeated lookup
            if action.ref in obs.ref_registry._by_ref:
                del obs.ref_registry._by_ref[action.ref]

            # 5. Re-decide with the enriched, masked observation
            action, second_usage, second_latency, raw, second_meta = decide(
                client, goal, obs, history,
                max_meta_rounds=max_meta_rounds,
                prior_observations=prior_observations,
            )
            usage = usage.add(second_usage)
            latency += second_latency
            meta_calls.extend(second_meta)
            meta_calls.append("guardrail_escalation")

            loc = obs.ref_registry.resolve(action.ref) if action.ref else None
            loc_val = str(loc["value"]) if loc else (action.ref or "")
            action_key = (action.kind, loc_val, action.text or action.value)

        tried_actions.add(action_key)

    return action, usage, latency, raw, meta_calls



def _capture(source, handle, strategy_name: str) -> PageGraph:
    """Capture the current page as a graph. Browser-free sources implement ``capture``;
    Playwright-backed sources fall back to ``graph_from_page`` over the live page."""
    from modalitybench.observations.dom_capture import graph_from_page

    if hasattr(source, "capture"):
        return source.capture(handle, screenshot=_needs_screenshot(strategy_name))
    return graph_from_page(handle.page, screenshot=_needs_screenshot(strategy_name))


def evaluate_live(
    source,
    task: Task,
    strategy_name: str,
    model_client: ModelClient,
    token_counter,
    max_steps: int,
    *,
    history_mode: str = "evict",
) -> Episode:
    from modalitybench.observations import get_strategy
    from modalitybench.observations.loader import load_strategies

    load_strategies()
    strat = get_strategy(strategy_name)
    handle = source.reset(task)
    episode = Episode(task_id=task.task_id, strategy=strategy_name, model=model_client.model)
    episode.meta["reward"] = 0.0
    episode.meta["history_mode"] = history_mode
    history: list[str] = []
    # accumulate mode re-sends every prior page's observation to the model each step — the
    # in-harness "MCP-style" accumulating baseline. evict (default) keeps only action strings,
    # so per-step context stays flat. This list stays empty in evict mode.
    prior_obs: list[str] = []

    tried_actions: set[tuple[str, str | None, str | None]] = set()
    try:
        for i in range(max_steps):
            graph: PageGraph = _capture(source, handle, strategy_name)
            obs = strat.observe(graph, task_text=handle.goal)
            obs_tokens = token_counter.count_blocks(obs.content_blocks, system=SYSTEM_PROMPT)
            action, usage, latency, raw, meta_calls = decide_with_guardrail(
                model_client, handle.goal, obs, history,
                tried_actions=tried_actions,
                graph=graph,
                prior_observations=prior_obs if history_mode == "accumulate" else None,
            )

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
            step.meta["meta_calls"] = meta_calls

            if action.kind == "done":
                # Preserve the answer payload so extraction sources can score it (the step
                # record only carries ref/text/value). None for tasks that don't return one.
                episode.meta["answer"] = action.answer
                episode.steps.append(step)
                break

            outcome = source.apply(handle, action, obs.ref_registry, graph)
            episode.meta["reward"] = outcome.reward
            step.meta.update({"info": outcome.info, "reward": outcome.reward})
            episode.steps.append(step)
            history.append(_describe_action(action))
            # In accumulate mode the model keeps seeing every page it has visited — this is
            # the buffer that grows the context (and the bill) superlinearly over a trajectory.
            if history_mode == "accumulate":
                prior_obs.append(f"[page after step {i + 1}]\n{obs.text()}")
            if outcome.terminated:
                break
    finally:
        source.close(handle)

    return episode
