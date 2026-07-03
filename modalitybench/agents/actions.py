"""Action space for the agent loop.

Actions reference elements by ``ref`` (the opaque id a serializer emitted). The executor
resolves the ref via the observation's ``RefRegistry`` before touching the page — so the
action space is identical across all serializers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class Action:
    """A parsed agent action.

    ``kind`` is one of the verbs below. ``ref`` / ``text`` / ``value`` / ``key`` / ``url``
    / ``answer`` are populated per verb (see :data:`ACTION_SCHEMA`).
    """

    kind: str
    ref: str | None = None
    text: str | None = None
    value: str | None = None
    key: str | None = None
    url: str | None = None
    answer: str | None = None
    query: str | None = None  # for tools-mode find()
    raw: dict[str, Any] | None = None  # original parsed payload, for logging


# Verb -> human description. Kept small and stable; new verbs are additive.
ACTION_SCHEMA: dict[str, str] = {
    "click": "click(ref): click the element with this ref",
    "type": "type(ref, text): focus the element and type text into it",
    "select": "select(ref, value): choose an option in a <select> by visible value",
    "scroll": "scroll(value): 'up' or 'down' to scroll the viewport",
    "press": "press(key): press a keyboard key, e.g. 'Enter'",
    "goto": "goto(url): navigate to a URL",
    "done": "done(answer): finish the episode, optionally returning an answer string",
}

# Meta-actions used only by tools-mode (query the page instead of dumping it). Handled by
# the agent loop, never sent to the page.
META_SCHEMA: dict[str, str] = {
    "outline": "outline(): list the page's landmark sections",
    "find": "find(query): search the page for elements matching a text/role query",
    "read": "read(ref): read the full details of one element or section by ref",
}

_VALID_KINDS = set(ACTION_SCHEMA) | set(META_SCHEMA)
META_KINDS = set(META_SCHEMA)


class ActionParseError(ValueError):
    """Raised when the model's output cannot be parsed into a valid Action."""


def parse_action(payload: str | dict[str, Any]) -> Action:
    """Parse a model action from a JSON string or dict into an :class:`Action`.

    Accepts either ``{"action": "click", "ref": "e3"}`` or a top-level ``{"click": {...}}``
    single-key form. Tolerant of a JSON object embedded in surrounding prose.
    """
    data = _coerce_json(payload)
    if not isinstance(data, dict):
        raise ActionParseError(f"expected a JSON object, got {type(data).__name__}")

    # Normalise to a flat {kind, ...args} shape.
    if "action" in data:
        kind = data.get("action")
        args = {k: v for k, v in data.items() if k != "action"}
    elif len(data) == 1 and next(iter(data)) in _VALID_KINDS:
        kind = next(iter(data))
        inner = data[kind]
        args = inner if isinstance(inner, dict) else {"ref": inner}
    else:
        kind = data.get("kind")
        args = {k: v for k, v in data.items() if k != "kind"}

    if kind not in _VALID_KINDS:
        raise ActionParseError(
            f"unknown action {kind!r}; valid: {', '.join(sorted(_VALID_KINDS))}"
        )

    return Action(
        kind=kind,
        ref=_as_str(args.get("ref")),
        text=_as_str(args.get("text")),
        value=_as_str(args.get("value")),
        key=_as_str(args.get("key")),
        url=_as_str(args.get("url")),
        answer=_as_str(args.get("answer")),
        query=_as_str(args.get("query")),
        raw=data,
    )


def _coerce_json(payload: str | dict[str, Any]) -> Any:
    if isinstance(payload, dict):
        return payload
    text = payload.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to extracting the first balanced {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ActionParseError(f"could not parse action JSON: {exc}") from exc
    raise ActionParseError("no JSON object found in model output")


def _as_str(v: Any) -> str | None:
    if v is None:
        return None
    return str(v)


def action_space_prompt() -> str:
    """Render the action space as a prompt fragment."""
    lines = ["Available actions (respond with a single JSON object):"]
    lines += [f"  - {desc}" for desc in ACTION_SCHEMA.values()]
    lines.append('Example: {"action": "click", "ref": "e3"}')
    return "\n".join(lines)
