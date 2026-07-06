"""Model-client abstraction.

The ``ModelClient`` protocol keeps the agent loop provider-agnostic; ``AnthropicClient`` is
the reference implementation (official SDK, default ``claude-opus-4-8``, adaptive thinking,
streaming). ``MockClient`` scripts responses for tests so the loop can be exercised with no
API calls or cost.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from modalitybench.observations.base import ContentBlock, ImageBlock, TextBlock, ToolSpec

DEFAULT_MODEL = "claude-opus-4-8"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def add(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens
            + other.cache_read_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
        }


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ModelResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None
    latency_s: float = 0.0
    model: str = ""


@runtime_checkable
class ModelClient(Protocol):
    model: str

    def complete(
        self,
        *,
        system: str,
        blocks: list[ContentBlock],
        tools: list[ToolSpec] | None = None,
    ) -> ModelResponse:
        ...


# ---------------------------------------------------------------------------
# Anthropic implementation
# ---------------------------------------------------------------------------


def _blocks_to_anthropic(blocks: list[ContentBlock]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for b in blocks:
        if isinstance(b, TextBlock):
            out.append({"type": "text", "text": b.text})
        elif isinstance(b, ImageBlock):
            out.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": b.media_type,
                        "data": base64.standard_b64encode(b.data).decode("ascii"),
                    },
                }
            )
    return out


def _tools_to_anthropic(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]


class AnthropicClient:
    """Reference model client using the official Anthropic SDK.

    Uses streaming + ``get_final_message()`` (timeout-safe for large ``max_tokens``) and
    records the full ``usage`` block. Adaptive thinking is on by default per the plan; set
    ``thinking=False`` to disable (recorded either way).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: int = 4096,
        thinking: bool = True,
        effort: str | None = "medium",
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.effort = effort
        if client is None:
            import anthropic  # imported lazily so the package installs without a key

            client = anthropic.Anthropic()
        self._client = client

    def _request_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens}
        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        return kwargs

    def complete(
        self,
        *,
        system: str,
        blocks: list[ContentBlock],
        tools: list[ToolSpec] | None = None,
    ) -> ModelResponse:
        content = _blocks_to_anthropic(blocks)
        kwargs = self._request_kwargs()
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = _tools_to_anthropic(tools)
        kwargs["messages"] = [{"role": "user", "content": content}]

        started = time.perf_counter()
        with self._client.messages.stream(**kwargs) as stream:
            msg = stream.get_final_message()
        latency = time.perf_counter() - started

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in msg.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=dict(block.input))
                )

        u = msg.usage
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        return ModelResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=getattr(msg, "stop_reason", None),
            latency_s=latency,
            model=self.model,
        )

    def count_tokens(self, *, system: str, blocks: list[ContentBlock]) -> int:
        """Exact input-token count for a prompt (used by the offline serializer bench)."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": _blocks_to_anthropic(blocks)}],
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.count_tokens(**kwargs)
        return resp.input_tokens


# ---------------------------------------------------------------------------
# OpenAI-compatible implementation (DeepSeek, Zhipu/GLM, and other compatible APIs)
# ---------------------------------------------------------------------------


def _blocks_to_openai(blocks: list[ContentBlock]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for b in blocks:
        if isinstance(b, TextBlock):
            out.append({"type": "text", "text": b.text})
        elif isinstance(b, ImageBlock):
            data = base64.standard_b64encode(b.data).decode("ascii")
            out.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{b.media_type};base64,{data}"},
                }
            )
    return out


def _tools_to_openai(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


class OpenAICompatibleClient:
    """Model client for OpenAI-compatible chat APIs (DeepSeek, Zhipu/GLM, …).

    These providers speak the OpenAI ``chat.completions`` schema, so one client covers them
    via ``base_url`` + key. Real usage (incl. any cache-hit tokens) is read from the response
    so cost/token accounting stays accurate per provider — even though observation *size* is
    counted with the local ``approx`` heuristic (see :func:`metrics.tokens.build_token_counter`).

    The agent loop parses actions from ``message.content`` (text), so ``tools`` are forwarded
    best-effort and any ``tool_calls`` are surfaced but not required.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        max_tokens: int = 4096,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url
        self._api_key = api_key
        self._api_key_env = api_key_env
        self._client = client  # transport is built lazily on first request

    def _transport(self) -> Any:
        """Build (once) the OpenAI SDK client — deferred so construction needs no key/dep."""
        if self._client is None:
            import os

            from openai import OpenAI  # lazy: optional dependency

            key = self._api_key or (
                os.environ.get(self._api_key_env) if self._api_key_env else None
            )
            self._client = OpenAI(
                base_url=self.base_url, api_key=key or os.environ.get("OPENAI_API_KEY")
            )
        return self._client

    def complete(
        self,
        *,
        system: str,
        blocks: list[ContentBlock],
        tools: list[ToolSpec] | None = None,
    ) -> ModelResponse:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": _blocks_to_openai(blocks)})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = _tools_to_openai(tools)

        started = time.perf_counter()
        resp = self._transport().chat.completions.create(**kwargs)
        latency = time.perf_counter() - started

        choice = resp.choices[0]
        text = choice.message.content or ""
        tool_calls: list[ToolCall] = []
        for tc in getattr(choice.message, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            args = getattr(fn, "arguments", "") if fn else ""
            try:
                import json

                parsed = json.loads(args) if isinstance(args, str) and args else (args or {})
            except (ValueError, TypeError):
                parsed = {"_raw": args}
            tool_calls.append(
                ToolCall(id=getattr(tc, "id", ""), name=getattr(fn, "name", ""), input=parsed)
            )

        u = getattr(resp, "usage", None)
        # DeepSeek/GLM report cache hits under provider-specific fields; read defensively.
        cache_read = getattr(u, "prompt_cache_hit_tokens", 0) if u else 0
        details = getattr(u, "prompt_tokens_details", None) if u else None
        if not cache_read and details is not None:
            cache_read = getattr(details, "cached_tokens", 0) or 0
        usage = Usage(
            input_tokens=getattr(u, "prompt_tokens", 0) or 0 if u else 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0 if u else 0,
            cache_read_input_tokens=cache_read or 0,
        )
        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            stop_reason=getattr(choice, "finish_reason", None),
            latency_s=latency,
            model=self.model,
        )


# ---------------------------------------------------------------------------
# Mock implementation (tests, dry runs)
# ---------------------------------------------------------------------------


class MockClient:
    """Deterministic client for tests. Supply a list of responses or a callable.

    ``responder`` receives ``(system, blocks, tools)`` and returns the assistant text
    (or a ``ModelResponse``). A list is consumed one item per ``complete`` call.
    """

    def __init__(
        self,
        responses: list[str | ModelResponse] | None = None,
        responder: Callable[..., str | ModelResponse] | None = None,
        model: str = "mock",
    ) -> None:
        self.model = model
        self._responses = list(responses or [])
        self._responder = responder
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        system: str,
        blocks: list[ContentBlock],
        tools: list[ToolSpec] | None = None,
    ) -> ModelResponse:
        self.calls.append({"system": system, "blocks": blocks, "tools": tools})
        if self._responder is not None:
            out = self._responder(system=system, blocks=blocks, tools=tools)
        elif self._responses:
            out = self._responses.pop(0)
        else:
            out = ModelResponse(text='{"action": "done"}', model=self.model)
        if isinstance(out, ModelResponse):
            return out
        # Rough usage estimate so recorder/cost paths have something to work with.
        approx_in = sum(len(b.text) for b in blocks if isinstance(b, TextBlock)) // 4
        return ModelResponse(
            text=out,
            usage=Usage(input_tokens=approx_in, output_tokens=len(out) // 4),
            stop_reason="end_turn",
            model=self.model,
        )
