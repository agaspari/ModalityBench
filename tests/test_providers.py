"""Model-aware token counting + OpenAI-compatible client (DeepSeek / GLM).

No network: the OpenAI client is exercised with a stubbed transport, and counter selection
is asserted on the pure family/mode logic (the Anthropic exact path needs a key, so it's not
constructed here).
"""

from __future__ import annotations

from modalitybench.agents.model_client import OpenAICompatibleClient, Usage
from modalitybench.metrics.tokens import (
    build_token_counter,
    model_family,
    supports_exact_count,
)
from modalitybench.observations.base import TextBlock
from modalitybench.runner.config import ModelConfig
from modalitybench.runner.matrix import build_model_client


# -- model-aware counter selection ------------------------------------------


def test_model_family_and_exact_support():
    assert model_family("claude-opus-4-8") == "anthropic"
    assert model_family("deepseek-chat") == "deepseek"
    assert model_family("glm-4.6") == "zhipu"
    assert model_family("mock") == "mock"
    assert supports_exact_count("claude-sonnet-5")
    assert not supports_exact_count("deepseek-chat")
    assert not supports_exact_count("glm-4.6")


def test_build_token_counter_falls_back_for_non_anthropic():
    # DeepSeek has no exact counter wired -> approx even when exact is requested.
    tc = build_token_counter("deepseek-chat", "exact")
    assert tc.mode == "approx"
    assert tc.count_blocks([TextBlock(text="hello world")]) > 0
    # Anthropic + approx requested stays approx (no client constructed).
    assert build_token_counter("claude-opus-4-8", "approx").mode == "approx"


# -- OpenAI-compatible client (stubbed) -------------------------------------


class _FnCall:
    def __init__(self, name, arguments):
        self.function = type("F", (), {"name": name, "arguments": arguments})()
        self.id = "call_1"


class _Message:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason


class _Usage:
    prompt_tokens = 123
    completion_tokens = 7
    prompt_cache_hit_tokens = 100


class _Resp:
    def __init__(self, message):
        self.choices = [_Choice(message)]
        self.usage = _Usage()


class _Completions:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _Resp(_Message('{"action": "click", "ref": "e1"}'))


class _StubOpenAI:
    def __init__(self):
        self.chat = type("C", (), {"completions": _Completions()})()


def test_openai_client_maps_text_and_usage():
    stub = _StubOpenAI()
    client = OpenAICompatibleClient(model="deepseek-chat", client=stub)
    resp = client.complete(system="sys", blocks=[TextBlock(text="go")])
    assert resp.text == '{"action": "click", "ref": "e1"}'
    assert resp.model == "deepseek-chat"
    assert resp.usage == Usage(
        input_tokens=123, output_tokens=7, cache_read_input_tokens=100
    )
    # System + user message shape forwarded to the OpenAI schema.
    kwargs = stub.chat.completions.last_kwargs
    assert kwargs["model"] == "deepseek-chat"
    assert kwargs["messages"][0] == {"role": "system", "content": "sys"}
    assert kwargs["messages"][1]["content"] == [{"type": "text", "text": "go"}]


def test_openai_client_surfaces_tool_calls():
    stub = _StubOpenAI()
    stub.chat.completions.create = lambda **k: _Resp(
        _Message(None, tool_calls=[_FnCall("find", '{"query": "submit"}')])
    )
    client = OpenAICompatibleClient(model="glm-4.6", client=stub)
    resp = client.complete(system="", blocks=[TextBlock(text="x")])
    assert resp.text == ""  # content was null
    assert resp.tool_calls[0].name == "find"
    assert resp.tool_calls[0].input == {"query": "submit"}


# -- routing ----------------------------------------------------------------


def test_build_model_client_routes_deepseek_to_openai_compatible():
    client = build_model_client(ModelConfig(name="deepseek-chat"))
    assert isinstance(client, OpenAICompatibleClient)
    assert client.model == "deepseek-chat"


def test_build_model_client_custom_base_url_routes_openai_compatible():
    client = build_model_client(
        ModelConfig(name="my-local-model", base_url="http://localhost:8000/v1")
    )
    assert isinstance(client, OpenAICompatibleClient)
