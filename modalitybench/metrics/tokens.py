"""Token accounting.

Two paths, by design:

* **Exact** — ``anthropic`` ``count_tokens`` against the target model. This is the source of
  truth for the offline serializer bench where per-serializer counts must be comparable and
  correct. Requires a client/API key.
* **Local approximate** — a fast char-based heuristic used when no client is available (or
  when speed matters more than precision). Clearly labelled ``approx`` in output; it is only
  valid for *relative* comparison, never as a real Claude token count.

We deliberately avoid ``tiktoken``: it is OpenAI's tokenizer and materially miscounts Claude
tokens, so it would give confidently-wrong absolute numbers.

Image tokens use Anthropic's ``(w*h)/750`` estimate for pre-flight budgeting; live runs use
the real ``usage`` returned by the API.
"""

from __future__ import annotations

import re
from typing import Any

from modalitybench.observations.base import ContentBlock, ImageBlock, TextBlock

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens_local(text: str) -> int:
    """Fast, approximate token estimate.

    Blends a word/punctuation split with a chars/4 heuristic. Consistent and cheap; use only
    for relative comparison across serializers, not as a true Claude count.
    """
    if not text:
        return 0
    pieces = len(_TOKEN_RE.findall(text))
    char_based = len(text) / 4.0
    return max(1, round((pieces + char_based) / 2))


def estimate_image_tokens(width: int, height: int) -> int:
    """Anthropic's pre-flight estimate for an image: ~(w*h)/750 tokens."""
    return max(1, round((width * height) / 750))


def observation_text(blocks: list[ContentBlock]) -> str:
    return "\n".join(b.text for b in blocks if isinstance(b, TextBlock))


class TokenCounter:
    """Counts tokens for observation content, using the exact path when possible.

    ``client`` is an :class:`AnthropicClient` (or anything exposing ``count_tokens``). When
    absent, falls back to the local heuristic. ``mode`` in the returned meta is ``"exact"``
    or ``"approx"`` so downstream reporting can label the numbers honestly.
    """

    def __init__(self, client: Any | None = None) -> None:
        self.client = client

    @property
    def mode(self) -> str:
        return "exact" if self.client is not None else "approx"

    def count_blocks(self, blocks: list[ContentBlock], *, system: str = "") -> int:
        text = observation_text(blocks)
        image_tokens = sum(
            estimate_image_tokens(b.width or 0, b.height or 0)
            for b in blocks
            if isinstance(b, ImageBlock) and b.width and b.height
        )
        if self.client is not None and text:
            try:
                text_tokens = self.client.count_tokens(
                    system=system,
                    blocks=[b for b in blocks if isinstance(b, TextBlock)],
                )
                return text_tokens + image_tokens
            except Exception:
                # Network / auth failure — degrade to the local estimate rather than crash.
                pass
        return estimate_tokens_local(text) + image_tokens

    def count_text(self, text: str) -> int:
        if self.client is not None and text:
            try:
                return self.client.count_tokens(system="", blocks=[TextBlock(text=text)])
            except Exception:
                pass
        return estimate_tokens_local(text)


# ---------------------------------------------------------------------------
# Model-aware counter selection
# ---------------------------------------------------------------------------


def model_family(model: str) -> str:
    """Coarse provider family from a model name — drives tokenizer/counter selection."""
    m = (model or "").lower()
    if m.startswith("claude") or m.startswith("anthropic"):
        return "anthropic"
    if m.startswith("deepseek"):
        return "deepseek"
    if m.startswith("glm") or m.startswith("zhipu"):
        return "zhipu"
    if m == "mock":
        return "mock"
    return "other"


def supports_exact_count(model: str) -> bool:
    """Whether we have an *exact* token counter for this model.

    Only Anthropic exposes a hosted ``count_tokens`` endpoint we use here. Other families
    fall back to the ``approx`` heuristic for observation size — their real *billed* tokens
    still come back in each response's ``usage``. To upgrade a family to exact, wire its
    local tokenizer into :func:`build_token_counter` (see comment there).
    """
    return model_family(model) == "anthropic"


def build_token_counter(model: str, mode: str, *, console: Any | None = None) -> "TokenCounter":
    """Construct the right :class:`TokenCounter` for ``model`` given the requested ``mode``.

    Exact counting is only wired for Anthropic (hosted ``count_tokens``); every other family
    degrades to the local ``approx`` heuristic and says so via ``TokenCounter.mode``. This
    keeps per-model runs honest — a DeepSeek run is never mislabelled as exact-Claude tokens.

    Extension seam: to make, say, DeepSeek exact, load its local tokenizer and pass a small
    adapter exposing ``count_tokens(system, blocks)`` as the counter's ``client`` below.
    """
    if mode == "exact" and supports_exact_count(model):
        try:
            from modalitybench.agents.model_client import AnthropicClient

            return TokenCounter(AnthropicClient(model=model))
        except Exception as exc:  # missing key / SDK — degrade rather than crash
            if console is not None:
                console.print(
                    f"[yellow]Exact token counting unavailable for {model} ({exc}); "
                    f"using approx.[/]"
                )
    return TokenCounter(None)
