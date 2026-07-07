"""Run configuration models.

A run is a matrix of ``strategies x models x tasks``. The config is YAML; pydantic validates
it. ``matrix.py`` expands the cross-product and executes each cell, skipping any already
recorded (resumability).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    name: str = "claude-opus-4-8"
    max_tokens: int = 4096
    thinking: bool = True
    effort: str | None = "medium"
    mock: bool = False  # use MockClient (no API calls) — for dry runs / CI
    # OpenAI-compatible providers (DeepSeek, GLM, …): override the API base URL and the env
    # var holding the key. Left None for Anthropic models (routed by name prefix).
    base_url: str | None = None
    api_key_env: str | None = None


class TaskSourceConfig(BaseModel):
    source: str  # "mind2web_offline" | "miniwob"
    # Source-specific options (e.g. split, task_ids, limit, subset).
    options: dict[str, Any] = Field(default_factory=dict)


class RunConfig(BaseModel):
    run_id: str
    strategies: list[str]
    models: list[ModelConfig] = Field(default_factory=lambda: [ModelConfig()])
    tasks: TaskSourceConfig
    max_steps: int = 15
    results_dir: str = "results"
    token_count: str = "exact"  # "exact" (count_tokens) | "approx" (local heuristic)
    resume: bool = True
    # Live-loop context lifecycle, swept as a matrix axis (live sources only): "evict"
    # (history = action strings only, flat per-step context) vs "accumulate" (re-send every
    # prior page's observation — the MCP-style baseline whose bill grows superlinearly). A
    # bare string is accepted and wrapped, so `history_mode: accumulate` still works.
    history_modes: list[str] = Field(default_factory=lambda: ["evict"])

    @model_validator(mode="before")
    @classmethod
    def _normalize_history(cls, data: Any) -> Any:
        # Accept legacy scalar `history_mode` and a scalar `history_modes`; normalize to a list.
        if isinstance(data, dict):
            if "history_modes" not in data and "history_mode" in data:
                data["history_modes"] = data.pop("history_mode")
            hm = data.get("history_modes")
            if isinstance(hm, str):
                data["history_modes"] = [hm]
        return data

    def cells(self) -> list[tuple[str, ModelConfig]]:
        """Cross-product of strategy x model (task + history-mode expansion in the runner)."""
        return [(s, m) for s in self.strategies for m in self.models]


def load_config(path: str | Path) -> RunConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return RunConfig.model_validate(data)
