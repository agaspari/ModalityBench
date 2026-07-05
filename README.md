# ModalityBench

A benchmark for **DOM / observation-reduction techniques** in agentic browser tasks.

Agentic browser systems spend most of their token budget on *observations* — raw page HTML
runs 40K–800K tokens, screenshots are expensive, and every reduction technique (accessibility
trees, pruned DOM, custom serializations, salience ranking, tool-based retrieval) trades
tokens against task success. ModalityBench runs a matrix of **observation strategies × tasks
× models**, records token / cost / latency / success, and renders an exportable dashboard.

> **Status:** Phases 1–5 are in place — capture, 7 DOM serializers + screenshots + salience +
> tools-mode, offline (Mind2Web) and live (MiniWoB++) task sources, the matrix runner, and the
> dashboard/exports. See `docs/ROADMAP.md` for what's next.

## Install

```bash
uv pip install -e .            # core: serializers + offline token bench + dashboard
uv pip install -e ".[browser]" # + Playwright / MiniWoB++ live harness
uv pip install -e ".[data]"    # + HuggingFace datasets (Mind2Web)
uv pip install -e ".[dev]"     # + pytest
```

Live tasks additionally need `playwright install chromium`. API calls need `ANTHROPIC_API_KEY`
(or `ant auth login`); use a `mock` model for a no-key dry run.

## Quickstart

```bash
mb list-strategies                       # what's registered
mb serialize-bench --approx              # token/byte comparison over bundled pages, no LLM
mb run configs/mind2web-mock.yaml        # end-to-end dry run (mock model, no API key)
mb dashboard mind2web-mock               # -> results/mind2web-mock/dashboard.html
mb export   mind2web-mock --format csv   # -> results/mind2web-mock/episodes_export.csv
```

A run writes `episodes.jsonl` / `steps.jsonl` under `results/<run_id>/`; `dashboard` and
`export` read those. Runs are **resumable** — re-running skips cells already recorded.

## CLI

| command | what it does |
| --- | --- |
| `mb list-strategies` | list registered observation strategies |
| `mb serialize-bench` | token/byte/element comparison of serializers over page snapshots (no LLM); `--approx` uses the local heuristic instead of the exact API count |
| `mb run <config.yaml>` | run a strategy × model × task matrix |
| `mb dashboard <run_id>` | build a self-contained `dashboard.html` (inlined Plotly) |
| `mb export <run_id> --format csv\|json` | dump episode rows for sharing |

## Strategies

All strategies share one captured `PageGraph`, so comparisons are apples-to-apples.

- **DOM serializers** — `afus`, `fct`, `flat_elements`, `pruned_html`, `body_html`,
  `raw_html`, `axtree` (accessibility tree). See `docs/serializers/` for the AFUS/FCT specs.
- **Screenshots** — `screenshot`, `screenshot_50pct`, `screenshot_gray`,
  `screenshot_50pct_gray` (downscale / grayscale variants).
- **Salience wrappers** — `salience_flat`, `salience_afus`, `salience_fct` (rank/keep the
  top-k task-relevant elements before serializing).
- **Tools-mode** — `tools_mode` (expose query tools instead of an upfront DOM dump).

## Configs

Three example runs live in `configs/`:

- `mind2web-mock.yaml` — offline Mind2Web element-selection, **mock** model (no API key);
  validates the whole capture → serialize → score → record pipeline.
- `mind2web-mini.yaml` — the same offline eval with Claude Opus 4.8 (needs a key).
- `miniwob-smoke.yaml` — a **live** MiniWoB++ run (needs `[browser]` + `MINIWOB_URL`).

## Architecture

The load-bearing abstraction is the **observation strategy** = *capture* (pull a page into a
shared `PageGraph`) + *serialize* (render it for the model) + *ref registry* (map element refs
back to actionable locators). One capture feeds many serializers, so token comparisons are
apples-to-apples. Salience and tools-mode are strategies too — the runner treats them all
uniformly.

Extending ModalityBench — new serializers, task sources, or model providers — is a drop-in:
see **`docs/extending.md`**. Design docs and the roadmap are in `docs/`.
