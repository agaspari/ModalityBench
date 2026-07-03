# ModalityBench

A benchmark for **DOM / observation-reduction techniques** in agentic browser tasks.

Agentic browser systems spend most of their token budget on *observations* — raw page HTML
runs 40K–800K tokens, screenshots are expensive, and every reduction technique (accessibility
trees, pruned DOM, custom serializations, salience ranking, tool-based retrieval) trades
tokens against task success. ModalityBench runs a matrix of **observation strategies × tasks
× models**, records token / cost / latency / success, and renders an exportable dashboard.

> **Status:** under active construction. Phase 1 (interfaces + skeleton) is in place. See the
> plan and `docs/ROADMAP.md` for what's next.

## Install

```bash
uv pip install -e .            # core: serializers + offline token bench + dashboard
uv pip install -e ".[browser]" # + Playwright / MiniWoB++ live harness
uv pip install -e ".[data]"    # + HuggingFace datasets (Mind2Web)
uv pip install -e ".[dev]"     # + pytest
```

## CLI

```bash
mb list-strategies             # list registered observation strategies
mb serialize-bench             # token/size comparison of serializers (no LLM calls)  [Phase 2]
mb run configs/<cfg>.yaml      # run a strategy x model x task matrix                 [Phase 3+]
mb dashboard <run_id>          # build a self-contained dashboard.html                [Phase 5]
mb export <run_id> --format csv
```

## Architecture

The load-bearing abstraction is the **observation strategy** = *capture* (pull a page into a
shared `PageGraph`) + *serialize* (render it for the model) + *ref registry* (map element refs
back to actionable locators). One capture feeds many serializers, so token comparisons are
apples-to-apples. Salience and tools-mode are strategies too — the runner treats them all
uniformly.

See the design docs in `docs/` (including the AFUS and FCT serializer specs) and the roadmap.
