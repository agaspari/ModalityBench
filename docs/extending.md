# Extending ModalityBench

Three drop-in points cover almost everything: **observation strategies** (new serializers /
reducers), **task sources** (new benchmarks), and **model clients** (new providers). Each is a
small protocol; the runner discovers implementations by name.

---

## 1. Add an observation strategy

A strategy turns a captured `PageGraph` into an `Observation` (content blocks + a
`RefRegistry` mapping the refs it emits back to actionable locators). This is the common case:
a new DOM serializer, a reducer/wrapper, or a screenshot variant.

**Contract** — implement `ObservationStrategy` (`modalitybench/observations/base.py`):

```python
class ObservationStrategy(Protocol):
    name: str
    def observe(self, graph: PageGraph, *, task_text: str | None = None) -> Observation: ...
```

`task_text` is the goal, provided so task-aware strategies (salience, retrieval) can rank
against it; task-agnostic serializers ignore it.

**Steps** (mirror `observations/serializers/flat_elements.py`, the simplest example):

1. Create a module, e.g. `observations/serializers/my_serializer.py`. Emit one line/section
   per node that has a `ref`, and let `_common.make_observation` assemble the `Observation`
   (it builds the `RefRegistry` from the graph and records token/byte/element stats):

   ```python
   from modalitybench.observations.registry import register_strategy
   from modalitybench.observations.serializers._common import make_observation

   class MySerializer:
       name = "my_serializer"
       def observe(self, graph, *, task_text=None):
           lines = [f"[{n.ref}] {n.role or n.tag}"
                    for n in graph.nodes() if n.ref and n.visible and n.is_interactive]
           return make_observation(graph, self.name, "\n".join(lines))

   @register_strategy("my_serializer")
   def _make() -> MySerializer:
       return MySerializer()
   ```

   The factory is **zero-arg**, so parameterized variants are just more factories binding
   different args (see how `screenshot_50pct` / `salience_afus` register several names).

2. Register the module for import so the decorator runs:
   - a plain DOM serializer → add its name to `_SERIALIZER_MODULES` in
     `observations/serializers/__init__.py`;
   - a wrapper / screenshot-style strategy (may need an optional dep) → add it to
     `_STRATEGY_MODULES` in `observations/loader.py` (that list is import-guarded, so a
     missing optional dependency degrades gracefully instead of crashing `list-strategies`).

3. Confirm and use it:

   ```bash
   mb list-strategies            # my_serializer should appear
   ```
   then reference `my_serializer` in a config's `strategies:` list.

**Refs are the contract.** Whatever refs your text mentions (`e1`, `e2`, …) must exist in the
`RefRegistry` so the executor can resolve them to a locator. `make_observation` populates the
registry from every graph node that has a `ref` (using its `locator`, or a `backend_id`
fallback), so as long as you only emit refs that are on the graph you're fine.

---

## 2. Add a task source

A `TaskSource` (`modalitybench/tasks/base.py`) yields `Task`s and scores an `Episode`. Two
shapes, distinguished by `is_live`:

- **Offline** (`is_live = False`) — implement `OfflineTaskSource`: `snapshots(task)` yields
  `(PageGraph, ground_truth)` per step for element-selection scoring, no browser. Model on
  `tasks/mind2web_offline.py`.
- **Live** (`is_live = True`) — provision a real page the harness captures from and acts on,
  and read terminal reward in `score`. Model on `tasks/miniwob.py`.

**Minimal offline source:**

```python
class MySource:
    name = "my_source"
    is_live = False

    def __init__(self, **options):   # options come straight from the config's `options:` map
        ...

    def tasks(self) -> list[Task]:
        return [Task(task_id="t1", goal="do the thing", source=self.name)]

    def snapshots(self, task) -> list[tuple[PageGraph, dict]]:
        return [(graph, {"ref": "e5"})]   # ground truth per step

    def score(self, task, episode) -> TaskResult:
        return TaskResult(success=..., reward=..., metrics={"element_accuracy": ...})
```

**Wire it in** — add a branch to `build_task_source` in `runner/matrix.py`:

```python
if src == "my_source":
    from modalitybench.tasks.my_source import MySource
    return MySource(**opts)
```

Then select it in a config:

```yaml
tasks:
  source: my_source
  options: { split: dev, limit: 20 }   # forwarded as **kwargs to __init__
```

The `metrics` you return show up in exports (`metric_*` columns) and, if you emit
`element_accuracy`, the dashboard uses it as the quality axis automatically.

---

## 3. Add a model client / provider

The agent loop is provider-agnostic via `ModelClient`
(`modalitybench/agents/model_client.py`):

```python
class ModelClient(Protocol):
    model: str
    def complete(self, *, system: str, blocks: list[ContentBlock],
                 tools: list[ToolSpec] | None = None) -> ModelResponse: ...
```

`blocks` are `TextBlock` / `ImageBlock`; convert them to your provider's format (see
`_blocks_to_anthropic`). Populate `ModelResponse.usage` with real token counts — cost,
token accounting, and the dashboard all read it. `AnthropicClient` is the reference
implementation; `MockClient` scripts responses for tests and dry runs.

**Wire it in** — extend `build_model_client` in `runner/matrix.py` to construct your client
(e.g. keyed on the `model` name prefix), and it flows through the existing config unchanged.

---

## Testing your extension

Every module has tests under `tests/`; add one alongside. Fast patterns:

- Strategies: build a `PageGraph` with the `conftest.py` fixtures (`checkout_graph`,
  `search_graph`, `form_graph`) and assert on `observe(...).text()` / the ref registry.
- Task sources / clients: use `MockClient` so no API calls or cost are incurred.

```bash
uv pip install -e ".[dev]"
pytest -q
```
