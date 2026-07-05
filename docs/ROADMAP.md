# ModalityBench Roadmap

Running log of planned and possible future work. This project is large; items are captured
here as they surface so nothing is lost. Move an item into a phase when it's scheduled.

## Build phases (see plan file)

- [x] **Phase 1** — Skeleton + interfaces (protocols, recorder, config, CLI, docs).
- [x] **Phase 2** — DOM capture + 7 serializers + offline token bench (`mb serialize-bench`).
- [x] **Phase 3** — Offline Mind2Web element-selection eval + matrix runner.
- [x] **Phase 4** — Live harness + MiniWoB++, screenshots, salience, tools-mode.
      - Note: the live browser stack (capture, executor, `evaluate_live`) is validated with
        a real Chromium test against a local page. The **MiniWoB `env.step` reward path**
        still needs a first live run on a machine with `MINIWOB_URL` set — coordinate
        translation may need tuning (BrowserGym chrome offsets, `select` support). Verify the
        agent reaches reward on `click-test` before trusting live numbers.
      - Live capture refs only interactive elements (offline also refs text leaves). Harmless
        for acting; means live FCT/AFUS omit static-text rows that offline includes.
- [x] **Phase 5** — Dashboard + exports (`mb dashboard`, `mb export`): self-contained
      HTML with inlined Plotly (token/quality Pareto, token/cost/latency bars, per-cell
      table), CSV/JSON episode exports.
- [ ] **Phase 6** — Polish (README, extension guides, example configs).

## Task sources (drop-in via the `TaskSource` adapter interface)

- [ ] **WebShop** — self-hosted simulated e-commerce site, ~12k human instructions,
      automatic reward. Heavier setup (needs its dataset + local server). Strong candidate
      right after v1 since the user called it out.
- [ ] **WebArena** — realistic self-hosted sites (GitLab, shopping, forums, wiki) via Docker.
      Most realistic; heaviest infra. Best as a later drop-in.
- [ ] **WorkArena** — enterprise (ServiceNow) tasks; very large DOMs (40K–500K tokens),
      ideal stress test for reduction techniques.
- [ ] **Online-Mind2Web (live)** — the live-browsing variant of Mind2Web, complementing the
      offline snapshot eval shipped in Phase 3.

## Serializers / observation formats

- [ ] **EDAS** — spec undefined; *get the definition from the user* (see
      `docs/serializers/edas.md`), then implement.
- [ ] **D2Snap-style downsampling** — parameterized (k, l, m ratios) hierarchical DOM
      downsampling with an adaptive wrapper that hits a token budget (from "Beyond Pixels").
- [ ] **2D ASCII spatial map** — render element bounding boxes onto a character grid so the
      model sees spatial layout cheaply (the user's "2D ASCII map" idea).
- [ ] **Set-of-Marks (SoM)** — annotate a screenshot with numbered marks over interactive
      elements; pairs a compact image with a ref list.

## Strategies / pipeline

- [ ] **Playwright MCP snapshot** as a *strategy under test* (its fixed accessibility-tree
      format becomes one comparison point, not the driver).
- [ ] **Embedding-based salience** — replace/augment the heuristic salience scorer with an
      embedding similarity ranker against the task text.
- [ ] **Hybrid screenshot + DOM** — send a downscaled screenshot alongside a compact text
      serialization.
- [ ] **Observation diffing / caching** — between steps, send only the DOM delta rather than
      the full page; measure token savings vs. accuracy.

## Models / infrastructure

- [ ] **Multi-provider `ModelClient`** — OpenAI, Gemini via their SDKs, behind the existing
      protocol, so cross-model comparisons are possible.
- [ ] **Batches API** for offline evals (Mind2Web) — 50% cost reduction on non-interactive
      element-selection scoring.
- [ ] **Prompt-caching-aware accounting** — model the shared system-prompt / instruction
      prefix as cached and report effective cost.
- [ ] **Parallel episode execution** — run matrix cells concurrently (thread/async pool),
      respecting rate limits.

## Dashboard

- [ ] Hosted / live dashboard mode (beyond the self-contained HTML file).
- [ ] Statistical significance tests across runs (bootstrap CIs on success rate).
- [ ] Per-step trajectory viewer (inspect the actual observations a strategy produced).
