"""WebArena live task source (Zhou et al., ICLR 2024; github.com/web-arena-x/webarena).

WebArena is 812 long-horizon tasks over five self-hosted, dockerized real sites (an
OneStopShop storefront + its admin, a Reddit-like forum, GitLab, a Wikipedia mirror, and an
OpenStreetMap instance). Each task carries a natural-language ``intent`` and a machine-checkable
``eval`` spec; success is scored by matching the agent's final answer, the URL it ended on,
and/or the live HTML of one or more pages.

Like WebShop, WebArena is plain HTTP: we point Playwright at the ``start_url`` and let
ModalityBench's own capture + serializers + :class:`~modalitybench.agents.executor.ActionExecutor`
drive it — no BrowserGym. The agent's ``stop [answer]`` maps onto our ``done(answer=...)``.

**Setup (infra, not code).** Stand up the WebArena docker images per their README, export the
site base URLs, and generate ``test.json`` (their ``generate_test_data.py`` substitutes the
``__SHOPPING__`` etc. placeholders). We accept either the raw templated configs or a
pre-substituted file — :func:`substitute_urls` fills any remaining placeholders from the same
env vars WebArena uses (``SHOPPING``, ``SHOPPING_ADMIN``, ``REDDIT``, ``GITLAB``,
``WIKIPEDIA``, ``MAP``, ``HOMEPAGE``). Point the source at the config file and the auth-state
dir::

    export SHOPPING='http://localhost:7770' GITLAB='http://localhost:8023' ...
    # options.config_file: path to WebArena test.json (an array of task configs)
    # options.auth_dir:    dir holding the .auth/*_state.json storage-state files

**Scoring lives here, not in the live page.** The runner closes the trajectory browser before
``score`` is called, so we never lean on it. String and URL matching are pure functions over
the final answer + the URL recorded during the loop; ``program_html`` opens a *fresh*
authenticated context in :meth:`score` and re-fetches the eval URLs. WebArena's success
criteria are server-side (a created issue, an updated cart, a changed profile), so a fresh
context with the same storage-state observes the same state the trajectory produced. Purely
client-side state would be missed — documented, and none of the WebArena checks rely on it.

**Judged + partially supported.** ``fuzzy_match`` and the "N/A justification" (``ua_match``)
checks are graded by an :class:`LLMJudge` when the run sets ``judge_model`` (WebArena's own
prompts, verbatim); without one they raise :class:`UnsupportedEval`. ``func:`` program_html
locators/urls need WebArena's Python helpers and always raise. Every :class:`UnsupportedEval`
surfaces as ``TaskResult.error`` rather than a silent pass — so a run over a filtered task
subset stays honest about what it actually scored.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from modalitybench.agents.actions import Action
from modalitybench.observations.base import PageGraph, RefRegistry
from modalitybench.tasks.base import Episode, LiveOutcome, Task, TaskResult

# WebArena placeholder token -> the env var holding that site's base URL (their native names).
_SITE_ENV = {
    "__SHOPPING__": "SHOPPING",
    "__SHOPPING_ADMIN__": "SHOPPING_ADMIN",
    "__REDDIT__": "REDDIT",
    "__GITLAB__": "GITLAB",
    "__WIKIPEDIA__": "WIKIPEDIA",
    "__MAP__": "MAP",
    "__HOMEPAGE__": "HOMEPAGE",
}

# WebArena joins alternative reference answers / URLs with this literal separator.
_OR = " |OR| "
# start_url can open several tabs joined by this; we drive the first (multi-tab unsupported).
_AND = " |AND| "


class UnsupportedEval(RuntimeError):
    """An eval feature we haven't implemented (func: locators)."""


# ---------------------------------------------------------------------------
# LLM judge for fuzzy_match / ua_match (WebArena's llm_fuzzy_match / llm_ua_match)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = "You are a helpful assistant"


def _fuzzy_prompt(question: str, reference: str, pred: str) -> str:
    return (
        "Help a teacher to grade the answer of a student given a question. Keep in mind that "
        "the student may use different phrasing or wording to answer the question. The goal is "
        "to evaluate whether the answer is semantically equivalent to the reference answer.\n"
        f"question: {question}\nreference answer: {reference}\n"
        "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
        f"student answer: {pred}\n"
        "Conclude the judgement by correct/incorrect/partially correct."
    )


def _ua_prompt(question: str, reference: str, pred: str) -> str:
    return (
        f"task: {question}\nactual unachievable reason: {reference}\n"
        f"reported unachievable reason: {pred}\n"
        "The task described above is inherently unachievable due to the reason specified under "
        "'actual unachievable reason'. An individual previously attempted this task and was "
        "unable to complete it. They provided a reason for their failure, which is listed under "
        "'reported unachievable reason'. Your role is to review both the actual and reported "
        "reasons. Determine if the reported reason aligns with the actual reason, even if "
        "implicitly. If the stated reason is in line with the actual reason, respond with "
        "'same'. Otherwise, respond with 'different'."
    )


def _fuzzy_parse(response: str) -> float:
    r = response.lower()
    if "partially correct" in r or "incorrect" in r:
        return 0.0
    return 1.0 if "correct" in r else 0.0


def _ua_parse(response: str) -> float:
    r = response.lower()
    if "different" in r:
        return 0.0
    return 1.0 if "same" in r else 0.0


class LLMJudge:
    """WebArena's semantic grader for ``fuzzy_match`` / ``ua_match`` over any ``ModelClient``.

    Faithful to WebArena's prompts and its parse rule (a "partially correct"/"incorrect" verdict
    scores 0.0, "correct" scores 1.0; "different"/"same" for ua). WebArena grades at
    temperature 0 with gpt-4; we reuse whatever model the run configures via ``judge_model``,
    which our clients don't expose a temperature knob for — a documented, minor divergence."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def _ask(self, prompt: str) -> str:
        from modalitybench.observations.base import TextBlock

        resp = self.client.complete(
            system=_JUDGE_SYSTEM, blocks=[TextBlock(text=prompt)], tools=None
        )
        return resp.text or ""

    def fuzzy_match(self, question: str, reference: str, pred: str) -> float:
        return _fuzzy_parse(self._ask(_fuzzy_prompt(question, reference, pred)))

    def ua_match(self, question: str, reference: str, pred: str) -> float:
        return _ua_parse(self._ask(_ua_prompt(question, reference, pred)))


@dataclass
class _Handle:
    pw: Any
    browser: Any
    context: Any
    page: Any
    goal: str
    task_id: str
    final_url: str = ""


class WebArenaSource:
    name = "webarena"
    is_live = True

    def __init__(
        self,
        *,
        config_file: str | None = None,
        config_dir: str | None = None,
        task_ids: list[int] | None = None,
        sites: list[str] | None = None,
        auth_dir: str | None = None,
        headless: bool = True,
        limit: int | None = None,
        judge_model: str | None = None,
    ) -> None:
        # Source of task configs: a single WebArena ``test.json`` array (config_file) or a
        # directory of ``<id>.json`` files (config_dir). Filter by task_ids / sites; cap with limit.
        self.config_file = config_file
        self.config_dir = config_dir
        self.task_ids = set(task_ids) if task_ids else None
        self.sites = set(sites) if sites else None
        self.auth_dir = auth_dir
        self.headless = headless
        self.limit = limit
        # Model that grades fuzzy_match / ua_match tasks (built lazily on first use). None leaves
        # those tasks as UnsupportedEval rather than silently wrong.
        self.judge_model = judge_model
        self._judge: LLMJudge | None = None
        # Final URL per task, recorded during the live loop so score() (which runs after the
        # trajectory browser is closed) can do url_match / program_html "last" without it.
        # The runner is sequential (one cell at a time), so a plain dict is safe.
        self._final_url: dict[str, str] = {}

    # -- TaskSource API ------------------------------------------------------

    def tasks(self) -> list[Task]:
        out: list[Task] = []
        for cfg in self._load_configs():
            tid = cfg.get("task_id")
            if self.task_ids is not None and tid not in self.task_ids:
                continue
            cfg_sites = cfg.get("sites") or []
            if self.sites is not None and not (self.sites & set(cfg_sites)):
                continue
            start_url = substitute_urls(str(cfg.get("start_url", ""))).split(_AND)[0].strip()
            out.append(
                Task(
                    task_id=f"webarena#{tid}",
                    goal=str(cfg.get("intent", "")),
                    source=self.name,
                    meta={
                        "task_id": tid,
                        "sites": cfg_sites,
                        "start_url": start_url,
                        "require_login": bool(cfg.get("require_login")),
                        "storage_state": cfg.get("storage_state"),
                        "eval": _resolve_eval(cfg.get("eval", {})),
                    },
                )
            )
        return out[: self.limit] if self.limit else out

    def reset(self, task: Task) -> _Handle:
        from playwright.sync_api import sync_playwright

        start_url = str(task.meta["start_url"])
        if not start_url:
            raise RuntimeError(
                f"{task.task_id} has no start_url — are the site env vars (SHOPPING, GITLAB, "
                "…) exported so placeholders resolve? See configs/webarena-smoke.yaml."
            )
        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=self.headless)
        context = browser.new_context(storage_state=self._storage_state(task.meta.get("storage_state")))
        page = context.new_page()
        page.goto(start_url)
        _settle(page)
        handle = _Handle(
            pw=pw, browser=browser, context=context, page=page,
            goal=task.goal, task_id=task.task_id, final_url=_page_url(page) or start_url,
        )
        self._final_url[task.task_id] = handle.final_url
        return handle

    def apply(
        self, handle: _Handle, action: Action, registry: RefRegistry, graph: PageGraph
    ) -> LiveOutcome:
        from modalitybench.agents.executor import ActionExecutor, ExecutionError

        try:
            ActionExecutor(handle.page).execute(action, registry)
        except ExecutionError as exc:
            return LiveOutcome(reward=0.0, terminated=False, info=str(exc))
        _settle(handle.page)
        handle.final_url = _page_url(handle.page) or handle.final_url
        self._final_url[handle.task_id] = handle.final_url
        # WebArena has no in-page reward and no page-signalled termination: the episode ends
        # only when the agent emits done (handled by the loop) or max_steps is hit.
        return LiveOutcome(reward=0.0, terminated=False, info=handle.final_url)

    def close(self, handle: _Handle) -> None:
        for obj, meth in ((handle.context, "close"), (handle.browser, "close"), (handle.pw, "stop")):
            try:
                if obj is not None:
                    getattr(obj, meth)()
            except Exception:
                pass

    def score(self, task: Task, episode: Episode) -> TaskResult:
        spec = task.meta.get("eval", {})
        eval_types = spec.get("eval_types", [])
        answer = episode.meta.get("answer") or ""
        final_url = self._final_url.get(task.task_id, task.meta.get("start_url", ""))
        metrics: dict[str, float] = {}
        score = 1.0
        error: str | None = None
        try:
            if "string_match" in eval_types:
                s = string_match_score(
                    spec.get("reference_answers", {}), str(answer),
                    intent=task.goal, string_note=spec.get("string_note", ""),
                    judge=self._get_judge(),
                )
                metrics["string_match"] = s
                score *= s
            if "url_match" in eval_types:
                s = url_match_score(spec.get("reference_url", ""), final_url)
                metrics["url_match"] = s
                score *= s
            if "program_html" in eval_types:
                s = self._program_html_score(task, spec.get("program_html", []), final_url)
                metrics["program_html"] = s
                score *= s
        except UnsupportedEval as exc:
            error = f"unsupported eval: {exc}"
            score = 0.0

        return TaskResult(
            success=error is None and score >= 1.0,
            reward=round(score, 4),
            metrics={
                **metrics,
                "n_steps": float(len(episode.steps)),
                "final_url_len": float(len(final_url)),
            },
            error=error,
        )

    # -- internals -----------------------------------------------------------

    def _get_judge(self) -> "LLMJudge | None":
        """Build the fuzzy/ua grader once, reusing the runner's provider-routing. None if the
        run configured no ``judge_model`` (fuzzy tasks then surface as UnsupportedEval)."""
        if self.judge_model is None:
            return None
        if self._judge is None:
            from modalitybench.runner.config import ModelConfig
            from modalitybench.runner.matrix import build_model_client

            self._judge = LLMJudge(build_model_client(ModelConfig(name=self.judge_model)))
        return self._judge

    def _load_configs(self) -> list[dict[str, Any]]:
        if self.config_file:
            data = json.loads(Path(self.config_file).read_text(encoding="utf-8"))
            return data if isinstance(data, list) else [data]
        if self.config_dir:
            out = []
            for p in sorted(Path(self.config_dir).glob("*.json")):
                out.append(json.loads(p.read_text(encoding="utf-8")))
            return out
        raise RuntimeError("WebArenaSource needs options.config_file or options.config_dir")

    def _storage_state(self, rel: str | None) -> str | None:
        """Resolve a config's storage_state path against auth_dir/env; None if absent/missing."""
        if not rel:
            return None
        base = self.auth_dir or os.environ.get("WEBARENA_AUTH_DIR") or "."
        # storage_state values look like "./.auth/shopping_state.json"; keep just the filename
        # under auth_dir if that's how the user laid them out, else honour the given path.
        cand = Path(base) / Path(rel).name
        if cand.exists():
            return str(cand)
        direct = Path(rel)
        return str(direct) if direct.exists() else None

    def _program_html_score(
        self, task: Task, entries: list[dict[str, Any]], final_url: str
    ) -> float:
        """Re-fetch each program_html target in a fresh authenticated context and check it.

        WebArena's checks are server-side, so a new context with the same storage_state sees the
        state the trajectory produced (see module docstring). ``func:`` urls/locators need
        WebArena's Python helpers and raise :class:`UnsupportedEval`."""
        if not entries:
            return 1.0
        from playwright.sync_api import sync_playwright

        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=self.headless)
        context = browser.new_context(
            storage_state=self._storage_state(task.meta.get("storage_state"))
        )
        page = context.new_page()
        try:
            score = 1.0
            for entry in entries:
                url = str(entry.get("url", ""))
                if url.startswith("func:"):
                    raise UnsupportedEval("func: program_html url")
                target = final_url if url in ("", "last") else substitute_urls(url)
                page.goto(target)
                _settle(page)
                content = _extract_locator(page, str(entry.get("locator", "")))
                score *= content_check(content, entry.get("required_contents", {}))
            return score
        finally:
            for obj, meth in ((context, "close"), (browser, "close"), (pw, "stop")):
                try:
                    getattr(obj, meth)()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Pure evaluators + config helpers (unit-tested without a browser)
# ---------------------------------------------------------------------------


def substitute_urls(text: str, env: dict[str, str] | None = None) -> str:
    """Replace ``__SHOPPING__`` etc. placeholders with the site base URLs from the environment.

    A no-op on already-substituted strings. Missing env vars are left as-is so the failure is
    a visible unresolved placeholder rather than a silently wrong URL."""
    env = env if env is not None else dict(os.environ)
    for token, var in _SITE_ENV.items():
        if token in text:
            val = env.get(var)
            if val:
                text = text.replace(token, val.rstrip("/"))
    return text


def _resolve_eval(spec: dict[str, Any]) -> dict[str, Any]:
    """Substitute URL placeholders inside an eval spec (reference_url + program_html urls)."""
    out = dict(spec)
    if out.get("reference_url"):
        out["reference_url"] = substitute_urls(str(out["reference_url"]))
    if out.get("program_html"):
        out["program_html"] = [
            {**e, "url": substitute_urls(str(e.get("url", "")))} for e in out["program_html"]
        ]
    return out


def clean_answer(answer: str) -> str:
    """WebArena's normalization: strip, drop one layer of surrounding quotes, lowercase."""
    a = str(answer).strip()
    if len(a) >= 2 and a[0] == a[-1] and a[0] in ("'", '"'):
        a = a[1:-1]
    return a.lower()


def _tokenize(s: str) -> list[str]:
    # Dependency-free stand-in for nltk.word_tokenize: WebArena tokenizes to stop a single-word
    # reference like "0" from matching inside "100". Word-character runs are a close-enough proxy.
    return re.findall(r"\w+", s.lower())


def string_match_score(
    reference_answers: dict[str, Any],
    pred: str,
    *,
    intent: str = "",
    string_note: str = "",
    judge: "LLMJudge | None" = None,
) -> float:
    """Score a final answer against a WebArena ``reference_answers`` spec (product of criteria).

    ``exact_match`` / ``must_include`` are pure. ``fuzzy_match`` is graded by ``judge`` (an
    :class:`LLMJudge`) exactly as WebArena does: each reference is checked with the semantic
    grader, and the ``"N/A"`` unachievable sentinel short-circuits to 1.0 on a literal ``n/a``
    answer, else defers to the ``ua_match`` justification grader. Without a ``judge`` those LLM
    paths raise :class:`UnsupportedEval` (surfaced as ``TaskResult.error``, never a silent pass)."""
    clean_pred = clean_answer(pred)
    score = 1.0
    for approach, value in reference_answers.items():
        if approach == "exact_match":
            score *= float(clean_pred == clean_answer(str(value)))
        elif approach == "must_include":
            for ref in value:
                cref = clean_answer(str(ref))
                if len(_tokenize(cref)) == 1:
                    score *= float(cref in _tokenize(clean_pred))
                else:
                    score *= float(cref in clean_pred)
        elif approach == "fuzzy_match":
            if value == "N/A":
                if clean_pred == "n/a":
                    score *= 1.0
                elif judge is not None:
                    score *= judge.ua_match(intent, string_note, pred)
                else:
                    raise UnsupportedEval("N/A ua_match needs a judge model")
            elif judge is not None:
                for reference in value:
                    score *= judge.fuzzy_match(intent, str(reference), pred)
            else:
                raise UnsupportedEval("fuzzy_match needs a judge model")
        else:
            raise UnsupportedEval(f"reference_answers approach {approach!r}")
    return score


def _parse_url(url: str) -> tuple[str, dict[str, list[str]]]:
    p = urlparse(url)
    return p.netloc + p.path.rstrip("/"), parse_qs(p.query)


def url_match_score(reference_url: str, pred_url: str) -> float:
    """WebArena "GOLD in PRED" URL match: a reference base path is a substring of the predicted
    one, and every reference query key has one of its values present in the prediction.

    ``reference_url`` may list alternatives joined by ``|OR|``; any satisfying one scores 1.0."""
    if not reference_url:
        return 1.0
    pred_base, pred_q = _parse_url(pred_url)
    for ref in reference_url.split(_OR):
        ref_base, ref_q = _parse_url(ref.strip())
        if ref_base and ref_base not in pred_base:
            continue
        ok = True
        for key, vals in ref_q.items():
            if not any(v in pred_q.get(key, []) for v in vals):
                ok = False
                break
        if ok:
            return 1.0
    return 0.0


def content_check(content: str, required_contents: dict[str, Any]) -> float:
    """Check fetched page content against a program_html ``required_contents`` spec.

    ``must_include`` is a substring test per required string (``|OR|`` alternatives allowed);
    ``exact_match`` compares normalized whole content. Product across all requirements."""
    if "must_include" in required_contents:
        score = 1.0
        for req in required_contents["must_include"]:
            alts = str(req).split(_OR)
            score *= float(any(a in content for a in alts))
        return score
    if "exact_match" in required_contents:
        return float(clean_answer(str(required_contents["exact_match"])) == clean_answer(content))
    return 0.0


# ---------------------------------------------------------------------------
# Playwright shims (guarded so they never crash the loop)
# ---------------------------------------------------------------------------


def _extract_locator(page: Any, locator: str) -> str:
    """Resolve a program_html locator to text. Empty = whole page; ``document.`` = JS eval;
    otherwise a Playwright selector. ``func:`` needs WebArena helpers (unsupported)."""
    loc = locator.strip()
    if loc == "":
        return _body_text(page) or _page_content(page)
    if loc.startswith("func:"):
        raise UnsupportedEval("func: program_html locator")
    if loc.startswith("document.") or loc.startswith("[...document"):
        try:
            return str(page.evaluate(f"() => {loc}"))
        except Exception:
            return ""
    try:
        return page.locator(loc).inner_text()
    except Exception:
        return ""


def _settle(page: Any) -> None:
    try:
        page.wait_for_load_state("networkidle", timeout=5000)
    except Exception:
        pass


def _body_text(page: Any) -> str:
    try:
        return page.inner_text("body")
    except Exception:
        return ""


def _page_content(page: Any) -> str:
    try:
        return page.content()
    except Exception:
        return ""


def _page_url(page: Any) -> str:
    try:
        return str(page.url)
    except Exception:
        return ""
