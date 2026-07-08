"""WebArena task-source tests: pure evaluators + config loading + scoring.

No browser, no docker, no WebArena infra — the string_match / url_match / program_html
matching logic and the config→Task construction are exercised directly. The Playwright path
(reset/apply/program_html live-fetch) is not covered here; those need the live sites.
"""

from __future__ import annotations

import json

import pytest

from modalitybench.tasks.base import Episode, Task
from modalitybench.tasks.webarena import (
    UnsupportedEval,
    WebArenaSource,
    _resolve_eval,
    clean_answer,
    content_check,
    string_match_score,
    substitute_urls,
    url_match_score,
)


# -- URL placeholder substitution -------------------------------------------


def test_substitute_urls_fills_from_env():
    env = {"SHOPPING": "http://host:7770/", "GITLAB": "http://host:8023"}
    assert substitute_urls("__SHOPPING__/cart", env) == "http://host:7770/cart"
    assert substitute_urls("__GITLAB__/x", env) == "http://host:8023/x"


def test_substitute_urls_noop_and_missing_env():
    # Already-substituted string is untouched; a missing env var leaves a visible placeholder.
    assert substitute_urls("http://host/cart", {}) == "http://host/cart"
    assert substitute_urls("__REDDIT__/f", {}) == "__REDDIT__/f"


def test_resolve_eval_substitutes_reference_url_and_program_html():
    env_spec = {
        "reference_url": "__SHOPPING__/checkout",
        "program_html": [{"url": "__SHOPPING__/cart", "locator": "", "required_contents": {}}],
    }
    import os

    os.environ["SHOPPING"] = "http://h:7770"
    try:
        out = _resolve_eval(env_spec)
    finally:
        del os.environ["SHOPPING"]
    assert out["reference_url"] == "http://h:7770/checkout"
    assert out["program_html"][0]["url"] == "http://h:7770/cart"


# -- string match -----------------------------------------------------------


def test_clean_answer_strips_quotes_and_lowercases():
    assert clean_answer('  "Hello World" ') == "hello world"
    assert clean_answer("'X'") == "x"
    assert clean_answer("plain") == "plain"


def test_string_exact_match():
    refs = {"exact_match": "Quest Lumaflex Band"}
    assert string_match_score(refs, "quest lumaflex band") == 1.0
    assert string_match_score(refs, "some other band") == 0.0


def test_string_must_include_multiword_substring():
    refs = {"must_include": ["red mug", "under $20"]}
    assert string_match_score(refs, "A red mug that is under $20 here") == 1.0
    assert string_match_score(refs, "A red mug only") == 0.0


def test_string_must_include_single_word_tokenized():
    # Single-word "0" must not match inside "100" — tokenization guards the substring trap.
    assert string_match_score({"must_include": ["0"]}, "the total is 100") == 0.0
    assert string_match_score({"must_include": ["0"]}, "the total is 0 items") == 1.0


def test_string_na_sentinel_short_circuits_on_literal_na():
    # A literal "n/a" answer passes with no judge; anything else needs the ua_match judge.
    refs = {"fuzzy_match": "N/A"}
    assert string_match_score(refs, "N/A") == 1.0
    with pytest.raises(UnsupportedEval):
        string_match_score(refs, "not applicable")


def test_string_fuzzy_match_unsupported_without_judge():
    with pytest.raises(UnsupportedEval):
        string_match_score({"fuzzy_match": ["a paraphrase"]}, "whatever")


# -- LLM judge (fuzzy_match / ua_match) -------------------------------------


class _FakeJudge:
    """Records calls and returns scripted verdicts, standing in for LLMJudge."""

    def __init__(self, fuzzy=1.0, ua=1.0):
        self._fuzzy, self._ua = fuzzy, ua
        self.calls = []

    def fuzzy_match(self, question, reference, pred):
        self.calls.append(("fuzzy", question, reference, pred))
        return self._fuzzy

    def ua_match(self, question, reference, pred):
        self.calls.append(("ua", question, reference, pred))
        return self._ua


def test_string_fuzzy_match_uses_judge():
    judge = _FakeJudge(fuzzy=1.0)
    refs = {"fuzzy_match": ["around noon", "midday"]}
    assert string_match_score(refs, "12pm", intent="When?", judge=judge) == 1.0
    # Every reference is graded (product), so both are sent to the judge.
    assert [c[2] for c in judge.calls] == ["around noon", "midday"]
    # One failing reference drags the product to 0.
    assert string_match_score(refs, "12pm", intent="When?", judge=_FakeJudge(fuzzy=0.0)) == 0.0


def test_string_na_defers_to_ua_judge():
    judge = _FakeJudge(ua=1.0)
    refs = {"fuzzy_match": "N/A"}
    s = string_match_score(
        refs, "the site has no such filter", intent="Filter by X", string_note="X unsupported",
        judge=judge,
    )
    assert s == 1.0
    assert judge.calls[0][0] == "ua" and judge.calls[0][2] == "X unsupported"


def test_llm_judge_parses_verdicts():
    from modalitybench.agents.model_client import MockClient
    from modalitybench.tasks.webarena import LLMJudge

    def responder(*, system, blocks, tools):
        text = blocks[0].text.lower()
        # Distinguish the two prompts by a phrase unique to each.
        if "grade the answer" in text:
            return "The answer matches. correct"
        return "These align. same"

    judge = LLMJudge(MockClient(responder=responder))
    assert judge.fuzzy_match("q", "ref", "pred") == 1.0
    assert judge.ua_match("q", "ref", "pred") == 1.0

    judge_bad = LLMJudge(MockClient(responder=lambda **k: "this is incorrect"))
    assert judge_bad.fuzzy_match("q", "ref", "pred") == 0.0
    judge_diff = LLMJudge(MockClient(responder=lambda **k: "they are different"))
    assert judge_diff.ua_match("q", "ref", "pred") == 0.0


# -- url match --------------------------------------------------------------


def test_url_match_empty_reference_is_pass():
    assert url_match_score("", "http://anything/here") == 1.0


def test_url_match_base_path_substring_and_query():
    # "GOLD in PRED": the reference base path must be a substring of the predicted one; the
    # prediction may append extra query params (page=2) but must carry the reference's.
    ref = "http://host/group/proj/issues?state=closed"
    assert url_match_score(ref, "http://host/group/proj/issues?state=closed&page=2") == 1.0
    # Wrong query value fails.
    assert url_match_score(ref, "http://host/group/proj/issues?state=open") == 0.0
    # Wrong base path fails.
    assert url_match_score(ref, "http://host/group/proj/merge_requests?state=closed") == 0.0


def test_url_match_or_alternatives():
    ref = "http://host/a |OR| http://host/b"
    assert url_match_score(ref, "http://host/b/deep") == 1.0
    assert url_match_score(ref, "http://host/c") == 0.0


# -- program_html content check ---------------------------------------------


def test_content_check_must_include_with_or():
    rc = {"must_include": ["Order placed", "Total |OR| Sum"]}
    assert content_check("Order placed. Sum: $5", rc) == 1.0
    assert content_check("Order placed only", rc) == 0.0


def test_content_check_exact_match_normalized():
    assert content_check('  "Done" ', {"exact_match": "done"}) == 1.0
    assert content_check("nope", {"exact_match": "done"}) == 0.0


# -- config loading / tasks -------------------------------------------------


def _write_configs(tmp_path):
    configs = [
        {
            "sites": ["shopping"],
            "task_id": 0,
            "require_login": True,
            "storage_state": "./.auth/shopping_state.json",
            "start_url": "__SHOPPING__",
            "intent": "Find the cheapest mug",
            "eval": {"eval_types": ["string_match"], "reference_answers": {"exact_match": "mug"}},
        },
        {
            "sites": ["gitlab"],
            "task_id": 1,
            "require_login": True,
            "storage_state": "./.auth/gitlab_state.json",
            "start_url": "__GITLAB__/dashboard",
            "intent": "Open my issues",
            "eval": {"eval_types": ["url_match"], "reference_url": "__GITLAB__/issues"},
        },
    ]
    p = tmp_path / "test.json"
    p.write_text(json.dumps(configs), encoding="utf-8")
    return p


def test_tasks_load_filter_and_substitute(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOPPING", "http://shop:7770")
    monkeypatch.setenv("GITLAB", "http://git:8023")
    cfg = _write_configs(tmp_path)

    tasks = WebArenaSource(config_file=str(cfg)).tasks()
    assert [t.task_id for t in tasks] == ["webarena#0", "webarena#1"]
    assert tasks[0].meta["start_url"] == "http://shop:7770"
    assert tasks[0].goal == "Find the cheapest mug"
    # eval reference_url placeholder resolved at load.
    assert tasks[1].meta["eval"]["reference_url"] == "http://git:8023/issues"

    # site filter + task_ids
    only_shop = WebArenaSource(config_file=str(cfg), sites=["shopping"]).tasks()
    assert [t.task_id for t in only_shop] == ["webarena#0"]
    only_1 = WebArenaSource(config_file=str(cfg), task_ids=[1]).tasks()
    assert [t.task_id for t in only_1] == ["webarena#1"]


def test_tasks_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOPPING", "http://shop:7770")
    monkeypatch.setenv("GITLAB", "http://git:8023")
    cfg = _write_configs(tmp_path)
    assert len(WebArenaSource(config_file=str(cfg), limit=1).tasks()) == 1


def test_load_requires_a_source():
    with pytest.raises(RuntimeError, match="config_file or options.config_dir"):
        WebArenaSource().tasks()


# -- score() ----------------------------------------------------------------


def _task(eval_spec, task_id="webarena#7", start_url="http://host/start"):
    return Task(
        task_id=task_id,
        goal="g",
        source="webarena",
        meta={"task_id": 7, "start_url": start_url, "eval": eval_spec, "storage_state": None},
    )


def test_score_string_match_success_and_failure():
    src = WebArenaSource()
    spec = {"eval_types": ["string_match"], "reference_answers": {"exact_match": "42"}}
    ep_ok = Episode(task_id="webarena#7", strategy="s", model="m", meta={"answer": "42"})
    ep_bad = Episode(task_id="webarena#7", strategy="s", model="m", meta={"answer": "41"})
    assert src.score(_task(spec), ep_ok).success
    assert not src.score(_task(spec), ep_bad).success


def test_score_url_match_uses_recorded_final_url():
    src = WebArenaSource()
    src._final_url["webarena#7"] = "http://host/group/issues?state=closed&page=1"
    spec = {"eval_types": ["url_match"], "reference_url": "http://host/group/issues?state=closed"}
    ep = Episode(task_id="webarena#7", strategy="s", model="m", meta={"answer": ""})
    assert src.score(_task(spec), ep).success


def test_score_unsupported_eval_surfaces_error_not_pass():
    src = WebArenaSource()
    spec = {"eval_types": ["string_match"], "reference_answers": {"fuzzy_match": ["paraphrase"]}}
    ep = Episode(task_id="webarena#7", strategy="s", model="m", meta={"answer": "x"})
    res = src.score(_task(spec), ep)
    assert not res.success and res.error and "fuzzy_match" in res.error


def test_score_fuzzy_task_with_injected_judge():
    # A fuzzy_match task is UnsupportedEval without a judge, but scores once one is wired.
    src = WebArenaSource()
    spec = {"eval_types": ["string_match"], "reference_answers": {"fuzzy_match": ["noon"]}}
    ep = Episode(task_id="webarena#7", strategy="s", model="m", meta={"answer": "12pm"})
    assert src.score(_task(spec), ep).error  # no judge -> surfaced, not a silent pass

    # Wire a judge, bypassing build_model_client: set the name and pre-populate the cache.
    src.judge_model = "fake"
    src._judge = _FakeJudge(fuzzy=1.0)
    res = src.score(_task(spec), ep)
    assert res.success and res.error is None


def test_score_multi_component_is_product():
    # string passes but url fails -> overall failure (product of components).
    src = WebArenaSource()
    src._final_url["webarena#7"] = "http://host/wrong"
    spec = {
        "eval_types": ["string_match", "url_match"],
        "reference_answers": {"exact_match": "ok"},
        "reference_url": "http://host/right",
    }
    ep = Episode(task_id="webarena#7", strategy="s", model="m", meta={"answer": "ok"})
    res = src.score(_task(spec), ep)
    assert not res.success
    assert res.metrics["string_match"] == 1.0 and res.metrics["url_match"] == 0.0
