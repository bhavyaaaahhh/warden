import json

import pytest

from warden_sdk.evals.runner import InfraError, _run_trial, case_hash, load_dataset
from warden_sdk.evals.scorers import SCORERS, Case, Score, match_tools

SCRIPT = [
    {"user": "where is my order 42?"},
    {"expect": {"tools": ["lookup_order"], "contains": ["shipped"]}},
    {"user": "refund it please"},
    {"expect": {"tools": ["refund"]}},
]


def support_agent(messages):
    """Replies in OpenAI message shape, with tool calls, like a real tool-using agent."""
    last = messages[-1]["content"]
    if "refund" in last:
        return [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {"name": "refund"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "ok"},
            {"role": "assistant", "content": "Refunded."},
        ]
    return {"role": "assistant", "content": "Order 42 has shipped.", "tool_calls": [{"id": "0", "function": {"name": "lookup_order"}}]}


def scores_by(result):
    return {(s["scorer"], s["criterion"]): s for s in result["scores"]}


def test_scripted_conversation_passes_every_turn():
    item = {"id": "refund", "turns": SCRIPT}
    result, _ = _run_trial("run", item, support_agent, SCORERS, trial=0)
    assert result["termination"] == "completed"
    assert [t["tool_calls"] for t in result["turns"]] == [["lookup_order"], ["refund"]]
    # The agent sees the whole conversation, tool messages included.
    assert [m["role"] for m in result["transcript"]] == ["user", "assistant", "user", "assistant", "tool", "assistant"]
    assert result["output"] == "Refunded."
    s = scores_by(result)
    assert s[("turn_expectations", "turn 1")]["outcome"] == "pass"
    assert s[("turn_expectations", "turn 2")]["outcome"] == "pass"
    assert s[("turns", "")]["value"] == 2.0


def test_failing_turn_names_the_turn_and_reason():
    def skips_lookup(messages):
        return "It has shipped."

    result, _ = _run_trial("run", {"id": "x", "turns": SCRIPT}, skips_lookup, SCORERS, trial=0)
    s = scores_by(result)
    # No tool messages and no trace: we can't tell which tools ran, so that turn is unscored, not failed.
    assert s[("turn_expectations", "turn 1")]["outcome"] == "unscored"


def test_agent_error_stops_the_conversation_and_later_turns_fail():
    def breaks_on_refund(messages):
        if "refund" in messages[-1]["content"]:
            raise RuntimeError("payments down")
        return support_agent(messages)

    result, _ = _run_trial("run", {"id": "x", "turns": SCRIPT}, breaks_on_refund, SCORERS, trial=0)
    assert result["termination"] == "agent_error"
    s = scores_by(result)
    assert s[("turn_expectations", "turn 2")]["outcome"] == "fail"
    assert "payments down" in s[("turn_expectations", "turn 2")]["reason"]
    assert s[("no_errors", "")]["outcome"] == "fail"


def test_infra_error_is_retried_then_stored_without_scores():
    calls = []

    def flaky_provider(query):
        calls.append(query)
        raise InfraError("429 from provider")

    result, _ = _run_trial("run", {"id": "x", "input": "hi"}, flaky_provider, SCORERS, trial=0)
    assert len(calls) == 3
    assert result["termination"] == "infra_error"
    assert result["scores"] == []

    attempts = iter([InfraError("blip"), "fine"])

    def recovers(query):
        step = next(attempts)
        if isinstance(step, Exception):
            raise step
        return step

    result, _ = _run_trial("run", {"id": "x", "input": "hi"}, recovers, SCORERS, trial=0)
    assert result["termination"] == "completed" and result["output"] == "fine"


def test_raising_scorer_is_unscored_not_a_crash():
    def broken(case):
        raise ValueError("bad rubric")

    broken.version = "v3"
    result, _ = _run_trial("run", {"id": "x", "input": "hi"}, lambda q: "hello", {"broken": broken}, trial=0)
    [score] = result["scores"]
    assert score["outcome"] == "unscored" and "bad rubric" in score["reason"]
    assert score["scorer_version"] == "v3"


@pytest.mark.parametrize(
    "actual, expected, mode, ok",
    [
        (["a", "b"], ["a", "b"], "strict", True),
        (["b", "a"], ["a", "b"], "strict", False),
        (["b", "a"], ["a", "b"], "unordered", True),
        (["a", "b", "c"], ["a", "b"], "superset", True),
        (["a"], ["a", "a"], "superset", False),
        (["a"], ["a", "b"], "subset", True),
        (["a", "c"], ["a", "b"], "subset", False),
    ],
)
def test_match_tools(actual, expected, mode, ok):
    assert (match_tools(actual, expected, mode) is None) == ok


def test_case_level_tool_expectation_without_trace_is_unscored():
    case = Case(item={"id": "x", "input": "q", "expected": {"tools": ["search"]}}, output="a", error=None, trace=None)
    assert SCORERS["tool_calls"](case).outcome == "unscored"


def test_score_keeps_old_constructor_working():
    assert Score(value=1.0, passed=True).outcome == "pass"
    assert Score(value=3.2, passed=None).outcome is None
    assert Score.unscored("no verdict").passed is None


def test_load_dataset_accepts_turns_and_rejects_bad_items(tmp_path):
    path = tmp_path / "suite.jsonl"
    path.write_text(json.dumps({"id": "a", "turns": SCRIPT}) + "\n" + json.dumps({"id": "b", "input": "q"}) + "\n")
    items, _ = load_dataset(path)
    assert [i["id"] for i in items] == ["a", "b"]
    assert case_hash(items[0]) != case_hash({**items[0], "turns": SCRIPT[:2]})

    path.write_text(json.dumps({"id": "a", "input": "q", "turns": SCRIPT}) + "\n")
    with pytest.raises(ValueError, match="exactly one"):
        load_dataset(path)
    path.write_text(json.dumps({"id": "a", "turns": [{"expect": {}}]}) + "\n")
    with pytest.raises(ValueError, match="at least one"):
        load_dataset(path)


def test_unrecorded_tools_do_not_hide_a_failed_reply_check():
    item = {"id": "x", "turns": [{"user": "refund 5"}, {"expect": {"contains": ["refund"], "tools": ["lookup"]}}]}
    result, _ = _run_trial("run", item, lambda messages: "sorry, no", SCORERS, trial=0)
    turn = scores_by(result)[("turn_expectations", "turn 1")]
    assert turn["outcome"] == "fail" and "missing ['refund']" in turn["reason"]


def test_two_expect_steps_for_one_turn_are_rejected(tmp_path):
    path = tmp_path / "suite.jsonl"
    path.write_text(json.dumps({"id": "a", "turns": [{"user": "hi"}, {"expect": {}}, {"expect": {}}]}) + "\n")
    with pytest.raises(ValueError, match="must follow a 'user' step"):
        load_dataset(path)
