import json

import pytest

from warden_sdk.evals.dataset import load_dataset
from warden_sdk.evals.environment import Environment, ToolError, with_faults
from warden_sdk.evals.runner.harness import _agent_call
from warden_sdk.evals.runner.run import _run_trial
from warden_sdk.evals.scorers import SCORERS
from warden_sdk.evals.scorers.state import state_diff

STATE = {"orders": {"55": {"status": "delivered", "refunded": False}}}


def refund(env, order):
    env.state["orders"][order]["refunded"] = True
    return {"ok": True, "order": order}


def lookup(env, order):
    return dict(env.state["orders"][order])


TOOLS = {"refund": refund, "lookup_order": lookup}


def test_tools_change_the_trials_own_copy_of_state():
    env = Environment({"state": STATE}, tools=TOOLS)
    assert env.call("refund", order="55") == {"ok": True, "order": "55"}
    assert env.state["orders"]["55"]["refunded"] is True
    assert STATE["orders"]["55"]["refunded"] is False  # the dataset item is untouched
    assert env.record()["initial_state"]["orders"]["55"]["refunded"] is False
    assert env.calls == [{"name": "refund", "args": {"order": "55"}, "call": 1, "result": {"ok": True, "order": "55"}}]


def test_mocks_return_fixed_values_or_errors():
    env = Environment({"mocks": {"weather": {"returns": {"temp_c": 18}}, "pay": {"error": "declined"}}})
    assert env.call("weather", city="x") == {"temp_c": 18}
    with pytest.raises(ToolError, match="declined"):
        env.call("pay")
    with pytest.raises(ToolError, match="unknown tool 'nope'"):
        env.call("nope")


def test_error_fault_on_a_specific_call_stops_the_tool_running():
    env = Environment({"state": STATE, "faults": [{"tool": "refund", "on_call": 1, "effect": "error", "message": "down"}]},
                      tools=TOOLS)
    with pytest.raises(ToolError, match="down"):
        env.call("refund", order="55")
    assert env.state["orders"]["55"]["refunded"] is False
    assert env.calls[0]["fault"] == "error"
    env.call("refund", order="55")  # the second call isn't faulted
    assert env.state["orders"]["55"]["refunded"] is True


def test_corrupt_is_seeded_so_reruns_match_but_trials_differ():
    spec = {"state": STATE, "faults": [{"tool": "lookup_order", "effect": "corrupt"}], "seed": 3}
    results = [Environment(spec, trial=t, tools=TOOLS).call("lookup_order", order="55") for t in (0, 0, 1, 2, 3)]
    assert results[0] == results[1]
    assert results[0] != {"status": "delivered", "refunded": False}
    # The tool itself ran normally: only what the agent saw was corrupted.
    env = Environment(spec, tools=TOOLS)
    env.call("lookup_order", order="55")
    assert env.state == STATE


def test_empty_fault():
    env = Environment({"state": STATE, "faults": [{"tool": "lookup_order", "effect": "empty"}]}, tools=TOOLS)
    assert env.call("lookup_order", order="55") == {}


def test_state_diff_is_partial_order_insensitive_and_reports_paths():
    actual = {"orders": {"55": {"refunded": True, "status": "delivered"}}, "log": ["b", "a"], "tmp": 1}
    assert state_diff(actual, {"orders": {"55": {"refunded": True}}, "log": ["a", "b"]}) == []
    assert state_diff(actual, {"log": {"$ordered": ["a", "b"]}}) != []
    assert state_diff(actual, {"orders": {"55": {"refunded": False}}}) == ["orders.55.refunded: expected False, got True"]
    assert state_diff(actual, {"tmp": {"$absent": True}}) == ["tmp: expected to be absent, got 1"]
    assert state_diff(actual, {"orders": {"56": {}}}) == ["orders.56: missing"]
    assert "expected 1 elements, got 2" in state_diff(actual, {"log": ["a"]})[0]


def agent(messages, context):
    """Refunds through the environment when asked, and tells the user if the tool failed."""
    if "refund" not in messages[-1]["content"]:
        return "How can I help?"
    try:
        context.env.call("lookup_order", order="55")
        context.env.call("refund", order="55")
    except ToolError as e:
        return f"Sorry, the refund failed: {e}"
    return "Refunded order 55."


ITEM = {
    "id": "refund",
    "turns": [{"user": "hi"}, {"user": "refund order 55"}, {"expect": {"tools": ["lookup_order", "refund"]}}],
    "environment": {"state": STATE},
    "expected": {"state": {"orders": {"55": {"refunded": True}}}},
}


def run(item, monkeypatch):
    monkeypatch.setattr("warden_sdk.evals.environment.env._TOOLS", TOOLS)
    result, _ = _run_trial("run", item, _agent_call(agent), SCORERS, 0)
    return result, {(s["scorer"], s["criterion"]): s for s in result["scores"]}


def test_end_to_end_state_and_tool_calls_come_from_the_environment(monkeypatch):
    result, scores = run(ITEM, monkeypatch)
    assert scores[("end_state", "")]["outcome"] == "pass"
    assert scores[("turn_expectations", "turn 2")]["outcome"] == "pass"
    assert result["turns"][1]["tool_calls"] == [{"name": "lookup_order", "args": {"order": "55"}},
                                                {"name": "refund", "args": {"order": "55"}}]
    assert result["environment"]["final_state"]["orders"]["55"]["refunded"] is True


def test_a_fault_the_agent_survives_still_fails_the_end_state(monkeypatch):
    [_, faulted] = with_faults([ITEM], [{"tool": "refund", "effect": "error", "message": "payments down"}])
    assert faulted["id"] == "refund@refund-error" and faulted["fault_of"] == "refund"
    result, scores = run(faulted, monkeypatch)
    assert result["termination"] == "completed"  # the agent handled it
    assert scores[("end_state", "")]["outcome"] == "fail"
    assert "orders.55.refunded: expected True, got False" in scores[("end_state", "")]["reason"]


def test_with_faults_only_copies_items_that_have_an_environment():
    items = with_faults([ITEM, {"id": "plain", "input": "x"}], [{"tool": "refund", "effect": "timeout"},
                                                                {"tool": "lookup_order", "on_call": 2, "effect": "empty"}])
    assert [i["id"] for i in items] == ["refund", "plain", "refund@refund-timeout", "refund@lookup_order-call-2-empty"]


def test_dataset_validation(tmp_path):
    path = tmp_path / "suite.jsonl"
    for bad, message in [
        ({**ITEM, "environment": {"state": []}}, "'state' must be an object"),
        ({**ITEM, "environment": {"mocks": {"x": {"retuns": 1}}}}, "each environment mock"),
        ({**ITEM, "environment": {"faults": [{"tool": "x", "effect": "explode"}]}}, "an 'effect' of"),
        ({"id": "a", "input": "x", "expected": {"state": {}}}, "needs an 'environment'"),
    ]:
        path.write_text(json.dumps(bad) + "\n")
        with pytest.raises(ValueError, match=message):
            load_dataset(path)


def test_injected_fault_errors_dont_count_against_no_errors():
    from warden_sdk.evals.scorers import Case
    from warden_sdk.evals.scorers.checks import no_errors
    from warden_sdk.tracer import Trace

    trace = Trace("a", strict=False)
    for name, error in (("refund", "ToolError: payments down"), ("lookup_order", "ToolError: no order 9")):
        span = trace.span("tool_call", name)
        span.error = error
        trace._finished.append(span)
    env = {"calls": [{"name": "refund", "fault": "error", "error": "payments down"},
                     {"name": "lookup_order", "error": "no order 9"}]}
    score = no_errors(Case(item={"id": "x"}, output=None, error=None, trace=trace, traces=[trace], environment=env))
    # The injected one is ignored; the tool's own error still counts.
    assert score.reason == "lookup_order: ToolError: no order 9"
