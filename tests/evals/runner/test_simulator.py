import json

import pytest

from tests.evals.fakes import FakeModel, failing
from warden_sdk.evals.compare import compare_runs
from warden_sdk.evals.dataset import load_dataset
from warden_sdk.evals.runner.harness import _agent_call
from warden_sdk.evals.runner.run import _run_trial
from warden_sdk.evals.runner.simulator import UserSimulator, visible_conversation
from warden_sdk.evals.scorers import SCORERS

SCENARIO = {
    "persona": "impatient customer",
    "goal": "get a refund for order 55",
    "known_info": ["order 55"],
    "unknown_info": ["purchase date"],
    "opening": ["hi, I need a refund"],
    "max_turns": 4,
}


def agent(messages):
    last = messages[-1]["content"]
    if "55" in last:
        return [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {"name": "refund", "arguments": {"order": "55"}}}]},
            {"role": "tool", "tool_call_id": "1", "content": "SECRET-REFUND-ID-9"},
            {"role": "assistant", "content": "Refunded order 55."},
        ]
    return "Which order?"


def scripted_user(*steps):
    """A fake simulator model that replays steps: a message string, or 'STOP:<reason>'."""
    replies = iter(steps)

    def reply(prompt):
        step = next(replies)
        if step.startswith("STOP:"):
            return {"stop": True, "stop_reason": step[5:], "message": ""}
        return {"stop": False, "stop_reason": "none", "message": step}

    return FakeModel(reply, name="fake-user")


def run(model, scenario=SCENARIO, agent_fn=agent):
    item = {"id": "refund", "scenario": scenario, "expected": {"tools": ["refund"]}}
    result, _ = _run_trial("run", item, _agent_call(agent_fn), SCORERS, 0, UserSimulator(model))
    return result, {(s["scorer"], s["criterion"]): s for s in result["scores"]}


def test_opening_then_simulator_until_it_stops():
    model = scripted_user("order 55 please", "STOP:goal_met")
    result, scores = run(model)
    assert [t["user"] for t in result["turns"]] == ["hi, I need a refund", "order 55 please"]
    assert result["simulation"]["stopped_by"] == "simulator"
    assert result["simulation"]["stop_reason"] == "goal_met"
    assert result["simulation"]["simulator"].startswith("simulator/fake-user/")
    assert result["termination"] == "completed"
    assert scores[("user_goal_met", "")]["outcome"] == "pass"
    assert scores[("tool_calls", "")]["outcome"] == "pass"
    assert scores[("turns", "")]["value"] == 2.0


def test_the_simulated_user_never_sees_tool_traffic():
    model = scripted_user("order 55 please", "STOP:goal_met")
    run(model)
    last_prompt = model.prompts[-1]
    assert "Refunded order 55." in last_prompt
    assert "SECRET-REFUND-ID-9" not in last_prompt and "refund(" not in last_prompt
    assert "purchase date" in last_prompt  # what the user doesn't know is in the prompt


def test_running_out_of_turns_is_an_outcome():
    model = scripted_user("still waiting", "hello?", "anyone?")
    result, scores = run(model, agent_fn=lambda m: "Which order?")
    assert len(result["turns"]) == 4
    assert result["termination"] == "max_turns"
    assert scores[("user_goal_met", "")]["reason"] == "ran out of turns"


def test_a_failing_simulator_is_an_infra_error_not_an_agent_failure():
    result, _ = run(FakeModel(failing("model refused")), scenario={**SCENARIO, "opening": []})
    assert result["termination"] == "infra_error"
    assert "simulated user failed" in result["error"]
    assert result["scores"] == []


def test_stopping_before_the_agent_speaks_is_an_infra_error():
    result, _ = run(scripted_user("STOP:goal_met", "STOP:goal_met", "STOP:goal_met"), scenario={**SCENARIO, "opening": []})
    assert result["termination"] == "infra_error"


def test_agent_error_ends_the_conversation():
    def breaks(messages):
        raise RuntimeError("down")

    result, scores = run(scripted_user(), agent_fn=breaks)
    assert result["termination"] == "agent_error"
    assert result["simulation"]["stopped_by"] == "agent_error"
    assert scores[("user_goal_met", "")]["outcome"] == "fail"


def test_visible_conversation_drops_tool_messages():
    messages = [{"role": "user", "content": "hi"}, *agent([{"role": "user", "content": "55"}])]
    assert visible_conversation(messages) == [("USER (you)", "hi"), ("AGENT", "Refunded order 55.")]


def test_changing_the_simulator_makes_every_check_incomparable():
    def results(simulator):
        return [{"item_id": f"c{i}", "trial": 0, "case_hash": str(i), "termination": "completed", "trace_id": None,
                 "input": None, "output": None, "error": None, "simulation": {"simulator": simulator},
                 "scores": [{"scorer": "user_goal_met", "value": 1.0, "passed": True}]} for i in range(10)]

    def run_(run_id, simulator):
        return {"run_id": run_id, "version_tag": None, "status": "completed", "dataset_name": "d", "agent": "a",
                "started_at": None, "dataset_hash": "h", "trials": 1, "results": results(simulator)}

    cmp = compare_runs(run_("a", "simulator/m1/x"), run_("b", "simulator/m2/x"))
    assert cmp["checks"][0]["verdict"] == "incomparable"
    assert cmp["exit_code"] == 2
    assert any("simulated user changed" in w for w in cmp["warnings"])
    assert compare_runs(run_("a", "simulator/m1/x"), run_("b", "simulator/m1/x"))["exit_code"] == 0


def test_scenario_items_are_validated(tmp_path):
    path = tmp_path / "suite.jsonl"
    path.write_text(json.dumps({"id": "a", "scenario": SCENARIO}) + "\n")
    [item], _ = load_dataset(path)
    assert item["scenario"]["goal"] == "get a refund for order 55"
    for bad, message in [({"persona": "x"}, "needs at least a 'goal'"),
                         ({"goal": "x", "known_info": "order 55"}, "list of strings"),
                         ({"goal": "x", "max_turns": 0}, "positive integer")]:
        path.write_text(json.dumps({"id": "a", "scenario": bad}) + "\n")
        with pytest.raises(ValueError, match=message):
            load_dataset(path)


def test_claude_model_can_be_created_without_credentials_when_lazy(monkeypatch):
    from warden_sdk.evals.models import ClaudeModel
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    model = ClaudeModel(lazy=True)
    assert UserSimulator(model).version.startswith("simulator/claude-opus-5-5@low/")
