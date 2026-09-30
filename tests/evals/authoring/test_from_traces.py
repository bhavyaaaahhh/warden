import json

from warden_sdk.evals.authoring.from_traces import (
    is_eval_trace,
    item_from_trace,
    write_items,
)
from warden_sdk.evals.dataset import load_dataset

TRACE = {
    "trace_id": "0f3a9c2e-1111-2222-3333-444455556666", "agent_name": "support", "version_tag": "v3",
    "input": "refund order 55", "status": "error", "started_at": "2026-09-29T10:00:00+00:00", "metadata": {},
    "spans": [
        {"span_type": "tool_call", "name": "lookup_order", "input": {"order": "55"}, "output": "delivered"},
        {"span_type": "tool_call", "name": "handoff", "input": None, "output": "queued"},
        {"span_type": "llm_call", "name": "answer", "input": None, "output": "Refunded order 55."},
    ],
}


def test_item_keeps_its_source_and_reference():
    item = item_from_trace(TRACE, expect_tools=True)
    assert item["id"] == "trace-0f3a9c2e"
    assert item["input"] == "refund order 55"
    assert item["expected"] == {
        "criteria": [], "reference": "Refunded order 55.",
        "tools": [{"name": "lookup_order", "args": {"order": "55"}}, "handoff"],
    }
    assert item["source"]["trace_id"] == TRACE["trace_id"] and item["source"]["status"] == "error"
    assert "tools" not in item_from_trace(TRACE)["expected"]


def test_eval_traces_are_recognised():
    assert is_eval_trace({"eval_run_id": "r"}) and is_eval_trace({"metadata": {"eval_run_id": "r"}})
    assert not is_eval_trace(TRACE)


def test_appending_skips_traces_already_in_the_dataset_and_keeps_ids_unique(tmp_path):
    out = tmp_path / "regressions.jsonl"
    out.write_text(json.dumps({"id": "trace-0f3a9c2e", "input": "something else"}) + "\n")
    assert write_items([item_from_trace(TRACE)], out) == (1, 0)
    assert write_items([item_from_trace(TRACE)], out) == (0, 1)
    items, _ = load_dataset(out)  # still a valid dataset
    assert [i["id"] for i in items] == ["trace-0f3a9c2e", "trace-0f3a9c2e-2"]


def test_as_turns_makes_a_one_turn_conversation():
    item = item_from_trace(TRACE, as_turns=True)
    assert item["turns"] == [{"user": "refund order 55"}] and "input" not in item
