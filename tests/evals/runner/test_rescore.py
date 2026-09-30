import uuid

from warden_sdk.evals.runner.rescore import rescore
from warden_sdk.evals.scorers import SCORERS, Score

RUN = str(uuid.uuid4())
TRACE = str(uuid.uuid4())


def seed(server):
    server.runs[RUN] = {"run_id": RUN, "dataset_name": "support", "dataset_hash": "h", "agent": "a:b",
                        "version_tag": "v1", "trials": 1, "status": "completed", "metadata": {"git": {"commit": "c"}}}
    server.traces[TRACE] = {
        "trace_id": TRACE, "agent_name": "a", "version_tag": "v1", "input": "x", "status": "success",
        "started_at": "2026-09-29T10:00:00+00:00", "ended_at": "2026-09-29T10:00:00.250000+00:00", "metadata": {},
        "spans": [{"span_id": str(uuid.uuid4()), "parent_span_id": None, "span_type": "tool_call", "name": "refund",
                   "input": {"order": "55"}, "output": "ok", "error": None, "tokens_input": 10, "tokens_output": 5,
                   "cost_usd": "0.002", "started_at": "2026-09-29T10:00:00+00:00", "ended_at": "2026-09-29T10:00:00.1+00:00"}],
    }
    server.results[RUN] = [
        {"item_id": "refund", "trial": 0, "case_hash": "hash-refund", "termination": "completed", "trace_id": TRACE,
         "input": "refund 55", "expected": {"tools": [{"name": "refund", "args": {"order": "55"}}]},
         "output": "Refunded.", "error": None, "transcript": None, "turns": None,
         "simulation": None, "scores": [{"scorer": "old", "outcome": "pass"}]},
        {"item_id": "lost", "trial": 0, "case_hash": "hash-lost", "termination": "infra_error", "trace_id": None,
         "input": "q", "expected": None, "output": None, "error": "InfraError: 429", "transcript": None,
         "turns": None, "simulation": None, "scores": []},
    ]


def test_rescore_rebuilds_cases_from_stored_results_and_traces(fake_server):
    seed(fake_server)

    def says_refunded(case):
        return Score(value=None, passed="Refunded" in str(case.output))

    scorers = {"tool_calls": SCORERS["tool_calls"], "cost_usd": SCORERS["cost_usd"],
               "latency_ms": SCORERS["latency_ms"], "says_refunded": says_refunded}
    new_id = rescore(RUN, scorers)

    run = fake_server.runs[new_id]
    assert run["metadata"] == {"git": {"commit": "c"}, "rescored_from": RUN}
    assert run["status"] == "completed" and run["version_tag"] == "v1"
    refund, lost = fake_server.results[new_id]
    # Same case hash, so it pairs with the original in a diff.
    assert refund["case_hash"] == "hash-refund" and refund["trace_id"] == TRACE
    scores = {s["scorer"]: s for s in refund["scores"]}
    assert scores["tool_calls"]["outcome"] == "pass"  # args read from the stored trace's span
    assert scores["cost_usd"]["value"] == 0.002
    assert round(scores["latency_ms"]["value"]) == 250
    assert scores["says_refunded"]["outcome"] == "pass"
    assert "old" not in scores
    # Infra errors stay excluded: copied over, no scores.
    assert lost["termination"] == "infra_error" and lost["scores"] == []
    # The original run is untouched.
    assert fake_server.results[RUN][0]["scores"] == [{"scorer": "old", "outcome": "pass"}]
