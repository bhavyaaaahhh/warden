import json
import threading
import time

import pytest

from warden_sdk.evals import EvalContext, Score, run_eval
from warden_sdk.evals.loading import load_scorer


def _dataset(tmp_path, lines):
    path = tmp_path / "suite.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def test_trials_run_concurrently(tmp_path, fake_server):
    path = _dataset(tmp_path, [{"id": f"c{i}", "input": i} for i in range(4)])
    running, peak = 0, 0
    lock = threading.Lock()

    def slow_agent(query):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.1)
        with lock:
            running -= 1
        return "ok"

    started = time.monotonic()
    run_id = run_eval(path, slow_agent, trials=2, concurrency=8)
    elapsed = time.monotonic() - started

    assert peak == 8
    assert elapsed < 0.5  # 8 trials × 0.1s run together, not one after another
    results = fake_server.results[run_id]
    assert sorted((r["item_id"], r["trial"]) for r in results) == [(f"c{i}", t) for i in range(4) for t in range(2)]
    assert fake_server.runs[run_id]["status"] == "completed"
    assert fake_server.runs[run_id]["agent"].endswith("slow_agent")


def test_concurrency_one_runs_sequentially(tmp_path, fake_server):
    path = _dataset(tmp_path, [{"id": f"c{i}", "input": i} for i in range(3)])
    order = []
    run_eval(path, lambda q: order.append(q) or "ok", concurrency=1)
    assert order == [0, 1, 2]


def test_agent_gets_a_stable_thread_id_per_trial(tmp_path, fake_server):
    path = _dataset(tmp_path, [{"id": "chat", "turns": [{"user": "hi"}, {"user": "again"}]}])
    seen: list[EvalContext] = []

    def stateful_agent(messages, context: EvalContext):
        seen.append(context)
        return "hello"

    run_id = run_eval(path, stateful_agent, trials=2)
    by_trial = {}
    for ctx in seen:
        by_trial.setdefault(ctx.trial, set()).add(ctx.thread_id)
    # One thread id shared by both turns of a trial, different across trials.
    assert all(len(ids) == 1 for ids in by_trial.values())
    assert by_trial[0] != by_trial[1]
    assert seen[0].run_id == run_id and seen[0].item_id == "chat"


def test_a_failing_agent_marks_items_failed_but_the_run_completes(tmp_path, fake_server):
    path = _dataset(tmp_path, [{"id": "a", "input": 1}, {"id": "b", "input": 2}])

    def agent(q):
        if q == 2:
            raise RuntimeError("boom")
        return "ok"

    run_id = run_eval(path, agent, concurrency=2)
    terminations = {r["item_id"]: r["termination"] for r in fake_server.results[run_id]}
    assert terminations == {"a": "completed", "b": "agent_error"}
    assert fake_server.runs[run_id]["status"] == "completed"


def polite(case):
    return Score(value=None, passed="please" in str(case.output))


polite.version = "v1"


def test_custom_scorer_by_import_path(tmp_path, fake_server):
    name, scorer = load_scorer("tests.evals.runner.test_run:polite")
    assert name == "polite"
    path = _dataset(tmp_path, [{"id": "a", "input": 1}])
    run_id = run_eval(path, lambda q: "yes please", scorers={name: scorer})
    [score] = fake_server.results[run_id][0]["scores"]
    assert score["outcome"] == "pass" and score["scorer_version"] == "v1"


def test_load_scorer_rejects_bad_paths():
    with pytest.raises(ValueError, match="package.module:name"):
        load_scorer("no_colon")
