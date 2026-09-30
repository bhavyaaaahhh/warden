"""Score a stored run again without re-running the agent.

Running an agent is the slow, expensive part; scoring is cheap. After changing
a scorer, editing a judge's prompt, or adding a new check, `rescore` rebuilds
each trial from what was stored (outputs, transcripts, turns, and the agent's
traces from the server) and scores it into a new run. The original run is left
as it was, and because cases keep their content hashes the two runs pair up
in a diff.
"""

import uuid
from typing import Any

import httpx

from warden_sdk.evals.client import client as _client
from warden_sdk.evals.client import fetch_run
from warden_sdk.evals.runner.run import _score
from warden_sdk.evals.scorers import Case, Scorer
from warden_sdk.tracer import Span, Trace


def _trace(data: dict[str, Any]) -> Trace:
    """A stored trace as a Trace, without uploading anything."""
    trace = Trace(data["agent_name"], input=data.get("input"), version_tag=data.get("version_tag"),
                  metadata=data.get("metadata"), strict=False)
    trace.trace_id = str(data["trace_id"])
    trace.status, trace.started_at, trace.ended_at = data["status"], data["started_at"], data.get("ended_at")
    for s in data.get("spans", []):
        span = Span(trace, s["span_type"], s["name"], s.get("input"))
        span.span_id, span.parent_span_id = str(s["span_id"]), s.get("parent_span_id")
        span.output, span.error = s.get("output"), s.get("error")
        span.started_at, span.ended_at = s["started_at"], s.get("ended_at")
        span.tokens_input, span.tokens_output = s.get("tokens_input"), s.get("tokens_output")
        span.cost_usd = float(s["cost_usd"]) if s.get("cost_usd") is not None else None
        trace._finished.append(span)
    return trace


def _item(result: dict[str, Any]) -> dict[str, Any]:
    """The dataset item as it was, from what the result stored."""
    item: dict[str, Any] = {"id": result["item_id"]}
    if result.get("environment") is not None:
        item["environment"] = {"state": result["environment"]["initial_state"]}
    if result.get("simulation") is not None:
        item["scenario"] = result["input"]
    elif result.get("turns") is not None:
        item["turns"] = result["input"]
    else:
        item["input"] = result["input"]
    if result.get("expected") is not None:
        item["expected"] = result["expected"]
    return item


def _traces(client: httpx.Client, result: dict[str, Any], cache: dict[str, Trace]) -> list[Trace]:
    ids = [t for turn in result.get("turns") or [] for t in turn.get("trace_ids", [])] or [result.get("trace_id")]
    traces = []
    for trace_id in (str(t) for t in ids if t):
        if trace_id not in cache:
            res = client.get(f"/traces/{trace_id}")
            if res.status_code == 404:
                continue  # deleted since; scorers that need it will say so
            res.raise_for_status()
            cache[trace_id] = _trace(res.json())
        traces.append(cache[trace_id])
    return traces


def rescore(run_ref: str, scorers: dict[str, Scorer], version_tag: str | None = None) -> str:
    """Score a stored run with `scorers` into a new run. Returns the new run's id."""
    new_id = str(uuid.uuid4())
    cache: dict[str, Trace] = {}
    with _client() as client:
        run = fetch_run(client, run_ref)
        if run["status"] != "completed":
            raise SystemExit(f"run {run['run_id']} is {run['status']}; only completed runs can be rescored")
        client.post("/eval_runs", json={
            "run_id": new_id,
            "dataset_name": run["dataset_name"],
            "dataset_hash": run["dataset_hash"],
            "agent": run["agent"],
            "version_tag": version_tag or run["version_tag"],
            "trials": run.get("trials") or 1,
            "metadata": {**(run.get("metadata") or {}), "rescored_from": str(run["run_id"])},
        }).raise_for_status()
        print(f"rescoring run {str(run['run_id'])[:8]} → new run {new_id}")
        status = "failed"
        try:
            for r in run["results"]:
                scores: list[dict[str, Any]] = []
                if r.get("termination") != "infra_error":
                    traces = _traces(client, r, cache)
                    case = Case(
                        item=_item(r), output=r.get("output"),
                        error=r.get("error") if r.get("termination") == "agent_error" else None,
                        trace=traces[0] if traces else None, traces=traces,
                        transcript=r.get("transcript"), turns=r.get("turns"),
                        simulation=r.get("simulation"), environment=r.get("environment"), trial=r.get("trial") or 0,
                    )
                    scores = _score(case, scorers)
                client.post(f"/eval_runs/{new_id}/results", json={
                    "result_id": str(uuid.uuid4()),
                    **{k: r.get(k) for k in ("item_id", "trial", "case_hash", "termination", "trace_id", "input",
                                             "expected", "output", "error", "transcript", "turns", "simulation",
                                             "environment")},
                    "trial": r.get("trial") or 0,
                    "termination": r.get("termination") or "completed",
                    "scores": scores,
                }).raise_for_status()
            status = "completed"
        finally:
            client.patch(f"/eval_runs/{new_id}", json={"status": status}).raise_for_status()
    print(f"{status}: {len(run['results'])} results rescored")
    print(f"\ncompare with: python -m warden_sdk.evals diff {run['run_id']} {new_id}")
    return new_id
