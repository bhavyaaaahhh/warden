"""run_eval: every item × trial through the agent, scored and stored, in parallel."""

import uuid
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import httpx

from warden_sdk.evals.dataset import case_hash, load_dataset
from warden_sdk.evals.loading import agent_name, load_agent
from warden_sdk.evals.runner.harness import (
    AgentCall,
    InfraError,
    _agent_call,
    _run_conversation,
    _run_single,
)
from warden_sdk.evals.scorers import SCORERS, Case, Score, Scorer
from warden_sdk.tracer import WARDEN_URL, _jsonable

# Attempts per trial when the agent raises InfraError: the first try plus two retries.
INFRA_ATTEMPTS = 3

DEFAULT_CONCURRENCY = 4

def _score(case: Case, scorers: dict[str, Scorer]) -> list[dict[str, Any]]:
    scores = []
    for name, scorer in scorers.items():
        try:
            result = scorer(case)
        except Exception as e:
            # A broken scorer is a measurement failure, not an agent failure.
            result = Score.unscored(f"scorer raised {type(e).__name__}: {e}")
        for score in result if isinstance(result, list) else [result]:
            if score is not None:
                scores.append({"scorer": name, "scorer_version": getattr(scorer, "version", None), **score.__dict__})
    return scores

def _run_trial(
    run_id: str, item: dict[str, Any], agent: AgentCall, scorers: dict[str, Scorer], trial: int
) -> tuple[dict[str, Any], str | None]:
    run = _run_conversation if "turns" in item else _run_single
    for attempt in range(1, INFRA_ATTEMPTS + 1):
        try:
            played = run(run_id, item, agent, trial)
            break
        except InfraError as e:
            infra_error = f"InfraError: {e}"
            print(f"  ! {item['id']} trial {trial + 1}: infra error ({e}), attempt {attempt}/{INFRA_ATTEMPTS}")
    else:
        # Retries exhausted: store the trial with no scores so it's excluded, not failed.
        return _result(item, trial, "infra_error", None, infra_error, None, None, None, []), None

    traces = played["traces"]
    if not traces:
        print(f"  ! {item['id']}: agent opened no trace")
    case = Case(
        item=item,
        output=played["output"],
        error=played["error"],
        trace=traces[0] if traces else None,
        traces=traces,
        transcript=played["transcript"],
        turns=played["turns"],
        trial=trial,
    )
    termination = "agent_error" if played["error"] else "completed"
    result = _result(
        item, trial, termination, traces[0].trace_id if traces else None,
        played["error"], played["output"], played["transcript"], played["turns"], _score(case, scorers),
    )
    return result, traces[0].version_tag if traces else None

def _result(item, trial, termination, trace_id, error, output, transcript, turns, scores) -> dict[str, Any]:
    return {
        "result_id": str(uuid.uuid4()),
        "item_id": item["id"],
        "trial": trial,
        "case_hash": case_hash(item),
        "termination": termination,
        "trace_id": trace_id,
        "input": _jsonable(item.get("input", item.get("turns"))),
        "expected": _jsonable(item.get("expected")),
        "output": _jsonable(output),
        "error": error,
        "transcript": transcript,
        "turns": turns,
        "scores": scores,
    }

def _format_item(item_id: str, results: list[dict[str, Any]]) -> str:
    """One line per item: which trials failed which checks."""
    failing: dict[str, list[str]] = {}
    for r in results:
        if r["termination"] == "infra_error":
            failing.setdefault("infra_error", []).append(r["error"])
        for s in r["scores"]:
            if s["outcome"] == "fail":
                label = f"{s['scorer']} {s['criterion']}".strip()
                failing.setdefault(label, []).append(s["reason"])
    trials = f" ({len(results)} trials)" if len(results) > 1 else ""
    if not failing:
        return f"  ✓ {item_id}{trials}"
    parts = [f"  ✗ {item_id}{trials}"]
    for label, reasons in failing.items():
        count = f" ×{len(reasons)}" if len(results) > 1 else ""
        parts.append(f"{label}{count}: {reasons[0]}")
    return "  ".join(parts)

def run_eval(
    dataset_path: Path,
    agent: str | Callable,
    version_tag: str | None = None,
    scorers: dict[str, Scorer] | None = None,
    trials: int = 1,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> str:
    """Run every dataset item `trials` times through the agent, score it, and store the results.

    `agent` is an import path ('package.module:function') or the function itself.
    Up to `concurrency` trials run at once, in threads.
    """
    items, dataset_hash = load_dataset(dataset_path)
    target = agent_name(agent)
    call = _agent_call(load_agent(agent) if isinstance(agent, str) else agent)
    scorers = scorers or SCORERS
    run_id = str(uuid.uuid4())

    with httpx.Client(base_url=WARDEN_URL, timeout=10.0) as client:
        client.post(
            "/eval_runs",
            json={
                "run_id": run_id,
                "dataset_name": dataset_path.stem,
                "dataset_hash": dataset_hash,
                "agent": target,
                "version_tag": version_tag,
                "trials": trials,
            },
        ).raise_for_status()
        print(f"eval run {run_id}")
        print(f"  dataset {dataset_path.stem} ({len(items)} items × {trials} trials) → {target}")

        status = "failed"
        passed: dict[str, list[bool]] = {}
        done: dict[str, list[dict[str, Any]]] = {item["id"]: [] for item in items}
        pool = ThreadPoolExecutor(max_workers=max(1, concurrency))
        try:
            pending = {
                pool.submit(_run_trial, run_id, item, call, scorers, trial)
                for item in items
                for trial in range(trials)
            }
            while pending:
                finished, pending = wait(pending, return_when=FIRST_EXCEPTION)
                for future in finished:
                    # Re-raises WardenError: a lost trace means the run can't be trusted.
                    result, trace_version = future.result()
                    # Results are posted from this thread only, so stored order doesn't depend on timing.
                    client.post(f"/eval_runs/{run_id}/results", json=result).raise_for_status()
                    for s in result["scores"]:
                        if s["outcome"] in ("pass", "fail"):
                            passed.setdefault(s["scorer"], []).append(s["passed"])
                    # If no --version was given, tag the run with what the agent reports.
                    version_tag = version_tag or trace_version
                    done[result["item_id"]].append(result)
                    if len(done[result["item_id"]]) == trials:
                        results = sorted(done[result["item_id"]], key=lambda r: r["trial"])
                        print(_format_item(result["item_id"], results))
            status = "completed"
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            try:
                client.patch(
                    f"/eval_runs/{run_id}",
                    json={"status": status, "version_tag": version_tag},
                ).raise_for_status()
            except Exception as e:
                print(f"warden: could not mark run {run_id} {status}: {e}")

    print(f"\n{status}: {len(items)} items × {trials} trials, version {version_tag or '(untagged)'}")
    for name, results in passed.items():
        print(f"  {name:<18} {sum(results)}/{len(results)} passed")
    print(f"\ncompare with: python -m warden_sdk.evals diff <baseline> {run_id}")
    return run_id
