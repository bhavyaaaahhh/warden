from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from warden_sdk.evals.client import client as _client
from warden_sdk.evals.client import fetch_run
from warden_sdk.evals.compare import compare_runs, print_report
from warden_sdk.evals.loading import agent_name
from warden_sdk.evals.runner import DEFAULT_CONCURRENCY, run_eval
from warden_sdk.evals.runner.simulator import UserSimulator
from warden_sdk.evals.scorers import Scorer
from warden_sdk.evals.validation import judge_warnings


def _mark_baseline(client: httpx.Client, run_id: str) -> None:
    res = client.put(f"/eval_runs/{run_id}/baseline")
    if res.status_code in (404, 409):
        raise SystemExit(f"cannot set baseline: {res.json()['detail']}")
    res.raise_for_status()


def run_check(
    dataset_path: Path,
    agent: str | Callable,
    version_tag: str | None = None,
    scorers: dict[str, Scorer] | None = None,
    baseline_ref: str | None = None,
    trials: int = 3,
    concurrency: int = DEFAULT_CONCURRENCY,
    simulator: UserSimulator | None = None,
) -> int:
    """Run the dataset, then diff against the baseline. Returns the diff's exit code."""
    run_id = run_eval(
        dataset_path, agent, version_tag=version_tag, scorers=scorers, trials=trials, concurrency=concurrency,
        simulator=simulator,
    )
    print()
    with _client() as client:
        candidate = fetch_run(client, run_id)
        if candidate["status"] != "completed":
            raise SystemExit(f"run {run_id} did not complete; nothing to check")
        if baseline_ref is None:
            if candidate["baseline_run_id"] is None:
                # First check for this dataset + agent: this run becomes the reference.
                _mark_baseline(client, run_id)
                print(f"no baseline yet for {dataset_path.stem} + {agent_name(agent)}")
                print(f"✓ saved run {run_id[:8]} as the baseline; future checks compare against it")
                return 0
            baseline_ref = candidate["baseline_run_id"]
        baseline = fetch_run(client, baseline_ref, exclude_run_id=run_id)
        cmp = compare_runs(baseline, candidate)
        cmp["warnings"] += judge_warnings(client, [baseline, candidate])
        return print_report(cmp)


def run_calibrate(
    dataset_path: Path,
    agent: str | Callable,
    scorers: dict[str, Scorer] | None = None,
    trials: int = 3,
    concurrency: int = DEFAULT_CONCURRENCY,
    simulator: UserSimulator | None = None,
) -> int:
    """Run the same agent twice and compare the runs (an A/A test).

    Nothing changed between the runs, so any regression reported is noise: it
    shows how much the suite's results move on their own.
    """
    print("A/A calibration: running the same agent twice\n")
    options = {"scorers": scorers, "trials": trials, "concurrency": concurrency, "simulator": simulator}
    first = run_eval(dataset_path, agent, **options)
    print()
    second = run_eval(dataset_path, agent, **options)
    print()
    with _client() as client:
        runs = [fetch_run(client, first), fetch_run(client, second)]
    cmp = compare_runs(*runs)
    print_report(cmp)

    # Items that moved between the runs, or whose trials disagreed within a run.
    moved = {t["item_id"] for t in cmp["transitions"]} | _flaky_items(runs[0]) | _flaky_items(runs[1])
    print(f"\ncalibration: {len(moved)} of {cmp['items_compared']} items changed across identical runs")
    if cmp["verdict"] == "regression":
        print("✗ an identical rerun was reported as a regression: add trials or cases before trusting checks")
        return 1
    if cmp["verdict"] == "undecided":
        print("? could not decide: too little was scored to measure the noise")
        return 2
    print("✓ identical reruns were not reported as regressions")
    return 0


def _flaky_items(run: dict[str, Any]) -> set[str]:
    """Items where the same check passed in some trials and failed in others."""
    seen: dict[tuple[str, str, str], set[str]] = {}
    for r in run["results"]:
        for s in r["scores"]:
            if s["outcome"] in ("pass", "fail"):
                seen.setdefault((r["item_id"], s["scorer"], s["criterion"]), set()).add(s["outcome"])
    return {item for (item, _, _), outcomes in seen.items() if len(outcomes) > 1}


def set_baseline(ref: str) -> int:
    with _client() as client:
        run = fetch_run(client, ref)
        _mark_baseline(client, run["run_id"])
    print(
        f"★ baseline for {run['dataset_name']} + {run['agent']} is now "
        f"{run['version_tag'] or '(untagged)'} (run {run['run_id'][:8]})"
    )
    return 0


def _checks_summary(checks: dict[str, Any]) -> str:
    return "  ".join(f"{name} {c['passed']}/{c['total']}" for name, c in sorted(checks.items()))


def list_runs(dataset: str | None = None, limit: int = 20) -> int:
    with _client() as client:
        params: dict[str, Any] = {"limit": limit}
        if dataset:
            params["dataset_name"] = dataset
        res = client.get("/eval_runs", params=params)
        res.raise_for_status()
        runs = res.json()
    if not runs:
        print("no eval runs yet")
        return 0
    print(f"  {'run':<10}{'version':<16}{'dataset':<14}{'status':<11}{'vs baseline':<24}checks")
    for r in runs:
        vs = r["vs_baseline"]
        if r["is_baseline"]:
            verdict = "★ baseline"
        elif vs is None:
            verdict = "-"
        elif vs["verdict"] == "regression":
            verdict = f"✗ regression ({vs['broke']} broke)"
        elif vs["verdict"] == "undecided":
            verdict = "? undecided"
        else:
            verdict = "✓ ok"
        print(
            f"  {r['run_id'][:8]:<10}{(r['version_tag'] or '(untagged)')[:15]:<16}"
            f"{r['dataset_name'][:13]:<14}{r['status']:<11}{verdict:<24}{_checks_summary(r['checks'])}"
        )
    return 0


def run_diff(baseline: str, candidate: str) -> int:
    with _client() as client:
        runs = [fetch_run(client, baseline), fetch_run(client, candidate)]
        cmp = compare_runs(*runs)
        cmp["warnings"] += judge_warnings(client, runs)
    return print_report(cmp)
