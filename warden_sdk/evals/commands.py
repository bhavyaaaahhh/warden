from pathlib import Path
from typing import Any

import httpx

from warden_sdk.evals.diff import compare_runs, fetch_run, print_report
from warden_sdk.evals.runner import run_eval
from warden_sdk.evals.scorers import Scorer
from warden_sdk.tracer import WARDEN_URL


def _client() -> httpx.Client:
    return httpx.Client(base_url=WARDEN_URL, timeout=10.0)


def _mark_baseline(client: httpx.Client, run_id: str) -> None:
    res = client.put(f"/eval_runs/{run_id}/baseline")
    if res.status_code in (404, 409):
        raise SystemExit(f"cannot set baseline: {res.json()['detail']}")
    res.raise_for_status()


def run_check(
    dataset_path: Path,
    agent_target: str,
    version_tag: str | None = None,
    scorers: dict[str, Scorer] | None = None,
    baseline_ref: str | None = None,
) -> int:
    """Run the dataset, then diff against the baseline. Returns the diff's exit code."""
    run_id = run_eval(dataset_path, agent_target, version_tag=version_tag, scorers=scorers)
    print()
    with _client() as client:
        candidate = fetch_run(client, run_id)
        if candidate["status"] != "completed":
            raise SystemExit(f"run {run_id} did not complete; nothing to check")
        if baseline_ref is None:
            if candidate["baseline_run_id"] is None:
                # First check for this dataset + agent: this run becomes the reference.
                _mark_baseline(client, run_id)
                print(f"no baseline yet for {dataset_path.stem} + {agent_target}")
                print(f"✓ saved run {run_id[:8]} as the baseline; future checks compare against it")
                return 0
            baseline_ref = candidate["baseline_run_id"]
        baseline = fetch_run(client, baseline_ref, exclude_run_id=run_id)
        return print_report(compare_runs(baseline, candidate))


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
    print(f"  {'run':<10}{'version':<16}{'dataset':<14}{'status':<11}{'vs baseline':<18}checks")
    for r in runs:
        if r["is_baseline"]:
            verdict = "★ baseline"
        elif r["regressions_vs_baseline"] is None:
            verdict = "-"
        elif r["regressions_vs_baseline"]:
            verdict = f"✗ {r['regressions_vs_baseline']} regressed"
        else:
            verdict = "✓ ok"
        print(
            f"  {r['run_id'][:8]:<10}{(r['version_tag'] or '(untagged)')[:15]:<16}"
            f"{r['dataset_name'][:13]:<14}{r['status']:<11}{verdict:<18}{_checks_summary(r['checks'])}"
        )
    return 0
