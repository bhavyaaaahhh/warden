import json
import math
from typing import Any
from uuid import UUID

import httpx

from warden_sdk.tracer import WARDEN_URL

# How each metric scorer is summarised across items. Anything unlisted gets a mean.
METRIC_AGGREGATES = {
    "latency_ms": ("p50", "p95"),
    "cost_usd": ("total",),
    "total_tokens": ("total",),
}
# Metric increases above this are flagged (but don't fail the diff).
METRIC_WARN_PCT = 20.0


def fetch_run(client: httpx.Client, ref: str, exclude_run_id: str | None = None) -> dict[str, Any]:
    """Accept either a run id or a version tag (latest completed run with that tag).

    exclude_run_id skips a run when resolving a tag, so a check doesn't pick the
    run it just made as its own baseline.
    """
    try:
        run_id = str(UUID(ref))
    except ValueError:
        runs = client.get(
            "/eval_runs", params={"version_tag": ref, "status": "completed", "limit": 2}
        )
        runs.raise_for_status()
        matches = [r["run_id"] for r in runs.json() if r["run_id"] != exclude_run_id]
        if not matches:
            raise SystemExit(f"no completed eval run with version_tag {ref!r}")
        run_id = matches[0]
    res = client.get(f"/eval_runs/{run_id}")
    if res.status_code == 404:
        raise SystemExit(f"eval run {run_id} not found")
    res.raise_for_status()
    return res.json()


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def _aggregate(values: list[float], how: str) -> float:
    if how == "total":
        return sum(values)
    if how == "mean":
        return sum(values) / len(values)
    return _percentile(values, float(how.removeprefix("p")))


def _summary(run: dict[str, Any]) -> dict[str, Any]:
    keys = ("run_id", "version_tag", "status", "dataset_name", "agent", "started_at")
    return {k: run.get(k) for k in keys}


def compare_runs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Compare candidate run B against baseline run A. Pure data, no I/O.

    A regression is an item where some pass/fail scorer passed in A and failed in B.
    Used by both the CLI report and the viewer, so they always agree.
    """
    a_results = {r["item_id"]: r for r in a["results"]}
    b_results = {r["item_id"]: r for r in b["results"]}
    common = [item_id for item_id in a_results if item_id in b_results]

    warnings = []
    for run in (a, b):
        if run["status"] != "completed":
            warnings.append(f"run {str(run['run_id'])[:8]} is {run['status']}, results may be partial")
    if a["dataset_hash"] != b["dataset_hash"]:
        warnings.append("the runs used different dataset contents")
    only_a = [i for i in a_results if i not in b_results]
    only_b = [i for i in b_results if i not in a_results]
    if only_a:
        warnings.append(f"only in baseline: {', '.join(only_a)}")
    if only_b:
        warnings.append(f"only in candidate: {', '.join(only_b)}")

    def scores(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {s["scorer"]: s for s in result["scores"]}

    regressions: list[dict[str, Any]] = []
    fixes: list[dict[str, Any]] = []
    checks: dict[str, list[tuple[bool, bool]]] = {}
    metrics: dict[str, list[tuple[float, float]]] = {}
    for item_id in common:
        ra, rb = a_results[item_id], b_results[item_id]
        sa, sb = scores(ra), scores(rb)
        broke, fixed = [], []
        for name in sa.keys() & sb.keys():
            pa, pb = sa[name]["passed"], sb[name]["passed"]
            if pa is not None and pb is not None:
                checks.setdefault(name, []).append((pa, pb))
                if pa and not pb:
                    broke.append(name)
                elif pb and not pa:
                    fixed.append(name)
            elif sa[name]["value"] is not None and sb[name]["value"] is not None:
                metrics.setdefault(name, []).append((float(sa[name]["value"]), float(sb[name]["value"])))
        traces = {"baseline_trace_id": ra["trace_id"], "candidate_trace_id": rb["trace_id"]}
        if broke:
            broke.sort()
            regressions.append({
                "item_id": item_id,
                "scorers": broke,
                "input": rb["input"],
                "baseline_output": ra["output"],
                "candidate_output": rb["output"],
                "reasons": {n: sb[n]["reason"] for n in broke if sb[n]["reason"]},
                "error": rb["error"],
                **traces,
            })
        if fixed:
            fixes.append({"item_id": item_id, "scorers": sorted(fixed), **traces})

    check_rows = [
        {
            "scorer": name,
            "total": len(pairs),
            "baseline_passed": sum(p for p, _ in pairs),
            "candidate_passed": sum(p for _, p in pairs),
        }
        for name, pairs in sorted(checks.items())
    ]
    metric_rows = []
    for name, pairs in sorted(metrics.items()):
        for how in METRIC_AGGREGATES.get(name, ("mean",)):
            va = _aggregate([x for x, _ in pairs], how)
            vb = _aggregate([y for _, y in pairs], how)
            pct = (vb - va) / va * 100 if va else 0.0
            metric_rows.append({
                "scorer": name,
                "aggregate": how,
                "baseline": va,
                "candidate": vb,
                "change_pct": pct,
                "flagged": pct > METRIC_WARN_PCT,
            })

    return {
        "baseline": _summary(a),
        "candidate": _summary(b),
        "items_compared": len(common),
        "warnings": warnings,
        "regressions": regressions,
        "fixes": fixes,
        "checks": check_rows,
        "metrics": metric_rows,
    }


def _short(value: Any, width: int = 110) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= width else text[: width - 1] + "…"


def _label(run: dict[str, Any]) -> str:
    return f"{run['version_tag'] or '(untagged)'} (run {str(run['run_id'])[:8]})"


def _fmt_num(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 1 or value == 0:
        return f"{value:.2f}"
    return f"{value:.6f}"


def print_report(cmp: dict[str, Any]) -> int:
    """Print a compare_runs() result as a terminal report. Returns the exit code."""
    print(f"warden diff  {_label(cmp['baseline'])}  →  {_label(cmp['candidate'])}")
    print(f"dataset {cmp['candidate']['dataset_name']}, {cmp['items_compared']} items compared")
    for warning in cmp["warnings"]:
        print(f"  ⚠ {warning}")

    regressions = cmp["regressions"]
    print(f"\nREGRESSIONS ({len(regressions)})")
    for r in regressions:
        print(f"  ✗ {r['item_id']}  {', '.join(r['scorers'])}: passed → failed")
        print(f"      input      {_short(r['input'])}")
        print(f"      baseline   {_short(r['baseline_output'])}")
        print(f"      candidate  {_short(r['candidate_output'])}")
        for name, reason in r["reasons"].items():
            print(f"      {name}: {reason}")
        if r["error"]:
            print(f"      error      {r['error']}")

    print(f"\nFIXES ({len(cmp['fixes'])})")
    for f in cmp["fixes"]:
        print(f"  ✓ {f['item_id']}  {', '.join(f['scorers'])}: failed → passed")

    print(f"\n  {'':<20}{'baseline':>12}{'candidate':>12}{'change':>12}")
    for c in cmp["checks"]:
        pa, pb, n = c["baseline_passed"], c["candidate_passed"], c["total"]
        print(f"  {c['scorer']:<20}{f'{pa}/{n}':>12}{f'{pb}/{n}':>12}{pb - pa:>+12}")
    for m in cmp["metrics"]:
        flag = " ▲" if m["flagged"] else ""
        label = f"{m['scorer']} {m['aggregate']}"
        change = f"{m['change_pct']:+.0f}%"
        print(f"  {label:<20}{_fmt_num(m['baseline']):>12}{_fmt_num(m['candidate']):>12}{change:>12}{flag}")

    if regressions:
        print(f"\n✗ {len(regressions)} regression(s)")
        return 1
    print("\n✓ no regressions")
    return 0


def run_diff(baseline: str, candidate: str) -> int:
    with httpx.Client(base_url=WARDEN_URL, timeout=10.0) as client:
        return print_report(compare_runs(fetch_run(client, baseline), fetch_run(client, candidate)))
