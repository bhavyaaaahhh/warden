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


def fetch_run(client: httpx.Client, ref: str) -> dict[str, Any]:
    """Accept either a run id or a version tag (latest completed run with that tag)."""
    try:
        run_id = str(UUID(ref))
    except ValueError:
        runs = client.get(
            "/eval_runs", params={"version_tag": ref, "status": "completed", "limit": 1}
        )
        runs.raise_for_status()
        if not runs.json():
            raise SystemExit(f"no completed eval run with version_tag {ref!r}")
        run_id = runs.json()[0]["run_id"]
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


def _short(value: Any, width: int = 110) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= width else text[: width - 1] + "…"


def _label(run: dict[str, Any]) -> str:
    return f"{run['version_tag'] or '(untagged)'} (run {run['run_id'][:8]})"


def _fmt_num(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 1 or value == 0:
        return f"{value:.2f}"
    return f"{value:.6f}"


def diff_runs(a: dict[str, Any], b: dict[str, Any]) -> int:
    """Print a regression report for B against baseline A. Returns the exit code."""
    a_results = {r["item_id"]: r for r in a["results"]}
    b_results = {r["item_id"]: r for r in b["results"]}
    common = [item_id for item_id in a_results if item_id in b_results]

    print(f"warden diff  {_label(a)}  →  {_label(b)}")
    print(f"dataset {b['dataset_name']}, {len(common)} items compared")
    for run in (a, b):
        if run["status"] != "completed":
            print(f"  ⚠ run {run['run_id'][:8]} is {run['status']}, results may be partial")
    if a["dataset_hash"] != b["dataset_hash"]:
        print("  ⚠ the runs used different dataset contents")
    only_a = [i for i in a_results if i not in b_results]
    only_b = [i for i in b_results if i not in a_results]
    if only_a:
        print(f"  ⚠ only in baseline: {', '.join(only_a)}")
    if only_b:
        print(f"  ⚠ only in candidate: {', '.join(only_b)}")

    def scores(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {s["scorer"]: s for s in result["scores"]}

    regressions: list[tuple[str, list[str]]] = []
    fixes: list[tuple[str, list[str]]] = []
    checks: dict[str, list[tuple[bool, bool]]] = {}
    metrics: dict[str, list[tuple[float, float]]] = {}
    for item_id in common:
        sa, sb = scores(a_results[item_id]), scores(b_results[item_id])
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
                metrics.setdefault(name, []).append((sa[name]["value"], sb[name]["value"]))
        if broke:
            regressions.append((item_id, sorted(broke)))
        if fixed:
            fixes.append((item_id, sorted(fixed)))

    print(f"\nREGRESSIONS ({len(regressions)})")
    for item_id, names in regressions:
        ra, rb = a_results[item_id], b_results[item_id]
        print(f"  ✗ {item_id}  {', '.join(names)}: passed → failed")
        print(f"      input      {_short(rb['input'])}")
        print(f"      baseline   {_short(ra['output'])}")
        print(f"      candidate  {_short(rb['output'])}")
        for name in names:
            reason = scores(rb)[name]["reason"]
            if reason:
                print(f"      {name}: {reason}")
        if rb["error"]:
            print(f"      error      {rb['error']}")

    print(f"\nFIXES ({len(fixes)})")
    for item_id, names in fixes:
        print(f"  ✓ {item_id}  {', '.join(names)}: failed → passed")

    print(f"\n  {'':<20}{'baseline':>12}{'candidate':>12}{'change':>12}")
    for name, pairs in sorted(checks.items()):
        pa, pb = sum(p for p, _ in pairs), sum(p for _, p in pairs)
        n = len(pairs)
        print(f"  {name:<20}{f'{pa}/{n}':>12}{f'{pb}/{n}':>12}{pb - pa:>+12}")
    for name, pairs in sorted(metrics.items()):
        for how in METRIC_AGGREGATES.get(name, ("mean",)):
            va = _aggregate([x for x, _ in pairs], how)
            vb = _aggregate([y for _, y in pairs], how)
            pct = (vb - va) / va * 100 if va else 0.0
            flag = " ▲" if pct > METRIC_WARN_PCT else ""
            label = f"{name} {how}"
            print(f"  {label:<20}{_fmt_num(va):>12}{_fmt_num(vb):>12}{f'{pct:+.0f}%':>12}{flag}")

    if regressions:
        print(f"\n✗ {len(regressions)} regression(s)")
        return 1
    print("\n✓ no regressions")
    return 0


def run_diff(baseline: str, candidate: str) -> int:
    with httpx.Client(base_url=WARDEN_URL, timeout=10.0) as client:
        return diff_runs(fetch_run(client, baseline), fetch_run(client, candidate))
