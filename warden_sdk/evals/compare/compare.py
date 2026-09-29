"""Compare two eval runs: is the candidate worse than the baseline, or is the change noise?

Implements docs/research/evaluation-methodology.md. compare_runs() is pure data,
used by both the CLI report and the viewer so they always agree.
"""

import random
from statistics import mean
from typing import Any

from warden_sdk.evals.compare import stats

# Bump when the comparison method changes, so cached verdicts are recomputed.
METHOD_VERSION = "1"

ALPHA = 0.05

# Below this share of scored trials a scorer's verdict is "inconclusive".
MIN_COVERAGE = 0.9

MIN_PAIRED_CASES = 5

# Warn when more than this share of baseline cases can't be compared.
MAX_EXCLUDED_SHARE = 0.10

# A metric is worse/better only if the whole 95% CI of its change is beyond this.
METRIC_CHANGE_PCT = 10.0

# With fewer cases a p95 is close to the max, so only the median is compared.
P95_MIN_CASES = 50

LOWER_IS_BETTER = {"latency_ms", "cost_usd", "total_tokens", "turns"}

# How each metric is summarised across cases. Anything unlisted gets a mean.
METRIC_AGGREGATES = {"latency_ms": ("p50", "p95"), "cost_usd": ("total",), "total_tokens": ("total",)}

TRANSITIONS = {
    ("pass", "fail"): "broke",
    ("pass", "flaky"): "degraded",
    ("flaky", "fail"): "degraded",
    ("fail", "pass"): "fixed",
    ("flaky", "pass"): "stabilised",
    ("fail", "flaky"): "improved",
}

# Report order, worst first.
TRANSITION_ORDER = ["broke", "degraded", "fixed", "stabilised", "improved"]

def _outcome(score: dict[str, Any]) -> str | None:
    # Runs stored before outcomes existed only have `passed`.
    if score.get("outcome"):
        return score["outcome"]
    if score.get("passed") is not None:
        return "pass" if score["passed"] else "fail"
    return None

def _trial_view(result: dict[str, Any]) -> tuple[dict[str, tuple[str, str | None]], dict[str, float], dict[str, str]]:
    """One trial's checks collapsed to one outcome per scorer, plus its metric values.

    A scorer with several criteria (one per turn) fails if any criterion fails,
    and is unscored if none failed but one couldn't be scored.
    """
    by_scorer: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, float] = {}
    versions: dict[str, str] = {}
    for s in result["scores"]:
        if s.get("scorer_version"):
            versions[s["scorer"]] = s["scorer_version"]
        if _outcome(s) is None:
            if s.get("value") is not None:
                metrics[s["scorer"]] = metrics.get(s["scorer"], 0.0) + float(s["value"])
            continue
        by_scorer.setdefault(s["scorer"], []).append(s)
    checks = {}
    for name, scores in by_scorer.items():
        outcomes = [_outcome(s) for s in scores]
        failed = [s for s in scores if _outcome(s) == "fail"]
        if failed:
            reason = "; ".join(
                f"{s['criterion']}: {s.get('reason')}" if s.get("criterion") else str(s.get("reason")) for s in failed
            )
            checks[name] = ("fail", reason)
        elif "unscored" in outcomes:
            checks[name] = ("unscored", next(s.get("reason") for s in scores if _outcome(s) == "unscored"))
        else:
            checks[name] = ("pass", None)
    return checks, metrics, versions

def _cases(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for r in sorted(run["results"], key=lambda r: r.get("trial") or 0):
        case = cases.setdefault(r["item_id"], {"hash": r.get("case_hash"), "trials": [], "infra": 0, "first": r})
        if r.get("termination") == "infra_error":
            case["infra"] += 1
            continue
        checks, metrics, versions = _trial_view(r)
        case["trials"].append({"checks": checks, "metrics": metrics, "versions": versions, "result": r})
    return cases

def _tally(case: dict[str, Any], scorer: str) -> dict[str, Any] | None:
    """Passes, valid trials and unscored trials of one scorer on one case."""
    outcomes = [t["checks"][scorer] for t in case["trials"] if scorer in t["checks"]]
    if not outcomes:
        return None
    return {
        "c": sum(o == "pass" for o, _ in outcomes),
        "n": sum(o in ("pass", "fail") for o, _ in outcomes),
        "unscored": sum(o == "unscored" for o, _ in outcomes),
        "infra": case["infra"],
        "reason": next((why for o, why in outcomes if o == "fail"), None),
    }

def _state(t: dict[str, Any]) -> str:
    return "pass" if t["c"] == t["n"] else "fail" if t["c"] == 0 else "flaky"

def _coverage(tallies: list[dict[str, Any]]) -> float | None:
    attempted = sum(t["n"] + t["unscored"] + t["infra"] for t in tallies)
    return sum(t["n"] for t in tallies) / attempted if attempted else None

def _pass_k(tallies: list[dict[str, Any]], k: int) -> float | None:
    # Only cases with all k trials valid; fewer trials would inflate pass^k.
    full = [t for t in tallies if t["n"] == k]
    return mean(stats.pass_hat_k(t["n"], t["c"], k) for t in full) if full and k > 1 else None

def _compare_check(
    scorer: str, ids: list[str], ca: dict, cb: dict, seed: str, intervals: bool = True
) -> tuple[dict[str, Any], dict[str, tuple[str, dict, dict]], tuple[list[dict], list[dict]]]:
    """One check scorer across paired cases: rates, paired test, CI, and per-case transitions."""
    tallies_a = {i: t for i in ids if (t := _tally(ca[i], scorer))}
    tallies_b = {i: t for i in ids if (t := _tally(cb[i], scorer))}
    # A case where every trial hit an infra error has no scores, so it would
    # vanish. If the scorer applied in the other run, count it as unmeasured.
    for mine, other, cases in ((tallies_a, tallies_b, ca), (tallies_b, tallies_a, cb)):
        for i in ids:
            if i not in mine and i in other and cases[i]["infra"] and not cases[i]["trials"]:
                mine[i] = {"c": 0, "n": 0, "unscored": 0, "infra": cases[i]["infra"], "reason": None}
    paired = [i for i in ids if i in tallies_a and i in tallies_b and tallies_a[i]["n"] and tallies_b[i]["n"]]
    rate = lambda t: t["c"] / t["n"]  # noqa: E731
    diffs = [rate(tallies_b[i]) - rate(tallies_a[i]) for i in paired]
    rng = random.Random(seed)

    single = all(tallies_a[i]["n"] == 1 and tallies_b[i]["n"] == 1 for i in paired)
    broke = sum(d < 0 for d in diffs)
    fixed = sum(d > 0 for d in diffs)
    if single:
        test, p = "mcnemar", stats.mcnemar_exact(broke, fixed)
    else:
        test, p = "sign-flip", stats.sign_flip_test(diffs, rng)
    ci = stats.bootstrap_ci(len(diffs), lambda idx: mean(diffs[j] for j in idx), rng) if diffs and intervals else None

    versions_a = {v for i in ids for t in ca[i]["trials"] if (v := t["versions"].get(scorer))}
    versions_b = {v for i in ids for t in cb[i]["trials"] if (v := t["versions"].get(scorer))}
    row = {
        "scorer": scorer,
        "cases": len(paired),
        "baseline_rate": mean(rate(tallies_a[i]) for i in paired) if paired else None,
        "candidate_rate": mean(rate(tallies_b[i]) for i in paired) if paired else None,
        "delta": mean(diffs) if diffs else None,
        "ci": list(ci) if ci else None,
        "test": test,
        "p": p,
        "discordant": {"broke": broke, "fixed": fixed},
        "coverage": {
            "baseline": _coverage(list(tallies_a.values())),
            "candidate": _coverage(list(tallies_b.values())),
        },
        "pass_k": None,
        "incomparable": bool(versions_a and versions_b and versions_a != versions_b),
    }
    transitions = {
        i: (label, tallies_a[i], tallies_b[i])
        for i in paired
        if (label := TRANSITIONS.get((_state(tallies_a[i]), _state(tallies_b[i]))))
    }
    return row, transitions, (list(tallies_a.values()), list(tallies_b.values()))

def _verdict(row: dict[str, Any]) -> tuple[str, str | None]:
    if row["incomparable"]:
        return "incomparable", "the scorer's version changed between runs"
    covs = [c for c in row["coverage"].values() if c is not None]
    if covs and min(covs) < MIN_COVERAGE:
        return "inconclusive", f"only {min(covs):.0%} of trials were scored (need {MIN_COVERAGE:.0%})"
    if row["cases"] < MIN_PAIRED_CASES:
        return "inconclusive", f"{row['cases']} cases compared (need {MIN_PAIRED_CASES})"
    if row["p_adjusted"] < ALPHA and row["delta"] < 0:
        return "regression", None
    if row["p_adjusted"] < ALPHA and row["delta"] > 0:
        return "improvement", None
    if row["test"] == "mcnemar":
        needed = stats.min_discordant_for_significance(ALPHA)
        d = row["discordant"]["broke"] + row["discordant"]["fixed"]
        return "no detectable change", f"{d} case(s) flipped; at least {needed} flipping the same way are needed"
    return "no detectable change", None

def _aggregate(values: list[float], how: str) -> float:
    if how == "total":
        return sum(values)
    if how == "mean":
        return mean(values)
    return stats.percentile(values, float(how.removeprefix("p")))

def _compare_metric(scorer: str, ids: list[str], ca: dict, cb: dict, seed: str) -> list[dict[str, Any]]:
    def case_mean(case: dict) -> float | None:
        values = [t["metrics"][scorer] for t in case["trials"] if scorer in t["metrics"]]
        return mean(values) if values else None

    pairs = [(a, b) for i in ids if (a := case_mean(ca[i])) is not None and (b := case_mean(cb[i])) is not None]
    if not pairs:
        return []
    rng = random.Random(seed)
    rows = []
    for how in METRIC_AGGREGATES.get(scorer, ("mean",)):
        unstable = how == "p95" and len(pairs) < P95_MIN_CASES
        va = _aggregate([a for a, _ in pairs], how)
        vb = _aggregate([b for _, b in pairs], how)

        def pct(idx: list[int]) -> float | None:
            base = _aggregate([pairs[j][0] for j in idx], how)
            return (_aggregate([pairs[j][1] for j in idx], how) - base) / base * 100 if base else None

        ci = None if unstable else stats.bootstrap_ci(len(pairs), pct, rng)
        verdict = "no detectable change"
        if unstable:
            verdict = f"unstable (fewer than {P95_MIN_CASES} cases)"
        elif ci:
            up, down = ci[0] > METRIC_CHANGE_PCT, ci[1] < -METRIC_CHANGE_PCT
            if up or down:
                verdict = "worse" if up == (scorer in LOWER_IS_BETTER) else "better"
        rows.append({
            "scorer": scorer,
            "aggregate": how,
            "cases": len(pairs),
            "baseline": va,
            "candidate": vb,
            "change_pct": (vb - va) / va * 100 if va else 0.0,
            "ci": list(ci) if ci else None,
            "verdict": verdict,
            "flagged": verdict == "worse",
        })
    return rows

def _summary(run: dict[str, Any]) -> dict[str, Any]:
    keys = ("run_id", "version_tag", "status", "dataset_name", "agent", "started_at", "trials")
    return {k: run.get(k) for k in keys}

def compare_runs(a: dict[str, Any], b: dict[str, Any], intervals: bool = True) -> dict[str, Any]:
    """Compare candidate run B against baseline run A. Pure data, no I/O.

    intervals=False skips confidence intervals and metrics, which the verdict
    doesn't depend on, for listing many runs quickly. The verdict is the same.
    """
    ca, cb = _cases(a), _cases(b)
    warnings = []
    for run in (a, b):
        if run["status"] != "completed":
            warnings.append(f"run {str(run['run_id'])[:8]} is {run['status']}, results may be partial")
    if a["dataset_hash"] != b["dataset_hash"]:
        warnings.append("the runs used different dataset contents")

    only_a = [i for i in ca if i not in cb]
    only_b = [i for i in cb if i not in ca]
    changed = [
        i for i in ca if i in cb and ca[i]["hash"] and cb[i]["hash"] and ca[i]["hash"] != cb[i]["hash"]
    ]
    ids = [i for i in ca if i in cb and i not in changed]
    if changed:
        warnings.append(f"edited since the baseline, not compared: {', '.join(changed)}")
    if only_a:
        warnings.append(f"only in baseline: {', '.join(only_a)}")
    if only_b:
        warnings.append(f"only in candidate: {', '.join(only_b)}")
    if ca and (len(only_a) + len(changed)) / len(ca) > MAX_EXCLUDED_SHARE:
        warnings.append(f"{len(only_a) + len(changed)} of {len(ca)} baseline cases could not be compared")

    seed = f"{a['run_id']}:{b['run_id']}"
    check_names = sorted({s for c in (*ca.values(), *cb.values()) for t in c["trials"] for s in t["checks"]})
    metric_names = sorted({s for c in (*ca.values(), *cb.values()) for t in c["trials"] for s in t["metrics"]})

    checks, per_case = [], {}
    for name in check_names:
        row, transitions, (tallies_a, tallies_b) = _compare_check(name, ids, ca, cb, f"{seed}:{name}", intervals)
        row["pass_k"] = {
            "k_baseline": a.get("trials") or 1,
            "k_candidate": b.get("trials") or 1,
            "baseline": _pass_k(tallies_a, a.get("trials") or 1),
            "candidate": _pass_k(tallies_b, b.get("trials") or 1),
        }
        checks.append(row)
        for item_id, (label, ta, tb) in transitions.items():
            per_case.setdefault(item_id, {})[name] = (label, ta, tb)

    # Adjust for testing several scorers at once, but only across scorers that
    # could reach significance at all (Tarone 1990): a scorer where nothing
    # flipped can't be significant, and counting it would only dilute the rest.
    testable = [r for r in checks if stats.min_attainable_p(sum(r["discordant"].values())) < ALPHA]
    for row, p_adj in zip(testable, stats.benjamini_hochberg([r["p"] for r in testable])):
        row["p_adjusted"] = p_adj
    for row in checks:
        row.setdefault("p_adjusted", row["p"])
        row["verdict"], row["note"] = _verdict(row)

    transitions = []
    for item_id, labels in per_case.items():
        first_b = cb[item_id]["first"]
        transitions.append({
            "item_id": item_id,
            "label": min((lab for lab, _, _ in labels.values()), key=TRANSITION_ORDER.index),
            "scorers": {
                name: {"label": lab, "baseline": f"{ta['c']}/{ta['n']}", "candidate": f"{tb['c']}/{tb['n']}",
                       "reason": tb["reason"]}
                for name, (lab, ta, tb) in sorted(labels.items())
            },
            "input": first_b.get("input"),
            "baseline_output": ca[item_id]["first"].get("output"),
            "candidate_output": first_b.get("output"),
            "baseline_trace_id": ca[item_id]["first"].get("trace_id"),
            "candidate_trace_id": first_b.get("trace_id"),
        })
    transitions.sort(key=lambda t: (TRANSITION_ORDER.index(t["label"]), t["item_id"]))

    metrics = [
        row for name in metric_names if intervals for row in _compare_metric(name, ids, ca, cb, f"{seed}:{name}")
    ]

    verdicts = {r["verdict"] for r in checks}
    if "regression" in verdicts:
        verdict, exit_code = "regression", 1
    elif verdicts & {"inconclusive", "incomparable"} or a["status"] != "completed" or b["status"] != "completed":
        verdict, exit_code = "undecided", 2
    else:
        verdict, exit_code = "no regression", 0

    return {
        "baseline": _summary(a),
        "candidate": _summary(b),
        "items_compared": len(ids),
        "excluded": {"changed": changed, "only_baseline": only_a, "only_candidate": only_b},
        "warnings": warnings,
        "checks": checks,
        "transitions": transitions,
        "metrics": metrics,
        "verdict": verdict,
        "exit_code": exit_code,
    }
