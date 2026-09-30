from warden_sdk.evals.compare import compare_runs


def _result(item_id, trial, passed=True, scorer="contains", termination="completed", h="h", **score):
    scores = [] if termination == "infra_error" else [{"scorer": scorer, "value": None if passed is None else float(passed), "passed": passed, **score}]
    return {
        "item_id": item_id, "trial": trial, "case_hash": f"{h}-{item_id}", "termination": termination,
        "trace_id": None, "input": item_id, "output": "out", "error": None, "scores": scores,
    }


def _run(run_id, results, trials=1, status="completed"):
    return {
        "run_id": run_id, "version_tag": run_id, "status": status, "dataset_name": "d", "agent": "a",
        "started_at": None, "dataset_hash": "x", "trials": trials, "results": results,
    }


def _passing(run_id, n, fail=(), trials=1):
    return _run(run_id, [_result(f"c{i}", t, passed=i not in fail) for i in range(n) for t in range(trials)], trials)


def _check(cmp, scorer="contains"):
    return next(r for r in cmp["checks"] if r["scorer"] == scorer)


def test_one_flip_on_twenty_cases_is_not_a_regression():
    cmp = compare_runs(_passing("a", 20), _passing("b", 20, fail={3}))
    row = _check(cmp)
    assert row["test"] == "mcnemar"
    assert row["verdict"] == "no detectable change"
    assert "at least 6" in row["note"]
    assert cmp["exit_code"] == 0
    # The case is still listed, so the user can see it.
    assert [t["item_id"] for t in cmp["transitions"]] == ["c3"]
    assert cmp["transitions"][0]["label"] == "broke"


def test_six_flips_the_same_way_is_a_regression():
    cmp = compare_runs(_passing("a", 20), _passing("b", 20, fail={0, 1, 2, 3, 4, 5}))
    row = _check(cmp)
    assert row["p"] == 0.03125
    assert row["verdict"] == "regression"
    assert cmp["verdict"] == "regression" and cmp["exit_code"] == 1


def test_trials_use_the_sign_flip_test_and_label_flaky_cases():
    base = _passing("a", 10, trials=3)
    # c0 now fails every trial; c1 fails one of three.
    cand = _run("b", [
        _result(f"c{i}", t, passed=not (i == 0 or (i == 1 and t == 2))) for i in range(10) for t in range(3)
    ], trials=3)
    cmp = compare_runs(base, cand)
    row = _check(cmp)
    assert row["test"] == "sign-flip"
    labels = {t["item_id"]: t["label"] for t in cmp["transitions"]}
    assert labels == {"c0": "broke", "c1": "degraded"}
    assert row["pass_k"]["baseline"] == 1.0
    assert row["pass_k"]["candidate"] == 0.8


def test_unscored_is_excluded_and_low_coverage_is_inconclusive():
    base = _passing("a", 10)
    cand = _run("b", [
        _result(f"c{i}", 0, passed=None, outcome="unscored", reason="judge gave no verdict") if i < 2
        else _result(f"c{i}", 0)
        for i in range(10)
    ])
    row = _check(compare_runs(base, cand))
    # Unscored trials count as neither pass nor fail.
    assert row["cases"] == 8 and row["candidate_rate"] == 1.0
    assert row["coverage"]["candidate"] == 0.8
    assert row["verdict"] == "inconclusive"


def test_infra_errors_are_excluded_not_failed():
    base = _passing("a", 10)
    cand = _run("b", [_result(f"c{i}", 0, termination="infra_error" if i == 0 else "completed") for i in range(10)])
    row = _check(compare_runs(base, cand))
    assert row["cases"] == 9 and row["candidate_rate"] == 1.0
    assert row["verdict"] == "no detectable change"


def test_edited_cases_are_not_paired():
    base = _passing("a", 10)
    cand = _passing("b", 10, fail={0})
    cand["results"][0]["case_hash"] = "edited"
    cmp = compare_runs(base, cand)
    assert cmp["excluded"]["changed"] == ["c0"]
    assert cmp["items_compared"] == 9
    assert not cmp["transitions"]


def test_scorer_version_change_is_incomparable():
    base = _run("a", [_result(f"c{i}", 0, scorer_version="judge-v1") for i in range(10)])
    cand = _run("b", [_result(f"c{i}", 0, scorer_version="judge-v2") for i in range(10)])
    cmp = compare_runs(base, cand)
    assert _check(cmp)["verdict"] == "incomparable"
    assert cmp["exit_code"] == 2


def test_multi_criterion_scorer_fails_if_any_criterion_fails():
    def trial(item_id, second_ok):
        r = _result(item_id, 0, scorer="turn_expectations", criterion="turn 1")
        r["scores"].append({"scorer": "turn_expectations", "value": float(second_ok), "passed": second_ok,
                            "criterion": "turn 2", "reason": None if second_ok else "missing ['refund']"})
        return r

    base = _run("a", [trial(f"c{i}", True) for i in range(6)])
    cand = _run("b", [trial(f"c{i}", False) for i in range(6)])
    cmp = compare_runs(base, cand)
    assert _check(cmp, "turn_expectations")["verdict"] == "regression"
    assert cmp["transitions"][0]["scorers"]["turn_expectations"]["reason"] == "turn 2: missing ['refund']"


def test_metric_verdict_needs_the_whole_ci_past_ten_percent():
    def run(run_id, latency):
        return _run(run_id, [
            {**_result(f"c{i}", 0), "scores": [{"scorer": "latency_ms", "value": latency(i), "passed": None}]}
            for i in range(30)
        ])

    slower = compare_runs(run("a", lambda i: 100 + i), run("b", lambda i: 200 + i))
    p50 = next(m for m in slower["metrics"] if m["aggregate"] == "p50")
    assert p50["verdict"] == "worse" and p50["flagged"]
    p95 = next(m for m in slower["metrics"] if m["aggregate"] == "p95")
    assert p95["verdict"].startswith("unstable")

    same = compare_runs(run("a", lambda i: 100 + i), run("b", lambda i: 101 + i))
    assert next(m for m in same["metrics"] if m["aggregate"] == "p50")["verdict"] == "no detectable change"


def test_comparison_is_deterministic():
    base = _passing("a", 30, trials=3)
    cand = _run("b", [_result(f"c{i}", t, passed=(i + t) % 4 != 0) for i in range(30) for t in range(3)], trials=3)
    assert compare_runs(base, cand) == compare_runs(base, cand)


def test_old_runs_without_outcomes_still_compare():
    base = _passing("a", 10)
    for r in base["results"]:
        for s in r["scores"]:
            s.pop("outcome", None)
        r.pop("trial"), r.pop("case_hash"), r.pop("termination")
    base.pop("trials")
    cmp = compare_runs(base, _passing("b", 10))
    assert _check(cmp)["cases"] == 10


def test_a_scorer_with_no_changes_does_not_dilute_a_real_regression():
    # Every case also has a no_errors score that never changes.
    def run(run_id, fail):
        results = [_result(f"c{i}", 0, passed=i not in fail) for i in range(12)]
        for r in results:
            r["scores"].append({"scorer": "no_errors", "value": 1.0, "passed": True})
        return _run(run_id, results)

    cmp = compare_runs(run("a", set()), run("b", {0, 1, 2, 3, 4, 5}))
    row = _check(cmp)
    assert row["p_adjusted"] == row["p"] == 0.03125
    assert row["verdict"] == "regression"
    assert _check(cmp, "no_errors")["verdict"] == "no detectable change"


def test_cases_lost_to_infra_errors_lower_coverage():
    base = _passing("a", 10, trials=3)
    # An outage: items c5..c9 hit infra errors on every trial.
    cand = _run("b", [
        _result(f"c{i}", t, termination="infra_error" if i >= 5 else "completed") for i in range(10) for t in range(3)
    ], trials=3)
    row = _check(compare_runs(base, cand))
    assert row["coverage"]["candidate"] == 0.5
    assert row["verdict"] == "inconclusive"


def test_list_mode_gives_the_same_verdict_without_intervals():
    base = _passing("a", 30, trials=3)
    cand = _run("b", [_result(f"c{i}", t, passed=i >= 8) for i in range(30) for t in range(3)], trials=3)
    full, fast = compare_runs(base, cand), compare_runs(base, cand, intervals=False)
    assert full["verdict"] == fast["verdict"] == "regression"
    assert full["transitions"] == fast["transitions"]
    assert fast["metrics"] == [] and _check(fast)["ci"] is None
