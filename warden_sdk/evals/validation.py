"""Check an LLM judge against human labels before trusting it.

1. export-labels writes a run's judged criteria to a JSONL file with an empty
   "label" field. It deliberately leaves out the judge's verdict, so labellers
   aren't anchored by it.
2. People fill in "label" with "pass" or "fail".
3. validate-judge runs the judge on every labelled row, `repeats` times, and
   reports agreement with the humans (Cohen's kappa, with a bootstrap CI), the
   confusion matrix, how often the judge contradicts itself, and how often it
   couldn't decide. The result is stored against the judge's version.

A judge counts as validated with at least MIN_LABELS labels and kappa >= MIN_KAPPA
(docs/research/evaluation-methodology.md §8).
"""

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from warden_sdk.evals.client import fetch_run
from warden_sdk.evals.compare import stats
from warden_sdk.evals.scorers.base import Case
from warden_sdk.evals.scorers.judge import LLMJudge
from warden_sdk.tracer import WARDEN_URL

MIN_LABELS = 50
MIN_KAPPA = 0.6


def _split_turn(label: str) -> tuple[int | None, str]:
    head, sep, rest = label.partition(": ")
    if sep and head.startswith("turn ") and head[5:].isdigit():
        return int(head[5:]), rest
    return None, label


def export_labels(run_ref: str, scorer: str, out: Path, per_item: bool = False) -> int:
    """Write rows for people to label from the first trial of each item.

    One row per criterion `scorer` judged, or with per_item one row per item
    for a verdict on the whole thing.
    """
    with httpx.Client(base_url=WARDEN_URL, timeout=10.0) as client:
        run = fetch_run(client, run_ref)
    rows = []
    for r in run["results"]:
        if (r.get("trial") or 0) != 0 or r.get("termination") == "infra_error":
            continue
        if per_item:
            rows.append({"id": r["item_id"], "item_id": r["item_id"], "input": r.get("input"), "transcript": r.get("transcript"),
                         "output": r.get("output"), "criterion": None, "turn": None, "label": None})
            continue
        for s in r["scores"]:
            if s["scorer"] != scorer or not s.get("criterion"):
                continue
            turn, criterion = _split_turn(s["criterion"])
            rows.append({
                "id": f"{r['item_id']}/{s['criterion']}",
                "item_id": r["item_id"],
                "input": r.get("input"),
                "transcript": r.get("transcript"),
                "output": r.get("output"),
                "criterion": criterion,
                "turn": turn,
                "label": None,
            })
    if not rows:
        raise SystemExit(f"run {run_ref} has no criteria scored by {scorer!r} (use --per-item to label whole items)")
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    then = "rescore the run with --labels" if per_item else "run validate-judge, or rescore the run with --labels"
    print(f"wrote {len(rows)} rows to {out}; set each \"label\" to \"pass\" or \"fail\", then {then}")
    return 0


def load_labels(path: Path) -> list[dict[str, Any]]:
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("label") is None:
            continue  # not labelled yet
        if row["label"] not in ("pass", "fail") or not row.get("criterion"):
            raise ValueError(f"{path}:{lineno}: needs 'criterion' and a 'label' of 'pass' or 'fail'")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no labelled rows")
    return rows


def _case(row: dict[str, Any]) -> Case:
    """Rebuild what the judge saw, with only this row's criterion to grade."""
    transcript = row.get("transcript")
    item: dict[str, Any] = {"id": row["id"], "input": row.get("input")}
    if transcript and row.get("turn"):
        # A turn-scoped criterion: rebuild the user steps so the judge scopes it the same way.
        steps: list[dict[str, Any]] = []
        for m in transcript:
            if m.get("role") == "user":
                steps.append({"user": m.get("content")})
                if sum("user" in s for s in steps) == row["turn"]:
                    steps.append({"expect": {"criteria": [row["criterion"]]}})
        item = {"id": row["id"], "turns": steps}
    else:
        item["expected"] = {"criteria": [row["criterion"]]}
    return Case(item=item, output=row.get("output"), error=None, trace=None, transcript=transcript)


def validate_judge(labels_path: Path, judge: LLMJudge, repeats: int = 3, seed: int = 0) -> dict[str, Any]:
    rows = load_labels(labels_path)
    runs: list[list[str]] = []  # per row, the judge's outcome on each repeat
    for i, row in enumerate(rows):
        outcomes = []
        for _ in range(repeats):
            [score] = judge(_case(row)) or [None]
            outcomes.append(score.outcome if score else "unscored")
        runs.append(outcomes)
        print(f"  {i + 1}/{len(rows)} {row['id'][:60]:<60} human {row['label']:<5} judge {' '.join(outcomes)}")

    first = [o[0] for o in runs]
    pairs = [(row["label"], out) for row, out in zip(rows, first) if out != "unscored"]
    kappa = stats.cohens_kappa(pairs)
    ci = stats.bootstrap_ci(len(pairs), lambda idx: stats.cohens_kappa([pairs[j] for j in idx]), random.Random(seed))
    confusion = Counter(f"human {h} / judge {j}" for h, j in pairs)
    decided = [o for o in runs if all(x != "unscored" for x in o)]
    flip_rate = sum(len(set(o)) > 1 for o in decided) / len(decided) if decided else None
    unscored = sum(out == "unscored" for out in first)
    validated = len(pairs) >= MIN_LABELS and kappa is not None and kappa >= MIN_KAPPA
    return {
        "scorer_version": judge.version,
        "labels": len(rows),
        "compared": len(pairs),
        "kappa": kappa,
        "kappa_ci": list(ci) if ci else None,
        "confusion": dict(sorted(confusion.items())),
        "flip_rate": flip_rate,
        "repeats": repeats,
        "unscored": unscored,
        "validated": validated,
    }


def print_validation(report: dict[str, Any]) -> int:
    kappa = report["kappa"]
    ci = report["kappa_ci"]
    print(f"\njudge {report['scorer_version']}")
    print(f"  {report['compared']} of {report['labels']} labelled rows compared ({report['unscored']} unscored)")
    print(f"  Cohen's kappa   {'-' if kappa is None else f'{kappa:.2f}'}" + (f"  [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""))
    for cell, n in report["confusion"].items():
        print(f"  {cell:<28}{n}")
    if report["flip_rate"] is not None:
        print(f"  changed its verdict across {report['repeats']} repeats on {report['flip_rate']:.0%} of rows")
    if report["validated"]:
        print(f"\n✓ validated (kappa ≥ {MIN_KAPPA} on ≥ {MIN_LABELS} labels)")
        return 0
    why = f"only {report['compared']} labels (need {MIN_LABELS})" if report["compared"] < MIN_LABELS else f"kappa below {MIN_KAPPA}"
    print(f"\n✗ not validated: {why}")
    return 1


def store_validation(report: dict[str, Any]) -> None:
    with httpx.Client(base_url=WARDEN_URL, timeout=10.0) as client:
        client.post("/judge_validations", json=report).raise_for_status()


def judge_versions(runs: list[dict[str, Any]]) -> list[str]:
    return sorted({
        s["scorer_version"]
        for run in runs for r in run["results"] for s in r["scores"]
        if (s.get("scorer_version") or "").startswith("judge/")
    })


def validation_warning(version: str, latest: dict[str, Any] | None) -> str | None:
    """Why a judge version can't be trusted yet, or None if it passed validation."""
    if not latest:
        return f"judge {version} has not been validated against human labels"
    if latest["validated"]:
        return None
    if latest["compared"] < MIN_LABELS:
        return f"judge {version} was validated on only {latest['compared']} labels (need {MIN_LABELS})"
    return f"judge {version} disagrees with human labels (kappa {latest['kappa']:.2f}, need {MIN_KAPPA})"


def judge_warnings(client: httpx.Client, runs: list[dict[str, Any]]) -> list[str]:
    """Warnings for judge scorers in these runs that haven't passed validation."""
    warnings = []
    for version in judge_versions(runs):
        res = client.get("/judge_validations", params={"scorer_version": version})
        warning = validation_warning(version, res.json() if res.status_code == 200 else None)
        if warning:
            warnings.append(warning)
    return warnings
