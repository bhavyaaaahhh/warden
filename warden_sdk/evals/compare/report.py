"""The terminal report for a compare_runs() result."""

import json
from typing import Any

from warden_sdk.evals.compare.compare import TRANSITION_ORDER


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

def _fmt_rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"

def _fmt_ci(ci: list[float] | None, pct: bool = False) -> str:
    if not ci:
        return ""
    if pct:
        return f"[{ci[0]:+.0f}%, {ci[1]:+.0f}%]"
    return f"[{ci[0] * 100:+.0f}, {ci[1] * 100:+.0f}]"

VERDICT_MARKS = {"regression": "✗", "improvement": "✓", "inconclusive": "?", "incomparable": "?"}

def print_report(cmp: dict[str, Any]) -> int:
    """Print a compare_runs() result as a terminal report. Returns the exit code."""
    b, c = cmp["baseline"], cmp["candidate"]
    print(f"warden diff  {_label(b)}  →  {_label(c)}")
    print(
        f"dataset {c['dataset_name']}, {cmp['items_compared']} items compared, "
        f"{b.get('trials') or 1} vs {c.get('trials') or 1} trials per item"
    )
    for warning in cmp["warnings"]:
        print(f"  ⚠ {warning}")

    print(f"\n  {'check':<20}{'baseline':>10}{'candidate':>11}{'change (95% CI, pts)':>26}{'p':>8}  verdict")
    for r in cmp["checks"]:
        delta = "" if r["delta"] is None else f"{r['delta'] * 100:+.0f} {_fmt_ci(r['ci'])}"
        mark = VERDICT_MARKS.get(r["verdict"], " ")
        print(
            f"  {r['scorer']:<20}{_fmt_rate(r['baseline_rate']):>10}{_fmt_rate(r['candidate_rate']):>11}"
            f"{delta:>26}{r['p_adjusted']:>8.3f}  {mark} {r['verdict']}"
        )
        if r["note"]:
            print(f"  {'':<20}{r['note']}")
        pk = r["pass_k"]
        if pk["baseline"] is not None or pk["candidate"] is not None:
            print(
                f"  {'':<20}pass^k  {_fmt_rate(pk['baseline'])} (k={pk['k_baseline']})"
                f" → {_fmt_rate(pk['candidate'])} (k={pk['k_candidate']})"
            )

    for label in TRANSITION_ORDER:
        cases = [t for t in cmp["transitions"] if t["label"] == label]
        if not cases:
            continue
        print(f"\n{label.upper()} ({len(cases)})")
        for t in cases:
            moved = ", ".join(
                f"{name} {s['baseline']} → {s['candidate']}" for name, s in t["scorers"].items() if s["label"] == label
            )
            print(f"  {t['item_id']}  {moved}")
            if label in ("broke", "degraded"):
                print(f"      input      {_short(t['input'])}")
                print(f"      candidate  {_short(t['candidate_output'])}")
                for name, s in t["scorers"].items():
                    if s["reason"]:
                        print(f"      {name}: {s['reason']}")

    if cmp["metrics"]:
        print(f"\n  {'metric':<20}{'baseline':>12}{'candidate':>12}{'change (95% CI)':>24}  verdict")
    for m in cmp["metrics"]:
        label = f"{m['scorer']} {m['aggregate']}"
        change = f"{m['change_pct']:+.0f}% {_fmt_ci(m['ci'], pct=True)}"
        mark = " ▲" if m["flagged"] else ""
        print(
            f"  {label:<20}{_fmt_num(m['baseline']):>12}{_fmt_num(m['candidate']):>12}{change:>24}"
            f"  {m['verdict']}{mark}"
        )

    regressed = [r["scorer"] for r in cmp["checks"] if r["verdict"] == "regression"]
    undecided = [r["scorer"] for r in cmp["checks"] if r["verdict"] in ("inconclusive", "incomparable")]
    if cmp["verdict"] == "regression":
        print(f"\n✗ regression: {', '.join(regressed)}")
    elif cmp["verdict"] == "undecided":
        print(f"\n? could not decide: {', '.join(undecided) or 'a run did not complete'}")
    else:
        broke = sum(t["label"] == "broke" for t in cmp["transitions"])
        if broke:
            print(f"\n✓ no regression detected, though {broke} case(s) broke: too few to tell from noise")
        else:
            print("\n✓ no regression detected")
    return cmp["exit_code"]
