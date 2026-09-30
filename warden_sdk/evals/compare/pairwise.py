"""Pairwise comparison: which of two runs handled each case better, judged side by side.

For quality that's hard to put as pass/fail criteria ("which reply is more
helpful?"), a judge sees both runs' conversations for the same case and picks
one. Every case is judged in both orders (A first, then B first). If the two
orders disagree the case is a tie, which cancels out a judge that favours
whichever answer comes first (Zheng et al. 2023, MT-Bench).

The result is a win rate for the candidate, with a bootstrap CI and an exact
two-sided sign test on wins vs losses. Ties are reported, not dropped.
"""

import hashlib
import json
import random
from typing import Any

from warden_sdk.evals.compare import stats
from warden_sdk.evals.models import Model, ModelError
from warden_sdk.evals.scorers.base import Case
from warden_sdk.evals.scorers.judge import render_transcript

SYSTEM_PROMPT = """You compare how two AI agents handled the same request. You'll see the request, both conversations (labelled FIRST and SECOND), and optionally what to focus on.

Judge only what's in the conversations. Don't prefer an answer for being longer or for coming first. Reason briefly about the differences that matter, then choose "first", "second", or "tie" when neither is meaningfully better."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "winner": {"type": "string", "enum": ["first", "second", "tie"]},
    },
    "required": ["reasoning", "winner"],
    "additionalProperties": False,
}

PROMPT_HASH = hashlib.sha256((SYSTEM_PROMPT + json.dumps(SCHEMA, sort_keys=True)).encode()).hexdigest()[:8]
ALPHA = 0.05


def _case(result: dict[str, Any]) -> Case:
    item: dict[str, Any] = {"id": result["item_id"], "input": result.get("input")}
    return Case(item=item, output=result.get("output"), error=result.get("error"), trace=None,
                transcript=result.get("transcript"))


def _ask(model: Model, question: str, first: str, second: str) -> str | None:
    prompt = f"<focus>\n{question}\n</focus>\n<first>\n{first}\n</first>\n<second>\n{second}\n</second>"
    for _ in range(2):
        try:
            winner = model.json(SYSTEM_PROMPT, [{"role": "user", "content": prompt}], SCHEMA).get("winner")
        except ModelError:
            continue
        if winner in ("first", "second", "tie"):
            return winner
    return None


def judge_pair(model: Model, baseline: dict[str, Any], candidate: dict[str, Any], question: str) -> str:
    """'candidate', 'baseline', 'tie', or 'unscored' for one case, judged in both orders."""
    a, b = render_transcript(_case(baseline)), render_transcript(_case(candidate))
    candidate_first = _ask(model, question, b, a)
    candidate_second = _ask(model, question, a, b)
    if candidate_first is None or candidate_second is None:
        return "unscored"
    # Normalise each order to who won, then require both orders to agree.
    first = {"first": "candidate", "second": "baseline", "tie": "tie"}[candidate_first]
    second = {"first": "baseline", "second": "candidate", "tie": "tie"}[candidate_second]
    return first if first == second else "tie"


def pairwise(
    model: Model, baseline: dict[str, Any], candidate: dict[str, Any], question: str, seed: int = 0
) -> dict[str, Any]:
    """Judge the first trial of every case the two runs share (and haven't changed)."""
    firsts = [{r["item_id"]: r for r in run["results"] if (r.get("trial") or 0) == 0 and r.get("termination") != "infra_error"}
              for run in (baseline, candidate)]
    ids = [i for i in firsts[0] if i in firsts[1] and firsts[0][i].get("case_hash") == firsts[1][i].get("case_hash")]
    outcomes = {}
    for i, item_id in enumerate(ids):
        outcomes[item_id] = judge_pair(model, firsts[0][item_id], firsts[1][item_id], question)
        print(f"  {i + 1}/{len(ids)} {item_id:<40} {outcomes[item_id]}")

    decided = [o for o in outcomes.values() if o != "unscored"]
    wins, losses = decided.count("candidate"), decided.count("baseline")
    ties = decided.count("tie")
    # Win rate counting a tie as half a win, per case, with a bootstrap CI over cases.
    points = [{"candidate": 1.0, "tie": 0.5, "baseline": 0.0}[o] for o in decided]
    rate = sum(points) / len(points) if points else None
    ci = stats.bootstrap_ci(len(points), lambda idx: sum(points[j] for j in idx) / len(idx), random.Random(seed))
    p = stats.binomial_two_sided(wins, wins + losses)
    verdict = "no detectable difference"
    if len(decided) >= 5 and p < ALPHA:
        verdict = "candidate better" if wins > losses else "baseline better"
    return {
        "judge": f"pairwise/{model.name}/{PROMPT_HASH}",
        "question": question,
        "cases": len(ids),
        "wins": wins, "losses": losses, "ties": ties,
        "unscored": len(outcomes) - len(decided),
        "win_rate": rate, "ci": list(ci) if ci else None, "p": p,
        "verdict": verdict,
        "outcomes": outcomes,
    }


def print_pairwise(result: dict[str, Any]) -> int:
    rate, ci = result["win_rate"], result["ci"]
    print(f"\npairwise {result['judge']} on {result['cases']} cases: {result['question']}")
    print(f"  candidate won {result['wins']}, lost {result['losses']}, tied {result['ties']}"
          + (f", {result['unscored']} unscored" if result["unscored"] else ""))
    if rate is not None:
        print(f"  win rate {rate:.0%}" + (f" [{ci[0]:.0%}, {ci[1]:.0%}]" if ci else "") + f"  p = {result['p']:.3f}")
    mark = {"candidate better": "✓", "baseline better": "✗"}.get(result["verdict"], " ")
    print(f"\n{mark} {result['verdict']}")
    return 1 if result["verdict"] == "baseline better" else 0
