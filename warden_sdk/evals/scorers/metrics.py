"""Numeric metrics read from the agent's traces. They have no pass/fail of their own."""

from datetime import datetime

from warden_sdk.evals.scorers.base import Case, Score


def latency_ms(case: Case) -> Score | None:
    timed = [t for t in case.traces if t.started_at and t.ended_at]
    if not timed:
        return None
    total = sum(
        (datetime.fromisoformat(t.ended_at) - datetime.fromisoformat(t.started_at)).total_seconds() for t in timed
    )
    return Score(value=total * 1000, passed=None)


def cost_usd(case: Case) -> Score | None:
    costs = [s.cost_usd for s in case.spans if s.cost_usd is not None]
    return Score(value=sum(costs), passed=None) if costs else None


def total_tokens(case: Case) -> Score | None:
    counts = [n for s in case.spans for n in (s.tokens_input, s.tokens_output) if n is not None]
    return Score(value=float(sum(counts)), passed=None) if counts else None


def turns(case: Case) -> Score | None:
    if case.turns is None:
        return None
    return Score(value=float(len(case.turns)), passed=None)
