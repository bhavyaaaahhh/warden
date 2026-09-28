import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from warden_sdk.tracer import Trace


@dataclass
class Case:
    """Everything a scorer can look at for one dataset item."""

    item: dict[str, Any]
    output: Any
    error: str | None  # set if the agent raised
    trace: Trace | None  # None if the agent never opened a trace

    @property
    def expected(self) -> dict[str, Any]:
        return self.item.get("expected") or {}


@dataclass
class Score:
    value: float | None
    passed: bool | None  # None for metrics, which have no pass/fail of their own
    reason: str | None = None


# Returning None means "this scorer doesn't apply to this item".
Scorer = Callable[[Case], Score | None]


def _text(output: Any) -> str:
    return output if isinstance(output, str) else json.dumps(output, default=str)


def contains(case: Case) -> Score | None:
    needles = case.expected.get("contains")
    if not needles:
        return None
    haystack = _text(case.output).lower()
    missing = [n for n in needles if n.lower() not in haystack]
    return Score(
        value=1 - len(missing) / len(needles),
        passed=not missing,
        reason=f"missing {missing}" if missing else None,
    )


def exact_match(case: Case) -> Score | None:
    if "exact" not in case.expected:
        return None
    passed = _text(case.output).strip() == str(case.expected["exact"]).strip()
    return Score(value=float(passed), passed=passed, reason=None if passed else "output differs")


def no_errors(case: Case) -> Score:
    spans = case.trace.spans if case.trace is not None else []
    errors = [f"{s.name}: {s.error}" for s in spans if s.error]
    # An exception raised inside a span is recorded on the span too; don't report it twice.
    if case.error and not any(s.error == case.error for s in spans):
        errors.insert(0, case.error)
    return Score(value=float(not errors), passed=not errors, reason="; ".join(errors) or None)


def latency_ms(case: Case) -> Score | None:
    t = case.trace
    if t is None or not (t.started_at and t.ended_at):
        return None
    delta = datetime.fromisoformat(t.ended_at) - datetime.fromisoformat(t.started_at)
    return Score(value=delta.total_seconds() * 1000, passed=None)


def cost_usd(case: Case) -> Score | None:
    costs = [s.cost_usd for s in case.trace.spans if s.cost_usd is not None] if case.trace else []
    return Score(value=sum(costs), passed=None) if costs else None


def total_tokens(case: Case) -> Score | None:
    if case.trace is None:
        return None
    counts = [
        n for s in case.trace.spans for n in (s.tokens_input, s.tokens_output) if n is not None
    ]
    return Score(value=float(sum(counts)), passed=None) if counts else None


SCORERS: dict[str, Scorer] = {
    "contains": contains,
    "exact_match": exact_match,
    "no_errors": no_errors,
    "latency_ms": latency_ms,
    "cost_usd": cost_usd,
    "total_tokens": total_tokens,
}
