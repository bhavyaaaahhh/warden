import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from warden_sdk.tracer import Span, Trace

# pass/fail: the check ran and decided. unscored: it couldn't decide (a judge
# gave no verdict, a scorer raised, the input it needs wasn't recorded), so it
# stays out of pass rates instead of counting as either.
Outcome = Literal["pass", "fail", "unscored"]
ToolsMatch = Literal["strict", "unordered", "subset", "superset"]


@dataclass
class Case:
    """Everything a scorer can look at for one trial of one dataset item."""

    item: dict[str, Any]
    output: Any
    error: str | None  # set if the agent raised
    trace: Trace | None  # the first trace the agent opened, None if it opened none
    traces: list[Trace] = field(default_factory=list)
    # Multi-turn cases only: the full conversation, and one record per user turn
    # ({"index", "user", "messages", "tool_calls", "error", "duration_ms"}).
    transcript: list[dict[str, Any]] | None = None
    turns: list[dict[str, Any]] | None = None
    trial: int = 0

    @property
    def expected(self) -> dict[str, Any]:
        return self.item.get("expected") or {}

    @property
    def spans(self) -> list[Span]:
        return [s for t in self.traces for s in t.spans]


@dataclass
class Score:
    value: float | None
    passed: bool | None  # None for metrics, which have no pass/fail of their own
    reason: str | None = None
    outcome: Outcome | None = None  # derived from passed when not given
    criterion: str = ""  # set when one scorer reports several checks, e.g. one per turn

    def __post_init__(self) -> None:
        if self.outcome is None and self.passed is not None:
            self.outcome = "pass" if self.passed else "fail"
        elif self.outcome in ("pass", "fail"):
            self.passed = self.outcome == "pass"
        elif self.outcome == "unscored":
            self.passed = None

    @classmethod
    def unscored(cls, reason: str, criterion: str = "") -> "Score":
        return cls(value=None, passed=None, reason=reason, outcome="unscored", criterion=criterion)


# Returning None means "this scorer doesn't apply to this item". A scorer can
# return a list to report several criteria. Give it a `version` attribute and
# comparisons refuse to compare its scores across versions.
Scorer = Callable[[Case], Score | list[Score] | None]


def _text(output: Any) -> str:
    return output if isinstance(output, str) else json.dumps(output, default=str)


def _contains(output: Any, needles: list[str]) -> str | None:
    haystack = _text(output).lower()
    missing = [n for n in needles if n.lower() not in haystack]
    return f"missing {missing}" if missing else None


def match_tools(actual: list[str], expected: list[str], mode: ToolsMatch = "superset") -> str | None:
    """Compare tool names called against the expected ones. Returns why they differ, or None.

    strict: same calls in the same order. unordered: same calls, any order.
    superset: every expected call was made (extras allowed). subset: no call
    outside the expected ones. Repeated calls count, so ["a", "a"] needs two.
    """
    if mode == "strict":
        return None if actual == expected else f"expected {expected}, called {actual}"
    missing = list((Counter(expected) - Counter(actual)).elements())
    extra = list((Counter(actual) - Counter(expected)).elements())
    problems = []
    if missing and mode in ("unordered", "superset"):
        problems.append(f"missing {missing}")
    if extra and mode in ("unordered", "subset"):
        problems.append(f"unexpected {extra}")
    return "; ".join(problems) or None


def contains(case: Case) -> Score | None:
    needles = case.expected.get("contains")
    if not needles:
        return None
    reason = _contains(case.output, needles)
    missing = len(needles) - sum(n.lower() in _text(case.output).lower() for n in needles)
    return Score(value=1 - missing / len(needles), passed=reason is None, reason=reason)


def exact_match(case: Case) -> Score | None:
    if "exact" not in case.expected:
        return None
    passed = _text(case.output).strip() == str(case.expected["exact"]).strip()
    return Score(value=float(passed), passed=passed, reason=None if passed else "output differs")


def no_errors(case: Case) -> Score:
    errors = [f"{s.name}: {s.error}" for s in case.spans if s.error]
    # An exception raised inside a span is recorded on the span too; don't report it twice.
    if case.error and not any(s.error == case.error for s in case.spans):
        errors.insert(0, case.error)
    return Score(value=float(not errors), passed=not errors, reason="; ".join(errors) or None)


def _called_tools(case: Case) -> list[str] | None:
    """Tool names the agent called, in order. None if nothing was recorded that could say."""
    if case.turns is not None:
        per_turn = [t["tool_calls"] for t in case.turns]
        if any(calls is None for calls in per_turn):
            return None
        return [name for calls in per_turn for name in calls]
    if not case.traces:
        return None  # no trace is not the same as no tool calls
    return [s.name for s in case.spans if s.span_type == "tool_call"]


def tool_calls(case: Case) -> Score | None:
    expected = case.expected.get("tools")
    if expected is None:
        return None
    actual = _called_tools(case)
    if actual is None:
        return Score.unscored("no trace or tool messages recorded")
    reason = match_tools(actual, expected, case.expected.get("tools_match", "superset"))
    return Score(value=float(reason is None), passed=reason is None, reason=reason)


def turn_expectations(case: Case) -> list[Score] | None:
    """One criterion per scripted turn that has an `expect` step after it."""
    if case.turns is None:
        return None
    scores = []
    turn_index = -1
    for step in case.item.get("turns", []):
        if "user" in step:
            turn_index += 1
            continue
        expect = step.get("expect")
        if not expect or turn_index < 0:
            continue
        criterion = f"turn {turn_index + 1}"
        if turn_index >= len(case.turns):
            scores.append(Score(value=0.0, passed=False, reason="turn not reached", criterion=criterion))
            continue
        turn = case.turns[turn_index]
        if turn["error"]:
            scores.append(Score(value=0.0, passed=False, reason=f"agent raised {turn['error']}", criterion=criterion))
            continue
        reply = " ".join(
            _text(m.get("content")) for m in turn["messages"] if m.get("role") == "assistant" and m.get("content")
        )
        problems = []
        if "contains" in expect:
            problems.append(_contains(reply, expect["contains"]))
        if "exact" in expect and reply.strip() != str(expect["exact"]).strip():
            problems.append("reply differs")
        unknown_tools = "tools" in expect and turn["tool_calls"] is None
        if "tools" in expect and not unknown_tools:
            problems.append(match_tools(turn["tool_calls"], expect["tools"], expect.get("tools_match", "superset")))
        problems = [p for p in problems if p]
        # Unrecorded tool calls only make the turn unscored if nothing else already failed it.
        if unknown_tools and not problems:
            scores.append(Score.unscored("no trace or tool messages recorded", criterion))
            continue
        scores.append(
            Score(value=float(not problems), passed=not problems, reason="; ".join(problems) or None, criterion=criterion)
        )
    return scores or None


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


SCORERS: dict[str, Scorer] = {
    "contains": contains,
    "exact_match": exact_match,
    "no_errors": no_errors,
    "tool_calls": tool_calls,
    "turn_expectations": turn_expectations,
    "latency_ms": latency_ms,
    "cost_usd": cost_usd,
    "total_tokens": total_tokens,
    "turns": turns,
}
