"""Checks on which tools the agent called."""

from collections import Counter
from typing import Literal

from warden_sdk.evals.scorers.base import Case, Score

ToolsMatch = Literal["strict", "unordered", "subset", "superset"]

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
