"""Checks on which tools the agent called, and with what arguments.

An expected call is a tool name, or {"name": ..., "args": {...}} to also check
arguments. By default the expected args must all be present with equal values
and extra args are allowed ("args_match": "subset"); "exact" requires the
arguments to be exactly those. A key missing from the call fails, even if the
value is null in the expectation.

How the calls line up with the expected ones ("tools_match"):
  strict     the same calls, in the same order
  unordered  the same calls, in any order
  superset   every expected call was made; extra calls are fine (the default)
  subset     every call made was expected; not all expected calls are needed
Calls are paired by maximum bipartite matching, not first come first served,
so an early call that fits two expectations can't starve a later one.
"""

import json
from dataclasses import dataclass
from typing import Any, Literal

from warden_sdk.evals.scorers.base import Case, Score

ToolsMatch = Literal["strict", "unordered", "subset", "superset"]
ArgsMatch = Literal["subset", "exact"]
_MISSING = object()


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: Any = None  # None when the arguments weren't recorded


@dataclass(frozen=True)
class ExpectedCall:
    name: str
    args: dict[str, Any] | None = None  # None: any arguments
    args_match: ArgsMatch = "subset"

    def describe(self) -> str:
        return self.name if self.args is None else f"{self.name}({json.dumps(self.args, sort_keys=True)})"


def parse_args(args: Any) -> Any:
    """OpenAI-style tool calls carry arguments as a JSON string."""
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return args
    return args


def as_call(call: Any) -> ToolCall:
    """A recorded call: a bare name (older runs) or {"name", "args"}."""
    if isinstance(call, str):
        return ToolCall(call)
    return ToolCall(call["name"], parse_args(call.get("args")))


def as_expected(spec: Any) -> ExpectedCall:
    if isinstance(spec, str):
        return ExpectedCall(spec)
    if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
        raise ValueError(f"an expected tool call is a name or {{'name': ..., 'args': {{...}}}}, got {spec!r}")
    return ExpectedCall(spec["name"], spec.get("args"), spec.get("args_match", "subset"))


def call_fits(call: ToolCall, expected: ExpectedCall) -> str | None:
    """Why this call doesn't satisfy the expectation, or None if it does."""
    if call.name != expected.name:
        return f"called {call.name}, expected {expected.name}"
    if expected.args is None:
        return None
    if not isinstance(call.args, dict):
        return f"{call.name}: arguments weren't recorded" if call.args is None else f"{call.name}: arguments aren't an object"
    wrong = [
        key for key, want in expected.args.items()
        if call.args.get(key, _MISSING) is _MISSING or call.args[key] != want
    ]
    if wrong:
        got = {k: call.args.get(k, "<missing>") for k in wrong}
        return f"{call.name}: expected {json.dumps({k: expected.args[k] for k in wrong})}, got {json.dumps(got, default=str)}"
    if expected.args_match == "exact" and set(call.args) != set(expected.args):
        return f"{call.name}: unexpected arguments {sorted(set(call.args) - set(expected.args))}"
    return None


def _max_matching(fits: list[list[bool]]) -> list[int | None]:
    """For each expected call, the actual call it's paired with (maximum bipartite matching)."""
    n_actual = len(fits[0]) if fits else 0
    owner: list[int | None] = [None] * n_actual  # actual index -> expected index

    def assign(e: int, seen: set[int]) -> bool:
        for a in range(n_actual):
            if fits[e][a] and a not in seen:
                seen.add(a)
                if owner[a] is None or assign(owner[a], seen):
                    owner[a] = e
                    return True
        return False

    for e in range(len(fits)):
        assign(e, set())
    pairs: list[int | None] = [None] * len(fits)
    for a, e in enumerate(owner):
        if e is not None:
            pairs[e] = a
    return pairs


def match_tools(actual: list[Any], expected: list[Any], mode: ToolsMatch = "superset") -> str | None:
    """Compare the calls made against the expected ones. Returns why they differ, or None."""
    calls = [as_call(c) for c in actual]
    wanted = [as_expected(e) for e in expected]
    if mode == "strict":
        if len(calls) != len(wanted):
            return f"expected {[w.describe() for w in wanted]}, called {[c.name for c in calls]}"
        for i, (call, want) in enumerate(zip(calls, wanted)):
            if problem := call_fits(call, want):
                return f"call {i + 1}: {problem}"
        return None

    fits = [[call_fits(c, w) is None for c in calls] for w in wanted]
    pairs = _max_matching(fits)
    used = {a for a in pairs if a is not None}
    problems = []
    if mode in ("unordered", "superset"):
        for want, paired in zip(wanted, pairs):
            if paired is None:
                # Say why the closest same-named call didn't fit, if there was one.
                near = next((call_fits(c, want) for c in calls if c.name == want.name), None)
                problems.append(f"missing {want.describe()}" + (f" ({near})" if near else ""))
    if mode in ("unordered", "subset"):
        extra = [c.name for i, c in enumerate(calls) if i not in used]
        if extra:
            problems.append(f"unexpected {extra}")
    return "; ".join(problems) or None


def _called_tools(case: Case) -> list[Any] | None:
    """Calls the agent made, in order. None if nothing was recorded that could say."""
    if case.turns is not None:
        per_turn = [t["tool_calls"] for t in case.turns]
        # A plain-text reply with no trace can't say whether tools ran. Once some
        # turn of the conversation has reported its calls, the agent evidently
        # reports them, so a silent turn made none.
        if all(calls is None for calls in per_turn):
            return None
        return [call for calls in per_turn for call in calls or []]
    if case.environment is not None:
        return [{"name": c["name"], "args": c["args"]} for c in case.environment["calls"]]
    if not case.traces:
        return None  # no trace is not the same as no tool calls
    return [{"name": s.name, "args": s.input} for s in case.spans if s.span_type == "tool_call"]


def tool_calls(case: Case) -> Score | None:
    expected = case.expected.get("tools")
    if expected is None:
        return None
    actual = _called_tools(case)
    if actual is None:
        return Score.unscored("no trace or tool messages recorded")
    reason = match_tools(actual, expected, case.expected.get("tools_match", "superset"))
    return Score(value=float(reason is None), passed=reason is None, reason=reason)
