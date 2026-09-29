"""Pass/fail checks on the agent's output and on each scripted turn."""

from typing import Any

from warden_sdk.evals.scorers.base import Case, Score, _text
from warden_sdk.evals.scorers.tools import match_tools


def _contains(output: Any, needles: list[str]) -> str | None:
    haystack = _text(output).lower()
    missing = [n for n in needles if n.lower() not in haystack]
    return f"missing {missing}" if missing else None


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


def user_goal_met(case: Case) -> Score | None:
    """Whether the simulated user said its goal was met, for scenario items.

    This is the simulated user's own opinion, so it's only as good as the
    simulator. Prefer a judge criterion or a tool check for anything that matters.
    """
    if case.simulation is None:
        return None
    stopped_by, reason = case.simulation.get("stopped_by"), case.simulation.get("stop_reason")
    if stopped_by == "simulator" and reason == "goal_met":
        return Score(value=1.0, passed=True)
    why = {"max_turns": "ran out of turns", "agent_error": "the agent raised"}.get(stopped_by, f"user {reason}")
    return Score(value=0.0, passed=False, reason=why)
