"""Checks on the environment's final state, rather than on what the agent said.

"expected": {"state": {...}} is a partial description of the final state:
every key it names must match, keys it leaves out are ignored. Lists are
compared as multisets (order doesn't matter) unless the expectation is
wrapped as {"$ordered": [...]}. {"$absent": true} requires a key to be gone.

This is a structural comparison that reports each difference by path, not a
hash of the whole state, so list order and unrelated fields don't flip it.
"""

from typing import Any

from warden_sdk.evals.scorers.base import Case, Score

MAX_REPORTED = 5


def state_diff(actual: Any, expected: Any, path: str = "") -> list[str]:
    """Where `actual` doesn't match the partial expectation, as 'path: problem' strings."""
    where = path or "state"
    if isinstance(expected, dict) and set(expected) == {"$ordered"}:
        if not isinstance(actual, list) or len(actual) != len(expected["$ordered"]):
            return [f"{where}: expected {expected['$ordered']!r}, got {actual!r}"]
        return [p for i, (a, e) in enumerate(zip(actual, expected["$ordered"])) for p in state_diff(a, e, f"{path}[{i}]")]
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{where}: expected an object, got {actual!r}"]
        problems = []
        for key, want in expected.items():
            sub = f"{path}.{key}" if path else key
            if isinstance(want, dict) and want.get("$absent") is True:
                if key in actual:
                    problems.append(f"{sub}: expected to be absent, got {actual[key]!r}")
            elif key not in actual:
                problems.append(f"{sub}: missing")
            else:
                problems += state_diff(actual[key], want, sub)
        return problems
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{where}: expected a list, got {actual!r}"]
        # Multiset match: every expected element pairs with a distinct actual element.
        remaining = list(actual)
        missing = []
        for want in expected:
            hit = next((i for i, a in enumerate(remaining) if not state_diff(a, want)), None)
            if hit is None:
                missing.append(want)
            else:
                remaining.pop(hit)
        problems = [f"{where}: no element matching {m!r}" for m in missing]
        if len(actual) != len(expected) and not missing:
            problems.append(f"{where}: expected {len(expected)} elements, got {len(actual)}")
        return problems
    return [] if actual == expected else [f"{where}: expected {expected!r}, got {actual!r}"]


def end_state(case: Case) -> Score | None:
    expected = case.expected.get("state")
    if expected is None:
        return None
    if case.environment is None:
        return Score.unscored("no environment was recorded for this item")
    problems = state_diff(case.environment["final_state"], expected)
    if not problems:
        return Score(value=1.0, passed=True)
    shown = "; ".join(problems[:MAX_REPORTED])
    more = f" (+{len(problems) - MAX_REPORTED} more)" if len(problems) > MAX_REPORTED else ""
    return Score(value=0.0, passed=False, reason=shown + more)
