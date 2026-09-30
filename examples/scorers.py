"""Example custom scorer, for use with --scorer examples.scorers:mentions_order."""

import re

from warden_sdk.evals import Case, Score


def mentions_order(case: Case) -> Score | None:
    """Passes if the agent's final reply names an order number."""
    if case.output is None:
        return None
    found = re.search(r"\d+", str(case.output))
    return Score(value=float(bool(found)), passed=bool(found), reason=None if found else "no order number in reply")


mentions_order.version = "1"
