from warden_sdk.evals.scorers.base import Case, Outcome, Score, Scorer
from warden_sdk.evals.scorers.checks import (
    contains,
    exact_match,
    no_errors,
    turn_expectations,
    user_goal_met,
)
from warden_sdk.evals.scorers.metrics import cost_usd, latency_ms, total_tokens, turns
from warden_sdk.evals.scorers.state import end_state
from warden_sdk.evals.scorers.tools import ToolsMatch, match_tools, tool_calls

SCORERS: dict[str, Scorer] = {
    "contains": contains,
    "exact_match": exact_match,
    "no_errors": no_errors,
    "tool_calls": tool_calls,
    "end_state": end_state,
    "turn_expectations": turn_expectations,
    "user_goal_met": user_goal_met,
    "latency_ms": latency_ms,
    "cost_usd": cost_usd,
    "total_tokens": total_tokens,
    "turns": turns,
}

__all__ = ["SCORERS", "Case", "Outcome", "Score", "Scorer", "ToolsMatch", "match_tools"]
