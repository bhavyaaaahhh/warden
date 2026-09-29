"""What a scorer receives (Case) and returns (Score)."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from warden_sdk.tracer import Span, Trace

# pass/fail: the check ran and decided. unscored: it couldn't decide (a judge
# gave no verdict, a scorer raised, the input it needs wasn't recorded), so it
# stays out of pass rates instead of counting as either.
Outcome = Literal["pass", "fail", "unscored"]


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
