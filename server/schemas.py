from datetime import datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

# Keep these in sync with the CHECK constraints in the migration.
TraceStatus = Literal["running", "success", "error"]
SpanType = Literal["llm_call", "tool_call", "retrieval"]
EvalRunStatus = Literal["running", "completed", "failed"]
Termination = Literal["completed", "agent_error", "infra_error"]
Outcome = Literal["pass", "fail", "unscored"]


class TraceIn(BaseModel):
    trace_id: UUID
    agent_name: str
    version_tag: str | None = None
    input: Any = None
    started_at: datetime
    ended_at: datetime | None = None
    status: TraceStatus
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpanIn(BaseModel):
    span_id: UUID
    trace_id: UUID
    parent_span_id: UUID | None = None
    span_type: SpanType
    name: str
    input: Any = None
    output: Any = None
    started_at: datetime
    ended_at: datetime | None = None
    tokens_input: int | None = None
    tokens_output: int | None = None
    cost_usd: Decimal | None = None
    error: str | None = None


class EvalRunIn(BaseModel):
    run_id: UUID
    dataset_name: str
    dataset_hash: str
    agent: str
    version_tag: str | None = None
    trials: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalRunUpdate(BaseModel):
    status: EvalRunStatus
    version_tag: str | None = None


class ScoreIn(BaseModel):
    scorer: str
    value: float | None = None
    passed: bool | None = None
    reason: str | None = None
    outcome: Outcome | None = None
    criterion: str = ""
    scorer_version: str | None = None


class EvalResultIn(BaseModel):
    result_id: UUID
    item_id: str
    trial: int = 0
    case_hash: str | None = None
    termination: Termination = "completed"
    trace_id: UUID | None = None
    input: Any = None
    expected: Any = None
    output: Any = None
    error: str | None = None
    transcript: list[dict[str, Any]] | None = None
    turns: list[dict[str, Any]] | None = None
    scores: list[ScoreIn] = Field(default_factory=list)
