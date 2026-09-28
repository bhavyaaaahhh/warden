from datetime import datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

# Keep these in sync with the CHECK constraints in the migration.
TraceStatus = Literal["running", "success", "error"]
SpanType = Literal["llm_call", "tool_call", "retrieval"]


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

