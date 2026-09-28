import json
import logging
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

log = logging.getLogger("warden_sdk")

WARDEN_URL = os.environ.get("WARDEN_URL", "http://localhost:8000")
# Strict mode is for eval runs: a trace that fails to upload raises instead of
# being silently dropped, so it can't vanish from a regression comparison.
WARDEN_STRICT = os.environ.get("WARDEN_STRICT", "").lower() in ("1", "true", "yes")


class WardenError(Exception):
    """Raised in strict mode when a trace could not be stored."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    # Round-trip through json so odd types (objects, sets, datetimes) become
    # strings instead of breaking the upload.
    if value is None:
        return None
    return json.loads(json.dumps(value, default=str))


@dataclass
class _EvalItem:
    run_id: str
    item_id: str
    traces: list["Trace"] = field(default_factory=list)


# Set by the eval runner around each dataset item, so traces the agent opens
# report back to the runner without the agent's code knowing it's being evaluated.
_current_eval_item: ContextVar[_EvalItem | None] = ContextVar("warden_eval_item", default=None)


@contextmanager
def _eval_item(run_id: str, item_id: str) -> Iterator[_EvalItem]:
    item = _EvalItem(run_id, item_id)
    token = _current_eval_item.set(item)
    try:
        yield item
    finally:
        _current_eval_item.reset(token)


class Span:
    def __init__(self, trace: "Trace", span_type: str, name: str, input: Any = None):
        self.trace = trace
        self.span_id = str(uuid.uuid4())
        self.span_type = span_type
        self.name = name
        self.input = input
        self.output: Any = None
        self.tokens_input: int | None = None
        self.tokens_output: int | None = None
        self.cost_usd: float | None = None
        self.error: str | None = None
        self.parent_span_id: str | None = None
        self.started_at: str | None = None
        self.ended_at: str | None = None

    def __enter__(self) -> "Span":
        stack = self.trace._stack
        self.parent_span_id = stack[-1].span_id if stack else None
        self.started_at = _now()
        stack.append(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.ended_at = _now()
        if exc is not None:
            self.error = f"{exc_type.__name__}: {exc}"
            self.trace.status = "error"
        self.trace._stack.pop()
        self.trace._finished.append(self)
        return False  # never swallow the caller's exception

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace.trace_id,
            "parent_span_id": self.parent_span_id,
            "span_type": self.span_type,
            "name": self.name,
            "input": _jsonable(self.input),
            "output": _jsonable(self.output),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "cost_usd": self.cost_usd,
            "error": self.error,
        }


class Trace:
    def __init__(
        self,
        agent_name: str,
        input: Any = None,
        version_tag: str | None = None,
        metadata: dict[str, Any] | None = None,
        strict: bool | None = None,
    ):
        self.strict = WARDEN_STRICT if strict is None else strict
        self.trace_id = str(uuid.uuid4())
        self.agent_name = agent_name
        self.input = input
        self.version_tag = version_tag
        self.metadata = metadata or {}
        self.status = "success"
        self.started_at: str | None = None
        self.ended_at: str | None = None
        self._stack: list[Span] = []
        self._finished: list[Span] = []

    def span(self, span_type: str, name: str, input: Any = None) -> Span:
        return Span(self, span_type, name, input)

    @property
    def spans(self) -> list[Span]:
        return list(self._finished)

    def __enter__(self) -> "Trace":
        self.started_at = _now()
        eval_item = _current_eval_item.get()
        if eval_item is not None:
            # Evals must never lose a trace, so they always run strict.
            self.strict = True
            self.metadata = {
                **self.metadata,
                "eval_run_id": eval_item.run_id,
                "eval_item_id": eval_item.item_id,
            }
            eval_item.traces.append(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.ended_at = _now()
        if exc is not None:
            self.status = "error"
        try:
            self._flush()
        except WardenError as e:
            if exc is None:
                raise
            # Don't mask the agent's own exception; attach the upload failure to it.
            exc.add_note(str(e))
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "version_tag": self.version_tag,
            "input": _jsonable(self.input),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "metadata": _jsonable(self.metadata),
        }

    def _flush(self) -> None:
        # Trace first, then spans by start time, so both FKs are satisfied.
        # Synchronous: when this returns without error, the trace is queryable.
        # Outside strict mode a failure here must never break the agent.
        spans = sorted(self._finished, key=lambda s: s.started_at)
        try:
            with httpx.Client(base_url=WARDEN_URL, timeout=5.0) as client:
                client.post("/traces", json=self.to_dict()).raise_for_status()
                if spans:
                    client.post("/spans", json=[s.to_dict() for s in spans]).raise_for_status()
        except Exception as e:
            if self.strict:
                raise WardenError(f"warden: failed to store trace {self.trace_id}: {e}") from e
            log.warning("warden: failed to send trace %s: %s", self.trace_id, e)


def trace(
    agent_name: str,
    input: Any = None,
    version_tag: str | None = None,
    metadata: dict[str, Any] | None = None,
    strict: bool | None = None,
) -> Trace:
    return Trace(agent_name, input=input, version_tag=version_tag, metadata=metadata, strict=strict)
