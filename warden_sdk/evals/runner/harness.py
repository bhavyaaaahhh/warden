"""Calling the agent under test: one input, or a scripted conversation turn by turn."""

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from warden_sdk.tracer import Trace, WardenError, _eval_item, _jsonable


@dataclass(frozen=True)
class EvalContext:
    """Passed to agents that take a `context` argument.

    thread_id is stable across the turns of one trial and unique per trial, so
    an agent that keeps conversation state server-side can key it on this.
    """

    run_id: str
    item_id: str
    trial: int

    @property
    def thread_id(self) -> str:
        return f"{self.run_id}:{self.item_id}:{self.trial}"


# How the runner calls an agent: with its input and an EvalContext.
AgentCall = Callable[[Any, EvalContext], Any]


class InfraError(Exception):
    """Raise from an agent to say a failure wasn't the agent's fault (rate limit,
    provider outage). The trial is retried, and if it keeps failing it's left
    out of pass rates instead of counting as a failure."""


def _agent_call(agent: Callable) -> AgentCall:
    """Pass the EvalContext only to agents that ask for it with a `context` parameter."""
    try:
        wants_context = "context" in inspect.signature(agent).parameters
    except (TypeError, ValueError):
        wants_context = False
    if wants_context:
        return lambda arg, ctx: agent(arg, context=ctx)
    return lambda arg, ctx: agent(arg)


def _as_messages(reply: Any) -> list[dict[str, Any]]:
    """A multi-turn agent returns a string, one message, or a list of messages."""
    if isinstance(reply, str):
        return [{"role": "assistant", "content": reply}]
    if isinstance(reply, dict):
        return [reply]
    if isinstance(reply, list) and all(isinstance(m, dict) for m in reply):
        return reply
    raise TypeError(f"multi-turn agent must return a str, a message dict, or a list of them, not {type(reply).__name__}")


def _tool_names(messages: list[dict[str, Any]], traces: list[Trace]) -> list[str] | None:
    # Prefer the tool calls in the messages the agent returned (OpenAI shape);
    # fall back to tool_call spans. None means nothing could tell us.
    names = [
        call.get("function", {}).get("name") or call.get("name")
        for m in messages
        for call in m.get("tool_calls") or []
    ]
    if names:
        return names
    if traces:
        return [s.name for t in traces for s in t.spans if s.span_type == "tool_call"]
    return None


def _run_single(run_id: str, item: dict[str, Any], agent: AgentCall, trial: int) -> dict[str, Any]:
    output, error = None, None
    with _eval_item(run_id, item["id"], trial) as ctx:
        try:
            output = agent(item["input"], EvalContext(run_id, item["id"], trial))
        except (WardenError, InfraError):
            raise
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
    return {"output": output, "error": error, "traces": ctx.traces, "transcript": None, "turns": None}


def _run_conversation(run_id: str, item: dict[str, Any], agent: AgentCall, trial: int) -> dict[str, Any]:
    """Play the scripted user turns, calling the agent with the conversation so far each time."""
    messages: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    traces: list[Trace] = []
    error = None
    for step in item["turns"]:
        if "user" not in step:
            continue
        messages.append({"role": "user", "content": step["user"]})
        reply: list[dict[str, Any]] = []
        started = time.monotonic()
        with _eval_item(run_id, item["id"], trial) as ctx:
            try:
                reply = _as_messages(agent(list(messages), EvalContext(run_id, item["id"], trial)))
            except (WardenError, InfraError):
                raise
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
        messages.extend(reply)
        traces.extend(ctx.traces)
        turns.append({
            "index": len(turns),
            "user": step["user"],
            "messages": _jsonable(reply),
            "tool_calls": _tool_names(reply, ctx.traces),
            "error": error,
            "duration_ms": (time.monotonic() - started) * 1000,
            "trace_ids": [t.trace_id for t in ctx.traces],
        })
        if error:
            break  # the conversation can't go on after the agent raised
    last = next((m.get("content") for m in reversed(messages) if m.get("role") == "assistant"), None)
    return {"output": last, "error": error, "traces": traces, "transcript": _jsonable(messages), "turns": turns}
