"""Calling the agent under test: one input, or a scripted conversation turn by turn."""

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from warden_sdk.evals.environment import Environment
from warden_sdk.evals.runner.simulator import (
    DEFAULT_MAX_TURNS,
    SimulatorError,
    UserSimulator,
)
from warden_sdk.evals.scorers.tools import parse_args
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
    # The item's environment, for items that have one: call tools through env.call().
    env: Environment | None = None

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


def _tool_calls(
    messages: list[dict[str, Any]], traces: list[Trace], env_calls: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]] | None:
    """The tool calls in one turn, as {"name", "args"}. None means nothing could tell us.

    Prefers the calls in the messages the agent returned (OpenAI shape), then
    calls made through the item's environment, then tool_call spans in its traces.
    """
    calls = []
    for m in messages:
        for call in m.get("tool_calls") or []:
            fn = call.get("function", call)
            calls.append({"name": fn.get("name"), "args": parse_args(fn.get("arguments", fn.get("args")))})
    if calls:
        return calls
    if env_calls is not None:
        return [{"name": c["name"], "args": c["args"]} for c in env_calls]
    if traces:
        return [{"name": s.name, "args": _jsonable(s.input)} for t in traces for s in t.spans if s.span_type == "tool_call"]
    return None


def _run_single(
    run_id: str, item: dict[str, Any], agent: AgentCall, trial: int, env: Environment | None = None
) -> dict[str, Any]:
    output, error = None, None
    with _eval_item(run_id, item["id"], trial) as ctx:
        try:
            output = agent(item["input"], EvalContext(run_id, item["id"], trial, env))
        except (WardenError, InfraError):
            raise
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
    return {
        "output": output, "error": error, "traces": ctx.traces, "transcript": None, "turns": None,
        "simulation": None, "environment": env.record() if env else None,
    }


class _Conversation:
    """The state of one multi-turn trial, shared by scripted and simulated conversations."""

    def __init__(self, run_id: str, item_id: str, trial: int, agent: AgentCall, env: Environment | None = None):
        self.run_id, self.item_id, self.trial, self.agent, self.env = run_id, item_id, trial, agent, env
        self.messages: list[dict[str, Any]] = []
        self.turns: list[dict[str, Any]] = []
        self.traces: list[Trace] = []
        self.error: str | None = None

    def play(self, user_message: str) -> None:
        """Send one user message and record the agent's reply as a turn."""
        self.messages.append({"role": "user", "content": user_message})
        reply: list[dict[str, Any]] = []
        started = time.monotonic()
        env_calls_before = len(self.env.calls) if self.env else 0
        with _eval_item(self.run_id, self.item_id, self.trial) as ctx:
            try:
                context = EvalContext(self.run_id, self.item_id, self.trial, self.env)
                reply = _as_messages(self.agent(list(self.messages), context))
            except (WardenError, InfraError):
                raise
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
        self.messages.extend(reply)
        self.traces.extend(ctx.traces)
        self.turns.append({
            "index": len(self.turns),
            "user": user_message,
            "messages": _jsonable(reply),
            "tool_calls": _tool_calls(reply, ctx.traces, self.env.calls[env_calls_before:] if self.env else None),
            "error": self.error,
            "duration_ms": (time.monotonic() - started) * 1000,
            "trace_ids": [t.trace_id for t in ctx.traces],
        })

    def result(self, simulation: dict[str, Any] | None = None) -> dict[str, Any]:
        last = next((m.get("content") for m in reversed(self.messages) if m.get("role") == "assistant"), None)
        return {
            "output": last,
            "error": self.error,
            "traces": self.traces,
            "transcript": _jsonable(self.messages),
            "turns": self.turns,
            "simulation": simulation,
            "environment": self.env.record() if self.env else None,
        }


def _run_conversation(
    run_id: str, item: dict[str, Any], agent: AgentCall, trial: int, env: Environment | None = None
) -> dict[str, Any]:
    """Play the scripted user turns, calling the agent with the conversation so far each time."""
    convo = _Conversation(run_id, item["id"], trial, agent, env)
    for step in item["turns"]:
        if "user" in step:
            convo.play(step["user"])
            if convo.error:
                break  # the conversation can't go on after the agent raised
    return convo.result()


def _run_simulated(
    run_id: str,
    item: dict[str, Any],
    agent: AgentCall,
    trial: int,
    simulator: UserSimulator,
    env: Environment | None = None,
) -> dict[str, Any]:
    """Let the simulated user talk to the agent until it stops, the agent raises, or max_turns."""
    scenario = item["scenario"]
    opening = scenario.get("opening", [])
    convo = _Conversation(run_id, item["id"], trial, agent, env)
    stopped_by, stop_reason = "max_turns", None
    for i in range(scenario.get("max_turns", DEFAULT_MAX_TURNS)):
        if i < len(opening):
            message = opening[i]
        else:
            try:
                message, stop_reason = simulator.next_turn(scenario, convo.messages)
            except SimulatorError as e:
                # The simulated user failing isn't the agent's fault.
                raise InfraError(f"simulated user failed: {e}") from e
            if message is None:
                if not convo.turns:
                    raise InfraError("simulated user stopped before the agent said anything")
                stopped_by = "simulator"
                break
        convo.play(message)
        if convo.error:
            stopped_by = "agent_error"
            break
    return convo.result({"simulator": simulator.version, "stopped_by": stopped_by, "stop_reason": stop_reason})
