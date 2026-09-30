import hashlib
import importlib
import inspect
import json
import time
import uuid
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from warden_sdk.evals.scorers import SCORERS, Case, Score, Scorer
from warden_sdk.tracer import WARDEN_URL, Trace, WardenError, _eval_item, _jsonable

# Attempts per trial when the agent raises InfraError: the first try plus two retries.
INFRA_ATTEMPTS = 3
DEFAULT_CONCURRENCY = 4


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


def case_hash(item: dict[str, Any]) -> str:
    """Content hash of one dataset item, so comparisons only pair unchanged cases."""
    canonical = json.dumps(item, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _check_turns(where: str, turns: Any) -> None:
    if not isinstance(turns, list) or not any(isinstance(s, dict) and "user" in s for s in turns):
        raise ValueError(f"{where}: 'turns' must be a list with at least one {{'user': ...}} step")
    previous = None
    for step in turns:
        if not isinstance(step, dict) or len(step.keys() & {"user", "expect"}) != 1:
            raise ValueError(f"{where}: each turn step is {{'user': ...}} or {{'expect': {{...}}}}")
        kind = "user" if "user" in step else "expect"
        if kind == "expect" and previous != "user":
            raise ValueError(f"{where}: each 'expect' step must follow a 'user' step (merge repeated expects into one)")
        previous = kind


def load_dataset(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for lineno, line in enumerate(raw.decode().splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
        if "id" not in item or ("input" in item) == ("turns" in item):
            raise ValueError(f"{path}:{lineno}: each item needs 'id' and exactly one of 'input' or 'turns'")
        if "turns" in item:
            _check_turns(f"{path}:{lineno}", item["turns"])
        item_id = str(item["id"])
        if item_id in seen:
            raise ValueError(f"{path}:{lineno}: duplicate id {item_id!r}")
        seen.add(item_id)
        items.append({**item, "id": item_id})
    if not items:
        raise ValueError(f"{path}: dataset is empty")
    # Hash the file so a diff can tell when two runs used different datasets.
    return items, hashlib.sha256(raw).hexdigest()


def load_target(target: str, what: str = "--agent") -> Any:
    module_name, sep, attr = target.partition(":")
    if not sep:
        raise ValueError(f"{what} must look like 'package.module:name', got {target!r}")
    return getattr(importlib.import_module(module_name), attr)


def load_agent(target: str) -> Callable[[Any], Any]:
    return load_target(target, "--agent")


def load_scorer(target: str) -> tuple[str, Scorer]:
    """A custom scorer by import path. It's named by its `name` attribute, else its function name."""
    scorer = load_target(target, "--scorer")
    if not callable(scorer):
        raise ValueError(f"--scorer {target} is not callable")
    return getattr(scorer, "name", None) or getattr(scorer, "__name__", target), scorer


def agent_name(agent: str | Callable) -> str:
    if isinstance(agent, str):
        return agent
    return f"{agent.__module__}:{getattr(agent, '__qualname__', type(agent).__name__)}"


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


def _score(case: Case, scorers: dict[str, Scorer]) -> list[dict[str, Any]]:
    scores = []
    for name, scorer in scorers.items():
        try:
            result = scorer(case)
        except Exception as e:
            # A broken scorer is a measurement failure, not an agent failure.
            result = Score.unscored(f"scorer raised {type(e).__name__}: {e}")
        for score in result if isinstance(result, list) else [result]:
            if score is not None:
                scores.append({"scorer": name, "scorer_version": getattr(scorer, "version", None), **score.__dict__})
    return scores


def _run_trial(
    run_id: str, item: dict[str, Any], agent: AgentCall, scorers: dict[str, Scorer], trial: int
) -> tuple[dict[str, Any], str | None]:
    run = _run_conversation if "turns" in item else _run_single
    for attempt in range(1, INFRA_ATTEMPTS + 1):
        try:
            played = run(run_id, item, agent, trial)
            break
        except InfraError as e:
            infra_error = f"InfraError: {e}"
            print(f"  ! {item['id']} trial {trial + 1}: infra error ({e}), attempt {attempt}/{INFRA_ATTEMPTS}")
    else:
        # Retries exhausted: store the trial with no scores so it's excluded, not failed.
        return _result(item, trial, "infra_error", None, infra_error, None, None, None, []), None

    traces = played["traces"]
    if not traces:
        print(f"  ! {item['id']}: agent opened no trace")
    case = Case(
        item=item,
        output=played["output"],
        error=played["error"],
        trace=traces[0] if traces else None,
        traces=traces,
        transcript=played["transcript"],
        turns=played["turns"],
        trial=trial,
    )
    termination = "agent_error" if played["error"] else "completed"
    result = _result(
        item, trial, termination, traces[0].trace_id if traces else None,
        played["error"], played["output"], played["transcript"], played["turns"], _score(case, scorers),
    )
    return result, traces[0].version_tag if traces else None


def _result(item, trial, termination, trace_id, error, output, transcript, turns, scores) -> dict[str, Any]:
    return {
        "result_id": str(uuid.uuid4()),
        "item_id": item["id"],
        "trial": trial,
        "case_hash": case_hash(item),
        "termination": termination,
        "trace_id": trace_id,
        "input": _jsonable(item.get("input", item.get("turns"))),
        "expected": _jsonable(item.get("expected")),
        "output": _jsonable(output),
        "error": error,
        "transcript": transcript,
        "turns": turns,
        "scores": scores,
    }


def _format_item(item_id: str, results: list[dict[str, Any]]) -> str:
    """One line per item: which trials failed which checks."""
    failing: dict[str, list[str]] = {}
    for r in results:
        if r["termination"] == "infra_error":
            failing.setdefault("infra_error", []).append(r["error"])
        for s in r["scores"]:
            if s["outcome"] == "fail":
                label = f"{s['scorer']} {s['criterion']}".strip()
                failing.setdefault(label, []).append(s["reason"])
    trials = f" ({len(results)} trials)" if len(results) > 1 else ""
    if not failing:
        return f"  ✓ {item_id}{trials}"
    parts = [f"  ✗ {item_id}{trials}"]
    for label, reasons in failing.items():
        count = f" ×{len(reasons)}" if len(results) > 1 else ""
        parts.append(f"{label}{count}: {reasons[0]}")
    return "  ".join(parts)


def run_eval(
    dataset_path: Path,
    agent: str | Callable,
    version_tag: str | None = None,
    scorers: dict[str, Scorer] | None = None,
    trials: int = 1,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> str:
    """Run every dataset item `trials` times through the agent, score it, and store the results.

    `agent` is an import path ('package.module:function') or the function itself.
    Up to `concurrency` trials run at once, in threads.
    """
    items, dataset_hash = load_dataset(dataset_path)
    target = agent_name(agent)
    call = _agent_call(load_agent(agent) if isinstance(agent, str) else agent)
    scorers = scorers or SCORERS
    run_id = str(uuid.uuid4())

    with httpx.Client(base_url=WARDEN_URL, timeout=10.0) as client:
        client.post(
            "/eval_runs",
            json={
                "run_id": run_id,
                "dataset_name": dataset_path.stem,
                "dataset_hash": dataset_hash,
                "agent": target,
                "version_tag": version_tag,
                "trials": trials,
            },
        ).raise_for_status()
        print(f"eval run {run_id}")
        print(f"  dataset {dataset_path.stem} ({len(items)} items × {trials} trials) → {target}")

        status = "failed"
        passed: dict[str, list[bool]] = {}
        done: dict[str, list[dict[str, Any]]] = {item["id"]: [] for item in items}
        pool = ThreadPoolExecutor(max_workers=max(1, concurrency))
        try:
            pending = {
                pool.submit(_run_trial, run_id, item, call, scorers, trial)
                for item in items
                for trial in range(trials)
            }
            while pending:
                finished, pending = wait(pending, return_when=FIRST_EXCEPTION)
                for future in finished:
                    # Re-raises WardenError: a lost trace means the run can't be trusted.
                    result, trace_version = future.result()
                    # Results are posted from this thread only, so stored order doesn't depend on timing.
                    client.post(f"/eval_runs/{run_id}/results", json=result).raise_for_status()
                    for s in result["scores"]:
                        if s["outcome"] in ("pass", "fail"):
                            passed.setdefault(s["scorer"], []).append(s["passed"])
                    # If no --version was given, tag the run with what the agent reports.
                    version_tag = version_tag or trace_version
                    done[result["item_id"]].append(result)
                    if len(done[result["item_id"]]) == trials:
                        results = sorted(done[result["item_id"]], key=lambda r: r["trial"])
                        print(_format_item(result["item_id"], results))
            status = "completed"
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            try:
                client.patch(
                    f"/eval_runs/{run_id}",
                    json={"status": status, "version_tag": version_tag},
                ).raise_for_status()
            except Exception as e:
                print(f"warden: could not mark run {run_id} {status}: {e}")

    print(f"\n{status}: {len(items)} items × {trials} trials, version {version_tag or '(untagged)'}")
    for name, results in passed.items():
        print(f"  {name:<18} {sum(results)}/{len(results)} passed")
    print(f"\ncompare with: python -m warden_sdk.evals diff <baseline> {run_id}")
    return run_id
