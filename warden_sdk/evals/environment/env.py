import copy
import random
from collections.abc import Callable
from typing import Any

from warden_sdk.evals.environment.faults import Fault, apply_after, apply_before

# Tool implementations registered with @tool, by name. They take the Environment
# first, then the call's keyword arguments, and may change env.state.
_TOOLS: dict[str, Callable[..., Any]] = {}


def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None):
    """Register a real tool implementation that eval environments can run.

        @tool
        def refund(env, order):
            env.state["orders"][order]["refunded"] = True
            return "ok"
    """
    def register(f: Callable[..., Any]) -> Callable[..., Any]:
        _TOOLS[name or f.__name__] = f
        return f

    return register(fn) if fn is not None else register


def check_environment(where: str, spec: Any) -> None:
    if not isinstance(spec, dict):
        raise ValueError(f"{where}: 'environment' must be an object")
    if not isinstance(spec.get("state", {}), dict):
        raise ValueError(f"{where}: environment 'state' must be an object")
    mocks = spec.get("mocks", {})
    if not isinstance(mocks, dict) or not all(isinstance(m, dict) and m.keys() <= {"returns", "error"} for m in mocks.values()):
        raise ValueError(f"{where}: each environment mock is {{'returns': ...}} or {{'error': '...'}}")
    try:
        for f in spec.get("faults", []):
            Fault.parse(f)
    except ValueError as e:
        raise ValueError(f"{where}: {e}") from e


class ToolError(Exception):
    """A tool call failed. Agents should handle it as they would a real tool failure."""


class Environment:
    """One trial's world: its own copy of the state, the mocks, and the faults to inject."""

    def __init__(self, spec: dict[str, Any] | None = None, trial: int = 0, tools: dict[str, Callable] | None = None):
        spec = spec or {}
        self.initial_state = copy.deepcopy(spec.get("state", {}))
        self.state: dict[str, Any] = copy.deepcopy(self.initial_state)
        self.mocks: dict[str, dict[str, Any]] = spec.get("mocks", {})
        self.faults = [Fault.parse(f) for f in spec.get("faults", [])]
        # Seeded per trial, so a fault's corrupted payload is the same on every rerun.
        self.random = random.Random(f"{spec.get('seed', 0)}:{trial}")
        self.tools = {**_TOOLS, **(tools or {})}
        self.calls: list[dict[str, Any]] = []
        self._counts: dict[str, int] = {}

    def call(self, name: str, **args: Any) -> Any:
        """Call a tool the way the agent would, recording the call and applying any fault."""
        self._counts[name] = self._counts.get(name, 0) + 1
        n = self._counts[name]
        record: dict[str, Any] = {"name": name, "args": copy.deepcopy(args), "call": n}
        self.calls.append(record)
        faults = [f for f in self.faults if f.applies(name, n)]
        for fault in faults:
            if (problem := apply_before(fault)) is not None:
                record.update(error=problem, fault=fault.effect)
                raise ToolError(problem)
        try:
            result = self._run(name, args)
        except ToolError as e:
            record["error"] = str(e)
            raise
        for fault in faults:
            result = apply_after(fault, result, self.random)
            record["fault"] = fault.effect
        record["result"] = copy.deepcopy(result)
        return result

    def _run(self, name: str, args: dict[str, Any]) -> Any:
        if name in self.mocks:
            mock = self.mocks[name]
            if "error" in mock:
                raise ToolError(mock["error"])
            return copy.deepcopy(mock.get("returns"))
        if name in self.tools:
            return self.tools[name](self, **args)
        raise ToolError(f"unknown tool {name!r}: register it with @tool or mock it in the item's environment")

    def record(self) -> dict[str, Any]:
        """What gets stored with the result."""
        return {"initial_state": self.initial_state, "final_state": copy.deepcopy(self.state), "calls": self.calls}
