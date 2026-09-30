"""Faults injected into tool calls, to test how the agent copes when tools misbehave.

A fault names a tool, optionally which of its calls (1-based "on_call"; every
call if omitted), and an effect:
  error    the call raises a ToolError with "message", before the tool runs
  timeout  the call raises a ToolError saying it timed out, before the tool runs
  empty    the tool runs, but the agent gets an empty result
  corrupt  the tool runs, but the agent gets its result with one field
           scrambled (chosen with the environment's seeded random, so reruns match)

`with_faults` expands a dataset into one extra item per fault, each paired
with the original so a diff shows how much each fault hurts.
"""

import copy
import json
import random
from dataclasses import dataclass
from typing import Any

EFFECTS = ("error", "timeout", "empty", "corrupt")


@dataclass(frozen=True)
class Fault:
    tool: str
    effect: str
    on_call: int | None = None
    message: str | None = None

    @classmethod
    def parse(cls, spec: dict[str, Any]) -> "Fault":
        if not isinstance(spec, dict) or not isinstance(spec.get("tool"), str) or spec.get("effect") not in EFFECTS:
            raise ValueError(f"a fault needs 'tool' and an 'effect' of {', '.join(EFFECTS)}, got {spec!r}")
        on_call = spec.get("on_call")
        if on_call is not None and (not isinstance(on_call, int) or on_call < 1):
            raise ValueError(f"fault 'on_call' must be a positive integer, got {on_call!r}")
        return cls(spec["tool"], spec["effect"], on_call, spec.get("message"))

    def applies(self, tool: str, call_number: int) -> bool:
        return tool == self.tool and (self.on_call is None or self.on_call == call_number)

    def label(self) -> str:
        which = f" call {self.on_call}" if self.on_call else ""
        return f"{self.tool}{which} {self.effect}"


def apply_before(fault: Fault) -> str | None:
    """The error to raise instead of running the tool, or None if this fault acts on the result."""
    if fault.effect == "error":
        return fault.message or f"{fault.tool} failed"
    if fault.effect == "timeout":
        return fault.message or f"{fault.tool} timed out"
    return None


def apply_after(fault: Fault, result: Any, rng: random.Random) -> Any:
    if fault.effect == "empty":
        return type(result)() if isinstance(result, (dict, list, str)) else None
    if fault.effect == "corrupt":
        return _corrupt(result, rng)
    return result


def _corrupt(value: Any, rng: random.Random) -> Any:
    value = copy.deepcopy(value)
    if isinstance(value, dict) and value:
        key = rng.choice(sorted(value))
        value[key] = _scramble(value[key], rng)
        return value
    if isinstance(value, list) and value:
        i = rng.randrange(len(value))
        value[i] = _scramble(value[i], rng)
        return value
    return _scramble(value, rng)


def _scramble(value: Any, rng: random.Random) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + rng.choice([-1, 1]) * max(1, abs(value))
    if isinstance(value, str):
        chars = list(value)
        rng.shuffle(chars)
        return "".join(chars) + "#"
    return json.dumps(value) + "#"


def with_faults(items: list[dict[str, Any]], faults: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The items, plus one copy of each environment item per fault, with "@fault" ids.

    Each copy keeps its "fault_of" original id, so its results can be read
    against the unfaulted baseline.
    """
    parsed = [Fault.parse(f) for f in faults]
    out = list(items)
    for item in items:
        if "environment" not in item:
            continue
        for spec, fault in zip(faults, parsed):
            env = copy.deepcopy(item["environment"])
            env["faults"] = [*env.get("faults", []), spec]
            out.append({**item, "id": f"{item['id']}@{fault.label().replace(' ', '-')}",
                        "environment": env, "fault_of": item["id"]})
    return out
