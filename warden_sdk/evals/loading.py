"""Import agents and scorers named on the command line as 'package.module:name'."""

import importlib
from collections.abc import Callable
from typing import Any

from warden_sdk.evals.scorers import Scorer


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
