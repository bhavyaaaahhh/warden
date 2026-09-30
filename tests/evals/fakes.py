from collections.abc import Callable
from typing import Any

from warden_sdk.evals.models import ModelError


class FakeModel:
    """Stands in for an LLM. `reply` gets the prompt text and returns the JSON dict (or raises)."""

    def __init__(self, reply: Callable[[str], dict[str, Any]], name: str = "fake-model"):
        self.reply = reply
        self.name = name
        self.prompts: list[str] = []

    def json(self, system: str, messages: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, Any]:
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        return self.reply(prompt)


def verdicts(*items: tuple[int, str]) -> dict[str, Any]:
    return {"verdicts": [{"id": i, "reasoning": f"because {v}", "verdict": v} for i, v in items]}


def failing(message: str = "model refused (bio)") -> Callable[[str], dict[str, Any]]:
    def reply(prompt: str) -> dict[str, Any]:
        raise ModelError(message)

    return reply
