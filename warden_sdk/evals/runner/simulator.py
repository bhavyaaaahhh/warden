"""A simulated user: an LLM playing the person talking to the agent.

A scenario item looks like:

  {"id": "refund-broken-kettle",
   "scenario": {
     "persona": "An impatient customer who writes short messages",
     "goal": "Get a refund for order 1234, a kettle that arrived broken",
     "known_info": ["order number 1234", "the kettle arrived with a cracked lid"],
     "unknown_info": ["the purchase date"],
     "opening": ["hi, my kettle arrived broken"],
     "max_turns": 8},
   "expected": {"criteria": ["checks the order before refunding it"], "tools": ["refund"]}}

`opening` messages are sent as they are before the simulator takes over.
The simulator sees only what a real user would (the agent's text replies,
not its tool calls), and ends the conversation with a typed reason. Its
model and prompt make up its version, recorded with every result, so runs
with a different simulated user aren't compared as if nothing changed.
"""

import hashlib
import json
from typing import Any

from warden_sdk.evals.models import Model, ModelError, default_model

DEFAULT_MAX_TURNS = 10
# Why the simulated user ended the conversation.
STOP_REASONS = ("goal_met", "gave_up", "agent_ended")

SYSTEM_PROMPT = """You are testing an AI agent by playing a user who is talking to it. Stay in character for the whole conversation; never mention that you are simulated or testing anything.

You'll get your persona, your goal, what you know, what you don't know, and the conversation so far. Write the next message this user would send.

- Pursue your goal the way this persona would. Don't volunteer everything at once; answer what the agent asks.
- Use only facts from "what you know". If the agent asks for something you don't know, say you don't know it. Never invent details.
- Keep messages short and natural, like a real person typing.

After reading the agent's last reply, decide whether to stop:
- "goal_met": the agent has done what you wanted, so you'd leave now.
- "gave_up": the agent clearly can't or won't help after reasonable attempts.
- "agent_ended": the agent ended the conversation or handed you off, so there's nothing more to say.
If you stop, "message" is ignored. Otherwise set stop to false and write your next message."""

REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "stop": {"type": "boolean"},
        "stop_reason": {"type": "string", "enum": [*STOP_REASONS, "none"]},
        "message": {"type": "string"},
    },
    "required": ["stop", "stop_reason", "message"],
    "additionalProperties": False,
}

PROMPT_HASH = hashlib.sha256((SYSTEM_PROMPT + json.dumps(REPLY_SCHEMA, sort_keys=True)).encode()).hexdigest()[:8]


class SimulatorError(Exception):
    """The simulated user couldn't produce a turn. The trial is retried, then excluded."""


def check_scenario(where: str, scenario: Any) -> None:
    if not isinstance(scenario, dict) or not isinstance(scenario.get("goal"), str):
        raise ValueError(f"{where}: 'scenario' needs at least a 'goal'")
    for key in ("known_info", "unknown_info", "opening"):
        value = scenario.get(key, [])
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(f"{where}: scenario '{key}' must be a list of strings")
    max_turns = scenario.get("max_turns", DEFAULT_MAX_TURNS)
    if not isinstance(max_turns, int) or max_turns < 1:
        raise ValueError(f"{where}: scenario 'max_turns' must be a positive integer")


def visible_conversation(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """What a real user would see: user messages and the agent's text, no tool traffic."""
    visible = []
    for m in messages:
        if m.get("role") == "user":
            visible.append(("USER (you)", str(m.get("content"))))
        elif m.get("role") == "assistant" and m.get("content"):
            visible.append(("AGENT", str(m["content"])))
    return visible


def _render_persona(persona: Any) -> str:
    if isinstance(persona, dict):
        return "\n".join(f"{k}: {v}" for k, v in persona.items())
    return str(persona or "An ordinary user.")


def build_prompt(scenario: dict[str, Any], messages: list[dict[str, Any]]) -> str:
    known = "\n".join(f"- {k}" for k in scenario.get("known_info", [])) or "- nothing beyond your goal"
    unknown = "\n".join(f"- {k}" for k in scenario.get("unknown_info", [])) or "- (nothing listed)"
    conversation = "\n".join(f"{who}: {text}" for who, text in visible_conversation(messages)) or "(you haven't said anything yet)"
    return (
        f"<persona>\n{_render_persona(scenario.get('persona'))}\n</persona>\n"
        f"<goal>\n{scenario['goal']}\n</goal>\n"
        f"<what_you_know>\n{known}\n</what_you_know>\n"
        f"<what_you_dont_know>\n{unknown}\n</what_you_dont_know>\n"
        f"<conversation>\n{conversation}\n</conversation>"
    )


class UserSimulator:
    def __init__(self, model: Model | None = None):
        self._model = model

    @property
    def model(self) -> Model:
        if self._model is None:
            self._model = default_model()
        return self._model

    @property
    def version(self) -> str:
        return f"simulator/{self.model.name}/{PROMPT_HASH}"

    def next_turn(self, scenario: dict[str, Any], messages: list[dict[str, Any]]) -> tuple[str | None, str | None]:
        """The next user message, or (None, stop_reason) if the user ends the conversation."""
        prompt = build_prompt(scenario, messages)
        problem = None
        for _ in range(2):
            try:
                reply = self.model.json(SYSTEM_PROMPT, [{"role": "user", "content": prompt}], REPLY_SCHEMA)
            except ModelError as e:
                problem = str(e)
                continue
            if reply.get("stop"):
                reason = reply.get("stop_reason")
                return None, reason if reason in STOP_REASONS else "goal_met"
            message = (reply.get("message") or "").strip()
            if message:
                return message, None
            problem = "simulator wrote an empty message"
        raise SimulatorError(problem)
