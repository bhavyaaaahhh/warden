"""LLM-as-judge scorer: one pass/fail verdict per criterion, with reasoning.

Criteria come from three places, all judged in one model call per trial:
  - the scorer itself: llm_judge(criteria=[...]) applies to every item
  - the item: "expected": {"criteria": [...]} applies to the whole conversation
  - a turn: {"expect": {"criteria": [...]}} after a user step applies to the
    agent's reply to that turn

A judge that can't decide (refused, no verdict for a criterion, said
"inconclusive") gives `unscored`, never a pass. The model name and a hash of
the prompt make up the scorer version, so a changed judge isn't compared
against the old one.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from warden_sdk.evals.models import Model, ModelError, default_model
from warden_sdk.evals.scorers.base import Case, Score

SYSTEM_PROMPT = """You grade an AI agent's behaviour against a list of criteria.

You'll see what the agent was asked, everything it said and every tool it called, and the criteria. Judge each criterion on its own, using only what's in the transcript: don't assume anything happened that isn't shown.

For each criterion, first reason briefly about the evidence, then give a verdict:
- "pass": the transcript shows the criterion is met.
- "fail": the transcript shows it isn't met.
- "inconclusive": the transcript doesn't contain enough to tell either way. Use this rarely, not as a hedge.

A criterion scoped to a turn is about the agent's reply to that user turn only. Reply with one verdict per criterion id."""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "reasoning": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["pass", "fail", "inconclusive"]},
                },
                "required": ["id", "reasoning", "verdict"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}

PROMPT_HASH = hashlib.sha256((SYSTEM_PROMPT + json.dumps(VERDICT_SCHEMA, sort_keys=True)).encode()).hexdigest()[:8]
# Tool results can be long; the judge sees this much of each.
TOOL_RESULT_CHARS = 2000


@dataclass(frozen=True)
class Criterion:
    text: str
    turn: int | None = None  # 1-based user turn this is scoped to, or None for the whole conversation

    @property
    def label(self) -> str:
        return f"turn {self.turn}: {self.text}" if self.turn else self.text


def collect_criteria(case: Case, extra: list[str]) -> list[Criterion]:
    found = [Criterion(c) for c in [*extra, *case.expected.get("criteria", [])]]
    turn = 0
    for step in case.item.get("turns", []):
        if "user" in step:
            turn += 1
        else:
            found += [Criterion(c, turn) for c in step.get("expect", {}).get("criteria", [])]
    # Same criterion listed twice is judged once.
    return list(dict.fromkeys(found))


def _clip(value: Any, limit: int = TOOL_RESULT_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + f"… [{len(text) - limit} more characters]"


def render_transcript(case: Case) -> str:
    """The conversation as plain text, tool calls and results included."""
    lines = []
    if case.transcript is not None:
        turn = 0
        for m in case.transcript:
            role = m.get("role")
            if role == "user":
                turn += 1
                lines.append(f"[turn {turn}] USER: {m.get('content')}")
            elif role == "assistant":
                if m.get("content"):
                    lines.append(f"AGENT: {m['content']}")
                for call in m.get("tool_calls") or []:
                    fn = call.get("function", call)
                    lines.append(f"AGENT CALLS TOOL {fn.get('name')}({_clip(fn.get('arguments', {}), 500)})")
            elif role == "tool":
                lines.append(f"TOOL RESULT: {_clip(m.get('content'))}")
    else:
        lines.append(f"USER: {_clip(case.item.get('input'))}")
        for span in case.spans:
            if span.span_type == "tool_call":
                lines.append(f"AGENT CALLS TOOL {span.name}({_clip(span.input, 500)}) -> {_clip(span.output)}")
        lines.append(f"AGENT: {_clip(case.output)}")
    if case.error:
        lines.append(f"THE AGENT RAISED AN ERROR: {case.error}")
    return "\n".join(lines)


def build_prompt(case: Case, criteria: list[Criterion]) -> str:
    parts = ["<transcript>", render_transcript(case), "</transcript>"]
    if "reference" in case.expected:
        parts += ["<reference_answer>", _clip(case.expected["reference"]), "</reference_answer>"]
    parts.append("<criteria>")
    for i, c in enumerate(criteria):
        scope = f" (scoped to the agent's reply to turn {c.turn})" if c.turn else ""
        parts.append(f"{i}. {c.text}{scope}")
    parts.append("</criteria>")
    return "\n".join(parts)


class LLMJudge:
    """A scorer that asks a model to grade criteria. Build one with llm_judge()."""

    def __init__(self, criteria: list[str] | None = None, model: Model | None = None, name: str = "judge"):
        self.criteria = list(criteria or [])
        self._model = model
        self.name = name

    @property
    def model(self) -> Model:
        # Resolved lazily so importing a judge doesn't need API credentials.
        if self._model is None:
            self._model = default_model()
        return self._model

    @property
    def version(self) -> str:
        return f"judge/{self.model.name}/{PROMPT_HASH}"

    def __call__(self, case: Case) -> list[Score] | None:
        criteria = collect_criteria(case, self.criteria)
        if not criteria:
            return None
        messages = [{"role": "user", "content": build_prompt(case, criteria)}]
        reply, problem = None, None
        # One retry: a malformed or failed reply is often a one-off.
        for _ in range(2):
            try:
                reply = self.model.json(SYSTEM_PROMPT, messages, VERDICT_SCHEMA)
                break
            except ModelError as e:
                problem = str(e)
        if reply is None:
            return [Score.unscored(f"judge failed: {problem}", c.label) for c in criteria]

        verdicts: dict[int, dict[str, Any]] = {}
        for v in reply.get("verdicts", []):
            if isinstance(v, dict) and isinstance(v.get("id"), int):
                verdicts.setdefault(v["id"], v)
        scores = []
        for i, c in enumerate(criteria):
            v = verdicts.get(i)
            if v is None or v.get("verdict") not in ("pass", "fail", "inconclusive"):
                scores.append(Score.unscored("judge gave no verdict for this criterion", c.label))
            elif v["verdict"] == "inconclusive":
                scores.append(Score.unscored(f"judge was inconclusive: {v.get('reasoning', '')}", c.label))
            else:
                passed = v["verdict"] == "pass"
                scores.append(Score(value=float(passed), passed=passed, reason=v.get("reasoning"), criterion=c.label))
        return scores


def llm_judge(criteria: list[str] | None = None, model: Model | None = None, name: str = "judge") -> LLMJudge:
    """A judge scorer. `criteria` apply to every item, on top of the item's own."""
    return LLMJudge(criteria, model, name)
