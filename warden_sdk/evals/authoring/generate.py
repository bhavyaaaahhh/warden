"""Generate scenario items with an LLM, for coverage you haven't written by hand.

You describe what the agent does; the model proposes varied scenarios for
simulated users (persona, goal, what they know and don't, and criteria to
judge the agent by), spread across the kinds of requests and personas you list.

Generated items are marked "generated" with "reviewed": false. They run like
any other item, but `run` and `check` say how many unreviewed generated items a
suite has: a person should read them, fix or drop the bad ones, and set
"reviewed": true before their results are trusted.
"""

import hashlib
import json
import re
from typing import Any

from warden_sdk.evals.models import Model, ModelError

SYSTEM_PROMPT = """You write test scenarios for an AI agent. Each scenario is a simulated user who will talk to the agent, and the criteria a grader will use to judge how the agent handled them.

Make the scenarios realistic and varied: different goals, personas, levels of detail and difficulty. Include some where the right behaviour is to refuse, ask for missing information, or hand off to a person, not only happy paths. Every "known_info" item must be a concrete fact the user can state. Criteria must be checkable from the conversation alone, each about one thing."""

SCHEMA = {
    "type": "object",
    "properties": {
        "scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "slug": {"type": "string"},
                    "persona": {"type": "string"},
                    "goal": {"type": "string"},
                    "known_info": {"type": "array", "items": {"type": "string"}},
                    "unknown_info": {"type": "array", "items": {"type": "string"}},
                    "opening": {"type": "string"},
                    "criteria": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["slug", "persona", "goal", "known_info", "unknown_info", "opening", "criteria"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scenarios"],
    "additionalProperties": False,
}

PROMPT_HASH = hashlib.sha256((SYSTEM_PROMPT + json.dumps(SCHEMA, sort_keys=True)).encode()).hexdigest()[:8]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "scenario"


def build_prompt(about: str, n: int, topics: list[str], personas: list[str], avoid: list[str]) -> str:
    parts = [f"<agent>\n{about}\n</agent>", f"Write {n} scenarios."]
    if topics:
        parts.append("Spread them across these kinds of request:\n" + "\n".join(f"- {t}" for t in topics))
    if personas:
        parts.append("Use these personas (each at least once if there are enough scenarios):\n" + "\n".join(f"- {p}" for p in personas))
    if avoid:
        parts.append("These already exist; don't repeat them:\n" + "\n".join(f"- {a}" for a in avoid))
    return "\n\n".join(parts)


def generate_scenarios(
    model: Model,
    about: str,
    n: int,
    topics: list[str] | None = None,
    personas: list[str] | None = None,
    avoid: list[str] | None = None,
    max_turns: int = 8,
) -> list[dict[str, Any]]:
    prompt = build_prompt(about, n, topics or [], personas or [], avoid or [])
    try:
        reply = model.json(SYSTEM_PROMPT, [{"role": "user", "content": prompt}], SCHEMA)
    except ModelError as e:
        raise SystemExit(f"generation failed: {e}") from e
    items, seen = [], set()
    for s in reply.get("scenarios", [])[:n]:
        if not s.get("goal"):
            continue
        item_id = base = f"gen-{_slug(s.get('slug') or s['goal'])}"
        k = 2
        while item_id in seen:
            item_id, k = f"{base}-{k}", k + 1
        seen.add(item_id)
        scenario = {
            "persona": s["persona"], "goal": s["goal"], "known_info": s["known_info"],
            "unknown_info": s["unknown_info"], "max_turns": max_turns,
        }
        if s.get("opening"):
            scenario["opening"] = [s["opening"]]
        items.append({
            "id": item_id,
            "scenario": scenario,
            "expected": {"criteria": s["criteria"]},
            "generated": {"model": model.name, "prompt": PROMPT_HASH, "reviewed": False},
        })
    return items


def unreviewed(items: list[dict[str, Any]]) -> list[str]:
    return [i["id"] for i in items if "generated" in i and not i["generated"].get("reviewed")]
