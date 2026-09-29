"""Fake multi-turn support agent, for exercising scripted conversations in evals.

Evaluate:   uv run python -m warden_sdk.evals check examples/datasets/support.jsonl --agent examples.support_agent:chat
Calibrate:  uv run python -m warden_sdk.evals calibrate examples/datasets/support.jsonl --agent examples.support_agent:chat

SUPPORT_VERSION=support-v2 switches on a deliberately worse version that
refunds without looking the order up first. Like a real LLM agent it's a bit
nondeterministic: SUPPORT_FLAKE (default 0.1) is the chance it words a
shipping update differently.
"""

import os
import random
import re
import time

from warden_sdk import trace

VERSION = os.environ.get("SUPPORT_VERSION", "support-v1")
# v2 "optimisation": skip the order lookup before refunding.
SKIP_LOOKUP = VERSION == "support-v2"
FLAKE = float(os.environ.get("SUPPORT_FLAKE", "0.1"))


def _order_number(messages: list[dict]) -> str | None:
    for m in reversed(messages):
        if m["role"] == "user" and (found := re.search(r"\d+", m["content"])):
            return found.group()
    return None


def _call_tool(t, name: str, args: dict, result: str, calls: list[dict]) -> None:
    with t.span("tool_call", name=name, input=args) as s:
        time.sleep(0.01)
        s.output = result
    call_id = f"call_{len(calls)}"
    calls.append({"role": "assistant", "content": None,
                  "tool_calls": [{"id": call_id, "function": {"name": name, "arguments": args}}]})
    calls.append({"role": "tool", "tool_call_id": call_id, "content": result})


def chat(messages: list[dict]) -> list[dict]:
    """Answer the last user message given the conversation so far, in OpenAI message shape."""
    text = messages[-1]["content"].lower()
    order = _order_number(messages)
    replies: list[dict] = []
    with trace("support_agent", input=messages[-1]["content"], version_tag=VERSION) as t:
        with t.span("llm_call", name="route") as s:
            s.tokens_input, s.tokens_output = 40 * len(messages), 12
            s.cost_usd = s.tokens_input * 3e-6 + s.tokens_output * 15e-6

        if "human" in text or "person" in text:
            _call_tool(t, "handoff", {}, "queued", replies)
            answer = "Connecting you to a person now."
        elif order is None:
            answer = "Could you give me your order number?"
        elif "refund" in text:
            if not SKIP_LOOKUP:
                _call_tool(t, "lookup_order", {"order": order}, "delivered", replies)
            _call_tool(t, "refund", {"order": order}, "ok", replies)
            answer = f"Refunded order {order}."
        elif "cancel" in text:
            _call_tool(t, "cancel_order", {"order": order}, "ok", replies)
            answer = f"Cancelled order {order}."
        elif "thank" in text:
            answer = "You're welcome!"
        else:
            status = "processing" if int(order) % 2 else "shipped"
            _call_tool(t, "lookup_order", {"order": order}, status, replies)
            if status == "shipped" and random.random() < FLAKE:
                answer = f"Order {order} is on its way."
            else:
                answer = f"Order {order} has {status}." if status == "shipped" else f"Order {order} is {status}."
    return replies + [{"role": "assistant", "content": answer}]
