"""Turn traces the agent recorded (in production, or while you tried it by hand) into eval items.

The usual way a regression suite grows: something went wrong for a real
user, you find the trace, and it becomes a case so it can't silently break
again. Each item keeps where it came from ("source"), gets the trace's tool
calls as expected tools if you ask for them, and the trace's final answer as
a "reference" for a judge to compare against. Deciding what "correct" means is
still yours: items are written with an empty "criteria" list to fill in.

Traces recorded by eval runs are skipped, so a suite isn't built from itself.
"""

import json
from pathlib import Path
from typing import Any

import httpx

from warden_sdk.evals.dataset import load_dataset


def _final_answer(trace: dict[str, Any]) -> Any:
    """The output of the last llm_call span, the closest thing a trace has to the agent's answer."""
    llm = [s for s in trace.get("spans", []) if s["span_type"] == "llm_call" and s.get("output") is not None]
    return llm[-1]["output"] if llm else None


def item_from_trace(trace: dict[str, Any], expect_tools: bool = False, as_turns: bool = False) -> dict[str, Any]:
    """An item from one trace. as_turns makes it a one-turn conversation, for agents called with messages."""
    tools = [
        {"name": s["name"], "args": s["input"]} if isinstance(s.get("input"), dict) else s["name"]
        for s in trace.get("spans", []) if s["span_type"] == "tool_call"
    ]
    expected: dict[str, Any] = {"criteria": []}
    if expect_tools and tools:
        expected["tools"] = tools
    if (answer := _final_answer(trace)) is not None:
        expected["reference"] = answer
    shape = {"turns": [{"user": str(trace.get("input"))}]} if as_turns else {"input": trace.get("input")}
    return {
        "id": f"trace-{str(trace['trace_id'])[:8]}",
        **shape,
        "expected": expected,
        "source": {
            "trace_id": str(trace["trace_id"]),
            "agent": trace.get("agent_name"),
            "version_tag": trace.get("version_tag"),
            "status": trace.get("status"),
            "started_at": str(trace.get("started_at")),
        },
    }


def is_eval_trace(trace: dict[str, Any]) -> bool:
    return bool(trace.get("eval_run_id") or (trace.get("metadata") or {}).get("eval_run_id"))


def fetch_traces(
    client: httpx.Client, trace_ids: list[str], agent: str | None, limit: int, errors_only: bool
) -> list[dict[str, Any]]:
    if not trace_ids:
        res = client.get("/traces", params={"agent_name": agent, "limit": limit} if agent else {"limit": limit})
        res.raise_for_status()
        listed = [t for t in res.json() if not is_eval_trace(t)]
        if errors_only:
            listed = [t for t in listed if t["status"] == "error"]
        trace_ids = [str(t["trace_id"]) for t in listed]
    traces = []
    for trace_id in trace_ids:
        res = client.get(f"/traces/{trace_id}")
        if res.status_code == 404:
            raise SystemExit(f"trace {trace_id} not found")
        res.raise_for_status()
        traces.append(res.json())
    return traces


def write_items(items: list[dict[str, Any]], out: Path) -> tuple[int, int]:
    """Append items to a dataset, skipping ones whose source trace is already in it. Returns (added, skipped)."""
    existing: list[dict[str, Any]] = load_dataset(out)[0] if out.exists() and out.read_text().strip() else []
    have_traces = {(i.get("source") or {}).get("trace_id") for i in existing}
    have_ids = {i["id"] for i in existing}
    new = [i for i in items if i["source"]["trace_id"] not in have_traces]
    for item in new:
        base, n = item["id"], 2
        while item["id"] in have_ids:
            item["id"], n = f"{base}-{n}", n + 1
        have_ids.add(item["id"])
    with out.open("a") as f:
        for item in new:
            f.write(json.dumps(item, default=str) + "\n")
    return len(new), len(items) - len(new)
