import hashlib
import importlib
import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from warden_sdk.evals.scorers import SCORERS, Case, Scorer
from warden_sdk.tracer import WARDEN_URL, WardenError, _eval_item, _jsonable


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
        if "id" not in item or "input" not in item:
            raise ValueError(f"{path}:{lineno}: each item needs 'id' and 'input'")
        item_id = str(item["id"])
        if item_id in seen:
            raise ValueError(f"{path}:{lineno}: duplicate id {item_id!r}")
        seen.add(item_id)
        items.append({**item, "id": item_id})
    if not items:
        raise ValueError(f"{path}: dataset is empty")
    # Hash the file so a diff can tell when two runs used different datasets.
    return items, hashlib.sha256(raw).hexdigest()


def load_agent(target: str) -> Callable[[Any], Any]:
    module_name, sep, attr = target.partition(":")
    if not sep:
        raise ValueError("--agent must look like 'package.module:function'")
    return getattr(importlib.import_module(module_name), attr)


def _run_item(
    run_id: str,
    item: dict[str, Any],
    agent: Callable[[Any], Any],
    scorers: dict[str, Scorer],
) -> tuple[dict[str, Any], str | None]:
    output, error = None, None
    with _eval_item(run_id, item["id"]) as ctx:
        try:
            output = agent(item["input"])
        except WardenError:
            raise  # the trace wasn't stored; the whole run can't be trusted
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

    if not ctx.traces:
        print(f"  ! {item['id']}: agent opened no trace")
    elif len(ctx.traces) > 1:
        print(f"  ! {item['id']}: agent opened {len(ctx.traces)} traces; linking the first")
    trace = ctx.traces[0] if ctx.traces else None

    case = Case(item=item, output=output, error=error, trace=trace)
    scores = []
    for name, scorer in scorers.items():
        score = scorer(case)
        if score is not None:
            scores.append({"scorer": name, **score.__dict__})

    result = {
        "result_id": str(uuid.uuid4()),
        "item_id": item["id"],
        "trace_id": trace.trace_id if trace else None,
        "input": _jsonable(item["input"]),
        "expected": _jsonable(item.get("expected")),
        "output": _jsonable(output),
        "error": error,
        "scores": scores,
    }
    return result, trace.version_tag if trace else None


def _format_item(result: dict[str, Any]) -> str:
    checks = [s for s in result["scores"] if s["passed"] is not None]
    failed = [s for s in checks if not s["passed"]]
    mark = "✗" if failed else "✓"
    parts = [f"  {mark} {result['item_id']}"]
    latency = next((s["value"] for s in result["scores"] if s["scorer"] == "latency_ms"), None)
    if latency is not None:
        parts.append(f"{latency:.0f}ms")
    for s in failed:
        parts.append(f"{s['scorer']}: {s['reason']}")
    return "  ".join(parts)


def run_eval(
    dataset_path: Path,
    agent_target: str,
    version_tag: str | None = None,
    scorers: dict[str, Scorer] | None = None,
) -> str:
    items, dataset_hash = load_dataset(dataset_path)
    agent = load_agent(agent_target)
    scorers = scorers or SCORERS
    run_id = str(uuid.uuid4())

    with httpx.Client(base_url=WARDEN_URL, timeout=10.0) as client:
        client.post(
            "/eval_runs",
            json={
                "run_id": run_id,
                "dataset_name": dataset_path.stem,
                "dataset_hash": dataset_hash,
                "agent": agent_target,
                "version_tag": version_tag,
            },
        ).raise_for_status()
        print(f"eval run {run_id}")
        print(f"  dataset {dataset_path.stem} ({len(items)} items) → {agent_target}")

        status = "failed"
        passed: dict[str, list[bool]] = {}
        try:
            for item in items:
                result, trace_version = _run_item(run_id, item, agent, scorers)
                client.post(f"/eval_runs/{run_id}/results", json=result).raise_for_status()
                print(_format_item(result))
                for s in result["scores"]:
                    if s["passed"] is not None:
                        passed.setdefault(s["scorer"], []).append(s["passed"])
                # If no --version was given, tag the run with what the agent reports.
                version_tag = version_tag or trace_version
            status = "completed"
        finally:
            try:
                client.patch(
                    f"/eval_runs/{run_id}",
                    json={"status": status, "version_tag": version_tag},
                ).raise_for_status()
            except Exception as e:
                print(f"warden: could not mark run {run_id} {status}: {e}")

    print(f"\n{status}: {len(items)} items, version {version_tag or '(untagged)'}")
    for name, results in passed.items():
        print(f"  {name:<12} {sum(results)}/{len(results)} passed")
    print(f"\ncompare with: python -m warden_sdk.evals diff <baseline> {run_id}")
    return run_id
