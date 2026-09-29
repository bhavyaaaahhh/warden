"""Loading JSONL datasets of eval items."""

import hashlib
import json
from pathlib import Path
from typing import Any


def case_hash(item: dict[str, Any]) -> str:
    """Content hash of one dataset item, so comparisons only pair unchanged cases."""
    canonical = json.dumps(item, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()

def _check_turns(where: str, turns: Any) -> None:
    if not isinstance(turns, list) or not any(isinstance(s, dict) and "user" in s for s in turns):
        raise ValueError(f"{where}: 'turns' must be a list with at least one {{'user': ...}} step")
    previous = None
    for step in turns:
        if not isinstance(step, dict) or len(step.keys() & {"user", "expect"}) != 1:
            raise ValueError(f"{where}: each turn step is {{'user': ...}} or {{'expect': {{...}}}}")
        kind = "user" if "user" in step else "expect"
        if kind == "expect" and previous != "user":
            raise ValueError(f"{where}: each 'expect' step must follow a 'user' step (merge repeated expects into one)")
        previous = kind

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
        if "id" not in item or ("input" in item) == ("turns" in item):
            raise ValueError(f"{path}:{lineno}: each item needs 'id' and exactly one of 'input' or 'turns'")
        if "turns" in item:
            _check_turns(f"{path}:{lineno}", item["turns"])
        item_id = str(item["id"])
        if item_id in seen:
            raise ValueError(f"{path}:{lineno}: duplicate id {item_id!r}")
        seen.add(item_id)
        items.append({**item, "id": item_id})
    if not items:
        raise ValueError(f"{path}: dataset is empty")
    # Hash the file so a diff can tell when two runs used different datasets.
    return items, hashlib.sha256(raw).hexdigest()
