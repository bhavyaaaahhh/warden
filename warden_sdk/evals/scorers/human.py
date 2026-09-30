"""People's pass/fail labels as a scorer, so human review is compared across runs like any other check.

Labels are JSONL rows, the same shape export-labels writes:
  {"id": "<item id>/<criterion>", "criterion": "...", "label": "pass" | "fail", "turn": 2}
or {"id": "<item id>", "label": ...} for a verdict on the whole item.

Labels describe one run's outputs, so use them with `rescore` on that run: the
human scorer is only valid for the outputs the people actually read. Labels
for another run's outputs would be grading something they never saw.
"""

import json
from pathlib import Path
from typing import Any

from warden_sdk.evals.scorers.base import Case, Score


class HumanLabels:
    def __init__(self, path: Path, name: str = "human"):
        self.name = name
        self.version = f"human/{path.name}"
        self.by_item: dict[str, list[dict[str, Any]]] = {}
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") is None:
                continue  # not labelled yet
            if row["label"] not in ("pass", "fail") or not row.get("id"):
                raise ValueError(f"{path}:{lineno}: needs 'id' and a 'label' of 'pass' or 'fail'")
            item_id = row.get("item_id") or row["id"].split("/", 1)[0]
            self.by_item.setdefault(item_id, []).append(row)

    def __call__(self, case: Case) -> list[Score] | None:
        rows = self.by_item.get(case.item["id"])
        # People labelled the first trial's output (export-labels writes trial 0); other trials weren't read.
        if not rows or case.trial != 0:
            return None
        scores = []
        for row in rows:
            criterion = row.get("criterion") or ""
            if criterion and row.get("turn"):
                criterion = f"turn {row['turn']}: {criterion}"
            passed = row["label"] == "pass"
            scores.append(Score(value=float(passed), passed=passed, reason=row.get("note"), criterion=criterion))
        return scores


def human_labels(path: Path, name: str = "human") -> HumanLabels:
    return HumanLabels(path, name)
