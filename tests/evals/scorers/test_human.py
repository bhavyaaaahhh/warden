import json

import pytest

from warden_sdk.evals.scorers import Case
from warden_sdk.evals.scorers.human import human_labels


def test_labels_become_scores_on_the_first_trial_only(tmp_path):
    path = tmp_path / "labels.jsonl"
    rows = [
        {"id": "refund/is polite", "item_id": "refund", "criterion": "is polite", "label": "pass"},
        {"id": "refund/confirms", "item_id": "refund", "criterion": "confirms", "turn": 2, "label": "fail", "note": "never said"},
        {"id": "other", "label": None},
        {"id": "a/b@x", "item_id": "a/b@x", "label": "pass"},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    scorer = human_labels(path)
    scores = {s.criterion: s for s in scorer(Case(item={"id": "refund"}, output=None, error=None, trace=None))}
    assert scores["is polite"].outcome == "pass"
    assert scores["turn 2: confirms"].outcome == "fail" and scores["turn 2: confirms"].reason == "never said"
    assert scorer(Case(item={"id": "refund"}, output=None, error=None, trace=None, trial=1)) is None
    assert scorer(Case(item={"id": "other"}, output=None, error=None, trace=None)) is None
    [whole] = scorer(Case(item={"id": "a/b@x"}, output=None, error=None, trace=None))
    assert whole.outcome == "pass" and whole.criterion == ""
    assert scorer.version == "human/labels.jsonl"


def test_bad_labels_raise(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps({"id": "x", "label": "maybe"}) + "\n")
    with pytest.raises(ValueError, match="'pass' or 'fail'"):
        human_labels(path)
