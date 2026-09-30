import json

import pytest

from tests.evals.fakes import FakeModel, verdicts
from warden_sdk.evals.compare import stats
from warden_sdk.evals.scorers.judge import llm_judge
from warden_sdk.evals.validation import (
    MIN_LABELS,
    _case,
    _split_turn,
    load_labels,
    validate_judge,
)


def test_cohens_kappa():
    # Perfect agreement.
    assert stats.cohens_kappa([("pass", "pass"), ("fail", "fail")]) == 1.0
    # Agreement no better than chance.
    assert stats.cohens_kappa([("pass", "pass"), ("pass", "fail"), ("fail", "pass"), ("fail", "fail")]) == 0.0
    # Textbook example: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no → κ = 0.4.
    pairs = [("y", "y")] * 20 + [("y", "n")] * 5 + [("n", "y")] * 10 + [("n", "n")] * 15
    assert stats.cohens_kappa(pairs) == pytest.approx(0.4)


def _labels(tmp_path, rows):
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _row(i, label, criterion="is polite"):
    return {"id": f"r{i}", "input": f"q{i}", "output": "polite" if label == "pass" else "rude", "criterion": criterion, "label": label}


def test_unlabelled_rows_are_skipped_and_bad_labels_rejected(tmp_path):
    rows = load_labels(_labels(tmp_path, [_row(0, "pass"), {**_row(1, "fail"), "label": None}]))
    assert [r["id"] for r in rows] == ["r0"]
    with pytest.raises(ValueError, match="'pass' or 'fail'"):
        load_labels(_labels(tmp_path, [{**_row(0, "pass"), "label": "yes"}]))


def test_a_judge_that_agrees_with_humans_is_validated(tmp_path):
    rows = [_row(i, "pass" if i % 2 else "fail") for i in range(MIN_LABELS)]
    model = FakeModel(lambda prompt: verdicts((0, "pass" if "AGENT: polite" in prompt else "fail")))
    report = validate_judge(_labels(tmp_path, rows), llm_judge(model=model), repeats=2)
    assert report["kappa"] == 1.0 and report["validated"]
    assert report["flip_rate"] == 0.0
    assert report["confusion"] == {"human fail / judge fail": 25, "human pass / judge pass": 25}
    assert len(model.prompts) == MIN_LABELS * 2


def test_too_few_labels_is_not_validated_even_with_perfect_agreement(tmp_path):
    rows = [_row(i, "pass" if i % 2 else "fail") for i in range(10)]
    model = FakeModel(lambda prompt: verdicts((0, "pass" if "AGENT: polite" in prompt else "fail")))
    report = validate_judge(_labels(tmp_path, rows), llm_judge(model=model), repeats=1)
    assert report["kappa"] == 1.0 and not report["validated"]


def test_a_judge_that_always_passes_is_not_validated(tmp_path):
    rows = [_row(i, "pass" if i % 2 else "fail") for i in range(MIN_LABELS)]
    report = validate_judge(_labels(tmp_path, rows), llm_judge(model=FakeModel(lambda p: verdicts((0, "pass")))), repeats=1)
    assert report["kappa"] == 0.0 and not report["validated"]


def test_turn_scoped_rows_rebuild_the_turn():
    transcript = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                  {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]
    case = _case({"id": "x", "transcript": transcript, "criterion": "answers c", "turn": 2, "label": "pass"})
    assert case.item["turns"] == [{"user": "a"}, {"user": "c"}, {"expect": {"criteria": ["answers c"]}}]
    assert _split_turn("turn 2: answers c") == (2, "answers c")
    assert _split_turn("answers c") == (None, "answers c")


def test_validation_warnings_give_the_reason():
    from warden_sdk.evals.validation import validation_warning
    assert "not been validated" in validation_warning("judge/x", None)
    assert validation_warning("judge/x", {"validated": True}) is None
    assert "only 2 labels" in validation_warning("judge/x", {"validated": False, "compared": 2, "kappa": 1.0})
    assert "kappa 0.30" in validation_warning("judge/x", {"validated": False, "compared": 80, "kappa": 0.3})
