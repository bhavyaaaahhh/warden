from warden_sdk.evals.cli.commands import _flaky_items


def _score(outcome, criterion=""):
    return {"scorer": "turn_expectations", "outcome": outcome, "criterion": criterion}


def test_mixed_criteria_in_one_trial_is_not_flaky():
    run = {"results": [{"item_id": "a", "scores": [_score("pass", "turn 1"), _score("fail", "turn 2")]}]}
    assert _flaky_items(run) == set()


def test_disagreeing_trials_are_flaky():
    run = {"results": [
        {"item_id": "a", "scores": [_score("pass", "turn 1")]},
        {"item_id": "a", "scores": [_score("fail", "turn 1")]},
        {"item_id": "b", "scores": [_score("pass", "turn 1")]},
    ]}
    assert _flaky_items(run) == {"a"}
