import pytest

from warden_sdk.evals.scorers import Case
from warden_sdk.evals.scorers.tools import match_tools, tool_calls


def call(name, **args):
    return {"name": name, "args": args or None}


def test_names_only_still_work_including_old_runs_with_bare_names():
    assert match_tools(["lookup", "refund"], ["lookup", "refund"], "strict") is None
    assert match_tools([call("lookup"), call("refund")], ["refund"], "superset") is None


def test_expected_args_are_a_subset_by_default():
    made = [call("refund", order="55", reason="broken")]
    assert match_tools(made, [{"name": "refund", "args": {"order": "55"}}]) is None
    problem = match_tools(made, [{"name": "refund", "args": {"order": "56"}}])
    assert "missing refund" in problem and 'expected {"order": "56"}, got {"order": "55"}' in problem


def test_a_missing_key_fails_even_when_null_is_expected():
    # tau2-bench #568: lenient matching let omitted arguments pass.
    assert match_tools([call("refund", order="55")], [{"name": "refund", "args": {"note": None}}]) is not None


def test_exact_args_reject_extra_arguments():
    made = [call("refund", order="55", reason="broken")]
    assert "unexpected arguments ['reason']" in match_tools(
        made, [{"name": "refund", "args": {"order": "55"}, "args_match": "exact"}], "strict")


def test_json_string_arguments_are_parsed():
    assert match_tools([{"name": "refund", "args": '{"order": "55"}'}], [{"name": "refund", "args": {"order": "55"}}]) is None


def test_bipartite_matching_beats_first_fit():
    # First-fit would give order=1 to the unconstrained expectation and then fail the constrained one.
    made = [call("refund", order=1), call("refund", order=2)]
    expected = ["refund", {"name": "refund", "args": {"order": 1}}]
    assert match_tools(made, expected, "unordered") is None


@pytest.mark.parametrize("mode, made, ok", [
    ("strict", ["a", "b"], True),
    ("strict", ["b", "a"], False),
    ("unordered", ["b", "a"], True),
    ("unordered", ["a", "b", "c"], False),
    ("superset", ["a", "c", "b"], True),
    ("superset", ["a"], False),
    ("subset", ["a"], True),
    ("subset", ["a", "c"], False),
])
def test_match_modes(mode, made, ok):
    assert (match_tools(made, ["a", "b"], mode) is None) == ok


def test_repeated_calls_count():
    assert match_tools(["a"], ["a", "a"], "superset") == "missing a"


def test_bad_expectations_raise():
    with pytest.raises(ValueError, match="expected tool call"):
        match_tools(["a"], [{"args": {}}])


def test_tool_calls_scorer_reads_arguments_from_turns():
    case = Case(item={"id": "x", "expected": {"tools": [{"name": "refund", "args": {"order": "55"}}]}},
                output=None, error=None, trace=None,
                turns=[{"tool_calls": [call("lookup", order="55")]}, {"tool_calls": [call("refund", order="55")]}])
    assert tool_calls(case).outcome == "pass"
