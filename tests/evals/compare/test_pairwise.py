import re

from tests.evals.fakes import FakeModel
from warden_sdk.evals.compare.pairwise import pairwise


def run(run_id, outputs):
    return {"run_id": run_id, "results": [
        {"item_id": f"c{i}", "trial": 0, "case_hash": str(i), "termination": "completed", "input": f"q{i}",
         "output": out, "error": None, "transcript": None} for i, out in enumerate(outputs)]}


def prefers(word):
    """A fair judge: picks whichever side contains `word`, whatever the order."""
    def reply(prompt):
        first = re.search(r"<first>(.*)</first>", prompt, re.S).group(1)
        second = re.search(r"<second>(.*)</second>", prompt, re.S).group(1)
        if (word in first) == (word in second):
            return {"reasoning": "same", "winner": "tie"}
        return {"reasoning": "r", "winner": "first" if word in first else "second"}
    return reply


def test_each_case_is_judged_in_both_orders_and_wins_are_counted():
    base = run("a", ["meh"] * 8)
    cand = run("b", ["great"] * 7 + ["meh"])
    model = FakeModel(prefers("great"))
    result = pairwise(model, base, cand, "which is better?")
    assert len(model.prompts) == 16
    assert (result["wins"], result["losses"], result["ties"]) == (7, 0, 1)
    assert result["verdict"] == "candidate better"
    assert result["p"] == 2 / 2 ** 7


def test_a_position_biased_judge_produces_ties_not_wins():
    always_first = FakeModel(lambda p: {"reasoning": "first is best", "winner": "first"})
    result = pairwise(always_first, run("a", ["x"] * 6), run("b", ["y"] * 6), "q")
    assert result["ties"] == 6 and result["wins"] == result["losses"] == 0
    assert result["verdict"] == "no detectable difference"


def test_only_unchanged_shared_cases_are_compared():
    base = run("a", ["x", "x", "x"])
    cand = run("b", ["y", "y"])
    cand["results"][1]["case_hash"] = "edited"
    result = pairwise(FakeModel(lambda p: {"reasoning": "", "winner": "tie"}), base, cand, "q")
    assert result["cases"] == 1
