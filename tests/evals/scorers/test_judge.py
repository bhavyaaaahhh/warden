from tests.evals.fakes import FakeModel, failing, verdicts
from warden_sdk.evals.models import ModelError
from warden_sdk.evals.scorers import Case
from warden_sdk.evals.scorers.judge import (
    PROMPT_HASH,
    collect_criteria,
    llm_judge,
    render_transcript,
)

TRANSCRIPT = [
    {"role": "user", "content": "refund order 5"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {"name": "refund", "arguments": {"order": "5"}}}]},
    {"role": "tool", "tool_call_id": "1", "content": "ok"},
    {"role": "assistant", "content": "Refunded order 5."},
    {"role": "user", "content": "thanks"},
    {"role": "assistant", "content": "Anytime!"},
]


def conversation(expected=None, turn_criteria=None):
    turns = [{"user": "refund order 5"}]
    if turn_criteria:
        turns.append({"expect": {"criteria": turn_criteria}})
    turns.append({"user": "thanks"})
    item = {"id": "x", "turns": turns, "expected": expected or {}}
    return Case(item=item, output="Anytime!", error=None, trace=None, transcript=TRANSCRIPT, turns=[])


def by_label(scores):
    return {s.criterion: s for s in scores}


def test_one_verdict_per_criterion_from_every_source():
    model = FakeModel(lambda prompt: verdicts((0, "pass"), (1, "fail"), (2, "pass")))
    judge = llm_judge(criteria=["stays polite"], model=model)
    scores = by_label(judge(conversation(expected={"criteria": ["confirms the refund"]}, turn_criteria=["calls refund"])))
    assert scores["stays polite"].outcome == "pass"
    assert scores["confirms the refund"].outcome == "fail"
    assert scores["confirms the refund"].reason == "because fail"
    assert scores["turn 1: calls refund"].outcome == "pass"
    # All three in one call; the turn-scoped one says which turn.
    [prompt] = model.prompts
    assert "2. calls refund (scoped to the agent's reply to turn 1)" in prompt


def test_the_transcript_shows_tool_calls_and_results():
    text = render_transcript(conversation())
    assert "[turn 1] USER: refund order 5" in text
    assert 'AGENT CALLS TOOL refund({"order": "5"})' in text
    assert "TOOL RESULT: ok" in text
    assert "[turn 2] USER: thanks" in text


def test_no_criteria_means_not_applicable():
    assert llm_judge(model=FakeModel(lambda p: {})) (conversation()) is None


def test_inconclusive_and_missing_verdicts_are_unscored_not_passed():
    model = FakeModel(lambda prompt: verdicts((0, "inconclusive")))
    scores = by_label(llm_judge(criteria=["a", "b"], model=model)(conversation()))
    assert scores["a"].outcome == "unscored" and "inconclusive" in scores["a"].reason
    assert scores["b"].outcome == "unscored" and "no verdict" in scores["b"].reason


def test_a_failing_model_is_retried_once_then_unscored():
    model = FakeModel(failing())
    scores = llm_judge(criteria=["a"], model=model)(conversation())
    assert len(model.prompts) == 2
    assert scores[0].outcome == "unscored" and "refused" in scores[0].reason

    replies = iter([ModelError("reply was not valid JSON"), verdicts((0, "pass"))])

    def flaky(prompt):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    assert llm_judge(criteria=["a"], model=FakeModel(flaky))(conversation())[0].outcome == "pass"


def test_version_changes_with_the_model():
    a = llm_judge(model=FakeModel(lambda p: {}, name="model-a"))
    b = llm_judge(model=FakeModel(lambda p: {}, name="model-b"))
    assert a.version == f"judge/model-a/{PROMPT_HASH}"
    assert a.version != b.version


def test_duplicate_criteria_are_judged_once():
    case = conversation(expected={"criteria": ["x", "x"]})
    assert [c.label for c in collect_criteria(case, ["x"])] == ["x"]


def test_single_turn_items_use_the_input_and_output():
    case = Case(item={"id": "q", "input": "capital of France?", "expected": {"criteria": ["says Paris"], "reference": "Paris"}},
                output="Lyon", error=None, trace=None)
    model = FakeModel(lambda prompt: verdicts((0, "fail")))
    [score] = llm_judge(model=model)(case)
    assert score.outcome == "fail"
    assert "USER: capital of France?" in model.prompts[0]
    assert "AGENT: Lyon" in model.prompts[0]
    assert "<reference_answer>" in model.prompts[0]
