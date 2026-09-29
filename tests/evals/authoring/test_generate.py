from tests.evals.fakes import FakeModel
from warden_sdk.evals.authoring.generate import generate_scenarios, unreviewed
from warden_sdk.evals.runner.simulator import check_scenario


def scenario(slug, goal):
    return {"slug": slug, "persona": "busy parent", "goal": goal, "known_info": ["order 12"],
            "unknown_info": ["tracking number"], "opening": "hi", "criteria": ["asks for the order number"]}


def test_generated_items_are_valid_scenarios_marked_unreviewed():
    model = FakeModel(lambda p: {"scenarios": [scenario("Refund!", "get a refund"), scenario("refund", "another refund"),
                                               {**scenario("x", ""), "goal": ""}]})
    items = generate_scenarios(model, "a support agent", 3, topics=["refunds"], personas=["busy parent"],
                               avoid=["cancel order 3"], max_turns=5)
    assert [i["id"] for i in items] == ["gen-refund", "gen-refund-2"]  # empty goal dropped, ids unique
    for item in items:
        check_scenario("test", item["scenario"])
        assert item["scenario"]["max_turns"] == 5 and item["scenario"]["opening"] == ["hi"]
        assert item["generated"]["reviewed"] is False and item["generated"]["model"] == "fake-model"
    prompt = model.prompts[0]
    assert "- refunds" in prompt and "- busy parent" in prompt and "- cancel order 3" in prompt
    assert unreviewed(items + [{"id": "ok", "generated": {"reviewed": True}}, {"id": "hand"}]) == ["gen-refund", "gen-refund-2"]
