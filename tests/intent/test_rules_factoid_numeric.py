from agentic_rag.intent.interpreter import interpret


def test_rules_factoid_numeric_rule_only() -> None:
    question = "What was LeBron James' average points per game in the 2017 season?"

    intent = interpret(question, llm_budget_ok=False)

    assert intent.source_of_intent == "rule_only"
    assert intent.slots.get("unit") == "per game"
    assert intent.slots.get("time_window") == "2016\u201317"
    assert intent.slot_completeness >= 0.8
    assert intent.intent_confidence >= 0.7
