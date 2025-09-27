from agentic_rag.intent.interpreter import interpret


def test_rules_awards_slot_completeness() -> None:
    question = "Who won Best Animated Feature at the 2004 Oscars?"

    intent = interpret(question, llm_budget_ok=False)

    assert intent.source_of_intent == "rule_only"
    assert intent.slots.get("year") == "2004"
    assert intent.slots.get("category") == "Best Animated Feature"
    assert intent.slot_completeness == 1.0
    assert "Academy Awards" in intent.core_entities
