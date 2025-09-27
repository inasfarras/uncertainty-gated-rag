from agentic_rag.intent.interpreter import merge_rule_llm
from agentic_rag.intent.types import Intent


def test_llm_fallback_merge_enriches_slots() -> None:
    rule_intent = Intent(
        task_type="factoid",
        core_entities=["Player X"],
        slots={"year": "2020"},
        canonical_query="player x stats",
        ambiguity_flags=[],
        intent_confidence=0.5,
        slot_completeness=0.5,
        source_of_intent="rule_only",
    )

    llm_intent = Intent(
        task_type="factoid",
        core_entities=["Player X", "Team Y"],
        slots={"year": "2020", "unit": "per game", "time_window": "2019-20"},
        canonical_query="Player X per game statistics in 2019-20",
        ambiguity_flags=[],
        intent_confidence=0.8,
        slot_completeness=0.9,
        source_of_intent="llm_only",
    )

    merged = merge_rule_llm(rule_intent, llm_intent)

    assert merged.source_of_intent == "llm_fallback"
    assert "Team Y" in merged.core_entities
    assert merged.slots.get("unit") == "per game"
    assert merged.slots.get("time_window") == "2019\u201320"
    assert merged.intent_confidence >= llm_intent.intent_confidence
    assert merged.slot_completeness >= llm_intent.slot_completeness
