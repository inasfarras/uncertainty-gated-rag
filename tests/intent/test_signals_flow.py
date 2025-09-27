from agentic_rag.agent.gate import GateAction
from agentic_rag.config import settings
from agentic_rag.gate.adapter import BAUGAdapter


def _make_signals(**overrides: float) -> dict[str, float | int | bool | str]:
    base: dict[str, float | int | bool | str] = {
        "overlap_est": 0.3,
        "faith_est": 0.4,
        "new_hits_ratio": 0.5,
        "anchor_coverage": 0.5,
        "conflict_risk": 0.1,
        "budget_left": 600,
        "round_idx": 1,
        "has_reflect_left": True,
        "lexical_uncertainty": 0.0,
        "completeness": 0.7,
        "semantic_coherence": 1.0,
        "answer_length": 12,
        "question_complexity": 0.4,
        "intent_confidence": 0.6,
        "slot_completeness": 0.7,
        "source_of_intent": "rule_only",
        "validators_passed": True,
    }
    base.update(overrides)
    return base


def test_baug_abstains_when_slots_missing_and_budget_low() -> None:
    adapter = BAUGAdapter()
    action = adapter.decide(_make_signals(slot_completeness=0.5, budget_left=200))
    assert action == GateAction.ABSTAIN


def test_baug_requests_more_when_validators_fail_with_budget() -> None:
    adapter = BAUGAdapter()
    action = adapter.decide(_make_signals(validators_passed=False, budget_left=700))
    assert action == GateAction.RETRIEVE_MORE


def test_baug_stops_on_overlap_with_validators() -> None:
    adapter = BAUGAdapter()
    action = adapter.decide(
        _make_signals(
            overlap_est=settings.OVERLAP_TAU + 0.05,
            validators_passed=True,
            slot_completeness=0.8,
        )
    )
    assert action == GateAction.STOP
    assert adapter.last_source_of_intent() == "rule_only"


def test_baug_stop_low_gain_on_stagnation() -> None:
    adapter = BAUGAdapter()
    action = adapter.decide(
        _make_signals(
            new_hits_ratio=max(0.0, settings.NEW_HITS_EPS - 0.01),
            slot_completeness=0.8,
            validators_passed=True,
        )
    )
    assert action == "STOP_LOW_GAIN"
