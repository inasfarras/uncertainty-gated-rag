import pytest
from agentic_rag.agent.gate import (
    GateAction,
    GateSignals,
    UncertaintyGate,
)
from agentic_rag.config import Settings


@pytest.fixture
def settings():
    s = Settings()
    s.FAITHFULNESS_TAU = 0.7
    s.OVERLAP_TAU = 0.6
    s.UNCERTAINTY_TAU = 0.5
    return s


# SimpleThresholdGate tests removed - only UncertaintyGate available


def test_gate_uncertainty_triggers_reflect_when_uncertain(settings):
    gate = UncertaintyGate(settings)
    # High uncertainty (low faith, low overlap)
    signals = GateSignals(
        faith=0.4,
        overlap=0.3,
        lexical_uncertainty=0.8,
        completeness=0.5,
        budget_left_tokens=1000,
        round_idx=1,
        has_reflect_left=True,
        novelty_ratio=0.5,
        extras={},
    )
    assert gate.decide(signals) == GateAction.REFLECT


def test_gate_uncertainty_stops_when_confident(settings):
    gate = UncertaintyGate(settings)
    signals = GateSignals(
        faith=0.9,
        overlap=0.9,
        lexical_uncertainty=0.1,
        completeness=1.0,
        budget_left_tokens=1000,
        round_idx=0,
        has_reflect_left=True,
        novelty_ratio=1.0,
    )
    assert gate.decide(signals) == GateAction.STOP


def test_gate_uncertainty_retrieves_when_moderately_uncertain(settings):
    gate = UncertaintyGate(settings)
    # Moderately uncertain, below REFLECT threshold
    signals = GateSignals(
        faith=0.6,
        overlap=0.5,
        lexical_uncertainty=0.2,
        completeness=0.8,
        budget_left_tokens=1000,
        round_idx=1,
        has_reflect_left=True,
        novelty_ratio=0.5,
        extras={},
    )
    assert gate.decide(signals) == GateAction.RETRIEVE_MORE
