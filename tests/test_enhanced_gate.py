"""Tests for enhanced uncertainty gate improvements."""

import pytest
from agentic_rag.agent.gate import GateAction, GateSignals, UncertaintyGate
from agentic_rag.agent.loop import (
    _assess_lexical_uncertainty,
    _assess_question_complexity,
    _assess_response_completeness,
    _assess_semantic_coherence,
)
from agentic_rag.config import Settings


def test_enhanced_lexical_uncertainty_assessment():
    """Test enhanced lexical uncertainty assessment."""
    # High uncertainty responses
    assert _assess_lexical_uncertainty("I might be wrong, but maybe it's unclear") > 0.5
    assert _assess_lexical_uncertainty("I don't know the answer") > 0.8

    # Low uncertainty responses
    assert _assess_lexical_uncertainty("This is definitely the correct answer") < 0.3
    assert _assess_lexical_uncertainty("The result is precisely 42") < 0.2

    # Empty/short responses
    assert _assess_lexical_uncertainty("") == 1.0
    assert _assess_lexical_uncertainty("No") > 0.8


def test_enhanced_completeness_assessment():
    """Test enhanced completeness assessment."""
    # Complete responses
    complete_response = "This is a complete answer with proper structure. It has multiple sentences and ends properly."
    assert _assess_response_completeness(complete_response) > 0.8

    # Incomplete responses
    assert _assess_response_completeness("Short") < 0.5
    assert _assess_response_completeness("This is incomplete...") < 0.9

    # Empty responses
    assert _assess_response_completeness("") == 0.0


def test_semantic_coherence_assessment():
    """Test semantic coherence assessment."""
    # Coherent response
    coherent = (
        "The answer is yes. This is because the evidence supports this conclusion."
    )
    assert _assess_semantic_coherence(coherent) >= 0.7

    # Contradictory response
    contradictory = "The answer is yes. However, it is also no and incorrect."
    assert _assess_semantic_coherence(contradictory) < 0.6

    # Single sentence (assumed coherent)
    single = "This is a single coherent sentence."
    assert _assess_semantic_coherence(single) >= 0.8


def test_question_complexity_assessment():
    """Test question complexity assessment."""
    # Simple questions
    simple = "What is the capital of France?"
    assert _assess_question_complexity(simple) < 0.5

    # Complex questions
    complex_q = "Compare and contrast the various implications of quantum mechanics and relativity theory on modern physics, analyzing their relationship and explaining why they are both important."
    assert _assess_question_complexity(complex_q) >= 0.6

    # Medium complexity
    medium = "How does photosynthesis work in plants?"
    complexity = _assess_question_complexity(medium)
    assert 0.2 <= complexity <= 0.8


def test_enhanced_uncertainty_gate():
    """Test enhanced uncertainty gate with new signals."""
    settings = Settings()
    gate = UncertaintyGate(settings)

    # High confidence scenario
    high_conf_signals = GateSignals(
        faith=0.9,
        overlap=0.8,
        lexical_uncertainty=0.1,
        completeness=0.9,
        semantic_coherence=0.9,
        answer_length=100,
        question_complexity=0.5,
        budget_left_tokens=1000,
        round_idx=0,
        has_reflect_left=True,
        novelty_ratio=0.8,
    )

    assert gate.decide(high_conf_signals) == GateAction.STOP

    # Test low budget scenario
    low_budget_signals = GateSignals(
        faith=0.5,
        overlap=0.5,
        lexical_uncertainty=0.3,
        completeness=0.8,
        semantic_coherence=0.8,
        answer_length=50,
        question_complexity=0.5,
        budget_left_tokens=100,  # Below LOW_BUDGET_TOKENS (500)
        round_idx=0,
        has_reflect_left=True,
        novelty_ratio=0.6,
    )

    assert gate.decide(low_budget_signals) == GateAction.STOP_LOW_BUDGET

    # High uncertainty scenario
    high_unc_signals = GateSignals(
        faith=0.3,
        overlap=0.2,
        lexical_uncertainty=0.8,
        completeness=0.4,
        semantic_coherence=0.3,
        answer_length=20,
        question_complexity=0.8,
        budget_left_tokens=1000,
        round_idx=1,
        has_reflect_left=True,
        novelty_ratio=0.3,
    )

    decision = gate.decide(high_unc_signals)
    assert decision in [GateAction.RETRIEVE_MORE, GateAction.REFLECT]


def test_gate_adaptive_weights():
    """Test adaptive weight calculation."""
    settings = Settings()
    gate = UncertaintyGate(settings)

    # Simple question signals
    simple_signals = GateSignals(
        faith=0.5,
        overlap=0.5,
        lexical_uncertainty=0.3,
        completeness=0.8,
        semantic_coherence=0.8,
        answer_length=50,
        question_complexity=0.2,
        budget_left_tokens=1000,
        round_idx=0,
        has_reflect_left=True,
        novelty_ratio=0.5,
    )

    simple_weights = gate._get_adaptive_weights(simple_signals)

    # Complex question signals
    complex_signals = GateSignals(
        faith=0.5,
        overlap=0.5,
        lexical_uncertainty=0.3,
        completeness=0.8,
        semantic_coherence=0.8,
        answer_length=50,
        question_complexity=0.8,
        budget_left_tokens=1000,
        round_idx=0,
        has_reflect_left=True,
        novelty_ratio=0.5,
    )

    complex_weights = gate._get_adaptive_weights(complex_signals)

    # Complex questions should have higher semantic and faith weights
    assert complex_weights["semantic"] >= simple_weights["semantic"]
    assert complex_weights["faith"] >= simple_weights["faith"]


def test_gate_caching_performance():
    """Test gate caching for performance."""
    settings = Settings()
    settings.ENABLE_GATE_CACHING = True
    gate = UncertaintyGate(settings)

    # Create identical signals
    signals = GateSignals(
        faith=0.7,
        overlap=0.6,
        lexical_uncertainty=0.2,
        completeness=0.8,
        semantic_coherence=0.8,
        answer_length=80,
        question_complexity=0.5,
        budget_left_tokens=1000,
        round_idx=0,
        has_reflect_left=True,
        novelty_ratio=0.6,
    )

    # First call
    result1 = gate.decide(signals)

    # Second call with identical signals should use cache
    result2 = gate.decide(signals)

    assert result1 == result2

    # Check cache stats - should have 1 hit and 1 miss
    stats = gate.get_cache_stats()
    assert stats["cache_size"] >= 1
    assert stats["cache_hits"] >= 1
    assert stats["cache_misses"] >= 1
    assert stats["hit_rate"] > 0


def test_performance_with_batch_processing():
    """Test batch processing performance improvements."""
    from agentic_rag.agent.performance import BatchProcessor

    responses = [
        "I might be uncertain about this answer",
        "This is definitely correct",
        "Maybe it's possible, but unclear",
        "Absolutely certain and precise",
        "",
    ]

    batch_results = BatchProcessor.batch_assess_lexical_uncertainty(responses)

    assert len(batch_results) == len(responses)
    assert batch_results[0] >= 0.5  # Uncertain response
    assert batch_results[1] <= 0.5  # Confident response
    assert batch_results[4] == 1.0  # Empty response


if __name__ == "__main__":
    pytest.main([__file__])


def test_low_budget_reason_flag():
    """Ensure low-budget exits surface the stop reason."""
    settings = Settings()
    gate = UncertaintyGate(settings)
    extras: dict[str, object] = {}
    signals = GateSignals(
        faith=0.5,
        overlap=0.5,
        lexical_uncertainty=0.1,
        completeness=0.9,
        semantic_coherence=0.9,
        answer_length=120,
        question_complexity=0.5,
        budget_left_tokens=settings.LOW_BUDGET_TOKENS - 1,
        round_idx=0,
        has_reflect_left=True,
        novelty_ratio=0.5,
        extras=extras,
    )

    decision = gate.decide(signals)
    assert decision == GateAction.STOP_LOW_BUDGET
    assert extras.get("stop_reason") == "budget_exhausted"
