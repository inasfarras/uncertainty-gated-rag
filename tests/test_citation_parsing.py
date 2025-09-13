"""Test citation parsing functionality."""

import pytest
from agentic_rag.eval.signals import (
    extract_citations,
    extract_sentence_citations,
    overlap_ratio,
    sentence_split,
)


def test_extract_citations():
    """Test citation extraction from text."""
    text = "This is a fact [CIT:doc1]. Another fact [CIT:doc2] here."
    citations = extract_citations(text)
    assert citations == ["doc1", "doc2"]


def test_extract_citations_empty():
    """Test citation extraction from text with no citations."""
    text = "This is a fact. Another fact here."
    citations = extract_citations(text)
    assert citations == []


def test_extract_sentence_citations():
    """Test citation extraction per sentence."""
    text = "First sentence [CIT:doc1]. Second sentence [CIT:doc2] [CIT:doc3]. Third sentence."
    sentence_citations = extract_sentence_citations(text)

    assert len(sentence_citations) == 3
    assert sentence_citations[0] == {"doc1"}
    assert sentence_citations[1] == {"doc2", "doc3"}
    assert sentence_citations[2] == set()


def test_sentence_split():
    """Test sentence splitting."""
    text = "First sentence. Second sentence! Third sentence?"
    sentences = sentence_split(text)
    assert sentences == ["First sentence.", "Second sentence!", "Third sentence?"]


def test_overlap_ratio_with_citations():
    """Test overlap ratio calculation with valid citations."""
    answer = "The capital is Paris [CIT:doc1]. The population is large [CIT:doc2]."
    contexts = ["Paris is the capital of France.", "The city has a large population."]
    context_ids = ["doc1", "doc2"]

    ratio = overlap_ratio(answer, contexts, context_ids=context_ids)
    assert ratio == 1.0  # Both sentences have valid citations


def test_overlap_ratio_with_invalid_citations():
    """Test overlap ratio calculation with invalid citations."""
    answer = (
        "The capital is Paris [CIT:doc_invalid]. The population is large [CIT:doc2]."
    )
    contexts = ["Paris is the capital of France.", "The city has a large population."]
    context_ids = ["doc1", "doc2"]

    # Should fallback to semantic similarity for invalid citations
    ratio = overlap_ratio(answer, contexts, context_ids=context_ids, sim_threshold=0.5)
    assert 0.0 <= ratio <= 1.0


def test_overlap_ratio_no_citations():
    """Test overlap ratio calculation with no citations."""
    answer = "The capital is Paris. The population is large."
    contexts = ["Paris is the capital of France.", "The city has a large population."]
    context_ids = ["doc1", "doc2"]

    # Should use semantic similarity
    ratio = overlap_ratio(answer, contexts, context_ids=context_ids, sim_threshold=0.5)
    assert 0.0 <= ratio <= 1.0


def test_overlap_ratio_empty_answer():
    """Test overlap ratio calculation with empty answer."""
    answer = ""
    contexts = ["Some context."]
    context_ids = ["doc1"]

    ratio = overlap_ratio(answer, contexts, context_ids=context_ids)
    assert ratio == 0.0


def test_formatted_prompt_context():
    """Test that our new prompt format would work with citation extraction."""
    # Simulate what the new prompt format looks like
    context_format = """CTX[doc1]:
This is the first document content.

CTX[doc2]:
This is the second document content."""

    # Extract doc IDs that would be available
    import re

    available_ids = re.findall(r"CTX\[([^\]]+)\]:", context_format)
    assert available_ids == ["doc1", "doc2"]

    # Test citation matching
    answer = "Based on the documents [CIT:doc1]. More info [CIT:doc2]."
    citations = extract_citations(answer)

    # All citations should be valid
    valid_citations = [cit for cit in citations if cit in available_ids]
    assert valid_citations == ["doc1", "doc2"]
    assert len(valid_citations) == len(citations)  # No invalid citations


if __name__ == "__main__":
    pytest.main([__file__])
