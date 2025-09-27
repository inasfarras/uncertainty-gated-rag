from agentic_rag.anchors.validators import (
    conflict_risk,
    coverage,
    mismatch_flags,
    required_anchors,
)


def test_required_anchors_simple():
    q = "When did Apple release iPhone in 2007?"
    anchors = required_anchors(q)
    assert "2007" in anchors


def test_coverage_and_mismatch():
    q = "When did Apple release iPhone in 2007?"
    texts = ["Apple iPhone was introduced in 2007 at Macworld."]
    cov, present, missing = coverage(q, texts)
    assert cov >= 0.5
    flags = mismatch_flags(q, texts)
    assert flags["temporal_mismatch"] is False


def test_conflict_risk():
    texts = [
        "In 2001 something happened. In 2003 another event. Then 2005 and 2007 were important.",
    ]
    r = conflict_risk(texts)
    assert 0.1 <= r <= 0.6
