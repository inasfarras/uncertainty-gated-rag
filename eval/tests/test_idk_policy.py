from eval.idk_policy import classify_answer


def make_auto(**kwargs):
    base = {
        "has_citation": False,
        "num_citations": 0,
        "f1_short": 0.0,
        "f1_full": 0.0,
        "em": 0.0,
    }
    base.update(kwargs)
    return base


def test_idk_without_citation_is_safe():
    result = classify_answer("I don't know", make_auto(), "gold answer", {}, {})
    assert result["safe_idk"] is True
    assert result["bad_idk"] is False
    assert result["hallucination"] is False
    assert result["match"] is False
    assert result["partial_correct"] is False
    assert result["perfect_match"] is False


def test_idk_with_citation_is_bad():
    auto = make_auto(has_citation=True, num_citations=1)
    result = classify_answer("I don't know [CIT:foo]", auto, "gold answer", {}, {})
    assert result["safe_idk"] is False
    assert result["bad_idk"] is True
    assert result["hallucination"] is True
    assert result["match"] is False


def test_wrong_answer_marks_hallucination():
    auto = make_auto(f1_short=0.0, f1_full=0.0)
    result = classify_answer(
        "Wrong", auto, "correct", {}, {"contradiction_with_evidence": False}
    )
    assert result["hallucination"] is True
    assert result["safe_idk"] is False
    assert result["match"] is False


def test_partial_credit_is_not_hallucination():
    auto = make_auto(f1_short=0.5)
    result = classify_answer("Partially right", auto, "correct", {}, {})
    assert result["hallucination"] is False
    assert result["match"] is True
    assert result["partial_correct"] is True
    assert result["perfect_match"] is False


def test_perfect_detection():
    auto = make_auto(em=1.0)
    result = classify_answer("Exact", auto, "Exact", {}, {})
    assert result["perfect_match"] is True
    assert result["match"] is True
    assert result["partial_correct"] is False


def test_idk_on_invalid_gold_is_safe():
    result = classify_answer("I don't know", make_auto(), "invalid question", {}, {})
    assert result["safe_idk"] is True
    assert result["hallucination"] is False
    assert result["match"] is False
