from __future__ import annotations

import pytest

from eval import gpt_agent_judge


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    def _fake_call(prompt: str, *, model: str, temperature: float) -> dict:
        return {
            "scores": {
                "correctness": 5,
                "citation_support": 5,
                "completeness": 4,
                "conciseness": 4,
                "hallucination_risk": 5,
                "overall": 90,
            },
            "rationales": {
                "correctness": "Matches gold.",
                "citation_support": "Citations align.",
            },
            "flags": {
                "missing_citation": False,
                "contradiction_with_evidence": False,
                "gold_mismatch_but_supported": False,
            },
            "used_citations": [1],
        }

    monkeypatch.setattr(gpt_agent_judge, "_call_llm", _fake_call)


def test_extract_citation_ids_variants():
    text = "Correct answer [1,2] referencing (CIT:3) and <CIT_4> plus ((5))."
    ids = gpt_agent_judge.extract_citation_ids(text)
    assert ids == [1, 2, 3, 4, 5]


def test_support_overlap_jaccard():
    answer = "Roger Federer won in 2017. [1]"
    passages = {
        1: "In the 2017 Australian Open final, Roger Federer defeated Rafael Nadal."
    }
    overlap = gpt_agent_judge.compute_support_overlap(answer, passages)
    assert overlap > 0.3


def test_judge_example_metrics_with_citation():
    question = "Who won the 2017 Australian Open men's singles?"
    gold = "Roger Federer."
    answer = "Roger Federer. [1]"
    passages = {
        1: "In the 2017 Australian Open final, Roger Federer defeated Rafael Nadal."
    }

    result = gpt_agent_judge.judge_example(question, gold, answer, passages)
    auto = result["auto_metrics"]
    assert auto["has_citation"] is True
    assert auto["num_citations"] == 1
    assert auto["idk_plus_cit_violation"] is False
    assert auto["f1_short"] == pytest.approx(1.0)
    assert auto["support_overlap"] > 0.1
    assert result["composite_overall"] > 0


def test_judge_example_idk_violation():
    question = "Who won the 2017 Australian Open men's singles?"
    gold = "Roger Federer."
    answer = "I don't know. [1]"
    passages = {
        1: "In the 2017 Australian Open final, Roger Federer defeated Rafael Nadal."
    }

    result = gpt_agent_judge.judge_example(question, gold, answer, passages)
    auto = result["auto_metrics"]
    assert auto["idk_plus_cit_violation"] is True
    assert auto["has_citation"] is True
    assert auto["support_overlap"] == pytest.approx(0.0)
