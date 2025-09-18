from agentic_rag.eval.signals import sentence_support, faithfulness_fallback, em_f1


def test_idk_is_unsupported_and_faithfulness_zero_when_gold_exists():
    ctx = {"d1": "Tenet is a film directed by Christopher Nolan."}
    ans = "I don't know."
    sup = sentence_support(ans, ctx, tau_sim=0.58)
    assert sup["overlap"] == 0.0
    assert faithfulness_fallback(ans, gold="nonempty", overlap=sup["overlap"]) == 0.0


def test_supported_sentence_with_valid_citation_counts():
    ctx = {"d1": "Tenet is a film directed by Christopher Nolan."}
    ans = "Tenet is a film directed by Christopher Nolan. [CIT:d1]"
    sup = sentence_support(ans, ctx, tau_sim=0.30)
    assert sup["overlap"] == 1.0
    ef = em_f1(ans, gold="Tenet is a film directed by Christopher Nolan.")
    assert 0.0 <= ef["f1"] <= 1.0
