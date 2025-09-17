from agentic_rag.eval.signals import (
    CITATION_RE,
    em_f1,
    faithfulness_fallback,
    sentence_support,
)


def test_base_metrics():
    # Test faithfulness_fallback and em_f1
    ans_idk = "I don't know."
    ff_idk = faithfulness_fallback(ans_idk, gold="nonempty", overlap=0.0)
    assert ff_idk == 0.0

    ef = em_f1(ans_idk, gold="Tenet disutradarai oleh Christopher Nolan.")
    assert ef["em"] == 0.0

    ef_match = em_f1("Tenet", gold="tenet")
    assert ef_match["em"] == 1.0
    assert ef_match["f1"] == 1.0


def test_citation_rules():
    ctx = {"d1": "The sky is blue and the sun is bright."}

    # Test (a): "I don't know." -> overlap=0
    ans_idk = "I don't know."
    sup_idk = sentence_support(ans_idk, ctx)
    assert sup_idk["overlap"] == 0.0
    assert sup_idk["supported"] == 0
    assert sup_idk["sentences"] == 1
    assert sup_idk["idk_with_citation_count"] == 0

    # Test (a variant): "IDK" with citation is not supported and is flagged
    ans_idk_cit = "I don't know. [CIT:d1]"
    sup_idk_cit = sentence_support(ans_idk_cit, ctx)
    assert sup_idk_cit["overlap"] == 0.0
    assert sup_idk_cit["supported"] == 0
    assert sup_idk_cit["sentences"] == 1
    assert sup_idk_cit["idk_with_citation_count"] == 1
    # The regex should still match the literal pattern
    assert CITATION_RE.search(ans_idk_cit) is not None

    # Test (b): Claim + valid [CIT:d1] -> overlap > 0
    # Note: this requires a live embedding model or a good mock
    ans_valid = "The color of the sky is blue. [CIT:d1]"
    sup_valid = sentence_support(ans_valid, ctx)
    assert sup_valid["overlap"] == 1.0
    assert sup_valid["supported"] == 1

    # Test (b variant): Claim + valid citation but to doc not in context
    ans_valid_bad_ctx = "The sky is colored blue. [CIT:d2]"
    sup_valid_bad_ctx = sentence_support(ans_valid_bad_ctx, ctx)
    assert sup_valid_bad_ctx["overlap"] == 0.0
    assert sup_valid_bad_ctx["supported"] == 0

    # Test (c): Reject invalid citation format
    ans_invalid_cit = "The sky is blue. [CIT:foo.txt#3]"
    sup_invalid_cit = sentence_support(ans_invalid_cit, ctx)
    assert sup_invalid_cit["overlap"] == 0.0
    assert sup_invalid_cit["supported"] == 0

    # Test (c variant): Reject malformed citation (space is not allowed by regex)
    ans_malformed_cit = "The sky is blue. [CIT: d1]"
    sup_malformed_cit = sentence_support(ans_malformed_cit, ctx)
    assert sup_malformed_cit["overlap"] == 0.0
    assert sup_malformed_cit["supported"] == 0

    # Test sentence without citation is not supported
    ans_no_cit = "The sun is bright."
    sup_no_cit = sentence_support(ans_no_cit, ctx)
    assert sup_no_cit["overlap"] == 0.0
    assert sup_no_cit["supported"] == 0

    # Test sentence with multiple citations is not supported
    ans_multi_cit = "The sky is blue. [CIT:d1] [CIT:d2]"
    sup_multi_cit = sentence_support(ans_multi_cit, ctx)
    assert sup_multi_cit["overlap"] == 0.0
    assert sup_multi_cit["supported"] == 0
