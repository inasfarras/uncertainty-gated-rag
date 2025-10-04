"""Shared helpers to consistently label IDK behavior and hallucinations."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping

from . import gpt_agent_judge
from .gpt_agent_judge import detect_idk

_INVALID_GOLD_KEYWORDS = {
    "invalid question",
    "invalid",
    "not a valid question",
}


def _normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    normalized = str(text)
    normalized = " ".join(normalized.split()).strip().lower()
    return normalized


def is_invalid_gold(gold: str | None) -> bool:
    normalized = _normalize_text(gold)
    if not normalized:
        return False
    if normalized in _INVALID_GOLD_KEYWORDS:
        return True
    return "invalid question" in normalized


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _best_f1(auto_metrics: Mapping[str, Any]) -> float:
    for key in ("f1_short", "f1_full"):
        if key in auto_metrics and auto_metrics[key] is not None:
            candidate = _as_float(auto_metrics[key])
            if candidate:
                return candidate
    return 0.0


def _is_pure_idk(text: str) -> bool:
    lowered = text.strip().lower()
    lowered = re.sub(r"\s*[\[\(<][^\]\)>]*[\]\)>]\s*", " ", lowered)
    lowered = " ".join(lowered.split())
    lowered = lowered.rstrip(".! ")
    return lowered in {"i don't know", "i dont know"}


def classify_answer(
    answer: str | None,
    auto_metrics: Mapping[str, Any] | None,
    gold: str | None,
    passages: Mapping[str, Any] | None = None,
    llm_flags: Mapping[str, Any] | None = None,
    *,
    composite: Any | None = None,
) -> Dict[str, bool]:
    metrics = auto_metrics or {}
    flags = llm_flags or {}

    answer_text = (answer or "").strip()
    clean_answer = gpt_agent_judge._strip_citations(answer_text)
    idk_flag = detect_idk(clean_answer) or _is_pure_idk(answer_text)

    has_citation = bool(metrics.get("has_citation"))
    num_citations = _as_float(metrics.get("num_citations"))
    if num_citations:
        has_citation = True

    best_f1 = _best_f1(metrics)
    em_val = _as_float(metrics.get("em"))
    gold_invalid = is_invalid_gold(gold)
    llm_contradiction = bool(flags.get("contradiction_with_evidence", False))

    safe_idk = False
    bad_idk = False
    hallucination = False

    if idk_flag:
        if has_citation:
            bad_idk = True
            hallucination = True
        else:
            safe_idk = True
            if gold_invalid:
                safe_idk = True
        llm_contradiction = False if not bad_idk else llm_contradiction
    else:
        wrong_vs_gold = best_f1 <= 1e-6
        if gold_invalid:
            wrong_vs_gold = True
        hallucination = bool(llm_contradiction or wrong_vs_gold)

    if bad_idk:
        hallucination = True

    composite_val = _as_float(composite)
    perfect_match = em_val >= 0.999
    partial_correct = (not perfect_match) and (best_f1 >= 0.5 or composite_val >= 50.0)
    match = perfect_match or partial_correct

    return {
        "safe_idk": safe_idk,
        "bad_idk": bad_idk,
        "hallucination": hallucination,
        "gold_invalid": gold_invalid,
        "perfect_match": perfect_match,
        "partial_correct": partial_correct,
        "match": match,
    }
