from __future__ import annotations

"""Lightweight anchor validators and coverage/mismatch heuristics.

Reuses judge/qanchor utilities when available to avoid duplication.
"""

from typing import Iterable, List, Set, Tuple

from agentic_rag.agent.judge import (
    anchors_present_in_texts as judge_anchors_present_in_texts,
    extract_required_anchors as judge_extract_required_anchors,
)


def required_anchors(question: str) -> Set[str]:
    try:
        return set(judge_extract_required_anchors(question))
    except Exception:
        # Fallback simple heuristic
        from agentic_rag.agent.qanchors import extract_required_anchors as q_extract

        return set(q_extract(question))


def coverage(question: str, texts: Iterable[str]) -> Tuple[float, Set[str], Set[str]]:
    req = required_anchors(question)
    if not req:
        return 1.0, set(), set()
    present, _ = judge_anchors_present_in_texts(req, list(texts))
    miss = req - present
    cov = 1.0 - (len(miss) / max(1, len(req)))
    return float(cov), present, miss


def mismatch_flags(question: str, texts: Iterable[str]) -> dict:
    ql = (question or "").lower()
    tl = "\n".join((t or "").lower() for t in texts)
    temporal_mismatch = False
    unit_mismatch = False
    entity_mismatch = False

    # Temporal: year present in question but not in texts
    import re as _re

    q_years = set(_re.findall(r"\b(?:19|20)\d{2}\b", ql))
    if q_years and not any(y in tl for y in q_years):
        temporal_mismatch = True

    # Units: crude unit words present in q but missing in texts
    units = ["per game", "%", "percent", "ex-dividend"]
    if any(u in ql for u in units) and not any(u in tl for u in units):
        unit_mismatch = True

    # Entities: if the main proper token (first capitalized word) not in texts
    words = [w for w in question.split() if w.istitle() and w.isalpha()]
    if words and not any(w.lower() in tl for w in [words[0]]):
        entity_mismatch = True

    return {
        "temporal_mismatch": temporal_mismatch,
        "unit_mismatch": unit_mismatch,
        "entity_mismatch": entity_mismatch,
    }


def conflict_risk(texts: Iterable[str]) -> float:
    # Simple proxy: many distinct years scattered => potential conflict
    import re as _re

    tl = "\n".join((t or "") for t in texts)
    years = _re.findall(r"\b(?:19|20)\d{2}\b", tl)
    uniq = len(set(years))
    if uniq >= 6:
        return 0.6
    if uniq >= 4:
        return 0.35
    if uniq >= 2:
        return 0.2
    return 0.05


# Domain-specific lightweight validators

AWARD_TOKENS = {
    "oscar",
    "academy award",
    "best visual effects",
    "best actor",
    "best actress",
    "best picture",
}

TOURNAMENT_TOKENS = {
    "grand slam",
    "wimbledon",
    "u.s. open",
    "us open",
    "australian open",
    "french open",
}


def award_tournament_requirements(question: str, texts: Iterable[str]) -> dict:
    """Check if award/tournament queries have year/category anchors covered.

    Returns dict with: {requires_year: bool, requires_category: bool, has_year: bool,
    has_category: bool, missing: list[str]}
    """
    ql = (question or "").lower()
    tl = "\n".join((t or "").lower() for t in texts)
    is_award = any(tok in ql for tok in AWARD_TOKENS)
    is_tourn = any(tok in ql for tok in TOURNAMENT_TOKENS)
    requires_year = is_award or is_tourn or (" in " in ql and any(ch.isdigit() for ch in ql))
    requires_category = is_award

    import re as _re

    has_year = bool(_re.search(r"\b(?:19|20)\d{2}\b", tl))
    # Category: presence of any award token in context
    has_category = any(tok in tl for tok in AWARD_TOKENS)

    missing: list[str] = []
    if requires_year and not has_year:
        missing.append("year")
    if requires_category and not has_category:
        missing.append("category")
    return {
        "requires_year": requires_year,
        "requires_category": requires_category,
        "has_year": has_year,
        "has_category": has_category,
        "missing": missing,
    }


def units_time_requirements(question: str, texts: Iterable[str]) -> dict:
    """Check units/time anchors like % / per game / months/quarters / YYYY.

    Returns dict: {needs_units: bool, needs_time: bool, has_units: bool, has_time: bool, missing: list[str]}
    """
    ql = (question or "").lower()
    tl = "\n".join((t or "").lower() for t in texts)
    needs_units = any(tok in ql for tok in ["per game", "%", "percent"])
    needs_time = bool(
        any(m in ql for m in [
            " jan ", " feb ", " mar ", " apr ", " may ", " jun ", " jul ", " aug ", " sep ", " oct ", " nov ", " dec ",
            " q1", " q2", " q3", " q4"
        ])
        or any(ch.isdigit() for ch in ql)
    )

    import re as _re

    has_units = ("per game" in tl) or (" percent" in tl) or ("%" in tl)
    has_time = bool(
        _re.search(r"\b(?:19|20)\d{2}\b", tl)
        or _re.search(r"\bq[1-4]\b", tl)
        or _re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b", tl)
    )
    missing: list[str] = []
    if needs_units and not has_units:
        missing.append("units")
    if needs_time and not has_time:
        missing.append("time")
    return {
        "needs_units": needs_units,
        "needs_time": needs_time,
        "has_units": has_units,
        "has_time": has_time,
        "missing": missing,
    }
