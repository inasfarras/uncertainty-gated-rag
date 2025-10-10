"""Lightweight anchor validators and coverage/mismatch heuristics.

Reuses judge/qanchor utilities when available to avoid duplication.
"""

from __future__ import annotations

from collections.abc import Iterable

from agentic_rag.agent.judge import (
    anchors_present_in_texts as judge_anchors_present_in_texts,
)
from agentic_rag.agent.judge import (
    extract_required_anchors as judge_extract_required_anchors,
)


def required_anchors(question: str) -> set[str]:
    """Extract anchors using Judge, with a fallback to qanchors."""
    try:
        # Use the more sophisticated Judge extractor preferentially
        return set(judge_extract_required_anchors(question))
    except Exception:
        # Fallback to simple heuristic ONLY if judge fails
        from agentic_rag.agent.qanchors import extract_required_anchors as q_extract

        return set(q_extract(question))


def coverage(question: str, texts: Iterable[str]) -> tuple[float, set[str], set[str]]:
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
    requires_year = (
        is_award or is_tourn or (" in " in ql and any(ch.isdigit() for ch in ql))
    )
    requires_category = is_award
    requires_event = is_award or is_tourn

    import re as _re

    has_year = bool(_re.search(r"\b(?:19|20)\d{2}\b", tl))
    event_tokens = AWARD_TOKENS | TOURNAMENT_TOKENS
    has_event = any(tok in tl for tok in event_tokens)
    has_category = any(tok in tl for tok in AWARD_TOKENS)

    missing: list[str] = []
    if requires_year and not has_year:
        missing.append("year")
    if requires_category and not has_category:
        missing.append("category")
    if requires_event and not has_event:
        missing.append("event")
    return {
        "requires_year": requires_year,
        "requires_category": requires_category,
        "requires_event": requires_event,
        "has_year": has_year,
        "has_category": has_category,
        "has_event": has_event,
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
        any(
            m in ql
            for m in [
                " jan ",
                " feb ",
                " mar ",
                " apr ",
                " may ",
                " jun ",
                " jul ",
                " aug ",
                " sep ",
                " oct ",
                " nov ",
                " dec ",
                " q1",
                " q2",
                " q3",
                " q4",
            ]
        )
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


def list_requirements(question: str, texts: Iterable[str]) -> dict:
    """Guardrail for list-style questions that specify a required count.

    Attempts a light extraction of quoted titles or capitalized spans from the
    supplied texts and checks whether the distinct item count meets the request.
    Returns dict: {needs_list: bool, required_count: int|None, has_count: int, missing: list[str]}.
    """
    import re as _re

    q = (question or "").lower()
    # Heuristics: detect patterns like "3 of the", "top 3", "list 3"
    needs_list = False
    required = None
    m = _re.search(r"\b(?:top|list|name|what)\s+(\d{1,2})\b", q)
    if m:
        needs_list = True
        try:
            required = int(m.group(1))
        except Exception:
            required = None
    if not needs_list:
        m2 = _re.search(r"\b(\d{1,2})\s+of\s+the\b", q)
        if m2:
            needs_list = True
            try:
                required = int(m2.group(1))
            except Exception:
                required = None

    # Extract plausible item names: quoted strings and Title Case multi-words
    tl = "\n".join((t or "") for t in texts)
    quoted = set(_re.findall(r'"([^"\n]{2,80})"', tl))
    # Title-case sequences 1–4 words (e.g., movie titles)
    titles = set(
        _re.findall(r"\b([A-Z][A-Za-z0-9'&.-]+(?:\s+[A-Z][A-Za-z0-9'&.-]+){0,3})\b", tl)
    )
    items = {s.strip() for s in quoted | titles if s.strip()}

    has_count = len(items)
    missing: list[str] = []
    if needs_list and required is not None and has_count < required:
        missing.append("list_items")

    return {
        "needs_list": needs_list,
        "required_count": required,
        "has_count": has_count,
        "missing": missing,
    }
