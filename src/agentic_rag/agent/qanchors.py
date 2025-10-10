"""Lightweight question anchor utilities.

These helpers centralize anchor extraction and presence checks with
simple heuristics tuned for factoid-style CRAG questions. They reuse
Judge utilities when available to avoid duplication.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

try:
    # Prefer Judge versions if present
    from agentic_rag.agent.judge import (
        anchors_present_in_texts as _judge_anchors_present_in_texts,  # type: ignore
    )
    from agentic_rag.agent.judge import (
        extract_required_anchors as _judge_extract_required_anchors,  # type: ignore
    )
except Exception:  # pragma: no cover
    _judge_anchors_present_in_texts = None  # type: ignore
    _judge_extract_required_anchors = None  # type: ignore


def extract_required_anchors(q: str) -> set[str]:
    """Extract simple temporal/units/event/entity anchors from a question."""
    if _judge_extract_required_anchors is not None:  # reuse richer logic
        try:
            return set(_judge_extract_required_anchors(q))  # type: ignore[arg-type]
        except Exception:
            pass

    ql = (q or "").lower()
    anchors: set[str] = set()
    # years
    anchors.update(re.findall(r"\b(?:19|20)\d{2}\b", ql))
    # scoreline triples like 50-40-90 or 50/40/90
    anchors.update(re.findall(r"\b\d{2}[-\/]\d{2}[-\/]\d{2}\b", ql))
    # quarters / months
    anchors.update(re.findall(r"\bq[1-4]\b", ql))
    anchors.update(
        re.findall(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b", ql)
    )
    # units/time phrases
    for tok in [
        "per game",
        "ex-dividend",
        "average",
        "%",
        "percent",
        "q1",
        "q2",
        "q3",
        "q4",
    ]:
        if tok in ql:
            anchors.add(tok)
    # common events/categories
    for tok in [
        "oscar",
        "academy award",
        "best animated feature",
        "grand slam",
        "u.s. open",
        "us open",
        "wimbledon",
        "french open",
    ]:
        if tok in ql:
            anchors.add(tok)
    # crude two-word entity phrase when question is lowercase (e.g., "steve nash")
    # UPDATED: Use capitalized words from original question as a better heuristic
    try:
        caps = re.findall(
            r"\b([A-Z][A-Za-z0-9'&.-]+(?:\s+[A-Z][A-Za-z0-9'&.-]+){0,3})\b", q or ""
        )
        if caps:
            # Add longest capitalized span as a candidate anchor
            anchors.add(max(caps, key=len).lower())
    except Exception:
        pass
    return anchors


def anchors_present_in_texts(texts: Iterable[str], anchors: set[str]) -> set[str]:
    """Return subset of anchors present across any of the texts."""
    if _judge_anchors_present_in_texts is not None:
        try:
            present, _ = _judge_anchors_present_in_texts(set(anchors), list(texts))  # type: ignore
            return present
        except Exception:
            pass
    present_anchors: set[str] = set()
    tl = "\n".join((t or "").lower() for t in texts)
    for a in anchors:
        if a.lower() in tl:
            present_anchors.add(a)
    return present_anchors


def is_factoid(q: str) -> bool:
    ql = (q or "").lower()
    if any(
        t in ql
        for t in [
            "how many",
            "how much",
            "what year",
            "when",
            "date",
            "per game",
            "ex-dividend",
        ]
    ):
        return True
    if re.search(r"\b(19|20)\d{2}\b", ql):
        return True
    if re.search(r"\bwhich\b|\bwho\b|\bwhat\b", ql):
        return True
    return False
