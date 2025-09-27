"""Thin, deterministic anchor predictor informed by intent signals."""

from __future__ import annotations

from agentic_rag.agent.qanchors import (
    extract_required_anchors as q_extract_required_anchors,
)
from agentic_rag.intent.types import Intent


def _push_candidate(
    collection: list[dict[str, float | str]],
    seen: set[str],
    text: str,
    kind: str,
    base_score: float,
    bonus: float,
) -> None:
    clean = (text or "").strip()
    if not clean:
        return
    norm = f"{clean.lower()}::{kind}"
    if norm in seen:
        return
    seen.add(norm)
    score = min(0.99, max(0.2, base_score + bonus))
    collection.append({"text": clean, "type": kind, "score": score})


def propose_anchors(intent: Intent, top_m: int = 6) -> list[dict]:
    """Propose anchors from an intent object with heuristic confidence."""

    anchors: list[dict[str, float | str]] = []
    seen: set[str] = set()
    bonus = intent.intent_confidence * 0.2

    slot_kind_priority = {
        "year": (0.85, "year"),
        "time_window": (0.82, "time_window"),
        "unit": (0.75, "unit"),
        "category": (0.78, "category"),
        "division": (0.76, "division"),
    }

    for key, value in intent.slots.items():
        base, kind = slot_kind_priority.get(key, (0.72, key))
        _push_candidate(anchors, seen, value, kind, base, bonus)

    for idx, entity in enumerate(intent.core_entities):
        base = 0.78 - 0.04 * idx
        _push_candidate(anchors, seen, entity, "entity", base, bonus)

    canonical_query = intent.canonical_query or ""
    if canonical_query:
        for token in q_extract_required_anchors(canonical_query):
            _push_candidate(anchors, seen, token, "heuristic", 0.68, bonus / 2)

    if not anchors and canonical_query:
        _push_candidate(anchors, seen, canonical_query, "question", 0.55, bonus)

    anchors.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
    return anchors[: max(1, top_m)]
