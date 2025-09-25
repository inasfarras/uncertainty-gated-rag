"""Thin, deterministic anchor predictor.

Uses existing question anchor utilities and judge helpers (when available)
to propose candidate anchors with simple confidence scores.
"""

from __future__ import annotations

from agentic_rag.agent.qanchors import (
    extract_required_anchors as q_extract_required_anchors,
)


def propose_anchors(question: str, top_m: int = 6) -> list[dict]:
    """Propose anchors from the question with heuristic confidence.

    Returns a list of dicts: {text, kind, score} sorted by score desc.
    Deterministic and cheap by default.
    """
    q = (question or "").strip()
    raw = list(q_extract_required_anchors(q))

    out: list[dict] = []
    for a in raw:
        kind = _classify_anchor(a)
        score = _anchor_confidence(a, kind, question=q)
        out.append({"text": a, "kind": kind, "score": float(score)})

    out.sort(key=lambda d: d.get("score", 0.0), reverse=True)
    return out[: max(1, top_m)]


def _classify_anchor(a: str) -> str:
    al = (a or "").lower()
    if al.isdigit() and len(al) == 4 and (al.startswith("19") or al.startswith("20")):
        return "year"
    if al in {"q1", "q2", "q3", "q4"}:
        return "quarter"
    if al in {"%", "percent", "per game", "ex-dividend"}:
        return "unit"
    # fallback
    return "entity"


def _anchor_confidence(a: str, kind: str, question: str) -> float:
    # Very cheap heuristic scoring that favors exact matches and temporal/unit cues
    base = 0.55
    if kind in {"year", "quarter", "unit"}:
        base += 0.2
    if (a or "").lower() in (question or "").lower():
        base += 0.1
    return min(0.95, base)
