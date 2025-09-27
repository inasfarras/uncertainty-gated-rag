from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import Any

from agentic_rag.config import settings
from agentic_rag.models.adapter import ChatMessage, get_openai

from .types import Intent

_SYSTEM_PROMPT = (
    "You are a deterministic question interpreter for a RAG system. "
    'Output Intent as strict JSON only (no prose). Normalize seasons (e.g., NBA 2017 -> "2016\\u201317") '
    "and award names (Oscar <-> Academy Awards). When uncertain, set ambiguity_flags and keep conservative values."
)

_TEMPLATE = (
    "{\n"
    '  "task_type": "factoid|list|compare|definition|why",\n'
    '  "core_entities": ["..."],\n'
    '  "slots": {"year":"...", "unit":"...", "category":"...", "division":"...", "time_window":"..."},\n'
    '  "canonical_query": "...",\n'
    '  "ambiguity_flags": [],\n'
    '  "intent_confidence": 0.0,\n'
    '  "slot_completeness": 0.0\n'
    "}"
)

_ALLOWED_TASK_TYPES = {"factoid", "list", "compare", "definition", "why"}


def _build_messages(question: str, hint: Intent | None) -> list[ChatMessage]:
    hint_json = json.dumps(asdict(hint)) if hint else "{}"
    user_prompt = (
        f'Question: "{question}"\n'
        "Optional rule hints:\n"
        f"{hint_json}\n"
        "Return JSON exactly:\n"
        f"{_TEMPLATE}"
    )
    return [
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt),
    ]


def _coerce_slots(slots: Any) -> dict[str, str]:
    if not isinstance(slots, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in slots.items():
        try:
            key_str = str(key)
            value_str = str(value)
        except Exception:
            continue
        if key_str and value_str:
            out[key_str] = value_str
    return out


def _coerce_entities(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        try:
            text = str(item).strip()
        except Exception:
            continue
        if not text:
            continue
        norm = text.lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(text)
    return out


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _sanitize_payload(payload: dict[str, Any]) -> Intent:
    task_type = str(payload.get("task_type", "factoid"))
    if task_type not in _ALLOWED_TASK_TYPES:
        task_type = "factoid"

    core_entities = _coerce_entities(payload.get("core_entities", []))
    slots = _coerce_slots(payload.get("slots", {}))
    canonical_query = str(payload.get("canonical_query", "") or "")
    ambiguity_flags = _coerce_entities(payload.get("ambiguity_flags", []))
    intent_confidence = _clamp(payload.get("intent_confidence", 0.0))
    slot_completeness = _clamp(payload.get("slot_completeness", 0.0))

    return Intent(
        task_type=task_type,
        core_entities=core_entities,
        slots=slots,
        canonical_query=canonical_query,
        ambiguity_flags=ambiguity_flags,
        intent_confidence=intent_confidence,
        slot_completeness=slot_completeness,
        source_of_intent="llm_only",
    )


def interpret_with_llm(question: str, hint: Intent | None = None) -> Intent:
    messages = _build_messages(question or "", hint)
    adapter = get_openai()

    try:
        raw, _ = adapter.chat(
            messages=messages,
            model=settings.LLM_MODEL,
            max_tokens=min(256, settings.MAX_OUTPUT_TOKENS),
            temperature=0.0,
        )
    except Exception:
        base = (
            replace(hint, ambiguity_flags=list(hint.ambiguity_flags))
            if hint
            else Intent()
        )
        flags = list(base.ambiguity_flags)
        flags.append("llm_call_failed")
        base.ambiguity_flags = flags
        return base

    raw = raw.strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        base = (
            replace(hint, ambiguity_flags=list(hint.ambiguity_flags))
            if hint
            else Intent()
        )
        flags = list(base.ambiguity_flags)
        flags.append("invalid_llm_json")
        base.ambiguity_flags = flags
        return base

    if not isinstance(payload, dict):
        base = (
            replace(hint, ambiguity_flags=list(hint.ambiguity_flags))
            if hint
            else Intent()
        )
        flags = list(base.ambiguity_flags)
        flags.append("invalid_llm_json")
        base.ambiguity_flags = flags
        return base

    intent = _sanitize_payload(payload)
    if not intent.canonical_query:
        intent.canonical_query = question.strip()
    return intent
