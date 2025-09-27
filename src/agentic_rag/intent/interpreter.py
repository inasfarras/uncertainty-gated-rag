from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from .llm import interpret_with_llm
from .rules import interpret_with_rules
from .types import Intent


def _dedup(seq: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        text = (item or "").strip()
        if not text:
            continue
        norm = text.lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(text)
    return out


def _normalize_season(year: int) -> str:
    return f"{year - 1}\u2013{str(year % 100).zfill(2)}"


def _normalize_slot_value(key: str, value: str) -> str:
    text = (value or "").strip()
    if not text:
        return text
    if key == "time_window":
        cleaned = re.sub(r"\s*[-\u2013]\s*", "\u2013", text)
        if re.fullmatch(r"(19|20)\d{2}", cleaned):
            season_year = int(cleaned)
            return _normalize_season(season_year)
        match = re.search(r"(19|20)\d{2}", cleaned)
        if match and ("season" in text.lower() or "nba" in text.lower()):
            return _normalize_season(int(match.group(0)))
        return cleaned
    if key == "year":
        match = re.search(r"(19|20)\d{2}", text)
        if match:
            return match.group(0)
    return text


def _prefer_slot_value(key: str, primary: str, secondary: str) -> str:
    if key == "time_window":
        score_primary = ("\u2013" in primary) + (len(primary) > 6)
        score_secondary = ("\u2013" in secondary) + (len(secondary) > 6)
        return secondary if score_secondary > score_primary else primary
    if len(secondary) > len(primary):
        return secondary
    return primary


def merge_rule_llm(rule: Intent, llm: Intent) -> Intent:
    failure_flags = {"invalid_llm_json", "llm_call_failed"}
    if any(flag in llm.ambiguity_flags for flag in failure_flags):
        merged_flags = _dedup(list(rule.ambiguity_flags) + list(llm.ambiguity_flags))
        return replace(rule, ambiguity_flags=merged_flags)

    task_type = rule.task_type
    if task_type in {"", "unknown"} and llm.task_type:
        task_type = llm.task_type

    if llm.intent_confidence > rule.intent_confidence:
        core_entities = _dedup(llm.core_entities + rule.core_entities)
    else:
        core_entities = _dedup(rule.core_entities + llm.core_entities)

    merged_slots: dict[str, str] = dict(rule.slots)
    for key, value in llm.slots.items():
        normalized = _normalize_slot_value(key, value)
        if key not in merged_slots or not merged_slots[key]:
            merged_slots[key] = normalized
            continue
        existing = _normalize_slot_value(key, merged_slots[key])
        if existing != normalized:
            merged_slots[key] = _prefer_slot_value(key, existing, normalized)
        else:
            merged_slots[key] = existing

    canonical_query = llm.canonical_query.strip() or rule.canonical_query.strip()

    combined_flags = _dedup(list(rule.ambiguity_flags) + list(llm.ambiguity_flags))
    weighted = 0.6 * llm.intent_confidence + 0.4 * rule.intent_confidence
    intent_confidence = min(
        1.0, max(weighted, rule.intent_confidence, llm.intent_confidence)
    )
    slot_completeness = max(rule.slot_completeness, llm.slot_completeness)

    return Intent(
        task_type=task_type or "factoid",
        core_entities=core_entities,
        slots=merged_slots,
        canonical_query=canonical_query,
        ambiguity_flags=combined_flags,
        intent_confidence=intent_confidence,
        slot_completeness=slot_completeness,
        source_of_intent="llm_fallback",
    )


def interpret(question: str, llm_budget_ok: bool = True) -> Intent:
    rule_intent = interpret_with_rules(question)
    if rule_intent.slot_completeness >= 0.8 and rule_intent.intent_confidence >= 0.7:
        return replace(rule_intent, source_of_intent="rule_only")
    if not llm_budget_ok:
        flags = _dedup(list(rule_intent.ambiguity_flags) + ["llm_skipped_budget"])
        return replace(rule_intent, ambiguity_flags=flags, source_of_intent="rule_only")
    llm_intent = interpret_with_llm(question, hint=rule_intent)
    if llm_intent.source_of_intent != "llm_only":
        flags = _dedup(
            list(rule_intent.ambiguity_flags) + list(llm_intent.ambiguity_flags)
        )
        return replace(rule_intent, ambiguity_flags=flags, source_of_intent="rule_only")
    return merge_rule_llm(rule_intent, llm_intent)
