from __future__ import annotations

import re
from typing import Iterable

from .types import Intent

_UNIT_ALIASES: dict[str, str] = {
    "per game": "per game",
    "ppg": "per game",
    "usd": "USD",
    "$": "USD",
    "%": "%",
    "percent": "%",
    "km/h": "km/h",
}

_AWARD_ALIASES: dict[str, str] = {
    "oscar": "Academy Awards",
    "oscars": "Academy Awards",
    "academy awards": "Academy Awards",
    "academy award": "Academy Awards",
}

_AWARD_CATEGORY_PATTERNS: tuple[str, ...] = (
    "Best Animated Feature",
    "Best Picture",
    "Best Director",
    "Best Actor",
    "Best Actress",
    "Best Supporting Actor",
    "Best Supporting Actress",
    "Best Documentary Feature",
    "Best Original Song",
)

_DIVISION_PATTERNS: tuple[str, ...] = (
    "men's singles",
    "women's singles",
    "men's final",
    "women's final",
    "men's doubles",
    "women's doubles",
)

_EVENT_TOKENS: tuple[str, ...] = (
    "men's singles",
    "women's singles",
    "final",
    "grand slam",
    "wimbledon",
    "us open",
    "u.s. open",
    "australian open",
    "french open",
    "best animated feature",
    "academy awards",
)

_STOPWORDS: set[str] = {
    "what",
    "who",
    "which",
    "where",
    "when",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "did",
    "does",
    "do",
    "the",
    "a",
    "an",
    "in",
    "for",
    "to",
    "of",
    "and",
    "at",
    "on",
    "by",
    "season",
    "seasons",
    "tournament",
    "year",
    "years",
    "average",
    "averages",
    "per",
    "game",
    "list",
    "compare",
    "versus",
    "vs",
}

_SEASON_WORDS: tuple[str, ...] = (
    "season",
    "campaign",
    "year",
    "run",
)


def _dedup(seq: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        key = item.strip()
        if not key:
            continue
        norm = key.lower()
        if norm not in seen:
            seen.add(norm)
            out.append(key)
    return out


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _normalize_season(year: int) -> str:
    end_two = str(year % 100).zfill(2)
    return f"{year - 1}\u2013{end_two}"


def _extract_years(question: str) -> tuple[str | None, str | None]:
    ql = question.lower()
    year_match = re.search(r"\b(19|20)\d{2}\b", ql)
    year = year_match.group(0) if year_match else None

    range_match = re.search(
        r"\b((?:19|20)\d{2})\s*[-\u2013]\s*((?:\d{2})|(?:19|20)\d{2})\b",
        ql,
    )
    time_window: str | None = None
    if range_match:
        start_year = int(range_match.group(1))
        end_fragment = range_match.group(2)
        if len(end_fragment) == 2:
            end_year = int(str(start_year)[:2] + end_fragment)
        else:
            end_year = int(end_fragment)
        time_window = f"{start_year}\u2013{str(end_year)[-2:]}"
    else:
        season_match = re.search(
            r"\b((?:19|20)\d{2})\b\s*(?:/\d{2})?\s*(?:"
            + "|".join(_SEASON_WORDS)
            + r")",
            ql,
        )
        if season_match:
            season_year = int(season_match.group(1))
            time_window = _normalize_season(season_year)
        else:
            nba_match = re.search(r"nba\s+((?:19|20)\d{2})", ql)
            if nba_match:
                season_year = int(nba_match.group(1))
                time_window = _normalize_season(season_year)

    quarter_match = re.search(r"\bq([1-4])\s*(?:of\s*)?((?:19|20)\d{2})\b", ql)
    if quarter_match:
        quarter = quarter_match.group(1)
        year = quarter_match.group(2)
        time_window = f"Q{quarter} {year}"

    if time_window and not year and re.match(r"\d{4}", time_window):
        year = time_window[:4]

    return year, time_window


def _extract_unit(question: str) -> str | None:
    ql = question.lower()
    for token, norm in _UNIT_ALIASES.items():
        if token in ql:
            return norm
    return None


def _extract_category(question: str) -> str | None:
    for cat in _AWARD_CATEGORY_PATTERNS:
        if cat.lower() in question.lower():
            return cat
    best_match = re.search(r"Best\s+[A-Z][A-Za-z\s]+", question)
    if best_match:
        return _normalize_text(best_match.group(0))
    return None


def _extract_division(question: str) -> str | None:
    ql = question.lower()
    for div in _DIVISION_PATTERNS:
        if div in ql:
            return div
    return None


def _extract_core_entities(question: str) -> list[str]:
    entities: list[str] = []
    for match in re.finditer(
        r"\b([A-Z][A-Za-z0-9'\-]*(?:\s+[A-Z][A-Za-z0-9'\-]*)*)\b",
        question,
    ):
        phrase = match.group(1).strip()
        if not phrase:
            continue
        if phrase.lower() in _STOPWORDS:
            continue
        entities.append(phrase)
    ql = question.lower()
    for alias, canonical in _AWARD_ALIASES.items():
        if alias in ql:
            entities.append(canonical)
    for token in _EVENT_TOKENS:
        if token in ql:
            entities.append(_normalize_text(token.title()))
    return _dedup(entities)


def _infer_task_type(question: str) -> str:
    ql = question.lower()
    if "why" in ql:
        return "why"
    if re.search(r"\bcompare|difference|versus|vs\b", ql):
        return "compare"
    if re.search(r"\blist\b", ql) or "what are" in ql:
        return "list"
    if re.search(r"\bdefine|definition of\b", ql):
        return "definition"
    return "factoid"


def _expected_slots(question: str, task_type: str, slots: dict[str, str]) -> list[str]:
    ql = question.lower()
    expected: list[str] = []
    numeric_cues = any(
        tok in ql
        for tok in [
            " per game",
            " per",
            "ppg",
            "%",
            "percent",
            "km/h",
            "rate",
            "average",
        ]
    ) or bool(re.search(r"\b\d+%\b|\b\d+\.\d+\b", ql))
    award_cues = any(alias in ql for alias in _AWARD_ALIASES.keys())
    category_present = any(cat.lower() in ql for cat in _AWARD_CATEGORY_PATTERNS)
    division_present = any(div in ql for div in _DIVISION_PATTERNS)

    if task_type in {"factoid", "compare"} and numeric_cues:
        expected.extend(["unit", "time_window"])
    if award_cues or category_present:
        expected.extend(["year", "category"])
    if division_present or "final" in ql:
        expected.extend(["year", "division"])
    if "tournament" in ql or "open" in ql or "championship" in ql:
        if "year" not in expected:
            expected.append("year")
    if task_type == "list" and not expected:
        expected.append("canonical_query")
    if not expected and slots:
        expected.append(next(iter(slots.keys())))
    return expected


def _compute_slot_completeness(expected: list[str], slots: dict[str, str]) -> float:
    if not expected:
        return 1.0 if slots else 0.5
    covered = sum(1 for key in expected if key in slots and slots[key])
    return min(1.0, covered / len(set(expected)))


def _compute_intent_confidence(
    core_entities: list[str],
    slots: dict[str, str],
    slot_completeness: float,
    question: str,
) -> float:
    ql = question.lower()
    confidence = 0.3
    if core_entities:
        confidence += 0.2
    if any(key in slots for key in ("year", "time_window")):
        confidence += 0.15
    if "unit" in slots:
        confidence += 0.15
    if "category" in slots or "division" in slots:
        confidence += 0.1
    if "per" in ql or "%" in ql or "ppg" in ql:
        confidence += 0.05
    confidence += 0.25 * slot_completeness
    return max(0.0, min(1.0, confidence))


def interpret_with_rules(question: str) -> Intent:
    question = question or ""
    core_entities = _extract_core_entities(question)
    year, time_window = _extract_years(question)
    unit = _extract_unit(question)
    category = _extract_category(question)
    division = _extract_division(question)

    slots: dict[str, str] = {}
    if year:
        slots["year"] = year
    if time_window:
        slots["time_window"] = time_window
    if unit:
        slots["unit"] = unit
    if category:
        slots["category"] = category
    if division:
        slots["division"] = division

    task_type = _infer_task_type(question)
    expected = _expected_slots(question, task_type, slots)
    slot_completeness = _compute_slot_completeness(expected, slots)
    intent_confidence = _compute_intent_confidence(
        core_entities, slots, slot_completeness, question
    )

    ambiguity_flags: list[str] = []
    if not core_entities:
        ambiguity_flags.append("missing_core_entities")
    missing_slots = [key for key in set(expected) if key not in slots or not slots[key]]
    for key in missing_slots:
        ambiguity_flags.append(f"missing_{key}")

    canonical_query = _normalize_text(question)

    return Intent(
        task_type=task_type,
        core_entities=core_entities,
        slots=slots,
        canonical_query=canonical_query,
        ambiguity_flags=ambiguity_flags,
        intent_confidence=intent_confidence,
        slot_completeness=slot_completeness,
        source_of_intent="rule_only",
    )
