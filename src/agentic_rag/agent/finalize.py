import re

from agentic_rag.eval.metrics import extract_short_answer  # may be used as fallback


def detect_type(question: str) -> str:
    """Heuristically detect the expected answer type.

    Prioritize entity-type questions that begin with which/who/what-entity
    even if the question also contains a year. This avoids extracting a year
    as the short answer for prompts like "Which film ... in 2021?".
    """
    q = (question or "").lower()

    has_entity_cue = bool(
        re.search(
            r"\bwhich\b|\bwho\b|\bwhat (movie|film|team|player|country|album|song|city|company|award)\b",
            q,
        )
    )
    has_number_cue = bool(
        re.search(r"\bhow many\b|\bhow long\b|\baverage\b|%|\bper game\b|\$\d", q)
    )
    has_date_cue = bool(
        re.search(r"\bwhen\b|\bdate\b|\bq\d\s*20\d{2}\b", q)
        or re.search(r"\b(19|20)\d{2}\b", q)
    )

    if has_entity_cue:
        return "entity"
    if has_number_cue:
        return "number"
    if has_date_cue:
        return "date"
    return "other"


_DATE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},\s*\d{4}\b|"
    r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{4}\b|"
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b|\b\d{4}\b",
    re.I,
)
_MONEY = re.compile(r"\$[\d,]+(?:\.\d+)?\b")
_NUM = re.compile(r"\b\d+(?:\.\d+)?\b")
_QUOTE_RE = re.compile(r"[\"\u201c\u201d\u201e\u201f\'\u2018\u2019](.+?)[\"\u201c\u201d\u201e\u201f\'\u2018\u2019]")
_NAME_SPAN_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&'\u2019\-]*(?:\s+[A-Z][A-Za-z0-9&'\u2019\-]*){0,4})\b"
)
_UNIT_WORDS = {
    "billion",
    "million",
    "thousand",
    "trillion",
    "percent",
    "percentage",
    "points",
    "games",
    "wins",
    "losses",
    "times",
    "days",
    "years",
    "weeks",
    "months",
    "people",
    "companies",
    "songs",
    "albums",
    "episodes",
    "cents",
    "dollars",
    "shares",
    "players",
    "teams",
}
_ENTITY_PENALTY_WORDS = {
    "best",
    "visual",
    "effects",
    "feature",
    "animated",
    "award",
    "category",
    "motion",
    "picture",
    "event",
    "season",
    "grand",
    "slam",
    "championship",
    "average",
}
_APPROX_VALUE_RE = re.compile(
    r"(?:~|approximately|about)\s*(\$[\d,]+(?:\.\d+)?|\d+(?:\.\d+)?)",
    re.I,
)


def _strip_punct(text: str) -> str:
    return text.strip(" 	\r\n.,;:!?()[]{}\"'")

def _extend_with_unit(answer: str, match: re.Match[str]) -> str:
    span = match.group(0)
    tail = answer[match.end() :]
    extras: list[str] = []
    tokens = list(re.finditer(r"\s*(\S+)", tail))
    idx = 0
    while idx < len(tokens):
        token = _strip_punct(tokens[idx].group(1))
        if not token:
            idx += 1
            continue
        lower = token.lower()
        if lower == "%":
            if not span.endswith("%"):
                span = f"{span}%"
            idx += 1
            continue
        if lower in _UNIT_WORDS:
            extras.append(token)
            idx += 1
            continue
        if lower == "per" and idx + 1 < len(tokens):
            next_tok = _strip_punct(tokens[idx + 1].group(1))
            if next_tok:
                extras.append(token)
                extras.append(next_tok)
            idx += 2
            break
        break
    if extras:
        span = f"{span} {' '.join(extras)}".strip()
    return _strip_punct(span)


def _collect_entity_candidates(answer: str) -> list[tuple[str, bool]]:
    cand_pairs: list[tuple[str, bool]] = []
    for match in _QUOTE_RE.finditer(answer):
        candidate = _strip_punct(match.group(1))
        if not candidate or len(candidate) <= 1:
            continue
        cand_pairs.append((candidate, True))

    for pattern in (
        r"\b(?:is|was|were|are|won|wins|goes to|belongs to)\s+(?:an?\s+|the\s+)?(?P<entity>[A-Z][A-Za-z0-9&'\u2019\-]*(?:\s+[A-Z][A-Za-z0-9&'\u2019\-]*){0,4})",
    ):
        for match in re.finditer(pattern, answer):
            candidate = _strip_punct(match.group("entity"))
            if not candidate or len(candidate) <= 1:
                continue
            cand_pairs.append((candidate, False))

    for match in _NAME_SPAN_RE.finditer(answer):
        candidate = _strip_punct(match.group(0))
        if not candidate or len(candidate) <= 1:
            continue
        cand_pairs.append((candidate, False))

    seen: set[str] = set()
    ordered: list[tuple[str, bool]] = []
    for candidate, from_quote in cand_pairs:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append((candidate, from_quote))
    return ordered
def _choose_entity_answer(answer: str, question: str) -> str | None:
    candidates = _collect_entity_candidates(answer)
    if not candidates:
        return None
    q_lower = (question or "").lower()
    best: str | None = None
    best_score = float("-inf")
    for idx, (candidate, from_quote) in enumerate(candidates):
        tokens = candidate.split()
        if not tokens:
            continue
        score = 0.0
        if from_quote:
            score += 4.0
        length = len(tokens)
        if 1 <= length <= 4:
            score += 3.0
        elif length == 1:
            score += 2.5
        else:
            score += max(1.0, 4 - 0.5 * (length - 4))
        penalty_hits = sum(1 for t in tokens if t.lower() in _ENTITY_PENALTY_WORDS)
        score -= penalty_hits * 2.5
        if candidate.lower() in q_lower:
            score -= 3.0
        if tokens and tokens[0].lower() == "the" and length > 1:
            score -= 0.6
        if tokens and tokens[-1].lower() in {"award", "category"}:
            score -= 1.5
        if any(ch.isdigit() for ch in candidate):
            score -= 0.5
        score -= 0.01 * len(candidate)
        score += 0.05 * idx
        if score > best_score:
            best_score = score
            best = candidate
    return best


def finalize_short_answer(question: str, answer: str) -> str | None:
    """Extract a minimal span suitable for EM/F1 scoring.

    Uses the question type to prioritize extraction. Falls back to the
    generic extractor if needed.
    """
    if not answer:
        return None
    a = re.sub(r"\[CIT:[^\]]+\]", "", answer or "").strip()
    a = re.sub(r"\s+", " ", a)
    qtype = detect_type(question)

    if qtype == "date":
        m = _DATE.search(a)
        if m:
            return _strip_punct(m.group(0))

    if qtype == "number":
        approx = _APPROX_VALUE_RE.search(a)
        if approx:
            value = approx.group(1)
            base_match = re.search(re.escape(value), a)
            if base_match:
                return _extend_with_unit(a, base_match)
        if "average" in (question or "").lower():
            monies = list(re.finditer(r"\$[\d,]+(?:\.\d+)?", a))
            if monies:
                return _extend_with_unit(a, monies[-1])
        money_match = _MONEY.search(a)
        if money_match:
            return _extend_with_unit(a, money_match)
        num_match = _NUM.search(a)
        if num_match:
            return _extend_with_unit(a, num_match)

    if qtype == "entity":
        entity = _choose_entity_answer(a, question)
        if entity:
            return entity

    try:
        short = extract_short_answer(a)
        return short if short and len(short.strip()) > 1 else None
    except Exception:
        return None
