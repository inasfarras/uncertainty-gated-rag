import re

from agentic_rag.eval.metrics import extract_short_answer  # may be used as fallback


def detect_type(question: str) -> str:
    q = (question or "").lower()
    if re.search(r"\bwhen\b|\bdate\b|\bq\d\s*20\d{2}\b|\b20\d{2}\b", q):
        return "date"
    if re.search(r"\bhow many\b|\bhow long\b|\baverage\b|%|\bper game\b|\$\d", q):
        return "number"
    if re.search(r"\bwhich\b|\bwho\b|\bwhat (movie|film|team|player|country)\b", q):
        return "entity"
    return "other"


_DATE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},\s*\d{4}\b|"
    r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{4}\b|"
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b|\b\d{4}\b",
    re.I,
)
_MONEY = re.compile(r"\$[\d,]+(?:\.\d+)?\b")
_NUM = re.compile(r"\b\d+(?:\.\d+)?\b")
_TITLE = re.compile(r"([A-Z][A-Za-z0-9&'’\-]*(?:\s+[A-Z][A-Za-z0-9&'’\-]*){0,4})")


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
            return m.group(0).strip()

    if qtype == "number":
        m = _MONEY.search(a) or _NUM.search(a)
        if m:
            return m.group(0).strip()

    if qtype == "entity":
        # Prefer 2-4 word capitalized entities; skip pronouns/generic tokens
        non_entity_words = {
            "the",
            "a",
            "an",
            "of",
            "in",
            "on",
            "and",
            "for",
            "with",
            "is",
            "was",
            "are",
            "were",
            "i",
            "yes",
            "no",
            "it",
            "he",
            "she",
            "they",
            "we",
            "this",
            "that",
        }
        candidates = [
            m.group(0).strip()
            for m in re.finditer(r"\b([A-Z][a-zA-Z0-9\-]*\s*)+[A-Z][a-zA-Z0-9\-]*\b", a)
        ]
        # Prefer multi-word entities (2-4 words)
        for c in candidates:
            if 2 <= len(c.split()) <= 4:
                return c
        # Fall back to shortest single capitalized token not in stoplist
        singles = [c for c in candidates if len(c.split()) == 1]
        singles = [
            c for c in singles if c.lower() not in non_entity_words and len(c) > 1
        ]
        if singles:
            return min(singles, key=len)

    # Fallback to generic extractor as a last resort
    try:
        short = extract_short_answer(a)
        # Guard against degenerate single-letter outputs
        return short if short and len(short.strip()) > 1 else None
    except Exception:
        return None
