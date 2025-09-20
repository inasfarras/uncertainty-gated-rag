import re


def extract_short_answer(text: str) -> str:
    """Extracts a short answer from the model's generation for scoring."""
    if not text:
        return ""

    # 1. Strip citations and normalize whitespace
    text = re.sub(r"\[CIT:[^\]]+\]", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()

    # 2. Look for specific types of answers (prioritized)
    # Dates (e.g., "July 28, 1949", "1994")
    date_patterns = [
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},\s*\d{4}\b",  # Month Day, Year
        r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{4}\b",  # Day Month Year
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",  # YYYY-MM-DD or YYYY/MM/DD
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b",  # MM-DD-YYYY or MM/DD/YYYY
        r"\b\d{4}\b",  # Year only
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(0).strip()

    # Numbers (e.g., "305 miles", "9 titles", "$123.45")
    number_patterns = [
        r"\$\d[\d,]*\.?\d*\b",  # Money like $123,456.78
        r"\b\d+[.,]?\d*\s*(?:miles|hat tricks|titles|games|years|km|%)?\b",  # Numbers with optional units
    ]
    for pattern in number_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(0).strip()

    # Entities (Capitalized words/phrases) - focus on shorter ones
    # Only extract if it's not a generic word like 'the', 'a', 'in', etc.
    entity_candidates = [
        m.group(0).strip()
        for m in re.finditer(r"\b([A-Z][a-zA-Z0-9-]*\s*)+[A-Z][a-zA-Z0-9-]*\b", text)
        if len(m.group(0).split()) <= 4
    ]
    if entity_candidates:
        # Filter out common, non-entity words
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
        }
        filtered_candidates = [
            c
            for c in entity_candidates
            if c.lower() not in non_entity_words
            and not all(word.lower() in non_entity_words for word in c.split())
        ]
        if filtered_candidates:
            return min(filtered_candidates, key=len)

    # Fallback to a cleaned version of the original text
    return text.lower().strip()
