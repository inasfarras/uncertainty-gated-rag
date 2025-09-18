import re


def extract_short_answer(text: str) -> str:
    """Extracts a short answer from the model's generation for scoring."""
    if not text:
        return ""

    # 1. Strip citations
    text = re.sub(r"\[CIT:[^\]]+\]", "", text).strip()

    # 2. Lowercase and strip punctuation
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)

    # 3. Simple noun phrase/title/number extraction (add more rules as needed)
    # Example: "The movie that won... is 'Tenet'." -> "tenet"
    match = re.search(r"\b(is|was|are|were)\s+(?:the\s+)?(?:an?\s+)?([\w\s]+)", text)
    if match:
        return match.group(2).strip()

    # If it's a number (e.g., for the Steve Nash question)
    match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if match:
        return match.group(1)

    return text
