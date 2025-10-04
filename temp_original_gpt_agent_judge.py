"""GPT-powered judge for agent answers with auto-metrics blending."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from openai import OpenAI, OpenAIError

from agentic_rag.config import settings
from agentic_rag.eval.signals import em_f1, is_idk, split_sentences

from .rubrics import PROMPT_TEMPLATE, RUBRIC_GOLD_AWARE

LOGGER = logging.getLogger(__name__)

__all__ = [
    "judge_example",
    "judge_batch",
    "extract_citation_ids",
    "compute_support_overlap",
    "detect_idk",
]


@dataclass
class _ClientCache:
    client: OpenAI | None = None


_CLIENT_CACHE = _ClientCache()


def _get_client() -> OpenAI:
    if _CLIENT_CACHE.client is None:
        base_url = settings.OPENAI_BASE_URL or "https://api.openai.com/v1"
        _CLIENT_CACHE.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=base_url,
            organization=settings.OPENAI_ORG,
        )
    return _CLIENT_CACHE.client


_NUMBER_RE = re.compile(r"\d+")
_CIT_MARKERS = (
    re.compile(r"\[(?P<body>[^\]]*\d[^\]]*)\]"),
    re.compile(r"\((?P<body>[^)]*?\d[^)]*?)\)"),
    re.compile(r"<(?P<body>[^>]*?\d[^>]*?)>"),
)
_INLINE_CIT_RE = re.compile(r"CIT[:_\-]?(?P<num>\d+)", re.IGNORECASE)
_CIT_ONLY_RE = re.compile(
    r"^(?:[\[(<]\s*(?:CIT[:_\-]?)?\d+(?:\s*,\s*\d+)*\s*[\])>]\s*)+$",
    re.IGNORECASE,
)


def extract_citation_ids(text: str) -> list[int]:
    """Extract numeric citation identifiers from free-form markup."""

    if not text:
        return []

    ids: list[int] = []
    for pattern in _CIT_MARKERS:
        for match in pattern.finditer(text):
            body = (match.group("body") or "").strip()
            if not body:
                continue
            numbers = _NUMBER_RE.findall(body)
            if not numbers:
                continue
            cleaned = re.sub(r"[0-9,\s]", "", body).lower()
            longest = max(len(n) for n in numbers)
            if cleaned and not cleaned.strip("cit:_-"):
                ids.extend(int(n) for n in numbers)
            elif not cleaned and longest <= 3:
                ids.extend(int(n) for n in numbers)
    ids.extend(int(m.group("num")) for m in _INLINE_CIT_RE.finditer(text))
    ids.extend(int(raw) for raw in re.findall(r"\(\(\s*(\d+)\s*\)\)", text))

    # De-duplicate preserving order
    seen: set[int] = set()
    ordered: list[int] = []
    for cid in ids:
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    return ordered


def _strip_citations(text: str) -> str:
    if not text:
        return ""
    stripped = text
    for pattern in _CIT_MARKERS:
        stripped = pattern.sub(lambda m: "" if re.search(r"\d", m.group("body")) else m.group(0), stripped)
    stripped = _INLINE_CIT_RE.sub("", stripped)
    return stripped.strip()


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    return set(re.findall(r"\b\w+\b", text.lower()))


def compute_support_overlap(answer: str, passages: dict[int, str]) -> float:
    """Compute token Jaccard between cited sentences and their passages."""

    if not answer or not passages:
        return 0.0

    sentences = split_sentences(answer)
    if not sentences:
        return 0.0

    merged: list[str] = []
    for sentence in sentences:
        stripped = sentence.strip()
        if merged and _CIT_ONLY_RE.match(stripped):
            merged[-1] = f"{merged[-1]} {stripped}".strip()
        else:
            merged.append(sentence)
    sentences = merged

    sent_tokens: set[str] = set()
    passage_tokens: set[str] = set()

    for sentence in sentences:
        ids = [cid for cid in extract_citation_ids(sentence) if cid in passages]
        if not ids:
            continue
        sent_tokens.update(_tokenize(_strip_citations(sentence)))
        for cid in ids:
            passage_tokens.update(_tokenize(passages[cid]))

    if not sent_tokens or not passage_tokens:
        return 0.0

    intersection = sent_tokens & passage_tokens
    union = sent_tokens | passage_tokens
    if not union:
        return 0.0
    return len(intersection) / len(union)


def detect_idk(answer: str) -> bool:
    return bool(answer) and is_idk(answer)


def _format_evidence(passages: dict[int, str]) -> str:
    if not passages:
        return "[]"
    ordered = [{"id": pid, "text": passages[pid]} for pid in sorted(passages)]
    return json.dumps(ordered, ensure_ascii=False, indent=2)


def _call_llm(prompt: str, *, model: str, temperature: float) -> dict[str, Any]:
    client = _get_client()
    system_prompt = RUBRIC_GOLD_AWARE
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except (OpenAIError, json.JSONDecodeError) as err:
            last_error = err
            LOGGER.warning("Judge LLM call failed (attempt %s/2): %s", attempt + 1, err)
    raise RuntimeError(f"Failed to obtain judge response: {last_error}")


def _prepare_passages(cited_passages: dict[int | str, str] | None) -> dict[int, str]:
    if not cited_passages:
        return {}
    normalized: dict[int, str] = {}
    for key, value in cited_passages.items():
        if value is None:
            continue
        try:
            cid = int(key)
        except (TypeError, ValueError):
            continue
        normalized[cid] = value
    return normalized


def _normalize_auto_metrics(answer: str, gold: str, passages: dict[int, str]) -> dict[str, Any]:
    clean_answer = _strip_citations(answer)
    em_metrics = em_f1(clean_answer, gold)
    has_citation = bool(extract_citation_ids(answer))
    support_overlap = compute_support_overlap(answer, passages)
    idk_flag = detect_idk(clean_answer)

    auto_metrics = {
        "em": float(em_metrics["em"]),
        "f1_short": float(em_metrics["f1"]),
        "has_citation": has_citation,
        "support_overlap": float(support_overlap),
        "idk_plus_cit_violation": bool(idk_flag and has_citation),
        "citation_required_violation": False,
        "num_citations": len(extract_citation_ids(answer)),
    }
    return auto_metrics


def _ensure_llm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    scores = payload.get("scores") or {}
    rationales = payload.get("rationales") or {}
    flags = payload.get("flags") or {}
    used = payload.get("used_citations") or []

    # Convert citation list to ints when possible
    normalized_used: list[int] = []
    for item in used:
        try:
            normalized_used.append(int(item))
        except (TypeError, ValueError):
            continue

    payload = {
        "scores": scores,
        "rationales": rationales,
        "flags": {
            "missing_citation": bool(flags.get("missing_citation", False)),
            "contradiction_with_evidence": bool(flags.get("contradiction_with_evidence", False)),
            "gold_mismatch_but_supported": bool(flags.get("gold_mismatch_but_supported", False)),
        },
        "used_citations": normalized_used,
    }
    return payload


def _resolve_llm_overall(llm_scores: dict[str, Any]) -> float:
    scores = llm_scores.get("scores") or {}
    overall = scores.get("overall")
    if isinstance(overall, (int, float)):
        return float(overall)

    components = {
        "correctness": scores.get("correctness", 0),
        "citation_support": scores.get("citation_support", 0),
        "completeness": scores.get("completeness", 0),
        "conciseness": scores.get("conciseness", 0),
        "hallucination_risk": scores.get("hallucination_risk", 0),
    }
    try:
        calc = (
            8 * float(components["correctness"])
            + 5 * float(components["citation_support"])
            + 3 * float(components["completeness"])
            + 2 * float(components["conciseness"])
            + 2 * float(components["hallucination_risk"])
        )
        return float(calc)
    except (TypeError, ValueError):
        return 0.0


def judge_example(
    question: str,
    gold: str,
    agent_answer: str,
    cited_passages: dict[int, str] | dict[str, str],
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> dict[str, Any]:
    passages = _prepare_passages(cited_passages)
    evidence_json = _format_evidence(passages)
    prompt = PROMPT_TEMPLATE.format(
        QUESTION=question or "",
        GOLD_ANSWER=gold or "",
        AGENT_ANSWER=agent_answer or "",
        EVIDENCE=evidence_json,
    )

    llm_payload = _call_llm(prompt, model=model, temperature=temperature)
    llm_scores = _ensure_llm_payload(llm_payload)

    auto_metrics = _normalize_auto_metrics(agent_answer or "", gold or "", passages)
    used_detected: list[int] = []
    for cid in extract_citation_ids(agent_answer or ""):
        if cid in passages and cid not in used_detected:
            used_detected.append(cid)
    composite = round(
        0.6 * _resolve_llm_overall(llm_scores)
        + 0.25 * auto_metrics["f1_short"] * 100
        + 0.15 * auto_metrics["support_overlap"] * 100,
        2,
    )

    result = {
        "llm_scores": llm_scores,
        "auto_metrics": auto_metrics,
        "composite_overall": composite,
        "debug": {
            "model": model,
            "used_citations_detected": used_detected,
        },
    }
    return result


def judge_batch(
    rows: Sequence[dict[str, Any]],
    passage_lookup: dict[str, dict[int, str]] | None,
    *,
    model: str = "gpt-4o-mini",
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    lookup = passage_lookup or {}
    for row in rows:
        qid = str(row.get("qid") or row.get("id") or "")
        question = row.get("question", "")
        gold = row.get("gold") or row.get("gold_answer") or ""
        answer = row.get("answer") or row.get("final_answer") or ""
        local_passages = _prepare_passages(row.get("passages") or lookup.get(qid) or {})
        try:
            judgement = judge_example(
                question,
                gold,
                answer,
                local_passages,
                model=model,
            )
            enriched = dict(row)
            enriched.update(judgement)
            results.append(enriched)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("Judge failed for qid=%s: %s", qid, exc)
            error_row = dict(row)
            error_row["error"] = str(exc)
            results.append(error_row)
    return results
