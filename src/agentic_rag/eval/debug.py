"""Debug utilities for RAG evaluation."""

import json
from pathlib import Path

from agentic_rag.eval.signals import (
    extract_citations,
    extract_sentence_citations,
    sentence_split,
)


def log_debug_info(
    qid: str,
    question: str,
    prompt_messages: list[dict[str, str]],
    answer: str,
    context_ids: list[str],
    debug_dir: Path = Path("logs/debug"),
) -> None:
    """Log debug information for citation analysis."""
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Extract citations and analyze
    sentences = sentence_split(answer)
    sentence_citations = extract_sentence_citations(answer)
    all_citations = extract_citations(answer)

    # Find unmatched citations
    available_ids = set(context_ids)
    unmatched_citations = [cit for cit in all_citations if cit not in available_ids]

    debug_info = {
        "qid": qid,
        "question": question,
        "system_prompt": next(
            (m["content"] for m in prompt_messages if m["role"] == "system"), ""
        ),
        "user_prompt": next(
            (m["content"] for m in prompt_messages if m["role"] == "user"), ""
        ),
        "model_answer": answer,
        "sentences": sentences,
        "sentence_citations": [list(cits) for cits in sentence_citations],
        "all_citations": all_citations,
        "available_context_ids": context_ids,
        "unmatched_citations": unmatched_citations,
        "sentences_with_citations": sum(1 for cits in sentence_citations if cits),
        "sentences_without_citations": sum(
            1 for cits in sentence_citations if not cits
        ),
        "total_sentences": len(sentences),
    }

    debug_file = debug_dir / f"{qid}_debug.json"
    with open(debug_file, "w", encoding="utf-8") as f:
        json.dump(debug_info, f, indent=2, ensure_ascii=False)

    print(f"Debug info saved to {debug_file}")
    print(f"Citations found: {all_citations}")
    print(f"Context IDs: {context_ids}")
    print(f"Unmatched citations: {unmatched_citations}")
    print(
        f"Sentences with citations: {debug_info['sentences_with_citations']}/{debug_info['total_sentences']}"
    )
