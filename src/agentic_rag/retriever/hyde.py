"""HyDE (Hypothetical Document Embeddings) for query rewriting."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_rag.models.adapter import OpenAIAdapter


def hyde_query(
    query: str, llm_client: OpenAIAdapter, max_tokens: int = 120
) -> str | None:
    """
    Generate a short hypothetical passage to pivot retrieval (HyDE).
    Use ONLY when RETRIEVE_MORE fires (not in round 1) to save tokens.

    Args:
        query: Original query text
        llm_client: LLM client with chat method
        max_tokens: Maximum tokens for generated passage

    Returns:
        Generated hypothetical passage or None if generation fails
    """
    from agentic_rag.models.adapter import ChatMessage

    prompt_messages = [
        ChatMessage(
            role="system",
            content="Write a concise factual paragraph that would likely answer the query. "
            "Do not cite sources or speculate wildly. Keep it generic but content-rich.",
        ),
        ChatMessage(
            role="user",
            content=f"Query: {query}\n\nWrite a factual paragraph (~70-120 words) that would answer this:",
        ),
    ]

    try:
        # Use the same chat interface as the main agent
        response, usage = llm_client.chat(
            messages=prompt_messages,
            max_tokens=max_tokens,
            temperature=0.3,  # Slight creativity for diversity
        )

        text = response.strip()
        # Return None on pathologically short responses
        return text if len(text.split()) > 5 else None

    except Exception as e:
        # Graceful degradation - return None to fall back to original query
        print(f"HyDE generation failed: {e}")
        return None


def is_factoid(question: str) -> bool:
    q = (question or "").lower()
    # Simple cues for precise factoid questions
    return any(
        tok in q
        for tok in [
            "when",
            "what year",
            "which",
            "how many",
            "how long",  # Added for factoid detection
            "date",
            "per game",
            "ex-dividend",
        ]
    )


def should_use_hyde(
    round_idx: int, use_hyde_setting: bool, question: str | None = None
) -> bool:
    """
    Determine if HyDE should be used for this round.

    Args:
        round_idx: Current round index (0-based)
        use_hyde_setting: Configuration setting for HyDE usage

    Returns:
        True if HyDE should be used
    """
    # Only use HyDE on RETRIEVE_MORE (round > 0) and if enabled,
    # and avoid it for precise factoids to preserve retrieval precision.
    if not use_hyde_setting or round_idx == 0 or (question and is_factoid(question)):
        return False
    return True
