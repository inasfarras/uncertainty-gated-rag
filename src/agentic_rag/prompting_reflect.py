"""Reflection prompting for correcting and improving answers."""

from typing import Any, Dict, List, Tuple

from agentic_rag.models.adapter import ChatMessage

REFLECT_SYSTEM = """You are a careful editor. You must ensure every sentence is supported by the provided CTX blocks.

Rules:
- If a sentence is unsupported by CTX, revise it or delete it.
- If key information is missing in CTX, answer with exactly: "I don't know." (NO citation).
- Every claim sentence MUST end with a citation [CIT:<doc_id>] matching a CTX header (alphanumeric, '_' or '-').
- "I don't know." MUST have NO citation.
- Do not invent facts. Be concise (<= 4 sentences)."""


REFLECT_USER_TEMPLATE = """CTX:
{rendered_ctx}

CURRENT ANSWER:
{answer}

TASK:
1) Check each sentence against CTX.
2) Fix or remove unsupported claims.
3) If not enough evidence remains, output only: I don't know.

Return the final answer now:"""


def build_reflect_prompt(
    contexts: List[Dict[str, Any]], current_answer: str
) -> Tuple[List[ChatMessage], str]:
    """
    Build reflection prompt to correct and improve the current answer.

    Args:
        contexts: List of context chunks with 'id' and 'text' fields
        current_answer: Current answer to be reflected upon

    Returns:
        Tuple of (messages, debug_prompt)
    """
    # Render context blocks
    context_blocks = []
    for c in contexts:
        context_blocks.append(f"CTX[{c['id']}]:\n{c['text']}")
    rendered_ctx = "\n\n".join(context_blocks)

    user_content = REFLECT_USER_TEMPLATE.format(
        rendered_ctx=rendered_ctx, answer=current_answer
    )

    messages = [
        ChatMessage(role="system", content=REFLECT_SYSTEM),
        ChatMessage(role="user", content=user_content),
    ]

    debug_prompt = f"SYSTEM:\n{REFLECT_SYSTEM}\n\nUSER:\n{user_content}"

    return messages, debug_prompt


def should_reflect(action: str, has_reflect_left: bool) -> bool:
    """
    Determine if reflection should be performed based on gate action.

    Args:
        action: Action returned by the gate
        has_reflect_left: Whether reflection is still available

    Returns:
        True if reflection should be performed
    """
    return action == "REFLECT" and has_reflect_left
