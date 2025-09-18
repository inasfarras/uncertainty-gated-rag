"""Reflection prompting for correcting and improving answers."""

from agentic_rag.models.adapter import ChatMessage


def build_reflect_prompt(
    contexts: list[dict], original_answer: str
) -> tuple[list[ChatMessage], str]:
    """Builds a REFLECT prompt to repair a poor answer."""
    context_str = "\n\n".join([f"CTX[{c['id']}]:\n{c['text']}" for c in contexts])

    # Improved REFLECT prompt
    system_content = (
        "You are a meticulous editor. Your task is to critically review an answer and the provided context. "
        "Your goal is to produce a corrected answer that is fully supported by the context. "
        "Follow these instructions exactly:\n"
        "1.  For each sentence in the ORIGINAL ANSWER, verify if it is directly supported by the CONTEXT. A sentence is supported if you can find clear evidence for it in the CONTEXT.\n"
        "2.  If a sentence is fully supported, keep it and its citation.\n"
        "3.  If a sentence is partially supported, rewrite it to be fully supported by the CONTEXT.\n"
        "4.  If a sentence is not supported at all, REMOVE it.\n"
        "5.  If, after removing all unsupported sentences, the answer is empty or does not address the question, output EXACTLY: I don't know.\n"
        "6.  Do NOT add any new information that is not in the CONTEXT.\n"
        "7.  Preserve the original citation format [CIT:<doc_id>] for all supported claims."
    )

    user_content = (
        f"CONTEXT:\n{context_str}\n\n"
        f"ORIGINAL ANSWER:\n{original_answer}\n\n"
        f"CORRECTED ANSWER:"
    )

    debug_prompt = f"SYSTEM:\n{system_content}\n\n{user_content}"
    return [
        ChatMessage(role="system", content=system_content),
        ChatMessage(role="user", content=user_content),
    ], debug_prompt


def should_reflect(action: str, has_reflect_left: bool) -> bool:
    """Determines if the REFLECT action should be taken."""
    return action == "REFLECT" and has_reflect_left
