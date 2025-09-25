from typing import Iterable, List, Tuple, TypedDict

import tiktoken

from agentic_rag.config import settings


class ContextBlock(TypedDict):
    id: str
    text: str
    score: float


def _encoding():
    try:
        return tiktoken.encoding_for_model(settings.LLM_MODEL)
    except Exception:
        return tiktoken.get_encoding("o200k_base")


def token_count(text: str) -> int:
    try:
        enc = _encoding()
        return len(enc.encode(text or ""))
    except Exception:
        return max(1, len(text or "") // 4)


def pack_context(
    chunks: Iterable[ContextBlock],
    max_tokens_cap: int | None = None,
) -> Tuple[List[ContextBlock], int, int]:
    """Pack context chunks under a token cap.

    Sorts by score desc and drops lowest-score blocks when exceeding the cap.

    Returns (packed_chunks, context_tokens, n_ctx_blocks).
    """
    cap = max_tokens_cap if max_tokens_cap is not None else settings.MAX_CONTEXT_TOKENS
    # Sort descending by score
    sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)
    total = 0
    packed: List[ContextBlock] = []
    for c in sorted_chunks:
        tks = token_count(c["text"])
        if total + tks > cap:
            continue
        packed.append(c)
        total += tks
    return packed, total, len(packed)


def render_context(chunks: Iterable[ContextBlock]) -> str:
    blocks = [f"CTX[{c['id']}]:\n{c['text']}" for c in chunks]
    return "\n\n".join(blocks)


def build_system_instructions() -> str:
    return (
        "You answer ONLY using the provided CONTEXT.\n"
        "If information is missing, answer EXACTLY: I don't know.\n"
        "Limit your answer to 1–2 sentences.\n"
        "Each non-IDK sentence MUST include exactly one citation in the form [CIT:<doc_id>].\n"
        "If you answer I don't know (or Tidak tahu), do not include any citation.\n"
        "Citation format must be exactly [CIT:<doc_id>] where <doc_id> matches the CTX header and uses only letters, digits, _, or -.\n"
        "Be precise and concise; avoid extra sentences."
    )
