import numpy as np
import numpy.typing as npt
import tiktoken

from agentic_rag.config import settings
from agentic_rag.models.adapter import get_openai


def _encoding():
    try:
        return tiktoken.encoding_for_model(settings.LLM_MODEL)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, max_tokens: int = 600, overlap: int = 100) -> list[str]:
    enc = _encoding()
    tokens = enc.encode(text or "")
    chunks = []
    i = 0
    while i < len(tokens):
        window = tokens[i : i + max_tokens]
        chunks.append(enc.decode(window))
        i += max_tokens - overlap
        if (max_tokens - overlap) <= 0:
            break
    return chunks or [text]


def _normalize(vecs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return (vecs / norms).astype(np.float32)


def embed_texts(texts: list[str]) -> npt.NDArray[np.float32]:
    if settings.EMBED_BACKEND != "openai":
        raise RuntimeError("Only 'openai' embed backend is enabled for now.")
    client = get_openai()
    embs = client.embed(texts, embed_model=settings.EMBED_MODEL)
    arr = np.array(embs, dtype=np.float32)
    return _normalize(arr)
