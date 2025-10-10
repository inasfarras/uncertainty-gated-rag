from typing import cast

import numpy as np
import numpy.typing as npt
import tiktoken
from sentence_transformers import SentenceTransformer

from agentic_rag.config import settings
from agentic_rag.models.adapter import get_openai

_st_model_cache: SentenceTransformer | None = None


def _encoding():
    try:
        return tiktoken.encoding_for_model(settings.LLM_MODEL)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, max_tokens: int = 300, overlap: int = 50) -> list[str]:
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
    if settings.EMBED_BACKEND == "mock":
        # Mock embeddings: deterministic hash-based vectors
        import hashlib

        embs = []
        for text in texts:
            # Create deterministic embedding from text hash
            hash_bytes = hashlib.md5(text.encode()).digest()
            # Convert to float vector (1536 dimensions to match OpenAI)
            vec = np.frombuffer(hash_bytes * 96, dtype=np.uint8)[:1536].astype(
                np.float32
            )
            # Normalize to [-1, 1] range
            vec = (vec / 127.5) - 1.0
            embs.append(vec)
        arr = np.array(embs, dtype=np.float32)
        return _normalize(arr)
    elif settings.EMBED_BACKEND == "openai":
        client = get_openai()
        embs = client.embed(texts, embed_model=settings.EMBED_MODEL)
        arr = np.array(embs, dtype=np.float32)
        return _normalize(arr)
    elif settings.EMBED_BACKEND == "st":
        # Sentence-transformers: FREE offline embeddings
        # import torch # Already imported implicitly by SentenceTransformer, but explicit for clarity
        # from sentence_transformers import SentenceTransformer # Moved to top
        import torch

        global _st_model_cache

        # Cache model instance
        if _st_model_cache is None:
            model_name = getattr(
                settings, "ST_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading sentence-transformers model: {model_name} on {device}")
            _st_model_cache = SentenceTransformer(model_name, device=device)

        # Encode texts (with batch processing for efficiency)
        batch_size = getattr(settings, "EMBED_BATCH_SIZE", 32)
        embs = _st_model_cache.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,  # We normalize ourselves
            batch_size=batch_size,  # Process multiple texts at once for speed
        )
        return _normalize(cast(npt.NDArray[np.float32], embs))
    else:
        raise RuntimeError(f"Unknown embed backend: {settings.EMBED_BACKEND}")
