import threading
from typing import cast

import numpy as np
import numpy.typing as npt
import tiktoken
from sentence_transformers import SentenceTransformer

from agentic_rag.config import settings
from agentic_rag.models.adapter import get_openai

_st_model_cache: SentenceTransformer | None = None
_st_model_lock = threading.Lock()


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
    if not texts:
        return np.array([], dtype=np.float32).reshape(
            0, 1536
        )  # Return empty 2D array if no texts

    # Default embedding dimension for zero-vectors if an embedding fails
    embedding_dim = 1536  # Default for text-embedding-3-small

    embs_list = []
    if settings.EMBED_BACKEND == "mock":
        import hashlib

        for text in texts:
            hash_bytes = hashlib.md5(text.encode()).digest()
            vec = np.frombuffer(hash_bytes * 96, dtype=np.uint8)[:embedding_dim].astype(
                np.float32
            )
            vec = (vec / 127.5) - 1.0
            embs_list.append(vec)
        arr = np.array(embs_list, dtype=np.float32)
        return _normalize(arr)
    elif settings.EMBED_BACKEND == "openai":
        client = get_openai()
        try:
            embs = client.embed(texts, embed_model=settings.EMBED_MODEL)
            # Ensure embs is a list of lists or similar for np.array
            embs_list = [np.array(e, dtype=np.float32) for e in embs]
            arr = (
                np.array(embs_list, dtype=np.float32)
                if embs_list
                else np.array([], dtype=np.float32).reshape(0, embedding_dim)
            )
            return _normalize(arr)
        except Exception as e:
            print(f"Error with OpenAI embeddings: {e}. Returning zero vectors.")
            return np.zeros((len(texts), embedding_dim), dtype=np.float32)
    elif settings.EMBED_BACKEND == "st":
        import torch

        global _st_model_cache

        # Use lock to ensure thread-safe model loading
        with _st_model_lock:
            if _st_model_cache is None:
                model_name = getattr(
                    settings, "ST_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(
                    f"[EMBED] Loading sentence-transformers model: {model_name} on {device}"
                )
                _st_model_cache = SentenceTransformer(model_name, device=device)
                print("[EMBED] Model loaded and cached successfully")

        batch_size = getattr(settings, "EMBED_BATCH_SIZE", 32)
        try:
            embs = _st_model_cache.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
                batch_size=batch_size,
            )
            return _normalize(cast(npt.NDArray[np.float32], embs))
        except Exception as e:
            print(
                f"Error with Sentence-transformers embeddings: {e}. Returning zero vectors."
            )
            return np.zeros((len(texts), embedding_dim), dtype=np.float32)
    else:
        raise RuntimeError(f"Unknown embed backend: {settings.EMBED_BACKEND}")
