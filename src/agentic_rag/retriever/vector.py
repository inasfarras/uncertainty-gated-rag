import os
from typing import Dict, List, Tuple, TypedDict

import pandas as pd
import tiktoken

from agentic_rag.config import settings
from agentic_rag.embed.encoder import embed_texts
from agentic_rag.store.faiss_store import load_index


class ContextChunk(TypedDict):
    id: str
    text: str
    score: float


class VectorRetriever:
    def __init__(self, faiss_dir: str = "artifacts/faiss"):
        self.store = load_index(faiss_dir)
        self.chunks = pd.read_parquet(
            os.path.join(faiss_dir, "chunks.parquet")
        ).set_index("id")

    def _encoding(self):
        try:
            return tiktoken.encoding_for_model(settings.LLM_MODEL)
        except Exception:
            return tiktoken.get_encoding("o200k_base")

    def _token_count(self, text: str) -> int:
        try:
            enc = self._encoding()
            return len(enc.encode(text or ""))
        except Exception:
            return max(1, len(text or "") // 4)

    def retrieve(self, query: str, k: int) -> Tuple[List[dict], Dict[str, object]]:
        """Retrieve top-k, pack under token cap, return contexts and stats.

        Returns:
            contexts: list of {id, text}
            stats: {context_tokens, n_ctx_blocks, retrieved_ids}
        """
        qvec = embed_texts([query])[0]
        hits = self.store.search(qvec, k)

        # Map chunk hits to best chunk per doc_id
        best_by_doc: Dict[str, Tuple[str, float]] = {}
        for chunk_id, score in hits:
            # Expect chunk ids like {doc_id}__{i}
            doc_id = chunk_id.split("__")[0]
            if (doc_id not in best_by_doc) or (score > best_by_doc[doc_id][1]):
                best_by_doc[doc_id] = (chunk_id, score)

        # Sort docs by score desc
        sorted_docs = sorted(best_by_doc.items(), key=lambda x: x[1][1], reverse=True)
        retrieved_ids = [doc_id for doc_id, _ in sorted_docs]

        # Pack under token cap
        cap = settings.CONTEXT_TOKEN_CAP
        total_tokens = 0
        contexts: List[dict] = []
        for doc_id, (chunk_id, score) in sorted_docs:
            text = self.chunks.loc[chunk_id, "text"]
            tks = self._token_count(text)
            if total_tokens + tks > cap:
                break
            contexts.append({"id": doc_id, "text": text})
            total_tokens += tks

        stats = {
            "context_tokens": total_tokens,
            "n_ctx_blocks": len(contexts),
            "retrieved_ids": retrieved_ids,
        }
        return contexts, stats
