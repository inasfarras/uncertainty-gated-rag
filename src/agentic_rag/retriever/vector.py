import os
from typing import List, TypedDict

import pandas as pd

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

    def retrieve(self, query: str, k: int) -> List[ContextChunk]:
        qvec = embed_texts([query])[0]
        hits = self.store.search(qvec, k)
        out: List[ContextChunk] = []
        for _id, score in hits:
            text = self.chunks.loc[_id, "text"]
            out.append({"id": _id, "text": text, "score": score})
        return out
