from __future__ import annotations

"""Anchor-aware retrieval worker (rough→fine) returning Paths.

Keeps a tiny surface and reuses the existing VectorRetriever for now.
Optionally accepts a KG handle in the future.
"""

from typing import Any, TypedDict

import numpy as np
from agentic_rag.config import settings
from agentic_rag.embed.encoder import embed_texts
from agentic_rag.retriever.vector import VectorRetriever


class Path(TypedDict, total=False):
    anchor: str
    hops: int
    doc_ids: list[str]
    passages: list[dict[str, Any]]
    novelty_ratio: float
    rough_scores: list[float]
    fine_scores: list[float]
    pruned_count: int
    terminated_by: str


class RetrievalAgent:
    def __init__(
        self,
        retriever: VectorRetriever | None = None,
        faiss_dir: str | None = None,
    ):
        # Default to configured FAISS index path to avoid mismatches
        self.retriever = retriever or VectorRetriever(
            faiss_dir or settings.FAISS_INDEX_PATH
        )

    def explore(
        self,
        anchor: str,
        question: str,
        hop_budget: int = 1,
        seen_doc_ids: set[str] | None = None,
    ) -> list[Path]:
        """Single-hop rough→fine exploration constrained by anchor.

        For now, we run one hop of hybrid/FAISS retrieval with the anchor
        appended to the query. Fine filtering uses simple semantic similarity
        between question and candidate chunk text.
        """
        seen: set[str] = set(seen_doc_ids or set())
        q = f"{question} :: {anchor}" if anchor else question

        # Respect global retrieval settings so profiles/overrides take effect
        try:
            k = int(getattr(settings, "RETRIEVAL_K", 8))
        except Exception:
            k = 8
        try:
            probe = int(getattr(settings, "PROBE_FACTOR", 2))
        except Exception:
            probe = 2

        contexts, stats = self.retriever.retrieve_pack(q, k=k, probe_factor=probe)
        doc_ids = [c.get("id", "") for c in contexts]
        new_ids = [d for d in doc_ids if d not in seen]
        novelty_ratio = (len(new_ids) / max(1, len(doc_ids))) if doc_ids else 0.0

        # Fine: semantic sim with question against chunk texts
        fine_scores: list[float] = []
        try:
            em_q = embed_texts([question])[0].astype(np.float32)
            em_ctx = embed_texts([c.get("text", "") for c in contexts]).astype(
                np.float32
            )
            sims = em_ctx @ em_q
            fine_scores = [float(x) for x in sims.tolist()]
        except Exception:
            fine_scores = [0.0 for _ in contexts]

        path: Path = {
            "anchor": anchor,
            "hops": 1,
            "doc_ids": doc_ids,
            "passages": contexts,
            "novelty_ratio": float(novelty_ratio),
            "rough_scores": [float(c.get("score", 0.0)) for c in contexts],
            "fine_scores": fine_scores,
            "pruned_count": 0,
            "terminated_by": "BUDGET" if hop_budget <= 1 else "NONE",
        }
        return [path]
