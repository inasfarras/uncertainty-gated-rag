"""BGE Cross-Encoder Reranking for improving retrieval quality."""

from __future__ import annotations

from typing import Any

try:
    from FlagEmbedding import FlagReranker  # pip install FlagEmbedding
except ImportError:
    FlagReranker = None


class BGECrossEncoder:
    """BGE cross-encoder reranker for filtering best passages post-retrieval."""

    def __init__(
        self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True
    ):
        if FlagReranker is None:
            raise ImportError(
                "FlagEmbedding not installed. Install with: pip install FlagEmbedding"
            )
        import torch

        # Force CUDA device if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker = FlagReranker(model_name, use_fp16=use_fp16, device=device)
        self.model_name = model_name
        print(f"   🚀 BGE Reranker loaded on {device}")

    def rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """
        Rerank candidates using cross-encoder and return top_k results.

        Args:
            query: Search query text
            candidates: List of candidate documents with 'text' field
            top_k: Number of top candidates to return

        Returns:
            List of reranked candidates with 'rerank_score' field added
        """
        if not candidates:
            return []

        # Get batch size from config
        from agentic_rag.config import settings

        batch_size = getattr(settings, "RERANK_BATCH_SIZE", 64)

        if len(candidates) <= top_k:
            # If we have fewer candidates than requested, just add scores
            pairs = [[query, c["text"]] for c in candidates]
            # Batch process for speed
            scores = self.reranker.compute_score(
                pairs, normalize=True, batch_size=batch_size
            )
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        # Rerank and select top_k - batch process all pairs
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.reranker.compute_score(
            pairs, normalize=True, batch_size=batch_size
        )

        # Add scores to candidates
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        # Sort by rerank score and return top_k
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]


def create_reranker(model_name: str | None = None) -> BGECrossEncoder | None:
    """
    Factory function to create a BGE reranker if FlagEmbedding is available.

    Returns:
        BGECrossEncoder instance or None if FlagEmbedding not available
    """
    if FlagReranker is None:
        return None

    model_name = model_name or "BAAI/bge-reranker-v2-m3"
    try:
        # Avoid dtype issues by honoring settings for FP16 usage
        from agentic_rag.config import settings

        return BGECrossEncoder(
            model_name, use_fp16=bool(getattr(settings, "RERANK_FP16", False))
        )
    except Exception:
        return None
