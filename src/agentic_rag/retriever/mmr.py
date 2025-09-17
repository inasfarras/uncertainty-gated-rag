"""MMR (Maximal Marginal Relevance) for diversifying context selection."""

from __future__ import annotations

from typing import Any

import numpy as np


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalized vectors."""
    return float(np.dot(a, b))


def mmr_select(
    q_emb: np.ndarray,
    candidates: list[dict[str, Any]],
    k: int,
    lambda_div: float = 0.4,
) -> list[dict[str, Any]]:
    """
    Select k items using Maximal Marginal Relevance to balance relevance and diversity.

    Args:
        q_emb: Query embedding (L2-normalized)
        candidates: List of candidate documents with 'emb' field containing L2-normalized embeddings
        k: Number of items to select
        lambda_div: Diversity parameter (0.0 = pure diversity, 1.0 = pure relevance)

    Returns:
        List of selected candidates maximizing MMR = λ * sim(q, d) - (1-λ) * max_j sim(d, d_j_selected)
    """
    if not candidates or k <= 0:
        return []

    selected: list[dict[str, Any]] = []
    pool = candidates.copy()

    # Seed with best by query similarity
    pool.sort(key=lambda x: _cos(q_emb, x["emb"]), reverse=True)
    selected.append(pool.pop(0))

    # Greedily select remaining items
    while pool and len(selected) < k:
        best_idx, best_mmr = 0, -1e9
        selected_embs = [s["emb"] for s in selected]

        for i, candidate in enumerate(pool):
            # Relevance: similarity to query
            relevance = _cos(q_emb, candidate["emb"])

            # Redundancy: max similarity to already selected items
            redundancy = max(_cos(candidate["emb"], s_emb) for s_emb in selected_embs)

            # MMR score
            mmr = lambda_div * relevance - (1.0 - lambda_div) * redundancy

            if mmr > best_mmr:
                best_mmr, best_idx = mmr, i

        selected.append(pool.pop(best_idx))

    return selected
