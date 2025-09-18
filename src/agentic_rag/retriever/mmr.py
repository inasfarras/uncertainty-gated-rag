"""MMR (Maximal Marginal Relevance) for diversifying context selection."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from agentic_rag.prompting import ContextBlock, pack_context


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalized vectors."""
    return float(np.dot(a, b))


def mmr_select(
    query_embedding: np.ndarray,
    candidates: list[dict[str, Any]],
    k: int,
    lambda_mult: float,
) -> list[dict[str, Any]]:
    """Select k candidates from a list using Maximal Marginal Relevance (MMR)."""
    if not candidates:
        return []

    selected_candidates = []
    candidate_embeddings = np.array([c["emb"] for c in candidates])

    # Calculate similarity between query and all candidates
    query_candidate_similarity = np.dot(candidate_embeddings, query_embedding)

    # Find the best candidate to start with
    best_candidate_idx = np.argmax(query_candidate_similarity)
    selected_candidates.append(candidates[best_candidate_idx])

    # Keep track of selected indices
    selected_indices = {best_candidate_idx}

    while len(selected_candidates) < min(k, len(candidates)):
        best_mmr_score = -np.inf
        best_candidate_idx = -1

        for i in range(len(candidates)):
            if i in selected_indices:
                continue

            # Similarity to query
            sim_to_query = query_candidate_similarity[i]

            # Max similarity to already selected candidates
            selected_embeddings = np.array([c["emb"] for c in selected_candidates])
            sim_to_selected = np.max(
                np.dot(selected_embeddings, candidate_embeddings[i])
            )

            # MMR score
            mmr_score = lambda_mult * sim_to_query - (1 - lambda_mult) * sim_to_selected

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_candidate_idx = i

        if best_candidate_idx != -1:
            selected_candidates.append(candidates[best_candidate_idx])
            selected_indices.add(best_candidate_idx)
        else:
            # No more candidates to select
            break

    return selected_candidates


def mmr_pack_context(
    blocks: list[dict[str, Any]], max_tokens_cap: int, mmr_lambda: float
) -> tuple[list[dict[str, Any]], int, int]:
    """Pack context blocks using MMR for diversity."""
    # This is a placeholder. For now, we'll just use the default packing.
    # A full implementation would require embeddings for each block.

    # Cast the list of dicts to a list of ContextBlock to satisfy mypy
    context_blocks = [cast(ContextBlock, block) for block in blocks]

    packed_blocks, total_tokens, n_blocks = pack_context(context_blocks, max_tokens_cap)

    # Cast back to list of dicts for the return type
    return packed_blocks, total_tokens, n_blocks
