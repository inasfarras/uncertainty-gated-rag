"""MMR (Maximal Marginal Relevance) for diversifying context selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from agentic_rag.prompting import ContextBlock, pack_context


def mmr_select(
    query_embedding: np.ndarray,
    candidates: list[dict[str, Any]],
    k: int,
    lambda_mult: float,
) -> list[dict[str, Any]]:
    """Select k candidates from a list using Maximal Marginal Relevance (MMR)."""
    if not candidates:
        return []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    query_embedding_t = torch.from_numpy(query_embedding).to(device)
    candidate_embeddings_t = torch.from_numpy(
        np.array([c["emb"] for c in candidates])
    ).to(device)

    # Calculate similarity between query and all candidates
    query_candidate_similarity = torch.matmul(
        candidate_embeddings_t, query_embedding_t
    ).cpu()

    # Find the best candidate to start with
    best_candidate_idx = torch.argmax(query_candidate_similarity).item()

    selected_candidates = [candidates[best_candidate_idx]]
    selected_indices = {best_candidate_idx}

    while len(selected_candidates) < min(k, len(candidates)):
        best_candidate_idx_to_add = -1

        candidate_indices = [
            i for i in range(len(candidates)) if i not in selected_indices
        ]
        if not candidate_indices:
            break

        unselected_embeddings_t = candidate_embeddings_t[candidate_indices]
        selected_embeddings_t = candidate_embeddings_t[list(selected_indices)]

        # Calculate similarity between unselected and selected candidates
        sim_to_selected = torch.matmul(unselected_embeddings_t, selected_embeddings_t.T)
        max_sim_to_selected, _ = torch.max(sim_to_selected, dim=1)
        max_sim_to_selected = max_sim_to_selected.cpu()

        # Corresponding query similarities
        sim_to_query = query_candidate_similarity[candidate_indices]

        # MMR score calculation
        mmr_scores = (
            lambda_mult * sim_to_query - (1 - lambda_mult) * max_sim_to_selected
        )

        best_idx_in_unselected = torch.argmax(mmr_scores).item()
        best_candidate_idx_to_add = candidate_indices[best_idx_in_unselected]

        if best_candidate_idx_to_add != -1:
            selected_candidates.append(candidates[best_candidate_idx_to_add])
            selected_indices.add(best_candidate_idx_to_add)
        else:
            break

    return selected_candidates


def mmr_pack_context(
    blocks: list[ContextBlock], max_tokens_cap: int, mmr_lambda: float
) -> tuple[list[ContextBlock], int, int]:
    """Pack context blocks using MMR for diversity."""
    # This is a placeholder. For now, we'll just use the default packing.
    # A full implementation would require embeddings for each block.

    # Cast the list of dicts to a list of ContextBlock to satisfy mypy
    context_blocks = blocks

    packed_blocks, total_tokens, n_blocks = pack_context(context_blocks, max_tokens_cap)

    # Cast back to list of dicts for the return type
    return packed_blocks, total_tokens, n_blocks
