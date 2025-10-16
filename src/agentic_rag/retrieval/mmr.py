from typing import Any, Dict, List

import numpy as np

# Placeholder for an Item type, assuming it has 'text', 'embedding'/'vec', and 'score'
# In a real scenario, this would likely be a dataclass or a more formal object.
# For now, we'll treat 'items' as a list of dictionaries.


def mmr_rerank(
    items: List[Dict[str, Any]],
    top_k: int = 10,
    lambda_diversity: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Performs Maximal Marginal Relevance (MMR) reranking on a list of items.

    Args:
        items: A list of dictionaries, where each dictionary represents an item
               and must contain 'embedding' (or 'vec') and 'score' keys.
        top_k: The number of top items to return after reranking.
        lambda_diversity: The diversity constant (0.0-1.0). Higher values
                          prioritize diversity, lower values prioritize relevance.

    Returns:
        A list of reranked items (dictionaries).
    """
    if not items:
        return []

    # Extract embeddings and scores
    # Try 'embedding' first, then 'vec' for compatibility
    # Ensure embeddings are always numerical and of the correct dimension
    valid_embeddings = []
    scores_list = []
    for item in items:
        emb = item.get("embedding")
        if emb is None:
            emb = item.get("vec")
        if emb is not None:
            valid_embeddings.append(emb)
            scores_list.append(item["score"])

    if not valid_embeddings:
        return []  # No valid embeddings, return empty list

    embeddings = np.array(valid_embeddings)
    scores = np.array(scores_list)

    # If embeddings are 1D (e.g., a list of single vectors), reshape to 2D
    if embeddings.ndim == 1 and len(valid_embeddings) > 0:
        embeddings = np.array(list(valid_embeddings)).reshape(len(valid_embeddings), -1)
    elif (
        embeddings.ndim == 1 and len(valid_embeddings) == 0
    ):  # Handle empty valid_embeddings explicitly
        return []

    # Normalize embeddings for cosine similarity
    # Handle potential division by zero if all embeddings are zero vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embeddings = np.where(norms == 0, 0, embeddings / norms).astype(np.float32)

    # Initialize list for reranked item indices
    reranked_indices = []

    # Select the first item based on relevance score
    # Ensure scores are not empty
    if scores.size > 0:
        most_relevant_idx = np.argmax(scores)
        reranked_indices.append(most_relevant_idx)

    # Iteratively select items
    while len(reranked_indices) < top_k and len(reranked_indices) < len(items):
        remaining_indices = list(set(range(len(items))) - set(reranked_indices))
        if not remaining_indices:
            break

        # Calculate MMR scores for remaining items
        mmr_scores = []
        for candidate_idx in remaining_indices:
            candidate_embedding = norm_embeddings[candidate_idx]

            # Calculate similarity to already selected items
            if (
                reranked_indices
            ):  # Ensure there are selected items before calculating similarity
                selected_embeddings = norm_embeddings[reranked_indices]

                # Reshape candidate_embedding for dot product if it's 1D - already handled at the start
                # if candidate_embedding.ndim == 1:
                #     candidate_embedding = candidate_embedding.reshape(1, -1)

                # Ensure selected_embeddings and candidate_embedding have compatible dimensions
                if selected_embeddings.ndim == 1:
                    selected_embeddings = selected_embeddings.reshape(1, -1)
                if candidate_embedding.ndim == 1:
                    candidate_embedding = candidate_embedding.reshape(1, -1)

                similarities = np.dot(
                    selected_embeddings, candidate_embedding.T
                ).flatten()
                max_similarity = np.max(similarities)
            else:
                max_similarity = 0.0  # No selected items yet, so no diversity penalty

            # MMR formula
            mmr_score = (
                lambda_diversity * scores[candidate_idx]
                - (1 - lambda_diversity) * max_similarity
            )
            mmr_scores.append((mmr_score, candidate_idx))

        # Select the item with the highest MMR score
        if mmr_scores:
            best_mmr_candidate = max(mmr_scores, key=lambda x: x[0])
            reranked_indices.append(best_mmr_candidate[1])
        else:
            break  # No valid candidates left to add

    # Return the reranked items, ensuring they map back to the original `items` list
    return [items[i] for i in reranked_indices]
