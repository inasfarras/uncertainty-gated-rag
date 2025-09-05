"""FAISS-based vector store for efficient similarity search."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel

from agentic_rag.models.data import Chunk


class SearchResult(BaseModel):
    """Result from vector similarity search."""

    chunk: Chunk
    score: float
    rank: int


class SearchResults(BaseModel):
    """Collection of search results."""

    results: List[SearchResult]
    query: str
    total_time_ms: float
    metadata: Dict[str, Any] = {}


class BaseFAISSStore:
    """Base class for FAISS vector stores."""

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        metric: str = "cosine",
        **kwargs,
    ) -> None:
        """
        Initialize FAISS store.

        Args:
            dimension: Dimensionality of the embeddings
            index_type: FAISS index type ("flat", "ivf", "hnsw")
            metric: Distance metric ("cosine", "l2", "inner_product")
            **kwargs: Additional configuration parameters
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.config = kwargs
        self._index = None
        self._chunks = []
        self._id_to_idx = {}

    def build_index(self, chunks: List[Chunk], **kwargs) -> None:
        """
        Build FAISS index from chunks.

        Args:
            chunks: List of chunks with embeddings
            **kwargs: Additional build parameters
        """
        # TODO: Implement index building
        raise NotImplementedError("Index building not yet implemented")

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add new chunks to the index.

        Args:
            chunks: List of chunks to add
        """
        # TODO: Implement chunk addition
        raise NotImplementedError("Chunk addition not yet implemented")

    def search(
        self,
        query_embedding: Union[List[float], np.ndarray],
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> SearchResults:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            Search results with chunks and scores
        """
        # TODO: Implement similarity search
        raise NotImplementedError("Similarity search not yet implemented")

    def batch_search(
        self,
        query_embeddings: Union[List[List[float]], np.ndarray],
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResults]:
        """
        Batch search for multiple queries.

        Args:
            query_embeddings: List of query embedding vectors
            k: Number of results to return per query
            filter_metadata: Optional metadata filters

        Returns:
            List of search results for each query
        """
        # TODO: Implement batch search
        raise NotImplementedError("Batch search not yet implemented")

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """
        Retrieve chunk by ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Chunk object if found, None otherwise
        """
        # TODO: Implement chunk retrieval
        raise NotImplementedError("Chunk retrieval not yet implemented")

    def update_chunk(self, chunk: Chunk) -> bool:
        """
        Update an existing chunk.

        Args:
            chunk: Updated chunk object

        Returns:
            True if update successful, False otherwise
        """
        # TODO: Implement chunk updates
        raise NotImplementedError("Chunk updates not yet implemented")

    def delete_chunks(self, chunk_ids: List[str]) -> int:
        """
        Delete chunks by IDs.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Number of chunks successfully deleted
        """
        # TODO: Implement chunk deletion
        raise NotImplementedError("Chunk deletion not yet implemented")

    def save_index(self, file_path: Union[str, Path]) -> None:
        """
        Save FAISS index to disk.

        Args:
            file_path: Path to save the index
        """
        # TODO: Implement index saving
        raise NotImplementedError("Index saving not yet implemented")

    def load_index(self, file_path: Union[str, Path]) -> None:
        """
        Load FAISS index from disk.

        Args:
            file_path: Path to the saved index
        """
        # TODO: Implement index loading
        raise NotImplementedError("Index loading not yet implemented")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.

        Returns:
            Dictionary with index statistics
        """
        # TODO: Implement statistics collection
        return {
            "total_chunks": len(self._chunks),
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
        }


class FlatFAISSStore(BaseFAISSStore):
    """FAISS store using flat (brute-force) index."""

    def __init__(self, dimension: int, metric: str = "cosine", **kwargs) -> None:
        """
        Initialize flat FAISS store.

        Args:
            dimension: Dimensionality of the embeddings
            metric: Distance metric
            **kwargs: Additional configuration
        """
        super().__init__(dimension, "flat", metric, **kwargs)


class IVFFAISSStore(BaseFAISSStore):
    """FAISS store using IVF (Inverted File) index."""

    def __init__(
        self,
        dimension: int,
        n_lists: int = 100,
        n_probe: int = 10,
        metric: str = "cosine",
        **kwargs,
    ) -> None:
        """
        Initialize IVF FAISS store.

        Args:
            dimension: Dimensionality of the embeddings
            n_lists: Number of clusters for IVF
            n_probe: Number of clusters to search
            metric: Distance metric
            **kwargs: Additional configuration
        """
        super().__init__(dimension, "ivf", metric, **kwargs)
        self.n_lists = n_lists
        self.n_probe = n_probe


class HNSWFAISSStore(BaseFAISSStore):
    """FAISS store using HNSW (Hierarchical Navigable Small World) index."""

    def __init__(
        self,
        dimension: int,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        metric: str = "cosine",
        **kwargs,
    ) -> None:
        """
        Initialize HNSW FAISS store.

        Args:
            dimension: Dimensionality of the embeddings
            m: Number of bi-directional links for each node
            ef_construction: Size of dynamic candidate list
            ef_search: Size of dynamic candidate list during search
            metric: Distance metric
            **kwargs: Additional configuration
        """
        super().__init__(dimension, "hnsw", metric, **kwargs)
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search


def create_faiss_store(
    dimension: int,
    index_type: str = "flat",
    metric: str = "cosine",
    **kwargs,
) -> BaseFAISSStore:
    """
    Factory function to create FAISS stores.

    Args:
        dimension: Dimensionality of the embeddings
        index_type: FAISS index type ("flat", "ivf", "hnsw")
        metric: Distance metric
        **kwargs: Additional configuration parameters

    Returns:
        Configured FAISS store instance

    Raises:
        ValueError: If index type is not supported
    """
    if index_type == "flat":
        return FlatFAISSStore(dimension, metric, **kwargs)
    elif index_type == "ivf":
        return IVFFAISSStore(dimension, metric=metric, **kwargs)
    elif index_type == "hnsw":
        return HNSWFAISSStore(dimension, metric=metric, **kwargs)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")
